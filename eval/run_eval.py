#!/usr/bin/env python3
"""Unified benchmark runner for HybridLM checkpoints.

All multiple-choice benchmarks share one batched log-likelihood scorer:
examples are pre-tokenized once, every (example, option) pair is flattened
into a job list, jobs are length-bucketed, and 32 sequences are scored per
forward pass. GSM8K is the sole generation benchmark (batched greedy decode).

Usage:
    python eval/run_eval.py --ckpt ckpt_sft.pt
    python eval/run_eval.py --ckpt ckpt_sft.pt --benchmarks hellaswag piqa
"""

import argparse
import os
import sys
import time
from importlib import import_module

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# FlashKDA CUTLASS kernel is unavailable in some envs; train.py reads this at import.
os.environ["FLA_FLASH_KDA"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# load_dataset() checks for updates online even on a cache hit; the pod cannot
# reach huggingface.co (timeout) and hf-mirror's API 403s. Offline mode turns
# an infinite hang into an immediate exception, which the per-benchmark
# SKIPPED handler (run_eval.py:304) catches as a fast failure.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import fone
from scripts.loader import load_checkpoint, load_tokenizer, prompt_fn  # noqa: E402

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
LETTERS = ["A", "B", "C", "D"]
MC_BATCH = 32
GEN_BATCH = 16
GEN_MAX_NEW = 256


# --- Dataset adapters: normalize every benchmark to {"prompt", "options", "label"} ---
# Benchmark modules import lazily so a broken optional dep in one doesn't kill the run.

def _load_module(name):
    return import_module(f"eval.{name}")


def _load_context_mc(name):
    m = _load_module(name)
    return [
        {"prompt": d["context"], "options": d["options"], "label": d["label"]}
        for d in m.load_dataset()
    ]


def load_piqa():
    m = _load_module("piqa")
    return [
        {
            "prompt": d["goal"].rstrip(),
            "options": [" " + d["sol1"], " " + d["sol2"]],
            "label": d["label"],
        }
        for d in m.load_dataset()
    ]


def _load_arc(config):
    m = _load_module("arc")
    items = []
    for d in m.load_dataset(config):
        labels = d["choices"]["label"]
        items.append(
            {
                "prompt": d["question"].rstrip(),
                "options": [" " + t for t in d["choices"]["text"]],
                "label": labels.index(d["answerKey"]),
            }
        )
    return items


def load_boolq():
    m = _load_module("boolq")
    items = []
    for d in m.load_dataset():  # streaming
        items.append(
            {
                "prompt": f"Passage: {d['passage']}\nQuestion: {d['question']}?\nAnswer:",
                "options": [" no", " yes"],
                "label": int(d["answer"]),
            }
        )
    return items


def load_openbookqa():
    m = _load_module("openbookqa")
    items = []
    for d in m.load_dataset():  # streaming
        labels = d["choices"]["label"]
        texts = d["choices"]["text"]
        prompt = (
            f"Question: {d['question_stem']}\n"
            + "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            + "\nAnswer:"
        )
        items.append(
            {
                "prompt": prompt,
                "options": [f" {l}" for l in labels],
                "label": labels.index(d["answerKey"]),
            }
        )
    return items


def load_mmlu():
    m = _load_module("mmlu")
    items = []
    for q in m.load_dataset():
        opts = " ".join(f"{L}. {c}" for L, c in zip(LETTERS, q["choices"]))
        items.append(
            {
                "prompt": f"{q['question']} {opts}",
                "options": list(LETTERS),  # score bare letter tokens, matching mmlu.py
                "label": q["answer"],
            }
        )
    return items


MC_BENCHMARKS = {
    "hellaswag": ("HellaSwag", lambda: _load_context_mc("hellaswag")),
    "piqa": ("PIQA", load_piqa),
    "arc-easy": ("ARC-Easy", lambda: _load_arc("ARC-Easy")),
    "arc-challenge": ("ARC-Challenge", lambda: _load_arc("ARC-Challenge")),
    "winogrande": ("WinoGrande", lambda: _load_context_mc("winogrande")),
    "boolq": ("BoolQ", load_boolq),
    "openbookqa": ("OpenBookQA", load_openbookqa),
    "mmlu": ("MMLU", load_mmlu),
    "ceval": ("C-Eval (zh)", lambda: _load_module("ceval").load_items()),
}
# hellaswag/piqa: unreachable from this machine (pod HF egress broken; curl -4
# also times out on huggingface.co). Not a signal judgement -- run
# `--benchmarks hellaswag piqa` on a box with egress. The at-chance note in
# score_matrix SKIP_REASON applies to the English MC set as a whole and is a
# separate reason.
ALL_BENCHMARKS = [b for b in MC_BENCHMARKS if b not in ("hellaswag", "piqa")] + ["gsm8k"]


# --- Batched log-likelihood scorer (shared by all multiple-choice benchmarks) ---

@torch.no_grad()
def score_mc(model, tok, items, device, batch_size=MC_BATCH, num_id=None):
    """Mean accuracy over multiple-choice items. See score_mc_items for the contract.

    A MEAN AND NOTHING ELSE, which is why score_mc_items exists beside it: a standard
    error needs the per-item outcomes, and every caller of this function was throwing
    them away inside the return statement. Kept as the name three existing callers use
    (lambada_zh, math_v2_like, run_eval's own benchmarks) and now a thin wrapper, so the
    per-item path and the mean path cannot disagree -- the alternative was a second copy
    of the batching loop, which is the shape §95 names (a test or reader that
    reimplements the shipped function defends nothing against the two diverging).
    """
    preds, labels, scored = score_mc_items(model, tok, items, device, batch_size, num_id)
    if not len(labels):
        return 0.0
    n_bad = int((~scored).sum())
    if n_bad:
        # NOT counted as wrong and NOT counted as right: an item whose options all failed to
        # tokenize was never scored. Printed because a mean over a silently smaller set is a
        # different metric wearing the same name.
        print(f"score_mc: {n_bad} of {len(labels)} item(s) had no scoreable option and are "
              f"excluded from the mean (they would otherwise count CORRECT whenever "
              f"label == 0)", flush=True)
    keep = scored
    if not int(keep.sum()):
        return 0.0
    return (preds[keep] == labels[keep]).float().mean().item()


@torch.no_grad()
def score_mc_items(model, tok, items, device, batch_size=MC_BATCH, num_id=None):
    """Score multiple-choice items by continuation log-likelihood: (preds, labels).

    NO_GRAD IS ON THE FUNCTION, not left to the caller. score_mc has carried the decorator
    since it was written; this function was split out of it (:159) and the decorator did not
    come with it, so every caller inherited the obligation silently. MEASURED 2026-09-06:
    scoring 1000 four-way items on a 206M checkpoint allocated 94.49 GiB and died with
    `Tried to allocate 14.00 MiB`, because the activations of 4,000 forward sequences were
    retained for a backward pass nobody was going to run. The same shapes under no_grad peak
    at 3.46 GiB. eval/api_cloze.py:410 is the other caller and it does not wrap the call
    either -- it has not OOMed because its item count is smaller, which is the failure mode
    a per-caller obligation produces: it works until the caller that stresses it arrives.

    Per-item tensors, so a caller can compute a standard error, a per-stratum mean, or a
    paired difference. score_mc is the mean of this.

    Each item: {"prompt": str, "options": [str, ...], "label": int}.
    Prediction = argmax over options of sum log-prob(option tokens | prompt).

    SUMMED, NOT LENGTH-NORMALISED, and a caller building its own items has to know:
    options of unequal token length are not comparable here, because a longer option
    accumulates more negative log-prob. Every existing caller equalises length itself
    (lambada_zh requires len(distractor) == len(target); math_v2_like uses a
    same-token-length digit edit). A 4-way item whose options differ in token count is
    measuring length, not knowledge.

    All (item, option) pairs are flattened into jobs, sorted by total length
    (length bucketing minimizes padding), and scored `batch_size` sequences per
    forward pass. Right-padding is safe: causal attention never looks right, so
    pad tokens never affect logits at real positions.
    """
    if not items:
        # EMPTY TENSORS, not 0.0. This branch belonged to the mean and the split left it
        # behind, so score_mc_items([]) returned a float and the caller's unpack raised
        # TypeError. Caught by the equivalence test, not by reading: the two returns are 60
        # lines apart and each looked right in isolation.
        z = torch.zeros(0, dtype=torch.long)
        return z, z, torch.zeros(0, dtype=torch.bool)

    # Pre-tokenize once; flatten (item, option) pairs into jobs.
    # A FoNE checkpoint has only ever seen numbers as [NUM] carrying a value, so its
    # prompts go through fone; plain BPE puts numeric questions off-distribution.
    def enc(texts):
        if num_id is None:
            return [e.ids for e in tok.encode_batch(texts)], [[0.0] * 0 for _ in texts]
        return fone.encode_prompts(texts, tok, num_id)

    jobs = []  # (prompt_ids, prompt_vals, option_ids, option_vals, item_idx, option_idx)
    prompts, opts_flat, index = [], [], []
    for i, it in enumerate(items):
        prompts.append(it["prompt"])
        for j, opt in enumerate(it["options"]):
            opts_flat.append(opt)
            index.append((i, j))
    p_all, pv_all = enc(prompts)
    o_all, ov_all = enc(opts_flat)
    for k, (i, j) in enumerate(index):
        if o_all[k]:  # empty continuation would score 0 and corrupt argmax
            pv = pv_all[i] if num_id is not None else [0.0] * len(p_all[i])
            ov = ov_all[k] if num_id is not None else [0.0] * len(o_all[k])
            jobs.append((p_all[i], pv, o_all[k], ov, i, j))

    n_items = len(items)
    max_opts = max(len(it["options"]) for it in items)
    scores = torch.full((n_items, max_opts), -1e9, dtype=torch.float32, device=device)

    jobs.sort(key=lambda job: len(job[0]) + len(job[2]))

    for s in range(0, len(jobs), batch_size):
        chunk = jobs[s : s + batch_size]
        max_len = max(len(p) + len(o) for p, _, o, _, _, _ in chunk)
        B = len(chunk)
        x = torch.zeros((B, max_len), dtype=torch.long, device=device)
        v = torch.zeros((B, max_len), device=device) if num_id is not None else None
        rows, positions, targets = [], [], []
        for b, (p_ids, p_val, o_ids, o_val, _, _) in enumerate(chunk):
            full = p_ids + o_ids
            x[b, : len(full)] = torch.tensor(full, dtype=torch.long, device=device)
            if num_id is not None:
                v[b, : len(full)] = torch.tensor(p_val + o_val, device=device)
            pl = len(p_ids)
            for k, t in enumerate(o_ids):
                rows.append(b)
                positions.append(max(pl - 1 + k, 0))  # logit predicting token k of the option
                targets.append(t)

        logits = model(x, num_vals=v)[0]
        rows_t = torch.tensor(rows, device=device)
        pos_t = torch.tensor(positions, device=device)
        tgt_t = torch.tensor(targets, device=device)

        token_lp = logits[rows_t, pos_t].log_softmax(-1)
        token_lp = token_lp[torch.arange(len(tgt_t), device=device), tgt_t]
        job_scores = torch.zeros(B, dtype=torch.float32, device=device)
        job_scores.scatter_add_(0, rows_t, token_lp)  # sum token log-probs per job
        for b, (_, _, _, _, ii, jj) in enumerate(chunk):
            scores[ii, jj] = job_scores[b]

    preds = scores.argmax(dim=1).cpu()
    labels = torch.tensor([it["label"] for it in items], dtype=torch.long)
    # AN ITEM WHOSE OPTIONS ALL FAILED TO TOKENIZE IS NOT A CORRECT ANSWER. Its whole row of
    # `scores` stays at the -1e9 fill (:198), argmax returns the FIRST maximum, so preds is 0
    # and the item counts CORRECT whenever label == 0 -- a silent free point, biased toward
    # whichever slot the caller happens to put the gold in. Measured 2026-09-05 with a
    # tokenizer that returns [] for every option: preds [0,0] against labels [0,2] scored 1 of
    # 2 correct, and the one it "got right" was never scored at all.
    #
    # Returned as a mask rather than dropped, so the caller decides: score_mc excludes them
    # from the mean (an unscoreable item is not a wrong answer either), and a caller building
    # its own items can refuse outright. Silence is the one option that is wrong.
    scored = (scores > -1e9).any(dim=1).cpu()
    return preds, labels, scored


# --- GSM8K: the only generation benchmark (batched greedy decoding) ---

@torch.no_grad()
def run_gsm8k(model, tok, device, batch_size=GEN_BATCH, temperature=0.0, fmt=None):
    """Greedy generation, exact match on the last number. Reuses gsm8k.py's
    generator/extractor; prompts are pre-tokenized here before decoding.

    fmt is REQUIRED in practice: it comes from prompt_fn(classify(cfg, name)) at the call
    site, the one place holding the cfg. It has no default of format_prompt because that
    default is precisely the defect -- a base checkpoint scored in ChatML reads zero on an
    unseen prefix rather than on capability (AGENTS.md:200; 1.6% vs 94.4% fence rate,
    eval/score_code_exec.py:9-31). Passing None raises rather than guessing.
    """
    if fmt is None:
        raise ValueError("run_gsm8k needs fmt=prompt_fn(classify(cfg, name)); a default "
                         "would silently score a base checkpoint in ChatML")
    m = _load_module("gsm8k")
    rows = list(m.load_dataset())
    prompts = [tok.encode(fmt(r["question"])).ids for r in rows]
    golds = [
        float(r["answer"].split("####")[-1].replace(",", "").strip()) for r in rows
    ]

    correct = total = 0
    for s in range(0, len(rows), batch_size):
        out_ids = m.generate_batch(model, prompts[s : s + batch_size], GEN_MAX_NEW, device, temperature)
        for ids, gold in zip(out_ids, golds[s : s + batch_size]):
            pred = m.extract_number(tok.decode(ids))
            total += 1
            if pred is not None and abs(pred - gold) < 1e-4:
                correct += 1
    return correct / max(total, 1)


# --- Model / tokenizer loading ---

def load_model(ckpt_path, device):
    model, cfg = load_checkpoint(ckpt_path, device=device)
    return model.to(torch.bfloat16), cfg


def main():
    parser = argparse.ArgumentParser(description="Unified benchmark runner for HybridLM")
    parser.add_argument("--ckpt", required=True, help="checkpoint path, e.g. ckpt_sft.pt")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=f"subset to run (default: all). Choices: {', '.join(ALL_BENCHMARKS)}",
    )
    parser.add_argument("--batch", type=int, default=MC_BATCH, help="MC scoring batch size")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="GSM8K sampling temperature; 0 = greedy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer", default=TOK_PATH, help="vocabulary the checkpoint was trained on")
    args = parser.parse_args()

    names = args.benchmarks or ALL_BENCHMARKS
    bad = [n for n in names if n not in ALL_BENCHMARKS]
    if bad:
        parser.error(f"unknown benchmarks {bad}; choose from {ALL_BENCHMARKS}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU (much slower).")
        device = "cpu"
    else:
        device = args.device

    model, cfg = load_model(args.ckpt, device)
    tok = load_tokenizer(args.tokenizer, cfg)
    num_id = getattr(cfg, "num_id", None) if getattr(cfg, "fone", False) else None
    if num_id is not None:
        print(f"FoNE checkpoint: prompts encode numbers as [NUM] (id {num_id})")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {args.ckpt}: {n_params:.1f}M params, bf16, device={device}")

    results = {}  # display name -> (accuracy, seconds)
    for key in names:
        if key == "gsm8k":
            if num_id is not None:
                # This path decodes through tok, which prints the [NUM] token instead of
                # the number it stands for; eval/math_hard.py has the FoNE decode.
                print("  GSM8K: skipped on a FoNE checkpoint (use eval/math_hard.py)")
                continue
            display = f"GSM8K (t={args.temperature})"
            t0 = time.perf_counter()
            from score_matrix import classify

            _fmt = prompt_fn(classify(cfg, os.path.basename(args.ckpt)))
            print(f"  GSM8K prompt format: {_fmt.__name__}", flush=True)
            acc = run_gsm8k(model, tok, device, temperature=args.temperature, fmt=_fmt)
        else:
            display, loader = MC_BENCHMARKS[key]
            t0 = time.perf_counter()
            try:
                acc = score_mc(model, tok, loader(), device, args.batch, num_id)
            except Exception as e:
                # One benchmark's fetch must not take the rest of the suite with it: the pod
                # cannot reach the HF Hub, so hellaswag's ReadTimeout aborted run_eval.py
                # before piqa ran, and the caller saw a single "FAILED" for two missing sets.
                print(f"  {display}: SKIPPED ({type(e).__name__}: {str(e)[:90]})", flush=True)
                continue
        elapsed = time.perf_counter() - t0
        results[display] = (acc, elapsed)
        print(f"  {display}: {acc:.1%} ({elapsed:.1f}s)", flush=True)

    print()
    print(f"{'Benchmark':<16} {'Accuracy':>8}")
    print("─" * 24)
    for name, (acc, _) in results.items():
        print(f"{name:<16} {acc:>7.1%}")
    print("─" * 24)
    avg = sum(a for a, _ in results.values()) / len(results)
    print(f"{'Average':<16} {avg:>7.1%}")


if __name__ == "__main__":
    main()
