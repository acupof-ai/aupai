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
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# FlashKDA CUTLASS kernel is unavailable in some envs; train.py reads this at import.
os.environ["FLA_FLASH_KDA"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from train import HybridLM  # noqa: E402

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|think|>", "<|/think|>"]
LETTERS = ["A", "B", "C", "D"]
MC_BATCH = 32       # sequences per forward pass for log-likelihood scoring
GEN_BATCH = 16      # prompts per batch for GSM8K greedy generation
GEN_MAX_NEW = 256


# --- Dataset adapters: normalize every benchmark to {"prompt", "options", "label"} ---
# Benchmark modules are imported lazily so a broken optional dependency in one
# module doesn't kill the whole run.

def _load_module(name):
    return import_module(f"eval.{name}")


def load_hellaswag():
    m = _load_module("hellaswag")
    return [
        {"prompt": d["context"], "options": d["options"], "label": d["label"]}
        for d in m.load_dataset()
    ]


def load_winogrande():
    m = _load_module("winogrande")
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
    "hellaswag": ("HellaSwag", load_hellaswag),
    "piqa": ("PIQA", load_piqa),
    "arc-easy": ("ARC-Easy", lambda: _load_arc("ARC-Easy")),
    "arc-challenge": ("ARC-Challenge", lambda: _load_arc("ARC-Challenge")),
    "winogrande": ("WinoGrande", load_winogrande),
    "boolq": ("BoolQ", load_boolq),
    "openbookqa": ("OpenBookQA", load_openbookqa),
    "mmlu": ("MMLU", load_mmlu),
}
ALL_BENCHMARKS = list(MC_BENCHMARKS) + ["gsm8k"]


# --- Batched log-likelihood scorer (shared by all multiple-choice benchmarks) ---

@torch.no_grad()
def score_mc(model, tok, items, device, batch_size=MC_BATCH):
    """Score multiple-choice items by continuation log-likelihood.

    Each item: {"prompt": str, "options": [str, ...], "label": int}.
    Prediction = argmax over options of sum log-prob(option tokens | prompt).

    All (item, option) pairs are flattened into jobs, sorted by total length
    (length bucketing minimizes padding), and scored `batch_size` sequences per
    forward pass. Right-padding is safe: causal attention never looks right, so
    pad tokens never affect logits at real positions.
    """
    if not items:
        return 0.0

    # Pre-tokenize once; flatten (item, option) pairs into jobs.
    jobs = []  # (prompt_ids, option_ids, item_idx, option_idx)
    for i, it in enumerate(items):
        p_ids = tok.encode(it["prompt"]).ids
        for j, opt in enumerate(it["options"]):
            o_ids = tok.encode(opt).ids
            if o_ids:  # empty continuation would score 0 and corrupt argmax
                jobs.append((p_ids, o_ids, i, j))

    n_items = len(items)
    max_opts = max(len(it["options"]) for it in items)
    scores = torch.full((n_items, max_opts), -1e9, dtype=torch.float32, device=device)

    jobs.sort(key=lambda job: len(job[0]) + len(job[1]))  # length-bucketed batches

    for s in range(0, len(jobs), batch_size):
        chunk = jobs[s : s + batch_size]
        max_len = max(len(p) + len(o) for p, o, _, _ in chunk)
        B = len(chunk)
        x = torch.zeros((B, max_len), dtype=torch.long, device=device)
        rows, positions, targets, job_ids = [], [], [], []
        for b, (p_ids, o_ids, _, _) in enumerate(chunk):
            full = p_ids + o_ids
            x[b, : len(full)] = torch.tensor(full, dtype=torch.long, device=device)
            pl = len(p_ids)
            for k, t in enumerate(o_ids):
                rows.append(b)
                positions.append(max(pl - 1 + k, 0))  # logit predicting token k of the option
                targets.append(t)
                job_ids.append(b)

        logits = model(x)[0]  # (B, T, V) float32, softcapped
        rows_t = torch.tensor(rows, device=device)
        pos_t = torch.tensor(positions, device=device)
        tgt_t = torch.tensor(targets, device=device)
        job_t = torch.tensor(job_ids, device=device)

        token_lp = logits[rows_t, pos_t].log_softmax(-1)  # (N_tokens, V)
        token_lp = token_lp[torch.arange(len(tgt_t), device=device), tgt_t]
        job_scores = torch.zeros(B, dtype=torch.float32, device=device)
        job_scores.scatter_add_(0, job_t, token_lp)  # sum token log-probs per job
        for b, (_, _, ii, jj) in enumerate(chunk):
            scores[ii, jj] = job_scores[b]

    preds = scores.argmax(dim=1).cpu()
    labels = torch.tensor([it["label"] for it in items], dtype=torch.long)
    return (preds == labels).float().mean().item()


# --- GSM8K: the only generation benchmark (batched greedy decoding) ---

@torch.no_grad()
def run_gsm8k(model, tok, device, batch_size=GEN_BATCH):
    """Greedy generation, exact match on the last number. Reuses gsm8k.py's
    generator/extractor; prompts are pre-tokenized here before decoding."""
    m = _load_module("gsm8k")
    rows = list(m.load_dataset())
    prompts = [tok.encode(f"问：{r['question']}\n答：").ids for r in rows]
    golds = [
        float(r["answer"].split("####")[-1].replace(",", "").strip()) for r in rows
    ]

    correct = total = 0
    for s in range(0, len(rows), batch_size):
        out_ids = m.generate_batch(model, prompts[s : s + batch_size], GEN_MAX_NEW, device)
        for ids, gold in zip(out_ids, golds[s : s + batch_size]):
            pred = m.extract_number(tok.decode(ids))
            total += 1
            if pred is not None and abs(pred - gold) < 1e-4:
                correct += 1
    return correct / max(total, 1)


# --- Model / tokenizer loading ---

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    cfg.grad_ckpt = False
    model = HybridLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model = model.to(torch.bfloat16)
    model.eval()
    return model


def load_tokenizer():
    tok = Tokenizer.from_file(TOK_PATH)
    for t in SPECIAL_TOKENS:
        if tok.token_to_id(t) is None:
            tok.add_special_tokens([t])
    return tok


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
    parser.add_argument("--device", default="cuda")
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

    model = load_model(args.ckpt, device)
    tok = load_tokenizer()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {args.ckpt}: {n_params:.1f}M params, bf16, device={device}")

    results = {}  # display name -> (accuracy, seconds)
    for key in names:
        if key == "gsm8k":
            display = "GSM8K"
            t0 = time.perf_counter()
            acc = run_gsm8k(model, tok, device)
        else:
            display, loader = MC_BENCHMARKS[key]
            t0 = time.perf_counter()
            acc = score_mc(model, tok, loader(), device, args.batch)
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
