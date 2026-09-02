#!/usr/bin/env python3
# restartable: writes only the output checkpoint, at the end. An interrupt costs the run.
# One card, no DDP -- 212M params at seq 1024 fits, and a single-card run has no rank-skew
# failure mode to debug on a control arm.
"""SFT a HuggingFace causal LM on the shared control text pack. One card, ~1h.

    python3 scripts/sft_hf_control.py \
        --model data/controls/pythia-160m-step2000 \
        --pack data/sft/control_sft_text.jsonl \
        --out ckpt_sft_pythia160m_control.pt
    python3 scripts/sft_hf_control.py --selftest      # no card, no model needed

WHY THIS FILE EXISTS. sft_math.py reads our own checkpoint format and builds our own model;
Pythia-160m is a HF GPTNeoXForCausalLM. Rather than teach sft_math.py a second model
family, this reproduces the parts of it that must be IDENTICAL for the comparison to mean
anything, and states plainly the parts that cannot be.

IDENTICAL to our arm, by construction:
  the example set and its order   -- the same control_sft_text.jsonl, same sha256
  the ChatML template             -- scripts.loader.format_example, imported not copied
  the loss mask                   -- prompt masked to -100, completion supervised
                                     including its <|im_end|> (the stop token must be
                                     supervised or the model never stops)
  the packing rule                -- whole examples only, never split, over-length dropped,
                                     right-padded with eos carrying no loss
  the LR schedule SHAPE           -- train.lr_mult, imported: absolute warmup then a cosine
                                     warmdown over the last cfg.warmdown of the run
  epochs                          -- default 1, the sft_math.py default

NECESSARILY DIFFERENT, and this belongs in the report header rather than a footnote:
  the tokenizer   ours vs a 50,304-entry NeoX BPE. The same TEXT yields a different token
                  count per side; both numbers get reported. A shared .pt is impossible and
                  check_sft_ready.py:check_vocab would correctly refuse one.
  the optimizer   our arm uses Muon (lr 0.01) on 2D matrices with AdamW on embeddings and
                  scalars. Muon on a foreign model is a second intervention on top of the
                  data, so this arm uses AdamW throughout. Any claim from this comparison
                  must therefore be about data and architecture TOGETHER, not either alone.
  the LR          AND IT IS NOT DERIVABLE FROM OURS, which the first version of this file got
                  wrong. It set 1e-2 = Cfg.embed_lr (0.1) x sft_math --lr_scale (0.1) and
                  called that "the same magnitude". Two errors in one: Cfg.embed_lr applies
                  to the EMBEDDING GROUP ONLY (Muon carries the 2D matrices at 0.01, and
                  Muon's update is orthogonalised, so its lr is not on the same scale as
                  AdamW's at all), and a per-group lr from our optimizer says nothing about a
                  foreign model's every-parameter lr. Measured on a CPU smoke test over 4
                  steps: 1e-2 DIVERGES (loss 11.52 -> 13.27), 1e-4 descends (-> 2.49), 2e-5
                  is too slow (-> 7.60). The default is therefore 1e-4, chosen by measurement
                  on this model, and the report must say the arms' LRs were each set for their
                  own optimizer rather than matched.
  the ChatML tokens  ours are single special tokens; the NeoX BPE has no <|im_start|>, so
                  they tokenize as several ordinary tokens. Added to the tokenizer as real
                  special tokens here so the two sides both learn a dedicated stop symbol --
                  the alternative, letting them split, would handicap the control for a
                  reason unrelated to the question. NO resize: Pythia's config.vocab_size
                  (50,304) already exceeds its tokenizer (50,277), so the two ids land on
                  pretrained-but-unaddressed rows. See the resize comment in main().
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from loader import IM_END, format_example  # noqa: E402

SEQ = 1024  # Pythia-160m's max_position_embeddings is 2048; 1024 halves the step cost and
#: still holds the overwhelming majority of these examples whole. Reported, not silent:
#: the drop count from this choice is printed and lands in the report.


def lr_mult_shape(step, total, warmup, warmdown, final_frac):
    """train.lr_mult's schedule, re-expressed with explicit numbers.

    train.py imports torch and builds a model at module scope, so importing it on a
    tokenizer-only selftest is not free; the function is 8 lines and its parameters are
    named here so the two schedules can be asserted equal (see selftest).
    """
    if step < warmup:
        return (step + 1) / warmup
    wd_steps = max(1, int(warmdown * total))
    wd_start = total - wd_steps
    if step < wd_start:
        return 1.0
    progress = min(1.0, (step - wd_start) / wd_steps)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def read_pack(path, holdout_every=0):
    """The shared pack -> [(prompt, completion)] through OUR template.

    holdout_every=N reserves every Nth EXAMPLE as a validation split and returns
    (train, val). The split is by example index in the shared text, not by packed row: row
    boundaries depend on the tokenizer, so a row-level split would put different content in
    each arm's validation set and the two numbers would not be comparable.

    An lr scan MUST be selected on held-out loss. The training loss of the last step rewards
    whichever lr memorised the most, which is precisely the artefact the scan exists to rule
    out ("we won because the control was undertuned" would just become "the control's lr was
    picked to overfit").
    """
    train, val = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pair = format_example(d["question"], d["answer"])
            (val if holdout_every and i % holdout_every == 0 else train).append(pair)
    return (train, val) if holdout_every else (train, [])


def pack_rows(pairs, tok, eos_id, seq):
    """Greedy pack into (seq+1)-token rows, prompt masked. Mirrors prepare_sft.pack_and_save.

    Returns (input_ids, labels, stats). One example never spans two rows; an example longer
    than a row is DROPPED, not truncated -- a truncated example's completion has no prompt
    for the mask to key on.
    """
    import torch

    row_len = seq + 1
    rows_i, rows_l = [], []
    cur_i, cur_l = [], []
    n_drop = n_pad = 0

    def flush():
        nonlocal cur_i, cur_l, n_pad
        if not cur_i:
            return
        pad = row_len - len(cur_i)
        n_pad += pad
        rows_i.append(cur_i + [eos_id] * pad)
        rows_l.append(cur_l + [-100] * pad)
        cur_i, cur_l = [], []

    # Batch-encode: one call per side beats 2N calls on a 500k-row pack.
    p_ids = tok([p for p, _ in pairs], add_special_tokens=False)["input_ids"]
    c_ids = tok([c for _, c in pairs], add_special_tokens=False)["input_ids"]
    for pi, ci in zip(p_ids, c_ids, strict=True):  # one per pair on both sides
        need = len(pi) + len(ci)
        if need > row_len:
            n_drop += 1
            continue
        if len(cur_i) + need > row_len:
            flush()
        cur_i += pi + ci
        cur_l += [-100] * len(pi) + list(ci)
    flush()
    stats = {"rows": len(rows_i), "dropped_overlong": n_drop, "pad_tokens": n_pad,
             "supervised": sum(1 for r in rows_l for v in r if v != -100),
             "total": len(rows_i) * row_len}
    return (torch.tensor(rows_i, dtype=torch.long),
            torch.tensor(rows_l, dtype=torch.long), stats)


def eval_loss(model, ids, lab, batch, device):
    """Mean loss over the validation rows. The number an lr scan is selected on."""
    import torch

    model.eval()
    tot = n = 0.0
    with torch.no_grad():
        for lo in range(0, ids.shape[0], batch):
            bi = ids[lo:lo + batch].to(device)
            bl = lab[lo:lo + batch].to(device)
            out = model(input_ids=bi[:, :-1], labels=bl[:, 1:])
            tot += out.loss.item() * bi.shape[0]
            n += bi.shape[0]
    model.train()
    return tot / max(n, 1)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="data/controls/pythia-160m-step2000")
    ap.add_argument("--pack", default="data/sft/control_sft_text.jsonl")
    ap.add_argument("--out", default="ckpt_sft_pythia160m_control.pt")
    ap.add_argument("--epochs", type=int, default=1, help="sft_math.py's default")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=6, help="effective batch = batch * accum")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="AdamW lr. NOT derived from our Cfg -- see the LR note in the module "
                         "docstring. 1e-4 is where a CPU smoke test on this model actually "
                         "descends (11.52 -> 2.49 over 4 steps); 0.01 DIVERGES (-> 13.27)")
    ap.add_argument("--warmup", type=int, default=20, help="train.Cfg.warmup, absolute steps")
    ap.add_argument("--warmdown", type=float, default=0.65, help="train.Cfg.warmdown")
    ap.add_argument("--final_lr_frac", type=float, default=0.05, help="train.Cfg.final_lr_frac")
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke tests only")
    ap.add_argument("--device", default="cuda",
                    help="'cpu' runs the real forward/backward without a card -- the only way "
                         "to prove this loop executes before it holds one")
    ap.add_argument("--holdout_every", type=int, default=50,
                    help="reserve every Nth EXAMPLE for validation (default 50 = 2%%). The lr "
                         "scan is selected on this loss, never on training loss")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = a.model if os.path.isabs(a.model) else os.path.join(ROOT, a.model)
    pack_path = a.pack if os.path.isabs(a.pack) else os.path.join(ROOT, a.pack)
    for p in (model_dir, pack_path):
        if not os.path.exists(p):
            print(f"CANNOT RUN: {p} does not exist")
            return 2

    pack_sha = sha256_of(pack_path)
    print(f"pack {pack_path}\n  sha256 {pack_sha}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    # Our ChatML markers are single special tokens on our side; give this side dedicated
    # tokens too, so neither arm is handicapped on the stop symbol.
    vocab_before = len(tok)
    added = tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", IM_END]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # The ids the tokenizer assigned, recorded per fb's ruling: an id assigned at run time is
    # not reconstructable from the config afterwards, and a scorer that re-tokenizes with a
    # freshly built tokenizer would get different ones without any error.
    chatml_ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|im_start|>", IM_END)}
    print(f"tokenizer {vocab_before} -> {len(tok)} entries (+{added} ChatML specials) "
          f"{chatml_ids}", flush=True)

    pairs, val_pairs = read_pack(pack_path, holdout_every=a.holdout_every)
    ids, lab, st = pack_rows(pairs, tok, tok.eos_token_id, a.seq)
    v_ids, v_lab, v_st = (pack_rows(val_pairs, tok, tok.eos_token_id, a.seq)
                          if val_pairs else (None, None, {}))
    if val_pairs:
        print(f"validation {len(val_pairs):,} examples -> {v_st['rows']:,} rows "
              f"(every {a.holdout_every}th example, held out of training)", flush=True)
    tokens = st["rows"] * (a.seq + 1)
    print(f"examples {len(pairs):,} -> rows {st['rows']:,} x {a.seq+1} = {tokens:,} tokens\n"
          f"  dropped over-length {st['dropped_overlong']:,}   pad {st['pad_tokens']:,}\n"
          f"  supervised {st['supervised']:,} of {st['total']:,} "
          f"({100.0*st['supervised']/max(st['total'],1):.1f}%)", flush=True)
    if st["supervised"] == 0 or st["supervised"] == st["total"]:
        print("REFUSING: the mask supervises nothing or everything")
        return 1

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    # NO resize. Pythia's config.vocab_size is 50,304 (padded for kernel alignment) while
    # its tokenizer holds 50,277 entries, so the two ChatML ids land at 50,277/50,278 --
    # INSIDE the pretrained embedding, on rows that were allocated and trained but never
    # addressed by a token. resize_token_embeddings(len(tok)) would resize DOWN to 50,279
    # and throw away 25 trained rows: verified on the pod, 50,304 -> 50,279. The rows the
    # markers land on are randomly-initialised in effect (no token ever mapped to them), so
    # this is what the resize was meant to achieve, minus the truncation.
    emb_rows = model.get_input_embeddings().weight.shape[0]
    max_id = max(chatml_ids.values())
    if max_id >= emb_rows:
        print(f"CANNOT RUN: ChatML id {max_id} is outside the embedding ({emb_rows} rows). "
              f"This model needs a resize UP -- add it deliberately rather than letting "
              f"resize_token_embeddings(len(tok)) also truncate.")
        return 2
    print(f"embedding {emb_rows} rows, ChatML ids at {sorted(chatml_ids.values())} -- inside, "
          f"no resize (a resize to len(tok)={len(tok)} would DISCARD "
          f"{emb_rows - len(tok)} trained rows)", flush=True)
    model.gradient_checkpointing_enable()
    model.to(a.device).train()
    # fused AdamW is a CUDA-only kernel; the cpu path exists to smoke-test this loop.
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.01, fused=(a.device == "cuda"))
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]

    n = ids.shape[0]
    per_epoch = n // (a.batch * a.accum)
    total = per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    print(f"steps {total} ({per_epoch}/epoch x {a.epochs}), effective batch "
          f"{a.batch*a.accum} rows", flush=True)
    if total < 1:
        print("REFUSING: fewer than one optimizer step -- batch*accum exceeds the pack")
        return 1

    # Same order both arms see: prepare_sft shuffles with seed 0 at pack time, and this pack
    # is already in that order, so read it straight through.
    # Measured BEFORE any step: an lr scan compares deltas, and a scan whose arms started
    # from different baselines would compare two different quantities.
    val_before = eval_loss(model, v_ids, v_lab, a.batch, a.device) if val_pairs else None
    if val_before is not None:
        print(f"held-out loss before training {val_before:.4f}", flush=True)

    t0 = time.time()
    step = 0
    losses = []
    for _ep in range(a.epochs):
        for s in range(per_epoch):
            if step >= total:
                break
            m = lr_mult_shape(step, total, a.warmup, a.warmdown, a.final_lr_frac)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * m
            opt.zero_grad(set_to_none=True)
            acc_loss = 0.0
            for k in range(a.accum):
                lo = (s * a.accum + k) * a.batch
                bi = ids[lo:lo + a.batch].to(a.device)
                bl = lab[lo:lo + a.batch].to(a.device)
                # train.py's convention: predict token t+1 from tokens <= t.
                out = model(input_ids=bi[:, :-1], labels=bl[:, 1:])
                (out.loss / a.accum).backward()
                acc_loss += out.loss.item() / a.accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(acc_loss)
            step += 1
            if step % 20 == 0 or step == 1:
                el = time.time() - t0
                print(f"  step {step}/{total}  loss {acc_loss:.4f}  lr {opt.param_groups[0]['lr']:.2e}"
                      f"  {el:.0f}s  eta {el/step*(total-step)/60:.0f}min", flush=True)

    val_after = eval_loss(model, v_ids, v_lab, a.batch, a.device) if val_pairs else None
    if val_after is not None:
        print(f"\nheld-out loss {val_before:.4f} -> {val_after:.4f} "
              f"(this is the number an lr scan is selected on)", flush=True)

    out_path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    model.save_pretrained(out_path + ".hf")
    tok.save_pretrained(out_path + ".hf")
    meta = {
        "arm": "control", "base": a.model, "base_revision_note":
            "step2000 inferred from the download URL path; the config carries no step field",
        "pack": os.path.relpath(pack_path, ROOT), "pack_sha256": pack_sha,
        "examples": len(pairs), "rows": st["rows"], "seq": a.seq, "tokens": tokens,
        "dropped_overlong": st["dropped_overlong"],
        "vocab_before": vocab_before, "vocab_after": len(tok),
        "chatml_token_ids": chatml_ids,
        "embedding_rows": emb_rows,
        "resized": False,
        "resize_note": "config.vocab_size 50304 > tokenizer 50277, so the ChatML ids land "
                       "inside the pretrained embedding; resize_token_embeddings(len(tok)) "
                       "would have shrunk it to 50279 and discarded 25 trained rows",
        "supervised_frac": round(st["supervised"] / max(st["total"], 1), 4),
        "steps": step, "epochs": a.epochs, "effective_batch": a.batch * a.accum,
        "optimizer": "AdamW (our arm uses Muon on 2D matrices -- NOT the same optimizer)",
        "lr": a.lr, "lr_note": "set by CPU smoke test on this model, NOT matched to our arm -- 1e-2 diverges here", "warmup": a.warmup, "warmdown": a.warmdown,
        "final_lr_frac": a.final_lr_frac,
        "final_train_loss": losses[-1] if losses else None,
        "first_train_loss": losses[0] if losses else None,
        # The selection number. Training loss is deliberately NOT it: selecting an lr on
        # training loss picks whichever lr memorised hardest, which is the artefact the scan
        # exists to rule out.
        "held_out_loss": val_after,
        "held_out_loss_before": val_before,
        "held_out_examples": len(val_pairs),
        "holdout_every": a.holdout_every,
        "wall_s": round(time.time() - t0),
    }
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {out_path}.hf  meta {out_path}.meta.json")
    print(json.dumps(meta, indent=2))
    return 0


def selftest():
    """The three things here that could be wrong without any error appearing.

    No card, no model download: a fake tokenizer stands in, because what is being checked is
    the mask, the packing and the schedule -- none of which involve weights.
    """
    fails = []

    # 1. the LR schedule must equal train.lr_mult exactly, since the whole point of
    #    re-expressing it is that the two arms share a shape.
    try:
        sys.path.insert(0, ROOT)
        from train import Cfg, lr_mult
        for total in (100, 733):
            for step in (0, 1, 19, 20, 21, total // 2, total - 2, total - 1, total, total + 5):
                mine = lr_mult_shape(step, total, Cfg.warmup, Cfg.warmdown, Cfg.final_lr_frac)
                theirs = lr_mult(step, total, Cfg)
                if abs(mine - theirs) > 1e-12:
                    fails.append(f"schedule differs at step {step}/{total}: {mine} vs {theirs}")
                    break
    except Exception as e:  # noqa: BLE001
        fails.append(f"could not compare against train.lr_mult: {type(e).__name__}: {e} "
                     f"-- the shape claim is then UNVERIFIED, not verified")

    # 2. the mask: prompt tokens -100, completion tokens supervised and equal to the input.
    try:
        import torch  # noqa: F401

        class FakeTok:
            """One id per whitespace token; ids are stable per word so labels are checkable."""
            def __init__(self):
                self.vocab = {}

            def __call__(self, texts, add_special_tokens=False):
                out = []
                for t in texts:
                    ids = []
                    for w in t.split():
                        ids.append(self.vocab.setdefault(w, len(self.vocab) + 10))
                    out.append(ids)
                return {"input_ids": out}

        ft = FakeTok()
        pairs = [("p1 p2 p3", "c1 c2"), ("q1", "d1 d2 d3")]
        ids, lab, st = pack_rows(pairs, ft, 0, 16)
        sup = (lab != -100)
        if not bool((lab[sup] == ids[sup]).all()):
            fails.append("a supervised label differs from its input token")
        # exactly the completion tokens are supervised: 2 + 3
        if int(sup.sum()) != 5:
            fails.append(f"expected 5 supervised tokens, got {int(sup.sum())}")
        # and the first prompt token of row 0 is masked
        if lab[0, 0] != -100:
            fails.append("the first prompt token is supervised")
        # an over-length example is dropped, never truncated
        _, _, st2 = pack_rows([("x " * 40, "y")], ft, 0, 8)
        if st2["dropped_overlong"] != 1 or st2["rows"] != 0:
            fails.append(f"over-length example not dropped: {st2}")
    except Exception as e:  # noqa: BLE001
        fails.append(f"mask check could not run: {type(e).__name__}: {e}")

    # 3. our template must actually put the stop token in the SUPERVISED half.
    p, c = format_example("q", "a")
    if not c.endswith(IM_END):
        fails.append(f"the completion does not end with the stop token: {c!r}")
    if IM_END in p.split("<|im_start|>assistant")[-1]:
        fails.append("the prompt's trailing assistant marker is followed by a stop token")

    # 4. the held-out split must be disjoint from training, deterministic, and by EXAMPLE.
    #    A scan selected on a validation set that leaked into training measures memorisation,
    #    which is the whole reason the scan exists.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        pp = os.path.join(d, "p.jsonl")
        with open(pp, "w", encoding="utf-8") as f:
            for i in range(20):
                f.write(json.dumps({"question": f"q{i}", "answer": f"a{i}"}) + "\n")
        tr, va = read_pack(pp, holdout_every=5)
        if len(tr) + len(va) != 20:
            fails.append(f"split lost or duplicated examples: {len(tr)} + {len(va)} != 20")
        if len(va) != 4:
            fails.append(f"expected 4 held-out of 20 at every-5th, got {len(va)}")
        if set(tr) & set(va):
            fails.append("the held-out split overlaps training")
        if read_pack(pp, holdout_every=5) != (tr, va):
            fails.append("the split is not deterministic across calls")
        tr0, va0 = read_pack(pp)
        if va0 or len(tr0) != 20:
            fails.append(f"holdout_every=0 must hold nothing back: {len(tr0)}, {len(va0)}")

    for f in fails:
        print(f"  SELFTEST FAIL {f}")
    if fails:
        print(f"\n{len(fails)} selftest failure(s)")
        return 1
    print("sft_hf_control selftest OK (schedule == train.lr_mult, mask exact, stop supervised, "
          "held-out split disjoint and deterministic)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
