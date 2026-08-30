#!/usr/bin/env python3
"""math-hard v2 likelihood twin (panel metric 4).

Scores the ANSWER SPAN conditioned on the solution prefix: prompt = instruction
+ output up to and including "\\boxed{", options = [gold, wrong]. Both options
sit at the same span position, so length is controlled by construction; the
remaining length confound is tokenization (number BPE), handled by a per-pair
token-count assert -- pairs that fail are skipped and counted, never scored.

Wrong-answer construction: one deterministic digit edit on the answer's last
digit (d -> (d+1) % 10). Same character length for every v2 answer shape
(plain int, "8秒,位置30", "x=-3,y=-1", "(x-12)(x+12)"), and an off-by-one
digit is a plausible arithmetic error, not garbage (panel: wrong answers must
be plausible). Gold correctness is guaranteed by construction-from-answer
upstream.

Floor 50% (2-way). Known-answer: --swap inverts option order; the gold content
must keep winning (position-bias control), and the normal-vs-swapped spread on
the strongest checkpoint must be >= 60pt (panel freeze rule).

Usage:
    python eval/math_v2_like.py --ckpt ckpt_0830v1_3.24b.pt.ep1 --out /tmp/m.json
    python eval/math_v2_like.py --ckpt ... --swap
    python eval/math_v2_like.py --selftest
"""

import argparse
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["FLA_FLASH_KDA"] = "0"

from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
V2_PATH = os.path.join(ROOT, "data", "synthetic", "math_hard_eval_v2_1k.jsonl")

_DIGIT = re.compile(r"\d")


def make_wrong(answer):
    """One deterministic digit edit on the last digit: d -> (d+1) % 10.
    Returns None if the answer has no digit (nothing plausible to perturb)."""
    m = None
    for m in _DIGIT.finditer(answer):
        pass
    if m is None:
        return None
    d = int(m.group())
    return answer[: m.start()] + str((d + 1) % 10) + answer[m.end() :]


def build_items(tok, path=V2_PATH):
    """Return (items, skipped) where items are score_mc-ready with label 0.

    skipped counts: no_digit (no digit to perturb), tok_len (gold and wrong
    tokenize to different lengths -- the number-BPE confound the panel bans),
    no_boxed (output has no \\boxed{ span)."""
    items, skipped = [], {"no_digit": 0, "tok_len": 0, "no_boxed": 0}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        out = d["output"]
        marker = "\\boxed{"
        i = out.rfind(marker)
        if i < 0 or not out.rstrip().endswith("}"):
            skipped["no_boxed"] += 1
            continue
        gold = out[i + len(marker) : out.rstrip().rfind("}")]
        wrong = make_wrong(gold)
        if wrong is None:
            skipped["no_digit"] += 1
            continue
        g_ids = tok.encode(gold, add_special_tokens=False).ids
        w_ids = tok.encode(wrong, add_special_tokens=False).ids
        if len(g_ids) != len(w_ids):
            skipped["tok_len"] += 1
            continue
        items.append({
            "prompt": d["instruction"].rstrip() + "\n" + out[: i + len(marker)],
            "options": [gold, wrong],
            "label": 0,
            "family": d.get("type", "unknown"),
        })
    return items, skipped


def score_detailed(model, tok, cfg, items, device, swap=False, batch=32):
    """Per-family accuracy + overall. Floor 50% by construction."""
    from eval.run_eval import score_mc
    num_id = getattr(cfg, "num_id", None) if getattr(cfg, "fone", False) else None
    mc_items = [
        {"prompt": it["prompt"],
         "options": [it["options"][1], it["options"][0]] if swap else it["options"],
         "label": 1 if swap else 0}
        for it in items
    ]
    # score_mc returns a scalar; per-family needs separate calls (18 families,
    # each one batched pass -- cheap).
    by_family, correct = {}, 0
    fam_idx = {}
    for k, it in enumerate(items):
        fam_idx.setdefault(it["family"], []).append(k)
    for fam, idxs in fam_idx.items():
        acc = score_mc(model, tok, [mc_items[k] for k in idxs], device,
                       batch_size=batch, num_id=num_id)
        by_family[fam] = {"acc": acc, "n": len(idxs)}
        correct += round(acc * len(idxs))
    return correct / len(items), by_family


def _selftest():
    """Construction invariants (char-level stub tokenizer) + plumbing."""
    import eval.math_v2_like as m  # noqa: F401
    cases = {
        "14": "15", "6": "7", "9": "0", "180": "181",
        "8秒,位置30": "8秒,位置31", "x=-3,y=-1": "x=-3,y=-2",
        "k=-5,b=4": "k=-5,b=5", "(x-12)(x+12)": "(x-12)(x+13)",
        "x₁=-8,x₂=-1": "x₁=-8,x₂=-2",
    }
    for gold, want in cases.items():
        got = make_wrong(gold)
        assert got == want, (gold, got, want)
        assert len(got) == len(gold), gold
        assert got != gold
    assert make_wrong("无数字") is None
    print("math_v2_like self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--data", default=V2_PATH)
    ap.add_argument("--out")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return
    if not a.ckpt:
        ap.error("--ckpt required (or --selftest)")

    model, cfg = load_checkpoint(a.ckpt, device=a.device, dtype=torch.bfloat16)
    tok = load_tokenizer(TOK_PATH, cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {a.ckpt}: {n_params:.1f}M params, device={a.device}", flush=True)

    items, skipped = build_items(tok, a.data)
    print(f"items: {len(items)} usable, skipped {skipped}", flush=True)
    if not items:
        ap.error("no usable items after token-length alignment")

    overall, by_family = score_detailed(model, tok, cfg, items, a.device, swap=a.swap)
    result = {
        "ckpt": a.ckpt, "n_params": n_params, "swap": a.swap,
        "n_items": len(items), "skipped": skipped,
        "overall": overall,
        "families": dict(sorted(by_family.items())),
    }
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
