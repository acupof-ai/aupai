#!/usr/bin/env python3
"""Reproduce the deterministic seed-42 200-sample of filtered survivors used by the
residual hand-read, and count how many of those 200 answers contain an arithmetic
equation (P.match). Answers aupai-6e's question: is ~45% of the 200 actually eq-bearing?
Not a re-sample: same seed, same filtered file (prefer the subagent's /tmp copy if present,
else the repo's freshly-filtered file, which must be the same 499,423 survivors)."""
import json
import random
import re

P = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([+*\-/x×])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)")


def has_eq(t):
    return bool(P.search(t or ""))


def main():
    cands = ["/tmp/control_sft_filtered.jsonl",
             "/work/aupai/data/sft/control_sft_text_train_filtered.jsonl"]
    fn = next((f for f in cands if __import__("os").path.exists(f)), None)
    assert fn, "no survivor file"
    rows_ids = []          # row order -> id (the 200 were sampled by ROW INDEX)
    eq_hold = {}           # id -> has equation
    with open(fn, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows_ids.append(d.get("id"))
            eq_hold[d.get("id")] = has_eq(d.get("answer") or "")
    rng = random.Random(42)
    idx = rng.sample(range(len(rows_ids)), 200)
    sample_ids = [rows_ids[i] for i in idx]
    with_eq = sum(1 for i in sample_ids if eq_hold[i])
    print(f"survivor_file={fn} n_survivors={len(rows_ids)}")
    print(f"sample_n=200 eq_holding={with_eq} eq_share={with_eq/200:.1%}")
    print(f"ids(first10)={sample_ids[:10]}")


if __name__ == "__main__":
    main()
