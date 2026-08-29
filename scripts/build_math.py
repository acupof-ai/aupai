#!/usr/bin/env python3
"""Merge every math source into one verified corpus, with per-source caps.

Caps matter: sft_v3 regressed to 26.8% because Ape210K's one-line `列式：a/b = c`
rows were 32% of the mix and collapsed multi-step reasoning (EXPERIMENTS.md).

Every row must pass: a numeric \\boxed{} answer, no wrong two-operand arithmetic
step (scripts/eqcheck.py — chained expressions are skipped, not scored), not a
holdout question (scripts/holdout.py), and not a duplicate question.

  python scripts/build_math.py [--out data/math/math_all.jsonl]
"""

import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from algorithms.rlvr_reward import extract_boxed, normalize_answer, to_number  # noqa: E402
from eqcheck import check_steps  # noqa: E402
from holdout import is_holdout  # noqa: E402

D = os.path.join(ROOT, "data")
# (path, cap) — cap None keeps everything. Multi-step Chinese sources dominate;
# single-line arithmetic drills are capped at ~10% of the mix.
SOURCES = [
    (f"{D}/math/belle.jsonl", None),
    (f"{D}/math/mxode.jsonl", None),
    (f"{D}/synthetic/math_short_v2.jsonl", None),
    (f"{D}/math/math23k.jsonl", None),
    (f"{D}/math/gsm8k_zh.jsonl", None),
    (f"{D}/synthetic/math_short_v1.jsonl", None),
    (f"{D}/workbatch/school_math_short.jsonl", None),
    (f"{D}/math/ape210k.jsonl", 40_000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/math/math_all.jsonl")
    ap.add_argument("--sources", help="comma-separated paths, overrides the default list")
    a = ap.parse_args()
    sources = [(p, None) for p in a.sources.split(",")] if a.sources else SOURCES

    random.seed(42)
    seen = set()
    total = drop_hold = drop_dup = drop_ans = drop_eq = 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as out:
        for path, cap in sources:
            if not os.path.exists(path):
                print(f"  missing (skipped): {path}")
                continue
            lines = open(path, encoding="utf-8").readlines()
            if cap:
                random.shuffle(lines)
            kept = 0
            for line in lines:
                if cap and kept >= cap:
                    break
                r = json.loads(line)
                q, ans = r["instruction"].strip(), r["output"].strip()
                if is_holdout(q):
                    drop_hold += 1
                    continue
                key = hashlib.md5(q.encode()).digest()
                if key in seen:
                    drop_dup += 1
                    continue
                if to_number(normalize_answer(extract_boxed(ans))) is None:
                    drop_ans += 1
                    continue
                if check_steps(ans)[1]:
                    drop_eq += 1
                    continue
                seen.add(key)
                out.write(json.dumps({"instruction": q, "output": ans}, ensure_ascii=False) + "\n")
                kept += 1
            print(f"  {os.path.basename(path)[:-6]:<22} kept {kept:>7}" + (f"  (cap {cap})" if cap else ""))
            total += kept
    print(
        f"TOTAL {total}  |  dropped: holdout {drop_hold}, dup {drop_dup}, "
        f"no-numeric-answer {drop_ans}, bad-arithmetic {drop_eq}"
    )
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
