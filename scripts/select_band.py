#!/usr/bin/env python3
"""Pick the RL prompt set from measured per-instance solve rates.

RL only learns where the group disagrees: a prompt the model always gets right and one it always
gets wrong both give every sample the same reward, so the advantage is zero and the step is wasted.
The previous run spent half its compute that way -- 30-55% of groups came back all-right or
all-wrong. The fix is to train on the prompts this model solves *sometimes*.

    python scripts/select_band.py data/rl/rl_band.jsonl [--min 800]
"""

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATES = os.path.join(ROOT, "data", "rl", "instance_rates.jsonl")
# Widest first is wrong: start at the band that actually carries signal and open up only if it is
# too thin to train on. A few hundred prompts are memorized long before 500 RL steps.
BANDS = ((0.2, 0.8), (0.15, 0.85), (0.1, 0.9), (0.05, 0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rates", default=RATES)
    ap.add_argument("--min", type=int, default=800, help="widen the band until it holds this many")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.rates, encoding="utf-8") if l.strip()]
    if not rows:
        print(f"select_band: {a.rates} is empty", file=sys.stderr)
        return 1

    hist = Counter(min(int(r["pass_at_k"] * 10), 9) for r in rows)
    print("solve-rate deciles: " + " ".join(f"{d / 10:.1f}:{hist[d]}" for d in range(10)))
    degen = hist[0] + hist[9]
    print(f"all-or-none: {degen}/{len(rows)} = {degen / len(rows):.1%} (zero advantage for RL)")

    for lo, hi in BANDS:
        band = [r for r in rows if lo <= r["pass_at_k"] <= hi]
        if len(band) >= a.min:
            break
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in band:
            f.write(
                json.dumps({"instruction": r["instruction"], "answer": r["answer"]}, ensure_ascii=False)
                + "\n"
            )
    # Which band was used is part of the result: a run that had to reach 5-95% to find enough
    # prompts is a different experiment from one that trained on a clean 20-80%.
    print(f"band {lo:.0%}-{hi:.0%}: {len(band)}/{len(rows)} instances -> {a.out}")
    if len(band) < a.min:
        print(f"[warn] only {len(band)} prompts even at the widest band; RL has little to work with")
    return 0


if __name__ == "__main__":
    sys.exit(main())
