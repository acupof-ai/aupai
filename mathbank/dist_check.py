#!/usr/bin/env python3
"""Compare a math batch against the eval on surface statistics, and flag what diverges.

Every check the pipeline had was added after something went wrong: answer length
after sft_k4 measured harmful, level mix to match the eval, forward references
after they turned out to be 9.8% of a batch. That is a scar list, not a test. It
missed a 1.85x skew in how often a solution contains a fraction -- 37.3% in
math_short_v8 against 20.2% in the eval's own solutions -- which the model then
reproduced at 51.1% of its generations.

This compares many axes at once so the next skew does not need someone to think of
it first. It says nothing about whether a batch is CORRECT; verify() and eqcheck.py
do that. It only says whether it LOOKS like the thing the model is scored on.

    python mathbank/dist_check.py data/synthetic/math_short_v9.jsonl
    python mathbank/dist_check.py batch.jsonl --ref data/eval/math_test_500.jsonl
"""

import argparse
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "data", "synthetic", "math_hard_eval_1k.jsonl")

NUM = re.compile(r"-?\d+(?:\.\d+)?")
FRAC = re.compile(r"\d+\s*/\s*\d+|frac")
DEC = re.compile(r"\d+\.\d+")
OPS = re.compile(r"[+\-×÷*]")
UNIT = re.compile(r"[千米米元个只本天时分秒kg克吨升]")


def stats(rows):
    """Surface statistics of a batch. Each is a rate or a median, so batches of
    different sizes compare directly."""
    n = len(rows)
    out = {}
    sol = [r.get("output", "") for r in rows]
    q = [r.get("instruction", "") for r in rows]
    out["solution contains a fraction"] = sum(bool(FRAC.search(s)) for s in sol) / n
    out["solution contains a decimal"] = sum(bool(DEC.search(s)) for s in sol) / n
    out["question contains a unit word"] = sum(bool(UNIT.search(s)) for s in q) / n
    lens = sorted(len(s) for s in sol)
    out["solution length (median chars)"] = lens[n // 2]
    out["solution length (p90 chars)"] = lens[int(n * 0.9)]
    steps = sorted(s.count("\n") + 1 for s in sol)
    out["solution steps (median lines)"] = steps[n // 2]
    nums = [len(NUM.findall(s)) for s in sol]
    out["numbers per solution (median)"] = sorted(nums)[n // 2]
    qn = [len(NUM.findall(s)) for s in q]
    out["numbers per question (median)"] = sorted(qn)[n // 2]
    ops = [c for s in sol for c in OPS.findall(s)]
    for o, label in (("+", "plus"), ("-", "minus"), ("×", "times"), ("÷", "divide")):
        out[f"operator share: {label}"] = ops.count(o) / max(1, len(ops))
    mags = [abs(float(m)) for s in sol for m in NUM.findall(s)[:20]]
    out["number magnitude (median)"] = statistics.median(mags) if mags else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--ref", default=EVAL, help="what the model is scored on")
    ap.add_argument(
        "--tol",
        type=float,
        default=0.30,
        help="flag an axis whose ratio to the reference leaves [1-tol, 1+tol]; 0.30 is loose "
        "enough that ordinary sampling noise does not fire it and tight enough to have caught "
        "the 1.85x fraction skew",
    )
    a = ap.parse_args()

    b = [json.loads(x) for x in open(a.batch, encoding="utf-8") if x.strip()]
    r = [json.loads(x) for x in open(a.ref, encoding="utf-8") if x.strip()]
    sb, sr = stats(b), stats(r)

    print(f"{os.path.basename(a.batch)} ({len(b)} rows) against {os.path.basename(a.ref)} ({len(r)})\n")
    print(f"{'axis':<34}{'batch':>10}{'eval':>10}{'ratio':>9}")
    bad = []
    for k in sb:
        vb, vr = sb[k], sr[k]
        ratio = vb / vr if vr else float("inf") if vb else 1.0
        flag = "" if (1 - a.tol) <= ratio <= (1 + a.tol) else "  <-- OFF"
        if flag:
            bad.append((k, ratio))
        fmt = "{:>10.1%}" if vb <= 1 and vr <= 1 else "{:>10.1f}"
        print(f"{k:<34}" + fmt.format(vb) + fmt.format(vr) + f"{ratio:>9.2f}{flag}")
    print()
    if bad:
        print(f"{len(bad)} axes diverge by more than {a.tol:.0%}:")
        for k, ratio in sorted(bad, key=lambda x: -abs(x[1] - 1)):
            print(f"  {k}: {ratio:.2f}x")
        return 1
    print(f"every axis within {a.tol:.0%} of the eval")
    return 0


def _demo():
    """The instrument on a case whose answer is known: the eval against itself is
    flat, and v8's fraction rate is the skew this was built to catch."""
    rows = [json.loads(x) for x in open(EVAL, encoding="utf-8") if x.strip()]
    s = stats(rows)
    assert abs(s["solution contains a fraction"] - 0.202) < 0.01, s["solution contains a fraction"]
    # A RANDOM half, not the first half: the eval file is ordered by level, so its
    # first half is 29.5% fractions against 20.2% overall. Anything that slices it
    # with rows[:n] gets a biased sample; the shard scripts stride, which is fine.
    import random

    r = rows[:]
    random.Random(0).shuffle(r)
    half = stats(r[: len(r) // 2])
    for k in s:
        if s[k]:
            assert 0.75 <= half[k] / s[k] <= 1.33, f"{k} unstable across halves: {half[k]} vs {s[k]}"
    print("dist_check self-test OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        sys.exit(main())
