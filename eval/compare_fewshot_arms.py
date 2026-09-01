#!/usr/bin/env python3
"""Compare two few-shot arms on the SAME problems (de, 2026-09-01).

The eval set excludes the demos, so it shrinks as demos grow: 497 at 3 demos, 492 at
8. Comparing the arms' headline rates directly compares populations as well as
prompts. `--eval-from` pins the set going forward, but the 3-demo arm ran before that
flag existed, so this re-aggregates it to the shared problem set instead of rerunning
a card's worth of work to fix a bookkeeping difference.

Restricting to the intersection is the right move rather than dropping problems to
match a count: the arms then answer the same questions, and the denominator is a fact
about the comparison rather than about either arm.

Reports, in this order and never the other:
  1. format rate -- the manipulation check. An accuracy under a failed manipulation
     is not interpretable, which is the discipline whose absence produced the sampled
     arm's ambiguity.
  2. accuracy, each arm, on the shared set
  3. a 200-permutation shuffled control per arm, because a single shuffle gave
     2.8/2.4/3.6% on three seeds and any one alone is a number that cannot be priced
  4. the paired difference, McNemar-style: of the problems where the arms DISAGREE,
     how many go each way. The unpaired difference of two rates on the same problems
     wastes the pairing and overstates its own uncertainty.

    python3 eval/compare_fewshot_arms.py --a preds_l1_d3....jsonl --b preds_l1_d8....jsonl
"""

import argparse
import json
import os
import random
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_BOX = re.compile(r"\\boxed\{([^}]*)\}")


def rows_of(path):
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("q") is not None:
                out[r["q"]] = r
    return out


def boxed(text):
    m = _BOX.search(text or "")
    return m.group(1).strip() if m else None


def _eq(a, b):
    if a is None or b is None:
        return False
    a, b = a.strip(), b.strip()
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-9
    except ValueError:
        return a == b


def arm_stats(rows, qs, golds, seed=0, n_perm=200):
    """(format_rate, hits, control_mean, control_sd) over the shared question set."""
    preds = [boxed(rows[q].get("gen", "")) for q in qs]
    fmt = sum(1 for p in preds if p is not None) / len(qs)
    gold = [golds.get(q) for q in qs]
    hits = [_eq(p, g) for p, g in zip(preds, gold, strict=True)]
    ctrl = []
    for s in range(n_perm):
        sh = gold[:]
        random.Random(seed + s).shuffle(sh)
        ctrl.append(sum(_eq(p, g) for p, g in zip(preds, sh, strict=True)))
    return fmt, hits, statistics.mean(ctrl), statistics.pstdev(ctrl)


def selftest():
    assert boxed(r"所以 \boxed{7}") == "7"
    assert boxed("没有") is None
    assert _eq("7", "7.0") and not _eq("7", "9")
    assert _eq("1,234", "1234"), "thousands separators must not create a mismatch"
    # the pairing: two arms, 4 problems, disagreeing on exactly 2 in opposite
    # directions. A raw rate difference is 0 and hides that anything moved; the
    # discordant counts are 1 and 1, which is the honest description.
    a = [True, True, False, False]
    b = [True, False, True, False]
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    assert (a_only, b_only) == (1, 1), (a_only, b_only)
    assert sum(a) == sum(b), "the fixture's point is equal rates over different problems"
    print("selftest OK: parses boxed answers, compares numerically, and the paired "
          "counts distinguish 'nothing moved' from 'equal rates, different problems'")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="arm A preds jsonl")
    ap.add_argument("--b", help="arm B preds jsonl")
    ap.add_argument("--a-name", default="arm A")
    ap.add_argument("--b-name", default="arm B")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "eval", "math_test_500.jsonl"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.a and args.b):
        ap.error("--a and --b required (or --selftest)")

    A, B = rows_of(args.a), rows_of(args.b)
    for nm, r, p in (("A", A, args.a), ("B", B, args.b)):
        if r is None:
            print(f"arm {nm}: no such file {p}")
            return 1
    with open(args.data, encoding="utf-8") as f:
        golds = {}
        for line in f:
            if line.strip():
                d = json.loads(line)
                golds[d["instruction"]] = boxed(d.get("output", ""))

    qs = sorted(set(A) & set(B))
    if not qs:
        print("the two arms share no problems; nothing to compare")
        return 1
    print(f"{args.a_name}: {len(A)} problems   {args.b_name}: {len(B)} problems   "
          f"shared: {len(qs)}")
    if len(A) != len(qs) or len(B) != len(qs):
        print(f"  restricted to the {len(qs)} shared problems -- the arms' own "
              f"denominators differ because the eval set excludes each arm's demos, "
              f"and comparing them unrestricted compares populations too")

    fa, ha, ca, sa = arm_stats(A, qs, golds)
    fb_, hb, cb, sb = arm_stats(B, qs, golds)
    n = len(qs)

    # 1. manipulation check FIRST.
    print("\nformat rate (the manipulation check, reported before accuracy)")
    print(f"  {args.a_name:14s} {fa:.1%}")
    print(f"  {args.b_name:14s} {fb_:.1%}")
    if min(fa, fb_) < 0.20:
        print("  -> BELOW the pre-registered 20% gate for at least one arm: the "
              "manipulation did not take, and the accuracies below are not "
              "interpretable as capability.")

    print(f"\naccuracy on the {n} shared problems, against a 200-permutation control")
    for nm, h, c, s in ((args.a_name, ha, ca, sa), (args.b_name, hb, cb, sb)):
        k = sum(h)
        z = (k - c) / s if s else float("nan")
        print(f"  {nm:14s} {k}/{n} = {k / n:.2%}   control {c / n:.2%} "
              f"(sd {s / n:.2%})   {(k - c) / n * 100:+.2f}pt, z={z:.2f}")

    # 4. paired comparison. Same problems, so the pairing is free information.
    both = sum(1 for x, y in zip(ha, hb, strict=True) if x and y)
    a_only = sum(1 for x, y in zip(ha, hb, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(ha, hb, strict=True) if y and not x)
    print(f"\npaired: both {both}, {args.a_name} only {a_only}, {args.b_name} only {b_only}")
    d = a_only + b_only
    if d == 0:
        print("  the arms are identical on every shared problem")
    else:
        # exact binomial sign test on the discordant pairs
        from math import comb
        k = min(a_only, b_only)
        p = sum(comb(d, i) for i in range(k + 1)) / (2 ** d) * 2
        print(f"  {d} discordant pair(s), sign test p = {min(1.0, p):.3f}"
              + ("  -- not distinguishable" if p > 0.05 else "  -- a real difference"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
