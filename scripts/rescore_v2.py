#!/usr/bin/env python3
"""Rescore math_hard_eval_v2 predictions with a family-aware grader.

Why: eval/math_hard.py's stock reward_fn does exact-string-or-numeric matching.
v2's symbolic answers are form-brittle under it -- "(x+2)(x-3)" vs gold
"(x-3)(x+2)", or "x=2或x=3" vs gold "x₁=2,x₂=3" -- and would read as 0 even
when the model solved the problem. A resolution probe with a scorer that
artificially zeros families gives a false "no signal" reading, so grade each
family by its semantics:

  set-families   (quadratic / abs / system):  compare the SET of numbers
  ordered-pair   (linear_func / number_line): compare numbers in order
  factorize:      compare root multiset extracted from the linear factors
  everything else: numeric (stock reward_fn semantics, units stripped)

Usage: python3 scripts/rescore_v2.py <preds.jsonl> <v2.jsonl>
"""
import json
import re
import sys

sys.path.insert(0, ".")
from algorithms.rlvr_reward import extract_boxed, reward_fn  # noqa: E402

SET_FAM = {"quadratic", "abs_equation", "system_2var"}
ORDER2_FAM = {"linear_func", "number_line_moving"}
ROOTS_FAM = {"factorize"}
NUM_RE = re.compile(r"-?\d+\.?\d*")


def numbers(s):
    return [float(x) for x in NUM_RE.findall(str(s))]


def roots_of_factors(s):
    """Root multiset of (x±a)(x±b)... ; None if no factors found."""
    facs = re.findall(r"\(\s*x\s*([+-])\s*(\d+(?:\.\d+)?)\s*\)", str(s))
    if not facs:
        return None
    return sorted(-v if sign == "+" else v for sign, v in [(sgn, float(n)) for sgn, n in facs])


def grade(fam, gen, gold):
    boxed = extract_boxed(gen)  # the pred rows store gen[-300:]; the boxed answer sits at the end
    if boxed is None:
        return 0.0
    if fam in SET_FAM:
        p, g = sorted(numbers(boxed)), sorted(numbers(gold))
        return 1.0 if p == g and p else 0.0
    if fam in ORDER2_FAM:
        p, g = numbers(boxed), numbers(gold)
        return 1.0 if len(p) == 2 and len(g) == 2 and p == g else 0.0
    if fam in ROOTS_FAM:
        p, g = roots_of_factors(boxed), roots_of_factors(gold)
        return 1.0 if p is not None and p == g else 0.0
    return reward_fn(f"\\boxed{{{boxed}}}", gold)


def main():
    preds_p, v2_p = sys.argv[1], sys.argv[2]
    fam = {json.loads(l)["instruction"]: json.loads(l).get("type", "?")
           for l in open(v2_p, encoding="utf-8")}
    CLUSTER = {f: "symbolic" for f in SET_FAM | ORDER2_FAM | ROOTS_FAM |
               {"fractional_eq", "inverse_prop", "quadratic_func", "variance",
                "floor_gauss", "perfect_square_pattern"}}
    CLUSTER.update({"pythagoras": "geometry", "right_triangle_3060": "geometry",
                    "similar_triangle": "geometry"})
    CLUSTER.update({"number_line_moving": "elementary-gap", "cryptarithm": "elementary-gap",
                    "number_array_cross": "elementary-gap", "hound_hare": "elementary-gap"})
    by, cl, stock = {}, {}, [0, 0]
    for line in open(preds_p, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("greedy"):
            continue
        f = fam.get(r["q"], "?")
        ok = grade(f, r["gen"], r["answer"])
        by.setdefault(f, [0, 0]); by[f][0] += ok; by[f][1] += 1
        c = CLUSTER.get(f, "?")
        cl.setdefault(c, [0, 0]); cl[c][0] += ok; cl[c][1] += 1
        stock[0] += r["ok"]; stock[1] += 1
    n = sum(v[1] for v in by.values())
    tot = sum(v[0] for v in by.values())
    print(f"OVERALL v2 pass@1 (family-aware): {tot}/{n} = {tot / n:.1%}")
    print(f"  stock reward_fn for reference:  {stock[0]}/{stock[1]} = {stock[0] / stock[1]:.1%}")
    for c in ("symbolic", "geometry", "elementary-gap"):
        v = cl.get(c, [0, 0])
        print(f"  cluster {c:16s}: {v[0]}/{v[1]} = {v[0] / max(v[1], 1):.1%}")
    print("per-family:")
    for f in sorted(by):
        v = by[f]
        print(f"  {f:26s} {v[0]:3.0f}/{v[1]:3d} = {v[0] / v[1]:5.1%}")


if __name__ == "__main__":
    main()
