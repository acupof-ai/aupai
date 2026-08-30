#!/usr/bin/env python3
"""Known-answer suite for rlvr_reward (the RLVR checker).

Same construction as the MinHash known-answer suite: pairs where the right
answer is known by construction, in named variant classes, so a checker
change that breaks any class is caught. A bad filter gives a wrong survival
rate; a bad reward function is what the model optimises against, so its
blind spots become training signal.

Two modes:
  python3 algorithms/test_rlvr_reward_suite.py            # variant pairs
  python3 algorithms/test_rlvr_reward_suite.py --roundtrip # full-data positive control

Variant pair classes:
  SHOULD  - reward must equal expected; a mismatch is a BUG (exit 1)
  GAP     - known wrong behavior, documented; mismatch means it changed (flag)

Findings the suite locks in (measured 2026-08-30, rlvr_clean.jsonl 217,953 rows):
  - round-trip 99.9904% pre-fix; post-fix 99.9977% (217,948/217,953). The
    pre-fix 21 failures = 9x yes/no '对' + 12x others; post-fix 5 = 4x
    multi-answer blocks + 1x 米米 (7 of the 12 flipped because the truncated
    \\frac tolerance was mangling their GTs)
  - FIXED 2026-08-30: '对' removed from the unit-stripping list (it collided
    with 一对; asymmetric vs 错 — RL paid 错 never 对, a directional bias)
  - FIXED 2026-08-30: tolerance is now type-split — integer/fraction gold
    compares exactly, decimal gold gets absolute 1e-4 (the old relative 1e-4
    made 10000 vs 10001 score 1.0; answers are exact quantities)
  - FIXED 2026-08-30: unbalanced \\boxed refused (extract returns None) and
    truncated-GT tolerance removed; bad GTs are refused at load_problems
  - standing rule: every checker that becomes a reward gets the GT round
    trip before a single RL step (load_problems now enforces it)
  - GAPs (left as FNs deliberately — every normaliser extension is a new way
    to mark a wrong answer correct): parenthesised negatives, mixed numbers
    (3\\frac{1}{2}), 万 (100万 vs 1000000), set order, doubled units (米米)
  - data debt: 5 rows in rlvr_clean fail the post-fix round trip — 4x
    multi-answer \\begin{aligned} blocks + 1x '\\text{米米}' (the doubled-unit
    GAP exists in real data, 1 row); load_problems refuses them loudly.
    Wider contract debt: 361 GTs (0.17%) are solution-form aligned blocks
    with no extractable final answer — they pass the round trip only because
    the gen is artificially the GT itself; in training the reward can never
    fire on them. Flagged to 3b as a data requirement (extract the final
    value at build time), not a reward change.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlvr_reward import reward_fn

# (gen_text, gt_raw, expected, note)
SHOULD = [
    # -- positive: genuinely equal, surface variants --
    (r"\boxed{240厘米}", "240厘米", 1.0, "int + unit both sides"),
    (r"\boxed{\frac{1}{2}}", r"\dfrac{1}{2}", 1.0, "frac vs dfrac"),
    (r"\boxed{0.5}", r"\dfrac{1}{2}", 1.0, "decimal vs fraction"),
    (r"\boxed{\text{无解}}", "无解", 1.0, "\\text wrapping"),
    (r"\boxed{1,000}", "1000", 1.0, "thousands comma"),
    (r"\boxed{-5}", "-5", 1.0, "negative"),
    (r"\boxed{50\%}", "50\\%", 1.0, "percent equal forms"),
    (r"\boxed{3.1400}", "3.14", 1.0, "decimal trailing zeros"),
    (r"\boxed{ 240 }", "240", 1.0, "whitespace"),
    (r"\boxed{100}\boxed{240}", "240", 1.0, "last boxed wins"),
    (r"\boxed{错}", "错", 1.0, "yes/no 错 (the working half)"),
    (r"\boxed{对}", "对", 1.0, "yes/no 对 (regression: was eaten by 一对 unit strip)"),
    (r"\boxed{x>3}", "x>3", 1.0, "expression answer"),
    (r"\boxed{\{2,3,5\}}", r"\{2,3,5\}", 1.0, "set equal forms"),
    (r"\boxed{240厘米}", "240", 1.0, "unit in gen only"),
    (r"\boxed{240.0}", "240", 1.0, "decimal-form pred equal to integer gold"),
    (r"\boxed{0.5}", r"\dfrac{1}{2}", 1.0, "decimal pred exactly equal to fraction gold"),
    # -- negative: genuinely different, must not collapse --
    (r"\boxed{50\%}", "0.5", 0.0, "percent vs decimal (regression guard)"),
    (r"\boxed{5}", "-5", 0.0, "sign flip"),
    (r"\boxed{\frac{3}{2}}", r"\dfrac{2}{3}", 0.0, "reciprocal"),
    (r"\boxed{\frac{1}{3}}", r"\dfrac{1}{2}", 0.0, "near-miss fraction"),
    (r"\boxed{10001}", "10000", 0.0, "integer gold exact (regression: relative tol)"),
    (r"\boxed{240.0001}", "240", 0.0, "integer gold exact — near-miss decimal refused"),
    (r"\boxed{0.3333}", r"\dfrac{1}{3}", 0.0, "fraction gold exact — decimal approximation refused"),
    (r"\boxed{10}", r"\dfrac{10", 0.0, "truncated GT must not match its prefix (regression)"),
    (r"\boxed{240", "240", 0.0, "unbalanced model boxed refused (regression)"),
    ("答案是240", "240", 0.0, "no boxed -> format contract"),
    (r"答案是240，\boxed{0}", "240", 0.0, "boxed contradicts text"),
    (r"\boxed{有解}", "无解", 0.0, "different non-numeric"),
    (r"\boxed{0.1002}", "0.1", 0.0, "decimal beyond 1e-4 absolute tolerance"),
]

# (gen_text, gt_raw, current_behavior, should_be, note)
GAP = [
    (r"\boxed{(-3)}", "-3", 0.0, 1.0,
     "FN: parenthesised negatives never match numerically (float raises, frac "
     "regex requires '/'). Models parenthesise negatives often. Left as FN: "
     "fixing it means extending the normaliser, and every extension is a new "
     "way to mark a wrong answer correct"),
    (r"\boxed{3\frac{1}{2}}", "3.5", 0.0, 1.0,
     "mixed number -> 3(1)/(2), no numeric parse. Rare in GT (3.9% frac, "
     "mixed rarer) but a real FN"),
    (r"\boxed{100万}", "1000000", 0.0, 1.0,
     "万/亿 not parsed; Chinese-numeral answers are exact-match only"),
    (r"\boxed{\{5,3,2\}}", r"\{2,3,5\}", 0.0, 1.0,
     "set order matters after comma deletion; sets are unordered"),
    (r"\boxed{米米}", "米米", 0.0, 1.0,
     "unit strip runs twice -> empty -> None. Edge; same class as the 对 bug"),
]


def run_suite():
    bugs = flags = 0
    print("== SHOULD (mismatch = bug) ==")
    for gen, gt, exp, note in SHOULD:
        got = reward_fn(gen, gt)
        ok = got == exp
        if not ok:
            bugs += 1
        print(f"  {'OK ' if ok else 'BUG'} {got} exp {exp} | {note}")
    print("== GAP (change = flag) ==")
    for gen, gt, cur, should, note in GAP:
        got = reward_fn(gen, gt)
        changed = got != cur
        if changed:
            flags += 1
        print(f"  {'CHG' if changed else 'gap'} {got} (was {cur}, should {should}) | {note[:70]}")
    print(f"\n{bugs} bugs, {flags} gap-changes")
    return bugs


def run_roundtrip():
    """Positive control on real data: the GT's own answer must score 1.0."""
    path = "/work/aupai/data/rl/rlvr_clean.jsonl"
    n = fail = 0
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        gt = str(d.get("answer") or "")
        n += 1
        if reward_fn(f"\\boxed{{{gt}}}", gt) != 1.0:
            fail += 1
    print(f"round-trip {path}: {n - fail}/{n} = {(n - fail) / n:.4%} pass")
    return fail


if __name__ == "__main__":
    if "--roundtrip" in sys.argv:
        sys.exit(1 if run_roundtrip() else 0)
    sys.exit(1 if run_suite() else 0)
