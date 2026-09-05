#!/usr/bin/env python3
"""Build the 4-way likelihood set for experiment 1's primary readout.

    python3 mathbank/emit_novel_ops_4way.py            # write S_test_4way.jsonl
    python3 mathbank/emit_novel_ops_4way.py --selftest # verify every property it must have

WHY LIKELIHOOD AND NOT EXACT MATCH (e1's MDE, fb ruling 2026-09-05). Exact match on a
generation has no headroom off a 0 floor at 200M: a curve that starts at 0 and stays at 0
cannot distinguish "the skill was not acquired" from "the readout cannot see it". Scoring
four fixed options by likelihood has a 25% floor and moves before the model can produce a
correct string unprompted.

WHY THESE DISTRACTORS. Each is the answer under a DIFFERENT MISREADING of the same rule,
so a model that picks one has told us which rule it applied -- the options are diagnostic,
not filler. The gold is the add-once fold. The ladder, in priority order, with how often
each is reachable over the 1000 items:

    add_until      467   add 10 until non-negative instead of exactly once
    no_carry       726   ignore the carry rule entirely
    wrong_order    984   evaluate left-to-right instead of right-to-left
    swap_operands  500   read the rule as 3b - 2a + 1
    sign_slip      298   read the minus as a plus
    carry_20         0   add 20 instead of 10 (never needed; kept for regeneration)
    coeff_swap      25   read the rule as 2a - 3b + 1
    no_plus_one      0   drop the trailing +1 (never needed; kept for regeneration)

An item takes the first three whose value is distinct from gold and from each other. The
ladder exists because the obvious three collide often: only 457 of 1000 items have four
distinct values under (gold, add_until, no_carry, wrong_order), and 25 items still need
the seventh rung. Falling back to a random integer would make those items measure
something else, so every option stays a rule-misreading.

POSITION. Options are shuffled per item with a fixed seed and the label distribution is
reported, because a set whose gold sits at index 0 more often than chance is a set a model
can score above floor without reading the problem.
"""

import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "probes", "novel_ops", "S_test.jsonl")
OUT = os.path.join(ROOT, "data", "probes", "novel_ops", "S_test_4way.jsonl")
SHUFFLE_SEED = 20260907  # distinct from TEST_SEED/POOL_SEED: a third stream, not a continuation
N_OPTIONS = 4
EXPR = re.compile(r"\d+(?:\s*@\s*\d+)+")


def _ops(instruction):
    m = EXPR.search(instruction)
    if not m:
        raise SystemExit(f"REFUSING: no `X @ Y` expression in {instruction!r}")
    return [int(x) for x in m.group(0).split("@")]


def _fold_r(ops, f):
    v = ops[-1]
    for x in reversed(ops[:-1]):
        v = f(x, v)
    return v


def _fold_l(ops, f):
    v = ops[0]
    for x in ops[1:]:
        v = f(v, x)
    return v


def _once(a, b):
    v = 3 * a - 2 * b + 1
    return v + 10 if v < 0 else v


def _until(a, b):
    v = 3 * a - 2 * b + 1
    while v < 0:
        v += 10
    return v


#: (name, value function). Order is priority; see the module docstring for hit rates.
LADDER = [
    ("add_until", lambda o: _fold_r(o, _until)),
    ("no_carry", lambda o: _fold_r(o, lambda a, b: 3 * a - 2 * b + 1)),
    ("wrong_order", lambda o: _fold_l(o, _once)),
    ("swap_operands", lambda o: _fold_r(o, lambda a, b: 3 * b - 2 * a + 1)),
    ("sign_slip", lambda o: _fold_r(o, lambda a, b: (lambda v: v + 10 if v < 0 else v)(3 * a + 2 * b + 1))),
    ("carry_20", lambda o: _fold_r(o, lambda a, b: (lambda v: v + 20 if v < 0 else v)(3 * a - 2 * b + 1))),
    ("coeff_swap", lambda o: _fold_r(o, lambda a, b: (lambda v: v + 10 if v < 0 else v)(2 * a - 3 * b + 1))),
    ("no_plus_one", lambda o: _fold_r(o, lambda a, b: (lambda v: v + 10 if v < 0 else v)(3 * a - 2 * b))),
]


def build_item(row, rng):
    """One 4-way item. RAISES if three distinct distractors cannot be found."""
    ops = _ops(row["instruction"])
    gold = _fold_r(ops, _once)
    if gold != row["answer"]:
        raise SystemExit(
            f"REFUSING: the add-once fold gives {gold} but the source row's answer is "
            f"{row['answer']}. The 4-way set would then teach a different rule than the "
            f"source set scores: {row['instruction']!r}")
    seen = {gold}
    picked = []
    for name, fn in LADDER:
        v = fn(ops)
        if v not in seen:
            seen.add(v)
            picked.append((name, v))
        if len(picked) == N_OPTIONS - 1:
            break
    if len(picked) < N_OPTIONS - 1:
        raise SystemExit(
            f"REFUSING: only {len(picked)} distinct distractor(s) for {row['instruction']!r}. "
            f"Every rung of the ladder collides with gold or with another rung; add a rung "
            f"that is still a misreading of the rule rather than padding with a random int.")
    opts = [("gold", gold)] + picked
    rng.shuffle(opts)
    return {
        "family": "S",
        "program": row["program"],
        "instruction": row["instruction"],
        "operands": ops,
        "options": [v for _n, v in opts],
        "label": [n for n, _v in opts].index("gold"),
        "option_kinds": [n for n, _v in opts],
        "answer": gold,
    }


def build(src=SRC):
    rows = [json.loads(line) for line in list(open(src, encoding="utf-8"))[1:]]
    rng = random.Random(SHUFFLE_SEED)
    return [build_item(r, rng) for r in rows]


def _label_hist(items):
    return dict(sorted(collections.Counter(i["label"] for i in items).items()))


def write(out=OUT, src=SRC):
    items = build(src)
    kinds = collections.Counter(k for i in items for k in i["option_kinds"] if k != "gold")
    header = {
        "_header": True, "family": "S", "kind": "test_4way", "n": len(items),
        "source": "data/probes/novel_ops/S_test.jsonl",
        "generator": "mathbank/emit_novel_ops_4way.py",
        "shuffle_seed": SHUFFLE_SEED,
        "readout": ("4-way likelihood: score each option, pick the argmax, compare to `label`. "
                    "Floor is 25%. Chosen over exact match because exact match has no headroom "
                    "off a 0 floor at 200M (e1's MDE, fb ruling 2026-09-05)."),
        "distractors_are_diagnostic": ("each option is the answer under a different MISREADING of "
                                       "the rule, so which option a model picks says which rule it "
                                       "applied. option_kinds names them per item."),
        "distractor_usage": dict(sorted(kinds.items())),
        "label_distribution": _label_hist(items),
        "absence_basis": ("inherited from the source set: the operator, its rule and its phrasing "
                          "were invented 2026-09-05, after every corpus in the mix was built. "
                          "facts/contamination.json#cont.novel_ops_frozen_sets"),
    }
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i in items:
            fh.write(json.dumps(i, ensure_ascii=False) + "\n")
    return len(items), header


def _selftest():
    fails = []
    items = build()
    if len(items) != 1000:
        fails.append(f"{len(items)} items, want 1000")
    # 1. EVERY DISTRACTOR IS ITS READING'S FOLD, checked against an independent recomputation.
    #    This is the assertion fb asked for: without it "option_kinds" is a label nobody verified.
    by_name = dict(LADDER)
    for it in items:
        for kind, val in zip(it["option_kinds"], it["options"]):
            if kind == "gold":
                if val != it["answer"]:
                    fails.append(f"gold option {val} != answer {it['answer']}")
                continue
            if by_name[kind](it["operands"]) != val:
                fails.append(f"{kind} option {val} is not that reading's fold")
                break
    # 2. FOUR DISTINCT OPTIONS. A repeated value makes two choices indistinguishable and the
    #    floor is no longer 25%.
    dup = [it for it in items if len(set(it["options"])) != N_OPTIONS]
    if dup:
        fails.append(f"{len(dup)} item(s) with a repeated option, e.g. {dup[0]['options']}")
    # 3. THE LABEL POINTS AT THE GOLD, after the shuffle.
    bad = [it for it in items if it["options"][it["label"]] != it["answer"]]
    if bad:
        fails.append(f"{len(bad)} item(s) whose label does not point at the gold")
    # 4. GOLD IS THE ADD-ONCE FOLD, not merely copied from the source row.
    for it in items:
        if _fold_r(it["operands"], _once) != it["answer"]:
            fails.append(f"gold is not the add-once fold: {it['instruction']!r}")
            break
    # 5. POSITION CARRIES NO SIGNAL. Under a uniform shuffle each of the 4 positions holds the
    #    gold ~250 times; the chi-square 99.9% critical value at df=3 is 16.27, so a set whose
    #    gold clusters at one index fails here rather than inflating the floor silently.
    hist = _label_hist(items)
    if sorted(hist) != list(range(N_OPTIONS)):
        fails.append(f"some position never holds the gold: {hist}")
    else:
        exp = len(items) / N_OPTIONS
        chi = sum((hist[k] - exp) ** 2 / exp for k in hist)
        if chi > 16.27:
            fails.append(f"gold position is not uniform: chi2={chi:.1f} > 16.27 (df=3, p=0.001), {hist}")
    # 6. DETERMINISM: the same seed twice is the same set, or a rebuild silently rescores.
    if [i["options"] for i in build()] != [i["options"] for i in items]:
        fails.append("build() is not reproducible at its own seed")
    # 7. THE DISTINCTNESS CHECK MUST HAVE POWER: an item built from a ladder whose rungs all
    #    return gold must refuse, not emit three copies of it.
    saved = LADDER[:]
    try:
        LADDER[:] = [(n, (lambda o: _fold_r(o, _once))) for n, _f in saved]
        try:
            build_item({"instruction": items[0]["instruction"], "program": "diamond_chain",
                        "answer": items[0]["answer"]}, random.Random(0))
            fails.append("build_item accepted a ladder that produces no distinct distractor")
        except SystemExit:
            pass
    finally:
        LADDER[:] = saved
    for f in fails:
        print(f"BUG {f}", file=sys.stderr)
    print(f"emit_novel_ops_4way selftest: {'PASS (7 worlds)' if not fails else f'{len(fails)} BUG(S)'}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    n, h = write(a.out)
    print(f"wrote {os.path.relpath(a.out, ROOT)}: {n} items")
    print(f"  label distribution: {h['label_distribution']}")
    print(f"  distractor usage:   {h['distractor_usage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
