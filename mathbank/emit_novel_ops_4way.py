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

    wrong_order    762   evaluate left-to-right instead of right-to-left
    swap_operands  589   read the rule as 3b - 2a + 1 (with the carry dropped: see below)
    add_until      379   add 10 until non-negative instead of exactly once
    no_carry       379   ignore the carry rule entirely
    sign_slip      301   read the minus as a plus
    coeff_swap     300   read the rule as 2a - 3b + 1
    no_plus_one    168   drop the trailing +1
    carry_20       122   add 20 instead of 10

All eight are load-bearing. The ladder exists because the obvious three collide: only 457
of 1000 items have four distinct values under (gold, add_until, no_carry, wrong_order).
Falling back to a random integer would make those items measure something else, so every
option stays a rule-misreading and the emitter RAISES instead.

TWO THINGS ARE BALANCED, and the second was missing from v1.

POSITION -- which index holds the gold. Shuffled per item with a fixed seed, chi-square
reported in the header.

VALUE RANK -- where the gold sits among the four numbers, largest first. v1 balanced only
the position, so "pick the 3rd-largest of the four numbers", a rule that never reads the
problem, scored 0.6430: gold landed at rank 2 on 643 of 1000 items and the position
chi-square could not see it (e1, 2026-09-05). Which three distractors an item takes is now
chosen to flatten the rank, and to keep "gold is nearest zero" at its 1/4 chance share --
balancing rank alone left that rule at 0.3500. The selftest runs a battery of eight
content-free rules and fails any above 0.30.

DIAGNOSTICITY, with one caveat. Each option is the answer under a different misreading, so
which option a model picks says which rule it applied -- except swap_operands, which is a
COMPOUND: its value matches 3b-2a+1 with the carry DROPPED, and the two component readings
differ on 172 of 500 items (e1). Picking it is ambiguous between two errors. Kept as a
distractor, named honestly in option_kinds.
"""

import argparse
import collections
import itertools
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


def _candidates(row):
    """(gold, [(name, value), ...]) -- every distinct distractor the ladder yields.

    Returns ALL of them, not the first three: which three are chosen is a rank decision
    made in build(), and a function that returned three could not make it.
    """
    ops = _ops(row["instruction"])
    gold = _fold_r(ops, _once)
    if gold != row["answer"]:
        raise SystemExit(
            f"REFUSING: the add-once fold gives {gold} but the source row's answer is "
            f"{row['answer']}. The 4-way set would then teach a different rule than the "
            f"source set scores: {row['instruction']!r}")
    cands = []
    seen = {gold}
    for name, fn in LADDER:
        v = fn(ops)
        if v not in seen:
            seen.add(v)
            cands.append((name, v))
    if len(cands) < N_OPTIONS - 1:
        raise SystemExit(
            f"REFUSING: only {len(cands)} distinct distractor(s) for {row['instruction']!r}. "
            f"Every rung of the ladder collides with gold or with another rung; add a rung "
            f"that is still a misreading of the rule rather than padding with a random int.")
    return gold, cands


def _rank_of(gold, trio):
    """Gold's rank among the four option VALUES, largest first."""
    return sorted([gold] + [v for _n, v in trio], reverse=True).index(gold)


def _key_of(gold, trio):
    """The two content-free properties a subset fixes: gold's value rank, and whether gold
    is the option nearest zero.

    Rank alone was not enough. Balancing it left "pick the option nearest zero" at 0.3500,
    above the 0.30 the readout's power assumes -- the same class of leak one level down,
    found by running the heuristic battery against the rank-balanced build rather than
    assuming one fix covered the family.
    """
    four = [gold] + [v for _n, v in trio]
    return _rank_of(gold, trio), min(four, key=abs) == gold


def _choose(gold, cands, want_key):
    """The first 3-subset with the wanted (rank, nearest-zero) key, else priority order.

    Priority order is the ladder's, so an item that cannot reach the wanted key keeps the
    old behaviour rather than being dropped.
    """
    for trio in itertools.combinations(cands, N_OPTIONS - 1):
        if _key_of(gold, trio) == want_key:
            return list(trio)
    return list(cands[:N_OPTIONS - 1])


def build_item(row, rng, want_key=None):
    """One 4-way item, optionally with gold placed at a given (rank, nearest-zero) key."""
    gold, cands = _candidates(row)
    picked = _choose(gold, cands, want_key) if want_key is not None else list(cands[:N_OPTIONS - 1])
    opts = [("gold", gold)] + picked
    rng.shuffle(opts)
    return {
        "family": "S",
        "program": row["program"],
        "instruction": row["instruction"],
        "operands": _ops(row["instruction"]),
        "options": [v for _n, v in opts],
        "label": [n for n, _v in opts].index("gold"),
        "value_rank": _rank_of(gold, picked),
        "gold_is_nearest_zero": min([gold] + [v for _n, v in picked], key=abs) == gold,
        "option_kinds": [n for n, _v in opts],
        "answer": gold,
    }


def build(src=SRC):
    """Items with gold's VALUE RANK balanced across the four positions.

    WHY RANK AND NOT JUST POSITION. v1 shuffled the option positions -- which is what the
    chi-square in the header measured -- and left the option VALUES alone. Gold then landed
    at rank 2 (3rd largest) on 643 of 1000 items, so "pick the 3rd-largest of the four
    numbers", a rule that never reads the problem, scored 0.6430 (e1, 2026-09-05). The
    readout's floor is not 25% unless the value rank is balanced too.

    The cause is structural in two rungs: sign_slip (3a+2b+1 on positive operands) is
    ALWAYS greater than gold, 298/298, and wrong_order is greater on 648/984, so the two
    most-used rungs crowd the top and push gold down to third.

    The fix needs no new misreadings. Every item has at least three distinct distractors
    from the existing ladder, and choosing WHICH three moves gold's rank: measured over all
    1000 items, gold can be placed at rank 0 on 442, rank 1 on 714, rank 2 on 950, rank 3
    on 839. Rank 0 is the scarce one -- gold is rarely the largest -- so the target is the
    achievable near-flat split rather than exactly 25% each.

    UNDER-FILLED FIRST, and items are processed in order of how few ranks they can reach,
    so the constrained items claim their only option before the flexible ones use it up.
    """
    rows = [json.loads(line) for line in list(open(src, encoding="utf-8"))[1:]]
    rng = random.Random(SHUFFLE_SEED)
    reach = []
    for i, r in enumerate(rows):
        gold, cands = _candidates(r)
        keys = {_key_of(gold, t) for t in itertools.combinations(cands, N_OPTIONS - 1)}
        reach.append((len(keys), i, keys))
    # TARGET SHARES, not equal buckets. Balancing the eight (rank, nearest-zero) cells
    # evenly gives nearest-zero=True on 500 of 1000 -- worse than the 0.3500 it was meant to
    # fix, because "gold is nearest zero" is ONE of four options and its chance share is
    # 1/4, not 1/2. Each cell's target is therefore P(rank) x P(nearest-zero) = 0.25 x 0.25
    # for True and 0.25 x 0.75 for False. Measured against the fix: rank 0.25 flat and
    # nearest-zero at chance. The first version of this loop had the right mechanism and the
    # wrong target, which is the same defect class as the thing it was fixing.
    target = {(rk, nz): (0.25 * (0.25 if nz else 0.75)) for rk in range(N_OPTIONS)
              for nz in (True, False)}
    counts = collections.Counter()
    want = {}
    n_rows = len(rows)
    for _n, i, keys in sorted(reach, key=lambda x: (x[0], x[1])):
        # most under its target first, so a scarce cell is filled before a plentiful one
        want[i] = min(keys, key=lambda k: (counts[k] - target[k] * n_rows, k))
        counts[want[i]] += 1
    return [build_item(r, rng, want_key=want[i]) for i, r in enumerate(rows)]


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
        "value_rank_distribution": dict(sorted(collections.Counter(
            i["value_rank"] for i in items).items())),
        "content_free_floor": ("every rule that reads only the four numbers scores at chance: "
                               "3rd-largest 0.2500, nearest-zero 0.2490, smallest 0.2490, "
                               "largest 0.2500. v1 let 3rd-largest score 0.6430 because gold "
                               "landed at value rank 2 on 643 of 1000 items (e1, 2026-09-05); "
                               "the option positions were shuffled but the option VALUES were "
                               "not chosen, and the header's chi-square could not see it."),
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
    # 5b. NO CONTENT-FREE RULE BEATS THE FLOOR. The assertion v1 did not have, and the
    #     reason it shipped a set where "pick the 3rd-largest of the four numbers" scored
    #     0.6430 (e1, 2026-09-05): the chi-square above measures the label POSITION, and a
    #     heuristic over the option VALUES is invisible to it. Every rule here reads only the
    #     four numbers, never the problem, so any of them above 0.30 is headroom the readout
    #     does not have. Run the battery, not one rule -- balancing rank alone left
    #     nearest-zero at 0.3500.
    battery = {
        "largest": max,
        "smallest": min,
        "2nd_largest": lambda o: sorted(o, reverse=True)[1],
        "3rd_largest": lambda o: sorted(o, reverse=True)[2],
        "nearest_zero": lambda o: min(o, key=abs),
        "farthest_zero": lambda o: max(o, key=abs),
        "only_negative": lambda o: ([v for v in o if v < 0] or [o[0]])[0],
        "median_high": lambda o: sorted(o)[2],
    }
    for name, fn in battery.items():
        hit = sum(1 for it in items if it["options"][it["label"]] == fn(it["options"])) / len(items)
        if hit > 0.30:
            fails.append(f"content-free rule {name!r} scores {hit:.4f} on the set, above the "
                         f"0.30 the readout's power assumes -- gold is distinguishable from "
                         f"the distractors by value alone")
    # 5c. GOLD'S VALUE RANK IS FLAT. The property 5b's 3rd_largest case depends on, asserted
    #     directly so a regression names the cause rather than only the symptom.
    rk = collections.Counter(it["value_rank"] for it in items)
    if max(rk.values()) / len(items) > 0.30:
        fails.append(f"gold's value rank is not flat: {dict(sorted(rk.items()))}")
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
    print(f"emit_novel_ops_4way selftest: {'PASS (9 worlds)' if not fails else f'{len(fails)} BUG(S)'}")
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
