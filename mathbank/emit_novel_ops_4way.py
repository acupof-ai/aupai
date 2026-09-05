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
import math
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
            f"source set scores: {row['instruction']!r}"
        )
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
            f"that is still a misreading of the rule rather than padding with a random int."
        )
    return gold, cands


def _rank_of(gold, trio):
    """Gold's rank among the four option VALUES, largest first."""
    return sorted([gold] + [v for _n, v in trio], reverse=True).index(gold)


def _key_of(gold, trio, anchor=None):
    """The three content-free properties a subset fixes, as a tuple of MARGINALS.

    Each was found by running the heuristic battery against the previous build rather than
    assuming one fix covered the family:

      value rank      -- v1 left "pick the 3rd-largest" at 0.6430.
      nearest zero    -- balancing rank alone left "pick the nearest zero" at 0.3500.
      proximity rank  -- balancing both left "pick the option closest to 3*operands[0]" at
                         0.4190 (e1, 2026-09-05). Mechanism, not coincidence: gold is
                         3*a1 - 2*(the rest) + 1, so a1 enters with coefficient 3 while every
                         rung perturbs the LATER operands' contribution, leaving gold nearer
                         3*a1 than its distractors. Worse with chain length -- 0.5360 on
                         diamond_chain4 against 0.3020 on diamond_chain.
    """
    four = [gold] + [v for _n, v in trio]
    prox = 0 if anchor is None else sorted(range(len(four)), key=lambda i: abs(four[i] - anchor)).index(0)
    return _rank_of(gold, trio), min(four, key=abs) == gold, prox


def _choose(gold, cands, want_key, anchor):
    """The first 3-subset with the wanted key, else priority order.

    Priority order is the ladder's, so an item that cannot reach the wanted key keeps the
    old behaviour rather than being dropped.
    """
    for trio in itertools.combinations(cands, N_OPTIONS - 1):
        if _key_of(gold, trio, anchor) == want_key:
            return list(trio)
    return list(cands[: N_OPTIONS - 1])


def build_item(row, rng, want_key=None, anchor=None):
    """One 4-way item, optionally with gold placed at a given content-free key."""
    gold, cands = _candidates(row)
    picked = _choose(gold, cands, want_key, anchor) if want_key is not None else list(cands[: N_OPTIONS - 1])
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
        "proximity_rank": _key_of(gold, picked, anchor)[2] if anchor is not None else None,
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

    PER PROGRAM, NOT POOLED (v4, 2026-09-05). v3 balanced the three marginals over all 1000
    items at once and the battery was clean at every rule: max 0.2920 pooled. Split by
    program it was not. "Closest to 3*sum(operands)" scored 0.4980 on diamond_chain4 and
    0.0140 on diamond_chain -- z = +12.81 and z = -12.19 -- and the two halves cancelled to
    0.2560, which the pooled battery read as chance (e1, 2026-09-05). An aggregate cannot
    see a partition whose halves lean in opposite directions, and the pooled build CREATES
    that shape: the two programs reach different key sets, so a pooled target is met by
    over-filling one program's reachable keys against the other's.

    The assignment below is unchanged; it runs once per program. Cost: chain4's worst rule
    falls 0.4980 -> 0.2880 and chain's 0.3840 -> 0.3300.

    THE FLOOR IS 0.33, NOT 0.25, and that is structural. Gold can be the largest of four on
    only 23 of 500 diamond_chain items: a 3-operand chain has two rungs (sign_slip,
    wrong_order) that exceed gold by construction. Balancing chain's value rank as far as it
    goes gives [23,150,162,165], so a rank-reading rule keeps ~0.33. Closing that would need
    a rung that UNDERSHOOTS gold on 3-operand chains, and per fb's ruling of 2026-09-05 no
    further rungs are added: 0.330 / 0.288 is recorded as the measured prior floor and the
    operational floor is the no-injection control arm's own score.

    The battery detects a leak; it cannot certify the absence of the next one. The rules are
    all "gold is an affine functional of the operands and each distractor perturbs one term",
    an unbounded family -- rank, then proximity to 3*a1, then proximity to 3*sum were closed
    in that order and each fix exposed the next.
    """
    with open(src, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in list(fh)[1:]]
    rng = random.Random(SHUFFLE_SEED)
    reach = collections.defaultdict(list)
    for i, r in enumerate(rows):
        gold, cands = _candidates(r)
        anchor = 3 * _ops(r["instruction"])[0]
        keys = {_key_of(gold, t, anchor) for t in itertools.combinations(cands, N_OPTIONS - 1)}
        reach[r["program"]].append((len(keys), i, keys))
    want = {}
    for prog in sorted(reach):
        want.update(_assign(reach[prog]))
    return [
        build_item(r, rng, want_key=want[i], anchor=3 * _ops(r["instruction"])[0]) for i, r in enumerate(rows)
    ]


def _assign(group):
    """Greedy marginal-target assignment over one group of (n_keys, row_index, keys).

    MARGINALS, NOT CELLS. Targeting the 16 (value_rank, proximity_rank) cells evenly gives
    value_rank [196,208,256,340] and proximity_rank [340,295,243,122] -- worse on BOTH axes
    than balancing either alone (e1, 2026-09-05), because two cells are structurally
    unreachable: gold cannot be the 2nd-largest value AND the farthest from 3*a1, since
    being far from 3*a1 in either direction pushes it toward an extreme of the value order.
    An even-cell target is infeasible and chasing it distorts every marginal. Targeting the
    marginals directly and letting the joint fall out reaches flat on all three.

    This is the SECOND time an even-cell target was the wrong shape here -- the first left
    nearest-zero at 0.4950 by asking for 1/2 where chance is 1/4. Both are the same error:
    a target stronger than the constraint needs, imposed on a space that cannot supply it.
    """
    marg = [collections.Counter() for _ in range(3)]
    want = {}
    n = len(group)
    tgt = [
        {k: n / N_OPTIONS for k in range(N_OPTIONS)},
        {True: n / N_OPTIONS, False: n * (N_OPTIONS - 1) / N_OPTIONS},
        {k: n / N_OPTIONS for k in range(N_OPTIONS)},
    ]

    def _cost(key):
        # how far over target each marginal would be if this key were taken
        return sum((marg[j][key[j]] + 1 - tgt[j][key[j]]) for j in range(3))

    # Most-constrained items first: they claim their only option before the flexible ones
    # use it up.
    for _n, i, keys in sorted(group, key=lambda x: (x[0], x[1])):
        want[i] = min(keys, key=lambda k: (_cost(k), k))
        for j in range(3):
            marg[j][want[i][j]] += 1
    return want


def _label_hist(items):
    return dict(sorted(collections.Counter(i["label"] for i in items).items()))


def write(out=OUT, src=SRC):
    items = build(src)
    kinds = collections.Counter(k for i in items for k in i["option_kinds"] if k != "gold")
    by_prog = collections.defaultdict(list)
    for it in items:
        by_prog[it["program"]].append(it)
    header = {
        "_header": True,
        "family": "S",
        "kind": "test_4way",
        "n": len(items),
        "source": "data/probes/novel_ops/S_test.jsonl",
        "generator": "mathbank/emit_novel_ops_4way.py",
        "shuffle_seed": SHUFFLE_SEED,
        "readout": (
            "4-way likelihood: score each option, pick the argmax, compare to `label`. "
            "Score PER PROGRAM and never pool: the floor differs by program (0.33 "
            "diamond_chain, 0.29 diamond_chain4) and a pooled mean cancelled a "
            "z=+12.81 cell against a z=-12.19 one in v3. Chosen over exact match "
            "because exact match has no headroom off a 0 floor at 200M (e1's MDE, fb "
            "ruling 2026-09-05)."
        ),
        "distractors_are_diagnostic": (
            "each option is the answer under a different MISREADING of "
            "the rule, so which option a model picks says which rule it "
            "applied. option_kinds names them per item."
        ),
        "distractor_usage": dict(sorted(kinds.items())),
        "label_distribution": _label_hist(items),
        "value_rank_distribution": {
            p: dict(sorted(collections.Counter(i["value_rank"] for i in sub).items()))
            for p, sub in sorted(by_prog.items())
        },
        "proximity_rank_distribution": {
            p: dict(sorted(collections.Counter(i["proximity_rank"] for i in sub).items()))
            for p, sub in sorted(by_prog.items())
        },
        "content_free_floor": (
            "THE FLOOR IS NOT 0.25. 23 rules that read the options and the operands but never "
            "apply the rule are measured per program; the highest is 0.330 on diamond_chain "
            "(`smallest`) and 0.288 on diamond_chain4 (`closest_sum`). Both are at the feasible "
            "edge: gold can be the largest of four on only 23 of diamond_chain's 500 items, "
            "because sign_slip and wrong_order both exceed gold and a 3-operand chain leaves too "
            "little room below it, so the best reachable value-rank marginal is [23,159,159,159] "
            "-> 0.318 (e1, 2026-09-05). Closing that would need a ladder rung that undershoots "
            "gold on 3-operand chains; fb ruled 2026-09-05 that no further rungs are added. "
            "Four leaks were closed in turn, each found by running the battery against the "
            "previous build: v1's 3rd-largest 0.6430 (gold at value rank 2 on 643/1000), v2's "
            "nearest-zero 0.3500, v2's closest-to-3*operands[0] 0.4190, and v3's closest-to-"
            "3*sum(operands) 0.4980 on diamond_chain4 against 0.0140 on diamond_chain, which "
            "pooled to 0.2560 and read as clean. e1 (lessons-58) found the first, third and "
            "fourth. The battery DETECTS a leak and cannot certify the absence of the next one: "
            "the rules are all 'gold is an affine functional of the operands and each distractor "
            "perturbs one term', an unbounded family."
        ),
        "content_free_max_by_program": {
            p: max(
                (
                    sum(1 for it in sub if it["options"][it["label"]] == fn(it["options"], it["operands"]))
                    / len(sub),
                    name,
                )
                for name, fn in _battery().items()
            )
            for p, sub in sorted(by_prog.items())
        },
        "absence_basis": (
            "inherited from the source set: the operator, its rule and its phrasing "
            "were invented 2026-09-05, after every corpus in the mix was built. "
            "facts/contamination.json#cont.novel_ops_frozen_sets"
        ),
    }
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i in items:
            fh.write(json.dumps(i, ensure_ascii=False) + "\n")
    return len(items), header


def _battery():
    """Content-free rules: each reads the four OPTIONS and the OPERANDS and never applies
    the operator. One definition, read by both the selftest and write()'s header, so a rule
    added after a leak is found is measured in the artifact too.

    THE BATTERY DOES NOT GATE diamond_chain, and adding a rule to it is not a request to
    close that rule. `nearest_mean_of_options` scores 0.3640 on diamond_chain (e1,
    2026-09-05, reproduced here) and is in this dict so the recorded maximum is the true
    one -- the maximum over the rules that happen to be here is otherwise a number that
    reads as a floor and is not one. Balancing it would move the displaced mass to whatever
    statistic reads it next: gold cannot be the largest of four on more than 23 of chain's
    500 items, so SOME statistic always finds where the missing rank-0 mass went. Its pick
    profile shows the mechanism -- it selects value rank 1 on 354 items and rank 2 on 146,
    never rank 0 or 3, and chain's gold sits at [23,150,162,165] so ranks 1-2 hold 312/500
    against a flat 250. On chain4, where the rank marginal IS flat, the same rule scores
    0.1840. Not a new leak family: the same infeasibility read by a spread statistic.
    """
    return {
        "largest": lambda o, p: max(o),
        "smallest": lambda o, p: min(o),
        "2nd_largest": lambda o, p: sorted(o, reverse=True)[1],
        "3rd_largest": lambda o, p: sorted(o, reverse=True)[2],
        "nearest_zero": lambda o, p: min(o, key=abs),
        "farthest_zero": lambda o, p: max(o, key=abs),
        "only_negative": lambda o, p: ([v for v in o if v < 0] or [o[0]])[0],
        "median_high": lambda o, p: sorted(o)[2],
        "nearest_mean_of_options": lambda o, p: min(o, key=lambda v: abs(v - sum(o) / len(o))),
        "farthest_mean_of_options": lambda o, p: max(o, key=lambda v: abs(v - sum(o) / len(o))),
        "closest_median_of_options": lambda o, p: min(o, key=lambda v: abs(v - sorted(o)[1])),
        "fewest_digits": lambda o, p: min(o, key=lambda v: (len(str(abs(v))), v)),
        "most_digits": lambda o, p: max(o, key=lambda v: (len(str(abs(v))), v)),
        "nonneg_then_smallest": lambda o, p: min([v for v in o if v >= 0] or list(o)),
        "closest_3a1": lambda o, p: min(o, key=lambda v: abs(v - 3 * p[0])),
        "closest_a1": lambda o, p: min(o, key=lambda v: abs(v - p[0])),
        "closest_a2": lambda o, p: min(o, key=lambda v: abs(v - p[1])),
        "closest_alast": lambda o, p: min(o, key=lambda v: abs(v - p[-1])),
        "closest_sum": lambda o, p: min(o, key=lambda v: abs(v - sum(p))),
        "closest_3sum": lambda o, p: min(o, key=lambda v: abs(v - 3 * sum(p))),
        "closest_mean": lambda o, p: min(o, key=lambda v: abs(v - sum(p) / len(p))),
        "closest_3a1_m2a2": lambda o, p: min(o, key=lambda v: abs(v - (3 * p[0] - 2 * p[1]))),
        "closest_3a1_m2a2_p1": lambda o, p: min(o, key=lambda v: abs(v - (3 * p[0] - 2 * p[1] + 1))),
        "closest_2a1": lambda o, p: min(o, key=lambda v: abs(v - 2 * p[0])),
        "closest_msum": lambda o, p: min(o, key=lambda v: abs(v + sum(p))),
        "closest_prod": lambda o, p: min(o, key=lambda v: abs(v - p[0] * p[1])),
        "closest_prod_all": lambda o, p: min(o, key=lambda v: abs(v - math.prod(p))),
        "closest_m2alast": lambda o, p: min(o, key=lambda v: abs(v + 2 * p[-1])),
        "even_else_first": lambda o, p: ([v for v in o if v % 2 == 0] or list(o))[0],
    }


def _selftest():
    fails = []
    items = build()
    if len(items) != 1000:
        fails.append(f"{len(items)} items, want 1000")
    # 1. EVERY DISTRACTOR IS ITS READING'S FOLD, checked against an independent recomputation.
    #    This is the assertion fb asked for: without it "option_kinds" is a label nobody verified.
    by_name = dict(LADDER)
    for it in items:
        for kind, val in zip(it["option_kinds"], it["options"], strict=True):
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
    # 5b. NO CONTENT-FREE RULE BEATS THE FLOOR, PER PROGRAM. The assertion v1 did not have,
    #     and the reason it shipped a set where "pick the 3rd-largest of the four numbers"
    #     scored 0.6430 (e1, 2026-09-05): the chi-square above measures the label POSITION,
    #     and a heuristic over the option VALUES is invisible to it. Every rule here reads
    #     only the four numbers, never the problem. A rule may read the OPTIONS and the
    #     OPERANDS; what it may not do is apply the rule.
    #
    #     SPLIT BY PROGRAM, and that is the whole point of this version. v3 passed this
    #     battery pooled at max 0.2920 while "closest to 3*sum" sat at 0.4980 on
    #     diamond_chain4 and 0.0140 on diamond_chain: z = +12.81 and z = -12.19, cancelling
    #     to 0.2560 (e1, 2026-09-05). A pooled mean cannot see a partition whose halves lean
    #     opposite ways, so the aggregate assertion certified a set that was half broken.
    #
    #     THE THRESHOLDS ARE THE MEASURED FLOORS, NOT 0.30, AND diamond_chain's IS 0.3640.
    #     Gold is the largest of four on only 23 of chain's 500 items, because two rungs
    #     exceed it by construction on a 3-operand chain, so its rank marginal cannot be
    #     flatter than [23,159,159,159]. The displaced mass lands in the middle ranks and a
    #     spread statistic reads it there: `nearest_mean_of_options` scores 0.3640 on chain
    #     and 0.1840 on chain4, where the marginal IS flat (e1, 2026-09-05, reproduced).
    #     Balancing that rule too would move the mass to whatever statistic reads it next --
    #     rank, prox-3a1, prox-3sum and this one were each closed in turn and each fix
    #     exposed the next, which is the non-convergence the header names. fb ruled
    #     2026-09-05: no further rungs, no v5.
    #
    #     SO THIS ASSERTION IS A REGRESSION GUARD, NOT A CERTIFICATE. The thresholds sit a
    #     little above each program's measured maximum so a rebuild that makes the set WORSE
    #     goes red, and nothing here says a rule under the threshold is absent -- the battery
    #     samples an unbounded family. The operational floor is the no-injection control
    #     arm's own score, which absorbs every member including the unenumerated ones.
    FLOOR = {"diamond_chain": 0.40, "diamond_chain4": 0.32}
    battery = _battery()
    by_prog = collections.defaultdict(list)
    for it in items:
        by_prog[it["program"]].append(it)
    for prog, sub in sorted(by_prog.items()):
        floor = FLOOR.get(prog)
        if floor is None:
            fails.append(
                f"program {prog!r} has no recorded content-free floor -- a new "
                f"program needs its own measured ceiling before it can be scored"
            )
            continue
        for name, fn in battery.items():
            hit = sum(
                1 for it in sub if it["options"][it["label"]] == fn(it["options"], it["operands"])
            ) / len(sub)
            if hit > floor:
                fails.append(
                    f"content-free rule {name!r} scores {hit:.4f} on {prog} "
                    f"({len(sub)} items), above its {floor:.2f} floor -- gold is "
                    f"distinguishable from the distractors by value alone. Pooled "
                    f"means hide this: v3 read 0.2920 overall with 0.4980 here."
                )
    # 5c. GOLD'S VALUE RANK IS FLAT, PER PROGRAM. The property 5b's 3rd_largest case depends
    #     on, asserted directly so a regression names the cause rather than only the symptom.
    #     Per program for 5b's reason, and at each program's own reachable bound: chain can
    #     put gold at rank 0 on 23 of 500 items, so its rank marginal cannot be flatter than
    #     [23,159,159,159] and a 0.30 threshold would be a permanent red.
    for prog, sub in sorted(by_prog.items()):
        for field in ("value_rank", "proximity_rank"):
            c = collections.Counter(it[field] for it in sub)
            if max(c.values()) / len(sub) > 0.36:
                fails.append(f"gold's {field} is not flat on {prog}: {dict(sorted(c.items()))}")
    # 6. DETERMINISM: the same seed twice is the same set, or a rebuild silently rescores.
    if [i["options"] for i in build()] != [i["options"] for i in items]:
        fails.append("build() is not reproducible at its own seed")
    # 7. THE DISTINCTNESS CHECK MUST HAVE POWER: an item built from a ladder whose rungs all
    #    return gold must refuse, not emit three copies of it.
    saved = LADDER[:]
    try:
        LADDER[:] = [(n, (lambda o: _fold_r(o, _once))) for n, _f in saved]
        try:
            build_item(
                {
                    "instruction": items[0]["instruction"],
                    "program": "diamond_chain",
                    "answer": items[0]["answer"],
                },
                random.Random(0),
            )
            fails.append("build_item accepted a ladder that produces no distinct distractor")
        except SystemExit:
            pass
    finally:
        LADDER[:] = saved
    # 8. THE PER-PROGRAM BATTERY MUST CATCH WHAT THE POOLED ONE MISSED. World 5b's threshold
    #    is only worth having if a pooled-clean set fails it, so the world is v3's actual
    #    build shape -- one _assign over all 1000 rows instead of one per program -- which
    #    passed the pooled battery at max 0.2920 and hid closest_3sum at 0.4980 on chain4.
    #    Rebuilt here from the real source rows by the real functions, not hand-written: the
    #    only difference from build() is the grouping, which is exactly the defect.
    with open(SRC, encoding="utf-8") as fh:
        _rows = [json.loads(line) for line in list(fh)[1:]]
    _reach = []
    for _i, _r in enumerate(_rows):
        _g, _c = _candidates(_r)
        _a = 3 * _ops(_r["instruction"])[0]
        _reach.append(
            (len(_c), _i, {_key_of(_g, _t, _a) for _t in itertools.combinations(_c, N_OPTIONS - 1)})
        )
    _want = _assign(_reach)
    _rng = random.Random(SHUFFLE_SEED)
    _pooled = [
        build_item(_r, _rng, want_key=_want[_i], anchor=3 * _ops(_r["instruction"])[0])
        for _i, _r in enumerate(_rows)
    ]
    _sub = [it for it in _pooled if it["program"] == "diamond_chain4"]
    _hit = sum(
        1
        for it in _sub
        if it["options"][it["label"]] == min(it["options"], key=lambda v: abs(v - 3 * sum(it["operands"])))
    ) / len(_sub)
    if _hit <= FLOOR["diamond_chain4"]:
        fails.append(
            f"world 8 tests nothing: the pooled build scores {_hit:.4f} for "
            f"closest_3sum on diamond_chain4, at or under the {FLOOR['diamond_chain4']:.2f} "
            f"floor, so 5b would pass it and the per-program split is unverified"
        )
    _pooled_all = sum(
        1
        for it in _pooled
        if it["options"][it["label"]] == min(it["options"], key=lambda v: abs(v - 3 * sum(it["operands"])))
    ) / len(_pooled)
    if _pooled_all > 0.30:
        fails.append(
            f"world 8 tests nothing: the pooled build's closest_3sum is "
            f"{_pooled_all:.4f} POOLED, so v3's aggregate battery would have caught "
            f"it too and this world does not isolate the per-program split"
        )
    for f in fails:
        print(f"BUG {f}", file=sys.stderr)
    print(f"emit_novel_ops_4way selftest: {'PASS (11 worlds)' if not fails else f'{len(fails)} BUG(S)'}")
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
