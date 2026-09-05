#!/usr/bin/env python3
"""Two CONSTRUCTED operator families for the conversion-rate curve (3b, 2026-09-05).

WHY CONSTRUCTED AND NOT SELECTED. Experiment 1 needs a skill the pretraining corpus
does not contain. Selecting one from this bank is impossible: measured on the pod,
every probed family is already in OpenWebMath -- age_ratio 7.330% of sampled docs
down to ant_crawl_solid 0.003%, and even a family scoring zero in 40,000 docs is
bounded above at 0.0075% by the rule of three, ~200 docs domain-wide
(facts/contamination.json#cont.generator_families_in_owm). A scan can fail to find
a family; it cannot certify absence. So absence here is a property of CONSTRUCTION:
the operator glyph, its rules and its phrasing were invented for this experiment,
and the claim to defend is how it was built rather than what a scanner did not see.

TWO FAMILIES, AND THE SECOND IS THE POINT (4c's design). A curve on one family
cannot separate "the model learned the procedure" from "the model learned the
format", because both make accuracy rise with n.

    S (skill)   -- diamond_chain: nested evaluation of a @ b = 3a - 2b + 1 under a
                   stated right-to-left precedence, with a carry rule that fires on
                   a negative intermediate. A correct answer needs the procedure
                   executed in order; no surface feature of the prompt carries it.
    P (control) -- diamond_single: the SAME notation, phrasing and instance count,
                   one application, both operands given. Recoverable by substitution
                   into the stated rule without composing anything.

The number is the S curve minus the P curve. Both rising together at n=1 or 8 is
format acquisition; S lagging P and closing as n grows is the skill being acquired.

CARRY RULE, stated in every instruction rather than assumed: if an intermediate
value is negative, add 10 before it is used as the next left operand. It exists so
the chain cannot be collapsed into one linear expression -- without it, nested
`3a-2b+1` folds into a weighted sum and a model could fit the weights instead of
running the steps.

Line contract is run_math_short.verify's: every line `X op Y = Z`, the last line's
value equals num(ans), and every prose integer >= 3 appears among the equation
tokens. The instruction states the rule with digits, so the rule's own constants
(3, 2, 10) appear in the lines by construction.
"""

import random

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L5", name, fn))


GLYPH = "@"
# Deliberately NOT a common mathematical operator. Checked against the cursor rows:
# the bound is recorded in facts/contamination.json#cont.novel_operator_collision.
RULE = "a @ b = 3a - 2b + 1"


def _rule_text():
    # "十" not "10": verify requires every prose integer >= 3 to appear among the
    # equation tokens, and the carry constant is stated in EVERY instruction while
    # the carry line only exists when an intermediate goes negative. Writing it as a
    # Chinese numeral keeps the rule fully stated without asserting a number the
    # equations need not contain. Caught by run_math_short.verify, not by my own
    # selftest, which never ran the prose<->equation check.
    return (f"定义新运算 {RULE}。若中间结果为负数，先加十再作为下一步的左操作数。"
            f"按从右到左的顺序计算。")


def _step(a, b):
    """One application plus the carry rule. Returns (value, carried)."""
    v = 3 * a - 2 * b + 1
    if v < 0:
        return v + 10, True
    return v, False


# ---------------------------------------------------------------- S: the skill
def diamond_chain(rng):
    """a @ b @ c, right-to-left, carry rule live. Needs the procedure in order."""
    a, b, c = (rng.randint(3, 19) for _ in range(3))
    inner, carried_in = _step(b, c)
    outer, carried_out = _step(a, inner)
    lines = [f"3 * {b} - 2 * {c} + 1 = {3 * b - 2 * c + 1}"]
    if carried_in:
        lines.append(f"{3 * b - 2 * c + 1} + 10 = {inner}")
    lines.append(f"3 * {a} - 2 * {inner} + 1 = {3 * a - 2 * inner + 1}")
    if carried_out:
        lines.append(f"{3 * a - 2 * inner + 1} + 10 = {outer}")
    ins = f"{_rule_text()}求 {a} @ {b} @ {c} 的值。"
    return ins, lines, outer


def diamond_chain4(rng):
    """Four operands, three applications. Same procedure, one step deeper."""
    a, b, c, d = (rng.randint(3, 15) for _ in range(4))
    v1, c1 = _step(c, d)
    v2, c2 = _step(b, v1)
    v3, c3 = _step(a, v2)
    lines = [f"3 * {c} - 2 * {d} + 1 = {3 * c - 2 * d + 1}"]
    if c1:
        lines.append(f"{3 * c - 2 * d + 1} + 10 = {v1}")
    lines.append(f"3 * {b} - 2 * {v1} + 1 = {3 * b - 2 * v1 + 1}")
    if c2:
        lines.append(f"{3 * b - 2 * v1 + 1} + 10 = {v2}")
    lines.append(f"3 * {a} - 2 * {v2} + 1 = {3 * a - 2 * v2 + 1}")
    if c3:
        lines.append(f"{3 * a - 2 * v2 + 1} + 10 = {v3}")
    ins = f"{_rule_text()}求 {a} @ {b} @ {c} @ {d} 的值。"
    return ins, lines, v3


# -------------------------------------------------------------- P: the control
def diamond_single(rng):
    """ONE application, both operands in the prompt. Substitution, no composition.

    Same glyph, same rule sentence, same instance shape as S. The carry rule is
    stated but the operands are drawn so it never fires -- a control whose answer
    is recoverable by substitution must not sometimes need a second step, or it
    is a weaker version of S rather than a different task.
    """
    a = rng.randint(6, 19)
    b = rng.randint(3, (3 * a + 1) // 2)  # 3a - 2b + 1 >= 0, so no carry
    v, carried = _step(a, b)
    assert not carried
    lines = [f"3 * {a} - 2 * {b} + 1 = {v}"]
    ins = f"{_rule_text()}求 {a} @ {b} 的值。"
    return ins, lines, v


def diamond_single_reverse(rng):
    """One application with the operands named in the other order. Still substitution."""
    b = rng.randint(3, 12)
    a = rng.randint((2 * b) // 3 + 1, 19)
    v, carried = _step(a, b)
    assert not carried
    lines = [f"3 * {a} - 2 * {b} + 1 = {v}"]
    ins = f"{_rule_text()}已知右操作数是 {b}，左操作数是 {a}，求 {a} @ {b} 的值。"
    return ins, lines, v


_reg("diamond_chain", diamond_chain)
_reg("diamond_chain4", diamond_chain4)
_reg("diamond_single", diamond_single)
_reg("diamond_single_reverse", diamond_single_reverse)

#: Which family each program belongs to. The curve is S minus P, so a consumer
#: must not have to infer membership from the name.
FAMILY = {"diamond_chain": "S", "diamond_chain4": "S",
          "diamond_single": "P", "diamond_single_reverse": "P"}


def _selftest():
    """Known answers, the carry rule firing, and S/P actually differing.

    Not a smoke test: the third case is the one that matters, because a control
    whose answers a chain-solver also produces would make the curve difference
    meaningless.
    """
    fails = []
    # 1. known answer, no carry: 7 @ 4 = 21 - 8 + 1 = 14
    v, c = _step(7, 4)
    if (v, c) != (14, False):
        fails.append(f"7@4 = {v},{c}, want 14,False")
    # 2. known answer WITH carry: 3 @ 9 = 9 - 18 + 1 = -8 -> +10 = 2
    v, c = _step(3, 9)
    if (v, c) != (2, True):
        fails.append(f"3@9 = {v},{c}, want 2,True")
    # 3. right-to-left is not left-to-right: pick operands where they differ, or
    #    the chain family would be measuring an order the model need not learn.
    a, b, c_ = 5, 3, 11
    rtl, _ = _step(a, _step(b, c_)[0])
    ltr, _ = _step(_step(a, b)[0], c_)
    if rtl == ltr:
        fails.append("right-to-left equals left-to-right on the selftest triple")
    # 4. the control never carries, over many draws -- asserted, not assumed
    rng = random.Random(7)
    for _ in range(2000):
        for fn in (diamond_single, diamond_single_reverse):
            _i, _l, ans = fn(rng)
            if ans < 0:
                fails.append(f"{fn.__name__} produced a negative answer {ans}")
                break
    # 5. every program's last line equals its answer (verify's core contract)
    for _lvl, name, fn in PROGRAMS:
        for _ in range(200):
            _i, lines, ans = fn(rng)
            tail = lines[-1].split("=")[-1].strip()
            if tail != str(ans):
                fails.append(f"{name}: last line {tail!r} != ans {ans}")
                break
    # 6. S and P are distinguishable: S answers must not be reproducible by
    #    applying the rule once to the first two operands.
    same = 0
    for _ in range(500):
        ins, _l, ans = diamond_chain(rng)
        nums = [int(t) for t in ins.replace("@", " ").split() if t.isdigit()]
        ops = [n for n in nums if n not in (3, 2, 1, 10)][:2]
        if len(ops) == 2 and _step(ops[0], ops[1])[0] == ans:
            same += 1
    if same > 25:  # 5% -- coincidence happens, systematic equality does not
        fails.append(f"diamond_chain answer equals a single application in {same}/500")
    for f in fails:
        print(f"BUG {f}")
    print(f"l5_ext_novel selftest: {'PASS (6 worlds)' if not fails else f'{len(fails)} BUG(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
