#!/usr/bin/env python3
"""L4 programs: 8+ steps or a linear equation / reverse reasoning / constraints."""
import random
from fractions import Fraction
from mathcommon import GOODS, NAMES, num

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L4", name, fn))


# linear: kx + c = res → x  (reverse arithmetic, no x in equation lines)
def linear_reverse(rng):
    a = rng.randint(2, 9)
    x = rng.randint(3, 20)
    c = rng.randint(5, 60)
    res = a * x + c
    ins = f"某数的{a}倍再加上{c}等于{res}，求这个数。"
    lines = [
        f"这个数的{a}倍 = {res} - {c} = {res - c}",
        f"这个数 = {res - c} ÷ {a} = {x}",
    ]
    return ins, lines, x


_reg("linear_reverse", linear_reverse)


# linear: x ÷ a - c = b → x
def linear_div(rng):
    a = rng.randint(2, 9)
    x = a * rng.randint(4, 30)
    c = rng.randint(2, 20)
    b = x // a - c
    while b <= 0:
        c = rng.randint(1, max(1, x // a - 1))
        b = x // a - c
    ins = f"一个数除以{a}，再减去{c}，正好等于{b}。这个数是多少？"
    lines = [
        f"这个数除以{a}的商 = {b} + {c} = {b + c}",
        f"这个数 = {b + c} × {a} = {x}",
    ]
    return ins, lines, x


_reg("linear_div", linear_div)


# combined work rate → time (fractional)
def work_rate(rng):
    total = rng.randint(3, 9) * 20
    r1, r2 = rng.randint(3, 12) * 5, rng.randint(3, 12) * 5
    s = r1 + r2
    t = Fraction(total, s)
    lines = [
        f"{r1} + {r2} = {s}件/时",
        f"{total} ÷ {s} = {num(t)}时",
    ]
    ins = f"甲每小时做{r1}件零件，乙每小时做{r2}件，两人合做{total}件要几小时？"
    return ins, lines, t


_reg("work_rate", work_rate)


# reverse spending: final left ↔ original, three stages
def reverse_spend(rng):
    # original -> 用去一半 -> 又花固定 -> 剩 x
    left = rng.randint(5, 40)
    spend2 = rng.randint(3, 30)
    after_half = left + spend2
    orig = after_half * 2
    ins = f"小明用去零花钱的一半又花掉{spend2}元后，还剩{left}元，原来有多少元？"
    lines = [
        f"{left} + {spend2} = {after_half}元",
        f"{after_half} × 2 = {orig}元",
    ]
    return ins, lines, orig


_reg("reverse_spend", reverse_spend)


# two-step percent then reverse to original
def pct_reverse(rng):
    orig = rng.randint(200, 900) * 10
    p = rng.randint(10, 60)
    after = orig - Fraction(orig * p, 100)
    ins = f"一件商品降价{p}%后卖{num(after)}元，原价是多少元？"
    lines = [
        f"{num(after)} ÷ ({100 - p}/100) = {num(orig)}元",
    ]
    return ins, lines, orig


_reg("pct_reverse", pct_reverse)


# consecutive numbers sum
def consec_sum(rng):
    k = rng.randint(3, 6)
    start = rng.randint(2, 20)
    vals = list(range(start, start + k))
    s = sum(vals)
    ins = f"求自然数 {'、'.join(map(str, vals))} 的和是多少？"
    lines = [f"和 = {' + '.join(map(str, vals))} = {s}"]
    return ins, lines, s


_reg("consec_sum", consec_sum)


# multi-constraint: sum & difference → each number
def sum_diff(rng):
    a = rng.randint(20, 80)
    b = rng.randint(10, a - 10)
    s, d = a + b, a - b
    who = rng.choice(["甲", "乙"])
    A = Fraction(s + d, 2)
    B = Fraction(s - d, 2)
    ins = f"甲、乙两数的和是{s}，差是{d}。{who}是多少？"
    if who == "甲":
        lines = [f"两数和加差 = {s} + {d} = {s + d}", f"甲数 = {s + d} ÷ 2 = {num(A)}"]
        return ins, lines, A
    lines = [f"两数和减差 = {s} - {d} = {s - d}", f"乙数 = {s - d} ÷ 2 = {num(B)}"]
    return ins, lines, B


_reg("sum_diff", sum_diff)


# budget: 3 spends + split remainder (many steps)
def budget_deep(rng):
    total = rng.randint(400, 900) * 10
    s1 = total // rng.randint(4, 6)
    s2 = total // rng.randint(5, 7)
    s3 = total // rng.randint(6, 8)
    used = s1 + s2 + s3
    rest = total - used
    g = rng.randint(3, 6)
    each = Fraction(rest, g)
    ins = (f"一笔{total}元经费，第一项用{s1}元，第二项用{s2}元，第三项用{s3}元，"
           f"剩余的分给{g}个部门，每部门多少元？")
    lines = [
        f"{total} - {s1} = {total - s1}元",
        f"{total - s1} - {s2} = {total - s1 - s2}元",
        f"{total - s1 - s2} - {s3} = {rest}元",
        f"{rest} ÷ {g} = {num(each)}元",
    ]
    return ins, lines, each


_reg("budget_deep", budget_deep)


if __name__ == "__main__":
    rng = random.Random(4)
    from run_math_short import verify
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines}"
            ok += 1
    print(f"L4 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")