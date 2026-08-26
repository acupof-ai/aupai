#!/usr/bin/env python3
"""L4 ext2: reverse reasoning, constraints, rates, ages, fractions — 62 families.

Every program: fn(rng) -> (instruction, lines, ans). Lines solve FORWARD from
givens to the asked value (no x variable). All exact arithmetic via Fraction.
"""
import random
from fractions import Fraction
from mathcommon import (ANIMALS, FOOD, FRUITS, GOODS, NAMES, PLACE, STATIONERY,
                        UNIT_FRUIT, UNIT_N, UNIT_ZHI, num)

PROGRAMS = []

_TAILS = [
    "请你把算式和结果都写出来。",
    "请把你的计算过程完整地写出来。",
    "请你列算式计算，并写出最后结果。",
    "请把计算过程和结果都写出来。",
    "请你列式计算，并把结果写出来。",
]


def _reg(name, fn):
    def wrapped(rng):
        ins, lines, ans = fn(rng)
        return ins + rng.choice(_TAILS), lines, ans
    PROGRAMS.append(("L4", name, wrapped))


# 1. 某数的k倍减去c等于r → 某数
def rev_kx_minus_c(rng):
    k = rng.randint(3, 9)
    x = rng.randint(4, 30)
    c = rng.randint(5, 40)
    r = k * x - c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}在做数学题时遇到一个数，它的{k}倍减去{c}等于{r}，请你帮他算出这个数是多少。",
        f"数学兴趣课上，老师出了一道题：一个数的{k}倍减去{c}，所得的差正好是{r}。这个数是多少？你会算吗？",
        f"{name}心里想了一个数，把这个数乘{k}再减去{c}，得数是{r}。请你猜一猜，{name}想的这个数是多少？",
    ])
    lines = [
        f"这个数的{k}倍 = {r} + {c} = {r + c}",
        f"这个数 = {r + c} ÷ {k} = {x}",
    ]
    return ins, lines, x


_reg("rev_kx_minus_c", rev_kx_minus_c)


# 2. 某数的一半加上c等于r → 某数
def rev_half_plus_c(rng):
    x = rng.randint(5, 30) * 2
    c = rng.randint(3, 30)
    r = x // 2 + c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数的一半加上{c}等于{r}。{name}想了半天没算出来，请你帮他算一算这个数是多少。",
        f"{name}在作业本上看到这样一道题：把一个数除以2，再加上{c}，得数是{r}。这个数是多少？请你列式算一算。",
        f"智慧老人给{name}出了一道题：某数的一半与{c}的和是{r}，求这个数。你能帮{name}算出来吗？",
    ])
    lines = [
        f"这个数的一半 = {r} - {c} = {x // 2}",
        f"这个数 = {x // 2} × 2 = {x}",
    ]
    return ins, lines, x


_reg("rev_half_plus_c", rev_half_plus_c)


# 3. 某数除以a再加c等于b → 某数
def rev_div_plus_c(rng):
    a = rng.randint(2, 9)
    q = rng.randint(5, 30)
    c = rng.randint(2, 15)
    x = a * q
    b = q + c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上，老师在黑板上写了一道题：一个数除以{a}，再加上{c}，正好等于{b}。{name}没算出来，你能帮他算出这个数吗？",
        f"{name}做练习时遇到一道题：把一个数除以{a}后又加上{c}，得数是{b}。这个数是多少？请你列式算一算。",
        f"爸爸给{name}出了一道思考题：某数除以{a}的商加上{c}等于{b}，求这个数。你会算吗？动手算一算吧。",
    ])
    lines = [
        f"这个数除以{a}的商 = {b} - {c} = {q}",
        f"这个数 = {q} × {a} = {x}",
    ]
    return ins, lines, x


_reg("rev_div_plus_c", rev_div_plus_c)


# 4. 某数加上c后再乘a等于r → 某数
def rev_add_then_mult(rng):
    a = rng.randint(3, 9)
    x = rng.randint(5, 30)
    c = rng.randint(4, 20)
    r = a * (x + c)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道思考题：一个数加上{c}后，再乘{a}，结果是{r}。{name}想了很久，你能帮他算出这个数吗？",
        f"{name}在练习册上看到一道题：把一个数加上{c}，再乘{a}，得数是{r}。这个数是多少？请你列式算一算。",
        f"爷爷给{name}出了一道题：某数与{c}的和的{a}倍是{r}，求这个数。你能算出来吗？快动手试一试。",
    ])
    lines = [
        f"这个数加{c}的和 = {r} ÷ {a} = {x + c}",
        f"这个数 = {x + c} - {c} = {x}",
    ]
    return ins, lines, x


_reg("rev_add_then_mult", rev_add_then_mult)


# 5. 某数减去c后再除以a等于b → 某数
def rev_sub_then_div(rng):
    a = rng.randint(2, 9)
    b = rng.randint(4, 25)
    c = rng.randint(3, 20)
    x = a * b + c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上，老师出了一道逆向思考题：一个数减去{c}后，再除以{a}，正好等于{b}。{name}算不出来，你能帮帮他吗？",
        f"{name}做家庭作业时遇到一道题：把一个数减去{c}，再除以{a}，得数是{b}。这个数是多少？请你列式算一算。",
        f"爸爸给{name}出了一道题：某数减去{c}的差除以{a}等于{b}，求这个数。你会算吗？请算给{name}看一看。",
    ])
    lines = [
        f"这个数减{c}的差 = {b} × {a} = {a * b}",
        f"这个数 = {a * b} + {c} = {x}",
    ]
    return ins, lines, x


_reg("rev_sub_then_div", rev_sub_then_div)


# 6. 某数的k1倍与k2倍之和是s → 某数
def multi_coeff_sum(rng):
    k1 = rng.randint(2, 6)
    k2 = rng.randint(2, 6)
    for _ in range(50):
        if k2 != k1:
            break
        k2 = rng.randint(2, 6)
    else:
        k2 = k1 + 1 if k1 < 6 else k1 - 1
    x = rng.randint(5, 30)
    s = (k1 + k2) * x
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数的{k1}倍与它的{k2}倍相加，和是{s}。{name}想了半天，你能帮他算出这个数吗？",
        f"{name}在练习册上看到一道题：把一个数分别乘{k1}和乘{k2}，再把两个积相加，得数是{s}。这个数是多少？请列式算一算。",
        f"智慧老人给{name}出了一道题：某数的{k1}倍加上它的{k2}倍等于{s}，求这个数。你能算出来吗？快动手试一试。",
    ])
    lines = [
        f"倍数之和 = {k1} + {k2} = {k1 + k2}",
        f"这个数 = {s} ÷ {k1 + k2} = {x}",
    ]
    return ins, lines, x


_reg("multi_coeff_sum", multi_coeff_sum)


# 7. 某数的k1倍比它的k2倍多d → 某数
def multi_coeff_diff(rng):
    k2 = rng.randint(2, 5)
    k1 = k2 + rng.randint(1, 3)
    x = rng.randint(5, 40)
    d = (k1 - k2) * x
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数的{k1}倍比它的{k2}倍多{d}。{name}没有算出来，你能帮他算出这个数是多少吗？",
        f"{name}在作业本上看到一道题：把一个数分别乘{k1}和乘{k2}，第一个积比第二个积多{d}。这个数是多少？请列式算一算。",
        f"爷爷给{name}出了一道思考题：某数的{k1}倍减去它的{k2}倍等于{d}，求这个数。你会算吗？快动手算一算。",
    ])
    lines = [
        f"倍数之差 = {k1} - {k2} = {k1 - k2}",
        f"这个数 = {d} ÷ {k1 - k2} = {x}",
    ]
    return ins, lines, x


_reg("multi_coeff_diff", multi_coeff_diff)


# 8. 合做时间（分数）
def work_together_frac(rng):
    total = rng.randint(40, 120)
    r1 = rng.randint(12, 40)
    r2 = rng.randint(12, 40)
    s = r1 + r2
    t = Fraction(total, s)
    ins = rng.choice([
        f"工厂要加工一批零件共{total}件，甲师傅每小时做{r1}件，乙师傅每小时做{r2}件。两人合做，几小时可以完成这批零件？",
        f"车间主任把加工{total}件零件的任务交给甲、乙两人，甲每小时做{r1}件，乙每小时做{r2}件。两人合做几小时能做完？",
        f"一批零件共{total}件，甲每小时能做{r1}件，乙每小时能做{r2}件。两人一起合做，需要几小时才能完成任务？",
    ])
    lines = [
        f"{r1} + {r2} = {s}件/时",
        f"{total} ÷ {s} = {num(t)}时",
    ]
    return ins, lines, t


_reg("work_together_frac", work_together_frac)


# 9. 合做时间已知，求一人效率
def work_together_reverse(rng):
    t = rng.randint(2, 6)
    r1 = rng.randint(15, 40)
    r2 = rng.randint(12, 35)
    total = t * (r1 + r2)
    ins = rng.choice([
        f"工厂要加工{total}件零件，甲、乙两位师傅合做{t}小时完成。已知甲每小时做{r1}件，乙每小时做多少件？你能算出来吗？",
        f"一批零件共{total}件，甲、乙两人合做{t}小时正好完成。甲每小时做{r1}件，乙每小时做多少件？请你列式算一算。",
        f"车间把{total}件零件交给甲、乙两人，他们合做{t}小时完成了任务。甲每小时做{r1}件，乙每小时做多少件？",
    ])
    lines = [
        f"{total} ÷ {t} = {r1 + r2}件/时",
        f"{r1 + r2} - {r1} = {r2}件/时",
    ]
    return ins, lines, r2


_reg("work_together_reverse", work_together_reverse)


# 10. 甲先做再合做，求剩余时间
def work_first_then_together(rng):
    r1 = rng.randint(15, 45)
    r2 = rng.randint(15, 45)
    a = rng.randint(1, 3)
    rest = rng.randint(40, 200)
    total = a * r1 + rest
    s = r1 + r2
    t = Fraction(rest, s)
    ins = rng.choice([
        f"工厂要加工{total}件零件，甲每小时做{r1}件，乙每小时做{r2}件。甲先单独做{a}小时，剩下的由两人合做，还要几小时才能完成？",
        f"一批零件共{total}件，甲每小时做{r1}件，乙每小时做{r2}件。甲先做{a}小时后，乙也加入一起做，还要几小时才能做完？",
        f"车间要完成{total}件零件，甲每小时做{r1}件，乙每小时做{r2}件。甲单独做{a}小时后两人合做，还需要几小时完成？",
    ])
    lines = [
        f"{r1} × {a} = {a * r1}件",
        f"{total} - {a * r1} = {rest}件",
        f"{r1} + {r2} = {s}件/时",
        f"{rest} ÷ {s} = {num(t)}时",
    ]
    return ins, lines, t


_reg("work_first_then_together", work_first_then_together)


# 11. 用去一半还多c元，剩left元 → 原
def reverse_spend_half_more(rng):
    left = rng.randint(10, 60)
    c = rng.randint(3, 20)
    after_half = left + c
    orig = after_half * 2
    name = rng.choice(NAMES)
    obj = rng.choice(STATIONERY + FOOD)
    ins = rng.choice([
        f"{name}去文具店买{obj}，用去零花钱的一半还多{c}元，现在口袋里还剩{left}元。他原来有多少元零花钱？你能算出来吗？",
        f"{name}攒了一笔零花钱，买{obj}花了一半再多{c}元，还剩{left}元。他原来有多少元零花钱？请你列式算一算。",
        f"周末{name}用零花钱的一半还多{c}元买了{obj}，回家数了数还剩{left}元。他原来有多少元零花钱？快帮他算一算。",
    ])
    lines = [
        f"{left} + {c} = {after_half}元",
        f"{after_half} × 2 = {orig}元",
    ]
    return ins, lines, orig


_reg("reverse_spend_half_more", reverse_spend_half_more)


# 12. 先用去1/3，再用去余下的一半，剩left → 原
def reverse_spend_fraction_chain(rng):
    k = rng.randint(6, 30)
    left = 2 * k
    after_first = 4 * k
    orig = 6 * k
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}攒了一笔零花钱，第一周用去全部的1/3，第二周用去余下的一半，这时还剩{left}元。他原来有多少元零花钱？你能算出来吗？",
        f"妈妈给了{name}一笔钱，他先用去总数的1/3，又用去余下的一半，最后还剩{left}元。这笔钱原来有多少元？请列式算一算。",
        f"{name}的零花钱，第一周花了全部的1/3，第二周花了余下的一半，数一数还剩{left}元。他原来有多少元零花钱？快帮他算一算。",
    ])
    lines = [
        f"{left} × 2 = {after_first}元",
        f"{after_first} × 3 ÷ 2 = {orig}元",
    ]
    return ins, lines, orig


_reg("reverse_spend_fraction_chain", reverse_spend_fraction_chain)


# 13. 降价p%后卖a元 → 原价
def pct_reverse_down(rng):
    k = rng.randint(20, 90)
    orig = 100 * k
    p = rng.choice([10, 15, 20, 25, 30, 40])
    after = k * (100 - p)
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店搞促销，一件{obj}降价{p}%后售价是{after}元。{rng.choice(NAMES)}想知道这件{obj}的原价是多少元，你能帮他算一算吗？",
        f"商场把一件{obj}的价格下调{p}%出售，现在卖{after}元。这件{obj}的原价是多少元？请你列式算一算。",
        f"一件{obj}的价格降低{p}%以后是{after}元。妈妈问这件{obj}原来卖多少元，你能算出来吗？",
    ])
    lines = [
        f"{after} ÷ ((100 - {p})/100) = {orig}元",
    ]
    return ins, lines, orig


_reg("pct_reverse_down", pct_reverse_down)


# 14. 涨价p%后卖a元 → 原价
def pct_reverse_up(rng):
    k = rng.randint(20, 90)
    orig = 100 * k
    p = rng.choice([10, 15, 20, 25])
    after = k * (100 + p)
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"由于进货价上涨，商店把一件{obj}涨价{p}%出售，现在售价是{after}元。这件{obj}的原价是多少元？你能算出来吗？",
        f"商场把一件{obj}的价格上调{p}%后卖{after}元。爸爸问这件{obj}的原价是多少元，请你列式算一算。",
        f"一件{obj}提价{p}%以后售价为{after}元。这件{obj}原来卖多少元？你能帮售货员算一算吗？",
    ])
    lines = [
        f"{after} ÷ ((100 + {p})/100) = {orig}元",
    ]
    return ins, lines, orig


_reg("pct_reverse_up", pct_reverse_up)


# 15. 先涨p%再降p%后卖a元 → 原价
def pct_up_then_down(rng):
    k = rng.randint(2, 5)
    orig = 10000 * k
    p = rng.choice([10, 20])
    mid = orig * (100 + p) // 100
    a = mid * (100 - p) // 100
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店里一件{obj}先涨价{p}%，再降价{p}%，最后售价是{a}元。妈妈想知道这件{obj}的原价是多少元，你能帮她算一算吗？",
        f"一件{obj}的价格先上调{p}%，又下调{p}%，最后卖{a}元。这件{obj}的原价是多少元？请你列式算一算。",
        f"商场把一件{obj}先提价{p}%，再降价{p}%促销，最后售价{a}元。这件{obj}的原价是多少元？你会算吗？",
    ])
    lines = [
        f"{a} ÷ ((100 - {p})/100) = {mid}元",
        f"{mid} ÷ ((100 + {p})/100) = {orig}元",
    ]
    return ins, lines, orig


_reg("pct_up_then_down", pct_up_then_down)


# 16. 和差问题
def sum_diff(rng):
    a = rng.randint(20, 80)
    b = rng.randint(10, a - 10)
    s, d = a + b, a - b
    who = rng.choice(["甲", "乙"])
    ins = rng.choice([
        f"数学课上老师出了一道和差问题：甲、乙两个数的和是{s}，差是{d}。{who}数是多少？{rng.choice(NAMES)}没算出来，你能帮帮他吗？",
        f"已知甲、乙两数之和为{s}，两数之差为{d}。爸爸让{rng.choice(NAMES)}求{who}数，他想了很久，你能列式算一算吗？",
        f"甲、乙两个数相加得{s}，相减得{d}。{who}数是多少？请你列算式算一算，把结果告诉{rng.choice(NAMES)}。",
    ])
    if who == "甲":
        lines = [f"两数和加差 = {s} + {d} = {s + d}", f"甲数 = {s + d} ÷ 2 = {a}"]
        return ins, lines, a
    lines = [f"两数和减差 = {s} - {d} = {s - d}", f"乙数 = {s - d} ÷ 2 = {b}"]
    return ins, lines, b


_reg("sum_diff", sum_diff)


# 17. 和倍问题
def sum_multiple(rng):
    b = rng.randint(8, 40)
    k = rng.randint(2, 6)
    a = k * b
    s = a + b
    who = rng.choice(["甲", "乙"])
    ins = rng.choice([
        f"数学课上老师出了一道和倍问题：甲、乙两个数的和是{s}，甲数正好是乙数的{k}倍。{who}数是多少？你能算出来吗？",
        f"已知甲、乙两数之和为{s}，且甲数是乙数的{k}倍。{rng.choice(NAMES)}想求{who}数，你能列式帮他算一算吗？",
        f"甲、乙两数的和是{s}，甲数等于乙数的{k}倍。{who}数是多少？请你列算式算一算，看谁算得又对又快。",
    ])
    lines = [f"总份数 = {k} + 1 = {k + 1}", f"乙数 = {s} ÷ {k + 1} = {b}"]
    if who == "甲":
        lines.append(f"甲数 = {b} × {k} = {a}")
        return ins, lines, a
    return ins, lines, b


_reg("sum_multiple", sum_multiple)


# 18. 差倍问题
def diff_multiple(rng):
    b = rng.randint(8, 40)
    k = rng.randint(2, 6)
    a = k * b
    d = a - b
    who = rng.choice(["甲", "乙"])
    ins = rng.choice([
        f"数学课上老师出了一道差倍问题：甲数比乙数多{d}，且甲数正好是乙数的{k}倍。{who}数是多少？你能算出来吗？",
        f"已知甲、乙两数的差是{d}，甲数是乙数的{k}倍。{rng.choice(NAMES)}想求{who}数，你能列式帮他算一算吗？",
        f"甲数比乙数大{d}，甲数等于乙数的{k}倍。{who}数是多少？请你列算式算一算，看谁算得又对又快。",
    ])
    lines = [f"份数差 = {k} - 1 = {k - 1}", f"乙数 = {d} ÷ {k - 1} = {b}"]
    if who == "甲":
        lines.append(f"甲数 = {b} × {k} = {a}")
        return ins, lines, a
    return ins, lines, b


_reg("diff_multiple", diff_multiple)


# 19. 连续自然数之和 → 最小数
def consec_sum_first(rng):
    k = rng.choice([3, 5])
    start = rng.randint(3, 20)
    vals = list(range(start, start + k))
    s = sum(vals)
    mid = start + (k - 1) // 2
    off = (k - 1) // 2
    ins = rng.choice([
        f"数学课上老师出了一道题：{k}个连续自然数的和是{s}，其中最小的一个是多少？{rng.choice(NAMES)}没算出来，你能帮帮他吗？",
        f"有{k}个连续的自然数，它们的和是{s}。爸爸让{rng.choice(NAMES)}求最小的数，你能列式算一算吗？",
        f"连续{k}个自然数相加得{s}，这{k}个数中最小的是几？请你列算式算一算，看谁算得又对又快。",
    ])
    lines = [
        f"中间的数 = {s} ÷ {k} = {mid}",
        f"最小的数 = {mid} - {off} = {start}",
    ]
    return ins, lines, start


_reg("consec_sum_first", consec_sum_first)


# 20. 连续偶数之和 → 最大数
def consec_even_max(rng):
    start = rng.randint(2, 9) * 2
    vals = [start, start + 2, start + 4, start + 6]
    s = sum(vals)
    mx = start + 6
    ins = rng.choice([
        f"数学课上老师出了一道题：4个连续偶数的和是{s}，其中最大的一个是多少？{rng.choice(NAMES)}没算出来，你能帮帮他吗？",
        f"有4个连续的偶数，它们相加得{s}。爸爸让{rng.choice(NAMES)}求最大的偶数，你能列式算一算吗？",
        f"连续4个偶数的和是{s}，这4个数中最大的是几？请你列算式算一算，看谁算得又对又快。",
    ])
    lines = [
        f"最小偶数的4倍 = {s} - 12 = {4 * start}",
        f"最小偶数 = {4 * start} ÷ 4 = {start}",
        f"最大偶数 = {start} + 6 = {mx}",
    ]
    return ins, lines, mx


_reg("consec_even_max", consec_even_max)


# 21. 经费按分数用三周，剩余平分
def budget_fractions_split(rng):
    k = rng.randint(10, 30)
    total = 60 * k
    s1, s2, s3 = 15 * k, 12 * k, 10 * k
    rest = 23 * k
    g = rng.randint(2, 6)
    each = Fraction(rest, g)
    ins = rng.choice([
        f"一笔{total}元的活动经费，第一周用去全部的1/4，第二周用去全部的1/5，第三周用去全部的1/6，剩下的平均分给{g}个小组。每个小组分到多少元？",
        f"一笔{total}元的经费，第一周花了总数的1/4，第二周花了总数的1/5，第三周花了总数的1/6，余下的平均分给{g}个小组。每组得多少元？",
        f"活动经费共{total}元，三周分别用去总数的1/4、1/5和1/6，剩下的钱平均分给{g}个小组。每个小组分到多少元？",
    ])
    lines = [
        f"{total} ÷ 4 = {s1}元",
        f"{total} ÷ 5 = {s2}元",
        f"{total} ÷ 6 = {s3}元",
        f"{total} - {s1} = {total - s1}元",
        f"{total - s1} - {s2} = {total - s1 - s2}元",
        f"{total - s1 - s2} - {s3} = {rest}元",
        f"{rest} ÷ {g} = {num(each)}元",
    ]
    return ins, lines, each


_reg("budget_fractions_split", budget_fractions_split)


# 22. 甲比乙的k倍多c，甲乙和s → 甲
def multiple_plus_c_sum(rng):
    b = rng.randint(5, 30)
    k = rng.randint(2, 5)
    c = rng.randint(3, 20)
    a = k * b + c
    s = a + b
    who = rng.choice(["甲", "乙"])
    ins = rng.choice([
        f"甲、乙两个数，甲数比乙数的{k}倍还多{c}，甲、乙两数的和是{s}。{who}数是多少？{rng.choice(NAMES)}没算出来，你能帮帮他吗？",
        f"已知甲数等于乙数的{k}倍加{c}，甲、乙两数之和是{s}。{rng.choice(NAMES)}想求{who}数，你能列式帮他算一算吗？",
        f"甲数比乙数的{k}倍多{c}，且甲、乙两数的和为{s}。{who}数是多少？请你列算式算一算，看谁算得又对又快。",
    ])
    lines = [
        f"乙数的{k + 1}倍 = {s} - {c} = {s - c}",
        f"总份数 = {k} + 1 = {k + 1}",
        f"乙数 = {s - c} ÷ {k + 1} = {b}",
    ]
    if who == "甲":
        lines.append(f"甲数 = {b} × {k} + {c} = {a}")
        return ins, lines, a
    return ins, lines, b


_reg("multiple_plus_c_sum", multiple_plus_c_sum)


# 23. 今年父是子k倍，a年后父是子m倍 → 子今年
def age_future_ratio(rng):
    x = a = father = None
    for _ in range(80):
        m = rng.choice([2, 3])
        k = rng.randint(m + 1, m + 3)
        t = rng.randint(3, 6)
        x = (m - 1) * t
        a = (k - m) * t
        father = k * x
        if 20 <= father - x <= 50 and father <= 65 and a >= 2:
            break
    else:
        m, k, t, x, a, father = 3, 5, 4, 8, 8, 40
    ins = rng.choice([
        f"今年父亲的年龄是儿子的{k}倍，{a}年后父亲的年龄是儿子的{m}倍。儿子今年多少岁？你能算出来吗？请列式算一算。",
        f"父亲今年的岁数是儿子的{k}倍，再过{a}年，父亲的岁数是儿子的{m}倍。儿子今年几岁？你会算吗？快动手试一试。",
        f"今年父亲年龄是儿子的{k}倍，{a}年以后父亲年龄是儿子的{m}倍。儿子今年多少岁？请你列算式算一算。",
    ])
    lines = [
        f"倍数差 = {k} - {m} = {k - m}",
        f"父比子多的倍数 = {m} - 1 = {m - 1}",
        f"多的倍数乘年数 = ({m} - 1) × {a} = {(m - 1) * a}",
        f"({m} - 1) × {a} ÷ {k - m} = {x}岁",
    ]
    return ins, lines, x


_reg("age_future_ratio", age_future_ratio)


# 24. 今年父f岁子s岁，几年前父是子的k倍
def age_past_ratio(rng):
    s = a = f = k = None
    for _ in range(80):
        s = rng.randint(8, 14)
        k = rng.randint(3, 6)
        q = rng.randint(3, min(8, s - 2))
        a = s - q
        f = k * q + a
        if 20 <= (k - 1) * q <= 52 and f <= 70:
            break
    else:
        s, k, q, a, f = 14, 5, 6, 8, 38
    ins = rng.choice([
        f"今年父亲{f}岁，儿子{s}岁。多少年前父亲的年龄正好是儿子的{k}倍？你能算出来吗？请列式算一算。",
        f"父亲今年{f}岁，儿子今年{s}岁。几年前父亲的岁数是儿子的{k}倍？你会算吗？快动手试一试。",
        f"今年父亲{f}岁、儿子{s}岁，多少年前父亲年龄是儿子的{k}倍？请你列算式算一算，看谁算得又对又快。",
    ])
    lines = [
        f"儿子岁数的{k}倍 = {k} × {s} = {k * s}",
        f"倍差乘年数 = {k * s} - {f} = {(k - 1) * a}",
        f"倍数差 = {k} - 1 = {k - 1}",
        f"{(k - 1) * a} ÷ {k - 1} = {a}年",
    ]
    return ins, lines, a


_reg("age_past_ratio", age_past_ratio)


# 25. 父子年龄和s，a年前父是子的k倍 → 子今年
def age_sum_past(rng):
    x = father = s = None
    for _ in range(80):
        q = rng.randint(4, 8)
        k = rng.randint(3, 5)
        a = rng.randint(2, 4)
        x = q + a
        d = (k - 1) * q
        father = x + d
        s = father + x
        if d >= 20 and father <= 65:
            break
    else:
        q, k, a, x, d, father, s = 6, 4, 3, 9, 18, 27, 36
    ins = rng.choice([
        f"今年父子俩的年龄和是{s}岁，{a}年前父亲的年龄是儿子的{k}倍。儿子今年多少岁？你能算出来吗？请列式算一算。",
        f"父亲和儿子今年的年龄加起来是{s}岁，{a}年前父亲的岁数是儿子的{k}倍。儿子今年几岁？你会算吗？快动手试一试。",
        f"今年父子年龄之和为{s}岁，{a}年前父亲年龄是儿子的{k}倍。儿子今年多少岁？请你列算式算一算。",
    ])
    lines = [
        f"倍数差 = {k} - 1 = {k - 1}",
        f"倍数差乘年数 = {a} × {k - 1} = {a * (k - 1)}",
        f"年龄和加调整量 = {s} + {a * (k - 1)} = {s + a * (k - 1)}",
        f"倍数和 = {k} + 1 = {k + 1}",
        f"{s + a * (k - 1)} ÷ {k + 1} = {x}岁",
    ]
    return ins, lines, x


_reg("age_sum_past", age_sum_past)


# 26. 两管同开注满时间
def two_pipe_together(rng):
    a = rng.randint(3, 12)
    b = rng.randint(3, 12)
    rate = Fraction(a + b, a * b)
    t = Fraction(a * b, a + b)
    ins = rng.choice([
        f"一个水池有甲、乙两个进水管。单开甲管{a}小时注满，单开乙管{b}小时注满。两管同时打开，几小时能注满？你会算吗？",
        f"水池有两个进水管，甲管单独开{a}小时注满，乙管单独开{b}小时注满。两管齐开，几小时注满？请列式算一算。",
        f"注满一个水池，甲管要{a}小时，乙管要{b}小时。管理员把两管同时打开，需要几小时才能注满水池？",
    ])
    lines = [
        f"两管合开每小时注水量 = 1 ÷ {a} + 1 ÷ {b} = {num(rate)}池/时",
        f"1 ÷ ({num(rate)}) = {num(t)}时",
    ]
    return ins, lines, t


_reg("two_pipe_together", two_pipe_together)


# 27. 进水管与排水管同开
def pipe_fill_drain(rng):
    a = rng.randint(3, 8)
    b = a + rng.randint(3, 10)
    rate = Fraction(b - a, a * b)
    t = Fraction(a * b, b - a)
    ins = rng.choice([
        f"一个水池有一个进水管和一个排水管。单开进水管{a}小时注满，单开排水管{b}小时排空。两管同时打开，几小时能注满？你会算吗？",
        f"水池的进水管单开{a}小时注满，排水管单开{b}小时把满池水排空。两管齐开，几小时注满？请列式算一算。",
        f"一个空水池，进水管{a}小时能注满，排水管{b}小时能排空。工人把两管同时打开，几小时后水池能注满？",
    ])
    lines = [
        f"每小时净注水量 = 1 ÷ {a} - 1 ÷ {b} = {num(rate)}池/时",
        f"1 ÷ ({num(rate)}) = {num(t)}时",
    ]
    return ins, lines, t


_reg("pipe_fill_drain", pipe_fill_drain)


# 28. 两管合开t小时满，甲单开a小时满 → 乙单开时间
def pipe_combined_reverse(rng):
    t, a, b = rng.choice([
        (2, 3, 6), (3, 4, 12), (4, 5, 20), (4, 6, 12), (5, 6, 30),
    ])
    rate_b = Fraction(1, b)
    ins = rng.choice([
        f"一个水池有甲、乙两个进水管。两管同时打开，{t}小时能注满。单开甲管{a}小时注满，单开乙管几小时注满？你会算吗？",
        f"水池有两个进水管，齐开{t}小时注满；只开甲管要{a}小时注满。只开乙管要几小时？请列式算一算。",
        f"注满一个水池，甲、乙两管合开需{t}小时，甲管单开需{a}小时。乙管单开需几小时？你能算出来吗？",
    ])
    lines = [
        f"乙管每小时注水量 = 1 ÷ {t} - 1 ÷ {a} = {num(rate_b)}池/时",
        f"1 ÷ ({num(rate_b)}) = {b}时",
    ]
    return ins, lines, b


_reg("pipe_combined_reverse", pipe_combined_reverse)


# 29. 三数平均m，前两数平均n → 第三数
def avg_third_number(rng):
    a = b = c = m = n = None
    for _ in range(80):
        a = rng.randint(10, 50)
        b = rng.randint(10, 50)
        if (a + b) % 2 != 0:
            b += 1
        n = (a + b) // 2
        m = rng.randint(20, 50)
        c = 3 * m - a - b
        if 10 <= c <= 60:
            break
    else:
        a, b, n, m, c = 20, 30, 25, 30, 40
    ins = rng.choice([
        f"数学课上老师出了一道题：甲、乙、丙三个数的平均数是{m}，其中甲、乙两个数的平均数是{n}。丙数是多少？你能算出来吗？",
        f"三个数的平均数是{m}，前两个数的平均数是{n}。第三个数是多少？{rng.choice(NAMES)}没算出来，请列式帮他算一算。",
        f"甲、乙、丙三个数平均为{m}，甲、乙两数平均为{n}。丙数是多少？请你列算式算一算，看谁算得又对又快。",
    ])
    lines = [
        f"三数总和 = {m} × 3 = {3 * m}",
        f"前两数之和 = {n} × 2 = {a + b}",
        f"丙数 = {3 * m} - {a + b} = {c}",
    ]
    return ins, lines, c


_reg("avg_third_number", avg_third_number)


# 30. 平均身高：加入新队员
def avg_new_member(rng):
    n = a = b = new = None
    for _ in range(80):
        n = rng.randint(4, 8)
        a = rng.randint(130, 160)
        b = rng.randint(132, 165)
        new = (n + 1) * b - n * a
        if 120 <= new <= 190:
            break
    else:
        n, a, b, new = 5, 140, 142, 152
    ins = rng.choice([
        f"体操小组原有{n}名队员，平均身高是{a}厘米。新加入1名队员后，全组的平均身高变为{b}厘米。新队员的身高是多少厘米？你会算吗？",
        f"合唱队原有{n}人，平均身高{a}厘米。又来了1名新队员后，平均身高变成{b}厘米。新队员身高多少厘米？请列式算一算。",
        f"一个{n}人的小队，平均身高是{a}厘米。加入1名队员后平均身高为{b}厘米。新加入的队员身高多少厘米？你能算出来吗？",
    ])
    lines = [
        f"{a} × {n} = {a * n}厘米",
        f"{b} × {n + 1} = {b * (n + 1)}厘米",
        f"{b * (n + 1)} - {a * n} = {new}厘米",
    ]
    return ins, lines, new


_reg("avg_new_member", avg_new_member)


# 31. 鸡兔同笼
def chicken_rabbit(rng):
    c = rng.randint(5, 20)
    r = rng.randint(5, 20)
    h = c + r
    l = 2 * c + 4 * r
    who = rng.choice(["小鸡", "小兔"])
    ins = rng.choice([
        f"农场里有小鸡和小兔共{h}只，数一数它们的腿共有{l}条。{who}有多少只？你能算出来吗？请列式算一算。",
        f"笼子里有鸡和兔共{h}只，腿一共有{l}条。{who}有多少只？{rng.choice(NAMES)}数不清，请你列式帮他算一算。",
        f"院子里小鸡和小兔一共{h}只，它们的腿加起来有{l}条。{who}有多少只？请你列算式算一算。",
    ])
    lines = [
        f"{h} × 2 = {2 * h}条",
        f"{l} - {2 * h} = {l - 2 * h}条",
        f"4 - 2 = 2条",
        f"{l - 2 * h} ÷ 2 = {r}只",
        f"{h} - {r} = {c}只",
    ]
    if who == "小兔":
        lines[-2], lines[-1] = lines[-1], lines[-2]
        return ins, lines, r
    return ins, lines, c


_reg("chicken_rabbit", chicken_rabbit)


# 32. 自行车与三轮车
def bike_trike_wheels(rng):
    b = rng.randint(5, 20)
    t = rng.randint(5, 20)
    h = b + t
    l = 2 * b + 3 * t
    who = rng.choice(["自行车", "三轮车"])
    ins = rng.choice([
        f"停车场里有自行车和三轮车共{h}辆，数一数车轮共有{l}个。{who}有多少辆？你能算出来吗？请列式算一算。",
        f"车棚里停放着自行车和三轮车共{h}辆，车轮一共{l}个。{who}有多少辆？{rng.choice(NAMES)}数不清，请列式帮他算一算。",
        f"停车场上自行车和三轮车一共{h}辆，车轮总数是{l}个。{who}有多少辆？请你列算式算一算。",
    ])
    lines = [
        f"{h} × 2 = {2 * h}个",
        f"{l} - {2 * h} = {l - 2 * h}个",
        f"3 - 2 = 1个",
        f"{l - 2 * h} ÷ 1 = {t}辆",
        f"{h} - {t} = {b}辆",
    ]
    if who == "三轮车":
        lines[-2], lines[-1] = lines[-1], lines[-2]
        return ins, lines, t
    return ins, lines, b


_reg("bike_trike_wheels", bike_trike_wheels)


# 33. 植树问题（两端都栽）
def plant_trees_forward(rng):
    d = rng.randint(3, 8)
    k = rng.randint(10, 30)
    L = d * k
    trees = k + 1
    ins = rng.choice([
        f"在一条长{L}米的小路一侧栽树，每隔{d}米栽一棵，两端都要栽。一共要栽多少棵树？你会算吗？请列式算一算。",
        f"一条马路长{L}米，在路的一边从头至尾每隔{d}米栽一棵树。一共要栽多少棵？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"园林工人要在长{L}米的小路旁栽树，每隔{d}米栽一棵，两端各栽一棵。共需多少棵树苗？请你列算式算一算。",
    ])
    lines = [
        f"{L} ÷ {d} = {k}段",
        f"{k} + 1 = {trees}棵",
    ]
    return ins, lines, trees


_reg("plant_trees_forward", plant_trees_forward)


# 34. 植树问题逆推路长
def plant_trees_reverse(rng):
    n = rng.randint(12, 40)
    d = rng.randint(3, 8)
    L = (n - 1) * d
    ins = rng.choice([
        f"园林工人在小路一侧栽树，每隔{d}米栽一棵，两端都栽，一共栽了{n}棵。这条小路长多少米？你会算吗？请列式算一算。",
        f"在一条马路的一边从头到尾栽树，每隔{d}米栽一棵，共栽了{n}棵。这条马路长多少米？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"小路旁每隔{d}米栽一棵树，两端都栽，一共栽了{n}棵。小路全长多少米？请你列算式算一算。",
    ])
    lines = [
        f"{n} - 1 = {n - 1}段",
        f"{n - 1} × {d} = {L}米",
    ]
    return ins, lines, L


_reg("plant_trees_reverse", plant_trees_reverse)


# 35. 锯木头
def saw_wood(rng):
    k = rng.randint(3, 6)
    m = rng.randint(5, 9)
    p = rng.randint(2, 5)
    t = (k - 1) * p
    total = (m - 1) * p
    ins = rng.choice([
        f"一根木头锯成{k}段需要{t}分钟，每次锯的时间相同。锯成{m}段需要多少分钟？你会算吗？请列式算一算。",
        f"把一根木头锯成{k}段用了{t}分钟，照这样计算，锯成{m}段要用多少分钟？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"一根木头锯成{k}段需{t}分钟，每次锯的时间一样。锯成{m}段需要几分钟？请你列算式算一算。",
    ])
    lines = [
        f"{k} - 1 = {k - 1}次",
        f"{t} ÷ {k - 1} = {p}分",
        f"{m} - 1 = {m - 1}次",
        f"{m - 1} × {p} = {total}分",
    ]
    return ins, lines, total


_reg("saw_wood", saw_wood)


# 36. 爬楼梯
def climb_stairs(rng):
    a = rng.randint(3, 6)
    b = rng.randint(7, 12)
    p = rng.randint(8, 20)
    t = (a - 1) * p
    total = (b - 1) * p
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}从1楼走到{a}楼用了{t}秒。照这样的速度，他从1楼走到{b}楼需要多少秒？你会算吗？请列式算一算。",
        f"{name}爬楼梯，从1楼到{a}楼用了{t}秒。以同样的速度，他从1楼到{b}楼要用多少秒？请你列算式算一算。",
        f"从1楼走到{a}楼，{name}用了{t}秒。照这样算，他从1楼走到{b}楼需要多少秒？你能算出来吗？",
    ])
    lines = [
        f"{a} - 1 = {a - 1}层",
        f"{t} ÷ {a - 1} = {p}秒",
        f"{b} - 1 = {b - 1}层",
        f"{b - 1} × {p} = {total}秒",
    ]
    return ins, lines, total


_reg("climb_stairs", climb_stairs)


# 37. 相遇时间
def meet_time(rng):
    v1 = rng.randint(40, 80)
    v2 = rng.randint(40, 80)
    t = rng.randint(2, 5)
    s = (v1 + v2) * t
    ins = rng.choice([
        f"甲、乙两车同时从相距{s}千米的两地相向开出，甲车每小时行{v1}千米，乙车每小时行{v2}千米。经过几小时两车相遇？你会算吗？",
        f"两地相距{s}千米，甲、乙两车分别从两地同时出发相向而行，甲车每小时{v1}千米，乙车每小时{v2}千米。几小时后相遇？请列式算一算。",
        f"甲、乙两车从相距{s}千米的两地同时相向而行，甲车速度每小时{v1}千米，乙车每小时{v2}千米。经过几小时相遇？你能算出来吗？",
    ])
    lines = [
        f"{v1} + {v2} = {v1 + v2}千米/时",
        f"{s} ÷ {v1 + v2} = {t}时",
    ]
    return ins, lines, t


_reg("meet_time", meet_time)


# 38. 相遇问题求速度
def meet_speed_reverse(rng):
    v1 = rng.randint(40, 80)
    v2 = rng.randint(40, 80)
    t = rng.randint(2, 5)
    s = (v1 + v2) * t
    ins = rng.choice([
        f"甲、乙两车同时从相距{s}千米的两地相向开出，经过{t}小时相遇。已知甲车每小时行{v1}千米，乙车每小时行多少千米？你会算吗？",
        f"两地相距{s}千米，甲、乙两车同时出发相向而行，{t}小时后相遇。甲车每小时行{v1}千米，乙车每小时行多少千米？请列式算一算。",
        f"甲、乙两车从相距{s}千米的两地同时相向开出，{t}小时相遇。甲车每小时{v1}千米，乙车每小时行多少千米？你能算出来吗？",
    ])
    lines = [
        f"{s} ÷ {t} = {v1 + v2}千米/时",
        f"{v1 + v2} - {v1} = {v2}千米/时",
    ]
    return ins, lines, v2


_reg("meet_speed_reverse", meet_speed_reverse)


# 39. 追及时间
def chase_time(rng):
    v1 = rng.randint(3, 6)
    v2 = v1 + rng.randint(1, 4)
    t = rng.randint(5, 20)
    a = (v2 - v1) * t
    ins = rng.choice([
        f"甲在乙前面{a}米处，甲每秒跑{v1}米，乙每秒跑{v2}米，两人同时向前跑。乙经过多少秒能追上甲？你会算吗？请列式算一算。",
        f"甲、乙两人赛跑，甲先跑出{a}米，甲每秒跑{v1}米，乙每秒跑{v2}米。乙多少秒后追上甲？请你列算式算一算。",
        f"乙在甲后面{a}米，两人同时同向跑步，甲每秒{v1}米，乙每秒{v2}米。乙几秒后追上甲？你能算出来吗？",
    ])
    lines = [
        f"{v2} - {v1} = {v2 - v1}米/秒",
        f"{a} ÷ {v2 - v1} = {t}秒",
    ]
    return ins, lines, t


_reg("chase_time", chase_time)


# 40. 往返时间
def round_trip_time(rng):
    v1 = t1 = dist = v2 = t2 = None
    for _ in range(80):
        v1 = rng.randint(50, 70)
        t1 = rng.randint(8, 15)
        dist = v1 * t1
        cands = [v for v in range(60, 101) if dist % v == 0]
        if cands:
            v2 = rng.choice(cands)
            t2 = dist // v2
            break
    else:
        v1, t1, dist, v2, t2 = 60, 10, 600, 75, 8
    name = rng.choice(NAMES)
    place = rng.choice(PLACE)
    ins = rng.choice([
        f"{name}从家走到{place}，每分钟走{v1}米，用了{t1}分钟。放学原路返回，每分钟走{v2}米，返回需要多少分钟？你会算吗？",
        f"{name}从家到{place}，每分钟走{v1}米，{t1}分钟走到。原路返回时每分钟走{v2}米，返回用了多少分钟？请列式算一算。",
        f"{name}步行去{place}，每分钟{v1}米，走了{t1}分钟。回来时每分钟{v2}米，返回需要多少分钟？你能算出来吗？",
    ])
    lines = [
        f"{v1} × {t1} = {dist}米",
        f"{dist} ÷ {v2} = {t2}分",
    ]
    return ins, lines, t2


_reg("round_trip_time", round_trip_time)


# 41. 工程：甲先做c天，再合做
def project_first_then(rng):
    a = rng.randint(8, 16)
    b = rng.randint(6, 14)
    c = rng.randint(1, a // 3)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    done = c * ra
    rest = 1 - done
    combined = ra + rb
    t = rest / combined
    ins = rng.choice([
        f"一项工程，甲队单独做{a}天完成，乙队单独做{b}天完成。甲队先单独做{c}天，剩下的两队合做，还要多少天完成？你会算吗？",
        f"一项工程，甲独做{a}天完成，乙独做{b}天完成。甲先做{c}天后，甲、乙合做，还需多少天完成？请列式算一算。",
        f"完成一项工程，甲队要{a}天，乙队要{b}天。甲队先做{c}天，余下的由两队合做，还要几天完成？你能算出来吗？",
    ])
    lines = [
        f"甲队每天完成量 = 1 ÷ {a} = {num(ra)}",
        f"乙队每天完成量 = 1 ÷ {b} = {num(rb)}",
        f"两队合做每天完成量 = {num(ra)} + {num(rb)} = {num(combined)}",
        f"甲队{c}天完成量 = {c} × {num(ra)} = {num(done)}",
        f"剩余工程量 = 1 - {num(done)} = {num(rest)}",
        f"{num(rest)} ÷ ({num(combined)}) = {num(t)}天",
    ]
    return ins, lines, t


_reg("project_first_then", project_first_then)


# 42. 浓度：加盐提高浓度
def concentration_add_salt(rng):
    g = salt = water = salt_new = add = p = q = None
    for _ in range(100):
        k = rng.randint(2, 8)
        g = 100 * k
        p = rng.choice([5, 10, 15, 20])
        q = rng.choice([20, 25, 30, 40, 50])
        if q <= p + 5:
            continue
        salt = k * p
        water = g - salt
        num_ = water * q
        if num_ % (100 - q) == 0:
            salt_new = num_ // (100 - q)
            add = salt_new - salt
            if add > 0:
                break
    else:
        g, p, q, salt, water, salt_new, add = 200, 10, 25, 20, 180, 60, 40
    ins = rng.choice([
        f"科学课上，老师拿来一杯{g}克的盐水，含盐率是{p}%。要使含盐率变成{q}%，需要再加入多少克盐？你会算吗？请列式算一算。",
        f"一杯盐水共{g}克，含盐率为{p}%。要使含盐率提高到{q}%，应加盐多少克？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"有{g}克盐水，含盐率是{p}%。要使含盐率达到{q}%，需要加入多少克盐？请你列算式算一算。",
    ])
    lines = [
        f"{g} × {p}/100 = {salt}克",
        f"{g} - {salt} = {water}克",
        f"后来水占的百分率 = 100 - {q} = {100 - q}",
        f"{water} × {q} ÷ {100 - q} = {salt_new}克",
        f"{salt_new} - {salt} = {add}克",
    ]
    return ins, lines, add


_reg("concentration_add_salt", concentration_add_salt)


# 43. 按比例配制（已知总量）
def ratio_parts_total(rng):
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    m = rng.randint(20, 60)
    t = (a + b) * m
    juice = a * m
    water = b * m
    who = rng.choice(["果汁", "水"])
    ins = rng.choice([
        f"一种饮料由果汁和水按{a}比{b}的比例配制而成。要配制{t}克这样的饮料，需要{who}多少克？你会算吗？请列式算一算。",
        f"一种饮料中果汁与水的比是{a}比{b}。要配制{t}克这种饮料，需要{who}多少克？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"按{a}比{b}的比例用果汁和水配制饮料，要配{t}克，需要{who}多少克？请你列算式算一算。",
    ])
    lines = [
        f"{a} + {b} = {a + b}份",
        f"{t} ÷ {a + b} = {m}克",
    ]
    if who == "果汁":
        lines.append(f"{m} × {b} = {water}克")
        lines.append(f"{m} × {a} = {juice}克")
        return ins, lines, juice
    lines.append(f"{m} × {a} = {juice}克")
    lines.append(f"{m} × {b} = {water}克")
    return ins, lines, water


_reg("ratio_parts_total", ratio_parts_total)


# 44. 三人按比例分奖金
def ratio_three_shares(rng):
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    c = rng.randint(2, 6)
    m = rng.randint(10, 40)
    t = (a + b + c) * m
    who = rng.choice(["第一", "第二", "第三"])
    share = {"第一": a * m, "第二": b * m, "第三": c * m}[who]
    ins = rng.choice([
        f"公司把一笔奖金分给三名员工，三人按{a}比{b}比{c}的比例分配一笔{t}元的奖金，{who}人得多少元？你会算吗？",
        f"一笔{t}元的奖金按{a}比{b}比{c}分给甲、乙、丙三个人，{who}个人分到多少元？请列式算一算。",
        f"甲、乙、丙三人按{a}比{b}比{c}的比例分{t}元奖金，{who}人（甲为第一）得多少元？你能算出来吗？",
    ])
    lines = [
        f"{a} + {b} + {c} = {a + b + c}份",
        f"{t} ÷ {a + b + c} = {m}元",
    ]
    order = {"第一": (a, b, c), "第二": (b, a, c), "第三": (c, a, b)}[who]
    first, other1, other2 = order
    lines.append(f"{m} × {other1} = {other1 * m}元")
    lines.append(f"{m} × {other2} = {other2 * m}元")
    lines.append(f"{m} × {first} = {first * m}元")
    return ins, lines, first * m


_reg("ratio_three_shares", ratio_three_shares)


# 45. 打折逆推原价
def discount_reverse(rng):
    d = rng.randint(6, 9)
    k = rng.randint(20, 80)
    orig = 10 * k
    a = k * d
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店搞促销，一件{obj}打{d}折出售，售价是{a}元。这件{obj}的原价是多少元？你会算吗？请列式算一算。",
        f"商场把一件{obj}按原价的{d}折出售，卖了{a}元。原价是多少元？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"一件{obj}打{d}折后的价格是{a}元。这件{obj}原来卖多少元？请你列算式算一算。",
    ])
    lines = [
        f"{a} × 10 ÷ {d} = {orig}元",
    ]
    return ins, lines, orig


_reg("discount_reverse", discount_reverse)


# 46. 利润定价
def profit_price(rng):
    k = rng.randint(2, 8)
    c = 100 * k
    p = rng.choice([10, 20, 25, 40, 50])
    profit = k * p
    price = c + profit
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店以{c}元的价格购进一件{obj}，想获得{p}%的利润。这件{obj}应定价多少元？你会算吗？请列式算一算。",
        f"一件{obj}的成本是{c}元，要想获得{p}%的利润，售价应定为多少元？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"一件{obj}的进价是{c}元，商店想获得{p}%的利润。这件{obj}应卖多少元？请你列算式算一算。",
    ])
    lines = [
        f"{c} × {p}/100 = {profit}元",
        f"{c} + {profit} = {price}元",
    ]
    return ins, lines, price


_reg("profit_price", profit_price)


# 47. 利息问题
def simple_interest(rng):
    p = rng.randint(2, 10) * 1000
    r = rng.choice([2, 3, 4, 5])
    n = rng.randint(2, 5)
    per = p * r // 100
    interest = per * n
    total = p + interest
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}把{p}元压岁钱存入银行，年利率是{r}%，存满{n}年。到期时一共可以取回多少元？你会算吗？请列式算一算。",
        f"{name}将{p}元存入银行，定期{n}年，年利率{r}%。到期后一共能取回多少元？请你列算式算一算。",
        f"银行一年期年利率是{r}%，{name}把{p}元存了{n}年。到期时本息共多少元？你能算出来吗？",
    ])
    lines = [
        f"{p} × {r}/100 = {per}元",
        f"{per} × {n} = {interest}元",
        f"{p} + {interest} = {total}元",
    ]
    return ins, lines, total


_reg("simple_interest", simple_interest)


# 48. 多步还原
def restore_multi_op(rng):
    q = a = x = b = d = e = c = None
    for _ in range(100):
        q = rng.randint(8, 30)
        a = rng.randint(2, 10)
        x = q - a
        b = rng.randint(2, 5)
        d = rng.randint(2, 6)
        e = rng.randint(3, 12)
        c = b * q - d * e
        if x >= 3 and 3 <= c <= 20:
            break
    else:
        q, a, x, b, d, e, c = 10, 4, 6, 3, 4, 6, 6
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道还原问题：一个数加上{a}，再乘{b}，减去{c}，最后除以{d}，结果是{e}。这个数是多少？你会算吗？",
        f"{name}在练习册上看到一道题：把一个数先加{a}，再乘{b}，然后减{c}，最后除以{d}，得数是{e}。这个数是多少？请列式算一算。",
        f"爷爷给{name}出了一道题：某数加上{a}后乘{b}，减去{c}，再除以{d}，正好等于{e}。这个数是多少？你能算出来吗？",
    ])
    lines = [
        f"除以{d}之前的数 = {e} × {d} = {e * d}",
        f"减去{c}之前的数 = {e * d} + {c} = {b * q}",
        f"乘{b}之前的数 = {b * q} ÷ {b} = {q}",
        f"原来的数 = {q} - {a} = {x}",
    ]
    return ins, lines, x


_reg("restore_multi_op", restore_multi_op)


# 49. 倍数与转移后相等
def transfer_equal(rng):
    k = rng.choice([3, 5])
    c = rng.randint(4, 18)
    if k == 5 and c % 2 == 1:
        c += 1
    b = 2 * c // (k - 1)
    a = k * b
    who = rng.choice(["甲", "乙"])
    fruit = rng.choice(FRUITS)
    ins = rng.choice([
        f"甲筐{fruit}的个数是乙筐的{k}倍。如果从甲筐拿{c}个放入乙筐，两筐{fruit}个数相等。{who}筐原来有多少个？你会算吗？",
        f"甲筐的{fruit}是乙筐的{k}倍，从甲筐取出{c}个放入乙筐后，两筐一样多。{who}筐原来有多少个？请列式算一算。",
        f"甲筐{fruit}个数是乙筐的{k}倍，把甲筐的{c}个放到乙筐，两筐{fruit}相等。{who}筐原有多少个？你能算出来吗？",
    ])
    lines = [
        f"倍数差 = {k} - 1 = {k - 1}",
        f"甲筐比乙筐多的个数 = {c} × 2 = {2 * c}",
        f"{2 * c} ÷ {k - 1} = {b}个",
        f"{b} × {k} = {a}个",
    ]
    if who == "甲":
        return ins, lines, a
    lines[-2], lines[-1] = lines[-1], lines[-2]
    return ins, lines, b


_reg("transfer_equal", transfer_equal)


# 50. 给a个后相等，已知总和
def give_equal_sum(rng):
    half = rng.randint(15, 60)
    a = rng.randint(3, min(15, half - 3))
    s = 2 * half
    jia = half + a
    yi = half - a
    who = rng.choice(["甲", "乙"])
    fruit = rng.choice(FRUITS)
    ins = rng.choice([
        f"甲、乙两人共有{s}个{fruit}，甲给乙{a}个后两人一样多。{who}原来有多少个？你会算吗？请列式算一算。",
        f"甲、乙一共有{s}个{fruit}，如果甲给乙{a}个，两人的{fruit}就同样多。{who}原来有多少个？请你列算式算一算。",
        f"两堆{fruit}共{s}个，从第一堆拿{a}个到第二堆后两堆相等。{('甲' if who == '甲' else '乙')}堆原来有多少个？你能算出来吗？",
    ])
    lines = [
        f"{s} ÷ 2 = {half}个",
    ]
    if who == "甲":
        lines.append(f"{half} - {a} = {yi}个")
        lines.append(f"{half} + {a} = {jia}个")
        return ins, lines, jia
    lines.append(f"{half} + {a} = {jia}个")
    lines.append(f"{half} - {a} = {yi}个")
    return ins, lines, yi


_reg("give_equal_sum", give_equal_sum)


# 51. 有余数除法逆推
def division_remainder(rng):
    a = rng.randint(3, 9)
    q = rng.randint(5, 20)
    r = rng.randint(1, a - 1)
    x = a * q + r
    ins = rng.choice([
        f"数学课上老师出了一道除法题：一个数除以{a}，商是{q}，余数是{r}。这个数是多少？你会算吗？请列式算一算。",
        f"{rng.choice(NAMES)}在练习册上看到一道题：某数除以{a}得商{q}余{r}，求这个数。请你列式帮他算一算。",
        f"两个数相除，除数是{a}，商是{q}，余数是{r}。被除数是多少？你能算出来吗？快动手试一试。",
    ])
    lines = [
        f"除数乘商的积 = {a} × {q} = {a * q}",
        f"被除数 = {a * q} + {r} = {x}",
    ]
    return ins, lines, x


_reg("division_remainder", division_remainder)


# 52. 周期问题：循环彩灯
def cycle_lights(rng):
    k = rng.randint(3, 19)
    n = 3 * k + 1
    red = k + 1
    ins = rng.choice([
        f"节日彩灯按红、黄、蓝3盏一组的顺序循环排列，前{n}盏灯中，红灯有多少盏？你会算吗？请列式算一算。",
        f"彩灯按红、黄、蓝3盏一组循环排列，前{n}盏里红灯共有多少盏？{rng.choice(NAMES)}数不清，请你列式帮他算一算。",
        f"一排彩灯按红、黄、蓝每3盏一组循环排列，前{n}盏灯中有多少盏红灯？你能算出来吗？",
    ])
    lines = [
        f"凑整后的灯数 = {n} + 2 = {n + 2}",
        f"{n + 2} ÷ 3 = {red}盏",
    ]
    return ins, lines, red


_reg("cycle_lights", cycle_lights)


# 53. 方阵最外层人数
def square_outer(rng):
    n = rng.randint(6, 20)
    outer = 4 * (n - 1)
    ins = rng.choice([
        f"运动会上，同学们排成一个实心方阵，最外层每边有{n}人。最外层一共有多少人？你会算吗？请列式算一算。",
        f"学生排成实心方阵做广播操，最外层每边{n}人。最外层共有多少人？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"操场上同学们排成实心方阵，最外层一边有{n}人。最外层一共多少人？请你列算式算一算。",
    ])
    lines = [
        f"{n} - 1 = {n - 1}人",
        f"{n - 1} × 4 = {outer}人",
    ]
    return ins, lines, outer


_reg("square_outer", square_outer)


# 54. 看书计划
def book_reading_plan(rng):
    a = rng.randint(12, 30)
    b = rng.randint(3, 8)
    c = rng.randint(3, 7)
    d = rng.randint(10, 25)
    p = a * b + c * d
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一本书共{p}页，{name}前{b}天每天看{a}页，剩下的计划{c}天看完。剩下的平均每天要看多少页？你会算吗？",
        f"{name}看一本{p}页的故事书，前{b}天每天看{a}页，余下的要在{c}天内看完。平均每天要看多少页？请列式算一算。",
        f"一本故事书有{p}页，{name}已经看了{b}天，每天看{a}页。剩下的{c}天看完，每天要看多少页？你能算出来吗？",
    ])
    lines = [
        f"{a} × {b} = {a * b}页",
        f"{p} - {a * b} = {p - a * b}页",
        f"{p - a * b} ÷ {c} = {d}页",
    ]
    return ins, lines, d


_reg("book_reading_plan", book_reading_plan)


# 55. 盈亏问题
def profit_loss_kids(rng):
    a = rng.randint(5, 12)
    d = rng.randint(1, 3)
    b = a + d
    k = rng.randint(4, 12)
    m = rng.randint(3, min(15, d * k - 1))
    n = d * k - m
    total = a * k + m
    fruit = rng.choice(FRUITS)
    ins = rng.choice([
        f"老师把一些{fruit}分给小朋友，每人分{a}个，多出{m}个；每人分{b}个，又还差{n}个。一共有多少个小朋友？你会算吗？",
        f"把一袋{fruit}分给小朋友，每人{a}个多{m}个，每人{b}个少{n}个。有多少个小朋友？请列式算一算。",
        f"小朋友分{fruit}，每人分{a}个剩{m}个，每人分{b}个缺{n}个。一共有多少个小朋友？你能算出来吗？",
    ])
    lines = [
        f"每人多分的个数 = {b} - {a} = {d}",
        f"盈亏总数差 = {m} + {n} = {m + n}",
        f"{a} × {k} + {m} = {total}个",
        f"{m + n} ÷ {d} = {k}人",
    ]
    return ins, lines, k


_reg("profit_loss_kids", profit_loss_kids)


# 56. 三筐苹果还原
def three_jar_redistribute(rng):
    K = rng.randint(20, 50)
    s = 3 * K
    a = rng.randint(3, 10)
    b = rng.randint(3, min(12, K - 3))
    j1 = K + a
    j2 = K + b - a
    j3 = K - b
    which = rng.choice(["一", "二", "三"])
    ans = {"一": j1, "二": j2, "三": j3}[which]
    fruit = rng.choice(FRUITS)
    ins = rng.choice([
        f"三筐{fruit}共{s}个。先从第一筐拿{a}个放入第二筐，再从第二筐拿{b}个放入第三筐，这时三筐{fruit}同样多。第{which}筐原来有多少个？你会算吗？",
        f"三筐{fruit}一共{s}个，从第一筐取{a}个放进第二筐，再从第二筐取{b}个放进第三筐后，三筐一样多。第{which}筐原有多少个？请列式算一算。",
        f"有三筐{fruit}共{s}个，第一筐给第二筐{a}个，第二筐再给第三筐{b}个，这时三筐个数相等。第{which}筐原来有多少个？你能算出来吗？",
    ])
    lines = [
        f"{s} ÷ 3 = {K}个",
    ]
    baskets = {"一": (f"{K} + {a} = {j1}个", j1),
               "二": (f"{K} + {b} - {a} = {j2}个", j2),
               "三": (f"{K} - {b} = {j3}个", j3)}
    for w in ("一", "二", "三"):
        if w != which:
            lines.append(baskets[w][0])
    lines.append(baskets[which][0])
    return ins, lines, baskets[which][1]


_reg("three_jar_redistribute", three_jar_redistribute)


# 57. 绳子分数还原
def rope_fraction_reverse(rng):
    k = rng.randint(3, 12)
    x = 3 * k
    L = 10 * k
    ins = rng.choice([
        f"一根绳子，第一次用去全长的2/5，第二次用去余下的一半，这时还剩{x}米。这根绳子原来长多少米？你会算吗？请列式算一算。",
        f"一根绳子先用去全长的2/5，又用去剩下的一半，最后还剩{x}米。绳子原来长多少米？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"一根绳子，第一次剪去全长的2/5，第二次剪去余下的一半，还剩{x}米。这根绳子原来长多少米？你能算出来吗？",
    ])
    lines = [
        f"{x} × 2 = {2 * x}米",
        f"{2 * x} × 5 ÷ 3 = {L}米",
    ]
    return ins, lines, L


_reg("rope_fraction_reverse", rope_fraction_reverse)


# 58. 两位数数字交换
def digits_swap(rng):
    a = rng.randint(1, 7)
    diff = rng.randint(1, 9 - a)
    b = a + diff
    s = a + b
    d = 9 * diff
    orig = 10 * a + b
    ins = rng.choice([
        f"一个两位数，十位数字与个位数字的和是{s}。把两个数字交换位置后，新数比原数大{d}。原来的两位数是多少？你会算吗？",
        f"一个两位数，个位与十位数字之和是{s}，交换两个数字后所得的数比原数大{d}。求这个两位数。请列式算一算。",
        f"有一个两位数，十位数字加个位数字等于{s}，把十位和个位交换后新数比原数大{d}。这个两位数是多少？你能算出来吗？",
    ])
    lines = [
        f"数字之差 = {d} ÷ 9 = {diff}",
        f"十位数字的2倍 = {s} - {diff} = {2 * a}",
        f"十位数字 = {2 * a} ÷ 2 = {a}",
        f"个位数字 = {a} + {diff} = {b}",
        f"原两位数 = {a} × 10 + {b} = {orig}",
    ]
    return ins, lines, orig


_reg("digits_swap", digits_swap)


# 59. 往返平均速度
def avg_speed_round_trip(rng):
    v1 = t1 = dist = v2 = t2 = None
    for _ in range(100):
        v1 = rng.randint(40, 60)
        t1 = rng.randint(10, 20)
        dist = v1 * t1
        cands = [t for t in range(6, 16) if dist % t == 0 and dist // t > v1]
        if cands:
            t2 = rng.choice(cands)
            v2 = dist // t2
            break
    else:
        v1, t1, dist, t2, v2 = 50, 12, 600, 10, 60
    avg = Fraction(2 * dist, t1 + t2)
    place = rng.choice(PLACE)
    ins = rng.choice([
        f"小明从家出发去{place}，每分钟走{v1}米，用了{t1}分钟。原路返回时每分钟走{v2}米。往返的平均速度是每分钟多少米？你会算吗？",
        f"小明从家到{place}，每分钟行{v1}米，{t1}分钟到达。回来时每分钟行{v2}米。往返平均每分钟行多少米？请列式算一算。",
        f"小明去{place}时每分钟走{v1}米，用了{t1}分钟；返回时每分钟走{v2}米。往返的平均速度是多少？你能算出来吗？",
    ])
    lines = [
        f"{v1} × {t1} = {dist}米",
        f"{dist} ÷ {t2} = {v2}米/分",
        f"{dist} × 2 = {2 * dist}米",
        f"{t1} + {t2} = {t1 + t2}分",
        f"{2 * dist} ÷ {t1 + t2} = {num(avg)}米/分",
    ]
    return ins, lines, avg


_reg("avg_speed_round_trip", avg_speed_round_trip)


# 60. 比与差
def ratio_diff(rng):
    a = rng.randint(3, 7)
    b = rng.randint(2, a - 1)
    m = rng.randint(4, 20)
    d = (a - b) * m
    jia = a * m
    yi = b * m
    who = rng.choice(["甲", "乙"])
    ins = rng.choice([
        f"数学课上老师出了一道题：甲、乙两个数的比是{a}比{b}，甲数比乙数多{d}。{who}数是多少？你会算吗？请列式算一算。",
        f"甲数与乙数的比为{a}比{b}，且甲数比乙数大{d}。{who}数是多少？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"甲、乙两数之比是{a}比{b}，两数之差是{d}。{who}数是多少？请你列算式算一算，看谁算得又对又快。",
    ])
    lines = [
        f"份数差 = {a} - {b} = {a - b}",
        f"每份是多少 = {d} ÷ {a - b} = {m}",
    ]
    if who == "甲":
        lines.append(f"乙数 = {m} × {b} = {yi}")
        lines.append(f"甲数 = {m} × {a} = {jia}")
        return ins, lines, jia
    lines.append(f"甲数 = {m} × {a} = {jia}")
    lines.append(f"乙数 = {m} × {b} = {yi}")
    return ins, lines, yi


_reg("ratio_diff", ratio_diff)


# 61. 竞赛得分
def quiz_scoring(rng):
    n = rng.randint(10, 20)
    a = rng.randint(5, 10)
    b = rng.randint(2, 5)
    c = rng.randint(6, n - 2)
    s = a * c - b * (n - c)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一次知识竞赛共有{n}道题，答对一题得{a}分，答错一题倒扣{b}分。{name}答完了所有题，共得{s}分。他答对了几道题？你会算吗？",
        f"竞赛共{n}道题，答对一题得{a}分，答错一题扣{b}分。{name}全部答完得{s}分，他答对了几道题？请列式算一算。",
        f"知识竞赛有{n}道题，答对一题加{a}分，答错一题减{b}分。{name}答完所有题得{s}分，答对几道题？你能算出来吗？",
    ])
    lines = [
        f"答对答错的分差 = {a} + {b} = {a + b}",
        f"得分加回倒扣的分 = {s} + {b} × {n} = {s + b * n}",
        f"{s + b * n} ÷ {a + b} = {c}题",
    ]
    return ins, lines, c


_reg("quiz_scoring", quiz_scoring)


# 62. 修路实际天数
def road_actual_days(rng):
    a = t = total = b = d = None
    for _ in range(100):
        a = rng.randint(40, 80)
        t = rng.randint(12, 30)
        total = a * t
        cands = [dd for dd in range(8, 26) if total % dd == 0]
        if cands:
            d = rng.choice(cands)
            b = total // d - a
            if 5 <= b <= 40:
                break
    else:
        a, t, total, d, b = 60, 20, 1200, 15, 20
    ins = rng.choice([
        f"工程队原计划每天修{a}米，{t}天修完一条路。实际每天比原计划多修{b}米，实际多少天修完？你会算吗？请列式算一算。",
        f"修一条路，计划每天修{a}米，{t}天完成。实际每天多修{b}米，实际用了多少天？{rng.choice(NAMES)}不会算，请你帮帮他。",
        f"工程队修一条路，原计划每天{a}米、{t}天完工。实际每天修{a + b}米，比原计划多{b}米，实际多少天完工？你能算出来吗？",
    ])
    lines = [
        f"{a} × {t} = {total}米",
        f"{a} + {b} = {a + b}米",
        f"{total} ÷ {a + b} = {d}天",
    ]
    return ins, lines, d


_reg("road_actual_days", road_actual_days)


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
    print(f"L4 ext2 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
