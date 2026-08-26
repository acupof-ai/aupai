#!/usr/bin/env python3
"""L3 ext4: 64 distinct 5-7 step families (percent/fraction/ratio/decimal/geometry).

Every program: fn(rng) -> (instruction, lines, ans); >=3 equation lines;
>=4 phrasings; all non-integer arithmetic via Fraction, rendered with num().
Verified against run_math_short.verify.
"""
import math
import random
import re
from fractions import Fraction
from mathcommon import (ANIMALS, FOOD, FRUITS, GOODS, NAMES, PLACE, STATIONERY,
                         UNIT_FRUIT, UNIT_N, UNIT_ZHI, num)

PROGRAMS = []

# Chinese label per bare line (a line whose RHS carries no unit/Chinese), in order.
_LABELS = {}


def _reg(name, fn):
    def wrapped(rng):
        ins, lines, ans = fn(rng)
        labels = _LABELS.get(name, ())
        out, i = [], 0
        for ln in lines:
            parts = ln.split("=")
            if len(parts) == 2 and not re.search(r"[一-鿿]", parts[1]):
                out.append(f"{labels[i]} = {ln}")
                i += 1
            else:
                out.append(ln)
        return ins, out, ans
    PROGRAMS.append(("L3", name, wrapped))


# 1. two days sell p1%, p2% of the ORIGINAL stock, left X -> original
def pct_two_stage_reverse(rng):
    p1 = rng.choice([10, 15, 20, 25, 30])
    p2 = rng.choice([10, 15, 20, 25, 30])
    for _ in range(50):
        if p1 + p2 < 80:
            break
        p2 = rng.choice([10, 15, 20, 25, 30])
    k = rng.randint(2, 20)
    total = 100 * k
    X = (100 - p1 - p2) * k
    unit = rng.choice(["吨", "千克", "本", "升", "件"])
    obj = rng.choice(["货物", "水果", "图书", "饮料", "商品"])
    ins = rng.choice([
        f"商店有一批{obj}，第一天卖出{p1}%，第二天卖出{p2}%，还剩{X}{unit}，这批{obj}原来有多少{unit}？",
        f"仓库运来一批{obj}，上午运走{p1}%，下午运走{p2}%，还剩{X}{unit}，这批{obj}共多少{unit}？",
        f"书店新进一批{obj}，第一天卖出总数的{p1}%，第二天卖出总数的{p2}%，还剩{X}{unit}，新进多少{unit}？",
        f"一批{obj}，第一次运走{p1}%，第二次运走{p2}%，还剩{X}{unit}，这批{obj}原有多少{unit}？",
    ])
    lines = [
        f"{p1} + {p2} = {p1 + p2}",
        f"100 - {p1 + p2} = {100 - p1 - p2}",
        f"{X} × 100 = {X * 100}",
        f"{X * 100} ÷ {100 - p1 - p2} = {total}{unit}",
    ]
    return ins, lines, total


_LABELS["pct_two_stage_reverse"] = ["两天卖出的百分比", "剩下的百分比", "剩下的量扩大100倍"]
_reg("pct_two_stage_reverse", pct_two_stage_reverse)


# 2. first 1/d, then p% of the REMAINDER, left X -> original
def frac_pct_reverse_original(rng):
    d = rng.choice([3, 4, 5, 6])
    p = rng.choice([10, 20, 25, 40, 50, 60, 75])
    k = rng.randint(2, 30)
    total = d * 100 * k
    X = (d - 1) * (100 - p) * k
    unit = rng.choice(["吨", "千克", "本", "升"])
    obj = rng.choice(["货物", "大米", "图书", "饮料"])
    ins = rng.choice([
        f"仓库有一批{obj}，第一次运走1/{d}，第二次运走余下的{p}%，还剩{X}{unit}，这批{obj}原来有多少{unit}？",
        f"一根绳子，第一次剪去全长的1/{d}，第二次剪去剩下的{p}%，还剩{X}米，绳子原来长多少米？",
        f"书店有一批书，第一天卖出1/{d}，第二天卖出余下的{p}%，还剩{X}本，这批书原来有多少本？",
        f"食堂有一批{obj}，第一周用去1/{d}，第二周用去余下的{p}%，还剩{X}{unit}，原来有多少{unit}？",
    ])
    lines = [
        f"{d} - 1 = {d - 1}",
        f"100 - {p} = {100 - p}",
        f"{X} × {d} = {X * d}",
        f"{X * d} × 100 = {X * d * 100}",
        f"{X * d * 100} ÷ ({d - 1} × {100 - p}) = {total}{unit}",
    ]
    return ins, lines, total


_LABELS["frac_pct_reverse_original"] = ["第一次后剩下的份数", "第二次后剩下的百分比",
                                        "剩下的量乘d", "再乘100"]
_reg("frac_pct_reverse_original", frac_pct_reverse_original)


# 3. A has p% more than B; A gives B x yuan then equal -> B's original
def pct_transfer_diff(rng):
    p = rng.choice([10, 20, 25, 40, 50])
    k = rng.choice([2, 4, 6, 8, 10])
    B = 100 * k
    x = p * k // 2
    obj = rng.choice(["元", "本书", "颗糖", "张邮票"])
    who = rng.choice([("甲", "乙"), ("哥哥", "弟弟"), ("小明", "小红"), ("姐姐", "妹妹")])
    a, b = who
    ins = rng.choice([
        f"{a}的钱数比{b}多{p}%，{a}给{b}{x}{obj}后两人钱数相等，{b}原来有多少{obj}？",
        f"{a}的{obj}比{b}多{p}%，{a}拿出{x}{obj}给{b}后两人一样多，{b}原有多少{obj}？",
        f"{a}比{b}多{p}%的{obj}，{a}给{b}{x}{obj}后两人相等，{b}原来有多少{obj}？",
        f"{a}的钱比{b}多{p}%，{a}给{b}{x}元后两人钱数相同，{b}原来有多少元？",
    ])
    lines = [
        f"{x} × 2 = {2 * x}",
        f"{2 * x} × 100 = {2 * x * 100}",
        f"{2 * x * 100} ÷ {p} = {B}{obj}",
    ]
    return ins, lines, B


_LABELS["pct_transfer_diff"] = ["两人钱数的差", "差扩大100倍"]
_reg("pct_transfer_diff", pct_transfer_diff)


# 4. used a/b of a rope PLUS c extra meters, left X -> rope length
def fraction_plus_excess(rng):
    b = rng.choice([3, 4, 5, 6])
    a = rng.randint(1, b - 1)
    c = rng.randint(2, 20)
    lo = c // (b - a) + 1
    k = rng.randint(lo, lo + 19)
    L = b * k
    X = (b - a) * k - c
    obj = rng.choice(["绳子", "彩带", "铁丝", "电线"])
    unit = rng.choice(["米", "米", "米", "米"])
    ins = rng.choice([
        f"一根{obj}，第一次用去全长的{a}/{b}还多{c}米，还剩{X}米，这根{obj}全长多少米？",
        f"一根{obj}长若干米，用去它的{a}/{b}多{c}米后，还剩{X}米，{obj}原来长多少米？",
        f"一根{obj}，剪去全长的{a}/{b}又{c}米，还剩{X}米，这根{obj}长多少米？",
        f"一根{obj}，用去{a}/{b}还多{c}米，剩下{X}米，{obj}全长多少米？",
    ])
    lines = [
        f"{X} + {c} = {X + c}米",
        f"{b} - {a} = {b - a}",
        f"{X + c} × {b} = {(X + c) * b}米",
        f"{(X + c) * b} ÷ {b - a} = {L}米",
    ]
    return ins, lines, L


_LABELS["fraction_plus_excess"] = ["剩下的份数差"]
_reg("fraction_plus_excess", fraction_plus_excess)


# 5. A*(1/a) == B*(1/b), A - B = X -> A
def frac_equal_parts_diff(rng):
    a = rng.randint(3, 8)
    b = rng.randint(2, a - 1)
    k = rng.randint(2, 30)
    A = a * k
    X = (a - b) * k
    obj = rng.choice(["", "吨货物", "元钱", "本书"])
    ins = rng.choice([
        f"甲数的1/{a}等于乙数的1/{b}，甲数比乙数多{X}，甲数是多少？",
        f"甲堆货物的1/{a}等于乙堆货物的1/{b}，甲堆比乙堆多{X}吨，甲堆有多少吨？",
        f"小明钱数的1/{a}等于小红钱数的1/{b}，小明比小红多{X}元，小明有多少元？",
        f"甲仓存粮的1/{a}等于乙仓存粮的1/{b}，甲仓比乙仓多{X}吨，甲仓存粮多少吨？",
    ])
    lines = [
        f"{a} - {b} = {a - b}",
        f"{X} × {a} = {X * a}",
        f"{X * a} ÷ {a - b} = {A}",
    ]
    return ins, lines, A


_LABELS["frac_equal_parts_diff"] = ["份数差", "差乘a", "甲数"]
_reg("frac_equal_parts_diff", frac_equal_parts_diff)


# 6. three days use 1/a, 1/b, 1/c of the ORIGINAL, left X -> original
def three_fractions_remainder(rng):
    a = rng.choice([3, 4, 5, 6])
    b = rng.choice([3, 4, 5, 6])
    c = rng.choice([3, 4, 5, 6])
    for _ in range(50):
        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) < 1:
            break
        c = rng.choice([3, 4, 5, 6])
    k = rng.randint(2, 15)
    abc = a * b * c
    rest = abc - a * b - a * c - b * c
    X = rest * k
    total = abc * k
    unit = rng.choice(["吨", "千克", "本", "升"])
    obj = rng.choice(["货物", "大米", "图书", "饮料"])
    ins = rng.choice([
        f"一批{obj}，第一天运走1/{a}，第二天运走1/{b}，第三天运走1/{c}，还剩{X}{unit}，这批{obj}原来有多少{unit}？",
        f"仓库有一批{obj}，第一周用去1/{a}，第二周用去1/{b}，第三周用去1/{c}，还剩{X}{unit}，原有多少{unit}？",
        f"一根绳子，第一次剪去1/{a}，第二次剪去1/{b}，第三次剪去1/{c}，还剩{X}米，绳子原长多少米？",
        f"书店有一批书，第一天卖出1/{a}，第二天卖出1/{b}，第三天卖出1/{c}，还剩{X}本，这批书原有多少本？",
    ])
    lines = [
        f"{a} × {b} × {c} = {abc}",
        f"{abc} - {a * b} - {a * c} - {b * c} = {rest}",
        f"{X} × {abc} = {X * abc}",
        f"{X * abc} ÷ {rest} = {total}{unit}",
    ]
    return ins, lines, total


_LABELS["three_fractions_remainder"] = ["分母的积", "剩下的份数", "剩下的量乘分母积"]
_reg("three_fractions_remainder", three_fractions_remainder)


# 7. cut 1/a, then 1/b of the REMAINDER, left X -> original (two-stage reverse)
def reverse_two_stage(rng):
    a = rng.choice([3, 4, 5, 6])
    b = rng.choice([2, 3, 4, 5])
    k = rng.randint(2, 30)
    total = a * b * k
    X = (a - 1) * (b - 1) * k
    obj = rng.choice(["绳子", "彩带", "铁丝", "电线"])
    ins = rng.choice([
        f"一根{obj}，第一次剪去全长的1/{a}，第二次剪去余下的1/{b}，还剩{X}米，这根{obj}原来长多少米？",
        f"一根{obj}，第一次用去1/{a}，第二次用去剩下的1/{b}，还剩{X}米，{obj}原长多少米？",
        f"一根{obj}，先剪去1/{a}，再剪去余下的1/{b}，还剩{X}米，原来长多少米？",
        f"一根{obj}，第一次截去1/{a}，第二次截去余下的1/{b}，还剩{X}米，这根{obj}长多少米？",
    ])
    lines = [
        f"{a} - 1 = {a - 1}",
        f"{b} - 1 = {b - 1}",
        f"{X} × {a} × {b} = {X * a * b}",
        f"{X * a * b} ÷ ({a - 1} × {b - 1}) = {total}米",
    ]
    return ins, lines, total


_LABELS["reverse_two_stage"] = ["第一次后剩下的份数", "第二次后剩下的份数", "剩下的量乘分母积"]
_reg("reverse_two_stage", reverse_two_stage)


# 8. used p% PLUS/MINUS c extra, left X -> original
def pct_excess_short_reverse(rng):
    p = rng.choice([10, 20, 25, 30, 40])
    c = rng.randint(2, 30)
    lo = c // (100 - p) + 1
    k = rng.randint(lo, lo + 29)
    L = 100 * k
    more = rng.choice([True, False])
    if more:
        X = (100 - p) * k - c
        used = f"用去它的{p}%还多{c}千克"
    else:
        X = (100 - p) * k + c
        used = f"用去它的{p}%少{c}千克"
    obj = rng.choice(["一桶油", "一袋大米", "一筐水果", "一批货物"])
    unit = rng.choice(["千克", "千克", "千克", "吨"])
    ins = rng.choice([
        f"{obj}，{used}，还剩{X}{unit}，这{obj[1:]}原来重多少{unit}？",
        f"{obj}重若干{unit}，{used}后，还剩{X}{unit}，原来重多少{unit}？",
        f"{obj}，{used}，剩下{X}{unit}，原来有多少{unit}？",
        f"{obj}，{used}，还剩{X}{unit}，原来重多少{unit}？",
    ])
    if more:
        lines = [
            f"{X} + {c} = {X + c}{unit}",
            f"100 - {p} = {100 - p}",
            f"{X + c} × 100 = {(X + c) * 100}",
            f"{(X + c) * 100} ÷ {100 - p} = {L}{unit}",
        ]
    else:
        lines = [
            f"{X} - {c} = {X - c}{unit}",
            f"100 - {p} = {100 - p}",
            f"{X - c} × 100 = {(X - c) * 100}",
            f"{(X - c) * 100} ÷ {100 - p} = {L}{unit}",
        ]
    return ins, lines, L


_LABELS["pct_excess_short_reverse"] = ["剩下的百分比", "被除数扩大100倍"]
_reg("pct_excess_short_reverse", pct_excess_short_reverse)


# 9. A:B:C = a:b:c, A+B = S -> C
def ratio_three_sum_given(rng):
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    c = rng.randint(2, 6)
    k = rng.randint(3, 30)
    S = (a + b) * k
    C = c * k
    obj = rng.choice(["", "本书", "元钱", "吨货物"])
    ins = rng.choice([
        f"甲、乙、丙三个数的比是{a}:{b}:{c}，甲、乙两数之和是{S}，丙数是多少？",
        f"三个班分得的图书本数比是{a}:{b}:{c}，一班和二班共分得{S}本，三班分得多少本？",
        f"甲、乙、丙三人的钱数比是{a}:{b}:{c}，甲、乙共有{S}元，丙有多少元？",
        f"三堆货物的重量比是{a}:{b}:{c}，甲、乙两堆共重{S}吨，丙堆重多少吨？",
    ])
    lines = [
        f"{a} + {b} = {a + b}",
        f"{S} ÷ {a + b} = {k}",
        f"{c} × {k} = {C}",
    ]
    return ins, lines, C


_LABELS["ratio_three_sum_given"] = ["份数和", "每份", "丙数"]
_reg("ratio_three_sum_given", ratio_three_sum_given)


# 10. A:B = a:b, B:C = c:d, A = C - X -> B
def continued_ratio_diff(rng):
    pairs = [(2, 3), (3, 4), (4, 5), (3, 5), (2, 5), (5, 6), (2, 7), (3, 7)]
    a, b = rng.choice(pairs)
    c, d = rng.choice(pairs)
    for _ in range(50):
        if b * d > a * c:
            break
        c, d = rng.choice(pairs)
    k = rng.randint(2, 20)
    X = (b * d - a * c) * k
    B = b * c * k
    ins = rng.choice([
        f"甲数与乙数的比是{a}:{b}，乙数与丙数的比是{c}:{d}，甲数比丙数少{X}，乙数是多少？",
        f"甲、乙的钱数比是{a}:{b}，乙、丙的钱数比是{c}:{d}，甲比丙少{X}元，乙有多少元？",
        f"三个组的人数比中，一组与二组的比是{a}:{b}，二组与三组的比是{c}:{d}，一组比三组少{X}人，二组有多少人？",
        f"甲:乙={a}:{b}，乙:丙={c}:{d}，丙比甲多{X}，乙数是多少？",
    ])
    lines = [
        f"{a} × {c} = {a * c}",
        f"{b} × {d} = {b * d}",
        f"{b * d} - {a * c} = {b * d - a * c}",
        f"{X} ÷ {b * d - a * c} = {k}",
        f"{b} × {c} × {k} = {B}",
    ]
    return ins, lines, B


_LABELS["continued_ratio_diff"] = ["甲的份数", "丙的份数", "份数差", "每份", "乙数"]
_reg("continued_ratio_diff", continued_ratio_diff)


# 11. A:B = a:b, both spend x, new ratio c:(c+1) -> A's original
def ratio_spend_equal(rng):
    a = rng.randint(4, 7)
    b = rng.randint(2, a - 1)
    c = rng.randint(2, 5)
    d = c + 1
    for _ in range(50):
        if a * d > b * c:
            break
        b = rng.randint(2, a - 1)
    k = rng.randint(2, 15)
    x = k * (a * d - b * c)
    A = a * k
    obj = rng.choice(["元", "本书", "颗糖"])
    ins = rng.choice([
        f"甲、乙两人的钱数比是{a}:{b}，两人各花去{x}{obj}后，钱数比变成{c}:{d}，甲原来有多少{obj}？",
        f"甲、乙两堆货物的比是{a}:{b}，各运走{x}吨后，剩下的比是{c}:{d}，甲堆原有多少吨？",
        f"小明和小红的邮票数比是{a}:{b}，两人都用掉{x}张后，比是{c}:{d}，小明原来有多少张？",
        f"甲、乙两数的比是{a}:{b}，两数各减去{x}后，比是{c}:{d}，甲数是多少？",
    ])
    lines = [
        f"{a} × {d} = {a * d}",
        f"{b} × {c} = {b * c}",
        f"{a * d} - {b * c} = {a * d - b * c}",
        f"{d} - {c} = {d - c}",
        f"{x} ÷ {a * d - b * c} = {k}",
        f"{a} × {k} = {A}{obj}",
    ]
    return ins, lines, A


_LABELS["ratio_spend_equal"] = ["甲的新份数积", "乙的新份数积", "份数差", "后项差", "每份"]
_reg("ratio_spend_equal", ratio_spend_equal)


# 12. father:son = a:b now, 10 years later = c:d -> son's age now
_AGE_RATIO = []
for a in range(3, 10):
    for b in range(1, 6):
        for d in range(1, 5):
            for c in range(d + 1, d + 5):
                den = a * d - b * c
                if den <= 0:
                    continue
                val = 10 * (c - d)
                if val % den:
                    continue
                k = val // den
                if 28 <= a * k <= 56 and 3 <= b * k <= 18:
                    _AGE_RATIO.append((a, b, c, d, k))


def ratio_age_years(rng):
    a, b, c, d, k = rng.choice(_AGE_RATIO)
    son = b * k
    rel = rng.choice([("父亲", "儿子"), ("爸爸", "小明"), ("母亲", "女儿"),
                      ("妈妈", "小红"), ("爷爷", "孙子"), ("奶奶", "孙女")])
    fa, ch = rel
    ins = rng.choice([
        f"今年{fa}与{ch}的年龄比是{a}:{b}，10年后两人年龄比是{c}:{d}，{ch}今年多少岁？",
        f"今年{fa}和{ch}的年龄比为{a}:{b}，10年后比为{c}:{d}，{ch}今年几岁？",
        f"{fa}与{ch}今年的年龄比是{a}:{b}，再过10年两人年龄比是{c}:{d}，{ch}今年多少岁？",
        f"今年{fa}年龄是{ch}的{a}/{b}倍，10年后{fa}年龄是{ch}的{c}/{d}倍，{ch}今年多少岁？",
    ])
    lines = [
        f"{c} - {d} = {c - d}",
        f"{c - d} × 10 = {(c - d) * 10}",
        f"{a} × {d} = {a * d}",
        f"{b} × {c} = {b * c}",
        f"{a * d} - {b * c} = {a * d - b * c}",
        f"{(c - d) * 10} ÷ {a * d - b * c} = {k}",
        f"{b} × {k} = {son}岁",
    ]
    return ins, lines, son


_LABELS["ratio_age_years"] = ["后项差", "差乘10", "父的份数积", "子的份数积", "份数差", "每份年龄"]
_reg("ratio_age_years", ratio_age_years)


# 13. A:B:C = a:b:c, A gives C x then all equal -> A's original
def ratio_three_transfer(rng):
    a = rng.randint(4, 8)
    b = rng.randint(1, 5)
    c = rng.randint(1, 5)
    for _ in range(50):
        if 2 * a > b + c:
            break
        b = rng.randint(1, 5)
        c = rng.randint(1, 5)
    diff = 2 * a - b - c
    k = rng.choice([3, 6, 9, 12])
    for _ in range(50):
        if (k * diff) % 3 == 0:
            break
        k = rng.choice([3, 6, 9, 12])
    x = k * diff // 3
    A = a * k
    obj = rng.choice(["元", "本书", "颗糖", "张邮票"])
    ins = rng.choice([
        f"甲、乙、丙三人的钱数比是{a}:{b}:{c}，甲给丙{x}{obj}后三人钱数相等，甲原来有多少{obj}？",
        f"甲、乙、丙三堆货物的比是{a}:{b}:{c}，从甲堆运{x}吨到丙堆后三堆相等，甲堆原有多少吨？",
        f"三人的邮票数比是{a}:{b}:{c}，甲给丙{x}张后三人一样多，甲原来有多少张？",
        f"甲、乙、丙的钱数比为{a}:{b}:{c}，甲拿出{x}元给丙后三人钱数相同，甲原有多少元？",
    ])
    lines = [
        f"{a} + {b} + {c} = {a + b + c}",
        f"2 × {a} - {b} - {c} = {diff}",
        f"{x} × 3 = {3 * x}",
        f"{3 * x} ÷ {diff} = {k}",
        f"{a} × {k} = {A}{obj}",
    ]
    return ins, lines, A


_LABELS["ratio_three_transfer"] = ["份数和", "甲比平均数多的份数", "转移量的3倍", "每份"]
_reg("ratio_three_transfer", ratio_three_transfer)


# 14. income ratio a:b, expense ratio c:(c-1), both save x -> A's income
def income_expense_ratio(rng):
    a = rng.randint(4, 9)
    b = rng.randint(3, a - 1)
    c = rng.randint(3, 8)
    d = c - 1
    for _ in range(50):
        if b * c > a * d:
            break
        a = rng.randint(4, 9)
        b = rng.randint(3, a - 1)
        c = rng.randint(3, 8)
        d = c - 1
    k = rng.randint(10, 60)
    x = k * (b * c - a * d)
    A = a * k
    ins = rng.choice([
        f"甲、乙两人月收入之比是{a}:{b}，月支出之比是{c}:{d}，两人每月都结余{x}元，甲每月收入多少元？",
        f"甲、乙两个工厂的收入比是{a}:{b}，支出比是{c}:{d}，两厂都结余{x}万元，甲厂收入多少万元？",
        f"兄弟两人月收入比为{a}:{b}，花销比为{c}:{d}，每月都剩{x}元，哥哥月收入多少元？",
        f"甲、乙两店的营业额比是{a}:{b}，成本比是{c}:{d}，两店都盈利{x}元，甲店营业额多少元？",
    ])
    lines = [
        f"{a} - {b} = {a - b}",
        f"{c} - {d} = {c - d}",
        f"{b} × {c} = {b * c}",
        f"{a} × {d} = {a * d}",
        f"{b * c} - {a * d} = {b * c - a * d}",
        f"{x} ÷ {b * c - a * d} = {k}",
        f"{a} × {k} = {A}元",
    ]
    return ins, lines, A


_LABELS["income_expense_ratio"] = ["收入份数差", "支出份数差", "乙的份数积", "甲的份数积",
                                   "份数积的差", "每份收入"]
_reg("income_expense_ratio", income_expense_ratio)


# 15. A:B = a:b, both add x, new sum S -> A
def ratio_add_sum(rng):
    a = rng.randint(2, 7)
    b = rng.randint(2, 7)
    for _ in range(50):
        if a != b:
            break
        b = rng.randint(2, 7)
    k = rng.randint(3, 30)
    x = rng.randint(2, 20)
    S = (a + b) * k + 2 * x
    A = a * k
    obj = rng.choice(["", "本书", "元钱", "吨货物"])
    ins = rng.choice([
        f"甲、乙两数的比是{a}:{b}，两数各增加{x}后，和是{S}，甲数是多少？",
        f"甲、乙两堆货物的比是{a}:{b}，各运来{x}吨后，两堆共重{S}吨，甲堆原有多少吨？",
        f"小明和小红的邮票数比是{a}:{b}，两人都又收集了{x}张后，共有{S}张，小明原来有多少张？",
        f"甲、乙两班人数比是{a}:{b}，两班都转入{x}人后，共有{S}人，甲班原有多少人？",
    ])
    lines = [
        f"{x} × 2 = {2 * x}",
        f"{S} - {2 * x} = {S - 2 * x}",
        f"{a} + {b} = {a + b}",
        f"{S - 2 * x} ÷ {a + b} = {k}",
        f"{a} × {k} = {A}",
    ]
    return ins, lines, A


_LABELS["ratio_add_sum"] = ["增加的和", "原来两数的和", "份数和", "每份", "甲数"]
_reg("ratio_add_sum", ratio_add_sum)


# 16. meet after t hours, total distance D, one speed known -> other speed
def meet_speed_find(rng):
    v1 = rng.randint(40, 80)
    v2 = rng.randint(40, 80)
    for _ in range(50):
        if v1 != v2:
            break
        v2 = rng.randint(40, 80)
    t = rng.randint(2, 6)
    D = (v1 + v2) * t
    who = rng.choice([("甲", "乙"), ("小明", "小红"), ("快车", "慢车"), ("哥哥", "弟弟")])
    a, b = who
    ins = rng.choice([
        f"甲、乙两地相距{D}千米，{a}、{b}两人同时相向而行，{t}小时后相遇，{a}每小时走{v1}千米，{b}每小时走多少千米？",
        f"两地相距{D}千米，{a}车和{b}车同时相向开出，{t}小时后相遇，{a}车每小时行{v1}千米，{b}车每小时行多少千米？",
        f"{a}、{b}从相距{D}千米的两地同时出发相向而行，{t}小时相遇，{a}每小时行{v1}千米，{b}每小时行多少千米？",
        f"甲、乙两城相距{D}千米，{a}、{b}两车同时相向而行，{t}小时后相遇，{a}车速度是每小时{v1}千米，{b}车速度是多少？",
    ])
    lines = [
        f"{v1} + {v2} = {v1 + v2}千米/时",
        f"{v1 + v2} × {t} = {D}千米",
        f"{D} ÷ {t} = {v1 + v2}千米/时",
        f"{v1 + v2} - {v1} = {v2}千米/时",
    ]
    return ins, lines, v2


_reg("meet_speed_find", meet_speed_find)


# 17. round trip total time T, two speeds -> one-way distance
def roundtrip_find_distance(rng):
    v1 = rng.choice([40, 50, 60, 80])
    v2 = rng.choice([40, 50, 60, 80])
    for _ in range(50):
        if v1 != v2:
            break
        v2 = rng.choice([40, 50, 60, 80])
    g = math.gcd(v1, v2)
    lcm = v1 * v2 // g
    k = rng.randint(1, 4)
    D = lcm * k
    T = D // v1 + D // v2
    ins = rng.choice([
        f"一艘船往返于两港之间，去时每小时行{v1}千米，返回时每小时行{v2}千米，往返一次共需{T}小时，两港相距多少千米？",
        f"一辆汽车往返甲、乙两地，去时每小时{v1}千米，回来每小时{v2}千米，往返共用{T}小时，甲、乙两地相距多少千米？",
        f"小明骑车去外婆家，去时每小时行{v1}千米，原路返回每小时行{v2}千米，往返一共用了{T}小时，家到外婆家多少千米？",
        f"一艘轮船往返两个码头，顺水每小时{v1}千米，逆水每小时{v2}千米，往返一次用{T}小时，两个码头相距多少千米？",
    ])
    lines = [
        f"{v1} + {v2} = {v1 + v2}",
        f"{v1} × {v2} = {v1 * v2}",
        f"{T} × {v1 * v2} = {T * v1 * v2}",
        f"{T * v1 * v2} ÷ {v1 + v2} = {D}千米",
    ]
    return ins, lines, D


_LABELS["roundtrip_find_distance"] = ["速度和", "速度积", "时间乘速度积"]
_reg("roundtrip_find_distance", roundtrip_find_distance)


# 18. A ahead by X, B catches in t minutes, A's speed known -> B's speed
def catch_speed_find(rng):
    vA = rng.randint(50, 80)
    t = rng.randint(2, 10)
    k = rng.randint(2, 20)
    X = k * t
    vB = vA + k
    who = rng.choice([("甲", "乙"), ("小明", "小红"), ("哥哥", "弟弟"), ("快车", "慢车")])
    a, b = who
    ins = rng.choice([
        f"{a}在{b}前面{X}米，{b}以每分钟{vB}米的速度追赶，{t}分钟后追上{a}，{a}每分钟走多少米？",
        f"{a}每分钟走{vA}米，{b}在{a}后面{X}米，{b}用{t}分钟追上{a}，{b}每分钟走多少米？",
        f"弟弟在哥哥前面{X}米，哥哥每分钟跑{vB}米，{t}分钟后追上弟弟，弟弟每分钟跑多少米？",
        f"{a}、{b}两人相距{X}米，{b}在后追赶，{t}分钟追上，{a}每分钟行{vA}米，{b}每分钟行多少米？",
    ])
    lines = [
        f"{vB} - {vA} = {k}米/分",
        f"{k} × {t} = {X}米",
        f"{vA} + {k} = {vB}米/分",
    ]
    return ins, lines, vB


_reg("catch_speed_find", catch_speed_find)


# 19. two legs: first speed v1 for t1, rest at v2 -> total time
def two_leg_time_find(rng):
    v1 = rng.randint(40, 70)
    t1 = rng.randint(2, 5)
    v2 = rng.randint(50, 90)
    k = rng.randint(2, 8)
    rem = v2 * k
    D = v1 * t1 + rem
    t2 = Fraction(rem, v2)
    total = t1 + t2
    ins = rng.choice([
        f"甲、乙两地相距{D}千米，一辆汽车先以每小时{v1}千米的速度行驶{t1}小时，余下的路程每小时行{v2}千米，全程共需多少小时？",
        f"一段路长{D}千米，小明骑车先走{t1}小时，每小时{v1}千米，剩下的路每小时行{v2}千米，全程用多少小时？",
        f"两地相距{D}千米，火车先以每小时{v1}千米的速度行了{t1}小时，余下路程每小时行{v2}千米，行完全程要几小时？",
        f"甲、乙两城相距{D}千米，汽车第一小时起以每小时{v1}千米行了{t1}小时，之后每小时行{v2}千米，全程需多少小时？",
    ])
    lines = [
        f"{v1} × {t1} = {v1 * t1}千米",
        f"{D} - {v1 * t1} = {rem}千米",
        f"{rem} ÷ {v2} = {num(t2)}小时",
        f"{t1} + {num(t2)} = {num(total)}小时",
    ]
    return ins, lines, total


_reg("two_leg_time_find", two_leg_time_find)


# 20. planned speed v, actual slower by delta, distance D -> delay
def speed_delay_find(rng):
    v = rng.choice([50, 60, 75, 80, 90])
    delta = rng.choice([10, 15, 20, 25, 30])
    for _ in range(50):
        if v - delta > 0:
            break
        delta = rng.choice([10, 15, 20, 25, 30])
    lcm = (v * (v - delta)) // math.gcd(v, v - delta)
    k = rng.randint(1, 3)
    D = lcm * k
    Tp = Fraction(D, v)
    Ta = Fraction(D, v - delta)
    delay = Ta - Tp
    ins = rng.choice([
        f"一辆汽车计划每小时行{v}千米，实际每小时比计划慢{delta}千米，两地相距{D}千米，实际比计划晚到多少小时？",
        f"小明原计划每小时骑{v}千米，实际每小时少骑{delta}千米，到{D}千米外的外婆家，实际比计划晚几小时？",
        f"一列火车计划时速{v}千米，因天气实际时速减少{delta}千米，行驶{D}千米后，比计划晚到多少小时？",
        f"一辆货车计划每小时行{v}千米，实际每小时行{v - delta}千米，行{D}千米比计划多用多少小时？",
    ])
    lines = [
        f"{v} - {delta} = {v - delta}千米/时",
        f"{D} ÷ {v} = {num(Tp)}小时",
        f"{D} ÷ {v - delta} = {num(Ta)}小时",
        f"{num(Ta)} - {num(Tp)} = {num(delay)}小时",
    ]
    return ins, lines, delay


_reg("speed_delay_find", speed_delay_find)


# 21. two trains pass each other: lengths L1, L2, speeds v1, v2 -> time
def train_meet_pass(rng):
    L1 = rng.randint(100, 300)
    L2 = rng.randint(100, 300)
    v1 = rng.choice([15, 20, 25, 30])
    v2 = rng.choice([15, 20, 25, 30])
    for _ in range(50):
        if (L1 + L2) % (v1 + v2) == 0:
            break
        v2 = rng.choice([15, 20, 25, 30])
    L = L1 + L2
    v = v1 + v2
    t = Fraction(L, v)
    ins = rng.choice([
        f"两列火车相向而行，甲车长{L1}米，乙车长{L2}米，甲车每秒行{v1}米，乙车每秒行{v2}米，从车头相遇到车尾离开需要多少秒？",
        f"一列慢车长{L1}米，一列快车长{L2}米，两车相向而行，慢车每秒{v1}米，快车每秒{v2}米，从相遇到错开需要多少秒？",
        f"甲、乙两列火车长分别为{L1}米和{L2}米，相向行驶，速度分别为每秒{v1}米和{v2}米，两车从车头相遇到车尾相离共需多少秒？",
        f"两列火车相向而行，车长分别是{L1}米和{L2}米，速度分别是每秒{v1}米和{v2}米，从车头相遇到车尾离开要几秒钟？",
    ])
    lines = [
        f"{L1} + {L2} = {L}米",
        f"{v1} + {v2} = {v}米/秒",
        f"{L} ÷ {v} = {num(t)}秒",
    ]
    return ins, lines, t


_reg("train_meet_pass", train_meet_pass)


# 22. wheel diameter d, n turns -> distance (pi = 3.14)
def wheel_distance(rng):
    a = rng.choice([5, 6, 7, 8, 10, 12])
    d = Fraction(a, 10)
    n = rng.randint(5, 30)
    r = d / 2
    C = 2 * Fraction(314, 100) * r
    D = C * n
    obj = rng.choice(["自行车", "三轮车", "手推车", "玩具车"])
    ins = rng.choice([
        f"一辆{obj}的车轮直径是{num(d)}米，车轮转了{n}圈（π取3.14），{obj}前进了多少米？",
        f"一种车轮的直径是{num(d)}米，滚动{n}圈（π取3.14），车轮前进多少米？",
        f"{obj}车轮的外直径是{num(d)}米，在路上滚动{n}周（π取3.14），车轮压过的路面长多少米？",
        f"一个直径{num(d)}米的车轮，转{n}圈（π取3.14），一共前进多少米？",
    ])
    lines = [
        f"{num(d)} ÷ 2 = {num(r)}米",
        f"2 × 3.14 × {num(r)} = {num(C)}米",
        f"{num(C)} × {n} = {num(D)}米",
    ]
    return ins, lines, D


_reg("wheel_distance", wheel_distance)


# 23. vA = (a/b) vB, same distance, A saves t hours -> B's time
def speed_ratio_time(rng):
    a = rng.randint(3, 7)
    b = rng.randint(2, a - 1)
    k = rng.randint(1, 6)
    tB = a * k
    tA = b * k
    t = (a - b) * k
    ins = rng.choice([
        f"甲车速度是乙车的{a}/{b}倍，行驶同一段路，甲车比乙车少用{t}小时，乙车行驶这段路用多少小时？",
        f"小明骑车的速度是步行的{a}/{b}倍，从家到学校骑车比步行少用{t}分钟，步行要多少分钟？",
        f"快车速度是慢车的{a}/{b}倍，行完一段路快车比慢车少用{t}小时，慢车行完这段路要多少小时？",
        f"哥哥的速度是弟弟的{a}/{b}倍，同样的路程哥哥比弟弟少用{t}分钟，弟弟要用多少分钟？",
    ])
    lines = [
        f"{a} - {b} = {a - b}",
        f"{t} × {a} = {t * a}小时",
        f"{t * a} ÷ {a - b} = {tB}小时",
    ]
    return ins, lines, tB


_LABELS["speed_ratio_time"] = ["份数差", "时间差乘a", "乙的时间"]
_reg("speed_ratio_time", speed_ratio_time)


# 24. walk v1, ride v2, riding saves t minutes -> distance
def walk_ride_distance(rng):
    v1 = rng.choice([60, 70, 80, 90])
    v2 = rng.choice([180, 210, 240, 270, 300])
    diff = v2 - v1
    p = v1 * v2
    gg = math.gcd(p, diff)
    m = rng.randint(1, 5)
    D = (p // gg) * m
    t = (diff // gg) * m
    ins = rng.choice([
        f"小明步行每分钟走{v1}米，骑车每分钟行{v2}米，从家到学校骑车比步行少用{t}分钟，家到学校有多少米？",
        f"小红步行每分钟{v1}米，骑车每分钟{v2}米，骑车上班比步行少用{t}分钟，她家到单位多少米？",
        f"从家到公园，步行每分钟{v1}米，骑车每分钟{v2}米，骑车比步行少用{t}分钟，家到公园多少米？",
        f"小华步行速度每分钟{v1}米，骑车速度每分钟{v2}米，走同一段路骑车比步行少用{t}分钟，这段路长多少米？",
    ])
    lines = [
        f"{v2} - {v1} = {diff}米/分",
        f"{v1} × {v2} = {p}",
        f"{t} × {p} = {t * p}",
        f"{t * p} ÷ {diff} = {D}米",
    ]
    return ins, lines, D


_LABELS["walk_ride_distance"] = ["速度差", "速度积", "时间乘速度积"]
_reg("walk_ride_distance", walk_ride_distance)


# 25. A alone a days, B alone b days, alternate one day each (A first) -> total days
_ALT = []
for a in range(4, 25):
    for b in range(4, 25):
        if a == b:
            continue
        pair = Fraction(1, a) + Fraction(1, b)
        n = int(1 / pair)
        r = 1 - pair * n
        last = r * a
        if 0 <= last <= 1:
            _ALT.append((a, b, n, last))


def work_alternating(rng):
    a, b, n, last = rng.choice(_ALT)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    pair = ra + rb
    r = 1 - pair * n
    total = 2 * n + last
    ins = rng.choice([
        f"一项工程，甲队单独做{a}天完成，乙队单独做{b}天完成，两队交替做（甲先做，一天一轮），多少天能完成？",
        f"修一条路，甲单独修{a}天完成，乙单独修{b}天完成，甲、乙轮流各做一天（甲先），几天修完？",
        f"一批零件，师傅单独做{a}天完成，徒弟单独做{b}天完成，师徒轮流做（师傅先，每人一天），共需多少天？",
        f"一个水池，甲管单独注满需{a}小时，乙管需{b}小时，两管轮流开（甲先，各开一小时），多少小时注满？",
    ])
    lines = [
        f"1 ÷ {a} = {num(ra)}",
        f"1 ÷ {b} = {num(rb)}",
        f"{num(ra)} + {num(rb)} = {num(pair)}",
        f"{num(pair)} × {n} = {num(pair * n)}",
        f"1 - {num(pair * n)} = {num(r)}",
        f"{num(r)} ÷ ({num(ra)}) = {num(last)}",
        f"2 × {n} + {num(last)} = {num(total)}天",
    ]
    return ins, lines, total


_LABELS["work_alternating"] = ["甲的效率", "乙的效率", "一轮的效率", "n轮完成的量",
                               "剩下的量", "甲还要做的天数"]
_reg("work_alternating", work_alternating)


# 26. three teams together t days, A alone a, B alone b -> C alone
_WORK_C = []
for t in range(3, 9):
    for a in range(8, 25):
        for b in range(8, 25):
            rc = Fraction(1, t) - Fraction(1, a) - Fraction(1, b)
            if rc > 0 and rc.numerator == 1 and rc.denominator <= 30:
                _WORK_C.append((t, a, b, rc.denominator))


def work_c_alone(rng):
    t, a, b, c = rng.choice(_WORK_C)
    m = rng.choice([1, 2, 3])
    t, a, b, c = t * m, a * m, b * m, c * m
    rt = Fraction(1, t)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    rc = rt - ra - rb
    ins = rng.choice([
        f"一项工程，甲、乙、丙三队合作{t}天完成，甲队单独做{a}天完成，乙队单独做{b}天完成，丙队单独做多少天完成？",
        f"修一条路，甲、乙、丙三队合修{t}天完成，甲队单独修{a}天完成，乙队单独修{b}天完成，丙队单独修要多少天？",
        f"一批零件，师徒三人合作{t}天完成，师傅单独做{a}天完成，大徒弟单独做{b}天完成，小徒弟单独做要多少天？",
        f"一个水池，甲、乙、丙三管同开{t}小时注满，单开甲管{a}小时注满，单开乙管{b}小时注满，单开丙管多少小时注满？",
    ])
    lines = [
        f"1 ÷ {t} = {num(rt)}",
        f"1 ÷ {a} = {num(ra)}",
        f"1 ÷ {b} = {num(rb)}",
        f"{num(rt)} - {num(ra)} - {num(rb)} = {num(rc)}",
        f"1 ÷ ({num(rc)}) = {c}天",
    ]
    return ins, lines, c


_LABELS["work_c_alone"] = ["合作效率", "甲的效率", "乙的效率", "丙的效率"]
_reg("work_c_alone", work_c_alone)


# 27. A alone a, B alone b, C alone c -> together time
def work_three_together(rng):
    a = rng.randint(6, 20)
    b = rng.randint(6, 20)
    c = rng.randint(6, 20)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    rc = Fraction(1, c)
    s = ra + rb + rc
    t = Fraction(1, s)
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成，乙单独做{b}天完成，丙单独做{c}天完成，三队合作多少天完成？",
        f"修一条路，甲队单独修{a}天完成，乙队单独修{b}天完成，丙队单独修{c}天完成，三队合修多少天完成？",
        f"一批零件，师傅单独做{a}天完成，甲徒弟单独做{b}天完成，乙徒弟单独做{c}天完成，三人合作多少天完成？",
        f"一个水池，甲管单独注满需{a}小时，乙管需{b}小时，丙管需{c}小时，三管同开多少小时注满？",
    ])
    lines = [
        f"1 ÷ {a} = {num(ra)}",
        f"1 ÷ {b} = {num(rb)}",
        f"1 ÷ {c} = {num(rc)}",
        f"{num(ra)} + {num(rb)} + {num(rc)} = {num(s)}",
        f"1 ÷ ({num(s)}) = {num(t)}天",
    ]
    return ins, lines, t


_LABELS["work_three_together"] = ["甲的效率", "乙的效率", "丙的效率", "三人效率和"]
_reg("work_three_together", work_three_together)


# 28. A alone a, B alone b, A works x days alone then B joins -> total time
def work_join_later(rng):
    a = rng.randint(10, 24)
    b = rng.randint(6, 20)
    x = rng.randint(2, a - 2)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    done = ra * x
    rem = 1 - done
    rate = ra + rb
    t = Fraction(rem, rate)
    total = x + t
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成，乙单独做{b}天完成，甲先做{x}天后乙加入合作，完成这项工程共需多少天？",
        f"修一条路，甲队单独修{a}天完成，乙队单独修{b}天完成，甲队先修{x}天后两队合修，修完共需多少天？",
        f"一批零件，师傅单独做{a}天完成，徒弟单独做{b}天完成，师傅先做{x}天后师徒合作，完成任务共需多少天？",
        f"一个水池，甲管单独注满需{a}小时，乙管需{b}小时，甲管先开{x}小时后两管同开，注满共需多少小时？",
    ])
    lines = [
        f"1 ÷ {a} = {num(ra)}",
        f"{num(ra)} × {x} = {num(done)}",
        f"1 - {num(done)} = {num(rem)}",
        f"1 ÷ {b} = {num(rb)}",
        f"{num(ra)} + {num(rb)} = {num(rate)}",
        f"{num(rem)} ÷ ({num(rate)}) = {num(t)}",
        f"{x} + {num(t)} = {num(total)}天",
    ]
    return ins, lines, total


_LABELS["work_join_later"] = ["甲的效率", "甲先做的量", "剩余工程量", "乙的效率", "合作效率", "合作天数"]
_reg("work_join_later", work_join_later)


# 29. two equal-weight solutions, concentrations c1%, c2% -> mixed concentration
def conc_mix_equal_weight(rng):
    s0 = rng.choice([100, 200, 300, 400])
    c1 = rng.choice([5, 10, 15, 20])
    c2 = rng.choice([10, 15, 20, 25, 30])
    salt1 = Fraction(s0 * c1, 100)
    salt2 = Fraction(s0 * c2, 100)
    salt = salt1 + salt2
    pct = Fraction(salt * 100, 2 * s0)
    ins = rng.choice([
        f"甲、乙两杯盐水同样重，各重{s0}克，甲杯含盐{c1}%，乙杯含盐{c2}%，两杯混合后含盐率是多少？",
        f"两瓶糖水各重{s0}克，甲瓶含糖{c1}%，乙瓶含糖{c2}%，混合后糖占糖水的百分之几？",
        f"甲、乙两杯果汁各{s0}克，甲杯果汁浓度{c1}%，乙杯浓度{c2}%，混合后浓度是多少？",
        f"两杯水各重{s0}克，第一杯含盐{c1}%，第二杯含盐{c2}%，倒在一起后含盐率是多少？",
    ])
    lines = [
        f"{s0} × {c1}/100 = {num(salt1)}克",
        f"{s0} × {c2}/100 = {num(salt2)}克",
        f"{num(salt1)} + {num(salt2)} = {num(salt)}克",
        f"{s0} + {s0} = {2 * s0}克",
        f"{num(salt)} ÷ {2 * s0} × 100 = {num(pct)}",
    ]
    return ins, lines, pct


_LABELS["conc_mix_equal_weight"] = ["混合后的含盐率"]
_reg("conc_mix_equal_weight", conc_mix_equal_weight)


# 30. two solutions s1 c1% and s2 c2% -> mixed concentration
def conc_mix_two_pct(rng):
    s1 = rng.choice([100, 200, 300])
    s2 = rng.choice([100, 200, 300, 400])
    c1 = rng.choice([5, 10, 15, 20])
    c2 = rng.choice([15, 20, 25, 30])
    salt1 = Fraction(s1 * c1, 100)
    salt2 = Fraction(s2 * c2, 100)
    salt = salt1 + salt2
    s = s1 + s2
    pct = Fraction(salt * 100, s)
    ins = rng.choice([
        f"甲杯有{s1}克含盐{c1}%的盐水，乙杯有{s2}克含盐{c2}%的盐水，两杯混合后含盐率是多少？",
        f"把{s1}克含糖{c1}%的糖水和{s2}克含糖{c2}%的糖水混合，混合后糖占糖水的百分之几？",
        f"甲瓶{s1}克盐水浓度{c1}%，乙瓶{s2}克盐水浓度{c2}%，混合后的浓度是多少？",
        f"第一杯{s1}克盐水含盐{c1}%，第二杯{s2}克盐水含盐{c2}%，倒在一起后含盐率是多少？",
    ])
    lines = [
        f"{s1} × {c1}/100 = {num(salt1)}克",
        f"{s2} × {c2}/100 = {num(salt2)}克",
        f"{num(salt1)} + {num(salt2)} = {num(salt)}克",
        f"{s1} + {s2} = {s}克",
        f"{num(salt)} ÷ {s} × 100 = {num(pct)}",
    ]
    return ins, lines, pct


_LABELS["conc_mix_two_pct"] = ["混合后的含盐率"]
_reg("conc_mix_two_pct", conc_mix_two_pct)


# 31. grapes dry: moisture a% -> b%, weight drops X -> original weight
def grape_dry_moisture(rng):
    a = rng.choice([80, 85, 90, 95])
    b = rng.choice([60, 65, 70, 75])
    for _ in range(50):
        if a > b:
            break
        b = rng.choice([60, 65, 70, 75])
    k = rng.randint(2, 20)
    W = (100 - b) * k
    X = (a - b) * k
    obj = rng.choice(["葡萄", "葡萄干原料", "鲜果", "水草"])
    ins = rng.choice([
        f"一批{obj}的含水率是{a}%，晒干后含水率降到{b}%，重量减少了{X}千克，这批{obj}原来重多少千克？",
        f"一批{obj}含水率{a}%，晾晒后含水率变为{b}%，重量减轻了{X}千克，原来重多少千克？",
        f"一批{obj}测得含水率{a}%，烘干后含水率{b}%，重量减少{X}千克，烘干前重多少千克？",
        f"一批{obj}含水{a}%，晒干后含水{b}%，重量少了{X}千克，这批{obj}原来有多少千克？",
    ])
    lines = [
        f"{a} - {b} = {a - b}",
        f"100 - {b} = {100 - b}",
        f"{X} × {100 - b} = {X * (100 - b)}",
        f"{X * (100 - b)} ÷ {a - b} = {W}千克",
    ]
    return ins, lines, W


_LABELS["grape_dry_moisture"] = ["含水率的差", "干果占的百分比", "减少的重量乘干果占比"]
_reg("grape_dry_moisture", grape_dry_moisture)


# 32. pour x grams from cup 1 (c1%) into cup 2 (c2%) -> cup 2's new concentration
def conc_transfer_cup(rng):
    s1 = rng.choice([200, 300, 400])
    s2 = rng.choice([100, 200, 300])
    c1 = rng.choice([10, 15, 20, 25])
    c2 = rng.choice([5, 10, 15])
    x = rng.choice([20, 40, 50, 60, 100])
    salt2 = Fraction(s2 * c2, 100)
    saltx = Fraction(x * c1, 100)
    salt = salt2 + saltx
    s2x = s2 + x
    pct = Fraction(salt * 100, s2x)
    ins = rng.choice([
        f"甲杯有{s1}克含盐{c1}%的盐水，乙杯有{s2}克含盐{c2}%的盐水，从甲杯倒{x}克到乙杯，乙杯现在的含盐率是多少？",
        f"甲瓶有{s1}克含糖{c1}%的糖水，乙瓶有{s2}克含糖{c2}%的糖水，把甲瓶{x}克倒入乙瓶，乙瓶现在含糖百分之几？",
        f"第一杯{s1}克盐水浓度{c1}%，第二杯{s2}克盐水浓度{c2}%，从第一杯倒{x}克进第二杯，第二杯现在的浓度是多少？",
        f"甲容器有{s1}克含盐{c1}%的盐水，乙容器有{s2}克含盐{c2}%的盐水，将甲中{x}克盐水倒入乙中，乙中含盐率是多少？",
    ])
    lines = [
        f"{s1} - {x} = {s1 - x}克",
        f"{s2} × {c2}/100 = {num(salt2)}克",
        f"{x} × {c1}/100 = {num(saltx)}克",
        f"{num(salt2)} + {num(saltx)} = {num(salt)}克",
        f"{s2} + {x} = {s2x}克",
        f"{num(salt)} ÷ {s2x} × 100 = {num(pct)}",
    ]
    return ins, lines, pct


_LABELS["conc_transfer_cup"] = ["甲杯剩下的盐水", "乙杯现在的含盐率"]
_reg("conc_transfer_cup", conc_transfer_cup)


# 33. parallelogram: base b, height = (a/d) of base, cost per m2 -> total
def parallelogram_cost(rng):
    b = rng.randint(4, 12)
    a = rng.randint(1, 3)
    d = rng.choice([2, 3, 4])
    for _ in range(50):
        if a < d:
            break
        a = rng.randint(1, 3)
    h = Fraction(b * a, d)
    area = b * h
    c = rng.randint(10, 50)
    cost = area * c
    obj = rng.choice(["菜地", "花圃", "草坪", "玻璃"])
    ins = rng.choice([
        f"一个平行四边形{obj}，底是{b}米，高是底的{a}/{d}，每平方米{c}元，这块{obj}值多少元？",
        f"平行四边形的底长{b}米，高是底的{a}/{d}，每平方米{c}元，面积是多少平方米？一共多少元？",
        f"一块平行四边形地，底{b}米，高是底的{a}/{d}，每平方米种{c}元的草皮，共需多少元？",
        f"平行四边形底为{b}米，高为底的{a}/{d}，每平方米{c}元，它的面积和总价各是多少？",
    ])
    lines = [
        f"{b} × {a}/{d} = {num(h)}米",
        f"{b} × {num(h)} = {num(area)}平方米",
        f"{num(area)} × {c} = {num(cost)}元",
    ]
    return ins, lines, cost


_reg("parallelogram_cost", parallelogram_cost)


# 34. circle circumference C -> area (pi = 3.14)
def circle_area_from_circumference(rng):
    r = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
    C = 2 * Fraction(314, 100) * r
    r2 = r * r
    area = Fraction(314, 100) * r2
    obj = rng.choice(["水池", "花坛", "草坪", "广场"])
    ins = rng.choice([
        f"一个圆形{obj}的周长是{num(C)}米（π取3.14），它的占地面积是多少平方米？",
        f"圆形{obj}的周长为{num(C)}米（π取3.14），面积是多少平方米？",
        f"一个圆的周长是{num(C)}米（π取3.14），这个圆的面积是多少平方米？",
        f"沿圆形{obj}走一圈是{num(C)}米（π取3.14），{obj}的面积是多少平方米？",
        f"圆形{obj}一圈长{num(C)}米（π取3.14），它的占地面积是多少平方米？",
        f"一个圆形{obj}，量得周长是{num(C)}米（π取3.14），面积是多少平方米？",
        f"圆形{obj}的周长是{num(C)}米（π取3.14），它占地多少平方米？",
        f"绕圆形{obj}一周是{num(C)}米（π取3.14），{obj}面积是多少平方米？",
    ])
    lines = [
        f"{num(C)} ÷ 2 ÷ 3.14 = {num(r)}米",
        f"{num(r)} × {num(r)} = {num(r2)}平方米",
        f"3.14 × {num(r2)} = {num(area)}平方米",
    ]
    return ins, lines, area


_reg("circle_area_from_circumference", circle_area_from_circumference)


# 35. cylinder surface area (pi = 3.14)
def cylinder_surface_area(rng):
    r = rng.choice([5, 10, 15, 20])
    h = rng.randint(4, 20)
    r2 = r * r
    base2 = 2 * Fraction(314, 100) * r2
    lat = 2 * Fraction(314, 100) * r * h
    S = base2 + lat
    obj = rng.choice(["水桶", "油桶", "罐头", "柱子"])
    ins = rng.choice([
        f"一个圆柱形{obj}，底面半径{r}分米，高{h}分米（π取3.14），做这个{obj}（有盖）至少需要多少平方分米铁皮？",
        f"圆柱底面半径{r}米，高{h}米（π取3.14），它的表面积是多少平方米？",
        f"一个圆柱形{obj}，底面半径{r}厘米，高{h}厘米（π取3.14），表面积是多少平方厘米？",
        f"圆柱形罐头底面半径{r}分米，高{h}分米（π取3.14），做一个有盖罐头至少要多少平方分米材料？",
    ])
    lines = [
        f"{r} × {r} = {r2}",
        f"2 × 3.14 × {r2} = {num(base2)}平方分米",
        f"2 × 3.14 × {r} × {h} = {num(lat)}平方分米",
        f"{num(base2)} + {num(lat)} = {num(S)}平方分米",
    ]
    return ins, lines, S


_LABELS["cylinder_surface_area"] = ["半径的平方"]
_reg("cylinder_surface_area", cylinder_surface_area)


# 36. cone volume x unit cost (pi = 3.14)
def cone_volume_cost(rng):
    r = rng.choice([3, 4, 5, 6, 10])
    h = rng.randint(3, 12)
    c = rng.randint(10, 50)
    r2 = r * r
    cyl = Fraction(314, 100) * r2 * h
    V = cyl / 3
    cost = V * c
    obj = rng.choice(["沙堆", "麦堆", "石子堆", "煤堆"])
    ins = rng.choice([
        f"一个圆锥形{obj}，底面半径{r}米，高{h}米（π取3.14），每立方米{c}元，这堆{obj}值多少元？",
        f"圆锥形{obj}底面半径{r}米，高{h}米（π取3.14），每立方米{c}元，一共值多少元？",
        f"一个圆锥底面半径{r}米，高{h}米（π取3.14），它的体积是多少立方米？每立方米{c}元，值多少元？",
        f"工地有一个圆锥形{obj}，底面半径{r}米，高{h}米（π取3.14），每立方米{c}元，这堆{obj}共值多少元？",
    ])
    lines = [
        f"{r} × {r} = {r2}",
        f"3.14 × {r2} × {h} = {num(cyl)}立方米",
        f"{num(cyl)} ÷ 3 = {num(V)}立方米",
        f"{num(V)} × {c} = {num(cost)}元",
    ]
    return ins, lines, cost


_LABELS["cone_volume_cost"] = ["半径的平方"]
_reg("cone_volume_cost", cone_volume_cost)


# 37. two squares side ratio a:b, area difference X -> big area
def square_ratio_area_diff(rng):
    a = rng.randint(3, 7)
    b = rng.randint(2, a - 1)
    k = rng.randint(2, 12)
    a2 = a * a
    b2 = b * b
    X = (a2 - b2) * k * k
    big = a2 * k * k
    obj = rng.choice(["正方形", "花坛", "地砖", "玻璃"])
    ins = rng.choice([
        f"大、小两个{obj}边长的比是{a}:{b}，面积相差{X}平方米，大{obj}的面积是多少平方米？",
        f"两个正方形边长比为{a}:{b}，面积差是{X}平方米，大正方形面积是多少平方米？",
        f"大、小两块正方形{obj}，边长比是{a}:{b}，面积相差{X}平方米，大{obj}面积是多少？",
        f"甲、乙两个正方形的边长比是{a}:{b}，乙的面积比甲少{X}平方米，甲的面积是多少平方米？",
    ])
    lines = [
        f"{a} × {a} = {a2}",
        f"{b} × {b} = {b2}",
        f"{a2} - {b2} = {a2 - b2}",
        f"{X} ÷ {a2 - b2} = {k * k}",
        f"{a2} × {k * k} = {big}平方米",
    ]
    return ins, lines, big


_LABELS["square_ratio_area_diff"] = ["大边长平方", "小边长平方", "面积份数差", "每份面积"]
_reg("square_ratio_area_diff", square_ratio_area_diff)


# 38. cuboid volume V, length l, width w -> height, then surface area
def cuboid_height_surface(rng):
    l = rng.randint(4, 12)
    w = rng.randint(3, 10)
    h = rng.randint(2, 8)
    V = l * w * h
    S = 2 * (l * w + l * h + w * h)
    obj = rng.choice(["木箱", "仓库", "水池", "礼盒"])
    ins = rng.choice([
        f"一个长方体{obj}的体积是{V}立方米，长{l}米、宽{w}米，它的高是多少米？表面积是多少平方米？",
        f"长方体体积为{V}立方米，长{l}米，宽{w}米，高是多少米？表面积是多少平方米？",
        f"一个长方体{obj}，体积{V}立方米，底面长{l}米、宽{w}米，高多少米？表面积多少平方米？",
        f"长方体的体积是{V}立方米，长和宽分别是{l}米和{w}米，求它的高和表面积。",
    ])
    lines = [
        f"{l} × {w} = {l * w}平方米",
        f"{V} ÷ {l * w} = {h}米",
        f"{l} × {h} = {l * h}平方米",
        f"{w} × {h} = {w * h}平方米",
        f"{l * w} + {l * h} + {w * h} = {l * w + l * h + w * h}平方米",
        f"2 × {l * w + l * h + w * h} = {S}平方米",
    ]
    return ins, lines, S


_reg("cuboid_height_surface", cuboid_height_surface)


# 39. cuboid edge ratio a:b:c, total edge L -> surface area
def cuboid_ratio_surface(rng):
    a = rng.randint(3, 6)
    b = rng.randint(2, 5)
    c = rng.randint(1, 4)
    k = rng.randint(2, 8)
    L = 4 * (a + b + c) * k
    s = a + b + c
    L4 = L // 4
    ab = a * b
    bc = b * c
    ac = a * c
    pairs = ab + bc + ac
    pk2 = pairs * k * k
    S = 2 * pk2
    obj = rng.choice(["木箱", "仓库", "水池", "模型"])
    ins = rng.choice([
        f"一个长方体{obj}长、宽、高的比是{a}:{b}:{c}，棱长总和是{L}米，它的表面积是多少平方米？",
        f"长方体的长宽高之比为{a}:{b}:{c}，棱长之和是{L}米，表面积是多少平方米？",
        f"一个长方体{obj}，长:宽:高={a}:{b}:{c}，全部棱长共{L}米，表面积是多少平方米？",
        f"长方体长、宽、高的比是{a}:{b}:{c}，棱长总和为{L}米，它的表面积是多少？",
    ])
    lines = [
        f"{L} ÷ 4 = {L4}",
        f"{a} + {b} + {c} = {s}",
        f"{L4} ÷ {s} = {k}",
        f"{a} × {b} = {ab}",
        f"{b} × {c} = {bc}",
        f"{a} × {c} = {ac}",
        f"{ab} + {bc} + {ac} = {pairs}",
        f"{pairs} × {k} × {k} = {pk2}",
        f"2 × {pk2} = {S}平方米",
    ]
    return ins, lines, S


_LABELS["cuboid_ratio_surface"] = ["棱长和除以4", "长宽高份数和", "每份长度", "上下底面份数积",
                                   "前后两面份数积", "左右两面份数积", "三对面积份数和", "份数面积"]
_reg("cuboid_ratio_surface", cuboid_ratio_surface)


# 40. same moment: pole a m shadows b m, building shadow X m -> building height
def shadow_height(rng):
    nice = {Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(3), Fraction(4), Fraction(5)}
    for _ in range(50):
        a = rng.choice([2, 3, 4, 5])
        X = rng.choice([10, 15, 20, 25, 30, 40, 50])
        h = rng.choice([6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 30, 36, 40, 45, 50])
        b = Fraction(a * X, h)
        if b in nice:
            break
    ins = rng.choice([
        f"同一时刻，一根{a}米长的竹竿影长是{num(b)}米，一座楼的影长是{X}米，楼高多少米？",
        f"在同一时间，小明量得{a}米高的树影长{num(b)}米，又量得一座塔的影长是{X}米，塔高多少米？",
        f"同一时刻，{a}米的杆子影长{num(b)}米，一座大楼的影子长{X}米，大楼高多少米？",
        f"阳光下，{a}米高的旗杆影长{num(b)}米，旁边一座楼的影长是{X}米，这座楼高多少米？",
    ])
    lines = [
        f"{a} × {X} = {a * X}",
        f"{a * X} ÷ ({num(b)}) = {num(h)}米",
    ]
    return ins, lines, h


_LABELS["shadow_height"] = ["竿高乘楼影"]
_reg("shadow_height", shadow_height)


# 41. open-top cuboid fish tank: 5 faces of glass x unit cost
def open_box_surface(rng):
    l = rng.randint(4, 12)
    w = rng.randint(3, 10)
    h = rng.randint(2, 8)
    c = rng.randint(10, 60)
    lw = l * w
    lh = l * h
    wh = w * h
    S = lw + 2 * lh + 2 * wh
    cost = S * c
    obj = rng.choice(["鱼缸", "水箱", "水槽", "货箱"])
    ins = rng.choice([
        f"一个无盖长方体{obj}，长{l}米、宽{w}米、高{h}米，每平方米玻璃{c}元，做这个{obj}需要多少元？",
        f"做一个无盖的长方体{obj}，长{l}米，宽{w}米，高{h}米，每平方米材料{c}元，共需多少元？",
        f"一个无盖长方体{obj}，长{l}米、宽{w}米、高{h}米，每平方米{c}元，做{obj}至少要花多少元？",
        f"长方体无盖{obj}的长、宽、高分别是{l}米、{w}米、{h}米，每平方米玻璃{c}元，制作需要多少元？",
    ])
    lines = [
        f"{l} × {w} = {lw}平方米",
        f"{l} × {h} = {lh}平方米",
        f"{w} × {h} = {wh}平方米",
        f"{lw} + 2 × {lh} + 2 × {wh} = {S}平方米",
        f"{S} × {c} = {cost}元",
    ]
    return ins, lines, cost


_reg("open_box_surface", open_box_surface)


# 42. triangle area S, base b -> height; flowers per m2 -> total
def triangle_height_flowers(rng):
    b = rng.choice([4, 6, 8, 10, 12])
    h = rng.choice([3, 4, 5, 6, 8, 10])
    S = Fraction(b * h, 2)
    c = rng.randint(3, 12)
    Sc = S * c
    obj = rng.choice(["花圃", "菜地", "草坪", "果园"])
    ins = rng.choice([
        f"一块三角形{obj}的面积是{num(S)}平方米，底是{b}米，高是多少米？如果每平方米种{c}棵花，一共种多少棵？",
        f"三角形{obj}面积为{num(S)}平方米，底边长{b}米，高是多少米？每平方米种{c}棵，共种多少棵？",
        f"一个三角形{obj}，面积是{num(S)}平方米，底是{b}米，它的高是多少米？每平方米种{c}棵花，一共多少棵？",
        f"三角形{obj}的面积是{num(S)}平方米，底{b}米，高多少米？每平方米种{c}棵，这块{obj}一共种多少棵？",
    ])
    lines = [
        f"{num(S)} × 2 = {num(2 * S)}平方米",
        f"{num(2 * S)} ÷ {b} = {num(h)}米",
        f"{num(S)} × {c} = {num(Sc)}棵",
    ]
    return ins, lines, Sc


_reg("triangle_height_flowers", triangle_height_flowers)


# 43. trapezoid area S, two bases a, b -> height
def trapezoid_find_height(rng):
    a = rng.randint(3, 10)
    b = rng.randint(a + 1, 15)
    h = rng.choice([2, 4, 6, 8, 10])
    S = Fraction((a + b) * h, 2)
    obj = rng.choice(["果园", "菜地", "花圃", "草坪"])
    ins = rng.choice([
        f"一个梯形{obj}的面积是{num(S)}平方米，上底{a}米、下底{b}米，高是多少米？",
        f"梯形{obj}面积为{num(S)}平方米，上底{a}米，下底{b}米，高是多少米？",
        f"一个梯形{obj}，面积是{num(S)}平方米，两底分别是{a}米和{b}米，高是多少米？",
        f"梯形的面积是{num(S)}平方米，上底与下底之和是{a + b}米（上底{a}米、下底{b}米），高是多少米？",
    ])
    lines = [
        f"{a} + {b} = {a + b}米",
        f"{num(S)} × 2 = {num(2 * S)}平方米",
        f"{num(2 * S)} ÷ {a + b} = {num(h)}米",
    ]
    return ins, lines, h


_reg("trapezoid_find_height", trapezoid_find_height)


# 44. semicircle diameter d -> perimeter (pi = 3.14)
def semicircle_perimeter(rng):
    d = rng.choice([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    r = Fraction(d, 2)
    arc = Fraction(314, 100) * r
    P = arc + d
    obj = rng.choice(["花坛", "草坪", "广场", "水池"])
    ins = rng.choice([
        f"一个半圆形{obj}的直径是{d}米（π取3.14），它的周长是多少米？",
        f"半圆形{obj}直径为{d}米（π取3.14），周长是多少米？",
        f"一个半圆的直径是{d}米（π取3.14），这个半圆的周长是多少米？",
        f"一块半圆形{obj}，直径{d}米（π取3.14），围一圈篱笆长多少米？",
        f"半圆形{obj}的直径是{d}米（π取3.14），它的周长是多少米？",
        f"一个半圆形{obj}，直径长{d}米（π取3.14），周长是多少米？",
        f"半圆的直径为{d}米（π取3.14），它的周长是多少米？",
        f"一块半圆形{obj}直径{d}米（π取3.14），沿边缘围一圈要多少米？",
    ])
    lines = [
        f"{d} ÷ 2 = {num(r)}米",
        f"3.14 × {num(r)} = {num(arc)}米",
        f"{num(arc)} + {d} = {num(P)}米",
    ]
    return ins, lines, P


_reg("semicircle_perimeter", semicircle_perimeter)


# 45. semicircle area x unit cost (pi = 3.14)
def semicircle_area_cost(rng):
    d = rng.choice([4, 6, 8, 10, 12, 14, 16, 18, 20, 24])
    r = Fraction(d, 2)
    r2 = r * r
    A = Fraction(314, 100) * r2 / 2
    c = rng.randint(10, 50)
    cost = A * c
    obj = rng.choice(["花坛", "草坪", "菜地", "广场"])
    ins = rng.choice([
        f"一个半圆形{obj}的直径是{d}米（π取3.14），每平方米{c}元，种满{obj}需要多少元？",
        f"半圆形{obj}直径为{d}米（π取3.14），每平方米{c}元，铺满要多少元？",
        f"一块半圆形{obj}，直径{d}米（π取3.14），每平方米投入{c}元，共需多少元？",
        f"半圆的直径是{d}米（π取3.14），每平方米{c}元，这个半圆的面积是多少平方米？共需多少元？",
    ])
    lines = [
        f"{d} ÷ 2 = {num(r)}米",
        f"{num(r)} × {num(r)} = {num(r2)}平方米",
        f"3.14 × {num(r2)} ÷ 2 = {num(A)}平方米",
        f"{num(A)} × {c} = {num(cost)}元",
    ]
    return ins, lines, cost


_reg("semicircle_area_cost", semicircle_area_cost)


# 46. cuboid volume x unit storage cost
def cuboid_volume_cost(rng):
    l = rng.randint(3, 10)
    w = rng.randint(2, 8)
    h = rng.randint(2, 6)
    c = rng.randint(10, 80)
    V = l * w * h
    cost = V * c
    obj = rng.choice(["仓库", "货箱", "水池", "沙坑"])
    ins = rng.choice([
        f"一个长方体{obj}，长{l}米、宽{w}米、高{h}米，每立方米货物{c}元，装满货物值多少元？",
        f"长方体{obj}长{l}米，宽{w}米，高{h}米，每立方米{c}元，装满需要多少元？",
        f"一个长方体{obj}，长{l}米、宽{w}米、深{h}米，每立方米{c}元，存满货物共值多少元？",
        f"长方体{obj}的长、宽、高分别是{l}米、{w}米、{h}米，每立方米{c}元，装满要多少元？",
    ])
    lines = [
        f"{l} × {w} = {l * w}平方米",
        f"{l * w} × {h} = {V}立方米",
        f"{V} × {c} = {cost}元",
    ]
    return ins, lines, cost


_reg("cuboid_volume_cost", cuboid_volume_cost)


# 47. cube iron cast into cuboid with base area S -> length; unit price -> value
def cast_reshape(rng):
    a = rng.randint(3, 8)
    S = rng.choice([4, 6, 8, 9, 12, 16, 18, 24, 25, 36])
    for _ in range(50):
        if (a ** 3) % S == 0:
            break
        S = rng.choice([4, 6, 8, 9, 12, 16, 18, 24, 25, 36])
    V = a ** 3
    L = Fraction(V, S)
    c = rng.randint(10, 50)
    cost = V * c
    obj = rng.choice(["铁块", "铜块", "铝块", "钢锭"])
    ins = rng.choice([
        f"把一个棱长{a}米的正方体{obj}铸造成一个底面积是{S}平方米的长方体，长方体的长是多少米？如果每立方米{c}元，这块{obj}值多少元？",
        f"棱长{a}米的正方体{obj}熔铸成底面积{S}平方米的长方体，长是多少米？每立方米{c}元，原值多少元？",
        f"一个正方体{obj}棱长{a}米，铸成底面积{S}平方米的长方体钢材，长多少米？每立方米{c}元，共值多少元？",
        f"把棱长{a}米的正方体{obj}锻造成底面积为{S}平方米的长方体，长是多少米？每立方米{c}元，这块{obj}值多少元？",
    ])
    lines = [
        f"{a} × {a} = {a * a}平方米",
        f"{a * a} × {a} = {V}立方米",
        f"{V} × {c} = {cost}元",
        f"{V} ÷ {S} = {num(L)}米",
    ]
    return ins, lines, L


_reg("cast_reshape", cast_reshape)


# 48. cylinder full of water poured into cuboid pool -> depth (pi = 3.14)
def cylinder_pour_depth(rng):
    r = rng.choice([5, 10, 15, 20])
    h = rng.randint(4, 16)
    a = rng.choice([5, 10, 15, 20])
    b = rng.choice([4, 5, 8, 10])
    r2 = r * r
    V = Fraction(314, 100) * r2 * h
    depth = V / (a * b)
    ins = rng.choice([
        f"一个圆柱形水桶，底面半径{r}米，高{h}米（π取3.14），装满水后全部倒入长{a}米、宽{b}米的长方体水池，水深多少米？",
        f"圆柱形容器底面半径{r}米，高{h}米（π取3.14），装满的水倒入长{a}米、宽{b}米的长方体水槽，水面高多少米？",
        f"一个圆柱形水桶底面半径{r}米，桶高{h}米（π取3.14），把满桶水倒进长{a}米、宽{b}米的长方体水池，水深多少米？",
        f"圆柱形水缸底面半径{r}米，高{h}米（π取3.14），装满水后倒入长{a}米、宽{b}米的长方体池子，水深多少米？",
    ])
    lines = [
        f"{r} × {r} = {r2}",
        f"3.14 × {r2} × {h} = {num(V)}立方米",
        f"{a} × {b} = {a * b}平方米",
        f"{num(V)} ÷ {a * b} = {num(depth)}米",
    ]
    return ins, lines, depth


_LABELS["cylinder_pour_depth"] = ["半径的平方"]
_reg("cylinder_pour_depth", cylinder_pour_depth)


# 49. parallelogram area, cost, and the other base's height
def parallelogram_other_height(rng):
    a = rng.randint(4, 12)
    h1 = rng.randint(2, 8)
    b = rng.randint(3, 10)
    c = rng.randint(10, 50)
    S = a * h1
    cost = S * c
    h2 = Fraction(S, b)
    obj = rng.choice(["菜地", "花圃", "草坪", "玻璃"])
    ins = rng.choice([
        f"一个平行四边形{obj}，底是{a}米，这条底上的高是{h1}米，面积是多少平方米？如果每平方米{c}元，整块{obj}值多少元？另一条底是{b}米，这条底上的高是多少米？",
        f"平行四边形底长{a}米，高{h1}米，面积是多少？每平方米{c}元，共值多少元？另一条底{b}米，对应的高是多少米？",
        f"一块平行四边形{obj}，底{a}米，高{h1}米，面积是多少平方米？每平方米{c}元，值多少元？底{b}米上的高是多少米？",
        f"平行四边形的底是{a}米，高是{h1}米，面积和总价各是多少（每平方米{c}元）？另一条底{b}米，高是多少米？",
    ])
    lines = [
        f"{a} × {h1} = {S}平方米",
        f"{S} × {c} = {cost}元",
        f"{S} ÷ {b} = {num(h2)}米",
    ]
    return ins, lines, h2


_reg("parallelogram_other_height", parallelogram_other_height)


# 50. rectangle perimeter C, length a -> area
def rectangle_perimeter_area(rng):
    a = rng.randint(4, 12)
    w = rng.randint(2, a - 1)
    C = 2 * (a + w)
    area = a * w
    obj = rng.choice(["菜地", "花圃", "操场", "地毯"])
    ins = rng.choice([
        f"一块长方形{obj}，周长是{C}米，长是{a}米，面积是多少平方米？",
        f"长方形{obj}的周长为{C}米，长{a}米，面积是多少平方米？",
        f"一个长方形{obj}，周长{C}米，长是{a}米，它的面积是多少平方米？",
        f"长方形的周长是{C}米，长{a}米，宽和面积各是多少？",
    ])
    lines = [
        f"{C} ÷ 2 = {C // 2}米",
        f"{C // 2} - {a} = {w}米",
        f"{a} × {w} = {area}平方米",
    ]
    return ins, lines, area


_reg("rectangle_perimeter_area", rectangle_perimeter_area)


# 51. isoceles triangle: perimeter P, waist a, height h -> area
def isoceles_area(rng):
    a = rng.randint(5, 15)
    h = rng.randint(3, 12)
    base = rng.choice([4, 6, 8, 10, 12, 16])
    P = 2 * a + base
    area = Fraction(base * h, 2)
    obj = rng.choice(["花圃", "菜地", "草坪", "广场"])
    ins = rng.choice([
        f"一个等腰三角形{obj}的周长是{P}米，腰长{a}米，底边上的高是{h}米，它的面积是多少平方米？",
        f"等腰三角形周长{P}米，腰{a}米，底边上的高{h}米，面积是多少平方米？",
        f"一个等腰三角形，周长是{P}米，腰长{a}米，高{h}米，面积是多少平方米？",
        f"等腰三角形{obj}周长{P}米，腰{a}米，底边上的高{h}米，面积是多少？",
    ])
    lines = [
        f"{a} × 2 = {2 * a}米",
        f"{P} - {2 * a} = {base}米",
        f"{base} × {h} = {base * h}平方米",
        f"{base * h} ÷ 2 = {num(area)}平方米",
    ]
    return ins, lines, area


_reg("isoceles_area", isoceles_area)


# 52. triangle interior angle ratio a:b:c -> largest angle
def triangle_angle_ratio(rng):
    a = rng.randint(1, 4)
    b = rng.randint(2, 5)
    c = rng.randint(2, 6)
    s = a + b + c
    per = Fraction(180, s)
    hi = max(a, b, c)
    angle = per * hi
    ins = rng.choice([
        f"一个三角形3个内角的比是{a}:{b}:{c}，最大的角是多少度？",
        f"三角形三个内角度数的比是{a}:{b}:{c}，最大角是多少度？",
        f"一个三角形3个内角度数比为{a}:{b}:{c}，最大的内角是多少度？",
        f"三角形的三个角的比是{a}:{b}:{c}，最大角是多少度？",
    ])
    lines = [
        f"{a} + {b} + {c} = {s}",
        f"180 ÷ {s} = {num(per)}度",
        f"{num(per)} × {hi} = {num(angle)}度",
    ]
    return ins, lines, angle


_LABELS["triangle_angle_ratio"] = ["3个角的份数和"]
_reg("triangle_angle_ratio", triangle_angle_ratio)


# 53. cylinder lateral area x paint cost (pi = 3.14)
def cylinder_lateral_cost(rng):
    r = rng.choice([5, 10, 15, 20])
    h = rng.randint(4, 20)
    c = rng.randint(10, 50)
    base = 2 * Fraction(314, 100) * r
    L = base * h
    cost = L * c
    obj = rng.choice(["烟囱", "柱子", "水管", "桥墩"])
    ins = rng.choice([
        f"一个圆柱形{obj}，底面半径{r}米，高{h}米（π取3.14），每平方米油漆{c}元，油漆侧面需要多少元？",
        f"圆柱形{obj}底面半径{r}米，高{h}米（π取3.14），侧面每平方米{c}元，刷漆共需多少元？",
        f"一个圆柱{obj}，半径{r}米，高{h}米（π取3.14），每平方米{c}元，油漆它的侧面要多少元？",
        f"圆柱形{obj}的底面半径{r}米，高{h}米（π取3.14），侧面油漆每平方米{c}元，一共需要多少元？",
    ])
    lines = [
        f"2 × 3.14 × {r} = {num(base)}米",
        f"{num(base)} × {h} = {num(L)}平方米",
        f"{num(L)} × {c} = {num(cost)}元",
    ]
    return ins, lines, cost


_reg("cylinder_lateral_cost", cylinder_lateral_cost)


# 54. father+son age sum S, father = (a/b)*son + c -> son's age
def age_fraction_excess(rng):
    a = rng.randint(2, 5)
    b = rng.randint(2, 6)
    for _ in range(50):
        if a != b:
            break
        b = rng.randint(2, 6)
    c = rng.randint(1, 5)
    k = rng.randint(3, 12)
    son = b * k
    S = (a + b) * k + c
    rel = rng.choice([("父亲", "儿子"), ("爸爸", "小明"), ("母亲", "女儿"), ("妈妈", "小红")])
    fa, ch = rel
    ins = rng.choice([
        f"{fa}和{ch}的年龄和是{S}岁，{fa}的年龄比{ch}的{a}/{b}倍多{c}岁，{ch}今年多少岁？",
        f"今年{fa}与{ch}年龄和为{S}岁，{fa}年龄是{ch}的{a}/{b}倍多{c}岁，{ch}今年几岁？",
        f"{fa}、{ch}今年共{S}岁，{fa}的年龄比{ch}的{a}/{b}倍还多{c}岁，{ch}今年多少岁？",
        f"今年{fa}和{ch}的年龄和是{S}岁，{fa}比{ch}年龄的{a}/{b}倍多{c}岁，{ch}今年多少岁？",
    ])
    lines = [
        f"{S} - {c} = {S - c}岁",
        f"{a} + {b} = {a + b}",
        f"{S - c} × {b} = {(S - c) * b}",
        f"{(S - c) * b} ÷ {a + b} = {son}岁",
    ]
    return ins, lines, son


_LABELS["age_fraction_excess"] = ["份数和", "年龄和减c后乘b"]
_reg("age_fraction_excess", age_fraction_excess)


# 55. two-digit number: digit sum S, swapped number is X larger -> original
def digit_swap_diff(rng):
    d = rng.randint(1, 5)
    S = rng.randint(5, 17)
    for _ in range(50):
        if S > d and (S - d) % 2 == 0:
            break
        S = rng.randint(5, 17)
    a = (S - d) // 2
    b = (S + d) // 2
    X = 9 * d
    number = 10 * a + b
    ins = rng.choice([
        f"一个两位数，十位数字与个位数字之和是{S}，交换两个数字的位置后比原数大{X}，原数是多少？",
        f"一个两位数，个位数字与十位数字的和是{S}，数字交换位置后得到的新数比原数大{X}，求原数。",
        f"小明想了一个两位数，两个数字之和是{S}，把十位和个位交换后比原数大{X}，这个两位数是多少？",
        f"一个两位数，数字和为{S}，倒转数比原数大{X}，这个两位数是几？",
        f"一个两位数，十位与个位数字相加得{S}，交换数字位置后比原数大{X}，原数是多少？",
        f"有一个两位数，数字之和是{S}，把两个数字调换后比原数大{X}，这个两位数是多少？",
    ])
    lines = [
        f"{X} ÷ 9 = {d}",
        f"{S} - {d} = {S - d}",
        f"{S - d} ÷ 2 = {a}",
        f"{S} - {a} = {b}",
        f"{a} × 10 + {b} = {number}",
    ]
    return ins, lines, number


_LABELS["digit_swap_diff"] = ["两个数字的差", "十位数字的2倍", "十位数字", "个位数字", "这个两位数"]
_reg("digit_swap_diff", digit_swap_diff)


# 56. three consecutive odd/even/natural numbers sum S -> largest
def three_consecutive(rng):
    kind = rng.choice(["odd", "even", "natural"])
    mid = rng.randint(5, 30)
    if kind == "odd":
        if mid % 2 == 0:
            mid += 1
        step = 2
        word = "奇数"
    elif kind == "even":
        if mid % 2 == 1:
            mid += 1
        step = 2
        word = "偶数"
    else:
        step = 1
        word = "自然数"
    S = 3 * mid
    maxv = mid + step
    minv = mid - step
    ins = rng.choice([
        f"3个连续{word}的和是{S}，最大的是多少？",
        f"3个连续{word}的和为{S}，最大的一个是多少？",
        f"已知3个连续{word}的和是{S}，最大的数是几？",
        f"3个连续{word}相加得{S}，最大的是多少？",
    ])
    lines = [
        f"{S} ÷ 3 = {mid}",
        f"{mid} - {step} = {minv}",
        f"{mid} + {step} = {maxv}",
    ]
    return ins, lines, maxv


_LABELS["three_consecutive"] = ["中间的数", "最小的数", "最大的数"]
_reg("three_consecutive", three_consecutive)


# 57. (a/b) of a number is X less than (c/d) of it -> the number
def fraction_diff_find_number(rng):
    b = rng.choice([3, 4, 5, 6])
    a = rng.randint(1, b - 1)
    d = rng.choice([2, 3, 4, 5])
    c = rng.randint(1, 8)
    for _ in range(50):
        if c * b > a * d:
            break
        c = rng.randint(1, 8)
    k = rng.randint(2, 20)
    N = b * d * k
    X = (c * b - a * d) * k
    ins = rng.choice([
        f"一个数的{a}/{b}比它的{c}/{d}少{X}，这个数是多少？",
        f"某数的{a}/{b}比它的{c}/{d}少{X}，求这个数。",
        f"一个数的{a}/{b}比这个数的{c}/{d}少{X}，这个数是几？",
        f"甲数的{a}/{b}比甲数的{c}/{d}少{X}，甲数是多少？",
    ])
    lines = [
        f"{c} × {b} = {c * b}",
        f"{a} × {d} = {a * d}",
        f"{c * b} - {a * d} = {c * b - a * d}",
        f"{X} × {b} × {d} = {X * b * d}",
        f"{X * b * d} ÷ {c * b - a * d} = {N}",
    ]
    return ins, lines, N


_LABELS["fraction_diff_find_number"] = ["c/d的分子积", "a/b的分子积", "份数差", "差乘分母积", "这个数"]
_reg("fraction_diff_find_number", fraction_diff_find_number)


# 58. A = p% of B, B = q% of C, A = X -> C
def pct_chain_find(rng):
    pairs = [(p, q) for p in [10, 20, 25, 40, 50, 80]
             for q in [10, 20, 25, 40, 50, 80] if (p * q) % 100 == 0]
    p, q = rng.choice(pairs)
    k = rng.randint(1, 3)
    C = 100 * k
    X = p * q * k // 100
    ins = rng.choice([
        f"甲数是乙数的{p}%，乙数是丙数的{q}%，甲数是{X}，丙数是多少？",
        f"甲是乙的{p}%，乙是丙的{q}%，甲是{X}，丙是多少？",
        f"甲数等于乙数的{p}%，乙数等于丙数的{q}%，已知甲数是{X}，丙数是多少？",
        f"甲是乙的{p}%，乙是丙的{q}%，如果甲是{X}，丙是多少？",
    ])
    lines = [
        f"{p} × {q} = {p * q}",
        f"{X} × 10000 = {X * 10000}",
        f"{X * 10000} ÷ {p * q} = {C}",
    ]
    return ins, lines, C


_LABELS["pct_chain_find"] = ["两个百分比的积", "甲数扩大10000倍", "丙数"]
_reg("pct_chain_find", pct_chain_find)


# 59. A + B = S, A = p% of B -> B
def pct_of_sum_find(rng):
    p = rng.choice([10, 20, 25, 40, 50, 60, 75])
    k = rng.randint(2, 30)
    S = (100 + p) * k
    B = 100 * k
    ins = rng.choice([
        f"甲、乙两数之和是{S}，甲数是乙数的{p}%，乙数是多少？",
        f"甲、乙两数共{S}，甲数是乙数的{p}%，乙数是多少？",
        f"两个数的和是{S}，其中一个数是另一个数的{p}%，较小的数是多少？",
        f"甲、乙两数的和为{S}，甲数等于乙数的{p}%，乙数是多少？",
    ])
    lines = [
        f"100 + {p} = {100 + p}",
        f"{S} × 100 = {S * 100}",
        f"{S * 100} ÷ {100 + p} = {B}",
    ]
    return ins, lines, B


_LABELS["pct_of_sum_find"] = ["份数和", "总和扩大100倍", "乙数"]
_reg("pct_of_sum_find", pct_of_sum_find)


# 60. father = (a/b) * son, father - son = X -> son's age
def age_fraction_diff(rng):
    a = rng.randint(3, 7)
    b = rng.randint(2, a - 1)
    k = rng.randint(2, 12)
    X = (a - b) * k
    son = b * k
    rel = rng.choice([("父亲", "儿子"), ("爸爸", "小明"), ("母亲", "女儿"), ("爷爷", "孙子")])
    fa, ch = rel
    ins = rng.choice([
        f"{fa}的年龄是{ch}的{a}/{b}倍，{fa}比{ch}大{X}岁，{ch}今年多少岁？",
        f"今年{fa}年龄是{ch}的{a}/{b}倍，{fa}比{ch}大{X}岁，{ch}今年几岁？",
        f"{fa}的年龄是{ch}的{a}/{b}倍，两人相差{X}岁，{ch}今年多少岁？",
        f"今年{fa}比{ch}大{X}岁，{fa}年龄正好是{ch}的{a}/{b}倍，{ch}今年多少岁？",
    ])
    lines = [
        f"{a} - {b} = {a - b}",
        f"{X} × {b} = {X * b}岁",
        f"{X * b} ÷ {a - b} = {son}岁",
    ]
    return ins, lines, son


_LABELS["age_fraction_diff"] = ["份数差"]
_reg("age_fraction_diff", age_fraction_diff)


# 61. two-digit number: tens - units = d, number + reverse = S -> number
def digit_reverse_sum(rng):
    d = rng.randint(1, 6)
    m = rng.randint(5, 19)
    for _ in range(50):
        if m > d and (m + d) % 2 == 0:
            break
        m = rng.randint(5, 19)
    a = (m + d) // 2
    b = (m - d) // 2
    S = 11 * m
    number = 10 * a + b
    ins = rng.choice([
        f"一个两位数，十位数字比个位数字大{d}，这个两位数与它的倒转数之和是{S}，求这个两位数。",
        f"一个两位数，个位数字比十位数字小{d}，原数与倒转数的和是{S}，这个两位数是多少？",
        f"小明想了一个两位数，十位数字比个位数字大{d}，这个数与它的倒转数相加得{S}，这个数是多少？",
        f"一个两位数，十位数字与个位数字之差是{d}，它与倒转数之和是{S}，求这个两位数。",
    ])
    lines = [
        f"{S} ÷ 11 = {m}",
        f"{m} + {d} = {m + d}",
        f"{m + d} ÷ 2 = {a}",
        f"{m} - {a} = {b}",
        f"{a} × 10 + {b} = {number}",
    ]
    return ins, lines, number


_LABELS["digit_reverse_sum"] = ["数字和", "十位数字的2倍", "十位数字", "个位数字", "这个两位数"]
_reg("digit_reverse_sum", digit_reverse_sum)


# 62. decimal point shift one place, difference X -> original number
def decimal_shift(rng):
    k = rng.randint(2, 30)
    X = 9 * k
    left = rng.choice([True, False])
    if left:
        N = 10 * k
        ins = rng.choice([
            f"一个数的小数点向左移动一位后比原数小{X}，原数是多少？",
            f"把一个数的小数点左移一位，得到的数比原数小{X}，原数是多少？",
            f"一个数缩小到原来的1/10后比原数小{X}，这个数是多少？",
            f"某数的小数点向左移动一位，比原数少{X}，求原数。",
        ])
        lines = [
            f"10 - 1 = 9",
            f"{X} × 10 = {10 * X}",
            f"{10 * X} ÷ 9 = {N}",
        ]
    else:
        N = k
        ins = rng.choice([
            f"一个数的小数点向右移动一位后比原数大{X}，移动后的新数是多少？",
            f"把一个数的小数点右移一位，得到的数比原数大{X}，新数是多少？",
            f"一个数扩大到原来的10倍后比原数大{X}，扩大后的数是多少？",
            f"某数的小数点向右移动一位，比原数多{X}，移动后的数是多少？",
        ])
        lines = [
            f"10 - 1 = 9",
            f"{X} ÷ 9 = {N}",
            f"{N} × 10 = {10 * N}",
        ]
    return ins, lines, 10 * N if not left else N


_LABELS["decimal_shift"] = ["倍数差", "原数", "新数"]
_reg("decimal_shift", decimal_shift)


# 63. (a/b) of A == (c/d) of B, B = X -> A
def fraction_chain_find(rng):
    a = rng.randint(2, 5)
    b = rng.randint(3, 7)
    c = rng.randint(2, 6)
    d = rng.randint(2, 6)
    k = rng.randint(2, 20)
    X = d * a * k
    A = c * b * k
    ins = rng.choice([
        f"甲数的{a}/{b}等于乙数的{c}/{d}，乙数是{X}，甲数是多少？",
        f"甲数的{a}/{b}与乙数的{c}/{d}相等，乙数是{X}，甲数是多少？",
        f"甲的{a}/{b}等于乙的{c}/{d}，已知乙是{X}，甲是多少？",
        f"甲数乘{a}/{b}等于乙数乘{c}/{d}，乙数是{X}，甲数是多少？",
    ])
    lines = [
        f"{X} × {c} = {X * c}",
        f"{X * c} × {b} = {X * c * b}",
        f"{d} × {a} = {d * a}",
        f"{X * c * b} ÷ {d * a} = {A}",
    ]
    return ins, lines, A


_LABELS["fraction_chain_find"] = ["乙数乘c", "再乘b", "分母的积", "甲数"]
_reg("fraction_chain_find", fraction_chain_find)


# 64. A is p% more/less than B -> B is what percent less/more than A
def pct_relative(rng):
    p = rng.choice([5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80])
    up = rng.choice([True, False])
    if up:
        q = Fraction(100 * p, 100 + p)
        base = 100 + p
        ins = rng.choice([
            f"甲数比乙数多{p}%，乙数比甲数少百分之几？",
            f"甲比乙多{p}%，乙比甲少百分之几？",
            f"一种商品先涨价{p}%，再降价百分之几才能回到原价？",
            f"甲数比乙数大{p}%，乙数比甲数小百分之几？",
            f"哥哥的钱比弟弟多{p}%，弟弟的钱比哥哥少百分之几？",
            f"今年产量比去年多{p}%，去年比今年少百分之几？",
        ])
    else:
        q = Fraction(100 * p, 100 - p)
        base = 100 - p
        ins = rng.choice([
            f"乙数比甲数少{p}%，甲数比乙数多百分之几？",
            f"乙比甲少{p}%，甲比乙多百分之几？",
            f"一种商品先降价{p}%，再涨价百分之几才能回到原价？",
            f"甲数比乙数小{p}%，乙数比甲数大百分之几？",
            f"弟弟的钱比哥哥少{p}%，哥哥的钱比弟弟多百分之几？",
            f"去年产量比今年少{p}%，今年比去年多百分之几？",
        ])
    lines = [
        f"100 + {p} = {base}" if up else f"100 - {p} = {base}",
        f"{p} × 100 = {100 * p}",
        f"{100 * p} ÷ {base} = {num(q)}",
    ]
    return ins, lines, q


_LABELS["pct_relative"] = ["标准量的份数", "被除数扩大100倍", "相差的百分比"]
_reg("pct_relative", pct_relative)


if __name__ == "__main__":
    rng = random.Random(3)
    from run_math_short import verify
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines} {ans}"
            ok += 1
    print(f"L3 ext4 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
