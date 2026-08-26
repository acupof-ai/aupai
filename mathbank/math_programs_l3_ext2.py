#!/usr/bin/env python3
"""L3 ext2: 66 distinct 5-7 step families (percent/fraction/ratio/decimal).

Every program: fn(rng) -> (instruction, lines, ans); >=3 equation lines;
>=3 phrasings; all non-integer arithmetic via Fraction, rendered with num().
"""
import math
import random
from fractions import Fraction
from mathcommon import (ANIMALS, FOOD, FRUITS, GOODS, NAMES, PLACE, STATIONERY,
                         UNIT_FRUIT, UNIT_N, UNIT_ZHI, num)

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L3", name, fn))


# 1. increase by p%, then use 1/d of the new stock -> remaining
def pct_up_frac_left(rng):
    base = rng.randint(20, 80) * 10
    pr = rng.randint(10, 40)
    den = rng.choice([2, 3, 4])
    inc = Fraction(base * pr, 100)
    after = base + inc
    used = Fraction(after, den)
    left = after - used
    unit = rng.choice(["吨", "本", "千克", "升"])
    ins = rng.choice([
        f"仓库原有{base}{unit}货物，运进{pr}%后又用掉现有货物的1/{den}，还剩多少{unit}？",
        f"书店原有{base}{unit}书，补货增加了{pr}%，随后卖出1/{den}，还剩多少{unit}？",
        f"食堂原有{base}{unit}大米，买进{pr}%后又吃掉1/{den}，还剩多少{unit}？",
    ])
    lines = [
        f"{base} × {pr}/100 = {num(inc)}{unit}",
        f"{base} + {num(inc)} = {num(after)}{unit}",
        f"{num(after)} × 1/{den} = {num(used)}{unit}",
        f"{num(after)} - {num(used)} = {num(left)}{unit}",
    ]
    return ins, lines, left


_reg("pct_up_frac_left", pct_up_frac_left)


# 2. use 1/d first, then p% of the remainder -> second amount
def frac_first_pct_second(rng):
    total = rng.randint(12, 60) * 10
    den = rng.choice([2, 3, 4])
    pr = rng.randint(10, 60)
    first = Fraction(total, den)
    rem = total - first
    second = Fraction(rem * pr, 100)
    unit = rng.choice(["吨", "本", "千克", "升"])
    ins = rng.choice([
        f"仓库有{total}{unit}货物，第一天运走1/{den}，第二天运走剩余部分的{pr}%，第二天运走多少{unit}？",
        f"一根绳子长{total}{unit}，第一次剪去1/{den}，第二次剪去剩下的{pr}%，第二次剪去多少{unit}？",
        f"书店运来{total}{unit}书，上午卖出1/{den}，下午卖出剩下的{pr}%，下午卖出多少{unit}？",
    ])
    lines = [
        f"{total} × 1/{den} = {num(first)}{unit}",
        f"{total} - {num(first)} = {num(rem)}{unit}",
        f"{num(rem)} × {pr}/100 = {num(second)}{unit}",
    ]
    return ins, lines, second


_reg("frac_first_pct_second", frac_first_pct_second)


# 3. two successive percent decreases -> final
def pct_down_twice(rng):
    base = rng.randint(30, 90) * 10
    p = rng.randint(10, 30)
    q = rng.randint(10, 30)
    cut1 = Fraction(base * p, 100)
    a1 = base - cut1
    cut2 = Fraction(a1 * q, 100)
    a2 = a1 - cut2
    unit = rng.choice(["元", "吨", "件", "千克"])
    ins = rng.choice([
        f"一件商品原价{base}{unit}，先降价{p}%，再降价{q}%，现价多少{unit}？",
        f"仓库有{base}{unit}货物，第一次运走{p}%，第二次又运走余下的{q}%，还剩多少{unit}？",
        f"一桶油重{base}{unit}，第一次用去{p}%，第二次用去剩下的{q}%，还剩多少{unit}？",
    ])
    lines = [
        f"{base} × {p}/100 = {num(cut1)}{unit}",
        f"{base} - {num(cut1)} = {num(a1)}{unit}",
        f"{num(a1)} × {q}/100 = {num(cut2)}{unit}",
        f"{num(a1)} - {num(cut2)} = {num(a2)}{unit}",
    ]
    return ins, lines, a2


_reg("pct_down_twice", pct_down_twice)


# 4. up p% then down q% -> final value
def pct_up_down_final(rng):
    base = rng.randint(30, 90) * 10
    p = rng.randint(10, 30)
    q = rng.randint(10, 40)
    up = Fraction(base * p, 100)
    after = base + up
    down = Fraction(after * q, 100)
    final = after - down
    unit = rng.choice(["元", "吨", "件"])
    ins = rng.choice([
        f"一件商品原价{base}{unit}，先涨价{p}%，再降价{q}%，现价多少{unit}？",
        f"仓库有{base}{unit}货物，先增加{p}%，再减少{q}%，现在有多少{unit}？",
        f"某股票原价{base}{unit}，先上涨{p}%，再下跌{q}%，现在价值多少{unit}？",
    ])
    lines = [
        f"{base} × {p}/100 = {num(up)}{unit}",
        f"{base} + {num(up)} = {num(after)}{unit}",
        f"{num(after)} × {q}/100 = {num(down)}{unit}",
        f"{num(after)} - {num(down)} = {num(final)}{unit}",
    ]
    return ins, lines, final


_reg("pct_up_down_final", pct_up_down_final)


# 5. meet problem -> distance one side traveled
def meet_distance_each(rng):
    d = rng.randint(5, 20) * 100
    v1 = rng.randint(40, 90)
    v2 = rng.randint(40, 90)
    for _ in range(50):
        if v1 != v2:
            break
        v2 = rng.randint(40, 90)
    s = v1 + v2
    t = Fraction(d, s)
    d1 = v1 * t
    ins = rng.choice([
        f"A、B两地相距{d}千米，甲车每小时行{v1}千米，乙车每小时行{v2}千米，两车同时相向开出，相遇时甲车行了多少千米？",
        f"甲、乙两人从相距{d}千米的两地同时相向而行，甲每小时走{v1}千米，乙每小时走{v2}千米，相遇时甲走了多少千米？",
        f"两艘轮船从相距{d}千米的两港同时相向开出，快船每小时行{v1}千米，慢船每小时行{v2}千米，相遇时快船行了多少千米？",
    ])
    lines = [
        f"速度和 = {v1} + {v2} = {s}千米/时",
        f"相遇时间 = {d} ÷ {s} = {num(t)}小时",
        f"{v1} × {num(t)} = {num(d1)}千米",
    ]
    return ins, lines, d1


_reg("meet_distance_each", meet_distance_each)


# 6. meet point -> distance past the midpoint
def meet_midpoint_gap(rng):
    d = rng.randint(5, 20) * 100
    v1 = rng.randint(50, 90)
    v2 = rng.randint(30, v1 - 10)
    s = v1 + v2
    t = Fraction(d, s)
    d1 = v1 * t
    mid = Fraction(d, 2)
    gap = d1 - mid
    ins = rng.choice([
        f"甲、乙两车从相距{d}千米的两地同时相向而行，甲车每小时行{v1}千米，乙车每小时行{v2}千米，相遇点距中点多少千米？",
        f"甲、乙两人从相距{d}米的两地同时相向出发，甲每分钟走{v1}米，乙每分钟走{v2}米，相遇时甲走过中点多少米？",
        f"两艘船从相距{d}千米的两港同时相向开出，快船每小时行{v1}千米，慢船每小时行{v2}千米，相遇地点离中点多少千米？",
    ])
    unit = "千米" if d >= 500 else "米"
    lines = [
        f"速度和 = {v1} + {v2} = {s}",
        f"相遇时间 = {d} ÷ {s} = {num(t)}",
        f"{v1} × {num(t)} = {num(d1)}{unit}",
        f"{d} ÷ 2 = {num(mid)}{unit}",
        f"{num(d1)} - {num(mid)} = {num(gap)}{unit}",
    ]
    return ins, lines, gap


_reg("meet_midpoint_gap", meet_midpoint_gap)


# 7. catch problem -> distance the chaser traveled
def catch_distance(rng):
    gap = rng.randint(6, 20) * 10
    vf = rng.randint(70, 120)
    vs = rng.randint(40, vf - 15)
    dv = vf - vs
    t = Fraction(gap, dv)
    df = vf * t
    ins = rng.choice([
        f"甲在乙前面{gap}米，甲每秒跑{vs}米，乙每秒跑{vf}米，乙追上甲时乙跑了多少米？",
        f"快车在慢车后面{gap}千米，慢车每小时行{vs}千米，快车每小时行{vf}千米，快车追上慢车时行了多少千米？",
        f"哥哥在弟弟前面{gap}米，弟弟每分钟走{vs}米，哥哥每分钟走{vf}米，哥哥追上弟弟时走了多少米？",
    ])
    unit = "千米" if gap >= 100 else "米"
    lines = [
        f"速度差 = {vf} - {vs} = {dv}",
        f"追上时间 = {gap} ÷ {dv} = {num(t)}",
        f"{vf} × {num(t)} = {num(df)}{unit}",
    ]
    return ins, lines, df


_reg("catch_distance", catch_distance)


# 8. catch with time head start -> catch time
def catch_head_start(rng):
    v1 = rng.randint(50, 80)
    v2 = rng.randint(v1 + 10, 120)
    t0 = rng.randint(2, 6)
    head = v1 * t0
    dv = v2 - v1
    t = Fraction(head, dv)
    ins = rng.choice([
        f"甲每分钟走{v1}米，出发{t0}分钟后乙从同一地点出发去追，乙每分钟走{v2}米，乙出发后多少分钟追上甲？",
        f"慢车每小时行{v1}千米，先开出{t0}小时后快车从同一站出发，快车每小时行{v2}千米，快车出发后多少小时追上慢车？",
        f"弟弟每分钟跑{v1}米，先跑{t0}分钟后哥哥出发，哥哥每分钟跑{v2}米，哥哥出发后多少分钟追上弟弟？",
    ])
    lines = [
        f"{v1} × {t0} = {head}米",
        f"{v2} - {v1} = {dv}米/分",
        f"{head} ÷ {dv} = {num(t)}分",
    ]
    return ins, lines, t


_reg("catch_head_start", catch_head_start)


# 9. ratio a:b with total -> one part
def ratio_two_part(rng):
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    for _ in range(50):
        if a != b:
            break
        b = rng.randint(2, 9)
    total = (a + b) * rng.randint(4, 30)
    per = total // (a + b)
    A = a * per
    obj = rng.choice(["本书", "颗糖", "朵花", "个苹果"])
    ins = rng.choice([
        f"甲、乙两人分得的{obj}数量比是{a}:{b}，两人一共分得{total}{obj}，甲分得多少{obj}？",
        f"把{total}{obj}按{a}:{b}分给甲、乙两人，甲分得多少{obj}？",
        f"甲、乙两数的比是{a}:{b}，它们的和是{total}，甲数是多少？",
    ])
    lines = [
        f"总份数 = {a} + {b} = {a + b}",
        f"每份 = {total} ÷ {a + b} = {per}",
        f"甲分得 = {a} × {per} = {A}",
    ]
    return ins, lines, A


_reg("ratio_two_part", ratio_two_part)


# 10. ratio a:b:c with total -> largest part
def ratio_three_part(rng):
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    c = rng.randint(1, 5)
    s = a + b + c
    total = s * rng.randint(4, 30)
    per = total // s
    big = max(a, b, c)
    v = big * per
    obj = rng.choice(["元奖金", "本书", "颗糖", "棵树苗"])
    ins = rng.choice([
        f"把{total}{obj}按{a}:{b}:{c}分给甲、乙、丙三人，分得最多的人得到多少{obj}？",
        f"甲、乙、丙三人分得的{obj}数量比是{a}:{b}:{c}，一共{total}{obj}，最多的一份是多少{obj}？",
        f"三个数的比是{a}:{b}:{c}，它们的和是{total}，最大的数是多少？",
    ])
    lines = [
        f"总份数 = {a} + {b} + {c} = {s}",
        f"每份 = {total} ÷ {s} = {per}",
        f"最多的一份 = {big} × {per} = {v}",
    ]
    return ins, lines, v


_reg("ratio_three_part", ratio_three_part)


# 11. ratio a:b with difference given -> larger part
def ratio_difference_larger(rng):
    a = rng.randint(3, 9)
    b = rng.randint(2, a - 1)
    d = a - b
    diff = d * rng.randint(4, 30)
    per = diff // d
    larger = a * per
    obj = rng.choice(["元", "本书", "颗糖"])
    ins = rng.choice([
        f"甲、乙两人的钱数比是{a}:{b}，甲比乙多{diff}{obj}，甲有多少{obj}？",
        f"甲、乙两数的比是{a}:{b}，甲数比乙数大{diff}，甲数是多少？",
        f"哥哥和弟弟的邮票数比是{a}:{b}，哥哥比弟弟多{diff}张，哥哥有多少张？",
    ])
    lines = [
        f"份数差 = {a} - {b} = {d}",
        f"每份 = {diff} ÷ {d} = {per}",
        f"甲 = {a} × {per} = {larger}",
    ]
    return ins, lines, larger


_reg("ratio_difference_larger", ratio_difference_larger)


# 12. ratio split where one side also gets a fixed extra
def ratio_extra_fixed(rng):
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    per0 = rng.randint(4, 20)
    extra = rng.randint(5, 50)
    total = (a + b) * per0 + extra
    rem = total - extra
    per = Fraction(rem, a + b)
    A = a * per + extra
    obj = rng.choice(["元", "本书", "颗糖"])
    ins = rng.choice([
        f"把{total}{obj}按{a}:{b}分给甲、乙，分配时甲先拿走{extra}{obj}，剩下的再按比分配，甲共得多少{obj}？",
        f"奖金共{total}{obj}，先给甲{extra}{obj}，其余的按{a}:{b}分给甲、乙，甲一共得到多少{obj}？",
        f"甲、乙分{total}{obj}，甲先分得{extra}{obj}，剩下的按{a}:{b}分，甲最终得到多少{obj}？",
    ])
    lines = [
        f"{total} - {extra} = {num(rem)}{obj}",
        f"总份数 = {a} + {b} = {a + b}",
        f"{num(rem)} ÷ {a + b} = {num(per)}{obj}",
        f"{a} × {num(per)} + {extra} = {num(A)}{obj}",
    ]
    return ins, lines, A


_reg("ratio_extra_fixed", ratio_extra_fixed)


# 13. profit split by investment x time (weighted ratio)
def invest_weighted(rng):
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    m1 = rng.randint(2, 6)
    m2 = rng.randint(2, 6)
    wa = a * m1
    wb = b * m2
    ws = wa + wb
    profit = ws * rng.randint(2, 20)
    per = Fraction(profit, ws)
    share = wa * per
    ins = rng.choice([
        f"甲投资{a}万元{m1}个月，乙投资{b}万元{m2}个月，共获利{profit}万元，按投资金额乘时间分配，甲分得多少万元？",
        f"甲、乙合伙做生意，甲出{a}万元经营{m1}个月，乙出{b}万元经营{m2}个月，盈利{profit}万元按出资乘时间分配，甲得多少万元？",
        f"甲投入{a}万元{m1}个月，乙投入{b}万元{m2}个月，年终盈利{profit}万元，按资金乘时间的比分配，甲分得多少万元？",
    ])
    lines = [
        f"甲的权重 = {a} × {m1} = {wa}",
        f"乙的权重 = {b} × {m2} = {wb}",
        f"权重和 = {wa} + {wb} = {ws}",
        f"{profit} ÷ {ws} = {num(per)}万元",
        f"{wa} × {num(per)} = {num(share)}万元",
    ]
    return ins, lines, share


_reg("invest_weighted", invest_weighted)


# 14. fraction of whole, two stages, second stage amount
def frac_two_stage_second(rng):
    total = rng.randint(12, 60) * 10
    d1 = rng.choice([2, 3, 4])
    d2 = rng.choice([2, 3])
    first = Fraction(total, d1)
    rem = total - first
    second = Fraction(rem, d2)
    unit = rng.choice(["吨", "本", "千克", "升"])
    ins = rng.choice([
        f"仓库有{total}{unit}货物，第一天运走1/{d1}，第二天运走剩余部分的1/{d2}，第二天运走多少{unit}？",
        f"一根绳子长{total}{unit}，第一次剪去1/{d1}，第二次剪去剩下的1/{d2}，第二次剪去多少{unit}？",
        f"食堂运来{total}{unit}煤，第一周烧去1/{d1}，第二周烧去余下的1/{d2}，第二周烧去多少{unit}？",
    ])
    lines = [
        f"{total} × 1/{d1} = {num(first)}{unit}",
        f"{total} - {num(first)} = {num(rem)}{unit}",
        f"{num(rem)} × 1/{d2} = {num(second)}{unit}",
    ]
    return ins, lines, second


_reg("frac_two_stage_second", frac_two_stage_second)


# 15. three successive fraction uses -> remaining
def frac_three_stage_left(rng):
    total = rng.randint(24, 80) * 10
    d1 = rng.choice([2, 3, 4])
    d2 = rng.choice([2, 3, 4])
    d3 = rng.choice([2, 3, 4])
    r1 = Fraction(total * (d1 - 1), d1)
    r2 = Fraction(r1 * (d2 - 1), d2)
    r3 = Fraction(r2 * (d3 - 1), d3)
    unit = rng.choice(["吨", "本", "千克", "升"])
    ins = rng.choice([
        f"仓库有{total}{unit}货物，第一天运走1/{d1}，第二天运走余下的1/{d2}，第三天运走余下的1/{d3}，还剩多少{unit}？",
        f"一根绳子长{total}{unit}，第一次剪去1/{d1}，第二次剪去剩下的1/{d2}，第三次剪去剩下的1/{d3}，还剩多少{unit}？",
        f"食堂有{total}{unit}大米，第一天吃了1/{d1}，第二天吃了剩下的1/{d2}，第三天吃了剩下的1/{d3}，还剩多少{unit}？",
    ])
    lines = [
        f"{total} × ({d1} - 1)/{d1} = {num(r1)}{unit}",
        f"{num(r1)} × ({d2} - 1)/{d2} = {num(r2)}{unit}",
        f"{num(r2)} × ({d3} - 1)/{d3} = {num(r3)}{unit}",
    ]
    return ins, lines, r3


_reg("frac_three_stage_left", frac_three_stage_left)


# 16. nested fraction: 乙 gets 1/d of total, gives 1/e of that away -> kept
def frac_nested_kept(rng):
    total = rng.randint(12, 60) * 10
    d = rng.choice([2, 3, 4])
    e = rng.choice([2, 3, 4])
    first = Fraction(total, d)
    given = Fraction(first, e)
    kept = first - given
    unit = rng.choice(["元", "本", "千克"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}有{total}{unit}糖，分给弟弟1/{d}，弟弟又把自己得到的1/{e}分给妹妹，弟弟还剩多少{unit}？",
        f"甲有{total}{unit}货物，运给乙1/{d}，乙又把其中的1/{e}转运给丙，乙还剩多少{unit}？",
        f"书店有{total}{unit}书，分给一班1/{d}，一班又把其中的1/{e}分给二班，一班还剩多少{unit}？",
    ])
    lines = [
        f"{total} × 1/{d} = {num(first)}{unit}",
        f"{num(first)} × 1/{e} = {num(given)}{unit}",
        f"{num(first)} - {num(given)} = {num(kept)}{unit}",
    ]
    return ins, lines, kept


_reg("frac_nested_kept", frac_nested_kept)


# 17. reverse: after two fraction uses, remaining R given -> original total
def frac_reverse_original(rng):
    d1 = rng.choice([2, 3, 4])
    d2 = rng.choice([2, 3, 4])
    k = rng.randint(2, 10)
    R = (d1 - 1) * (d2 - 1) * k
    r1 = Fraction(R * d2, d2 - 1)
    total = Fraction(r1 * d1, d1 - 1)
    unit = rng.choice(["吨", "本", "千克"])
    ins = rng.choice([
        f"仓库有一批货物，第一天运走1/{d1}，第二天运走余下的1/{d2}，还剩{R}{unit}，这批货物原来有多少{unit}？",
        f"一根绳子，第一次剪去1/{d1}，第二次剪去剩下的1/{d2}，还剩{R}{unit}，这根绳子原来长多少{unit}？",
        f"食堂有一批煤，第一周烧去1/{d1}，第二周烧去余下的1/{d2}，还剩{R}{unit}，原来有多少{unit}？",
    ])
    lines = [
        f"{R} × {d2} ÷ ({d2} - 1) = {num(r1)}{unit}",
        f"{num(r1)} - {R} = {num(Fraction(r1, d2))}{unit}",
        f"{num(r1)} × {d1} ÷ ({d1} - 1) = {num(total)}{unit}",
    ]
    return ins, lines, total


_reg("frac_reverse_original", frac_reverse_original)


# 18. part is p% of whole -> complement (the rest)
def part_pct_complement(rng):
    p = rng.choice([10, 20, 25, 40, 50, 60, 75])
    whole = rng.randint(2, 10) * 100
    part = Fraction(whole * p, 100)
    girls = whole - part
    ins = rng.choice([
        f"六年级有学生{whole}人，其中男生占{p}%，女生有多少人？",
        f"书店运来{whole}本书，其中故事书占{p}%，其余的是科技书，科技书有多少本？",
        f"果园有果树{whole}棵，其中苹果树占{p}%，其余的是梨树，梨树有多少棵？",
    ])
    lines = [
        f"1%的量 = {whole} ÷ 100 = {whole // 100}",
        f"占{p}%的部分 = {whole // 100} × {p} = {num(part)}",
        f"其余 = {whole} - {num(part)} = {num(girls)}",
    ]
    return ins, lines, girls


_reg("part_pct_complement", part_pct_complement)


# 19. part = whole/d + extra; then a fraction of the rest joins a club
def part_frac_extra_club(rng):
    d = rng.choice([3, 4, 5])
    k = rng.randint(4, 15)
    whole = d * k
    e = rng.randint(2, 9)
    part = k + e
    f = rng.choice([2, 3])
    rest = whole - part
    club = Fraction(rest, f)
    ins = rng.choice([
        f"全班有{whole}人，参加合唱队的人数比全班的1/{d}多{e}人，其余同学的1/{f}参加美术组，美术组有多少人？",
        f"图书馆有{whole}本书，故事书比总数的1/{d}多{e}本，其余书的1/{f}是漫画书，漫画书有多少本？",
        f"果园有{whole}棵果树，苹果树比总数的1/{d}多{e}棵，其余果树的1/{f}是桃树，桃树有多少棵？",
    ])
    lines = [
        f"每份 = {whole} × 1/{d} = {k}",
        f"比1/{d}多{e}的 = {k} + {e} = {part}",
        f"其余 = {whole} - {part} = {rest}",
        f"其余的1/{f} = {rest} × 1/{f} = {num(club)}",
    ]
    return ins, lines, club


_reg("part_frac_extra_club", part_frac_extra_club)


# 20. rectangle area -> number of square tiles
def area_tiles(rng):
    s = rng.choice([2, 3, 4])
    l = s * rng.randint(3, 7)
    w = s * rng.randint(3, 7)
    area = l * w
    tile = s * s
    n = area // tile
    ins = rng.choice([
        f"一间教室长{l}米、宽{w}米，用边长{s}米的方砖铺地，需要多少块？",
        f"一块长方形地长{l}米、宽{w}米，分成边长{s}米的正方形小菜地，可以分成多少块？",
        f"客厅长{l}米、宽{w}米，铺边长{s}米的正方形地砖，一共需要多少块？",
    ])
    lines = [
        f"{l} × {w} = {area}平方米",
        f"{s} × {s} = {tile}平方米",
        f"{area} ÷ {tile} = {n}块",
    ]
    return ins, lines, n


_reg("area_tiles", area_tiles)


# 21. triangle area x fraction x unit cost
def triangle_frac_cost(rng):
    b = rng.randint(4, 12)
    h = rng.randint(4, 12)
    d = rng.choice([2, 3])
    c = rng.randint(10, 50)
    area = Fraction(b * h, 2)
    planted = Fraction(area, d)
    total = planted * c
    ins = rng.choice([
        f"一块三角形菜地，底{b}米、高{h}米，用它的1/{d}种白菜，每平方米收白菜{c}元，种白菜的地值多少元？",
        f"三角形花圃底{b}米、高{h}米，其中1/{d}种玫瑰，每平方米玫瑰卖{c}元，玫瑰地一共值多少元？",
        f"一块三角形土地底{b}米、高{h}米，划出1/{d}种草坪，每平方米草坪{c}元，草坪共值多少元？",
    ])
    lines = [
        f"{b} × {h} ÷ 2 = {num(area)}平方米",
        f"{num(area)} × 1/{d} = {num(planted)}平方米",
        f"{num(planted)} × {c} = {num(total)}元",
    ]
    return ins, lines, total


_reg("triangle_frac_cost", triangle_frac_cost)


# 22. trapezoid area x unit cost
def trapezoid_cost(rng):
    a = rng.randint(3, 10)
    b = rng.randint(3, 10)
    h = rng.randint(4, 12)
    c = rng.randint(10, 50)
    s = a + b
    area = Fraction(s * h, 2)
    total = area * c
    ins = rng.choice([
        f"一块梯形菜地，上底{a}米、下底{b}米、高{h}米，每平方米种菜收入{c}元，这块地一共收入多少元？",
        f"梯形花坛上底{a}米、下底{b}米、高{h}米，每平方米铺草皮{c}元，铺满需要多少元？",
        f"一块梯形土地上底{a}米、下底{b}米、高{h}米，每平方米种果树收入{c}元，总收入多少元？",
    ])
    lines = [
        f"{a} + {b} = {s}米",
        f"{s} × {h} ÷ 2 = {num(area)}平方米",
        f"{num(area)} × {c} = {num(total)}元",
    ]
    return ins, lines, total


_reg("trapezoid_cost", trapezoid_cost)


# 23. border area (outer - inner rectangle) x cost
def border_cost(rng):
    L = rng.randint(8, 16)
    W = rng.randint(8, 16)
    bw = rng.choice([1, 2])
    l = L - 2 * bw
    w = W - 2 * bw
    c = rng.randint(10, 40)
    outer = L * W
    inner = l * w
    border = outer - inner
    total = border * c
    ins = rng.choice([
        f"一个长方形花坛长{L}米、宽{W}米，四周修{bw}米宽的小路，小路每平方米{c}元，修小路共需多少元？",
        f"一块长方形地长{L}米、宽{W}米，中间留长{l}米、宽{w}米的草坪，四周铺地砖，每平方米{c}元，共需多少元？",
        f"长方形游泳池长{L}米、宽{W}米，四周铺{bw}米宽的防滑砖，每平方米{c}元，一共需要多少元？",
    ])
    lines = [
        f"{L} × {W} = {outer}平方米",
        f"{l} × {w} = {inner}平方米",
        f"{outer} - {inner} = {border}平方米",
        f"{border} × {c} = {total}元",
    ]
    return ins, lines, total


_reg("border_cost", border_cost)


# 24. square perimeter -> side -> area -> cost
def square_perimeter_cost(rng):
    side = rng.randint(4, 12)
    P = side * 4
    area = side * side
    c = rng.randint(10, 50)
    total = area * c
    ins = rng.choice([
        f"一块正方形菜地的周长是{P}米，每平方米种菜收入{c}元，这块地一共收入多少元？",
        f"正方形花坛周长{P}米，每平方米种花{c}元，种满花需要多少元？",
        f"一块正方形地周长{P}米，每平方米铺地砖{c}元，铺满需要多少元？",
    ])
    lines = [
        f"{P} ÷ 4 = {side}米",
        f"{side} × {side} = {area}平方米",
        f"{area} × {c} = {total}元",
    ]
    return ins, lines, total


_reg("square_perimeter_cost", square_perimeter_cost)


# 25. tank volume x fraction depth -> liters
def tank_volume_frac(rng):
    l = rng.randint(2, 8)
    w = rng.randint(2, 8)
    h = rng.randint(2, 8)
    d = rng.choice([2, 3, 4])
    base = l * w
    vol = base * h
    water = Fraction(vol, d)
    ins = rng.choice([
        f"一个长方体水箱，从里面量长{l}分米、宽{w}分米、高{h}分米，注入1/{d}的水，水有多少升？",
        f"长方体鱼缸长{l}分米、宽{w}分米、高{h}分米，装水到容量的1/{d}，装了多少升水？",
        f"一个长方体水池长{l}米、宽{w}米、深{h}米，蓄水到容量的1/{d}，蓄水多少升？",
    ])
    lines = [
        f"底面积 = {l} × {w} = {base}",
        f"容积 = {base} × {h} = {vol}",
        f"{vol} × 1/{d} = {num(water)}升",
    ]
    return ins, lines, water


_reg("tank_volume_frac", tank_volume_frac)


# 26. two shared costs split evenly
def group_two_costs(rng):
    n = rng.randint(3, 6)
    q1 = rng.randint(2, 6)
    p1 = rng.randint(3, 20)
    q2 = rng.randint(2, 6)
    p2 = rng.randint(2, 10)
    t1 = q1 * p1
    t2 = q2 * p2
    grand = t1 + t2
    each = Fraction(grand, n)
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"{n}个同学合买{q1}份{p1}元的{obj}和{q2}份{p2}元的饮料，平均每人付多少元？",
        f"{n}人一起买{q1}本{p1}元的笔记本和{q2}支{p2}元的铅笔，平均每人出多少元？",
        f"{n}个朋友聚餐，点了{q1}份{p1}元的菜和{q2}份{p2}元的主食，平均每人付多少元？",
    ])
    lines = [
        f"{q1} × {p1} = {t1}元",
        f"{q2} × {p2} = {t2}元",
        f"{t1} + {t2} = {grand}元",
        f"{grand} ÷ {n} = {num(each)}元",
    ]
    return ins, lines, each


_reg("group_two_costs", group_two_costs)


# 27. one person pays a fixed extra, rest split evenly
def group_one_extra(rng):
    n = rng.randint(3, 6)
    base = rng.randint(5, 30)
    extra = rng.randint(5, 50)
    total = base * n + extra
    jia = base + extra
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{n}人聚餐共{total}元，{name}比其他人多付{extra}元，其余人付的一样多，{name}付了多少元？",
        f"{n}个同学合买礼物共花{total}元，{name}多出{extra}元，其余同学出的钱相同，{name}出了多少元？",
        f"一笔{total}元的费用由{n}人分摊，{name}多承担{extra}元，其余每人出的一样多，{name}承担多少元？",
    ])
    lines = [
        f"{total} - {extra} = {base * n}元",
        f"{base * n} ÷ {n} = {base}元",
        f"{base} + {extra} = {jia}元",
    ]
    return ins, lines, jia


_reg("group_one_extra", group_one_extra)


# 28. weighted average of three groups
def weighted_average_three(rng):
    n1 = rng.randint(2, 8)
    n2 = rng.randint(2, 8)
    n3 = rng.randint(2, 8)
    a1 = rng.randint(70, 95)
    a2 = rng.randint(70, 95)
    a3 = rng.randint(70, 95)
    s1 = n1 * a1
    s2 = n2 * a2
    s3 = n3 * a3
    total = s1 + s2 + s3
    n = n1 + n2 + n3
    avg = Fraction(total, n)
    ins = rng.choice([
        f"三个班参加考试，一班{n1}人平均{a1}分，二班{n2}人平均{a2}分，三班{n3}人平均{a3}分，三个班的总平均分是多少分？",
        f"三个小组植树，一组{n1}人平均每人植{a1}棵，二组{n2}人平均每人植{a2}棵，三组{n3}人平均每人植{a3}棵，平均每人植多少棵？",
        f"甲队{n1}人平均体重{a1}千克，乙队{n2}人平均体重{a2}千克，丙队{n3}人平均体重{a3}千克，三队平均体重多少千克？",
    ])
    lines = [
        f"第一项 = {n1} × {a1} = {s1}",
        f"第二项 = {n2} × {a2} = {s2}",
        f"第三项 = {n3} × {a3} = {s3}",
        f"总和 = {s1} + {s2} + {s3} = {total}",
        f"总人数 = {n1} + {n2} + {n3} = {n}",
        f"平均数 = {total} ÷ {n} = {num(avg)}",
    ]
    return ins, lines, avg


_reg("weighted_average_three", weighted_average_three)


# 29. two discounts -> total saved
def discount_saved(rng):
    price = rng.randint(20, 90) * 10
    d1 = rng.choice([7, 8, 9])
    d2 = rng.choice([7, 8, 9])
    a1 = Fraction(price * d1, 10)
    a2 = Fraction(a1 * d2, 10)
    saved = price - a2
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}原价{price}元，先打{d1}折，再打{d2}折，一共便宜了多少元？",
        f"一台电器原价{price}元，先降价到{d1}折，再打{d2}折，比原价便宜多少元？",
        f"某商品原价{price}元，先按{d1}折出售，再按{d2}折出售，共优惠多少元？",
    ])
    lines = [
        f"{price} × {d1}/10 = {num(a1)}元",
        f"{num(a1)} × {d2}/10 = {num(a2)}元",
        f"{price} - {num(a2)} = {num(saved)}元",
    ]
    return ins, lines, saved


_reg("discount_saved", discount_saved)


# 30. three chained discounts
def discount_three(rng):
    price = rng.randint(30, 90) * 10
    d1 = rng.choice([8, 9])
    d2 = rng.choice([8, 9])
    d3 = rng.choice([8, 9])
    a1 = Fraction(price * d1, 10)
    a2 = Fraction(a1 * d2, 10)
    a3 = Fraction(a2 * d3, 10)
    ins = rng.choice([
        f"一件商品原价{price}元，先打{d1}折，再打{d2}折，最后再打{d3}折，现在多少元？",
        f"一台电器原价{price}元，连续三次打折，分别打{d1}折、{d2}折、{d3}折，现价多少元？",
        f"某商品原价{price}元，先按{d1}折出售，顾客持会员卡再打{d2}折，用优惠券再打{d3}折，最终多少元？",
    ])
    lines = [
        f"{price} × {d1}/10 = {num(a1)}元",
        f"{num(a1)} × {d2}/10 = {num(a2)}元",
        f"{num(a2)} × {d3}/10 = {num(a3)}元",
    ]
    return ins, lines, a3


_reg("discount_three", discount_three)


# 31. discount then fixed coupon -> final price
def discount_coupon(rng):
    price = rng.randint(30, 90) * 10
    d = rng.choice([8, 9])
    C = rng.choice([20, 30, 50, 100])
    a1 = Fraction(price * d, 10)
    for _ in range(50):
        if C < a1:
            break
        C = rng.choice([20, 30, 50])
    cut = price - a1
    final = a1 - C
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}原价{price}元，先打{d}折，再用优惠券减{C}元，现在多少元？",
        f"某电器原价{price}元，先按{d}折出售，持会员卡再减{C}元，现价多少元？",
        f"一件商品原价{price}元，先打{d}折，参加活动再减{C}元，实际付多少元？",
    ])
    lines = [
        f"{price} × {d}/10 = {num(a1)}元",
        f"{price} - {num(a1)} = {num(cut)}元",
        f"{num(a1)} - {C} = {num(final)}元",
    ]
    return ins, lines, final


_reg("discount_coupon", discount_coupon)


# 32. compare two discount schemes -> difference
def discount_compare(rng):
    P = rng.randint(50, 90) * 10
    d1 = 9
    d2 = 8
    C = rng.choice([10, 20, 30])
    a = Fraction(P * d1, 10)
    b1 = Fraction(P * d2, 10)
    b = b1 - C
    diff = a - b
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}原价{P}元，甲店直接打{d1}折，乙店先打{d2}折再减{C}元，乙店比甲店便宜多少元？",
        f"某商品原价{P}元，方案一打{d1}折出售，方案二先打{d2}折再减{C}元，两种方案相差多少元？",
        f"一台电器原价{P}元，柜台A打{d1}折，柜台B先打{d2}折再减{C}元，买B比买A少花多少元？",
    ])
    lines = [
        f"{P} × {d1}/10 = {num(a)}元",
        f"{P} × {d2}/10 = {num(b1)}元",
        f"{num(b1)} - {C} = {num(b)}元",
        f"{num(a)} - {num(b)} = {num(diff)}元",
    ]
    return ins, lines, diff


_reg("discount_compare", discount_compare)


# 33. fractional hours of travel -> round-trip time
def speed_roundtrip_time(rng):
    v = rng.randint(3, 12)
    t = rng.choice([Fraction(1, 2), Fraction(1, 3), Fraction(1, 4),
                    Fraction(2, 3), Fraction(3, 4)])
    d = v * t
    total = 2 * d
    t2 = Fraction(total, v)
    ins = rng.choice([
        f"小明骑车每小时行{v}千米，从家到图书馆用了{num(t)}小时，往返一共需要多少小时？",
        f"一艘船每小时行{v}千米，从甲港到乙港用了{num(t)}小时，往返一次需要多少小时？",
        f"小红步行每小时走{v}千米，从家到学校用了{num(t)}小时，上学和放学一共要走多少小时？",
    ])
    lines = [
        f"{v} × {num(t)} = {num(d)}千米",
        f"{num(d)} × 2 = {num(total)}千米",
        f"{num(total)} ÷ {v} = {num(t2)}小时",
    ]
    return ins, lines, t2


_reg("speed_roundtrip_time", speed_roundtrip_time)


# 34. m/s x minutes -> km (unit conversion)
def speed_ms_km(rng):
    v = rng.randint(5, 15)
    t = rng.randint(2, 10)
    per_min = v * 60
    d = per_min * t
    km = Fraction(d, 1000)
    ins = rng.choice([
        f"一辆汽车每秒行{v}米，行了{t}分钟，行了多少千米？",
        f"一列火车每秒行驶{v}米，通过一座桥用了{t}分钟，这段路程是多少千米？",
        f"小明骑车每秒行{v}米，骑了{t}分钟，一共行了多少千米？",
    ])
    lines = [
        f"{v} × 60 = {per_min}米",
        f"{per_min} × {t} = {d}米",
        f"{d} ÷ 1000 = {num(km)}千米",
    ]
    return ins, lines, km


_reg("speed_ms_km", speed_ms_km)


# 35. round trip at two speeds -> average speed
def roundtrip_avg(rng):
    D = rng.randint(6, 30) * 10
    v1 = rng.randint(30, 90)
    v2 = rng.randint(30, 90)
    for _ in range(50):
        if v1 != v2:
            break
        v2 = rng.randint(30, 90)
    t1 = Fraction(D, v1)
    t2 = Fraction(D, v2)
    tt = t1 + t2
    total = 2 * D
    avg = Fraction(total, tt)
    ins = rng.choice([
        f"甲、乙两地相距{D}千米，一辆汽车去时每小时行{v1}千米，返回时每小时行{v2}千米，往返的平均速度是多少千米/时？",
        f"一艘船在相距{D}千米的两港间航行，顺水每小时行{v1}千米，逆水每小时行{v2}千米，往返平均速度是多少？",
        f"小明去外婆家路程{D}千米，去时骑车每小时{v1}千米，回来步行每小时{v2}千米，往返平均速度是多少千米/时？",
    ])
    lines = [
        f"{D} ÷ {v1} = {num(t1)}小时",
        f"{D} ÷ {v2} = {num(t2)}小时",
        f"{num(t1)} + {num(t2)} = {num(tt)}小时",
        f"{D} × 2 = {total}千米",
        f"{total} ÷ ({num(tt)}) = {num(avg)}千米/时",
    ]
    return ins, lines, avg


_reg("roundtrip_avg", roundtrip_avg)


# 36. boat upstream/downstream -> downstream distance
def upstream_downstream(rng):
    b = rng.randint(10, 30)
    c = rng.randint(2, 8)
    t = rng.choice([Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), 1, 2])
    down = b + c
    up = b - c
    d = down * t
    ins = rng.choice([
        f"一艘船在静水中每小时行{b}千米，水流速度每小时{c}千米，顺水航行{num(t)}小时，行了多少千米？",
        f"轮船在静水中速度每小时{b}千米，水速每小时{c}千米，顺水走{num(t)}小时，路程是多少千米？",
        f"一条河水流速度每小时{c}千米，船在静水中每小时行{b}千米，顺水航行{num(t)}小时，行了多少千米？",
    ])
    lines = [
        f"{b} + {c} = {down}千米/时",
        f"{b} - {c} = {up}千米/时",
        f"{down} × {num(t)} = {num(d)}千米",
    ]
    return ins, lines, d


_reg("upstream_downstream", upstream_downstream)


# 37. train crossing bridge: full-cross time and fully-on-bridge time
def train_bridge_two(rng):
    L = rng.randint(8, 20) * 10
    B = rng.randint(30, 60) * 10
    v = rng.randint(10, 30)
    total = L + B
    t1 = Fraction(total, v)
    on = B - L
    t2 = Fraction(on, v)
    ins = rng.choice([
        f"一列火车长{L}米，以每秒{v}米的速度通过一座长{B}米的大桥，火车完全在桥上的时间是多少秒？",
        f"火车长{L}米，桥长{B}米，火车每秒行{v}米，从车头进桥到车尾离桥后，整列火车都在桥上的时间是多少秒？",
        f"一列长{L}米的火车以每秒{v}米的速度穿过长{B}米的隧道，火车完全在隧道里的时间是多少秒？",
    ])
    lines = [
        f"{L} + {B} = {total}米",
        f"{total} ÷ {v} = {num(t1)}秒",
        f"{B} - {L} = {on}米",
        f"{on} ÷ {v} = {num(t2)}秒",
    ]
    return ins, lines, t2


_reg("train_bridge_two", train_bridge_two)


# 38. budget minus two expenses, split the rest
def budget_two_split(rng):
    T = rng.randint(50, 150) * 10
    e1 = rng.randint(5, 20) * 10
    e2 = rng.randint(5, 20) * 10
    for _ in range(50):
        if e1 + e2 < T:
            break
        e1 = rng.randint(5, 20) * 10
        e2 = rng.randint(5, 20) * 10
    g = rng.randint(2, 4)
    rest = T - e1 - e2
    each = Fraction(rest, g)
    ins = rng.choice([
        f"一笔{T}元的活动经费，买奖品用了{e1}元，买水果用了{e2}元，余下的平均分给{g}个小组，每个小组得多少元？",
        f"学校拨款{T}元，买图书用去{e1}元，买体育用品用去{e2}元，剩下的平均分给{g}个班，每班多少元？",
        f"一笔预算{T}元，设备花了{e1}元，材料花了{e2}元，剩余的分给{g}个小组，每组多少元？",
    ])
    lines = [
        f"{T} - {e1} = {T - e1}元",
        f"{T - e1} - {e2} = {rest}元",
        f"{rest} ÷ {g} = {num(each)}元",
    ]
    return ins, lines, each


_reg("budget_two_split", budget_two_split)


# 39. budget: fractions to two categories, rest split
def budget_fractions_split(rng):
    T = rng.randint(24, 80) * 10
    d1 = rng.choice([2, 3, 4])
    d2 = rng.choice([3, 4, 5])
    g = rng.randint(2, 3)
    a = Fraction(T, d1)
    b = Fraction(T, d2)
    rest = T - a - b
    each = Fraction(rest, g)
    ins = rng.choice([
        f"一笔{T}元的经费，1/{d1}用于设备，1/{d2}用于材料，余下的平均分给{g}个小组，每组多少元？",
        f"学校有经费{T}元，1/{d1}买图书，1/{d2}买仪器，剩下的平均分给{g}个社团，每个社团多少元？",
        f"一笔预算{T}元，1/{d1}用于装修，1/{d2}用于家具，其余的平均分给{g}个部门，每个部门多少元？",
    ])
    lines = [
        f"{T} × 1/{d1} = {num(a)}元",
        f"{T} × 1/{d2} = {num(b)}元",
        f"{T} - {num(a)} - {num(b)} = {num(rest)}元",
        f"{num(rest)} ÷ {g} = {num(each)}元",
    ]
    return ins, lines, each


_reg("budget_fractions_split", budget_fractions_split)


# 40. budget: two percents, rest to a third party
def budget_pcts_rest(rng):
    T = rng.randint(30, 90) * 10
    p = rng.randint(10, 30)
    q = rng.randint(10, 40)
    a = Fraction(T * p, 100)
    b = Fraction(T * q, 100)
    c = T - a - b
    ins = rng.choice([
        f"一笔{T}元的奖金，{p}%给甲，{q}%给乙，其余的给丙，丙得多少元？",
        f"学校有经费{T}元，{p}%用于买书，{q}%用于买器材，其余的存入银行，存了多少元？",
        f"一笔捐款{T}元，{p}%给一班，{q}%给二班，剩下的给三班，三班得多少元？",
    ])
    lines = [
        f"{T} × {p}/100 = {num(a)}元",
        f"{T} × {q}/100 = {num(b)}元",
        f"{T} - {num(a)} - {num(b)} = {num(c)}元",
    ]
    return ins, lines, c


_reg("budget_pcts_rest", budget_pcts_rest)


# 41. mix two prices -> average price
def mix_avg_price(rng):
    q1 = rng.randint(2, 10)
    q2 = rng.randint(2, 10)
    p1 = rng.randint(3, 12)
    p2 = rng.randint(10, 20)
    c1 = q1 * p1
    c2 = q2 * p2
    total = c1 + c2
    q = q1 + q2
    avg = Fraction(total, q)
    fruit = rng.choice(FRUITS)
    ins = rng.choice([
        f"商店把{q1}千克每千克{p1}元的{fruit}和{q2}千克每千克{p2}元的{fruit}混合出售，混合后每千克多少元？",
        f"买{q1}千克{p1}元的糖和{q2}千克{p2}元的糖混在一起，平均每千克多少元？",
        f"甲种{fruit}{q1}千克每千克{p1}元，乙种{fruit}{q2}千克每千克{p2}元，混合后平均每千克多少元？",
    ])
    lines = [
        f"{q1} × {p1} = {c1}元",
        f"{q2} × {p2} = {c2}元",
        f"{c1} + {c2} = {total}元",
        f"{q1} + {q2} = {q}千克",
        f"{total} ÷ {q} = {num(avg)}元",
    ]
    return ins, lines, avg


_reg("mix_avg_price", mix_avg_price)


# 42. mixture: average given, find the cheaper quantity
def mix_find_cheaper(rng):
    gap = rng.choice([2, 3, 4, 6])
    cand = [q for q in range(4, 13) if math.gcd(q, gap) > 1]
    Q = rng.choice(cand)
    g = math.gcd(Q, gap)
    x = Q // g
    p1 = rng.randint(3, 20 - gap)
    p2 = p1 + gap
    A = p2 - gap // g
    total = Q * A
    alldear = Q * p2
    diff = alldear - total
    gap = p2 - p1
    ins = rng.choice([
        f"用每千克{p1}元和每千克{p2}元的两种糖混合成{Q}千克什锦糖，平均每千克{A}元，便宜的糖用了多少千克？",
        f"把每千克{p1}元的茶叶和每千克{p2}元的茶叶混合成{Q}千克，平均每千克{A}元，便宜的茶叶有多少千克？",
        f"甲、乙两种{FRUITS[0]}，甲每千克{p1}元，乙每千克{p2}元，混合{Q}千克后平均每千克{A}元，甲种用了多少千克？",
    ])
    lines = [
        f"{Q} × {A} = {total}元",
        f"{Q} × {p2} = {alldear}元",
        f"{alldear} - {total} = {diff}元",
        f"{p2} - {p1} = {gap}元",
        f"{diff} ÷ {gap} = {x}千克",
    ]
    return ins, lines, x


_reg("mix_find_cheaper", mix_find_cheaper)


# 43. mix two solutions -> salt per 100 grams
def conc_mix_per100(rng):
    s1 = rng.randint(5, 30) * 10
    s2 = rng.randint(5, 30) * 10
    c1 = rng.choice([5, 10, 15, 20, 25])
    c2 = rng.choice([5, 10, 15, 20, 25])
    salt1 = Fraction(s1 * c1, 100)
    salt2 = Fraction(s2 * c2, 100)
    salt = salt1 + salt2
    total = s1 + s2
    p = Fraction(salt * 100, total)
    ins = rng.choice([
        f"把{s1}克含盐{c1}%的盐水和{s2}克含盐{c2}%的盐水混合，混合后每100克盐水含盐多少克？",
        f"甲杯有{s1}克含盐{c1}%的盐水，乙杯有{s2}克含盐{c2}%的盐水，混合后每100克含盐多少克？",
        f"两种盐水分别重{s1}克和{s2}克，浓度为{c1}%和{c2}%，混合后每100克盐水含盐多少克？",
    ])
    lines = [
        f"{s1} × {c1}/100 = {num(salt1)}克",
        f"{s2} × {c2}/100 = {num(salt2)}克",
        f"{num(salt1)} + {num(salt2)} = {num(salt)}克",
        f"{s1} + {s2} = {total}克",
        f"{num(salt)} ÷ {total} × 100 = {num(p)}克",
    ]
    return ins, lines, p


_reg("conc_mix_per100", conc_mix_per100)


# 44. add water to a solution -> salt per 100 grams
def conc_add_water(rng):
    s = rng.randint(5, 30) * 10
    c = rng.choice([5, 10, 15, 20, 25])
    w = rng.randint(2, 20) * 10
    salt = Fraction(s * c, 100)
    total = s + w
    p = Fraction(salt * 100, total)
    ins = rng.choice([
        f"有{s}克含盐{c}%的盐水，加入{w}克水后，每100克盐水含盐多少克？",
        f"一杯{s}克的盐水含盐{c}%，再加入{w}克水，这时每100克盐水含盐多少克？",
        f"把{w}克水加入{s}克含盐{c}%的盐水中，混合后每100克含盐多少克？",
    ])
    lines = [
        f"{s} × {c}/100 = {num(salt)}克",
        f"{s} + {w} = {total}克",
        f"{num(salt)} ÷ {total} × 100 = {num(p)}克",
    ]
    return ins, lines, p


_reg("conc_add_water", conc_add_water)


# 45. evaporate water -> salt per 100 grams
def conc_evaporate(rng):
    s = rng.randint(10, 30) * 10
    c = rng.choice([5, 10, 15, 20])
    e = rng.randint(2, 8) * 10
    salt = Fraction(s * c, 100)
    total = s - e
    p = Fraction(salt * 100, total)
    ins = rng.choice([
        f"有{s}克含盐{c}%的盐水，蒸发掉{e}克水后，每100克盐水含盐多少克？",
        f"一杯{s}克的盐水含盐{c}%，放在太阳下蒸发了{e}克水，这时每100克盐水含盐多少克？",
        f"把{s}克含盐{c}%的盐水加热，蒸发{e}克水后，每100克盐水含盐多少克？",
    ])
    lines = [
        f"{s} × {c}/100 = {num(salt)}克",
        f"{s} - {e} = {total}克",
        f"{num(salt)} ÷ {total} × 100 = {num(p)}克",
    ]
    return ins, lines, p


_reg("conc_evaporate", conc_evaporate)


# 46. two workers together -> time
def work_together(rng):
    a = rng.randint(3, 12)
    b = rng.randint(3, 12)
    for _ in range(50):
        if a != b:
            break
        b = rng.randint(3, 12)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    rate = ra + rb
    t = Fraction(1, rate)
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成，乙单独做{b}天完成，两人合作多少天完成？",
        f"修一条路，甲队单独修{a}天完成，乙队单独修{b}天完成，两队合修多少天完成？",
        f"打一份稿件，甲单独打{a}小时完成，乙单独打{b}小时完成，两人合打多少小时完成？",
    ])
    lines = [
        f"甲的效率 = 1 ÷ {a} = {num(ra)}",
        f"乙的效率 = 1 ÷ {b} = {num(rb)}",
        f"效率和 = {num(ra)} + {num(rb)} = {num(rate)}",
        f"1 ÷ ({num(rate)}) = {num(t)}天",
    ]
    return ins, lines, t


_reg("work_together", work_together)


# 47. work together for t0 days, then one leaves -> remaining time
def work_one_leaves(rng):
    a = rng.choice([6, 8, 10, 12])
    b = rng.choice([4, 6, 8])
    t0 = 1
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    rate = ra + rb
    done = rate * t0
    rem = 1 - done
    t = Fraction(rem, ra)
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成，乙单独做{b}天完成，两人合作{t0}天后乙离开，剩下的甲还要做多少天？",
        f"修一条路，甲队单独修{a}天完成，乙队单独修{b}天完成，两队合修{t0}天后乙队调走，甲队还要修多少天？",
        f"一批零件，甲单独做{a}天完成，乙单独做{b}天完成，两人合作{t0}天后乙有事离开，甲还需多少天完成？",
    ])
    lines = [
        f"甲的效率 = 1 ÷ {a} = {num(ra)}",
        f"乙的效率 = 1 ÷ {b} = {num(rb)}",
        f"效率和 = {num(ra)} + {num(rb)} = {num(rate)}",
        f"已完成 = {num(rate)} × {t0} = {num(done)}",
        f"剩下 = 1 - {num(done)} = {num(rem)}",
        f"({num(rem)}) ÷ ({num(ra)}) = {num(t)}天",
    ]
    return ins, lines, t


_reg("work_one_leaves", work_one_leaves)


# 48. daily rate x days = p% of plan -> remaining amount
def rate_pct_remaining(rng):
    rate = rng.randint(20, 80) * 10
    days = rng.randint(3, 7)
    p = rng.choice([20, 25, 40, 50, 60])
    made = rate * days
    total = Fraction(made * 100, p)
    rem = total - made
    unit = rng.choice(["件", "吨"])
    ins = rng.choice([
        f"工厂每天生产{rate}{unit}，{days}天完成了全部计划的{p}%，这批货物还剩多少{unit}？",
        f"车间每天加工{rate}{unit}零件，{days}天完成了计划的{p}%，还要生产多少{unit}才能完成？",
        f"工厂每天生产{rate}{unit}，{days}天完成了全部计划的{p}%，还剩多少{unit}没完成？",
    ])
    lines = [
        f"{rate} × {days} = {made}{unit}",
        f"{made} ÷ {p} × 100 = {num(total)}{unit}",
        f"{num(total)} - {made} = {num(rem)}{unit}",
    ]
    return ins, lines, rem


_reg("rate_pct_remaining", rate_pct_remaining)


# 49. two production phases at different rates -> second phase time
def rate_two_phase(rng):
    r1 = rng.choice([10, 12, 15, 20])
    r2 = rng.choice([10, 12, 15, 20])
    d = rng.choice([2, 3, 4])
    k = rng.randint(1, 2)
    T = d * r1 * r2 * k
    made = Fraction(T, d)
    t1 = Fraction(made, r1)
    rem = T - made
    t2 = Fraction(rem, r2)
    unit = rng.choice(["件", "吨"])
    ins = rng.choice([
        f"一批货物共{T}{unit}，先生产1/{d}，每天生产{r1}{unit}，剩下的每天生产{r2}{unit}，还要多少天完成？",
        f"修一条长{T}米的路，先修了1/{d}，每天修{r1}米，余下的每天修{r2}米，还需多少天？",
        f"工厂要加工{T}{unit}产品，先完成1/{d}，每天加工{r1}{unit}，剩下的每天加工{r2}{unit}，还要几天？",
    ])
    lines = [
        f"{T} × 1/{d} = {num(made)}{unit}",
        f"{num(made)} ÷ {r1} = {num(t1)}天",
        f"{T} - {num(made)} = {num(rem)}{unit}",
        f"{num(rem)} ÷ {r2} = {num(t2)}天",
    ]
    return ins, lines, t2


_reg("rate_two_phase", rate_two_phase)


# 50. three pipes together -> fill time
def pipes_three(rng):
    a = rng.randint(3, 10)
    b = rng.randint(3, 10)
    c = rng.randint(3, 10)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    rc = Fraction(1, c)
    rate = ra + rb + rc
    t = Fraction(1, rate)
    ins = rng.choice([
        f"一个水池有三个进水管，单开甲管{a}小时注满，单开乙管{b}小时注满，单开丙管{c}小时注满，三管齐开多少小时注满？",
        f"三个水管注水，甲管单独{a}小时注满水池，乙管{b}小时，丙管{c}小时，同时开多少小时注满？",
        f"蓄水池有甲、乙、丙三个进水管，甲管{a}小时注满，乙管{b}小时注满，丙管{c}小时注满，三管同开几小时注满？",
    ])
    lines = [
        f"甲管效率 = 1 ÷ {a} = {num(ra)}",
        f"乙管效率 = 1 ÷ {b} = {num(rb)}",
        f"丙管效率 = 1 ÷ {c} = {num(rc)}",
        f"总效率 = {num(ra)} + {num(rb)} + {num(rc)} = {num(rate)}",
        f"1 ÷ ({num(rate)}) = {num(t)}小时",
    ]
    return ins, lines, t


_reg("pipes_three", pipes_three)


# 51. fill pipe + drain pipe -> fill time
def pipe_fill_drain(rng):
    a = rng.randint(3, 8)
    b = rng.randint(a + 2, a + 6)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    net = ra - rb
    t = Fraction(1, net)
    ins = rng.choice([
        f"一个水池，进水管{a}小时注满，排水管{b}小时排空，两管同开多少小时注满？",
        f"蓄水池有一个进水管和一个出水管，进水管{a}小时注满，出水管{b}小时放完，同时打开多少小时注满？",
        f"一个水箱，甲管{a}小时注满，乙管{b}小时把水排完，两管齐开多少小时注满？",
    ])
    lines = [
        f"进水管效率 = 1 ÷ {a} = {num(ra)}",
        f"排水管效率 = 1 ÷ {b} = {num(rb)}",
        f"净效率 = {num(ra)} - {num(rb)} = {num(net)}",
        f"1 ÷ ({num(net)}) = {num(t)}小时",
    ]
    return ins, lines, t


_reg("pipe_fill_drain", pipe_fill_drain)


# 52. map scale + speed -> travel time
def map_scale_speed(rng):
    m = rng.randint(2, 10)
    k = rng.choice([1000000, 2000000, 4000000, 5000000])
    v = rng.randint(40, 100)
    cm = m * k
    km = Fraction(cm, 100000)
    t = Fraction(km, v)
    ins = rng.choice([
        f"在比例尺1:{k}的地图上，量得两地距离{m}厘米，一辆汽车每小时行{v}千米，需要多少小时？",
        f"地图的比例尺是1:{k}，量得甲、乙两地相距{m}厘米，汽车以每小时{v}千米的速度行驶，几小时到达？",
        f"一幅地图比例尺为1:{k}，图上两地距离{m}厘米，骑车每小时行{v}千米，要多少小时？",
    ])
    lines = [
        f"{m} × {k} = {cm}厘米",
        f"{cm} ÷ 100000 = {num(km)}千米",
        f"{num(km)} ÷ {v} = {num(t)}小时",
    ]
    return ins, lines, t


_reg("map_scale_speed", map_scale_speed)


# 53. scale -> real area
def scale_area(rng):
    l = rng.randint(2, 8)
    w = rng.randint(2, 8)
    k = rng.choice([100, 200, 500, 1000])
    L = l * k
    W = w * k
    area = L * W
    m2 = Fraction(area, 10000)
    ins = rng.choice([
        f"在比例尺1:{k}的图纸上，长方形操场长{l}厘米、宽{w}厘米，操场实际面积是多少平方米？",
        f"图纸比例尺1:{k}，量得一块长方形地长{l}厘米、宽{w}厘米，实际面积多少平方米？",
        f"一幅图的比例尺是1:{k}，图上长方形长{l}厘米、宽{w}厘米，实际面积是多少平方米？",
    ])
    lines = [
        f"{l} × {k} = {L}厘米",
        f"{w} × {k} = {W}厘米",
        f"{L} × {W} = {area}平方厘米",
        f"{area} ÷ 10000 = {num(m2)}平方米",
    ]
    return ins, lines, m2


_reg("scale_area", scale_area)


# 54. cost -> markup -> discount -> profit
def markup_discount_profit(rng):
    C = rng.randint(5, 30) * 10
    p = rng.choice([30, 40, 50, 60])
    d = rng.choice([8, 9])
    markup = Fraction(C * p, 100)
    price = C + markup
    sell = Fraction(price * d, 10)
    profit = sell - C
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}的成本是{C}元，按成本加价{p}%定价，再打{d}折出售，实际赚了多少元？",
        f"某商品成本{C}元，先加价{p}%标价，再按标价打{d}折卖出，利润是多少元？",
        f"一台电器成本{C}元，商家加价{p}%定价，促销时打{d}折，每件赚多少元？",
    ])
    lines = [
        f"{C} × {p}/100 = {num(markup)}元",
        f"{C} + {num(markup)} = {num(price)}元",
        f"{num(price)} × {d}/10 = {num(sell)}元",
        f"{num(sell)} - {C} = {num(profit)}元",
    ]
    return ins, lines, profit


_reg("markup_discount_profit", markup_discount_profit)


# 55. price + tax -> per item -> total for q items
def tax_qty_total(rng):
    P = rng.randint(10, 60)
    p = rng.choice([3, 5, 6, 9, 13])
    q = rng.randint(2, 8)
    tax = Fraction(P * p, 100)
    each = P + tax
    total = each * q
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}售价{P}元，另按{p}%缴纳消费税，买{q}件一共要付多少元？",
        f"某商品单价{P}元，税率{p}%，买{q}件含税总价是多少元？",
        f"一本书定价{P}元，按{p}%收税，买{q}本一共需要多少元？",
    ])
    lines = [
        f"{P} × {p}/100 = {num(tax)}元",
        f"{P} + {num(tax)} = {num(each)}元",
        f"{num(each)} × {q} = {num(total)}元",
    ]
    return ins, lines, total


_reg("tax_qty_total", tax_qty_total)


# 56. tiered commission
def commission_tiered(rng):
    S1 = rng.randint(10, 50) * 100
    S = S1 + rng.randint(5, 30) * 100
    p1 = rng.choice([1, 2, 3])
    p2 = rng.choice([5, 8, 10])
    e1 = Fraction(S1 * p1, 100)
    rest = S - S1
    e2 = Fraction(rest * p2, 100)
    earn = e1 + e2
    ins = rng.choice([
        f"销售员的提成规则：销售额{S1}元以内按{p1}%提成，超过部分按{p2}%提成，本月销售额{S}元，提成共多少元？",
        f"保险佣金：保费{S1}元以内按{p1}%，超出部分按{p2}%，共收保费{S}元，佣金多少元？",
        f"中介佣金按分段计算：{S1}元以内收{p1}%，以上部分收{p2}%，一笔成交额{S}元，佣金共多少元？",
    ])
    lines = [
        f"{S1} × {p1}/100 = {num(e1)}元",
        f"{S} - {S1} = {rest}元",
        f"{rest} × {p2}/100 = {num(e2)}元",
        f"{num(e1)} + {num(e2)} = {num(earn)}元",
    ]
    return ins, lines, earn


_reg("commission_tiered", commission_tiered)


# 57. fractional quantities at integer prices -> change
def shopping_frac_qty(rng):
    q1 = rng.choice([Fraction(3, 2), Fraction(5, 2), Fraction(1, 2), Fraction(1, 4)])
    q2 = rng.choice([Fraction(1, 2), Fraction(3, 2), Fraction(3, 4)])
    p1 = rng.randint(4, 16)
    p2 = rng.randint(4, 16)
    c1 = p1 * q1
    c2 = p2 * q2
    total = c1 + c2
    change = 100 - total
    fruit = rng.choice(FRUITS)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}买了{num(q1)}千克每千克{p1}元的{fruit}，又买了{num(q2)}千克每千克{p2}元的香蕉，付给100元，找回多少元？",
        f"妈妈买{num(q1)}千克{p1}元的{fruit}和{num(q2)}千克{p2}元的橘子，付100元，应找回多少元？",
        f"买{num(q1)}千克每千克{p1}元的{fruit}，{num(q2)}千克每千克{p2}元的葡萄，给100元，找回多少元？",
    ])
    lines = [
        f"{p1} × {num(q1)} = {num(c1)}元",
        f"{p2} × {num(q2)} = {num(c2)}元",
        f"{num(c1)} + {num(c2)} = {num(total)}元",
        f"100 - {num(total)} = {num(change)}元",
    ]
    return ins, lines, change


_reg("shopping_frac_qty", shopping_frac_qty)


# 58. rope cut into three pieces, two fractions of the whole
def rope_three_pieces(rng):
    L = rng.randint(12, 40)
    d = rng.choice([3, 4, 5, 6])
    e = rng.choice([3, 4, 5, 6])
    for _ in range(50):
        if Fraction(1, d) + Fraction(1, e) < 1:
            break
        e = rng.choice([3, 4, 5, 6])
    p1 = Fraction(L, d)
    p2 = Fraction(L, e)
    p3 = L - p1 - p2
    ins = rng.choice([
        f"一根绳子长{L}米，第一次剪去全长的1/{d}，第二次剪去全长的1/{e}，剩下的第三段长多少米？",
        f"一条彩带长{L}米，做蝴蝶结用去1/{d}，包装礼物用去1/{e}，还剩多少米？",
        f"一根铁丝长{L}米，第一次截去1/{d}，第二次截去1/{e}，还剩多少米？",
    ])
    lines = [
        f"{L} × 1/{d} = {num(p1)}米",
        f"{L} × 1/{e} = {num(p2)}米",
        f"{L} - {num(p1)} - {num(p2)} = {num(p3)}米",
    ]
    return ins, lines, p3


_reg("rope_three_pieces", rope_three_pieces)


# 59. age ratio + difference -> age in t years
def age_ratio_future(rng):
    a = rng.choice([4, 5, 6, 7])
    b = rng.choice([2, 3])
    d = a - b
    D = d * rng.randint(2, 6)
    per = D // d
    son = b * per
    t = rng.choice([2, 3, 4, 5, 6, 10])
    future = son + t
    ins = rng.choice([
        f"今年父亲与儿子的年龄比是{a}:{b}，父亲比儿子大{D}岁，{t}年后儿子多少岁？",
        f"爸爸和小明的年龄比是{a}:{b}，爸爸比小明大{D}岁，{t}年后小明多少岁？",
        f"今年母子年龄比是{a}:{b}，母亲比儿子大{D}岁，{t}年后儿子多少岁？",
    ])
    lines = [
        f"份数差 = {a} - {b} = {d}",
        f"{D} ÷ {d} = {per}岁",
        f"{b} × {per} = {son}岁",
        f"{son} + {t} = {future}岁",
    ]
    return ins, lines, future


_reg("age_ratio_future", age_ratio_future)


# 60. alloy ratio split + add metal -> new amount
def alloy_add(rng):
    M = rng.randint(6, 20)
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    x = rng.randint(1, 5)
    s = a + b
    per = Fraction(M, s)
    cu = a * per
    newcu = cu + x
    ins = rng.choice([
        f"一块合金重{M}千克，其中铜和锌的比是{a}:{b}，再加入{x}千克铜，新合金中铜有多少千克？",
        f"有{M}千克合金，铜与锌的比为{a}:{b}，熔入{x}千克铜后，铜有多少千克？",
        f"一块铜锌合金重{M}千克，铜和锌的比是{a}:{b}，加入{x}千克铜，现在铜有多少千克？",
    ])
    lines = [
        f"总份数 = {a} + {b} = {s}",
        f"{M} ÷ {s} = {num(per)}千克",
        f"{a} × {num(per)} = {num(cu)}千克",
        f"{num(cu)} + {x} = {num(newcu)}千克",
    ]
    return ins, lines, newcu


_reg("alloy_add", alloy_add)


# 61. fractions of two different wholes, summed
def frac_two_wholes_sum(rng):
    A = rng.randint(20, 80)
    B = rng.randint(20, 80)
    d = rng.choice([2, 3, 4])
    e = rng.choice([2, 3, 4])
    x = Fraction(A, d)
    y = Fraction(B, e)
    s = x + y
    unit = rng.choice(["元", "本", "千克"])
    ins = rng.choice([
        f"甲有{A}{unit}，乙有{B}{unit}，甲拿出自己的1/{d}，乙拿出自己的1/{e}，两人一共拿出多少{unit}？",
        f"一班捐书{A}本，二班捐书{B}本，一班捐出1/{d}，二班捐出1/{e}，两班共捐出多少本？",
        f"甲仓有{A}{unit}粮食，乙仓有{B}{unit}，甲仓运出1/{d}，乙仓运出1/{e}，一共运出多少{unit}？",
    ])
    lines = [
        f"{A} × 1/{d} = {num(x)}{unit}",
        f"{B} × 1/{e} = {num(y)}{unit}",
        f"{num(x)} + {num(y)} = {num(s)}{unit}",
    ]
    return ins, lines, s


_reg("frac_two_wholes_sum", frac_two_wholes_sum)


# 62. percent of a percent -> leftover
def pct_of_pct_left(rng):
    T = rng.randint(20, 80) * 10
    p = rng.choice([10, 20, 25])
    q = rng.choice([10, 20, 25, 50])
    first = Fraction(T * p, 100)
    amount = Fraction(first * q, 100)
    left = first - amount
    unit = rng.choice(["元", "吨", "本"])
    ins = rng.choice([
        f"一笔钱共{T}{unit}，先拿出{p}%，再从拿出的钱中用掉{q}%，拿出的钱还剩多少{unit}？",
        f"仓库有{T}{unit}货物，先运走{p}%，再从运走的货物中卖掉{q}%，运走的货物还剩多少{unit}？",
        f"学校有经费{T}{unit}，先拨出{p}%，又从拨出的钱中花掉{q}%，拨出的钱还剩多少{unit}？",
    ])
    lines = [
        f"{T} × {p}/100 = {num(first)}{unit}",
        f"{num(first)} × {q}/100 = {num(amount)}{unit}",
        f"{num(first)} - {num(amount)} = {num(left)}{unit}",
    ]
    return ins, lines, left


_reg("pct_of_pct_left", pct_of_pct_left)


# 63. meet with delayed start -> time after the second starts
def meet_delayed(rng):
    D = rng.randint(10, 60) * 10
    v1 = rng.randint(40, 90)
    v2 = rng.randint(40, 90)
    t0 = rng.choice([1, 2])
    head = v1 * t0
    rem = D - head
    s = v1 + v2
    t = Fraction(rem, s)
    ins = rng.choice([
        f"甲、乙两车从相距{D}千米的两地相向而行，甲车每小时行{v1}千米，先出发{t0}小时后乙车以每小时{v2}千米出发，乙车出发后多少小时相遇？",
        f"两地相距{D}千米，甲每小时走{v1}千米，先走{t0}小时后乙从对面以每小时{v2}千米走来，乙出发后多少小时相遇？",
        f"A、B两地相距{D}千米，慢车每小时行{v1}千米，先开{t0}小时后快车从B地以每小时{v2}千米相向开出，快车出发后多少小时相遇？",
    ])
    lines = [
        f"{v1} × {t0} = {head}千米",
        f"{D} - {head} = {rem}千米",
        f"{v1} + {v2} = {s}千米/时",
        f"{rem} ÷ {s} = {num(t)}小时",
    ]
    return ins, lines, t


_reg("meet_delayed", meet_delayed)


# 64. same distance, two speeds -> arrival time gap
def arrival_gap(rng):
    D = rng.randint(6, 30) * 10
    v1 = rng.randint(30, 60)
    v2 = rng.randint(v1 + 10, 100)
    t1 = Fraction(D, v1)
    t2 = Fraction(D, v2)
    diff = t1 - t2
    ins = rng.choice([
        f"甲、乙两地相距{D}千米，慢车每小时行{v1}千米，快车每小时行{v2}千米，两车同时从甲地出发，快车比慢车早到多少小时？",
        f"小明家到学校{D}千米，步行每小时{v1}千米，骑车每小时{v2}千米，骑车比步行早到多少小时？",
        f"一艘船航行{D}千米，顺水每小时{v2}千米，逆水每小时{v1}千米，逆水比顺水多用多少小时？",
    ])
    lines = [
        f"{D} ÷ {v1} = {num(t1)}小时",
        f"{D} ÷ {v2} = {num(t2)}小时",
        f"{num(t1)} - {num(t2)} = {num(diff)}小时",
    ]
    return ins, lines, diff


_reg("arrival_gap", arrival_gap)


# 65. two percent increases, summed
def two_pct_increases(rng):
    M = rng.randint(10, 60) * 10
    N = rng.randint(10, 60) * 10
    p = rng.choice([5, 10, 15, 20])
    q = rng.choice([5, 10, 15, 20])
    i1 = Fraction(M * p, 100)
    i2 = Fraction(N * q, 100)
    inc = i1 + i2
    ins = rng.choice([
        f"甲店上月营业额{M}元，本月增加{p}%；乙店上月营业额{N}元，本月增加{q}%，两店营业额一共增加多少元？",
        f"甲厂上月生产{M}件，本月增产{p}%；乙厂上月生产{N}件，本月增产{q}%，两厂一共增产多少件？",
        f"甲仓有粮{M}吨，今年增产{p}%；乙仓有粮{N}吨，今年增产{q}%，两仓一共增产多少吨？",
    ])
    lines = [
        f"甲增加 = {M} × {p}/100 = {num(i1)}",
        f"乙增加 = {N} × {q}/100 = {num(i2)}",
        f"一共增加 = {num(i1)} + {num(i2)} = {num(inc)}",
    ]
    return ins, lines, inc


_reg("two_pct_increases", two_pct_increases)


# 66. two travel legs at different speeds -> average speed
def two_leg_avg(rng):
    v1 = rng.randint(30, 60)
    v2 = rng.randint(60, 100)
    t1 = rng.randint(1, 4)
    t2 = rng.randint(1, 4)
    d1 = v1 * t1
    d2 = v2 * t2
    total = d1 + d2
    tt = t1 + t2
    avg = Fraction(total, tt)
    ins = rng.choice([
        f"一辆汽车先以每小时{v1}千米的速度行驶{t1}小时，又以每小时{v2}千米的速度行驶{t2}小时，全程平均速度是多少千米/时？",
        f"小明骑车前{t1}小时每小时行{v1}千米，后{t2}小时每小时行{v2}千米，平均每小时行多少千米？",
        f"一列火车先以每小时{v1}千米的速度开了{t1}小时，再以每小时{v2}千米的速度开了{t2}小时，平均速度是多少千米/时？",
    ])
    lines = [
        f"{v1} × {t1} = {d1}千米",
        f"{v2} × {t2} = {d2}千米",
        f"{d1} + {d2} = {total}千米",
        f"{t1} + {t2} = {tt}小时",
        f"{total} ÷ {tt} = {num(avg)}千米/时",
    ]
    return ins, lines, avg


_reg("two_leg_avg", two_leg_avg)


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
    print(f"L3 ext2 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
