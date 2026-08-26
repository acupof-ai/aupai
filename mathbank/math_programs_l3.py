#!/usr/bin/env python3
"""L3 programs: 5-7 steps; multi-entity, ratio, percent, fractions, meet, area."""
import random
from fractions import Fraction
from mathcommon import GOODS, NAMES, num, pct

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L3", name, fn))


# percent then fraction then combine
def percent_frac(rng):
    base = rng.randint(50, 300) * 10
    pr = rng.randint(10, 40)
    inc = Fraction(base * pr, 100)
    after = base + inc
    den = rng.choice([2, 3, 4])
    left = after - Fraction(after, den)
    obj = rng.choice(["美元", "吨", "毫升", "名学生"])
    t = rng.randrange(2)
    ins = [
        f"仓库原有{base}{obj}，运进{pr}%后又用掉其中的1/{den}，还剩多少{obj}？",
        f"一件商品原价{base}元，先涨价{pr}%，再打{den}分之1的折扣，现价多少？",
    ][t] if t == 0 else f"原有{base}{obj}，先增加{pr}%，再用去1/{den}，剩余多少？"
    lines = [
        f"{base} × {pr/100} = {num(inc)}{obj}",
        f"{base} + {num(inc)} = {num(after)}{obj}",
        f"{num(after)} × {1}/{den} = {num(Fraction(after, den))}{obj}",
        f"{num(after)} - {num(Fraction(after, den))} = {num(left)}{obj}",
    ]
    return ins, lines, left


_reg("percent_frac", percent_frac)


# meet/catch: distance ÷ speed-sum → time, then fractions of that
def meet_time(rng):
    d = rng.randint(3, 40) * 10
    v1, v2 = rng.randint(2, 9) * 10, rng.randint(2, 9) * 10
    s = v1 + v2
    t = Fraction(d, s)
    lines = [
        f"{v1} + {v2} = {s}米/分",
        f"{d} ÷ {s} = {num(t)}分",
    ]
    ins = f"A、B两地相距{d}米，甲每分钟走{v1}米，乙每分钟走{v2}米，同时相向而行，几分钟相遇？"
    return ins, lines, t


_reg("meet_time", meet_time)


# ratio → per share → one part
def ratio_split(rng):
    a, b = rng.randint(2, 8), rng.randint(2, 8)
    while a == b:
        b = rng.randint(2, 8)
    total = (a + b) * rng.randint(4, 30)
    per = total // (a + b)
    A, B = a * per, b * per
    who = rng.choice(["甲", "乙"])
    k = a if who == "甲" else b
    v = A if who == "甲" else B
    ins = f"甲、乙两数的比是{a}:{b}，它们的和是{total}。{who}是多少？"
    lines = [
        f"每份 = {total} ÷ ({a}+{b}) = {per}",
        f"{who}数 = {k} × {per} = {v}",
    ]
    return ins, lines, v


_reg("ratio_split", ratio_split)


# fraction of a whole, two stages
def frac_two(rng):
    total = rng.randint(120, 500) * 10
    d1 = rng.choice([2, 3, 4])
    first = Fraction(total, d1)
    rem = total - first
    den = rng.choice([2, 3])
    second = Fraction(rem, den)
    obj = rng.choice(["棵树", "本书", "升水", "个零件"])
    ins = f"仓库有{total}{obj}，第一天用去1/{d1}，第二天用去剩余部分的1/{den}，第二天用去多少？"
    lines = [
        f"{total} × {1}/{d1} = {num(first)}{obj}",
        f"{total} - {num(first)} = {num(rem)}{obj}",
        f"{num(rem)} × {1}/{den} = {num(second)}{obj}",
    ]
    return ins, lines, second


_reg("frac_two", frac_two)


# area of a room from length×width then cost per unit → total cost (5-6 steps)
def area_cost(rng):
    w = rng.randint(4, 12)
    rng_len = rng.randint(2, 4)
    l = w + rng_len
    a = l * w
    per = rng.randint(15, 60)
    cost = a * per
    ins = f"一间教室长{l}米、宽{w}米，铺每平方米{per}元的地砖，一共需要多少元？"
    lines = [
        f"{l} × {w} = {a}平方米",
        f"{a} × {per} = {cost}元",
    ]
    return ins, lines, cost


_reg("area_cost", area_cost)


# buy N items total then split cost evenly
def group_pay(rng):
    people = rng.randint(3, 6)
    q, p = rng.randint(2, 6), rng.randint(3, 20)
    total = q * p
    each = Fraction(total, people)
    obj = rng.choice(GOODS)
    ins = f"{people}个同学合买{q}份{p}元的{obj}，平均每人付多少元？"
    lines = [
        f"{q} × {p} = {total}元",
        f"{total} ÷ {people} = {num(each)}元",
    ]
    return ins, lines, each


_reg("group_pay", group_pay)


# percent: discount stack (原价→九折→再减)
def discount_chain(rng):
    price = rng.randint(200, 900) * 10
    d1 = rng.choice([8, 9])          # 八折 / 九折
    d2 = rng.choice([8, 9])
    a1 = Fraction(price * d1, 10)
    a2 = Fraction(a1 * d2, 10)
    ins = f"一台笔记本原价{price}元，先打{d1}折，再打{d2}折，现在多少元？"
    lines = [
        f"{price} × {d1}/10 = {num(a1)}元",
        f"{num(a1)} × {d2}/10 = {num(a2)}元",
    ]
    return ins, lines, a2


_reg("discount_chain", discount_chain)


# production rate × time, then percent target
def output_target(rng):
    rate = rng.randint(30, 80) * 10
    days = rng.randint(3, 7)
    made = rate * days
    target = Fraction(made * 100, rng.randint(40, 70))
    obj = rng.choice(["件", "吨"])
    ins = f"工厂每天生产{rate}{obj}，{days}天后已知完成全部计划的{rng.randint(40, 70)}%，这批货共多少{obj}？"
    frac_plan = None
    # rebuild with known pct for exact lines:
    p0 = rng.randint(40, 70)
    target = Fraction(made * 100, p0)
    ins = f"工厂每天生产{rate}{obj}，{days}天共生产了全部计划的{p0}%，全部计划是多少{obj}？"
    lines = [
        f"{rate} × {days} = {made}{obj}",
        f"{made} ÷ {p0/100} = {num(target)}{obj}",
    ]
    return ins, lines, target


_reg("output_target", output_target)


# fraction to whole (已知部分求整体) reverse
def part_to_whole(rng):
    den = rng.choice([3, 4, 5])
    whole = rng.randint(15, 60) * den
    part = whole // den
    obj = rng.choice(["学生", "棵树", "本书"])
    ins = f"一个班{obj}中有{part}人是合唱队，正好占全班的1/{den}，全班多少人？"
    lines = [
        f"{part} × {den} = {whole}人",
    ]
    return ins, lines, whole


_reg("part_to_whole", part_to_whole)


# speed with unit conversion: km/h → m/min then distance
def speed_unit(rng):
    kmh = rng.randint(3, 9)
    t_h = rng.choice([Fraction(1, 2), Fraction(2, 3), Fraction(1, 4), Fraction(3, 4)])
    d = kmh * t_h
    ins = f"骑车每小时走{kmh}千米，骑{t_h}小时，走了多少千米？"
    lines = [
        f"{kmh} × {num(t_h) if t_h.denominator != 1 else t_h} = {num(d)}千米",
    ]
    return ins, lines, d


_reg("speed_unit", speed_unit)


# multi-step: total budget - two fixed + share rest
def budget_rest(rng):
    total = rng.randint(300, 900) * 10
    f1, f2 = total // rng.randint(5, 9), total // rng.randint(6, 10)
    used = f1 + f2
    rest = total - used
    each = Fraction(rest, rng.randint(2, 4))
    ins = f"一笔{total}元的预算，设备用了{f1}元，人工用了{f2}元，剩余的分给{2}个小组，每组多少？"
    g = rng.randint(2, 4)
    each = Fraction(rest, g)
    ins = f"一笔{total}元的预算，设备用了{f1}元，人工用了{f2}元，剩余的分给{g}个小组，每组多少元？"
    lines = [
        f"{total} - {f1} = {total - f1}元",
        f"{total - f1} - {f2} = {rest}元",
        f"{rest} ÷ {g} = {num(each)}元",
    ]
    return ins, lines, each


_reg("budget_rest", budget_rest)


if __name__ == "__main__":
    rng = random.Random(3)
    from run_math_short import verify
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines}"
            ok += 1
    print(f"L3 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")