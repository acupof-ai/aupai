#!/usr/bin/env python3
"""L2 programs: 3-4 arithmetic steps; combine, convert units, remainders, average."""
import random
from fractions import Fraction
from mathcommon import GOODS, NAMES, num

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L2", name, fn))


# two-item money total
def two_buy(rng):
    q1, q2 = rng.randint(2, 8), rng.randint(2, 8)
    p1, p2 = rng.randint(2, 12), rng.randint(2, 12)
    o1, o2 = rng.sample(GOODS, 2)
    n1 = rng.choice(NAMES)
    c1, c2 = q1 * p1, q2 * p2
    s = c1 + c2
    t = rng.randrange(3)
    ins = [
        f"{n1}买了{q1}个{p1}元的{o1}，又买了{q2}个{p2}元的{o2}，一共多少钱？",
        f"{q1}个{o1}每个{p1}元，{q2}个{o2}每个{p2}元，{n1}共花了多少元？",
        f"{n1}购物：{o1}买了{q1}个（{p1}元/个），{o2}买了{q2}个（{p2}元/个）。共付多少？",
    ][t]
    lines = [
        f"第一种总价 = {q1} × {p1} = {c1}元",
        f"第二种总价 = {q2} × {p2} = {c2}元",
        f"一共 = {c1} + {c2} = {s}元",
    ]
    return ins, lines, s


_reg("two_buy", two_buy)


# unit conversion: 时↔分 / 米↔厘米 / 千克↔克
def unit_conv(rng):
    a = rng.randrange(3)
    if a == 0:
        h = rng.randint(1, 12)
        ins = f"{h}小时等于多少分钟？"
        lines = [f"分钟数 = {h} × 60 = {h * 60}分"]
        return ins, lines, h * 60
    if a == 1:
        m = rng.randint(1, 9) * 10
        ins = f"{m}厘米等于多少米？（1米=100厘米）"
        lines = [f"米数 = {m} ÷ 100 = {num(Fraction(m, 100))}米"]
        return ins, lines, Fraction(m, 100)
    kg = rng.randint(1, 9) * 100
    ins = f"{kg}克等于多少千克？（1千克=1000克）"
    lines = [f"千克数 = {kg} ÷ 1000 = {num(Fraction(kg, 1000))}千克"]
    return ins, lines, Fraction(kg, 1000)


_reg("unit_conv", unit_conv)


# remainder: n books into boxes of k
def rem_boxes(rng):
    k = rng.randint(3, 9)
    full = rng.randint(3, 8)
    left = rng.randint(1, k - 1)
    n = k * full + left
    lines = [
        f"装满的 = {full} × {k} = {k * full}本",
        f"剩下 = {n} - {k * full} = {left}本",
    ]
    ins = f"有{n}本书，每{k}本装一盒，装满{full}盒后还剩几本？"
    return ins, lines, left  # boxed = 剩余


_reg("rem_boxes", rem_boxes)


# average of k scores (integral by construction)
def average_of(rng):
    k = rng.randint(3, 5)
    vals = [rng.randint(30, 100) for _ in range(k - 1)]
    last = rng.randint(30, 100)
    for _ in range(50):
        if (sum(vals) + last) % k == 0:
            break
        last = rng.randint(30, 100)
    vals.append(last)
    s = sum(vals)
    avg = s // k
    n1 = rng.choice(NAMES)
    ins = f"{n1}{k}次测试成绩分别是{'、'.join(map(str, vals))}分，平均分是多少？"
    lines = [
        f"总分 = {' + '.join(map(str, vals))} = {s}分",
        f"平均分 = {s} ÷ {k} = {avg}分",
    ]
    return ins, lines, avg


_reg("average_of", average_of)


# rectangle perimeter
def perimeter(rng):
    a, b = rng.randint(5, 40), rng.randint(5, 40)
    p2 = 2 * (a + b)
    t = rng.randrange(2)
    ins = [
        f"一个长方形长{a}米、宽{b}米，周长是多少米？",
        f"长{b}米、宽{a}米的长方形，周长几米？",
    ][t]
    lines = [
        f"长加宽 = {a} + {b} = {a + b}米",
        f"周长 = {a + b} × 2 = {p2}米",
    ]
    return ins, lines, p2


_reg("perimeter", perimeter)


# simple distance = speed × time, then two legs sum (or convert)
def distance_two(rng):
    v1, t1 = rng.randint(2, 30), rng.randint(1, 5)
    v2, t2 = rng.randint(2, 30), rng.randint(1, 5)
    d1, d2 = v1 * t1, v2 * t2
    d = d1 + d2
    n1 = rng.choice(NAMES)
    ins = (f"{n1}先以时速{v1}千米骑了{t1}小时，再以时速{v2}千米骑了{t2}小时，"
           f"一共骑了多少千米？")
    lines = [
        f"第一段 = {v1} × {t1} = {d1}千米",
        f"第二段 = {v2} × {t2} = {d2}千米",
        f"总路程 = {d1} + {d2} = {d}千米",
    ]
    return ins, lines, d


_reg("distance_two", distance_two)


# buy with change: total - give = change + items
def buy_change(rng):
    q1, p1 = rng.randint(2, 5), rng.randint(3, 12)
    q2, p2 = rng.randint(1, 4), rng.randint(2, 10)
    c1, c2 = q1 * p1, q2 * p2
    s = c1 + c2
    while s >= 80:
        q1, q2 = rng.randint(2, 4), rng.randint(1, 3)
        c1, c2 = q1 * p1, q2 * p2
        s = c1 + c2
    pay = 50 if s < 40 else 100
    change = pay - s
    o1, o2 = rng.sample(GOODS, 2)
    n1 = rng.choice(NAMES)
    ins = f"{n1}买了{q1}个{p1}元的{o1}和{q2}个{p2}元的{o2}，付{pay}元，应找回多少？"
    lines = [
        f"第一项 = {q1} × {p1} = {c1}元",
        f"第二项 = {q2} × {p2} = {c2}元",
        f"总价 = {c1} + {c2} = {s}元",
        f"找零 = {pay} - {s} = {change}元",
    ]
    return ins, lines, change


_reg("buy_change", buy_change)


# three addends
def add_three(rng):
    a, b, c = (rng.randint(8, 60) for _ in range(3))
    obj = rng.choice(["人", "本", "个"])
    t = rng.randrange(2)
    n1 = rng.choice(NAMES)
    unit = "片" if t == 1 else obj
    ins = [
        f"三批物资分别为{a}{obj}、{b}{obj}、{c}{obj}，一共多少{obj}？",
        f"{n1}三天分别收集了{a}、{b}、{c}片树叶，共多少片？",
    ][t]
    ab, s = a + b, a + b + c
    lines = [
        f"前两天 = {a} + {b} = {ab}{unit}",
        f"一共 = {ab} + {c} = {s}{unit}",
    ]
    return ins, lines, s


_reg("add_three", add_three)


# age: now = younger + diff ; then + years
def age_math(rng):
    young = rng.randint(5, 15)
    diff = rng.randint(20, 35)
    now_older = young + diff
    yrs = rng.randint(2, 10)
    t = rng.randrange(2)
    who = "爸爸" if t == 0 else "哥哥"
    ins = [
        f"小明{young}岁，爸爸比小明大{diff}岁。{yrs}年后爸爸几岁？",
        f"弟弟{young}岁，哥哥比弟弟大{diff}岁，{yrs}年后哥哥多少岁？",
    ][t]
    older_now = young + diff
    lines = [
        f"{who}今年 = {young} + {diff} = {older_now}岁",
        f"{yrs}年后 = {older_now} + {yrs} = {older_now + yrs}岁",
    ]
    return ins, lines, older_now + yrs


_reg("age_math", age_math)


if __name__ == "__main__":
    rng = random.Random(2)
    from run_math_short import verify
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines}"
            ok += 1
    print(f"L2 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")