#!/usr/bin/env python3
"""L1 programs: single operation (1-2 arithmetic steps). ~15 families.

Each fun(rng) -> (instruction, lines[list[str]], ans:int|Fraction). At least 3
sentence phrasings per family; names/objects from mathcommon pools; L1 ranges.
Every line is `X op Y = Z`; last line's value must equal num(ans) (verified).
"""

import sys
from fractions import Fraction
from mathcommon import (
    ANIMALS, FOOD, GOODS, NAMES, STATIONERY, UNIT_N, UNIT_ZHI,
    num,
)

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L1", name, fn))


# 1. buy_one: q × price = total ----------------------------------------------
def buy_one(rng):
    q = rng.randint(2, 9)
    p = rng.choice([2, 3, 5, 8]) * rng.randint(1, 4)
    obj = rng.choice(GOODS)
    unit = rng.choice([UNIT_N, UNIT_ZHI])
    n = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n}买了{q}{unit}{obj}，每个{p}元，一共花了多少钱？",
        f"{obj}每个{p}元，{n}一共买了{q}{unit}，总共多少元？",
        f"{n}买了{p}元一个的{obj}{q}{unit}，一共要付多少元？",
    ][t]
    lines = [f"总价 = {q} × {p} = {num(q * p)}元"]
    return ins, lines, q * p


_reg("buy_one", buy_one)


# 2. sum_two: a + b = total --------------------------------------------------
def sum_two(rng):
    a, b = rng.randint(10, 80), rng.randint(10, 80)
    obj = rng.choice(["人", "本书", "个球", "棵树"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{a}{obj}，{n2}有{b}{obj}，两人共有多少？",
        f"第一队{a}{obj}，第二队{b}{obj}，一共多少{obj}？",
        f"上午来了{a}{obj}，下午来了{b}{obj}，全天共来多少？",
    ][t]
    lines = [f"一共 = {a} + {b} = {a + b}{obj}"]
    return ins, lines, a + b


_reg("sum_two", sum_two)


# 3. left_after: n - x = left ------------------------------------------------
def left_after(rng):
    n = rng.randint(15, 90)
    x = rng.randint(3, n - 3)
    obj = rng.choice(["块糖", "本书", "个苹果", "枝铅笔", "张纸"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{n}{obj}，吃掉了{x}{obj}，还剩多少？",
        f"盒子里有{n}{obj}，拿出{x}{obj}后剩多少？",
        f"原有{n}{obj}，分走了{x}{obj}，剩余几个？",
    ][t]
    lines = [f"剩下 = {n} - {x} = {n - x}{obj}"]
    return ins, lines, n - x


_reg("left_after", left_after)


# 4. split_even: n ÷ k = each (exact) ---------------------------------------
def split_even(rng):
    k = rng.randint(2, 6)
    each = rng.randint(2, 12)
    n = k * each
    obj = rng.choice(["块饼干", "个面包", "本练习册", "颗糖", "只蛋"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n}{obj}平均分给{k}个小朋友，每人几个？",
        f"{k}个小朋友平分{n}{obj}，每人各得多少？",
        f"把{n}{obj}均分成{k}份且每份相等，每份是多少？",
    ][t]
    lines = [f"每份 = {n} ÷ {k} = {each}{obj}"]
    return ins, lines, each


_reg("split_even", split_even)


# 5. price_each: t ÷ q = unit price ------------------------------------------
def price_each(rng):
    q = rng.randint(2, 8)
    p = rng.randint(3, 15)
    t = q * p
    obj = rng.choice(GOODS)
    unit = rng.choice([UNIT_N, UNIT_ZHI])
    n = rng.choice(NAMES)
    t_ = rng.randrange(3)
    ins = [
        f"{n}花{t}元买了{q}{unit}{obj}，每个多少元？",
        f"{obj}共{t}元，共{q}{unit}，单价是多少？",
        f"{t}元买了{q}{unit}{obj}，平均每个几元？",
    ][t_]
    lines = [f"单价 = {t} ÷ {q} = {num(Fraction(t, q))}元"]
    return ins, lines, Fraction(t, q)


_reg("price_each", price_each)


# 6. pay_change: bill - paid = change ----------------------------------------
def pay_change(rng):
    t = rng.choice([10, 20, 50, 60, 100])
    cost = rng.randint(2, t - 1)
    obj = rng.choice(GOODS)
    n = rng.choice(NAMES)
    tt = rng.randrange(3)
    ins = [
        f"{n}买{obj}要付{cost}元，付了一张{t}元，找回多少元？",
        f"商品共{cost}元，{n}付了{t}元，应找零多少？",
        f"应付{cost}元，{n}给了{t}元，还需找回几元？",
    ][tt]
    lines = [f"找零 = {t} - {cost} = {t - cost}元"]
    return ins, lines, t - cost


_reg("pay_change", pay_change)


# 7. weeks_days: w × 7 = days / days ÷ 7 = weeks ------------------------------
def weeks_days(rng):
    w = rng.randint(2, 8)
    a = rng.randrange(2)
    if a == 0:
        ins = f"{w}个星期一共有多少天？"
        lines = [f"天数 = {w} × 7 = {w * 7}天"]
        return ins, lines, w * 7
    days = w * 7
    ins = f"{days}天是多少个星期？"
    lines = [f"星期数 = {days} ÷ 7 = {w}个星期"]
    return ins, lines, w


_reg("weeks_days", weeks_days)


# 8. groups_kids: g × k = total ---------------------------------------------
def groups_kids(rng):
    g, k = rng.randint(2, 8), rng.randint(3, 15)
    obj = rng.choice(["队", "组", "排", "班"])
    kids = rng.choice(["小朋友", "同学", "学生"])
    t = rng.randrange(3)
    ins = [
        f"有{g}{obj}，每{obj}{k}个{kids}，一共多少{kids}？",
        f"每{obj}坐{k}人，共{g}{obj}，总共有几人？",
        f"{g}{obj}，每{obj}{k}名，共多少名？",
    ][t]
    lines = [f"总数 = {g} × {k} = {g * k}人"]
    return ins, lines, g * k


_reg("groups_kids", groups_kids)


# 9. half_double: double / half ---------------------------------------------
def half_double(rng):
    v = rng.randint(6, 60)
    a = rng.randrange(2)
    if a == 0:
        ins = f"{v}的两倍是多少？"
        lines = [f"两倍 = {v} × 2 = {v * 2}"]
        return ins, lines, v * 2
    even = v - (v % 2)  # ensure half integral
    ins = f"{even}的一半是多少？"
    lines = [f"一半 = {even} ÷ 2 = {even // 2}"]
    return ins, lines, even // 2


_reg("half_double", half_double)


# 10. time_pass: start + duration = end (1-step, no wrap) ---------------------
def time_pass(rng):
    start = rng.randint(0, 21)
    dur = rng.randint(1, 22 - start)
    end = start + dur
    n1 = rng.choice(NAMES)
    act = rng.choice(["看书", "写作业", "跑步", "弹琴"])
    t = rng.randrange(3)
    ins = [
        f"{n1}{start}点开始{act}，用了{dur}小时，几点结束？",
        f"{act}从{start}时进行{dur}小时，结束时是几时？",
        f"{n1}从{start}点开始，{dur}小时后到几点？",
    ][t]
    lines = [f"结束 = {start} + {dur} = {end}点"]
    return ins, lines, end


_reg("time_pass", time_pass)


# 11. animal_legs: count × legs = total (known small) -------------------------
def animal_legs(rng):
    kind, legs = rng.choice([("小鸡", 2), ("小狗", 4), ("小猫", 4), ("小鸭", 2), ("八爪鱼", 8)])
    cnt = rng.randint(2, 12)
    t = rng.randrange(2)
    ins = [
        f"{cnt}只{kind}一共有几条腿？",
        f"每只{kind}有{legs}条腿，{cnt}只共有多少条腿？",
    ][t]
    lines = [f"腿数 = {cnt} × {legs} = {cnt * legs}条"]
    return ins, lines, cnt * legs


_reg("animal_legs", animal_legs)


# 12. score_sum: three small numbers -----------------------------------------
def score_sum(rng):
    a, b, c = (rng.randint(5, 95) for _ in range(3))
    s = a + b + c
    n1 = rng.choice(NAMES)
    ins = f"{n1}三次口算分别得了{a}、{b}、{c}分，三次一共多少分？"
    lines = [
        f"和 = {a} + {b} = {a + b}分",
        f"总和 = {a + b} + {c} = {s}分",
    ]
    return ins, lines, s


_reg("score_sum", score_sum)


# 13. buy_book_change: pay then change (2-step) -------------------------------
def book_change(rng):
    price, money = rng.randint(3, 40), rng.choice([50, 100])
    n1 = rng.choice(NAMES)
    obj = rng.choice(["文具", "玩具", "学习用品"])
    ins = f"{n1}买{obj}花了{price}元，付了{money}元，应找回多少钱？"
    lines = [f"找零 = {money} - {price} = {money - price}元"]
    return ins, lines, money - price


_reg("book_change", book_change)


if __name__ == "__main__":
    import random
    rng = random.Random(1)
    ok = 0
    from run_math_short import verify
    for _lvl, name, fn in PROGRAMS:
        for _ in range(30):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines}"
            ok += 1
    print(f"L1 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
    import run_math_short  # noqa: F401  (ensure import-path availability)