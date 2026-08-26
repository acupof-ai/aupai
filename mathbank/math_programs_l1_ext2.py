#!/usr/bin/env python3
"""L1 extension bank #2: 58 more single/two-step families (1-2 arithmetic steps).

Each fn(rng) -> (instruction, lines[list[str]], ans:int|Fraction). Every line is
`label = expr = value[unit]` (or `expr = value[unit]`); the last line's value
must equal num(ans). At least 3 phrasings per family; names/objects/units drawn
from mathcommon pools; all numbers <= 100. Verified by run_math_short.verify.
"""

import random
from fractions import Fraction

from mathcommon import (
    ANIMALS, FOOD, FRUITS, GOODS, NAMES, PLACE, STATIONERY,
    UNIT_N, UNIT_ZHI, num,
)

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L1", name, fn))


# 1. compare_more_less: difference of two quantities --------------------------
def compare_more_less(rng):
    a = rng.randint(20, 90)
    b = rng.randint(5, a - 3)
    obj = rng.choice(["张卡片", "颗糖", "本书", "个苹果", "支铅笔"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{a}{obj}，{n2}有{b}{obj}，{n1}比{n2}多多少{obj}？",
        f"{n1}有{a}{obj}，{n2}有{b}{obj}，{n2}比{n1}少多少{obj}？",
        f"{n1}攒了{a}{obj}，{n2}攒了{b}{obj}，两人相差多少{obj}？",
    ][t]
    lines = [f"相差 = {a} - {b} = {a - b}{obj}"]
    return ins, lines, a - b


_reg("compare_more_less", compare_more_less)


# 2. compare_find: find the bigger/smaller quantity ----------------------------
def compare_find(rng):
    a = rng.randint(15, 80)
    b = rng.randint(3, 12)
    obj = rng.choice(["张卡片", "颗糖", "本书", "朵花", "个气球"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    if t == 0:
        ins = f"{n1}有{a}{obj}，{n2}比{n1}多{b}{obj}，{n2}有多少{obj}？"
        lines = [f"{n2} = {a} + {b} = {a + b}{obj}"]
        return ins, lines, a + b
    if t == 1:
        ins = f"{n1}有{a}{obj}，{n2}比{n1}少{b}{obj}，{n2}有多少{obj}？"
        lines = [f"{n2} = {a} - {b} = {a - b}{obj}"]
        return ins, lines, a - b
    if rng.randrange(2):
        ins = f"{n1}有{a}{obj}，比{n2}多{b}{obj}，{n2}有多少{obj}？"
        lines = [f"{n2} = {a} - {b} = {a - b}{obj}"]
        return ins, lines, a - b
    ins = f"{n1}有{a}{obj}，比{n2}少{b}{obj}，{n2}有多少{obj}？"
    lines = [f"{n2} = {a} + {b} = {a + b}{obj}"]
    return ins, lines, a + b


_reg("compare_find", compare_find)


# 3. compare_then_sum: one side differs, find the sum --------------------------
def compare_then_sum(rng):
    a = rng.randint(12, 45)
    b = rng.randint(3, 9)
    obj = rng.choice(["张卡片", "颗糖", "本书", "个苹果"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    if t == 0:
        c = a + b
        ins = f"{n1}有{a}{obj}，{n2}比{n1}多{b}{obj}，两人一共有多少{obj}？"
        lines = [f"{n2} = {a} + {b} = {c}{obj}",
                 f"一共 = {a} + {c} = {a + c}{obj}"]
        return ins, lines, a + c
    if t == 1:
        c = a - b
        ins = f"{n1}有{a}{obj}，{n2}比{n1}少{b}{obj}，两人一共有多少{obj}？"
        lines = [f"{n2} = {a} - {b} = {c}{obj}",
                 f"一共 = {a} + {c} = {a + c}{obj}"]
        return ins, lines, a + c
    c = a + b
    ins = f"{n1}有{a}{obj}，比{n2}少{b}{obj}，两人一共有多少{obj}？"
    lines = [f"{n2} = {a} + {b} = {c}{obj}",
             f"一共 = {a} + {c} = {a + c}{obj}"]
    return ins, lines, a + c


_reg("compare_then_sum", compare_then_sum)


# 4. times_as_many: k times a quantity -----------------------------------------
def times_as_many(rng):
    a = rng.randint(3, 20)
    k = rng.randint(2, 5)
    obj = rng.choice(["张卡片", "颗糖", "本书", "支铅笔", "个苹果"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{a}{obj}，{n2}的{obj}是{n1}的{k}倍，{n2}有多少{obj}？",
        f"第一个书架有{a}{obj}，第二个书架的{obj}是第一个的{k}倍，第二个书架有多少{obj}？",
        f"一年级捐了{a}{obj}，二年级捐的是一年级的{k}倍，二年级捐了多少{obj}？",
    ][t]
    lines = [f"{n2 if t == 0 else '二年级' if t == 2 else '第二个书架'} = {a} × {k} = {a * k}{obj}"]
    return ins, lines, a * k


_reg("times_as_many", times_as_many)


# 5. times_divide: known multiple, find the base -------------------------------
def times_divide(rng):
    k = rng.randint(2, 5)
    b = rng.randint(3, 18)
    a = k * b
    obj = rng.choice(["张卡片", "颗糖", "本书", "支铅笔", "个苹果"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{a}{obj}，正好是{n2}的{k}倍，{n2}有多少{obj}？",
        f"苹果树有{a}棵，是梨树的{k}倍，梨树有多少棵？",
        f"美术组有{a}人，是书法组的{k}倍，书法组有多少人？",
    ][t]
    unit = obj if t == 0 else "棵" if t == 1 else "人"
    label = n2 if t == 0 else "梨树" if t == 1 else "书法组"
    lines = [f"{label} = {a} ÷ {k} = {b}{unit}"]
    return ins, lines, b


_reg("times_divide", times_divide)


# 6. times_ratio: how many times one is of another -----------------------------
def times_ratio(rng):
    b = rng.randint(2, 9)
    k = rng.randint(2, 6)
    a = b * k
    obj = rng.choice(["张卡片", "颗糖", "本书", "支铅笔"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{a}{obj}，{n2}有{b}{obj}，{n1}的{obj}是{n2}的几倍？",
        f"苹果树{a}棵，梨树{b}棵，苹果树是梨树的几倍？",
        f"美术组{a}人，书法组{b}人，美术组人数是书法组的几倍？",
    ][t]
    lines = [f"倍数 = {a} ÷ {b} = {k}倍"]
    return ins, lines, k


_reg("times_ratio", times_ratio)


# 7. times_sum: base plus its multiple ------------------------------------------
def times_sum(rng):
    a = rng.randint(4, 15)
    k = rng.randint(2, 4)
    obj = rng.choice(["张卡片", "颗糖", "本书", "个苹果"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{a}{obj}，{n2}的是{n1}的{k}倍，两人共有多少{obj}？",
        f"苹果树{a}棵，梨树是苹果树的{k}倍，两种树共多少棵？",
        f"红气球{a}个，黄气球是红气球的{k}倍，两种气球共多少个？",
    ][t]
    unit = obj if t == 0 else "棵" if t == 1 else "个"
    label2 = n2 if t == 0 else "梨树" if t == 1 else "黄气球"
    lines = [f"{label2} = {a} × {k} = {a * k}{unit}",
             f"一共 = {a} + {a * k} = {a * (k + 1)}{unit}"]
    return ins, lines, a * (k + 1)


_reg("times_sum", times_sum)


# 8. times_diff: multiple minus the base ----------------------------------------
def times_diff(rng):
    a = rng.randint(4, 15)
    k = rng.randint(2, 4)
    obj = rng.choice(["张卡片", "颗糖", "本书", "个苹果"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{a}{obj}，{n2}的是{n1}的{k}倍，{n1}比{n2}少多少{obj}？",
        f"苹果树{a}棵，梨树是苹果树的{k}倍，苹果树比梨树少多少棵？",
        f"红气球{a}个，黄气球是红气球的{k}倍，红气球比黄气球少多少个？",
    ][t]
    unit = obj if t == 0 else "棵" if t == 1 else "个"
    label2 = n2 if t == 0 else "梨树" if t == 1 else "黄气球"
    lines = [f"{label2} = {a} × {k} = {a * k}{unit}",
             f"少 = {a * k} - {a} = {a * (k - 1)}{unit}"]
    return ins, lines, a * (k - 1)


_reg("times_diff", times_diff)


# 9. times_offset: k times plus/minus a few -------------------------------------
def times_offset(rng):
    a = rng.randint(4, 12)
    k = rng.randint(2, 4)
    c = rng.randint(1, a * k - 2)
    obj = rng.choice(["张卡片", "颗糖", "本书", "个苹果"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    if t == 0:
        ins = f"{n1}有{a}{obj}，{n2}的比{n1}的{k}倍多{c}{obj}，{n2}有多少{obj}？"
        lines = [f"{n2} = {a} × {k} + {c} = {a * k + c}{obj}"]
        return ins, lines, a * k + c
    if t == 1:
        ins = f"{n1}有{a}{obj}，{n2}的比{n1}的{k}倍少{c}{obj}，{n2}有多少{obj}？"
        lines = [f"{n2} = {a} × {k} - {c} = {a * k - c}{obj}"]
        return ins, lines, a * k - c
    ins = f"苹果树{a}棵，梨树比苹果树的{k}倍多{c}棵，梨树有多少棵？"
    lines = [f"梨树 = {a} × {k} + {c} = {a * k + c}棵"]
    return ins, lines, a * k + c


_reg("times_offset", times_offset)


# 10. remainder_split: quotient given, find the remainder -----------------------
def remainder_split(rng):
    k = rng.randint(3, 6)
    q = rng.randint(3, 8)
    r = rng.randint(1, k - 1)
    n = k * q + r
    obj, unit = rng.choice([("故事书", "本"), ("苹果", "个"), ("糖", "颗"),
                            ("饼干", "块"), ("铅笔", "支")])
    t = rng.randrange(3)
    ins = [
        f"有{n}{unit}{obj}，每{k}{unit}装一盒，装满{q}盒后还剩多少{unit}？",
        f"老师把{n}{unit}{obj}平均分给{k}个小组，每组分得{q}{unit}，还剩多少{unit}？",
        f"{n}{unit}{obj}，每{k}{unit}放一盘，放满{q}盘后还剩多少{unit}？",
    ][t]
    lines = [f"分掉 = {q} × {k} = {q * k}{unit}",
             f"剩下 = {n} - {q * k} = {r}{unit}"]
    return ins, lines, r


_reg("remainder_split", remainder_split)


# 11. remainder_pad: how many more to fill the last set -------------------------
def remainder_pad(rng):
    k = rng.randint(3, 6)
    q = rng.randint(3, 8)
    r = rng.randint(1, k - 1)
    n = k * q + r
    obj = rng.choice(["苹果", "鸡蛋", "月饼", "面包"])
    t = rng.randrange(3)
    ins = [
        f"有{n}个{obj}，每{k}个装一盒，至少再添几个正好装满？",
        f"{n}个{obj}，每{k}个放一盘，至少再拿几个正好放满？",
        f"把{n}个{obj}每{k}个装一袋，至少再添几个正好装完？",
    ][t]
    lines = [f"剩下 = {n} - {q} × {k} = {r}个",
             f"再添 = {k} - {r} = {k - r}个"]
    return ins, lines, k - r


_reg("remainder_pad", remainder_pad)


# 12. boats_ceil: round up (进一法) ----------------------------------------------
def boats_ceil(rng):
    k = rng.randint(3, 6)
    q = rng.randint(3, 8)
    r = rng.randint(1, k - 1)
    n = k * q + r
    t = rng.randrange(3)
    ins = [
        f"{n}人过河，每条船坐{k}人，至少需要几条船？",
        f"{n}个同学去划船，每条船最多坐{k}人，至少要租几条船？",
        f"{n}人乘车去春游，每辆车坐{k}人，至少需要几辆车？",
    ][t]
    unit = "条船" if t < 2 else "辆车"
    label = "船数" if t < 2 else "车数"
    lines = [f"坐满 = {q} × {k} = {q * k}人",
             f"剩下 = {n} - {q * k} = {r}人",
             f"{label} = {q} + 1 = {q + 1}{unit}"]
    return ins, lines, q + 1


_reg("boats_ceil", boats_ceil)


# 13. contain_div: measurement division (包含除) ---------------------------------
def contain_div(rng):
    b = rng.randint(2, 9)
    q = rng.randint(2, 9)
    a = b * q
    t = rng.randrange(3)
    if t == 0:
        food = rng.choice(FOOD)
        ins = f"每个{food}{b}元，{a}元能买几个{food}？"
        lines = [f"个数 = {a} ÷ {b} = {q}个"]
    elif t == 1:
        ins = f"一本故事书有{a}页，每天看{b}页，几天可以看完？"
        lines = [f"天数 = {a} ÷ {b} = {q}天"]
    else:
        ins = f"{a}个同学去划船，每条船坐{b}人，需要几条船？"
        lines = [f"船数 = {a} ÷ {b} = {q}条"]
    return ins, lines, q


_reg("contain_div", contain_div)


# 14. fraction_part: find 1/k of a quantity -------------------------------------
def fraction_part(rng):
    k = rng.randint(3, 6)
    each = rng.randint(2, 12)
    a = k * each
    obj, unit = rng.choice([("一袋糖", "颗"), ("一袋苹果", "个"),
                            ("一袋饼干", "块"), ("一盒巧克力", "块")])
    t = rng.randrange(3)
    ins = [
        f"{obj}有{a}{unit}，小丽吃了它的1/{k}，吃了多少{unit}？",
        f"{obj}共{a}{unit}，拿出它的1/{k}分给同学，分出多少{unit}？",
        f"{obj}有{a}{unit}，吃掉它的1/{k}，吃了多少{unit}？",
    ][t]
    lines = [f"部分 = {a} ÷ {k} = {each}{unit}"]
    return ins, lines, each


_reg("fraction_part", fraction_part)


# 15. fraction_divide: quotient is a proper fraction -----------------------------
def fraction_divide(rng):
    b = rng.randint(3, 6)
    a = rng.randint(2, b - 1)
    t = rng.randrange(3)
    ins = [
        f"把{a}个月饼平均分给{b}个小朋友，每人分几个？",
        f"{a}个西瓜平均分给{b}人，每人分到几个？",
        f"把{a}块蛋糕平均分给{b}个小朋友，每人分几块？",
    ][t]
    unit = "个" if t < 2 else "块"
    ans = Fraction(a, b)
    lines = [f"每人 = {a} ÷ {b} = {num(ans)}{unit}"]
    return ins, lines, ans


_reg("fraction_divide", fraction_divide)


# 16. fraction_add: same-denominator fraction add/sub ----------------------------
def fraction_add(rng):
    d = rng.randint(5, 9)
    a = rng.randint(2, d - 2)
    b = rng.randint(1, min(a - 1, d - 1 - a))
    t = rng.randrange(3)
    if t == 0:
        ins = f"一块蛋糕平均分成{d}份，小明吃了{a}份，小红吃了{b}份，两人共吃了这块蛋糕的几分之几？"
        ans = Fraction(a + b, d)
        lines = [f"共吃 = {a}/{d} + {b}/{d} = {num(ans)}块"]
    elif t == 1:
        ins = f"一块蛋糕平均分成{d}份，小明吃了{a}份，还剩这块蛋糕的几分之几？"
        ans = Fraction(d - a, d)
        lines = [f"剩下 = 1 - {a}/{d} = {num(ans)}块"]
    else:
        ins = f"一块蛋糕平均分成{d}份，小明吃了{a}份，小红吃了{b}份，小明比小红多吃几分之几？"
        ans = Fraction(a - b, d)
        lines = [f"多吃 = {a}/{d} - {b}/{d} = {num(ans)}块"]
    return ins, lines, ans


_reg("fraction_add", fraction_add)


# 17. work_rate: 1/a of the job per day ------------------------------------------
def work_rate(rng):
    a = rng.randint(3, 9)
    t = rng.randrange(3)
    ins = [
        f"一项工程，甲队单独做{a}天完成，每天完成这项工程的几分之几？",
        f"修一条路，单独修{a}天修完，每天修这条路的几分之几？",
        f"一批货物{a}次运完，每次运这批货物的几分之几？",
    ][t]
    label = "每次" if t == 2 else "每天"
    ans = Fraction(1, a)
    lines = [f"{label} = 1 ÷ {a} = {num(ans)}"]
    return ins, lines, ans


_reg("work_rate", work_rate)


# 18. speed_unit: distance / time = speed ----------------------------------------
def speed_unit(rng):
    t = rng.randrange(3)
    if t == 0:
        b = rng.randint(2, 6)
        q = rng.randint(5, 15)
        a = b * q
        ins = f"小明骑自行车{b}小时行了{a}千米，每小时行多少千米？"
        lines = [f"速度 = {a} ÷ {b} = {q}千米"]
        return ins, lines, q
    if t == 1:
        b = rng.randint(2, 5)
        q = rng.randint(4, 12)
        a = b * q
        ins = f"一只蜗牛{b}小时爬了{a}米，平均每小时爬多少米？"
        lines = [f"速度 = {a} ÷ {b} = {q}米"]
        return ins, lines, q
    b = rng.randint(3, 5)
    a = rng.randint(b + 1, b * 6)
    if a % b == 0:
        a += 1
    ans = Fraction(a, b)
    ins = f"小明走路{b}小时走了{a}千米，平均每小时走多少千米？"
    lines = [f"速度 = {a} ÷ {b} = {num(ans)}千米"]
    return ins, lines, ans


_reg("speed_unit", speed_unit)


# 19. guiyi: normalize then multiply (归一) ---------------------------------------
def guiyi(rng):
    b = rng.randint(2, 5)
    p = rng.randint(2, 8)
    a = b * p
    c = rng.randint(2, 9)
    obj, unit = rng.choice([("笔记本", "本"), ("铅笔", "支"),
                            ("苹果", "个"), ("包子", "个")])
    t = rng.randrange(3)
    ins = [
        f"{b}{unit}{obj}共{a}元，买{c}{unit}要多少元？",
        f"妈妈买{b}{unit}{obj}用了{a}元，照这样计算，买{c}{unit}要多少元？",
        f"{a}元买了{b}{unit}{obj}，买{c}{unit}同样的{obj}要多少元？",
    ][t]
    lines = [f"单价 = {a} ÷ {b} = {p}元",
             f"总价 = {p} × {c} = {p * c}元"]
    return ins, lines, p * c


_reg("guiyi", guiyi)


# 20. guizong: total fixed, redistribute (归总) -----------------------------------
def guizong(rng):
    a = rng.randint(3, 9)
    b = rng.randint(3, 8)
    total = a * b
    c = rng.randint(2, 9)
    for _ in range(50):
        if total % c == 0:
            break
        c = rng.randint(2, 9)
    q = total // c
    t = rng.randrange(3)
    ins = [
        f"小明每天看{a}页故事书，{b}天正好看完。如果每天看{c}页，几天看完？",
        f"一批书，每箱装{a}本，{b}箱正好装完。如果每箱装{c}本，要装几箱？",
        f"同学们做操，每行站{a}人，正好站{b}行。如果每行站{c}人，要站几行？",
    ][t]
    unit = ["天", "箱", "行"][t]
    label = ["天数", "箱数", "行数"][t]
    lines = [f"总数 = {a} × {b} = {total}",
             f"{label} = {total} ÷ {c} = {q}{unit}"]
    return ins, lines, q


_reg("guizong", guizong)


# 21. buy_two: one line item plus one single item ------------------------------
def buy_two(rng):
    q = rng.randint(2, 6)
    p = rng.randint(3, 9)
    c = rng.randint(3, 20)
    t = rng.randrange(3)
    if t == 0:
        fruit = rng.choice(FRUITS)
        ins = f"{fruit}每斤{p}元，妈妈买了{q}斤，又买了一袋{c}元的饼干，一共花了多少元？"
        lines = [f"总价 = {q} × {p} + {c} = {q * p + c}元"]
        return ins, lines, q * p + c
    if t == 1:
        obj = rng.choice(STATIONERY)
        unit = rng.choice([UNIT_N, UNIT_ZHI])
        ins = f"{obj}每个{p}元，小明买了{q}{unit}，还买了一个{c}元的文具盒，一共要付多少元？"
        lines = [f"总价 = {q} × {p} + {c} = {q * p + c}元"]
        return ins, lines, q * p + c
    adult = rng.randint(8, 20)
    child = rng.randint(4, adult - 1)
    ins = f"动物园成人票每张{adult}元，儿童票每张{child}元，买{q}张成人票和1张儿童票共多少元？"
    lines = [f"成人票 = {q} × {adult} = {q * adult}元",
             f"一共 = {q * adult} + {child} = {q * adult + child}元"]
    return ins, lines, q * adult + child


_reg("buy_two", buy_two)


# 22. spend_left: money minus a purchase ----------------------------------------
def spend_left(rng):
    b = rng.randint(2, 6)
    p = rng.randint(3, 9)
    cost = b * p
    a = cost + rng.randint(3, 30)
    n = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n}带了{a}元，买{b}支钢笔，每支{p}元，还剩多少元？",
        f"妈妈带{a}元去买菜，买了{b}斤苹果，每斤{p}元，还剩多少元？",
        f"{n}有{a}元，买{b}个笔记本，每个{p}元，还剩多少元？",
    ][t]
    lines = [f"剩下 = {a} - {b} × {p} = {a - cost}元"]
    return ins, lines, a - cost


_reg("spend_left", spend_left)


# 23. shortage: how much more money is needed -----------------------------------
def shortage(rng):
    b = rng.randint(2, 5)
    p = rng.randint(8, 20)
    cost = b * p
    a = cost - rng.randint(3, 15)
    t = rng.randrange(3)
    ins = [
        f"每个书包{p}元，小明想买{b}个，可是他只带了{a}元，还差多少元？",
        f"一盒彩笔{p}元，买{b}盒，只带了{a}元，还差多少元？",
        f"每辆玩具汽车{p}元，买{b}辆，带了{a}元，还差多少元？",
    ][t]
    lines = [f"还差 = {b} × {p} - {a} = {cost - a}元"]
    return ins, lines, cost - a


_reg("shortage", shortage)


# 24. cartons_offset: packs with a leftover/deficit ------------------------------
def cartons_offset(rng):
    a = rng.randint(6, 15)
    b = rng.randint(3, 6)
    c = rng.randint(3, 9)
    t = rng.randrange(3)
    if t == 0:
        ins = f"每箱苹果{a}千克，装了{b}箱，还剩{c}千克没装，一共有多少千克苹果？"
        lines = [f"一共 = {a} × {b} + {c} = {a * b + c}千克"]
        return ins, lines, a * b + c
    if t == 1:
        ins = f"每箱苹果{a}千克，装了{b}箱，吃掉{c}千克后，还剩多少千克？"
        lines = [f"剩下 = {a} × {b} - {c} = {a * b - c}千克"]
        return ins, lines, a * b - c
    ins = f"商店运来{b}箱苹果，每箱{a}千克，卖出{c}千克，还剩多少千克？"
    lines = [f"剩下 = {a} × {b} - {c} = {a * b - c}千克"]
    return ins, lines, a * b - c


_reg("cartons_offset", cartons_offset)


# 25. add_sub_mix: add then subtract (bus / library / savings) -------------------
def add_sub_mix(rng):
    a = rng.randint(20, 60)
    b = rng.randint(3, 15)
    c = rng.randint(3, b + 5)
    t = rng.randrange(3)
    if t == 0:
        ins = f"车上原有{a}人，到站后上车{b}人，下车{c}人，现在车上有多少人？"
        lines = [f"现在 = {a} + {b} - {c} = {a + b - c}人"]
    elif t == 1:
        ins = f"图书角原有{a}本书，又买来{b}本，借走{c}本，现在有多少本？"
        lines = [f"现在 = {a} + {b} - {c} = {a + b - c}本"]
    else:
        n = rng.choice(NAMES)
        ins = f"{n}的存钱罐里有{a}元，妈妈又给了{b}元，他买文具用去{c}元，还剩多少元？"
        lines = [f"剩下 = {a} + {b} - {c} = {a + b - c}元"]
    return ins, lines, a + b - c


_reg("add_sub_mix", add_sub_mix)


# 26. sub_twice: two successive subtractions -------------------------------------
def sub_twice(rng):
    a = rng.randint(60, 99)
    b = rng.randint(10, 30)
    c = rng.randint(10, 30)
    t = rng.randrange(3)
    ins = [
        f"一本故事书有{a}页，第一天看了{b}页，第二天看了{c}页，还剩多少页？",
        f"一根绳子长{a}米，第一次用去{b}米，第二次用去{c}米，还剩多少米？",
        f"妈妈带{a}元，买肉用去{b}元，买菜用去{c}元，还剩多少元？",
    ][t]
    unit = ["页", "米", "元"][t]
    lines = [f"剩下 = {a} - {b} - {c} = {a - b - c}{unit}"]
    return ins, lines, a - b - c


_reg("sub_twice", sub_twice)


# 27. liancheng: three factors ----------------------------------------------------
def liancheng(rng):
    a = rng.randint(2, 4)
    b = rng.randint(2, 4)
    c = rng.randint(3, 9)
    for _ in range(50):
        if a * b * c <= 100:
            break
        c = rng.randint(3, 9)
    t = rng.randrange(3)
    ins = [
        f"学校有{a}个年级，每个年级{b}个班，每班栽{c}棵树，一共栽多少棵树？",
        f"超市运来{a}车苹果，每车{b}箱，每箱{c}千克，一共运来多少千克？",
        f"小明每天写{a}页字，每页写{b}行，每行{c}个字，每天写多少个字？",
    ][t]
    unit = ["棵", "千克", "个"][t]
    lines = [f"一共 = {a} × {b} × {c} = {a * b * c}{unit}"]
    return ins, lines, a * b * c


_reg("liancheng", liancheng)


# 28. double_divide: two successive divisions -------------------------------------
def double_divide(rng):
    t = rng.randrange(3)
    if t == 0:
        c = rng.randint(2, 4)
        b = rng.randint(2, 4)
        each = rng.randint(2, 8)
        a = b * c * each
        ins = f"学校买来{a}本故事书，分给{b}个年级，每个年级{c}个班，每班分几本？"
        lines = [f"每年级 = {a} ÷ {b} = {a // b}本",
                 f"每班 = {a // b} ÷ {c} = {each}本"]
        return ins, lines, each
    if t == 1:
        each = rng.randint(2, 8)
        a = each * 4
        ins = f"一根绳子长{a}米，对折再对折，每段长多少米？"
        lines = [f"对折一次 = {a} ÷ 2 = {a // 2}米",
                 f"再对折 = {a // 2} ÷ 2 = {each}米"]
        return ins, lines, each
    c = rng.randint(2, 4)
    b = rng.randint(2, 5)
    each = rng.randint(2, 7)
    a = b * c * each
    ins = f"{a}个同学做操，平均分成{b}个方队，每个方队{c}行，每行几人？"
    lines = [f"每方队 = {a} ÷ {b} = {a // b}人",
             f"每行 = {a // b} ÷ {c} = {each}人"]
    return ins, lines, each


_reg("double_divide", double_divide)


# 29. avg_two: average of two numbers ---------------------------------------------
def avg_two(rng):
    a = rng.randint(10, 40)
    b = rng.randint(10, 40)
    if (a + b) % 2:
        b += 1
    s = a + b
    t = rng.randrange(3)
    ins = [
        f"小明两次拍球分别拍了{a}下和{b}下，平均每次拍多少下？",
        f"小华两天看书{a}页和{b}页，平均每天看多少页？",
        f"小军两次数学测验得{a}分和{b}分，平均分是多少分？",
    ][t]
    unit = ["下", "页", "分"][t]
    lines = [f"总和 = {a} + {b} = {s}{unit}",
             f"平均 = {s} ÷ 2 = {s // 2}{unit}"]
    return ins, lines, s // 2


_reg("avg_two", avg_two)


# 30. avg_three: average of three numbers ------------------------------------------
def avg_three(rng):
    a = b = c = 1
    s = 1
    for _ in range(50):
        a = rng.randint(5, 25)
        b = rng.randint(5, 25)
        c = rng.randint(5, 25)
        s = a + b + c
        if s % 3 == 0:
            break
    t = rng.randrange(3)
    ins = [
        f"小丽三次跳绳分别跳了{a}下、{b}下、{c}下，平均每次跳多少下？",
        f"小明三天看书{a}页、{b}页、{c}页，平均每天看多少页？",
        f"小华三次口算分别得{a}分、{b}分、{c}分，平均分是多少分？",
    ][t]
    unit = ["下", "页", "分"][t]
    lines = [f"总分 = {a} + {b} + {c} = {s}{unit}",
             f"平均 = {s} ÷ 3 = {s // 3}{unit}"]
    return ins, lines, s // 3


_reg("avg_three", avg_three)


# 31. avg_reverse: average times count = total -------------------------------------
def avg_reverse(rng):
    a = rng.randint(5, 15)
    k = rng.randint(3, 7)
    t = rng.randrange(3)
    ins = [
        f"小明平均每次跳{a}下绳，跳了{k}次，一共跳了多少下？",
        f"书店平均每天卖出{a}套书，{k}天共卖出多少套？",
        f"小华平均每天写{a}个大字，{k}天一共写多少个？",
    ][t]
    unit = ["下", "套", "个"][t]
    lines = [f"一共 = {a} × {k} = {a * k}{unit}"]
    return ins, lines, a * k


_reg("avg_reverse", avg_reverse)


# 32. diff_then_share: subtract then split evenly -----------------------------------
def diff_then_share(rng):
    k = rng.randint(2, 5)
    each = rng.randint(3, 10)
    diff = k * each
    b = rng.randint(3, 30)
    a = b + diff
    t = rng.randrange(3)
    ins = [
        f"小明有{a}张卡片，送给小红{b}张后，剩下的平均分给{k}个好朋友，每人分几张？",
        f"一根绳子长{a}米，先用去{b}米，剩下的平均剪成{k}段，每段长多少米？",
        f"妈妈买了{a}个苹果，吃掉{b}个，剩下的平均分给{k}个小朋友，每人分几个？",
    ][t]
    unit = ["张", "米", "个"][t]
    label = ["每人", "每段", "每人"][t]
    lines = [f"剩下 = {a} - {b} = {diff}{unit}",
             f"{label} = {diff} ÷ {k} = {each}{unit}"]
    return ins, lines, each


_reg("diff_then_share", diff_then_share)


# 33. transfer_equal: transfer to make two sides equal -------------------------------
def transfer_equal(rng):
    b = rng.randint(5, 20)
    d = rng.randint(2, 10) * 2
    a = b + d
    t = rng.randrange(3)
    ins = [
        f"小明有{a}张卡片，小红有{b}张，小明给小红多少张后两人一样多？",
        f"甲筐有{a}千克苹果，乙筐有{b}千克，从甲筐拿多少千克到乙筐，两筐一样多？",
        f"哥哥有{a}元，弟弟有{b}元，哥哥给弟弟多少元后两人钱数一样多？",
    ][t]
    unit = ["张", "千克", "元"][t]
    lines = [f"相差 = {a} - {b} = {d}{unit}",
             f"给的 = {d} ÷ 2 = {d // 2}{unit}"]
    return ins, lines, d // 2


_reg("transfer_equal", transfer_equal)


# 34. half_plus: half of a quantity plus another --------------------------------------
def half_plus(rng):
    half = rng.randint(4, 20)
    a = half * 2
    b = rng.randint(3, 9)
    t = rng.randrange(3)
    ins = [
        f"一根绳子长{a}米，第一次用去一半，第二次用去{b}米，两次一共用去多少米？",
        f"一袋大米{a}千克，第一天吃了一半，第二天吃了{b}千克，两天共吃多少千克？",
        f"小明有{a}元，买文具用去一半，买零食用去{b}元，一共用去多少元？",
    ][t]
    unit = ["米", "千克", "元"][t]
    lines = [f"一半 = {a} ÷ 2 = {half}{unit}",
             f"共用 = {half} + {b} = {half + b}{unit}"]
    return ins, lines, half + b


_reg("half_plus", half_plus)


# 35. tare_half: net weight then half ---------------------------------------------------
def tare_half(rng):
    b = rng.randint(2, 8)
    half = rng.randint(4, 15)
    a = b + half * 2
    t = rng.randrange(3)
    ins = [
        f"一筐苹果连筐重{a}千克，筐重{b}千克，苹果的一半重多少千克？",
        f"一桶油连桶重{a}千克，桶重{b}千克，油的一半重多少千克？",
        f"一箱橘子连箱重{a}千克，箱子重{b}千克，橘子的一半重多少千克？",
    ][t]
    lines = [f"净重 = {a} - {b} = {a - b}千克",
             f"一半 = {a - b} ÷ 2 = {half}千克"]
    return ins, lines, half


_reg("tare_half", tare_half)


# 36. suit_buy: one set times the number of sets ----------------------------------------
def suit_buy(rng):
    a = rng.randint(12, 30)
    b = rng.randint(8, a - 3)
    c = rng.randint(2, 3)
    t = rng.randrange(3)
    ins = [
        f"一件上衣{a}元，一条裤子{b}元，买{c}套这样的衣服要多少元？",
        f"一张桌子{a}元，一把椅子{b}元，买{c}套桌椅要多少元？",
        f"一盒彩笔{a}元，一个画本{b}元，买{c}套要多少元？",
    ][t]
    lines = [f"一套 = {a} + {b} = {a + b}元",
             f"{c}套 = {a + b} × {c} = {(a + b) * c}元"]
    return ins, lines, (a + b) * c


_reg("suit_buy", suit_buy)


# 37. diff_times: difference then a multiple ---------------------------------------------
def diff_times(rng):
    b = rng.randint(3, 9)
    rest = rng.randint(5, 15)
    a = b + rest
    c = rng.randint(2, 4)
    t = rng.randrange(3)
    ins = [
        f"小明有{a}张卡片，小红比小明少{b}张，小军的卡片是小红的{c}倍，小军有多少张？",
        f"一年级有{a}人，二年级比一年级少{b}人，三年级人数是二年级的{c}倍，三年级有多少人？",
        f"苹果树有{a}棵，梨树比苹果树少{b}棵，桃树是梨树的{c}倍，桃树有多少棵？",
    ][t]
    unit = ["张", "人", "棵"][t]
    label1 = ["小红", "二年级", "梨树"][t]
    label2 = ["小军", "三年级", "桃树"][t]
    lines = [f"{label1} = {a} - {b} = {rest}{unit}",
             f"{label2} = {rest} × {c} = {rest * c}{unit}"]
    return ins, lines, rest * c


_reg("diff_times", diff_times)


# 38. queue: line-up with the self counted once -------------------------------------------
def queue(rng):
    a = rng.randint(4, 12)
    b = rng.randint(4, 12)
    t = rng.randrange(3)
    if t == 0:
        ins = f"同学们排队做操，小明前面有{a}人，后面有{b}人，这一队一共有多少人？"
        lines = [f"一共 = {a} + {b} + 1 = {a + b + 1}人"]
        return ins, lines, a + b + 1
    if t == 1:
        ins = f"同学们排队，从前面数小明排第{a}，从后面数小明排第{b}，这一队一共有多少人？"
        lines = [f"一共 = {a} + {b} - 1 = {a + b - 1}人"]
        return ins, lines, a + b - 1
    animal = rng.choice(ANIMALS)
    ins = f"小动物们排队，{animal}前面有{a}只，后面有{b}只，这一队一共有多少只小动物？"
    lines = [f"一共 = {a} + {b} + 1 = {a + b + 1}只"]
    return ins, lines, a + b + 1


_reg("queue", queue)


# 39. cut_pieces: cuts vs pieces, then average length --------------------------------------
def cut_pieces(rng):
    n = rng.randint(3, 6)
    pieces = n + 1
    each = rng.randint(3, 12)
    a = pieces * each
    t = rng.randrange(3)
    ins = [
        f"一根绳子长{a}米，剪了{n}次，平均每段长多少米？",
        f"一根铁丝长{a}米，剪了{n}次，平均每段长多少米？",
        f"一根木头长{a}米，锯了{n}次，平均每段长多少米？",
    ][t]
    lines = [f"段数 = {n} + 1 = {pieces}段",
             f"每段 = {a} ÷ {pieces} = {each}米"]
    return ins, lines, each


_reg("cut_pieces", cut_pieces)


# 40. interval_count: n objects have n-1 intervals ------------------------------------------
def interval_count(rng):
    n = rng.randint(4, 9)
    t = rng.randrange(3)
    if t == 0:
        ins = f"小明从1楼爬到{n}楼，要爬几层楼梯？"
        lines = [f"层数 = {n} - 1 = {n - 1}层"]
        return ins, lines, n - 1
    if t == 1:
        ins = f"一根木头锯成{n}段，要锯几次？"
        lines = [f"次数 = {n} - 1 = {n - 1}次"]
        return ins, lines, n - 1
    ins = f"{n}人进行乒乓球淘汰赛，要赛多少场才能决出冠军？"
    lines = [f"场次 = {n} - 1 = {n - 1}场"]
    return ins, lines, n - 1


_reg("interval_count", interval_count)


# 41. interval_times: intervals times per-interval amount -------------------------
def interval_times(rng):
    n = rng.randint(3, 6)
    b = rng.randint(5, 20)
    t = rng.randrange(3)
    if t == 0:
        ins = f"小明从1楼走到{n}楼，每层楼要走{b}秒，一共要走多少秒？"
        lines = [f"层数 = {n} - 1 = {n - 1}层",
                 f"秒数 = {n - 1} × {b} = {(n - 1) * b}秒"]
    elif t == 1:
        ins = f"一根木头锯成{n}段，每锯一次要{b}分钟，一共要多少分钟？"
        lines = [f"次数 = {n} - 1 = {n - 1}次",
                 f"分钟 = {n - 1} × {b} = {(n - 1) * b}分钟"]
    else:
        ins = f"时钟{n}点敲{n}下，每隔{b}秒敲一下，多少秒敲完？"
        lines = [f"间隔 = {n} - 1 = {n - 1}个",
                 f"秒数 = {n - 1} × {b} = {(n - 1) * b}秒"]
    return ins, lines, (n - 1) * b


_reg("interval_times", interval_times)


# 42. clock_strike: strike time divided by intervals -------------------------------
def clock_strike(rng):
    n = rng.randint(4, 6)
    k = rng.randint(2, 5)
    total = (n - 1) * k
    t = rng.randrange(3)
    ins = [
        f"时钟{n}点敲{n}下，{total}秒敲完，每隔几秒敲一下？",
        f"时钟敲{n}下，用了{total}秒，每两下之间相隔多少秒？",
        f"广场的大钟{n}点敲{n}下，{total}秒敲完，每隔多少秒敲一下？",
    ][t]
    lines = [f"间隔 = {n} - 1 = {n - 1}个",
             f"每次 = {total} ÷ {n - 1} = {k}秒"]
    return ins, lines, k


_reg("clock_strike", clock_strike)


# 43. saishi: round-robin games / line segments (1+2+...+(n-1)) ----------------------
def saishi(rng):
    n = rng.randint(4, 6)
    total = n * (n - 1) // 2
    tail = " + ".join(str(i) for i in range(n - 1, 0, -1))
    t = rng.randrange(3)
    if t == 0:
        ins = f"{n}个小朋友进行乒乓球比赛，每两人比赛一场，一共要赛多少场？"
        lines = [f"每人赛 = {n} - 1 = {n - 1}场",
                 f"场次 = {tail} = {total}场"]
    elif t == 1:
        ins = f"一条线段上有{n}个点（含两个端点），这条线段上共有多少条线段？"
        lines = [f"间隔 = {n} - 1 = {n - 1}个",
                 f"条数 = {tail} = {total}条"]
    else:
        ins = f"{n}个同学聚会，每两人握一次手，一共要握多少次手？"
        lines = [f"每人握 = {n} - 1 = {n - 1}次",
                 f"次数 = {tail} = {total}次"]
    return ins, lines, total


_reg("saishi", saishi)


# 44. ordered_pairs: directed pairs (cards / captain & vice) -------------------------
def ordered_pairs(rng):
    n = rng.randint(4, 6)
    t = rng.randrange(3)
    if t == 0:
        ins = f"{n}个小朋友，每两人互寄一张贺卡，一共寄了多少张贺卡？"
        lines = [f"贺卡 = {n} × {n - 1} = {n * (n - 1)}张"]
    elif t == 1:
        ins = f"从{n}名同学中选正、副班长各一名，有多少种不同选法？"
        lines = [f"选法 = {n} × {n - 1} = {n * (n - 1)}种"]
    else:
        ins = f"{n}个球队，每两队之间进行两场比赛，一共要赛多少场？"
        lines = [f"场次 = {n} × {n - 1} = {n * (n - 1)}场"]
    return ins, lines, n * (n - 1)


_reg("ordered_pairs", ordered_pairs)


# 45. peici: matching tops & bottoms (a x b) ------------------------------------------
def peici(rng):
    a = rng.randint(2, 4)
    b = rng.randint(2, 4)
    t = rng.randrange(3)
    ins = [
        f"小明有{a}件上衣和{b}条裤子，一件上衣配一条裤子，有多少种不同穿法？",
        f"早餐有{a}种点心和{b}种饮料，各选一种，有多少种不同搭配？",
        f"从家到学校有{a}条路，从学校到公园有{b}条路，从家经过学校到公园有多少种走法？",
    ][t]
    lines = [f"搭配 = {a} × {b} = {a * b}种"]
    return ins, lines, a * b


_reg("peici", peici)


# 46. permutation: n! arrangements ------------------------------------------------------
def permutation(rng):
    n = rng.randint(3, 4)
    total = 1
    for i in range(2, n + 1):
        total *= i
    expr = " × ".join(str(i) for i in range(n, 0, -1))
    t = rng.randrange(3)
    ins = [
        f"{n}个小朋友排成一排照相，有多少种不同排法？",
        f"用{n}个不同的数字排成一排，有多少种不同排法？",
        f"{n}本不同的书排成一排，有多少种不同排法？",
    ][t]
    lines = [f"排法 = {expr} = {total}种"]
    return ins, lines, total


_reg("permutation", permutation)


# 47. overlap: two sets with intersection (a + b - c) -----------------------------------
def overlap(rng):
    a = rng.randint(6, 15)
    b = rng.randint(6, 15)
    c = rng.randint(2, min(a, b) - 1)
    t = rng.randrange(3)
    if t == 0:
        ins = f"二（1）班参加美术组的有{a}人，参加音乐组的有{b}人，两组都参加的有{c}人，参加这两个组的共有多少人？"
        lines = [f"一共 = {a} + {b} - {c} = {a + b - c}人"]
    elif t == 1:
        ins = f"两块木板各长{a}厘米和{b}厘米，钉在一起重叠{c}厘米，钉好后的木板长多少厘米？"
        lines = [f"钉好 = {a} + {b} - {c} = {a + b - c}厘米"]
    else:
        ins = f"会打篮球的有{a}人，会踢足球的有{b}人，两样都会的有{c}人，会这两样的共有多少人？"
        lines = [f"一共 = {a} + {b} - {c} = {a + b - c}人"]
    return ins, lines, a + b - c


_reg("overlap", overlap)


# 48. perimeter_regular: regular polygon perimeter ----------------------------------------
def perimeter_regular(rng):
    k = rng.randint(3, 6)
    a = rng.randint(3, 15)
    for _ in range(50):
        if a * k <= 100:
            break
        a = rng.randint(3, 15)
    shape = {3: "等边三角形", 4: "正方形", 5: "正五边形", 6: "正六边形"}[k]
    t = rng.randrange(3)
    ins = [
        f"{shape}的边长是{a}厘米，它的周长是多少厘米？",
        f"一个{shape}，每条边长{a}厘米，周长是多少厘米？",
        f"{shape}花坛，边长{a}米，绕它走一圈是多少米？",
    ][t]
    unit = "厘米" if t < 2 else "米"
    lines = [f"周长 = {a} × {k} = {a * k}{unit}"]
    return ins, lines, a * k


_reg("perimeter_regular", perimeter_regular)


# 49. perimeter_rect: rectangle perimeter ((a + b) x 2) ------------------------------------
def perimeter_rect(rng):
    a = rng.randint(6, 20)
    b = rng.randint(4, a - 1)
    t = rng.randrange(3)
    ins = [
        f"长方形长{a}厘米，宽{b}厘米，它的周长是多少厘米？",
        f"一个长方形操场，长{a}米，宽{b}米，绕操场跑一圈是多少米？",
        f"长方形菜地长{a}米，宽{b}米，四周围上篱笆，篱笆长多少米？",
    ][t]
    unit = ["厘米", "米", "米"][t]
    lines = [f"长加宽 = {a} + {b} = {a + b}{unit}",
             f"周长 = {a + b} × 2 = {(a + b) * 2}{unit}"]
    return ins, lines, (a + b) * 2


_reg("perimeter_rect", perimeter_rect)


# 50. perimeter_reverse: perimeter divided into equal sides ---------------------------------
def perimeter_reverse(rng):
    k = rng.randint(3, 6)
    a = rng.randint(3, 15)
    for _ in range(50):
        if a * k <= 100:
            break
        a = rng.randint(3, 15)
    c = a * k
    shape = {3: "等边三角形", 4: "正方形", 5: "正五边形", 6: "正六边形"}[k]
    t = rng.randrange(3)
    ins = [
        f"{shape}的周长是{c}厘米，它的边长是多少厘米？",
        f"一个{shape}花坛，周长是{c}米，边长是多少米？",
        f"用{c}厘米长的铁丝围成一个{shape}，每条边长多少厘米？",
    ][t]
    unit = ["厘米", "米", "厘米"][t]
    lines = [f"边长 = {c} ÷ {k} = {a}{unit}"]
    return ins, lines, a


_reg("perimeter_reverse", perimeter_reverse)


# 51. width_from_perimeter: rectangle width from perimeter and length ------------------------
def width_from_perimeter(rng):
    a = rng.randint(6, 18)
    b = rng.randint(4, 12)
    c = 2 * (a + b)
    t = rng.randrange(3)
    ins = [
        f"长方形的周长是{c}厘米，长是{a}厘米，宽是多少厘米？",
        f"用一根{c}厘米长的铁丝围成一个长方形，长是{a}厘米，宽是多少厘米？",
        f"长方形菜地周长{c}米，长{a}米，宽是多少米？",
    ][t]
    unit = ["厘米", "厘米", "米"][t]
    lines = [f"两条长 = {a} × 2 = {2 * a}{unit}",
             f"剩下 = {c} - {2 * a} = {2 * b}{unit}",
             f"宽 = {2 * b} ÷ 2 = {b}{unit}"]
    return ins, lines, b


_reg("width_from_perimeter", width_from_perimeter)


# 52. rectangle_area: area of rectangle/square ----------------------------------------------
def rectangle_area(rng):
    t = rng.randrange(3)
    if t == 0:
        a = rng.randint(5, 15)
        b = rng.randint(4, 12)
        for _ in range(50):
            if a * b <= 100:
                break
            b = rng.randint(4, 12)
        ins = f"长方形长{a}厘米，宽{b}厘米，它的面积是多少平方厘米？"
        lines = [f"面积 = {a} × {b} = {a * b}平方厘米"]
        return ins, lines, a * b
    if t == 1:
        a = rng.randint(5, 12)
        ins = f"正方形边长{a}厘米，它的面积是多少平方厘米？"
        lines = [f"面积 = {a} × {a} = {a * a}平方厘米"]
        return ins, lines, a * a
    a = rng.randint(6, 15)
    b = rng.randint(5, 12)
    for _ in range(50):
        if a * b <= 100:
            break
        b = rng.randint(5, 12)
    ins = f"长方形菜地长{a}米，宽{b}米，面积是多少平方米？"
    lines = [f"面积 = {a} × {b} = {a * b}平方米"]
    return ins, lines, a * b


_reg("rectangle_area", rectangle_area)


# 53. area_reverse: area divided by one side -------------------------------------------------
def area_reverse(rng):
    b = rng.randint(4, 12)
    q = rng.randint(4, 12)
    a = b * q
    t = rng.randrange(3)
    ins = [
        f"长方形的面积是{a}平方厘米，长是{b}厘米，宽是多少厘米？",
        f"长方形菜地面积{a}平方米，宽{b}米，长是多少米？",
        f"教室地面面积{a}平方米，宽{b}米，长是多少米？",
    ][t]
    unit = ["厘米", "米", "米"][t]
    lines = [f"另一边 = {a} ÷ {b} = {q}{unit}"]
    return ins, lines, q


_reg("area_reverse", area_reverse)


# 54. unit_convert: simple unit conversions (x10/x12/x15/x24) ---------------------------------
def unit_convert(rng):
    table = [
        ("天", "小时", 24, (2, 4)),
        ("年", "个月", 12, (2, 8)),
        ("公斤", "斤", 2, (3, 10)),
        ("斤", "两", 10, (2, 10)),
        ("米", "分米", 10, (2, 9)),
        ("分米", "厘米", 10, (2, 9)),
        ("厘米", "毫米", 10, (2, 9)),
        ("元", "角", 10, (2, 9)),
        ("角", "分", 10, (2, 9)),
        ("刻", "分钟", 15, (2, 6)),
    ]
    src, dst, f, (lo, hi) = rng.choice(table)
    q = rng.randint(lo, hi)
    for _ in range(50):
        if q * f <= 100:
            break
        q = rng.randint(lo, hi)
    t = rng.randrange(3)
    if t < 2:
        ins = [f"{q}{src}等于多少{dst}？", f"{q}{src}是多少{dst}？"][t]
        lines = [f"结果 = {q} × {f} = {q * f}{dst}"]
        return ins, lines, q * f
    total = q * f
    ins = f"{total}{dst}是多少{src}？"
    lines = [f"结果 = {total} ÷ {f} = {q}{src}"]
    return ins, lines, q


_reg("unit_convert", unit_convert)


# 55. convert_sub: whole unit minus a part (1m - a cm etc.) -----------------------------------
def convert_sub(rng):
    t = rng.randrange(3)
    if t == 0:
        a = rng.randint(20, 80)
        ins = f"一根绳子长1米，剪去{a}厘米，还剩多少厘米？"
        lines = [f"剩下 = 100 - {a} = {100 - a}厘米"]
        return ins, lines, 100 - a
    if t == 1:
        a = rng.randint(2, 8)
        n = rng.choice(NAMES)
        ins = f"{n}有1元钱，买铅笔用去{a}角，还剩多少角？"
        lines = [f"剩下 = 10 - {a} = {10 - a}角"]
        return ins, lines, 10 - a
    a = rng.randint(10, 50)
    ins = f"一部电影1小时，已经放了{a}分钟，还有多少分钟放完？"
    lines = [f"剩下 = 60 - {a} = {60 - a}分钟"]
    return ins, lines, 60 - a


_reg("convert_sub", convert_sub)


# 56. elapsed: end minus start (hours) ---------------------------------------------------------
def elapsed(rng):
    a = rng.randint(7, 10)
    b = rng.randint(a + 1, 12)
    t = rng.randrange(3)
    ins = [
        f"电影上午{a}时开始，上午{b}时结束，放映了多少小时？",
        f"小明{a}时到校，{b}时放学，他在校多少小时？",
        f"火车{a}时出发，{b}时到达，路上行了多少小时？",
    ][t]
    lines = [f"经过 = {b} - {a} = {b - a}小时"]
    return ins, lines, b - a


_reg("elapsed", elapsed)


# 57. rate_time: rate times time (pages/distance/work) ------------------------------------------
def rate_time(rng):
    t = rng.randrange(3)
    if t == 0:
        a = rng.randint(5, 15)
        b = rng.randint(3, 8)
        n = rng.choice(NAMES)
        ins = f"{n}每天看{a}页故事书，看了{b}天，一共看了多少页？"
        lines = [f"总页数 = {a} × {b} = {a * b}页"]
        return ins, lines, a * b
    if t == 1:
        a = rng.randint(30, 50)
        b = rng.randint(2, 3)
        for _ in range(50):
            if a * b <= 100:
                break
            a = rng.randint(30, 50)
        ins = f"汽车每小时行{a}千米，行了{b}小时，一共行了多少千米？"
        lines = [f"路程 = {a} × {b} = {a * b}千米"]
        return ins, lines, a * b
    a = rng.randint(8, 15)
    b = rng.randint(4, 8)
    ins = f"修路队每天修{a}米路，修了{b}天，一共修了多少米？"
    lines = [f"总长 = {a} × {b} = {a * b}米"]
    return ins, lines, a * b


_reg("rate_time", rate_time)


# 58. tens_ones: place value / yuan composition (a x 10 + b) ------------------------------------
def tens_ones(rng):
    a = rng.randint(2, 9)
    b = rng.randint(1, 9)
    t = rng.randrange(3)
    if t == 0:
        ins = f"一个数，十位上是{a}，个位上是{b}，这个数是多少？"
        lines = [f"这个数 = {a} × 10 + {b} = {a * 10 + b}"]
    elif t == 1:
        ins = f"{a}张10元和{b}张1元，一共是多少元？"
        lines = [f"一共 = {a} × 10 + {b} = {a * 10 + b}元"]
    else:
        ins = f"一个数由{a}个十和{b}个一组成，这个数是多少？"
        lines = [f"这个数 = {a} × 10 + {b} = {a * 10 + b}"]
    return ins, lines, a * 10 + b


_reg("tens_ones", tens_ones)


if __name__ == "__main__":
    rng = random.Random(1)
    ok = 0
    from run_math_short import verify
    for _lvl, name, fn in PROGRAMS:
        for _ in range(30):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines} {ans}"
            ok += 1
    print(f"L1 ext2 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
