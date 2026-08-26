#!/usr/bin/env python3
"""L1 extension bank #3: 55 more single/two-step families (1-2 arithmetic steps).

Distinct from math_programs_l1.py, _ext1, _ext2. Each fn(rng) -> (instruction,
lines[list[str]], ans:int|Fraction). Every line is `label = expr = value[unit]`;
the last line's value must equal num(ans). >=4 phrasings per family; names/
objects/units from mathcommon pools; all numbers <= 100 (constants 180/360
excepted). Verified by run_math_short.verify.
"""

import math
import random
from fractions import Fraction

from mathcommon import NAMES, UNIT_N, UNIT_ZHI, num

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L1", name, fn))


# 1. age_future_past: age +/- years -------------------------------------------
def age_future_past(rng):
    a = rng.randint(5, 12)
    b = rng.randint(2, 9)
    n = rng.choice(NAMES)
    t = rng.randrange(4)
    if t in (0, 2):
        ans = a + b
        ins = [
            f"{n}今年{a}岁，{b}年后他多少岁？",
            f"{n}今年{a}岁，再过{b}年他几岁？",
        ][t // 2]
        lines = [f"年龄 = {a} + {b} = {ans}岁"]
    else:
        ans = a - b
        ins = [
            f"{n}今年{a}岁，{b}年前他多少岁？",
            f"{n}今年{a}岁，{b}年前他几岁？",
        ][(t - 1) // 2]
        lines = [f"年龄 = {a} - {b} = {ans}岁"]
    return ins, lines, ans


_reg("age_future_past", age_future_past)


# 2. age_sum_future: sum of two ages plus 2*years ------------------------------
def age_sum_future(rng):
    a = rng.randint(28, 40)
    b = rng.randint(5, 12)
    c = rng.randint(2, 8)
    n = rng.choice(NAMES)
    kin = rng.choice(["爸爸", "妈妈", "爷爷", "奶奶"])
    t = rng.randrange(4)
    s = a + b
    total = s + 2 * c
    ins = [
        f"{kin}今年{a}岁，{n}今年{b}岁，{c}年后两人一共多少岁？",
        f"{kin}今年{a}岁，{n}今年{b}岁，再过{c}年两人共多少岁？",
        f"{kin}{a}岁，{n}{b}岁，{c}年以后两人年龄和是多少岁？",
        f"{kin}今年{a}岁，{n}今年{b}岁，{c}年后他们俩共多少岁？",
    ][t]
    lines = [f"今年和 = {a} + {b} = {s}岁",
             f"一共 = {s} + {c} × 2 = {total}岁"]
    return ins, lines, total


_reg("age_sum_future", age_sum_future)


# 3. age_gap_invariant: age difference is invariant ----------------------------
def age_gap_invariant(rng):
    a = rng.randint(28, 40)
    b = rng.randint(5, 12)
    c = rng.randint(3, 9)
    n = rng.choice(NAMES)
    kin = rng.choice(["爸爸", "妈妈"])
    t = rng.randrange(4)
    gap = a - b
    if t == 1:
        ka, kb, op = a - c, b - c, "-"
    else:
        ka, kb, op = a + c, b + c, "+"
    ins = [
        f"{kin}今年{a}岁，{n}今年{b}岁，{c}年后{kin}比{n}大几岁？",
        f"{kin}今年{a}岁，{n}今年{b}岁，{c}年前{kin}比{n}大几岁？",
        f"{kin}{a}岁，{n}{b}岁，再过{c}年{kin}比{n}大多少岁？",
        f"{kin}今年{a}岁，{n}今年{b}岁，{c}年以后两人相差多少岁？",
    ][t]
    lines = [f"{kin} = {a} {op} {c} = {ka}岁",
             f"{n} = {b} {op} {c} = {kb}岁",
             f"相差 = {ka} - {kb} = {gap}岁"]
    return ins, lines, gap


_reg("age_gap_invariant", age_gap_invariant)


# 4. age_sum_target: years until combined age reaches target --------------------
def age_sum_target(rng):
    a = rng.randint(7, 12)
    b = rng.randint(5, 11)
    d = rng.randint(2, 9)
    c = a + b + 2 * d
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(4)
    ins = [
        f"{n1}今年{a}岁，{n2}今年{b}岁，当两人年龄和是{c}岁时，是几年后？",
        f"{n1}今年{a}岁，{n2}今年{b}岁，几年后两人年龄和正好是{c}岁？",
        f"{n1}{a}岁，{n2}{b}岁，当两人共{c}岁时，是几年以后？",
        f"{n1}今年{a}岁，{n2}今年{b}岁，多少年后两人年龄和为{c}岁？",
    ][t]
    lines = [f"今年和 = {a} + {b} = {a + b}岁",
             f"差 = {c} - {a + b} = {2 * d}岁",
             f"年数 = {2 * d} ÷ 2 = {d}年"]
    return ins, lines, d


_reg("age_sum_target", age_sum_target)


# 5. date_span: inclusive day count (b - a + 1) --------------------------------
def date_span(rng):
    a = rng.randint(1, 10)
    b = rng.randint(a + 3, a + 20)
    t = rng.randrange(4)
    days = b - a + 1
    ins = [
        f"从{a}日到{b}日（含首尾两天）共有多少天？",
        f"学校从{a}日放假到{b}日，一共放假多少天？",
        f"某月{a}日到{b}日（包括这两天）一共有多少天？",
        f"从{a}日到{b}日，一共经过多少天（含首尾）？",
    ][t]
    lines = [f"相差 = {b} - {a} = {b - a}天",
             f"天数 = {b - a} + 1 = {days}天"]
    return ins, lines, days


_reg("date_span", date_span)


# 6. count_digits_write: digits written from 1 to a -----------------------------
def count_digits_write(rng):
    a = rng.randint(10, 54)
    t = rng.randrange(4)
    two = a - 9
    total = 9 + 2 * two
    ins = [
        f"从1写到{a}，一共写了多少个数字？",
        f"小明从1开始写数，一直写到{a}，他一共写了多少个数字？",
        f"一本书的页码从1到{a}，编页码一共用了多少个数字？",
        f"从1连续写到{a}，总共写出多少个数字？",
    ][t]
    lines = [f"一位数 = 9 × 1 = 9个",
             f"两位数 = {a} - 9 = {two}个",
             f"数字 = 9 + {two} × 2 = {total}个"]
    return ins, lines, total


_reg("count_digits_write", count_digits_write)


# 7. triangular_pile: trapezoid stack sum (a+b)*layers/2 ------------------------
def triangular_pile(rng):
    a = rng.randint(2, 6)
    b = rng.randint(a + 2, a + 8)
    if (a + b) % 2 == 1 and (b - a + 1) % 2 == 1:
        b += 1
    layers = b - a + 1
    total = (a + b) * layers // 2
    obj = rng.choice(["钢管", "圆木", "铅笔", "水泥管"])
    unit = "支" if obj == "铅笔" else "根"
    t = rng.randrange(4)
    ins = [
        f"一堆{obj}堆成梯形，最上层{a}{unit}，最下层{b}{unit}，一共多少{unit}？",
        f"工地的{obj}堆成梯形，上层{a}{unit}，下层{b}{unit}，共有多少{unit}？",
        f"{obj}堆成梯形，最上面一层{a}{unit}，最下面一层{b}{unit}，一共多少{unit}？",
        f"一批{obj}堆成梯形，顶层{a}{unit}，底层{b}{unit}，总共有多少{unit}？",
    ][t]
    lines = [f"层数 = {b} - {a} + 1 = {layers}层",
             f"一共 = ({a} + {b}) × {layers} ÷ 2 = {total}{unit}"]
    return ins, lines, total


_reg("triangular_pile", triangular_pile)


# 8. fold_layers: 2^a layers after a folds --------------------------------------
def fold_layers(rng):
    a = rng.randint(3, 6)
    paper = rng.choice(["一张纸", "一张报纸", "一张彩纸", "一张卡纸", "一张手工纸",
                        "一张宣纸", "一张锡纸", "一块布", "一张皱纹纸", "一张蜡光纸",
                        "一张牛皮纸", "一张瓦楞纸"])
    t = rng.randrange(4)
    layers = 2 ** a
    expr = " × ".join(["2"] * a)
    ins = [
        f"{paper}对折{a}次后，一共有多少层？",
        f"把{paper}对折{a}次，层数是多少？",
        f"{paper}连续对折{a}次，折完后有几层？",
        f"{paper}对折{a}次，能得到多少层？",
    ][t]
    lines = [f"对折次数 = {a} = {a}次",
             f"层数 = {expr} = {layers}层"]
    return ins, lines, layers


_reg("fold_layers", fold_layers)


# 9. money_combo: a*5 + b*2 yuan ------------------------------------------------
def money_combo(rng):
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    t = rng.randrange(4)
    total = a * 5 + b * 2
    ins = [
        f"小明有{a}张5元和{b}张2元的纸币，一共多少元？",
        f"小红攒了{a}张5元、{b}张2元，她一共有多少元？",
        f"钱包里有{a}张5元和{b}张2元，合计多少元？",
        f"{a}张5元加{b}张2元，一共是多少元？",
    ][t]
    lines = [f"一共 = {a} × 5 + {b} × 2 = {total}元"]
    return ins, lines, total


_reg("money_combo", money_combo)


# 10. buy_get_free: buy b get 1 free, need a, how many to buy -------------------
def buy_get_free(rng):
    b = rng.randint(2, 4)
    q = rng.randint(2, 6)
    a = (b + 1) * q
    obj = rng.choice(["苹果", "笔记本", "铅笔", "包子", "面包", "气球", "贴纸", "卡片"])
    unit = rng.choice([UNIT_N, UNIT_ZHI])
    t = rng.randrange(4)
    buy = a - q
    ins = [
        f"商店做活动，每买{b}{unit}送1{unit}。小明需要{a}{unit}{obj}，最少要买几{unit}？",
        f"{obj}搞促销，买{b}{unit}送1{unit}，要得到{a}{unit}，只需买几{unit}？",
        f"文具店每买{b}{unit}{obj}送1{unit}，小红想要{a}{unit}，她要买几{unit}？",
        f"超市{obj}买{b}{unit}送1{unit}，小明拿了{a}{unit}，其中需要付钱的有几{unit}？",
    ][t]
    lines = [f"赠送 = {a} ÷ ({b} + 1) = {q}{unit}",
             f"要买 = {a} - {q} = {buy}{unit}"]
    return ins, lines, buy


_reg("buy_get_free", buy_get_free)


# 11. mixed_legs: two kinds of animals/vehicles legs ----------------------------
def mixed_legs(rng):
    a = rng.randint(3, 12)
    b = rng.randint(3, 12)
    t = rng.randrange(4)
    if t == 0:
        ans = a * 2 + b * 4
        ins = f"院子里有{a}只鸡和{b}只兔，一共有多少条腿？"
        lines = [f"一共 = {a} × 2 + {b} × 4 = {ans}条"]
    elif t == 1:
        ans = a * 2 + b * 3
        ins = f"车棚里有{a}辆自行车和{b}辆三轮车，一共有多少个轮子？"
        lines = [f"一共 = {a} × 2 + {b} × 3 = {ans}个"]
    elif t == 2:
        ans = a * 2 + b * 4
        ins = f"农场里有{a}只鸭和{b}只羊，一共有多少条腿？"
        lines = [f"一共 = {a} × 2 + {b} × 4 = {ans}条"]
    else:
        ans = a * 2 + b * 4
        ins = f"停车场有{a}辆两轮摩托车和{b}辆四轮汽车，一共有多少个轮子？"
        lines = [f"一共 = {a} × 2 + {b} × 4 = {ans}个"]
    return ins, lines, ans


_reg("mixed_legs", mixed_legs)


# 12. floor_groups: round-down grouping (去尾法) ---------------------------------
def floor_groups(rng):
    b = rng.randint(3, 8)
    q = rng.randint(3, 9)
    r = rng.randint(1, b - 1)
    a = b * q + r
    t = rng.randrange(4)
    if t == 0:
        ins = f"{a}名同学参加活动，每{b}人一组，最多能分几组？"
        unit, label, u2 = "组", "组数", "人"
    elif t == 1:
        ins = f"{a}元钱买每支{b}元的钢笔，最多能买几支？"
        unit, label, u2 = "支", "支数", "元"
    elif t == 2:
        ins = f"一块布长{a}米，做一件衣服用{b}米，最多能做几件？"
        unit, label, u2 = "件", "件数", "米"
    else:
        ins = f"{a}个苹果，每{b}个装一盒，最多能装满几盒？"
        unit, label, u2 = "盒", "盒数", "个"
    lines = [f"分掉 = {q} × {b} = {q * b}{u2}",
             f"剩下 = {a} - {q * b} = {r}{u2}",
             f"{label} = ({a} - {r}) ÷ {b} = {q}{unit}"]
    return ins, lines, q


_reg("floor_groups", floor_groups)


# 13. tree_planting: planting along a line / around a circle --------------------
def tree_planting(rng):
    b = rng.randint(3, 9)
    q = rng.randint(3, 9)
    a = b * q
    t = rng.randrange(4)
    if t == 0:
        ins = f"一条路长{a}米，每隔{b}米栽一棵树，两端都栽，一共要栽多少棵？"
        lines = [f"段数 = {a} ÷ {b} = {q}段",
                 f"棵数 = {q} + 1 = {q + 1}棵"]
        return ins, lines, q + 1
    if t == 1:
        ins = f"一条路长{a}米，每隔{b}米栽一棵树，一端栽一端不栽，一共栽多少棵？"
        lines = [f"棵数 = {a} ÷ {b} = {q}棵"]
        return ins, lines, q
    if t == 2:
        ins = f"一条路长{a}米，每隔{b}米栽一棵树，两端都不栽，一共要栽多少棵？"
        lines = [f"段数 = {a} ÷ {b} = {q}段",
                 f"棵数 = {q} - 1 = {q - 1}棵"]
        return ins, lines, q - 1
    ins = f"圆形池塘周长{a}米，每隔{b}米栽一棵树，一共要栽多少棵？"
    lines = [f"棵数 = {a} ÷ {b} = {q}棵"]
    return ins, lines, q


_reg("tree_planting", tree_planting)


# 14. square_side_area: side from square area -----------------------------------
def square_side_area(rng):
    b = rng.randint(4, 9)
    a = b * b
    obj, unit = rng.choice([
        ("正方形花坛", "平方米"), ("正方形地砖", "平方分米"),
        ("正方形手帕", "平方分米"), ("正方形桌面", "平方米"),
        ("正方形画纸", "平方分米"), ("正方形卡片", "平方厘米"),
        ("正方形棋盘", "平方米"), ("正方形镜框", "平方分米"),
    ])
    t = rng.randrange(4)
    ins = [
        f"{obj}的面积是{a}{unit}，它的边长是多少{unit.replace('平方', '')}？",
        f"{obj}面积{a}{unit}，边长是多少{unit.replace('平方', '')}？",
        f"一个{obj}的面积为{a}{unit}，边长是多少{unit.replace('平方', '')}？",
        f"{obj}占地{a}{unit}，它的边长是多少{unit.replace('平方', '')}？",
    ][t]
    lu = unit.replace("平方", "")
    lines = [f"面积 = {b} × {b} = {a}{unit}",
             f"边长 = {a} ÷ {b} = {b}{lu}"]
    return ins, lines, b


_reg("square_side_area", square_side_area)


# 15. triangle_area: base * height / 2 ------------------------------------------
def triangle_area(rng):
    a = rng.randint(6, 15)
    b = rng.randint(4, 12)
    if a % 2 == 1 and b % 2 == 1:
        b += 1
    area = a * b // 2
    t = rng.randrange(4)
    ins = [
        f"三角形的底是{a}厘米，高是{b}厘米，它的面积是多少平方厘米？",
        f"一个三角形底{a}厘米，高{b}厘米，面积是多少平方厘米？",
        f"三角形菜地的底是{a}米，高是{b}米，这块菜地的面积是多少平方米？",
        f"三角形底长{a}厘米，高{b}厘米，它的面积是多少？",
    ][t]
    unit = "平方米" if t == 2 else "平方厘米"
    lines = [f"面积 = {a} × {b} ÷ 2 = {area}{unit}"]
    return ins, lines, area


_reg("triangle_area", triangle_area)


# 16. trapezoid_area: (upper + lower) * height / 2 ------------------------------
def trapezoid_area(rng):
    a = rng.randint(3, 8)
    b = rng.randint(a + 2, a + 6)
    h = rng.randint(4, 8)
    if (a + b) % 2 == 1 and h % 2 == 1:
        h += 1
    area = (a + b) * h // 2
    t = rng.randrange(4)
    lu = "米" if t == 2 else "厘米"
    ins = [
        f"梯形的上底是{a}{lu}，下底是{b}{lu}，高是{h}{lu}，它的面积是多少平方{lu}？",
        f"一个梯形上底{a}{lu}，下底{b}{lu}，高{h}{lu}，面积是多少平方{lu}？",
        f"梯形菜地上底{a}米，下底{b}米，高{h}米，这块菜地的面积是多少平方米？",
        f"梯形上底{a}{lu}，下底{b}{lu}，高{h}{lu}，它的面积是多少？",
    ][t]
    unit = f"平方{lu}"
    lines = [f"上下底和 = {a} + {b} = {a + b}{lu}",
             f"面积 = {a + b} × {h} ÷ 2 = {area}{unit}"]
    return ins, lines, area


_reg("trapezoid_area", trapezoid_area)


# 17. triangle_angle: third angle of a triangle ---------------------------------
def triangle_angle(rng):
    a = rng.randint(30, 80)
    b = rng.randint(30, 150 - a)
    c = 180 - a - b
    t = rng.randrange(4)
    ins = [
        f"三角形的两个内角分别是{a}度和{b}度，第三个角是多少度？",
        f"一个三角形有两个角是{a}度和{b}度，另一个角是多少度？",
        f"三角形中两个内角为{a}度和{b}度，求第三个内角的度数。",
        f"三角形的两个角分别是{a}度和{b}度，剩下的角是多少度？",
    ][t]
    lines = [f"两角和 = {a} + {b} = {a + b}度",
             f"第三角 = 180 - {a + b} = {c}度"]
    return ins, lines, c


_reg("triangle_angle", triangle_angle)


# 18. isosceles_angle: base angle from vertex angle -----------------------------
def isosceles_angle(rng):
    a = rng.randint(10, 90)
    if a % 2 == 1:
        a += 1
    base = (180 - a) // 2
    t = rng.randrange(6)
    ins = [
        f"等腰三角形的顶角是{a}度，它的每个底角是多少度？",
        f"一个等腰三角形的顶角为{a}度，底角是多少度？",
        f"等腰三角形顶角{a}度，两个底角各是多少度？",
        f"等腰三角形的顶角是{a}度，它的一个底角是多少度？",
        f"等腰三角形的顶角等于{a}度，它的底角是多少度？",
        f"一个等腰三角形，顶角是{a}度，它的一个底角是多少度？",
    ][t]
    lines = [f"底角和 = 180 - {a} = {180 - a}度",
             f"每个底角 = {180 - a} ÷ 2 = {base}度"]
    return ins, lines, base


_reg("isosceles_angle", isosceles_angle)


# 19. isosceles_vertex: vertex angle from a base angle --------------------------
def isosceles_vertex(rng):
    a = rng.randint(20, 80)
    vertex = 180 - 2 * a
    t = rng.randrange(4)
    ins = [
        f"等腰三角形的一个底角是{a}度，它的顶角是多少度？",
        f"一个等腰三角形的底角为{a}度，顶角是多少度？",
        f"等腰三角形的底角是{a}度，顶角是多少度？",
        f"等腰三角形一个底角{a}度，它的顶角是多少度？",
    ][t]
    lines = [f"两底角 = {a} × 2 = {2 * a}度",
             f"顶角 = 180 - {2 * a} = {vertex}度"]
    return ins, lines, vertex


_reg("isosceles_vertex", isosceles_vertex)


# 20. quad_angle: fourth angle of a quadrilateral -------------------------------
def quad_angle(rng):
    a = rng.randint(60, 100)
    b = rng.randint(60, 100)
    c = rng.randint(60, 100)
    d = 360 - a - b - c
    t = rng.randrange(4)
    ins = [
        f"四边形的三个内角分别是{a}度、{b}度、{c}度，第四个角是多少度？",
        f"一个四边形有三个角是{a}度、{b}度、{c}度，另一个角是多少度？",
        f"四边形中三个内角为{a}度、{b}度和{c}度，求第四个角的度数。",
        f"四边形的三个角分别是{a}度、{b}度、{c}度，剩下的角是多少度？",
    ][t]
    lines = [f"三角和 = {a} + {b} + {c} = {a + b + c}度",
             f"第四角 = 360 - {a + b + c} = {d}度"]
    return ins, lines, d


_reg("quad_angle", quad_angle)


# 21. angle_supplement: supplementary / complementary angle ----------------------
def angle_supplement(rng):
    a = rng.randint(25, 80)
    t = rng.randrange(4)
    if t < 2:
        ans = 180 - a
        ins = [
            f"一个角是{a}度，它的补角是多少度？",
            f"角{a}度的补角是多少度？",
        ][t]
        lines = [f"补角 = 180 - {a} = {ans}度"]
    else:
        ans = 90 - a
        ins = [
            f"一个角是{a}度，它的余角是多少度？",
            f"角{a}度的余角是多少度？",
        ][t - 2]
        lines = [f"余角 = 90 - {a} = {ans}度"]
    return ins, lines, ans


_reg("angle_supplement", angle_supplement)


# 22. exterior_angle: exterior angle equals sum of two remote interiors ----------
def exterior_angle(rng):
    a = rng.randint(30, 80)
    b = rng.randint(30, 150 - a)
    s = a + b
    t = rng.randrange(4)
    ins = [
        f"三角形的两个内角分别是{a}度和{b}度，与第三个角相邻的外角是多少度？",
        f"三角形两个内角为{a}度和{b}度，第三个角的外角是多少度？",
        f"三角形中两个角是{a}度和{b}度，求第三个角的外角度数。",
        f"三角形的两个内角分别是{a}度、{b}度，第三个顶点处的外角是多少度？",
    ][t]
    lines = [f"外角 = {a} + {b} = {s}度"]
    return ins, lines, s


_reg("exterior_angle", exterior_angle)


# 23. matchstick_polygon: (k-1)*a + 1 sticks for a joined polygons ---------------
def matchstick_polygon(rng):
    a = rng.randint(3, 10)
    shape, k = rng.choice([("正方形", 4), ("三角形", 3), ("六边形", 6),
                           ("八边形", 8), ("五边形", 5), ("七边形", 7)])
    t = rng.randrange(4)
    total = (k - 1) * a + 1
    ins = [
        f"用火柴棒摆{a}个相连的{shape}（相邻两个共用一条边），一共需要多少根火柴棒？",
        f"摆{a}个连在一起的{shape}，每两个之间共用一条边，需要多少根火柴棒？",
        f"照样子摆{a}个相连的{shape}，一共要用多少根火柴棒？",
        f"摆{a}个相连的{shape}，相邻两个共用一条边，共需多少根火柴棒？",
    ][t]
    lines = [f"火柴 = {a} × {k - 1} + 1 = {total}根"]
    return ins, lines, total


_reg("matchstick_polygon", matchstick_polygon)


# 24. sum_diff: sum-and-difference problem ---------------------------------------
def sum_diff(rng):
    a = rng.randint(20, 60)
    b = rng.randint(2, 10)
    if (a + b) % 2 == 1:
        b += 1
    big = (a + b) // 2
    small = (a - b) // 2
    obj = rng.choice(["本书", "张卡片", "颗糖", "个苹果"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(4)
    ins = [
        f"{n1}和{n2}共有{a}{obj}，{n1}比{n2}多{b}{obj}，{n1}有多少{obj}？",
        f"甲乙两人共有{a}{obj}，甲比乙多{b}{obj}，甲有多少{obj}？",
        f"{n1}和{n2}一共{a}{obj}，{n2}比{n1}少{b}{obj}，{n1}有多少{obj}？",
        f"两筐共有{a}{obj}，第一筐比第二筐多{b}{obj}，第一筐有多少{obj}？",
    ][t]
    lines = [f"少的 = ({a} - {b}) ÷ 2 = {small}{obj}",
             f"多的 = ({a} + {b}) ÷ 2 = {big}{obj}"]
    return ins, lines, big


_reg("sum_diff", sum_diff)


# 25. ratio_part: sum-multiple problem, find the smaller part --------------------
def ratio_part(rng):
    k = rng.randint(2, 4)
    b = rng.randint(3, 12)
    a = (k + 1) * b
    obj = rng.choice(["本书", "张卡片", "颗糖", "支铅笔"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(4)
    ins = [
        f"{n1}和{n2}共有{a}{obj}，{n1}的{obj}是{n2}的{k}倍，{n2}有多少{obj}？",
        f"甲乙共有{a}{obj}，甲的数量是乙的{k}倍，乙有多少{obj}？",
        f"苹果树和梨树共{a}棵，苹果树是梨树的{k}倍，梨树有多少棵？",
        f"{n1}和{n2}一共{a}{obj}，{n1}是{n2}的{k}倍，{n2}有多少{obj}？",
    ][t]
    unit = "棵" if t == 2 else obj
    lines = [f"少的 = {a} ÷ ({k} + 1) = {b}{unit}"]
    return ins, lines, b


_reg("ratio_part", ratio_part)


# 26. ratio_diff_part: difference-multiple problem -------------------------------
def ratio_diff_part(rng):
    k = rng.randint(2, 4)
    b = rng.randint(3, 12)
    a = (k - 1) * b
    obj = rng.choice(["本书", "张卡片", "颗糖", "支铅笔"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(4)
    ins = [
        f"{n1}比{n2}多{a}{obj}，{n1}的{obj}是{n2}的{k}倍，{n2}有多少{obj}？",
        f"甲比乙多{a}{obj}，甲的数量是乙的{k}倍，乙有多少{obj}？",
        f"苹果树比梨树多{a}棵，苹果树是梨树的{k}倍，梨树有多少棵？",
        f"{n1}比{n2}多{a}{obj}，{n1}是{n2}的{k}倍，{n2}有多少{obj}？",
    ][t]
    unit = "棵" if t == 2 else obj
    lines = [f"少的 = {a} ÷ ({k} - 1) = {b}{unit}"]
    return ins, lines, b


_reg("ratio_diff_part", ratio_diff_part)


# 27. missing_term_avg: last score given average and prior sum -------------------
def missing_term_avg(rng):
    k = rng.randint(3, 5)
    a = rng.randint(12, 20)
    d = rng.randint(1, 4)
    b = (k - 1) * (a - d)
    last = k * a - b
    t = rng.randrange(4)
    ins = [
        f"小明{k}次跳绳的平均成绩是每次{a}下，前{k - 1}次一共跳了{b}下，第{k}次跳了多少下？",
        f"小华{k}次测验平均每次{a}分，前{k - 1}次的总分是{b}分，最后一次得多少分？",
        f"小丽{k}次口算平均每次{a}题，前{k - 1}次共做{b}题，第{k}次做了多少题？",
        f"小军{k}次拍球平均每次{a}下，前{k - 1}次共拍{b}下，第{k}次拍了多少下？",
    ][t]
    unit = ["下", "分", "题", "下"][t]
    lines = [f"前几次 = {k} - 1 = {k - 1}次",
             f"总分 = {k} × {a} = {k * a}{unit}",
             f"第{k}次 = {k * a} - {b} = {last}{unit}"]
    return ins, lines, last


_reg("missing_term_avg", missing_term_avg)


# 28. weighted_avg: mix two kinds, average price ---------------------------------
def weighted_avg(rng):
    a = rng.randint(2, 4)
    c = rng.randint(2, 4)
    b = rng.randint(5, 12)
    d = rng.randint(5, 12)
    for _ in range(50):
        if (a * b + c * d) % (a + c) == 0:
            break
        b = rng.randint(5, 12)
        d = rng.randint(5, 12)
    total_price = a * b + c * d
    avg = total_price // (a + c)
    t = rng.randrange(4)
    ins = [
        f"甲种糖{a}千克，每千克{b}元；乙种糖{c}千克，每千克{d}元。混合后平均每千克多少元？",
        f"买{a}千克苹果每千克{b}元，又买{c}千克梨每千克{d}元，平均每千克水果多少元？",
        f"{a}千克甲糖和{c}千克乙糖混合，甲糖每千克{b}元，乙糖每千克{d}元，混合糖每千克多少元？",
        f"商店运来{a}千克奶糖（每千克{b}元）和{c}千克水果糖（每千克{d}元），平均每千克多少元？",
    ][t]
    lines = [f"总价 = {a} × {b} + {c} × {d} = {total_price}元",
             f"总重 = {a} + {c} = {a + c}千克",
             f"平均 = {total_price} ÷ {a + c} = {avg}元"]
    return ins, lines, avg


_reg("weighted_avg", weighted_avg)


# 29. combined_rate: (a + b) * c combined work / meeting -------------------------
def combined_rate(rng):
    a = rng.randint(8, 15)
    b = rng.randint(6, 12)
    c = rng.randint(2, 4)
    for _ in range(50):
        if (a + b) * c <= 100:
            break
        a = rng.randint(8, 15)
        b = rng.randint(6, 12)
    total = (a + b) * c
    t = rng.randrange(4)
    if t == 0:
        ins = f"甲每小时加工{a}个零件，乙每小时加工{b}个，两人合做{c}小时，一共加工多少个？"
        unit = "个"
    elif t == 1:
        ins = f"甲队每天修{a}米路，乙队每天修{b}米，两队合修{c}天，一共修多少米？"
        unit = "米"
    elif t == 2:
        ins = f"甲乙两人从两地相向而行，甲每小时行{a}千米，乙每小时行{b}千米，{c}小时后相遇，两地相距多少千米？"
        unit = "千米"
    else:
        ins = f"甲管每小时注水{a}吨，乙管每小时注水{b}吨，两管同时开{c}小时，共注水多少吨？"
        unit = "吨"
    lines = [f"每小时 = {a} + {b} = {a + b}{unit}",
             f"一共 = {a + b} × {c} = {total}{unit}"]
    return ins, lines, total


_reg("combined_rate", combined_rate)


# 30. chase_gap: (a - b) * c same-direction distance gap --------------------------
def chase_gap(rng):
    a = rng.randint(12, 20)
    b = rng.randint(6, 10)
    c = rng.randint(3, 8)
    for _ in range(50):
        if (a - b) * c <= 100:
            break
        a = rng.randint(12, 20)
        b = rng.randint(6, 10)
        c = rng.randint(3, 8)
    gap = (a - b) * c
    t = rng.randrange(4)
    ins = [
        f"甲每小时行{a}千米，乙每小时行{b}千米，两人同时同地同向出发，{c}小时后相距多少千米？",
        f"快车每小时行{a}千米，慢车每小时行{b}千米，同向而行{c}小时，快车比慢车多行多少千米？",
        f"小明每分钟走{a}米，小红每分钟走{b}米，同方向走{c}分钟后，两人相距多少米？",
        f"甲船每小时行{a}千米，乙船每小时行{b}千米，同时同向开出{c}小时，两船相距多少千米？",
    ][t]
    unit = "米" if t == 2 else "千米"
    lines = [f"速度差 = {a} - {b} = {a - b}{unit}",
             f"相距 = {a - b} × {c} = {gap}{unit}"]
    return ins, lines, gap


_reg("chase_gap", chase_gap)


# 31. train_tunnel: (train + tunnel) / speed = time ------------------------------
def train_tunnel(rng):
    a = rng.randint(20, 45)
    b = rng.randint(20, 45)
    c = rng.randint(4, 9)
    for _ in range(50):
        if (a + b) % c == 0 and a + b <= 100:
            break
        a = rng.randint(20, 45)
        b = rng.randint(20, 45)
        c = rng.randint(4, 9)
    s = a + b
    time = s // c
    t = rng.randrange(4)
    ins = [
        f"一列火车长{a}米，隧道长{b}米，火车每秒行{c}米，完全通过隧道要多少秒？",
        f"火车长{a}米，一座大桥长{b}米，火车每秒行驶{c}米，通过大桥需要多少秒？",
        f"一列长{a}米的火车以每秒{c}米的速度通过长{b}米的隧道，要多少秒？",
        f"火车长{a}米，隧道长{b}米，每秒行{c}米，从车头进入到车尾离开要多少秒？",
    ][t]
    lines = [f"路程 = {a} + {b} = {s}米",
             f"时间 = {s} ÷ {c} = {time}秒"]
    return ins, lines, time


_reg("train_tunnel", train_tunnel)


# 32. taxi_fare: flag-down + over-distance fare ----------------------------------
def taxi_fare(rng):
    a = rng.randint(8, 12)
    b = rng.randint(3, 4)
    c = rng.randint(2, 3)
    d = rng.randint(b + 2, b + 7)
    fare = a + (d - b) * c
    t = rng.randrange(4)
    ins = [
        f"出租车起步价{a}元（含{b}千米），超过部分每千米{c}元，行{d}千米要付多少元？",
        f"某市出租车起步价{a}元，{b}千米以内都是{a}元，超过{b}千米每千米{c}元，行{d}千米应付多少元？",
        f"打车起步价{a}元（{b}千米内），超出后每千米{c}元，小明行了{d}千米，共付多少元？",
        f"出租车{b}千米内收费{a}元，超过{b}千米每千米收{c}元，行{d}千米收费多少元？",
    ][t]
    lines = [f"超过 = {d} - {b} = {d - b}千米",
             f"加价 = {d - b} × {c} = {(d - b) * c}元",
             f"车费 = {a} + {(d - b) * c} = {fare}元"]
    return ins, lines, fare


_reg("taxi_fare", taxi_fare)


# 33. profit_total: (sell - cost) * count ----------------------------------------
def profit_total(rng):
    a = rng.randint(10, 25)
    b = rng.randint(a + 3, a + 15)
    c = rng.randint(3, 8)
    for _ in range(50):
        if (b - a) * c <= 100:
            break
        b = rng.randint(a + 3, a + 15)
        c = rng.randint(3, 8)
    profit = (b - a) * c
    obj = rng.choice(["玩具", "文具", "书包", "笔记本", "钢笔"])
    t = rng.randrange(4)
    ins = [
        f"商店每个{obj}的进价是{a}元，售价是{b}元，卖出{c}个，一共赚多少元？",
        f"一件{obj}进价{a}元，卖{b}元，老板卖出{c}件，共赚多少元？",
        f"超市购进{obj}每个{a}元，按每个{b}元出售，卖出{c}个，盈利多少元？",
        f"{obj}的批发价是{a}元，零售价是{b}元，卖出{c}个，一共赚多少元？",
    ][t]
    lines = [f"每个赚 = {b} - {a} = {b - a}元",
             f"一共 = {b - a} × {c} = {profit}元"]
    return ins, lines, profit


_reg("profit_total", profit_total)


# 34. save_months: (price - saved) / monthly = months ----------------------------
def save_months(rng):
    a = rng.randint(5, 12)
    d = rng.randint(2, 6)
    c = rng.randint(10, 28)
    b = c + a * d
    n = rng.choice(NAMES)
    obj = rng.choice(["玩具", "书包", "滑板", "拼图", "游戏机"])
    t = rng.randrange(4)
    ins = [
        f"{n}每月存{a}元，想买一个{b}元的{obj}，已经存了{c}元，还要存几个月？",
        f"{n}想买{b}元的{obj}，他每月存{a}元，现已存{c}元，还需存几个月？",
        f"一个{obj}售价{b}元，{n}每月能存{a}元，已经存了{c}元，再过几个月能买到？",
        f"{n}计划买{b}元的{obj}，每月存{a}元，已存{c}元，还要存几个月才够？",
    ][t]
    lines = [f"还差 = {b} - {c} = {b - c}元",
             f"月数 = {b - c} ÷ {a} = {d}个月"]
    return ins, lines, d


_reg("save_months", save_months)


# 35. plan_actual_diff: planned vs actual daily rate ------------------------------
def plan_actual_diff(rng):
    a = rng.randint(4, 12)
    b = rng.randint(4, 9)
    c = rng.randint(2, b - 1)
    for _ in range(50):
        if (a * b) % c == 0:
            break
        a = rng.randint(4, 12)
        b = rng.randint(4, 9)
        c = rng.randint(2, b - 1)
    total = a * b
    actual = total // c
    diff = actual - a
    t = rng.randrange(4)
    ins = [
        f"小明计划每天看{a}页故事书，{b}天正好看完。实际{c}天看完，实际每天比计划多看多少页？",
        f"一批零件计划每天做{a}个，{b}天完成。实际{c}天完成，实际每天比计划多做多少个？",
        f"修路队计划每天修{a}米，{b}天修完。实际{c}天修完，实际每天比计划多修多少米？",
        f"一本书计划每天读{a}页，{b}天读完。实际{c}天就读完了，实际每天比计划多读多少页？",
    ][t]
    unit = ["页", "个", "米", "页"][t]
    lines = [f"总量 = {a} × {b} = {total}{unit}",
             f"实际每天 = {total} ÷ {c} = {actual}{unit}",
             f"多看 = {actual} - {a} = {diff}{unit}"]
    return ins, lines, diff


_reg("plan_actual_diff", plan_actual_diff)


# 36. return_speed: distance there / time back = return speed ---------------------
def return_speed(rng):
    a = rng.randint(6, 10)
    b = rng.randint(3, 6)
    c = rng.randint(2, 8)
    for _ in range(50):
        if (a * b) % c == 0 and c != b:
            break
        a = rng.randint(6, 10)
        b = rng.randint(3, 6)
        c = rng.randint(2, 8)
    s = a * b
    v = s // c
    t = rng.randrange(4)
    ins = [
        f"小明去时每小时行{a}千米，{b}小时到达；原路返回用了{c}小时，返回时每小时行多少千米？",
        f"一辆汽车从甲地到乙地每小时行{a}千米，{b}小时到达。返回时用了{c}小时，返回每小时行多少千米？",
        f"小明骑车去公园每小时行{a}千米，{b}小时到达，步行返回用了{c}小时，步行每小时行多少千米？",
        f"从家到县城，去时每小时{a}千米，{b}小时到达；回来时用了{c}小时，回来每小时行多少千米？",
    ][t]
    lines = [f"路程 = {a} × {b} = {s}千米",
             f"返回速度 = {s} ÷ {c} = {v}千米"]
    return ins, lines, v


_reg("return_speed", return_speed)


# 37. next_page_start: pages read + 1 = next page ---------------------------------
def next_page_start(rng):
    a = rng.randint(6, 12)
    b = rng.randint(3, 8)
    n = rng.choice(NAMES)
    t = rng.randrange(4)
    seen = a * b
    nxt = seen + 1
    ins = [
        f"{n}每天看{a}页故事书，看了{b}天，接下来应从第几页开始看？",
        f"一本故事书，{n}每天看{a}页，看了{b}天，下一天从第几页看起？",
        f"{n}看一本书，每天{a}页，看了{b}天，接着该从第几页开始看？",
        f"故事书每天看{a}页，{n}看了{b}天，再看时从第几页看起？",
    ][t]
    lines = [f"已看 = {a} × {b} = {seen}页",
             f"下一页 = {seen} + 1 = {nxt}页"]
    return ins, lines, nxt


_reg("next_page_start", next_page_start)


# 38. remaining_days: leftover pages / per day = days left ------------------------
def remaining_days(rng):
    b = rng.randint(6, 10)
    c = rng.randint(3, 6)
    d = rng.randint(2, 5)
    a = b * (c + d)
    for _ in range(50):
        if a <= 100:
            break
        b = rng.randint(6, 10)
        c = rng.randint(3, 6)
        d = rng.randint(2, 5)
        a = b * (c + d)
    seen = b * c
    left = a - seen
    t = rng.randrange(4)
    ins = [
        f"一本故事书有{a}页，小明每天看{b}页，看了{c}天，余下的还要看几天？",
        f"一本书共{a}页，每天看{b}页，已经看了{c}天，剩下的还要几天看完？",
        f"故事书{a}页，小明每天读{b}页，读了{c}天后，余下的还需几天读完？",
        f"一本{a}页的书，每天看{b}页，看了{c}天，照这样，余下的还要看几天？",
    ][t]
    lines = [f"已看 = {b} × {c} = {seen}页",
             f"剩下 = {a} - {seen} = {left}页",
             f"还要 = {left} ÷ {b} = {d}天"]
    return ins, lines, d


_reg("remaining_days", remaining_days)


# 39. transfer_compare: after giving c, how much more does A have -----------------
def transfer_compare(rng):
    a = rng.randint(30, 45)
    b = rng.randint(12, 22)
    c = rng.randint(2, 5)
    for _ in range(50):
        if a - b - 2 * c >= 1:
            break
        a = rng.randint(30, 45)
        b = rng.randint(12, 22)
        c = rng.randint(2, 5)
    na = a - c
    nb = b + c
    diff = na - nb
    obj = rng.choice(["张卡片", "颗糖", "本书", "个苹果"])
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(4)
    ins = [
        f"{n1}有{a}{obj}，{n2}有{b}{obj}，{n1}给{n2}{c}{obj}后，{n1}比{n2}多多少{obj}？",
        f"甲筐有{a}千克苹果，乙筐有{b}千克，从甲筐拿{c}千克到乙筐，甲筐比乙筐多多少千克？",
        f"{n1}有{a}{obj}，{n2}有{b}{obj}，{n1}送给{n2}{c}{obj}后，两人相差多少{obj}？",
        f"哥哥有{a}{obj}，弟弟有{b}{obj}，哥哥给弟弟{c}{obj}后，哥哥比弟弟多多少{obj}？",
    ][t]
    unit = "千克" if t == 1 else obj
    la = "甲筐" if t == 1 else ("哥哥" if t == 3 else n1)
    lb = "乙筐" if t == 1 else ("弟弟" if t == 3 else n2)
    lines = [f"{la} = {a} - {c} = {na}{unit}",
             f"{lb} = {b} + {c} = {nb}{unit}",
             f"多 = {na} - {nb} = {diff}{unit}"]
    return ins, lines, diff


_reg("transfer_compare", transfer_compare)


# 40. calc_mistake: misread addend/subtrahend, correct the result -----------------
def calc_mistake(rng):
    t = rng.randrange(4)
    if t < 2:
        a = rng.randint(12, 35)
        b = a + rng.randint(1, 8)
        if b > 35:
            b = a - rng.randint(1, 8)
        d = rng.randint(25, 55)
        c = b + d
        ans = c - b + a
        ins = [
            f"小明做加法，把一个加数{a}错看成{b}，得到的和是{c}，正确的和是多少？",
            f"计算加法时，把加数{a}看成了{b}，算得和为{c}，正确的和应是多少？",
        ][t]
        lines = [f"另一加数 = {c} - {b} = {c - b}",
                 f"正确和 = {c - b} + {a} = {ans}"]
    else:
        a = rng.randint(12, 35)
        b = a + rng.randint(1, 8)
        if b > 35:
            b = a - rng.randint(1, 8)
        d = rng.randint(45, 80)
        c = d - b
        ans = c + b - a
        ins = [
            f"小明做减法，把减数{a}错看成{b}，得到的差是{c}，正确的差是多少？",
            f"计算减法时，把减数{a}看成了{b}，算得差为{c}，正确的差应是多少？",
        ][t - 2]
        lines = [f"被减数 = {c} + {b} = {c + b}",
                 f"正确差 = {c + b} - {a} = {ans}"]
    return ins, lines, ans


_reg("calc_mistake", calc_mistake)


# 41. multiply_mistake: misread factor, product grew by c ------------------------
def multiply_mistake(rng):
    a = rng.randint(3, 8)
    b = a + rng.randint(2, 5)
    d = rng.randint(3, 9)
    c = (b - a) * d
    t = rng.randrange(4)
    ins = [
        f"小明做乘法，把一个乘数{a}错看成{b}，结果积比正确答案多了{c}，另一个乘数是多少？",
        f"计算乘法时，把乘数{a}看成了{b}，积多了{c}，另一个乘数是多少？",
        f"小华做乘法，把一个因数{a}写成{b}，积比原来多{c}，另一个因数是多少？",
        f"乘法中把{a}看成{b}，得到的积比正确积多{c}，另一个乘数是几？",
    ][t]
    lines = [f"多了 = {b} - {a} = {b - a}",
             f"另一乘数 = {c} ÷ {b - a} = {d}"]
    return ins, lines, d


_reg("multiply_mistake", multiply_mistake)


# 42. consecutive_sum: k consecutive naturals sum to a, find the largest ----------
def consecutive_sum(rng):
    k = rng.choice([3, 5, 7])
    m = rng.randint(3, 100 // k)
    a = k * m
    big = m + (k - 1) // 2
    t = rng.randrange(4)
    ins = [
        f"{k}个连续自然数的和是{a}，其中最大的一个是多少？",
        f"有{k}个连续的自然数，它们的和是{a}，最大的数是多少？",
        f"{k}个连续自然数相加，和为{a}，这{k}个数中最大的是多少？",
        f"已知{k}个连续自然数的和是{a}，最大的那个数是几？",
    ][t]
    lines = [f"中间 = {a} ÷ {k} = {m}",
             f"最大 = {m} + {(k - 1) // 2} = {big}"]
    return ins, lines, big


_reg("consecutive_sum", consecutive_sum)


# 43. reverse_digits: swap tens and ones ------------------------------------------
def reverse_digits(rng):
    a = rng.randint(2, 9)
    b = a
    for _ in range(50):
        b = rng.randint(1, 9)
        if b != a:
            break
    new = b * 10 + a
    t = rng.randrange(4)
    ins = [
        f"一个两位数，十位上是{a}，个位上是{b}，把十位和个位交换后得到的数是多少？",
        f"两位数的十位数字是{a}，个位数字是{b}，交换两个数字的位置后是多少？",
        f"一个数由{a}个十和{b}个一组成，把它的十位和个位颠倒过来是多少？",
        f"十位是{a}、个位是{b}的两位数，倒过来写是多少？",
    ][t]
    lines = [f"新数 = {b} × 10 + {a} = {new}"]
    return ins, lines, new


_reg("reverse_digits", reverse_digits)


# 44. tiles_side: floor tiles from side length ------------------------------------
def tiles_side(rng):
    c = rng.randint(2, 3)
    a = c * rng.randint(3, 9)
    b = c * rng.randint(3, 9)
    na = a // c
    nb = b // c
    total = na * nb
    t = rng.randrange(4)
    ins = [
        f"教室长{a}米，宽{b}米，用边长{c}米的方砖铺地，需要多少块方砖？",
        f"一个房间长{a}米、宽{b}米，铺边长{c}米的正方形地砖，共需多少块？",
        f"长方形地面长{a}米，宽{b}米，用边长{c}米的方砖铺满，要多少块？",
        f"客厅长{a}米，宽{b}米，地砖是边长{c}米的正方形，一共需要多少块？",
    ][t]
    lines = [f"长块 = {a} ÷ {c} = {na}块",
             f"宽块 = {b} ÷ {c} = {nb}块",
             f"总块 = {na} × {nb} = {total}块"]
    return ins, lines, total


_reg("tiles_side", tiles_side)


# 45. fence_wall: fence length with one side against a wall ------------------------
def fence_wall(rng):
    a = rng.randint(10, 25)
    b = rng.randint(5, 12)
    t = rng.randrange(4)
    if t == 0:
        ans = a + 2 * b
        ins = f"长方形菜地长{a}米，宽{b}米，长边靠墙，其余三面围篱笆，篱笆长多少米？"
        lines = [f"篱笆 = {a} + {b} × 2 = {ans}米"]
    elif t == 1:
        ans = 2 * a + b
        ins = f"长方形菜地长{a}米，宽{b}米，宽边靠墙，其余三面围篱笆，篱笆长多少米？"
        lines = [f"篱笆 = {a} × 2 + {b} = {ans}米"]
    elif t == 2:
        ans = 3 * a
        ins = f"正方形菜地边长{a}米，一面靠墙，另外三面围篱笆，篱笆长多少米？"
        lines = [f"篱笆 = {a} × 3 = {ans}米"]
    else:
        ans = a + 2 * b
        ins = f"长方形菜地长{a}米，宽{b}米，一面靠墙，篱笆至少长多少米？"
        lines = [f"篱笆 = {a} + {b} × 2 = {ans}米"]
    return ins, lines, ans


_reg("fence_wall", fence_wall)


# 46. square_perimeter_area: area from square perimeter ----------------------------
def square_perimeter_area(rng):
    b = rng.randint(5, 12)
    p = 4 * b
    area = b * b
    obj = rng.choice(["正方形花坛", "正方形地砖", "正方形手帕", "正方形桌面",
                      "正方形画纸", "正方形卡片", "正方形棋盘", "正方形镜框"])
    t = rng.randrange(4)
    ins = [
        f"{obj}的周长是{p}米，它的面积是多少平方米？",
        f"一个{obj}的周长为{p}米，面积是多少平方米？",
        f"{obj}四周的总长是{p}米，它的面积是多少平方米？",
        f"用一根长{p}米的绳子围成一个{obj}，它的面积是多少平方米？",
    ][t]
    lines = [f"边长 = {p} ÷ 4 = {b}米",
             f"面积 = {b} × {b} = {area}平方米"]
    return ins, lines, area


_reg("square_perimeter_area", square_perimeter_area)


# 47. rect_perimeter_area: area from rectangle perimeter and length ----------------
def rect_perimeter_area(rng):
    a = rng.randint(12, 20) * 2
    b = rng.randint(6, 10)
    for _ in range(50):
        if a // 2 > b:
            break
        a = rng.randint(12, 20) * 2
        b = rng.randint(6, 10)
    w = a // 2 - b
    area = b * w
    t = rng.randrange(4)
    lu = "米" if t == 2 else "厘米"
    ins = [
        f"长方形的周长是{a}{lu}，长是{b}{lu}，它的面积是多少平方{lu}？",
        f"一个长方形周长{a}{lu}，长{b}{lu}，面积是多少平方{lu}？",
        f"长方形菜地周长{a}米，长{b}米，它的面积是多少平方米？",
        f"用长{a}{lu}的铁丝围成一个长方形，长是{b}{lu}，面积是多少平方{lu}？",
    ][t]
    unit = f"平方{lu}"
    lines = [f"长宽和 = {a} ÷ 2 = {a // 2}{lu}",
             f"宽 = {a // 2} - {b} = {w}{lu}",
             f"面积 = {b} × {w} = {area}{unit}"]
    return ins, lines, area


_reg("rect_perimeter_area", rect_perimeter_area)


# 48. fraction_remaining: whole minus 1/k of it -----------------------------------
def fraction_remaining(rng):
    b = rng.randint(3, 6)
    q = rng.randint(3, 12)
    a = b * q
    left = a - q
    obj = rng.choice(["一袋大米", "一袋面粉", "一桶油", "一袋苹果", "一筐梨"])
    t = rng.randrange(4)
    ins = [
        f"{obj}重{a}千克，吃掉了它的1/{b}，还剩多少千克？",
        f"{obj}有{a}千克，用去它的1/{b}，还剩多少千克？",
        f"{obj}共{a}千克，吃了1/{b}，剩下多少千克？",
        f"{obj}重{a}千克，用了它的1/{b}，还余下多少千克？",
    ][t]
    lines = [f"吃了 = {a} ÷ {b} = {q}千克",
             f"剩下 = {a} - {q} = {left}千克"]
    return ins, lines, left


_reg("fraction_remaining", fraction_remaining)


# 49. fraction_land: whole minus two unit fractions of it --------------------------
def fraction_land(rng):
    pairs = [(3, 4), (3, 6), (4, 6), (3, 5), (4, 5), (5, 6), (3, 8), (4, 8), (6, 8)]
    b, c = rng.choice(pairs)
    l = b * c // math.gcd(b, c)
    k = rng.randint(1, 100 // l)
    a = l * k
    x = a // b
    y = a // c
    left = a - x - y
    crop1, crop2 = rng.sample(["西红柿", "黄瓜", "茄子", "辣椒", "白菜", "萝卜"], 2)
    t = rng.randrange(4)
    ins = [
        f"一块地{a}平方米，其中1/{b}种{crop1}，1/{c}种{crop2}，剩下的种白菜，白菜种了多少平方米？",
        f"一块菜地{a}平方米，1/{b}种{crop1}，1/{c}种{crop2}，其余种白菜，白菜有多少平方米？",
        f"农场有{a}平方米地，用1/{b}种{crop1}，1/{c}种{crop2}，剩下的种白菜，种白菜多少平方米？",
        f"一块地共{a}平方米，1/{b}种{crop1}，1/{c}种{crop2}，剩下的种白菜，白菜地是多少平方米？",
    ][t]
    lines = [f"{crop1} = {a} ÷ {b} = {x}平方米",
             f"{crop2} = {a} ÷ {c} = {y}平方米",
             f"白菜 = {a} - {x} - {y} = {left}平方米"]
    return ins, lines, left


_reg("fraction_land", fraction_land)


# 50. inverse_two_step: (c - b) / a = x -------------------------------------------
def inverse_two_step(rng):
    a = rng.randint(3, 7)
    x = rng.randint(3, 10)
    b = rng.randint(5, 25)
    c = a * x + b
    t = rng.randrange(4)
    ins = [
        f"一个数乘{a}，再加上{b}，得到{c}，这个数是多少？",
        f"一个数的{a}倍加上{b}等于{c}，这个数是多少？",
        f"某数的{a}倍比{c}少{b}，某数是多少？",
        f"一个数乘{a}后加上{b}，结果是{c}，求这个数。",
    ][t]
    lines = [f"这个数 = ({c} - {b}) ÷ {a} = {x}"]
    return ins, lines, x


_reg("inverse_two_step", inverse_two_step)


# 51. avg_pooled: average of two pooled sums over 5 tries --------------------------
def avg_pooled(rng):
    a = rng.randint(20, 60)
    b = rng.randint(15, 40)
    for _ in range(50):
        if (a + b) % 5 == 0:
            break
        a = rng.randint(20, 60)
        b = rng.randint(15, 40)
    s = a + b
    avg = s // 5
    t = rng.randrange(4)
    ins = [
        f"小明前3次跳绳共跳{a}下，后2次共跳{b}下，平均每次跳多少下？",
        f"小华前3次数学测验共得{a}分，后2次共得{b}分，平均每次得多少分？",
        f"小丽前3天看书{a}页，后2天看书{b}页，平均每天看多少页？",
        f"小军前3次拍球共{a}下，后2次共{b}下，平均每次拍多少下？",
    ][t]
    unit = ["下", "分", "页", "下"][t]
    lines = [f"总和 = {a} + {b} = {s}{unit}",
             f"次数 = 3 + 2 = 5次",
             f"平均 = {s} ÷ 5 = {avg}{unit}"]
    return ins, lines, avg


_reg("avg_pooled", avg_pooled)


# 52. two_leg_distance: a*b + c*d two-part total ----------------------------------
def two_leg_distance(rng):
    a = rng.randint(3, 9)
    b = rng.randint(3, 9)
    c = rng.randint(2, 8)
    d = rng.randint(2, 8)
    for _ in range(50):
        if a * b + c * d <= 100:
            break
        a = rng.randint(3, 9)
        b = rng.randint(3, 9)
        c = rng.randint(2, 8)
        d = rng.randint(2, 8)
    p1 = a * b
    p2 = c * d
    total = p1 + p2
    t = rng.randrange(4)
    if t == 0:
        ins = f"一辆汽车先以每小时{a}千米的速度行了{b}小时，又以每小时{c}千米的速度行了{d}小时，一共行了多少千米？"
        unit = "千米"
    elif t == 1:
        ins = f"小明上午骑车每分钟行{a}米，行了{b}分钟；下午每分钟行{c}米，行了{d}分钟，全天共行多少米？"
        unit = "米"
    elif t == 2:
        ins = f"甲书架有{a}层，每层放{b}本书；乙书架有{c}层，每层放{d}本书，两个书架共放多少本书？"
        unit = "本"
    else:
        ins = f"商店上午卖出{a}箱苹果，每箱{b}千克；下午卖出{c}箱，每箱{d}千克，全天共卖出多少千克？"
        unit = "千克"
    lines = [f"第一段 = {a} × {b} = {p1}{unit}",
             f"第二段 = {c} × {d} = {p2}{unit}",
             f"一共 = {p1} + {p2} = {total}{unit}"]
    return ins, lines, total


_reg("two_leg_distance", two_leg_distance)


# 53. sum_then_share: (a + b) / c = each ------------------------------------------
def sum_then_share(rng):
    a = rng.randint(20, 50)
    b = rng.randint(10, 30)
    c = rng.randint(3, 6)
    for _ in range(50):
        if (a + b) % c == 0:
            break
        a = rng.randint(20, 50)
        b = rng.randint(10, 30)
        c = rng.randint(3, 6)
    s = a + b
    each = s // c
    t = rng.randrange(4)
    ins = [
        f"学校买来{a}本故事书和{b}本科技书，平均分给{c}个班，每班分多少本？",
        f"甲筐有{a}千克苹果，乙筐有{b}千克，把这些苹果平均装在{c}个箱子里，每箱装多少千克？",
        f"小明有{a}张卡片，小红有{b}张，两人的卡片合起来平均分给{c}个小朋友，每人分多少张？",
        f"食堂运来{a}千克大米和{b}千克面粉，平均分给{c}个食堂窗口，每个窗口分多少千克？",
    ][t]
    unit = ["本", "千克", "张", "千克"][t]
    label = ["每班", "每箱", "每人", "每个窗口"][t]
    lines = [f"总和 = {a} + {b} = {s}{unit}",
             f"{label} = {s} ÷ {c} = {each}{unit}"]
    return ins, lines, each


_reg("sum_then_share", sum_then_share)


# 54. two_product_diff: a*b - c*d --------------------------------------------------
def two_product_diff(rng):
    a = rng.randint(3, 9)
    b = rng.randint(3, 9)
    c = rng.randint(2, 8)
    d = rng.randint(2, 8)
    for _ in range(50):
        if a * b > c * d:
            break
        a = rng.randint(3, 9)
        b = rng.randint(3, 9)
        c = rng.randint(2, 8)
        d = rng.randint(2, 8)
    p1 = a * b
    p2 = c * d
    diff = p1 - p2
    t = rng.randrange(4)
    ins = [
        f"小明每天看{a}页书，看了{b}天；小红每天看{c}页，看了{d}天，小明比小红多看多少页？",
        f"甲车间每天加工{a}个零件，加工了{b}天；乙车间每天加工{c}个，加工了{d}天，甲比乙多加工多少个？",
        f"苹果树有{a}行，每行{b}棵；梨树有{c}行，每行{d}棵，苹果树比梨树多多少棵？",
        f"学校买来{a}箱粉笔，每箱{b}盒；用了{c}箱，每箱{d}盒，剩下的比用掉的多多少盒？",
    ][t]
    unit = ["页", "个", "棵", "盒"][t]
    lines = [f"前者 = {a} × {b} = {p1}{unit}",
             f"后者 = {c} × {d} = {p2}{unit}",
             f"多 = {p1} - {p2} = {diff}{unit}"]
    return ins, lines, diff


_reg("two_product_diff", two_product_diff)


# 55. adjacent_sum: sum of adjacent naturals/evens/odds ---------------------------
def adjacent_sum(rng):
    t = rng.randrange(6)
    if t == 0:
        a = rng.randint(5, 60)
        ans = 2 * a + 1
        ins = f"两个相邻的自然数是{a}和{a + 1}，它们的和是多少？"
        lines = [f"和 = {a} + {a + 1} = {ans}"]
    elif t == 1:
        a = rng.randint(5, 60)
        a -= a % 2
        ans = 2 * a + 2
        ins = f"两个相邻的偶数是{a}和{a + 2}，它们的和是多少？"
        lines = [f"和 = {a} + {a + 2} = {ans}"]
    elif t == 2:
        a = rng.randint(5, 60)
        a = a - (a % 2) + 1
        ans = 2 * a + 2
        ins = f"两个相邻的奇数是{a}和{a + 2}，它们的和是多少？"
        lines = [f"和 = {a} + {a + 2} = {ans}"]
    elif t == 3:
        a = rng.randint(5, 32)
        ans = 3 * a + 3
        ins = f"三个连续自然数{a}、{a + 1}、{a + 2}的和是多少？"
        lines = [f"和 = {a} + {a + 1} + {a + 2} = {ans}"]
    elif t == 4:
        a = rng.randint(5, 23)
        ans = 4 * a + 6
        ins = f"四个连续自然数{a}、{a + 1}、{a + 2}、{a + 3}的和是多少？"
        lines = [f"和 = {a} + {a + 1} + {a + 2} + {a + 3} = {ans}"]
    else:
        a = rng.randint(5, 18)
        ans = 5 * a + 10
        ins = f"五个连续自然数{a}、{a + 1}、{a + 2}、{a + 3}、{a + 4}的和是多少？"
        lines = [f"和 = {a} + {a + 1} + {a + 2} + {a + 3} + {a + 4} = {ans}"]
    return ins, lines, ans


_reg("adjacent_sum", adjacent_sum)


if __name__ == "__main__":
    rng = random.Random(1)
    bad = 0
    from run_math_short import verify
    for _lvl, name, fn in PROGRAMS:
        for _ in range(30):
            ins, lines, ans = fn(rng)
            out, ok = verify(ins, lines, ans)
            if not ok:
                print(f"FAIL {name}: {ins!r} {lines} {ans}")
                bad += 1
                break
    print(f"selfcheck {'PASSED' if bad == 0 else f'FAILED ({bad})'}: {len(PROGRAMS)} programs")
