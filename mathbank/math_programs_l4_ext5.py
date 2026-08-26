#!/usr/bin/env python3
"""L4 ext5: reverse reasoning, multi-constraint, combinatorics, schedules — 60 families.

Every program: fn(rng) -> (instruction, lines, ans). Lines solve FORWARD from
givens to the asked value (no x variable). All exact arithmetic via Fraction.
Every equation line is chained: 中文标签 = 表达式 = 值[单位].
"""
import math
import random
from fractions import Fraction
from mathcommon import (ANIMALS, FOOD, FRUITS, GOODS, NAMES, PLACE, STATIONERY,
                        UNIT_FRUIT, UNIT_N, UNIT_ZHI, num)

PROGRAMS = []

_TAILS = [
    "请你把算式和结果都写出来。",
    "请把你的计算过程完整地写出来。",
    "请你列式计算，并写出最后结果。",
    "请把计算过程和结果都写出来。",
    "请你列式计算，并把结果写出来。",
]


def _reg(name, fn):
    def wrapped(rng):
        ins, lines, ans = fn(rng)
        return ins + rng.choice(_TAILS), lines, ans
    PROGRAMS.append(("L4", name, wrapped))


# 1. 3个连续偶数的和是s → 最小/中间/最大
def consec_three_even(rng):
    k = rng.randint(2, 15)
    x = 2 * k
    s = 3 * x + 6
    who = rng.choice(["最小", "中间", "最大"])
    ans = {"最小": x, "中间": x + 2, "最大": x + 4}[who]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：3个连续偶数的和是{s}，其中{who}的一个是多少？",
        f"{name}在练习册上看到一道题：3个连续偶数相加得{s}，{who}的偶数是多少？请你帮他算一算。",
        f"三个连续偶数的和是{s}，这三个数中{who}的是几？",
        f"有3个连续的偶数，它们的和是{s}。{name}想知道{who}的一个是多少，请你列式算一算。",
    ])
    lines = [
        f"最小偶数的3倍 = {s} - 6 = {3 * x}",
        f"最小的偶数 = {3 * x} ÷ 3 = {x}",
        f"中间的偶数 = {x} + 2 = {x + 2}",
        f"最大的偶数 = {x + 2} + 2 = {x + 4}",
    ]
    order = {"最小": 0, "中间": 1, "最大": 2}[who]
    lines = lines[:1] + lines[1:order + 1] + lines[order + 1:] + [lines[order + 1]]
    return ins, lines, ans


_reg("consec_three_even", consec_three_even)


# 2. 约分后是p/q，分子分母差d → 原分数
def fraction_simplify_diff(rng):
    p, q = rng.choice([(2, 3), (3, 4), (2, 5), (3, 5), (4, 5), (5, 6), (4, 7), (5, 7)])
    t = rng.randint(2, 12)
    d = (q - p) * t
    n = p * t
    den = q * t
    ans = Fraction(n, den)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个分数约分后是{p}/{q}，原分数的分子与分母的差是{d}。这个分数是多少？",
        f"{name}遇到一道思考题：某分数约分后等于{p}/{q}，且分子和分母相差{d}。请你帮他算出这个分数。",
        f"一个分数化成最简分数是{p}/{q}，原来的分子与分母的差是{d}。原来的分数是多少？",
        f"分数{p}/{q}是一个分数约分后的结果，已知原分数分子分母的差是{d}。{name}想知道原分数是多少，请你算一算。",
    ])
    lines = [
        f"每份是多少 = {d} ÷ {q - p} = {t}",
        f"分子 = {t} × {p} = {n}",
        f"分母 = {t} × {q} = {den}",
        f"这个分数 = {n} ÷ {den} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("fraction_simplify_diff", fraction_simplify_diff)


# 3. 被减数+减数+差=s，减数是差的k倍 → 各数
def subtraction_relation(rng):
    t = rng.randint(3, 15)
    k = rng.randint(2, 5)
    cha = t
    jianshu = k * t
    beijian = (k + 1) * t
    s = 2 * beijian
    who = rng.choice(["被减数", "减数", "差"])
    ans = {"被减数": beijian, "减数": jianshu, "差": cha}[who]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：在一道减法算式里，被减数、减数与差的和是{s}，减数正好是差的{k}倍。{who}是多少？",
        f"{name}在练习册上看到一道题：减法算式中被减数、减数、差相加得{s}，减数是差的{k}倍。请你帮他算出{who}。",
        f"一道减法算式，被减数加减数加差等于{s}，减数是差的{k}倍。{who}是多少？",
        f"减法算式里，被减数、减数与差的和是{s}，减数是差的{k}倍。{name}想知道{who}是多少，请你列式算一算。",
    ])
    lines = [
        f"被减数 = {s} ÷ 2 = {beijian}",
        f"总份数 = {k} + 1 = {k + 1}",
        f"差 = {beijian} ÷ {k + 1} = {cha}",
        f"减数 = {cha} × {k} = {jianshu}",
    ]
    idx = {"被减数": 0, "减数": 3, "差": 2}[who]
    lines = lines[:idx] + lines[idx + 1:] + [lines[idx]]
    return ins, lines, ans


_reg("subtraction_relation", subtraction_relation)


# 4. 被除数+除数+商=s，商=k → 被除数/除数
def division_relation(rng):
    t = rng.randint(5, 20)
    k = rng.randint(2, 6)
    beichu = k * t
    chushu = t
    s = beichu + chushu + k
    who = rng.choice(["被除数", "除数"])
    ans = beichu if who == "被除数" else chushu
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：在一道没有余数的除法算式里，被除数、除数与商的和是{s}，商是{k}。{who}是多少？",
        f"{name}在练习册上看到一道题：除法算式中被除数、除数、商相加得{s}，商正好是{k}。请你帮他算出{who}。",
        f"一道整除的除法算式，被除数加除数加商等于{s}，商是{k}。{who}是多少？",
        f"除法算式里，被除数、除数与商的和是{s}，商是{k}。{name}想知道{who}是多少，请你列式算一算。",
    ])
    lines = [
        f"被除数与除数的和 = {s} - {k} = {beichu + chushu}",
        f"总份数 = {k} + 1 = {k + 1}",
        f"除数 = {beichu + chushu} ÷ {k + 1} = {chushu}",
        f"被除数 = {chushu} × {k} = {beichu}",
    ]
    if who == "除数":
        lines = lines[:2] + lines[3:] + [lines[2]]
    return ins, lines, ans


_reg("division_relation", division_relation)


# 5. 两个加数与和相加=s，大加数是小加数的k倍 → 加数
def addition_relation(rng):
    t = rng.randint(5, 25)
    k = rng.randint(2, 5)
    small = t
    big = k * t
    he = (k + 1) * t
    s = 2 * he
    who = rng.choice(["较大的加数", "较小的加数"])
    ans = big if who == "较大的加数" else small
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：在一道加法算式里，两个加数与和相加得{s}，较大的加数正好是较小的加数的{k}倍。{who}是多少？",
        f"{name}在练习册上看到一道题：加法算式中两个加数与和的和是{s}，一个加数是另一个的{k}倍。请你帮他算出{who}。",
        f"一道加法算式，两个加数与它们的和相加等于{s}，大加数是小加数的{k}倍。{who}是多少？",
        f"加法算式里，两个加数与和的和是{s}，较大加数是较小加数的{k}倍。{name}想知道{who}是多少，请你列式算一算。",
    ])
    lines = [
        f"两个加数的和 = {s} ÷ 2 = {he}",
        f"总份数 = {k} + 1 = {k + 1}",
        f"较小的加数 = {he} ÷ {k + 1} = {small}",
        f"较大的加数 = {small} × {k} = {big}",
    ]
    if who == "较小的加数":
        lines = lines[:2] + lines[3:] + [lines[2]]
    return ins, lines, ans


_reg("addition_relation", addition_relation)


# 6. 除以a余r1，除以b余r2 → 最小数
def crt_two_remainders(rng):
    a = b = r1 = r2 = x = None
    for _ in range(80):
        a = rng.randint(3, 9)
        b = rng.randint(a + 1, 12)
        if math.gcd(a, b) != 1:
            continue
        r1 = rng.randint(1, a - 1)
        r2 = rng.randint(1, b - 1)
        for k in range(b):
            if (r1 + a * k) % b == r2:
                x = r1 + a * k
                break
        if x is not None and x >= 3:
            break
        x = None
    else:
        a, b, r1, r2, x = 3, 4, 1, 2, 5
    k0 = (x - r1) // a
    q = x // b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数除以{a}余{r1}，除以{b}余{r2}。这个数最小是多少？",
        f"{name}遇到一道思考题：某数除以{a}余{r1}，除以{b}余{r2}。请你帮他算出满足条件的最小的数。",
        f"一个数分别除以{a}和{b}，余数分别是{r1}和{r2}。这个数最小是多少？",
        f"一筐苹果，{a}个一数余{r1}个，{b}个一数余{r2}个。这筐苹果最少有多少个？",
    ])
    lines = [
        f"试算的倍数 = {a} × {k0} = {a * k0}",
        f"验算除以{b} = {x} - {b} × {q} = {r2}",
        f"这个数 = {a * k0} + {r1} = {x}",
    ]
    return ins, lines, x


_reg("crt_two_remainders", crt_two_remainders)


# 7. 五个数平均m，前三平均a，后三平均b → 中间数
def avg_overlap_middle(rng):
    a = b = m = mid = None
    for _ in range(80):
        a = rng.randint(60, 90)
        b = rng.randint(60, 90)
        m = rng.randint(60, 90)
        mid = 3 * a + 3 * b - 5 * m
        if 10 <= mid <= 60:
            break
    else:
        a, b, m, mid = 70, 75, 72, 25
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：5个数的平均数是{m}，前3个数的平均数是{a}，后3个数的平均数是{b}。中间的数是多少？",
        f"{name}在练习册上看到一道题：五个数平均为{m}，前三个平均{a}，后三个平均{b}。请你帮他算出中间的数。",
        f"有5个数，它们的平均数是{m}，前3个数的平均数是{a}，后3个数的平均数是{b}。中间那个数是多少？",
        f"五个数的平均数是{m}，前三个数平均{a}，后三个数平均{b}。{name}想知道中间的数是多少，请你列式算一算。",
    ])
    lines = [
        f"前三个数的和 = {a} × 3 = {3 * a}",
        f"后三个数的和 = {b} × 3 = {3 * b}",
        f"五个数的和 = {m} × 5 = {5 * m}",
        f"中间的数 = {3 * a} + {3 * b} - {5 * m} = {mid}",
    ]
    return ins, lines, mid


_reg("avg_overlap_middle", avg_overlap_middle)


# 8. 6个数平均m，去掉两个后余下4个平均n，去掉的两数差d → 较大/较小
def avg_remove_two(rng):
    m = n = S = d = big = small = None
    for _ in range(80):
        m = rng.randint(20, 50)
        n = rng.randint(15, m - 2)
        S = 6 * m - 4 * n
        if S < 6:
            continue
        d = rng.randint(2, S - 2)
        if (S + d) % 2 == 0:
            big = (S + d) // 2
            small = (S - d) // 2
            break
    else:
        m, n, S, d, big, small = 30, 25, 80, 10, 45, 35
    who = rng.choice(["较大的数", "较小的数"])
    ans = big if who == "较大的数" else small
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：6个数的平均数是{m}，去掉其中两个数后，余下4个数的平均数是{n}。去掉的两个数相差{d}，{who}是多少？",
        f"{name}在练习册上看到一道题：六个数平均为{m}，去掉两个数后剩下四个数平均{n}，去掉的两数差{d}。请你帮他算出{who}。",
        f"有6个数，平均数是{m}，划去其中两个后，余下4个数的平均数是{n}。已知划去的两个数差{d}，{who}是多少？",
        f"六个数的平均数是{m}，去掉两个数后余下四个数的平均数是{n}，去掉的两个数相差{d}。{name}想知道{who}是多少，请你列式算一算。",
    ])
    lines = [
        f"6个数的和 = {m} × 6 = {6 * m}",
        f"余下4个数的和 = {n} × 4 = {4 * n}",
        f"去掉的两个数的和 = {6 * m} - {4 * n} = {S}",
        f"较大的数 = {S} + {d} = {S + d}",
        f"较大的数 = {S + d} ÷ 2 = {big}",
        f"较小的数 = {S} - {d} = {S - d}",
        f"较小的数 = {S - d} ÷ 2 = {small}",
    ]
    if who == "较大的数":
        lines = lines[:5]
    else:
        lines = lines[:3] + lines[5:]
    return ins, lines, ans


_reg("avg_remove_two", avg_remove_two)


# 9. 男生平均a女生平均b全班平均m，男生n人 → 女生人数
def weighted_avg_count(rng):
    m = rng.randint(70, 90)
    d1 = rng.randint(2, 8)
    d2 = rng.randint(2, 8)
    t = rng.randint(2, 6)
    boys = d2 * t
    girls = d1 * t
    a = m + d1
    b = m - d2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一次数学测验，全班男生的平均分是{a}分，女生的平均分是{b}分，全班的平均分是{m}分。已知男生有{boys}人，女生有多少人？",
        f"{name}班男生平均{a}分，女生平均{b}分，全班平均{m}分。男生有{boys}人，{name}想知道女生有多少人，请你算一算。",
        f"某班男生数学平均{a}分，女生平均{b}分，全班平均{m}分。如果男生有{boys}人，女生有多少人？",
        f"测验后统计：男生平均{a}分，女生平均{b}分，全班平均{m}分，男生共{boys}人。女生有多少人？",
    ])
    lines = [
        f"男生每人高出平均分 = {a} - {m} = {d1}分",
        f"男生共高出的分数 = {d1} × {boys} = {d1 * boys}分",
        f"女生每人低于平均分 = {m} - {b} = {d2}分",
        f"女生人数 = {d1 * boys} ÷ {d2} = {girls}人",
    ]
    return ins, lines, girls


_reg("weighted_avg_count", weighted_avg_count)


# 10. 蜘蛛8腿蜻蜓6腿2翅蝉6腿1翅，共n只腿l翅w → 各几只
def bugs_legs_wings(rng):
    spider = rng.randint(2, 8)
    dragon = rng.randint(2, 8)
    cicada = rng.randint(2, 8)
    n = spider + dragon + cicada
    l = 8 * spider + 6 * (dragon + cicada)
    w = 2 * dragon + cicada
    who = rng.choice(["蜘蛛", "蜻蜓", "蝉"])
    ans = {"蜘蛛": spider, "蜻蜓": dragon, "蝉": cicada}[who]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"蜘蛛有8条腿，蜻蜓有6条腿和2对翅膀，蝉有6条腿和1对翅膀。这三种昆虫共{n}只，腿共有{l}条，翅膀共有{w}对。{who}有多少只？",
        f"{name}在草地上捉到蜘蛛、蜻蜓、蝉共{n}只，数出腿共{l}条、翅膀共{w}对。蜘蛛8条腿，蜻蜓6条腿2对翅膀，蝉6条腿1对翅膀。{who}有多少只？",
        f"蜘蛛、蜻蜓、蝉共{n}只，它们的腿共{l}条，翅膀共{w}对。已知蜘蛛8条腿，蜻蜓6条腿2对翅膀，蝉6条腿1对翅膀。{who}有多少只？",
        f"生物小组采集到蜘蛛、蜻蜓、蝉共{n}只，腿共{l}条，翅膀共{w}对。蜘蛛8条腿，蜻蜓6条腿2对翅，蝉6条腿1对翅。{name}想知道{who}有多少只，请你算一算。",
    ])
    six = n - spider
    lines = [
        f"假设全是6条腿的总腿数 = {n} × 6 = {6 * n}条",
        f"蜘蛛比6条腿多的条数 = 8 - 6 = 2条",
        f"腿数的差 = {l} - {6 * n} = {l - 6 * n}条",
        f"蜘蛛的只数 = {l - 6 * n} ÷ 2 = {spider}只",
        f"蜻蜓和蝉的总数 = {n} - {spider} = {six}只",
        f"蜻蜓的只数 = {w} - {six} = {dragon}只",
        f"蝉的只数 = {six} - {dragon} = {cicada}只",
    ]
    idx = {"蜘蛛": 3, "蜻蜓": 5, "蝉": 6}[who]
    lines = lines[:idx] + lines[idx + 1:] + [lines[idx]]
    return ins, lines, ans


_reg("bugs_legs_wings", bugs_legs_wings)


# 11. 哥a岁弟b岁，弟长到哥现在年龄时哥多少岁/年龄和
def age_when_older(rng):
    a = rng.randint(10, 40)
    b = rng.randint(5, a - 3)
    d = a - b
    who = rng.choice(["哥哥的年龄", "两人的年龄和"])
    ans = 2 * a - b if who == "哥哥的年龄" else 3 * a - b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"哥哥今年{a}岁，弟弟今年{b}岁。当弟弟长到哥哥现在的年龄时，{who}是多少？",
        f"{name}今年{a}岁，弟弟今年{b}岁。弟弟长到哥哥现在这么大时，{who}是多少？请你列式算一算。",
        f"哥哥{a}岁，弟弟{b}岁。再过多少年弟弟有哥哥现在这么大？那时{who}是多少？",
        f"今年哥哥{a}岁、弟弟{b}岁。当弟弟和哥哥现在一样大时，{who}是多少？",
    ])
    lines = [
        f"年龄差 = {a} - {b} = {d}岁",
        f"弟弟长到哥哥现在的年龄 = {b} + {d} = {a}岁",
        f"那时哥哥的年龄 = {a} + {d} = {2 * a - b}岁",
        f"那时两人的年龄和 = {2 * a - b} + {a} = {3 * a - b}岁",
    ]
    if who == "哥哥的年龄":
        lines = lines[:3]
    return ins, lines, ans


_reg("age_when_older", age_when_older)


# 12. 相遇点距中点d，甲速v1乙速v2 → 距离
def meet_midpoint_distance(rng):
    diff = rng.randint(5, 20)
    k = rng.randint(2, 8)
    d = diff * k
    v2 = rng.randint(40, 80)
    v1 = v2 + diff
    s = 2 * k * (v1 + v2)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人同时从两地相向而行，甲每分钟走{v1}米，乙每分钟走{v2}米，相遇点距两地中点{d}米。两地相距多少米？",
        f"{name}和小红从两地同时出发相向而行，{name}每分钟行{v1}米，小红每分钟行{v2}米，相遇时距中点{d}米。两地相距多少米？",
        f"甲、乙两车从A、B两地同时相向开出，甲车每小时{v1}千米，乙车每小时{v2}千米，相遇点距中点{d}千米。A、B两地相距多少千米？",
        f"两人从两地同时相向走来，速度分别是每分钟{v1}米和{v2}米，在距中点{d}米处相遇。两地相距多少米？",
    ])
    lines = [
        f"速度差 = {v1} - {v2} = {diff}米/分",
        f"相遇时的路程差 = {d} × 2 = {2 * d}米",
        f"相遇时间 = {2 * d} ÷ {diff} = {2 * k}分",
        f"速度和 = {v1} + {v2} = {v1 + v2}米/分",
        f"两地距离 = {2 * k} × {v1 + v2} = {s}米",
    ]
    return ins, lines, s


_reg("meet_midpoint_distance", meet_midpoint_distance)


# 13. 甲跑一圈a分钟乙b分钟，起点第一次相遇时甲跑几圈
def circular_start_laps(rng):
    a = b = g = None
    for _ in range(50):
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        if b != a:
            g = math.gcd(a, b)
            break
    else:
        a, b, g = 4, 6, 2
    t = a * b // g
    laps = t // a
    laps_b = t // b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人在环形跑道上跑步，甲跑一圈要{a}分钟，乙跑一圈要{b}分钟。两人同时同地同向出发，第一次在起点相遇时，甲跑了多少圈？",
        f"环形跑道上，甲跑完一圈需{a}分钟，乙跑完一圈需{b}分钟。{name}和同学同时同地同向起跑，第一次在起点相遇时甲跑了几圈？",
        f"甲、乙绕操场跑步，跑一圈分别要{a}分钟和{b}分钟。两人同时同地出发，至少多少分钟后在起点相遇？此时甲跑了多少圈？",
        f"甲跑一圈用{a}分钟，乙跑一圈用{b}分钟，两人同时同地同向出发。第一次在起点相遇时，甲跑了多少圈？",
    ])
    lines = [
        f"起点相遇时间 = {a} × {b} ÷ {g} = {t}分钟",
        f"乙跑的圈数 = {t} ÷ {b} = {laps_b}圈",
        f"两人共跑的圈数 = {laps} + {laps_b} = {laps + laps_b}圈",
        f"甲跑的圈数 = {t} ÷ {a} = {laps}圈",
    ]
    return ins, lines, laps


_reg("circular_start_laps", circular_start_laps)


# 14. 火车长L速v，人速u，同向/相向 → 通过时间
def train_person_pass(rng):
    direction = rng.choice(["同向", "相向"])
    if direction == "同向":
        rel = rng.randint(5, 25)
        t = rng.randint(5, 20)
        L = rel * t
        v = rng.randint(rel + 1, rel + 20)
        u = v - rel
        lines = [
            f"火车与人的速度差 = {v} - {u} = {rel}米/秒",
            f"火车比人多行的路程 = {rel} × {t} = {L}米",
            f"通过的时间 = {L} ÷ {rel} = {t}秒",
        ]
        ins = rng.choice([
            f"一列火车长{L}米，每秒行{v}米，一个人以每秒{u}米的速度与火车同向行走。火车从车头遇到人到车尾离开人，需要多少秒？",
            f"铁路旁，一列长{L}米的火车以每秒{v}米的速度驶来，旁边一人以每秒{u}米的速度同向行走。{rng.choice(NAMES)}想知道火车完全通过这个人要几秒，请你算一算。",
            f"火车长{L}米，每秒行驶{v}米，一人在路旁以每秒{u}米的速度与火车同向步行。火车经过这个人需要多少秒？",
            f"一列长{L}米的火车每秒行{v}米，一人以每秒{u}米的速度沿铁路同向行走。从车头遇上人到车尾离开人，共需多少秒？",
        ])
    else:
        u = rng.randint(1, 5)
        v = rng.randint(8, 30)
        rel = v + u
        t = rng.randint(4, 15)
        L = rel * t
        lines = [
            f"火车与人的速度和 = {v} + {u} = {rel}米/秒",
            f"火车与人共行的路程 = {rel} × {t} = {L}米",
            f"通过的时间 = {L} ÷ {rel} = {t}秒",
        ]
        ins = rng.choice([
            f"一列火车长{L}米，每秒行{v}米，一个人以每秒{u}米的速度迎面走来。火车从车头遇到人到车尾离开人，需要多少秒？",
            f"铁路旁，一列长{L}米的火车以每秒{v}米的速度驶来，一人以每秒{u}米的速度迎面走向火车。{rng.choice(NAMES)}想知道火车完全通过这个人要几秒，请你算一算。",
            f"火车长{L}米，每秒行驶{v}米，一人在路旁以每秒{u}米的速度向火车方向走来。火车经过这个人需要多少秒？",
            f"一列长{L}米的火车每秒行{v}米，一人以每秒{u}米的速度迎面而行。从车头遇上人到车尾离开人，共需多少秒？",
        ])
    return ins, lines, t


_reg("train_person_pass", train_person_pass)


# 15. 甲乙合做a天乙丙b天甲丙c天 → 三队合做天数
def work_three_efficiency(rng):
    a = b = c = t = None
    for _ in range(80):
        a = rng.randint(4, 12)
        b = rng.randint(4, 15)
        c = rng.randint(4, 15)
        rate = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
        total = 2 / rate
        if total.denominator == 1 and total.numerator <= 30:
            t = total.numerator
            break
    else:
        a, b, c, t = 6, 10, 15, 6
    ra, rb, rc = Fraction(1, a), Fraction(1, b), Fraction(1, c)
    rate = (ra + rb + rc) / 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程，甲、乙两队合做{a}天完成，乙、丙两队合做{b}天完成，甲、丙两队合做{c}天完成。三队合做，多少天可以完成？",
        f"修一条路，甲乙合修{a}天完工，乙丙合修{b}天完工，甲丙合修{c}天完工。{name}想知道三队合修几天完工，请你算一算。",
        f"加工一批零件，师傅和徒弟合做{a}天完成，徒弟和{name}合做{b}天完成，师傅和{name}合做{c}天完成。三人合做几天完成？",
        f"一项工程，甲乙合做{a}天完成，乙丙合做{b}天完成，甲丙合做{c}天完成。三队一起做，多少天完成这项工程？",
    ])
    lines = [
        f"甲乙合做效率 = 1 ÷ {a} = {num(ra)}",
        f"乙丙合做效率 = 1 ÷ {b} = {num(rb)}",
        f"甲丙合做效率 = 1 ÷ {c} = {num(rc)}",
        f"三队合做效率 = ({num(ra)} + {num(rb)} + {num(rc)}) ÷ 2 = {num(rate)}",
        f"三队合做的天数 = 1 ÷ ({num(rate)}) = {t}天",
    ]
    return ins, lines, t


_reg("work_three_efficiency", work_three_efficiency)


# 16. 甲a乙b丙c三管，先开甲t小时再三管齐开 → 还需时间
def pipes_three_cycle(rng):
    a = rng.randint(4, 10)
    b = rng.randint(3, 10)
    c = rng.randint(3, 10)
    t = rng.randint(1, a - 2)
    ra, rb, rc = Fraction(1, a), Fraction(1, b), Fraction(1, c)
    done = t * ra
    rest = 1 - done
    comb = ra + rb + rc
    t2 = rest / comb
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个水池有甲、乙、丙三个进水管。单开甲管{a}小时注满，单开乙管{b}小时注满，单开丙管{c}小时注满。先开甲管{t}小时后，三管齐开，还要多少小时注满？",
        f"水池的甲管单开{a}小时注满，乙管单开{b}小时注满，丙管单开{c}小时注满。工人先开甲管{t}小时，然后把乙、丙两管也打开，{name}想知道还要几小时注满，请你算一算。",
        f"注满一个水池，甲管要{a}小时，乙管要{b}小时，丙管要{c}小时。先开甲管{t}小时，再三管齐开，还需要多少小时？",
        f"一个空水池，甲进水管{a}小时注满，乙{b}小时注满，丙{c}小时注满。先开甲管{t}小时后三管同开，几小时能注满？",
    ])
    lines = [
        f"甲管效率 = 1 ÷ {a} = {num(ra)}",
        f"乙管效率 = 1 ÷ {b} = {num(rb)}",
        f"丙管效率 = 1 ÷ {c} = {num(rc)}",
        f"甲管先注入的水量 = {t} × {num(ra)} = {num(done)}",
        f"剩余的水量 = 1 - {num(done)} = {num(rest)}",
        f"三管合效率 = {num(ra)} + {num(rb)} + {num(rc)} = {num(comb)}",
        f"还需要的时间 = {num(rest)} ÷ ({num(comb)}) = {num(t2)}时",
    ]
    return ins, lines, t2


_reg("pipes_three_cycle", pipes_three_cycle)


# 17. 满池水，排水管a小时排空，先排t小时后开进水管b小时满 → 还需排空时间
def drain_then_inlet(rng):
    a = rng.randint(3, 8)
    b = rng.randint(a + 3, 2 * a + 6)
    t = rng.randint(1, a - 1)
    rd, ri = Fraction(1, a), Fraction(1, b)
    rest = 1 - t * rd
    net = rd - ri
    t2 = rest / net
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个装满水的水池，排水管单开{a}小时排空，进水管单开{b}小时注满。先打开排水管{t}小时后，再打开进水管，还要多少小时才能把水排空？",
        f"满池水，排水管{a}小时可排空，进水管{b}小时可注满。管理员先开排水管{t}小时，然后把进水管也打开。{name}想知道还要几小时排空，请你算一算。",
        f"水池有一个排水管和一个进水管，单开排水管{a}小时排空满池，单开进水管{b}小时注满空池。先开排水管{t}小时后两管齐开，多少小时排空？",
        f"一个水池装满了水，排水管每小时排整池的1/{a}，进水管每小时注整池的1/{b}。先开排水管{t}小时，再开进水管，还需几小时排空？",
    ])
    lines = [
        f"排水管效率 = 1 ÷ {a} = {num(rd)}",
        f"进水管效率 = 1 ÷ {b} = {num(ri)}",
        f"先排出的水量 = {t} × {num(rd)} = {num(t * rd)}",
        f"剩余的水量 = 1 - {num(t * rd)} = {num(rest)}",
        f"净排水效率 = {num(rd)} - {num(ri)} = {num(net)}",
        f"还需要的时间 = {num(rest)} ÷ ({num(net)}) = {num(t2)}时",
    ]
    return ins, lines, t2


_reg("drain_then_inlet", drain_then_inlet)


# 18. 成本a，按成本增p%定价，打d折 → 实际利润
def profit_discount_actual(rng):
    k = rng.randint(2, 10)
    a = 100 * k
    p = rng.choice([20, 25, 40, 50, 60])
    if p == 25:
        d = rng.choice([6, 8])
    else:
        d = rng.randint(5, 9)
    price = k * (100 + p)
    sell = k * (100 + p) * d // 10
    profit = sell - a
    name = rng.choice(NAMES)
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店以{a}元的成本购进一件{obj}，按成本增加{p}%定价，后来又打{d}折出售。实际利润是多少元？",
        f"一件{obj}的成本是{a}元，商店先按成本增加{p}%定价，节日期间打{d}折卖出。{name}想知道实际赚了多少元，请你算一算。",
        f"某{obj}进价{a}元，商家按成本增加{p}%定价，实际按定价打{d}折出售。实际利润是多少元？",
        f"商店进了一件{obj}，成本{a}元，先按成本增加{p}%定价，再打{d}折促销。实际利润是多少元？",
    ])
    lines = [
        f"定价 = {a} × ({100 + p}/100) = {price}元",
        f"实际售价 = {price} × {d}/10 = {sell}元",
        f"实际利润 = {sell} - {a} = {profit}元",
    ]
    return ins, lines, profit


_reg("profit_discount_actual", profit_discount_actual)


# 19. 两件各卖a元，一件赚p%一件亏q%（p≠q）→ 合计亏多少
def profit_loss_diff_pct(rng):
    p, q = rng.choice([(20, 25), (25, 20)])
    if (p, q) == (20, 25):
        k = rng.randint(2, 20)
        a = 6 * k
        c1 = 5 * k
        c2 = 8 * k
    else:
        k = rng.randint(2, 20)
        a = 20 * k
        c1 = 16 * k
        c2 = 25 * k
    cost = c1 + c2
    sold = 2 * a
    loss = cost - sold
    name = rng.choice(NAMES)
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店卖出两件{obj}，每件都卖{a}元，其中一件赚{p}%，另一件亏{q}%。两件合起来亏了多少元？",
        f"老板把两件{obj}都卖了{a}元，一件赚了{p}%，另一件亏了{q}%。{name}想知道两件合计亏了多少元，请你算一算。",
        f"两件{obj}各卖{a}元，一件赚{p}%，一件亏{q}%。两件合起来是亏了多少元？",
        f"商店卖两件{obj}，售价都是{a}元，第一件赚{p}%，第二件亏{q}%。两件合计亏多少元？",
    ])
    lines = [
        f"赚的那件成本 = {a} ÷ ({100 + p}/100) = {c1}元",
        f"亏的那件成本 = {a} ÷ ({100 - q}/100) = {c2}元",
        f"两件的总成本 = {c1} + {c2} = {cost}元",
        f"两件的总售价 = {a} × 2 = {sold}元",
        f"合计亏的钱 = {cost} - {sold} = {loss}元",
    ]
    return ins, lines, loss


_reg("profit_loss_diff_pct", profit_loss_diff_pct)


# 20. 甲乙共s元，甲取a乙存b后相等 → 甲/乙原来
def deposits_equalize(rng):
    s = a = b = after = jia = yi = None
    for _ in range(80):
        s = rng.randint(100, 500) * 10
        a = rng.randint(10, 50)
        b = rng.randint(10, 50)
        if (s - a + b) % 2 == 0:
            after = (s - a + b) // 2
            yi = after - b
            if yi > 5:
                jia = after + a
                break
    else:
        s, a, b, after, jia, yi = 300, 30, 20, 145, 175, 125
    who = rng.choice(["甲", "乙"])
    ans = jia if who == "甲" else yi
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人共有{s}元存款。甲取出{a}元，乙存入{b}元后，两人的存款正好相等。{who}原来有多少元？",
        f"{name}和小红共有{s}元压岁钱，{name}取出{a}元，小红存入{b}元后，两人的钱一样多。{who}原来有多少元？",
        f"甲、乙在银行共存了{s}元，后来甲取出{a}元，乙又存入{b}元，这时两人存款相等。{who}原来存了多少元？",
        f"两人共有{s}元，甲取出{a}元、乙存入{b}元后，两人的钱数相等。{who}原来有多少元？",
    ])
    lines = [
        f"变化后的总钱数 = {s} - {a} + {b} = {s - a + b}元",
        f"相等时每人的钱 = {s - a + b} ÷ 2 = {after}元",
        f"甲原来的钱 = {after} + {a} = {jia}元",
        f"乙原来的钱 = {after} - {b} = {yi}元",
    ]
    if who == "甲":
        lines = lines[:3] + lines[4:]
    else:
        lines = lines[:2] + lines[3:]
    return ins, lines, ans


_reg("deposits_equalize", deposits_equalize)


# 21. 三层共s本，上比中多a，下比中少b → 各层
def shelves_three(rng):
    mid = rng.randint(20, 60)
    a = rng.randint(3, 15)
    b = rng.randint(3, 15)
    s = 3 * mid + a - b
    upper = mid + a
    lower = mid - b
    who = rng.choice(["上层", "中层", "下层"])
    ans = {"上层": upper, "中层": mid, "下层": lower}[who]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"书架上、中、下三层共有{s}本书。上层比中层多{a}本，下层比中层少{b}本。{who}有多少本书？",
        f"一个书架三层共放了{s}本书，上层比中层多{a}本，下层比中层少{b}本。{name}想知道{who}有多少本，请你算一算。",
        f"书架的上、中、下三层一共{s}本书，上层比中层多{a}本，下层比中层少{b}本。{who}放了多少本书？",
        f"图书角的书架三层共{s}本书，上层比中层多{a}本，下层比中层少{b}本。{who}有多少本书？",
    ])
    lines = [
        f"中层的3倍 = {s} - {a} + {b} = {3 * mid}本",
        f"中层的本数 = {3 * mid} ÷ 3 = {mid}本",
        f"上层的本数 = {mid} + {a} = {upper}本",
        f"下层的本数 = {mid} - {b} = {lower}本",
    ]
    idx = {"上层": 2, "中层": 1, "下层": 3}[who]
    lines = lines[:idx] + lines[idx + 1:] + [lines[idx]]
    return ins, lines, ans


_reg("shelves_three", shelves_three)


# 22. 两绳共s，第一根剪1/n第二根剪1/m后剩下相等 → 各原长
def ropes_fractions_equal(rng):
    n = m = None
    for _ in range(50):
        n = rng.randint(2, 5)
        m = rng.randint(2, 5)
        if n != m:
            break
    else:
        n, m = 2, 3
    k = rng.randint(2, 10)
    x = n * (m - 1) * k
    y = m * (n - 1) * k
    s = x + y
    who = rng.choice(["第一根", "第二根"])
    ans = x if who == "第一根" else y
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"两根绳子共长{s}米。第一根剪去全长的1/{n}，第二根剪去全长的1/{m}后，两根绳子剩下的长度相等。{who}原来长多少米？",
        f"甲、乙两根绳一共长{s}米，甲绳剪去1/{n}，乙绳剪去1/{m}，剩下的部分一样长。{name}想知道{('甲绳' if who == '第一根' else '乙绳')}原来的长度，请你算一算。",
        f"两根绳子共{s}米，第一根用去全长的1/{n}，第二根用去全长的1/{m}，余下的长度相等。{who}原来长多少米？",
        f"有两根绳子共长{s}米，把第一根剪去1/{n}、第二根剪去1/{m}后，两根一样长。{who}原来长多少米？",
    ])
    lines = [
        f"第一根的份数 = {n} × {m - 1} = {n * (m - 1)}",
        f"第二根的份数 = {m} × {n - 1} = {m * (n - 1)}",
        f"总份数 = {n * (m - 1)} + {m * (n - 1)} = {s // k}",
        f"每份的长度 = {s} ÷ {s // k} = {k}米",
        f"第一根原长 = {k} × {n * (m - 1)} = {x}米",
        f"第二根原长 = {k} × {m * (n - 1)} = {y}米",
    ]
    if who == "第一根":
        lines = lines[:5] + lines[6:]
    else:
        lines = lines[:4] + lines[5:]
    return ins, lines, ans


_reg("ropes_fractions_equal", ropes_fractions_equal)


# 23. 三次分别剪去1/n、余下1/m、余下1/k，剩r → 原长
def fraction_chain_three_reverse(rng):
    n = m = k = None
    for _ in range(50):
        n = rng.randint(2, 5)
        m = rng.randint(2, 5)
        k = rng.randint(2, 5)
        if n != m and m != k and n != k:
            break
    else:
        n, m, k = 2, 3, 4
    t = rng.randint(2, 8)
    r = (n - 1) * (m - 1) * (k - 1) * t
    L = n * m * k * t
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一根绳子，第一次剪去全长的1/{n}，第二次剪去余下的1/{m}，第三次剪去余下的1/{k}，最后还剩{r}米。这根绳子原来长多少米？",
        f"一根绳子长未知，{name}第一次剪去全长的1/{n}，第二次剪去剩下的1/{m}，第三次剪去又余下的1/{k}，最后剩{r}米。绳子原来长多少米？",
        f"一根绳子，先用去全长的1/{n}，再用去余下的1/{m}，最后用去又余下的1/{k}，还剩{r}米。这根绳子原来长多少米？",
        f"一根绳子分三次剪完，第一次剪1/{n}，第二次剪余下的1/{m}，第三次剪余下的1/{k}，最后剩{r}米。原长多少米？",
    ])
    lines = [
        f"第三次剪去的分率 = 1 ÷ {k} = {num(Fraction(1, k))}",
        f"第三次剪之前的长度 = {r} × {k} ÷ {k - 1} = {(n - 1) * (m - 1) * k * t}米",
        f"第二次剪去的分率 = 1 ÷ {m} = {num(Fraction(1, m))}",
        f"第二次剪之前的长度 = {(n - 1) * (m - 1) * k * t} × {m} ÷ {m - 1} = {(n - 1) * m * k * t}米",
        f"第一次剪去的分率 = 1 ÷ {n} = {num(Fraction(1, n))}",
        f"绳子的原长 = {(n - 1) * m * k * t} × {n} ÷ {n - 1} = {L}米",
    ]
    return ins, lines, L


_reg("fraction_chain_three_reverse", fraction_chain_three_reverse)


# 24. 第一次用a千克，第二次用余下的1/n，用去的比剩下的多d → 原重
def oil_uses_reverse(rng):
    n = rng.choice([3, 4])
    a = d = rest1 = None
    for _ in range(50):
        a = rng.randint(10, 40)
        d = rng.randint(2, a - 5)
        rest1 = n * (a - d) // (n - 2)
        if rest1 % n == 0:
            break
    else:
        n, a, d, rest1 = 3, 18, 9, 27
    L = rest1 + a
    second = rest1 // n
    left = rest1 - second
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一桶油，第一次用去{a}千克，第二次用去余下的1/{n}，这时用去的比剩下的多{d}千克。这桶油原来重多少千克？",
        f"一桶食用油，{name}第一次倒出{a}千克，第二次倒出余下的1/{n}，结果用去的比剩下的多{d}千克。这桶油原来重多少千克？",
        f"一桶油重未知，第一次用去{a}千克，第二次用去第一次后余下的1/{n}，这时用去的比剩下的多{d}千克。原来有多少千克油？",
        f"食堂买来一桶油，第一次用{a}千克，第二次用去余下的1/{n}，这时用去的比剩下的多{d}千克。这桶油原来多少千克？",
    ])
    lines = [
        f"第一次用后余下的 = {n} × ({a} - {d}) ÷ {n - 2} = {rest1}千克",
        f"第二次用去的 = {rest1} ÷ {n} = {second}千克",
        f"最后剩下的 = {rest1} - {second} = {left}千克",
        f"这桶油原来的质量 = {rest1} + {a} = {L}千克",
    ]
    return ins, lines, L


_reg("oil_uses_reverse", oil_uses_reverse)


# 25. 男生比全班的1/n多a，女生比男生少b → 全班/男生/女生
def class_fractions(rng):
    n = a = b = s = None
    for _ in range(50):
        n = rng.choice([3, 4])
        a = rng.randint(10, 30)
        b = rng.randint(2, a - 3)
        s = n * (2 * a - b) // (n - 2)
        if s % n == 0:
            break
    else:
        n, a, b, s = 3, 15, 6, 72
    boys = s // n + a
    girls = boys - b
    who = rng.choice(["全班", "男生", "女生"])
    ans = {"全班": s, "男生": boys, "女生": girls}[who]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"某班男生人数比全班人数的1/{n}多{a}人，女生人数比男生少{b}人。{who}有多少人？",
        f"{name}班的男生比全班人数的1/{n}多{a}人，女生比男生少{b}人。{name}想知道{who}有多少人，请你算一算。",
        f"一个班，男生人数比全班的1/{n}多{a}人，女生人数比男生少{b}人。{who}有多少人？",
        f"五年级某班，男生比全班人数的1/{n}多{a}人，女生比男生少{b}人。{who}有多少人？",
    ])
    lines = [
        f"全班人数的{n - 2}/{n}对应的人数 = 2 × {a} - {b} = {2 * a - b}人",
        f"全班人数 = {n} × {2 * a - b} ÷ {n - 2} = {s}人",
        f"男生人数 = {s} ÷ {n} + {a} = {boys}人",
        f"女生人数 = {boys} - {b} = {girls}人",
    ]
    idx = {"全班": 1, "男生": 2, "女生": 3}[who]
    lines = lines[:idx] + lines[idx + 1:] + [lines[idx]]
    return ins, lines, ans


_reg("class_fractions", class_fractions)


# 26. 甲是乙的k倍，甲运a乙运b后甲是乙的m倍 → 乙原/甲原
def piles_two_ratios(rng):
    x = rng.randint(10, 40)
    k = rng.randint(3, 6)
    m = rng.randint(2, k - 1)
    b = rng.randint(3, x - 3)
    a = (k - m) * x + m * b
    who = rng.choice(["乙堆", "甲堆"])
    ans = x if who == "乙堆" else k * x
    name = rng.choice(NAMES)
    goods = rng.choice(["煤", "沙子", "石子", "粮食", "化肥"])
    ins = rng.choice([
        f"甲堆{goods}的质量是乙堆的{k}倍。甲堆运走{a}吨，乙堆运走{b}吨后，甲堆剩下的正好是乙堆剩下的{m}倍。{who}原来有多少吨？",
        f"甲、乙两堆{goods}，甲堆是乙堆的{k}倍。从甲堆运走{a}吨、从乙堆运走{b}吨后，甲堆剩下的是乙堆剩下的{m}倍。{name}想知道{who}原来有多少吨，请你算一算。",
        f"甲堆{goods}是乙堆的{k}倍，甲运走{a}吨、乙运走{b}吨后，甲剩下的质量是乙剩下的{m}倍。{who}原来有多少吨？",
        f"两堆{goods}，甲堆质量是乙堆的{k}倍。甲堆用去{a}吨，乙堆用去{b}吨，甲堆剩下的是乙堆剩下的{m}倍。{who}原来有多少吨？",
    ])
    lines = [
        f"乙堆剩下的 = {x} - {b} = {x - b}吨",
        f"甲堆剩下的 = {x - b} × {m} = {m * (x - b)}吨",
        f"甲堆运走的 = {k * x} - {m * (x - b)} = {a}吨",
        f"甲堆原来的 = {x} × {k} = {k * x}吨",
        f"乙堆原来的 = {k * x} ÷ {k} = {x}吨",
    ]
    if who == "甲堆":
        lines = lines[:4]
    return ins, lines, ans


_reg("piles_two_ratios", piles_two_ratios)


# 27. 合金两金属比a:b，加入m千克第一种后比c:d → 原合金
def alloy_add_metal(rng):
    a, b, c, d = rng.choice([(1, 2, 2, 3), (2, 3, 3, 4), (2, 5, 1, 2),
                             (3, 5, 2, 3), (1, 3, 1, 2)])
    m1, m2 = rng.choice([("铜", "锌"), ("铜", "锡"), ("铅", "锡"), ("铝", "镁"), ("金", "银")])
    m = rng.randint(2, 8)
    k = d * m
    metal1 = a * k
    metal2 = b * k
    total = metal1 + metal2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一块{m1}{m2}合金中，{m1}与{m2}的质量比是{a}比{b}。加入{m}千克{m1}后，{m1}与{m2}的比变成{c}比{d}。这块合金原来重多少千克？",
        f"一块合金由{m1}和{m2}组成，质量比为{a}:{b}。加入{m}千克{m1}后，新合金中{m1}与{m2}的比是{c}:{d}。{name}想知道原合金的质量，请你算一算。",
        f"合金中{m1}和{m2}的比是{a}比{b}，再加入{m}千克{m1}，比变成{c}比{d}。原来这块合金重多少千克？",
        f"一块{m1}{m2}合金，{m1}与{m2}的比是{a}:{b}。加入{m}千克{m1}后，{m1}与{m2}的比为{c}:{d}。原合金重多少千克？",
    ])
    lines = [
        f"每份的质量 = {d} × {m} = {k}千克",
        f"原来{m1}的质量 = {a} × {k} = {metal1}千克",
        f"原来{m2}的质量 = {b} × {k} = {metal2}千克",
        f"加入{m1}后{m1}的质量 = {metal1} + {m} = {metal1 + m}千克",
        f"新的比 = {metal1 + m} ÷ {metal2} = {num(Fraction(c, d))}",
        f"原合金的质量 = {metal1} + {metal2} = {total}千克",
    ]
    return ins, lines, total


_reg("alloy_add_metal", alloy_add_metal)


# 28. 比例尺1:n，图上长a宽b → 实际面积
def map_scale_area(rng):
    n = rng.choice([100, 200, 500])
    a = rng.randint(2, 20)
    b = rng.randint(2, 20)
    fig_area = a * b
    area = fig_area * n * n // 10000
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一幅地图的比例尺是1比{n}，量得一块长方形地的图上长是{a}厘米、宽是{b}厘米。这块地的实际面积是多少平方米？",
        f"在比例尺为1:{n}的图纸上，一个长方形操场长{a}厘米、宽{b}厘米。{name}想知道操场的实际面积是多少平方米，请你算一算。",
        f"一张图纸的比例尺是1比{n}，图上长方形花坛长{a}厘米、宽{b}厘米。花坛的实际面积是多少平方米？",
        f"在1比{n}的图纸上，一块长方形菜地长{a}厘米、宽{b}厘米。这块菜地的实际面积是多少平方米？",
    ])
    lines = [
        f"图上面积 = {a} × {b} = {fig_area}平方厘米",
        f"实际面积 = {fig_area} × {n} × {n} ÷ 10000 = {area}平方米",
    ]
    return ins, lines, area


_reg("map_scale_area", map_scale_area)


# 29. 边长a分米的砖n块铺地，换边长b → 需要多少块
def tiles_paving(rng):
    a, b = rng.choice([(4, 2), (6, 3), (6, 2), (8, 4), (9, 3), (10, 5), (12, 6), (12, 4)])
    n = rng.randint(20, 200)
    area = a * a * n
    blocks = area // (b * b)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"用边长{a}分米的方砖铺一间教室，需要{n}块。如果改用边长{b}分米的方砖，需要多少块？",
        f"铺一间会议室，用边长{a}分米的方砖要{n}块。{name}想知道改用边长{b}分米的方砖需要多少块，请你算一算。",
        f"一间屋子用边长{a}分米的地砖铺地，正好用{n}块。如果换成边长{b}分米的地砖，需要多少块？",
        f"给教室铺地砖，边长{a}分米的砖需要{n}块。若改用边长{b}分米的砖，一共需要多少块？",
    ])
    lines = [
        f"教室的面积 = {a} × {a} × {n} = {area}平方分米",
        f"新砖的面积 = {b} × {b} = {b * b}平方分米",
        f"需要的块数 = {area} ÷ {b * b} = {blocks}块",
    ]
    return ins, lines, blocks


_reg("tiles_paving", tiles_paving)


# 30. 长2e宽e高e的长方体切成两个正方体 → 表面积增加/原/总
def cuboid_cut_surface(rng):
    e = rng.randint(3, 12)
    who = rng.choice(["增加的表面积", "原来的表面积", "两个正方体的总表面积"])
    ans = {"增加的表面积": 2 * e * e, "原来的表面积": 10 * e * e,
           "两个正方体的总表面积": 12 * e * e}[who]
    obj = rng.choice(["长方体木块", "长方体橡皮", "长方体蛋糕", "长方体豆腐", "长方体砖块"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个{obj}，长是{e}厘米的2倍，宽和高都是{e}厘米。把它切成两个棱长{e}厘米的正方体，{who}是多少平方厘米？",
        f"把一个长{2 * e}厘米、宽{e}厘米、高{e}厘米的{obj}切成两个正方体，{name}想知道{who}是多少平方厘米，请你算一算。",
        f"一个{obj}正好可以切成两个棱长{e}厘米的正方体。{who}是多少平方厘米？",
        f"长方体的长、宽、高分别是{2 * e}厘米、{e}厘米、{e}厘米，沿长的中点切成两个正方体。{who}是多少平方厘米？",
    ])
    lines = [
        f"切面的面积 = {e} × {e} = {e * e}平方厘米",
        f"表面积增加 = {e * e} × 2 = {2 * e * e}平方厘米",
        f"原来的表面积 = 2 × ({2 * e} × {e} + {2 * e} × {e} + {e} × {e}) = {10 * e * e}平方厘米",
        f"两个正方体的总表面积 = {10 * e * e} + {2 * e * e} = {12 * e * e}平方厘米",
    ]
    idx = {"增加的表面积": 1, "原来的表面积": 2, "两个正方体的总表面积": 3}[who]
    lines = lines[:idx] + lines[idx + 1:] + [lines[idx]]
    return ins, lines, ans


_reg("cuboid_cut_surface", cuboid_cut_surface)


# 31. n个棱长a的正方体拼成一排 → 表面积
def cubes_combine_surface(rng):
    n = rng.randint(2, 6)
    a = rng.randint(2, 10)
    long = n * a
    surf = 2 * (2 * n + 1) * a * a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"把{n}个棱长{a}厘米的正方体拼成一个长方体，拼成的长方体的表面积是多少平方厘米？",
        f"{name}把{n}个棱长{a}厘米的正方体木块排成一排粘成一个长方体，这个长方体的表面积是多少平方厘米？",
        f"用{n}个棱长{a}厘米的正方体拼成一个大长方体（排成一排），表面积是多少平方厘米？",
        f"{n}个棱长{a}厘米的正方体一字排开拼成一个长方体，求这个长方体的表面积。",
    ])
    lines = [
        f"拼成的长 = {n} × {a} = {long}厘米",
        f"上下两个面 = 2 × {long} × {a} = {2 * long * a}平方厘米",
        f"前后两个面 = 2 × {long} × {a} = {2 * long * a}平方厘米",
        f"左右两个面 = 2 × {a} × {a} = {2 * a * a}平方厘米",
        f"表面积 = {2 * long * a} + {2 * long * a} + {2 * a * a} = {surf}平方厘米",
    ]
    return ins, lines, surf


_reg("cubes_combine_surface", cubes_combine_surface)


# 32. 长方体长a宽b高c切最大正方体 → 正方体体积/剩余体积
def cuboid_max_cube(rng):
    a = rng.randint(8, 20)
    b = rng.randint(5, a - 2)
    c = rng.randint(3, b - 2)
    cube = c ** 3
    total = a * b * c
    rest = total - cube
    who = rng.choice(["正方体的体积", "剩余部分的体积"])
    ans = cube if who == "正方体的体积" else rest
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个长方体木块，长{a}厘米、宽{b}厘米、高{c}厘米。把它切成一个最大的正方体，{who}是多少立方厘米？",
        f"一块长方体木料长{a}厘米、宽{b}厘米、高{c}厘米，{name}想切出一个最大的正方体。{who}是多少立方厘米？",
        f"长方体的长、宽、高分别是{a}厘米、{b}厘米、{c}厘米，从中切下一个最大的正方体。{who}是多少立方厘米？",
        f"一个长{a}厘米、宽{b}厘米、高{c}厘米的长方体，切成最大的正方体后，{who}是多少立方厘米？",
    ])
    lines = [
        f"最大正方体的棱长 = {c} = {c}厘米",
        f"正方体的体积 = {c} × {c} × {c} = {cube}立方厘米",
        f"长方体的体积 = {a} × {b} × {c} = {total}立方厘米",
        f"剩余部分的体积 = {total} - {cube} = {rest}立方厘米",
    ]
    if who == "正方体的体积":
        lines = [lines[0], lines[2], lines[1]]
    return ins, lines, ans


_reg("cuboid_max_cube", cuboid_max_cube)


# 33. 实心方阵横竖各加一排增加a人 → 原方阵人数
def square_add_row_col(rng):
    a = rng.randint(5, 19) * 2 + 1
    n = (a - 1) // 2
    total = n * n
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"同学们排成一个实心方阵，后来横竖各增加一排，还需要增加{a}人。原来的方阵有多少人？",
        f"运动会上，同学们排成实心方阵。如果横竖各增加一排，还差{a}人。{name}想知道原来方阵有多少人，请你算一算。",
        f"一个实心方阵，横竖各增加一排后，共需增加{a}人。原来这个方阵有多少人？",
        f"学生排成实心方阵，若横竖各加一排，需要再加入{a}人。原来的方阵共有多少人？",
    ])
    lines = [
        f"原来每边人数的2倍 = {a} - 1 = {2 * n}人",
        f"原来每边的人数 = {2 * n} ÷ 2 = {n}人",
        f"原来方阵的人数 = {n} × {n} = {total}人",
    ]
    return ins, lines, total


_reg("square_add_row_col", square_add_row_col)


# 34. 实心方阵最外层a人 → 总人数
def solid_square_outer(rng):
    a = rng.randint(4, 15) * 4
    n = a // 4 + 1
    total = n * n
    scene = rng.choice([
        "同学们排成一个实心方阵",
        "棋子摆成一个实心方阵",
        "花盆摆成一个实心方阵",
        "士兵排成一个实心方阵",
    ])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{scene}，最外层一共有{a}人。这个实心方阵一共有多少人？",
        f"{scene}，最外层共{a}人。{name}想知道这个方阵一共有多少人，请你算一算。",
        f"{scene}，数出最外层有{a}人。这个方阵共有多少人？",
        f"{scene}，最外层一圈共{a}人。整个方阵有多少人？",
    ])
    lines = [
        f"最外层每边的人数 = {a} ÷ 4 + 1 = {n}人",
        f"实心方阵的总人数 = {n} × {n} = {total}人",
    ]
    return ins, lines, total


_reg("solid_square_outer", solid_square_outer)


# 35. 棋子摆正方形多a个，横竖各加一排少b个 → 棋子总数
def chess_square(rng):
    a = b = n = None
    for _ in range(50):
        a = rng.randint(3, 15)
        b = rng.randint(3, 15)
        if (a + b) % 2 == 1:
            n = (a + b - 1) // 2
            break
    else:
        a, b, n = 5, 6, 5
    total = n * n + a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}把一些棋子摆成正方形，多出{a}个；如果横竖各增加一排，又少{b}个。这些棋子一共有多少个？",
        f"一堆棋子摆成实心正方形多{a}个，若横竖各加一排则少{b}个。{name}想知道棋子共有多少个，请你算一算。",
        f"棋子摆成正方形方阵余{a}个，横竖各增加一排缺{b}个。棋子一共有多少个？",
        f"用棋子摆正方形，多{a}个；想把正方形每边增加一个，又少{b}个。棋子共有多少个？",
    ])
    lines = [
        f"横竖各加一排共需 = {a} + {b} = {a + b}个",
        f"原来每边的2倍 = {a + b} - 1 = {2 * n}个",
        f"原来每边的个数 = {2 * n} ÷ 2 = {n}个",
        f"原正方形的棋子 = {n} × {n} = {n * n}个",
        f"棋子的总数 = {n * n} + {a} = {total}个",
    ]
    return ins, lines, total


_reg("chess_square", chess_square)


# 36. n! 末尾0的个数
def trailing_zeros(rng):
    n = rng.randint(25, 100)
    c5 = n // 5
    c25 = n // 25
    ans = c5 + c25
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：{n}的阶乘（从1乘到{n}）的末尾有多少个连续的0？",
        f"{name}在练习册上看到一道题：从1一直乘到{n}，积的末尾有多少个0？请你帮他算一算。",
        f"从1一直乘到{n}，所得的积末尾有多少个连续的0？",
        f"{n}的阶乘的末尾有多少个0？",
    ])
    lines = [
        f"从1到{n}中5的倍数的个数 = {c5 * 5} ÷ 5 = {c5}个",
        f"从1到{n}中25的倍数的个数 = {c25 * 25} ÷ 25 = {c25}个",
        f"末尾0的个数 = {c5} + {c25} = {ans}个",
    ]
    return ins, lines, ans


_reg("trailing_zeros", trailing_zeros)


# 37. 斐波那契数列第n项/前n项和
def fibonacci_nth(rng):
    n = rng.randint(6, 14)
    who = rng.choice(["第n项", "前n项和"])
    fib = [0, 1, 1]
    for i in range(3, n + 3):
        fib.append(fib[-1] + fib[-2])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：斐波那契数列的前两项是1和1，从第三项起每一项都等于前两项之和。这个数列的第{n}项是多少？前{n}项的和是多少？（求{('第' + str(n) + '项') if who == '第n项' else '前' + str(n) + '项的和'}）",
        f"斐波那契数列：1，1，2，3，5，8，…从第三个数起，每个数等于前两个数的和。{name}想知道第{n}个数和前{n}个数的和，请你列式算出{('第' + str(n) + '项') if who == '第n项' else '前' + str(n) + '项的和'}。",
        f"有一个数列：1，1，2，3，5，8，13，…从第三项起每项等于前两项之和。第{n}项是多少？前{n}项的和是多少？求{('第' + str(n) + '项') if who == '第n项' else '前' + str(n) + '项的和'}。",
        f"兔子数列的前两项是1、1，以后每项等于前两项之和。这个数列的第{n}项是几？前{n}项的和是几？求{('第' + str(n) + '项') if who == '第n项' else '前' + str(n) + '项的和'}。",
    ])
    lines = []
    for i in range(3, n + 1):
        lines.append(f"第{i}项 = {fib[i - 1]} + {fib[i - 2]} = {fib[i]}")
    if who == "第n项":
        ans = fib[n]
    else:
        ans = fib[n + 2] - 1
        lines.append(f"第{n + 1}项 = {fib[n]} + {fib[n - 1]} = {fib[n + 1]}")
        lines.append(f"第{n + 2}项 = {fib[n + 1]} + {fib[n]} = {fib[n + 2]}")
        lines.append(f"前{n}项的和 = {fib[n + 2]} - 1 = {ans}")
    return ins, lines, ans


_reg("fibonacci_nth", fibonacci_nth)


# 38. 等比数列1,r,r²…第n项/前n项和
def gp_nth_sum(rng):
    r = rng.choice([2, 3])
    n = rng.randint(5, 10)
    who = rng.choice(["第n项", "前n项和"])
    term = r ** (n - 1)
    total = (r ** n - 1) // (r - 1)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数列，第一个数是1，以后每个数都是前一个数的{r}倍。这个数列的第{n}项是多少？前{n}项的和是多少？（求{('第' + str(n) + '项') if who == '第n项' else '前' + str(n) + '项的和'}）",
        f"数列1，{r}，{r * r}，{r ** 3}，…后一个数是前一个数的{r}倍。{name}想知道第{n}项和前{n}项的和，请你列式算出{('第' + str(n) + '项') if who == '第n项' else '前' + str(n) + '项的和'}。",
        f"有一个数列，首项是1，公比是{r}。第{n}项是多少？前{n}项的和是多少？求{('第' + str(n) + '项') if who == '第n项' else '前' + str(n) + '项的和'}。",
        f"一个数列：1，{r}，{r * r}，{r ** 3}，…每个数等于前一个数乘{r}。第{n}项是几？前{n}项的和是几？求{('第' + str(n) + '项') if who == '第n项' else '前' + str(n) + '项的和'}。",
    ])
    lines = []
    cur = 1
    for i in range(2, n + 1):
        nxt = cur * r
        lines.append(f"第{i}项 = {cur} × {r} = {nxt}")
        cur = nxt
    if who == "第n项":
        ans = term
    else:
        ans = total
        lines.append(f"前{n}项的和 = ({term} × {r} - 1) ÷ {r - 1} = {total}")
    return ins, lines, ans


_reg("gp_nth_sum", gp_nth_sum)


# 39. 定义新运算
def custom_operation(rng):
    x = rng.randint(2, 9)
    y = rng.randint(2, 9)
    z = rng.randint(2, 9)
    kind = rng.choice(["plus_minus", "minus_plus", "plus_plus"])
    if kind == "plus_minus":
        v = x * y + x - y
        ans = v * z + v - z
        defn = "a×b+a-b"
        expr1 = f"{x} × {y} + {x} - {y}"
        expr2 = f"{v} × {z} + {v} - {z}"
    elif kind == "minus_plus":
        v = x * y - x + y
        ans = v * z - v + z
        defn = "a×b-a+b"
        expr1 = f"{x} × {y} - {x} + {y}"
        expr2 = f"{v} × {z} - {v} + {z}"
    else:
        v = x * y + x + y
        ans = v * z + v + z
        defn = "a×b+a+b"
        expr1 = f"{x} × {y} + {x} + {y}"
        expr2 = f"{v} × {z} + {v} + {z}"
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师规定一种新运算：a*b = {defn}。照这样计算，({x}*{y})*{z} 等于多少？",
        f"{name}在练习册上看到一种新运算：a*b = {defn}。请你帮他算出 ({x}*{y})*{z} 的结果。",
        f"规定新运算 * ：a*b = {defn}。那么 ({x}*{y})*{z} 等于多少？",
        f"如果 a*b = {defn}，那么 ({x}*{y})*{z} 的结果是多少？",
    ])
    lines = [
        f"括号内的结果 = {expr1} = {v}",
        f"最终结果 = {expr2} = {ans}",
    ]
    return ins, lines, ans


_reg("custom_operation", custom_operation)


# 40. 两集合容斥
def inclusion_exclusion(rng):
    a = rng.randint(15, 35)
    b = rng.randint(15, 35)
    c = rng.randint(3, min(a, b) - 5)
    n = rng.randint(a + b - c + 5, a + b - c + 20)
    none = n - a - b + c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"某班有{n}人，参加数学兴趣小组的有{a}人，参加英语兴趣小组的有{b}人，两个小组都参加的有{c}人。两个小组都没参加的有多少人？",
        f"{name}班共{n}人，参加数学小组的{a}人，参加英语小组的{b}人，两组都参加的{c}人。两个小组都没参加的有多少人？",
        f"五年级一班有{n}人，参加美术组的{a}人，参加音乐组的{b}人，两组都参加的{c}人。两组都没参加的有多少人？",
        f"全班{n}人，参加书法小组的{a}人，参加绘画小组的{b}人，两个都参加的{c}人。两个小组都不参加的有多少人？",
    ])
    lines = [
        f"参加两组的人数和 = {a} + {b} = {a + b}人",
        f"至少参加一组的人数 = {a + b} - {c} = {a + b - c}人",
        f"两组都没参加的人数 = {n} - {a + b - c} = {none}人",
    ]
    return ins, lines, none


_reg("inclusion_exclusion", inclusion_exclusion)


# 41. 抽屉原理
def drawer_principle(rng):
    n = rng.randint(2, 5)
    a = rng.randint(n, n + 8)
    b = rng.randint(n, n + 8)
    c = rng.randint(n, n + 8)
    ans = (n - 1) * 3 + 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个袋子里有红球{a}个、黄球{b}个、蓝球{c}个。至少摸出多少个球，才能保证其中一定有{n}个同色的球？",
        f"盒中有红球{a}个、黄球{b}个、蓝球{c}个。{name}闭着眼睛摸球，至少摸出多少个才能保证有{n}个颜色相同？",
        f"布袋里放着红球{a}个、黄球{b}个、蓝球{c}个。至少取出多少个球，才能保证其中有{n}个同色球？",
        f"袋子里有红、黄、蓝三种颜色的球，分别有{a}个、{b}个、{c}个。至少摸出多少个，才能保证有{n}个同色的球？",
    ])
    lines = [
        f"球的总个数 = {a} + {b} + {c} = {a + b + c}个",
        f"每种颜色先摸的个数 = {n} - 1 = {n - 1}个",
        f"三种颜色共摸的个数 = {n - 1} × 3 = {3 * (n - 1)}个",
        f"至少要摸的个数 = {3 * (n - 1)} + 1 = {ans}个",
    ]
    return ins, lines, ans


_reg("drawer_principle", drawer_principle)


# 42. 烙饼问题
def pancake_time(rng):
    n = rng.randint(3, 12)
    a = rng.randint(2, 5)
    total = n * a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一口锅每次最多放2张饼，每张饼两面都要烙，每面{a}分钟。烙{n}张饼至少需要多少分钟？",
        f"妈妈烙饼，锅里每次最多放2张，每面烙{a}分钟。{name}想知道烙{n}张饼最少要多少分钟，请你算一算。",
        f"烙饼时锅每次能放2张饼，每面需{a}分钟。烙好{n}张饼至少需要多少分钟？",
        f"用一口平底锅烙饼，每次最多烙2张，两面都要烙，每面{a}分钟。烙{n}张饼最少要几分钟？",
    ])
    lines = [
        f"饼的总面数 = {n} × 2 = {2 * n}面",
        f"需要烙的次数 = {2 * n} ÷ 2 = {n}次",
        f"最少需要的时间 = {n} × {a} = {total}分钟",
    ]
    return ins, lines, total


_reg("pancake_time", pancake_time)


# 43. 排队打水最短等候时间
def queue_water_time(rng):
    times = sorted(rng.sample(range(1, 16), 4))
    t1, t2, t3, t4 = times
    total = t1 * 4 + t2 * 3 + t3 * 2 + t4
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"4个人各拿一个水桶去打水，只有一个水龙头，4个人打水分别需要{t1}、{t2}、{t3}、{t4}分钟。怎样安排顺序使4个人的等候时间总和最少？最少是多少分钟？",
        f"甲、乙、丙、丁4人排队打水，打水时间分别为{t1}分钟、{t2}分钟、{t3}分钟、{t4}分钟，水龙头只有一个。{name}想知道怎样安排顺序等候时间总和最少，最少是多少分钟？",
        f"4个人到一个水龙头前打水，所需时间分别是{t1}、{t2}、{t3}、{t4}分钟。按怎样的顺序打水，4人的等候时间总和最少？最少是多少分钟？",
        f"四人打水，时间分别为{t1}分钟、{t2}分钟、{t3}分钟、{t4}分钟，只有一个水龙头。怎样安排使总的等候时间最少？最少多少分钟？",
    ])
    lines = [
        f"第一人打水时4人共等候 = {t1} × 4 = {t1 * 4}分钟",
        f"第二人打水时3人共等候 = {t2} × 3 = {t2 * 3}分钟",
        f"第三人打水时2人共等候 = {t3} × 2 = {t3 * 2}分钟",
        f"第四人打水时1人等候 = {t4} × 1 = {t4}分钟",
        f"等候时间总和 = {t1 * 4} + {t2 * 3} + {t3 * 2} + {t4} = {total}分钟",
    ]
    return ins, lines, total


_reg("queue_water_time", queue_water_time)


# 44. 阶梯水价
def water_tiered(rng):
    a = rng.randint(5, 15)
    p = rng.randint(2, 4)
    q = rng.randint(p + 1, p + 4)
    t = rng.randint(a + 1, a + 20)
    cost = a * p + (t - a) * q
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"自来水公司规定：每户每月用水{a}吨以内（含{a}吨），每吨{p}元；超过{a}吨的部分，每吨{q}元。{name}家上月用水{t}吨，应交水费多少元？",
        f"某市实行阶梯水价：月用水量不超过{a}吨的部分每吨{p}元，超出部分每吨{q}元。{name}家上月用了{t}吨水，要交水费多少元？",
        f"水费标准：每月用水{a}吨以内每吨{p}元，超过{a}吨的部分每吨{q}元。一户上月用水{t}吨，应交多少元水费？",
        f"自来水收费：{a}吨以内每吨{p}元，超过{a}吨每吨{q}元。{name}家上月用水{t}吨，应付水费多少元？",
    ])
    lines = [
        f"第一档的水费 = {a} × {p} = {a * p}元",
        f"超出的吨数 = {t} - {a} = {t - a}吨",
        f"第二档的水费 = {t - a} × {q} = {(t - a) * q}元",
        f"应交的水费 = {a * p} + {(t - a) * q} = {cost}元",
    ]
    return ins, lines, cost


_reg("water_tiered", water_tiered)


# 45. 邮费
def postage(rng):
    a = rng.randint(1, 3)
    b = rng.randint(8, 15)
    c = rng.randint(2, 6)
    w = rng.randint(a + 1, a + 10)
    fee = b + (w - a) * c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"邮局规定：寄包裹首重{a}千克收费{b}元，超过{a}千克的部分每千克收{c}元（不足1千克按1千克算）。{name}寄一个{w}千克的包裹，应付邮费多少元？",
        f"快递收费：首重{a}千克{b}元，续重每千克{c}元。{name}的包裹重{w}千克，应付多少元快递费？",
        f"邮寄包裹，{a}千克以内收费{b}元，超过部分每千克{c}元。一个{w}千克的包裹应付邮费多少元？",
        f"物流公司规定：首重{a}千克收{b}元，续重每千克{c}元。寄{w}千克的货物，要付多少元？",
    ])
    lines = [
        f"续重的千克数 = {w} - {a} = {w - a}千克",
        f"续重的费用 = {w - a} × {c} = {(w - a) * c}元",
        f"应付的邮费 = {b} + {(w - a) * c} = {fee}元",
    ]
    return ins, lines, fee


_reg("postage", postage)


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
    print(f"L4 ext5 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
