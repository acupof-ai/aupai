#!/usr/bin/env python3
"""L3 extension bank 6: structurally novel elementary programs.

Each program: fn(rng) -> (instruction, lines, ans). Every line is an
equation `label = X op Y = Z[u]` (3-part) or `X op Y = Z[u]` (2-part,
pure-arithmetic LHS). Last line value must equal ans. Prose integers >=3
must appear among the equation tokens (enforced by run_math_short.verify).

Structures covered (all absent from l3 base + ext1..ext5):
sum-diff / diff-multiple / sum-multiple-with-remainder, reverse-operations,
mistake-in-calculation, periodic counting, LCM/GCD applications, same-
remainder, chicken-rabbit variants, average-change, travel variants
(messenger, double-meet, three-people, return-meet, shared-bike, escalator,
two-tunnels, post-meet time), work variants (join-later, close-early,
three-people, wage-split, finish-early reverse, rate-ratio), concentration
(add-to-target, mix-ratio, repeated-dilution), interest (tax, principal-
reverse), milling rate, relative fractions, transfers, family-age, counting
(round-robin, tickets, outfits, permutations, drawer, square-array), new
operation, arithmetic-series, manuscript tax, gears, scale-enlarge, circle
planting, geometry (track, annular cylinder, road-roller, cylinder cut,
casting, rectangle cut, cuboid cut, cubes join, trapezoid triangle,
rotation, painted cube).
"""

import random
from math import gcd, lcm
from fractions import Fraction

from mathcommon import num

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L3", name, fn))


def _d(f):
    """Render Fraction as terminating decimal (all denominators here are 2^a*5^b)."""
    return f"{float(Fraction(f)):.10f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Batch 1: number relations, reverse reasoning, cycles, LCM/GCD, chicken-rabbit
# ---------------------------------------------------------------------------

def sum_diff_basic(rng):
    a = rng.randrange(15, 56)       # smaller
    d = rng.randrange(4, 19)        # difference
    b = a + d                       # larger
    s = a + b
    ins = rng.choice([
        f"两个数的和是{s}，差是{d}，较大的数是多少？",
        f"已知两个数相加得{s}，相减得{d}，求较大的数。",
        f"甲、乙两数之和为{s}，之差为{d}，甲数比乙数大，甲数是多少？",
        f"两个数的和是{s}，大数比小数多{d}，大数是多少？",
    ])
    lines = [
        f"小数 = ({s} - {d}) / 2 = {a}",
        f"大数 = ({s} + {d}) / 2 = {b}",
    ]
    return ins, lines, b


_reg("sum_diff_basic", sum_diff_basic)


def sum_diff_three(rng):
    b = rng.randrange(20, 46)
    d1 = rng.randrange(3, 11)
    d2 = rng.randrange(2, 9)
    s = 3 * b + d1 - d2
    ins = rng.choice([
        f"甲、乙、丙三个数的和是{s}，甲比乙多{d1}，丙比乙少{d2}，乙数是多少？",
        f"三个数之和为{s}，其中甲数比乙数多{d1}，丙数比乙数少{d2}，求乙数。",
        f"甲、乙、丙共{s}，甲比乙大{d1}，丙比乙小{d2}，乙数是多少？",
        f"已知甲、乙、丙三个数的和是{s}，甲减乙等于{d1}，乙减丙等于{d2}，乙数是多少？",
    ])
    lines = [
        f"三数和调整 = {s} - {d1} + {d2} = {s - d1 + d2}",
        f"乙数 = {s - d1 + d2} / 3 = {b}",
    ]
    return ins, lines, b


_reg("sum_diff_three", sum_diff_three)


def diff_multiple(rng):
    k = rng.randrange(3, 7)
    b = rng.randrange(8, 26)
    d = (k - 1) * b
    ins = rng.choice([
        f"甲数是乙数的{k}倍，甲数比乙数多{d}，乙数是多少？",
        f"甲是乙的{k}倍，甲比乙多{d}，求乙数。",
        f"两个数，大数是小数的{k}倍，大数比小数多{d}，小数是多少？",
        f"甲数除以乙数商{k}，甲数比乙数多{d}，乙数是多少？",
    ])
    lines = [
        f"倍数差 = {k} - 1 = {k - 1}",
        f"甲数 = {k} × {b} = {k * b}",
        f"乙数 = {d} / {k - 1} = {b}",
    ]
    return ins, lines, b


_reg("diff_multiple", diff_multiple)


def sum_multiple_extra(rng):
    k = rng.randrange(2, 6)
    b = rng.randrange(10, 26)
    r = rng.randrange(1, 9)
    s = (k + 1) * b + r
    ins = rng.choice([
        f"甲、乙两数的和是{s}，甲数比乙数的{k}倍还多{r}，乙数是多少？",
        f"两个数之和为{s}，大数比小数的{k}倍多{r}，求小数。",
        f"甲、乙共{s}，甲比乙的{k}倍多{r}，乙数是多少？",
        f"两数和是{s}，其中一个数比另一个数的{k}倍多{r}，较小的数是多少？",
    ])
    lines = [
        f"和减去零头 = {s} - {r} = {s - r}",
        f"乙数 = {s - r} / ({k} + 1) = {b}",
    ]
    return ins, lines, b


_reg("sum_multiple_extra", sum_multiple_extra)


def reverse_operations(rng):
    x = rng.randrange(10, 36)
    a = rng.randrange(3, 13)
    b = rng.randrange(3, 8)
    c = rng.randrange(2, 11)
    r1 = x + a
    r2 = r1 * b
    r3 = r2 - c
    ins = rng.choice([
        f"一个数加上{a}，再乘{b}，然后减去{c}，结果是{r3}。这个数是多少？",
        f"某数先加{a}，再乘{b}，最后减{c}，得到{r3}。求某数。",
        f"一个数经过加{a}、乘{b}、减{c}后等于{r3}，这个数是多少？",
        f"把一个数加上{a}，所得的和乘{b}，再减去{c}，结果是{r3}。原数是多少？",
    ])
    lines = [
        f"逆推减变加 = {r3} + {c} = {r2}",
        f"逆推乘变除 = {r2} / {b} = {r1}",
        f"原数 = {r1} - {a} = {x}",
    ]
    return ins, lines, x


_reg("reverse_operations", reverse_operations)


def addition_misread(rng):
    m = rng.randrange(1, 9)
    n = rng.randrange(1, 9)
    while n == m:
        n = rng.randrange(1, 9)
    e = rng.randrange(60, 151)
    delta = n - m
    correct = e - delta
    ins = rng.choice([
        f"小明做加法时，把一个加数个位上的{m}看成了{n}，得到错误的和{e}。正确的和是多少？",
        f"计算加法时，把一个加数个位的{m}错写成{n}，结果得{e}。正确的和是多少？",
        f"小马虎做加法，把一个加数个位上的{m}看成{n}，算出和是{e}。正确的和应该是多少？",
        f"一道加法题，把一个加数个位上的{m}看成了{n}，得到的和是{e}。正确的和是多少？",
    ])
    lines = [
        f"和的变化 = {n} - {m} = {delta}",
        f"正确的和 = {e} - {delta} = {correct}",
    ]
    return ins, lines, correct


_reg("addition_misread", addition_misread)


def subtraction_misread(rng):
    m = rng.randrange(1, 9)
    n = rng.randrange(1, 9)
    while n == m:
        n = rng.randrange(1, 9)
    e = rng.randrange(40, 101)
    delta = n - m
    correct = e + delta
    ins = rng.choice([
        f"小明做减法时，把减数个位上的{m}看成了{n}，得到错误的差{e}。正确的差是多少？",
        f"计算减法时，把减数个位的{m}错写成{n}，结果差是{e}。正确的差是多少？",
        f"小马虎做减法，把减数个位上的{m}看成{n}，算出差是{e}。正确的差应该是多少？",
        f"一道减法题，把减数个位上的{m}看成了{n}，得到的差是{e}。正确的差是多少？",
    ])
    lines = [
        f"减数的变化 = {n} - {m} = {delta}",
        f"正确的差 = {e} + {delta} = {correct}",
    ]
    return ins, lines, correct


_reg("subtraction_misread", subtraction_misread)


def period_lanterns(rng):
    q = rng.randrange(5, 19)
    r = rng.randrange(1, 3)
    n = 3 * q + r
    red = q + 1
    ins = rng.choice([
        f"灯笼按“红、黄、蓝”的顺序循环排列，第1个是红色。前{n}个灯笼中共有多少个红灯笼？",
        f"节日灯笼按红、黄、蓝3个一组循环排列，第1个是红灯笼。前{n}个灯笼里有多少个红灯笼？",
        f"一排灯笼按红、黄、蓝、红、黄、蓝……的规律排列，前{n}个灯笼中红灯笼有多少个？",
        f"彩灯按“红、黄、蓝”循环排列，已知第1个是红色，前{n}个彩灯中共有多少个红色的？",
    ])
    lines = [
        f"整组灯笼 = {q} × 3 = {3 * q}个",
        f"余下灯笼 = {n} - {3 * q} = {r}个",
        f"红灯笼 = {q} + 1 = {red}个",
    ]
    return ins, lines, red


_reg("period_lanterns", period_lanterns)


def cycle_nth_weekday(rng):
    q = rng.randrange(2, 9)
    r = rng.randrange(1, 8)
    n = 7 * q + r
    ins = rng.choice([
        f"科技馆每周开放7天，从周一开始循环。第{n}个开放日是星期几（周一用1表示，周日用7表示）？",
        f"某展馆每周7天都开放，按周一到周日循环。第{n}个开放日是星期几（周一为1，周日为7）？",
        f"图书馆每周开放7天，从星期一开始循环开放。第{n}个开放日是星期几（用1到7表示，周一为1）？",
        f"一个每周7天开放的场馆，从周一开始循环。第{n}个开放日是星期几（周一记作1，周日记作7）？",
    ])
    lines = [
        f"经过的整周 = {q} × 7 = {7 * q}天",
        f"余下的天数 = {n} - {7 * q} = {r}天",
        f"星期序号 = {r} = {r}",
    ]
    return ins, lines, r


_reg("cycle_nth_weekday", cycle_nth_weekday)


def lcm_buses_simultaneous(rng):
    a = rng.choice([4, 6, 8])
    b = rng.choice([6, 8, 10, 12])
    c = rng.choice([8, 10, 12, 15])
    g1 = gcd(a, b)
    l1 = a * b // g1
    g2 = gcd(l1, c)
    l = l1 * c // g2
    ins = rng.choice([
        f"三路公交车分别每{a}分钟、{b}分钟、{c}分钟发一班车，同时发车后，至少再过多少分钟三路车又同时发车？",
        f"三条线路的公交车分别每隔{a}、{b}、{c}分钟发一班，早上同时发车后，至少再过多少分钟又同时发车？",
        f"三路公交车分别每{a}分钟、{b}分钟、{c}分钟发一班，某时刻同时发车，最少再过多少分钟再次同时发车？",
        f"三路车发车间隔分别为{a}分钟、{b}分钟、{c}分钟，若现在同时发车，至少再过多少分钟三路车第二次同时发车？",
    ])
    lines = [
        f"前两路间隔 = {a} × {b} / {g1} = {l1}分钟",
        f"三路同时间隔 = {l1} × {c} / {g2} = {l}分钟",
    ]
    return ins, lines, l


_reg("lcm_buses_simultaneous", lcm_buses_simultaneous)


def gcd_square_tiles(rng):
    a = rng.choice([12, 16, 18, 24, 30, 36])
    b = rng.choice([8, 10, 12, 16, 20, 24])
    g = gcd(a, b)
    na, nb = a // g, b // g
    tiles = na * nb
    ins = rng.choice([
        f"一间长{a}分米、宽{b}分米的长方形房间，用最大的正方形地砖铺地（整块），需要多少块地砖？",
        f"长方形地面长{a}米、宽{b}米，铺正方形地砖且不切割，地砖边长最大，共需多少块？",
        f"用正方形地砖铺一块长{a}分米、宽{b}分米的地面，要求地砖边长最大且都是整块，需要多少块？",
        f"一块长{a}米、宽{b}米的长方形地，铺边长最大的正方形地砖（整块铺），一共要多少块？",
    ])
    lines = [
        f"长可铺 = {a} / {g} = {na}块",
        f"宽可铺 = {b} / {g} = {nb}块",
        f"总块数 = {na} × {nb} = {tiles}块",
    ]
    return ins, lines, tiles


_reg("gcd_square_tiles", gcd_square_tiles)


def gcd_three_sticks(rng):
    a, b, c = rng.choice([
        (12, 18, 24), (16, 24, 32), (18, 24, 30),
        (12, 20, 28), (24, 36, 48), (15, 25, 40),
        (18, 30, 42), (20, 30, 40), (16, 28, 40),
        (30, 42, 54), (14, 21, 35), (22, 33, 44),
        (16, 24, 40), (18, 27, 36),
    ])
    g = gcd(gcd(a, b), c)
    segs = a // g + b // g + c // g
    ins = rng.choice([
        f"三根木棒分别长{a}厘米、{b}厘米、{c}厘米，要把它们截成同样长的小段且没有剩余，每段尽可能长，一共能截成多少段？",
        f"有三根铁丝，长度分别为{a}厘米、{b}厘米、{c}厘米，截成相等的小段且无剩余，每段最长，共可截多少段？",
        f"三根竹竿长{a}米、{b}米、{c}米，截成同样长的小段（无剩余），要使每段最长，一共截成多少段？",
        f"把长{a}厘米、{b}厘米、{c}厘米的三根木条截成同样长的小段，不能有剩余，每段最长，一共可截多少段？",
    ])
    lines = [
        f"三根棒总长 = {a} + {b} + {c} = {a + b + c}厘米",
        f"总段数 = {a} / {g} + {b} / {g} + {c} / {g} = {segs}段",
    ]
    return ins, lines, segs


_reg("gcd_three_sticks", gcd_three_sticks)


def remainder_same(rng):
    a = rng.choice([3, 4, 5, 6])
    b = rng.choice([4, 5, 6, 7])
    while b == a:
        b = rng.choice([4, 5, 6, 7])
    r = rng.randrange(1, min(a, b))
    g = gcd(a, b)
    l = a * b // g
    x = l + r
    ins = rng.choice([
        f"一个数除以{a}余{r}，除以{b}也余{r}，这个数最小是多少？",
        f"某数除以{a}余{r}，除以{b}余{r}，求满足条件的最小自然数。",
        f"一个数分别除以{a}和{b}，余数都是{r}，这个数最小是多少？",
        f"有一个数，除以{a}余{r}，除以{b}同样余{r}，这个数最小是几？",
    ])
    lines = [
        f"最小公倍数 = {a} × {b} / {g} = {l}",
        f"所求最小数 = {l} + {r} = {x}",
    ]
    return ins, lines, x


_reg("remainder_same", remainder_same)


def chicken_rabbit_swap(rng):
    c = rng.randrange(4, 15)
    r = rng.randrange(4, 15)
    h = c + r
    l_swap = 4 * c + 2 * r
    ins = rng.choice([
        f"鸡兔同笼，共有{h}个头。如果把鸡和兔的只数互换，脚就变成{l_swap}只。原来兔有多少只？",
        f"鸡兔同笼，共{h}个头。若鸡换成兔、兔换成鸡，则共有脚{l_swap}只。原来有多少只兔？",
        f"笼中鸡兔共{h}只，把鸡兔只数互换后，脚的总数为{l_swap}只。原来兔有多少只？",
        f"鸡兔同笼，头共{h}个。假如鸡与兔互换，脚共有{l_swap}只。原来兔有多少只？",
    ])
    lines = [
        f"原来的鸡 = ({l_swap} - 2 × {h}) / 2 = {c}只",
        f"原来的兔 = {h} - {c} = {r}只",
    ]
    return ins, lines, r


_reg("chicken_rabbit_swap", chicken_rabbit_swap)


def chicken_rabbit_leg_diff(rng):
    c = rng.randrange(4, 13)
    r = rng.randrange(c // 2 + 2, c + 5)
    h = c + r
    d = 4 * r - 2 * c
    ins = rng.choice([
        f"鸡兔同笼，共有{h}个头，兔的总脚数比鸡的总脚数多{d}只。鸡有多少只？",
        f"笼中鸡兔共{h}只，兔脚总数比鸡脚总数多{d}只。鸡有多少只？",
        f"鸡兔同笼，头共{h}个，鸡的总脚数比兔的总脚数少{d}只。鸡有多少只？",
        f"鸡和兔共{h}只，兔的脚数之和比鸡的脚数之和多{d}只。鸡有多少只？",
    ])
    lines = [
        f"兔的只数 = ({d} + 2 × {h}) / 6 = {r}只",
        f"鸡的只数 = {h} - {r} = {c}只",
    ]
    return ins, lines, c


_reg("chicken_rabbit_leg_diff", chicken_rabbit_leg_diff)


def average_change_one(rng):
    n = rng.choice([4, 5, 6])
    orig = rng.randrange(20, 51)
    d = rng.randrange(2, 6)
    a = rng.randrange(30, 50)
    b = a + d
    x = orig + n * d
    ins = rng.choice([
        f"{n}个数的平均数是{a}，把其中一个数改成{x}后，平均数变成{b}。这个数原来是多少？",
        f"有{n}个数，平均数为{a}。若将其中一个数改为{x}，平均数变为{b}，求被改的数。",
        f"{n}个数的平均数是{a}，其中一个数被改成{x}后，平均数是{b}。这个数原来是多少？",
        f"一组{n}个数的平均数为{a}，把其中一个数换成{x}后，平均数变为{b}。原来的数是多少？",
    ])
    lines = [
        f"总和增加 = {n} × ({b} - {a}) = {n * d}",
        f"原数 = {x} - {n * d} = {orig}",
    ]
    return ins, lines, orig


_reg("average_change_one", average_change_one)


# ---------------------------------------------------------------------------
# Batch 2: travel variants + work variants
# ---------------------------------------------------------------------------

def messenger_round_trip(rng):
    L, v, u = rng.choice([
        (180, 60, 120), (360, 60, 120), (540, 60, 120),
        (210, 70, 140), (420, 70, 140),
        (240, 80, 160), (480, 80, 160),
        (200, 50, 150), (400, 50, 150),
        (240, 60, 180), (480, 60, 180),
    ])
    t1 = L // (u - v)
    t2 = L // (u + v)
    ins = rng.choice([
        f"一支队伍长{L}米，以每分钟{v}米的速度前进。通讯员从队尾骑摩托车以每分钟{u}米的速度赶到队头传达命令，再立即返回队尾。通讯员往返一共需要多少分钟？",
        f"队伍长{L}米，前进速度为每分钟{v}米。通讯员从队尾以每分钟{u}米的速度跑步到队头送信，然后马上返回队尾。他往返共需多少分钟？",
        f"一列队伍长{L}米，正以每分钟{v}米的速度行进。队尾的通讯员以每分钟{u}米的速度赶到队头，又以原速返回队尾。一共需要多少分钟？",
        f"行军队伍长{L}米，每分钟前进{v}米。通讯员从队尾骑摩托车（每分钟{u}米）到队头传达命令后立即返回队尾。往返共用多少分钟？",
    ])
    lines = [
        f"追及时间 = {L} / ({u} - {v}) = {t1}分钟",
        f"相遇时间 = {L} / ({u} + {v}) = {t2}分钟",
        f"往返总时间 = {t1} + {t2} = {t1 + t2}分钟",
    ]
    return ins, lines, t1 + t2


_reg("messenger_round_trip", messenger_round_trip)


def double_meet_total(rng):
    a = rng.randrange(30, 81, 2)
    b = rng.randrange(10, 61, 2)
    s = (3 * a + b) // 2
    ins = rng.choice([
        f"甲、乙两人同时从A、B两地相向而行，第一次在距A地{a}千米处相遇。相遇后两人继续前进，到达对方出发地后立即返回，第二次在距A地{b}千米处相遇。A、B两地相距多少千米？",
        f"甲、乙分别从A、B两地同时出发相向而行，第一次相遇点距A地{a}千米。相遇后继续前行，各自到达终点后立即返回，在距A地{b}千米处第二次相遇。求A、B两地的距离。",
        f"甲、乙两车同时从A、B两城相向开出，第一次在距A城{a}千米处相遇。相遇后两车继续前进，分别到达B、A两城后立即返回，在距A城{b}千米处第二次相遇。A、B两城相距多少千米？",
        f"甲、乙两人同时从A、B两地相向而行，第一次相遇时甲走了{a}千米。相遇后两人继续走到对方出发地并立即返回，第二次相遇点距A地{b}千米。A、B两地相距多少千米？",
    ])
    lines = [
        f"甲共行路程 = 3 × {a} = {3 * a}千米",
        f"全程 = ({3 * a} + {b}) / 2 = {s}千米",
    ]
    return ins, lines, s


_reg("double_meet_total", double_meet_total)


def three_people_meet(rng):
    v1, v2, v3, dt = rng.choice([
        (6, 4, 2, 1), (6, 4, 2, 2), (8, 6, 2, 1), (8, 6, 4, 1),
        (7, 5, 3, 1), (9, 6, 3, 1), (9, 6, 3, 2), (10, 6, 2, 1),
        (10, 8, 4, 1), (12, 8, 4, 1), (12, 9, 6, 1),
    ])
    f = (v1 + v3) * (v2 + v3) // (v1 - v2)
    L = f * dt
    ins = rng.choice([
        f"甲、乙两人从A镇、丙从B镇同时出发相向而行。甲每小时行{v1}千米，乙每小时行{v2}千米，丙每小时行{v3}千米。甲与丙相遇后，再过{dt}小时乙与丙相遇。A、B两镇相距多少千米？",
        f"甲、乙同时从东村、丙从西村相向而行，甲速每小时{v1}千米，乙速每小时{v2}千米，丙速每小时{v3}千米。甲、丙相遇后{dt}小时，乙、丙相遇。东西两村相距多少千米？",
        f"甲、乙、丙三人，甲、乙从A地、丙从B地同时出发相向而行。甲每小时{v1}千米，乙每小时{v2}千米，丙每小时{v3}千米。甲和丙相遇后，又过了{dt}小时乙和丙才相遇。A、B两地相距多远？",
        f"甲、乙两人由A地、丙由B地同时相向出发，甲每小时行{v1}千米，乙每小时行{v2}千米，丙每小时行{v3}千米。甲遇到丙后{dt}小时，乙才遇到丙。A、B两地相距多少千米？",
    ])
    lines = [
        f"速度组合 = ({v1} + {v3}) × ({v2} + {v3}) / ({v1} - {v2}) = {f}千米",
        f"两镇距离 = {f} × {dt} = {L}千米",
    ]
    return ins, lines, L


_reg("three_people_meet", three_people_meet)


def school_return_meet(rng):
    v2, u, t, S = rng.choice([
        (60, 180, 5, 900), (50, 150, 6, 900), (70, 210, 4, 840),
        (60, 120, 5, 1200), (80, 240, 3, 720), (90, 180, 4, 1440),
        (50, 100, 6, 1200), (70, 140, 5, 1400), (60, 180, 10, 1800),
        (100, 200, 3, 1200), (80, 160, 5, 1600), (90, 270, 4, 1080),
        (60, 240, 5, 800), (50, 200, 6, 800),
    ])
    ins = rng.choice([
        f"哥哥和弟弟同时从家出发去学校，弟弟每分钟走{v2}米。哥哥走了{t}分钟后发现忘带课本，立即以原速返回家中取课本（取课本时间不计），然后骑自行车以每分钟{u}米的速度追赶弟弟，结果两人同时到达学校。家到学校有多少米？",
        f"兄弟二人同时从家去学校，弟弟步行每分钟{v2}米。哥哥出发{t}分钟后发现忘带作业，马上原速回家去取，取到后立即骑车以每分钟{u}米的速度赶往学校，与弟弟同时到校。家到学校多少米？",
        f"弟弟每分钟走{v2}米，哥哥和他同时从家出发去学校。哥哥走了{t}分钟后想起忘带东西，立即原速回家，拿到后骑车以每分钟{u}米的速度追弟弟，两人同时到校。家到学校有多少米？",
        f"哥哥与弟弟同时离家去学校，弟弟的速度是每分钟{v2}米。哥哥行{t}分钟后因故原速返回家中，再骑车以每分钟{u}米的速度去学校，恰好和弟弟同时到达。家到学校多少米？",
    ])
    lines = [
        f"弟弟用时 = {S} / {v2} = {num(Fraction(S, v2))}分钟",
        f"家到学校 = 2 × {t} × {u} × {v2} / ({u} - {v2}) = {S}米",
    ]
    return ins, lines, S


_reg("school_return_meet", school_return_meet)


def both_ride_walk(rng):
    S = rng.choice([36, 72, 108, 144, 180, 216, 252, 288, 324, 360])
    x = 2 * S // 3
    w = S - x
    ins = rng.choice([
        f"甲、乙两人同时从A地出发到B地，全程{S}千米。甲骑自行车每小时12千米，乙步行每小时6千米。甲骑到途中某地把自行车放下，立即以每小时4千米步行前进；乙走到放车处骑自行车继续前进，两人恰好同时到达B地。甲骑车行了多少千米？",
        f"甲、乙同时从A地去B地，路程{S}千米。甲骑车每小时12千米，乙步行每小时6千米。甲在途中放下车改为每小时4千米步行，乙走到放车点再骑车，两人同时到达。甲骑车行了多少千米？",
        f"全程{S}千米，甲、乙两人同时从A地出发。甲骑自行车（12千米/时），乙步行（6千米/时）。甲骑一段路后放下车步行（4千米/时），乙走到车处骑车前行，两人同时到达B地。甲骑车行了多少千米？",
        f"甲、乙从A地到B地（全程{S}千米），甲骑车12千米/时、步行4千米/时，乙步行6千米/时。甲在途中某地放下车步行，乙走到该地骑车，两人同时到达。甲骑车行了多少千米？",
    ])
    lines = [
        f"甲骑车用时 = {x} / 12 = {x // 12}小时",
        f"甲步行用时 = {w} / 4 = {w // 4}小时",
        f"乙步行用时 = {x} / 6 = {x // 6}小时",
        f"乙骑车用时 = {w} / 12 = {w // 12}小时",
        f"甲骑车路程 = 2 × {S} / 3 = {x}千米",
    ]
    return ins, lines, x


_reg("both_ride_walk", both_ride_walk)


def escalator_up_down(rng):
    t = rng.choice([10, 12, 15, 18, 20, 24, 25, 30, 36, 40])
    N = 4 * t
    ins = rng.choice([
        f"商场自动扶梯匀速向上运行。小明每秒向上走3级，{t}秒到达楼上；以每秒5级向下走，{t}秒到达楼下。扶梯的可见部分有多少级？",
        f"自动扶梯匀速向上行驶。小明每秒向上走3级，{t}秒到楼上；若每秒向下走5级，{t}秒到楼下。扶梯可见部分共多少级？",
        f"商场的扶梯匀速向上运行。小明沿扶梯向上每秒走3级，{t}秒到达楼上；沿扶梯向下每秒走5级，{t}秒到达楼下。扶梯有多少级可见？",
        f"一部匀速向上的自动扶梯，小明每秒向上走3级，{t}秒到达楼上；每秒向下走5级，{t}秒到达楼下。扶梯可见部分有多少级？",
    ])
    lines = [
        f"扶梯速度 = (5 - 3) / 2 = 1级/秒",
        f"可见级数 = (1 + 3) × {t} = {N}级",
    ]
    return ins, lines, N


_reg("escalator_up_down", escalator_up_down)


def train_two_tunnels(rng):
    B1, B2, t1, t2 = rng.choice([
        (300, 500, 40, 60), (250, 450, 30, 50), (400, 700, 50, 80),
        (360, 600, 40, 60), (500, 800, 55, 85), (300, 540, 30, 50),
        (450, 750, 50, 80), (600, 900, 70, 100), (480, 720, 50, 70),
        (350, 650, 40, 70), (550, 850, 60, 90),
    ])
    v = (B2 - B1) // (t2 - t1)
    L = v * t1 - B1
    ins = rng.choice([
        f"一列火车通过长{B1}米的隧道用了{t1}秒，通过长{B2}米的隧道用了{t2}秒（车速不变）。这列火车长多少米？",
        f"火车以同样的速度通过两条隧道：通过{B1}米长的隧道用时{t1}秒，通过{B2}米长的隧道用时{t2}秒。火车长多少米？",
        f"一列火车穿过长{B1}米的隧道需{t1}秒，穿过长{B2}米的隧道需{t2}秒，速度不变。求火车的长度。",
        f"火车通过{B1}米的隧道用了{t1}秒，以同样速度通过{B2}米的隧道用了{t2}秒。这列火车长多少米？",
    ])
    lines = [
        f"火车速度 = ({B2} - {B1}) / ({t2} - {t1}) = {v}米/秒",
        f"火车长度 = {v} × {t1} - {B1} = {L}米",
    ]
    return ins, lines, L


_reg("train_two_tunnels", train_two_tunnels)


def meet_remaining_speed(rng):
    v1, v2, S = rng.choice([
        (6, 3, 36), (10, 5, 60), (12, 6, 72), (8, 4, 48),
        (14, 7, 84), (18, 9, 108), (22, 11, 132), (26, 13, 156),
        (16, 8, 96), (24, 12, 144),
        (8, 2, 40), (12, 3, 60), (16, 4, 80), (20, 5, 100),
        (6, 4, 60), (9, 6, 90), (12, 8, 120), (15, 10, 150),
        (10, 5, 90), (12, 6, 108), (14, 7, 126), (16, 8, 144), (8, 4, 72),
        (9, 3, 72), (12, 4, 96), (15, 5, 120), (18, 6, 144),
    ])
    t = S // (v1 + v2)
    t1 = v2 * t // v1
    brem = v1 * t // v2
    ins = rng.choice([
        f"甲、乙两车同时从相距{S}千米的A、B两地相向而行，甲车每小时行{v1}千米，乙车每小时行{v2}千米。两车相遇后，乙车还要多少小时才能到达A地？",
        f"甲、乙两人同时从相距{S}米的两地相向而行，甲每分钟走{v1}米，乙每分钟走{v2}米。相遇后乙还要走多少分钟才能到达甲的出发地？",
        f"A、B两地相距{S}千米，甲、乙两车同时从两地相向开出，甲车每小时{v1}千米，乙车每小时{v2}千米。相遇后乙车还需多少小时到达A地？",
        f"甲、乙两车同时从相距{S}千米的两地相向而行，速度分别为每小时{v1}千米和{v2}千米。相遇后乙车还要多少小时到达A地？",
    ])
    lines = [
        f"相遇时间 = {S} / ({v1} + {v2}) = {t}小时",
        f"甲剩余时间 = {v2} × {t} / {v1} = {t1}小时",
        f"乙剩余时间 = {v1} × {t} / {v2} = {brem}小时",
    ]
    return ins, lines, brem


_reg("meet_remaining_speed", meet_remaining_speed)


def work_efficiency_up(rng):
    a, b, n = rng.choice([
        (20, 30, 5), (24, 16, 4), (30, 20, 5), (30, 20, 10),
        (36, 12, 12), (40, 10, 10), (15, 10, 5), (18, 9, 3),
        (18, 9, 6), (24, 8, 8), (16, 12, 2), (28, 21, 7),
        (28, 21, 14), (32, 16, 8), (36, 18, 9), (36, 18, 18),
        (40, 24, 16), (45, 30, 15), (48, 16, 16), (48, 24, 12),
        (50, 30, 10),
    ])
    t = b * (a - n) // (a + b)
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成，乙单独做{b}天完成。甲先单独做{n}天后，乙加入一起做，还要多少天完成？",
        f"修一条路，甲队单独修{a}天完成，乙队单独修{b}天完成。甲队先修{n}天，乙队加入合修，还需多少天？",
        f"一批零件，师傅单独做{a}天完成，徒弟单独做{b}天完成。师傅先做{n}天后徒弟加入，还要多少天完成？",
        f"一项工程甲独做{a}天完成，乙独做{b}天完成。甲先做{n}天，剩下的由甲乙合作，还需多少天完成？",
    ])
    lines = [
        f"总工作量 = {a} × {b} = {a * b}份",
        f"甲先做 = {b} × {n} = {b * n}份",
        f"剩余 = {a * b} - {b * n} = {b * (a - n)}份",
        f"合作效率 = {a} + {b} = {a + b}份/天",
        f"还需天数 = {b * (a - n)} / {a + b} = {t}天",
    ]
    return ins, lines, t


_reg("work_efficiency_up", work_efficiency_up)


def pipe_close_early(rng):
    a, b, t = rng.choice([
        (12, 8, 3), (15, 10, 3), (20, 12, 5), (20, 15, 4),
        (24, 16, 6), (18, 12, 3), (18, 12, 6), (16, 12, 4),
        (25, 15, 5), (30, 20, 6), (30, 20, 3), (30, 24, 5),
        (36, 24, 6), (40, 24, 5),
    ])
    rem = (a * b - (a + b) * t) // a
    ins = rng.choice([
        f"一个水池，甲管单独注满需{a}小时，乙管单独注满需{b}小时。两管同时开放{t}小时后甲管关闭，乙管还要多少小时才能注满？",
        f"水池有甲、乙两个进水管，单开甲管{a}小时注满，单开乙管{b}小时注满。两管齐开{t}小时后关掉甲管，乙管还需几小时注满？",
        f"注满一池水，甲管要{a}小时，乙管要{b}小时。两管同时注水{t}小时后甲管发生故障关闭，乙管还要多少小时注满？",
        f"甲管{a}小时、乙管{b}小时可分别注满一池水。两管同时开{t}小时后甲管关闭，只开乙管，还要多少小时注满？",
    ])
    lines = [
        f"总水量 = {a} × {b} = {a * b}份",
        f"已注水 = ({a} + {b}) × {t} = {(a + b) * t}份",
        f"剩余 = {a * b} - {(a + b) * t} = {a * b - (a + b) * t}份",
        f"乙管还需 = {a * b - (a + b) * t} / {a} = {rem}小时",
    ]
    return ins, lines, rem


_reg("pipe_close_early", pipe_close_early)


def work_three_join_later(rng):
    a, b, c, n = rng.choice([
        (12, 8, 6, 3), (15, 10, 6, 5), (20, 12, 15, 4), (20, 12, 15, 8),
        (24, 16, 12, 6), (30, 20, 12, 10), (30, 20, 12, 15), (18, 12, 9, 9),
        (20, 15, 12, 4), (20, 15, 12, 8), (30, 24, 20, 15),
    ])
    L = lcm(lcm(a, b), c)
    ra, rb, rc = L // a, L // b, L // c
    t = (L - ra * n) // (ra + rb + rc)
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成，乙单独做{b}天完成，丙单独做{c}天完成。甲先单独做{n}天，然后乙、丙加入一起做，还要多少天完成？",
        f"修一条路，甲队独修{a}天完成，乙队独修{b}天完成，丙队独修{c}天完成。甲队先修{n}天，乙、丙两队加入合修，还需多少天？",
        f"一批零件，甲、乙、丙单独做分别需{a}天、{b}天、{c}天。甲先做{n}天，乙和丙加入一起做，还要多少天完成？",
        f"一项工程，甲、乙、丙单独完成分别需要{a}天、{b}天、{c}天。甲先单独做{n}天，然后乙、丙加入合作，还需多少天才能完成？",
    ])
    lines = [
        f"甲的效率 = {L} / {a} = {ra}份/天",
        f"乙的效率 = {L} / {b} = {rb}份/天",
        f"丙的效率 = {L} / {c} = {rc}份/天",
        f"甲先做 = {ra} × {n} = {ra * n}份",
        f"剩余 = {L} - {ra * n} = {L - ra * n}份",
        f"三人效率和 = {ra} + {rb} + {rc} = {ra + rb + rc}份/天",
        f"还需天数 = {L - ra * n} / {ra + rb + rc} = {t}天",
    ]
    return ins, lines, t


_reg("work_three_join_later", work_three_join_later)


def work_wage_three(rng):
    a, b, c, W = rng.choice([
        (10, 15, 6, 900), (12, 8, 6, 1080), (15, 10, 6, 1200),
        (20, 12, 15, 2160), (24, 16, 12, 2592), (30, 20, 12, 3600),
        (18, 12, 9, 1458), (16, 12, 8, 1248), (10, 12, 15, 1350),
        (8, 12, 6, 1080), (12, 15, 10, 1350),
    ])
    wa, wb, wc = b * c, a * c, a * b
    tot = wa + wb + wc
    wage = W * wa // tot
    ins = rng.choice([
        f"甲、乙、丙三人单独完成一项工程分别需要{a}天、{b}天、{c}天。三人合作完成后共得工资{W}元，按各自的工作量分配，甲应得多少元？",
        f"一项工程，甲独做{a}天完成，乙独做{b}天完成，丙独做{c}天完成。三人合作完工，共得工资{W}元，按工作量分配，甲分得多少元？",
        f"加工一批零件，甲、乙、丙单独做分别需{a}天、{b}天、{c}天。三人合作完成后得工资{W}元，按工作量分配工资，甲应得多少元？",
        f"甲、乙、丙合做一项工程，单独做分别需{a}天、{b}天、{c}天。完工后共得工资{W}元，按各人完成的工作量分配，甲应得多少元？",
    ])
    lines = [
        f"甲的工作量 = {b} × {c} = {wa}份",
        f"乙的工作量 = {a} × {c} = {wb}份",
        f"丙的工作量 = {a} × {b} = {wc}份",
        f"总份数 = {wa} + {wb} + {wc} = {tot}份",
        f"甲应得 = {W} × {wa} / {tot} = {wage}元",
    ]
    return ins, lines, wage


_reg("work_wage_three", work_wage_three)


def work_restart(rng):
    a, n, t = rng.choice([
        (20, 5, 3), (24, 6, 4), (30, 6, 4), (30, 10, 5),
        (36, 6, 6), (15, 3, 2), (18, 3, 3), (18, 6, 2),
        (16, 4, 2), (16, 2, 2), (25, 5, 5), (25, 10, 3),
        (40, 8, 4), (40, 10, 6), (48, 12, 6), (20, 8, 2),
        (20, 4, 4), (24, 8, 4), (30, 8, 2), (36, 12, 6),
        (50, 10, 5),
    ])
    k = a - t - n
    b = a * k // (a - n - k)
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成。甲先单独做{n}天后乙加入合作，结果比甲单独做提前{t}天完成。乙单独做需要多少天？",
        f"修一条路，甲队单独修{a}天完成。甲队先修{n}天，乙队加入合修，比甲队单独修提前{t}天完成。乙队单独修需要多少天？",
        f"一批零件，甲单独做{a}天完成。甲先做{n}天后乙来帮忙，结果比甲单独做提前{t}天完成。乙单独做需要多少天？",
        f"一项工程甲独做{a}天完成。甲先独做{n}天，然后与乙合作，这样比甲独做提前{t}天完工。乙独做需要多少天？",
    ])
    lines = [
        f"实际工期 = {a} - {t} = {a - t}天",
        f"合作天数 = {a - t} - {n} = {k}天",
        f"乙单独做 = {a} × {k} / ({a} - {n} - {k}) = {b}天",
    ]
    return ins, lines, b


_reg("work_restart", work_restart)


def work_amount_compare(rng):
    a, b, t = rng.choice([
        (12, 8, 5), (15, 10, 5), (18, 12, 5), (20, 15, 7),
        (24, 16, 5), (16, 12, 7), (18, 6, 4), (25, 15, 8),
        (30, 20, 5), (14, 7, 6), (12, 9, 7), (16, 8, 6),
        (20, 5, 5), (24, 12, 3), (18, 9, 3), (15, 5, 4),
        (21, 14, 5), (28, 21, 7),
    ])
    total = a * t
    ct = total // (a + b)
    diff = (a - b) * ct
    ins = rng.choice([
        f"一批零件，甲每小时做{a}个，乙每小时做{b}个。这批零件由甲单独做{t}小时正好完成。若两人合作完成，甲比乙多做多少个？",
        f"加工一批零件，甲每小时做{a}个，乙每小时做{b}个。甲单独做{t}小时可以完成。两人合作完成时，甲比乙多做多少个？",
        f"一批零件由甲单独加工{t}小时完成，甲每小时做{a}个，乙每小时做{b}个。两人合作完成这批零件，甲比乙多做多少个？",
        f"甲每小时做{a}个零件，乙每小时做{b}个。一批零件甲单独做{t}小时完成。若甲、乙合作完成，甲比乙多做多少个零件？",
    ])
    lines = [
        f"零件总数 = {a} × {t} = {total}个",
        f"合作时间 = {total} / ({a} + {b}) = {ct}小时",
        f"甲做 = {a} × {ct} = {a * ct}个",
        f"乙做 = {b} × {ct} = {b * ct}个",
        f"甲比乙多 = {a * ct} - {b * ct} = {diff}个",
    ]
    return ins, lines, diff


_reg("work_amount_compare", work_amount_compare)


def conc_add_solute_target(rng):
    s, c1, c2 = rng.choice([
        (200, 10, 20), (300, 15, 25), (400, 10, 25), (250, 12, 20),
        (500, 8, 20), (300, 10, 25), (240, 10, 25), (360, 10, 25),
        (450, 10, 25), (150, 10, 25), (180, 10, 25), (200, 8, 20),
        (300, 16, 20), (400, 12, 20), (600, 10, 25), (300, 12, 20),
        (320, 10, 25), (450, 12, 20), (480, 10, 25),
    ])
    x = s * (c2 - c1) // (100 - c2)
    ins = rng.choice([
        f"有{s}克含盐{c1}%的盐水，要使盐水的含盐率变为{c2}%，需要再加入多少克盐？",
        f"一杯{s}克的盐水含盐{c1}%，要使含盐率提高到{c2}%，需加盐多少克？",
        f"现有{s}克含盐{c1}%的盐水，加入多少克盐后，含盐率恰好为{c2}%？",
        f"把{s}克含盐{c1}%的盐水的含盐率提高到{c2}%，需要加入多少克盐？",
    ])
    lines = [
        f"原有盐 = {s} × {c1} / 100 = {s * c1 // 100}克",
        f"加盐量 = {s} × ({c2} - {c1}) / (100 - {c2}) = {x}克",
    ]
    return ins, lines, x


_reg("conc_add_solute_target", conc_add_solute_target)


def conc_mix_ratio(rng):
    s1, c1, c2, c = rng.choice([
        (200, 10, 25, 15), (300, 10, 25, 15), (240, 10, 25, 16),
        (300, 15, 30, 20), (200, 15, 30, 20), (400, 10, 30, 20),
        (300, 12, 30, 20), (360, 10, 30, 22), (200, 8, 20, 12),
        (280, 10, 30, 16), (350, 10, 30, 16), (420, 10, 30, 16),
        (300, 8, 20, 14), (240, 15, 35, 20), (320, 15, 35, 25),
        (200, 10, 40, 16), (300, 10, 40, 16), (400, 10, 40, 16),
        (360, 10, 40, 16),
    ])
    s2 = s1 * (c - c1) // (c2 - c)
    ins = rng.choice([
        f"甲杯有{s1}克含盐{c1}%的盐水，乙杯有含盐{c2}%的盐水。把乙杯盐水倒入甲杯混合，要得到含盐{c}%的盐水，需要乙杯盐水多少克？",
        f"现有{s1}克含盐{c1}%的盐水，再加入多少克含盐{c2}%的盐水，才能配成含盐{c}%的盐水？",
        f"甲容器有{s1}克含盐{c1}%的盐水，乙容器是含盐{c2}%的盐水。要配制含盐{c}%的盐水，应取乙容器盐水多少克？",
        f"把含盐{c2}%的盐水加入{s1}克含盐{c1}%的盐水中，使混合后含盐{c}%，需要加入多少克？",
    ])
    lines = [
        f"混合后总重 = {s1} + {s2} = {s1 + s2}克",
        f"需乙杯盐水 = {s1} × ({c} - {c1}) / ({c2} - {c}) = {s2}克",
    ]
    return ins, lines, s2


_reg("conc_mix_ratio", conc_mix_ratio)


# ---------------------------------------------------------------------------
# Batch 3: concentration, interest, fractions, counting
# ---------------------------------------------------------------------------

def conc_repeated_dilution(rng):
    c, n, k = rng.choice([
        (16, 4, 2), (32, 4, 2), (48, 4, 2), (25, 5, 2), (50, 5, 2),
        (75, 5, 2), (16, 2, 3), (24, 2, 3), (32, 2, 3), (40, 2, 3),
        (48, 2, 3), (56, 2, 3), (64, 2, 3), (32, 2, 4), (48, 2, 4),
        (64, 2, 4), (80, 2, 4), (64, 4, 3), (64, 8, 2),
    ])
    c2 = c * (n - 1) ** k // n ** k
    mult = " × ".join([str(n - 1)] * k) + " / " + " / ".join([str(n)] * k)
    ins = rng.choice([
        f"一杯含盐{c}%的盐水，每次倒出其中的1/{n}，再用水加满。这样操作{k}次后，盐水的含盐率是多少？",
        f"有一杯含盐{c}%的盐水，每次倒出1/{n}后用水加满，操作{k}次后，含盐率变为多少？",
        f"容器里有含盐{c}%的盐水，每次倒出1/{n}再补满水，重复{k}次后，含盐率是多少？",
        f"一杯盐水含盐{c}%，每次倒出其中的1/{n}并加满水，{k}次后这杯盐水的含盐率是多少？",
    ])
    lines = [
        f"操作次数 = 1 × {k} = {k}次",
        f"每次剩余 = ({n} - 1) / {n} = {num(Fraction(n - 1, n))}",
        f"最终浓度 = {c} × {mult} = {c2}%",
    ]
    return ins, lines, c2


_reg("conc_repeated_dilution", conc_repeated_dilution)


def interest_tax(rng):
    P, r, t, s = rng.choice([
        (2000, 3, 2, 20), (5000, 3, 3, 20), (4000, 4, 2, 5),
        (6000, 3, 2, 5), (2000, 4, 3, 20), (10000, 3, 2, 20),
        (3000, 3, 2, 5), (5000, 4, 2, 20), (2500, 4, 2, 5),
        (3000, 5, 2, 20), (4000, 3, 3, 5), (6000, 4, 2, 20),
        (5000, 5, 3, 20), (2000, 5, 2, 5), (4500, 4, 2, 5),
        (3500, 4, 2, 20), (5500, 4, 2, 20), (7000, 3, 2, 5),
        (8000, 3, 2, 20), (2500, 3, 2, 20), (3200, 5, 2, 20),
        (3600, 5, 2, 5), (4800, 5, 2, 20), (6000, 5, 2, 5),
        (6400, 5, 2, 20), (7500, 4, 2, 20), (8000, 5, 2, 5),
    ])
    pretax = P * r * t // 100
    after = pretax * (100 - s) // 100
    ins = rng.choice([
        f"爸爸把{P}元存入银行，定期{t}年，年利率{r}%，到期后按{s}%缴纳利息税。税后利息是多少元？",
        f"妈妈存入银行{P}元，定期{t}年，年利率{r}%，利息税率为{s}%。到期后税后利息多少元？",
        f"把{P}元钱存入银行，存期{t}年，年利率{r}%，按{s}%缴纳利息税后，到期实得利息多少元？",
        f"银行一年期年利率{r}%，小明把{P}元存了{t}年定期，利息税{s}%。到期后税后利息是多少元？",
    ])
    lines = [
        f"税前利息 = {P} × {r} / 100 × {t} = {pretax}元",
        f"税后利息 = {pretax} × (100 - {s}) / 100 = {after}元",
    ]
    return ins, lines, after


_reg("interest_tax", interest_tax)


def interest_find_principal(rng):
    r, t, I = rng.choice([
        (3, 2, 120), (4, 3, 240), (5, 2, 300), (3, 3, 270),
        (4, 2, 160), (2, 3, 120), (5, 3, 375), (3, 4, 360),
        (6, 2, 360), (4, 5, 400), (3, 5, 300), (5, 4, 400),
        (2, 5, 200), (6, 3, 540), (4, 4, 320), (3, 2, 180),
        (5, 2, 200), (4, 3, 360), (6, 2, 240), (3, 4, 240),
    ])
    P = 100 * I // (r * t)
    ins = rng.choice([
        f"妈妈把一笔钱存入银行，定期{t}年，年利率{r}%，到期后得到利息{I}元（税前）。妈妈存入的本金是多少元？",
        f"爸爸存入银行一笔钱，定期{t}年，年利率{r}%，到期获利息{I}元。本金是多少元？",
        f"一笔存款存{t}年，年利率{r}%，到期利息为{I}元（未扣税）。这笔存款的本金是多少元？",
        f"小红把压岁钱存入银行，定期{t}年，年利率{r}%，到期得到利息{I}元。她存入了多少元？",
    ])
    lines = [
        f"年利息 = {P} × {r} / 100 = {P * r // 100}元",
        f"本金 = 100 × {I} / ({r} × {t}) = {P}元",
    ]
    return ins, lines, P


_reg("interest_find_principal", interest_find_principal)


def milling_rate(rng):
    w, r = rng.choice([
        (400, 85), (500, 80), (600, 85), (300, 85), (400, 80),
        (800, 85), (250, 80), (350, 80), (450, 80), (550, 80),
        (650, 80), (700, 85), (750, 80), (900, 85), (1000, 85),
        (200, 85), (150, 80), (1200, 85), (1600, 85), (2000, 85),
    ])
    flour = w * r // 100
    bran = w - flour
    diff = flour - bran
    ins = rng.choice([
        f"{w}千克小麦磨成面粉，出粉率是{r}%。面粉比麸皮多多少千克？",
        f"小麦的出粉率是{r}%，{w}千克小麦磨出的面粉比麸皮多多少千克？",
        f"把{w}千克小麦磨成面粉，出粉率为{r}%。面粉比麸皮多多少千克？",
        f"{w}千克小麦去磨面粉，出粉率{r}%，磨出的面粉比麸皮多多少千克？",
    ])
    lines = [
        f"面粉 = {w} × {r} / 100 = {flour}千克",
        f"麸皮 = {w} - {flour} = {bran}千克",
        f"面粉比麸皮多 = {flour} - {bran} = {diff}千克",
    ]
    return ins, lines, diff


_reg("milling_rate", milling_rate)


def fraction_relative(rng):
    n = rng.randrange(3, 9)
    k = rng.randrange(3, 10)
    b = n * k
    jia = b + k
    ins = rng.choice([
        f"乙有{b}元，甲比乙多1/{n}。乙比甲少几分之几？",
        f"甲数比乙数多1/{n}，乙数是{b}。乙数比甲数少几分之几？",
        f"乙有{b}张卡片，甲比乙多1/{n}。乙比甲少几分之几？",
        f"小明有{b}元，小红比小明多1/{n}。小明比小红少几分之几？",
    ])
    lines = [
        f"甲的钱 = {b} × (1 + 1/{n}) = {jia}元",
        f"乙比甲少 = {k} / {jia} = 1/{n + 1}",
    ]
    return ins, lines, Fraction(1, n + 1)


_reg("fraction_relative", fraction_relative)


def give_still_more(rng):
    a, b, d = rng.choice([
        (50, 30, 8), (80, 46, 10), (100, 60, 12), (60, 35, 5),
        (75, 41, 10), (90, 50, 8), (90, 54, 8), (120, 70, 10),
        (55, 30, 5), (65, 39, 6), (70, 40, 6), (85, 51, 12),
        (95, 61, 10), (110, 70, 8), (130, 80, 10), (50, 26, 8),
        (60, 32, 4), (70, 38, 8), (80, 50, 6), (100, 55, 5),
        (100, 56, 8), (100, 58, 4), (100, 62, 6), (100, 64, 8),
        (100, 66, 10), (100, 68, 12), (100, 70, 14), (100, 72, 16),
        (100, 74, 18), (100, 76, 20),
    ])
    x = (a - b - d) // 2
    ins = rng.choice([
        f"甲有{a}元，乙有{b}元。甲给乙多少元后，甲仍比乙多{d}元？",
        f"甲仓有粮{a}吨，乙仓有粮{b}吨。从甲仓运多少吨到乙仓后，甲仓还比乙仓多{d}吨？",
        f"哥哥有{a}元，弟弟有{b}元。哥哥给弟弟多少元后，哥哥仍比弟弟多{d}元？",
        f"甲筐有苹果{a}千克，乙筐有{b}千克。从甲筐取多少千克放入乙筐，甲筐还比乙筐多{d}千克？",
    ])
    lines = [
        f"原来相差 = {a} - {b} = {a - b}元",
        f"甲给乙 = ({a - b} - {d}) / 2 = {x}元",
    ]
    return ins, lines, x


_reg("give_still_more", give_still_more)


def three_people_equal(rng):
    S, a, b = rng.choice([
        (120, 10, 8), (150, 15, 10), (180, 20, 12), (90, 8, 6),
        (210, 25, 15), (240, 30, 18), (270, 30, 20), (300, 40, 25),
        (360, 50, 30), (450, 60, 40), (600, 80, 50),
    ])
    each = S // 3
    bing = each - b
    ins = rng.choice([
        f"甲、乙、丙三人共有{S}元。甲给乙{a}元，乙给丙{b}元后，三人的钱数正好相等。丙原来有多少元？",
        f"甲、乙、丙共有{S}元钱。甲给乙{a}元、乙给丙{b}元后，三人钱数相等。丙原来有多少元？",
        f"三人共有{S}元，甲给乙{a}元，乙给丙{b}元，这时三人钱数相同。丙原来有多少元？",
        f"甲、乙、丙原来共有{S}元。经过甲给乙{a}元、乙给丙{b}元后，三人的钱一样多。丙原来有多少元？",
    ])
    lines = [
        f"最后每人 = {S} / 3 = {each}元",
        f"甲原来 = {each} + {a} = {each + a}元",
        f"乙原来 = {each} + {b} - {a} = {each + b - a}元",
        f"丙原来 = {each} - {b} = {bing}元",
    ]
    return ins, lines, bing


_reg("three_people_equal", three_people_equal)


def family_age_sum(rng):
    a, Sp, c = rng.choice([
        (8, 47, 5), (10, 50, 6), (6, 44, 4), (12, 55, 7),
        (8, 41, 3), (10, 52, 8), (7, 45, 5), (9, 48, 4),
        (11, 54, 6), (12, 51, 5), (15, 60, 10), (14, 58, 8),
        (16, 62, 9), (18, 66, 12), (20, 70, 13),
    ])
    S = Sp + 2 * a + c
    ins = rng.choice([
        f"今年小明一家三口的年龄和是{S}岁，而{a}年前全家年龄和是{Sp}岁。小明今年多少岁？",
        f"今年小华家三口人的年龄和是{S}岁，{a}年前全家年龄和为{Sp}岁。小华今年多少岁？",
        f"一家三口今年的年龄和是{S}岁，{a}年前这个家庭的年龄和是{Sp}岁。孩子今年多少岁？",
        f"今年爸爸、妈妈和小红的年龄和是{S}岁，{a}年前全家年龄和是{Sp}岁。小红今年多少岁？",
    ])
    lines = [
        f"父母年龄共增 = 2 × {a} = {2 * a}岁",
        f"孩子今年 = {S} - {Sp} - {2 * a} = {c}岁",
    ]
    return ins, lines, c


_reg("family_age_sum", family_age_sum)


def bookshelf_transfer(rng):
    S, x = rng.choice([
        (120, 15), (150, 20), (200, 30), (180, 25), (240, 35),
        (160, 20), (220, 30), (260, 40), (300, 45), (140, 18),
        (170, 25), (190, 28), (210, 32), (250, 36), (280, 42),
    ])
    half = S // 2
    up = half + x
    down = half - x
    ins = rng.choice([
        f"一个书架上、下两层共有{S}本书。如果从上层拿{x}本放到下层，两层的书就同样多。上层原来有多少本书？",
        f"书架上、下两层共放书{S}本。从上层拿{x}本到下层后，两层书数相等。上层原来有多少本？",
        f"书架两层共有{S}本书，把上层的{x}本移到下层，两层就一样多。上层原来有多少本书？",
        f"一个书架有上、下两层，共{S}本书。若从上层取出{x}本放入下层，则两层书数相等。上层原有多少本？",
    ])
    lines = [
        f"两层相等时 = {S} / 2 = {half}本",
        f"下层原来 = {half} - {x} = {down}本",
        f"上层原来 = {half} + {x} = {up}本",
    ]
    return ins, lines, up


_reg("bookshelf_transfer", bookshelf_transfer)


def round_robin_matches(rng):
    n = rng.randrange(6, 17)
    m = n * (n - 1) // 2
    ins = rng.choice([
        f"有{n}支球队进行单循环比赛（每两队之间赛一场），一共要比赛多少场？",
        f"学校举行足球赛，共有{n}个队参加，比赛采用单循环制（每两队赛一场）。一共要赛多少场？",
        f"{n}个同学进行乒乓球比赛，每两人之间都要赛一场，一共要比赛多少场？",
        f"一次象棋比赛有{n}人参加，每两人都要下一盘，一共要下多少盘？",
    ])
    lines = [
        f"每队比赛 = {n} - 1 = {n - 1}场",
        f"比赛总场数 = {n} × {n - 1} / 2 = {m}场",
    ]
    return ins, lines, m


_reg("round_robin_matches", round_robin_matches)


def train_tickets_count(rng):
    n = rng.randrange(5, 16)
    one_way = n * (n - 1) // 2
    total = n * (n - 1)
    ins = rng.choice([
        f"一条铁路线上有{n}个车站，一共需要准备多少种不同的车票（往返车票不同）？",
        f"某铁路沿线共有{n}个车站，往返车票不同，一共要准备多少种车票？",
        f"一列火车在{n}个车站之间运行，每个车站都要有到其他各站的车票，往返票不同，共需多少种车票？",
        f"铁路线上有{n}个站，任意两站之间都要有车票，且往返车票不同，一共需要多少种车票？",
    ])
    lines = [
        f"单程车票 = {n} × ({n} - 1) / 2 = {one_way}种",
        f"往返车票 = {one_way} × 2 = {total}种",
    ]
    return ins, lines, total


_reg("train_tickets_count", train_tickets_count)


def outfits_combinations(rng):
    a = rng.randrange(3, 7)
    b = rng.randrange(3, 6)
    ins = rng.choice([
        f"小明有{a}件上衣和{b}条裤子，一件上衣配一条裤子，一共有多少种不同的穿法？",
        f"衣柜里有{a}件上衣、{b}条裤子，每次穿一件上衣和一条裤子，共有多少种搭配方法？",
        f"小红有{a}件不同的上衣和{b}条不同的裤子，一件上衣配一条裤子，有多少种不同的穿法？",
        f"有{a}件上衣和{b}条裤子，任意一件上衣与一条裤子搭配，共有多少种搭配？",
    ])
    lines = [
        f"每件上衣搭配 = 1 × {b} = {b}种",
        f"每条裤子搭配 = 1 × {a} = {a}种",
        f"搭配总数 = {a} × {b} = {a * b}种",
    ]
    return ins, lines, a * b


_reg("outfits_combinations", outfits_combinations)


def permutation_three_digits(rng):
    n = rng.randrange(4, 12)
    total = n * (n - 1) * (n - 2)
    ins = rng.choice([
        f"用1到{n}这{n}个数字组成没有重复数字的三位数，一共可以组成多少个？",
        f"从1、2、…、{n}这{n}个数字中任取三个组成没有重复数字的三位数，共可组成多少个？",
        f"用1~{n}这{n}个数字，能组成多少个没有重复数字的三位数？",
        f"从1到{n}的{n}个数字中选三个不同的数字组成三位数，一共能组成多少个？",
    ])
    lines = [
        f"百位选择 = 1 × {n} = {n}种",
        f"十位选择 = {n} - 1 = {n - 1}种",
        f"个位选择 = {n} - 2 = {n - 2}种",
        f"三位数总数 = {n} × {n - 1} × {n - 2} = {total}个",
    ]
    return ins, lines, total


_reg("permutation_three_digits", permutation_three_digits)


def drawer_socks(rng):
    b = rng.randrange(4, 11)
    c = rng.randrange(4, 11)
    ans = b + c + 2
    ins = rng.choice([
        f"抽屉里有红色、黑色、白色三种袜子混在一起，其中黑色{b}只、白色{c}只。至少摸出多少只，才能保证其中有2只红色袜子？",
        f"布袋里有红、黑、白三种袜子，黑色{b}只、白色{c}只。至少摸出多少只，才能保证摸到2只红色袜子？",
        f"盒子里混放着红、黑、白三种袜子，已知黑色{b}只、白色{c}只。至少拿出多少只，才能保证其中有2只红袜子？",
        f"抽屉中混放着三种颜色的袜子：黑色{b}只、白色{c}只，其余是红色。至少摸出多少只才能保证有2只红色袜子？",
    ])
    lines = [
        f"最不利情况 = {b} + {c} = {b + c}只",
        f"至少摸出 = {b + c} + 2 = {ans}只",
    ]
    return ins, lines, ans


_reg("drawer_socks", drawer_socks)


def square_array_outer(rng):
    n = rng.randrange(6, 21)
    outer = 4 * (n - 1)
    ins = rng.choice([
        f"同学们排成一个{n}行{n}列的方阵，最外层一共有多少人？",
        f"学生排成{n}行{n}列的实心方阵，最外层有多少名学生？",
        f"运动会开幕式上，同学们组成了{n}行{n}列的方阵，最外层一共有多少人？",
        f"一个{n}行{n}列的方阵队伍，最外层共有多少人？",
    ])
    lines = [
        f"每边去掉角 = {n} - 1 = {n - 1}人",
        f"最外层人数 = 4 × {n - 1} = {outer}人",
    ]
    return ins, lines, outer


_reg("square_array_outer", square_array_outer)


def new_operation(rng):
    a, b, c = rng.choice([
        (4, 3, 2), (4, 3, 3), (4, 3, 4), (4, 3, 5),
        (5, 4, 2), (5, 4, 3), (5, 4, 4), (5, 4, 5),
        (6, 5, 2), (6, 5, 3), (6, 5, 4), (6, 5, 5),
        (4, 4, 2), (4, 4, 3), (4, 4, 4), (4, 4, 5),
        (5, 3, 2), (5, 3, 3), (5, 3, 4), (5, 3, 5),
        (6, 4, 2), (6, 4, 3), (6, 4, 4), (6, 4, 5),
    ])
    inner = a * b - a - b
    outer = inner * c - inner - c
    ins = rng.choice([
        f"规定新运算“※”：a※b = a×b − a − b。求（{a}※{b}）※{c}的值。",
        f"定义运算“※”：a※b = a×b − a − b。计算（{a}※{b}）※{c}。",
        f"对于任意数a、b，规定a※b = a×b − a − b。求（{a}※{b}）※{c}等于多少？",
        f"新运算“※”定义为a※b = a×b − a − b，求（{a}※{b}）※{c}的结果。",
    ])
    lines = [
        f"括号内 = {a} × {b} - {a} - {b} = {inner}",
        f"原式 = {inner} × {c} - {inner} - {c} = {outer}",
    ]
    return ins, lines, outer


_reg("new_operation", new_operation)


# ---------------------------------------------------------------------------
# Batch 4: geometry + misc applications
# ---------------------------------------------------------------------------

def arithmetic_pile_sum(rng):
    a = rng.randrange(3, 9)
    n = rng.randrange(5, 11)
    b = a + n - 1
    total = (a + b) * n // 2
    ins = rng.choice([
        f"一堆钢管，最上层有{a}根，最下层有{b}根，每相邻两层相差1根。这堆钢管一共有多少根？",
        f"建筑工地有一堆钢管，最上层{a}根，最下层{b}根，自上而下每层多1根。这堆钢管共多少根？",
        f"一堆圆木堆成梯形，最上层有{a}根，最下层有{b}根，每相邻两层差1根。这堆圆木一共有多少根？",
        f"仓库里有一堆钢管，最上层{a}根，最下层{b}根，每层相差1根。这堆钢管共有多少根？",
    ])
    lines = [
        f"层数 = {b} - {a} + 1 = {n}层",
        f"钢管总数 = ({a} + {b}) × {n} / 2 = {total}根",
    ]
    return ins, lines, total


_reg("arithmetic_pile_sum", arithmetic_pile_sum)


def manuscript_fee_tax(rng):
    M = 800 + 50 * rng.randrange(8, 61)
    tax = (M - 800) * 14 // 100
    ins = rng.choice([
        f"李老师获得稿酬{M}元，按规定超过800元的部分按14%缴纳个人所得税。李老师应纳税多少元？",
        f"王叔叔发表文章得稿酬{M}元，国家规定超过800元的部分按14%纳税。他应缴纳个人所得税多少元？",
        f"张老师得到一笔稿酬{M}元，其中超过800元的部分要按14%缴纳个人所得税。张老师应纳税多少元？",
        f"小明的爸爸获得稿酬{M}元，按税法规定超过800元的部分按14%缴纳个人所得税。他应纳税多少元？",
    ])
    lines = [
        f"应纳税部分 = {M} - 800 = {M - 800}元",
        f"应纳税 = {M - 800} × 14 / 100 = {tax}元",
    ]
    return ins, lines, tax


_reg("manuscript_fee_tax", manuscript_fee_tax)


def gear_teeth_turns(rng):
    a, b, n = rng.choice([
        (36, 12, 20), (40, 16, 20), (48, 12, 15), (42, 14, 20),
        (45, 15, 20), (48, 16, 20), (56, 14, 25), (36, 18, 30),
        (40, 20, 30), (48, 24, 30), (60, 15, 20), (60, 12, 15),
        (32, 16, 25), (30, 15, 25), (28, 14, 30), (24, 12, 40),
        (50, 20, 18), (45, 18, 24), (36, 24, 40), (42, 28, 40),
    ])
    small = a * n // b
    ins = rng.choice([
        f"一对互相咬合的齿轮，大齿轮有{a}个齿，小齿轮有{b}个齿。大齿轮每分钟转{n}转，小齿轮每分钟转多少转？",
        f"两个互相咬合的齿轮，大齿轮{a}个齿，小齿轮{b}个齿。大齿轮每分钟转{n}周，小齿轮每分钟转多少周？",
        f"机器上有一对咬合的齿轮，大齿轮有{a}个齿，小齿轮有{b}个齿。大齿轮每分钟转{n}转，小齿轮每分钟转多少转？",
        f"大小两个齿轮互相咬合，大齿轮有{a}个齿，小齿轮有{b}个齿。大齿轮每分钟转{n}转，小齿轮每分钟转多少转？",
    ])
    lines = [
        f"大齿轮每分钟转过 = {a} × {n} = {a * n}齿",
        f"小齿轮转速 = {a * n} / {b} = {small}转/分",
    ]
    return ins, lines, small


_reg("gear_teeth_turns", gear_teeth_turns)


def enlarge_area_increase(rng):
    a = rng.randrange(4, 11)
    b = rng.randrange(3, 9)
    k = rng.randrange(2, 5)
    old = a * b
    new = old * k * k
    inc = new - old
    ins = rng.choice([
        f"一个长方形长{a}厘米、宽{b}厘米，把它的长和宽都放大到原来的{k}倍，面积增加多少平方厘米？",
        f"长方形的长是{a}厘米，宽是{b}厘米。将长和宽分别扩大到原来的{k}倍，面积比原来增加多少平方厘米？",
        f"一个长方形长{a}厘米、宽{b}厘米，把它的长和宽都放大{k}倍，面积增加了多少平方厘米？",
        f"把长{a}厘米、宽{b}厘米的长方形的长和宽都扩大到原来的{k}倍，面积增加多少平方厘米？",
    ])
    lines = [
        f"放大后长 = {a} × {k} = {a * k}厘米",
        f"放大后宽 = {b} × {k} = {b * k}厘米",
        f"原面积 = {a} × {b} = {old}平方厘米",
        f"放大后面积 = {a * k} × {b * k} = {new}平方厘米",
        f"面积增加 = {new} - {old} = {inc}平方厘米",
    ]
    return ins, lines, inc


_reg("enlarge_area_increase", enlarge_area_increase)


def circle_tree_peach(rng):
    L, d = rng.choice([
        (120, 4), (150, 5), (180, 6), (240, 4), (300, 5), (360, 6),
        (180, 3), (240, 6), (300, 6), (420, 6), (480, 4), (600, 5),
        (270, 5), (360, 4), (450, 5), (540, 6), (150, 3), (210, 3),
        (330, 5), (390, 6),
    ])
    n = L // d
    ins = rng.choice([
        f"一个圆形湖的周长是{L}米，沿湖边每隔{d}米栽一棵桃树，一共要栽多少棵桃树？",
        f"沿一个周长{L}米的圆形池塘边每隔{d}米栽一棵树，一共要栽多少棵树？",
        f"圆形花坛周长{L}米，沿坛边每隔{d}米栽一棵月季，一共要栽多少棵？",
        f"一个圆形操场周长{L}米，沿周围每隔{d}米插一面彩旗，一共要插多少面彩旗？",
    ])
    lines = [
        f"湖的周长 = {d} × {n} = {L}米",
        f"植树棵数 = {L} / {d} = {n}棵",
    ]
    return ins, lines, n


_reg("circle_tree_peach", circle_tree_peach)


def stadium_track_laps(rng):
    a = rng.choice([80, 90, 100, 110, 120])
    n = rng.choice([3, 4, 5, 6, 7, 8])
    perimeter = 2 * a + 157
    total = perimeter * n
    ins = rng.choice([
        f"一个运动场的跑道由两条直道和两个半圆形弯道组成，直道长{a}米，弯道的直径是50米（π取3.14）。沿跑道跑{n}圈，一共跑多少米？",
        f"运动场跑道的直道长{a}米，两端弯道合成一个直径50米的圆（π取3.14）。跑{n}圈一共多少米？",
        f"学校操场跑道由两条{a}米的直道和两个半圆形弯道组成，弯道直径50米（π取3.14）。沿跑道跑{n}圈是多少米？",
        f"一个标准运动场，直道各长{a}米，弯道是直径50米的半圆（π取3.14）。沿跑道跑{n}圈，共跑多少米？",
    ])
    lines = [
        f"弯道总长 = 3.14 × 50 = 157米",
        f"一圈周长 = 2 × {a} + 157 = {perimeter}米",
        f"{n}圈路程 = {perimeter} × {n} = {total}米",
    ]
    return ins, lines, total


_reg("stadium_track_laps", stadium_track_laps)


def annular_cylinder_volume(rng):
    D, d, L = rng.choice([
        (10, 6, 25), (10, 6, 50), (10, 6, 75), (10, 6, 100),
        (12, 8, 25), (12, 8, 50), (12, 8, 100),
        (20, 10, 20), (20, 10, 40),
        (8, 4, 25), (8, 4, 50), (8, 4, 100),
        (14, 10, 25), (14, 10, 50),
        (16, 8, 25), (16, 8, 50),
        (6, 4, 20), (6, 4, 40), (6, 4, 100),
    ])
    ring = (D * D - d * d) // 4
    area = Fraction(157, 50) * ring
    V = area * L
    ins = rng.choice([
        f"一根空心钢管，外直径{D}厘米，内直径{d}厘米，长{L}厘米。这根钢管的体积是多少立方厘米（π取3.14）？",
        f"一根圆柱形空心钢管，外直径{D}厘米，内直径{d}厘米，管长{L}厘米。它的体积是多少立方厘米（π取3.14）？",
        f"空心钢管的外直径是{D}厘米，内直径是{d}厘米，长{L}厘米。求钢管的体积（π取3.14）。",
        f"一根钢管外直径{D}厘米、内直径{d}厘米、长{L}厘米，这根钢管的体积是多少立方厘米（π取3.14）？",
    ])
    lines = [
        f"钢管横截面积 = 3.14 × ({D} × {D} - {d} × {d}) / 4 = {_d(area)}平方厘米",
        f"钢管体积 = {_d(area)} × {L} = {_d(V)}立方厘米",
    ]
    return ins, lines, V


_reg("annular_cylinder_volume", annular_cylinder_volume)


def road_roller_area(rng):
    d, w, n = rng.choice([
        (1, 2, 10), (1, 2, 25), (1.5, 2, 10), (1.5, 2, 20),
        (2, 1.5, 10), (2, 1.5, 20), (1.2, 2, 10), (1.2, 2, 25),
        (1.5, 1.8, 10), (1.5, 1.8, 20), (1.6, 1.5, 10), (1.6, 1.5, 25),
        (0.8, 2, 10), (0.8, 2, 25), (1, 1.8, 10), (1, 1.8, 25),
    ])
    circ = Fraction(157, 50) * Fraction(str(d))
    area = circ * Fraction(str(w)) * n
    ins = rng.choice([
        f"压路机前轮直径{d}米，轮宽{w}米，每分钟转{n}周。每分钟压路多少平方米？",
        f"一台压路机的前轮是圆柱形，直径{d}米，轮宽{w}米。前轮每分钟转{n}周，每分钟压路多少平方米？",
        f"压路机前轮直径{d}米，宽{w}米，每分钟滚动{n}周。每分钟能压多少平方米的路面？",
        f"一种压路机前轮直径{d}米，轮宽{w}米，每分钟转{n}周。它每分钟压路多少平方米？",
    ])
    lines = [
        f"前轮周长 = 3.14 × {d} = {_d(circ)}米",
        f"每分钟压路 = {_d(circ)} × {w} × {n} = {_d(area)}平方米",
    ]
    return ins, lines, area


_reg("road_roller_area", road_roller_area)


def cylinder_cut_surface(rng):
    r = rng.randrange(2, 7)
    h = rng.randrange(5, 16)
    inc = 4 * r * h
    ins = rng.choice([
        f"一个圆柱底面半径{r}分米，高{h}分米。沿底面直径把它切成两半，表面积增加多少平方分米？",
        f"圆柱形木料底面半径{r}分米，高{h}分米。沿底面直径纵切成两块，表面积增加多少平方分米？",
        f"一个圆柱的底面半径是{r}分米，高是{h}分米。沿着底面直径把它切成两个半圆柱，表面积增加多少平方分米？",
        f"把底面半径{r}分米、高{h}分米的圆柱沿底面直径切成两半，表面积增加多少平方分米？",
    ])
    lines = [
        f"一个切面面积 = 2 × {r} × {h} = {2 * r * h}平方分米",
        f"增加的表面积 = 2 × {2 * r * h} = {inc}平方分米",
    ]
    return ins, lines, inc


_reg("cylinder_cut_surface", cylinder_cut_surface)


def cylinder_to_cone_height(rng):
    r, R, h = rng.choice([
        (2, 3, 3), (2, 3, 6), (2, 3, 9), (2, 3, 12),
        (3, 6, 4), (3, 6, 8), (3, 6, 12), (3, 6, 16),
        (4, 2, 1), (4, 2, 2), (4, 2, 3),
        (6, 3, 1), (6, 3, 2), (6, 3, 3),
        (3, 9, 6), (3, 9, 9), (3, 9, 12), (3, 9, 15),
        (4, 6, 3), (4, 6, 6), (4, 6, 9),
        (5, 10, 4), (5, 10, 8), (5, 10, 12),
        (8, 4, 1), (8, 4, 2),
        (10, 5, 1), (10, 5, 2),
    ])
    H = 3 * r * r * h // (R * R)
    vc = Fraction(157, 50) * r * r * h
    ins = rng.choice([
        f"把一个底面半径{r}厘米、高{h}厘米的圆柱形铁块熔铸成一个底面半径{R}厘米的圆锥形零件，这个圆锥形零件的高是多少厘米？",
        f"将底面半径{r}厘米、高{h}厘米的圆柱熔铸成底面半径{R}厘米的圆锥，圆锥的高是多少厘米？",
        f"一个圆柱形铁块底面半径{r}厘米、高{h}厘米，把它熔铸成底面半径{R}厘米的圆锥，圆锥高多少厘米？",
        f"把底面半径{r}厘米、高{h}厘米的圆柱钢材熔铸成底面半径{R}厘米的圆锥形零件，这个零件高多少厘米？",
    ])
    lines = [
        f"圆柱体积 = 3.14 × {r} × {r} × {h} = {_d(vc)}立方厘米",
        f"圆锥的高 = 3 × {r} × {r} × {h} / ({R} × {R}) = {H}厘米",
    ]
    return ins, lines, H


_reg("cylinder_to_cone_height", cylinder_to_cone_height)


def rectangle_cut_square(rng):
    a = rng.randrange(10, 31)
    b = rng.randrange(6, a - 1)
    p = 2 * a
    ins = rng.choice([
        f"一张长方形纸，长{a}厘米、宽{b}厘米。从这张纸上剪下一个最大的正方形，剩下图形的周长是多少厘米？",
        f"一块长方形布长{a}厘米、宽{b}厘米，剪下一个最大的正方形后，剩下布的周长是多少厘米？",
        f"在长{a}厘米、宽{b}厘米的长方形纸上剪一个最大的正方形，余下部分的周长是多少厘米？",
        f"一张长{a}厘米、宽{b}厘米的长方形纸，剪去一个最大的正方形，剩下的长方形周长是多少厘米？",
    ])
    lines = [
        f"剩余长方形长 = {a} - {b} = {a - b}厘米",
        f"剩余周长 = 2 × ({a - b} + {b}) = {p}厘米",
    ]
    return ins, lines, p


_reg("rectangle_cut_square", rectangle_cut_square)


def cuboid_cut_surface(rng):
    a = rng.randrange(6, 16)
    b = rng.randrange(4, 11)
    inc = 2 * a * b
    ins = rng.choice([
        f"一个长方体长{a}厘米、宽{b}厘米，把它切成两个长方体（切面平行于上、下底面），表面积增加多少平方厘米？",
        f"把一个长{a}厘米、宽{b}厘米的长方体木料沿水平方向切成两个长方体，表面积增加多少平方厘米？",
        f"一个长方体长{a}厘米、宽{b}厘米，沿平行于底面的方向把它切成两个长方体，表面积增加多少平方厘米？",
        f"有一块长{a}厘米、宽{b}厘米的长方体豆腐，沿水平方向切成两块，表面积增加多少平方厘米？",
    ])
    lines = [
        f"切面面积 = {a} × {b} = {a * b}平方厘米",
        f"增加的表面积 = 2 × {a * b} = {inc}平方厘米",
    ]
    return ins, lines, inc


_reg("cuboid_cut_surface", cuboid_cut_surface)


def cubes_join_surface(rng):
    n = rng.randrange(3, 9)
    a = rng.randrange(2, 6)
    s = (4 * n + 2) * a * a
    ins = rng.choice([
        f"把{n}个棱长{a}厘米的正方体排成一排拼成一个长方体，这个长方体的表面积是多少平方厘米？",
        f"将{n}个棱长{a}厘米的正方体木块拼成一个长方体（排成一排），长方体的表面积是多少平方厘米？",
        f"{n}个棱长{a}厘米的正方体排成一行拼成一个长方体，拼成的长方体表面积是多少平方厘米？",
        f"把{n}个棱长是{a}厘米的正方体拼成一个长方体（排成一排），这个长方体的表面积是多少平方厘米？",
    ])
    lines = [
        f"长方体长 = {n} × {a} = {n * a}厘米",
        f"表面积 = ({n * a} × {a} + {n * a} × {a} + {a} × {a}) × 2 = {s}平方厘米",
    ]
    return ins, lines, s


_reg("cubes_join_surface", cubes_join_surface)


def trapezoid_max_triangle(rng):
    a = rng.randrange(4, 13, 2)
    b = rng.randrange(a + 2, a + 11)
    h = rng.choice([4, 6, 8, 10])
    rem = a * h // 2
    ins = rng.choice([
        f"一个梯形上底{a}厘米、下底{b}厘米、高{h}厘米。从这个梯形中剪去一个最大的三角形，剩下的面积是多少平方厘米？",
        f"梯形的上底是{a}厘米，下底是{b}厘米，高是{h}厘米。剪去一个最大的三角形后，余下的面积是多少平方厘米？",
        f"一块梯形纸板，上底{a}厘米、下底{b}厘米、高{h}厘米。剪去一个最大的三角形，剩下的面积是多少平方厘米？",
        f"在一个上底{a}厘米、下底{b}厘米、高{h}厘米的梯形中剪去最大的三角形，剩余部分的面积是多少平方厘米？",
    ])
    lines = [
        f"最大三角形面积 = {b} × {h} / 2 = {b * h // 2}平方厘米",
        f"梯形面积 = ({a} + {b}) × {h} / 2 = {(a + b) * h // 2}平方厘米",
        f"剩余面积 = {(a + b) * h // 2} - {b * h // 2} = {rem}平方厘米",
    ]
    return ins, lines, rem


_reg("trapezoid_max_triangle", trapezoid_max_triangle)


def rotate_cylinder_volume(rng):
    a = rng.randrange(4, 11)
    b = rng.randrange(2, 6)
    V = Fraction(157, 50) * b * b * a
    ins = rng.choice([
        f"一个长{a}厘米、宽{b}厘米的长方形，以长边为轴旋转一周，得到的圆柱体积是多少立方厘米（π取3.14）？",
        f"把长{a}厘米、宽{b}厘米的长方形绕它的长边旋转一周，形成的圆柱体积是多少立方厘米（π取3.14）？",
        f"一张长{a}厘米、宽{b}厘米的长方形纸，以长边为轴旋转一周，得到的立体图形体积是多少立方厘米（π取3.14）？",
        f"长方形的长是{a}厘米，宽是{b}厘米，以长边为轴旋转一周，所得圆柱的体积是多少立方厘米（π取3.14）？",
    ])
    lines = [
        f"底面半径 = {b} × 1 = {b}厘米",
        f"圆柱体积 = 3.14 × {b} × {b} × {a} = {_d(V)}立方厘米",
    ]
    return ins, lines, V


_reg("rotate_cylinder_volume", rotate_cylinder_volume)


def cube_paint_faces(rng):
    n = rng.randrange(4, 15)
    two = 12 * (n - 2)
    ins = rng.choice([
        f"一个正方体木块，表面涂满红色后，切成棱长1厘米的小正方体（原正方体棱长{n}厘米）。两面涂色的小正方体有多少个？",
        f"把一个棱长{n}厘米的正方体表面涂上颜色，然后切成棱长1厘米的小正方体。两面涂色的小正方体有多少个？",
        f"一个棱长{n}厘米的正方体，六个面都涂上红色，再切成棱长1厘米的小正方体。两面涂色的小正方体有多少个？",
        f"正方体木块棱长{n}厘米，表面涂漆后切成棱长1厘米的小正方体。两面涂漆的小正方体有多少个？",
    ])
    lines = [
        f"每条棱上两面涂色 = {n} - 2 = {n - 2}个",
        f"两面涂色总数 = 12 × {n - 2} = {two}个",
    ]
    return ins, lines, two


_reg("cube_paint_faces", cube_paint_faces)


if __name__ == "__main__":
    from run_math_short import verify

    rng = random.Random(1234)
    fails = 0
    bare = 0
    small = []
    for level, name, fn in PROGRAMS:
        seen = set()
        for i in range(40):
            ins, lines, ans = fn(rng)
            seen.add(ins)
            for ln in lines:
                if "=" not in ln:
                    bare += 1
            out, ok = verify(ins, lines, ans)
            if not ok:
                fails += 1
                print(f"FAIL {name} #{i}: {ins}")
                for ln in lines:
                    print(f"   {ln}")
                print(f"   ans={ans}")
                break
        if len(seen) < 20:
            small.append((name, len(seen)))
    print(f"programs={len(PROGRAMS)} fails={fails} bare={bare}")
    for name, u in small:
        print(f"  small-space: {name} ({u} unique)")
