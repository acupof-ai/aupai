#!/usr/bin/env python3
"""L4 ext7: novel structures — clock hands (overlap/right-angle/straight/
angle/fast-slow/mirror/overlap-count), weekday cycles, leap years, snail
wells, doubling lilies, torn book pages, page digit totals, toggled lamps,
Hanoi towers, fake-coin weighings, take-away winning moves, ferry crossings,
empty-bottle exchange, dog shuttle runs, second-meeting points, nth-meeting
positions, up/down escalators, reverse bus headways, divisibility blanks,
three-prime even sums, square-difference numbers, gcd cutting/grouping,
divisor counts, three-modulus CRT, telescoping fractions, unit-fraction
splits, odd sums, square/cube sum formulas, magic squares, folded-box
volume, lengthwise-cut cylinders, cone water fractions, fence max area,
unfolded-box volume, stock two-day swings, reverse tax, freight optimization,
store-discount comparison, max-profit pricing, overtime wages, folded-paper
thickness, chessboard rice, minimum coins, grid paths, lever balance, dice
probability, subset weights, cuboid growth, rectangle perimeter-area.

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
    PROGRAMS.append(("L4", name, fn))


def _factor(n):
    """Prime factorization as '2 × 2 × 3' string."""
    parts = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            parts.append(str(d))
            n //= d
        d += 1
    if n > 1:
        parts.append(str(n))
    return " × ".join(parts)


def _common_factors(a, b):
    """Product string of gcd's prime factors, e.g. '2 × 2 × 5' for gcd 20."""
    return _factor(math.gcd(a, b))


# 1. 几点后时针分针首次重合
def clock_overlap_time(rng):
    h = rng.randint(2, 11)
    ans = Fraction(60 * h, 11)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{h}点整以后，时针与分针第一次重合是在{h}点多少分？",
        f"{name}看钟时发现{h}点整两针不重合，那么{h}点过后时针与分针第一次重合要经过多少分钟？",
        f"从{h}点整开始，再过多少分钟时针与分针第一次重合？请列式算一算。",
        f"{h}点整时，时针指向{h}、分针指向12。两针第一次重合是在{h}点多少分？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"时针每分钟走 = 1 ÷ 12 = 1/12格",
        f"分针每分钟比时针多走 = 1 - 1/12 = 11/12格",
        f"重合需要追赶的格数 = {h} × 5 = {5 * h}格",
        f"第一次重合所需时间 = {5 * h} ÷ (11 ÷ 12) = {num(ans)}分",
    ]
    return ins, lines, ans


_reg("clock_overlap_time", clock_overlap_time)


# 2. 几点后两针第一次成直角
def clock_right_angle_time(rng):
    h = rng.choice([4, 5, 6, 7, 8, 10])
    ans = Fraction(60 * h - 180, 11)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{h}点整以后，时针与分针第一次成90°角是在{h}点多少分？",
        f"从{h}点整开始，再过多少分钟两针第一次成90°角？{name}想知道，请你算一算。",
        f"{h}点整时两针的夹角大于90°，那么{h}点过后两针第一次成90°角要经过多少分钟？",
        f"{name}观察钟面：{h}点整以后，时针与分针第一次成90°角是在多少分钟后？请列式计算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{h}点整的夹角 = 30 × {h} = {30 * h}度",
        f"分针每分钟比时针多走 = 6 - 0.5 = 5.5度",
        f"第一次成90°角所需时间 = ({30 * h} - 90) ÷ 5.5 = {num(ans)}分",
    ]
    return ins, lines, ans


_reg("clock_right_angle_time", clock_right_angle_time)


# 3. 几点后两针第一次成平角
def clock_straight_time(rng):
    h = rng.choice([7, 8, 9, 10, 11])
    ans = Fraction(60 * h - 360, 11)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{h}点整以后，时针与分针第一次成180°平角是在{h}点多少分？",
        f"从{h}点整开始，再过多少分钟两针第一次成一条直线（180°）？",
        f"{h}点整时两针夹角还不到180°，{name}问：{h}点过后两针第一次成180°角要多少分钟？",
        f"{h}点整以后，时针与分针第一次方向相反（成180°角）是在多少分钟后？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{h}点整的夹角 = 30 × {h} = {30 * h}度",
        f"分针每分钟比时针多走 = 6 - 0.5 = 5.5度",
        f"第一次成180°角所需时间 = ({30 * h} - 180) ÷ 5.5 = {num(ans)}分",
    ]
    return ins, lines, ans


_reg("clock_straight_time", clock_straight_time)


# 4. 某点某分两针的夹角
def clock_angle_at_time(rng):
    while True:
        h = rng.randint(1, 12)
        m = rng.choice([10, 20, 30, 40, 50])
        a = abs(30 * h + Fraction(m, 2) - 6 * m)
        if a > 180:
            a = 360 - a
        if 0 < a <= 180:
            break
    ans = a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{h}点{m}分时，时针与分针的较小夹角是多少度？",
        f"{name}问：钟面上{h}点{m}分这一时刻，时针与分针的夹角（取较小的）是多少度？",
        f"求{h}点{m}分时，钟面上时针与分针所成的较小角的度数。",
        f"{h}点{m}分，时针与分针的夹角是多少度？（取较小的那个角）",
    ]) + rng.choice(_TAILS)
    raw = abs(30 * h + Fraction(m, 2) - 6 * m)
    pos_h = 30 * h + Fraction(m, 2)
    lines = [
        f"时针的位置 = 30 × {h} + 0.5 × {m} = {num(pos_h)}度",
        f"分针的位置 = 6 × {m} = {6 * m}度",
    ]
    if pos_h >= 6 * m:
        lines.append(f"两针的夹角 = {num(pos_h)} - {6 * m} = {num(raw)}度")
    else:
        lines.append(f"两针的夹角 = {6 * m} - {num(pos_h)} = {num(raw)}度")
    if raw > 180:
        lines.append(f"较小夹角 = 360 - {num(raw)} = {num(ans)}度")
    return ins, lines, ans


_reg("clock_angle_at_time", clock_angle_at_time)


# 5. 慢钟显示时刻求实际时间
def clock_fast_slow(rng):
    k = rng.choice([10, 12, 15, 20])
    t = 240 * 60 // (60 - k)
    hh = 8 + t // 60
    mm = t % 60
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一只钟每小时慢{k}分钟，早上8点整对准了标准时间。当这只钟第一次显示中午12点整时，实际时间是几点几分？",
        f"{name}家的钟每小时慢{k}分钟，早上8点对准。当天这只钟走到12点整时，实际已经是几点几分了？",
        f"一座钟每小时比标准时间慢{k}分钟，早上8点整校准。当钟面显示12点整时，实际时间是几点几分？请列式算一算。",
        f"有一只慢钟，每小时慢{k}分钟，早上8点整与标准时间对准。当它显示12点整时，实际时间是多少？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"慢钟从8点走到12点共走 = 4 × 60 = 240分",
        f"慢钟速度是标准钟的 = (60 - {k}) ÷ 60 = {num(Fraction(60 - k, 60))}",
    ]
    if mm > 0:
        lines.append(f"实际时刻的分钟 = {t} - {t // 60} × 60 = {mm}分")
        lines.append(f"实际经过的整小时 = ({t} - {mm}) ÷ 60 = {t // 60}小时")
    else:
        lines.append(f"实际经过的整小时 = {t} ÷ 60 = {t // 60}小时")
    lines.append(f"实际时刻 = 8 + {t // 60} = {hh}点")
    lines.append(f"实际经过的时间 = 240 ÷ ({num(Fraction(60 - k, 60))}) = {t}分")
    return ins, lines, t


_reg("clock_fast_slow", clock_fast_slow)


# 6. 镜中时间与实际时间之差
def clock_mirror_time(rng):
    h = rng.randint(2, 5)
    m = rng.choice([15, 20, 30, 40, 45, 50])
    T = 60 * h + m
    A = 720 - T
    D = A - T
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}从镜子里看到钟面是{h}点{m}分，实际时间与镜子里看到的时间相差多少分钟？",
        f"小明对着镜子看钟，镜中显示{h}点{m}分。实际钟面时间和镜中时间相差多少分钟？",
        f"镜子里的钟面是{h}点{m}分，实际时间与镜中时间相差多少分钟？请列式算一算。",
        f"{name}在镜中看到时钟显示{h}点{m}分。实际时间与镜中时间相差多少分钟？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"镜中时刻 = 60 × {h} + {m} = {T}分",
        f"实际时刻 = 720 - {T} = {A}分",
        f"两个时刻相差 = {A} - {T} = {D}分",
    ]
    return ins, lines, D


_reg("clock_mirror_time", clock_mirror_time)


# 7. 一昼夜两针重合/成直角次数
def clock_overlaps_per_day(rng):
    kind = rng.choice(["重合", "成直角"])
    if kind == "重合":
        ans = 22
        ins = rng.choice([
            "从0点到24点的一昼夜里，时针与分针一共重合多少次？",
            "一昼夜（24小时）中，钟面上时针与分针共有多少次重合？请列式算一算。",
            "请你算一算：24小时内时针与分针重合的总次数是多少？",
            "一昼夜之间，时针和分针会重合多少次？写出你的计算过程。",
        ])
        lines = [
            "两次重合的间隔 = 60 ÷ (1 - 1/12) = 720/11分",
            "12小时 = 720 = 720分",
            "12小时内重合次数 = 720 ÷ (720 ÷ 11) = 11次",
            "24 ÷ 12 = 2个12小时",
            "一昼夜重合次数 = 2 × 11 = 22次",
        ]
    else:
        ans = 44
        ins = rng.choice([
            "从0点到24点的一昼夜里，时针与分针一共多少次成直角？",
            "一昼夜（24小时）中，钟面上时针与分针共有多少次成直角？请列式算一算。",
            "请你算一算：24小时内时针与分针成直角的总次数是多少？",
            "一昼夜之间，时针和分针会有多少次成直角？写出你的计算过程。",
        ])
        lines = [
            "相邻两次成直角的间隔 = 180 ÷ 5.5 = 360/11分",
            "12小时 = 720 = 720分",
            "12小时内成直角次数 = 720 ÷ (360 ÷ 11) = 22次",
            "24 ÷ 12 = 2个12小时",
            "一昼夜成直角次数 = 2 × 22 = 44次",
        ]
    return ins + rng.choice(_TAILS), lines, ans


_reg("clock_overlaps_per_day", clock_overlaps_per_day)


# 8. 再过 n 天是星期几
def weekday_cycle(rng):
    today = rng.randint(1, 7)
    n = rng.randint(20, 90)
    q, r = divmod(n, 7)
    s = today + r
    if s > 7:
        s -= 7
    ans = s
    week = ["", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"今天是{week[today]}，再过{n}天是星期几？（星期一用1表示，星期日用7表示）",
        f"{name}知道今天是{week[today]}，请你帮他算出再过{n}天是星期几，用数字1到7表示。",
        f"今天{week[today]}，{n}天后是星期几？答案用数字表示：星期一为1，星期日为7。",
        f"如果今天是{week[today]}，那么再过{n}天是星期几？（用1到7的数字回答）",
    ]) + rng.choice(_TAILS)
    lines = [
        f"7 × {q} = {7 * q}天",
        f"{n} - {7 * q} = {r}天",
    ]
    if today + r > 7:
        lines.append(f"{today} + {r} - 7 = {ans}")
    else:
        lines.append(f"{today} + {r} = {ans}")
    return ins, lines, ans


_reg("weekday_cycle", weekday_cycle)


# 9. 区间内闰年个数
def leap_year_count(rng):
    y1 = rng.choice([1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008])
    y2 = y1 + rng.choice([20, 24, 28, 32])
    if y2 > 2024:
        y1, y2 = 1980, 2004
    a, b = y1 // 4, y2 // 4
    ans = b - a + 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"从{y1}年到{y2}年（包括这两年），一共有多少个闰年？",
        f"{name}查日历：从{y1}年年初到{y2}年年底，共有多少个闰年？请列式算一算。",
        f"在{y1}年到{y2}年之间（含头尾两年），闰年一共有多少个？",
        f"请你算一算：{y1}年至{y2}年（包括这两年）共有多少个闰年？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{y1} ÷ 4 = {a}个",
        f"{y2} ÷ 4 = {b}个",
        f"{b} - {a} + 1 = {ans}个",
    ]
    return ins, lines, ans


_reg("leap_year_count", leap_year_count)


# 10. 蜗牛爬井
def snail_well(rng):
    a = rng.randint(3, 6)
    b = rng.randint(1, a - 1)
    d = a - b
    k = rng.randint(3, 9)
    h = a + k * d
    ans = k + 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一口井深{h}米，一只蜗牛从井底往上爬，白天爬{a}米，夜里滑下{b}米。它多少天能爬出井口？",
        f"蜗牛爬井：井深{h}米，白天向上爬{a}米，晚上滑下{b}米。{name}问：蜗牛第几天能爬出井？",
        f"一只蜗牛掉进{h}米深的井里，白天爬{a}米，夜里滑{b}米。照这样，它几天能爬出来？请列式算一算。",
        f"井深{h}米，蜗牛白天向上爬{a}米、晚上滑下{b}米。它在第几天白天能爬出井口？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"每天实际上升 = {a} - {b} = {d}米",
        f"最后一天前要爬到 = {h} - {a} = {h - a}米",
        f"前面需要的天数 = {h - a} ÷ {d} = {k}天",
        f"总天数 = {k} + 1 = {ans}天",
    ]
    return ins, lines, ans


_reg("snail_well", snail_well)


# 11. 睡莲翻倍，第几天铺了几分之几
def water_lily_double(rng):
    n = rng.randint(10, 16)
    d = rng.randint(2, 3)
    k = n - d
    ans = Fraction(1, 2 ** d)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"池塘里的睡莲每天长大一倍，第{n}天铺满整个池塘。第{k}天铺了池塘的几分之几？",
        f"睡莲的面积每天翻一倍，第{n}天刚好长满池塘。{name}问：第{k}天铺了池塘的几分之几？",
        f"一种水生植物每天面积扩大一倍，第{n}天铺满水面。第{k}天它铺了水面的几分之几？请列式算一算。",
        f"池塘睡莲每天长大一倍，{n}天铺满。第{k}天铺了整个池塘的几分之几？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"相差的天数 = {n} - {k} = {d}天",
    ]
    cur = Fraction(1)
    for i in range(d):
        nxt = cur / 2
        lines.append(f"往前倒推一天 = {num(cur)} ÷ 2 = {num(nxt)}")
        cur = nxt
    return ins, lines, ans


_reg("water_lily_double", water_lily_double)


# 12. 撕掉一页后剩余页码和
def book_missing_page(rng):
    n = rng.randint(20, 60)
    x = rng.randint(5, n - 5)
    total = n * (n + 1) // 2
    torn = 2 * x + 1
    S = total - torn
    ans = x
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一本书的页码从1到{n}，撕掉了一张（含两个连续页码），剩余页码之和是{S}。撕掉的较小页码是多少？",
        f"一本{n}页的书，页码为1、2、…、{n}。{name}撕掉一张后，剩下的页码和是{S}。撕掉的那张上较小的页码是几？",
        f"一本书共{n}页，页码从1编到{n}。撕掉一张纸（两个连续页码）后，余下页码之和为{S}。较小的被撕页码是多少？请列式算一算。",
        f"页码从1到{n}的书，被人撕掉一张，剩下的页码之和是{S}。撕掉的两个页码中较小的是多少？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"全部页码之和 = {n} × ({n} + 1) ÷ 2 = {total}",
        f"被撕掉的页码之和 = {total} - {S} = {torn}",
        f"较小的被撕页码 = ({torn} - 1) ÷ 2 = {ans}",
    ]
    return ins, lines, ans


_reg("book_missing_page", book_missing_page)


# 13. 编页码共用多少个数字
def page_digits_total(rng):
    n = rng.randint(100, 999)
    three = n - 99
    ans = 9 + 180 + three * 3
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"给一本书编页码，从1编到{n}，一共要用多少个数字？",
        f"一本{n}页的书，页码从1到{n}。{name}问：编页码一共用了多少个数字？",
        f"出版社给一本{n}页的书编页码（1、2、…、{n}），总共需要多少个数字？请列式算一算。",
        f"一本书共{n}页，从第1页到第{n}页编页码，一共用了多少个数字？",
    ]) + rng.choice(_TAILS)
    lines = [
        "一位数页码用数字 = 9 × 1 = 9个",
        "两位数页码用数字 = 90 × 2 = 180个",
        f"三位数页码的页数 = {n} - 99 = {three}页",
        f"三位数页码用数字 = {three} × 3 = {three * 3}个",
        f"数字总数 = 9 + 180 + {three * 3} = {ans}个",
    ]
    return ins, lines, ans


_reg("page_digits_total", page_digits_total)


# 14. 拉灯问题：完全平方数编号的灯亮着
def lights_toggle(rng):
    n = rng.choice([49, 64, 81, 100, 144])
    k = int(math.isqrt(n))
    ans = k
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{n}盏灯编号1到{n}，开始全灭。第1轮按所有编号的开关，第2轮按2的倍数，第3轮按3的倍数……第{n}轮按{n}的倍数。最后亮着几盏灯？",
        f"走廊有{n}盏灯，编号1～{n}，全部关闭。{name}第1轮按下所有开关，第2轮按2的倍数，第3轮按3的倍数，依此类推。最后有几盏灯亮着？",
        f"{n}盏灯排成一排，全是灭的。第k轮把编号为k的倍数的灯的开关都按一次，共进行{n}轮。最后亮着的灯有多少盏？请列式算一算。",
        f"有{n}盏灯，编号1到{n}，开始都不亮。一个人第1轮按全部开关，以后第m轮按m的倍数的开关。做完{n}轮后，亮着几盏灯？",
    ]) + rng.choice(_TAILS)
    lines = []
    for i in range(1, k + 1):
        lines.append(f"{i} × {i} = {i * i}号灯亮着")
    ones = " + ".join(["1"] * k)
    lines.append(f"亮着的灯总数 = {ones} = {ans}盏")
    return ins, lines, ans


_reg("lights_toggle", lights_toggle)


# 15. 汉诺塔移动次数
def tower_hanoi(rng):
    n = rng.randint(4, 7)
    ans = 2 ** n - 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"汉诺塔有{n}个圆盘，每次只能移动一个，大圆盘不能放在小圆盘上面。至少移动多少次才能把全部圆盘移到另一根柱子上？",
        f"{name}玩汉诺塔：{n}个圆盘套在一根柱子上，规则是每次移一个、大盘不能压小盘。全部移到另一根柱子至少要多少次？",
        f"经典的汉诺塔游戏：{n}个圆盘，三根柱子，每次移动一个且大圆盘不能放在小圆盘上。把{n}个圆盘整体移到另一根柱子，最少移动多少次？请列式算一算。",
        f"汉诺塔的{n}个圆盘要从一根柱子移到另一根，每次只能移最上面的一个，大盘不能放在小盘上。至少需要移动多少次？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{n} × 0 + 1 = 1次（1个圆盘）",
    ]
    cur = 1
    for i in range(2, n + 1):
        nxt = cur * 2 + 1
        lines.append(f"{cur} × 2 + 1 = {nxt}次（{i}个圆盘）")
        cur = nxt
    return ins, lines, ans


_reg("tower_hanoi", tower_hanoi)


# 16. 天平找次品（已知偏轻）
def weighing_fake_known(rng):
    n = rng.choice([9, 27, 81])
    k = 1 if n == 3 else 2 if n == 9 else 3 if n == 27 else 4
    item = rng.choice(["零件", "珍珠", "硬币", "糖果", "纽扣", "乒乓球"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"有{n}个{item}，其中1个是次品，比正品轻一些。用天平称，至少称几次一定能找出次品？",
        f"{n}个{item}里有1个较轻的次品，{name}用天平来称，至少称几次就能保证找到它？",
        f"一批{n}个{item}中混入1个较轻的次品。用天平称，最少称几次一定能把次品找出来？请列式算一算。",
        f"{n}个外观相同的{item}，1个偏轻是次品。用天平至少称几次，保证能找出次品？",
    ]) + rng.choice(_TAILS)
    lines = []
    cur = n
    while cur > 1:
        lines.append(f"{cur} ÷ 3 = {cur // 3}个")
        cur //= 3
    ones = " + ".join(["1"] * k)
    lines.append(f"称量次数 = {ones} = {k}次")
    return ins, lines, k


_reg("weighing_fake_known", weighing_fake_known)


# 17. 天平找次品（不知轻重）
def weighing_unknown_3balls(rng):
    n = rng.choice([3, 4, 5])
    cases = 2 * n
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"有{n}个球，其中1个是次品，但不知道次品比正品轻还是重。用天平至少称几次，一定能找出次品？",
        f"{n}个球里有1个次品（轻重未知），{name}用天平称，至少称几次才能保证找出次品？",
        f"{n}个外观一样的球，1个是次品，可能偏重也可能偏轻。用天平至少称几次一定能找到它？请列式算一算。",
        f"盒子里有{n}个球，1个是次品且不知轻重。用天平至少称几次，保证能把次品找出来？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"次品的可能情况 = {n} × 2 = {cases}种",
        "称1次最多分辨 = 3 = 3种",
    ]
    if cases <= 3:
        lines.append("1 = 1次")
        ans = 1
    elif cases <= 9:
        lines.append("3 × 3 = 9种")
        lines.append("1 + 1 = 2次")
        ans = 2
    else:
        lines.append("3 × 3 = 9种")
        lines.append("9 × 3 = 27种")
        lines.append("1 + 1 + 1 = 3次")
        ans = 3
    return ins, lines, ans


_reg("weighing_unknown_3balls", weighing_unknown_3balls)


# 18. 抢数游戏的必胜第一步
def take_stones_win(rng):
    k = rng.randint(2, 5)
    m = rng.randint(3, 12)
    r = rng.randint(1, k)
    n = m * (k + 1) + r
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"两人轮流报数，每次只能报1到{k}个数，谁报到{n}谁赢。先手第一次应该报几个数，才能保证获胜？",
        f"抢数游戏：每次报1～{k}个数，报到{n}者胜。{name}先手，他第一次报几个数就必胜？",
        f"两人玩报数游戏，每人每次报1至{k}个连续的数，谁先报到{n}谁赢。先手第一次报几个数能保证赢？请列式算一算。",
        f"游戏规则：轮流报数，每次报1到{k}个数，报到{n}的人获胜。先手第一次报几个数才有必胜策略？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"每轮两人报数的和 = {k} + 1 = {k + 1}个",
        f"{k + 1} × {m} = {(k + 1) * m}个",
        f"先手第一次应报 = {n} - {(k + 1) * m} = {r}个",
    ]
    return ins, lines, r


_reg("take_stones_win", take_stones_win)


# 19. 过河问题（船要划回来）
def ferry_crossings(rng):
    a = rng.randint(4, 7)
    k = rng.randint(2, 8)
    n = a + k * (a - 1)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{n}人要过河，河边只有一条小船，每次最多载{a}人，且每次都需要1人把船划回来。至少要渡几次才能全部过河？",
        f"{name}带{n}人过河，船每次最多坐{a}人，每次过河后得有1人划船回来。最少要渡几次？",
        f"一条船最多载{a}人，{n}人过河，每次过河后需1人把船划回。至少渡几次？请列式算一算。",
        f"{n}个人过河，只有一条能载{a}人的小船，每次过河后要1人划船返回。全部过河至少要几次？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"除最后一批外每次净过河 = {a} - 1 = {a - 1}人",
        f"最后一批前还剩 = {n} - {a} = {n - a}人",
        f"前面需要的次数 = {n - a} ÷ {a - 1} = {k}次",
        f"总次数 = {k} + 1 = {k + 1}次",
    ]
    return ins, lines, k + 1


_reg("ferry_crossings", ferry_crossings)


# 20. 空瓶换水
def empty_bottle(rng):
    a = rng.randint(3, 5)
    k = rng.randint(3, 8)
    n = k * (a - 1) + 1
    T = n + k
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"商店规定：{a}个空汽水瓶可以换1瓶汽水。小明买了{n}瓶汽水，他最多能喝到多少瓶汽水？",
        f"{name}买了{n}瓶汽水，喝完后每{a}个空瓶能换1瓶汽水。他一共最多能喝多少瓶？",
        f"汽水厂促销：{a}个空瓶换1瓶汽水。买{n}瓶汽水，最多可以喝到多少瓶？请列式算一算。",
        f"班级买了{n}瓶汽水，规定{a}个空瓶换1瓶。最多能喝多少瓶汽水？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"每换1瓶实际消耗空瓶 = {a} - 1 = {a - 1}个",
        f"可换的瓶数 = ({n} - 1) ÷ {a - 1} = {k}瓶",
        f"最多能喝 = {n} + {k} = {T}瓶",
    ]
    return ins, lines, T


_reg("empty_bottle", empty_bottle)


# 21. 狗在两人之间往返跑
def dog_between_two(rng):
    v1 = rng.randint(40, 80)
    v2 = rng.randint(30, v1 - 10)
    v3 = rng.randint(v1 + 20, v1 + 60)
    S = (v1 + v2) * rng.randint(2, 5)
    t = S // (v1 + v2)
    ans = v3 * t
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人相距{S}米，同时相向而行，甲每分钟走{v1}米，乙每分钟走{v2}米。甲带的一只狗每分钟跑{v3}米，在两人之间往返跑，直到两人相遇。狗一共跑了多少米？",
        f"{name}和爸爸相距{S}米，相向而行，{name}每分走{v1}米，爸爸每分走{v2}米。小狗每分跑{v3}米，在他们之间来回跑。相遇时小狗跑了多少米？",
        f"两地相距{S}米，甲乙相向而行，速度分别是每分{v1}米和{v2}米。一只狗以每分{v3}米的速度在两人间往返。两人相遇时狗跑了多少米？请列式算一算。",
        f"甲、乙从相距{S}米处相向而行，甲速{v1}米/分、乙速{v2}米/分。狗速{v3}米/分，在两人间往返奔跑。到相遇时狗共跑多少米？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"两人的速度和 = {v1} + {v2} = {v1 + v2}米/分",
        f"相遇时间 = {S} ÷ {v1 + v2} = {t}分",
        f"狗跑的路程 = {v3} × {t} = {ans}米",
    ]
    return ins, lines, ans


_reg("dog_between_two", dog_between_two)


# 22. 两车往返第二次相遇点
def two_cars_second_meet(rng):
    k = rng.randint(2, 6)
    m = rng.randint(2, 6)
    S = 5 * k * m
    v1, v2 = 3 * k, 2 * k
    t = 3 * m
    ans = S // 5
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两车分别从相距{S}千米的A、B两地同时相向开出，甲车每小时{v1}千米，乙车每小时{v2}千米。两车到达对方出发地后立即返回，第二次相遇点距A地多少千米？",
        f"A、B两地相距{S}千米，甲车{v1}千米/时、乙车{v2}千米/时，同时相向出发，到达后立即返回。{name}问：第二次相遇点离A地多远？",
        f"两车从相距{S}千米的两地相向而行，速度分别为{v1}和{v2}千米/时，各自到达终点后马上返回。第二次相遇点距A地多少千米？请列式算一算。",
        f"甲、乙两车同时从相距{S}千米的两地相向开出，甲速{v1}千米/时，乙速{v2}千米/时，到端点后立即返回。它们第二次相遇的地点距A地多少千米？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"两车的速度和 = {v1} + {v2} = {v1 + v2}千米/时",
        f"第二次相遇时共走路程 = 3 × {S} = {3 * S}千米",
        f"第二次相遇时间 = {3 * S} ÷ {v1 + v2} = {t}小时",
        f"甲车走的路程 = {v1} × {t} = {v1 * t}千米",
        f"相遇点距A地 = 2 × {S} - {v1 * t} = {ans}千米",
    ]
    return ins, lines, ans


_reg("two_cars_second_meet", two_cars_second_meet)


# 23. 环形跑道第 n 次相遇点
def circular_nth_meet_position(rng):
    for _ in range(100):
        d = rng.choice([20, 30, 40, 50, 60])
        L = d * rng.randint(8, 20)
        v2 = rng.randint(2, 8) * 10
        v1 = v2 + d
        n = rng.randint(2, 5)
        if (n * L) % d == 0 and (v1 * n) % d != 0:
            break
    else:
        d, L, v2, v1, n = 40, 400, 60, 100, 3
    t = n * L // d
    total = v1 * t
    q, pos = divmod(total, L)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人在周长{L}米的环形跑道上同点同向出发，甲每分跑{v1}米，乙每分跑{v2}米。第{n}次相遇点距起点多少米？",
        f"环形跑道周长{L}米，甲速{v1}米/分、乙速{v2}米/分，两人同点同向起跑。{name}问：第{n}次相遇时离起点多少米？",
        f"甲、乙沿周长{L}米的环形跑道同点同向跑步，甲每分{v1}米，乙每分{v2}米。第{n}次相遇的地点距起点多少米？请列式算一算。",
        f"在{L}米环形跑道上，甲、乙同点同向出发，速度分别为{v1}和{v2}米/分。第{n}次相遇点距起点多少米？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"两人的速度差 = {v1} - {v2} = {d}米/分",
        f"第{n}次相遇的路程差 = {n} × {L} = {n * L}米",
        f"相遇时间 = {n * L} ÷ {d} = {t}分",
        f"甲跑的路程 = {v1} × {t} = {total}米",
        f"整圈数 = {q} = {q}圈",
        f"相遇点距起点 = {total} - {q} × {L} = {pos}米",
    ]
    return ins, lines, pos


_reg("circular_nth_meet_position", circular_nth_meet_position)


# 24. 扶梯一顺一逆求速度
def escalator_up_down(rng):
    t1, t2, base = rng.choice([(20, 60, 60), (30, 60, 120), (24, 48, 96), (20, 40, 80)])
    k = rng.randint(1, 3)
    N = base * k
    v = N * (t2 - t1) // (2 * t1 * t2)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"商场的自动扶梯匀速运行。{name}站在扶梯上向上走，{t1}秒到楼上；他以同样的速度在这部扶梯上向下走，{t2}秒到楼下。扶梯每秒移动多少级？",
        f"一部自动扶梯，小明顺行{t1}秒到楼上，逆行{t2}秒到楼下（上下走的速度相同）。扶梯每秒移动多少级？",
        f"自动扶梯向上运行，{name}在扶梯上向上走用{t1}秒，向下走用{t2}秒（人走的速度不变）。扶梯的速度是每秒多少级？请列式算一算。",
        f"同一部扶梯，人向上走{t1}秒到顶，向下走{t2}秒到底，人走的速度相同。扶梯每秒移动多少级？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"顺行的速度和 = {N} ÷ {t1} = {N // t1}级/秒",
        f"逆行的速度差 = {N} ÷ {t2} = {N // t2}级/秒",
        f"扶梯的速度 = ({N // t1} - {N // t2}) ÷ 2 = {v}级/秒",
    ]
    return ins, lines, v


_reg("escalator_up_down", escalator_up_down)


# 25. 由相遇间隔反求发车间隔
def bus_interval_reverse(rng):
    for _ in range(100):
        v = rng.choice([300, 400, 500, 600])
        u = rng.randint(100, v - 100)
        t = rng.randint(3, 10)
        if ((v + u) * t) % v == 0:
            break
    else:
        v, u, t = 500, 100, 5
    T = (v + u) * t // v
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"公交车以每分钟{v}米的速度运行，{name}骑车每分钟行{u}米，与公交车相向而行。他每隔{t}分钟遇到一辆公交车。公交车每隔多少分钟发一辆？",
        f"一条线路上，公交车速每分{v}米，{name}骑车每分{u}米迎面而行，每{t}分钟遇到一辆车。发车间隔是多少分钟？",
        f"小明与公交车相向而行，车速{v}米/分，人速{u}米/分，每隔{t}分钟遇到一辆。公交车每隔几分钟发一班？请列式算一算。",
        f"公交车每分行{v}米，骑车人每分行{u}米，相向而行，每隔{t}分钟相遇一辆。求公交车的发车间隔。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"人与车的速度和 = {v} + {u} = {v + u}米/分",
        f"相邻两辆车的间距 = {v + u} × {t} = {(v + u) * t}米",
        f"发车间隔 = {(v + u) * t} ÷ {v} = {T}分钟",
    ]
    return ins, lines, T


_reg("bus_interval_reverse", bus_interval_reverse)


# 26. 填数字使被 9 整除
def divisibility_fill_digit(rng):
    while True:
        digits = [rng.randint(1, 9) for _ in range(4)]
        s = sum(digits)
        if s % 9 != 0:
            break
    x = 9 - s % 9
    pos = rng.randint(0, 4)
    d = digits[:pos] + ["□"] + digits[pos:]
    places = ["万位", "千位", "百位", "十位", "个位"]
    spaced = " ".join(map(str, d))
    known_desc = "、".join(f"{places[i]}是{digits[i] if i < pos else digits[i - 1]}"
                          for i in range(5) if i != pos)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个五位数，{known_desc}，{places[pos]}是□。这个数能被9整除，□里应填几？",
        f"{name}写了一个五位数：{spaced}。要使它能被9整除，□代表的数字是几？",
        f"五位数（{spaced}）是9的倍数，□里填几？请列式算一算。",
        f"在□里填一个数字，使五位数{spaced}能被9整除。□应填几？",
    ]) + rng.choice(_TAILS)
    q = s // 9 + 1
    lines = [
        f"已知四位数字的和 = {' + '.join(map(str, digits))} = {s}",
        f"大于{s}的最小9的倍数 = 9 × {q} = {9 * q}",
        f"□里的数字 = {9 * q} - {s} = {x}",
    ]
    return ins, lines, x


_reg("divisibility_fill_digit", divisibility_fill_digit)


# 27. 三个质数和为偶数
def prime_three_sum_even(rng):
    p, q = rng.choice([(3, 5), (3, 7), (3, 11), (5, 7), (5, 11), (7, 11),
                       (3, 13), (5, 13), (7, 13), (11, 13)])
    S = 2 + p + q
    ans = 2 * p * q
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"三个不同质数的和是{S}，其中两个质数分别是{p}和{q}。第三个质数是多少？这三个质数的积是多少？",
        f"{name}找到三个不同的质数，和为{S}，已知其中两个是{p}和{q}。第三个质数是几？三个质数的积是多少？",
        f"三个互不相同的质数相加得{S}，其中两个是{p}和{q}。求第三个质数，并求三个质数的积。请列式算一算。",
        f"已知三个不同质数的和是{S}，两个是{p}、{q}。第三个质数是多少？它们的积是多少？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"两个已知质数的和 = {p} + {q} = {p + q}",
        f"第三个质数 = {S} - {p + q} = 2",
        f"三个质数的积 = 2 × {p} × {q} = {ans}",
    ]
    return ins, lines, ans


_reg("prime_three_sum_even", prime_three_sum_even)


# 28. 加两个数都是平方数
def square_both_sides(rng):
    A, B = rng.choice([(100, 168), (200, 292), (100, 224), (150, 306),
                       (300, 520), (120, 316), (160, 320)])
    D = B - A
    half = D // 2
    n = (half + 2) // 2
    x = n * n - B
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个数加上{A}后是某个整数的平方，加上{B}后是另一个整数的平方。这个数是多少？",
        f"{name}在想一个数：它加上{A}是一个平方数，加上{B}是另一个平方数。这个数是多少？",
        f"某数加{A}等于一个整数的平方，加{B}等于另一个整数的平方。求这个数。请列式算一算。",
        f"一个数，加{A}是平方数，加{B}也是平方数（两个平方数不同）。这个数是多少？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"两个平方数的差 = {B} - {A} = {D}",
        f"两个底数的和 = {D} ÷ 2 = {half}",
        f"较大的底数 = ({half} + 2) ÷ 2 = {n}",
        f"较大的平方数 = {n} × {n} = {n * n}",
        f"所求的数 = {n * n} - {B} = {x}",
    ]
    return ins, lines, x


_reg("square_both_sides", square_both_sides)


# 29. 长方形纸裁最大正方形
def gcd_cut_squares(rng):
    a, b = rng.choice([(24, 18), (36, 24), (48, 36), (60, 48), (72, 48),
                       (84, 60), (96, 72), (60, 40), (75, 45), (80, 60)])
    g = math.gcd(a, b)
    ans = (a // g) * (b // g)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一张长方形纸长{a}厘米、宽{b}厘米，把它裁成同样大小的正方形且没有剩余。正方形边长最大是多少厘米？一共能裁多少块？",
        f"长{a}厘米、宽{b}厘米的长方形纸，裁成边长相等的正方形，无剩余。{name}想知道：边长最大是多少？能裁多少块？",
        f"把长{a}厘米、宽{b}厘米的长方形纸裁成大小相同的正方形（不剩纸）。正方形边长最大是多少厘米？共裁多少块？请列式算一算。",
        f"一张长{a}厘米、宽{b}厘米的纸，裁成同样大的正方形且没有剩余。最大边长是多少？能裁多少块？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{a}分解质因数 = {_factor(a)} = {a}",
        f"{b}分解质因数 = {_factor(b)} = {b}",
        f"正方形最大边长 = {_common_factors(a, b)} = {g}厘米",
        f"长能裁 = {a} ÷ {g} = {a // g}个",
        f"宽能裁 = {b} ÷ {g} = {b // g}个",
        f"总块数 = {a // g} × {b // g} = {ans}块",
    ]
    return ins, lines, ans


_reg("gcd_cut_squares", gcd_cut_squares)


# 30. 男女生混合分组
def gcd_mixed_groups(rng):
    a, b = rng.choice([(24, 18), (36, 24), (40, 24), (48, 36), (60, 48),
                       (60, 40), (75, 45), (84, 60), (48, 30), (36, 30)])
    g = math.gcd(a, b)
    ans = a // g + b // g
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"班级有男生{a}人、女生{b}人，要混合分组，使每组男生人数相同、女生人数也相同。最多能分几组？每组有多少人？",
        f"男生{a}人、女生{b}人参加活动，混合编组，每组男女生人数分别相等。{name}问：最多分几组？每组几人？",
        f"把{a}名男生和{b}名女生混合分组，每组男生一样多、女生一样多。最多能分几组？每组多少人？请列式算一算。",
        f"课外小组有男生{a}人、女生{b}人，分成人数相同的组（每组男、女生数分别相等）。最多分几组？每组几人？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{a}分解质因数 = {_factor(a)} = {a}",
        f"{b}分解质因数 = {_factor(b)} = {b}",
        f"最多的组数 = {_common_factors(a, b)} = {g}组",
        f"每组男生 = {a} ÷ {g} = {a // g}人",
        f"每组女生 = {b} ÷ {g} = {b // g}人",
        f"每组人数 = {a // g} + {b // g} = {ans}人",
    ]
    return ins, lines, ans


_reg("gcd_mixed_groups", gcd_mixed_groups)


# 31. 约数个数
def count_divisors(rng):
    n = rng.choice([120, 180, 240, 360, 480, 720, 840, 100, 144, 90])
    exps = []
    m = n
    d = 2
    while d * d <= m:
        e = 0
        while m % d == 0:
            m //= d
            e += 1
        if e:
            exps.append(e)
        d += 1
    if m > 1:
        exps.append(1)
    ans = 1
    for e in exps:
        ans *= e + 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{n}一共有多少个约数（包括1和它本身）？",
        f"{name}想知道{n}的约数总个数。请你用分解质因数的方法算一算。",
        f"一个数是{n}，它有多少个约数？请列式算一算。",
        f"求{n}的约数个数（含1和{n}）。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{n}分解质因数 = {_factor(n)} = {n}",
        f"约数个数 = {' × '.join(f'({e} + 1)' for e in exps)} = {ans}个",
    ]
    return ins, lines, ans


_reg("count_divisors", count_divisors)


# 32. 三余数物不知数
def crt_three_mod(rng):
    while True:
        a = rng.randint(0, 2)
        b = rng.randint(0, 4)
        c = rng.randint(0, 6)
        if a + b + c > 0:
            break
    X = 70 * a + 21 * b + 15 * c
    x = X % 105
    if x == 0:
        x = 105
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个数除以3余{a}，除以5余{b}，除以7余{c}。这个数最小是多少？",
        f"{name}在做一道题：某数除以3余{a}，除以5余{b}，除以7余{c}，求最小的这样的数。请你算一算。",
        f"有一个数，除以3余{a}，除以5余{b}，除以7余{c}。满足条件的最小数是多少？请列式算一算。",
        f"古代“物不知数”：一个数除以3余{a}，除以5余{b}，除以7余{c}。这个数最小是几？",
    ]) + rng.choice(_TAILS)
    lines = [
        "5 × 7 × 2 = 70",
        "3 × 7 = 21",
        "3 × 5 = 15",
        f"70 × {a} + 21 × {b} + 15 × {c} = {X}",
        f"所求最小数 = {X} - {105 * (X // 105)} = {x}",
    ]
    return ins, lines, x


_reg("crt_three_mod", crt_three_mod)


# 33. 分数裂项求和
def fraction_telescoping(rng):
    a = rng.randint(2, 4)
    b = a + rng.randint(3, 6)
    ans = Fraction(1, a) - Fraction(1, b + 1)
    name = rng.choice(NAMES)
    terms = " + ".join(f"1/（{k}×{k + 1}）" for k in range(a, b + 1))
    ins = rng.choice([
        f"计算：{terms}。",
        f"{name}遇到一道分数计算题：{terms}，请你帮他算出结果。",
        f"用简便方法计算：{terms}。请列式算一算。",
        f"求分数和 {terms} 的值。",
    ]) + rng.choice(_TAILS)
    sum_lhs = " + ".join(f"1 ÷ ({k} × {k + 1})" for k in range(a, b + 1))
    lines = [
        f"裂项相加 = {sum_lhs} = {num(ans)}",
        f"首尾相消 = 1 ÷ {a} - 1 ÷ {b + 1} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("fraction_telescoping", fraction_telescoping)


# 34. 分数单位拆分
def fraction_unit_split(rng):
    a = rng.randint(4, 8)
    x = a + 1
    y = a * x
    ans = y
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"把分数单位1/{a}拆成两个不同的分数单位之和，其中较大的一个是1/{x}，另一个分数单位的分母是多少？",
        f"1/{a} = 1/{x} + 1/（？）。{name}问：括号里应填几？",
        f"在等式1/{a} = 1/{x} + 1/□中，□代表的整数是多少？请列式算一算。",
        f"把1/{a}拆成两个分数单位的和，已知一个是1/{x}，另一个的分母是几？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"验证 = 1 ÷ {x} + 1 ÷ {y} = 1/{a}",
        f"两个分母的关系 = {a} + 1 = {x}",
        f"另一个分母 = {a} × {x} = {y}",
    ]
    return ins, lines, ans


_reg("fraction_unit_split", fraction_unit_split)


# 35. 连续奇数和
def odd_sum_square(rng):
    L = rng.choice([19, 29, 39, 49, 59, 69])
    n = (L + 1) // 2
    ans = n * n
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"从1开始的连续奇数，最后一个是{L}。这些奇数的和是多少？",
        f"一串连续奇数从1开始，到{L}结束。{name}想知道它们的和，请你算一算。",
        f"求从1开始、到{L}结束的所有连续奇数之和。请列式算一算。",
        f"从1起的连续奇数排到{L}，它们的和是多少？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"奇数的个数 = ({L} + 1) ÷ 2 = {n}",
        f"奇数的和 = {n} × {n} = {ans}",
    ]
    return ins, lines, ans


_reg("odd_sum_square", odd_sum_square)


# 36. 平方和公式
def square_sum_formula(rng):
    n = rng.randint(5, 10)
    P = n * (n + 1) * (2 * n + 1)
    ans = P // 6
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"计算从1到{n}的平方和：1² + 2² + … + {n}²。",
        f"{name}想知道1² + 2² + … + {n}²等于多少，请你算一算。",
        f"求1到{n}各数的平方之和。请列式算一算。",
        f"计算：1² + 2² + … + {n}² = ？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{n} + 1 = {n + 1}",
        f"2 × {n} + 1 = {2 * n + 1}",
        f"三数相乘 = {n} × {n + 1} × {2 * n + 1} = {P}",
        f"平方和 = {P} ÷ 6 = {ans}",
    ]
    return ins, lines, ans


_reg("square_sum_formula", square_sum_formula)


# 37. 立方和公式
def cube_sum_formula(rng):
    n = rng.randint(5, 10)
    S = n * (n + 1) // 2
    ans = S * S
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"计算从1到{n}的立方和：1³ + 2³ + … + {n}³。",
        f"{name}想知道1³ + 2³ + … + {n}³等于多少，请你算一算。",
        f"求1到{n}各数的立方之和。请列式算一算。",
        f"计算：1³ + 2³ + … + {n}³ = ？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"底数之和 = {n} × ({n} + 1) = {n * (n + 1)}",
        f"底数之和的一半 = {n * (n + 1)} ÷ 2 = {S}",
        f"立方和 = {S} × {S} = {ans}",
    ]
    return ins, lines, ans


_reg("cube_sum_formula", cube_sum_formula)


# 38. 三阶幻方
def magic_square(rng):
    c = rng.randint(5, 9)
    while True:
        a = rng.randint(1, 12)
        b = rng.randint(1, 12)
        x = 3 * c - a - b
        if x > 0 and len({a, b, c, x}) == 4:
            break
    ans = x
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"三阶幻方（每行、每列、每条对角线上三个数的和都相等）的中心数是{c}，其中一行已填{a}和{b}两个数，这行第三个数是多少？",
        f"一个三阶幻方的中心数是{c}，{name}在某一行填了{a}和{b}，这行还缺的数是几？",
        f"九宫格幻方的中心数是{c}，每行三个数的和相等。一行中有{a}和{b}两个数，第三个数是多少？请列式算一算。",
        f"三阶幻方中心填{c}，幻和（每行的和）是多少？若一行已有{a}、{b}两数，第三个数是几？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"幻和 = 3 × {c} = {3 * c}",
        f"已有两数的和 = {a} + {b} = {a + b}",
        f"第三个数 = {3 * c} - {a + b} = {ans}",
    ]
    return ins, lines, ans


_reg("magic_square", magic_square)


# 39. 铁皮折盒容积
def box_from_sheet(rng):
    a = rng.choice([30, 40, 50])
    b = rng.choice([20, 25, 30])
    c = rng.choice([3, 4, 5])
    vol = (a - 2 * c) * (b - 2 * c) * c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一块长方形铁皮长{a}厘米、宽{b}厘米，四个角各剪去一个边长{c}厘米的正方形，折成一个无盖铁盒。铁盒的容积是多少立方厘米？",
        f"长{a}厘米、宽{b}厘米的铁皮，四角剪去边长{c}厘米的正方形后折成无盖盒子。{name}问：盒子容积是多少？",
        f"用长{a}厘米、宽{b}厘米的长方形铁皮，四个角剪去边长{c}厘米的正方形，做成无盖铁盒。它的容积是多少？请列式算一算。",
        f"一张长{a}厘米、宽{b}厘米的铁皮，四角各剪去边长{c}厘米的正方形，弯折后焊成无盖铁盒。铁盒容积是多少立方厘米？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"铁盒的长 = {a} - 2 × {c} = {a - 2 * c}厘米",
        f"铁盒的宽 = {b} - 2 × {c} = {b - 2 * c}厘米",
        f"铁盒的底面积 = {a - 2 * c} × {b - 2 * c} = {(a - 2 * c) * (b - 2 * c)}平方厘米",
        f"铁盒的容积 = {(a - 2 * c) * (b - 2 * c)} × {c} = {vol}立方厘米",
    ]
    return ins, lines, vol


_reg("box_from_sheet", box_from_sheet)


# 40. 圆柱纵切表面积增加
def cylinder_lengthwise_cut(rng):
    r = rng.choice([3, 4, 5, 6])
    h = rng.choice([10, 12, 15, 20])
    d = 2 * r
    ans = 2 * d * h
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个圆柱底面半径{r}厘米、高{h}厘米，沿底面直径纵切成两半，表面积增加了多少平方厘米？",
        f"圆柱底面半径{r}厘米，高{h}厘米。{name}把它沿底面直径切成两半，表面积增加多少平方厘米？",
        f"把底面半径{r}厘米、高{h}厘米的圆柱沿直径纵切，表面积增加多少？请列式算一算。",
        f"圆柱沿底面直径切成两个半圆柱，半径{r}厘米、高{h}厘米。表面积共增加多少平方厘米？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"底面直径 = 2 × {r} = {d}厘米",
        f"一个切面的面积 = {d} × {h} = {d * h}平方厘米",
        f"增加的表面积 = 2 × {d * h} = {ans}平方厘米",
    ]
    return ins, lines, ans


_reg("cylinder_lengthwise_cut", cylinder_lengthwise_cut)


# 41. 圆锥装水体积比
def cone_water_fraction(rng):
    a, b = rng.choice([(1, 2), (1, 3), (2, 3), (1, 4), (3, 4)])
    ans = Fraction(a ** 3, b ** 3)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"圆锥形容器尖朝下，水面高度正好是容器高度的{a}/{b}。水的体积是容器容积的几分之几？",
        f"一个圆锥形容器尖朝下装水，水深是高的{a}/{b}。{name}问：水的体积占容器容积的几分之几？",
        f"圆锥容器尖朝下放着，水面高度为容器高度的{a}/{b}。水的体积是整个容器容积的几分之几？请列式算一算。",
        f"尖朝下的圆锥容器里装水，水深占高的{a}/{b}。水的体积是容积的几分之几？",
    ]) + rng.choice(_TAILS)
    f1 = Fraction(a, b)
    lines = [
        f"水深是高的 = {a} ÷ {b} = {num(f1)}",
        f"底面积之比 = {num(f1)} × {num(f1)} = {num(f1 ** 2)}",
        f"体积之比 = {num(f1 ** 2)} × {num(f1)} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("cone_water_fraction", cone_water_fraction)


# 42. 靠墙围长方形最大面积
def fence_max_area(rng):
    L = rng.choice([20, 24, 28, 32, 36, 40])
    w = L // 4
    l = L - 2 * w
    area = w * l
    a1 = (w - 1) * (l + 2)
    a3 = (w + 1) * (l - 2)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"用{L}米长的篱笆一面靠墙围一个长方形菜地，怎样围面积最大？最大面积是多少平方米？",
        f"{name}家有{L}米篱笆，一面靠墙围长方形鸡圈。怎样围面积最大？最大是多少平方米？",
        f"一面靠墙，用{L}米篱笆围一个长方形，面积最大是多少平方米？请列式算一算。",
        f"用{L}米篱笆靠墙围长方形菜地（墙足够长），最大面积是多少平方米？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"宽少1米的面积 = ({w} - 1) × ({l} + 2) = {a1}平方米",
        f"宽多1米的面积 = ({w} + 1) × ({l} - 2) = {a3}平方米",
        f"最优的宽 = {L} ÷ 4 = {w}米",
        f"对应的长 = {L} - 2 × {w} = {l}米",
        f"最大面积 = {w} × {l} = {area}平方米",
    ]
    return ins, lines, area


_reg("fence_max_area", fence_max_area)


# 43. 展开图求长方体体积
def unfold_box_volume(rng):
    b = rng.randint(3, 6)
    d = rng.randint(1, 3)
    a = b + d
    H = 4 * b + 2 * d
    c = rng.choice([5, 8, 10, 12])
    V = 2 * b + c
    vol = a * b * c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"长方体的十字形展开图中，横向总长是{H}厘米，纵向总长是{V}厘米，底面的长比宽多{d}厘米。这个长方体的体积是多少立方厘米？",
        f"一个长方体纸盒的十字形展开图：横向总长{H}厘米，纵向总长{V}厘米，底面长比宽多{d}厘米。{name}问：纸盒体积是多少？",
        f"长方体展开后呈十字形，量得横向总长{H}厘米、纵向总长{V}厘米，且底面长比宽多{d}厘米。求长方体体积。请列式算一算。",
        f"十字形展开的长方体，横向总长{H}厘米，纵向总长{V}厘米，底面长比宽多{d}厘米。它的体积是多少立方厘米？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"四条宽加两条长差 = {H} - 2 × {d} = {H - 2 * d}",
        f"底面的宽 = ({H} - 2 × {d}) ÷ 4 = {b}厘米",
        f"底面的长 = {b} + {d} = {a}厘米",
        f"长方体的高 = {V} - 2 × {b} = {c}厘米",
        f"体积 = {a} × {b} × {c} = {vol}立方厘米",
    ]
    return ins, lines, vol


_reg("unfold_box_volume", unfold_box_volume)


# 44. 股票两天涨跌
def stock_two_days(rng):
    while True:
        a = rng.choice([10, 20, 25])
        b = rng.choice([10, 20])
        P = (100 + a) * (100 - b)
        if P != 10000:
            break
    ans = Fraction(P, 10000)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一只股票第一天涨了{a}%，第二天跌了{b}%。两天后的价格是原价的几分之几？",
        f"{name}买的股票第一天上涨{a}%，第二天下跌{b}%。现在的价格是原来的几分之几？",
        f"某股票先涨{a}%再跌{b}%，现价是原价的几分之几？请列式算一算。",
        f"一只股票连续两天：第一天涨{a}%，第二天跌{b}%。现价与原价比是几分之几？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"第一天后是原价的 = 100 + {a} = {100 + a}",
        f"第二天后是原价的 = 100 - {b} = {100 - b}",
        f"两天的总倍数 = {100 + a} × {100 - b} = {P}",
        f"现价是原价的 = {P} ÷ 10000 = {num(ans)}",
    ]
    return ins, lines, ans


_reg("stock_two_days", stock_two_days)


# 45. 由纳税额反求应纳税所得额
def tax_reverse(rng):
    T = rng.choice([190, 290, 390, 490, 590, 690, 790, 890, 990])
    x2 = (T - 90) * 10
    x = 3000 + x2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"个人所得税规定：应纳税所得额不超过3000元的部分按3%纳税，超过3000元至12000元的部分按10%纳税。某人纳税{T}元，他的应纳税所得额是多少元？",
        f"个税税率：不超过3000元部分3%，3000元至12000元部分10%。{name}的爸爸纳税{T}元，他的应纳税所得额是多少元？",
        f"按规定，应纳税所得额3000元以内按3%纳税，3000元到12000元的部分按10%纳税。某人缴了{T}元税，应纳税所得额是多少？请列式算一算。",
        f"计税办法：不超过3000元的部分缴3%，超过3000元至12000元的部分缴10%。一人纳税{T}元，他的应纳税所得额是多少元？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"第一档最多纳税 = 3000 × 3 ÷ 100 = 90元",
        f"第二档的范围 = 12000 - 3000 = 9000元",
        f"第二档缴纳的税 = {T} - 90 = {T - 90}元",
        f"第二档的金额 = ({T} - 90) × 10 = {x2}元",
        f"应纳税所得额 = 3000 + {x2} = {x}元",
    ]
    return ins, lines, x


_reg("tax_reverse", tax_reverse)


# 46. 租车运货最省运费
def freight_optimize(rng):
    n, a, p, b, q = rng.choice([
        (27, 5, 100, 3, 65), (24, 5, 100, 3, 65), (33, 5, 100, 3, 65),
        (27, 4, 90, 3, 65), (33, 6, 120, 4, 90),
    ])
    combos = []
    for x in range(0, (n + a - 1) // a + 1):
        rest = n - a * x
        y = 0 if rest <= 0 else (rest + b - 1) // b
        combos.append((x, y, p * x + q * y))
    best = min(combos, key=lambda c: c[2])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"要运{n}吨货物，大车每辆载{a}吨、运费{p}元，小车每辆载{b}吨、运费{q}元。怎样租车最省？最少运费多少元？",
        f"工地有{n}吨货物要运。大车每辆运{a}吨，运费{p}元；小车每辆运{b}吨，运费{q}元。{name}怎样安排车辆最省钱？最少多少元？",
        f"运{n}吨货，大车限载{a}吨（每辆{p}元），小车限载{b}吨（每辆{q}元）。怎样租车运费最省？最省是多少元？请列式算一算。",
        f"一批{n}吨的货物，可用大车（每辆{a}吨，{p}元）和小车（每辆{b}吨，{q}元）运。怎样安排最省运费？最少多少元？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"全用大车的辆数 = {n} ÷ {a} = {num(Fraction(n, a))}辆",
        f"全用小车的辆数 = {n} ÷ {b} = {num(Fraction(n, b))}辆",
    ]
    for x, y, cost in combos:
        if (x, y, cost) == best:
            continue
        lines.append(f"大车{x}辆小车{y}辆 = {p} × {x} + {q} × {y} = {cost}元")
    lines.append(f"最省方案 = {p} × {best[0]} + {q} × {best[1]} = {best[2]}元")
    return ins, lines, best[2]


_reg("freight_optimize", freight_optimize)


# 47. 买十送一与原价对比
def purchase_optimize(rng):
    n = rng.choice([22, 33, 44, 55])
    price = rng.choice([40, 50, 60])
    groups = n // 11
    pay = n - groups
    diff = price * groups
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"学校要买{n}个足球，甲店每个{price}元不优惠；乙店每个{price}元，买十送一。到哪家店买便宜？便宜多少元？",
        f"买{n}个同样的球，A店每个{price}元；B店每个{price}元但买10个送1个。{name}去哪家店划算？便宜多少元？",
        f"体育用品店促销：每个球{price}元，买十送一。要买{n}个球，比不促销便宜多少元？请列式算一算。",
        f"买{n}个篮球，甲店一律每个{price}元，乙店买10个送1个（单价也是{price}元）。乙店比甲店便宜多少元？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"乙店每组的个数 = 10 + 1 = 11个",
        f"赠送的组数 = {n} ÷ 11 = {groups}组",
        f"乙店实际付款个数 = {n} - {groups} = {pay}个",
        f"甲店总价 = {n} × {price} = {n * price}元",
        f"乙店总价 = {pay} × {price} = {pay * price}元",
        f"乙店便宜 = {n * price} - {pay * price} = {diff}元",
    ]
    return ins, lines, diff


_reg("purchase_optimize", purchase_optimize)


# 48. 涨价减销的最大利润
def profit_max_price(rng):
    c, p0, n0 = rng.choice([
        (40, 50, 300), (40, 50, 500), (40, 50, 800), (30, 40, 500), (50, 60, 600),
    ])
    t = (n0 // 10 - (p0 - c)) // 2
    price = p0 + t
    sold = n0 - 10 * t
    ans = (price - c) * sold
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一种商品成本{c}元，定价{p0}元时每天可卖{n0}件。每涨价1元，每天少卖10件。定价多少元时利润最大？最大利润是多少元？",
        f"商店卖一种货，成本每件{c}元，卖{p0}元一件时每天卖{n0}件；每涨1元少卖10件。{name}问：定价多少利润最大？最大利润多少？",
        f"商品成本{c}元，售价{p0}元时每天售{n0}件，售价每提高1元销量减少10件。求最大利润及此时的定价。请列式算一算。",
        f"一种商品按{p0}元卖，每天卖{n0}件，成本{c}元。每涨价1元每天少卖10件。定价多少时每天利润最大？是多少元？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"卖完的涨价空间 = {n0} ÷ 10 = {n0 // 10}元",
        f"每件原来的利润 = {p0} - {c} = {p0 - c}元",
        f"最优涨价 = ({n0 // 10} - {p0 - c}) ÷ 2 = {t}元",
        f"最优定价 = {p0} + {t} = {price}元",
        f"每件利润 = {price} - {c} = {price - c}元",
        f"每天销量 = {n0} - 10 × {t} = {sold}件",
        f"最大利润 = {price - c} × {sold} = {ans}元",
    ]
    return ins, lines, ans


_reg("profit_max_price", profit_max_price)


# 49. 加班工资
def wage_overtime(rng):
    a = rng.choice([20, 30, 40])
    n = rng.choice([45, 50, 55])
    m = rng.choice([5, 8, 10])
    rate = a * 3 // 2
    normal = (n - m) * a
    ot = m * rate
    total = normal + ot
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"工人每小时工资{a}元，加班按1.5倍计算。他一周工作{n}小时，其中加班{m}小时，这周工资多少元？",
        f"{name}的爸爸时薪{a}元，加班时薪是正常的1.5倍。上周他工作{n}小时，其中{m}小时加班，工资共多少元？",
        f"一份工作正常时薪{a}元，加班按1.5倍发。某人一周上班{n}小时（含加班{m}小时），应得工资多少元？请列式算一算。",
        f"时薪{a}元，加班1.5倍。一周工作{n}小时、加班{m}小时，工资一共多少元？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"正常工作时间 = {n} - {m} = {n - m}小时",
        f"加班时薪 = {a} × 1.5 = {rate}元",
        f"正常工资 = {n - m} × {a} = {normal}元",
        f"加班工资 = {m} × {rate} = {ot}元",
        f"一周工资 = {normal} + {ot} = {total}元",
    ]
    return ins, lines, total


_reg("wage_overtime", wage_overtime)


# 50. 折纸厚度倍数
def fold_paper_thickness(rng):
    n = rng.choice([5, 6, 7, 8, 10])
    ans = 2 ** n
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一张纸对折{n}次后，厚度是原来的多少倍？",
        f"{name}把一张纸对折{n}次，厚度变成原来的多少倍？",
        f"一张纸连续对折{n}次，层数是原来一张的多少倍？请列式算一算。",
        f"把纸对折{n}次后，它的厚度是原来的多少倍？",
    ]) + rng.choice(_TAILS)
    lines = []
    cur = 1
    for i in range(1, n + 1):
        nxt = cur * 2
        lines.append(f"第{i}次对折后 = {cur} × 2 = {nxt}层")
        cur = nxt
    return ins, lines, ans


_reg("fold_paper_thickness", fold_paper_thickness)


# 51. 棋盘放米
def rice_chessboard(rng):
    ans = 2 ** 64 - 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"棋盘64格，第1格放1粒米，第2格放2粒，以后每格都是前一格的2倍。放满64格共需多少粒米？",
        f"国王赏米：棋盘第1格1粒，第2格2粒，每格翻倍，共64格。{name}问：一共要多少粒米？",
        f"在64格棋盘上放米，第1格1粒，第2格2粒，以后每格都是前一格的2倍。64格共放多少粒？请列式算一算。",
        f"棋盘64个格子，第1格放1粒米，之后每格放的米都是前一格的2倍。放满共需多少粒米？",
    ]) + rng.choice(_TAILS)
    lines = [
        "2 × 2 = 4",
        "4 × 4 = 16",
        "16 × 16 = 256",
        "256 × 256 = 65536",
        "65536 × 65536 = 4294967296",
        "4294967296 × 4294967296 = 18446744073709551616",
        f"64格总米粒数 = 18446744073709551616 - 1 = {ans}粒",
    ]
    return ins, lines, ans


_reg("rice_chessboard", rice_chessboard)


# 52. 硬币凑钱最少枚数
def coin_min_count(rng):
    n = rng.choice([8, 11, 13, 14, 16, 17, 18, 19, 21, 23])
    sols = []
    for x in range(1, n // 5 + 1):
        rest = n - 5 * x
        if rest >= 3 and rest % 3 == 0:
            sols.append((x, rest // 3, x + rest // 3))
    best = min(sols, key=lambda s: s[2])
    worst = max(sols, key=lambda s: s[2])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"用3元和5元的硬币凑{n}元（两种硬币都要用），最少要几枚硬币？",
        f"存钱罐里只有3元和5元的硬币。{name}要凑出{n}元（两种都用），最少用几枚？",
        f"用3元、5元硬币各若干枚凑{n}元，两种都必须用。最少需要多少枚硬币？请列式算一算。",
        f"凑{n}元钱，只能用3元和5元的硬币（两种都要用）。最少要几枚？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"枚数较多的凑法 = 5 × {worst[0]} + 3 × {worst[1]} = {n}元",
        f"这种凑法的枚数 = {worst[0]} + {worst[1]} = {worst[2]}枚",
        f"枚数最少的凑法 = 5 × {best[0]} + 3 × {best[1]} = {n}元",
        f"最少的枚数 = {best[0]} + {best[1]} = {best[2]}枚",
    ]
    return ins, lines, best[2]


_reg("coin_min_count", coin_min_count)


# 53. 网格最短路径条数
def grid_path_count(rng):
    m = rng.choice([2, 3, 4])
    n = rng.randint(m, m + 3)
    P = 1
    for i in range(1, m + 1):
        P *= n + i
    f = math.factorial(m)
    ans = P // f
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"从家到学校，要向东走{m}段、向北走{n}段，只能向东或向北走。最短路线一共有多少条？",
        f"街道网格中，从A点到B点需向东{m}段、向北{n}段。{name}问：只向东或向北走，最短路线有多少条？",
        f"从甲地到乙地，向东走{m}段、向北走{n}段，方向只能是东或北。共有多少条最短路线？请列式算一算。",
        f"网格路上，从家到学校要向东{m}段、向北{n}段。只许向东、向北走，最短路线有多少条？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"一共要走的段数 = {m} + {n} = {m + n}段",
        f"选{m}段向东的排列 = {' × '.join(str(n + i) for i in range(m, 0, -1))} = {P}",
        f"向东段数的全排列 = {' × '.join(str(i) for i in range(m, 0, -1))} = {f}",
        f"最短路线条数 = {P} ÷ {f} = {ans}条",
    ]
    return ins, lines, ans


_reg("grid_path_count", grid_path_count)


# 54. 杠杆平衡
def lever_balance(rng):
    for _ in range(100):
        W1 = rng.choice([20, 30, 40, 50, 60])
        L1 = rng.choice([3, 4, 5, 6])
        L2 = rng.choice([2, 3, 4])
        if (W1 * L1) % L2 == 0:
            break
    else:
        W1, L1, L2 = 30, 4, 3
    W2 = W1 * L1 // L2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"杠杆左边{L1}格处挂{W1}克的物体，右边{L2}格处挂多少克的物体，杠杆才能平衡？",
        f"跷跷板左边{L1}格处挂{W1}克的物体，右边{L2}格处挂多少克才能平衡？{name}想知道，请你算一算。",
        f"杠杆支点左边{L1}格挂{W1}克，右边{L2}格挂多少克可以平衡？请列式算一算。",
        f"一根杠杆，左{L1}格处挂{W1}克，右{L2}格处挂多重的物体能平衡？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"杠杆的总格数 = {L1} + {L2} = {L1 + L2}格",
        f"左边的力矩 = {W1} × {L1} = {W1 * L1}克·格",
        f"右边应挂的质量 = {W1 * L1} ÷ {L2} = {W2}克",
    ]
    return ins, lines, W2


_reg("lever_balance", lever_balance)


# 55. 骰子点数和的概率
def dice_sum_probability(rng):
    s = rng.choice([6, 7, 8])
    count = s - 1 if s <= 7 else 13 - s
    ans = Fraction(count, 36)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"同时掷两个骰子，点数之和等于{s}的可能性是几分之几？",
        f"掷两个骰子，{name}想知道点数和为{s}的可能性（用分数表示）。请你算一算。",
        f"两个骰子同时掷出，点数之和是{s}的情况占所有情况的几分之几？请列式算一算。",
        f"同时掷两枚骰子，点数和等于{s}的可能性是几分之几？",
    ]) + rng.choice(_TAILS)
    lines = [
        "所有可能的结果 = 6 × 6 = 36种",
    ]
    for i in range(1, 7):
        j = s - i
        if 1 <= j <= 6:
            lines.append(f"点数组合 = {i} + {j} = {s}")
    lines.append(f"可能性 = {count} ÷ 36 = {num(ans)}")
    return ins, lines, ans


_reg("dice_sum_probability", dice_sum_probability)


# 56. 砝码称重种数
def balance_weights(rng):
    w = rng.choice([(1, 2, 4), (1, 2, 4, 8), (1, 2, 4, 8, 16)])
    ans = 2 ** len(w) - 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"有{'、'.join(map(str, w))}克的砝码各一个，一共能称出多少种不同的质量？",
        f"一架天平配有{'、'.join(map(str, w))}克的砝码各一个。{name}问：能称出多少种不同质量？",
        f"用{'、'.join(map(str, w))}克的砝码各一个（砝码只能放一边），可以称出多少种不同的质量？请列式算一算。",
        f"盒中有{'、'.join(map(str, w))}克砝码各一个，共能称出多少种不同质量？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"砝码的总质量 = {' + '.join(map(str, w))} = {sum(w)}克",
        f"每个砝码可选可不选 = 2{' × 2' * (len(w) - 1)} = {2 ** len(w)}种",
        f"去掉都不选的情况 = {2 ** len(w)} - 1 = {ans}种",
    ]
    return ins, lines, ans


_reg("balance_weights", balance_weights)


# 57. 长方体长宽增加后的体积增量
def cuboid_volume_increase(rng):
    a, b, c = rng.choice([
        (8, 5, 6), (9, 6, 7), (10, 7, 8), (7, 5, 4), (12, 8, 5), (6, 5, 10),
    ])
    ans = 3 * b * c + 4 * a * c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个长方体长{a}厘米、宽{b}厘米、高{c}厘米。如果长增加3厘米、宽增加4厘米（高不变），体积比原来增加多少立方厘米？",
        f"长方体长{a}、宽{b}、高{c}（厘米）。{name}把长增加3厘米、宽增加4厘米，体积增加多少立方厘米？",
        f"一个长方体，长{a}厘米、宽{b}厘米、高{c}厘米。长增加3厘米同时宽增加4厘米，体积增加多少？请列式算一算。",
        f"长方体的长{a}厘米、宽{b}厘米、高{c}厘米。若长增加3厘米、宽增加4厘米，体积增加多少立方厘米？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"长增加3厘米多出的体积 = 3 × {b} × {c} = {3 * b * c}立方厘米",
        f"宽增加4厘米多出的体积 = {a} × 4 × {c} = {4 * a * c}立方厘米",
        f"一共增加的体积 = {3 * b * c} + {4 * a * c} = {ans}立方厘米",
    ]
    return ins, lines, ans


_reg("cuboid_volume_increase", cuboid_volume_increase)


# 58. 周长与差求长方形面积
def rectangle_perimeter_area(rng):
    P = rng.choice([24, 28, 32, 36, 40, 44])
    d = rng.choice([2, 4, 6])
    half = P // 2
    b = (half - d) // 2
    a = b + d
    area = a * b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个长方形的周长是{P}厘米，长比宽多{d}厘米。它的面积是多少平方厘米？",
        f"长方形周长{P}厘米，长比宽多{d}厘米。{name}想知道它的面积，请你算一算。",
        f"一个长方形周长为{P}厘米，长与宽的差是{d}厘米。求它的面积。请列式算一算。",
        f"长方形的周长是{P}厘米，长比宽长{d}厘米。面积是多少平方厘米？",
    ]) + rng.choice(_TAILS)
    lines = [
        f"长与宽的和 = {P} ÷ 2 = {half}厘米",
        f"两条宽的长度 = {half} - {d} = {half - d}厘米",
        f"宽 = {half - d} ÷ 2 = {b}厘米",
        f"长 = {b} + {d} = {a}厘米",
        f"面积 = {a} × {b} = {area}平方厘米",
    ]
    return ins, lines, area


_reg("rectangle_perimeter_area", rectangle_perimeter_area)


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
    print(f"L4 ext7 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
