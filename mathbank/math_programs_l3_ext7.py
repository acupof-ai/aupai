#!/usr/bin/env python3
"""L3 extension bank 7: structurally novel elementary programs.

Each program: fn(rng) -> (instruction, lines, ans). Every line is an
equation `label = X op Y = Z[u]` (3-part) or `X op Y = Z[u]` (2-part,
pure-arithmetic LHS). Last line value must equal ans. Prose integers >=3
must appear among the equation tokens (enforced by run_math_short.verify).

Structures covered (all absent from l3 base + ext1..ext6):
clock gain/slow, clock-hand path & sweep area, date weekday & day-count,
page digit occurrence, repeating decimal <-> fraction, nth repeating digit,
temperature difference, exchange rate, knockout matches, dividend-divisor-
quotient-remainder relation, factor count, prime pair, dice/spinner/coin
probability, draw-without-replacement, pigeonhole birthday, take-away
winning strategy, balance substitution, defective weighing, magic square,
cross number array, symbol elimination, max product split, max area fence,
pancake scheduling, queue waiting, cheapest rental, ticket plan compare,
two-set inclusion-exclusion, post-meet reach time, midpoint-meet reverse,
double-meet point gap, uphill/downhill average, goat grazing, track stagger,
wire reshape, cylinder-cone volume diff, cylinder reassemble, circle-
inscribed square, pie/bar/line chart reading, fraction unit to prime,
matchstick pattern, segment/rectangle counting, elevator floors, unit
fraction split, number-table position, calendar box, cross frame, median,
decimal->fraction diff, gcd/lcm pair, fold angle, coin combinations,
cylinder carve to cone.
"""

import random
from math import gcd
from fractions import Fraction

from mathcommon import num

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L3", name, fn))


def _d(f):
    """Exact decimal rendering for terminating Fractions, else n/d fallback."""
    f = Fraction(f)
    if f.denominator == 1:
        return str(f.numerator)
    n, d = abs(f.numerator), f.denominator
    sign = "-" if f.numerator < 0 else ""
    for k in range(1, 7):
        if (10 ** k) % d == 0:
            v = n * (10 ** k // d)
            s = str(v).zfill(k + 1)
            return f"{sign}{s[:-k]}.{s[-k:]}"
    return num(f)


# ---------------------------------------------------------------------------
# Batch 1: clock, calendar, digits, decimals, misc number relations
# ---------------------------------------------------------------------------

def clock_gain_seconds(rng):
    k = rng.choice([2, 3, 4, 5, 6])
    ans = 4 * k
    ins = rng.choice([
        f"一块手表每小时快{k}秒，早上8时对准标准时间，当天中午12时，这块表快了多少秒？",
        f"一只钟每小时快{k}秒，上午8时对准，到中午12时，它快了多少秒？",
        f"小明的表每小时比标准时间快{k}秒，早上8时调准，中午12时这块表快了多少秒？",
        f"一座钟每小时快{k}秒，8时对准后，到当天12时，这座钟快了多少秒？",
        f"一块表每小时快{k}秒，早上8点对准，中午12点时，它比标准时间快多少秒？",
        f"小红的手表每小时快{k}秒，上午8时对准标准时间，中午12时快了多少秒？",
    ])
    lines = [
        f"每小时快 = {k} = {k}秒",
        f"经过时间 = 12 - 8 = 4小时",
        f"4 × {k} = {ans}秒",
    ]
    return ins, lines, ans


_reg("clock_gain_seconds", clock_gain_seconds)


def clock_slow_actual(rng):
    m = rng.choice([10, 12, 15, 20])
    actual = Fraction(240 * 60, 60 - m)
    ins = rng.choice([
        f"一只钟每小时慢{m}分，早上8时对准，当这只钟指向中午12时时，实际经过了多少分钟？",
        f"一座钟每小时慢{m}分钟，上午8时调准，钟面走到12时时，实际时间经过了多少分钟？",
        f"小明家的钟每小时慢{m}分，早上8时对准，当钟显示12时时，实际已经过了多少分钟？",
        f"一只慢钟每小时慢{m}分，8时对准标准时间，钟指向12时时，实际经过多少分钟？",
        f"钟每小时慢{m}分钟，早上8点对准，当钟面是12点时，实际过了多少分钟？",
        f"一座旧钟每小时慢{m}分，上午8时对准，它指向中午12时时，实际经过了多少分钟？",
        f"有一只钟每小时慢{m}分，早上8时调准，当这只钟走到12时时，实际经过了多少分钟？",
        f"小华的闹钟每小时慢{m}分钟，8时对准，闹钟走到12时时，实际经过了多少分钟？",
    ])
    lines = [
        f"钟面经过 = 12 - 8 = 4小时",
        f"4 × 60 = 240分",
        f"钟速 = 60 - {m} = {60 - m}分/时",
        f"实际经过 = 240 × 60 ÷ {60 - m} = {num(actual)}分",
    ]
    return ins, lines, actual


_reg("clock_slow_actual", clock_slow_actual)


def clock_hour_hand_path(rng):
    r = rng.randint(6, 15)
    C = 2 * Fraction(314, 100) * r
    ans = 4 * Fraction(314, 100) * r
    ins = rng.choice([
        f"一个时钟的时针长{r}厘米，时针尖端一昼夜走过多少厘米（π取3.14）？",
        f"钟表时针长{r}厘米，一昼夜时针的尖端走过的路程是多少厘米（π取3.14）？",
        f"一只大钟的时针长{r}厘米，这根时针的尖端一昼夜走多少厘米（π取3.14）？",
        f"时钟时针长{r}厘米，时针尖端一昼夜所走的路程是多少厘米（π取3.14）？",
    ])
    lines = [
        f"一昼夜转 = 2 × 1 = 2圈",
        f"2 × 3.14 × {r} = {_d(C)}厘米",
        f"{_d(C)} × 2 = {_d(ans)}厘米",
    ]
    return ins, lines, ans


_reg("clock_hour_hand_path", clock_hour_hand_path)


def clock_minute_sweep_area(rng):
    r = rng.randint(6, 15)
    h = rng.randint(1, 9)
    A = Fraction(314, 100) * r * r
    ans = 3 * A
    ins = rng.choice([
        f"一只挂钟的分针长{r}厘米，从上午{h}时到上午{h + 3}时，分针扫过的面积是多少平方厘米（π取3.14）？",
        f"钟表分针长{r}厘米，从{h}时到{h + 3}时，分针扫过的面积是多少平方厘米（π取3.14）？",
        f"一个时钟的分针长{r}厘米，上午{h}时到上午{h + 3}时，分针扫过的面积共多少平方厘米（π取3.14）？",
        f"挂钟分针长{r}厘米，从{h}点到{h + 3}点，分针扫过的面积是多少平方厘米（π取3.14）？",
    ])
    lines = [
        f"经过时间 = {h + 3} - {h} = 3小时",
        f"3.14 × {r} × {r} = {_d(A)}平方厘米",
        f"3 × {_d(A)} = {_d(ans)}平方厘米",
    ]
    return ins, lines, ans


_reg("clock_minute_sweep_area", clock_minute_sweep_area)


def date_weekday_calc(rng):
    d = rng.randint(1, 18)
    delta = rng.randint(4, 26)
    d2 = d + delta
    w = rng.randint(1, 7)
    names = ["一", "二", "三", "四", "五", "六", "日"]
    total = w - 1 + delta
    q, r = divmod(total, 7)
    ans = r + 1
    ins = rng.choice([
        f"某月{d}日是星期{names[w - 1]}，这个月{d2}日是星期几（周一记为1，周日记为7）？",
        f"已知某月{d}日是星期{names[w - 1]}，该月{d2}日是星期几（用1到7表示，周一为1）？",
        f"某月{d}日是星期{names[w - 1]}，同个月的{d2}日是星期几（周一记作1，周日记作7）？",
        f"日历上某月{d}日是星期{names[w - 1]}，这个月{d2}日是星期几（周一为1，周日为7）？",
    ])
    lines = [
        f"{d2} - {d} = {delta}天",
        f"{w} - 1 + {delta} = {total}",
        f"7 × {q} = {7 * q}",
        f"{total} - {7 * q} = {r}",
        f"星期序号 = {r} + 1 = {ans}",
    ]
    return ins, lines, ans


_reg("date_weekday_calc", date_weekday_calc)


def date_diff_days(rng):
    month_len = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    m1 = rng.randint(1, 10)
    m2 = rng.randint(m1 + 1, 12)
    d1 = rng.randint(1, month_len[m1 - 1])
    d2 = rng.randint(1, month_len[m2 - 1])
    p1 = sum(month_len[:m1 - 1])
    p2 = sum(month_len[:m2 - 1])
    y1 = p1 + d1
    y2 = p2 + d2
    ans = y2 - y1
    ins = rng.choice([
        f"同一年里，从{m1}月{d1}日到{m2}月{d2}日（不含出发当天）相隔多少天？",
        f"某年{m1}月{d1}日到同年{m2}月{d2}日，中间相隔多少天？",
        f"从{m1}月{d1}日到{m2}月{d2}日（同一年，不算{m1}月{d1}日当天）共有多少天？",
        f"同一年的{m1}月{d1}日至{m2}月{d2}日相隔多少天？",
    ])
    lines = [
        f"{m1}月{d1}日是第 = {p1} + {d1} = {y1}天",
        f"{m2}月{d2}日是第 = {p2} + {d2} = {y2}天",
        f"{y2} - {y1} = {ans}天",
    ]
    return ins, lines, ans


_reg("date_diff_days", date_diff_days)


def page_digit_one_count(rng):
    N = rng.randint(100, 999)
    u = N // 10 + (1 if N % 10 >= 1 else 0)
    t = (N // 100) * 10 + max(0, min(10, N % 100 - 10 + 1))
    h = min(N - 100 + 1, 100)
    ans = u + t + h
    ins = rng.choice([
        f"给一本书编页码，从第1页编到第{N}页，数字“1”一共出现了多少次？",
        f"一本书的页码从1排到{N}，这些页码中数字“1”出现了多少次？",
        f"编一本{N}页的书的页码，数字“1”在页码中一共出现多少次？",
        f"一本故事书共{N}页，页码中数字“1”出现了多少次？",
    ])
    lines = [
        f"总页数 = {N} = {N}页",
        f"个位上的1 = {u} = {u}次",
        f"十位上的1 = {t} = {t}次",
        f"百位上的1 = {h} = {h}次",
        f"{u} + {t} + {h} = {ans}次",
    ]
    return ins, lines, ans


_reg("page_digit_one_count", page_digit_one_count)


def repeating_decimal_frac(rng):
    ab = rng.randint(10, 98)
    ans = Fraction(ab, 99)
    ins = rng.choice([
        f"把纯循环小数0.{ab}……（循环节是{ab}）化成分数。",
        f"循环小数0.{ab}……化成分数是多少？",
        f"0.{ab}……是一个纯循环小数，把它化成分数。",
        f"把循环小数0.{ab}……（循环节为{ab}）化成最简分数。",
    ])
    lines = [
        f"循环节位数 = 2 = 2位",
        f"分母 = 99 = 99",
        f"{ab} ÷ 99 = {num(ans)}",
    ]
    return ins, lines, ans


_reg("repeating_decimal_frac", repeating_decimal_frac)


def frac_repeating_nth_digit(rng):
    n = rng.randint(3, 50)
    cycle = "142857"
    q, r = divmod(n - 1, 6)
    ans = int(cycle[r])
    ins = rng.choice([
        f"把分数1/7化成小数，小数点后第{n}位上的数字是几？",
        f"1/7化成小数是一个循环小数，小数点后第{n}位数字是几？",
        f"分数1/7化成小数后，小数点后面第{n}位上的数字是多少？",
        f"1/7 = 0.142857……，小数点后第{n}位上的数字是几？",
    ])
    lines = [
        f"999999 ÷ 7 = 142857",
        f"循环节 = 142857 = 142857",
        f"{n} - 1 = {n - 1}",
        f"6 × {q} = {6 * q}",
        f"{n - 1} - {6 * q} = {r}",
        f"第{n}位数字 = {ans} = {ans}",
    ]
    return ins, lines, ans


_reg("frac_repeating_nth_digit", frac_repeating_nth_digit)


def temperature_diff(rng):
    a = rng.randint(3, 12)
    b = rng.randint(3, 12)
    ans = a + b
    ins = rng.choice([
        f"某地一天最高气温是{a}℃，最低气温是零下{b}℃，这天的温差是多少℃？",
        f"某城市白天最高气温{a}℃，夜间最低气温零下{b}℃，昼夜温差是多少℃？",
        f"甲地气温{a}℃，乙地气温零下{b}℃，两地气温相差多少℃？",
        f"冰箱冷藏室温度是{a}℃，冷冻室温度是零下{b}℃，两室温度相差多少℃？",
    ])
    lines = [
        f"零上温度 = {a} = {a}℃",
        f"零下温度 = {b} = {b}℃",
        f"温差 = {a} + {b} = {ans}℃",
    ]
    return ins, lines, ans


_reg("temperature_diff", temperature_diff)


def exchange_rate(rng):
    r = rng.choice([6, 7, 8])
    y = rng.randint(100, 900)
    if rng.random() < 0.5:
        ans = r * y
        ins = rng.choice([
            f"1美元可以兑换{r}元人民币，{y}美元可以兑换多少元人民币？",
            f"银行汇率：1美元兑换{r}元人民币，{y}美元能兑换多少元人民币？",
            f"按1美元兑换{r}元人民币计算，{y}美元可兑换人民币多少元？",
            f"美元兑人民币汇率为1:{r}，{y}美元可以兑换多少元人民币？",
        ])
        lines = [
            f"1美元 = {r} = {r}元",
            f"美元总额 = {y} = {y}美元",
            f"{r} × {y} = {ans}元",
        ]
    else:
        ans = Fraction(y, r)
        ins = rng.choice([
            f"1美元可以兑换{r}元人民币，{y}元人民币可以兑换多少美元？",
            f"银行汇率：1美元兑换{r}元人民币，{y}元人民币能兑换多少美元？",
            f"按1美元兑换{r}元人民币计算，{y}元人民币可兑换多少美元？",
            f"美元兑人民币汇率为1:{r}，{y}元人民币可以兑换多少美元？",
        ])
        lines = [
            f"1美元 = {r} = {r}元",
            f"人民币总额 = {y} = {y}元",
            f"{y} ÷ {r} = {num(ans)}美元",
        ]
    return ins, lines, ans


_reg("exchange_rate", exchange_rate)


def knockout_games(rng):
    n = rng.randint(8, 32)
    ans = n - 1
    ins = rng.choice([
        f"{n}支球队参加淘汰赛，每场比赛淘汰一支球队，决出冠军一共要赛多少场？",
        f"足球淘汰赛有{n}支球队参加，每场淘汰1队，决出冠军共需比赛多少场？",
        f"{n}名选手进行淘汰赛，每场比赛淘汰1人，决出冠军要进行多少场比赛？",
        f"一次淘汰赛共有{n}支球队，每赛一场淘汰一支，决出冠军一共要赛几场？",
    ])
    lines = [
        f"冠军只有 = 1 = 1支",
        f"需要淘汰 = {n} - 1 = {ans}支",
        f"比赛场数 = {ans} = {ans}场",
    ]
    return ins, lines, ans


_reg("knockout_games", knockout_games)


def dividend_relation(rng):
    q = rng.randint(3, 9)
    r = rng.randint(1, q - 1)
    d = rng.randint(5, 20)
    div = q * d + r
    S = div + d + q + r
    ins = rng.choice([
        f"在一道有余数的除法里，被除数、除数、商、余数的和是{S}，已知商是{q}，余数是{r}，被除数是多少？",
        f"一道除法算式中，被除数、除数、商、余数相加得{S}，商是{q}，余数是{r}，被除数是几？",
        f"有余数除法中，四数（被除数、除数、商、余数）之和为{S}，商{q}余{r}，求被除数。",
        f"被除数、除数、商、余数的和是{S}，商是{q}，余数是{r}，被除数是多少？",
    ])
    lines = [
        f"({S} - {q} - 2 × {r}) ÷ ({q} + 1) = {d}",
        f"{q} × {d} + {r} = {div}",
    ]
    return ins, lines, div


_reg("dividend_relation", dividend_relation)


_FACTOR_TABLE = [
    (12, "2 × 2 × 3", [2, 1]),
    (18, "2 × 3 × 3", [1, 2]),
    (20, "2 × 2 × 5", [2, 1]),
    (24, "2 × 2 × 2 × 3", [3, 1]),
    (36, "2 × 2 × 3 × 3", [2, 2]),
    (40, "2 × 2 × 2 × 5", [3, 1]),
    (45, "3 × 3 × 5", [2, 1]),
    (48, "2 × 2 × 2 × 2 × 3", [4, 1]),
    (54, "2 × 3 × 3 × 3", [1, 3]),
    (56, "2 × 2 × 2 × 7", [3, 1]),
    (72, "2 × 2 × 2 × 3 × 3", [3, 2]),
    (100, "2 × 2 × 5 × 5", [2, 2]),
    (84, "2 × 2 × 3 × 7", [2, 1, 1]),
    (80, "2 × 2 × 2 × 2 × 5", [4, 1]),
    (28, "2 × 2 × 7", [2, 1]),
]


def factor_count(rng):
    N, fac, exps = rng.choice(_FACTOR_TABLE)
    plus = [e + 1 for e in exps]
    ans = 1
    for p in plus:
        ans *= p
    ins = rng.choice([
        f"自然数{N}分解质因数是{N}={fac}，它一共有多少个因数？",
        f"已知{N}={fac}，自然数{N}的因数共有多少个？",
        f"把{N}分解质因数得{N}={fac}，{N}有多少个因数？",
        f"{N}={fac}，按这个分解式，{N}一共有多少个因数？",
    ])
    lines = [f"分解式 = {fac} = {N}"]
    for e, p in zip(exps, plus):
        lines.append(f"{e} + 1 = {p}")
    lhs = " × ".join(str(p) for p in plus)
    lines.append(f"{lhs} = {ans}个")
    return ins, lines, ans


_reg("factor_count", factor_count)


_PRIME_PAIRS = [(2, 3), (2, 5), (2, 7), (3, 5), (3, 7), (2, 11),
                (5, 7), (3, 11), (2, 13), (5, 11), (7, 11), (3, 13),
                (2, 17), (5, 13)]


def prime_pair(rng):
    p, q = rng.choice(_PRIME_PAIRS)
    S = p + q
    P = p * q
    ans = max(p, q)
    ins = rng.choice([
        f"两个质数的和是{S}，积是{P}，这两个质数中较大的是几？",
        f"两个质数相加得{S}，相乘得{P}，较大的质数是多少？",
        f"甲、乙都是质数，它们的和是{S}，积是{P}，较大的那个质数是几？",
        f"两个质数的积是{P}，和是{S}，其中较大的质数是多少？",
    ])
    lines = [
        f"积的分解 = {p} × {q} = {P}",
        f"验证和 = {p} + {q} = {S}",
        f"较大质数 = {ans} = {ans}",
    ]
    return ins, lines, ans


_reg("prime_pair", prime_pair)


# ---------------------------------------------------------------------------
# Batch 2: probability, pigeonhole, strategy, substitution, optimization
# ---------------------------------------------------------------------------

_DICE_COND = [
    ("偶数", 3), ("奇数", 3), ("大于4", 2), ("小于3", 2),
    ("等于3", 1), ("是3的倍数", 2), ("不是1", 5), ("大于2", 4),
    ("小于5", 4), ("是2的倍数", 3), ("等于6", 1), ("不是6", 5),
]


def dice_even_prob(rng):
    cond, c = rng.choice(_DICE_COND)
    ans = Fraction(c, 6)
    ins = rng.choice([
        f"掷一枚均匀的骰子，掷出的点数{cond}的可能性是几分之几？",
        f"任意掷一枚骰子，朝上的点数{cond}的可能性是多少？",
        f"掷一个均匀的正方体骰子（六个面分别是1到6），掷出{cond}的可能性是几分之几？",
        f"一枚骰子六个面上分别写着1到6，掷出{cond}的可能性是多少？",
        f"掷骰子一次，朝上的数{cond}的可能性为几分之几？",
        f"随便掷一枚均匀骰子，点数{cond}的可能性是多少？",
    ])
    lines = [
        f"点数{cond}的情况 = {c} = {c}种",
        f"总情况 = 6 = 6种",
        f"可能性 = {c} ÷ 6 = {num(ans)}",
    ]
    return ins, lines, ans


_reg("dice_even_prob", dice_even_prob)


_DICE_SUM = [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6),
             (8, 5), (9, 4), (10, 3), (11, 2), (12, 1)]


def two_dice_sum_prob(rng):
    s, c = rng.choice(_DICE_SUM)
    ans = Fraction(c, 36)
    ins = rng.choice([
        f"同时掷两枚均匀的骰子，点数之和等于{s}的可能性是几分之几？",
        f"掷两枚骰子，朝上的两个点数相加等于{s}的可能性是多少？",
        f"同时掷两个骰子，点数和为{s}的可能性是几分之几？",
        f"两枚骰子一起掷出，点数之和是{s}的可能性是多少？",
    ])
    lines = [
        f"和为{s}的组合 = {c} = {c}种",
        f"总情况 = 6 × 6 = 36种",
        f"可能性 = {c} ÷ 36 = {num(ans)}",
    ]
    return ins, lines, ans


_reg("two_dice_sum_prob", two_dice_sum_prob)


def spinner_prob(rng):
    n = rng.randint(6, 12)
    m = rng.randint(1, n - 1)
    ans = Fraction(m, n)
    ins = rng.choice([
        f"一个转盘被平均分成{n}份，其中{m}份涂成红色，其余涂成蓝色。转动转盘，指针停在红色区域的可能性是几分之几？",
        f"转盘平均分成{n}个相等的扇形，{m}个是红色，转动转盘一次，指针指向红色的可能性是多少？",
        f"一个圆被平均分成{n}份，{m}份涂红色，其余涂蓝色，指针停在红色区域的可能性是几分之几？",
        f"游戏转盘被平均分成{n}份，红色占{m}份，转动一次指针落在红色区域的可能性是多少？",
    ])
    lines = [
        f"总份数 = {n} = {n}份",
        f"红色份数 = {m} = {m}份",
        f"可能性 = {m} ÷ {n} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("spinner_prob", spinner_prob)


def draw_two_red_prob(rng):
    a = rng.randint(3, 8)
    b = rng.randint(3, 8)
    n = a + b
    fav = a * (a - 1)
    tot = n * (n - 1)
    ans = Fraction(fav, tot)
    ins = rng.choice([
        f"袋子里有{a}个红球和{b}个白球，从中任意摸出2个球（不放回），摸出的2个球都是红球的可能性是几分之几？",
        f"盒中有{a}个红球、{b}个白球，一次摸出2个且不放回，两个都是红球的可能性是多少？",
        f"口袋里装{a}个红球和{b}个白球，任意摸两个（不放回），摸到两个红球的可能性是几分之几？",
        f"袋子中有{a}个红球、{b}个白球，从中摸出2个球（不再放回），全是红球的可能性是多少？",
    ])
    lines = [
        f"总球数 = {a} + {b} = {n}个",
        f"摸两球总情况 = {n} × {n - 1} = {tot}种",
        f"两红情况 = {a} × {a - 1} = {fav}种",
        f"可能性 = {fav} ÷ {tot} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("draw_two_red_prob", draw_two_red_prob)


def pigeonhole_birthday(rng):
    q = rng.randint(1, 3)
    r = rng.randint(1, 11)
    n = 12 * q + r
    ans = q + 1
    ins = rng.choice([
        f"某校六年级有{n}名学生，他们中至少有几人在同一个月过生日？",
        f"兴趣小组有{n}名同学，至少有几名同学的生日在同一个月？",
        f"一个班有{n}名学生，至少有几人是同一个月出生的？",
        f"学校合唱队有{n}名队员，其中至少有几人在同一个月过生日？",
    ])
    lines = [
        f"一年有 = 12 = 12个月",
        f"12 × {q} = {12 * q}人",
        f"{n} - {12 * q} = {r}人",
        f"至少人数 = {q} + 1 = {ans}人",
    ]
    return ins, lines, ans


_reg("pigeonhole_birthday", pigeonhole_birthday)


def take_stones_win(rng):
    n = rng.randint(15, 40)
    while n % 3 == 0:
        n = rng.randint(15, 40)
    q, ans = divmod(n, 3)
    ins = rng.choice([
        f"有{n}颗石子，两人轮流取，每次可取1颗或2颗，取到最后一颗石子的人获胜。先取的人第一次取几颗才能必胜？",
        f"桌上有{n}颗石子，两人轮流拿，每次拿1颗或2颗，谁拿到最后一颗谁赢。先拿者第一次拿几颗必胜？",
        f"一堆石子共{n}颗，甲乙轮流取，每次取1至2颗，取到最后一颗者胜。先取者第一次应取几颗？",
        f"{n}颗石子排成一堆，两人轮流取，每次只能取1颗或2颗，取走最后一颗的人获胜。先取的人第一次取几颗必胜？",
    ])
    lines = [
        f"每轮可控 = 1 + 2 = 3颗",
        f"3 × {q} = {3 * q}颗",
        f"{n} - {3 * q} = {ans}颗",
        f"第一次取 = {ans} = {ans}颗",
    ]
    return ins, lines, ans


_reg("take_stones_win", take_stones_win)


def balance_chain(rng):
    a = rng.randint(2, 5)
    b = rng.randint(2, 5)
    ans = a * b
    style = rng.randrange(3)
    if style == 0:
        ins = rng.choice([
            f"1个西瓜的质量等于{a}个菠萝，1个菠萝的质量等于{b}个苹果。1个西瓜的质量等于多少个苹果？",
            f"如果1个西瓜和{a}个菠萝一样重，1个菠萝和{b}个苹果一样重，那么1个西瓜和多少个苹果一样重？",
            f"天平上1个西瓜正好等于{a}个菠萝，1个菠萝正好等于{b}个苹果，1个西瓜等于多少个苹果？",
            f"已知1个西瓜的质量 = {a}个菠萝的质量，1个菠萝的质量 = {b}个苹果的质量，1个西瓜的质量 = 多少个苹果的质量？",
        ])
    elif style == 1:
        ins = rng.choice([
            f"1头牛可以换{a}只羊，1只羊可以换{b}只兔。1头牛可以换多少只兔？",
            f"集市上1头牛换{a}只羊，1只羊换{b}只兔，1头牛能换多少只兔？",
            f"如果1头牛的价钱等于{a}只羊，1只羊的价钱等于{b}只兔，1头牛等于多少只兔？",
            f"用1头牛能换{a}只羊，用1只羊能换{b}只兔，用1头牛能换多少只兔？",
        ])
    else:
        ins = rng.choice([
            f"1本字典的厚度等于{a}本故事书，1本故事书的厚度等于{b}本练习本。1本字典等于多少本练习本的厚度？",
            f"如果1本字典和{a}本故事书一样厚，1本故事书和{b}本练习本一样厚，1本字典和多少本练习本一样厚？",
            f"1本字典厚 = {a}本故事书，1本故事书厚 = {b}本练习本，1本字典厚 = 多少本练习本？",
            f"摞起来1本字典与{a}本故事书一样高，1本故事书与{b}本练习本一样高，1本字典与多少本练习本一样高？",
        ])
    lines = [
        f"1个中间量 = {b} = {b}个最小量",
        f"中间量个数 = {a} = {a}个",
        f"所求 = {a} × {b} = {ans}个",
    ]
    return ins, lines, ans


_reg("balance_chain", balance_chain)


def defective_min_weigh(rng):
    n = rng.randint(4, 27)
    ans = 2 if n <= 9 else 3
    ins = rng.choice([
        f"有{n}个外形相同的零件，其中有1个次品（比正品轻）。用天平称，至少称几次就能保证找出次品？",
        f"{n}个零件里有1个较轻的次品，用天平至少称几次保证能把它找出来？",
        f"一批零件共{n}个，其中1个是次品且较轻，不用砝码，用天平至少称几次能保证找出次品？",
        f"有{n}颗外观一样的珠子，其中1颗较轻是次品，用天平至少称几次就一定能找出次品？",
    ])
    if ans == 2:
        lines = [
            f"零件总数 = {n} = {n}个",
            f"称1次最多辨 = 3 = 3个",
            f"3 × 3 = 9个",
            f"至少次数 = 2 = 2次",
        ]
    else:
        lines = [
            f"零件总数 = {n} = {n}个",
            f"称2次最多辨 = 3 × 3 = 9个",
            f"9 × 3 = 27个",
            f"至少次数 = 3 = 3次",
        ]
    return ins, lines, ans


_reg("defective_min_weigh", defective_min_weigh)


def magic_square_center(rng):
    H = rng.randint(15, 45)
    ans = Fraction(H, 3)
    ins = rng.choice([
        f"在三阶幻方中，每行、每列、每条对角线上三个数的和都相等，这个和是{H}。中间的数是多少？",
        f"一个三阶幻方的幻和（每行三个数的和）是{H}，最中间的数是几？",
        f"三阶幻方中，每行、每列、每条对角线的三个数相加都得{H}，正中间的数是多少？",
        f"用9个数组成一个三阶幻方，使每行、每列、每条对角线上三个数的和都是{H}，中间数是多少？",
    ])
    lines = [
        f"9个数总和 = 3 × {H} = {3 * H}",
        f"中间数 = {3 * H} ÷ 9 = {num(ans)}",
    ]
    return ins, lines, ans


_reg("magic_square_center", magic_square_center)


_CROSS_SETS = [
    ([1, 2, 3, 4, 5], 15),
    ([2, 3, 4, 5, 6], 20),
    ([3, 4, 5, 6, 7], 25),
]


def cross_array_center(rng):
    nums, total = rng.choice(_CROSS_SETS)
    lo, hi = (total + min(nums)) // 2, (total + max(nums)) // 2
    K = rng.randint(lo, hi)
    ans = 2 * K - total
    s = "、".join(str(x) for x in nums)
    ins = rng.choice([
        f"把{s}这五个数分别填入十字形数阵的五个圈中（中间一个圈，横、竖各两个圈），使横行三个数的和与竖行三个数的和都等于{K}。中间圈里应填几？",
        f"将{s}分别填入十字数阵（中间一格，上下左右各一格），使横行、竖行三个数的和都是{K}，中间一格填几？",
        f"把{s}填入十字形的五个圆圈里，中间的圆圈横、竖共用，要使横行与竖行三个数的和都等于{K}，中间应填几？",
        f"十字数阵有五个圈（中间一个，横两个，竖两个），把{s}填进去，使横行和竖行三个数的和都等于{K}，中间圈填多少？",
        f"用{s}这五个数填十字形数阵，中间数横、竖各算一次，若横行三个数与竖行三个数的和都等于{K}，中间数是几？",
        f"把{s}填进十字框（中间一个圈，四周四个圈，横竖各过中间），使横行、竖行三数之和都为{K}，中间圈填几？",
        f"将{s}填入十字形数阵图，要求横行三数之和等于竖行三数之和且都等于{K}，中间的数是多少？",
        f"把{s}分别放在十字形的五个位置上（中间位置横、竖共用），使横行、竖行三个数相加都得{K}，中间位置放几？",
        f"十字数阵图中，把{s}填入五个圈，使每条线上三个数的和都是{K}，中间圈里的数是几？",
        f"用{s}填十字形（中间一个圈，横、竖线上各两个圈），两条线上三个数的和都等于{K}，中间应填几？",
    ])
    lines = [
        f"五数总和 = {' + '.join(str(x) for x in nums)} = {total}",
        f"2 × {K} = {2 * K}",
        f"{2 * K} - {total} = {ans}",
    ]
    return ins, lines, ans


_reg("cross_array_center", cross_array_center)


def symbol_elimination(rng):
    tri = rng.randint(3, 9)
    sq = rng.randint(2, 9)
    A = 2 * tri + sq
    B = tri + sq
    ans = tri
    ins = rng.choice([
        f"已知△+△+□={A}，△+□={B}，求△代表的数。",
        f"如果△+△+□={A}，并且△+□={B}，那么△等于多少？",
        f"算式△+△+□={A}，△+□={B}，△代表的数是几？",
        f"已知△与□满足：△+△+□={A}，△+□={B}，求△。",
    ])
    lines = [
        f"△+△+□ = {A} = {A}",
        f"△+□ = {B} = {B}",
        f"({A}) - ({B}) = {ans}",
        f"△ = {ans} = {ans}",
    ]
    return ins, lines, ans


_reg("symbol_elimination", symbol_elimination)


def split_max_product(rng):
    N = rng.randint(5, 15) * 2
    h = N // 2
    ans = h * h
    ins = rng.choice([
        f"把{N}拆成两个自然数的和，再把这两个数相乘，乘积最大是多少？",
        f"两个自然数的和是{N}，这两个数的乘积最大是多少？",
        f"把{N}分成两个自然数相加，怎样分使它们的乘积最大？最大乘积是多少？",
        f"两个数相加等于{N}，这两个数相乘，积最大是多少？",
    ])
    lines = [
        f"{N} ÷ 2 = {h}",
        f"两个数都是 = {h} = {h}",
        f"{h} × {h} = {ans}",
    ]
    return ins, lines, ans


_reg("split_max_product", split_max_product)


def fence_max_area(rng):
    L = rng.randint(4, 10) * 4
    s = L // 4
    ans = s * s
    ins = rng.choice([
        f"用{L}米长的篱笆围一个长方形（长和宽都是整米数），怎样围面积最大？最大面积是多少平方米？",
        f"一根{L}米长的篱笆围成一个长方形或正方形，围成的图形面积最大是多少平方米？",
        f"用{L}米篱笆围一块长方形菜地（长、宽取整米数），最大面积是多少平方米？",
        f"园丁用{L}米的篱笆围长方形花圃，长和宽都是整米数，怎样围面积最大？最大是多少平方米？",
    ])
    lines = [
        f"正方形边长 = {L} ÷ 4 = {s}米",
        f"长和宽都是 = {s} = {s}米",
        f"{s} × {s} = {ans}平方米",
    ]
    return ins, lines, ans


_reg("fence_max_area", fence_max_area)


def pancake_minutes(rng):
    n = rng.randint(3, 7)
    m = rng.choice([2, 3, 4])
    ans = n * m
    ins = rng.choice([
        f"一只平底锅每次最多能放2张饼，每张饼要烙两面，每面需要{m}分钟。烙{n}张饼至少需要多少分钟？",
        f"平底锅一次最多烙2张饼，每张饼两面都要烙，每面{m}分钟，烙熟{n}张饼最少要几分钟？",
        f"用一只每次能放2张饼的平底锅烙饼，每张饼烙两面，每面{m}分钟，烙{n}张饼至少需要多少分钟？",
        f"一口锅每次最多烙2张饼，每张饼需烙正反两面，每面{m}分钟，烙{n}张饼最少需要多少分钟？",
    ])
    lines = [
        f"饼的总面数 = 2 × {n} = {2 * n}面",
        f"需要烙的次数 = {2 * n} ÷ 2 = {n}次",
        f"{n} × {m} = {ans}分钟",
    ]
    return ins, lines, ans


_reg("pancake_minutes", pancake_minutes)


def queue_min_wait(rng):
    t1 = rng.randint(1, 4)
    t2 = rng.randint(t1 + 1, t1 + 4)
    t3 = rng.randint(t2 + 1, t2 + 4)
    ans = 3 * t1 + 2 * t2 + t3
    ins = rng.choice([
        f"3个人各拿一个水桶去打水，打水时间分别需要{t1}分钟、{t2}分钟、{t3}分钟。如果只有一个水龙头，怎样安排使三人等候（含自己打水）的总时间最少？最少是多少分钟？",
        f"三人打水，用时分别为{t1}分、{t2}分、{t3}分，只有一个水龙头，按什么顺序三人花费的总时间（含各自打水时间）最少？最少多少分钟？",
        f"3个同学排队接水，每人接水时间分别是{t1}分钟、{t2}分钟、{t3}分钟，一个水龙头，怎样安排使三人等候总时间（含自己接水）最少？最少多少分钟？",
        f"甲乙丙三人打水，分别需要{t1}、{t2}、{t3}分钟，只有一个水龙头，如何安排顺序使三人从到达到打完水的总时间最少？最少是多少分钟？",
    ])
    lines = [
        f"最短的先打 = {t1} × 3 = {3 * t1}分钟",
        f"其次 = {t2} × 2 = {2 * t2}分钟",
        f"最长的最后 = {t3} × 1 = {t3}分钟",
        f"{3 * t1} + {2 * t2} + {t3} = {ans}分钟",
    ]
    return ins, lines, ans


_reg("queue_min_wait", queue_min_wait)


# ---------------------------------------------------------------------------
# Batch 3: optimization, inclusion-exclusion, travel, geometry, charts
# ---------------------------------------------------------------------------

def rent_car_cheapest(rng):
    a = rng.randint(7, 12)       # big car capacity
    x = rng.randint(60, 120)     # big car rent
    b = rng.randint(3, 6)        # small car capacity
    y = rng.randint(30, 70)      # small car rent
    n = rng.randint(20, 50)
    best = None
    for B in range((n + a - 1) // a + 1):
        rem = n - a * B
        C = 0 if rem <= 0 else (rem + b - 1) // b
        cost = B * x + C * y
        if best is None or cost < best[0]:
            best = (cost, B, C)
    ans, B, C = best
    ins = rng.choice([
        f"租车去春游：大车每辆限坐{a}人、租金{x}元，小车每辆限坐{b}人、租金{y}元。{n}人出行，怎样租车最省钱？最少需要多少元？",
        f"旅行社有两种车：大车限乘{a}人，每辆{x}元；小车限乘{b}人，每辆{y}元。{n}人乘车，最少要花多少元租车？",
        f"学校组织{n}名师生外出，可租大车（限坐{a}人，每辆{x}元）和小车（限坐{b}人，每辆{y}元），怎样租车费用最少？最少多少元？",
        f"大车限坐{a}人、租金{x}元，小车限坐{b}人、租金{y}元，{n}人参加活动，最省钱的租车方案需要多少元？",
    ])
    lines = [
        f"总人数 = {n} = {n}人",
        f"大车限坐 = {a} = {a}人",
        f"小车限坐 = {b} = {b}人",
        f"大车辆数 = {B} = {B}辆",
        f"小车辆数 = {C} = {C}辆",
        f"{B} × {x} = {B * x}元",
        f"{C} × {y} = {C * y}元",
        f"{B * x} + {C * y} = {ans}元",
    ]
    return ins, lines, ans


_reg("rent_car_cheapest", rent_car_cheapest)


def ticket_plan_compare(rng):
    A = rng.randint(50, 90)       # adult ticket
    B = rng.randint(20, 40)       # child ticket
    C = rng.randint(25, 45)       # group ticket per person
    m = rng.randint(6, 9)         # adults
    n = rng.randint(4, 7)         # children
    total = m + n
    P1 = m * A + n * B
    P2 = total * C
    P3 = 10 * C + (total - 10) * B
    ans = min(P1, P2, P3)
    ins = rng.choice([
        f"公园门票：成人票每张{A}元，儿童票每张{B}元；10人及以上可买团体票，每张{C}元。{m}个成人带{n}个儿童去游玩，怎样买票最省钱？最少多少元？",
        f"景区成人票{A}元一张，儿童票{B}元一张，团体票（10人以上）每张{C}元。{m}名成人和{n}名儿童怎样购票最合算？最少花多少元？",
        f"动物园成人票每张{A}元，儿童票每张{B}元，满10人可购每张{C}元的团体票。{m}个成人、{n}个儿童最少要花多少元买票？",
        f"博物馆门票成人{A}元、儿童{B}元，10人起可买团体票每人{C}元。{m}位家长带{n}名儿童，最少需要多少元？",
    ])
    lines = [
        f"方案一分开买 = {m} × {A} + {n} × {B} = {P1}元",
        f"方案二全团体 = {total} × {C} = {P2}元",
        f"方案三混合 = 10 × {C} + {total - 10} × {B} = {P3}元",
        f"最少 = {ans} = {ans}元",
    ]
    return ins, lines, ans


_reg("ticket_plan_compare", ticket_plan_compare)


def inclusion_exclusion_two(rng):
    for _ in range(100):
        n = rng.randint(40, 50)
        c = rng.randint(2, 8)
        neither = n - c
        a = rng.randint(neither // 2 + 1, neither - 2)
        b = rng.randint(neither // 2 + 1, neither - 2)
        both = a + b - neither
        if 1 <= both <= min(a, b) - 1:
            break
    ans = both
    ins = rng.choice([
        f"某班有{n}人，参加数学兴趣小组的有{a}人，参加英语兴趣小组的有{b}人，两个小组都不参加的有{c}人。两个小组都参加的有多少人？",
        f"六年级一班{n}名同学中，参加数学小组的{a}人，参加英语小组的{b}人，两科都不参加的{c}人。两科都参加的有几人？",
        f"一个班{n}人，订《数学报》的{a}人，订《英语报》的{b}人，两种都不订的{c}人。两种报纸都订的有多少人？",
        f"班级共{n}人，喜欢数学的{a}人，喜欢英语的{b}人，两科都不喜欢的{c}人。两科都喜欢的有多少人？",
    ])
    lines = [
        f"至少参加一个 = {n} - {c} = {neither}人",
        f"{a} + {b} = {a + b}人",
        f"都参加 = {a + b} - {neither} = {ans}人",
    ]
    return ins, lines, ans


_reg("inclusion_exclusion_two", inclusion_exclusion_two)


def meet_post_reach_time(rng):
    combos = [(v1, v2, k) for v1 in (40, 50, 60, 80) for v2 in (30, 40, 50, 60)
              for k in range(3, 9) if v1 != v2 and (v2 * k) % v1 == 0]
    v1, v2, k = rng.choice(combos)
    s = v1 + v2
    D = s * k
    d2 = v2 * k
    ans = Fraction(d2, v1)
    ins = rng.choice([
        f"甲、乙两人从相距{D}米的两地同时相向而行，甲每分钟走{v1}米，乙每分钟走{v2}米。相遇后甲还要走多少分钟才能到达B地？",
        f"A、B两地相距{D}米，甲、乙同时相向出发，甲每分钟行{v1}米，乙每分钟行{v2}米。相遇后甲再走几分钟到B地？",
        f"甲、乙从相距{D}米的两地相向而行，甲速{v1}米/分，乙速{v2}米/分，相遇后甲还要多少分钟到达对方出发地？",
        f"两地相距{D}米，小明每分钟走{v1}米，小红每分钟走{v2}米，同时相向而行，相遇后小明还要走多少分钟到B地？",
    ])
    lines = [
        f"速度和 = {v1} + {v2} = {s}米/分",
        f"相遇时间 = {D} ÷ {s} = {k}分钟",
        f"相遇时乙走 = {v2} × {k} = {d2}米",
        f"甲还需时间 = {d2} ÷ {v1} = {num(ans)}分钟",
    ]
    return ins, lines, ans


_reg("meet_post_reach_time", meet_post_reach_time)


_MIDPOINT_PAIRS = [
    (60, 40, 10), (50, 40, 18), (80, 60, 14), (75, 50, 10),
    (90, 60, 10), (60, 45, 14), (70, 50, 12), (80, 48, 8),
]


def meet_midpoint_reverse(rng):
    v1, v2, factor = rng.choice(_MIDPOINT_PAIRS)
    g = rng.randint(5, 20)
    dv = v1 - v2
    sv = v1 + v2
    D = 2 * g * sv // dv
    ans = D
    ins = rng.choice([
        f"甲、乙两车同时从A、B两地相向而行，甲车每小时行{v1}千米，乙车每小时行{v2}千米（甲车快），相遇点距中点{g}千米。A、B两地相距多少千米？",
        f"甲、乙两人同时从两地相向出发，甲每分钟走{v1}米，乙每分钟走{v2}米，相遇时甲走过中点{g}米。两地相距多少米？",
        f"快船每小时行{v1}千米，慢船每小时行{v2}千米，两船同时从两港相向开出，相遇点距两港中点{g}千米。两港相距多少千米？",
        f"甲、乙从A、B两地同时相向而行，甲速{v1}千米/时，乙速{v2}千米/时，相遇地点离中点{g}千米。求A、B两地距离。",
    ])
    lines = [
        f"速度差 = {v1} - {v2} = {dv}",
        f"速度和 = {v1} + {v2} = {sv}",
        f"路程差 = 2 × {g} = {2 * g}千米",
        f"全程 = {2 * g} × {sv} ÷ {dv} = {ans}千米",
    ]
    return ins, lines, ans


_reg("meet_midpoint_reverse", meet_midpoint_reverse)


_DOUBLE_MEET = [
    (100, 60, 40), (90, 50, 40), (120, 70, 50), (140, 80, 60),
    (105, 75, 50), (120, 90, 60), (70, 60, 45), (150, 90, 60),
    (135, 75, 60), (175, 75, 50), (180, 90, 60), (96, 60, 36),
]


def double_meet_gap(rng):
    D, v1, v2 = rng.choice(_DOUBLE_MEET)
    sv = v1 + v2
    a1 = D * v1 // sv
    pos = 2 * D - 3 * a1
    ans = a1 - pos
    ins = rng.choice([
        f"甲、乙两人同时从相距{D}米的A、B两地相向而行，甲每分钟走{v1}米，乙每分钟走{v2}米。第一次相遇后两人继续前进，到达对方出发地后立即返回。第一次相遇点与第二次相遇点之间相距多少米？",
        f"甲、乙从相距{D}米的两地同时相向出发，甲速{v1}米/分，乙速{v2}米/分，相遇后继续走到对方出发地立即返回。两次相遇点之间相距多少米？",
        f"A、B两地相距{D}米，甲、乙同时相向而行，甲每分钟{v1}米，乙每分钟{v2}米，第一次相遇后继续前行并立即返回。两次相遇点相距多少米？",
        f"甲、乙两人在相距{D}米的两地间往返行走（同时相向出发），甲速{v1}米/分，乙速{v2}米/分。第一次与第二次相遇点之间相距多少米？",
    ])
    lines = [
        f"速度和 = {v1} + {v2} = {sv}米/分",
        f"第一次相遇甲走 = {D} × {v1} ÷ {sv} = {a1}米",
        f"第二次相遇甲共走 = 3 × {a1} = {3 * a1}米",
        f"第二次相遇点距A = 2 × {D} - {3 * a1} = {pos}米",
        f"两次相遇点距离 = {a1} - {pos} = {ans}米",
    ]
    return ins, lines, ans


_reg("double_meet_gap", double_meet_gap)


def uphill_downhill_avg(rng):
    v1 = rng.choice([40, 50, 60])
    t1 = rng.randint(3, 8)
    d1 = v1 * t1
    v2 = rng.choice([80, 100, 120])
    t2 = rng.randint(2, 6)
    d2 = v2 * t2
    total = d1 + d2
    tt = t1 + t2
    ans = Fraction(total, tt)
    ins = rng.choice([
        f"小明骑自行车上坡，{d1}米的路程速度是每分钟{v1}米；下坡{d2}米，速度是每分钟{v2}米。他上、下坡的平均速度是每分钟多少米？",
        f"一段山路，上坡{d1}米，小明骑车每分钟行{v1}米；下坡{d2}米，每分钟行{v2}米。求小明上、下坡的平均速度。",
        f"小华骑车上坡{d1}米用每分钟{v1}米的速度，下坡{d2}米用每分钟{v2}米的速度，他全程的平均速度是多少米/分？",
        f"上坡路长{d1}米，骑车速度{v1}米/分；下坡路长{d2}米，速度{v2}米/分。上、下坡的平均速度是多少米每分？",
    ])
    lines = [
        f"上坡时间 = {d1} ÷ {v1} = {t1}分钟",
        f"下坡时间 = {d2} ÷ {v2} = {t2}分钟",
        f"总路程 = {d1} + {d2} = {total}米",
        f"总时间 = {t1} + {t2} = {tt}分钟",
        f"平均速度 = {total} ÷ {tt} = {num(ans)}米/分",
    ]
    return ins, lines, ans


_reg("uphill_downhill_avg", uphill_downhill_avg)


def goat_grazing_area(rng):
    a = rng.randint(6, 12)
    r = rng.randint(4, a)
    full = Fraction(314, 100) * r * r
    ans = full * Fraction(3, 4)
    ins = rng.choice([
        f"一只羊被拴在边长{a}米的正方形房屋的一个墙角，绳长{r}米（绳长不超过房屋边长）。这只羊能吃到草的面积是多少平方米（π取3.14）？",
        f"在边长{a}米的正方形草地一角的木桩上拴着一只羊，绳长{r}米，羊能吃到草的面积是多少平方米（π取3.14）？",
        f"一只羊用{r}米长的绳子拴在边长{a}米的正方形房屋墙角，它的吃草面积是多少平方米（π取3.14）？",
        f"正方形房屋边长{a}米，墙角拴羊，绳长{r}米，羊可以吃到多少平方米的草（π取3.14）？",
    ])
    lines = [
        f"边长比较 = {a} - {r} = {a - r}米",
        f"360 - 90 = 270度",
        f"270 ÷ 360 = 3/4",
        f"3.14 × {r} × {r} = {_d(full)}平方米",
        f"{_d(full)} × 3/4 = {_d(ans)}平方米",
    ]
    return ins, lines, ans


_reg("goat_grazing_area", goat_grazing_area)


def track_start_gap(rng):
    r = rng.randint(30, 40)
    d = rng.choice([1, Fraction(5, 4), Fraction(3, 2)])
    outer = 2 * Fraction(314, 100) * (r + d)
    inner = 2 * Fraction(314, 100) * r
    ans = outer - inner
    ins = rng.choice([
        f"田径场弯道部分是半圆，最内圈弯道半径为{r}米，每条跑道宽{_d(d)}米。跑一圈时，相邻两条跑道的起跑线应相差多少米（π取3.14）？",
        f"运动场弯道是半圆形，内圈弯道半径{r}米，跑道宽{_d(d)}米。跑一圈时，相邻跑道的起跑线要相差多少米（π取3.14）？",
        f"环形跑道的弯道是半圆，最内圈半径{r}米，道宽{_d(d)}米。跑一圈，相邻两道的起跑线相差多少米（π取3.14）？",
        f"田径场最内圈弯道半径为{r}米，每条跑道宽{_d(d)}米，跑一圈时外圈起跑线要比内圈提前多少米（π取3.14）？",
    ])
    lines = [
        f"外圈弯道半径 = {r} + {_d(d)} = {_d(r + d)}米",
        f"2 × 3.14 × {_d(r + d)} = {_d(outer)}米",
        f"2 × 3.14 × {r} = {_d(inner)}米",
        f"{_d(outer)} - {_d(inner)} = {_d(ans)}米",
    ]
    return ins, lines, ans


_reg("track_start_gap", track_start_gap)


def wire_reshape_area(rng):
    a = rng.randint(6, 12)
    b = rng.randint(4, a - 2)
    if (a + b) % 2 != 0:
        b += 1
    p = 2 * (a + b)
    s = p // 4
    asq = s * s
    ab = a * b
    ans = asq - ab
    ins = rng.choice([
        f"一根铁丝恰好能围成长{a}厘米、宽{b}厘米的长方形。如果把它改围成一个正方形，正方形的面积比长方形多多少平方厘米？",
        f"一根铁丝围成长{a}厘米、宽{b}厘米的长方形正好用完，改围成正方形后，面积增加了多少平方厘米？",
        f"用一根铁丝围成长{a}厘米、宽{b}厘米的长方形，若改围成正方形，面积会增加多少平方厘米？",
        f"一根铁丝可以围成长{a}厘米、宽{b}厘米的长方形，把它拉直后围成正方形，面积比原来多多少平方厘米？",
    ])
    lines = [
        f"周长 = 2 × ({a} + {b}) = {p}厘米",
        f"正方形边长 = {p} ÷ 4 = {s}厘米",
        f"{s} × {s} = {asq}平方厘米",
        f"{a} × {b} = {ab}平方厘米",
        f"{asq} - {ab} = {ans}平方厘米",
    ]
    return ins, lines, ans


_reg("wire_reshape_area", wire_reshape_area)


def cylinder_cone_vol_diff(rng):
    X = rng.randint(6, 30) * 2
    each = Fraction(X, 2)
    ans = 3 * each
    ins = rng.choice([
        f"一个圆柱和一个圆锥等底等高，它们的体积相差{X}立方分米。圆柱的体积是多少立方分米？",
        f"等底等高的圆柱和圆锥，体积相差{X}立方厘米，圆柱的体积是多少立方厘米？",
        f"一个圆柱与一个圆锥底面积和高都相等，圆柱比圆锥的体积多{X}立方米，圆柱体积是多少立方米？",
        f"等底等高的圆柱和圆锥体积之差是{X}立方分米，圆柱的体积是多少立方分米？",
    ])
    lines = [
        f"份数差 = 3 - 1 = 2份",
        f"每份 = {X} ÷ 2 = {num(each)}立方分米",
        f"圆柱体积 = 3 × {num(each)} = {num(ans)}立方分米",
    ]
    return ins, lines, ans


_reg("cylinder_cone_vol_diff", cylinder_cone_vol_diff)


def cylinder_reassemble_surface(rng):
    r = rng.randint(3, 10)
    h = rng.randint(5, 15)
    rh = r * h
    ans = 2 * rh
    ins = rng.choice([
        f"把一个底面半径{r}厘米、高{h}厘米的圆柱切成若干等份，拼成一个近似的长方体，表面积增加了多少平方厘米？",
        f"将底面半径{r}厘米、高{h}厘米的圆柱切拼成一个近似长方体，表面积比原来增加多少平方厘米？",
        f"一个圆柱底面半径{r}厘米、高{h}厘米，把它切开拼成近似的长方体，表面积增加了多少平方厘米？",
        f"把底面半径{r}厘米、高{h}厘米的圆柱体切拼成近似长方体后，表面积增加了多少平方厘米？",
    ])
    lines = [
        f"新增长方形面 = {r} × {h} = {rh}平方厘米",
        f"2 × {rh} = {ans}平方厘米",
    ]
    return ins, lines, ans


_reg("cylinder_reassemble_surface", cylinder_reassemble_surface)


def circle_inscribed_square_shaded(rng):
    d = rng.randint(3, 10) * 2
    r = d // 2
    circle = Fraction(314, 100) * r * r
    sq = Fraction(d * d, 2)
    ans = circle - sq
    ins = rng.choice([
        f"在直径{d}厘米的圆内画一个最大的正方形，圆内正方形以外的阴影面积是多少平方厘米（π取3.14）？",
        f"一个直径{d}厘米的圆里有一个最大的正方形，圆与正方形之间的面积是多少平方厘米（π取3.14）？",
        f"在直径为{d}厘米的圆中剪出最大的正方形，剩下的边角料面积是多少平方厘米（π取3.14）？",
        f"直径{d}厘米的圆内有一个最大的正方形，圆面积比正方形多多少平方厘米（π取3.14）？",
    ])
    lines = [
        f"{d} ÷ 2 = {r}厘米",
        f"3.14 × {r} × {r} = {_d(circle)}平方厘米",
        f"正方形面积 = {d} × {d} ÷ 2 = {num(sq)}平方厘米",
        f"{_d(circle)} - {num(sq)} = {_d(ans)}平方厘米",
    ]
    return ins, lines, ans


_reg("circle_inscribed_square_shaded", circle_inscribed_square_shaded)


def pie_chart_total(rng):
    a = rng.choice([30, 36, 40, 45, 60, 72, 90, 120])
    k = rng.randint(2, 8)
    x = a * k
    q = 360 // a
    ans = 360 * k
    ins = rng.choice([
        f"扇形统计图中，喜欢足球的同学所在扇形的圆心角是{a}度，已知喜欢足球的有{x}人。一共调查了多少人？",
        f"在一个扇形统计图里，表示文艺书的扇形圆心角是{a}度，已知文艺书有{x}本。图书一共有多少本？",
        f"扇形统计图中，成绩优秀的学生对应的圆心角为{a}度，成绩优秀的有{x}人。参加统计的学生共多少人？",
        f"某班学生最喜欢的运动的扇形统计图中，喜欢乒乓球的扇形圆心角是{a}度，喜欢乒乓球的有{x}人。这个班有多少人？",
    ])
    lines = [
        f"圆心角 = {a} = {a}度",
        f"360 ÷ {a} = {q}份",
        f"{x} × {q} = {ans}人",
    ]
    return ins, lines, ans


_reg("pie_chart_total", pie_chart_total)


def bar_chart_average(rng):
    nums = [rng.randint(4, 20) for _ in range(4)]
    e = rng.randint(4, 20)
    rem = (sum(nums) + e) % 5
    if rem != 0:
        e += 5 - rem
        if e > 25:
            e -= 5
    nums.append(e)
    s = sum(nums)
    ans = s // 5
    who = rng.choice(["小明", "小红", "小华", "小丽", "小军"])
    ins = rng.choice([
        f"条形统计图显示5名同学一学期读书本数分别为{nums[0]}本、{nums[1]}本、{nums[2]}本、{nums[3]}本、{nums[4]}本。平均每人读了多少本？",
        f"5名同学的身高分别是{nums[0]}厘米、{nums[1]}厘米、{nums[2]}厘米、{nums[3]}厘米、{nums[4]}厘米，他们的平均身高是多少厘米？",
        f"条形图记录了{who}等5名同学1分钟跳绳的个数：{nums[0]}、{nums[1]}、{nums[2]}、{nums[3]}、{nums[4]}，平均每人跳多少个？",
        f"5个同学的数学成绩分别为{nums[0]}分、{nums[1]}分、{nums[2]}分、{nums[3]}分、{nums[4]}分，平均分是多少分？",
    ])
    lines = [
        f"人数 = 5 = 5人",
        f"{nums[0]} + {nums[1]} + {nums[2]} + {nums[3]} + {nums[4]} = {s}",
        f"{s} ÷ 5 = {ans}",
    ]
    return ins, lines, ans


_reg("bar_chart_average", bar_chart_average)


# ---------------------------------------------------------------------------
# Batch 4: charts, patterns, counting, number theory, geometry, misc
# ---------------------------------------------------------------------------

def line_chart_growth(rng):
    a = rng.randint(20, 60)
    steps = [rng.randint(3, 15) for _ in range(4)]
    vals = [a]
    for s in steps:
        vals.append(vals[-1] + s)
    ans = max(steps)
    days = ["周一", "周二", "周三", "周四", "周五"]
    ins = rng.choice([
        f"折线统计图记录了商店周一到周五的销售额（元）：{vals[0]}、{vals[1]}、{vals[2]}、{vals[3]}、{vals[4]}。哪一天比前一天增加得最多？最多增加了多少元？",
        f"某店周一至周五的营业额（元）分别是{vals[0]}、{vals[1]}、{vals[2]}、{vals[3]}、{vals[4]}，相邻两天中，营业额最多增加了多少元？",
        f"折线图显示一周五天的用水量（吨）：{vals[0]}、{vals[1]}、{vals[2]}、{vals[3]}、{vals[4]}，第二天比第一天最多多多少吨？",
        f"某站周一到周五的客流量（人次）为{vals[0]}、{vals[1]}、{vals[2]}、{vals[3]}、{vals[4]}，相邻两天的最大增幅是多少人次？",
    ])
    lines = [
        f"{vals[1]} - {vals[0]} = {steps[0]}",
        f"{vals[2]} - {vals[1]} = {steps[1]}",
        f"{vals[3]} - {vals[2]} = {steps[2]}",
        f"{vals[4]} - {vals[3]} = {steps[3]}",
        f"最大增幅 = {ans} = {ans}",
    ]
    return ins, lines, ans


_reg("line_chart_growth", line_chart_growth)


def fraction_unit_to_prime(rng):
    b = rng.randint(5, 12)
    a = rng.randint(b - 3, b + 3)
    while a == b or a >= 2 * b:
        a = rng.randint(b - 3, b + 3)
    ans = 2 * b - a
    ins = rng.choice([
        f"分数{a}/{b}再添上几个它的分数单位（1/{b}）后等于最小的质数？",
        f"{a}/{b}的分数单位是1/{b}，再添上几个这样的单位就是最小的质数？",
        f"分数{a}/{b}再加上几个1/{b}等于最小的质数？",
        f"{a}/{b}再添多少个它的分数单位（1/{b}）后，结果是最小的质数？",
    ])
    lines = [
        f"分数单位 = 1 ÷ {b} = 1/{b}",
        f"最小的质数 = 2 = 2",
        f"2 × {b} = {2 * b}",
        f"{2 * b} - {a} = {ans}",
    ]
    return ins, lines, ans


_reg("fraction_unit_to_prime", fraction_unit_to_prime)


def matchstick_hex_pattern(rng):
    n = rng.randint(3, 15)
    ans = 5 * n + 1
    ins = rng.choice([
        f"用小棒摆六边形：摆1个用6根，摆2个用11根，摆3个用16根……照这样摆下去，摆{n}个六边形需要多少根小棒？",
        f"摆1个六边形用6根小棒，摆2个用11根，摆3个用16根，摆{n}个六边形需要多少根小棒？",
        f"小棒摆六边形，个数与根数的规律是6、11、16……摆{n}个六边形要用多少根小棒？",
        f"照样子摆六边形：1个用6根，2个用11根，3个用16根，摆{n}个需要多少根小棒？",
    ])
    lines = [
        f"11 - 6 = 5根",
        f"3 × 5 + 1 = 16根",
        f"5 × {n} + 1 = {ans}根",
    ]
    return ins, lines, ans


_reg("matchstick_hex_pattern", matchstick_hex_pattern)


def count_segments(rng):
    n = rng.randint(4, 12)
    ans = n * (n - 1) // 2
    ins = rng.choice([
        f"一条线段上有{n}个点（包括两端点），一共可以数出多少条线段？",
        f"线段AB上有{n}个点（含A、B），图中共有多少条线段？",
        f"一条直线上有{n}个点，每两个点确定一条线段，一共有多少条线段？",
        f"线段上共有{n}个点（包括端点），任意两点间的线段共有多少条？",
    ])
    lines = [
        f"点数 = {n} = {n}个",
        f"{n} - 1 = {n - 1}",
        f"{n} × {n - 1} ÷ 2 = {ans}条",
    ]
    return ins, lines, ans


_reg("count_segments", count_segments)


def count_rectangles_grid(rng):
    m = rng.randint(2, 5)
    n = rng.randint(2, 5)
    ans = m * (m + 1) * n * (n + 1) // 4
    ins = rng.choice([
        f"一个{m}行{n}列的方格网（由{m}×{n}个小正方形组成），一共可以数出多少个长方形（包括正方形）？",
        f"在{m}行{n}列的网格图中，一共能数出多少个长方形（含正方形）？",
        f"由{m}行{n}列小方格组成的网格，共有多少个长方形（正方形也算）？",
        f"数一数，{m}行{n}列的方格网中共有多少个长方形（包括正方形）？",
    ])
    lines = [
        f"{m} + 1 = {m + 1}",
        f"{n} + 1 = {n + 1}",
        f"{m + 1} × {n + 1} × {m} × {n} ÷ 4 = {ans}个",
    ]
    return ins, lines, ans


_reg("count_rectangles_grid", count_rectangles_grid)


def elevator_floor_diff(rng):
    f = rng.randint(1, 3)
    t = rng.randint(3, 12)
    ans = t + f
    ins = rng.choice([
        f"电梯从地下{f}层（记作-{f}层）上升到地上{t}层，电梯一共上升了多少层？",
        f"一幢楼的最底层是地下{f}层，电梯从-{f}层坐到{t}层，上升了多少层？",
        f"电梯在-{f}层，要到地上{t}层，需要上升多少层？",
        f"从地下{f}层乘电梯到地上{t}层，电梯共上升多少层？",
    ])
    lines = [
        f"地上层数 = {t} = {t}层",
        f"地下层数 = {f} = {f}层",
        f"{t} - (-{f}) = {ans}层",
    ]
    return ins, lines, ans


_reg("elevator_floor_diff", elevator_floor_diff)


def unit_fraction_split(rng):
    n = rng.randint(3, 12)
    ans = n * (n + 1)
    ins = rng.choice([
        f"在括号里填上不同的自然数：1/{n} = 1/{n + 1} + 1/(?)，括号里应填几？",
        f"把1/{n}拆成两个不同的分数单位之和：1/{n} = 1/{n + 1} + 1/(?)，求括号里的数。",
        f"已知1/{n} = 1/{n + 1} + 1/x，且x是自然数，x等于多少？",
        f"分数拆分：1/{n} = 1/{n + 1} + 1/(?)，问号处应填哪个自然数？",
    ])
    lines = [
        f"{n} + 1 = {n + 1}",
        f"{n} × {n + 1} = {ans}",
        f"1 ÷ {n + 1} + 1 ÷ {ans} = 1/{n}",
        f"括号里的数 = {ans} = {ans}",
    ]
    return ins, lines, ans


_reg("unit_fraction_split", unit_fraction_split)


def number_table_position(rng):
    k = rng.randint(4, 8)
    N = rng.randint(10, 60)
    q, r = divmod(N - 1, k)
    ans = q + 1
    ins = rng.choice([
        f"自然数从1开始按每行{k}个数依次排列，{N}排在第几行？",
        f"把自然数按每行{k}个数依次排列，{N}在第几行？",
        f"自然数按每行{k}个排列成数表，{N}位于第几行？",
        f"数表中每行排{k}个连续自然数，{N}排在第几行？",
    ])
    lines = [
        f"{N} - 1 = {N - 1}",
        f"{k} × {q} = {k * q}",
        f"{N - 1} - {k * q} = {r}",
        f"行数 = {q} + 1 = {ans}",
    ]
    return ins, lines, ans


_reg("number_table_position", number_table_position)


def calendar_box_sum(rng):
    a = rng.randint(1, 22)
    S = 4 * a + 16
    ans = a
    ins = rng.choice([
        f"在某月的日历上，用正方形框出相邻两行两列共四个数，这四个数的和是{S}，其中最小的数是几？",
        f"日历上用正方形框出2行2列四个数，它们的和是{S}，这四个数中最小的是多少？",
        f"在日历中框出2×2的四个数（上下相邻两行、左右相邻两列），四个数之和为{S}，最小的数是几？",
        f"某月日历上，一个正方形框住2行2列四个数，四个数相加得{S}，最小的数是多少？",
    ])
    lines = [
        f"4个数的平均数 = {S} ÷ 4 = {a + 4}",
        f"最小数 = {a + 4} - 4 = {ans}",
    ]
    return ins, lines, ans


_reg("calendar_box_sum", calendar_box_sum)


def cross_frame_sum(rng):
    x = rng.randint(5, 30)
    S = 5 * x
    ans = x
    ins = rng.choice([
        f"在自然数排列的数表中，用十字框框出5个数（中间一个数，它的上、下、左、右各一个数），这5个数的和是{S}。中间的数是几？",
        f"十字框在数表中框出5个数（中间一个，上下左右各一个），5个数的和是{S}，中间数是多少？",
        f"数表里用十字框框出5个数，中间数的上下左右各有一个数，这5个数相加得{S}，中间的数是几？",
        f"在排列整齐的数表中，十字框框出的5个数之和为{S}，正中间的数是多少？",
    ])
    lines = [
        f"十字框有5个数 = 5 = 5个",
        f"总和是中间数的倍数 = 5 = 5倍",
        f"中间数 = {S} ÷ 5 = {ans}",
    ]
    return ins, lines, ans


_reg("cross_frame_sum", cross_frame_sum)


def median_find(rng):
    nums = rng.sample(range(130, 171), 7)
    s = sorted(nums)
    ans = s[3]
    ins = rng.choice([
        f"7名同学的身高（厘米）分别为{nums[0]}、{nums[1]}、{nums[2]}、{nums[3]}、{nums[4]}、{nums[5]}、{nums[6]}，这组数据的中位数是多少？",
        f"7名队员的体重（千克）分别是{nums[0]}、{nums[1]}、{nums[2]}、{nums[3]}、{nums[4]}、{nums[5]}、{nums[6]}，中位数是多少？",
        f"7次数学测验的成绩（分）为{nums[0]}、{nums[1]}、{nums[2]}、{nums[3]}、{nums[4]}、{nums[5]}、{nums[6]}，这组数据的中位数是多少？",
        f"7天的气温（℃）分别是{nums[0]}、{nums[1]}、{nums[2]}、{nums[3]}、{nums[4]}、{nums[5]}、{nums[6]}，中位数是多少？",
    ])
    lines = [
        f"7 + 1 = 8",
        f"8 ÷ 2 = 4",
        f"7个数总和 = {' + '.join(str(x) for x in s)} = {sum(s)}",
        f"中位数 = {ans} = {ans}",
    ]
    return ins, lines, ans


_reg("median_find", median_find)


def decimal_to_fraction_diff(rng):
    ab = rng.randint(10, 99)
    g = gcd(ab, 100)
    p = ab // g
    q = 100 // g
    ans = q - p
    ins = rng.choice([
        f"把小数0.{ab}化成最简分数后，分母比分子多多少？",
        f"0.{ab}化成最简分数是几分之几？分母与分子的差是多少？",
        f"将0.{ab}化为最简分数，分母比分子大多少？",
        f"小数0.{ab}化成最简分数后，分子与分母的差是多少？",
    ])
    lines = [
        f"原小数 = {ab} ÷ 100 = {ab}/100",
        f"分子分母同除以 = {g} = {g}",
        f"{ab} ÷ {g} = {p}",
        f"100 ÷ {g} = {q}",
        f"{q} - {p} = {ans}",
    ]
    return ins, lines, ans


_reg("decimal_to_fraction_diff", decimal_to_fraction_diff)


def gcd_lcm_pair(rng):
    while True:
        g = rng.randint(2, 6)
        p = rng.randint(2, 6)
        q = rng.randint(2, 7)
        if p != q and gcd(p, q) == 1:
            break
    a = g * p
    l = g * p * q
    ans = g * q
    ins = rng.choice([
        f"甲、乙两数的最大公因数是{g}，最小公倍数是{l}，甲数是{a}，乙数是多少？",
        f"两个数的最大公因数是{g}，最小公倍数是{l}，其中一个数是{a}，另一个数是多少？",
        f"已知甲数={a}，甲、乙两数的最大公因数是{g}，最小公倍数是{l}，乙数是几？",
        f"甲、乙两个数，最大公因数是{g}，最小公倍数是{l}，如果甲数是{a}，乙数是多少？",
    ])
    lines = [
        f"最大公因数 = {g} = {g}",
        f"最小公倍数 = {l} = {l}",
        f"甲数 = {a} = {a}",
        f"{l} ÷ {a} = {q}",
        f"{g} × {q} = {ans}",
    ]
    return ins, lines, ans


_reg("gcd_lcm_pair", gcd_lcm_pair)


def fold_angle(rng):
    a = rng.randint(10, 35) * 2
    ans = (180 - a) // 2
    ins = rng.choice([
        f"把一张长方形纸的一角折起，已知∠1={a}°，求∠2的度数。",
        f"一张长方形纸折叠后（∠1与∠2在折痕两侧），量得∠1={a}°，∠2是多少度？",
        f"将长方形纸的一角沿折痕折起，如果∠1={a}°，那么∠2等于多少度？",
        f"长方形纸折叠一角，已知∠1为{a}度，∠2为多少度？",
    ])
    lines = [
        f"平角 = 180 = 180度",
        f"180 - {a} = {180 - a}度",
        f"({180 - a}) ÷ 2 = {ans}度",
    ]
    return ins, lines, ans


_reg("fold_angle", fold_angle)


_COIN_OUTCOMES = [
    ("两次都是正面朝上", 1),
    ("两次都是反面朝上", 1),
    ("恰好一次正面朝上", 2),
    ("至少有一次正面朝上", 3),
    ("两次朝上的面相同", 2),
    ("第一次正面朝上", 2),
]


def coin_two_toss(rng):
    desc, c = rng.choice(_COIN_OUTCOMES)
    ans = Fraction(c, 4)
    ins = rng.choice([
        f"抛一枚均匀的硬币，连续抛2次，{desc}的可能性是几分之几？",
        f"一枚硬币抛2次，{desc}的可能性是多少？",
        f"连续抛一枚均匀硬币2次，{desc}的可能性为几分之几？",
        f"把一枚硬币抛2次，{desc}的可能性是多少？",
        f"掷一枚均匀硬币2次，{desc}的可能性是几分之几？",
        f"一枚硬币连续抛2次，{desc}的可能性是多少？",
    ])
    lines = [
        f"总情况 = 2 × 2 = 4种",
        f"符合的情况 = {c} = {c}种",
        f"可能性 = {c} ÷ 4 = {num(ans)}",
    ]
    return ins, lines, ans


_reg("coin_two_toss", coin_two_toss)


def coin_combinations(rng):
    N = rng.randint(8, 12)
    counts = []
    for a5 in range(N // 5 + 1):
        rest = N - 5 * a5
        counts.append(rest // 2 + 1)
    ans = sum(counts)
    ins = rng.choice([
        f"用1元、2元、5元的纸币凑成{N}元（每种纸币都可以不用），一共有多少种不同的凑法？",
        f"现有1元、2元、5元纸币若干张，要凑出{N}元，共有多少种不同的凑法？",
        f"用面值1元、2元、5元的人民币凑成{N}元，有多少种不同的凑法？",
        f"凑{N}元钱，只用1元、2元、5元纸币（张数不限），共有多少种凑法？",
        f"用1元、2元、5元三种纸币凑出{N}元，一共有多少种凑法？",
        f"小明有1元、2元、5元纸币各若干张，他要凑出{N}元，有多少种不同的凑法？",
    ])
    lines = [
        f"总金额 = {N} = {N}元",
        f"2元面值 = 2 = 2元",
        f"5元面值 = 5 = 5元",
    ]
    for i, c in enumerate(counts):
        lines.append(f"5元用{i}张时 = {c} = {c}种")
    lhs = " + ".join(str(c) for c in counts)
    lines.append(f"{lhs} = {ans}种")
    return ins, lines, ans


_reg("coin_combinations", coin_combinations)


def cylinder_carve_cone(rng):
    r = rng.randint(3, 10)
    h = rng.randint(6, 15)
    V = Fraction(314, 100) * r * r * h
    ans = V * Fraction(2, 3)
    ins = rng.choice([
        f"把一个底面半径{r}厘米、高{h}厘米的圆柱形木料削成一个最大的圆锥，削去部分的体积是多少立方厘米（π取3.14）？",
        f"圆柱形木料底面半径{r}厘米、高{h}厘米，把它削成最大的圆锥，削去部分的体积是多少立方厘米（π取3.14）？",
        f"一个圆柱底面半径{r}厘米、高{h}厘米，削成一个与它等底等高的圆锥，削去的体积是多少立方厘米（π取3.14）？",
        f"把底面半径{r}厘米、高{h}厘米的圆柱削成最大的圆锥，削掉部分的体积是多少立方厘米（π取3.14）？",
    ])
    lines = [
        f"3.14 × {r} × {r} × {h} = {_d(V)}立方厘米",
        f"份数差 = 3 - 1 = 2份",
        f"{_d(V)} × 2 ÷ 3 = {_d(ans)}立方厘米",
    ]
    return ins, lines, ans


_reg("cylinder_carve_cone", cylinder_carve_cone)
