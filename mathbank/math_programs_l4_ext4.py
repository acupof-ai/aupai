#!/usr/bin/env python3
"""L4 ext4: trains, currents, clocks, number theory, geometry, mixtures — 62 families.

Every program: fn(rng) -> (instruction, lines, ans). Lines solve FORWARD from
givens to the asked value (no x variable). All exact arithmetic via Fraction.
Every equation line is chained: 中文标签 = 表达式 = 值[单位].
"""
import math
import random
from collections import Counter
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


# 1. 火车过桥时间
def train_bridge_time(rng):
    L = rng.randint(100, 400)
    B = rng.randint(300, 1200)
    v = rng.randint(10, 40)
    t = Fraction(L + B, v)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一列火车长{L}米，以每秒{v}米的速度通过一座长{B}米的大桥。从车头上桥到车尾离桥，一共需要多少秒？",
        f"一列火车全长{L}米，每秒行驶{v}米，要通过一座{B}米长的桥。{name}想知道从车头开上桥到车尾离开桥需要多少秒，请你帮他算一算。",
        f"铁路桥上，一列长{L}米的火车以每秒{v}米的速度前进，桥长{B}米。这列火车完全通过这座桥需要多少秒？",
        f"一列火车长{L}米，每秒行{v}米，全车通过长{B}米的隧道。从车头进入到车尾驶出，要用多少秒？",
    ])
    lines = [
        f"火车行驶的总路程 = {L} + {B} = {L + B}米",
        f"通过大桥的时间 = {L + B} ÷ {v} = {num(t)}秒",
    ]
    return ins, lines, t


_reg("train_bridge_time", train_bridge_time)


# 2. 火车过桥求速度
def train_bridge_speed_reverse(rng):
    L = rng.randint(100, 400)
    B = rng.randint(300, 1200)
    t = rng.randint(10, 40)
    v = Fraction(L + B, t)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一列火车长{L}米，通过一座长{B}米的大桥，从车头上桥到车尾离桥正好用了{t}秒。这列火车每秒行驶多少米？",
        f"一列全长{L}米的火车完全通过一座{B}米长的桥用了{t}秒。{name}想知道火车每秒行多少米，请你帮他算一算。",
        f"一列火车长{L}米，全车通过长{B}米的隧道用了{t}秒。这列火车平均每秒行驶多少米？",
        f"铁路旁，一列长{L}米的火车驶过一座长{B}米的桥，从前轮上桥到后轮离桥共{t}秒。火车的速度是每秒多少米？",
    ])
    lines = [
        f"火车行驶的总路程 = {L} + {B} = {L + B}米",
        f"火车的速度 = {L + B} ÷ {t} = {num(v)}米/秒",
    ]
    return ins, lines, v


_reg("train_bridge_speed_reverse", train_bridge_speed_reverse)


# 3. 两列火车相向错车
def trains_meet_pass(rng):
    a = rng.randint(100, 300)
    b = rng.randint(100, 300)
    v1 = rng.randint(15, 40)
    v2 = rng.randint(15, 40)
    t = Fraction(a + b, v1 + v2)
    ins = rng.choice([
        f"两列火车相向而行，甲车长{a}米，每秒行{v1}米；乙车长{b}米，每秒行{v2}米。从车头相遇到车尾离开，一共需要多少秒？",
        f"一列长{a}米的火车以每秒{v1}米的速度迎面开来，另一列长{b}米的火车每秒行{v2}米。{rng.choice(NAMES)}想知道两车从车头相遇到车尾相离要用多少秒，请你帮他算一算。",
        f"甲、乙两列火车在双线铁路上相向行驶，甲车长{a}米、速度每秒{v1}米，乙车长{b}米、速度每秒{v2}米。两车错车而过需要多少秒？",
        f"两列火车迎面相遇，第一列长{a}米，每秒行{v1}米；第二列长{b}米，每秒行{v2}米。从车头遇到车尾分开，共需多少秒？",
    ])
    lines = [
        f"两车的长度和 = {a} + {b} = {a + b}米",
        f"两车的速度和 = {v1} + {v2} = {v1 + v2}米/秒",
        f"错车需要的时间 = {a + b} ÷ {v1 + v2} = {num(t)}秒",
    ]
    return ins, lines, t


_reg("trains_meet_pass", trains_meet_pass)


# 4. 快车超过慢车
def trains_overtake(rng):
    a = rng.randint(100, 300)
    b = rng.randint(100, 300)
    v2 = rng.randint(10, 25)
    v1 = v2 + rng.randint(5, 20)
    t = Fraction(a + b, v1 - v2)
    ins = rng.choice([
        f"一列快车长{a}米，每秒行{v1}米；一列慢车长{b}米，每秒行{v2}米。两车同向行驶，快车从车头追上慢车到车尾超过慢车，需要多少秒？",
        f"铁路上，一列长{a}米的快车以每秒{v1}米的速度追赶前方一列长{b}米、每秒行{v2}米的慢车。{rng.choice(NAMES)}想知道快车完全超过慢车要多少秒，请你帮他算一算。",
        f"快车长{a}米，每秒行驶{v1}米；慢车长{b}米，每秒行驶{v2}米。两车同向而行，从快车车头追上慢车车尾，到快车车尾离开慢车车头，共需多少秒？",
        f"一列快车和一列慢车同向行驶，快车长{a}米、速度每秒{v1}米，慢车长{b}米、速度每秒{v2}米。快车超过慢车需要多少秒？",
    ])
    lines = [
        f"两车的长度和 = {a} + {b} = {a + b}米",
        f"两车的速度差 = {v1} - {v2} = {v1 - v2}米/秒",
        f"超车需要的时间 = {a + b} ÷ {v1 - v2} = {num(t)}秒",
    ]
    return ins, lines, t


_reg("trains_overtake", trains_overtake)


# 5. 顺水逆水行船时间
def boat_current_down_up(rng):
    v = rng.randint(20, 50)
    u = rng.randint(2, v // 3)
    s = rng.randint(60, 300)
    for _ in range(50):
        if s % (v + u) == 0 or s % (v - u) == 0:
            break
        s = rng.randint(60, 300)
    ask = rng.choice(["顺水", "逆水"])
    if ask == "顺水":
        t = Fraction(s, v + u)
        lines = [
            f"顺水的速度 = {v} + {u} = {v + u}千米/时",
            f"顺水行驶的时间 = {s} ÷ {v + u} = {num(t)}时",
        ]
    else:
        t = Fraction(s, v - u)
        lines = [
            f"逆水的速度 = {v} - {u} = {v - u}千米/时",
            f"逆水行驶的时间 = {s} ÷ {v - u} = {num(t)}时",
        ]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一艘船在静水中的速度是每小时{v}千米，水流速度是每小时{u}千米。这艘船在相距{s}千米的两个码头之间{ask}航行，需要多少小时？",
        f"一艘轮船静水速度为每小时{v}千米，河水每小时流{u}千米。两码头相距{s}千米，{name}想知道轮船{ask}航行需要几小时，请你帮他算一算。",
        f"船在静水中每小时行{v}千米，水流速度为每小时{u}千米。甲、乙两港相距{s}千米，这艘船{ask}航行要多少小时？",
        f"一艘船从上游码头到下游码头，静水速度每小时{v}千米，水速每小时{u}千米，两码头相距{s}千米。{ask}行完全程需要多少小时？",
    ])
    return ins, lines, t


_reg("boat_current_down_up", boat_current_down_up)


# 6. 顺水逆水速度求静水速度与水速
def boat_current_speeds(rng):
    u = rng.randint(2, 10)
    v = rng.randint(15, 45)
    a = v + u
    b = v - u
    ask = rng.choice(["静水", "水流"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一艘船顺水航行每小时行{a}千米，逆水航行每小时行{b}千米。这艘船在静水中的速度和水流速度各是多少？{ask}速度是每小时多少千米？",
        f"一艘轮船在顺水中每小时行{a}千米，在逆水中每小时行{b}千米。{name}想知道船的{ask}速度是每小时多少千米，请你帮他算一算。",
        f"甲、乙两港间，船顺水每小时行{a}千米，逆水每小时行{b}千米。这条河的{ask}速度是每小时多少千米？",
        f"一艘船顺流而下每小时行{a}千米，逆流而上每小时行{b}千米。船的{ask}速度是每小时多少千米？",
    ])
    if ask == "静水":
        lines = [
            f"顺水与逆水的速度和 = {a} + {b} = {a + b}千米/时",
            f"静水速度 = {a + b} ÷ 2 = {v}千米/时",
        ]
        return ins, lines, v
    lines = [
        f"顺水与逆水的速度差 = {a} - {b} = {a - b}千米/时",
        f"水流速度 = {a - b} ÷ 2 = {u}千米/时",
    ]
    return ins, lines, u


_reg("boat_current_speeds", boat_current_speeds)


# 7. 水流往返时间
def boat_round_trip_current(rng):
    u = d = t1 = t2 = v = s = None
    for _ in range(80):
        u = rng.randint(2, 10)
        d = rng.randint(2, 10)
        t1 = rng.randint(2, 8)
        if (2 * u * t1) % d == 0:
            v = u + d
            t2 = t1 + (2 * u * t1) // d
            s = (v + u) * t1
            break
    else:
        u, d, t1, v, t2, s = 5, 5, 3, 10, 6, 45
    total = t1 + t2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"两个码头相距{s}千米，一艘船在静水中每小时行{v}千米，水流每小时{u}千米。这艘船在两个码头之间往返一次，一共需要多少小时？",
        f"甲、乙两港相距{s}千米，轮船静水速度每小时{v}千米，水速每小时{u}千米。{name}乘船从甲港到乙港再返回甲港，一共要用多少小时？",
        f"一艘船静水速度为每小时{v}千米，河水每小时流{u}千米，两码头相距{s}千米。这艘船往返一次需要多少小时？",
        f"两个港口相距{s}千米，船在静水中每小时行{v}千米，水流速度每小时{u}千米。船在两港间往返一次共需多少小时？",
    ])
    lines = [
        f"顺水的速度 = {v} + {u} = {v + u}千米/时",
        f"顺水航行的时间 = {s} ÷ {v + u} = {t1}时",
        f"逆水的速度 = {v} - {u} = {v - u}千米/时",
        f"逆水航行的时间 = {s} ÷ {v - u} = {t2}时",
        f"往返一共的时间 = {t1} + {t2} = {total}时",
    ]
    return ins, lines, total


_reg("boat_round_trip_current", boat_round_trip_current)


# 8. 敲钟问题
def clock_striking(rng):
    k = rng.randint(2, 6)
    p = rng.randint(2, 6)
    t = (k - 1) * p
    m = rng.randint(k + 2, 12)
    total = (m - 1) * p
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一座大钟，{k}点整敲{k}下，用了{t}秒。照这样计算，{m}点整敲{m}下，需要多少秒？",
        f"钟楼的大钟{k}点敲{k}下，{t}秒敲完。{name}想知道{m}点敲{m}下要用多少秒，请你帮他算一算。",
        f"广场上的大钟，{k}点时敲{k}下用了{t}秒。那么{m}点时敲{m}下，一共需要多少秒？",
        f"一座报时钟，{k}点整敲{k}下，前后共用{t}秒。照这样的速度，{m}点整敲{m}下需要多少秒？",
    ])
    lines = [
        f"敲{k}下的间隔数 = {k} - 1 = {k - 1}个",
        f"每个间隔的时间 = {t} ÷ {k - 1} = {p}秒",
        f"敲{m}下的间隔数 = {m} - 1 = {m - 1}个",
        f"敲{m}下的时间 = {m - 1} × {p} = {total}秒",
    ]
    return ins, lines, total


_reg("clock_striking", clock_striking)


# 9. 三人合做工程
def work_three_together(rng):
    a = rng.randint(4, 15)
    b = rng.randint(4, 15)
    c = rng.randint(4, 15)
    ra, rb, rc = Fraction(1, a), Fraction(1, b), Fraction(1, c)
    rate = ra + rb + rc
    t = Fraction(1, rate)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程，甲队单独做{a}天完成，乙队单独做{b}天完成，丙队单独做{c}天完成。三队合做，多少天可以完成？",
        f"加工一批零件，师傅单独做{a}天完成，徒弟单独做{b}天完成，{name}单独做{c}天完成。三人合做需要多少天完成？",
        f"修一条路，甲队独做{a}天完工，乙队独做{b}天完工，丙队独做{c}天完工。三队一起修，多少天能完工？",
        f"一项工程，甲{a}天完成，乙{b}天完成，丙{c}天完成。三人合作，需要几天完成这项工程？",
    ])
    lines = [
        f"甲队每天完成量 = 1 ÷ {a} = {num(ra)}",
        f"乙队每天完成量 = 1 ÷ {b} = {num(rb)}",
        f"丙队每天完成量 = 1 ÷ {c} = {num(rc)}",
        f"三队合做每天完成量 = {num(ra)} + {num(rb)} + {num(rc)} = {num(rate)}",
        f"三队合做的天数 = 1 ÷ ({num(rate)}) = {num(t)}天",
    ]
    return ins, lines, t


_reg("work_three_together", work_three_together)


# 10. 三队合做求一队单独时间
def work_three_reverse(rng):
    t = a = b = c = None
    for _ in range(80):
        t = rng.randint(2, 4)
        a = rng.randint(6, 15)
        b = rng.randint(6, 15)
        rc = Fraction(1, t) - Fraction(1, a) - Fraction(1, b)
        if rc > 0 and rc.numerator == 1 and rc.denominator <= 20:
            c = rc.denominator
            break
    else:
        t, a, b, c = 2, 6, 6, 6
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程，甲队单独做{a}天完成，乙队单独做{b}天完成。甲、乙、丙三队合做{t}天完成。丙队单独做需要多少天完成？",
        f"加工一批零件，师傅{a}天完成，徒弟{b}天完成，师徒和{name}三人合做{t}天完成。{name}单独做需要多少天？",
        f"一项工程，甲独做{a}天完成，乙独做{b}天完成，甲、乙、丙三人合做{t}天完成。丙单独做需要多少天？",
        f"修一条路，甲队{a}天完工，乙队{b}天完工，三队合修{t}天完工。丙队单独修需要多少天完工？",
    ])
    lines = [
        f"三队合做每天完成量 = 1 ÷ {t} = {num(Fraction(1, t))}",
        f"甲队每天完成量 = 1 ÷ {a} = {num(Fraction(1, a))}",
        f"乙队每天完成量 = 1 ÷ {b} = {num(Fraction(1, b))}",
        f"丙队每天完成量 = {num(Fraction(1, t))} - {num(Fraction(1, a))} - {num(Fraction(1, b))} = {num(Fraction(1, c))}",
        f"丙队单独做的天数 = 1 ÷ ({num(Fraction(1, c))}) = {c}天",
    ]
    return ins, lines, c


_reg("work_three_reverse", work_three_reverse)


# 11. 按工作量分配工资
def wages_by_work(rng):
    a = rng.randint(6, 15)
    b = rng.randint(6, 15)
    k = rng.randint(10, 50)
    w = (a + b) * k
    jia = b * k
    yi = a * k
    who = rng.choice(["甲", "乙"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"加工一批零件，甲单独做{a}天完成，乙单独做{b}天完成。两人合做完成后共得工资{w}元，按完成的工作量分配，{who}应得多少元？",
        f"一项工程，甲队{a}天完成，乙队{b}天完成，两队合做完成后共得工资{w}元。{name}想知道按工作量分配，{who}队应得多少元，请你帮他算一算。",
        f"甲、乙两人合做一项工程，甲单独做{a}天完成，乙单独做{b}天完成，共得工资{w}元。按工作量分配，{who}应得多少元？",
        f"师徒合做一批零件，师傅{a}天完成，徒弟{b}天完成，完工后共得工资{w}元。按工作量分配，{who}应得多少元？",
    ])
    ra, rb = Fraction(1, a), Fraction(1, b)
    if who == "甲":
        lines = [
            f"甲每天完成量 = 1 ÷ {a} = {num(ra)}",
            f"乙每天完成量 = 1 ÷ {b} = {num(rb)}",
            f"两人合做每天完成量 = {num(ra)} + {num(rb)} = {num(ra + rb)}",
            f"甲应得的工资 = {w} × {num(ra)} ÷ ({num(ra + rb)}) = {jia}元",
        ]
        return ins, lines, jia
    lines = [
        f"甲每天完成量 = 1 ÷ {a} = {num(ra)}",
        f"乙每天完成量 = 1 ÷ {b} = {num(rb)}",
        f"两人合做每天完成量 = {num(ra)} + {num(rb)} = {num(ra + rb)}",
        f"乙应得的工资 = {w} × {num(rb)} ÷ ({num(ra + rb)}) = {yi}元",
    ]
    return ins, lines, yi


_reg("wages_by_work", wages_by_work)


# 12. 蜡烛燃烧倍数
def candles_burn(rng):
    if rng.random() < 0.5:
        a = rng.randint(3, 8)
        b = rng.randint(3, 2 * a - 1)
        t = Fraction(a * b, 2 * a - b)
        name = rng.choice(NAMES)
        ins = rng.choice([
            f"两根同样长的蜡烛，粗蜡烛全部燃完要{a}小时，细蜡烛全部燃完要{b}小时。同时点燃后，多少小时细蜡烛剩下的长度正好是粗蜡烛剩下的一半？",
            f"两根一样长的蜡烛，粗的{a}小时燃尽，细的{b}小时燃尽。{name}同时点燃两根蜡烛，多少小时后细蜡烛剩下的长度是粗蜡烛剩下的一半？",
            f"两根蜡烛长度相同，粗蜡烛可烧{a}小时，细蜡烛可烧{b}小时。同时点燃，多少小时后细蜡烛剩下的长度是粗蜡烛的一半？",
            f"停电了，{name}点燃两根同样长的蜡烛，粗的能烧{a}小时，细的能烧{b}小时。多少小时后细蜡烛剩下的长度是粗蜡烛剩下的一半？",
        ])
        lines = [
            f"粗蜡烛时间的2倍 = 2 × {a} = {2 * a}",
            f"时间差 = {2 * a} - {b} = {2 * a - b}",
            f"两根蜡烛可烧时间的积 = {a} × {b} = {a * b}",
            f"点燃的时间 = {a * b} ÷ {2 * a - b} = {num(t)}时",
        ]
        return ins, lines, t
    a = rng.randint(2, 5)
    b = rng.randint(2 * a + 1, 15)
    t = Fraction(a * b, b - 2 * a)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"两根同样长的蜡烛，粗蜡烛全部燃完要{a}小时，细蜡烛全部燃完要{b}小时。同时点燃后，多少小时粗蜡烛剩下的长度正好是细蜡烛剩下的2倍？",
        f"两根一样长的蜡烛，粗的{a}小时燃尽，细的{b}小时燃尽。{name}同时点燃两根蜡烛，多少小时后粗蜡烛剩下的长度是细蜡烛剩下的2倍？",
        f"两根蜡烛长度相同，粗蜡烛可烧{a}小时，细蜡烛可烧{b}小时。同时点燃，多少小时后粗蜡烛剩下的长度是细蜡烛的2倍？",
        f"两根同样长的蜡烛，粗的能烧{a}小时，细的能烧{b}小时。同时点燃，多少小时后粗蜡烛剩下的长度是细蜡烛剩下的2倍？",
    ])
    lines = [
        f"细蜡烛时间的2倍 = 2 × {a} = {2 * a}",
        f"时间差 = {b} - {2 * a} = {b - 2 * a}",
        f"两根蜡烛可烧时间的积 = {a} × {b} = {a * b}",
        f"点燃的时间 = {a * b} ÷ {b - 2 * a} = {num(t)}时",
    ]
    return ins, lines, t


_reg("candles_burn", candles_burn)


# 13. 牛吃草问题
def ox_grazing(rng):
    t1 = t2 = t3 = G = r = None
    for _ in range(80):
        t1 = rng.choice([3, 4, 5, 6])
        t2 = rng.choice([8, 10, 12, 15])
        t3 = rng.choice([4, 5, 6, 8, 10, 12])
        g0 = math.lcm(t1, t2, t3)
        k = rng.randint(1, 3)
        G = g0 * k
        r = rng.randint(8, 20)
        a = G // t1 + r
        b = G // t2 + r
        c = G // t3 + r
        if 15 <= a <= 40 and 15 <= b <= 40 and 15 <= c <= 40:
            break
    else:
        t1, t2, t3, G, r = 4, 12, 6, 120, 15
        a, b, c = 45, 25, 35
    ask = rng.choice(["周数", "牛数"])
    name = rng.choice(NAMES)
    if ask == "周数":
        ins = rng.choice([
            f"牧场上的草每天均匀生长。这片草可供{a}头牛吃{t1}周，或供{b}头牛吃{t2}周。照这样计算，可供{c}头牛吃多少周？",
            f"一片牧场的草匀速生长，{a}头牛{t1}周吃完，{b}头牛{t2}周吃完。{name}想知道{c}头牛几周能吃完，请你帮他算一算。",
            f"牧场长满草，草每天匀速生长。{a}头牛可吃{t1}周，{b}头牛可吃{t2}周。那么{c}头牛可以吃多少周？",
            f"一片匀速生长的草地，{a}头牛{t1}周把草吃完，{b}头牛{t2}周把草吃完。{c}头牛几周能把草吃完？",
        ])
        lines = [
            f"{a}头牛{t1}周吃的草 = {a} × {t1} = {a * t1}份",
            f"{b}头牛{t2}周吃的草 = {b} × {t2} = {b * t2}份",
            f"相差的草量 = {b * t2} - {a * t1} = {b * t2 - a * t1}份",
            f"相差的周数 = {t2} - {t1} = {t2 - t1}周",
            f"每周新长的草 = {b * t2 - a * t1} ÷ {t2 - t1} = {r}份",
            f"牧场原有的草 = {a * t1} - {r} × {t1} = {G}份",
            f"吃新草的牛数 = {c} - {r} = {c - r}头",
            f"可以吃的周数 = {G} ÷ {c - r} = {t3}周",
        ]
        return ins, lines, t3
    ins = rng.choice([
        f"牧场上的草每天均匀生长。这片草可供{a}头牛吃{t1}周，或供{b}头牛吃{t2}周。照这样计算，要在{t3}周内吃完，需要多少头牛？",
        f"一片牧场的草匀速生长，{a}头牛{t1}周吃完，{b}头牛{t2}周吃完。{name}想知道{t3}周吃完需要多少头牛，请你帮他算一算。",
        f"牧场长满草，草每天匀速生长。{a}头牛可吃{t1}周，{b}头牛可吃{t2}周。要在{t3}周内吃完，需要多少头牛？",
        f"一片匀速生长的草地，{a}头牛{t1}周把草吃完，{b}头牛{t2}周把草吃完。多少头牛{t3}周能把草吃完？",
    ])
    lines = [
        f"{a}头牛{t1}周吃的草 = {a} × {t1} = {a * t1}份",
        f"{b}头牛{t2}周吃的草 = {b} × {t2} = {b * t2}份",
        f"相差的草量 = {b * t2} - {a * t1} = {b * t2 - a * t1}份",
        f"相差的周数 = {t2} - {t1} = {t2 - t1}周",
        f"每周新长的草 = {b * t2 - a * t1} ÷ {t2 - t1} = {r}份",
        f"牧场原有的草 = {a * t1} - {r} × {t1} = {G}份",
        f"吃原有草的牛数 = {G} ÷ {t3} = {G // t3}头",
        f"一共需要的牛数 = {G // t3} + {r} = {c}头",
    ]
    return ins, lines, c


_reg("ox_grazing", ox_grazing)


# 14. 细菌翻倍
def bacteria_double(rng):
    a = rng.randint(2, 10)
    b = rng.randint(3, 30)
    k = rng.randint(1, 4)
    c = b * (2 ** k)
    ans = a + k
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一种细菌每小时数量增加1倍。{name}在{a}小时后观察到有{b}个，照这样计算，多少小时后细菌数量达到{c}个？",
        f"一种细菌每小时分裂1次，数量增加1倍。{a}小时后有{b}个，多少小时后有{c}个？",
        f"瓶子里有一种细菌，每小时数量增加1倍，{a}小时后瓶中有{b}个。{name}想知道几小时后瓶中有{c}个，请你帮他算一算。",
        f"一种细菌的数量每小时增加1倍，{a}小时后是{b}个。再过几小时正好是{c}个？",
    ])
    lines = []
    cur = b
    for i in range(k):
        nxt = cur * 2
        lines.append(f"第{a + i + 1}小时后的数量 = {cur} × 2 = {nxt}个")
        cur = nxt
    lines.append(f"一共经过的小时数 = {a} + {k} = {ans}小时")
    return ins, lines, ans


_reg("bacteria_double", bacteria_double)


# 15. 编页码用多少数字
def page_numbering(rng):
    p = rng.randint(15, 99)
    digits = 9 + 2 * (p - 9)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"给一本书编页码，从第1页编到第{p}页，一共需要多少个数字？",
        f"一本故事书共有{p}页，{name}给它编页码，从1编到{p}，一共要用多少个数字？",
        f"一本书有{p}页，页码从1开始依次编排。编完这本书的页码，一共需要多少个数字？",
        f"印刷厂给一本{p}页的书编页码，第1页到第{p}页，一共需要多少个数字？",
    ])
    lines = [
        f"两位数的页数 = {p} - 9 = {p - 9}页",
        f"两位数页码用的数字 = {p - 9} × 2 = {2 * (p - 9)}个",
        f"一共需要的数字 = 9 + {2 * (p - 9)} = {digits}个",
    ]
    return ins, lines, digits


_reg("page_numbering", page_numbering)


# 16. 数字个数求页数
def page_numbering_reverse(rng):
    p = rng.randint(15, 99)
    digits = 9 + 2 * (p - 9)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"给一本书编页码，从第1页开始编，一共用了{digits}个数字。这本书共有多少页？",
        f"{name}给一本故事书编页码，从1开始依次编排，一共用了{digits}个数字。这本书有多少页？",
        f"一本书的页码从1编起，编完共使用了{digits}个数字。这本书一共有多少页？",
        f"印刷厂给一本书编页码，一共用了{digits}个数字。这本书共有多少页？",
    ])
    lines = [
        f"两位数页码用的数字 = {digits} - 9 = {digits - 9}个",
        f"两位数的页数 = {digits - 9} ÷ 2 = {p - 9}页",
        f"这本书的页数 = {p - 9} + 9 = {p}页",
    ]
    return ins, lines, p


_reg("page_numbering_reverse", page_numbering_reverse)


# 17. 握手次数
def handshakes(rng):
    n = rng.randint(6, 30)
    ans = n * (n - 1) // 2
    scene = rng.choice([
        f"一次聚会上有{n}人，每两人之间握一次手",
        f"会议结束后，{n}名代表两两握手告别",
        f"毕业晚会上，{n}个同学每两人合影一次",
        f"象棋比赛中，{n}名选手进行单循环赛（每两人赛一场）",
    ])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{scene}，一共要握多少次手？",
        f"{scene}。{name}想知道一共要进行多少次，请你帮他算一算。",
        f"{scene}，总次数是多少？",
        f"{scene}。请你算一算一共多少次。",
    ])
    lines = [
        f"每人握手的次数 = {n} - 1 = {n - 1}次",
        f"所有人握手的次数 = {n} × {n - 1} = {n * (n - 1)}次",
        f"实际握手的次数 = {n * (n - 1)} ÷ 2 = {ans}次",
    ]
    return ins, lines, ans


_reg("handshakes", handshakes)


# 18. 互送礼物
def gifts_pair(rng):
    n = rng.randint(6, 30)
    ans = n * (n - 1)
    scene = rng.choice([
        f"新年到了，{n}个同学每两人之间互送一张贺卡",
        f"圣诞节前，{n}个小朋友每两人之间互送一件礼物",
        f"毕业时，{n}名同学每两人之间互写一封告别信",
        f"春节期间，{n}个亲戚每两人之间互发一条拜年短信",
    ])
    name = rng.choice(NAMES)
    if "贺卡" in scene:
        unit = "张"
        q = rng.choice([
            f"{scene}，一共要准备多少张贺卡？",
            f"{scene}。{name}想知道一共要准备多少张贺卡，请你帮他算一算。",
            f"{scene}，一共需要多少张贺卡？",
            f"{scene}。请你算一算一共要准备多少张贺卡。",
        ])
    elif "短信" in scene:
        unit = "条"
        q = rng.choice([
            f"{scene}，一共要发多少条短信？",
            f"{scene}。{name}想知道一共要发多少条短信，请你帮他算一算。",
            f"{scene}，一共需要多少条短信？",
            f"{scene}。请你算一算一共要发多少条短信。",
        ])
    elif "信" in scene:
        unit = "封"
        q = rng.choice([
            f"{scene}，一共要写多少封信？",
            f"{scene}。{name}想知道一共要写多少封信，请你帮他算一算。",
            f"{scene}，一共需要多少封信？",
            f"{scene}。请你算一算一共要写多少封信。",
        ])
    else:
        unit = "件"
        q = rng.choice([
            f"{scene}，一共要准备多少件礼物？",
            f"{scene}。{name}想知道一共要准备多少件礼物，请你帮他算一算。",
            f"{scene}，一共需要多少件礼物？",
            f"{scene}。请你算一算一共要准备多少件礼物。",
        ])
    lines = [
        f"每人送出的件数 = {n} - 1 = {n - 1}{unit}",
        f"一共送出的件数 = {n} × {n - 1} = {ans}{unit}",
    ]
    return q, lines, ans


_reg("gifts_pair", gifts_pair)


# 19. 周长与差求面积
def rect_perimeter_diff_area(rng):
    w = rng.randint(5, 25)
    d = rng.randint(2, 20)
    l = w + d
    p = 2 * (l + w)
    area = l * w
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个长方形的周长是{p}米，长比宽多{d}米。这个长方形的面积是多少平方米？",
        f"一块长方形菜地，周长是{p}米，长比宽多{d}米。{name}想知道这块菜地的面积是多少平方米，请你帮他算一算。",
        f"一个长方形操场，周长{p}米，长比宽多{d}米。它的面积是多少平方米？",
        f"用篱笆围一个长方形菜园，篱笆共长{p}米，菜园的长比宽多{d}米。菜园的面积是多少平方米？",
    ])
    lines = [
        f"长与宽的和 = {p} ÷ 2 = {p // 2}米",
        f"宽的2倍 = {p // 2} - {d} = {p // 2 - d}米",
        f"宽 = {p // 2 - d} ÷ 2 = {w}米",
        f"长 = {w} + {d} = {l}米",
        f"面积 = {l} × {w} = {area}平方米",
    ]
    return ins, lines, area


_reg("rect_perimeter_diff_area", rect_perimeter_diff_area)


# 20. 正方体棱长和求体积
def cube_edge_sum_volume(rng):
    e = rng.randint(2, 15)
    s = 12 * e
    vol = e ** 3
    scene = rng.choice([
        f"用一根长{s}厘米的铁丝正好焊成一个正方体框架",
        f"一个正方体的棱长之和是{s}厘米",
        f"用一根{s}厘米长的铁丝做一个正方体框架（接头处忽略不计）",
        f"手工课上，{rng.choice(NAMES)}用一根长{s}厘米的铁丝正好做成一个正方体框架",
    ])
    ins = rng.choice([
        f"{scene}，这个正方体的体积是多少立方厘米？",
        f"{scene}。它的体积是多少立方厘米？",
        f"{scene}，请你算出它的体积。",
        f"{scene}。这个正方体的体积是多少立方厘米？",
    ])
    lines = [
        f"棱长 = {s} ÷ 12 = {e}厘米",
        f"棱长的平方 = {e} × {e} = {e * e}",
        f"体积 = {e * e} × {e} = {vol}立方厘米",
    ]
    return ins, lines, vol


_reg("cube_edge_sum_volume", cube_edge_sum_volume)


# 21. 长方体棱长和求体积
def cuboid_edge_sum_volume(rng):
    a = rng.randint(2, 12)
    b = rng.randint(2, 12)
    c = rng.randint(2, 12)
    s = 4 * (a + b + c)
    vol = a * b * c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个长方体的棱长之和是{s}厘米，长是{a}厘米，宽是{b}厘米。它的体积是多少立方厘米？",
        f"用一根长{s}厘米的铁丝正好焊成一个长方体框架，长{a}厘米、宽{b}厘米。{name}想知道这个长方体的体积是多少立方厘米，请你帮他算一算。",
        f"一个长方体礼盒，棱长总和是{s}厘米，长{a}厘米，宽{b}厘米。它的体积是多少立方厘米？",
        f"一根{s}厘米长的铁丝正好做成一个长方体框架，量得长{a}厘米、宽{b}厘米。这个长方体的体积是多少立方厘米？",
    ])
    lines = [
        f"长宽高的和 = {s} ÷ 4 = {s // 4}厘米",
        f"高 = {s // 4} - {a} - {b} = {c}厘米",
        f"长乘宽的积 = {a} × {b} = {a * b}",
        f"体积 = {a * b} × {c} = {vol}立方厘米",
    ]
    return ins, lines, vol


_reg("cuboid_edge_sum_volume", cuboid_edge_sum_volume)


# 22. 三角形面积与底求高
def triangle_area_reverse(rng):
    a = rng.randint(4, 30)
    h = rng.randint(4, 30)
    if (a * h) % 2 == 1:
        h += 1
    s = a * h // 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个三角形的面积是{s}平方米，底是{a}米。这个三角形的高是多少米？",
        f"一块三角形菜地，面积是{s}平方米，底边长{a}米。{name}想知道这条底边上的高是多少米，请你帮他算一算。",
        f"三角形的面积是{s}平方厘米，底是{a}厘米。这条底边上的高是多少厘米？",
        f"一个三角形花坛，面积{s}平方米，底{a}米。这个底边上的高是多少米？",
    ])
    lines = [
        f"与它等底等高的平行四边形面积 = {s} × 2 = {2 * s}平方米",
        f"三角形的高 = {2 * s} ÷ {a} = {h}米",
    ]
    return ins, lines, h


_reg("triangle_area_reverse", triangle_area_reverse)


# 23. 梯形面积
def trapezoid_area(rng):
    a = b = h = None
    for _ in range(50):
        a = rng.randint(2, 20)
        b = rng.randint(2, 20)
        h = rng.randint(2, 20)
        if (a + b) * h % 2 == 0:
            break
    else:
        a, b, h = 3, 5, 4
    s = (a + b) * h
    ans = s // 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个梯形的上底是{a}米，下底是{b}米，高是{h}米。它的面积是多少平方米？",
        f"一块梯形菜地，上底{a}米、下底{b}米、高{h}米。{name}想知道这块菜地的面积是多少平方米，请你帮他算一算。",
        f"梯形的上底长{a}厘米，下底长{b}厘米，高{h}厘米。它的面积是多少平方厘米？",
        f"一个梯形花坛，上底{a}米，下底{b}米，高{h}米。这个花坛的面积是多少平方米？",
    ])
    lines = [
        f"上底与下底的和 = {a} + {b} = {a + b}米",
        f"上下底和乘高 = {a + b} × {h} = {s}平方米",
        f"梯形的面积 = {s} ÷ 2 = {ans}平方米",
    ]
    return ins, lines, ans


_reg("trapezoid_area", trapezoid_area)


# 24. 梯形面积求底
def trapezoid_base_reverse(rng):
    a = b = h = None
    for _ in range(50):
        a = rng.randint(2, 20)
        h = rng.randint(2, 20)
        b = rng.randint(2, 20)
        if (a + b) * h % 2 == 0:
            break
    else:
        a, b, h = 3, 5, 4
    s = (a + b) * h // 2
    who = rng.choice(["上底", "下底"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个梯形的面积是{s}平方米，高是{h}米，{who}是{a if who == '下底' else b}米。它的{'下底' if who == '上底' else '上底'}是多少米？",
        f"一块梯形菜地，面积{s}平方米，高{h}米，{who}{a if who == '下底' else b}米。{name}想知道另一底是多少米，请你帮他算一算。",
        f"梯形的面积是{s}平方厘米，高{h}厘米，{who}{a if who == '下底' else b}厘米。另一底是多少厘米？",
        f"一个梯形花坛面积{s}平方米，高{h}米，{who}{a if who == '下底' else b}米。另一底长多少米？",
    ])
    known = a if who == "下底" else b
    unknown = b if who == "下底" else a
    lines = [
        f"上下底的和 = {s} × 2 ÷ {h} = {2 * s // h}米",
        f"另一底 = {2 * s // h} - {known} = {unknown}米",
    ]
    return ins, lines, unknown


_reg("trapezoid_base_reverse", trapezoid_base_reverse)


# 25. 三角形内角
def triangle_angles(rng):
    mode = rng.choice(["两角", "等腰顶角", "等腰底角"])
    name = rng.choice(NAMES)
    if mode == "两角":
        a = rng.randint(30, 90)
        b = rng.randint(30, 90)
        for _ in range(50):
            if a + b < 170:
                break
            b = rng.randint(30, 90)
        else:
            a, b = 60, 60
        c = 180 - a - b
        ins = rng.choice([
            f"三角形的两个内角分别是{a}度和{b}度，第三个内角是多少度？",
            f"一个三角形中，两个角分别是{a}度和{b}度。{name}想知道第三个角是多少度，请你帮他算一算。",
            f"三角形的两个内角为{a}度和{b}度，另一个内角是多少度？",
            f"已知三角形两个内角分别是{a}度、{b}度，求第三个内角的度数。",
        ])
        lines = [
            f"两个内角的和 = {a} + {b} = {a + b}度",
            f"第三个内角 = 180 - {a + b} = {c}度",
        ]
        return ins, lines, c
    if mode == "等腰顶角":
        a = rng.randint(10, 60) * 2
        base = (180 - a) // 2
        ins = rng.choice([
            f"一个等腰三角形的顶角是{a}度，它的一个底角是多少度？",
            f"等腰三角形的顶角为{a}度。{name}想知道它的底角是多少度，请你帮他算一算。",
            f"一个等腰三角形，顶角{a}度，底角是多少度？",
            f"等腰三角形的顶角是{a}度，每个底角是多少度？",
        ])
        lines = [
            f"两个底角的和 = 180 - {a} = {180 - a}度",
            f"一个底角 = {180 - a} ÷ 2 = {base}度",
        ]
        return ins, lines, base
    a = rng.randint(30, 75)
    vertex = 180 - 2 * a
    ins = rng.choice([
        f"一个等腰三角形的一个底角是{a}度，它的顶角是多少度？",
        f"等腰三角形的一个底角为{a}度。{name}想知道它的顶角是多少度，请你帮他算一算。",
        f"一个等腰三角形，底角{a}度，顶角是多少度？",
        f"等腰三角形的每个底角是{a}度，顶角是多少度？",
    ])
    lines = [
        f"两个底角的和 = {a} × 2 = {2 * a}度",
        f"顶角 = 180 - {2 * a} = {vertex}度",
    ]
    return ins, lines, vertex


_reg("triangle_angles", triangle_angles)


# 26. 补角与余角的倍数
def angle_complement_multiple(rng):
    k = rng.randint(3, 8)
    c = rng.choice([0, 10, 20, 30, 40])
    x = Fraction(90 * k + c - 180, k - 1)
    name = rng.choice(NAMES)
    if c == 0:
        ins = rng.choice([
            f"一个角的补角正好是它的余角的{k}倍，这个角是多少度？",
            f"数学课上老师出了一道题：一个角的补角是它余角的{k}倍。{name}没算出来，请你帮他算一算这个角是多少度。",
            f"一个角的补角等于它的余角的{k}倍，求这个角的度数。",
            f"已知一个角的补角是它的余角的{k}倍，这个角是多少度？",
        ])
    else:
        ins = rng.choice([
            f"一个角的补角比它的余角的{k}倍还多{c}度，这个角是多少度？",
            f"数学课上老师出了一道题：一个角的补角比它余角的{k}倍多{c}度。{name}没算出来，请你帮他算一算这个角是多少度。",
            f"一个角的补角等于它的余角的{k}倍加{c}度，求这个角的度数。",
            f"已知一个角的补角比它的余角的{k}倍多{c}度，这个角是多少度？",
        ])
    lines = [
        f"余角的{k}倍 = 90 × {k} = {90 * k}度",
        f"补角与余角倍数的差 = {90 * k} + {c} - 180 = {90 * k + c - 180}度",
        f"倍数差 = {k} - 1 = {k - 1}",
        f"这个角 = {90 * k + c - 180} ÷ {k - 1} = {num(x)}度",
    ]
    return ins, lines, x


_reg("angle_complement_multiple", angle_complement_multiple)


# 27. 多边形外角
def polygon_exterior(rng):
    name = rng.choice(NAMES)
    if rng.random() < 0.5:
        n = rng.randint(3, 24)
        ans = Fraction(360, n)
        ins = rng.choice([
            f"一个正{n}边形的每个外角是多少度？",
            f"数学课上老师出了一道题：正{n}边形的每个外角是多少度？{name}没算出来，请你帮他算一算。",
            f"正{n}边形的一个外角等于多少度？",
            f"一个正{n}边形，它的每个外角是多少度？",
            f"正{n}边形的外角和是360度，每个外角是多少度？",
            f"已知正{n}边形的外角和为360度，它的每个外角是多少度？",
        ])
        lines = [
            f"每个外角 = 360 ÷ {n} = {num(ans)}度",
        ]
        return ins, lines, ans
    n = rng.choice([3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24])
    a = 360 // n
    ins = rng.choice([
        f"一个正多边形的每个外角是{a}度，这个正多边形是正几边形？",
        f"数学课上老师出了一道题：一个正多边形的每个外角都是{a}度，它是正几边形？{name}没算出来，请你帮他算一算。",
        f"正多边形的每个外角等于{a}度，它有多少条边？",
        f"一个正多边形，每个外角{a}度，这是正几边形？",
        f"正多边形的外角和是360度，每个外角{a}度，它是正几边形？",
        f"已知一个正多边形的外角和为360度，每个外角{a}度，它有多少条边？",
    ])
    lines = [
        f"边数 = 360 ÷ {a} = {n}条",
    ]
    return ins, lines, n


_reg("polygon_exterior", polygon_exterior)


# 28. 因数的个数
_DIV_POOL = [12, 18, 20, 24, 28, 30, 36, 40, 42, 44, 45, 48, 50, 54, 56, 60,
             63, 66, 70, 72, 75, 78, 80, 84, 88, 90, 96, 98, 100, 104, 105,
             108, 110, 112, 120, 126, 132, 135, 140, 150]


def count_divisors(rng):
    n = rng.choice(_DIV_POOL)
    x = n
    factors = []
    p = 2
    while p * p <= x:
        while x % p == 0:
            factors.append(p)
            x //= p
        p += 1 if p == 2 else 2
    if x > 1:
        factors.append(x)
    cnt = Counter(factors)
    lines = []
    cur = n
    for p, e in sorted(cnt.items()):
        for _ in range(e):
            nxt = cur // p
            lines.append(f"分解质因数 = {cur} ÷ {p} = {nxt}")
            cur = nxt
    exps = list(cnt.values())
    prod = 1
    for e in exps:
        prod *= (e + 1)
    terms = " × ".join(f"({e + 1})" for e in exps)
    lines.append(f"因数的个数 = {terms} = {prod}个")
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：{n}的因数一共有多少个？",
        f"{name}在练习册上看到一道题：{n}一共有多少个因数？请你帮他算一算。",
        f"自然数{n}的因数共有多少个？",
        f"你能很快算出{n}有多少个因数吗？",
    ])
    return ins, lines, prod


_reg("count_divisors", count_divisors)


# 29. 三个数的最大公约数
def gcd_three(rng):
    patterns = [(2, 2), (2, 3), (3, 2), (2, 2, 3), (2, 3, 3), (2, 2, 2),
                (3, 3), (2, 5), (3, 5), (2, 2, 5)]
    primes = rng.choice(patterns)
    g = 1
    for p in primes:
        g *= p
    m1 = m2 = m3 = None
    for _ in range(80):
        m1 = rng.randint(2, 9)
        m2 = rng.randint(2, 9)
        m3 = rng.randint(2, 9)
        if (math.gcd(m1, m2) == 1 and math.gcd(m2, m3) == 1
                and math.gcd(m1, m3) == 1):
            break
    else:
        m1, m2, m3 = 2, 3, 5
    a, b, c = g * m1, g * m2, g * m3
    lines = []
    ca, cb, cc = a, b, c
    for p in primes:
        na, nb, nc = ca // p, cb // p, cc // p
        lines.append(f"甲数除以{p} = {ca} ÷ {p} = {na}")
        lines.append(f"乙数除以{p} = {cb} ÷ {p} = {nb}")
        lines.append(f"丙数除以{p} = {cc} ÷ {p} = {nc}")
        ca, cb, cc = na, nb, nc
    prod = " × ".join(str(p) for p in primes)
    lines.append(f"最大公约数 = {prod} = {g}")
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙、丙三个数分别是{a}、{b}、{c}，它们的最大公约数是多少？",
        f"数学课上老师出了一道题：求{a}、{b}、{c}三个数的最大公约数。{name}没算出来，请你帮他算一算。",
        f"甲数是{a}，乙数是{b}，丙数是{c}，这三个数的最大公约数是多少？",
        f"用短除法求{a}、{b}、{c}的最大公约数，结果是多少？",
    ])
    return ins, lines, g


_reg("gcd_three", gcd_three)


# 30. 两个数的最小公倍数
def lcm_two(rng):
    a = rng.randint(4, 30)
    b = rng.randint(4, 30)
    g = math.gcd(a, b)
    l = a * b // g
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两个数分别是{a}和{b}，它们的最小公倍数是多少？",
        f"数学课上老师出了一道题：求{a}和{b}的最小公倍数。{name}没算出来，请你帮他算一算。",
        f"甲数是{a}，乙数是{b}，这两个数的最小公倍数是多少？",
        f"你能很快算出{a}和{b}的最小公倍数吗？",
    ])
    lines = [
        f"两数的乘积 = {a} × {b} = {a * b}",
        f"最小公倍数 = {a * b} ÷ {g} = {l}",
    ]
    return ins, lines, l


_reg("lcm_two", lcm_two)


# 31. 最大公约数、最小公倍数与和
def gcd_lcm_sum(rng):
    m = rng.randint(2, 9)
    n = rng.randint(2, 9)
    for _ in range(50):
        if n != m and math.gcd(m, n) == 1:
            break
        n = rng.randint(2, 9)
    else:
        n = m + 1 if m < 9 else m - 1
    g = rng.randint(2, 9)
    big, small = max(m, n), min(m, n)
    a, b = g * big, g * small
    s, l = a + b, g * m * n
    who = rng.choice(["甲", "乙"])
    ans = a if who == "甲" else b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两个数的最大公约数是{g}，最小公倍数是{l}，两数之和是{s}，且甲数比乙数大。{who}数是多少？",
        f"数学课上老师出了一道题：两个数的最大公约数是{g}，最小公倍数是{l}，它们的和是{s}。{name}没算出来，请你帮他算出{who}数（甲数较大）。",
        f"已知甲、乙两数的最大公约数为{g}，最小公倍数为{l}，两数之和为{s}。{who}数是多少（甲数比乙数大）？",
        f"两个数的最大公约数是{g}，最小公倍数是{l}，和是{s}。{who}数是多少（甲数较大）？",
    ])
    lines = [
        f"两数和除以最大公约数 = {s} ÷ {g} = {s // g}",
        f"最小公倍数除以最大公约数 = {l} ÷ {g} = {l // g}",
        f"两个互质数 = {big} × {small} = {big * small}",
        f"两个互质数的和 = {big} + {small} = {big + small}",
        f"{who}数 = {g} × {big if who == '甲' else small} = {ans}",
    ]
    return ins, lines, ans


_reg("gcd_lcm_sum", gcd_lcm_sum)


# 32. 同余问题
def crt_same_remainder(rng):
    a = rng.randint(3, 9)
    b = rng.randint(a + 1, 12)
    for _ in range(50):
        if math.gcd(a, b) == 1:
            break
        b = rng.randint(a + 1, 12)
    else:
        a, b = 3, 4
    r = rng.randint(1, min(a, b) - 1)
    x = a * b + r
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个数除以{a}余{r}，除以{b}也余{r}，这个数最小是多少？",
        f"数学课上老师出了一道题：某数除以{a}余{r}，除以{b}余{r}。{name}没算出来，请你帮他算出这个数最小是多少。",
        f"一筐苹果，{a}个一数余{r}个，{b}个一数也余{r}个，这筐苹果最少有多少个？",
        f"一个数分别除以{a}和{b}，余数都是{r}，这个数最小是多少？",
    ])
    lines = [
        f"两除数的积 = {a} × {b} = {a * b}",
        f"最小的数 = {a * b} + {r} = {x}",
    ]
    return ins, lines, x


_reg("crt_same_remainder", crt_same_remainder)


# 33. 倍数之和
def sum_multiples_upto(rng):
    m = rng.choice([3, 4, 6, 7, 8, 9])
    k = rng.randint(10, 40)
    N = m * k
    s = m * k * (k + 1) // 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"在1到{N}中，所有{m}的倍数之和是多少？",
        f"数学课上老师出了一道题：求1到{N}中所有{m}的倍数的和。{name}没算出来，请你帮他算一算。",
        f"从1到{N}的自然数中，{m}的倍数一共有多少个？它们的和是多少？",
        f"1到{N}之间所有{m}的倍数相加，和是多少？",
    ])
    lines = [
        f"倍数的个数 = {N} ÷ {m} = {k}个",
        f"首尾配对的和 = {k} + 1 = {k + 1}",
        f"{k}个倍数的和 = {k} × {k + 1} = {k * (k + 1)}",
        f"还原成原数 = {k * (k + 1)} × {m} = {k * (k + 1) * m}",
        f"总和 = {k * (k + 1) * m} ÷ 2 = {s}",
    ]
    return ins, lines, s


_reg("sum_multiples_upto", sum_multiples_upto)


# 34. 分数连乘剩余
def fraction_chain_remaining(rng):
    n = rng.randint(2, 5)
    m = rng.randint(2, 5)
    k = rng.randint(3, 20)
    L = n * m * k
    used1 = L // n
    after1 = L - used1
    used2 = after1 // m
    left = after1 - used2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一根绳子长{L}米，第一次用去全长的1/{n}，第二次用去余下的1/{m}，还剩多少米？",
        f"一根绳子长{L}米，{name}第一次剪去全长的1/{n}，第二次剪去剩下的1/{m}。还剩多少米？",
        f"一根{L}米长的绳子，先用去它的1/{n}，再用去余下的1/{m}，还剩下多少米？",
        f"一根绳子长{L}米，第一次用去1/{n}，第二次用去余下的1/{m}，最后还剩多少米？",
    ])
    lines = [
        f"第一次用去的长度 = {L} ÷ {n} = {used1}米",
        f"第一次剩下的长度 = {L} - {used1} = {after1}米",
        f"第二次用去的长度 = {after1} ÷ {m} = {used2}米",
        f"最后剩下的长度 = {after1} - {used2} = {left}米",
    ]
    return ins, lines, left


_reg("fraction_chain_remaining", fraction_chain_remaining)


# 35. 先涨后降（不同幅度）
def pct_up_then_down_diff(rng):
    a = mid = final = p = q = None
    for _ in range(80):
        k = rng.randint(2, 20)
        a = 100 * k
        p = rng.choice([10, 20, 25, 50])
        q = rng.choice([10, 20, 25, 50])
        mid = a * (100 + p) // 100
        f = Fraction(mid * (100 - q), 100)
        if f.denominator == 1:
            final = f.numerator
            break
    else:
        a, p, q, mid, final = 1000, 10, 20, 1100, 880
    obj = rng.choice(GOODS)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一件{obj}原价{a}元，先涨价{p}%，再降价{q}%，现价是多少元？",
        f"商店里一件{obj}原价{a}元，先把价格上调{p}%，又下调{q}%。{name}想知道现价是多少元，请你帮他算一算。",
        f"一件{obj}的原价是{a}元，先涨价{p}%，再降价{q}%出售。现价多少元？",
        f"一件{obj}原价{a}元，第一周涨价{p}%，第二周在涨价后的价格上降价{q}%。现在的售价是多少元？",
    ])
    lines = [
        f"涨价后的价格 = {a} × ({100 + p}/100) = {mid}元",
        f"降价后的价格 = {mid} × ({100 - q}/100) = {final}元",
    ]
    return ins, lines, final


_reg("pct_up_then_down_diff", pct_up_then_down_diff)


# 36. 买m送1
def buy_m_get_one_free(rng):
    buy = rng.randint(2, 5)
    k = rng.randint(2, 12)
    n = (buy + 1) * k
    a = rng.randint(2, 20)
    pay = buy * k * a
    obj = rng.choice(FOOD + STATIONERY)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"商店做活动，{obj}买{buy}送1。{name}买了{n}个{obj}，每个{a}元，一共要付多少元？",
        f"超市里{obj}买{buy}送1，{name}一共拿了{n}个，每个{a}元。他实际要付多少元？",
        f"文具店促销：{obj}买{buy}送1。{name}买{n}个，每个{a}元，共需付多少元？",
        f"一种{obj}买{buy}送1，{name}买了{n}个，每个{a}元，一共花了多少元？",
    ])
    lines = [
        f"每组的个数 = {buy} + 1 = {buy + 1}个",
        f"组数 = {n} ÷ {buy + 1} = {k}组",
        f"实际付钱的个数 = {k} × {buy} = {buy * k}个",
        f"一共要付的钱 = {buy * k} × {a} = {pay}元",
    ]
    return ins, lines, pay


_reg("buy_m_get_one_free", buy_m_get_one_free)


# 37. 买票方案比较
def ticket_scheme_compare(rng):
    n = rng.randint(2, 10)
    m = rng.randint(2, 10)
    a = rng.randint(20, 60)
    b = rng.randint(10, 40)
    g = rng.randint(15, 45)
    s1 = n * a + m * b
    s2 = (n + m) * g
    diff = abs(s1 - s2)
    place = rng.choice(["动物园", "植物园", "游乐园", "海洋馆", "科技馆"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{place}成人票每张{a}元，儿童票每张{b}元。{n}个成人带{m}个儿童去游玩，如果都买团体票，每张{g}元。买团体票比单独买票省多少元？",
        f"去{place}游玩，成人票{a}元一张，儿童票{b}元一张，团体票{g}元一张。{n}个成人和{m}个儿童一起去，买团体票比各自买票省多少元？",
        f"{name}一家和朋友共{n}个成人、{m}个儿童去{place}，成人票每张{a}元，儿童票每张{b}元，团体票每张{g}元。全部买团体票比单独买票省多少元？",
        f"{place}售票处写着：成人票{a}元，儿童票{b}元，团体票{g}元。{n}个成人、{m}个儿童怎样买票更省钱？买团体票比单独买票省多少元？",
    ])
    lines = [
        f"单独买成人票 = {n} × {a} = {n * a}元",
        f"单独买儿童票 = {m} × {b} = {m * b}元",
        f"单独买票共需 = {n * a} + {m * b} = {s1}元",
        f"总人数 = {n} + {m} = {n + m}人",
        f"买团体票共需 = {n + m} × {g} = {s2}元",
        f"节省的钱 = {max(s1, s2)} - {min(s1, s2)} = {diff}元",
    ]
    return ins, lines, diff


_reg("ticket_scheme_compare", ticket_scheme_compare)


# 38. 混合糖果单价
def candy_mix_price(rng):
    a = rng.randint(10, 50)
    b = rng.randint(10, 50)
    m = rng.randint(2, 6)
    n = rng.randint(2, 6)
    total = a * m + b * n
    weight = m + n
    price = Fraction(total, weight)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"商店把每千克{a}元的奶糖{m}千克和每千克{b}元的水果糖{n}千克混合成什锦糖。混合后的糖果每千克多少元？",
        f"一种什锦糖由每千克{a}元的奶糖{m}千克和每千克{b}元的水果糖{n}千克混合而成。{name}想知道混合后每千克多少元，请你帮他算一算。",
        f"食品店把{m}千克单价{a}元的糖和{n}千克单价{b}元的糖混在一起卖，混合糖每千克应卖多少元？",
        f"每千克{a}元的糖{m}千克与每千克{b}元的糖{n}千克混合，混合后的单价是每千克多少元？",
    ])
    lines = [
        f"奶糖的总价 = {a} × {m} = {a * m}元",
        f"水果糖的总价 = {b} × {n} = {b * n}元",
        f"混合糖的总价 = {a * m} + {b * n} = {total}元",
        f"混合糖的总质量 = {m} + {n} = {weight}千克",
        f"混合后的单价 = {total} ÷ {weight} = {num(price)}元",
    ]
    return ins, lines, price


_reg("candy_mix_price", candy_mix_price)


# 39. 扶梯可见级数
def escalator_visible(rng):
    a = rng.randint(1, 4)
    b = rng.randint(1, 3)
    t = rng.randint(10, 60)
    N = (a + b) * t
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"商场的自动扶梯匀速向上运行。{name}站在扶梯上每秒向上走{a}级，扶梯每秒向上移动{b}级，他用了{t}秒到达楼上。这个扶梯的可见部分共有多少级？",
        f"自动扶梯匀速向上，小明每秒走{a}级，扶梯每秒行{b}级，{t}秒后小明到达楼上。扶梯的可见部分有多少级？",
        f"一部向上的自动扶梯，{name}每秒向上走{a}级台阶，扶梯每秒上升{b}级，他{t}秒到达楼上。扶梯共有多少级可见台阶？",
        f"自动扶梯以每秒{b}级的速度向上运行，{name}在扶梯上以每秒{a}级的速度向上走，{t}秒后到达楼上。扶梯可见部分有多少级？",
    ])
    lines = [
        f"每秒上升的级数 = {a} + {b} = {a + b}级",
        f"扶梯的可见级数 = {a + b} × {t} = {N}级",
    ]
    return ins, lines, N


_reg("escalator_visible", escalator_visible)


# 40. 扶梯速度
def escalator_speed(rng):
    a = rng.randint(1, 4)
    t = rng.randint(10, 60)
    b = rng.randint(1, 3)
    N = (a + b) * t
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"商场的自动扶梯匀速向上运行，可见部分共{N}级。{name}站在扶梯上每秒向上走{a}级，用了{t}秒到达楼上。扶梯每秒向上移动多少级？",
        f"一部自动扶梯的可见部分有{N}级，小明每秒走{a}级，{t}秒到达楼上。扶梯每秒向上移动多少级？",
        f"自动扶梯可见部分共{N}级台阶，{name}每秒向上走{a}级，{t}秒后到达楼上。扶梯每秒上升多少级？",
        f"向上的自动扶梯有{N}级可见，{name}每秒走{a}级，他{t}秒到达楼上。扶梯的速度是每秒多少级？",
    ])
    lines = [
        f"每秒上升的级数 = {N} ÷ {t} = {N // t}级",
        f"扶梯的速度 = {N // t} - {a} = {b}级/秒",
    ]
    return ins, lines, b


_reg("escalator_speed", escalator_speed)


# 41. 狗在两人之间跑
def dog_runs_between(rng):
    v1 = rng.randint(40, 80)
    v2 = rng.randint(40, 80)
    v3 = rng.randint(60, 120)
    s = rng.randint(100, 1000)
    for _ in range(50):
        if s % (v1 + v2) == 0:
            break
        s = rng.randint(100, 1000)
    else:
        v1, v2, s = 50, 50, 600
    t = s // (v1 + v2)
    ans = t * v3
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人同时从相距{s}米的两地相向而行，甲每分钟走{v1}米，乙每分钟走{v2}米。甲带的一只狗每分钟跑{v3}米，狗在两人之间来回跑，直到两人相遇。这只狗一共跑了多少米？",
        f"两地相距{s}米，甲、乙两人同时出发相向而行，甲每分钟行{v1}米，乙每分钟行{v2}米。一只狗以每分钟{v3}米的速度在两人之间往返奔跑，两人相遇时狗一共跑了多少米？",
        f"甲、乙两人从相距{s}米的两地同时相向走来，甲每分钟走{v1}米，乙每分钟走{v2}米。{name}带的狗每分钟跑{v3}米，在两人间不停地跑，两人相遇时狗跑了多少米？",
        f"相距{s}米的两人同时相向而行，速度分别是每分钟{v1}米和{v2}米。一只狗每分钟跑{v3}米，在两人之间来回跑，直到相遇。狗一共跑了多少米？",
    ])
    lines = [
        f"两人的速度和 = {v1} + {v2} = {v1 + v2}米/分",
        f"两人相遇的时间 = {s} ÷ {v1 + v2} = {t}分",
        f"狗跑的路程 = {t} × {v3} = {ans}米",
    ]
    return ins, lines, ans


_reg("dog_runs_between", dog_runs_between)


# 42. 蜗牛爬井
def snail_climb_well(rng):
    a = rng.randint(3, 8)
    b = rng.randint(1, a - 1)
    d = rng.randint(2, 10)
    h = a + d * (a - b)
    days = d + 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一只蜗牛掉进一口深{h}米的井里，它白天向上爬{a}米，晚上滑下{b}米。这只蜗牛多少天能爬出井外？",
        f"井深{h}米，一只蜗牛从井底向上爬，白天爬{a}米，夜里滑下{b}米。{name}想知道蜗牛几天能爬出井，请你帮他算一算。",
        f"一只蜗牛在深{h}米的井底，白天向上爬{a}米，晚上滑下{b}米。它多少天可以爬出井口？",
        f"蜗牛爬井，井深{h}米，白天爬上{a}米，晚上滑下{b}米。这只蜗牛第几天能爬出井外？",
    ])
    lines = [
        f"最后一天前要爬的高度 = {h} - {a} = {h - a}米",
        f"每天实际上升的高度 = {a} - {b} = {a - b}米",
        f"爬完前面部分的天数 = {h - a} ÷ {a - b} = {d}天",
        f"一共需要的天数 = {d} + 1 = {days}天",
    ]
    return ins, lines, days


_reg("snail_climb_well", snail_climb_well)


# 43. 空心方阵
def hollow_square_troops(rng):
    n = rng.randint(6, 20)
    k = rng.randint(2, n // 2)
    ans = 4 * (n - k) * k
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"同学们排成一个{k}层的空心方阵，最外层每边有{n}人。这个方阵一共有多少人？",
        f"运动会上，同学们排成{k}层的空心方阵，最外层每边{n}人。{name}想知道这个方阵共有多少人，请你帮他算一算。",
        f"一个{k}层空心方阵，最外层每边站{n}人，方阵里一共有多少人？",
        f"学生排成{k}层的空心方阵表演，最外层每边{n}人。参加表演的一共有多少人？",
    ])
    lines = [
        f"最外层每边与层数的差 = {n} - {k} = {n - k}人",
        f"一层的人数 = {n - k} × 4 = {4 * (n - k)}人",
        f"方阵的总人数 = {4 * (n - k)} × {k} = {ans}人",
    ]
    return ins, lines, ans


_reg("hollow_square_troops", hollow_square_troops)


# 44. 三人年龄链
def age_three_chain(rng):
    b = rng.randint(2, 8)
    a = rng.randint(2, 8)
    bing = rng.randint(3, 15)
    yi = bing + b
    jia = yi + a
    s = jia + yi + bing
    who = rng.choice(["甲", "乙", "丙"])
    ans = {"甲": jia, "乙": yi, "丙": bing}[who]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙、丙三人的年龄和是{s}岁，甲比乙大{a}岁，乙比丙大{b}岁。{who}今年多少岁？",
        f"甲、乙、丙三人今年的年龄加起来是{s}岁，甲比乙大{a}岁，乙比丙大{b}岁。{name}想知道{who}今年多少岁，请你帮他算一算。",
        f"三个人的年龄和是{s}岁，其中甲比乙大{a}岁，乙比丙大{b}岁。{who}今年多少岁？",
        f"甲、乙、丙三人，甲比乙大{a}岁，乙比丙大{b}岁，三人年龄和是{s}岁。{who}今年多少岁？",
    ])
    lines = [
        f"甲比丙大的岁数 = {a} + {b} = {a + b}岁",
        f"丙年龄的3倍 = {s} - {a + b} - {b} = {s - a - 2 * b}岁",
        f"丙的年龄 = {s - a - 2 * b} ÷ 3 = {bing}岁",
        f"乙的年龄 = {bing} + {b} = {yi}岁",
        f"甲的年龄 = {yi} + {a} = {jia}岁",
    ]
    idx = {"甲": 4, "乙": 3, "丙": 2}[who]
    lines = lines[:idx] + lines[idx + 1:] + [lines[idx]]
    return ins, lines, ans


_reg("age_three_chain", age_three_chain)


# 45. 三数比与差求总和
def ratio_three_diff_total(rng):
    a = rng.randint(3, 8)
    c = rng.randint(2, a - 1)
    b = rng.randint(2, 8)
    m = rng.randint(3, 20)
    d = (a - c) * m
    total = (a + b + c) * m
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙、丙三个数的比是{a}比{b}比{c}，甲数比丙数多{d}。这三个数的和是多少？",
        f"甲、乙、丙三个数的比为{a}:{b}:{c}，已知甲数比丙数大{d}。{name}想知道这三个数的和是多少，请你帮他算一算。",
        f"三个数的比是{a}比{b}比{c}，其中最大数比最小数多{d}。这三个数的和是多少？",
        f"甲、乙、丙三个数，甲与乙的比是{a}比{b}，乙与丙的比是{b}比{c}，甲比丙多{d}。三个数的和是多少？",
    ])
    lines = [
        f"份数差 = {a} - {c} = {a - c}",
        f"每份是多少 = {d} ÷ {a - c} = {m}",
        f"总份数 = {a} + {b} + {c} = {a + b + c}",
        f"三个数的和 = {a + b + c} × {m} = {total}",
    ]
    return ins, lines, total


_reg("ratio_three_diff_total", ratio_three_diff_total)


# 46. 钟表快慢
def clock_gains_time(rng):
    a = m = t = None
    for _ in range(50):
        a = rng.choice([5, 10, 15, 20, 30, 60])
        m = rng.randint(1, 12)
        if (m * 60) % a == 0:
            t = m * 60 // a
            break
    else:
        a, m, t = 10, 2, 12
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一只钟每小时快{a}秒，今天中午对准后，多少小时后这只钟正好快{m}分钟？",
        f"小明家的钟每小时快{a}秒，对准标准时间后，经过多少小时这只钟会快{m}分钟？",
        f"一座钟每小时比标准时间快{a}秒，{name}中午把它对准，几小时后它快了{m}分钟？",
        f"一只走快的钟每小时快{a}秒，对准后，多少小时后它比标准时间快{m}分钟？",
    ])
    lines = [
        f"快的总秒数 = {m} × 60 = {m * 60}秒",
        f"需要的小时数 = {m * 60} ÷ {a} = {t}小时",
    ]
    return ins, lines, t


_reg("clock_gains_time", clock_gains_time)


# 47. 加权成绩
def weighted_score(rng):
    a = rng.randint(60, 99)
    b = rng.randint(60, 99)
    ans = Fraction(a * 3 + b * 7, 10)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"学期成绩由平时成绩和期末成绩组成，平时成绩占3/10，期末成绩占7/10。{name}的平时成绩是{a}分，期末成绩是{b}分，他的学期成绩是多少分？",
        f"某学科规定：平时成绩占3/10，期末成绩占7/10。{name}平时得{a}分，期末得{b}分，学期总评是多少分？",
        f"学期总评中，平时成绩占3/10，期末考试占7/10。{name}平时{a}分、期末{b}分，他的总评成绩是多少分？",
        f"一门课的成绩这样算：平时成绩占3/10，期末成绩占7/10。{name}平时考了{a}分，期末考了{b}分，这门课的成绩是多少分？",
    ])
    lines = [
        f"平时成绩的计入分 = {a} × (3/10) = {num(Fraction(a * 3, 10))}分",
        f"期末成绩的计入分 = {b} × (7/10) = {num(Fraction(b * 7, 10))}分",
        f"学期成绩 = {num(Fraction(a * 3, 10))} + {num(Fraction(b * 7, 10))} = {num(ans)}分",
    ]
    return ins, lines, ans


_reg("weighted_score", weighted_score)


# 48. 含水率
def mushroom_water(rng):
    a = rng.randint(5, 50)
    p = rng.choice([80, 85, 90, 95])
    q = rng.choice([60, 70, 75, 80])
    for _ in range(50):
        if q < p:
            break
        q = rng.choice([60, 70, 75, 80])
    else:
        p, q = 90, 75
    dry = Fraction(a * (100 - p), 100)
    ans = Fraction(dry * 100, 100 - q)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"仓库里有{a}千克蘑菇，含水率是{p}%。晾晒后含水率降到{q}%，晾晒后的蘑菇重多少千克？",
        f"一批蘑菇重{a}千克，含水率为{p}%。经过晾晒，含水率降为{q}%。{name}想知道晾晒后蘑菇的重量，请你帮他算一算。",
        f"新鲜蘑菇{a}千克，含水率{p}%，晒干后含水率是{q}%。晒干后的蘑菇重多少千克？",
        f"有{a}千克含水率{p}%的蘑菇，晾晒后含水率变为{q}%，这时蘑菇重多少千克？",
    ])
    lines = [
        f"干蘑菇的质量 = {a} × ({100 - p}/100) = {num(dry)}千克",
        f"晾晒后干蘑菇占的百分率 = 100 - {q} = {100 - q}",
        f"晾晒后的质量 = {num(dry)} × 100 ÷ {100 - q} = {num(ans)}千克",
    ]
    return ins, lines, ans


_reg("mushroom_water", mushroom_water)


# 49. 两次相遇求距离
def meet_twice_distance(rng):
    a = rng.randint(100, 400)
    b = rng.randint(50, 400)
    if (3 * a + b) % 2 == 1:
        b += 1
    s = (3 * a + b) // 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人同时从A、B两地相向而行，第一次在距A地{a}米处相遇，相遇后继续前进，分别到达B、A后立即返回，第二次在距A地{b}米处相遇。A、B两地相距多少米？",
        f"甲、乙两人从A、B两地同时出发相向而行，第一次相遇点距A地{a}米。相遇后两人继续走，到达对方出发地后立即返回，第二次相遇点距A地{b}米。{name}想知道A、B两地的距离，请你帮他算一算。",
        f"甲、乙两车同时从A、B两站相向开出，第一次在离A站{a}千米处相遇，之后继续前进，到达后立即返回，第二次在离A站{b}千米处相遇。A、B两站相距多少千米？",
        f"两人从A、B两地同时相向而行，第一次相遇时距A地{a}米，到达对面后立即返回，第二次相遇时距A地{b}米。A、B两地相距多少米？",
    ])
    lines = [
        f"甲走的总路程 = {a} × 3 = {3 * a}米",
        f"甲走的总路程与第二次距A地的和 = {3 * a} + {b} = {3 * a + b}米",
        f"A、B两地的距离 = {3 * a + b} ÷ 2 = {s}米",
    ]
    return ins, lines, s


_reg("meet_twice_distance", meet_twice_distance)


# 50. 环形跑道相遇
def circular_track_meet(rng):
    v1 = rng.randint(80, 200)
    v2 = rng.randint(80, 200)
    for _ in range(50):
        if v1 != v2:
            break
        v2 = rng.randint(80, 200)
    else:
        v1, v2 = 100, 120
    direction = rng.choice(["反向", "同向"])
    if direction == "反向":
        L = (v1 + v2) * rng.randint(2, 5)
        t = L // (v1 + v2)
        lines = [
            f"两人的速度和 = {v1} + {v2} = {v1 + v2}米/分",
            f"第一次相遇的时间 = {L} ÷ {v1 + v2} = {t}分",
        ]
    else:
        fast, slow = max(v1, v2), min(v1, v2)
        L = (fast - slow) * rng.randint(4, 12)
        t = L // (fast - slow)
        lines = [
            f"两人的速度差 = {fast} - {slow} = {fast - slow}米/分",
            f"第一次追上的时间 = {L} ÷ {fast - slow} = {t}分",
        ]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人在周长{L}米的环形跑道上同时同地出发，甲每分钟跑{v1}米，乙每分钟跑{v2}米，两人{direction}而行。经过多少分钟两人第一次相遇？",
        f"环形跑道周长{L}米，甲、乙两人同时同地{direction}出发，甲每分钟{v1}米，乙每分钟{v2}米。{name}想知道两人几分钟后第一次相遇，请你帮他算一算。",
        f"甲、乙两人绕周长{L}米的跑道同时同地{direction}跑步，甲每分钟跑{v1}米，乙每分钟跑{v2}米。多少分钟后两人第一次相遇？",
        f"在周长{L}米的环形跑道上，甲、乙两人同时同地出发，{direction}而行，速度分别为每分钟{v1}米和{v2}米。经过多少分钟第一次相遇？",
    ])
    return ins, lines, t


_reg("circular_track_meet", circular_track_meet)


# 51. 起点再次相遇
def laps_start_meet(rng):
    a = rng.randint(2, 12)
    b = rng.randint(2, 12)
    for _ in range(50):
        if b != a and math.gcd(a, b) == 1:
            break
        b = rng.randint(2, 12)
    else:
        a, b = 3, 4
    x = a * b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人在环形跑道上跑步，甲跑一圈要{a}分钟，乙跑一圈要{b}分钟。两人同时同地同向出发，至少多少分钟后两人又在起点相遇？",
        f"甲跑一圈用{a}分钟，乙跑一圈用{b}分钟，两人同时同地出发。{name}想知道至少多少分钟后两人再次在起点相遇，请你帮他算一算。",
        f"环形跑道上，甲跑完一圈需{a}分钟，乙跑完一圈需{b}分钟。两人同时同地同向起跑，至少几分钟后在起点再次相遇？",
        f"甲、乙两人绕操场跑步，跑一圈分别要{a}分钟和{b}分钟。两人同时同地出发，至少多少分钟后同时回到起点？",
    ])
    lines = [
        f"起点再次相遇的时间 = {a} × {b} = {x}分钟",
    ]
    return ins, lines, x


_reg("laps_start_meet", laps_start_meet)


# 52. 绳测井深
def rope_measure_well(rng):
    a = rng.randint(5, 15)
    b = rng.randint(2, (3 * a - 1) // 4)
    L = 12 * (a - b)
    h = L // 3 - a
    who = rng.choice(["井深", "绳长"])
    ans = h if who == "井深" else L
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"用一根绳子测量井深：把绳子3折来量，井外余{a}米；把绳子4折来量，井外余{b}米。{who}是多少米？",
        f"古人用绳测井深，绳3折入井，井外剩{a}米；绳4折入井，井外剩{b}米。{name}想知道{who}是多少米，请你帮他算一算。",
        f"一根绳子3折后去量井深，露出井口{a}米；4折后去量，露出井口{b}米。{who}是多少米？",
        f"用绳测井深，3折量时井外余{a}米，4折量时井外余{b}米。{who}是多少米？",
    ])
    lines = [
        f"两次井外余绳的差 = {a} - {b} = {a - b}米",
        f"绳子的长度 = {a - b} × 12 = {L}米",
        f"绳子3折后的长度 = {L} ÷ 3 = {L // 3}米",
        f"绳子4折后的长度 = {L} ÷ 4 = {L // 4}米",
        f"井深 = {L // 4} - {b} = {h}米",
    ]
    if who == "绳长":
        lines = lines[:1] + lines[2:] + [lines[1]]
    return ins, lines, ans


_reg("rope_measure_well", rope_measure_well)


# 53. 比例尺
def map_scale(rng):
    n = rng.choice([100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000])
    a = rng.randint(2, 30)
    for _ in range(50):
        if (a * n) % 100 == 0:
            break
        a = rng.randint(2, 30)
    else:
        n, a = 1000, 10
    name = rng.choice(NAMES)
    if rng.random() < 0.5:
        real_cm = a * n
        real_m = real_cm // 100
        ins = rng.choice([
            f"一幅地图的比例尺是1比{n}，量得A、B两地的图上距离是{a}厘米。A、B两地的实际距离是多少米？",
            f"在比例尺为1:{n}的地图上，{name}量得两地距离是{a}厘米。两地的实际距离是多少米？",
            f"一张地图的比例尺是1比{n}，图上{a}厘米表示实际距离多少米？",
            f"地图上{a}厘米的距离，在比例尺1比{n}的地图上，实际距离是多少米？",
        ])
        lines = [
            f"实际距离 = {a} × {n} = {real_cm}厘米",
            f"换算成米 = {real_cm} ÷ 100 = {real_m}米",
        ]
        return ins, lines, real_m
    s = a * n // 100
    fig = s * 100 // n
    ins = rng.choice([
        f"一幅地图的比例尺是1比{n}，A、B两地的实际距离是{s}米。A、B两地在这幅地图上的距离是多少厘米？",
        f"在比例尺为1:{n}的地图上，实际距离{s}米的两地，图上距离是多少厘米？",
        f"一张地图的比例尺是1比{n}，实际{s}米的距离在图上是多少厘米？",
        f"两地实际相距{s}米，画在比例尺1比{n}的地图上，应画多少厘米？",
    ])
    lines = [
        f"实际距离换算成厘米 = {s} × 100 = {s * 100}厘米",
        f"图上距离 = {s * 100} ÷ {n} = {fig}厘米",
    ]
    return ins, lines, fig


_reg("map_scale", map_scale)


# 54. 齿轮转动
def gear_rotation(rng):
    a = rng.randint(10, 60)
    n = rng.randint(2, 12)
    b = rng.randint(10, 60)
    for _ in range(50):
        if (a * n) % b == 0:
            break
        b = rng.randint(10, 60)
    else:
        a, n, b = 20, 3, 15
    ans = a * n // b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"两个互相咬合的齿轮，大齿轮有{a}个齿，小齿轮有{b}个齿。大齿轮转{n}圈，小齿轮转多少圈？",
        f"一对互相咬合的齿轮，主动轮{a}个齿，从动轮{b}个齿。主动轮转{n}圈，从动轮转多少圈？",
        f"两个咬合的齿轮，一个有{a}个齿，另一个有{b}个齿。{a}个齿的齿轮转{n}圈，另一个转多少圈？",
        f"钟表里两个咬合的齿轮分别有{a}个齿和{b}个齿，{a}个齿的齿轮转{n}圈，另一个齿轮转多少圈？",
    ])
    lines = [
        f"大齿轮转过的齿数 = {a} × {n} = {a * n}个",
        f"小齿轮转的圈数 = {a * n} ÷ {b} = {ans}圈",
    ]
    return ins, lines, ans


_reg("gear_rotation", gear_rotation)


# 55. 小数点移动
def decimal_point_shift(rng):
    k = rng.randint(2, 40)
    d = 9 * k
    x = Fraction(10 * d, 9)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"把一个小数的小数点向左移动一位后，得到的数比原数小{d}。这个小数原来是多少？",
        f"一个小数的小数点向左移动一位，新数比原数少{d}。{name}想知道原数是多少，请你帮他算一算。",
        f"把一个数的小数点向左移动一位，这个数就比原来小{d}。原来的数是多少？",
        f"一个小数，小数点左移一位后比原数小{d}。这个小数原来是多少？",
    ])
    lines = [
        f"倍数差 = 10 - 1 = 9",
        f"原数的十分之一 = {d} ÷ 9 = {k}",
        f"原来的小数 = {k} × 10 = {num(x)}",
    ]
    return ins, lines, x


_reg("decimal_point_shift", decimal_point_shift)


# 56. 分子分母同加
def fraction_add_k(rng):
    n = rng.randint(2, 7)
    d = rng.randint(n + 1, 12)
    p = rng.randint(n + 1, 9)
    q = rng.randint(max(p + 1, d + 1), 15)
    for _ in range(50):
        if p * d - q * n > 0 and (p * d - q * n) % (q - p) == 0:
            break
        p = rng.randint(n + 1, 9)
        q = rng.randint(max(p + 1, d + 1), 15)
    else:
        n, d, p, q = 2, 3, 3, 4
    k = (p * d - q * n) // (q - p)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"分数{n}/{d}的分子和分母同时加上一个相同的数后，等于{p}/{q}。加上的这个数是多少？",
        f"把分数{n}/{d}的分子、分母同时加上同一个数，所得的分数等于{p}/{q}。{name}想知道加上的数是多少，请你帮他算一算。",
        f"一个分数是{n}/{d}，分子分母同时加上一个相同的数后约分为{p}/{q}。加上的数是多少？",
        f"分数{n}/{d}的分子与分母同时加上某数后等于{p}/{q}，求某数。",
    ])
    lines = [
        f"新分子乘原分母 = {p} × {d} = {p * d}",
        f"新分母乘原分子 = {q} × {n} = {q * n}",
        f"分子分母的差 = {p * d} - {q * n} = {p * d - q * n}",
        f"新分母与新分子的差 = {q} - {p} = {q - p}",
        f"加上的数 = {p * d - q * n} ÷ {q - p} = {k}",
    ]
    return ins, lines, k


_reg("fraction_add_k", fraction_add_k)


# 57. 速度比求时间
def speed_ratio_time(rng):
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    for _ in range(50):
        if b != a:
            break
        b = rng.randint(2, 9)
    else:
        b = a + 1
    t = rng.randint(5, 40)
    for _ in range(50):
        if (a * t) % b == 0:
            break
        t = rng.randint(5, 40)
    else:
        a, b, t = 2, 3, 6
    ans = a * t // b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"走同一段路，甲、乙的速度比是{a}比{b}，甲用了{t}分钟，乙要用多少分钟？",
        f"甲、乙两人走同样长的一段路，速度比为{a}:{b}，甲走了{t}分钟。{name}想知道乙要走多少分钟，请你帮他算一算。",
        f"行同一段路程，甲与乙的速度比是{a}比{b}，甲用时{t}分钟，乙用时多少分钟？",
        f"甲、乙速度的比是{a}:{b}，走同一段路甲用{t}分钟，乙要用多少分钟？",
    ])
    lines = [
        f"甲的速度乘时间 = {a} × {t} = {a * t}",
        f"乙用的时间 = {a * t} ÷ {b} = {ans}分钟",
    ]
    return ins, lines, ans


_reg("speed_ratio_time", speed_ratio_time)


# 58. 平方和与立方和
def sum_squares_cubes(rng):
    n = rng.randint(3, 15)
    mode = rng.choice(["平方", "立方"])
    scene = rng.choice([
        "数学课上老师出了一道题",
        f"{rng.choice(NAMES)}在练习册上看到一道题",
        "兴趣课上老师出了一道思考题",
    ])
    if mode == "平方":
        ans = n * (n + 1) * (2 * n + 1) // 6
        ins = rng.choice([
            f"{scene}：1的平方加2的平方，一直加到{n}的平方，和是多少？",
            f"{scene}：求1² + 2² + … + {n}² 的和。",
            f"{scene}：从1的平方加到{n}的平方，得数是多少？",
            f"{scene}：前{n}个自然数的平方和是多少？",
        ])
        lines = [
            f"求和公式中的首末和 = {n} + 1 = {n + 1}",
            f"求和公式中的2倍加1 = 2 × {n} + 1 = {2 * n + 1}",
            f"分子 = {n} × {n + 1} × {2 * n + 1} = {n * (n + 1) * (2 * n + 1)}",
            f"平方和 = {n * (n + 1) * (2 * n + 1)} ÷ 6 = {ans}",
        ]
        return ins, lines, ans
    s1 = n * (n + 1) // 2
    ans = s1 * s1
    ins = rng.choice([
        f"{scene}：1的立方加2的立方，一直加到{n}的立方，和是多少？",
        f"{scene}：求1³ + 2³ + … + {n}³ 的和。",
        f"{scene}：从1的立方加到{n}的立方，得数是多少？",
        f"{scene}：前{n}个自然数的立方和是多少？",
    ])
    lines = [
        f"求和公式中的首末和 = {n} + 1 = {n + 1}",
        f"1到{n}的和 = {n} × {n + 1} ÷ 2 = {s1}",
        f"立方和 = {s1} × {s1} = {ans}",
    ]
    return ins, lines, ans


_reg("sum_squares_cubes", sum_squares_cubes)


# 59. 上下坡平均速度
def harmonic_mean_speed(rng):
    v1 = rng.randint(30, 80)
    v2 = rng.randint(30, 80)
    for _ in range(50):
        if v2 != v1:
            break
        v2 = rng.randint(30, 80)
    else:
        v1, v2 = 40, 60
    ans = Fraction(2 * v1 * v2, v1 + v2)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"小明上山每分钟走{v1}米，沿原路下山每分钟走{v2}米。小明上、下山的平均速度是每分钟多少米？",
        f"一段山路，{name}上山时每分钟行{v1}米，下山时每分钟行{v2}米。他往返的平均速度是每分钟多少米？",
        f"小华上山速度为每分钟{v1}米，下山速度为每分钟{v2}米，沿原路返回。他上、下山的平均速度是多少？",
        f"从山脚到山顶，上山每分钟走{v1}米，下山每分钟走{v2}米。往返的平均速度是每分钟多少米？",
    ])
    lines = [
        f"两个速度的积 = {v1} × {v2} = {v1 * v2}",
        f"两个速度积的2倍 = {v1 * v2} × 2 = {2 * v1 * v2}",
        f"两个速度的和 = {v1} + {v2} = {v1 + v2}",
        f"平均速度 = {2 * v1 * v2} ÷ {v1 + v2} = {num(ans)}米/分",
    ]
    return ins, lines, ans


_reg("harmonic_mean_speed", harmonic_mean_speed)


# 60. 三段路程平均速度
def avg_speed_three_segments(rng):
    t1 = rng.randint(1, 6)
    t2 = rng.randint(1, 6)
    t3 = rng.randint(1, 6)
    v1 = rng.randint(40, 80)
    v2 = rng.randint(40, 80)
    v3 = rng.randint(40, 80)
    s1, s2, s3 = v1 * t1, v2 * t2, v3 * t3
    S = s1 + s2 + s3
    T = t1 + t2 + t3
    ans = Fraction(S, T)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一辆汽车先以每小时{v1}千米的速度行驶{t1}小时，又以每小时{v2}千米的速度行驶{t2}小时，最后以每小时{v3}千米的速度行驶{t3}小时。这辆汽车全程的平均速度是每小时多少千米？",
        f"{name}骑车郊游，前{t1}小时每小时行{v1}千米，接着{t2}小时每小时行{v2}千米，最后{t3}小时每小时行{v3}千米。他全程的平均速度是每小时多少千米？",
        f"一辆货车分三段送货：第一段每小时{v1}千米行了{t1}小时，第二段每小时{v2}千米行了{t2}小时，第三段每小时{v3}千米行了{t3}小时。全程平均速度是多少？",
        f"汽车在三段路上的速度分别是每小时{v1}、{v2}、{v3}千米，行驶时间分别是{t1}、{t2}、{t3}小时。全程的平均速度是每小时多少千米？",
    ])
    lines = [
        f"第一段路程 = {v1} × {t1} = {s1}千米",
        f"第二段路程 = {v2} × {t2} = {s2}千米",
        f"第三段路程 = {v3} × {t3} = {s3}千米",
        f"总路程 = {s1} + {s2} + {s3} = {S}千米",
        f"总时间 = {t1} + {t2} + {t3} = {T}小时",
        f"平均速度 = {S} ÷ {T} = {num(ans)}千米/时",
    ]
    return ins, lines, ans


_reg("avg_speed_three_segments", avg_speed_three_segments)


# 61. 混合后稀释
def concentration_mix_dilute(rng):
    k1 = rng.randint(1, 10)
    k2 = rng.randint(1, 10)
    a, b = 100 * k1, 100 * k2
    p = rng.choice([5, 10, 15, 20, 25])
    q = rng.choice([5, 10, 15, 20, 25])
    q2 = water = None
    for _ in range(80):
        q2 = rng.choice([4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25])
        salt = k1 * p + k2 * q
        if salt * 100 % q2 == 0:
            new_total = salt * 100 // q2
            if new_total > a + b:
                water = new_total - a - b
                break
    else:
        a, b, p, q, q2, water = 200, 300, 10, 20, 5, 1100
    salt1 = a * p // 100
    salt2 = b * q // 100
    salt = salt1 + salt2
    total = a + b
    new_total = total + water
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲瓶有{a}克盐水，含盐率{p}%；乙瓶有{b}克盐水，含盐率{q}%。把两瓶盐水混合后，要使含盐率变为{q2}%，需要再加入多少克水？",
        f"两杯盐水，一杯{a}克、含盐率{p}%，另一杯{b}克、含盐率{q}%。{name}把它们倒在一起，要使含盐率变成{q2}%，应加水多少克？",
        f"把{a}克含盐率{p}%的盐水和{b}克含盐率{q}%的盐水混合，要使混合后的含盐率为{q2}%，需加水多少克？",
        f"甲容器有{a}克含盐{p}%的盐水，乙容器有{b}克含盐{q}%的盐水。混合后要使含盐率为{q2}%，需要加多少克水？",
    ])
    lines = [
        f"甲瓶盐的质量 = {a} × {p}/100 = {salt1}克",
        f"乙瓶盐的质量 = {b} × {q}/100 = {salt2}克",
        f"盐的总质量 = {salt1} + {salt2} = {salt}克",
        f"混合后盐水的总质量 = {a} + {b} = {total}克",
        f"稀释后盐水的总质量 = {salt} × 100 ÷ {q2} = {new_total}克",
        f"需要加入的水 = {new_total} - {total} = {water}克",
    ]
    return ins, lines, water


_reg("concentration_mix_dilute", concentration_mix_dilute)


# 62. 蒸发提浓
def concentration_evaporate(rng):
    k = rng.randint(1, 10)
    a = 100 * k
    p = rng.choice([5, 10, 15, 20])
    q = evap = None
    for _ in range(80):
        q = rng.choice([10, 12, 15, 16, 20, 24, 25, 30, 40])
        if q > p:
            salt = k * p
            if salt * 100 % q == 0:
                new_a = salt * 100 // q
                if new_a < a:
                    evap = a - new_a
                    break
    else:
        a, p, q, evap = 200, 10, 20, 100
    salt = a * p // 100
    new_a = a - evap
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"有{a}克盐水，含盐率是{p}%。要使含盐率变为{q}%，需要蒸发掉多少克水？",
        f"一杯{a}克的盐水，含盐率{p}%。{name}想通过蒸发水使含盐率变成{q}%，需要蒸发多少克水？",
        f"把{a}克含盐率{p}%的盐水蒸发一部分水后，含盐率变为{q}%。蒸发了多少克水？",
        f"现有{a}克含盐{p}%的盐水，要使含盐率提高到{q}%，需蒸发掉多少克水？",
    ])
    lines = [
        f"盐的质量 = {a} × {p}/100 = {salt}克",
        f"蒸发后盐水的质量 = {salt} × 100 ÷ {q} = {new_a}克",
        f"蒸发掉的水 = {a} - {new_a} = {evap}克",
    ]
    return ins, lines, evap


_reg("concentration_evaporate", concentration_evaporate)


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
    print(f"L4 ext4 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
