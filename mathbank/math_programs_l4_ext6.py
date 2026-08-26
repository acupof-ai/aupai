#!/usr/bin/env python3
"""L4 ext6: novel structures — two-bridge trains, drift logs, diameter-start
circular tracks, time-headstart chases, return-meet races, three-pedestrian
meet intervals, detour trips, troop messengers, bus dispatch headways,
delayed-start meets, asymmetric up/downhill, log-chase boats, two-rider
escalators, speed-ratio arrival gaps, stopover meets, alternating work shifts,
mid-project crew increases, leaking pools, repeated dilution, equalizing swaps,
partial-discount profits, two-item discounted costs, savings-plan comparison,
three-solution mixing, two-alloy blending, evaporate-then-salt, three-worker
leave, pipe schedules, sell-cattle grazing, digit sums/counts, decimal digit
sums, square-plus-two, prime sum-product, power digit sums, arithmetic-sequence
trailing zeros, remainder interval counts, hybrid tournaments, queue
arrangements, digit combinatorics, two-color drawer, grid square/rectangle
counts, cut-to-cube cuboids, painted cubes, cone-cylinder volume, wire
reshaping, pair-side square increase, triangle base increase, trapezoid
cuts, target averages, monkey peaches, denominator add, two-day reading
reverse, double-transfer groups, fractional three-people relations, candle
length ratios, three-set inclusion. 60 families.

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


# 1. 火车通过两座不同的桥，用时不同 → 车速与车长
def train_two_bridges(rng):
    v = rng.randint(12, 25)
    L = rng.randint(8, 30) * v
    a = rng.randint(2, 10) * v
    b = a + rng.randint(2, 8) * v
    t1 = (L + a) // v
    t2 = (L + b) // v
    who = rng.choice(["车速", "车长"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一列火车通过一座长{a}米的桥用了{t1}秒，以同样的速度通过一座长{b}米的桥用了{t2}秒。这列火车的{who}是多少？",
        f"一列火车匀速行驶，经过长{a}米的大桥用时{t1}秒，经过长{b}米的大桥用时{t2}秒。{name}想知道火车的{who}，请你帮他算一算。",
        f"铁路旁，一列火车通过{a}米的桥用{t1}秒，通过{b}米的桥用{t2}秒，车速不变。火车的{who}是多少？请列式算一算。",
        f"一列火车开过一座长{a}米的桥要{t1}秒，开过一座长{b}米的桥要{t2}秒。{name}没算出火车的{who}，你能帮帮他吗？",
    ])
    lines = [
        f"两座桥的长度差 = {b} - {a} = {b - a}米",
        f"通过时间的差 = {t2} - {t1} = {t2 - t1}秒",
        f"火车的速度 = {b - a} ÷ {t2 - t1} = {v}米/秒",
        f"火车的长度 = {v} × {t1} - {a} = {L}米",
    ]
    if who == "车速":
        lines = lines[:3] + [lines[2]]
        ans = v
    else:
        ans = L
    return ins, lines, ans


_reg("train_two_bridges", train_two_bridges)


# 2. 船顺水时间与逆水时间已知 → 木箱漂流时间
def log_float_drift(rng):
    u = rng.randint(3, 8)
    v = rng.randint(2, u - 1)
    g = math.lcm(u + v, u - v, v)
    m = rng.randint(1, 3)
    s = g * m
    t1 = s // (u + v)
    t2 = s // (u - v)
    T = s // v
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一艘船从A码头顺水航行到B码头用了{t1}小时，从B码头逆水返回A码头用了{t2}小时，两码头相距{s}千米。一只木箱从A码头顺水漂到B码头需要多少小时？",
        f"船在静水中速度不变，水流速度不变。船从A到B顺水行{t1}小时，从B到A逆水行{t2}小时，A、B相距{s}千米。{name}想知道一只木箱从A漂到B要几小时，请你算一算。",
        f"两个码头相距{s}千米，船顺水行驶全程要{t1}小时，逆水行驶全程要{t2}小时。一个木箱从上游码头顺水漂到下游码头，需要多少小时？",
        f"船顺水走{s}千米用{t1}小时，逆水走同样路程用{t2}小时。{name}把一只木箱放入水中，木箱顺水漂到下游码头要多少小时？请列式算一算。",
    ])
    lines = [
        f"顺水速度 = {s} ÷ {t1} = {u + v}千米/时",
        f"逆水速度 = {s} ÷ {t2} = {u - v}千米/时",
        f"水流速度 = ({u + v} - {u - v}) ÷ 2 = {v}千米/时",
        f"木箱漂流时间 = {s} ÷ {v} = {T}小时",
    ]
    return ins, lines, T


_reg("log_float_drift", log_float_drift)


# 3. 环形跑道直径两端同时出发 → 第一次相遇
def circular_diameter_start(rng):
    v1 = rng.randint(3, 9)
    v2 = rng.randint(3, 9)
    s = v1 + v2
    k = rng.randint(2, 6)
    C = 2 * s * k
    who = rng.choice(["相遇时间", "甲走的路程"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个环形跑道的周长是{C}米，甲、乙两人从跑道直径的两端同时出发，背向而行，甲每分钟跑{v1}米，乙每分钟跑{v2}米。两人第一次相遇时{who}是多少？",
        f"环形跑道周长{C}米，甲、乙分别站在直径两端，同时相向而行，甲速{v1}米/分，乙速{v2}米/分。{name}想知道第一次相遇时{who}，请你算一算。",
        f"甲、乙两人在周长{C}米的环形跑道上，从直径两端同时出发背向跑步，甲每分钟{v1}米，乙每分钟{v2}米。第一次相遇时{who}是多少？请列式算一算。",
        f"环形跑道一圈{C}米，甲、乙从直径两端同时出发相向而行，速度分别是{v1}米/分和{v2}米/分。{name}问：第一次相遇时{who}是多少？",
    ])
    lines = [
        f"两人出发时相距 = {C} ÷ 2 = {C // 2}米",
        f"两人的速度和 = {v1} + {v2} = {s}米/分",
        f"第一次相遇时间 = {C // 2} ÷ {s} = {k}分钟",
        f"相遇时甲走的路程 = {v1} × {k} = {v1 * k}米",
    ]
    if who == "相遇时间":
        lines = lines[:3] + [lines[2]]
        ans = k
    else:
        ans = v1 * k
    return ins, lines, ans


_reg("circular_diameter_start", circular_diameter_start)


# 4. 甲让乙先跑一段时间 → 追及
def chase_headstart_time(rng):
    v1 = rng.randint(5, 10)
    v2 = rng.randint(3, v1 - 2)
    k = rng.randint(2, 5)
    t0 = (v1 - v2) * k
    gap = v2 * t0
    t = v2 * k
    who = rng.choice(["追及时间", "追上点距起点"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲每秒跑{v1}米，乙每秒跑{v2}米。甲让乙先跑{t0}秒，然后甲出发去追乙。甲出发后{who}是多少？",
        f"乙在甲前面先跑{t0}秒，乙每秒跑{v2}米，甲每秒跑{v1}米，两人同时同向。{name}想知道甲出发后{who}，请你算一算。",
        f"甲、乙练习跑步，甲每秒{v1}米，乙每秒{v2}米。乙先跑{t0}秒后甲才出发，甲追上乙时{who}是多少？请列式算一算。",
        f"乙先跑{t0}秒，每秒跑{v2}米；甲随后出发，每秒跑{v1}米。{name}问：甲出发后{who}是多少？",
    ])
    lines = [
        f"乙先跑的路程 = {v2} × {t0} = {gap}米",
        f"两人的速度差 = {v1} - {v2} = {v1 - v2}米/秒",
        f"甲出发后的追及时间 = {gap} ÷ {v1 - v2} = {t}秒",
        f"追上点距起点的路程 = {v1} × {t} = {v1 * t}米",
    ]
    if who == "追及时间":
        lines = lines[:3] + [lines[2]]
        ans = t
    else:
        ans = v1 * t
    return ins, lines, ans


_reg("chase_headstart_time", chase_headstart_time)


# 5. 赛跑：甲到终点返回，在距终点b米处遇乙 → 乙速/乙路程
def race_return_meet(rng):
    m = rng.randint(3, 6)
    n = rng.randint(2, m - 1)
    k = rng.randint(5, 15)
    a = k * (m + n)
    b = k * (m - n)
    u = rng.randint(2, 6)
    v1 = m * u
    v2 = n * u
    who = rng.choice(["乙的速度", "相遇时乙跑的路程"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人在{a}米的跑道上赛跑，甲每秒跑{v1}米。甲到达终点后立即返回，在距终点{b}米处与乙相遇。{who}是多少？",
        f"甲、乙从同一起点出发赛跑，跑道长{a}米，甲每秒跑{v1}米。甲到终点后马上返回，在离终点{b}米的地方遇到乙。{name}想知道{who}，请你算一算。",
        f"一次{a}米赛跑，甲每秒跑{v1}米，甲到终点后立即返回，在距终点{b}米处与乙相遇。{who}是多少？请列式算一算。",
        f"甲、乙比赛跑步，全程{a}米，甲速{v1}米/秒。甲跑到终点后返回，在距终点{b}米处碰上乙。{name}问{who}是多少？",
    ])
    if who == "乙的速度":
        lines = [
            f"相遇时甲跑的路程 = {a} + {b} = {a + b}米",
            f"相遇时乙跑的路程 = {a} - {b} = {a - b}米",
            f"乙速是甲速的几分之几 = {a - b} ÷ {a + b} = {n}/{m}",
            f"乙的速度 = {v1} × {n} ÷ {m} = {v2}米/秒",
        ]
        ans = v2
    else:
        time = Fraction(a + b, v1)
        lines = [
            f"相遇时甲跑的路程 = {a} + {b} = {a + b}米",
            f"两人相遇用的时间 = {a + b} ÷ {v1} = {num(time)}秒",
            f"相遇时乙跑的路程 = {v2} × {num(time)} = {a - b}米",
        ]
        ans = a - b
    return ins, lines, ans


_reg("race_return_meet", race_return_meet)


# 6. 甲、乙从A，丙从B，甲丙相遇后t分钟乙丙相遇 → AB距离
def three_people_meet_interval(rng):
    S = T1 = None
    for _ in range(100):
        va = rng.randint(60, 90)
        vb = rng.randint(40, va - 10)
        vc = rng.randint(30, 60)
        t = rng.randint(1, 5)
        den = va - vb
        if (t * (vb + vc)) % den == 0:
            T1 = t * (vb + vc) // den
            S = (va + vc) * T1
            break
    else:
        va, vb, vc, t, T1, S = 70, 60, 50, 1, 11, 1320
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人从A地，丙从B地同时出发相向而行，甲每分钟走{va}米，乙每分钟走{vb}米，丙每分钟走{vc}米。甲与丙相遇后，又过{t}分钟乙与丙相遇。A、B两地相距多少米？",
        f"甲、乙从A地、丙从B地同时相向出发，甲速{va}米/分，乙速{vb}米/分，丙速{vc}米/分。甲和丙相遇{t}分钟后，乙和丙才相遇。{name}想知道A、B距离，请你算一算。",
        f"甲、乙、丙三人，甲、乙在A地，丙在B地，同时相向而行。甲每分{va}米，乙每分{vb}米，丙每分{vc}米，甲丙相遇{t}分钟后乙丙相遇。A、B相距多少米？请列式算一算。",
        f"甲、乙同从A地出发，丙从B地出发，相向而行，速度分别为每分钟{va}米、{vb}米、{vc}米。甲丙相遇后{t}分钟乙丙相遇。{name}问A、B两地相距多远？",
    ])
    lines = [
        f"甲丙相遇时乙丙还相距 = ({vb} + {vc}) × {t} = {(vb + vc) * t}米",
        f"甲丙每分钟比乙丙多走 = ({va} + {vc}) - ({vb} + {vc}) = {va - vb}米",
        f"甲丙相遇用的时间 = {(vb + vc) * t} ÷ {va - vb} = {T1}分钟",
        f"A、B两地的距离 = ({va} + {vc}) × {T1} = {S}米",
    ]
    return ins, lines, S


_reg("three_people_meet_interval", three_people_meet_interval)


# 7. 甲走一段后返回取物，两人同时到达 → 家到学校距离
def detour_school(rng):
    S = t = None
    for _ in range(100):
        v1 = rng.randint(60, 90)
        v2 = rng.randint(40, v1 - 10)
        a = rng.randint(2, 8) * 10
        if (2 * a) % (v1 - v2) == 0:
            t = 2 * a // (v1 - v2)
            S = v2 * t
            break
    else:
        v1, v2, a, S, t = 80, 60, 60, 360, 6
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人同时从家去学校，甲每分钟走{v1}米，乙每分钟走{v2}米。甲走了{a}米后发现忘带红领巾，立即返回家去取，再去学校，结果两人同时到达学校。家到学校有多少米？",
        f"甲、乙同时从家出发去学校，甲速{v1}米/分，乙速{v2}米/分。甲走出{a}米后返回家取东西，再去学校，两人恰好同时到达。{name}想知道家到学校的距离，请你算一算。",
        f"甲、乙都从家去学校，甲每分{v1}米，乙每分{v2}米。甲走{a}米后回家取红领巾再去学校，乙一直走，两人同时到。家到学校多少米？请列式算一算。",
        f"上学路上，甲每分钟走{v1}米，乙每分钟走{v2}米，两人同时出发。甲走{a}米后返回家里取东西，然后去学校，正好和乙同时到达。{name}问家到学校有多远？",
    ])
    lines = [
        f"甲比乙多走的路程 = {a} × 2 = {2 * a}米",
        f"两人的速度差 = {v1} - {v2} = {v1 - v2}米/分",
        f"两人走的时间 = {2 * a} ÷ {v1 - v2} = {t}分钟",
        f"家到学校的距离 = {v2} × {t} = {S}米",
    ]
    return ins, lines, S


_reg("detour_school", detour_school)


# 8. 传令兵从队尾到队头再返回 → 总时间
def messenger_troop(rng):
    v = rng.randint(2, 4)
    u = rng.randint(v + 3, v + 8)
    g = math.lcm(u - v, u + v)
    k = rng.randint(1, 3)
    L = g * k
    t1 = L // (u - v)
    t2 = L // (u + v)
    total = t1 + t2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一支队伍长{L}米，以每秒{v}米的速度前进。传令兵从队尾以每秒{u}米的速度跑到队头传达命令，再立即返回队尾。传令兵一共用了多少秒？",
        f"队伍长{L}米，前进速度是每秒{v}米。一名传令兵从队尾跑步到队头，速度每秒{u}米，到队头后马上跑回队尾。{name}想知道传令兵共用多少秒，请你算一算。",
        f"一列队伍以{v}米/秒前进，队伍长{L}米。传令兵以{u}米/秒从队尾赶到队头，又以同样速度返回队尾。他往返一共用了多少秒？请列式算一算。",
        f"行军队伍长{L}米，每秒走{v}米。传令兵每秒跑{u}米，从队尾到队头再回到队尾。{name}问他一共用了多少秒？",
    ])
    lines = [
        f"赶到队头的速度差 = {u} - {v} = {u - v}米/秒",
        f"从队尾到队头的时间 = {L} ÷ {u - v} = {t1}秒",
        f"返回队尾的速度和 = {u} + {v} = {u + v}米/秒",
        f"从队头到队尾的时间 = {L} ÷ {u + v} = {t2}秒",
        f"往返的总时间 = {t1} + {t2} = {total}秒",
    ]
    return ins, lines, total


_reg("messenger_troop", messenger_troop)


# 9. 公交车发车间隔 → 骑车人遇车间隔
def bus_dispatch_interval(rng):
    T = None
    for _ in range(100):
        v = rng.randint(300, 600)
        u = rng.randint(100, v - 100)
        t = rng.randint(3, 10)
        way = rng.choice(["相向", "同向"])
        den = v + u if way == "相向" else v - u
        if (v * t) % den == 0:
            T = v * t // den
            break
    else:
        v, u, t, way, T = 500, 200, 7, "相向", 5
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"公交车每隔{t}分钟发一辆，车速是每分钟{v}米。{name}骑车每分钟行{u}米，与公交车{way}而行。{name}每隔多少分钟遇到一辆公交车？",
        f"一条线路的公交车每隔{t}分钟发一班，车速为每分{v}米。{name}骑自行车每分{u}米，与公交车{way}而行。他每隔几分钟遇到一辆公交车？请你算一算。",
        f"公交车每隔{t}分钟开出一辆，每分行{v}米。{name}骑车每分行{u}米，与公交车{way}而行。{name}每隔多少分钟能遇到一辆公交车？请列式算一算。",
        f"马路上公交车每隔{t}分钟发一辆，速度是每分{v}米。{name}以每分{u}米的速度与公交车{way}骑行。他每隔几分钟遇到一辆公交车？",
    ])
    if way == "相向":
        lines = [
            f"相邻两辆公交车的间距 = {v} × {t} = {v * t}米",
            f"人与车的速度和 = {v} + {u} = {v + u}米/分",
            f"遇到公交车的间隔 = {v * t} ÷ {v + u} = {T}分钟",
        ]
    else:
        lines = [
            f"相邻两辆公交车的间距 = {v} × {t} = {v * t}米",
            f"人与车的速度差 = {v} - {u} = {v - u}米/分",
            f"遇到公交车的间隔 = {v * t} ÷ {v - u} = {T}分钟",
        ]
    return ins, lines, T


_reg("bus_dispatch_interval", bus_dispatch_interval)


# 10. 甲先行一段时间后乙出发 → 乙速/两地距离
def delayed_start_meet(rng):
    v1 = rng.randint(50, 90)
    v2 = rng.randint(40, 80)
    t = rng.randint(1, 3)
    s = rng.randint(2, 6)
    S = v1 * (t + s) + v2 * s
    who = rng.choice(["乙的速度", "两地距离"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"A、B两地相距{S}千米，甲、乙两车分别从A、B同时相向而行。甲车先行{t}小时后乙车才出发，乙车出发后又经过{s}小时两车相遇，甲车每小时行{v1}千米。{who}是多少？",
        f"甲、乙两车从相距{S}千米的两地相向而行，甲先出发{t}小时，乙再出发，乙出发{s}小时后两车相遇，甲车速{v1}千米/时。{name}想知道{who}，请你算一算。",
        f"两地相距{S}千米，甲车从A地先行{t}小时后，乙车从B地出发相向而行，又过{s}小时相遇，甲每小时{v1}千米。{who}是多少？请列式算一算。",
        f"甲、乙相向而行，两地相距{S}千米。甲先走{t}小时乙才出发，再经过{s}小时两人相遇，甲速{v1}千米/时。{name}问{who}是多少？",
    ])
    if who == "乙的速度":
        lines = [
            f"甲车走的路程 = {v1} × ({t} + {s}) = {v1 * (t + s)}千米",
            f"乙车走的路程 = {S} - {v1 * (t + s)} = {v2 * s}千米",
            f"乙车的速度 = {v2 * s} ÷ {s} = {v2}千米/时",
        ]
        ans = v2
    else:
        lines = [
            f"甲车走的路程 = {v1} × ({t} + {s}) = {v1 * (t + s)}千米",
            f"乙车走的路程 = {v2} × {s} = {v2 * s}千米",
            f"两地的距离 = {v1 * (t + s)} + {v2 * s} = {S}千米",
        ]
        ans = S
    return ins, lines, ans


_reg("delayed_start_meet", delayed_start_meet)


# 11. 去时上坡a下坡b，返回上下坡互换 → 往返总时间
def uphill_downhill(rng):
    v1 = rng.randint(3, 8) * 10
    v2 = v1 + rng.randint(1, 4) * 10
    g = math.lcm(v1, v2)
    a = g * rng.randint(1, 2)
    b = g * rng.randint(1, 2)
    t1 = a // v1 + b // v2
    t2 = b // v1 + a // v2
    t = t1 + t2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两地之间，去时的上坡路长{a}千米、下坡路长{b}千米。一辆汽车上坡每小时行{v1}千米，下坡每小时行{v2}千米。这辆汽车往返一次一共需要多少小时？",
        f"从甲地到乙地，上坡路{a}千米，下坡路{b}千米。汽车上坡速{v1}千米/时，下坡速{v2}千米/时。{name}想知道汽车往返一次的总时间，请你算一算。",
        f"甲、乙两地间上坡路长{a}千米、下坡路长{b}千米，汽车上坡每小时{v1}千米、下坡每小时{v2}千米。往返一次共需多少小时？请列式算一算。",
        f"一段路，去时上坡{a}千米、下坡{b}千米，返回时上坡变下坡、下坡变上坡。汽车上坡{v1}千米/时、下坡{v2}千米/时。{name}问往返共需几小时？",
    ])
    lines = [
        f"去时用的时间 = {a} ÷ {v1} + {b} ÷ {v2} = {t1}小时",
        f"返回用的时间 = {b} ÷ {v1} + {a} ÷ {v2} = {t2}小时",
        f"往返的总时间 = {t1} + {t2} = {t}小时",
    ]
    return ins, lines, t


_reg("uphill_downhill", uphill_downhill)


# 12. 船逆水行，木箱掉落，t分钟后发现，调头追 → 追上时间
def boat_log_chase(rng):
    v = rng.randint(10, 20) * 10
    u = rng.randint(2, 6) * 10
    t = rng.randint(5, 30)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一艘船逆水航行，船在静水中的速度是每分钟{v}米，水流速度是每分钟{u}米。船上一只木箱掉入水中，{t}分钟后船员才发现，立即调头去追。船调头后多少分钟能追上木箱？",
        f"船逆水行驶，静水速度{v}米/分，水速{u}米/分。一只木箱掉落水中顺流漂走，{t}分钟后船调头追木箱。{name}想知道调头后几分钟追上，请你算一算。",
        f"船在河中逆水航行，静水船速每分{v}米，水速每分{u}米。木箱掉后{t}分钟才发现，船马上调头追。调头后多少分钟追上木箱？请列式算一算。",
        f"逆水航行的船，静水速度每分{v}米，水流每分{u}米。木箱掉入水中，{t}分钟后船调头追木箱。{name}问调头后几分钟能追上？",
    ])
    lines = [
        f"船逆水速度 = {v} - {u} = {v - u}米/分",
        f"发现时船与木箱的距离 = ({v} - {u}) × {t} + {u} × {t} = {v * t}米",
        f"调头后船与木箱的速度差 = ({v} + {u}) - {u} = {v}米/分",
        f"调头后追上的时间 = {v * t} ÷ {v} = {t}分钟",
    ]
    return ins, lines, t


_reg("boat_log_chase", boat_log_chase)


# 13. 两人不同速度上扶梯 → 扶梯可见级数/扶梯速度
def escalator_two_riders(rng):
    u = t1 = t2 = N = None
    for _ in range(100):
        u = rng.randint(1, 3)
        a = rng.randint(2, 5)
        b = rng.randint(1, a - 1)
        t1 = rng.randint(10, 30)
        N = (a + u) * t1
        if N % (b + u) == 0:
            t2 = N // (b + u)
            break
    else:
        u, a, b, t1, t2, N = 2, 3, 2, 20, 25, 100
    who = rng.choice(["可见级数", "扶梯速度"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"商场的自动扶梯匀速向上运行。{name}每秒向上走{a}级，用了{t1}秒到达楼上；另一人每秒向上走{b}级，用了{t2}秒到达楼上。扶梯的{who}是多少？",
        f"自动扶梯向上运行，甲每秒走{a}级，{t1}秒到楼上；乙每秒走{b}级，{t2}秒到楼上。{name}想知道扶梯的{who}，请你算一算。",
        f"一部向上的自动扶梯，男孩每秒走{a}级，{t1}秒到达；女孩每秒走{b}级，{t2}秒到达。扶梯的{who}是多少？请列式算一算。",
        f"两个孩子上同一部向上的自动扶梯，一个每秒{a}级，用{t1}秒；另一个每秒{b}级，用{t2}秒。{name}问扶梯的{who}是多少？",
    ])
    lines = [
        f"甲走的级数 = {a} × {t1} = {a * t1}级",
        f"乙走的级数 = {b} × {t2} = {b * t2}级",
        f"扶梯的速度 = ({a * t1} - {b * t2}) ÷ ({t2} - {t1}) = {u}级/秒",
        f"扶梯的可见级数 = ({a} + {u}) × {t1} = {N}级",
    ]
    if who == "可见级数":
        ans = N
    else:
        lines = lines[:3] + [lines[2]]
        ans = u
    return ins, lines, ans


_reg("escalator_two_riders", escalator_two_riders)


# 14. 速度比与时间差 → 各自时间
def speed_ratio_arrival_diff(rng):
    a = rng.randint(3, 6)
    b = rng.randint(2, a - 1)
    k = rng.randint(2, 5)
    t_diff = (a - b) * k
    t甲 = b * k
    t乙 = a * k
    who = rng.choice(["甲", "乙"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"走同一段路，甲、乙的速度比是{a}比{b}，甲比乙早到{t_diff}分钟。{who}走这段路用了多少分钟？",
        f"甲、乙走同一段路，速度比为{a}:{b}，结果甲比乙少用{t_diff}分钟。{name}想知道{who}用了多少分钟，请你算一算。",
        f"同一段路，甲、乙速度比是{a}:{b}，乙比甲慢{t_diff}分钟。{who}走这段路要多少分钟？请列式算一算。",
        f"甲、乙两人走同一段路，速度比{a}比{b}，甲比乙早到{t_diff}分钟。{name}问{who}用了多少分钟？",
    ])
    lines = [
        f"时间差对应的份数 = {a} - {b} = {a - b}份",
        f"每份的时间 = {t_diff} ÷ {a - b} = {k}分钟",
        f"甲用的时间 = {b} × {k} = {t甲}分钟",
        f"乙用的时间 = {a} × {k} = {t乙}分钟",
    ]
    if who == "甲":
        lines = lines[:3] + [lines[2]]
        ans = t甲
    else:
        ans = t乙
    return ins, lines, ans


_reg("speed_ratio_arrival_diff", speed_ratio_arrival_diff)


# 15. 相向而行，甲中途停留 → 相遇总时间/甲路程
def meet_stopover(rng):
    v1 = rng.randint(50, 80)
    v2 = rng.randint(40, 70)
    t = rng.randint(2, 5)
    s = rng.randint(5, 15)
    r = rng.randint(2, 6)
    S = (v1 + v2) * t + v2 * s + (v1 + v2) * r
    total = t + s + r
    who = rng.choice(["相遇总时间", "相遇时甲走的路程"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"A、B两地相距{S}米，甲、乙两人同时相向而行，甲每分钟走{v1}米，乙每分钟走{v2}米。甲走了{t}分钟后修车停留了{s}分钟，再继续走。两人从出发到相遇一共用了多少分钟？",
        f"甲、乙从相距{S}米的两地相向而行，甲速{v1}米/分，乙速{v2}米/分。甲走{t}分钟后停留{s}分钟修车，然后继续。{name}想知道{who}，请你算一算。",
        f"两地相距{S}米，甲、乙相向而行，甲每分{v1}米，乙每分{v2}米。甲出发{t}分钟后停留{s}分钟，再继续走。{who}是多少？请列式算一算。",
        f"甲、乙两人在相距{S}米的路上相向而行，甲每分{v1}米，乙每分{v2}米。甲走{t}分钟后停了{s}分钟，再继续。{name}问{who}是多少？",
    ])
    lines = [
        f"前{t}分钟两人共走 = ({v1} + {v2}) × {t} = {(v1 + v2) * t}米",
        f"甲停留时乙走的路程 = {v2} × {s} = {v2 * s}米",
        f"剩下的路程 = {S} - {(v1 + v2) * t} - {v2 * s} = {(v1 + v2) * r}米",
        f"剩下路程合走的时间 = {(v1 + v2) * r} ÷ ({v1} + {v2}) = {r}分钟",
        f"从出发到相遇的总时间 = {t} + {s} + {r} = {total}分钟",
        f"相遇时甲走的路程 = {v1} × ({t} + {r}) = {v1 * (t + r)}米",
    ]
    if who == "相遇总时间":
        lines = lines[:5] + [lines[4]]
        ans = total
    else:
        lines = lines[:5] + [lines[5]]
        ans = v1 * (t + r)
    return ins, lines, ans


_reg("meet_stopover", meet_stopover)


# 16. 甲乙轮流做工程（甲先，每次1天）→ 总天数
def work_alternating(rng):
    total = rem = t = None
    for _ in range(100):
        a = rng.randint(6, 12)
        b = rng.randint(a + 2, a + 10)
        per = Fraction(1, a) + Fraction(1, b)
        r = int(1 / per)
        rem = 1 - r * per
        if rem == 0:
            total = 2 * r
            break
        t = rem * a
        if t <= 1 and t.denominator <= 6:
            total = 2 * r + t
            break
    else:
        a, b, r, rem, t, total = 10, 15, 6, Fraction(0), Fraction(0), 12
    per = Fraction(1, a) + Fraction(1, b)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程，甲队单独做{a}天完成，乙队单独做{b}天完成。甲、乙两队轮流做，甲队先做，每次各做1天。完成这项工程一共需要多少天？",
        f"甲单独做一项工程要{a}天，乙单独做要{b}天。甲、乙轮流施工，甲先做1天，乙再做1天，这样交替。{name}想知道共需多少天，请你算一算。",
        f"一项工程，甲{a}天完成，乙{b}天完成。甲、乙轮流做，甲先乙后，每次1天。完成工程共需多少天？请列式算一算。",
        f"甲队{a}天完成一项工程，乙队{b}天完成。两队轮流做，甲先做1天、乙再做1天，交替进行。{name}问共需多少天完成？",
    ])
    lines = [
        f"甲队每天完成 = 1 ÷ {a} = 1/{a}",
        f"乙队每天完成 = 1 ÷ {b} = 1/{b}",
        f"甲乙各做1天完成 = 1/{a} + 1/{b} = {num(per)}",
        f"{r}轮完成的工作量 = (1 ÷ {a} + 1 ÷ {b}) × {r} = {num(r * per)}",
    ]
    if rem == 0:
        lines.append(f"总天数 = 2 × {r} = {num(total)}天")
    else:
        lines += [
            f"剩余工作量 = 1 - {num(r * per)} = {num(rem)}",
            f"甲完成剩余需要 = {num(rem)} ÷ (1 ÷ {a}) = {num(t)}天",
            f"总天数 = {2 * r} + {num(t)} = {num(total)}天",
        ]
    return ins, lines, total


_reg("work_alternating", work_alternating)


# 17. 原计划a人b天，工作c天后增d人 → 提前几天
def worker_increase_early(rng):
    early = None
    for _ in range(100):
        a = rng.randint(8, 20)
        b = rng.randint(20, 40)
        c = rng.randint(5, b - 10)
        d = rng.randint(2, 8)
        if (a * (b - c)) % (a + d) == 0:
            rem_days = a * (b - c) // (a + d)
            early = b - c - rem_days
            break
    else:
        a, b, c, d, early = 12, 30, 10, 6, 5
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程原计划{a}人做{b}天完成。工作{c}天后，增加{d}人，每人工作效率相同。这样可以提前几天完成？",
        f"工程队原计划{a}人{b}天完成一项工程。做了{c}天后，又增加{d}人。{name}想知道能提前几天完成，请你算一算。",
        f"原计划{a}人用{b}天完成工程，开工{c}天后增加{d}人。照这样计算，可提前几天完成？请列式算一算。",
        f"一项工程，原计划{a}人做{b}天。工作{c}天后调来{d}人帮忙。{name}问可以提前几天完成？",
    ])
    lines = [
        f"总工作量 = {a} × {b} = {a * b}人天",
        f"{c}天后已完成 = {a} × {c} = {a * c}人天",
        f"剩余工作量 = {a * b} - {a * c} = {a * (b - c)}人天",
        f"增加后人数 = {a} + {d} = {a + d}人",
        f"剩余天数 = {a * (b - c)} ÷ {a + d} = {a * (b - c) // (a + d)}天",
        f"提前天数 = {b} - {c} - {a * (b - c) // (a + d)} = {early}天",
    ]
    return ins, lines, early


_reg("worker_increase_early", worker_increase_early)


# 18. 水池有裂缝，进水管与裂缝同时开 → 注满时间
def pool_leak(rng):
    t = None
    for _ in range(100):
        a = rng.randint(4, 10)
        b = rng.randint(a + 2, a + 8)
        if (a * b) % (b - a) == 0:
            t = a * b // (b - a)
            break
    else:
        a, b, t = 6, 10, 15
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个水池有一条裂缝。单开进水管，{a}小时能注满空池；满池水从裂缝漏完要{b}小时。进水管和裂缝同时开，多少小时能注满空池？",
        f"水池的进水管{a}小时可注满空池，池底裂缝{b}小时可漏完满池水。{name}想知道进水和漏水同时进行时几小时注满，请你算一算。",
        f"一个空水池，开进水管{a}小时注满；水池有裂缝，满池水{b}小时漏完。两管同时开，几小时注满？请列式算一算。",
        f"水池有裂缝，进水管{a}小时注满空池，裂缝{b}小时漏完满池。同时开进水管，几小时能注满？{name}问。",
    ])
    net = Fraction(1, a) - Fraction(1, b)
    lines = [
        f"进水管每小时注水 = 1 ÷ {a} = 1/{a}池",
        f"裂缝每小时漏水 = 1 ÷ {b} = 1/{b}池",
        f"每小时净注水 = 1/{a} - 1/{b} = {num(net)}池",
        f"注满时间 = {a} × {b} ÷ ({b} - {a}) = {t}小时",
    ]
    return ins, lines, t


_reg("pool_leak", pool_leak)


# 19. 盐水反复倒出加水 → 最终浓度/盐
def concentration_repeated_pour(rng):
    a, p, n = rng.choice([
        (200, 40, 2), (200, 40, 3), (200, 60, 2), (400, 40, 2),
        (400, 60, 2), (300, 40, 2), (200, 20, 2), (400, 20, 2),
    ])
    b = a // 2
    orig_salt = a * p // 100
    final_salt = orig_salt // (2 ** n)
    final_pct = Fraction(final_salt, a) * 100
    who = rng.choice(["最终浓度", "最终盐的质量"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一杯{a}克的盐水，含盐率是{p}%。倒出{b}克盐水后再用水加满，这样重复{n}次。这时盐水的{who}是多少？",
        f"有{a}克盐水，浓度为{p}%。每次倒出{b}克后加水补满，共操作{n}次。{name}想知道{who}，请你算一算。",
        f"{a}克盐水含盐率{p}%，倒出{b}克再加水{b}克，重复{n}次。这时{who}是多少？请列式算一算。",
        f"一杯{a}克{p}%的盐水，倒出{b}克后用水加满，再倒出{b}克再用水加满，共{n}次。{name}问{who}是多少？",
    ])
    div2 = " ÷ 2" * n
    lines = [
        f"原有盐的质量 = {a} × {p}/100 = {orig_salt}克",
        f"每次倒出后剩下的盐 = ({a} - {b}) ÷ {a} = 1/2",
        f"倒出{n}次后剩下的盐 = {orig_salt}{div2} = {final_salt}克",
        f"最终浓度 = {final_salt} ÷ {a} × 100 = {num(final_pct)}%",
    ]
    if who == "最终浓度":
        ans = final_pct
    else:
        lines = lines[:3] + [lines[2]]
        ans = final_salt
    return ins, lines, ans


_reg("concentration_repeated_pour", concentration_repeated_pour)


# 20. 两杯盐水互相交换多少克后浓度相等
def two_cups_swap(rng):
    a, b = rng.choice([
        (300, 600), (200, 300), (400, 600), (300, 450),
        (200, 600), (400, 1200), (600, 300), (450, 300),
    ])
    x = a * b // (a + b)
    p = rng.randint(5, 15)
    q = rng.randint(p + 2, p + 10)
    total_salt = Fraction(a * p, 100) + Fraction(b * q, 100)
    c = total_salt / (a + b)
    salt_a = a * c
    diff = salt_a - Fraction(a * p, 100)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲杯有{a}克盐水，含盐率{p}%；乙杯有{b}克盐水，含盐率{q}%。现在从两杯各取出相同质量的盐水互相交换，使两杯浓度相等。各取出多少克？",
        f"甲杯{a}克{p}%的盐水，乙杯{b}克{q}%的盐水。互相交换多少克盐水后，两杯浓度相等？{name}想知道，请你算一算。",
        f"两杯盐水，甲杯{a}克浓度{p}%，乙杯{b}克浓度{q}%。交换多少克（两杯交换量相同）后浓度相等？请列式算一算。",
        f"甲杯盛{a}克{p}%盐水，乙杯盛{b}克{q}%盐水。{name}从两杯各倒出同样多的盐水互换，结果浓度相等。各倒出多少克？",
    ])
    lines = [
        f"两杯盐的总质量 = {a} × {p}/100 + {b} × {q}/100 = {num(total_salt)}克",
        f"相等时每杯浓度 = {num(total_salt)} ÷ ({a} + {b}) = {num(c)}",
        f"甲杯最终盐的质量 = {a} × {num(c)} = {num(salt_a)}克",
        f"甲杯盐的变化 = {num(salt_a)} - {a} × {p}/100 = {num(diff)}克",
        f"交换的盐水质量 = {num(diff)} × 100 ÷ ({q} - {p}) = {x}克",
    ]
    return ins, lines, x


_reg("two_cups_swap", two_cups_swap)


# 21. 按定价卖一部分，剩余打折 → 总利润/利润率
def profit_partial_discount(rng):
    revenue = None
    for _ in range(100):
        c, p = rng.choice([
            (50, 20), (60, 20), (80, 25), (100, 20), (100, 30),
            (40, 25), (80, 50), (60, 50), (100, 40), (60, 25),
        ])
        price = c + c * p // 100
        a = rng.randint(10, 40)
        b = rng.randint(5, 30)
        d = rng.choice([7, 8, 9])
        if price * d % 10 == 0:
            dp = price * d // 10
            revenue = a * price + b * dp
            if revenue > (a + b) * c:
                break
    else:
        c, p, price, a, b, d, dp, revenue = 100, 20, 120, 20, 10, 8, 96, 3360
    cost = (a + b) * c
    profit = revenue - cost
    pct = Fraction(profit, cost) * 100
    who = rng.choice(["总利润", "总利润率"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"商店购进{a + b}件商品，每件成本{c}元，按成本增加{p}%定价。卖出{a}件后，剩下的打{d}折出售。卖完后{who}是多少？",
        f"一件商品成本{c}元，按{p}%的利润定价。商店卖出{a}件后，剩余{b}件打{d}折售完。{name}想知道{who}，请你算一算。",
        f"商店购{a + b}件商品，每件成本{c}元，按{p}%利润定价。卖了{a}件后，余下的打{d}折卖完。{who}是多少？请列式算一算。",
        f"某商品成本{c}元，按成本增{p}%定价。先卖{a}件，剩下{b}件打{d}折出售。{name}问{who}是多少？",
    ])
    lines = [
        f"商品总件数 = {a} + {b} = {a + b}件",
        f"每件定价 = {c} + {c} × {p}/100 = {price}元",
        f"按定价卖{a}件收入 = {a} × {price} = {a * price}元",
        f"打折后单价 = {price} × {d}/10 = {dp}元",
        f"打折卖{b}件收入 = {b} × {dp} = {b * dp}元",
        f"总收入 = {a * price} + {b * dp} = {revenue}元",
        f"总成本 = ({a} + {b}) × {c} = {cost}元",
        f"总利润 = {revenue} - {cost} = {profit}元",
        f"总利润率 = {profit} ÷ {cost} × 100 = {num(pct)}%",
    ]
    if who == "总利润":
        lines = lines[:8] + [lines[7]]
        ans = profit
    else:
        ans = pct
    return ins, lines, ans


_reg("profit_partial_discount", profit_partial_discount)


# 22. 两件商品成本共S，按不同利润率定价出售，总利润P → 各成本
def two_items_discounted(rng):
    c1, c2, p, q = rng.choice([
        (200, 300, 50, 20), (300, 200, 40, 50), (150, 250, 40, 20),
        (400, 100, 50, 30), (250, 350, 40, 20), (180, 220, 50, 25),
    ])
    S = c1 + c2
    P = c1 * p // 100 + c2 * q // 100
    diff = P - S * q // 100
    who = rng.choice(["甲商品成本", "乙商品成本"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两件商品成本共{S}元。甲商品按{p}%的利润定价，乙商品按{q}%的利润定价，都按定价出售，总利润是{P}元。{who}是多少元？",
        f"商店进甲、乙两件商品共花{S}元，甲按{p}%利润定价，乙按{q}%利润定价，全部按定价卖出，共获利{P}元。{name}想知道{who}，请你算一算。",
        f"甲、乙两件商品成本共{S}元，分别按{p}%和{q}%的利润定价出售，总利润{P}元。{who}是多少元？请列式算一算。",
        f"两件商品总成本{S}元，甲按{p}%利润定价，乙按{q}%利润定价，都按定价卖，共赚{P}元。{name}问{who}是多少？",
    ])
    lines = [
        f"假设全是乙商品的总利润 = {S} × {q}/100 = {S * q // 100}元",
        f"实际多赚的利润 = {P} - {S * q // 100} = {diff}元",
        f"甲比乙每件多赚 = {p} - {q} = {p - q}%",
        f"甲商品成本 = {diff} × 100 ÷ ({p} - {q}) = {c1}元",
        f"乙商品成本 = {S} - {c1} = {c2}元",
    ]
    if who == "甲商品成本":
        lines = lines[:4] + [lines[3]]
        ans = c1
    else:
        ans = c2
    return ins, lines, ans


_reg("two_items_discounted", two_items_discounted)


# 23. 存两年定期 vs 存一年再转存 → 利息比较
def interest_two_plans(rng):
    a, r2 = rng.choice([
        (10000, 2), (10000, 3), (10000, 4), (10000, 5),
        (5000, 2), (5000, 4), (5000, 6), (5000, 8),
        (2000, 5), (2000, 10), (4000, 5), (4000, 10), (1000, 10),
    ])
    r1 = rng.choice([2, 3, 4, 5])
    plan1 = a * r1 * 2 // 100
    i1 = a * r2 // 100
    i2 = (a + i1) * r2 // 100
    plan2 = i1 + i2
    diff = plan2 - plan1
    who = rng.choice(["方案一利息", "方案二利息", "利息差"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}有{a}元压岁钱。方案一：直接存2年，年利率{r1}%（单利）；方案二：先存1年，年利率{r2}%，到期后连本带息再存1年。{who}是多少元？",
        f"银行2年期存款年利率{r1}%，1年期年利率{r2}%。{a}元存2年，与先存1年再连本带息转存1年相比，{who}是多少？请你算一算。",
        f"妈妈有{a}元，方案一存2年年利率{r1}%；方案二先存1年年利率{r2}%，到期连本带息再存1年。{who}是多少元？请列式算一算。",
        f"{a}元钱，存2年期（年利率{r1}%）和存两个1年期（年利率{r2}%）相比，{who}是多少？{name}问。",
    ])
    lines = [
        f"方案一利息 = {a} × {r1}/100 × 2 = {plan1}元",
        f"方案二第一年利息 = {a} × {r2}/100 = {i1}元",
        f"方案二第二年利息 = ({a} + {i1}) × {r2}/100 = {i2}元",
        f"方案二利息 = {i1} + {i2} = {plan2}元",
        f"利息差 = {plan2} - {plan1} = {diff}元",
    ]
    if who == "方案一利息":
        lines = lines[:1] + [lines[0]]
        ans = plan1
    elif who == "方案二利息":
        lines = lines[:4] + [lines[3]]
        ans = plan2
    else:
        ans = diff
    return ins, lines, ans


_reg("interest_two_plans", interest_two_plans)


# 24. 三种盐水按质量比混合 → 浓度/盐
def mixture_three(rng):
    p, q, r = rng.sample([10, 15, 20, 25, 30], 3)
    a, b, c = rng.sample([2, 3, 4, 5], 3)
    s1 = Fraction(a * p, 100)
    s2 = Fraction(b * q, 100)
    s3 = Fraction(c * r, 100)
    salt_mass = s1 + s2 + s3
    mass = a + b + c
    conc_pct = salt_mass / mass * 100
    who = rng.choice(["混合浓度", "盐的总质量"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"三种盐水，浓度分别是{p}%、{q}%、{r}%，按{a}比{b}比{c}的质量比混合。混合后盐水的{who}是多少？",
        f"把浓度{p}%、{q}%、{r}%的三种盐水按{a}:{b}:{c}的质量比混合。{name}想知道{who}，请你算一算。",
        f"三种盐水浓度为{p}%、{q}%、{r}%，质量比{a}:{b}:{c}混合。{who}是多少？请列式算一算。",
        f"甲、乙、丙三种盐水浓度分别是{p}%、{q}%、{r}%，按{a}:{b}:{c}混合。{name}问{who}是多少？",
    ])
    lines = [
        f"第一种盐的质量 = {a} × {p}/100 = {num(s1)}克",
        f"第二种盐的质量 = {b} × {q}/100 = {num(s2)}克",
        f"第三种盐的质量 = {c} × {r}/100 = {num(s3)}克",
        f"盐的总质量 = {num(s1)} + {num(s2)} + {num(s3)} = {num(salt_mass)}克",
        f"盐水总质量 = {a} + {b} + {c} = {mass}克",
        f"混合浓度 = {num(salt_mass)} ÷ {mass} × 100 = {num(conc_pct)}%",
    ]
    if who == "混合浓度":
        ans = conc_pct
    else:
        lines = lines[:4] + [lines[3]]
        ans = salt_mass
    return ins, lines, ans


_reg("mixture_three", mixture_three)


# 25. 两块不同比的合金各取同样多熔化 → 新合金铜/锌占比
def alloy_two_mix(rng):
    a, b, c, d = rng.choice([
        (3, 2, 7, 3), (2, 3, 3, 7), (1, 4, 2, 3), (3, 7, 1, 4),
        (1, 2, 1, 3), (2, 1, 3, 2), (2, 3, 7, 3), (3, 1, 2, 3),
    ])
    cu = Fraction(a, a + b) + Fraction(c, c + d)
    zn = Fraction(b, a + b) + Fraction(d, c + d)
    who = rng.choice(["铜", "锌"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两块合金，甲中铜与锌的比是{a}比{b}，乙中铜与锌的比是{c}比{d}。现在从两块合金上各取质量相同的一部分熔化，新合金中{who}占几分之几？",
        f"甲合金铜锌比{a}:{b}，乙合金铜锌比{c}:{d}。各取同样多熔化在一起，{name}想知道新合金中{who}占几分之几，请你算一算。",
        f"两块合金，甲铜锌比{a}:{b}，乙铜锌比{c}:{d}。各取等质量熔化，新合金{who}占几分之几？请列式算一算。",
        f"甲块合金铜与锌比{a}:{b}，乙块铜与锌比{c}:{d}。各取同样多熔化，{name}问新合金中{who}占几分之几？",
    ])
    f1 = Fraction(a, a + b)
    f2 = Fraction(c, c + d)
    g1 = Fraction(b, a + b)
    g2 = Fraction(d, c + d)
    lines = [
        f"甲中铜占 = {a} ÷ ({a} + {b}) = {num(f1)}",
        f"乙中铜占 = {c} ÷ ({c} + {d}) = {num(f2)}",
        f"铜共占 = {num(f1)} + {num(f2)} = {num(cu)}",
        f"锌共占 = {num(g1)} + {num(g2)} = {num(zn)}",
        f"铜占新合金 = {num(cu)} ÷ ({num(cu)} + {num(zn)}) = {num(cu / 2)}",
        f"锌占新合金 = {num(zn)} ÷ ({num(cu)} + {num(zn)}) = {num(zn / 2)}",
    ]
    if who == "铜":
        lines = lines[:5] + [lines[4]]
        ans = cu / 2
    else:
        ans = zn / 2
    return ins, lines, ans


_reg("alloy_two_mix", alloy_two_mix)


# 26. 盐水蒸发后再加盐 → 最终浓度/盐
def evaporation_then_add(rng):
    final_pct = None
    for _ in range(100):
        a = rng.choice([200, 300, 400])
        p = rng.choice([10, 15, 20, 25])
        if a * p % 100 != 0:
            continue
        salt = a * p // 100
        b_evap = rng.randint(20, 80)
        M = a - b_evap
        if M <= 0 or 100 * salt % M != 0:
            continue
        q = 100 * salt // M
        if q >= 100:
            continue
        c_add = rng.randint(5, 30)
        final_pct = Fraction(salt + c_add, M + c_add) * 100
        break
    else:
        a, p, salt, b_evap, M, q, c_add, final_pct = 200, 20, 40, 40, 160, 25, 10, Fraction(500, 17)
    who = rng.choice(["最终浓度", "最终盐的质量"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一杯{a}克的盐水，含盐率{p}%。蒸发掉{b_evap}克水后，含盐率变为{q}%；再加入{c_add}克盐。这时盐水的{who}是多少？",
        f"有{a}克{p}%的盐水，蒸发{b_evap}克水后浓度变成{q}%，又加入{c_add}克盐。{name}想知道{who}，请你算一算。",
        f"{a}克盐水浓度{p}%，蒸发{b_evap}克水后浓度为{q}%，再加{c_add}克盐。{who}是多少？请列式算一算。",
        f"一杯{a}克{p}%的盐水，先蒸发{b_evap}克水（浓度变{q}%），再加入{c_add}克盐。{name}问{who}是多少？",
    ])
    lines = [
        f"原有盐的质量 = {a} × {p}/100 = {salt}克",
        f"蒸发后盐水质量 = {a} - {b_evap} = {M}克",
        f"蒸发后浓度 = {salt} ÷ {M} × 100 = {q}%",
        f"加盐后盐的质量 = {salt} + {c_add} = {salt + c_add}克",
        f"加盐后盐水质量 = {M} + {c_add} = {M + c_add}克",
        f"最终浓度 = ({salt} + {c_add}) ÷ ({M} + {c_add}) × 100 = {num(final_pct)}%",
    ]
    if who == "最终浓度":
        ans = final_pct
    else:
        lines = lines[:4] + [lines[3]]
        ans = salt + c_add
    return ins, lines, ans


_reg("evaporation_then_add", evaporation_then_add)


# 27. 三人合做t天后甲离开 → 乙丙还需几天/总天数
def work_three_leave(rng):
    t2 = None
    for _ in range(100):
        a = rng.randint(6, 12)
        b = rng.randint(8, 15)
        c = rng.randint(10, 20)
        t = rng.randint(2, 4)
        rate3 = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
        done = rate3 * t
        if done >= 1:
            continue
        rem = 1 - done
        rate2 = Fraction(1, b) + Fraction(1, c)
        t2 = rem / rate2
        if t2 > 0 and t2.denominator <= 6:
            break
    else:
        a, b, c, t, t2 = 10, 15, 20, 2, Fraction(34, 7)
    total = t + t2
    who = rng.choice(["还需时间", "总时间"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成，乙单独做{b}天完成，丙单独做{c}天完成。三人合做{t}天后，甲因事离开，由乙、丙继续做。完成工程{who}是多少天？",
        f"甲{a}天、乙{b}天、丙{c}天分别完成一项工程。三人合做{t}天后甲离开，乙丙继续。{name}想知道{who}，请你算一算。",
        f"工程甲{a}天完成，乙{b}天完成，丙{c}天完成。三人合做{t}天，甲走了，乙丙接着做。{who}是多少天？请列式算一算。",
        f"一项工程，甲、乙、丙单独做分别要{a}、{b}、{c}天。三人合做{t}天后甲离开，乙丙继续完成。{name}问{who}是多少？",
    ])
    done = (Fraction(1, a) + Fraction(1, b) + Fraction(1, c)) * t
    rem = 1 - done
    rate2 = Fraction(1, b) + Fraction(1, c)
    lines = [
        f"甲队效率 = 1 ÷ {a} = 1/{a}",
        f"乙队效率 = 1 ÷ {b} = 1/{b}",
        f"丙队效率 = 1 ÷ {c} = 1/{c}",
        f"三人合做{t}天完成 = (1 ÷ {a} + 1 ÷ {b} + 1 ÷ {c}) × {t} = {num(done)}",
        f"剩余工作量 = 1 - {num(done)} = {num(rem)}",
        f"乙丙效率和 = 1/{b} + 1/{c} = {num(rate2)}",
        f"乙丙还需时间 = {num(rem)} ÷ (1/{b} + 1/{c}) = {num(t2)}天",
        f"完成工程总时间 = {t} + {num(t2)} = {num(total)}天",
    ]
    if who == "还需时间":
        lines = lines[:7] + [lines[6]]
        ans = t2
    else:
        ans = total
    return ins, lines, ans


_reg("work_three_leave", work_three_leave)


# 28. 甲管先开t小时，再开乙，齐开s小时后关甲 → 乙还需多久
def pipes_schedule(rng):
    t3 = None
    for _ in range(100):
        a = rng.randint(4, 10)
        b = rng.randint(5, 12)
        t = rng.randint(1, 3)
        s = rng.randint(1, 3)
        phase1 = Fraction(t, a)
        phase2 = s * (Fraction(1, a) + Fraction(1, b))
        rem = 1 - phase1 - phase2
        if rem <= 0:
            continue
        t3 = rem * b
        if t3.denominator <= 6:
            break
    else:
        a, b, t, s, t3 = 6, 8, 1, 2, Fraction(2, 1)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个水池，单开甲管{a}小时注满，单开乙管{b}小时注满。先开甲管{t}小时后，再打开乙管，两管齐开{s}小时后关掉甲管。乙管还要开多少小时才能注满？",
        f"甲管{a}小时注满水池，乙管{b}小时注满。先开甲{t}小时，再开乙，两管一起开{s}小时后关甲。{name}想知道乙管还要开多久，请你算一算。",
        f"水池有甲、乙两管，甲{a}小时满，乙{b}小时满。甲先开{t}小时，然后甲乙齐开{s}小时，再关甲。乙还需几小时注满？请列式算一算。",
        f"注满水池，甲管要{a}小时，乙管要{b}小时。甲先开{t}小时，接着两管齐开{s}小时，然后关甲管。{name}问乙管还要几小时？",
    ])
    phase1 = Fraction(t, a)
    phase2 = s * (Fraction(1, a) + Fraction(1, b))
    rem = 1 - phase1 - phase2
    lines = [
        f"甲管每小时注水 = 1 ÷ {a} = 1/{a}",
        f"乙管每小时注水 = 1 ÷ {b} = 1/{b}",
        f"甲先开{t}小时注水 = 1/{a} × {t} = {num(phase1)}",
        f"两管齐开{s}小时注水 = (1/{a} + 1/{b}) × {s} = {num(phase2)}",
        f"剩余工作量 = 1 - {num(phase1)} - {num(phase2)} = {num(rem)}",
        f"乙管还需时间 = {num(rem)} ÷ (1 ÷ {b}) = {num(t3)}小时",
    ]
    return ins, lines, t3


_reg("pipes_schedule", pipes_schedule)


# 29. 牛吃草：吃t3周后卖掉d头牛 → 剩下的牛还能吃几周
def ox_grazing_sell(rng):
    t4 = None
    for _ in range(200):
        t1 = rng.choice([3, 4, 5, 6])
        t2 = rng.choice([8, 10, 12, 15])
        g0 = math.lcm(t1, t2)
        k = rng.randint(1, 3)
        G = g0 * k
        r = rng.randint(8, 20)
        a = G // t1 + r
        b = G // t2 + r
        c = rng.randint(max(r + 2, a - 8), a + 8)
        t3 = rng.randint(2, 5)
        if c * t3 >= G + r * t3:
            continue
        if c - r - 2 < 2:
            continue
        d = rng.randint(2, c - r - 2)
        den = c - d - r
        if den <= 0:
            continue
        num4 = G + r * t3 - c * t3
        if num4 % den == 0 and num4 > 0:
            t4 = num4 // den
            break
    else:
        t1, t2, G, r, a, b, c, t3, d, t4 = 4, 12, 120, 15, 45, 25, 40, 3, 10, 3
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"牧场上的草每天均匀生长。这片草可供{a}头牛吃{t1}周，或供{b}头牛吃{t2}周。开始有{c}头牛吃了{t3}周后，卖掉{d}头牛，剩下的牛还能吃多少周？",
        f"一片匀速生长的草地，{a}头牛{t1}周吃完，{b}头牛{t2}周吃完。{c}头牛吃了{t3}周后卖掉{d}头，剩下的牛还能吃几周？{name}想知道，请你算一算。",
        f"牧场草匀速生长，{a}头牛可吃{t1}周，{b}头牛可吃{t2}周。{c}头牛吃{t3}周后卖掉{d}头，余下的牛还能吃多少周？请列式算一算。",
        f"一片草地每天长草一样多，{a}头牛{t1}周吃完，{b}头牛{t2}周吃完。{c}头牛吃了{t3}周后卖掉{d}头。{name}问剩下的牛还能吃几周？",
    ])
    num4 = G + r * t3 - c * t3
    den = c - d - r
    lines = [
        f"{a}头牛{t1}周吃的草 = {a} × {t1} = {a * t1}份",
        f"{b}头牛{t2}周吃的草 = {b} × {t2} = {b * t2}份",
        f"每周新长的草 = ({b * t2} - {a * t1}) ÷ ({t2} - {t1}) = {r}份",
        f"牧场原有的草 = {a * t1} - {r} × {t1} = {G}份",
        f"{c}头牛{t3}周吃的草 = {c} × {t3} = {c * t3}份",
        f"这{t3}周新长的草 = {r} × {t3} = {r * t3}份",
        f"剩余的草 = {G} + {r * t3} - {c * t3} = {num4}份",
        f"卖{d}头后剩下的牛 = {c} - {d} = {c - d}头",
        f"吃新草后剩余牛数 = {c - d} - {r} = {den}头",
        f"剩下的牛还能吃 = {num4} ÷ {den} = {t4}周",
    ]
    return ins, lines, t4


_reg("ox_grazing_sell", ox_grazing_sell)


# 30. 蒸发a克水后浓度p%，再加b克盐浓度q% → 原浓度/原质量
def concentration_two_unknown(rng):
    orig_pct = None
    for _ in range(100):
        p = rng.choice([15, 20, 25])
        q = rng.choice([20, 25, 30])
        if q <= p:
            continue
        b_add = rng.choice([5, 10, 15, 20])
        M = b_add * (100 - q) // (q - p)
        if M * p % 100 != 0:
            continue
        salt = M * p // 100
        a_orig = rng.choice([50, 100, 150, 200])
        orig_mass = M + a_orig
        orig_pct = Fraction(salt, orig_mass) * 100
        if orig_pct.denominator <= 20:
            break
    else:
        p, q, b_add, M, salt, a_orig, orig_mass, orig_pct = 20, 25, 10, 150, 30, 50, 200, Fraction(15, 1)
    who = rng.choice(["原浓度", "原盐水质量"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一杯盐水，蒸发掉{a_orig}克水后，含盐率变为{p}%；再加入{b_add}克盐，含盐率变为{q}%。这杯盐水原来的{who}是多少？",
        f"有一杯盐水，蒸发{a_orig}克水后浓度是{p}%，再加{b_add}克盐后浓度是{q}%。{name}想知道原来的{who}，请你算一算。",
        f"一杯盐水先蒸发{a_orig}克水，浓度变{p}%；再加入{b_add}克盐，浓度变{q}%。原来的{who}是多少？请列式算一算。",
        f"盐水蒸发{a_orig}克水后浓度为{p}%，加入{b_add}克盐后浓度为{q}%。{name}问原来的{who}是多少？",
    ])
    lines = [
        f"蒸发后盐水质量 = {M} = {M}克",
        f"蒸发后盐的质量 = {M} × {p}/100 = {salt}克",
        f"加盐后盐的质量 = {salt} + {b_add} = {salt + b_add}克",
        f"加盐后盐水质量 = {M} + {b_add} = {M + b_add}克",
        f"加盐后浓度 = ({salt} + {b_add}) ÷ ({M} + {b_add}) × 100 = {q}%",
        f"原盐水质量 = {M} + {a_orig} = {orig_mass}克",
        f"原浓度 = {salt} ÷ {orig_mass} × 100 = {num(orig_pct)}%",
    ]
    if who == "原浓度":
        ans = orig_pct
    else:
        lines = lines[:6] + [lines[5]]
        ans = orig_mass
    return ins, lines, ans


_reg("concentration_two_unknown", concentration_two_unknown)


# 31. 1到n所有自然数的数字之和
def digit_sum_1_to_n(rng):
    n = rng.choice([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    def pos_sum(p):
        factor = 10 ** p
        higher = n // (factor * 10)
        cur = (n // factor) % 10
        lower = n % factor
        return 45 * higher * factor + cur * (cur - 1) // 2 * factor + cur * (lower + 1)

    s0 = pos_sum(0)
    s1 = pos_sum(1)
    s2 = pos_sum(2)
    s3 = pos_sum(3) if n >= 1000 else 0
    total = s0 + s1 + s2 + s3
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"从1到{n}的所有自然数，各位上的数字之和是多少？",
        f"{name}在练习册上看到一道题：1、2、…、{n}，这些数各位数字相加的总和是多少？请你帮他算一算。",
        f"求1到{n}（含1和{n}）所有自然数的数字之和。请列式算一算。",
        f"把1到{n}的所有整数的各位数字全部加起来，总和是多少？{name}问。",
    ])
    lines = [
        f"个位数字之和 = 45 × {n // 10} × 1 + {(n % 10) * (n % 10 - 1) // 2} × 1 + {n % 10} × 1 = {s0}",
        f"十位数字之和 = 45 × {n // 100} × 10 + {((n // 10) % 10) * (((n // 10) % 10) - 1) // 2} × 10 + {(n // 10) % 10} × 1 = {s1}",
        f"百位数字之和 = 45 × {n // 1000} × 100 + {((n // 100) % 10) * (((n // 100) % 10) - 1) // 2} × 100 + {(n // 100) % 10} × 1 = {s2}",
    ]
    if n >= 1000:
        lines.append(f"千位数字之和 = 45 × 0 × 1000 + 0 × 1000 + 1 × 1 = {s3}")
        lines.append(f"1到{n}的数字总和 = {s0} + {s1} + {s2} + {s3} = {total}")
    else:
        lines.append(f"1到{n}的数字总和 = {s0} + {s1} + {s2} = {total}")
    return ins, lines, total


_reg("digit_sum_1_to_n", digit_sum_1_to_n)


# 32. 1到n中数字"1"出现的次数
def digit_one_count(rng):
    n = rng.choice([100, 200, 300, 500, 1000])

    def count1(x):
        s, factor = 0, 1
        while factor <= x:
            higher = x // (factor * 10)
            cur = (x // factor) % 10
            lower = x % factor
            if cur < 1:
                s += higher * factor
            elif cur == 1:
                s += higher * factor + lower + 1
            else:
                s += (higher + 1) * factor
            factor *= 10
        return s

    total = count1(n)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"从1写到{n}，一共写了多少个数字\"1\"？",
        f"{name}在本子上从1写到{n}，数字\"1\"一共出现了多少次？请你算一算。",
        f"在1到{n}的所有自然数中，数字\"1\"出现了多少次？请列式算一算。",
        f"从1开始写数，一直写到{n}，其中数字\"1\"写了多少个？{name}问。",
    ])
    lines = [
        f"1~99中1的个数 = 10 + 10 = 20个",
    ]
    if n >= 199:
        lines.append(f"100~199中1的个数 = 100 + 10 + 10 = 120个")
        for i in range(2, n // 100):
            lines.append(f"{i * 100}~{i * 100 + 99}中1的个数 = 10 + 10 = 20个")
    elif n == 100:
        lines.append(f"100中1的个数 = 1 = 1个")
    if n == 1000:
        lines.append(f"1000中1的个数 = 1 = 1个")
    parts = [ln.split(" = ")[-1].replace("个", "") for ln in lines]
    lines.append(f"1到{n}中1的总个数 = {' + '.join(parts)} = {total}个")
    return ins, lines, total


_reg("digit_one_count", digit_one_count)


# 33. 1到n中数字"0"出现的次数
def digit_zero_count(rng):
    n = rng.choice([100, 200, 500, 1000])

    def count0(x):
        s, factor = 0, 1
        while factor <= x:
            higher = x // (factor * 10)
            cur = (x // factor) % 10
            lower = x % factor
            if higher == 0:
                pass
            elif cur == 0:
                s += (higher - 1) * factor + lower + 1
            else:
                s += higher * factor
            factor *= 10
        return s

    total = count0(n)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"从1写到{n}，一共写了多少个数字\"0\"？",
        f"{name}在本子上从1写到{n}，数字\"0\"一共出现了多少次？请你算一算。",
        f"在1到{n}的所有自然数中，数字\"0\"出现了多少次？请列式算一算。",
        f"从1开始写数，一直写到{n}，其中数字\"0\"写了多少个？{name}问。",
    ])
    lines = [
        f"1~99中0的个数 = 9 = 9个",
    ]
    if n >= 199:
        lines.append(f"100~199中0的个数 = 10 + 10 = 20个")
        for i in range(2, n // 100):
            lines.append(f"{i * 100}~{i * 100 + 99}中0的个数 = 10 + 10 = 20个")
    if n == 100:
        lines.append(f"100中0的个数 = 2 = 2个")
    elif n == 1000:
        lines.append(f"1000中0的个数 = 3 = 3个")
    elif n % 100 == 0:
        lines.append(f"{n}中0的个数 = 2 = 2个")
    parts = [ln.split(" = ")[-1].replace("个", "") for ln in lines]
    lines.append(f"1到{n}中0的总个数 = {' + '.join(parts)} = {total}个")
    return ins, lines, total


_reg("digit_zero_count", digit_zero_count)


# 34. 1/7 等分数化成小数，小数点后前n位数字之和
def fraction_digit_sum(rng):
    p, q, cycle = rng.choice([
        (1, 7, "142857"), (2, 7, "285714"), (3, 7, "428571"),
        (4, 7, "571428"), (5, 7, "714285"), (6, 7, "857142"),
        (1, 13, "076923"), (2, 13, "153846"),
    ])
    L = len(cycle)
    cycle_sum = sum(int(c) for c in cycle)
    n = rng.randint(20, 200)
    qq = n // L
    r = n % L
    partial = sum(int(c) for c in cycle[:r])
    total = cycle_sum * qq + partial
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"分数 {p}/{q} 化成小数后，小数点后前{n}位数字之和是多少？",
        f"{name}把 {p}/{q} 化成小数，小数点后面前{n}位数字相加，和是多少？请你算一算。",
        f"{p}/{q} 化成小数是一个循环小数，小数点后前{n}位数字之和是多少？请列式算一算。",
        f"计算 {p}/{q} 的小数点后前{n}位数字之和。{name}问。",
    ])
    lines = [
        f"分数 {p}/{q} 的循环节位数 = {L} = {L}位",
        f"一个循环节的数字和 = {' + '.join(cycle)} = {cycle_sum}",
        f"余数 = {n} - {L} × {qq} = {r}",
        f"完整循环节的数字和 = {cycle_sum} × {qq} = {cycle_sum * qq}",
    ]
    if r > 0:
        lines.append(f"余下{r}位的数字和 = {' + '.join(cycle[:r])} = {partial}")
        lines.append(f"数字总和 = {cycle_sum * qq} + {partial} = {total}")
    else:
        lines.append(f"数字总和 = {cycle_sum * qq} = {total}")
    return ins, lines, total


_reg("fraction_digit_sum", fraction_digit_sum)


# 35. 从 base^1 到 base^n 的个位数字之和
def power_digit_sum(rng):
    base = rng.choice([2, 3, 7, 8])
    cycle = {2: [2, 4, 8, 6], 3: [3, 9, 7, 1], 7: [7, 9, 3, 1], 8: [8, 4, 2, 6]}[base]
    n = rng.randint(20, 200)
    qq = n // 4
    r = n % 4
    partial = sum(cycle[:r])
    total = 20 * qq + partial
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"从{base}的1次方到{n}次方，所有结果的个位数字之和是多少？",
        f"{name}计算 {base}^1、{base}^2、…、{base}^{n} 的个位数字，把它们全部加起来，和是多少？请你算一算。",
        f"求{base}的1次方到{n}次方，所有结果的个位数字之和。请列式算一算。",
        f"把{base}的1次方、2次方、…、{n}次方的个位数字相加，总和是多少？{name}问。",
    ])
    lines = [
        f"个位数字一个循环的和 = {cycle[0]} + {cycle[1]} + {cycle[2]} + {cycle[3]} = 20",
        f"余数 = {n} - 4 × {qq} = {r}",
        f"完整循环的个位和 = 20 × {qq} = {20 * qq}",
    ]
    if r > 0:
        lines.append(f"余下{r}个的个位和 = {' + '.join(str(c) for c in cycle[:r])} = {partial}")
        lines.append(f"个位数字总和 = {20 * qq} + {partial} = {total}")
    else:
        lines.append(f"个位数字总和 = {20 * qq} = {total}")
    return ins, lines, total


_reg("power_digit_sum", power_digit_sum)


# 36. 一个数加A是平方数，加B也是平方数 → 这个数
def square_plus_two(rng):
    A = rng.choice([60, 80, 100, 120, 150])
    a = rng.randint(13, 25)
    b = rng.randint(a + 2, a + 8)
    x = a * a - A
    B = b * b - x
    d1 = b - a
    d2 = b + a
    D = d1 * d2
    who = rng.choice(["这个数", "较小的底数", "较大的底数"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个整数加上{A}是一个完全平方数，加上{B}也是一个完全平方数。{who}是多少？",
        f"{name}遇到一道题：某整数加{A}等于一个平方数，加{B}也等于一个平方数。{who}是多少？请你算一算。",
        f"有一个整数，它加上{A}是完全平方数，加上{B}也是完全平方数。{who}是多少？请列式算一算。",
        f"一个整数，加{A}后是某个整数的平方，加{B}后也是某个整数的平方。{name}问{who}是多少？",
    ])
    lines = [
        f"两个平方数的差 = {B} - {A} = {D}",
        f"{D} = {d1} × {d2} = {D}",
        f"较小的底数 = ({d2} - {d1}) ÷ 2 = {a}",
        f"较大的底数 = ({d2} + {d1}) ÷ 2 = {b}",
        f"这个数 = {a} × {a} - {A} = {x}",
    ]
    if who == "这个数":
        ans = x
    elif who == "较小的底数":
        lines = lines[:3] + [lines[2]]
        ans = a
    else:
        lines = lines[:4] + [lines[3]]
        ans = b
    return ins, lines, ans


_reg("square_plus_two", square_plus_two)


# 37. 两个质数的和与积已知 → 两质数
def prime_sum_product(rng):
    q1, q2 = rng.choice([
        (3, 5), (3, 7), (5, 7), (3, 11), (5, 11), (7, 11),
        (3, 13), (5, 13), (7, 13), (11, 13), (2, 3), (2, 5),
        (2, 7), (2, 11), (2, 13),
    ])
    s = q1 + q2
    p = q1 * q2
    d = q2 - q1
    who = rng.choice(["较大的质数", "较小的质数"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"两个质数的和是{s}，积是{p}。{who}是多少？",
        f"{name}在练习册上看到：两个质数相加得{s}，相乘得{p}。{who}是多少？请你算一算。",
        f"已知两个质数的和是{s}，积是{p}。{who}是多少？请列式算一算。",
        f"两个质数，和为{s}，积为{p}。{name}问{who}是多少？",
    ])
    lines = [
        f"两数和的平方 = {s} × {s} = {s * s}",
        f"积的4倍 = 4 × {p} = {4 * p}",
        f"两数差的平方 = {s * s} - {4 * p} = {d * d}",
        f"两数的差 = {d * d} ÷ {d} = {d}",
        f"较大的质数 = ({s} + {d}) ÷ 2 = {q2}",
        f"较小的质数 = ({s} - {d}) ÷ 2 = {q1}",
    ]
    if who == "较大的质数":
        lines = lines[:5] + [lines[4]]
        ans = q2
    else:
        ans = q1
    return ins, lines, ans


_reg("prime_sum_product", prime_sum_product)


# 38. 等差数列 1×4×7×…×k 的积末尾0的个数
def arithmetic_seq_zeros(rng):
    k = rng.choice([31, 40, 49, 58, 67, 76, 85, 94, 103])
    terms = list(range(1, k + 1, 3))

    def v(n, p):
        c = 0
        while n % p == 0:
            c += 1
            n //= p
        return c

    v5_terms = [t for t in terms if t % 5 == 0]
    v5 = sum(v(t, 5) for t in v5_terms)
    v2 = sum(v(t, 2) for t in terms if t % 2 == 0)
    v5_expr = " + ".join(str(v(t, 5)) for t in v5_terms)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"算式 1 × 4 × 7 × … × {k} 的积，末尾有多少个连续的0？",
        f"{name}计算 1 × 4 × 7 × … × {k}，积的末尾有多少个连续的0？请你算一算。",
        f"求 1 × 4 × 7 × … × {k} 的积末尾连续0的个数。请列式算一算。",
        f"1、4、7、…、{k} 这些数相乘，积的末尾有多少个连续的0？{name}问。",
    ])
    lines = [
        f"公差 = 7 - 4 = 3",
        f"这个数列的项数 = ({k} - 1) ÷ 3 + 1 = {len(terms)}项",
        f"含因数5的项数 = ({v5_terms[-1]} - {v5_terms[0]}) ÷ 15 + 1 = {len(v5_terms)}项",
        f"因数5的总个数 = {v5_expr} = {v5}个",
        f"因数2的总个数 = {v2} = {v2}个（更多）",
        f"末尾连续0的个数 = {v5} = {v5}个",
    ]
    return ins, lines, v5


_reg("arithmetic_seq_zeros", arithmetic_seq_zeros)


# 39. 除以a余r1、除以b余r2，在m到n之间有多少个
def remainder_interval_count(rng):
    count = 0
    for _ in range(100):
        a = rng.randint(3, 7)
        b = rng.randint(a + 1, 9)
        r1 = rng.randint(1, a - 1)
        r2 = rng.randint(1, b - 1)
        x0 = None
        for x in range(1, a * b + 1):
            if x % a == r1 and x % b == r2:
                x0 = x
                break
        if x0 is None:
            continue
        L = math.lcm(a, b)
        m = rng.randint(2, 5) * L
        n = m + rng.randint(0, L - 1)
        vals = [x for x in range(m, n + 1) if x % a == r1 and x % b == r2]
        if len(vals) >= 2:
            count = len(vals)
            break
    else:
        a, b, r1, r2, x0, L, m, n, vals, count = 3, 5, 2, 3, 8, 15, 100, 200, [113, 128, 143, 158, 173, 188], 6
    first, last = vals[0], vals[-1]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个数除以{a}余{r1}，除以{b}余{r2}。在{m}到{n}之间（包括{m}和{n}），这样的数有多少个？",
        f"{name}想找除以{a}余{r1}、除以{b}余{r2}的数，在{m}到{n}之间共有多少个？请你算一算。",
        f"某数除以{a}余{r1}，除以{b}余{r2}。{m}到{n}之间（含两端）这样的数有多少个？请列式算一算。",
        f"在{m}到{n}中，除以{a}余{r1}且除以{b}余{r2}的数有多少个？{name}问。",
    ])
    lines = [
        f"循环周期（{a}与{b}的最小公倍数） = {L} = {L}",
        f"满足条件的最小数 = {x0} = {x0}",
        f"范围（{m}到{n}）内第一个数 = {first} = {first}",
        f"范围内最后一个数 = {last} = {last}",
        f"数的个数 = ({last} - {first}) ÷ {L} + 1 = {count}个",
    ]
    return ins, lines, count


_reg("remainder_interval_count", remainder_interval_count)


# 40. n队分两组单循环 + 决赛 → 总场次/小组赛场次
def tournament_hybrid(rng):
    n = rng.choice([8, 12, 16, 20, 24])
    m = n // 2
    group = m * (m - 1) // 2
    total = 2 * group + 1
    who = rng.choice(["总场次", "小组赛场次"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{n}支球队分成两个小组进行单循环赛（每两队赛一场），各组第一名再进行一场决赛。整个比赛{who}是多少场？",
        f"足球赛有{n}支队，分两组单循环，每组第一进决赛。{name}想知道{who}，请你算一算。",
        f"{n}支球队分两组，组内单循环，两组第一决赛一场。{who}是多少场？请列式算一算。",
        f"一次球赛{n}支队，平均分成两组打单循环，再由两组第一名决赛。{name}问{who}是多少？",
    ])
    lines = [
        f"每组队数 = {n} ÷ 2 = {m}支",
        f"每组比赛场次 = {m} × ({m} - 1) ÷ 2 = {group}场",
        f"两个小组共赛 = 2 × {group} = {2 * group}场",
        f"决赛 = 1 = 1场",
        f"比赛总场次 = {2 * group} + 1 = {total}场",
    ]
    if who == "总场次":
        ans = total
    else:
        lines = lines[:3] + [lines[2]]
        ans = 2 * group
    return ins, lines, ans


_reg("tournament_hybrid", tournament_hybrid)


# 41. n人排队，甲乙相邻/不相邻/甲不在两端 → 排法数
def queue_arrangement(rng):
    n = rng.choice([4, 5])
    variant = rng.choice(["相邻", "不相邻", "甲不在两端"])

    def fact(k):
        r = 1
        for i in range(2, k + 1):
            r *= i
        return r

    def fact_expr(k):
        return " × ".join(str(i) for i in range(k, 1, -1)) + " × 1"

    name = rng.choice(NAMES)
    if variant == "相邻":
        f = fact(n - 1)
        total = 2 * f
        ins = rng.choice([
            f"{n}人排队，甲、乙必须相邻，一共有多少种排法？",
            f"{name}和{n - 1}个同学共{n}人排队，甲、乙必须站在一起，有多少种排法？请你算一算。",
            f"{n}个人排成一排，甲、乙两人必须相邻，共有多少种排法？请列式算一算。",
            f"{n}人排队，甲和乙要挨在一起，其余人任意排，有多少种排法？{name}问。",
        ])
        lines = [
            f"排队总人数 = {n} = {n}人",
            f"甲乙捆绑的排法 = 2 = 2种",
            f"捆绑后整体排列 = {fact_expr(n - 1)} = {f}种",
            f"总排法 = 2 × {f} = {total}种",
        ]
    elif variant == "不相邻":
        f = fact(n - 1)
        total = fact(n) - 2 * f
        ins = rng.choice([
            f"{n}人排队，甲、乙不能相邻，一共有多少种排法？",
            f"{name}和{n - 1}个同学共{n}人排队，甲、乙不能站在一起，有多少种排法？请你算一算。",
            f"{n}个人排成一排，甲、乙两人不相邻，共有多少种排法？请列式算一算。",
            f"{n}人排队，甲和乙不能挨在一起，有多少种排法？{name}问。",
        ])
        lines = [
            f"全排列 = {fact_expr(n)} = {fact(n)}种",
            f"甲乙相邻的排法 = 2 × {f} = {2 * f}种",
            f"甲乙不相邻的排法 = {fact(n)} - {2 * f} = {total}种",
        ]
    else:
        f = fact(n - 1)
        total = (n - 2) * f
        ins = rng.choice([
            f"{n}人排队，甲不能站在两端，一共有多少种排法？",
            f"{name}和{n - 1}个同学共{n}人排队，甲不站排头也不站排尾，有多少种排法？请你算一算。",
            f"{n}个人排成一排，甲不在两端，共有多少种排法？请列式算一算。",
            f"{n}人排队，甲不能站两端，其余人任意排，有多少种排法？{name}问。",
        ])
        lines = [
            f"甲的位置选择 = {n} - 2 = {n - 2}种",
            f"其余人的排列 = {fact_expr(n - 1)} = {f}种",
            f"总排法 = {n - 2} × {f} = {total}种",
        ]
    return ins, lines, total


_reg("queue_arrangement", queue_arrangement)


# 42. 用给定数字组成无重复三位数 → 总数/偶数/奇数
def digits_form_numbers(rng):
    digits = rng.choice([[1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5]])
    n = len(digits)
    variant = rng.choice(["三位数总数", "三位偶数", "三位奇数"])
    evens = [d for d in digits if d % 2 == 0]
    odds = [d for d in digits if d % 2 == 1]
    name = rng.choice(NAMES)
    ds = "、".join(str(d) for d in digits)
    ins = rng.choice([
        f"用数字{ds}组成没有重复数字的三位数，{('一共可以组成多少个' if variant == '三位数总数' else '其中' + variant[2:] + '有多少个')}？",
        f"{name}用{ds}这几个数字组成无重复数字的三位数，{('共多少个' if variant == '三位数总数' else variant[2:] + '有多少个')}？请你算一算。",
        f"从{ds}中选3个不同数字组成三位数，{('共多少个' if variant == '三位数总数' else variant[2:] + '有多少个')}？请列式算一算。",
        f"用{ds}组成没有重复数字的三位数，{('一共多少个' if variant == '三位数总数' else '其中' + variant[2:] + '多少个')}？{name}问。",
    ])
    lines = [
        f"可选数字共 = {n} = {n}个（{ds}）",
    ]
    if variant == "三位数总数":
        total = n * (n - 1) * (n - 2)
        lines += [
            f"百位选择 = {n} = {n}种",
            f"十位选择 = {n} - 1 = {n - 1}种",
            f"个位选择 = {n} - 2 = {n - 2}种",
            f"三位数总个数 = {n} × {n - 1} × {n - 2} = {total}个",
        ]
    elif variant == "三位偶数":
        e = len(evens)
        total = e * (n - 1) * (n - 2)
        lines += [
            f"个位选择（偶数） = {e} = {e}种",
            f"百位选择 = {n} - 1 = {n - 1}种",
            f"十位选择 = {n} - 2 = {n - 2}种",
            f"三位偶数个数 = {e} × {n - 1} × {n - 2} = {total}个",
        ]
    else:
        o = len(odds)
        total = o * (n - 1) * (n - 2)
        lines += [
            f"个位选择（奇数） = {o} = {o}种",
            f"百位选择 = {n} - 1 = {n - 1}种",
            f"十位选择 = {n} - 2 = {n - 2}种",
            f"三位奇数个数 = {o} × {n - 1} × {n - 2} = {total}个",
        ]
    return ins, lines, total


_reg("digits_form_numbers", digits_form_numbers)


# 43. 袋中两色球，至少摸多少保证两种颜色都有/k个同色
def drawer_two_colors(rng):
    a = rng.randint(5, 12)
    b = rng.randint(5, 12)
    variant = rng.choice(["两种颜色都有", "k个同色"])
    name = rng.choice(NAMES)
    if variant == "两种颜色都有":
        mx = max(a, b)
        ans = mx + 1
        ins = rng.choice([
            f"袋子里有红球{a}个、蓝球{b}个，至少摸出多少个球，才能保证两种颜色的球都有？",
            f"袋中有红球{a}个、蓝球{b}个。{name}至少摸出多少个球，才能保证摸到两种颜色？请你算一算。",
            f"口袋里有{a}个红球和{b}个蓝球，一次至少摸出多少个，才能保证两种颜色都有？请列式算一算。",
            f"袋中红球{a}个、蓝球{b}个，至少摸多少个才能保证红蓝都有？{name}问。",
        ])
        lines = [
            f"红球个数 = {a} = {a}个",
            f"蓝球个数 = {b} = {b}个",
            f"较多的一种球数 = {mx} = {mx}个",
            f"保证两种颜色都有 = {mx} + 1 = {ans}个",
        ]
    else:
        k = rng.randint(3, min(a, b))
        ans = 2 * (k - 1) + 1
        ins = rng.choice([
            f"袋子里有红球{a}个、蓝球{b}个，至少摸出多少个球，才能保证其中一定有{k}个同色的球？",
            f"袋中有红球{a}个、蓝球{b}个。{name}至少摸出多少个球，才能保证有{k}个颜色相同？请你算一算。",
            f"口袋里有{a}个红球和{b}个蓝球，一次至少摸出多少个，才能保证有{k}个同色？请列式算一算。",
            f"袋中红球{a}个、蓝球{b}个，至少摸多少个才能保证有{k}个同色球？{name}问。",
        ])
        lines = [
            f"红球个数 = {a} = {a}个",
            f"蓝球个数 = {b} = {b}个",
            f"最坏情况每种颜色摸 = {k} - 1 = {k - 1}个",
            f"两种颜色共摸 = 2 × {k - 1} = {2 * (k - 1)}个",
            f"保证{k}个同色 = {2 * (k - 1)} + 1 = {ans}个",
        ]
    return ins, lines, ans


_reg("drawer_two_colors", drawer_two_colors)


# 44. n×n方格中正方形的总数
def square_grid_count(rng):
    n = rng.choice([3, 4, 5, 6])
    total = n * (n + 1) * (2 * n + 1) // 6
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"在{n}×{n}的方格图中，一共可以数出多少个正方形？",
        f"{name}在{n}行{n}列的方格纸上数正方形，一共能数出多少个？请你算一算。",
        f"一个{n}×{n}的方格，大大小小的正方形共有多少个？请列式算一算。",
        f"{n}×{n}的方格图中有多少个正方形？{name}问。",
    ])
    lines = []
    for i in range(1, n + 1):
        cnt = n - i + 1
        lines.append(f"{i}×{i}的正方形 = {cnt} × {cnt} = {cnt * cnt}个")
    lines.append(f"正方形总数 = {n} × ({n} + 1) × (2 × {n} + 1) ÷ 6 = {total}个")
    return ins, lines, total


_reg("square_grid_count", square_grid_count)


# 45. m行n列方格中长方形的总数
def rectangle_grid_count(rng):
    m = rng.choice([2, 3, 4])
    n = rng.choice([2, 3, 4])
    cm = (m + 1) * m // 2
    cn = (n + 1) * n // 2
    total = cm * cn
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"在{m}行{n}列的方格图中，一共可以数出多少个长方形（包括正方形）？",
        f"{name}在{m}行{n}列的方格纸上数长方形，一共能数出多少个？请你算一算。",
        f"一个{m}行{n}列的方格，大大小小的长方形共有多少个？请列式算一算。",
        f"{m}×{n}的方格图中有多少个长方形？{name}问。",
    ])
    lines = [
        f"长边上的线段数 = ({m} + 1) × {m} ÷ 2 = {cm}条",
        f"宽边上的线段数 = ({n} + 1) × {n} ÷ 2 = {cn}条",
        f"长方形总数 = {cm} × {cn} = {total}个",
    ]
    return ins, lines, total


_reg("rectangle_grid_count", rectangle_grid_count)


# 46. 长方体高减少后变正方体，表面积减少 → 原体积
def cuboid_cut_to_cube(rng):
    a = rng.randint(2, 8)
    b = rng.randint(2, 6)
    S = 4 * a * b
    V = b * b * (b + a)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个长方体，如果高减少{a}厘米，就变成一个正方体，这时表面积比原来减少{S}平方厘米。原来长方体的体积是多少立方厘米？",
        f"一个长方体木块，高截去{a}厘米后正好变成一个正方体，表面积减少了{S}平方厘米。{name}想知道原来木块的体积，请你算一算。",
        f"长方体的高减少{a}厘米后成为一个正方体，表面积减少{S}平方厘米。原来长方体的体积是多少？请列式算一算。",
        f"一个长方体高缩短{a}厘米就成了正方体，表面积随之减少{S}平方厘米。原长方体体积是多少？{name}问。",
    ])
    lines = [
        f"减少的每个侧面面积 = {S} ÷ 4 = {S // 4}平方厘米",
        f"正方体的棱长 = {S // 4} ÷ {a} = {b}厘米",
        f"原来的高 = {b} + {a} = {b + a}厘米",
        f"原来的体积 = {b} × {b} × {b + a} = {V}立方厘米",
    ]
    return ins, lines, V


_reg("cuboid_cut_to_cube", cuboid_cut_to_cube)


# 47. 表面涂色的n×n×n正方体切开 → 各类涂色小方块数
def cube_painted(rng):
    n = rng.choice([3, 4, 5, 6])
    variant = rng.choice(["三面", "两面", "一面", "无色"])
    name = rng.choice(NAMES)
    ask = {"三面": "三面涂色", "两面": "两面涂色", "一面": "一面涂色",
           "无色": "六个面都没有涂色"}[variant]
    ins = rng.choice([
        f"一个{n}×{n}×{n}的正方体，六个面都涂上颜色，然后切成棱长为1的小正方体。{ask}的小正方体有多少个？",
        f"把一个棱长为{n}的正方体六个面刷上漆，再切成棱长为1的小方块。{name}想知道{ask}的小方块有多少个，请你算一算。",
        f"正方体棱长为{n}，表面全部涂色后切成1×1×1的小正方体。{ask}的小正方体共有多少个？请列式算一算。",
        f"一个{n}×{n}×{n}的立方体六面涂色后切成小立方体，{ask}的有多少个？{name}问。",
    ])
    lines = [f"正方体的面数 = 6 = 6个"]
    if variant == "三面":
        ans = 8
        lines += [
            f"正方体的棱长 = {n} = {n}",
            f"三面涂色的块数（在顶点处） = 8 = 8个",
        ]
    elif variant == "两面":
        ans = 12 * (n - 2)
        lines += [
            f"每条棱上两面涂色的块数 = {n} - 2 = {n - 2}个",
            f"正方体的棱数 = 12 = 12条",
            f"两面涂色的总块数 = 12 × {n - 2} = {ans}个",
        ]
    elif variant == "一面":
        ans = 6 * (n - 2) ** 2
        lines += [
            f"每个面中间涂色正方形的边长 = {n} - 2 = {n - 2}",
            f"每个面上一面涂色的块数 = {n - 2} × {n - 2} = {(n - 2) ** 2}个",
            f"一面涂色的总块数 = 6 × {(n - 2) ** 2} = {ans}个",
        ]
    else:
        ans = (n - 2) ** 3
        lines += [
            f"内部无色正方体的棱长 = {n} - 2 = {n - 2}",
            f"无色的块数 = {n - 2} × {n - 2} × {n - 2} = {ans}个",
        ]
    return ins, lines, ans


_reg("cube_painted", cube_painted)


# 48. 等底等高的圆柱与圆锥体积差 → 圆柱体积
def cone_cylinder_volume(rng):
    V = rng.randint(2, 10) * 10
    ans = 3 * V // 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个圆柱和一个圆锥等底等高，它们的体积相差{V}立方厘米。圆柱的体积是多少立方厘米？",
        f"等底等高的圆柱和圆锥，体积相差{V}立方厘米。{name}想知道圆柱的体积，请你算一算。",
        f"一个圆锥和一个圆柱底面积和高都相等，已知它们的体积差是{V}立方厘米。圆柱体积是多少？请列式算一算。",
        f"等底等高的圆柱比圆锥体积大{V}立方厘米。圆柱的体积是多少？{name}问。",
    ])
    lines = [
        f"体积相差的份数 = 3 - 1 = 2份",
        f"每份的体积 = {V} ÷ 2 = {V // 2}立方厘米",
        f"圆柱的体积 = 3 × {V // 2} = {ans}立方厘米",
    ]
    return ins, lines, ans


_reg("cone_cylinder_volume", cone_cylinder_volume)


# 49. 铁丝围长方形改围正方形 → 正方形面积
def wire_reshape(rng):
    while True:
        a = rng.randint(5, 20)
        b = rng.randint(5, 20)
        if (a + b) % 2 == 0:
            break
    peri = 2 * (a + b)
    side = (a + b) // 2
    area = side * side
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一根铁丝正好围成一个长{a}厘米、宽{b}厘米的长方形，如果把它改围成一个正方形，正方形的面积是多少平方厘米？",
        f"一根铁丝可以围成长{a}厘米、宽{b}厘米的长方形。{name}把它改围成正方形，面积是多少平方厘米？请你算一算。",
        f"用一根铁丝围成长{a}厘米、宽{b}厘米的长方形后，再改成围成一个最大的正方形。正方形的面积是多少？请列式算一算。",
        f"一根铁丝围成长{a}厘米宽{b}厘米的长方形恰好用完，若围成正方形，面积是多少平方厘米？{name}问。",
    ])
    lines = [
        f"铁丝的长度 = ({a} + {b}) × 2 = {peri}厘米",
        f"正方形的边长 = {peri} ÷ 4 = {side}厘米",
        f"正方形的面积 = {side} × {side} = {area}平方厘米",
    ]
    return ins, lines, area


_reg("wire_reshape", wire_reshape)


# 50. 正方形一组对边各增加a米，面积增加S → 原面积
def square_pair_sides(rng):
    a = rng.randint(2, 10)
    side = rng.randint(3, 12)
    S = a * side
    ans = side * side
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一块正方形地，把它的一组对边各增加{a}米，面积就增加{S}平方米。原来这块地的面积是多少平方米？",
        f"一个正方形，一组对边同时增加{a}米后，面积增加了{S}平方米。{name}想知道原来正方形的面积，请你算一算。",
        f"正方形的一组对边各延长{a}米，得到的长方形面积比原正方形多{S}平方米。原正方形面积是多少？请列式算一算。",
        f"一块正方形菜地，一组对边各增加{a}米，面积增加{S}平方米。原来菜地面积是多少？{name}问。",
    ])
    lines = [
        f"原来的边长 = {S} ÷ {a} = {side}米",
        f"原来的面积 = {side} × {side} = {ans}平方米",
    ]
    return ins, lines, ans


_reg("square_pair_sides", square_pair_sides)


# 51. 三角形底增加a米面积增加S → 原三角形面积
def triangle_base_increase(rng):
    a = rng.randint(2, 10)
    k = rng.randint(2, 8)
    S = k * a
    h = 2 * k
    base = rng.randint(5, 20)
    ans = base * h // 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个三角形的底是{base}米，如果底增加{a}米，面积就增加{S}平方米。原来三角形的面积是多少平方米？",
        f"一个三角形底长{base}米，把底延长{a}米后，面积增加了{S}平方米。{name}想知道原来三角形的面积，请你算一算。",
        f"三角形的底是{base}米，高不变，底增加{a}米后面积增加{S}平方米。原来三角形的面积是多少？请列式算一算。",
        f"一个三角形底{base}米，若底增加{a}米则面积增加{S}平方米。原三角形面积是多少？{name}问。",
    ])
    lines = [
        f"三角形的高 = 2 × {S} ÷ {a} = {h}米",
        f"原来的面积 = {base} × {h} ÷ 2 = {ans}平方米",
    ]
    return ins, lines, ans


_reg("triangle_base_increase", triangle_base_increase)


# 52. 梯形剪去最大平行四边形或最大三角形 → 剩余面积
def trapezoid_cut(rng):
    a = rng.randint(3, 10)
    b = a + rng.randint(2, 8)
    h = rng.randint(2, 6) * 2
    variant = rng.choice(["最大平行四边形", "最大三角形"])
    name = rng.choice(NAMES)
    if variant == "最大平行四边形":
        ans = (b - a) * h // 2
        ins = rng.choice([
            f"一个梯形的上底是{a}厘米，下底是{b}厘米，高是{h}厘米。如果剪去一个最大的平行四边形，剩下的面积是多少平方厘米？",
            f"梯形上底{a}厘米、下底{b}厘米、高{h}厘米。{name}从中剪去一个最大的平行四边形，剩下的面积是多少？请你算一算。",
            f"一个梯形上底{a}厘米，下底{b}厘米，高{h}厘米。剪去一个最大的平行四边形后，剩余部分的面积是多少平方厘米？请列式算一算。",
            f"梯形上底下底分别为{a}厘米和{b}厘米，高{h}厘米。剪去最大平行四边形后还剩多少面积？{name}问。",
        ])
        lines = [
            f"剩下三角形的底 = {b} - {a} = {b - a}厘米",
            f"剩下的面积 = {b - a} × {h} ÷ 2 = {ans}平方厘米",
        ]
    else:
        ans = a * h // 2
        ins = rng.choice([
            f"一个梯形的上底是{a}厘米，下底是{b}厘米，高是{h}厘米。如果剪去一个最大的三角形，剩下的面积是多少平方厘米？",
            f"梯形上底{a}厘米、下底{b}厘米、高{h}厘米。{name}从中剪去一个最大的三角形，剩下的面积是多少？请你算一算。",
            f"一个梯形上底{a}厘米，下底{b}厘米，高{h}厘米。剪去一个最大的三角形后，剩余部分的面积是多少平方厘米？请列式算一算。",
            f"梯形上底下底分别为{a}厘米和{b}厘米，高{h}厘米。剪去最大三角形后还剩多少面积？{name}问。",
        ])
        lines = [
            f"梯形的面积 = ({a} + {b}) × {h} ÷ 2 = {(a + b) * h // 2}平方厘米",
            f"最大三角形的面积 = {b} × {h} ÷ 2 = {b * h // 2}平方厘米",
            f"剩下的面积 = {(a + b) * h // 2} - {b * h // 2} = {ans}平方厘米",
        ]
    return ins, lines, ans


_reg("trapezoid_cut", trapezoid_cut)


# 53. 前n次平均分a，要使n+1次平均分达b → 下一次分数
def avg_target_score(rng):
    n = rng.randint(3, 8)
    a = rng.randint(75, 90)
    b = rng.randint(a + 2, 98)
    ans = (n + 1) * b - n * a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}前{n}次数学测验的平均分是{a}分。他想使前{n + 1}次测验的平均分达到{b}分，下一次测验必须考多少分？",
        f"小明前{n}次考试平均{a}分，要让前{n + 1}次的平均分提高到{b}分，下一次应考多少分？请你算一算。",
        f"{name}前{n}次测验平均成绩是{a}分。若前{n + 1}次的平均分要达到{b}分，他下一次测验需要得多少分？请列式算一算。",
        f"前{n}次数学测试平均分{a}分，{name}希望前{n + 1}次平均分达到{b}分。下一次测试他要考多少分？",
    ])
    lines = [
        f"前{n}次的总分 = {n} × {a} = {n * a}分",
        f"前{n + 1}次的总分 = {n + 1} × {b} = {(n + 1) * b}分",
        f"下一次的分数 = {(n + 1) * b} - {n * a} = {ans}分",
    ]
    return ins, lines, ans


_reg("avg_target_score", avg_target_score)


# 54. 三只猴子依次取走一半又a个，正好取完 → 原桃子数（还原）
def monkey_peaches(rng):
    a = rng.randint(1, 5)
    ans = 14 * a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"三只猴子分一堆桃子。第一只猴子取走一半又{a}个，第二只猴子取走余下的一半又{a}个，第三只猴子再取走余下的一半又{a}个，正好取完。这堆桃子原来有多少个？",
        f"一堆桃子，第一只小猴拿走一半多{a}个，第二只小猴拿走余下的一半多{a}个，第三只小猴拿走余下的一半多{a}个，正好拿完。{name}问这堆桃子原有多少个？请你算一算。",
        f"猴妈妈分桃：第一只取一半又{a}个，第二只取余下的一半又{a}个，第三只再取余下的一半又{a}个，正好分完。原有桃子多少个？请列式算一算。",
        f"三只猴子分桃，每只都取走当时桃子的一半又{a}个，第三只取完后正好没有剩余。原来有多少个桃子？{name}问。",
    ])
    lines = [
        f"第三只猴子取之前的桃子 = {a} × 2 = {2 * a}个",
        f"第二只猴子取之前的桃子 = ({2 * a} + {a}) × 2 = {6 * a}个",
        f"第一只猴子取之前的桃子 = ({6 * a} + {a}) × 2 = {14 * a}个",
    ]
    return ins, lines, ans


_reg("monkey_peaches", monkey_peaches)


# 55. 分数n/d分母加上x后约分得p/q → x
def fraction_denominator_add(rng):
    n, d, p, q = rng.choice([
        (3, 8, 1, 3), (5, 12, 1, 3), (7, 16, 1, 3), (4, 15, 2, 9),
        (3, 10, 1, 4), (7, 20, 1, 4), (5, 18, 1, 4), (9, 25, 3, 10),
        (7, 24, 1, 4),
    ])
    new_d = n * q // p
    ans = new_d - d
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"分数{n}/{d}的分母加上一个数后，约分得{p}/{q}。加上的数是多少？",
        f"把分数{n}/{d}的分母加上某数，约分后是{p}/{q}。{name}想知道加上的数，请你算一算。",
        f"分数{n}/{d}的分母加上多少后，分数值等于{p}/{q}？请列式算一算。",
        f"给{n}/{d}的分母加上一个数，新分数约分后为{p}/{q}。加上的数是多少？{name}问。",
    ])
    lines = [
        f"约分后的分母 = {n} × {q} ÷ {p} = {new_d}",
        f"加上的数 = {new_d} - {d} = {ans}",
    ]
    return ins, lines, ans


_reg("fraction_denominator_add", fraction_denominator_add)


# 56. 两天看书，每天看余下的几分之几多几页，剩r页 → 全书页数（还原）
def pages_read_reverse(rng):
    a, p, b, q, r = rng.choice([
        (4, 6, 3, 5, 15), (3, 8, 2, 4, 12), (5, 12, 2, 6, 14),
        (3, 6, 2, 5, 15), (4, 8, 2, 4, 16), (4, 10, 2, 8, 14),
        (5, 20, 2, 4, 16),
    ])
    y = (r + q) * b // (b - 1)
    ans = (y + p) * a // (a - 1)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}看一本故事书，第一天看了全书的1/{a}多{p}页，第二天看了余下的1/{b}多{q}页，还剩{r}页没看。这本书共有多少页？",
        f"小明看一本书，第一天看全书的1/{a}还多{p}页，第二天看余下的1/{b}还多{q}页，这时还剩{r}页。这本书共多少页？请你算一算。",
        f"一本故事书，第一天看了全书的1/{a}多{p}页，第二天看了剩下的1/{b}多{q}页，还剩{r}页。全书共多少页？请列式算一算。",
        f"{name}读一本书，第一天读全书的1/{a}多{p}页，第二天读余下的1/{b}多{q}页，最后剩{r}页。这本书有多少页？",
    ])
    lines = [
        f"第二天看之前的页数 = ({r} + {q}) × {b} ÷ ({b} - 1) = {y}页",
        f"全书的页数 = ({y} + {p}) × {a} ÷ ({a} - 1) = {ans}页",
    ]
    return ins, lines, ans


_reg("pages_read_reverse", pages_read_reverse)


# 57. 甲给乙a本两人相等；乙给甲b本后甲是乙的k倍 → 原各有多少
def book_swap_equal(rng):
    for _ in range(50):
        k = rng.choice([3, 4, 5])
        a = rng.randint(3, 10)
        b = rng.randint(2, 8)
        total = 2 * a + (k + 1) * b
        if total % (k - 1) == 0:
            y = total // (k - 1)
            break
    else:
        k, a, b = 3, 4, 2
        y = 8
    x = y + 2 * a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人各有一些图书。如果甲给乙{a}本，两人的图书就同样多；如果乙给甲{b}本，甲的图书就是乙的{k}倍。甲原来有多少本？",
        f"甲和乙各有若干本书。甲给乙{a}本后两人本数相等；而乙给甲{b}本后，甲的本数是乙的{k}倍。{name}问甲原来有多少本？请你算一算。",
        f"甲、乙两人的图书，甲给乙{a}本则两人一样多；乙给甲{b}本则甲是乙的{k}倍。甲原有图书多少本？请列式算一算。",
        f"甲给乙{a}本书后两人书数相同，若乙给甲{b}本书，甲的书就是乙的{k}倍。甲原来有多少本书？{name}问。",
    ])
    lines = [
        f"甲比乙多的本数 = 2 × {a} = {2 * a}本",
        f"乙给甲{b}本后甲比乙多 = {2 * a} + 2 × {b} = {2 * a + 2 * b}本",
        f"乙给甲{b}本后乙的本数 = {2 * a + 2 * b} ÷ ({k} - 1) = {y - b}本",
        f"乙原来的本数 = {y - b} + {b} = {y}本",
        f"甲原来的本数 = {y} + {2 * a} = {x}本",
    ]
    return ins, lines, x


_reg("book_swap_equal", book_swap_equal)


# 58. 甲是乙丙和的1/2，乙是甲丙和的1/3，丙有N元 → 甲有多少
def ratio_three_people_fraction(rng):
    k = rng.randint(2, 10)
    N = 5 * k
    ans = 4 * k
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙、丙三人各有一些钱。甲的钱数是乙、丙两人钱数和的1/2，乙的钱数是甲、丙两人钱数和的1/3，丙有{N}元。甲有多少元？",
        f"甲、乙、丙三人攒钱，甲的钱是乙丙总和的一半，乙的钱是甲丙总和的三分之一，已知丙有{N}元。{name}想知道甲有多少元，请你算一算。",
        f"甲的钱数等于乙、丙钱数之和的1/2，乙的钱数等于甲、丙钱数之和的1/3，丙有{N}元。甲有多少元？请列式算一算。",
        f"甲乙丙三人，甲的钱是乙丙之和的一半，乙的钱是甲丙之和的三分之一，丙有{N}元。甲有多少元？{name}问。",
    ])
    lines = [
        f"总钱数是甲的倍数 = 2 + 1 = 3倍",
        f"总钱数是乙的倍数 = 3 + 1 = 4倍",
        f"丙占总钱数的份数 = 12 - 4 - 3 = 5份",
        f"每份的钱数 = {N} ÷ 5 = {k}元",
        f"甲的钱数 = 4 × {k} = {ans}元",
    ]
    return ins, lines, ans


_reg("ratio_three_people_fraction", ratio_three_people_fraction)


# 59. 粗细蜡烛可燃a、b小时，点燃t小时后剩余相等 → 原长比
def candle_length_ratio(rng):
    a, b, t = rng.choice([
        (4, 3, 2), (5, 3, 2), (6, 4, 3), (5, 4, 2), (6, 5, 3),
        (8, 6, 4), (7, 5, 3), (6, 4, 2),
    ])
    rem_a = Fraction(a - t, a)
    rem_b = Fraction(b - t, b)
    ans = rem_b / rem_a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"有两支蜡烛，粗蜡烛全部点完要{a}小时，细蜡烛全部点完要{b}小时。同时点燃{t}小时后，两支蜡烛剩下的长度正好相等。原来粗蜡烛和细蜡烛的长度比是多少？",
        f"一根粗蜡烛可燃{a}小时，一根细蜡烛可燃{b}小时。{name}同时点燃它们，{t}小时后剩下的长度相等。原来两根蜡烛的长度比是多少？请你算一算。",
        f"粗蜡烛点完需{a}小时，细蜡烛点完需{b}小时，同时点燃{t}小时后剩余长度相同。原来粗、细蜡烛的长度比是多少？请列式算一算。",
        f"两支蜡烛，粗的{a}小时燃尽，细的{b}小时燃尽，同时点燃{t}小时后所剩长度相等。原来的长度比（粗比细）是多少？{name}问。",
    ])
    lines = [
        f"粗蜡烛剩下的长度 = 1 - {t}/{a} = {num(rem_a)}",
        f"细蜡烛剩下的长度 = 1 - {t}/{b} = {num(rem_b)}",
        f"粗蜡烛与细蜡烛的原长比 = {num(rem_b)} ÷ ({num(rem_a)}) = {num(ans)}",
    ]
    return ins, lines, ans


_reg("candle_length_ratio", candle_length_ratio)


# 60. 三集合容斥：班n人，三组人数与两两交集 → 都没参加的人数
def three_set_inclusion(rng):
    for _ in range(60):
        n = rng.randint(40, 60)
        g = rng.randint(1, 4)
        d = g + rng.randint(1, 5)
        e = g + rng.randint(1, 5)
        f = g + rng.randint(1, 5)
        a = d + e - g + rng.randint(0, 6)
        b = d + f - g + rng.randint(0, 6)
        c = e + f - g + rng.randint(0, 6)
        union = a + b + c - d - e - f + g
        if union <= n - 5:
            break
    else:
        n, g, d, e, f = 50, 2, 5, 6, 7
        a, b, c = 12, 14, 13
        union = a + b + c - d - e - f + g
    ans = n - union
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"某班有{n}人，参加数学小组的有{a}人，参加语文小组的有{b}人，参加英语小组的有{c}人。同时参加数学和语文的有{d}人，同时参加数学和英语的有{e}人，同时参加语文和英语的有{f}人，三个小组都参加的有{g}人。三个小组都没参加的有多少人？",
        f"一个班{n}名同学中，参加美术组{a}人、音乐组{b}人、体育组{c}人，同时参加美术和音乐的{d}人、美术和体育的{e}人、音乐和体育的{f}人，三组都参加的{g}人。{name}想知道三个组都没参加的人数，请你算一算。",
        f"某班{n}人，参加数学、语文、英语兴趣班的分别有{a}人、{b}人、{c}人，其中同时参加数学与语文的{d}人、数学与英语的{e}人、语文与英语的{f}人，三个都参加的{g}人。三个班都没参加的有多少人？请列式算一算。",
        f"全班{n}人，报数学班{a}人、语文班{b}人、英语班{c}人，两两都报的分别有{d}人、{e}人、{f}人，三个班都报的{g}人。没有报任何班的有多少人？{name}问。",
    ])
    lines = [
        f"至少参加一个小组的人数 = {a} + {b} + {c} - {d} - {e} - {f} + {g} = {union}人",
        f"三个小组都没参加的人数 = {n} - {union} = {ans}人",
    ]
    return ins, lines, ans


_reg("three_set_inclusion", three_set_inclusion)


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
    print(f"L4 ext6 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
