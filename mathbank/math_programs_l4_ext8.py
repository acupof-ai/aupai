#!/usr/bin/env python3
"""L4 ext8: novel structures — boat/current synthesis (two-ship meet with
current cancelling, same-speed offset, airplane headwind), train tunnel
interior time, pole-interval speed, shuttle pickup of a second walking
group, three-team rotation work, efficiency step-up, rainy-day work,
two-cup back-and-forth concentration, simple-vs-compound interest,
discount-interest loans, royalty income tax, insurance payout, auction
commission, coupon-vs-discount comparison, ant shortest paths on cylinder
and cuboid surfaces, goat grazing around a building, cone vertical cut,
circle-in-square / square-in-circle, rotated-body volume, cone net sector
angle, semicircle perimeter, cube corner cut, garden cross path, cylinder
horizontal cut, digit roots, base conversion, divisor sums, last-two-digit
cycles, stars-and-bars, circular permutations, ring coloring, combinations,
derangements, coin ways, ticket types, Nim equalizing, misere take-away,
truth-teller logic, ranking logic, age-order logic, advanced drawer
(pairs/months/cards/gloves), bridge-and-torch, cooking schedule, bucket
measuring, pages containing a digit, marble probability, bicycle gears,
wheel rolling, echo distance, matchstick patterns, dice hidden faces.

Every program: fn(rng) -> (instruction, lines, ans). Lines solve FORWARD from
givens to the asked value (no x variable). All exact arithmetic via Fraction.
Every equation line is chained: 中文标签 = 表达式 = 值[单位].
"""
import math
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


# 1. 上下游两船相向而行，水速在速度和中抵消
def boat_two_ships_meet(rng):
    u1 = rng.randint(12, 24)
    u2 = rng.randint(12, 24)
    v = rng.randint(2, 6)
    m = rng.randint(2, 4)
    s = m * (u1 + u2)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两船分别从上游A港和下游B港同时出发相向而行。甲船在静水中每小时行{u1}千米，乙船在静水中每小时行{u2}千米，水流速度是每小时{v}千米，两港相距{s}千米。两船出发后多少小时相遇？",
        f"A港在上游、B港在下游，相距{s}千米。甲船从A港顺水而下，乙船从B港逆水而上，两船同时出发。甲船静水速度每小时{u1}千米，乙船静水速度每小时{u2}千米，水速每小时{v}千米。{name}想知道两船几小时后相遇，请你算一算。",
        f"甲船从上游码头、乙船从下游码头同时相向开出，两码头相距{s}千米。甲船静水速度{u1}千米/时，乙船静水速度{u2}千米/时，水流速度{v}千米/时。两船多少小时后相遇？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"甲船顺水速度 = {u1} + {v} = {u1 + v}千米/时",
        f"乙船逆水速度 = {u2} - {v} = {u2 - v}千米/时",
        f"两船速度和 = ({u1} + {v}) + ({u2} - {v}) = {u1 + u2}千米/时",
        f"相遇时间 = {s} ÷ {u1 + u2} = {m}小时",
    ]
    return ins, lines, m


_reg("boat_two_ships_meet", boat_two_ships_meet)


# 2. 两船静水速度相同，一顺一逆，相遇点距中点
def boat_same_speed_offset(rng):
    u = rng.randint(15, 30)
    v = rng.randint(3, 8)
    t = rng.randint(2, 5)
    s = 2 * u * t
    ans = v * t
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两船在静水中速度相同，都是每小时{u}千米。甲船顺水、乙船逆水，同时从两港出发，{t}小时后相遇。水流速度是每小时{v}千米。相遇点距两港中点多少千米？",
        f"两船静水速度都是{u}千米/时，甲船顺水行驶、乙船逆水行驶，同时出发后{t}小时相遇，水速{v}千米/时。{name}问相遇点离中点多少千米，请你算一算。",
        f"甲船顺水、乙船逆水同时从两港相向开出，两船静水速度相同，均为{u}千米/时，水流速度{v}千米/时，{t}小时后两船相遇。相遇点距两港中点多少千米？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"甲船顺水速度 = {u} + {v} = {u + v}千米/时",
        f"乙船逆水速度 = {u} - {v} = {u - v}千米/时",
        f"两港间距离 = ({u} + {v}) × {t} + ({u} - {v}) × {t} = {s}千米",
        f"两港距离的一半 = {s} ÷ 2 = {u * t}千米",
        f"甲船行驶的路程 = ({u} + {v}) × {t} = {u * t + ans}千米",
        f"相遇点距中点 = {u * t + ans} - {u * t} = {ans}千米",
    ]
    return ins, lines, ans


_reg("boat_same_speed_offset", boat_same_speed_offset)


# 3. 飞机顺风逆风求风速
def plane_wind(rng):
    pairs = [(2, 3), (3, 4), (2, 4), (3, 5), (4, 5), (2, 5), (3, 6), (4, 6), (5, 6)]
    t1, t2 = rng.choice(pairs)
    lcm = math.lcm(t1, t2)
    k = rng.choice([60, 80, 100, 200, 300, 400])
    s = lcm * k
    a = s // t1
    b = s // t2
    if (a - b) % 2 != 0:
        k *= 2
        s = lcm * k
        a = s // t1
        b = s // t2
    v = (a - b) // 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一架飞机在两个机场之间飞行，顺风飞行{s}千米用了{t1}小时，逆风飞行同样的路程用了{t2}小时。风速是每小时多少千米？",
        f"飞机顺风飞{s}千米需{t1}小时，逆风飞同样路程需{t2}小时。{name}想知道风速是多少，你能算出来吗？",
        f"两个机场相距{s}千米，飞机顺风飞行全程要{t1}小时，逆风飞行全程要{t2}小时。风速是每小时多少千米？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"顺风速度 = {s} ÷ {t1} = {a}千米/时",
        f"逆风速度 = {s} ÷ {t2} = {b}千米/时",
        f"风速 = ({a} - {b}) ÷ 2 = {v}千米/时",
    ]
    return ins, lines, v


_reg("plane_wind", plane_wind)


# 4. 火车完全在隧道内的时间
def train_inside_tunnel(rng):
    v = rng.randint(15, 30)
    l = rng.randint(100, 300)
    m = rng.randint(20, 60)
    t = l + m * v
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一列火车长{l}米，每秒行驶{v}米，前方隧道长{t}米。从车尾进入隧道到车头离开隧道（整列火车完全在隧道内）需要多少秒？",
        f"火车长{l}米，速度{v}米/秒，隧道长{t}米。{name}想知道火车完全在隧道内行驶的时间，请你算一算。",
        f"一列长{l}米的火车以每秒{v}米的速度穿过长{t}米的隧道。火车车身全部在隧道内的时间是多少秒？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"隧道长度 = {t} = {t}米",
        f"火车长度 = {l} = {l}米",
        f"火车完全在隧道内行驶的路程 = {t} - {l} = {m * v}米",
        f"所需时间 = ({t} - {l}) ÷ {v} = {m}秒",
    ]
    return ins, lines, m


_reg("train_inside_tunnel", train_inside_tunnel)


# 5. 电线杆间隔测速
def train_poles(rng):
    k = rng.randint(5, 12)
    n = 2 * k + 1
    t = 5 * k
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"铁路旁每隔50米有一根电线杆。一列火车从第1根电线杆旁驶过，到第{n}根电线杆旁用了{t}秒。这列火车平均每小时行多少千米？",
        f"铁路边每两根电线杆之间相距50米。火车从第1根电线杆行驶到第{n}根电线杆用了{t}秒。{name}想知道火车的速度合每小时多少千米，请你算一算。",
        f"沿铁路每隔50米立一根电线杆。一列火车从第1根电线杆出发，到第{n}根电线杆恰好用了{t}秒。火车平均每小时行多少千米？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"从第1根到第{n}根的间隔数 = {n} - 1 = {2 * k}个",
        f"行驶的路程 = {2 * k} × 50 = {100 * k}米",
        f"火车速度 = {100 * k} ÷ {t} = 20米/秒",
        f"合每小时 = 20 × 3.6 = 72千米",
    ]
    return ins, lines, 72


_reg("train_poles", train_poles)


# 6. 汽车送第一批到终点后返回接第二批
def shuttle_pickup(rng):
    u, v, s, total = rng.choice([
        (30, 6, 15, 70), (40, 10, 20, 66), (30, 10, 15, 60), (48, 12, 24, 66),
        (30, 6, 30, 140), (40, 8, 20, 70), (60, 15, 30, 66), (36, 12, 18, 60),
        (60, 20, 30, 60), (50, 10, 25, 70), (40, 10, 40, 132), (30, 6, 45, 210),
    ])
    m1 = Fraction(s * 60, u)
    w = Fraction(v * m1, 60)
    d = s - w
    m2 = Fraction(d * 60, u + v)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"A、B两地相距{s}千米，一辆汽车和第二批步行的人同时从A地出发。汽车速度是每小时{u}千米，人步行每小时{v}千米。汽车先把第一批人送到B地，立即返回接途中步行的第二批人。第二批人从出发到到达B地一共需要多少分钟？",
        f"两地相距{s}千米，汽车速度{u}千米/时，步行速度{v}千米/时。汽车送第一批人到目的地后马上返回，接到步行的第二批人再开往目的地。{name}问第二批人全程共用多少分钟，请你算一算。",
        f"一辆汽车（速度{u}千米/时）先送一批人去{s}千米外的目的地，到达后立即返回接步行（{v}千米/时）的第二批人。第二批人从出发到到达目的地共需多少分钟？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"汽车送第一批到终点用时 = {s} ÷ {u} × 60 = {m1}分钟",
        f"此时第二批人步行的路程 = {v} × ({m1} ÷ 60) = {w}千米",
        f"汽车返回时与第二批人相距 = {s} - {w} = {d}千米",
        f"相向而行相遇用时 = {d} ÷ ({u} + {v}) × 60 = {m2}分钟",
        f"第二批人一共用时 = {m1} + 2 × {m2} = {total}分钟",
    ]
    return ins, lines, total


_reg("shuttle_pickup", shuttle_pickup)


# 7. 三队按顺序轮流工作
def work_three_rotation(rng):
    cfgs = [
        dict(a=10, b=15, c=20, per="13/60", q=4, qprod="13/15", rem="2/15",
             mids=[("甲做1天后剩余", "2/15 - 1/10", "1/30"),
                   ("乙完成剩余需要", "(1/30) ÷ (1/15)", "1/2")],
             total_lhs="4 × 3 + 1 + 1/2", total="27/2", ans=Fraction(27, 2)),
        dict(a=8, b=12, c=16, per="13/48", q=3, qprod="13/16", rem="3/16",
             mids=[("甲做1天后剩余", "3/16 - 1/8", "1/16"),
                   ("乙完成剩余需要", "(1/16) ÷ (1/12)", "3/4")],
             total_lhs="3 × 3 + 1 + 3/4", total="43/4", ans=Fraction(43, 4)),
        dict(a=8, b=10, c=12, per="37/120", q=3, qprod="37/40", rem="3/40",
             mids=[("甲完成剩余需要", "(3/40) ÷ (1/8)", "3/5")],
             total_lhs="3 × 3 + 3/5", total="48/5", ans=Fraction(48, 5)),
        dict(a=12, b=16, c=24, per="3/16", q=5, qprod="15/16", rem="1/16",
             mids=[("甲完成剩余需要", "(1/16) ÷ (1/12)", "3/4")],
             total_lhs="5 × 3 + 3/4", total="63/4", ans=Fraction(63, 4)),
        dict(a=10, b=20, c=30, per="11/60", q=5, qprod="11/12", rem="1/12",
             mids=[("甲完成剩余需要", "(1/12) ÷ (1/10)", "5/6")],
             total_lhs="5 × 3 + 5/6", total="95/6", ans=Fraction(95, 6)),
        dict(a=15, b=20, c=30, per="3/20", q=6, qprod="9/10", rem="1/10",
             mids=[("甲做1天后剩余", "1/10 - 1/15", "1/30"),
                   ("乙完成剩余需要", "(1/30) ÷ (1/20)", "2/3")],
             total_lhs="6 × 3 + 1 + 2/3", total="59/3", ans=Fraction(59, 3)),
        dict(a=12, b=18, c=9, per="1/4", q=4, exact=True, ans=12),
        dict(a=10, b=12, c=15, per="1/4", q=4, exact=True, ans=12),
    ]
    c = rng.choice(cfgs)
    a, b, cc = c["a"], c["b"], c["c"]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程，甲队单独做{a}天完成，乙队单独做{b}天完成，丙队单独做{cc}天完成。三队按甲、乙、丙的顺序轮流做，每天换一队。完成这项工程需要多少天？",
        f"甲队独做{a}天、乙队独做{b}天、丙队独做{cc}天可分别完成一项工程。现按甲、乙、丙的顺序轮流施工，每队做1天。{name}想知道完成工程共需多少天，请你算一算。",
        f"一项工程，甲{a}天完成、乙{b}天完成、丙{cc}天完成。三队按甲→乙→丙的顺序轮流各做1天。完成这项工程需要多少天？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"三队各做1天完成 = 1/{a} + 1/{b} + 1/{cc} = {c['per']}",
    ]
    if c.get("exact"):
        lines += [
            f"循环轮数 = 1 ÷ ({c['per']}) = {c['q']}轮",
            f"总天数 = {c['q']} × 3 = {3 * c['q']}天",
        ]
    else:
        lines += [
            f"{c['q']}轮完成 = {c['q']} × {c['per']} = {c['qprod']}",
            f"剩余工程量 = 1 - {c['qprod']} = {c['rem']}",
        ]
        for label, lhs, rhs in c["mids"]:
            lines.append(f"{label} = {lhs} = {rhs}天")
        lines.append(f"总天数 = {c['total_lhs']} = {c['total']}天")
    return ins, lines, c["ans"]


_reg("work_three_rotation", work_three_rotation)


# 8. 工程中途效率提高
def work_efficiency_up(rng):
    a = rng.choice([20, 24, 30, 36, 40])
    t = rng.choice([tt for tt in range(a // 4, a) if (a - tt) % 5 == 0])
    new_eff = Fraction(5, 4 * a)
    rem_days = (Fraction(a - t, a)) / new_eff
    actual = t + rem_days
    ans = a - actual
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程，原计划{a}天完成。工作{t}天后，施工队改进技术，工作效率提高了25%。这样可以提前几天完成？",
        f"工程队原计划{a}天完成一项工程，做了{t}天后效率提高25%。{name}想知道能提前几天完成，请你算一算。",
        f"一项工程按原效率需{a}天完成，施工{t}天后效率提高了25%。实际比原计划提前几天完成？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"原计划每天完成 = 1 ÷ {a} = 1/{a}",
        f"工作{t}天完成 = {t} × 1/{a} = {num(Fraction(t, a))}",
        f"剩余工作量 = 1 - {num(Fraction(t, a))} = {num(Fraction(a - t, a))}",
        f"效率提高25%后每天完成 = 1/{a} × 5/4 = {num(new_eff)}",
        f"剩余工作需要 = ({num(Fraction(a - t, a))}) ÷ ({num(new_eff)}) = {num(rem_days)}天",
        f"实际总天数 = {t} + {num(rem_days)} = {num(actual)}天",
        f"提前天数 = {a} - {num(actual)} = {num(ans)}天",
    ]
    return ins, lines, ans


_reg("work_efficiency_up", work_efficiency_up)


# 9. 晴天雨天工程（鸡兔同笼结构）
def work_rainy(rng):
    a, b, t = rng.choice([
        (12, 18, 14), (12, 18, 16), (10, 15, 12), (10, 15, 14), (8, 12, 10),
        (15, 20, 17), (15, 20, 19), (20, 30, 23), (20, 30, 27), (18, 24, 20),
        (18, 24, 22),
    ])
    y = b * (t - a) // (b - a)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程，晴天单独做{a}天完成，雨天单独做{b}天完成（雨天效率低）。施工队一共用了{t}天完成这项工程，施工期间有多少天是雨天？",
        f"某工程晴天{a}天可完成，雨天则需{b}天。工程队施工{t}天恰好完成，{name}想知道其中有几天是雨天，请你算一算。",
        f"一项工程，晴天每天完成1/{a}，雨天每天完成1/{b}。施工队共用{t}天完成，施工期间有多少天是雨天？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"假设{t}天全是晴天完成 = {t} ÷ {a} = {num(Fraction(t, a))}",
        f"比实际多出 = {num(Fraction(t, a))} - 1 = {num(Fraction(t - a, a))}",
        f"每个雨天比晴天少完成 = 1/{a} - 1/{b} = {num(Fraction(1, a) - Fraction(1, b))}",
        f"雨天天数 = ({num(Fraction(t - a, a))}) ÷ ({num(Fraction(1, a) - Fraction(1, b))}) = {y}天",
    ]
    return ins, lines, y


_reg("work_rainy", work_rainy)


# 10. 甲乙两杯盐水互倒
def concentration_backforth(rng):
    a, p, c, b, d = rng.choice([
        (200, 10, 50, 150, 100), (300, 20, 100, 100, 50), (100, 20, 25, 75, 50),
        (300, 10, 100, 100, 50), (500, 8, 125, 125, 125), (200, 20, 50, 150, 100),
    ])
    salt_a = Fraction(a * p, 100)
    salt_a_left = salt_a - Fraction(c * p, 100)
    salt_b = Fraction(c * p, 100)
    sol_b = b + c
    conc_b = salt_b / sol_b
    salt_back = Fraction(d) * conc_b
    salt_a_now = salt_a_left + salt_back
    sol_a_now = a - c + d
    ans = salt_a_now / sol_a_now * 100
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲杯有{a}克浓度为{p}%的盐水，乙杯有{b}克水。先从甲杯倒{c}克盐水到乙杯，搅匀后再从乙杯倒{d}克盐水到甲杯。这时甲杯盐水的浓度是百分之几？",
        f"甲杯盛{a}克{p}%的盐水，乙杯盛{b}克水。把甲杯的{c}克倒入乙杯，搅匀后再把乙杯的{d}克倒回甲杯。{name}想知道甲杯现在的浓度，请你算一算。",
        f"甲杯有{a}克浓度{p}%的盐水，乙杯有{b}克清水。先从甲杯取{c}克放入乙杯搅匀，再从乙杯取{d}克放回甲杯。甲杯盐水浓度变为百分之几？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"甲杯原有盐 = {a} × {p}/100 = {num(salt_a)}克",
        f"倒出{c}克后甲杯剩盐 = {num(salt_a)} - {c} × {p}/100 = {num(salt_a_left)}克",
        f"乙杯盐水质量 = {b} + {c} = {sol_b}克",
        f"乙杯中盐的质量 = {c} × {p}/100 = {num(salt_b)}克",
        f"乙杯盐水浓度 = {num(salt_b)} ÷ {sol_b} = {num(conc_b)}",
        f"倒回{d}克带入盐 = {d} × {num(conc_b)} = {num(salt_back)}克",
        f"甲杯现有盐 = {num(salt_a_left)} + {num(salt_back)} = {num(salt_a_now)}克",
        f"甲杯现有盐水 = {a} - {c} + {d} = {sol_a_now}克",
        f"甲杯浓度 = {num(salt_a_now)} ÷ {sol_a_now} × 100 = {num(ans)}%",
    ]
    return ins, lines, ans


_reg("concentration_backforth", concentration_backforth)


# 11. 单利与复利比较
def interest_compare(rng):
    p, r = rng.choice([
        (10000, 10), (20000, 5), (5000, 20), (8000, 25), (4000, 50),
        (10000, 5), (5000, 10), (20000, 10),
    ])
    simple = p * r // 50
    i1 = p * r // 100
    i2 = (p + i1) * r // 100
    compound = i1 + i2
    ans = compound - simple
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}把{p}元存入银行，年利率是{r}%，存期2年。按复利计算比按单利计算多得利息多少元？",
        f"银行年利率为{r}%，存入{p}元，存2年。{name}想知道复利比单利多得多少利息，请你算一算。",
        f"本金{p}元，年利率{r}%，存2年。单利和复利两种方式下，利息相差多少元？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"单利两年利息 = {p} × {r}/100 × 2 = {simple}元",
        f"复利第一年利息 = {p} × {r}/100 = {i1}元",
        f"复利第二年利息 = {p + i1} × {r}/100 = {i2}元",
        f"复利两年利息 = {i1} + {i2} = {compound}元",
        f"复利比单利多 = {compound} - {simple} = {ans}元",
    ]
    return ins, lines, ans


_reg("interest_compare", interest_compare)


# 12. 贴息贷款（利息先扣）
def loan_deduct(rng):
    p, r, y = rng.choice([
        (20000, 6, 2), (30000, 5, 2), (10000, 8, 3), (50000, 4, 2),
        (25000, 6, 2), (40000, 5, 3),
    ])
    per_year = p * r // 100
    interest = per_year * y
    ans = p - interest
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"某贷款公司办理贴息贷款：借款{p}元，年利率{r}%，借期{y}年，利息在放款时先扣除。{name}实际能拿到多少元？",
        f"借款{p}元，年利率{r}%，期限{y}年，按规定利息先从本金中扣除。{name}实际得到多少元？请你算一算。",
        f"一笔{p}元的贷款，年利率{r}%，借{y}年，利息预先扣除。借款人实际到手多少元？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"每年利息 = {p} × {r}/100 = {per_year}元",
        f"{y}年利息 = {per_year} × {y} = {interest}元",
        f"实际得到 = {p} - {interest} = {ans}元",
    ]
    return ins, lines, ans


_reg("loan_deduct", loan_deduct)


# 13. 稿酬个人所得税
def tax_royalty(rng):
    if rng.random() < 0.5:
        x = rng.choice([2000, 2500, 3000, 3500, 4000])
        base = x - 800
        ans = base * 14 // 100
        name = rng.choice(NAMES)
        ins = rng.choice([
            f"作家获得稿酬{x}元。按规定减除费用800元后，余额按20%的税率缴纳个人所得税，并按应纳税额减征30%。作家应缴纳个人所得税多少元？",
            f"{name}发表文章获得稿酬{x}元。税法规定：稿酬所得减除800元费用后，余额按20%税率纳税，并减征30%。应缴纳个人所得税多少元？",
            f"一笔稿酬{x}元，按规定减除800元费用后，余额的20%为应纳税额，再减征30%。实际应缴纳个人所得税多少元？请列式算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            f"应纳税所得额 = {x} - 800 = {base}元",
            f"应纳税额 = {base} × 20/100 × 70/100 = {ans}元",
        ]
    else:
        x = rng.choice([5000, 6000, 7500, 8000, 10000])
        base = x * 80 // 100
        ans = base * 14 // 100
        name = rng.choice(NAMES)
        ins = rng.choice([
            f"作家获得稿酬{x}元。按规定先减除20%的费用，余额按20%的税率缴纳个人所得税，并按应纳税额减征30%。作家应缴纳个人所得税多少元？",
            f"{name}获得稿酬{x}元。税法规定：稿酬所得先减除20%的费用，余额按20%税率纳税，并减征30%。应缴纳个人所得税多少元？",
            f"一笔稿酬{x}元，按规定先减除20%的费用，余额的20%为应纳税额，再减征30%。实际应缴纳个人所得税多少元？请列式算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            f"应纳税所得额 = {x} × (1 - 20/100) = {base}元",
            f"应纳税额 = {base} × 20/100 × 70/100 = {ans}元",
        ]
    return ins, lines, ans


_reg("tax_royalty", tax_royalty)


# 14. 保险投保与赔偿
def insurance_payout(rng):
    a, b, c = rng.choice([
        (200000, 1, 50000), (300000, 1, 60000), (150000, 2, 30000),
        (500000, 1, 80000), (100000, 3, 20000), (400000, 1, 50000),
    ])
    cover = a * 80 // 100
    fee = cover * b // 100
    pay = c * 80 // 100
    ans = c - pay + fee
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"某户家庭财产价值{a}元，按8成向保险公司投保，保险费率为{b}%。后发生火灾损失{c}元，保险公司按投保比例赔偿。这户人家实际损失多少元？",
        f"{name}家财产价值{a}元，按8成投保，保险费率{b}%。一次火灾损失{c}元，保险公司按投保比例赔付。扣除赔款和保险费后，实际损失多少元？",
        f"家庭财产{a}元，投保金额为8成，年保险费率{b}%。出险后损失{c}元，保险公司按投保比例赔偿。这户人家实际损失多少元？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"保险金额 = {a} × 80/100 = {cover}元",
        f"保险费 = {cover} × {b}/100 = {fee}元",
        f"保险公司赔偿 = {c} × 80/100 = {pay}元",
        f"实际损失 = {c} - {pay} + {fee} = {ans}元",
    ]
    return ins, lines, ans


_reg("insurance_payout", insurance_payout)


# 15. 拍卖佣金
def auction_commission(rng):
    a, b = rng.choice([
        (50000, 10), (80000, 5), (120000, 8), (60000, 12), (200000, 5), (150000, 6),
    ])
    fee = a * b // 100
    ans = a - fee
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一件拍品成交价{a}元，拍卖公司向买卖双方各收取{b}%的佣金。卖方实际收入多少元？",
        f"拍卖会上一件藏品以{a}元成交，拍卖公司向委托方和买受方各收{b}%的佣金。{name}想知道卖方实际收入多少元，请你算一算。",
        f"一件艺术品拍卖成交价{a}元，买卖双方各需支付{b}%的佣金。卖方扣除佣金后实际收入多少元？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"卖方应付佣金 = {a} × {b}/100 = {fee}元",
        f"买方应付佣金 = {a} × {b}/100 = {fee}元",
        f"卖方实际收入 = {a} - {fee} = {ans}元",
    ]
    return ins, lines, ans


_reg("auction_commission", auction_commission)


# 16. 满减与打折比较
def coupon_vs_discount(rng):
    x, a, b, c = rng.choice([
        (380, 300, 100, 75), (500, 400, 120, 8), (300, 200, 50, 85),
        (450, 400, 80, 8), (600, 500, 150, 8), (320, 300, 60, 85),
    ])
    pa = x - b
    p = c if c >= 10 else c * 10
    pb = x * p // 100
    diff = abs(pa - pb)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"商场促销，A方案：满{a}元减{b}元；B方案：全部商品打{c}折。买一件标价{x}元的商品，两种方案的付款金额相差多少元？",
        f"商店有两种优惠：A方案满{a}元减{b}元，B方案全部打{c}折。{name}买标价{x}元的商品，两种方案付款相差多少元？请你算一算。",
        f"商场促销活动：方案一满{a}元减{b}元，方案二全部商品打{c}折。一件标价{x}元的商品，两种方案的付款金额相差多少元？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"超过满减门槛 = {x} - {a} = {x - a}元",
        f"A方案付款 = {x} - {b} = {pa}元",
        f"B方案付款 = {x} × {p}/100 = {pb}元",
        f"两种方案相差 = {max(pa, pb)} - {min(pa, pb)} = {diff}元",
    ]
    return ins, lines, diff


_reg("coupon_vs_discount", coupon_vs_discount)


# 17. 蚂蚁爬圆柱侧面最短路径
def ant_cylinder(rng):
    h, c, d = rng.choice([
        (12, 32, 20), (15, 40, 25), (9, 24, 15), (24, 64, 40), (18, 48, 30),
        (6, 16, 10), (10, 48, 26), (7, 48, 25), (16, 24, 20), (8, 12, 10),
        (4, 6, 5), (24, 14, 25), (30, 32, 34),
    ])
    half = c // 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"圆柱高{h}厘米，底面周长{c}厘米。一只蚂蚁从底面圆周上的A点沿侧面爬到顶面圆周上与A点正对的B点，最短路径是多少厘米？",
        f"一个圆柱，高{h}厘米，底面周长{c}厘米。蚂蚁从下底边缘的A点爬到上底边缘与A正对的B点，只能沿侧面爬。{name}想知道最短路程，请你算一算。",
        f"圆柱的高是{h}厘米，底面周长是{c}厘米。一只蚂蚁从A点（底面圆周）沿侧面爬到与A正对的B点（顶面圆周），最短路径长多少厘米？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"侧面展开后水平距离 = {c} ÷ 2 = {half}厘米",
        f"侧面展开后垂直距离 = {h} = {h}厘米",
        f"最短路径的平方 = {half} × {half} + {h} × {h} = {half * half + h * h}",
        f"最短路径 = {d} = {d}厘米",
    ]
    return ins, lines, d


_reg("ant_cylinder", ant_cylinder)


# 18. 蚂蚁爬长方体表面最短路径
def ant_cuboid(rng):
    a, b, c, d = rng.choice([
        (9, 12, 20, 29), (12, 16, 21, 35), (20, 25, 28, 53), (10, 14, 18, 30),
        (8, 12, 15, 25), (7, 8, 8, 17), (11, 13, 20, 29), (13, 15, 21, 35),
        (22, 23, 28, 53), (9, 15, 18, 30), (14, 10, 18, 30), (16, 8, 18, 30),
    ])
    s1 = a + b
    s2 = a + c
    s3 = b + c
    q1 = s1 * s1 + c * c
    q2 = s2 * s2 + b * b
    q3 = s3 * s3 + a * a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"长方体长{a}厘米、宽{b}厘米、高{c}厘米。一只蚂蚁从一个顶点沿表面爬到对角的顶点，最短路径是多少厘米？",
        f"一个长方体，长{a}厘米、宽{b}厘米、高{c}厘米。蚂蚁从一个顶点沿表面爬到体对角线的另一个顶点。{name}想知道最短路程，请你算一算。",
        f"长方体盒子的长、宽、高分别是{a}厘米、{b}厘米、{c}厘米。一只蚂蚁从一个顶点沿盒子表面爬到对角顶点，最短路径是多少厘米？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"长与宽之和 = {a} + {b} = {s1}厘米",
        f"展开方式一的平方 = {s1} × {s1} + {c} × {c} = {q1}",
        f"展开方式二的平方 = {s2} × {s2} + {b} × {b} = {q2}",
        f"展开方式三的平方 = {s3} × {s3} + {a} × {a} = {q3}",
        f"最短路径 = {d} = {d}厘米",
    ]
    return ins, lines, d


_reg("ant_cuboid", ant_cuboid)


# 19. 羊拴在建筑墙角吃草面积
def goat_corner(rng):
    a, b, l, area = rng.choice([
        (8, 4, 20, 1256), (4, 2, 10, 314), (10, 2, 16, 785), (20, 10, 30, 2512),
        (16, 8, 40, 5024), (20, 10, 50, 7850), (10, 10, 20, 1099),
    ])
    big = Fraction(3, 4) * Fraction(157, 50) * l * l
    r1 = l - a
    r2 = l - b
    s1 = Fraction(1, 4) * Fraction(157, 50) * r1 * r1
    s2 = Fraction(1, 4) * Fraction(157, 50) * r2 * r2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"长方形建筑长{a}米、宽{b}米，一只羊拴在建筑外墙的一角，绳长{l}米。这只羊能吃到草的面积是多少平方米？（π取3.14）",
        f"一座长方形房子长{a}米、宽{b}米，墙角拴着一只羊，绳长{l}米。{name}想知道羊能吃到草的面积，请你算一算。（π取3.14）",
        f"长方形建筑外墙一角拴羊，建筑长{a}米、宽{b}米，拴羊绳长{l}米。羊能吃到的草地面积是多少平方米？（π取3.14）请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"大扇形面积 = 3/4 × 3.14 × {l} × {l} = {num(big)}平方米",
        f"绕过长边后绳长 = {l} - {a} = {r1}米",
        f"小扇形面积 = 1/4 × 3.14 × {r1} × {r1} = {num(s1)}平方米",
        f"绕过宽边后绳长 = {l} - {b} = {r2}米",
        f"小扇形面积 = 1/4 × 3.14 × {r2} × {r2} = {num(s2)}平方米",
        f"吃草总面积 = {num(big)} + {num(s1)} + {num(s2)} = {num(area)}平方米",
    ]
    return ins, lines, area


_reg("goat_corner", goat_corner)


# 20. 圆锥沿高纵切表面积增加
def cone_vertical_cut(rng):
    d, h = rng.choice([
        (10, 12), (8, 15), (12, 10), (6, 8), (10, 15), (14, 10), (16, 15),
        (12, 14), (18, 10), (20, 15),
    ])
    sec = d * h // 2
    ans = d * h
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"圆锥底面直径{d}厘米，高{h}厘米。沿底面直径和高把圆锥切成完全相同的两半，表面积增加多少平方厘米？",
        f"一个圆锥，底面直径{d}厘米，高{h}厘米。{name}沿底面直径和高把它切成两半，表面积增加了多少平方厘米？请你算一算。",
        f"圆锥的底面直径是{d}厘米，高是{h}厘米。沿底面直径和高切开（分成相同的两半），表面积增加多少平方厘米？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"截面三角形的底 = {d} = {d}厘米",
        f"截面三角形的高 = {h} = {h}厘米",
        f"一个截面的面积 = {d} × {h} ÷ 2 = {sec}平方厘米",
        f"表面积共增加 = {sec} × 2 = {ans}平方厘米",
    ]
    return ins, lines, ans


_reg("cone_vertical_cut", cone_vertical_cut)


# 21. 正方形中剪最大圆
def circle_in_square(rng):
    a = rng.choice([10, 20, 8, 12, 4, 6, 16])
    square = a * a
    r = a // 2
    circle = Fraction(157, 50) * r * r
    ans = Fraction(square) - circle
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"在边长{a}厘米的正方形纸上剪一个最大的圆，剩下部分的面积是多少平方厘米？（π取3.14）",
        f"一张边长{a}厘米的正方形纸，剪下一个最大的圆。{name}想知道剩下部分的面积，请你算一算。（π取3.14）",
        f"从边长{a}厘米的正方形中剪去最大的圆，剩余面积是多少平方厘米？（π取3.14）请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"正方形面积 = {a} × {a} = {square}平方厘米",
        f"圆的半径 = {a} ÷ 2 = {r}厘米",
        f"圆的面积 = 3.14 × {r} × {r} = {num(circle)}平方厘米",
        f"剩余面积 = {square} - {num(circle)} = {num(ans)}平方厘米",
    ]
    return ins, lines, ans


_reg("circle_in_square", circle_in_square)


# 22. 圆内最大正方形
def square_in_circle(rng):
    r = rng.choice([10, 5, 20, 4, 8, 2, 6])
    circle = Fraction(157, 50) * r * r
    diag = 2 * r
    square = diag * diag // 2
    ans = circle - square
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"在半径{r}厘米的圆内画一个最大的正方形，正方形以外的面积是多少平方厘米？（π取3.14）",
        f"一个半径{r}厘米的圆，里面画最大的正方形。{name}想知道正方形之外的面积，请你算一算。（π取3.14）",
        f"在半径为{r}厘米的圆中画一个面积最大的正方形，圆内正方形以外的面积是多少平方厘米？（π取3.14）请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"圆的面积 = 3.14 × {r} × {r} = {num(circle)}平方厘米",
        f"正方形对角线 = {r} × 2 = {diag}厘米",
        f"正方形面积 = {diag} × {diag} ÷ 2 = {square}平方厘米",
        f"剩余面积 = {num(circle)} - {square} = {num(ans)}平方厘米",
    ]
    return ins, lines, ans


_reg("square_in_circle", square_in_circle)


# 23. 直角三角形旋转成圆锥
def rotated_cone(rng):
    a, b = rng.choice([
        (4, 3), (6, 4), (8, 6), (3, 4), (10, 6), (12, 5), (9, 4), (6, 5),
        (15, 4), (15, 8), (20, 6), (21, 4), (18, 5), (12, 9), (15, 12),
        (6, 8), (10, 9), (20, 9), (24, 5), (24, 7), (24, 10),
    ])
    vol = Fraction(1, 3) * Fraction(157, 50) * b * b * a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"直角三角形两条直角边分别长{a}厘米和{b}厘米。绕{a}厘米的直角边旋转一周，得到的圆锥体积是多少立方厘米？（π取3.14）",
        f"一个直角三角形，两条直角边为{a}厘米和{b}厘米。以{a}厘米的直角边为轴旋转一周，{name}想知道所得圆锥的体积，请你算一算。（π取3.14）",
        f"直角三角形的两条直角边分别是{a}厘米和{b}厘米，绕{a}厘米的边旋转一周形成圆锥。圆锥的体积是多少立方厘米？（π取3.14）请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"旋转后圆锥的底面半径 = {b} = {b}厘米",
        f"圆锥的高 = {a} = {a}厘米",
        f"圆锥的体积 = 1/3 × 3.14 × {b} × {b} × {a} = {num(vol)}立方厘米",
    ]
    return ins, lines, vol


_reg("rotated_cone", rotated_cone)


# 24. 圆锥侧面展开扇形圆心角
def cone_net_angle(rng):
    l, r = rng.choice([
        (10, 3), (12, 4), (15, 6), (20, 5), (9, 3), (15, 5), (18, 6), (24, 6),
        (30, 10), (10, 4), (12, 3), (15, 9), (20, 12), (10, 6), (8, 3),
        (16, 6), (24, 9), (32, 12), (5, 2), (25, 10),
    ])
    arc = Fraction(157, 50) * 2 * r
    circ = Fraction(157, 50) * 2 * l
    ans = 360 * r // l
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"圆锥的母线长{l}厘米，底面半径{r}厘米。它的侧面展开图是一个扇形，这个扇形的圆心角是多少度？（π取3.14）",
        f"一个圆锥，母线长{l}厘米，底面半径{r}厘米。{name}把它的侧面展开成扇形，扇形的圆心角是多少度？（π取3.14）请你算一算。",
        f"圆锥母线为{l}厘米，底面半径为{r}厘米。侧面展开图（扇形）的圆心角是多少度？（π取3.14）请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"扇形弧长 = 2 × 3.14 × {r} = {num(arc)}厘米",
        f"扇形所在圆的周长 = 2 × 3.14 × {l} = {num(circ)}厘米",
        f"圆心角 = ({num(arc)}) ÷ ({num(circ)}) × 360 = {ans}度",
    ]
    return ins, lines, ans


_reg("cone_net_angle", cone_net_angle)


# 25. 半圆的周长
def semicircle_perimeter(rng):
    r = rng.choice([5, 10, 4, 8, 2, 6, 20, 15, 12, 25])
    arc = Fraction(157, 50) * r
    d = 2 * r
    ans = arc + d
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个半圆的半径是{r}厘米，它的周长是多少厘米？（π取3.14）",
        f"半圆的半径为{r}厘米。{name}想知道它的周长，请你算一算。（π取3.14）",
        f"半径{r}厘米的半圆，周长是多少厘米？（π取3.14）请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"半圆弧的长 = 3.14 × {r} = {num(arc)}厘米",
        f"半圆的直径 = {r} × 2 = {d}厘米",
        f"半圆的周长 = {num(arc)} + {d} = {num(ans)}厘米",
    ]
    return ins, lines, ans


_reg("semicircle_perimeter", semicircle_perimeter)


# 26. 正方体切去一角后的棱数/面数
def cube_cut_corner(rng):
    ask = rng.choice(["棱", "面"])
    name = rng.choice(NAMES)
    if ask == "棱":
        ins = rng.choice([
            "一个正方体，切去它的一个角（切面经过这个顶点相邻的三条棱的中点）。剩下的几何体有多少条棱？",
            f"正方体切去一个角，切面经过该顶点相邻的三条棱的中点。{name}想知道剩下几何体的棱数，请你算一算。",
            "把正方体的一个角切下（切面过相邻三条棱的中点），剩下的几何体有多少条棱？请列式算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            "正方体原有棱数 = 12 = 12条",
            "切面新增棱数 = 3 = 3条",
            "剩下几何体的棱数 = 12 + 3 = 15条",
        ]
        ans = 15
    else:
        ins = rng.choice([
            "一个正方体，切去它的一个角（切面经过这个顶点相邻的三条棱的中点）。剩下的几何体有多少个面？",
            f"正方体切去一个角，切面经过该顶点相邻的三条棱的中点。{name}想知道剩下几何体的面数，请你算一算。",
            "把正方体的一个角切下（切面过相邻三条棱的中点），剩下的几何体有多少个面？请列式算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            "正方体原有面数 = 6 = 6个",
            "切面新增面数 = 1 = 1个",
            "剩下几何体的面数 = 6 + 1 = 7个",
        ]
        ans = 7
    return ins, lines, ans


_reg("cube_cut_corner", cube_cut_corner)


# 27. 十字小路草地面积
def garden_path(rng):
    a, b, w = rng.choice([
        (20, 15, 2), (30, 20, 3), (25, 18, 3), (40, 25, 4), (16, 12, 2),
        (24, 18, 4), (36, 24, 4), (50, 30, 5), (32, 20, 2), (45, 30, 5),
        (28, 20, 4),
    ])
    la = a - w
    lb = b - w
    ans = la * lb
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"长方形草地长{a}米、宽{b}米，中间有两条宽{w}米的十字形小路（一条横着、一条竖着）。草地的面积是多少平方米？",
        f"一块长方形草地长{a}米、宽{b}米，中间修了两条宽{w}米的十字路。{name}想知道草地的面积，请你算一算。",
        f"长方形草地长{a}米、宽{b}米，正中间有一横一竖两条宽{w}米的小路。草地面积是多少平方米？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"草地的长 = {a} - {w} = {la}米",
        f"草地的宽 = {b} - {w} = {lb}米",
        f"草地面积 = {la} × {lb} = {ans}平方米",
    ]
    return ins, lines, ans


_reg("garden_path", garden_path)


# 28. 圆柱横切表面积增加
def cylinder_horizontal_cut(rng):
    r, h = rng.choice([
        (5, 20), (10, 30), (4, 15), (3, 10), (6, 25), (8, 16), (2, 8),
        (20, 40), (15, 30), (12, 24),
    ])
    base = Fraction(157, 50) * r * r
    ans = 2 * base
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"圆柱底面半径{r}厘米，高{h}厘米。平行于底面把它切成两段，表面积增加多少平方厘米？（π取3.14）",
        f"一个圆柱，底面半径{r}厘米，高{h}厘米。{name}平行于底面把它切成两段，表面积增加了多少平方厘米？（π取3.14）请你算一算。",
        f"圆柱底面半径是{r}厘米，高是{h}厘米。沿平行于底面的方向切成两段，表面积增加多少平方厘米？（π取3.14）请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"圆柱的高 = {h} = {h}厘米",
        f"圆柱的底面积 = 3.14 × {r} × {r} = {num(base)}平方厘米",
        f"横切后增加的底面数 = 2 = 2个",
        f"表面积增加 = {num(base)} × 2 = {num(ans)}平方厘米",
    ]
    return ins, lines, ans


_reg("cylinder_horizontal_cut", cylinder_horizontal_cut)


# 29. 数字根
def digit_root(rng):
    while True:
        k = rng.randint(6, 9)
        digits = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(k - 1)]
        n = int("".join(str(d) for d in digits))
        s1 = sum(digits)
        if s1 >= 10:
            break
    s2 = sum(int(d) for d in str(s1))
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"把一个数的各位数字相加，再把所得和的各位数字相加，直到得到一位数。{n}的最后结果是多少？",
        f"对{n}反复求各位数字之和，直到结果是一位数。{name}想知道这个一位数是多少，请你算一算。",
        f"一个数的数字根：把各位数字相加，和再相加，直到一位数。{n}的数字根是多少？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{n}的各位数字之和 = {' + '.join(str(d) for d in digits)} = {s1}",
    ]
    if s2 < 10:
        lines.append(f"{s1}的各位数字之和 = {' + '.join(str(s1))} = {s2}")
        ans = s2
    else:
        s3 = sum(int(d) for d in str(s2))
        lines.append(f"{s1}的各位数字之和 = {' + '.join(str(s1))} = {s2}")
        lines.append(f"{s2}的各位数字之和 = {' + '.join(str(s2))} = {s3}")
        ans = s3
    return ins, lines, ans


_reg("digit_root", digit_root)


# 30. 二进制化十进制
def binary_to_decimal(rng):
    k = rng.randint(5, 6)
    while True:
        bits = "1" + "".join(rng.choice("01") for _ in range(k - 1))
        if "0" in bits[1:]:
            break
    value = 0
    for ch in bits:
        value = value * 2 + int(ch)
    hi = bits[: k - 3]
    lo = bits[k - 3:]
    hi_terms = " + ".join(f"{ch} × {2 ** (k - 1 - i)}" for i, ch in enumerate(hi))
    lo_terms = " + ".join(f"{ch} × {2 ** (k - 1 - (len(hi) + i))}" for i, ch in enumerate(lo))
    hi_val = sum(int(ch) * 2 ** (k - 1 - i) for i, ch in enumerate(hi))
    lo_val = sum(int(ch) * 2 ** (k - 1 - (len(hi) + i)) for i, ch in enumerate(lo))
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"把二进制数{bits}化成十进制数是多少？",
        f"二进制数{bits}化成十进制数是多少？{name}想知道，请你算一算。",
        f"二进制数{bits}对应的十进制数是多少？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{bits}的高位部分 = {hi_terms} = {hi_val}",
        f"{bits}的低位部分 = {lo_terms} = {lo_val}",
        f"十进制数 = {hi_val} + {lo_val} = {value}",
    ]
    return ins, lines, value


_reg("binary_to_decimal", binary_to_decimal)


# 31. 十进制化五进制求数字和
def base5_digitsum(rng):
    while True:
        n = rng.randint(40, 200)
        digits = []
        x = n
        while x > 0:
            digits.append(x % 5)
            x //= 5
        digits = digits[::-1]
        if len(digits) == 3 and sum(digits) >= 3:
            break
    d2, d1, d0 = digits
    ds = sum(digits)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"把十进制数{n}化成五进制数，所得五进制数的各位数字之和是多少？",
        f"十进制数{n}化成五进制数后，各位数字相加的和是多少？{name}想知道，请你算一算。",
        f"将十进制数{n}改写成五进制数，这个五进制数的各位数字之和是多少？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"五进制表示 = {d2} × 25 + {d1} × 5 + {d0} × 1 = {n}",
        f"各位数字之和 = {d2} + {d1} + {d0} = {ds}",
    ]
    return ins, lines, ds


_reg("base5_digitsum", base5_digitsum)


# 32. 一个数所有因数之和
def factor_sum(rng):
    n, factors, sums, _ = rng.choice([
        (12, "2 × 2 × 3", [("1 + 2 + 4", 7), ("1 + 3", 4)], 28),
        (18, "2 × 3 × 3", [("1 + 2", 3), ("1 + 3 + 9", 13)], 39),
        (24, "2 × 2 × 2 × 3", [("1 + 2 + 4 + 8", 15), ("1 + 3", 4)], 60),
        (36, "2 × 2 × 3 × 3", [("1 + 2 + 4", 7), ("1 + 3 + 9", 13)], 91),
        (54, "2 × 3 × 3 × 3", [("1 + 2", 3), ("1 + 3 + 9 + 27", 40)], 120),
        (100, "2 × 2 × 5 × 5", [("1 + 2 + 4", 7), ("1 + 5 + 25", 31)], 217),
        (20, "2 × 2 × 5", [("1 + 2 + 4", 7), ("1 + 5", 6)], 42),
        (28, "2 × 2 × 7", [("1 + 2 + 4", 7), ("1 + 7", 8)], 56),
        (45, "3 × 3 × 5", [("1 + 3 + 9", 13), ("1 + 5", 6)], 78),
        (50, "2 × 5 × 5", [("1 + 2", 3), ("1 + 5 + 25", 31)], 93),
        (98, "2 × 7 × 7", [("1 + 2", 3), ("1 + 7 + 49", 57)], 171),
        (200, "2 × 2 × 2 × 5 × 5", [("1 + 2 + 4 + 8", 15), ("1 + 5 + 25", 31)], 465),
        (16, "2 × 2 × 2 × 2", [("1 + 2 + 4 + 8 + 16", 31)], 31),
        (81, "3 × 3 × 3 × 3", [("1 + 3 + 9 + 27 + 81", 121)], 121),
        (25, "5 × 5", [("1 + 5 + 25", 31)], 31),
        (49, "7 × 7", [("1 + 7 + 49", 57)], 57),
        (125, "5 × 5 × 5", [("1 + 5 + 25 + 125", 156)], 156),
        (64, "2 × 2 × 2 × 2 × 2 × 2", [("1 + 2 + 4 + 8 + 16 + 32 + 64", 127)], 127),
        (32, "2 × 2 × 2 × 2 × 2", [("1 + 2 + 4 + 8 + 16 + 32", 63)], 63),
    ])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{n}的所有因数之和是多少？",
        f"{name}想知道{n}的所有因数（约数）之和是多少，请你算一算。",
        f"求{n}的全部因数之和。请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{n}的质因数分解 = {factors} = {n}",
    ]
    for expr, val in sums:
        lines.append(f"含该质因数的因数和 = {expr} = {val}")
    if len(sums) == 1:
        ans = sums[0][1]
        lines.append(f"所有因数之和 = {sums[0][0]} = {ans}")
    else:
        ans = sums[0][1] * sums[1][1]
        lines.append(f"所有因数之和 = {sums[0][1]} × {sums[1][1]} = {ans}")
    return ins, lines, ans


_reg("factor_sum", factor_sum)


# 33. 7的n次方末两位数字之和
def last_two_digits(rng):
    n = rng.randint(10, 99)
    cycle = {1: (0, 7), 2: (4, 9), 3: (4, 3), 0: (0, 1)}
    r = n % 4
    d1, d2 = cycle[r]
    ans = d1 + d2
    q = n // 4
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"7的{n}次方的末两位数字之和是多少？",
        f"{name}想知道7^{n}的末两位数字之和，请你算一算。",
        f"7的{n}次方（即{n}个7连乘）的末两位数字之和是多少？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"7的幂的末两位周期 = 4 = 4",
        f"{n}除以4的余数 = {n} - {4 * q} = {r}",
        f"末两位数字和 = {d1} + {d2} = {ans}",
    ]
    return ins, lines, ans


_reg("last_two_digits", last_two_digits)


# 34. 隔板法分苹果
def stars_bars(rng):
    a, b = rng.choice([
        (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (6, 4), (7, 4), (8, 4),
        (9, 4), (7, 5), (8, 5), (9, 5), (10, 5), (8, 2), (9, 2), (10, 2),
    ])
    m = a - 1
    k = b - 1
    if k == 1:
        ans = m
        lines = [
            f"苹果之间的空隙 = {a} - 1 = {m}个",
            f"需要的隔板数 = {b} - 1 = 1个",
            f"不同的分法 = {m} = {m}种",
        ]
    elif k == 2:
        ans = m * (m - 1) // 2
        lines = [
            f"苹果之间的空隙 = {a} - 1 = {m}个",
            f"需要的隔板数 = {b} - 1 = 2个",
            f"不同的分法 = {m} × ({m} - 1) ÷ 2 = {ans}种",
        ]
    elif k == 3:
        ans = m * (m - 1) * (m - 2) // 6
        lines = [
            f"苹果之间的空隙 = {a} - 1 = {m}个",
            f"需要的隔板数 = {b} - 1 = 3个",
            f"不同的分法 = {m} × ({m} - 1) × ({m} - 2) ÷ 6 = {ans}种",
        ]
    else:
        ans = m * (m - 1) * (m - 2) * (m - 3) // 24
        lines = [
            f"苹果之间的空隙 = {a} - 1 = {m}个",
            f"需要的隔板数 = {b} - 1 = 4个",
            f"不同的分法 = {m} × ({m} - 1) × ({m} - 2) × ({m} - 3) ÷ 24 = {ans}种",
        ]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"把{a}个相同的苹果分给{b}个小朋友，每人至少分到1个，一共有多少种不同的分法？",
        f"{a}个相同的苹果分给{b}个小朋友，每人至少1个。{name}想知道有多少种分法，请你算一算。",
        f"将{a}个相同的苹果分给{b}个小朋友，要求每人至少分到1个，共有多少种不同的分法？请列式算一算。",
    ]) + rng.choice(_TAILS)
    return ins, lines, ans


_reg("stars_bars", stars_bars)


# 35. 圆桌排列
def circular_table(rng):
    n = rng.choice([4, 5, 6, 7])
    rest = n - 1
    ans = 1
    for i in range(2, n):
        ans *= i
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{n}个人围坐在圆桌旁，一共有多少种不同的坐法？（旋转后相同的算同一种）",
        f"{n}个人围圆桌而坐，旋转后相同的算同一种坐法。{name}想知道共有多少种坐法，请你算一算。",
        f"{n}个人围坐成一圈，有多少种不同的坐法？（旋转后相同算同一种）请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"围坐的人数 = {n} = {n}人",
        f"固定1人的位置 = 1 = 1种",
        f"其余{rest}人的排列 = {' × '.join(str(i) for i in range(rest, 0, -1))} = {ans}种",
    ]
    return ins, lines, ans


_reg("circular_table", circular_table)


# 36. 环形染色
def ring_coloring(rng):
    name = rng.choice(NAMES)
    ins = rng.choice([
        "如图，4个区域围成一圈（区域4与区域1也相邻），用3种颜色给区域染色，相邻区域的颜色不同。一共有多少种染色方法？",
        f"4个区域围成环形（首尾两个区域也相邻），用3种颜色染色，相邻区域颜色不同。{name}想知道共有多少种染法，请你算一算。",
        "把一个圆环分成4个区域，用3种颜色染色，相邻区域颜色不同。共有多少种染色方法？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        "区域1的颜色 = 3 = 3种",
        "区域2的颜色 = 2 = 2种",
        "区域3与区域1同色时区域4的选法 = 1 × 2 = 2种",
        "区域3与区域1不同色时区域4的选法 = 1 × 1 = 1种",
        "区域3、区域4的选法合计 = 2 + 1 = 3种",
        "染色方法总数 = 3 × 2 × 3 = 18种",
    ]
    return ins, lines, 18


_reg("ring_coloring", ring_coloring)


# 37. 组合选人参选
def choose_committee(rng):
    n, m = rng.choice([
        (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (6, 3), (7, 3), (8, 3),
        (9, 3), (10, 3), (8, 4), (9, 4), (10, 4),
    ])
    perm = 1
    for i in range(m):
        perm *= (n - i)
    fact = 1
    for i in range(2, m + 1):
        fact *= i
    ans = perm // fact
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"从{n}名同学中选出{m}人参加活动，一共有多少种不同的选法？",
        f"从{n}名同学中选出{m}人参加比赛。{name}想知道有多少种不同的选法，请你算一算。",
        f"从{n}名同学中选出{m}人组成代表队，共有多少种不同的选法？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"选出{m}人的排列数 = {' × '.join(str(n - i) for i in range(m))} = {perm}",
        f"{m}人的排列数 = {' × '.join(str(i) for i in range(m, 0, -1))} = {fact}",
        f"不同的选法 = {perm} ÷ {fact} = {ans}种",
    ]
    return ins, lines, ans


_reg("choose_committee", choose_committee)


# 38. 错装信封
def wrong_envelopes(rng):
    n = rng.choice([3, 4])
    name = rng.choice(NAMES)
    if n == 3:
        ins = rng.choice([
            "3封信装入3个写好地址的信封，全部装错的装法有多少种？",
            f"3封信和3个写好地址的信封，每封信都装错信封。{name}想知道有多少种装法，请你算一算。",
            "把3封信装入3个对应的信封，要求全部装错，共有多少种装法？请列式算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            "全部装法 = 3 × 2 × 1 = 6种",
            "至少1封装对的装法 = 3 × 2 = 6种",
            "至少2封装对的装法 = 3 = 3种",
            "3封全对的装法 = 1 = 1种",
            "全错装法 = 6 - 6 + 3 - 1 = 2种",
        ]
        ans = 2
    else:
        ins = rng.choice([
            "4封信装入4个写好地址的信封，全部装错的装法有多少种？",
            f"4封信和4个写好地址的信封，每封信都装错信封。{name}想知道有多少种装法，请你算一算。",
            "把4封信装入4个对应的信封，要求全部装错，共有多少种装法？请列式算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            "全部装法 = 4 × 3 × 2 × 1 = 24种",
            "至少1封装对的装法 = 4 × 6 = 24种",
            "至少2封装对的装法 = 6 × 2 = 12种",
            "至少3封装对的装法 = 4 = 4种",
            "4封全对的装法 = 1 = 1种",
            "全错装法 = 24 - 24 + 12 - 4 + 1 = 9种",
        ]
        ans = 9
    return ins, lines, ans


_reg("wrong_envelopes", wrong_envelopes)


# 39. 硬币凑钱
def coin_ways(rng):
    n = rng.randint(8, 15)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"用1分、2分、5分的硬币凑成{n}分（每种硬币都可以用任意枚），一共有多少种不同的凑法？",
        f"储蓄罐里有1分、2分、5分的硬币，要凑出{n}分钱。{name}想知道有多少种凑法，请你算一算。",
        f"用1分、2分、5分硬币凑{n}分，共有多少种不同的凑法？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = []
    total = 0
    for k in range(n // 5 + 1):
        r = n - 5 * k
        twos = r // 2
        ways = twos + 1
        total += ways
        if k == 0:
            lines.append(f"不含5分硬币的凑法 = {twos} + 1 = {ways}种")
        else:
            lines.append(f"含{k}个5分的凑法 = {twos} + 1 = {ways}种")
    lines.append(f"凑成{n}分的凑法总数 = {' + '.join(str((n - 5 * k) // 2 + 1) for k in range(n // 5 + 1))} = {total}种")
    return ins, lines, total


_reg("coin_ways", coin_ways)


# 40. 车站车票种类
def train_tickets(rng):
    n = rng.randint(5, 12)
    ans = n * (n - 1)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一条铁路线上有{n}个车站，需要准备多少种不同的车票？（往返车票视为不同）",
        f"铁路沿线有{n}个车站，任意两站之间都要有车票。{name}想知道共需准备多少种车票，请你算一算。",
        f"一条铁路有{n}个车站，每个车站到其他车站都要有车票，往返票不同。共需准备多少种车票？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"每个车站到其他车站 = {n} - 1 = {n - 1}种",
        f"车票总数 = {n} × {n - 1} = {ans}种",
    ]
    return ins, lines, ans


_reg("train_tickets", train_tickets)


# 41. 尼姆游戏：两堆取子必胜策略
def nim_equalize(rng):
    a = rng.randint(15, 40)
    b = rng.randint(8, a - 1)
    ans = a - b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"两堆石子分别有{a}个和{b}个，两人轮流从其中一堆取任意个（至少1个），取到最后一个的人获胜。先手第一次从多的一堆取多少个就必胜？",
        f"两堆石子{a}个和{b}个，每次从一堆中取任意个，取到最后一个者胜。{name}先手，他第一次取多少个能保证获胜？请你算一算。",
        f"有两堆石子，分别是{a}个和{b}个。两人轮流取，每次从一堆中取1个或多个，取到最后一个的赢。先手第一次从多的一堆取多少个必胜？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"多的一堆 = {a} = {a}个",
        f"少的一堆 = {b} = {b}个",
        f"取后两堆相等 = {b} = {b}个",
        f"先手应取 = {a} - {b} = {ans}个",
    ]
    return ins, lines, ans


_reg("nim_equalize", nim_equalize)


# 42. 巴什博弈（取到最后一个输）
def bash_misere(rng):
    k = rng.randint(2, 5)
    m = rng.randint(3, 12)
    r = rng.randint(1, k)
    n = m * (k + 1) + r + 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一堆石子共{n}个，两人轮流取，每次取1到{k}个，取到最后一个的人输。先手第一次取多少个才能保证获胜？",
        f"有{n}个石子，每次取1~{k}个，取到最后一个者输。{name}先手，第一次取几个必胜？请你算一算。",
        f"两人轮流取石子，共{n}个，每次取1至{k}个，取到最后一个的人输。先手第一次取多少个有必胜策略？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"每轮两人取子的和 = {k} + 1 = {k + 1}个",
        f"先手取{r}个后剩下 = {n} - {r} = {n - r}个",
        f"以后每轮取走 = {k + 1} = {k + 1}个",
        f"最后留给对手 = 1 = 1个",
        f"先手第一次应取 = {r} = {r}个",
    ]
    return ins, lines, r


_reg("bash_misere", bash_misere)


# 43. 真话假话：只有一人说真话
def truth_one_true(rng):
    patterns = [
        (("acc", 2), ("acc", 3), ("not", 2), ("deny",),
         [[3, 4], [1, 3, 4], [2, 4], [3]], 3),
        (("acc", 2), ("acc", 4), ("deny",), ("not", 2),
         [[3, 4], [1, 3, 4], [4], [2, 3]], 4),
        (("acc", 2), ("acc", 3), ("not", 2), ("acc", 3),
         [[3], [1, 3], [2, 4], [2, 4]], 3),
        (("acc", 2), ("acc", 4), ("not", 2), ("deny",),
         [[3, 4], [1, 3, 4], [3, 4], [3]], 3),
        (("acc", 4), ("acc", 3), ("deny",), ("not", 2),
         [[3, 4], [2, 3], [4], [1, 3, 4]], 4),
        (("acc", 2), ("deny",), ("acc", 4), ("not", 1),
         [[2, 4], [1], [2, 4], [2, 3, 4]], 1),
        (("acc", 2), ("acc", 4), ("deny",), ("not", 1),
         [[3, 4], [1, 3], [4], [2, 3, 4]], 4),
    ]
    s1, s2, s3, s4, scenarios, truth = rng.choice(patterns)

    def stmt_text(s):
        if s[0] == "acc":
            return f"是{s[1]}号打破的"
        if s[0] == "deny":
            return "不是我打破的"
        return f"{s[1]}号说的不对"

    name = rng.choice(NAMES)
    ins = rng.choice([
        f"教室里的玻璃被打破了，老师询问4位同学。1号说：\"{stmt_text(s1)}。\"2号说：\"{stmt_text(s2)}。\"3号说：\"{stmt_text(s3)}。\"4号说：\"{stmt_text(s4)}。\"已知只有1人说真话，说真话的是几号？",
        f"4位同学中有人打破了玻璃。1号说：\"{stmt_text(s1)}。\"2号说：\"{stmt_text(s2)}。\"3号说：\"{stmt_text(s3)}。\"4号说：\"{stmt_text(s4)}。\"{name}发现只有1人说真话。说真话的是几号？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = []
    for i in range(4):
        tellers = scenarios[i]
        t_str = "、".join(f"{t}号" for t in tellers)
        ones = " + ".join("1" for _ in tellers)
        lines.append(f"假设是{i + 1}号做的，说真话的有{t_str} = {ones} = {len(tellers)}人")
    lines.append(f"说真话的是{truth}号 = {truth} = {truth}号")
    return ins, lines, truth


_reg("truth_one_true", truth_one_true)


# 44. 真话假话：只有一人说假话
def truth_one_false(rng):
    patterns = [
        (("acc", 2), ("acc", 3), ("not", 2), ("deny",),
         [[3, 4], [1, 3, 4], [2, 4], [3]], 2),
        (("acc", 2), ("acc", 4), ("deny",), ("not", 2),
         [[3, 4], [1, 3, 4], [4], [2, 3]], 2),
        (("acc", 2), ("acc", 3), ("deny",), ("not", 1),
         [[3, 4], [1, 3], [2, 4], [3, 4]], 2),
        (("acc", 3), ("acc", 4), ("deny",), ("acc", 2),
         [[3], [3, 4], [1, 4], [2, 3]], 3),
        (("acc", 3), ("acc", 2), ("not", 1), ("deny",),
         [[3, 4], [2, 3, 4], [1, 4], [3]], 2),
        (("acc", 4), ("acc", 2), ("deny",), ("not", 1),
         [[3, 4], [2, 3, 4], [4], [3]], 2),
    ]
    s1, s2, s3, s4, scenarios, doer = rng.choice(patterns)

    def stmt_text(s):
        if s[0] == "acc":
            return f"是{s[1]}号打破的"
        if s[0] == "deny":
            return "不是我打破的"
        return f"{s[1]}号说的不对"

    name = rng.choice(NAMES)
    ins = rng.choice([
        f"教室里的玻璃被打破了，老师询问4位同学。1号说：\"{stmt_text(s1)}。\"2号说：\"{stmt_text(s2)}。\"3号说：\"{stmt_text(s3)}。\"4号说：\"{stmt_text(s4)}。\"已知只有1人说假话，打破玻璃的是几号？",
        f"4位同学中有人打破了玻璃。1号说：\"{stmt_text(s1)}。\"2号说：\"{stmt_text(s2)}。\"3号说：\"{stmt_text(s3)}。\"4号说：\"{stmt_text(s4)}。\"{name}发现只有1人说假话。打破玻璃的是几号？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = []
    for i in range(4):
        tellers = scenarios[i]
        t_str = "、".join(f"{t}号" for t in tellers)
        ones = " + ".join("1" for _ in tellers)
        lines.append(f"假设是{i + 1}号做的，说真话的有{t_str} = {ones} = {len(tellers)}人")
    lines.append(f"说假话的是{doer}号 = {doer} = {doer}号")
    return ins, lines, doer


_reg("truth_one_false", truth_one_false)


# 45. 比赛名次推理
def logic_ranking(rng):
    pat = rng.choice(["C", "D", "E", "F"])
    name = rng.choice(NAMES)
    if pat == "C":
        ins = rng.choice([
            "甲、乙、丙三人比赛跑步，分获第1、2、3名。甲说：\"我不是第1名。\"乙说：\"我不是第2名。\"丙说：\"我不是第3名。\"三人说的都对，并且甲不是第3名。乙是第几名？",
            f"甲、乙、丙三人比赛，名次分别是第1、2、3名。甲不是第1名，乙不是第2名，丙不是第3名，甲也不是第3名。{name}想知道乙是第几名，请你算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            "甲的名次 = 2 = 2名",
            "丙的名次 = 1 = 1名",
            "乙的名次 = 3 = 3名",
        ]
        ans = 3
    elif pat == "D":
        ins = rng.choice([
            "甲、乙、丙、丁四人比赛，分获1~4名。已知丙是第1名，甲和乙的名次相邻，丁不是第2名。丁是第几名？",
            f"甲、乙、丙、丁四人比赛，名次为第1到第4名。丙第1，甲、乙名次相邻，丁不是第2名。{name}想知道丁是第几名，请你算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            "丙的名次 = 1 = 1名",
            "甲、乙占第2和第3 = 2 = 2名",
            "丁的名次 = 4 = 4名",
        ]
        ans = 4
    elif pat == "E":
        ins = rng.choice([
            "甲、乙、丙、丁、戊五人比赛，分获1~5名。已知戊是第1名，甲是第2名，乙和丙的名次相邻，丁不是第3名。丁是第几名？",
            f"五人比赛，名次为第1到第5名。戊第1，甲第2，乙、丙名次相邻，丁不是第3名。{name}想知道丁是第几名，请你算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            "戊的名次 = 1 = 1名",
            "甲的名次 = 2 = 2名",
            "乙、丙占第3和第4 = 2 = 2名",
            "丁的名次 = 5 = 5名",
        ]
        ans = 5
    else:
        ins = rng.choice([
            "甲、乙、丙、丁四人比赛，分获1~4名。已知丁是第4名，甲和乙的名次相邻，丙不是第1名。丙是第几名？",
            f"四人比赛，名次为第1到第4名。丁第4，甲、乙名次相邻，丙不是第1名。{name}想知道丙是第几名，请你算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            "丁的名次 = 4 = 4名",
            "甲、乙占第1和第2 = 2 = 2名",
            "丙的名次 = 3 = 3名",
        ]
        ans = 3
    return ins, lines, ans


_reg("logic_ranking", logic_ranking)


# 46. 抽屉原理：保证n双同色袜子
def drawer_pairs(rng):
    n = rng.choice([2, 3, 4])
    a = rng.randint(2 * n + 1, 2 * n + 8)
    b = rng.randint(2 * n + 1, 2 * n + 8)
    ans = 4 * n - 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"袋子里有黑色袜子{a}只、白色袜子{b}只（不分左右）。至少摸出多少只，才能保证其中一定有{n}双同色的袜子？（一双=2只）",
        f"袋中有黑袜子{a}只、白袜子{b}只。{name}闭着眼睛摸，至少摸出多少只才能保证有{n}双同色袜子？请你算一算。",
        f"布袋里放着黑袜子{a}只、白袜子{b}只。至少取出多少只，才能保证其中有{n}双同色的袜子？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"黑色袜子 = {a} = {a}只",
        f"白色袜子 = {b} = {b}只",
        f"最坏情况每种颜色摸 = {n} × 2 - 1 = {2 * n - 1}只",
        f"两种颜色共摸 = {2 * n - 1} × 2 = {4 * n - 2}只",
        f"保证有{n}双同色 = {4 * n - 2} + 1 = {ans}只",
    ]
    return ins, lines, ans


_reg("drawer_pairs", drawer_pairs)


# 47. 抽屉原理：同月份生日
def drawer_months(rng):
    k = rng.choice([2, 3, 4])
    ans = 12 * (k - 1) + 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"至少有多少人，才能保证其中一定有{k}个人在同一个月出生？",
        f"要保证必有{k}个人同一个月过生日，至少需要多少人？{name}想知道，请你算一算。",
        f"在任意一群人中，至少要有多少人，才能保证其中有{k}个人出生在同一个月？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"最坏情况每个月出生 = {k} - 1 = {k - 1}人",
        f"12个月共 = {k - 1} × 12 = {12 * (k - 1)}人",
        f"至少需要 = {12 * (k - 1)} + 1 = {ans}人",
    ]
    return ins, lines, ans


_reg("drawer_months", drawer_months)


# 48. 抽屉原理：扑克牌同花色
def drawer_cards(rng):
    n = rng.choice([2, 3, 4])
    ans = 4 * n - 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一副扑克牌（含大王、小王），至少抽出多少张，才能保证其中一定有{n}张同花色的牌？",
        f"从一副扑克牌（含大小王）中抽牌，{name}至少抽多少张才能保证有{n}张同花色？请你算一算。",
        f"一副扑克牌含大王、小王。至少抽出多少张，才能保证其中有{n}张同一花色？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"最坏情况每种花色抽 = {n} - 1 = {n - 1}张",
        f"4种花色共抽 = {n - 1} × 4 = {4 * (n - 1)}张",
        f"大小王 = 2 = 2张",
        f"至少抽出 = {4 * (n - 1)} + 2 + 1 = {ans}张",
    ]
    return ins, lines, ans


_reg("drawer_cards", drawer_cards)


# 49. 抽屉原理：手套配对
def drawer_gloves(rng):
    a = rng.randint(3, 8)
    b = rng.randint(3, 8)
    ans = a + b + 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"箱子里有黑色手套{a}副、白色手套{b}副（手套分左右手）。闭着眼睛至少摸出多少只，才能保证一定有一副同色的手套？",
        f"盒中放着黑手套{a}副、白手套{b}副，混在一起。{name}闭眼摸手套，至少摸出多少只才能保证有一副同色的？请你算一算。",
        f"箱子里有{a}副黑手套和{b}副白手套（分左右手）。至少取出多少只，才能保证其中有一副同色手套？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"手套总数 = 2 × ({a} + {b}) = {2 * (a + b)}只",
        f"最坏情况全是同一只手 = {a} + {b} = {a + b}只",
        f"保证有一副同色 = {a + b} + 1 = {ans}只",
    ]
    return ins, lines, ans


_reg("drawer_gloves", drawer_gloves)


# 50. 过桥问题（手电筒）
def bridge_torch(rng):
    times, steps, ans = rng.choice([
        ((1, 2, 5, 10), [("1号和2号先过桥", 2), ("1号返回", 1), ("3号和4号过桥", 10),
                          ("2号返回", 2), ("1号和2号再过桥", 2)], 17),
        ((1, 3, 4, 6), [("1号和4号先过桥", 6), ("1号返回", 1), ("1号和3号过桥", 4),
                        ("1号返回", 1), ("1号和2号过桥", 3)], 15),
        ((1, 2, 4, 8), [("1号和2号先过桥", 2), ("1号返回", 1), ("3号和4号过桥", 8),
                        ("2号返回", 2), ("1号和2号再过桥", 2)], 15),
        ((2, 3, 5, 10), [("1号和2号先过桥", 3), ("1号返回", 2), ("3号和4号过桥", 10),
                         ("2号返回", 3), ("1号和2号再过桥", 3)], 21),
        ((1, 2, 3, 4), [("1号和2号先过桥", 2), ("1号返回", 1), ("3号和4号过桥", 4),
                        ("2号返回", 2), ("1号和2号再过桥", 2)], 11),
        ((1, 2, 5, 8), [("1号和2号先过桥", 2), ("1号返回", 1), ("3号和4号过桥", 8),
                        ("2号返回", 2), ("1号和2号再过桥", 2)], 15),
    ])
    t1, t2, t3, t4 = times
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"四人过桥，分别需要{t1}、{t2}、{t3}、{t4}分钟。天黑只有一只手电筒，每次最多过2人，过桥时间以慢者为准。四人全部过桥至少需要多少分钟？",
        f"甲、乙、丙、丁四人过桥，所需时间分别为{t1}、{t2}、{t3}、{t4}分钟。只有一只手电筒，每次最多2人同行。{name}想知道四人全部过桥的最少时间，请你算一算。",
        f"四人过桥时间分别是{t1}、{t2}、{t3}、{t4}分钟，一只手电筒，每次最多过2人（按慢者计时）。全部过桥至少要多少分钟？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"四人时间合计 = {t1} + {t2} + {t3} + {t4} = {t1 + t2 + t3 + t4}分钟",
    ]
    for label, val in steps:
        lines.append(f"{label} = {val} = {val}分钟")
    total_expr = " + ".join(str(v) for _, v in steps)
    lines.append(f"总时间 = {total_expr} = {ans}分钟")
    return ins, lines, ans


_reg("bridge_torch", bridge_torch)


# 51. 做饭统筹
def cook_schedule(rng):
    a, b, c, d, e, f = rng.choice([
        (1, 2, 2, 1, 1, 4), (2, 2, 2, 1, 1, 5), (1, 3, 3, 2, 2, 5),
        (2, 3, 3, 1, 1, 6), (1, 2, 2, 2, 2, 3), (1, 3, 2, 2, 1, 6),
    ])
    ans = a + b + d + f
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"妈妈做饭：洗锅{a}分钟，热锅{b}分钟（同时可以洗菜{c}分钟），热油{d}分钟（同时可以切菜{e}分钟），炒菜{f}分钟。做好这顿饭至少需要多少分钟？",
        f"做饭工序：洗锅{a}分钟，热锅{b}分钟（同时洗菜{c}分钟），热油{d}分钟（同时切菜{e}分钟），炒菜{f}分钟。{name}想知道最少用时，请你算一算。",
        f"做一顿饭：洗锅{a}分钟，热锅{b}分钟（这期间可洗菜{c}分钟），热油{d}分钟（这期间可切菜{e}分钟），炒菜{f}分钟。至少需要多少分钟？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"洗锅 = {a} = {a}分钟",
        f"热锅（同时洗菜）= {b} = {b}分钟",
        f"热油（同时切菜）= {d} = {d}分钟",
        f"炒菜 = {f} = {f}分钟",
        f"最少时间 = {a} + {b} + {d} + {f} = {ans}分钟",
    ]
    return ins, lines, ans


_reg("cook_schedule", cook_schedule)


# 52. 水桶量水
def bucket_measure(rng):
    pat = rng.choice(["354", "573"])
    name = rng.choice(NAMES)
    if pat == "354":
        ins = rng.choice([
            "有两个水桶，容量分别是3升和5升。只用这两个桶，怎样量出4升水？至少需要倒几次？",
            f"3升和5升的水桶各一个，{name}想用它们量出4升水。至少需要倒几次？请你算一算。",
            "两个空桶，容量3升和5升。只用这两个桶量出4升水，至少要倒几次？请列式算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            "5升桶水量 = 5 = 5升",
            "5升桶倒入3升桶后 = 5 - 3 = 2升",
            "3升桶倒空 = 3 - 3 = 0升",
            "5升桶的2升倒入3升桶 = 2 = 2升",
            "5升桶再装满 = 5 = 5升",
            "5升桶倒满3升桶后 = 5 - 1 = 4升",
            "最少次数 = 6 = 6次",
        ]
        ans = 6
    else:
        ins = rng.choice([
            "有两个水桶，容量分别是5升和7升。只用这两个桶，怎样量出3升水？至少需要倒几次？",
            f"5升和7升的水桶各一个，{name}想用它们量出3升水。至少需要倒几次？请你算一算。",
            "两个空桶，容量5升和7升。只用这两个桶量出3升水，至少要倒几次？请列式算一算。",
        ]) + rng.choice(_TAILS)
        lines = [
            "5升桶水量 = 5 = 5升",
            "5升桶倒入7升桶后 = 5 = 5升",
            "5升桶再装满 = 5 = 5升",
            "5升桶倒满7升桶后 = 5 - 2 = 3升",
            "最少次数 = 4 = 4次",
        ]
        ans = 4
    return ins, lines, ans


_reg("bucket_measure", bucket_measure)


# 53. 含数字6的页码
def pages_with_digit(rng):
    n, ones, tens, overlap, total = rng.choice([
        (100, 10, 10, 1, 19), (200, 20, 20, 2, 38), (150, 15, 10, 1, 24),
        (120, 12, 10, 1, 21), (80, 8, 10, 1, 17), (60, 6, 1, 0, 7),
        (50, 5, 0, 0, 5), (90, 9, 10, 1, 18), (70, 7, 10, 1, 16),
        (40, 4, 0, 0, 4), (30, 3, 0, 0, 3), (110, 11, 10, 1, 20),
    ])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一本书有{n}页（页码从1到{n}）。页码中含有数字\"6\"的有多少个？",
        f"一本书共{n}页，页码从1编到{n}。{name}想知道含有数字6的页码有多少个，请你算一算。",
        f"一本{n}页的书，页码1~{n}中，含有数字\"6\"的页码共有多少个？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"书的总页数 = {n} = {n}页",
        f"个位是6的页码 = {ones} = {ones}个",
        f"十位是6的页码 = {tens} = {tens}个",
        f"重复计算的页码 = {overlap} = {overlap}个",
        f"含数字6的页码 = {ones} + {tens} - {overlap} = {total}个",
    ]
    return ins, lines, total


_reg("pages_with_digit", pages_with_digit)


# 54. 摸球概率
def ball_probability(rng):
    a, b = rng.choice([
        (3, 2), (4, 3), (5, 3), (4, 2), (5, 2), (6, 4), (3, 3), (6, 3),
        (7, 3), (8, 4), (5, 5), (7, 5),
    ])
    total_c = (a + b) * (a + b - 1) // 2
    red_c = a * (a - 1) // 2
    ans = Fraction(red_c, total_c)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"袋子里有{a}个红球和{b}个白球（除颜色外完全相同）。从中摸出2个球，2个都是红球的概率是多少？",
        f"袋中有{a}个红球、{b}个白球。{name}闭眼摸出2个球，都是红球的概率是多少？请你算一算。",
        f"袋子里放着{a}个红球和{b}个白球，一次摸出2个。2个都是红球的概率是多少？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"球的总数 = {a} + {b} = {a + b}个",
        f"摸2个球的组合数 = {a + b} × {a + b - 1} ÷ 2 = {total_c}种",
        f"2个都是红球的组合数 = {a} × {a - 1} ÷ 2 = {red_c}种",
        f"概率 = {red_c} ÷ {total_c} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("ball_probability", ball_probability)


# 55. 自行车齿轮
def bike_gear(rng):
    a, b, d = rng.choice([
        (40, 20, 50), (32, 16, 50), (48, 24, 50), (36, 18, 50), (44, 22, 50),
        (50, 25, 50), (48, 16, 50), (45, 15, 50), (42, 14, 50), (39, 13, 50),
        (36, 12, 50), (40, 10, 50), (48, 12, 50), (36, 9, 50), (44, 11, 50),
        (52, 13, 50), (50, 10, 50), (45, 9, 50), (40, 8, 50),
    ])
    ratio = a // b
    circ = Fraction(157, 50) * d
    ans = circ * ratio
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"自行车前齿轮有{a}个齿，后齿轮有{b}个齿，车轮直径{d}厘米。蹬一圈自行车前进多少厘米？（π取3.14）",
        f"一辆自行车，前齿轮{a}齿、后齿轮{b}齿，车轮直径{d}厘米。{name}蹬一圈，车前进多少厘米？（π取3.14）请你算一算。",
        f"自行车前齿轮{a}个齿、后齿轮{b}个齿，车轮直径{d}厘米。蹬一圈前进多少厘米？（π取3.14）请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"前后齿轮齿数比 = {a} ÷ {b} = {ratio}",
        f"蹬一圈车轮转 = {ratio} = {ratio}圈",
        f"车轮周长 = 3.14 × {d} = {num(circ)}厘米",
        f"蹬一圈前进 = {num(circ)} × {ratio} = {num(ans)}厘米",
    ]
    return ins, lines, ans


_reg("bike_gear", bike_gear)


# 56. 车轮滚动距离
def wheel_rolls(rng):
    d, n = rng.choice([
        ("0.6", 100), ("0.5", 200), ("0.8", 50), ("1.2", 50), ("0.4", 250),
        ("0.7", 100), ("0.9", 100), ("1.5", 40), ("0.6", 50), ("0.5", 100),
    ])
    circ = Fraction(157, 50) * Fraction(d)
    ans = circ * n
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"自行车车轮的直径是{d}米，滚动{n}圈前进多少米？（π取3.14）",
        f"车轮直径{d}米，在地上滚动{n}圈。{name}想知道前进了多少米，请你算一算。（π取3.14）",
        f"一个车轮直径{d}米，滚动{n}圈能前进多少米？（π取3.14）请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"车轮周长 = 3.14 × {d} = {num(circ)}米",
        f"前进距离 = {num(circ)} × {n} = {num(ans)}米",
    ]
    return ins, lines, ans


_reg("wheel_rolls", wheel_rolls)


# 57. 回声测距
def echo_distance(rng):
    t = rng.choice([3, 4, 5, 6, 8, 10])
    v = 340
    ans = v * t // 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}朝对面山谷大喊一声，{t}秒后听到回声。声音在空气中每秒传播340米，{name}离山谷多少米？",
        f"小明对着山崖喊话，{t}秒后听到回声，声速是每秒340米。山崖离小明多少米？请你算一算。",
        f"朝山谷喊一声，{t}秒后听到回声。声音每秒传播340米，人到山谷的距离是多少米？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"声音传播的总路程 = 340 × {t} = {v * t}米",
        f"人到山谷的距离 = {v * t} ÷ 2 = {ans}米",
    ]
    return ins, lines, ans


_reg("echo_distance", echo_distance)


# 58. 火柴棒摆正方形
def matchstick_squares(rng):
    n = rng.randint(4, 15)
    ans = 3 * n + 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"用火柴棒摆正方形（连在一起摆），摆{n}个正方形需要多少根火柴棒？",
        f"把正方形连在一起摆，{name}摆{n}个正方形需要多少根火柴棒？请你算一算。",
        f"用火柴棒并排摆{n}个相连的正方形，一共需要多少根火柴棒？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"第一个正方形 = 4 = 4根",
        f"以后每个正方形 = 3 = 3根",
        f"{n}个正方形共需 = 4 + ({n} - 1) × 3 = {ans}根",
    ]
    return ins, lines, ans


_reg("matchstick_squares", matchstick_squares)


# 59. 骰子看不见的面
def dice_hidden_sum(rng):
    a, b, c = rng.choice([
        (1, 2, 3), (1, 2, 4), (1, 3, 5), (2, 3, 6), (1, 4, 5),
        (2, 4, 6), (3, 5, 6), (4, 5, 6),
    ])
    ans = 21 - (a + b + c)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个正方体骰子放在桌上，看得见的三个面上的数字分别是{a}、{b}、{c}（骰子相对两个面上的数字之和都是7）。看不见的三个面上的数字之和是多少？",
        f"骰子可见的三个面是{a}、{b}、{c}，相对面数字之和为7。{name}想知道看不见的三面之和，请你算一算。",
        f"一个骰子放在桌上，朝上和朝侧面的三个数字是{a}、{b}、{c}，骰子相对两面之和是7。看不见的三个数字之和是多少？请列式算一算。",
    ]) + rng.choice(_TAILS)
    lines = [
        f"{a}的对面 = 7 - {a} = {7 - a}",
        f"{b}的对面 = 7 - {b} = {7 - b}",
        f"{c}的对面 = 7 - {c} = {7 - c}",
        f"看不见的三面之和 = {7 - a} + {7 - b} + {7 - c} = {ans}",
    ]
    return ins, lines, ans


_reg("dice_hidden_sum", dice_hidden_sum)
