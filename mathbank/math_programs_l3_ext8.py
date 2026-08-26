#!/usr/bin/env python3
"""L3 extension bank 8: structurally novel elementary programs.

Each program: fn(rng) -> (instruction, lines, ans). Every line is an
equation `label = X op Y = Z[u]` (3-part) or `X op Y = Z[u]` (2-part,
pure-arithmetic LHS). Last line value must equal ans. Prose integers >=3
must appear among the equation tokens (enforced by run_math_short.verify).

Structures covered (all absent from l3 base + ext1..ext7):
boat current/raft/hat-chase/roundtrip-distance, train same-direction
overtake, work 2+1 cycle / three-person rotation / machine upgrade,
conc mix-then-evaporate / mutual swap / evaporate-then-add-salt /
three-way mix, profit restore-original / remainder-profit / clearance
batch, simple-vs-compound interest, two-map scale, sector-minus-triangle,
four-semicircle petals, annular sector, semicircle-minus-triangle,
equal-perimeter square-vs-circle, cuboid/cylinder net square, cube cut
surface, cone cast to cylinder, cone frustum, staircase section, grid
triangle cut-and-fill, fence against wall, ant shortest path, dice
opposite faces, mirror time, number-table nth value, triangle table row
sum, table first-n-rows sum, dot pattern, cube stack, square-grid
matches, count squares, median overlap average, drop-high-low average,
fair-game add balls, odd-product card probability, with-replacement
probability, rank/seat logic, age chain, ages-ago multiple, ages-future
multiple, age-sum future, max product digit arrangement, min sum given
product, three-way max product, min perimeter given area, prime split,
double surplus/shortage, dorm rooms, cutting stock, milk bulk, snail
well, candle burning, three-leg average speed.
"""

import random
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
# Batch 1: boats, trains, work, concentration, profit, interest
# ---------------------------------------------------------------------------

def boat_current_speed(rng):
    vu = rng.randint(10, 24)
    vd = rng.randint(vu + 4, 40)
    if (vd - vu) % 2:
        vd += 1
    cur = Fraction(vd - vu, 2)
    still = Fraction(vd + vu, 2)
    ins = rng.choice([
        f"一艘船顺水每小时行{vd}千米，逆水每小时行{vu}千米，水流速度是每小时多少千米？",
        f"轮船顺水速度为每小时{vd}千米，逆水速度为每小时{vu}千米，这条河的水流速度是多少？",
        f"一条船顺流每小时行{vd}千米，逆流每小时行{vu}千米，水流每小时多少千米？",
        f"船在顺水中每小时行{vd}千米，逆水中每小时行{vu}千米，水速是每小时多少千米？",
    ])
    lines = [
        f"{vd} + {vu} = {vd + vu}千米/时",
        f"静水速度 = {vd + vu} ÷ 2 = {num(still)}千米/时",
        f"{vd} - {vu} = {vd - vu}千米/时",
        f"水流速度 = {vd - vu} ÷ 2 = {num(cur)}千米/时",
    ]
    return ins, lines, cur


_reg("boat_current_speed", boat_current_speed)


def boat_raft(rng):
    a, b = rng.choice([(3, 5), (4, 6), (5, 7), (3, 9), (6, 10), (7, 9),
                       (4, 8), (6, 8), (2, 4), (2, 6), (5, 15), (7, 21),
                       (8, 12), (9, 15), (10, 14), (10, 30), (12, 18), (14, 28)])
    D = a * b
    vd = b
    vu = a
    cur = Fraction(b - a, 2)
    raft = Fraction(D, cur)
    ins = rng.choice([
        f"甲、乙两港相距{D}千米，一艘船顺水从甲港到乙港需{a}小时，逆水返回需{b}小时。一只木筏从甲港漂流到乙港需要多少小时？",
        f"两码头相距{D}千米，船顺水行完全程要{a}小时，逆水行完全程要{b}小时，木筏顺水漂流全程要多少小时？",
        f"一艘船在相距{D}千米的两港间航行，顺水{a}小时到达，逆水{b}小时到达，无动力的木筏漂流全程需多少小时？",
        f"甲港到乙港{D}千米，船顺流而下用{a}小时，逆流而上用{b}小时，一只木箱从甲港漂到乙港要多少小时？",
    ])
    lines = [
        f"{D} ÷ {a} = {vd}千米/时",
        f"{D} ÷ {b} = {vu}千米/时",
        f"{vd} - {vu} = {vd - vu}千米/时",
        f"水流速度 = {vd - vu} ÷ 2 = {num(cur)}千米/时",
        f"{D} ÷ {num(cur)} = {num(raft)}小时",
    ]
    return ins, lines, raft


_reg("boat_raft", boat_raft)


def boat_hat(rng):
    v = rng.randint(15, 30)
    w = rng.randint(3, 10)
    t = rng.randint(5, 30)
    ins = rng.choice([
        f"一人划船在静水中每小时行{v}千米，河水流速为每小时{w}千米。他的帽子掉入水中，{t}分钟后才发现，立即掉头（船在静水中速度不变），需要多少分钟追上帽子？",
        f"小船在静水中速度为每小时{v}千米，水流每小时{w}千米。帽子落水后船继续前行{t}分钟才发现，随后掉头追帽子，多少分钟能追上？",
        f"某船静水速度每小时{v}千米，水速每小时{w}千米。船上一只木箱掉落水中，{t}分钟后船员发现并掉头，船多久能追上木箱？",
        f"划船人静水速度每小时{v}千米，河水每小时流{w}千米。帽子被风吹落水中，{t}分钟后他发现并掉头，追上帽子需要多少分钟？",
    ])
    lines = [
        f"{v} + {w} - {w} = {v}千米/时",
        f"{v} - {w} + {w} = {v}千米/时",
        f"追及时间 = {t} × {v} ÷ {v} = {t}分钟",
    ]
    return ins, lines, t


_reg("boat_hat", boat_hat)


def boat_roundtrip_dist(rng):
    v, w, T = rng.choice([
        (25, 5, 5), (25, 5, 4), (25, 5, 6), (30, 6, 5), (25, 5, 3),
        (30, 10, 6), (20, 4, 5), (20, 4, 10), (25, 5, 10), (30, 6, 10),
        (25, 5, 8), (15, 5, 6), (15, 5, 3), (15, 5, 12), (25, 15, 5),
        (25, 15, 10), (25, 15, 4), (25, 15, 6), (25, 15, 8), (25, 15, 12),
        (25, 15, 3), (25, 15, 2), (25, 15, 7), (25, 15, 9), (25, 15, 11),
        (25, 15, 14), (25, 15, 16), (25, 15, 18), (25, 15, 20), (25, 15, 25),
    ])
    vd, vu = v + w, v - w
    avg = Fraction(vd * vu, v)
    D = avg * T / 2
    ins = rng.choice([
        f"一艘轮船在静水中每小时行{v}千米，水流速度为每小时{w}千米，往返甲、乙两港一次共需{T}小时。甲、乙两港相距多少千米？",
        f"船在静水中每小时行{v}千米，水速每小时{w}千米，往返两码头共用{T}小时，两码头相距多少千米？",
        f"某船静水速度每小时{v}千米，河水每小时流{w}千米，该船往返两港一次花了{T}小时，两港相距多少千米？",
        f"轮船静水速度为每小时{v}千米，水流每小时{w}千米，它在两港间往返一次需{T}小时，两港间的距离是多少千米？",
    ])
    lines = [
        f"{v} + {w} = {vd}千米/时",
        f"{v} - {w} = {vu}千米/时",
        f"{vd} + {vu} = {vd + vu}千米/时",
        f"{vd} × {vu} × 2 ÷ {vd + vu} = {num(avg)}千米/时",
        f"{num(avg)} × {T} = {num(avg * T)}千米",
        f"{num(avg * T)} ÷ 2 = {num(D)}千米",
    ]
    return ins, lines, D


_reg("boat_roundtrip_dist", boat_roundtrip_dist)


def train_same_dir(rng):
    L1, L2, v1, v2 = rng.choice([
        (200, 160, 25, 20), (150, 250, 30, 20), (180, 120, 25, 15),
        (220, 180, 28, 20), (160, 140, 25, 20), (300, 200, 30, 20),
        (120, 80, 20, 15), (250, 150, 30, 22), (180, 220, 27, 19),
        (140, 100, 22, 16), (200, 100, 25, 15), (160, 240, 29, 19),
        (100, 140, 20, 14), (180, 180, 30, 20), (240, 160, 32, 24),
        (150, 130, 24, 16), (170, 130, 26, 20), (190, 210, 30, 22),
    ])
    L = L1 + L2
    dv = v1 - v2
    t = Fraction(L, dv)
    ins = rng.choice([
        f"快车长{L1}米，每秒行{v1}米；慢车长{L2}米，每秒行{v2}米。两车同向并行，快车从追上慢车到完全超过需要多少秒？",
        f"一列快车长{L1}米，速度每秒{v1}米；一列慢车长{L2}米，速度每秒{v2}米。两车同向行驶，快车车头追上慢车车尾到快车车尾离开慢车车头需多少秒？",
        f"快车车长{L1}米，慢车车长{L2}米，两车同向而行，快车每秒{v1}米，慢车每秒{v2}米，快车超过慢车需要多少秒？",
        f"两列火车同向行驶，甲车长{L1}米每秒行{v1}米，乙车长{L2}米每秒行{v2}米，甲车从追上乙车到完全超过乙车要多少秒？",
    ])
    lines = [
        f"{L1} + {L2} = {L}米",
        f"{v1} - {v2} = {dv}米/秒",
        f"{L} ÷ {dv} = {num(t)}秒",
    ]
    return ins, lines, t


_reg("train_same_dir", train_same_dir)


def work_cycle_21(rng):
    a, b = rng.choice([(15, 12), (20, 15), (10, 15), (12, 18), (30, 20),
                       (18, 12), (24, 16), (12, 20), (40, 30), (18, 24)])
    ra, rb = Fraction(1, a), Fraction(1, b)
    cycle = 2 * ra + rb
    n = 0
    while (n + 1) * cycle < 1:
        n += 1
    rem = 1 - n * cycle
    jia = min(Fraction(2), rem / ra)
    rem2 = rem - jia * ra
    yi = rem2 / rb if rem2 > 0 else Fraction(0)
    total = 3 * n + jia + yi
    ins = rng.choice([
        f"一项工程，甲队单独做{a}天完成，乙队单独做{b}天完成。甲队做2天、乙队做1天为一个周期循环，共需多少天完成？",
        f"修一条路，甲单独修{a}天完成，乙单独修{b}天完成。甲先修2天、乙再修1天，按此循环，共需多少天？",
        f"一批零件，师傅单独做{a}天完成，徒弟单独做{b}天完成。师傅做2天、徒弟做1天轮流，共需多少天完成？",
        f"一个水池，甲管单独注满需{a}小时，乙管需{b}小时。甲管开2小时、乙管开1小时循环，多少小时注满？",
    ])
    lines = [
        f"1 ÷ {a} = {num(ra)}",
        f"1 ÷ {b} = {num(rb)}",
        f"2 × {num(ra)} + {num(rb)} = {num(cycle)}",
        f"{num(cycle)} × {n} = {num(cycle * n)}",
        f"1 - {num(cycle * n)} = {num(rem)}",
        f"{num(jia)} × {num(ra)} = {num(jia * ra)}",
    ]
    if yi > 0:
        lines.append(f"{num(rem)} - {num(jia * ra)} = {num(rem2)}")
        lines.append(f"({num(rem2)}) ÷ ({num(rb)}) = {num(yi)}天")
    lines.append(f"3 × {n} + {num(jia)} + {num(yi)} = {num(total)}天")
    return ins, lines, total


_reg("work_cycle_21", work_cycle_21)


def work_three_alt(rng):
    a, b, c = rng.choice([
        (12, 15, 20), (10, 15, 30), (12, 18, 36), (15, 20, 30),
        (20, 30, 60), (12, 24, 36), (15, 30, 45), (20, 24, 30),
        (10, 12, 15), (14, 21, 28), (18, 27, 54), (20, 40, 50),
    ])
    ra, rb, rc = Fraction(1, a), Fraction(1, b), Fraction(1, c)
    cycle = ra + rb + rc
    n = 0
    while (n + 1) * cycle < 1:
        n += 1
    rem = 1 - n * cycle
    rem0 = rem
    d1 = min(Fraction(1), rem / ra)
    rem -= d1 * ra
    d2 = min(Fraction(1), rem / rb) if rem > 0 else Fraction(0)
    rem -= d2 * rb
    d3 = rem / rc if rem > 0 else Fraction(0)
    total = 3 * n + d1 + d2 + d3
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成，乙单独做{b}天完成，丙单独做{c}天完成。三人按甲、乙、丙的顺序轮流各做一天，共需多少天完成？",
        f"修一条路，甲队独修{a}天完成，乙队独修{b}天完成，丙队独修{c}天完成。三队按甲乙丙顺序轮流各修一天，共需多少天？",
        f"一批零件，甲、乙、丙单独做分别需{a}天、{b}天、{c}天。三人按甲、乙、丙顺序轮流各做一天，共需多少天完成？",
        f"一个水池，甲管{a}小时注满，乙管{b}小时注满，丙管{c}小时注满。三管按甲乙丙顺序轮流各开一小时，多少小时注满？",
    ])
    lines = [
        f"1 ÷ {a} = {num(ra)}",
        f"1 ÷ {b} = {num(rb)}",
        f"1 ÷ {c} = {num(rc)}",
        f"{num(ra)} + {num(rb)} + {num(rc)} = {num(cycle)}",
        f"{num(cycle)} × {n} = {num(cycle * n)}",
        f"1 - {num(cycle * n)} = {num(rem0)}",
        f"{num(d1)} × {num(ra)} = {num(d1 * ra)}",
    ]
    if d2 > 0:
        lines.append(f"{num(rem0)} - {num(d1 * ra)} = {num(rem0 - d1 * ra)}")
        lines.append(f"{num(d2)} × {num(rb)} = {num(d2 * rb)}")
    if d3 > 0:
        lines.append(f"({num(rem0 - d1 * ra - d2 * rb)}) ÷ ({num(rc)}) = {num(d3)}天")
    lines.append(f"3 × {n} + {num(d1)} + {num(d2)} + {num(d3)} = {num(total)}天")
    return ins, lines, total


_reg("work_three_alt", work_three_alt)


def work_machine_upgrade(rng):
    a, b, t = rng.choice([
        (40, 10, 2), (45, 9, 2), (50, 10, 2), (30, 10, 3), (40, 8, 2),
        (45, 15, 2), (60, 12, 2), (36, 9, 2), (48, 12, 2), (50, 25, 2),
        (40, 5, 2), (35, 7, 2), (56, 8, 2), (45, 5, 3), (30, 6, 2),
        (24, 8, 3), (25, 5, 2), (20, 5, 3), (20, 4, 2), (18, 6, 2),
        (16, 8, 2), (15, 5, 2), (12, 6, 2), (10, 5, 2), (60, 10, 3),
    ])
    N = t * a * (a + b) // b
    ins = rng.choice([
        f"工厂加工一批零件，原计划每小时加工{a}个，实际每小时多加工{b}个，结果提前{t}小时完成。这批零件共有多少个？",
        f"车间生产一批零件，计划每小时做{a}个，改进方法后每小时多做{b}个，提前{t}小时完成任务。这批零件有多少个？",
        f"加工一批零件，原计划每小时加工{a}个，实际每小时加工{a + b}个，比计划提前{t}小时完成。这批零件共多少个？",
        f"一批零件，按原计划每小时加工{a}个则按时完成，实际每小时多加工{b}个，提前{t}小时完成。这批零件有多少个？",
    ])
    lines = [
        f"{a} + {b} = {a + b}个/时",
        f"{a} × {a + b} = {a * (a + b)}",
        f"{a * (a + b)} × {t} = {a * (a + b) * t}",
        f"{a * (a + b) * t} ÷ {b} = {N}个",
    ]
    return ins, lines, N


_reg("work_machine_upgrade", work_machine_upgrade)


def conc_mix_evap(rng):
    a, x, b, y, c = rng.choice([
        (200, 10, 300, 20, 100), (200, 10, 300, 20, 300),
        (300, 10, 200, 5, 300), (300, 10, 200, 5, 420),
        (400, 5, 100, 20, 300), (200, 15, 300, 10, 200),
        (200, 15, 300, 10, 380), (250, 8, 250, 12, 300),
        (100, 20, 400, 5, 100), (100, 20, 400, 5, 300),
        (300, 20, 200, 10, 100), (300, 20, 200, 10, 300),
        (300, 20, 200, 10, 340), (250, 8, 250, 12, 450),
        (200, 5, 300, 15, 200), (150, 10, 350, 20, 200),
    ])
    s1 = Fraction(a * x, 100)
    s2 = Fraction(b * y, 100)
    salt = s1 + s2
    total = a + b - c
    conc = salt * 100 / total
    ins = rng.choice([
        f"甲杯有{a}克含盐{x}%的盐水，乙杯有{b}克含盐{y}%的盐水，混合后蒸发掉{c}克水，此时盐水的含盐率是多少？",
        f"把{a}克含盐{x}%的盐水和{b}克含盐{y}%的盐水混合，再蒸发{c}克水，现在的含盐率是百分之几？",
        f"甲瓶{a}克盐水浓度{x}%，乙瓶{b}克盐水浓度{y}%，混合后蒸发{c}克水，浓度变为多少？",
        f"两杯盐水分别重{a}克（浓度{x}%）和{b}克（浓度{y}%），混合后蒸发{c}克水，求现在的含盐率。",
    ])
    lines = [
        f"{a} × {x}/100 = {num(s1)}克",
        f"{b} × {y}/100 = {num(s2)}克",
        f"{num(s1)} + {num(s2)} = {num(salt)}克",
        f"{a} + {b} - {c} = {total}克",
        f"{num(salt)} ÷ {total} × 100 = {num(conc)}%",
    ]
    return ins, lines, conc


_reg("conc_mix_evap", conc_mix_evap)


def conc_swap_equal(rng):
    a, b = rng.choice([
        (300, 200), (400, 100), (600, 300), (100, 300), (150, 100),
        (240, 160), (350, 150), (400, 600), (700, 300), (800, 200),
        (900, 300), (100, 400), (100, 700), (100, 900), (200, 600),
        (300, 600), (500, 750), (600, 900), (120, 180), (160, 240),
        (180, 270), (210, 280), (220, 330), (240, 360), (260, 390),
        (280, 420), (300, 450), (320, 480), (340, 510), (360, 540),
    ])
    x, y = rng.randint(5, 25), rng.randint(5, 25)
    swap = Fraction(a * b, a + b)
    ins = rng.choice([
        f"甲瓶有{a}克含盐{x}%的盐水，乙瓶有{b}克含盐{y}%的盐水。现在从两瓶中各取出相同质量的盐水互换，互换多少克后两瓶浓度相同？",
        f"甲杯{a}克盐水浓度{x}%，乙杯{b}克盐水浓度{y}%，互相交换多少克后两杯盐水浓度相同？",
        f"甲瓶{a}克{x}%的盐水与乙瓶{b}克{y}%的盐水，各取出多少克互换后，两瓶浓度恰好相同？",
        f"两瓶盐水分别重{a}克和{b}克，浓度为{x}%和{y}%，互相交换多少克后浓度相同？",
    ])
    lines = [
        f"{a} × {b} = {a * b}",
        f"{a} + {b} = {a + b}克",
        f"{a * b} ÷ {a + b} = {num(swap)}克",
    ]
    return ins, lines, swap


_reg("conc_swap_equal", conc_swap_equal)


def conc_evap_add_salt(rng):
    a, x, e, y = rng.choice([
        (400, 5, 100, 20), (400, 5, 200, 20), (500, 8, 100, 20),
        (500, 8, 250, 30), (300, 10, 150, 25), (300, 10, 200, 50),
        (200, 15, 50, 40), (200, 10, 50, 50), (600, 10, 100, 20),
        (600, 10, 200, 20), (300, 15, 100, 30), (400, 10, 150, 30),
        (500, 10, 200, 30), (250, 8, 50, 20), (350, 10, 150, 30),
    ])
    s0 = Fraction(a * x, 100)
    ae = a - e
    num_ = y * ae - 100 * s0
    den = 100 - y
    s = Fraction(num_, den)
    ins = rng.choice([
        f"有{a}克含盐{x}%的盐水，蒸发掉{e}克水后，要使盐水浓度变为{y}%，需要加入多少克盐？",
        f"一杯{a}克的盐水，浓度{x}%，蒸发{e}克水后，再加入多少克盐，浓度恰好为{y}%？",
        f"现有{a}克含盐{x}%的盐水，先蒸发{e}克水，再加入多少克盐，能使浓度达到{y}%？",
        f"把{a}克{x}%的盐水蒸发{e}克水后，加入多少克盐，盐水浓度变为{y}%？",
    ])
    lines = [
        f"{a} × {x}/100 = {num(s0)}克",
        f"{a} - {e} = {ae}克",
        f"{ae} × {y} = {ae * y}",
        f"{num(s0)} × 100 = {100 * s0}",
        f"{ae * y} - {100 * s0} = {num_}",
        f"100 - {y} = {den}",
        f"{num_} ÷ {den} = {num(s)}克",
    ]
    return ins, lines, s


_reg("conc_evap_add_salt", conc_evap_add_salt)


def conc_three_mix(rng):
    a, x, b, y, c, z = rng.choice([
        (300, 10, 200, 20, 500, 6), (300, 10, 200, 20, 500, 16),
        (300, 10, 200, 20, 500, 26), (300, 10, 200, 20, 500, 36),
        (300, 10, 200, 20, 500, 46), (200, 10, 300, 20, 500, 30),
        (250, 8, 250, 12, 500, 10), (250, 8, 250, 12, 500, 20),
        (250, 8, 250, 12, 500, 30), (250, 8, 250, 12, 500, 40),
        (250, 8, 250, 12, 500, 50), (200, 5, 300, 15, 500, 25),
        (200, 5, 300, 15, 500, 35), (200, 5, 300, 15, 500, 45),
        (200, 5, 300, 15, 500, 55), (200, 5, 300, 15, 500, 65),
        (200, 5, 300, 15, 500, 75), (200, 5, 300, 15, 500, 85),
        (200, 5, 300, 15, 500, 95), (100, 10, 200, 20, 700, 20),
    ])
    s1 = Fraction(a * x, 100)
    s2 = Fraction(b * y, 100)
    s3 = Fraction(c * z, 100)
    salt = s1 + s2 + s3
    total = a + b + c
    conc = salt * 100 / total
    ins = rng.choice([
        f"三杯盐水分别重{a}克、{b}克、{c}克，含盐率分别为{x}%、{y}%、{z}%。三杯混合后含盐率是多少？",
        f"甲、乙、丙三瓶盐水分别重{a}克、{b}克、{c}克，浓度分别是{x}%、{y}%、{z}%，混合后的浓度是多少？",
        f"把{a}克{x}%的盐水、{b}克{y}%的盐水和{c}克{z}%的盐水倒在一起，混合后含盐率是百分之几？",
        f"三种盐水分别重{a}克、{b}克、{c}克，浓度各为{x}%、{y}%、{z}%，混合后的含盐率是多少？",
    ])
    lines = [
        f"{a} × {x}/100 = {num(s1)}克",
        f"{b} × {y}/100 = {num(s2)}克",
        f"{c} × {z}/100 = {num(s3)}克",
        f"{num(s1)} + {num(s2)} + {num(s3)} = {num(salt)}克",
        f"{a} + {b} + {c} = {total}克",
        f"{num(salt)} ÷ {total} × 100 = {num(conc)}%",
    ]
    return ins, lines, conc


_reg("conc_three_mix", conc_three_mix)


def profit_restore_original(rng):
    x, a = rng.choice([
        (10, 4), (10, 9), (10, 15), (10, 25), (10, 36), (5, 2), (5, 5),
        (5, 10), (5, 20), (4, 2), (4, 5), (4, 10), (4, 16), (2, 2),
        (2, 5), (2, 8), (20, 25), (8, 16), (16, 64), (12, 9), (15, 9),
        (6, 9), (3, 9), (7, 49), (9, 81), (14, 49), (11, 121), (13, 169),
        (17, 289), (19, 361), (21, 441), (22, 121), (23, 529), (24, 36),
        (25, 16), (25, 25), (25, 36), (25, 49), (25, 64), (25, 81),
    ])
    P = Fraction(10000 * a, x * x)
    ins = rng.choice([
        f"一件商品先涨价{x}%，再降价{x}%，结果比原价少了{a}元。这件商品原价多少元？",
        f"某商品先提价{x}%，又降价{x}%，最终比原价少{a}元，原价是多少元？",
        f"一台电器先涨价{x}%，再降价{x}%，售价比原价低{a}元，原价多少元？",
        f"商品价格先上调{x}%，再下调{x}%，比原价少了{a}元，原价是多少元？",
    ])
    lines = [
        f"{x} × {x} = {x * x}",
        f"10000 × {a} = {10000 * a}",
        f"{10000 * a} ÷ {x * x} = {num(P)}元",
    ]
    return ins, lines, P


_reg("profit_restore_original", profit_restore_original)


def profit_remain_profit(rng):
    c, p, r, d = rng.choice([
        (20, 30, 10, 200), (25, 40, 8, 250), (25, 40, 8, 220),
        (25, 40, 8, 190), (25, 40, 8, 160), (25, 40, 8, 130),
        (25, 40, 8, 100), (25, 40, 8, 70), (25, 40, 8, 40),
        (30, 50, 10, 100), (30, 50, 10, 300), (30, 50, 10, 500),
        (30, 50, 10, 700), (30, 50, 10, 900), (40, 60, 5, 100),
        (40, 60, 5, 300), (40, 60, 5, 500), (40, 60, 5, 700),
        (15, 20, 10, 50), (15, 20, 10, 100), (15, 20, 10, 150),
        (15, 20, 10, 200), (10, 15, 20, 100), (10, 15, 20, 200),
        (50, 60, 10, 200), (50, 60, 10, 400), (50, 60, 10, 600),
        (50, 60, 10, 800), (50, 60, 10, 1000), (50, 60, 10, 1200),
    ])
    pc = p - c
    pr = p * r
    N = Fraction(d + pr, pc)
    ins = rng.choice([
        f"商店以每件{c}元购进一批商品，售价每件{p}元。当卖到还剩{r}件时，除收回全部成本外还获利{d}元。这批商品共有多少件？",
        f"商场以每件{c}元的价格购进一批商品，按每件{p}元出售。卖到还剩{r}件时，除成本外还盈利{d}元。这批商品有多少件？",
        f"商店购进一批单价{c}元的商品，按每件{p}元销售。当剩下{r}件没卖时，已收回成本并获利{d}元。这批商品共多少件？",
        f"一批商品每件成本{c}元，售价{p}元。卖到还剩{r}件时，除全部成本外净赚{d}元。这批商品共有多少件？",
    ])
    lines = [
        f"{p} - {c} = {pc}元",
        f"{p} × {r} = {pr}元",
        f"{d} + {pr} = {d + pr}元",
        f"{d + pr} ÷ {pc} = {num(N)}件",
    ]
    return ins, lines, N


_reg("profit_remain_profit", profit_remain_profit)


def profit_clearance(rng):
    c, x, n, y = rng.choice([
        (80, 50, 50, 5), (80, 50, 100, 5), (80, 50, 50, 8),
        (60, 50, 50, 5), (60, 50, 80, 5), (100, 20, 50, 5),
        (100, 20, 50, 8), (50, 80, 50, 5), (50, 60, 50, 5),
        (50, 60, 50, 8), (75, 40, 40, 8), (75, 40, 40, 6),
        (75, 40, 40, 4), (75, 40, 80, 8), (60, 50, 100, 8),
        (80, 25, 80, 8), (80, 25, 80, 5), (80, 25, 40, 8),
        (80, 25, 40, 5), (120, 50, 50, 8), (120, 50, 50, 5),
        (120, 50, 100, 8), (120, 50, 100, 5), (120, 50, 40, 8),
        (120, 50, 40, 5), (90, 50, 60, 8), (90, 50, 60, 5),
        (90, 50, 80, 8), (90, 50, 80, 5), (90, 50, 100, 8),
    ])
    markup = Fraction(c * x, 100)
    price = c + markup
    n1 = Fraction(n * 80, 100)
    n2 = n - n1
    price2 = Fraction(price * y, 10)
    rev1 = n1 * price
    rev2 = n2 * price2
    total_rev = rev1 + rev2
    total_cost = c * n
    profit = total_rev - total_cost
    ins = rng.choice([
        f"商店以每件{c}元购进{n}件衬衫，按获利{x}%定价。卖出80%后，剩下的打{y}折出售。全部卖完后共获利多少元？",
        f"商场以每件{c}元的价格进了{n}件商品，先按{x}%的利润定价出售。卖出80%后，余下的打{y}折卖完。总获利多少元？",
        f"一批{n}件商品，每件成本{c}元，按{x}%的利润定价。卖出80%后，剩下的打{y}折出售。这批商品共获利多少元？",
        f"商店购进{n}件单价{c}元的商品，按{x}%的利润率定价。卖出80%后，剩余商品打{y}折售完。实际获利多少元？",
    ])
    lines = [
        f"{c} × {x}/100 = {num(markup)}元",
        f"{c} + {num(markup)} = {num(price)}元",
        f"{n} × 80/100 = {num(n1)}件",
        f"{num(n1)} × {num(price)} = {num(rev1)}元",
        f"{n} - {num(n1)} = {num(n2)}件",
        f"{num(price)} × {y}/10 = {num(price2)}元",
        f"{num(n2)} × {num(price2)} = {num(rev2)}元",
        f"{num(rev1)} + {num(rev2)} = {num(total_rev)}元",
        f"{c} × {n} = {total_cost}元",
        f"{num(total_rev)} - {total_cost} = {num(profit)}元",
    ]
    return ins, lines, profit


_reg("profit_clearance", profit_clearance)


def simple_compound_diff(rng):
    P, r = rng.choice([
        (10000, 5), (10000, 10), (10000, 8), (10000, 6), (10000, 4),
        (10000, 20), (10000, 15), (10000, 12), (10000, 25), (8000, 5),
        (8000, 10), (8000, 15), (8000, 20), (6000, 5), (6000, 10),
        (6000, 15), (6000, 20), (5000, 10), (5000, 20), (5000, 8),
        (5000, 6), (5000, 4), (5000, 12), (4000, 5), (4000, 10),
        (4000, 15), (4000, 20), (2000, 5), (2000, 10), (2000, 15),
        (2000, 20), (1000, 10), (1000, 20), (1000, 25), (1000, 40),
        (2500, 8), (2500, 6), (2500, 4), (2500, 12), (2500, 20),
    ])
    i1 = Fraction(P * r, 100)
    diff = Fraction(i1 * r, 100)
    ins = rng.choice([
        f"小明把{P}元存入银行，年利率{r}%，存期2年。复利比单利多得多少元利息？",
        f"把{P}元钱存入银行，年利率{r}%，存2年。按复利计算比按单利计算多多少元利息？",
        f"妈妈把{P}元存入银行，年利率{r}%，定期2年。复利计息比单利计息多多少元？",
        f"一笔{P}元的存款，年利率{r}%，存2年，复利比单利多多少元利息？",
    ])
    lines = [
        f"{P} × {r}/100 = {num(i1)}元",
        f"{num(i1)} × {r}/100 = {num(diff)}元",
    ]
    return ins, lines, diff


_reg("simple_compound_diff", simple_compound_diff)

# ---------------------------------------------------------------------------
# Batch 2: scale, shaded areas, nets, solid geometry, misc classics
# ---------------------------------------------------------------------------

def scale_two_maps(rng):
    D, a, b = rng.choice([
        (600, 10000, 15000), (900, 10000, 15000), (1200, 10000, 15000),
        (600, 5000, 10000), (800, 10000, 16000), (1000, 10000, 20000),
        (1000, 5000, 20000), (750, 5000, 15000), (450, 5000, 15000),
        (450, 9000, 15000), (360, 6000, 9000), (360, 4000, 9000),
        (480, 6000, 8000), (480, 4000, 8000), (540, 6000, 9000),
        (540, 3000, 9000), (240, 3000, 4000), (240, 2000, 4000),
        (240, 2000, 3000), (240, 3000, 6000), (300, 3000, 5000),
        (300, 2000, 5000), (300, 2500, 5000), (300, 2000, 6000),
        (300, 3000, 6000), (300, 5000, 6000), (150, 3000, 5000),
        (150, 2000, 6000), (150, 3000, 6000), (150, 5000, 6000),
    ])
    Dcm = D * 100
    d1 = Fraction(Dcm, a)
    d2 = Fraction(Dcm, b)
    diff = d1 - d2
    ins = rng.choice([
        f"甲、乙两幅地图的比例尺分别是1:{a}和1:{b}。实际距离{D}米的一段路，在两幅图上的长度相差多少厘米？",
        f"在比例尺为1:{a}和1:{b}的两幅地图上，实际距离{D}米的公路，图上距离相差多少厘米？",
        f"同一实际距离{D}米，在比例尺1:{a}的图上长一些，在比例尺1:{b}的图上短一些，两幅图上的长度相差多少厘米？",
        f"甲图比例尺1:{a}，乙图比例尺1:{b}，实际距离{D}米的路程在两图上的长度差是多少厘米？",
    ])
    lines = [
        f"{D} × 100 = {Dcm}厘米",
        f"{Dcm} ÷ {a} = {num(d1)}厘米",
        f"{Dcm} ÷ {b} = {num(d2)}厘米",
        f"{num(d1)} - {num(d2)} = {num(diff)}厘米",
    ]
    return ins, lines, diff


_reg("scale_two_maps", scale_two_maps)


def sector_triangle_shaded(rng):
    r = rng.choice([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    circle = Fraction(314, 100) * r * r
    sector = circle / 4
    tri = Fraction(r * r, 2)
    shaded = sector - tri
    ins = rng.choice([
        f"一个圆心角为90度的扇形，半径是{r}厘米。扇形中两条半径与弧围成的最大等腰直角三角形之外的部分是阴影，阴影面积是多少平方厘米（π取3.14）？",
        f"圆心角90度、半径{r}厘米的扇形里，有一个两条直角边都是半径的等腰直角三角形，三角形以外的阴影面积是多少平方厘米（π取3.14）？",
        f"一个扇形的圆心角是90度，半径{r}厘米。以两条半径为直角边画一个最大的等腰直角三角形，剩余阴影面积是多少平方厘米（π取3.14）？",
        f"半径{r}厘米、圆心角90度的扇形中，最大的等腰直角三角形（直角边等于半径）之外的阴影部分面积是多少平方厘米（π取3.14）？",
    ])
    lines = [
        f"90 ÷ 360 = 1/4",
        f"3.14 × {r} × {r} = {_d(circle)}平方厘米",
        f"{_d(circle)} × 1/4 = {_d(sector)}平方厘米",
        f"{r} × {r} ÷ 2 = {_d(tri)}平方厘米",
        f"{_d(sector)} - {_d(tri)} = {_d(shaded)}平方厘米",
    ]
    return ins, lines, shaded


_reg("sector_triangle_shaded", sector_triangle_shaded)


def petal_shaded(rng):
    a = rng.choice([4, 6, 8, 10, 12, 14, 16, 18, 20])
    r = Fraction(a, 2)
    circle = Fraction(314, 100) * r * r
    two = circle * 2
    sq = a * a
    petals = two - sq
    ins = rng.choice([
        f"边长{a}厘米的正方形，以每条边为直径在正方形内画半圆，四个半圆重叠形成的花瓣形阴影面积是多少平方厘米（π取3.14）？",
        f"在边长{a}厘米的正方形里，以四条边为直径向内各画一个半圆，重叠部分（花瓣形）的面积是多少平方厘米（π取3.14）？",
        f"正方形边长{a}厘米，分别以四条边为直径在正方形内画半圆，求四个半圆重叠部分的面积（π取3.14）。",
        f"边长{a}厘米的正方形中，以每条边为直径向内作半圆，四个半圆两两重叠的花瓣面积是多少平方厘米（π取3.14）？",
    ])
    lines = [
        f"{a} ÷ 2 = {num(r)}厘米",
        f"3.14 × {num(r)} × {num(r)} = {_d(circle)}平方厘米",
        f"{_d(circle)} × 2 = {_d(two)}平方厘米",
        f"{a} × {a} = {sq}平方厘米",
        f"{_d(two)} - {sq} = {_d(petals)}平方厘米",
    ]
    return ins, lines, petals


_reg("petal_shaded", petal_shaded)


def ring_sector(rng):
    angle, R, r = rng.choice([
        (60, 10, 4), (60, 12, 6), (60, 9, 3), (60, 15, 3), (60, 8, 2),
        (60, 14, 8), (90, 10, 4), (90, 10, 6), (90, 12, 4), (90, 8, 4),
        (90, 9, 5), (90, 15, 5), (90, 14, 8), (90, 10, 8), (120, 10, 4),
        (120, 12, 6), (120, 9, 3), (120, 15, 3), (120, 8, 2), (120, 18, 6),
    ])
    frac = Fraction(angle, 360)
    diff = R * R - r * r
    area = Fraction(314, 100) * diff * frac
    ins = rng.choice([
        f"一个扇环，圆心角是{angle}度，外半径{R}厘米，内半径{r}厘米。它的面积是多少平方厘米（π取3.14）？",
        f"扇环的圆心角为{angle}度，外圆半径{R}厘米，内圆半径{r}厘米，求扇环的面积（π取3.14）。",
        f"一个圆环被截去一部分，剩下的扇环圆心角{angle}度，外半径{R}厘米，内半径{r}厘米，面积是多少平方厘米（π取3.14）？",
        f"圆心角{angle}度的扇环，外半径{R}厘米，内半径{r}厘米，它的面积是多少平方厘米（π取3.14）？",
    ])
    lines = [
        f"{angle} ÷ 360 = {num(frac)}",
        f"{R} × {R} = {R * R}",
        f"{r} × {r} = {r * r}",
        f"{R * R} - {r * r} = {diff}",
        f"3.14 × {diff} × {num(frac)} = {_d(area)}平方厘米",
    ]
    return ins, lines, area


_reg("ring_sector", ring_sector)


def semicircle_triangle(rng):
    r = rng.choice([4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20])
    circle = Fraction(314, 100) * r * r
    semi = circle / 2
    base = 2 * r
    tri = Fraction(base * r, 2)
    shaded = semi - tri
    ins = rng.choice([
        f"一个半圆的半径是{r}厘米。在半圆里画一个最大的三角形（以直径为底、半径为高），三角形以外的阴影面积是多少平方厘米（π取3.14）？",
        f"半径{r}厘米的半圆中，以直径为底、半径为高画一个最大的三角形，剩余阴影面积是多少平方厘米（π取3.14）？",
        f"一个半圆形纸片半径{r}厘米，从中剪去一个最大的三角形（底是直径、高是半径），剩下的面积是多少平方厘米（π取3.14）？",
        f"半圆的半径是{r}厘米，里面最大的三角形以直径为底、半径为高，三角形外的阴影面积是多少平方厘米（π取3.14）？",
    ])
    lines = [
        f"3.14 × {r} × {r} = {_d(circle)}平方厘米",
        f"{_d(circle)} ÷ 2 = {_d(semi)}平方厘米",
        f"{r} × 2 = {base}厘米",
        f"{base} × {r} ÷ 2 = {_d(tri)}平方厘米",
        f"{_d(semi)} - {_d(tri)} = {_d(shaded)}平方厘米",
    ]
    return ins, lines, shaded


_reg("semicircle_triangle", semicircle_triangle)


def equal_perimeter_diff(rng):
    C = rng.choice([62.8, 125.6, 31.4, 94.2, 188.4, 251.2, 25.12, 12.56,
                    18.84, 6.28, 15.7, 21.98, 28.26, 37.68, 43.96, 50.24])
    Cf = Fraction(str(C))
    r = Cf / Fraction(314, 100) / 2
    side = Cf / 4
    circle = Fraction(314, 100) * r * r
    square = side * side
    diff = circle - square
    ins = rng.choice([
        f"用两根同样长的铁丝分别围成一个正方形和一个圆，铁丝长{C}厘米。圆的面积比正方形大多少平方厘米（π取3.14）？",
        f"两根{C}厘米长的铁丝，一根围成正方形，一根围成圆，圆的面积比正方形多多少平方厘米（π取3.14）？",
        f"用长{C}厘米的铁丝分别围成正方形和圆（各一根），圆的面积比正方形的面积大多少平方厘米（π取3.14）？",
        f"一根铁丝长{C}厘米，围成圆；另一根同样长的铁丝围成正方形。圆的面积比正方形大多少平方厘米（π取3.14）？",
    ])
    lines = [
        f"{C} ÷ 3.14 ÷ 2 = {_d(r)}厘米",
        f"{C} ÷ 4 = {_d(side)}厘米",
        f"3.14 × {_d(r)} × {_d(r)} = {_d(circle)}平方厘米",
        f"{_d(side)} × {_d(side)} = {_d(square)}平方厘米",
        f"{_d(circle)} - {_d(square)} = {_d(diff)}平方厘米",
    ]
    return ins, lines, diff


_reg("equal_perimeter_diff", equal_perimeter_diff)


def cuboid_net_square(rng):
    a = rng.choice([3, 4, 5, 6, 7, 8, 9, 10])
    h = 4 * a
    base = a * a
    V = base * h
    ins = rng.choice([
        f"一个长方体，底面是边长{a}厘米的正方形，它的侧面展开后正好是一个正方形。这个长方体的体积是多少立方厘米？",
        f"一个长方体的底面是边长{a}厘米的正方形，把它的侧面展开得到一个正方形，求这个长方体的体积。",
        f"长方体底面为边长{a}厘米的正方形，侧面展开图是一个正方形，它的体积是多少立方厘米？",
        f"一个长方体，底面是边长{a}厘米的正方形，沿侧面的高剪开后展开正好是个正方形，这个长方体的体积是多少立方厘米？",
    ])
    lines = [
        f"{a} × 4 = {h}厘米",
        f"{a} × {a} = {base}平方厘米",
        f"{base} × {h} = {V}立方厘米",
    ]
    return ins, lines, V


_reg("cuboid_net_square", cuboid_net_square)


def cylinder_net_square(rng):
    a = rng.choice([6.28, 12.56, 18.84, 25.12, 3.14, 9.42, 15.7, 21.98,
                    28.26, 31.4, 37.68, 43.96])
    af = Fraction(str(a))
    r = af / Fraction(314, 100) / 2
    base = Fraction(314, 100) * r * r
    V = base * af
    ins = rng.choice([
        f"一个圆柱的侧面展开后是一个边长{a}厘米的正方形，这个圆柱的体积是多少立方厘米（π取3.14）？",
        f"把一个圆柱的侧面沿高剪开，展开后是边长{a}厘米的正方形，圆柱的体积是多少立方厘米（π取3.14）？",
        f"圆柱的侧面展开图是一个边长{a}厘米的正方形，求这个圆柱的体积（π取3.14）。",
        f"一个圆柱体，侧面展开后是边长{a}厘米的正方形，它的体积是多少立方厘米（π取3.14）？",
    ])
    lines = [
        f"{a} ÷ 3.14 ÷ 2 = {_d(r)}厘米",
        f"3.14 × {_d(r)} × {_d(r)} = {_d(base)}平方厘米",
        f"{_d(base)} × {a} = {_d(V)}立方厘米",
    ]
    return ins, lines, V


_reg("cylinder_net_square", cylinder_net_square)


def cube_cut_volume(rng):
    inc = rng.choice([32, 50, 72, 98, 128, 162, 200, 242, 288, 338,
                      392, 450, 512, 578, 648, 722, 800, 882, 968])
    face = inc // 2
    edge = int(face ** 0.5)
    V = edge ** 3
    ins = rng.choice([
        f"把一个正方体切成两个完全相同的长方体，表面积增加了{inc}平方厘米。原来正方体的体积是多少立方厘米？",
        f"一个正方体被切成两个一样的长方体，表面积增加{inc}平方厘米，原正方体的体积是多少立方厘米？",
        f"把正方体沿平行于一个面的方向切成两个相等的长方体，表面积增加了{inc}平方厘米，原正方体体积是多少立方厘米？",
        f"一个正方体切成两个完全相同的长方体后，表面积比原来增加{inc}平方厘米，原来正方体的体积是多少立方厘米？",
    ])
    lines = [
        f"{inc} ÷ 2 = {face}平方厘米",
        f"棱长 = {edge} = {edge}厘米",
        f"{edge} × {edge} × {edge} = {V}立方厘米",
    ]
    return ins, lines, V


_reg("cube_cut_volume", cube_cut_volume)


def cone_cylinder_base(rng):
    r, h = rng.choice([
        (6, 15), (9, 12), (12, 9), (3, 18), (6, 12), (9, 15), (12, 15),
        (15, 9), (3, 12), (6, 18), (9, 18), (12, 18), (15, 12), (15, 15),
        (15, 18), (18, 12), (18, 15), (18, 18), (3, 15), (3, 9),
    ])
    cyl = Fraction(314, 100) * r * r * h
    cone = cyl / 3
    H = Fraction(h, 3)
    ins = rng.choice([
        f"把一个底面半径{r}厘米、高{h}厘米的圆锥形铁块熔铸成一个与它等底的圆柱，这个圆柱的高是多少厘米（π取3.14）？",
        f"一个圆锥底面半径{r}厘米、高{h}厘米，把它熔铸成等底的圆柱，圆柱的高是多少厘米（π取3.14）？",
        f"圆锥形零件底面半径{r}厘米、高{h}厘米，将它熔铸成底面积相等的圆柱，圆柱的高是多少厘米（π取3.14）？",
        f"把底面半径{r}厘米、高{h}厘米的圆锥熔铸成一个和它等底的圆柱，圆柱的高是多少厘米（π取3.14）？",
    ])
    lines = [
        f"3.14 × {r} × {r} × {h} = {_d(cyl)}立方厘米",
        f"{_d(cyl)} ÷ 3 = {_d(cone)}立方厘米",
        f"{_d(cone)} ÷ 3.14 ÷ {r} ÷ {r} = {num(H)}厘米",
    ]
    return ins, lines, H


_reg("cone_cylinder_base", cone_cylinder_base)


def cone_frustum(rng):
    r, h = rng.choice([
        (9, 3), (9, 6), (9, 9), (9, 12), (9, 15), (9, 18), (9, 21),
        (9, 24), (9, 27), (9, 30), (6, 9), (6, 18), (6, 27), (3, 9),
        (3, 18), (3, 27), (12, 9), (12, 18), (12, 27), (15, 9),
    ])
    h3 = h // 3
    V = Fraction(314, 100) * r * r * h / 3
    small = V / 27
    frustum = V - small
    ins = rng.choice([
        f"一个圆锥形零件，底面半径{r}厘米，高{h}厘米。从顶点截去一个高{h3}厘米的小圆锥，剩下的圆台体积是多少立方厘米（π取3.14）？",
        f"圆锥底面半径{r}厘米、高{h}厘米，沿平行于底面的方向在距顶点{h3}厘米处截去小圆锥，剩余圆台的体积是多少立方厘米（π取3.14）？",
        f"一个圆锥底面半径{r}厘米、高{h}厘米，从顶点向下{h3}厘米处横着截去一个小圆锥，剩下部分的体积是多少立方厘米（π取3.14）？",
        f"圆锥形木块底面半径{r}厘米、高{h}厘米，截去顶点处高{h3}厘米的小圆锥后，圆台的体积是多少立方厘米（π取3.14）？",
    ])
    lines = [
        f"3.14 × {r} × {r} × {h} ÷ 3 = {_d(V)}立方厘米",
        f"{h} ÷ 3 = {h3}厘米",
        f"{h} ÷ {h3} = 3",
        f"3 × 3 × 3 = 27",
        f"{_d(V)} ÷ 27 = {_d(small)}立方厘米",
        f"{_d(V)} - {_d(small)} = {_d(frustum)}立方厘米",
    ]
    return ins, lines, frustum


_reg("cone_frustum", cone_frustum)


def staircase_section(rng):
    a, b, n = rng.choice([
        (3, 2, 5), (3, 3, 4), (2, 3, 6), (4, 3, 5), (3, 4, 5),
        (2, 2, 10), (3, 2, 8), (4, 2, 6), (5, 3, 4), (3, 5, 4),
        (2, 4, 7), (3, 3, 6), (2, 3, 8), (4, 4, 5), (5, 4, 5),
        (3, 2, 6), (4, 3, 6), (5, 3, 5), (3, 5, 5), (2, 5, 6),
    ])
    tri = n * (n + 1) // 2
    ab = a * b
    area = ab * tri
    ins = rng.choice([
        f"楼梯的截面图中，每级台阶宽{a}分米、高{b}分米，共{n}级。这个截面的面积是多少平方分米？",
        f"一个楼梯截面，每级台阶宽{a}分米、高{b}分米，一共有{n}级台阶，截面面积是多少平方分米？",
        f"如图，楼梯每级宽{a}分米、高{b}分米，共{n}级，求这个楼梯截面的面积。",
        f"楼梯截面由{n}级台阶组成，每级台阶宽{a}分米、高{b}分米，截面的总面积是多少平方分米？",
    ])
    lines = [
        f"{n} + 1 = {n + 1}",
        f"{n} × {n + 1} ÷ 2 = {tri}",
        f"{a} × {b} = {ab}平方分米",
        f"{ab} × {tri} = {area}平方分米",
    ]
    return ins, lines, area


_reg("staircase_section", staircase_section)


def grid_triangle_area(rng):
    x, u, v = rng.choice([
        (8, 2, 6), (10, 4, 5), (12, 3, 8), (9, 3, 6), (10, 2, 8),
        (12, 4, 5), (15, 5, 6), (8, 6, 5), (14, 6, 5), (16, 4, 6),
        (10, 6, 5), (12, 8, 6), (15, 3, 8), (18, 6, 5), (20, 8, 6),
    ])
    rect = x * v
    t1 = Fraction(u * v, 2)
    xu = x - u
    t2 = Fraction(xu * v, 2)
    area = rect - t1 - t2
    ins = rng.choice([
        f"方格纸上一个三角形，三个顶点的位置用数对表示为(0,0)、({x},0)、({u},{v})（每个小方格边长1厘米）。用割补法求这个三角形的面积。",
        f"在方格图中，三角形三个顶点分别在(0,0)、({x},0)、({u},{v})处（每格边长1厘米），求三角形的面积。",
        f"一个三角形的三个顶点坐标为(0,0)、({x},0)、({u},{v})（单位：厘米），用割补法求它的面积。",
        f"方格纸上，三角形的顶点在(0,0)、({x},0)、({u},{v})三个格点上（每格1厘米），这个三角形的面积是多少平方厘米？",
    ])
    lines = [
        f"{x} × {v} = {rect}平方厘米",
        f"{u} × {v} ÷ 2 = {num(t1)}平方厘米",
        f"{x} - {u} = {xu}",
        f"{xu} × {v} ÷ 2 = {num(t2)}平方厘米",
        f"{rect} - {num(t1)} - {num(t2)} = {num(area)}平方厘米",
    ]
    return ins, lines, area


_reg("grid_triangle_area", grid_triangle_area)


def fence_three_sides(rng):
    L = rng.choice([20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68,
                    72, 76, 80, 88, 96, 100, 120])
    w = L // 4
    l = L // 2
    area = l * w
    ins = rng.choice([
        f"用{L}米长的篱笆围一块长方形菜地，一面靠墙（只围三条边）。怎样围面积最大？最大面积是多少平方米？",
        f"张大爷用{L}米篱笆靠墙围一个长方形菜园，怎样围面积最大？最大是多少平方米？",
        f"用{L}米长的篱笆围一个长方形花圃，一边靠墙，怎样围面积最大？最大面积是多少平方米？",
        f"农场主用{L}米的篱笆围长方形菜地，一面利用房屋的墙，怎样围面积最大？最大面积多少平方米？",
    ])
    lines = [
        f"{L} ÷ 4 = {w}米",
        f"{w} × 2 = {l}米",
        f"{l} × {w} = {area}平方米",
    ]
    return ins, lines, area


_reg("fence_three_sides", fence_three_sides)


def ant_crawl(rng):
    a, b, c, ans = rng.choice([
        (3, 5, 6, 10), (6, 10, 12, 20), (9, 15, 18, 30), (12, 20, 24, 40),
        (15, 25, 30, 50), (18, 30, 36, 60), (21, 35, 42, 70),
        (24, 40, 48, 80), (27, 45, 54, 90), (30, 50, 60, 100),
        (9, 12, 20, 29), (12, 16, 21, 35), (18, 24, 40, 58),
        (24, 32, 42, 70), (3, 5, 6, 10), (6, 10, 12, 20),
    ])
    ab = a + b
    d2 = ab * ab + c * c
    ac = a + c
    e2 = ac * ac + b * b
    de = e2 - d2
    ins = rng.choice([
        f"一个长方体盒子，长{a}厘米、宽{b}厘米、高{c}厘米。一只蚂蚁从一个顶点沿表面爬到对角顶点，最短路线长多少厘米？",
        f"长方体木块长{a}厘米、宽{b}厘米、高{c}厘米，一只蚂蚁从一个顶点沿表面爬到相对的顶点，最短路程是多少厘米？",
        f"一个长方体房间长{a}米、宽{b}米、高{c}米，一只蚂蚁从墙角沿墙面爬到对角的墙角，最短要爬多少米？",
        f"长方体礼盒长{a}厘米、宽{b}厘米、高{c}厘米，蚂蚁从一个顶点沿表面爬到对角顶点，最短路线长多少厘米？",
    ])
    lines = [
        f"{a} + {b} = {ab}",
        f"({ab}) × ({ab}) + ({c}) × ({c}) = {d2}",
        f"({a} + {c}) × ({a} + {c}) + ({b}) × ({b}) = {e2}",
        f"{e2} - {d2} = {de}",
        f"最短 = {ans} = {ans}厘米",
    ]
    return ins, lines, ans


_reg("ant_crawl", ant_crawl)


def dice_opposite(rng):
    top = rng.randint(1, 6)
    bottom = 7 - top
    ins = rng.choice([
        f"一个正方体骰子，六个面上分别写着1、2、3、4、5、6，相对两个面上的数之和都是7。如果{top}朝上，朝下的数是几？",
        f"骰子六个面分别写1、2、3、4、5、6，相对面的和都是7。掷出后{top}点朝上，底面是几点？",
        f"一个正方体六个面上分别写着1到6，相对两个面的和都是7。当{top}朝上时，朝下的面上写的是几？",
        f"骰子的六个面写着1、2、3、4、5、6，每对相对面的和都是7。{top}朝上时，朝下的数是多少？",
        f"正方体骰子相对两面的和都是7，六个面写着1、2、3、4、5、6。{top}点朝上时，对面是几点？",
        f"一个骰子六个面上的数是1、2、3、4、5、6，相对面之和为7。朝上的是{top}，朝下的数是几？",
    ])
    lines = [
        f"1 + 6 = 7",
        f"2 + 5 = 7",
        f"3 + 4 = 7",
        f"{top} + {bottom} = 7",
        f"7 - {top} = {bottom}",
    ]
    return ins, lines, bottom


_reg("dice_opposite", dice_opposite)

# ---------------------------------------------------------------------------
# Batch 3: mirror time, tables, patterns, stats, probability, logic, ages
# ---------------------------------------------------------------------------

def mirror_time(rng):
    h = rng.randint(2, 9)
    m = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    mins = 60 * h + m
    ans = 720 - mins
    ins = rng.choice([
        f"小明从镜子里看到钟面的时间是{h}点{m}分，实际时间折合多少分钟（从0时算起的分钟数）？",
        f"镜子中看到的时钟显示{h}点{m}分，此时的实际时间折合多少分钟（0时起算）？",
        f"从镜子里看钟表是{h}点{m}分，实际时间是多少分钟（以午夜0时为0分计算）？",
        f"钟面在镜子中的像是{h}点{m}分，实际时间折合多少分钟（从0时开始算）？",
    ])
    lines = [
        f"12 × 60 = 720分",
        f"{h} × 60 + {m} = {mins}分",
        f"720 - {mins} = {ans}分",
    ]
    return ins, lines, ans


_reg("mirror_time", mirror_time)


def number_table_nth(rng):
    k = rng.randint(4, 8)
    n = rng.randint(5, 15)
    m = rng.randint(1, k)
    base = k * (n - 1)
    ans = base + m
    ins = rng.choice([
        f"自然数从1开始按每行{k}个数依次排列，第{n}行第{m}个数是多少？",
        f"把自然数按每行{k}个排列成数表，第{n}行的第{m}个数是几？",
        f"数表中每行排{k}个连续自然数，第{n}行第{m}个数是多少？",
        f"自然数从1起每行{k}个数排列，第{n}行第{m}个数是多少？",
    ])
    lines = [
        f"{n} - 1 = {n - 1}",
        f"{k} × {n - 1} = {base}",
        f"{k} × {n} = {k * n}",
        f"{base} + {m} = {ans}",
    ]
    return ins, lines, ans


_reg("number_table_nth", number_table_nth)


def triangle_table_row_sum(rng):
    n = rng.randint(3, 12)
    n2 = n * n
    ans = n * (n2 + 1) // 2
    ins = rng.choice([
        f"自然数排成三角形数表：第一行一个数，第二行两个数，第三行三个数……第{n}行所有数的和是多少？",
        f"把自然数排成三角形：第一行一个，第二行两个，第三行三个……第{n}行各数之和是多少？",
        f"三角形数表第一行是1，第二行两个数，第三行三个数……第{n}行所有数的和是多少？",
        f"自然数按三角形排列，第几行就有几个数，第{n}行所有数的和是多少？",
    ])
    lines = [
        f"{n} × {n} = {n2}",
        f"{n2} + 1 = {n2 + 1}",
        f"{n} × ({n2} + 1) ÷ 2 = {ans}",
    ]
    return ins, lines, ans


_reg("triangle_table_row_sum", triangle_table_row_sum)


def table_first_n_rows_sum(rng):
    k = rng.randint(3, 8)
    n = rng.randint(5, 15)
    kn = k * n
    ans = kn * (kn + 1) // 2
    ins = rng.choice([
        f"自然数从1开始按每行{k}个数排列成数表，前{n}行所有数的和是多少？",
        f"数表每行{k}个连续自然数，前{n}行的所有数之和是多少？",
        f"把自然数从1起每行{k}个排列，前{n}行全部数的和是多少？",
        f"自然数数表每行{k}个数，前{n}行所有数加起来的和是多少？",
    ])
    lines = [
        f"{k} × {n} = {kn}",
        f"{kn} + 1 = {kn + 1}",
        f"{kn} × {kn + 1} ÷ 2 = {ans}",
    ]
    return ins, lines, ans


_reg("table_first_n_rows_sum", table_first_n_rows_sum)


def dot_square_pattern(rng):
    n = rng.randint(5, 20)
    n2 = n * n
    n1 = n - 1
    n12 = n1 * n1
    ans = n2 - n12
    ins = rng.choice([
        f"用圆点摆正方形点阵：第一个图形每边两个点，第二个每边三个点……第{n}个图形比第{n - 1}个图形多多少个点？",
        f"正方形点阵第一个每边两个点，第二个每边三个点，照这样摆下去，第{n}个点阵比第{n - 1}个多几个点？",
        f"用圆点摆正方形，第一个每边两个点，第二个每边三个点……第{n}个正方形比第{n - 1}个多用多少个点？",
        f"摆正方形点阵，第一个每边两点，第二个每边三点，第{n}个比第{n - 1}个多多少个点？",
    ])
    lines = [
        f"{n} × {n} = {n2}个",
        f"{n} - 1 = {n1}",
        f"{n1} × {n1} = {n12}个",
        f"{n2} - {n12} = {ans}个",
    ]
    return ins, lines, ans


_reg("dot_square_pattern", dot_square_pattern)


def cube_stack_layers(rng):
    n = rng.randint(3, 10)
    n1 = n + 1
    n2 = 2 * n + 1
    prod = n * n1 * n2
    ans = prod // 6
    ins = rng.choice([
        f"把小正方体堆成金字塔形：第一层1个，第二层四个，第三层九个……照这样，第{n}层有{n}×{n}个。堆{n}层共需多少个小正方体？",
        f"小正方体堆成塔：第一层1个，第二层四个，第三层九个……第{n}层是{n}×{n}个，{n}层一共多少个？",
        f"堆正方体塔，第一层1个，第二层四个，第三层九个，每层个数是层数乘层数，堆{n}层共要多少个小正方体？",
        f"用小正方体堆金字塔，第一层1个，第二层四个，第三层九个……第{n}层{n}×{n}个，{n}层共需多少个？",
    ])
    lines = [
        f"{n} + 1 = {n1}",
        f"2 × {n} + 1 = {n2}",
        f"{n} × {n1} × {n2} = {prod}个",
        f"{prod} ÷ 6 = {ans}个",
    ]
    return ins, lines, ans


_reg("cube_stack_layers", cube_stack_layers)


def square_grid_matches(rng):
    k = rng.randint(3, 10)
    k1 = k + 1
    kk = k * k1
    ans = 2 * kk
    ins = rng.choice([
        f"用火柴棒摆成{k}行{k}列的小正方形网格（每个小格边长为一根火柴），一共需要多少根火柴棒？",
        f"摆一个{k}行{k}列的正方形网格，每个小正方形边长用一根火柴，共需多少根火柴棒？",
        f"用火柴棒摆{k}×{k}的方格网，每根火柴是小格的一条边，一共要用多少根火柴？",
        f"拼成{k}行{k}列的小正方形网格，每个小格边长一根火柴，需要多少根火柴棒？",
    ])
    lines = [
        f"{k} + 1 = {k1}",
        f"{k} × {k} = {k * k}个",
        f"{k} × {k1} = {kk}根",
        f"2 × {kk} = {ans}根",
    ]
    return ins, lines, ans


_reg("square_grid_matches", square_grid_matches)


def polygon_diagonals(rng):
    n = rng.choice([5, 6, 7, 8, 9, 10, 11, 12, 15, 20])
    n3 = n - 3
    ans = n * n3 // 2
    ins = rng.choice([
        f"一个{n}边形，从每个顶点出发可以画{n - 3}条对角线。这个{n}边形一共有多少条对角线？",
        f"从{n}边形的一个顶点出发能画{n - 3}条对角线，这个{n}边形对角线总数是多少条？",
        f"一个{n}边形共有多少条对角线（每个顶点可画{n - 3}条）？",
        f"n边形一个顶点可画n-3条对角线，当n={n}时，这个多边形共有多少条对角线？",
    ])
    lines = [
        f"{n} - 3 = {n3}条",
        f"{n} × {n3} = {n * n3}条",
        f"2 × {ans} = {n * n3}条",
        f"{n * n3} ÷ 2 = {ans}条",
    ]
    return ins, lines, ans


_reg("polygon_diagonals", polygon_diagonals)


def median_overlap(rng):
    a, b, c = rng.choice([
        (80, 90, 85), (70, 80, 75), (90, 80, 85), (60, 70, 65),
        (85, 95, 90), (75, 85, 80), (95, 85, 90), (88, 92, 90),
        (82, 88, 85), (78, 82, 80), (92, 88, 90), (72, 78, 75),
        (86, 94, 90), (68, 72, 70), (90, 95, 90), (85, 80, 80),
    ])
    s1 = 3 * a
    s2 = 3 * b
    ss = s1 + s2
    sum5 = ss - c
    avg = Fraction(sum5, 5)
    ins = rng.choice([
        f"小明五次数学测验，前三次的平均分是{a}分，后三次的平均分是{b}分，第三次测验得{c}分（五次成绩的中位数）。五次测验的平均分是多少？",
        f"小红五次考试成绩，前三次平均{a}分，后三次平均{b}分，第三次考了{c}分。五次的平均分是多少分？",
        f"小华五次测验，前三次平均分{a}，后三次平均分{b}，第三次得{c}分，五次的总平均分是多少？",
        f"小丽五次数学成绩，前三次平均{a}分，后三次平均{b}分，已知第三次是{c}分，她五次的平均分是多少？",
    ])
    lines = [
        f"3 × {a} = {s1}分",
        f"3 × {b} = {s2}分",
        f"{s1} + {s2} = {ss}分",
        f"{ss} - {c} = {sum5}分",
        f"{sum5} ÷ 5 = {num(avg)}分",
    ]
    return ins, lines, avg


_reg("median_overlap", median_overlap)


def remove_high_low(rng):
    while True:
        scores = rng.sample(range(70, 100), 5)
        s = sorted(scores)
        if sum(s[1:4]) % 3 == 0:
            break
    s1, s2, s3, s4, s5 = s
    sum3 = s2 + s3 + s4
    avg = sum3 // 3
    ins = rng.choice([
        f"评委给选手打分：{s1}、{s2}、{s3}、{s4}、{s5}分。按规则去掉一个最高分和一个最低分，选手的平均得分是多少？",
        f"五位评委打分分别为{s1}、{s2}、{s3}、{s4}、{s5}分，去掉最高分和最低分后，平均分是多少？",
        f"选手得分：{s1}、{s2}、{s3}、{s4}、{s5}分，去掉一个最高分和一个最低分，他的平均得分是多少？",
        f"比赛打分：{s1}、{s2}、{s3}、{s4}、{s5}分，按规则去掉最高和最低各一个，平均得分是多少分？",
    ])
    lines = [
        f"最高分 = {s5} = {s5}分",
        f"最低分 = {s1} = {s1}分",
        f"{s2} + {s3} + {s4} = {sum3}分",
        f"{sum3} ÷ 3 = {avg}分",
    ]
    return ins, lines, avg


_reg("remove_high_low", remove_high_low)


def fair_add_balls(rng):
    a, b = rng.choice([
        (3, 5), (4, 7), (5, 8), (6, 10), (7, 11), (8, 12), (5, 9),
        (6, 9), (7, 10), (8, 11), (9, 12), (10, 15), (12, 18), (10, 14),
        (12, 16), (14, 20), (15, 20), (16, 20), (18, 24), (20, 30),
    ])
    x = b - a
    ins = rng.choice([
        f"袋子里有{a}个红球和{b}个白球（白球比红球多）。再放入多少个红球后，摸到红球和白球的可能性相同？",
        f"盒中有{a}个红球、{b}个白球，再放入几个红球，摸到两种球的可能性就相同了？",
        f"口袋里装{a}个红球和{b}个白球，再放进多少个红球，摸球游戏才公平？",
        f"袋子中有{a}个红球、{b}个白球，再放入多少个红球后，摸到红球与白球的可能性相等？",
    ])
    lines = [
        f"{a} + {x} = {b}个",
        f"{a} + {b} = {a + b}个",
        f"{b} + {b} = {2 * b}个",
        f"{b} - {a} = {x}个",
    ]
    return ins, lines, x


_reg("fair_add_balls", fair_add_balls)


def cards_odd_product(rng):
    N = rng.choice([6, 8, 10, 12, 14, 16, 18, 20, 24])
    odd = N // 2
    fav = odd * (odd - 1) // 2
    tot = N * (N - 1) // 2
    ans = Fraction(fav, tot)
    ins = rng.choice([
        f"把写着1到{N}的{N}张卡片打乱，任意抽出两张，两张卡片上的数之积为奇数的可能性是几分之几？",
        f"有写着1到{N}的卡片各一张，任意抽两张，两张数的乘积是奇数的可能性是多少？",
        f"从1到{N}的{N}张卡片中任意摸出两张，两张卡片上数的积为奇数的可能性是几分之几？",
        f"将1至{N}的{N}张卡片洗匀，一次抽出两张，积为奇数的可能性是多少？",
    ])
    lines = [
        f"{N} ÷ 2 = {odd}张",
        f"{odd} × {odd - 1} ÷ 2 = {fav}种",
        f"{N} × {N - 1} ÷ 2 = {tot}种",
        f"{fav} ÷ {tot} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("cards_odd_product", cards_odd_product)


def prob_two_red_replacement(rng):
    a, b = rng.choice([
        (3, 2), (2, 3), (1, 1), (3, 3), (4, 1), (1, 4), (2, 2),
        (5, 3), (3, 5), (4, 4), (5, 5), (6, 2), (2, 6), (7, 2),
        (2, 7), (8, 2), (2, 8), (6, 4), (4, 6), (7, 3),
    ])
    n = a + b
    p1 = Fraction(a, n)
    ans = p1 * p1
    ins = rng.choice([
        f"袋子里有{a}个红球和{b}个白球，每次摸出一个球后放回，摇匀再摸。两次都摸到红球的可能性是几分之几？",
        f"盒中有{a}个红球、{b}个白球，摸出一个记下颜色后放回，再摸一个。两次都是红球的可能性是多少？",
        f"口袋里装{a}个红球和{b}个白球，有放回地摸两次，两次都摸到红球的可能性是几分之几？",
        f"袋子中有{a}个红球、{b}个白球，每次摸完放回，摸两次，两次都是红球的可能性是多少？",
    ])
    lines = [
        f"{a} + {b} = {n}个",
        f"{a} ÷ {n} = {num(p1)}",
        f"{num(p1)} × {num(p1)} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("prob_two_red_replacement", prob_two_red_replacement)


def logic_rank_chain(rng):
    chain = rng.sample([1, 2, 3, 4, 5], 5)
    a, b, c, d, e = chain
    target = rng.choice(chain)
    pos = chain.index(target) + 1
    before = pos - 1
    after = 5 - pos
    clues = rng.choice([
        f"{a}比{b}快，{b}比{c}快，{c}比{d}快，{d}比{e}快",
        f"{a}比{b}快，{c}比{d}快，{b}比{c}快，{d}比{e}快",
        f"{d}比{e}快，{c}比{d}快，{b}比{c}快，{a}比{b}快",
        f"{b}比{c}快，{a}比{b}快，{d}比{e}快，{c}比{d}快",
    ])
    ins = rng.choice([
        f"编号1到5的五名运动员赛跑，没有并列。已知：{clues}。{target}号运动员是第几名？",
        f"五名运动员编号1、2、3、4、5，成绩各不相同。已知：{clues}。{target}号跑了第几名？",
        f"1至5号运动员赛跑，已知：{clues}。{target}号运动员排第几名？",
        f"编号1到5的五人比赛跑步，无并列。已知：{clues}。{target}号是第几名？",
    ])
    lines = [
        f"2 + 3 = 5人",
        f"1 + 4 = 5人",
        f"{before} + 1 = {pos}名",
        f"{pos} + {after} = 5人",
        f"5 - {pos} = {after}人",
        f"5 - {after} = {pos}名",
    ]
    return ins, lines, pos


_reg("logic_rank_chain", logic_rank_chain)


def logic_seat(rng):
    seats = rng.sample([1, 2, 3, 4, 5], 5)
    a, b, c, d, e = seats
    target = rng.choice(seats)
    pos = seats.index(target) + 1
    before = pos - 1
    after = 5 - pos
    ins = rng.choice([
        f"编号1到5的五人坐一排五个座位。已知：{a}坐第一个座位，{c}坐第三个座位，{b}坐在{a}和{c}中间，{e}坐第五个座位，{d}坐剩下的座位。{target}坐在第几个座位？",
        f"五个编号1、2、3、4、5的人坐成一排。{a}坐第一个，{c}坐第三个，{b}坐在{a}和{c}之间，{e}坐第五个，{d}坐剩余的座位。{target}坐第几个？",
        f"一排五个座位坐五人，编号1到5。已知{a}坐第一位，{c}坐第三位，{b}在{a}和{c}中间，{e}坐第五位，{d}坐剩下的位置。{target}坐在第几位？",
        f"五人坐一排五个座位，编号1至5。{a}坐第一个座位，{c}坐第三个，{b}坐在{a}与{c}中间，{e}坐第五个，{d}坐其余座位。{target}坐第几个座位？",
    ])
    lines = [
        f"2 + 3 = 5个",
        f"1 + 4 = 5个",
        f"{before} + 1 = {pos}个",
        f"{pos} + {after} = 5个",
        f"5 - {pos} = {after}个",
        f"5 - {after} = {pos}个",
    ]
    return ins, lines, pos


_reg("logic_seat", logic_seat)


def age_chain(rng):
    d, c1, c2, c3 = rng.choice([
        (10, 1, 3, 2), (9, 2, 4, 3), (8, 1, 5, 2), (10, 2, 3, 5),
        (7, 3, 2, 4), (12, 1, 4, 2), (11, 2, 5, 3), (10, 3, 2, 4),
        (9, 1, 6, 3), (8, 2, 3, 5), (10, 4, 2, 3), (11, 1, 3, 4),
        (12, 2, 2, 3), (9, 3, 4, 2), (10, 2, 2, 5), (8, 4, 3, 2),
    ])
    p1 = d + c1
    p2 = p1 + c2
    ans = p2 + c3
    ins = rng.choice([
        f"甲、乙、丙、丁四人比年龄。丙比丁大{c1}岁，乙比丙大{c2}岁，甲比乙大{c3}岁，丁今年{d}岁。甲今年多少岁？",
        f"四个人比年龄：丙比丁大{c1}岁，乙比丙大{c2}岁，甲比乙大{c3}岁。丁今年{d}岁，甲今年多少岁？",
        f"丙比丁大{c1}岁，乙比丙大{c2}岁，甲比乙大{c3}岁，丁今年{d}岁。甲今年多少岁？",
        f"甲、乙、丙、丁四人，丙比丁大{c1}岁，乙比丙大{c2}岁，甲比乙大{c3}岁。如果丁{d}岁，甲多少岁？",
    ])
    lines = [
        f"{d} + {c1} = {p1}岁",
        f"{p1} + {c2} = {p2}岁",
        f"{p2} + {c3} = {ans}岁",
    ]
    return ins, lines, ans


_reg("age_chain", age_chain)

# ---------------------------------------------------------------------------
# Batch 4: ages, max/min, splits, surplus/shortage, schemes, classics
# ---------------------------------------------------------------------------

def age_years_ago(rng):
    a, b, k = rng.choice([
        (42, 12, 4), (39, 12, 4), (36, 12, 4), (33, 12, 4), (30, 12, 4),
        (45, 15, 4), (42, 15, 4), (39, 15, 4), (36, 15, 4), (33, 15, 4),
        (30, 15, 4), (40, 14, 3), (38, 14, 3), (36, 14, 3), (34, 14, 3),
        (32, 14, 3), (30, 14, 3), (28, 14, 3), (49, 13, 5), (45, 13, 5),
        (41, 13, 5), (37, 13, 5), (33, 13, 5), (29, 13, 5), (25, 13, 5),
        (44, 10, 6), (40, 10, 6), (36, 10, 6), (32, 10, 6), (28, 10, 6),
    ])
    kb = k * b
    diff = kb - a
    k1 = k - 1
    x = Fraction(diff, k1)
    ins = rng.choice([
        f"父亲今年{a}岁，儿子今年{b}岁。几年前父亲的年龄是儿子的{k}倍？",
        f"爸爸今年{a}岁，小明今年{b}岁，几年前爸爸的年龄是小明的{k}倍？",
        f"今年父亲{a}岁、儿子{b}岁，几年前父亲年龄是儿子的{k}倍？",
        f"妈妈今年{a}岁，女儿今年{b}岁，几年前妈妈的年龄是女儿的{k}倍？",
    ])
    lines = [
        f"{k} × {b} = {kb}岁",
        f"{kb} - {a} = {diff}岁",
        f"{k} - 1 = {k1}",
        f"{diff} ÷ {k1} = {num(x)}年",
    ]
    return ins, lines, x


_reg("age_years_ago", age_years_ago)


def age_future_multiple(rng):
    a, b, k = rng.choice([
        (40, 12, 2), (42, 12, 2), (45, 12, 2), (38, 12, 2), (36, 12, 2),
        (34, 12, 2), (33, 12, 2), (32, 12, 2), (31, 12, 2), (30, 12, 2),
        (40, 10, 3), (42, 10, 3), (44, 10, 3), (46, 10, 3), (48, 10, 3),
        (50, 10, 3), (38, 10, 3), (36, 10, 3), (34, 10, 3), (32, 10, 3),
        (45, 9, 4), (48, 9, 4), (51, 9, 4), (54, 9, 4), (57, 9, 4),
        (60, 9, 4), (42, 9, 4), (39, 9, 4), (50, 14, 3), (52, 14, 3),
    ])
    kb = k * b
    diff = a - kb
    k1 = k - 1
    x = Fraction(diff, k1)
    ins = rng.choice([
        f"父亲今年{a}岁，儿子今年{b}岁。几年后父亲的年龄是儿子的{k}倍？",
        f"爸爸今年{a}岁，小明今年{b}岁，几年后爸爸的年龄是小明的{k}倍？",
        f"今年父亲{a}岁、儿子{b}岁，几年后父亲年龄是儿子的{k}倍？",
        f"妈妈今年{a}岁，女儿今年{b}岁，几年后妈妈的年龄是女儿的{k}倍？",
    ])
    lines = [
        f"{k} × {b} = {kb}岁",
        f"{a} - {kb} = {diff}岁",
        f"{k} - 1 = {k1}",
        f"{diff} ÷ {k1} = {num(x)}年",
    ]
    return ins, lines, x


_reg("age_future_multiple", age_future_multiple)


def age_sum_future(rng):
    a, b, S = rng.choice([
        (12, 8, 30), (12, 8, 40), (12, 8, 50), (10, 6, 28), (10, 6, 36),
        (10, 6, 44), (14, 10, 40), (14, 10, 50), (14, 10, 60), (11, 7, 30),
        (11, 7, 40), (11, 7, 50), (13, 9, 40), (13, 9, 50), (13, 9, 60),
        (15, 11, 50), (15, 11, 60), (15, 11, 70), (16, 12, 60), (16, 12, 70),
        (16, 12, 80), (18, 12, 70), (18, 12, 80), (18, 12, 90), (20, 10, 60),
        (20, 10, 70), (20, 10, 80), (22, 14, 80), (22, 14, 90), (22, 14, 100),
    ])
    ab = a + b
    diff = S - ab
    x = diff // 2
    ins = rng.choice([
        f"今年甲{a}岁、乙{b}岁，几年后两人的年龄和是{S}岁？",
        f"小明今年{a}岁，小红今年{b}岁，几年后他们的年龄和为{S}岁？",
        f"今年哥哥{a}岁、妹妹{b}岁，几年后两人年龄和是{S}岁？",
        f"甲今年{a}岁，乙今年{b}岁，几年后甲乙年龄和是{S}岁？",
    ])
    lines = [
        f"{a} + {b} = {ab}岁",
        f"{S} - {ab} = {diff}岁",
        f"{diff} ÷ 2 = {x}年",
    ]
    return ins, lines, x


_reg("age_sum_future", age_sum_future)


def max_product_two_digit(rng):
    while True:
        ds = rng.sample(range(1, 10), 4)
        a, b, c, d = sorted(ds)
        p1 = (10 * d + a) * (10 * c + b)
        p2 = (10 * d + b) * (10 * c + a)
        if p1 != p2:
            break
    if p1 < p2:
        p1, p2 = p2, p1
        n1, n2 = 10 * d + b, 10 * c + a
        n3, n4 = 10 * d + a, 10 * c + b
    else:
        n1, n2 = 10 * d + a, 10 * c + b
        n3, n4 = 10 * d + b, 10 * c + a
    diff = p1 - p2
    ins = rng.choice([
        f"用{a}、{b}、{c}、{d}四个数字组成两位数乘两位数（每个数字用一次），积最大是多少？",
        f"用数字{a}、{b}、{c}、{d}各一次组成两个两位数相乘，最大的积是多少？",
        f"从{a}、{b}、{c}、{d}四个数字中选，每个用一次，组成两位数乘两位数，积最大是多少？",
        f"用{a}、{b}、{c}、{d}四个数字（每个用一次）组成两个两位数，乘积最大是多少？",
    ])
    lines = [
        f"{n3} × {n4} = {p2}",
        f"{p1} - {p2} = {diff}",
        f"{n1} × {n2} = {p1}",
    ]
    return ins, lines, p1


_reg("max_product_two_digit", max_product_two_digit)


def _closest_factors(P):
    best = (1, P)
    for f in range(2, int(P ** 0.5) + 1):
        if P % f == 0:
            best = (f, P // f)
    return best


def min_sum_given_product(rng):
    P = rng.choice([30, 36, 40, 42, 48, 54, 56, 60, 64, 66, 70, 72, 78,
                    80, 84, 88, 90, 96, 98, 99, 100, 102, 104, 105, 108,
                    110, 112, 120, 126, 132, 140, 144, 150])
    f1, f2 = _closest_factors(P)
    ans = f1 + f2
    g1, g2 = 1, P
    gsum = g1 + g2
    ins = rng.choice([
        f"两个自然数的乘积是{P}，这两个数的和最小是多少？",
        f"两个自然数相乘等于{P}，它们的和最小是多少？",
        f"两个自然数的积是{P}，和最小是多少？",
        f"哪两个自然数的乘积是{P}且和最小？最小的和是多少？",
    ])
    lines = [
        f"{g1} × {g2} = {P}",
        f"{g1} + {g2} = {gsum}",
        f"{f1} × {f2} = {P}",
        f"{f1} + {f2} = {ans}",
    ]
    return ins, lines, ans


_reg("min_sum_given_product", min_sum_given_product)


def split_three_max_product(rng):
    N = rng.choice([15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51,
                    54, 57, 60])
    h = N // 3
    h2 = h * h
    ans = h2 * h
    ins = rng.choice([
        f"把{N}拆成三个自然数的和，再把这三个数相乘，乘积最大是多少？",
        f"三个自然数的和是{N}，这三个数的乘积最大是多少？",
        f"把{N}分成三个自然数相加，怎样分使乘积最大？最大乘积是多少？",
        f"三个数相加等于{N}，这三个数相乘，积最大是多少？",
    ])
    lines = [
        f"{N} ÷ 3 = {h}",
        f"{h} + {h} + {h} = {N}",
        f"{h} × {h} = {h2}",
        f"{h2} × {h} = {ans}",
    ]
    return ins, lines, ans


_reg("split_three_max_product", split_three_max_product)


def min_perimeter_given_area(rng):
    A = rng.choice([30, 36, 40, 42, 48, 54, 56, 60, 64, 70, 72, 80, 84,
                    90, 96, 100, 110, 120, 126, 132, 140, 144, 150, 156,
                    160, 168, 180, 192, 200])
    f1, f2 = _closest_factors(A)
    s = f1 + f2
    ans = 2 * s
    g1, g2 = 1, A
    gsum = g1 + g2
    ins = rng.choice([
        f"一个长方形的面积是{A}平方米，长和宽都是整米数，它的周长最小是多少米？",
        f"长方形面积为{A}平方米，长、宽均为整数米，周长最小是多少米？",
        f"面积是{A}平方米的长方形，长和宽取整米数，最小周长是多少米？",
        f"一块长方形地面积{A}平方米，长和宽都是整米数，怎样围周长最小？最小是多少米？",
    ])
    lines = [
        f"{g1} × {g2} = {A}平方米",
        f"{g1} + {g2} = {gsum}米",
        f"{f1} × {f2} = {A}平方米",
        f"{f1} + {f2} = {s}米",
        f"{s} × 2 = {ans}米",
    ]
    return ins, lines, ans


_reg("min_perimeter_given_area", min_perimeter_given_area)


def _is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def split_primes(rng):
    a = rng.choice([10, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
                    36, 38, 40, 42, 44, 46, 48, 50])
    pairs = [(p, a - p) for p in range(2, a // 2 + 1)
             if _is_prime(p) and _is_prime(a - p)]
    count = len(pairs)
    ins = rng.choice([
        f"把{a}拆成两个质数之和，有几种不同的拆法？",
        f"两个质数相加等于{a}，有多少种不同的拆法？",
        f"把{a}写成两个质数的和，共有几种拆法？",
        f"{a}可以拆成哪两个质数之和？一共有几种拆法？",
    ])
    lines = [f"拆法{i + 1} = {p} + {q} = {a}" for i, (p, q) in enumerate(pairs)]
    lines.append(" + ".join(["1"] * count) + f" = {count}种")
    return ins, lines, count


_reg("split_primes", split_primes)


def double_surplus(rng):
    a, m, b, n = rng.choice([
        (5, 22, 7, 6), (4, 19, 6, 5), (6, 25, 8, 5), (5, 30, 8, 6),
        (4, 25, 7, 4), (6, 30, 9, 6), (7, 33, 10, 6), (5, 27, 9, 3),
        (6, 34, 10, 6), (8, 40, 12, 8), (5, 26, 8, 8), (4, 22, 6, 8),
        (6, 28, 8, 8), (7, 30, 9, 6), (5, 34, 9, 6), (6, 38, 10, 10),
        (8, 34, 10, 8), (5, 24, 7, 10), (4, 28, 8, 4), (6, 26, 9, 8),
    ])
    diff = m - n
    ba = b - a
    assert diff % ba == 0, f"double_surplus params not integral: {(a, m, b, n)}"
    people = diff // ba
    total = a * people + m
    ins = rng.choice([
        f"老师分糖果，每人分{a}个多{m}个，每人分{b}个多{n}个。有多少人？",
        f"把一批糖果分给小朋友，每人{a}个多{m}个，每人{b}个多{n}个，有几个小朋友？",
        f"分糖果时，每人分{a}个剩{m}个，每人分{b}个剩{n}个，共有多少人？",
        f"阿姨分糖果，每人{a}个多{m}个，每人{b}个多{n}个，一共有几人？",
    ])
    lines = [
        f"{m} - {n} = {diff}个",
        f"{b} - {a} = {ba}个",
        f"{a} × {people} + {m} = {total}个",
        f"{diff} ÷ {ba} = {people}人",
    ]
    return ins, lines, people


_reg("double_surplus", double_surplus)


def double_shortage(rng):
    a, m, b, n = rng.choice([
        (5, 8, 7, 2), (4, 10, 6, 4), (6, 12, 8, 4), (5, 15, 8, 3),
        (4, 14, 7, 2), (6, 16, 9, 4), (7, 18, 10, 3), (5, 20, 9, 4),
        (6, 22, 10, 6), (8, 22, 12, 6), (4, 18, 9, 3),
        (6, 14, 8, 6), (5, 17, 8, 5), (7, 20, 10, 5), (8, 26, 12, 6),
        (5, 12, 7, 4), (4, 16, 8, 4), (6, 20, 9, 5), (7, 16, 10, 4),
    ])
    diff = m - n
    ba = b - a
    assert diff % ba == 0, f"double_shortage params not integral: {(a, m, b, n)}"
    people = diff // ba
    total = a * people - m
    ins = rng.choice([
        f"老师分糖果，每人分{a}个还少{m}个，每人分{b}个还少{n}个。有多少人？",
        f"把一批糖果分给小朋友，每人{a}个少{m}个，每人{b}个少{n}个，有几个小朋友？",
        f"分糖果时，每人分{a}个缺{m}个，每人分{b}个缺{n}个，共有多少人？",
        f"阿姨分糖果，每人{a}个还差{m}个，每人{b}个还差{n}个，一共有几人？",
    ])
    lines = [
        f"{m} - {n} = {diff}个",
        f"{b} - {a} = {ba}个",
        f"{a} × {people} - {m} = {total}个",
        f"{diff} ÷ {ba} = {people}人",
    ]
    return ins, lines, people


_reg("double_shortage", double_shortage)


def dorm_rooms(rng):
    a, m, b, n = rng.choice([
        (6, 4, 8, 2), (6, 8, 8, 4), (4, 10, 6, 3), (8, 6, 10, 3),
        (7, 6, 10, 3), (8, 4, 12, 2), (6, 6, 9, 4),
        (5, 8, 7, 4), (6, 16, 8, 5), (4, 14, 6, 3), (8, 10, 10, 5),
        (6, 4, 8, 2), (5, 12, 7, 4), (6, 20, 8, 7),
        (8, 12, 12, 3), (6, 10, 8, 5), (5, 6, 7, 2), (4, 8, 6, 2),
    ])
    bn = b * n
    num = m + bn
    ba = b - a
    rooms = num // ba
    people = a * rooms + m
    ins = rng.choice([
        f"学生宿舍，每间住{a}人则有{m}人没床位，每间住{b}人则空出{n}间。宿舍有多少间？",
        f"安排学生住宿，每间住{a}人，{m}人没床位；每间住{b}人，空出{n}间。共有多少间宿舍？",
        f"学校分宿舍，每间{a}人则多{m}人，每间{b}人则空{n}间，宿舍有多少间？",
        f"新生住宿，每间住{a}人有{m}人没床位，每间住{b}人空余{n}间，一共有多少间宿舍？",
    ])
    lines = [
        f"{b} × {n} = {bn}个",
        f"{m} + {bn} = {num}个",
        f"{b} - {a} = {ba}个",
        f"{a} × {rooms} + {m} = {people}人",
        f"{num} ÷ {ba} = {rooms}间",
    ]
    return ins, lines, rooms


_reg("dorm_rooms", dorm_rooms)


def cut_stock(rng):
    x, y, pipes = rng.choice([
        (4, 3, 3), (6, 3, 3), (4, 6, 4), (8, 3, 4), (7, 2, 3),
        (10, 5, 5), (6, 6, 5), (4, 9, 5), (12, 6, 6), (3, 3, 3),
        (5, 2, 3), (2, 2, 2), (6, 9, 6), (9, 3, 4), (8, 8, 6),
        (5, 5, 4), (10, 10, 7), (3, 6, 3), (12, 3, 5), (9, 6, 5),
    ])
    total = 3 * x + 4 * y
    cap = pipes * 10
    waste = cap - total
    ins = rng.choice([
        f"一根钢管长10米，要截成3米和4米两种短管。需要3米的{x}根、4米的{y}根，至少要用多少根钢管？",
        f"把长10米的钢管截成3米和4米的短管，需要3米管{x}根、4米管{y}根，最少要多少根钢管？",
        f"仓库有10米长的钢管，要截出3米的{x}根和4米的{y}根，至少需要多少根？",
        f"10米长的钢管截成3米和4米两种，需3米的{x}根、4米的{y}根，最少用多少根钢管？",
    ])
    lines = [
        f"3 × {x} + 4 × {y} = {total}米",
        f"{total} + {waste} = {cap}米",
        f"{cap} ÷ 10 = {pipes}根",
    ]
    return ins, lines, pipes


_reg("cut_stock", cut_stock)


def milk_bulk(rng):
    A = rng.choice([45, 48, 50, 52, 55, 58, 60])
    B = rng.choice([3, 4])
    N = rng.choice([25, 26, 27, 28, 29, 30, 32, 35, 36, 40, 42, 44, 45,
                    46, 47, 48])
    retail = N * B
    rem = N - 24
    one_box = A + rem * B
    two_box = 2 * A
    opts = [("零售", retail), ("一箱", one_box), ("两箱", two_box)]
    opts.sort(key=lambda t: t[1])
    ans = opts[0][1]
    ins = rng.choice([
        f"牛奶整箱卖，每箱24瓶{A}元；零售每瓶{B}元。买{N}瓶最少要花多少元？",
        f"商店牛奶每箱24瓶，售价{A}元；单买每瓶{B}元。买{N}瓶最少花多少元？",
        f"牛奶整箱24瓶卖{A}元，零售每瓶{B}元。要买{N}瓶，最少需要多少元？",
        f"一款牛奶整箱24瓶{A}元，零卖每瓶{B}元。买{N}瓶最少要付多少元？",
    ])
    lines = [
        f"{N} × {B} = {retail}元",
        f"{N} - 24 = {rem}瓶",
        f"{A} + {rem} × {B} = {one_box}元",
        f"2 × {A} = {two_box}元",
        f"{opts[1][1]} - {opts[0][1]} = {opts[1][1] - opts[0][1]}元",
        f"{opts[2][1]} - {opts[0][1]} = {opts[2][1] - opts[0][1]}元",
        f"最少 = {ans} = {ans}元",
    ]
    return ins, lines, ans


_reg("milk_bulk", milk_bulk)


def snail_well(rng):
    h, up, down = rng.choice([
        (10, 3, 2), (12, 4, 2), (15, 5, 2), (9, 3, 1), (11, 4, 1),
        (10, 4, 2), (8, 3, 1), (14, 5, 3), (13, 4, 2), (16, 5, 3),
        (20, 6, 4), (18, 5, 2), (12, 5, 3), (10, 5, 3), (9, 4, 2),
        (7, 3, 1), (6, 3, 1), (5, 3, 1), (10, 3, 1), (12, 3, 1),
    ])
    net = up - down
    h_up = h - up
    days_before = -(-h_up // net)
    climbed = days_before * net
    ans = days_before + 1
    ins = rng.choice([
        f"蜗牛爬井，井深{h}米，白天向上爬{up}米，晚上滑下{down}米，几天能爬出井？",
        f"一只蜗牛爬{h}米深的井，白天爬{up}米，夜里滑下{down}米，几天能爬出井口？",
        f"蜗牛在{h}米深的井底，白天向上爬{up}米，晚上滑下{down}米，第几天能爬出井？",
        f"井深{h}米，蜗牛白天爬{up}米、晚上滑{down}米，它几天能爬出井？",
    ])
    lines = [
        f"{up} - {down} = {net}米",
        f"{h} - {up} = {h_up}米",
        f"{days_before} × {net} = {climbed}米",
        f"{climbed} + {up} = {climbed + up}米",
        f"{days_before} + 1 = {ans}天",
    ]
    return ins, lines, ans


_reg("snail_well", snail_well)


def candle_burning(rng):
    a, b, k = rng.choice([
        (4, 6, 2), (3, 6, 2), (6, 12, 2), (8, 12, 2), (5, 10, 3),
        (10, 20, 3), (4, 8, 2), (6, 8, 2), (6, 9, 2), (8, 16, 2),
        (10, 15, 2), (3, 4, 3), (6, 12, 3), (4, 6, 3), (5, 6, 3),
        (4, 5, 3), (8, 10, 3), (8, 12, 3), (9, 12, 2), (12, 18, 2),
    ])
    kb = k * b
    den = kb - a
    ab = a * b
    k1 = k - 1
    abk = ab * k1
    t = Fraction(abk, den)
    ins = rng.choice([
        f"甲蜡烛可燃{a}小时，乙蜡烛可燃{b}小时，同时点燃后，多少小时乙蜡烛剩下的长度是甲蜡烛的{k}倍？",
        f"两根蜡烛，粗的可燃{a}小时，细的可燃{b}小时，同时点燃。几小时后细蜡烛剩下的长度是粗蜡烛的{k}倍？",
        f"甲蜡烛点完要{a}小时，乙蜡烛点完要{b}小时，同时点燃，多少小时后乙剩下的是甲剩下的{k}倍？",
        f"两根同样长的蜡烛，甲可燃{a}小时，乙可燃{b}小时，同时点燃，几小时后乙的剩余长度是甲的{k}倍？",
    ])
    lines = [
        f"{k} × {b} = {kb}",
        f"{kb} - {a} = {den}",
        f"{a} × {b} = {ab}",
        f"{k} - 1 = {k1}",
        f"{ab} × {k1} = {abk}",
        f"{abk} ÷ {den} = {num(t)}小时",
    ]
    return ins, lines, t


_reg("candle_burning", candle_burning)


def avg_three_leg(rng):
    s, v1, v2, v3 = rng.choice([
        (60, 30, 20, 15), (60, 60, 30, 20), (120, 60, 40, 30),
        (30, 30, 15, 10), (60, 20, 15, 12), (90, 45, 30, 18),
        (120, 40, 30, 24), (120, 60, 40, 24), (60, 60, 40, 24),
        (60, 60, 20, 15), (60, 30, 20, 10), (150, 50, 30, 25),
        (90, 30, 18, 15), (60, 20, 12, 10), (120, 60, 30, 24),
        (60, 40, 30, 24), (90, 45, 18, 15), (120, 40, 24, 20),
        (60, 60, 15, 12), (150, 75, 50, 30),
    ])
    t1 = Fraction(s, v1)
    t2 = Fraction(s, v2)
    t3 = Fraction(s, v3)
    tt = t1 + t2 + t3
    total = 3 * s
    avg = Fraction(total, tt)
    ins = rng.choice([
        f"一辆汽车在三段同样长的路上分别以每小时{v1}、{v2}、{v3}千米的速度行驶，每段路长{s}千米。全程的平均速度是多少？",
        f"小明骑车走三段各{s}千米的路，速度分别为每小时{v1}、{v2}、{v3}千米，全程平均速度是多少？",
        f"一段路分三段，每段{s}千米，汽车分别以每小时{v1}、{v2}、{v3}千米行驶，全程平均速度是多少千米/时？",
        f"三段路程都是{s}千米，速度分别是每小时{v1}、{v2}、{v3}千米，求全程的平均速度。",
    ])
    lines = [
        f"{s} ÷ {v1} = {num(t1)}小时",
        f"{s} ÷ {v2} = {num(t2)}小时",
        f"{s} ÷ {v3} = {num(t3)}小时",
        f"{num(t1)} + {num(t2)} + {num(t3)} = {num(tt)}小时",
        f"{s} × 3 = {total}千米",
        f"{total} ÷ ({num(tt)}) = {num(avg)}千米/时",
    ]
    return ins, lines, avg


_reg("avg_three_leg", avg_three_leg)
