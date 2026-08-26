#!/usr/bin/env python3
"""L3 short-solution programs: minimal but complete labeled-equation chains.

Each program solves through a genuine shortcut (grouping assumption, pairing,
rounding, invariant, whole-substitution, telescoping, ...): solved by brute
force the problem needs 4+ lines; the trick collapses it. Every arithmetic
step is still present as a labeled two-operand equation — no bare equations,
no standalone prose, no 3+ operand chains.
"""
import random
from fractions import Fraction
from mathcommon import num

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L3", name, fn))


# 分组假设：1大和尚吃3个、3小和尚吃1个，1大3小为1组吃4个
def monks_buns_group(rng):
    big_eat = rng.choice([3, 4])
    small_group = big_eat
    group_size = 1 + small_group
    group_buns = big_eat + 1
    total = 100
    groups = total // group_buns
    small = groups * small_group
    ins = (f"寺院里{total}个和尚一起吃{total}个馒头，大和尚1人吃{big_eat}个，"
           f"小和尚{small_group}人吃1个，正好分完。小和尚有多少人？")
    lines = [
        f"每组和尚数 = 1 + {small_group} = {group_size}个",
        f"每组馒头数 = {big_eat} + 1 = {group_buns}个",
        f"组数 = {total} ÷ {group_buns} = {groups}组",
        f"小和尚人数 = {groups} × {small_group} = {small}人",
    ]
    return ins, lines, small


_reg("monks_buns_group", monks_buns_group)


# 配对：从1开始的连续奇数和 = 个数的平方
def odd_sum_square(rng):
    n = rng.randint(4, 12)
    last = 2 * n - 1
    total = n * n
    ins = (f"从1开始的连续奇数：1、3、5、……、{last}，一共{n}个。"
           f"把这些奇数全部加起来，和是多少？（提示：首尾两两配对，"
           f"每一对的和都相等）")
    lines = [
        f"1、3、5、……、{last}的个数 = {last} + 1 = {last + 1}",
        f"奇数个数 = {last + 1} ÷ 2 = {n}个",
        f"首末之和 = 1 + {last} = {1 + last}",
        f"首末和乘个数 = {1 + last} × {n} = {(1 + last) * n}",
        f"总和 = {(1 + last) * n} ÷ 2 = {total}",
    ]
    return ins, lines, total


_reg("odd_sum_square", odd_sum_square)


# 凑整：9+99+999+9999 = (10+100+1000+10000) - 4
def round_up_sum(rng):
    d = rng.choice([1, 2, 3])
    terms = [10 ** i - d for i in range(1, 5)]
    total_round = 11110
    extra = 4 * d
    ans = total_round - extra
    terms_str = " + ".join(str(t) for t in terms)
    ins = (f"用凑整的简便方法计算：{terms_str}。（提示：把每个数凑成整十、"
           f"整百、整千、整万，再减去多加的部分）")
    lines = [
        f"{terms_str}凑整 = 10 + 100 = 110",
        f"继续凑整 = 110 + 1000 = 1110",
        f"凑整的总和 = 1110 + 10000 = {total_round}",
        f"多加的部分 = 4 × {d} = {extra}",
        f"原式的和 = {total_round} - {extra} = {ans}",
    ]
    return ins, lines, ans


_reg("round_up_sum", round_up_sum)


# 基准数：102+99+101+98 = 100×4 + (正差-负差)
def base_number_sum(rng):
    base = rng.choice([100, 200, 50])
    pos = sorted(rng.sample([1, 2, 3], 2))
    neg = sorted(rng.sample([1, 2, 3], 2))
    devs = [pos[0], pos[1], -neg[0], -neg[1]]
    rng.shuffle(devs)
    nums = [base + d for d in devs]
    pos_sum = pos[0] + pos[1]
    neg_sum = neg[0] + neg[1]
    ans = base * 4 + pos_sum - neg_sum
    nums_str = " + ".join(str(x) for x in nums)
    ins = (f"用基准数法计算：{nums_str}。（提示：选{base}作基准数，把每个数"
           f"与{base}的差先合起来算）")
    lines = [
        f"{nums_str}的基准数 = {base} × 4 = {base * 4}",
        f"正差之和 = {pos[0]} + {pos[1]} = {pos_sum}",
        f"负差之和 = {neg[0]} + {neg[1]} = {neg_sum}",
        f"差的总和 = {pos_sum} - {neg_sum} = {pos_sum - neg_sum}",
        f"原式的和 = {base * 4} + {pos_sum - neg_sum} = {ans}",
    ]
    return ins, lines, ans


_reg("base_number_sum", base_number_sum)


# 借一还一：1+2+4+...+2^(n-1) = 2^n - 1
def geom_sum_borrow(rng):
    n = rng.randint(5, 9)
    last = 2 ** (n - 1)
    double_last = 2 * last
    ans = double_last - 1
    ins = (f"一个等比数列，首项是1，以后每一项都是前一项的2倍，第{n}项是{last}。"
           f"用把总和扩大2倍再减去原和的方法，求前{n}项的和。")
    lines = [
        f"第{n}项的2倍 = {last} × 2 = {double_last}",
        f"扩大2倍再减首项 = {double_last} - 1 = {ans}",
    ]
    return ins, lines, ans


_reg("geom_sum_borrow", geom_sum_borrow)


# 裂项相消：1/(1×2)+1/(2×3)+...+1/((n-1)×n) = 1 - 1/n
def telescoping_frac(rng):
    n = rng.choice([4, 5, 8, 10])
    ans = Fraction(n - 1, n)
    ins = (f"有一串分数，第一个的分母是1与2的积，以后每个分数的分母是相邻两个"
           f"自然数的积，最后一个的分母是{n - 1}与{n}的积。用裂项相消法求"
           f"这串分数的和。")
    lines = [
        "第一项1/(1×2) = 1 - 1/2 = 1/2",
        "第二项1/(2×3) = 1/2 - 1/3 = 1/6",
        f"末项1/({n - 1}×{n}) = 1/{n - 1} - 1/{n} = 1/{n * (n - 1)}",
        f"抵消后剩余 = 1 - 1/{n} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("telescoping_frac", telescoping_frac)


# 提取公因数：999×778+333×666 = 999×(778+222)
def factor_out_common(rng):
    k = rng.choice([3, 4])
    base = rng.choice([333, 222, 444])
    other = base * k
    a = rng.choice([778, 667, 556])
    b = 1000 - a
    ans = other * 1000
    ins = (f"用提取公因数的简便方法计算：{other} × {a} + {base} × {b * k}。"
           f"（提示：{base} × {k} = {other}，把第二个乘法项也变成含{other}的）")
    lines = [
        f"公因数 = {base} × {k} = {other}",
        f"提取后的另一项 = {b * k} ÷ {k} = {b}",
        f"括号内的和 = {a} + {b} = 1000",
        f"结果 = {other} × 1000 = {ans}",
    ]
    return ins, lines, ans


_reg("factor_out_common", factor_out_common)


# 平方差：a²-b² = (a+b)(a-b)
def diff_squares(rng):
    a = rng.randint(50, 99)
    b = a - rng.choice([1, 2, 3])
    ans = (a + b) * (a - b)
    ins = (f"用平方差公式计算：{a}² - {b}²。（提示：a²-b² = (a+b)×(a-b)，"
           f"这里两数之差很小，可直接口算）")
    lines = [
        f"两数之和 = {a} + {b} = {a + b}",
        f"两数之差 = {a} - {b} = {a - b}",
        f"平方差 = {a + b} × {a - b} = {ans}",
    ]
    return ins, lines, ans


_reg("diff_squares", diff_squares)


# 末位周期：2^n 的个位按 2、4、8、6 循环
def last_digit_cycle(rng):
    base = rng.choice([2, 3, 7, 8])
    cycle = {2: [2, 4, 8, 6], 3: [3, 9, 7, 1], 7: [7, 9, 3, 1], 8: [8, 4, 2, 6]}[base]
    n = rng.randint(30, 2026)
    rem = n % 4
    if rem == 0:
        rem = 4
    ans = cycle[rem - 1]
    rem = n - (n // 4) * 4
    cyc_str = "、".join(str(x) for x in cycle)
    ins = (f"我们知道{base}的n次方，个位数字按{cyc_str}循环（每乘4次就"
           f"重复一遍）。请你算一算，{base}的{n}次方的个位数字是几？")
    lines = [
        f"循环节{cyc_str}的周期 = 4 = 4个",
        f"{n} ÷ 4 = {n / 4:.6g}",
        f"整数部分 = {n // 4} = {n // 4}",
        f"整周期 = {n // 4} × 4 = {(n // 4) * 4}",
        f"余下的个数 = {n} - {(n // 4) * 4} = {rem}个",
        f"第{rem}个末位数字 = {ans} = {ans}",
    ]
    return ins, lines, ans


_reg("last_digit_cycle", last_digit_cycle)


# 公式：多边形内角和 = (边数-2)×180°
def polygon_interior_sum(rng):
    sides = rng.choice([6, 7, 8, 9, 10, 12])
    ans = (sides - 2) * 180
    name = {6: "六边形", 7: "七边形", 8: "八边形", 9: "九边形",
            10: "十边形", 12: "十二边形"}[sides]
    ins = (f"一个{name}（{sides}条边的多边形），它的内角和是多少度？"
           f"（提示：多边形内角和 = (边数-2)×180°）")
    lines = [
        f"边数减2 = {sides} - 2 = {sides - 2}",
        f"内角和 = {sides - 2} × 180 = {ans}°",
    ]
    return ins, lines, ans


_reg("polygon_interior_sum", polygon_interior_sum)


# 公式：正多边形每个外角 = 360° ÷ 边数
def regular_polygon_ext_angle(rng):
    sides = rng.choice([6, 8, 9, 10, 12, 15, 18, 20])
    ans = 360 // sides
    name = {6: "六边形", 8: "八边形", 9: "九边形", 10: "十边形",
            12: "十二边形", 15: "十五边形", 18: "十八边形",
            20: "二十边形"}[sides]
    ins = (f"一个正{name}（每条边都相等、每个内角也都相等的多边形），"
           f"它的每个外角是多少度？（提示：任意多边形的外角和都是360°）")
    lines = [
        f"外角和 = 360 = 360°",
        f"每个外角 = 360 ÷ {sides} = {ans}°",
    ]
    return ins, lines, ans


_reg("regular_polygon_ext_angle", regular_polygon_ext_angle)


# 整体法：捆4个等粗圆柱，绳长 = 4条直径 + 1个圆周长
def bundle_drums_rope(rng):
    r = rng.choice([25, 50, 75, 100])
    d = 2 * r
    straight = 4 * d
    arc = Fraction(314, 100) * 2 * r
    ans = straight + arc
    ins = (f"把4个半径{r}厘米的圆柱形油桶用绳子捆扎一圈（接头处忽略不计），"
           f"至少需要多少厘米的绳子？（提示：绳长 = 4条直径 + 1个圆周长）")
    lines = [
        f"桶的直径 = {r} × 2 = {d}厘米",
        f"4段直绳 = 4 × {d} = {straight}厘米",
        f"圆周长系数 = 2 × 3.14 = 6.28",
        f"4段圆弧 = 6.28 × {r} = {num(arc)}厘米",
        f"绳的总长 = {straight} + {num(arc)} = {num(ans)}厘米",
    ]
    return ins, lines, ans


_reg("bundle_drums_rope", bundle_drums_rope)


# 抵消：跑道起跑线差 = π × 道宽（200米跑只过一个弯道）
def track_lane_gap(rng):
    w = rng.choice([Fraction(6, 5), Fraction(5, 4), Fraction(3, 2), Fraction(5, 2)])
    gap = Fraction(314, 100) * w
    w_disp = f"{float(w):.6g}"
    gap_disp = f"{float(gap):.6g}"
    ins = (f"田径场的跑道宽{w_disp}米，200米赛跑要经过一个半圆形弯道。"
           f"如果起跑线在同一条线上，外道的运动员要多跑多少米？"
           f"（提示：多跑的距离 = π × 道宽）")
    lines = [
        f"200米赛跑起跑线差 = 3.14 × {w_disp} = {gap_disp}米",
    ]
    return ins, lines, gap


_reg("track_lane_gap", track_lane_gap)


# 割补：等腰直角三角形面积 = 斜边² ÷ 4
def iso_right_tri_area(rng):
    h = rng.choice([6, 8, 10, 12, 14, 16, 20])
    h2 = h * h
    ans = h2 // 4
    ins = (f"一个等腰直角三角形的斜边长{h}厘米，它的面积是多少平方厘米？"
           f"（提示：用4个这样的三角形可以拼成一个边长等于斜边的正方形）")
    lines = [
        f"斜边的平方 = {h} × {h} = {h2}",
        f"面积 = {h2} ÷ 4 = {ans}平方厘米",
    ]
    return ins, lines, ans


_reg("iso_right_tri_area", iso_right_tri_area)


# 公式：正方形内以顶点为圆心的最大扇形 = 1/4 个圆
def quarter_circle_sector(rng):
    s = rng.choice([4, 6, 8, 10, 12, 20])
    s2 = s * s
    circle = Fraction(314, 100) * s2
    ans = circle / 4
    c_disp = f"{float(circle):.6g}"
    a_disp = f"{float(ans):.6g}"
    ins = (f"在一个边长{s}厘米的正方形里，以一个顶点为圆心、边长为半径"
           f"画一个最大的扇形，这个扇形的面积是多少平方厘米？"
           f"（提示：这个扇形是四分之一个圆）")
    lines = [
        f"半径的平方 = {s} × {s} = {s2}",
        f"圆的面积 = 3.14 × {s2} = {c_disp}",
        f"扇形面积 = {c_disp} ÷ 4 = {a_disp}",
    ]
    return ins, lines, ans


_reg("quarter_circle_sector", quarter_circle_sector)


# 公式：长方形内最大半圆，半径 = 宽
def max_semicircle_rect(rng):
    w = rng.choice([4, 5, 6, 7, 8])
    L = 2 * w + rng.choice([2, 4, 6])
    w2 = w * w
    circle = Fraction(314, 100) * w2
    ans = circle / 2
    c_disp = f"{float(circle):.6g}"
    a_disp = f"{float(ans):.6g}"
    ins = (f"在一张长{L}厘米、宽{w}厘米的长方形纸上剪一个最大的半圆，"
           f"这个半圆的面积是多少平方厘米？（提示：半圆的半径等于纸的宽）")
    lines = [
        f"长{L}宽{w}的纸，最大半圆半径 = {w} = {w}厘米",
        f"半径的平方 = {w} × {w} = {w2}",
        f"圆的面积 = 3.14 × {w2} = {c_disp}",
        f"半圆的面积 = {c_disp} ÷ 2 = {a_disp}",
    ]
    return ins, lines, ans


_reg("max_semicircle_rect", max_semicircle_rect)


# 对应：网格最短路径 = C(总步数, 向北步数)
def grid_shortest_paths(rng):
    e = rng.choice([3, 4, 5, 6])
    total = e + 2
    p = total * (total - 1)
    ans = p // 2
    ins = (f"从A地到B地，必须向东走{e}个街区、向北走2个街区，途中只能"
           f"向东或向北走，一共有多少条不同的最短路线？"
           f"（提示：在总共{total}步中选2步向北，其余向东）")
    lines = [
        f"总步数 = {e} + 2 = {total}步",
        f"第二步的选法 = {total} - 1 = {total - 1}",
        f"选2步向北 = {total} × {total - 1} = {p}",
        f"路径数 = {p} ÷ 2 = {ans}条",
    ]
    return ins, lines, ans


_reg("grid_shortest_paths", grid_shortest_paths)


# 容斥：1到100k中含数字4的数 = 个位4 + 十位4 - 重复
def count_digit_four(rng):
    k = rng.choice([1, 2, 3, 5])
    bound = 100 * k
    units = 10 * k
    ans = 19 * k
    ins = (f"从1到{bound}的自然数中，含有数字4的数一共有多少个？"
           f"（提示：分别数个位是4和十位是4的数，注意像44这样个位、"
           f"十位都是4的数被算了两次）")
    lines = [
        f"1到{bound}中个位是4的数 = {units} = {units}个",
        f"十位是4的数 = {units} = {units}个",
        f"44等重复的数 = {k} = {k}个",
        f"含4的总数 = {units} + {units} = {2 * k * 10}",
        f"去掉重复 = {2 * k * 10} - {k} = {ans}个",
    ]
    return ins, lines, ans


_reg("count_digit_four", count_digit_four)


# 对应：数字和为s的两位数，十位定了个位就唯一
def count_digit_sum_twodigit(rng):
    s = rng.choice([7, 8, 9, 10, 11, 12])
    lo = max(1, s - 9)
    hi = min(9, s)
    ans = hi - lo + 1
    ins = (f"在所有的两位数中，十位数字与个位数字之和等于{s}的数一共有"
           f"多少个？（提示：先确定十位数字的可选范围，个位数字随之确定）")
    lines = [
        f"十位最小 = {s} - 9 = {lo}" if s > 9 else f"十位最小 = 1 = {lo}",
        f"十位最大 = {hi} = {hi}",
        f"可选范围 = {hi} - {lo} = {hi - lo}",
        f"十位的选法 = {hi - lo} + 1 = {ans}个",
        f"个位随之确定 = 1 = 1个",
        f"两位数个数 = {ans} × 1 = {ans}个",
    ]
    return ins, lines, ans


_reg("count_digit_sum_twodigit", count_digit_sum_twodigit)


# 极端：n把钥匙n把锁，最多试 (n-1)+(n-2)+...+1 次
def keys_locks_max(rng):
    n = rng.choice([4, 5, 6, 7, 8])
    ans = n * (n - 1) // 2
    ins = (f"{n}把钥匙开{n}把锁，但不知道哪把钥匙开哪把锁。最多要试多少次"
           f"才能保证把所有的锁都打开？（提示：第一把锁最多试{n - 1}次，"
           f"最后一把锁只剩一把钥匙，不用试）")
    lines = [f"第1把锁最多试 = {n} - 1 = {n - 1}次"]
    for i in range(2, n):
        lines.append(f"第{i}把锁最多试 = {n - i + 1} - 1 = {n - i}次")
    lines.append(f"第{n}把锁不用试 = 0 = 0次")
    lines.append(f"首项乘项数 = {n - 1} × {n} = {n * (n - 1)}")
    lines.append(f"总次数 = {n * (n - 1)} ÷ 2 = {ans}次")
    return ins, lines, ans


_reg("keys_locks_max", keys_locks_max)


# 整体法：分组循环 + 淘汰赛，总场次 = 小组赛场次 + (出线队数-1)
def tournament_group_knock(rng):
    G = rng.choice([3, 4, 6, 8])
    T = 4 * G
    group_matches = 6 * G
    out = 2 * G
    knock = out - 1
    ans = group_matches + knock
    ins = (f"{T}支球队分成{G}个小组，每组4支球队进行单循环赛，每组前2名"
           f"出线；出线的球队再进行淘汰赛决出冠军。一共要比赛多少场？"
           f"（提示：淘汰赛每淘汰1支球队需要1场比赛）")
    lines = [
        f"每组球队数 = {T} ÷ {G} = 4支",
        f"每组循环赛 = 4 × 3 = 12",
        f"每组场次 = 12 ÷ 2 = 6场",
        f"小组赛场次 = 6 × {G} = {group_matches}场",
        f"出线队伍 = 2 × {G} = {out}支",
        f"淘汰赛场次 = {out} - 1 = {knock}场",
        f"总场次 = {group_matches} + {knock} = {ans}场",
    ]
    return ins, lines, ans


_reg("tournament_group_knock", tournament_group_knock)


# 递推：切饼n刀最多 n(n+1)/2 + 1 块
def plane_max_pieces(rng):
    n = rng.choice([3, 4, 5, 6, 7, 8])
    p = n * (n + 1)
    half = p // 2
    ans = half + 1
    ins = (f"一张圆饼，切{n}刀，每一刀都与前面所有的刀相交，最多能切成"
           f"多少块？（提示：第{n}刀与前{n - 1}刀相交，最多增加{n}块）")
    lines = [
        f"第{n}刀与前{n - 1}刀相交，刀数加1 = {n} + 1 = {n + 1}",
        f"刀数乘加1 = {n} × {n + 1} = {p}",
        f"一半 = {p} ÷ 2 = {half}",
        f"最多块数 = {half} + 1 = {ans}块",
    ]
    return ins, lines, ans


_reg("plane_max_pieces", plane_max_pieces)


# 公式：实心方阵第k层，每边 = 外边 - 2(k-1)，层人数 = 4×(每边-1)
def square_array_layer(rng):
    n = rng.choice([8, 10, 12, 15, 16, 20])
    k = rng.choice([2, 3, 4])
    side = n - 2 * (k - 1)
    ans = 4 * (side - 1)
    ins = (f"运动会上，同学们排成{n}行{n}列的实心方阵，从外往里数第{k}层"
           f"一共有多少人？（提示：每向里一层，每边的人数减少2人）")
    lines = [
        f"减少的层数 = {k} - 1 = {k - 1}",
        f"每边共减少 = {k - 1} × 2 = {2 * (k - 1)}人",
        f"从外数第{k}层每边 = {n} - {2 * (k - 1)} = {side}人",
        f"每边去掉角 = {side} - 1 = {side - 1}人",
        f"第{k}层人数 = {side - 1} × 4 = {ans}人",
    ]
    return ins, lines, ans


_reg("square_array_layer", square_array_layer)


# 公式：空心方阵总数 = (最外层每边 - 层数) × 层数 × 4
def hollow_square_total(rng):
    n = rng.choice([8, 10, 12, 15])
    L = rng.choice([2, 3])
    inner = n - L
    one_layer = inner * L
    ans = one_layer * 4
    ins = (f"用棋子摆一个{L}层的空心方阵，最外层每边有{n}枚棋子，"
           f"一共要用多少枚棋子？（提示：空心方阵总数 = (最外层每边 - 层数)"
           f" × 层数 × 4）")
    lines = [
        f"每边减层数 = {n} - {L} = {inner}",
        f"一层平均人数 = {inner} × {L} = {one_layer}",
        f"总人数 = {one_layer} × 4 = {ans}人",
    ]
    return ins, lines, ans


_reg("hollow_square_total", hollow_square_total)


# 递推：上n级台阶的走法数 = 斐波那契数列
def fib_stairs(rng):
    n = rng.choice([4, 5, 6, 7, 8])
    fib = [1, 2]
    lines = [
        "到第1级 = 1 = 1种",
        "到第2级 = 2 = 2种",
    ]
    for i in range(3, n + 1):
        v = fib[i - 3] + fib[i - 2]
        lines.append(f"到第{i}级 = {fib[i - 3]} + {fib[i - 2]} = {v}种")
        fib.append(v)
    ans = fib[n - 1]
    ins = (f"小明上楼梯，每步可以上1级或2级。他要上{n}级台阶，一共有"
           f"多少种不同的上法？（提示：到第n级的上法数，等于到第{n - 1}级"
           f"与第{n - 2}级的上法数之和）")
    return ins, lines, ans


_reg("fib_stairs", fib_stairs)


# 分类：0到d组成无重复三位数，百位不能为0
def perm_with_zero(rng):
    d = rng.choice([4, 5, 6])
    two = d * d
    ans = two * (d - 1)
    ins = (f"从0到{d}共{d + 1}个数字中，选出三个互不相同的数字组成三位数，"
           f"一共可以组成多少个？（提示：百位上不能为0）")
    lines = [
        f"从0到{d}共{d + 1}个数字，百位可选 = {d + 1} - 1 = {d}个",
        f"十位可选 = {d + 1} - 1 = {d}个",
        f"个位可选 = {d} - 1 = {d - 1}个",
        f"前两位组合 = {d} × {d} = {two}",
        f"三位数个数 = {two} × {d - 1} = {ans}个",
    ]
    return ins, lines, ans


_reg("perm_with_zero", perm_with_zero)


# 捆绑法：n人排队，甲乙必须相邻 = 2 × (n-1)!
def queue_tie_adjacent(rng):
    n = rng.choice([3, 4, 5])
    prods = []
    f = 1
    for i in range(2, n):
        f *= i
        prods.append((i, f))
    fact = f
    ans = 2 * fact
    lines = [
        "甲乙内部排法 = 2 = 2种",
        f"捆绑后整体数 = {n} - 1 = {n - 1}个",
    ]
    if len(prods) == 1:
        lines.append(f"整体全排列 = {prods[0][0]} = {prods[0][1]}种")
    else:
        lines.append(f"整体全排列 = {prods[0][0]} × {prods[1][0]} = {prods[1][1]}")
        for j in range(2, len(prods)):
            lines.append(f"继续乘 = {prods[j - 1][1]} × {prods[j][0]} = {prods[j][1]}")
    lines.append(f"总排法 = 2 × {fact} = {ans}种")
    ins = (f"{n}个小朋友排队，其中甲和乙必须排在一起，一共有多少种不同的"
           f"排法？（提示：把甲、乙捆绑成一个整体，先排整体再排内部）")
    return ins, lines, ans


_reg("queue_tie_adjacent", queue_tie_adjacent)


# 插空法：b男g女排队，女生不相邻 = A(b,b) × C(b+1,g) × A(g,g)
def queue_gap_nonadj(rng):
    b = rng.choice([3, 4])
    g = 2
    b_fact = b * (b - 1) * (b - 2)
    gaps = b + 1
    choose = gaps * (gaps - 1) // 2
    ans = b_fact * choose * 2
    ins = (f"{b}个男生和{g}个女生排队，要求{g}个女生互不相邻，一共有"
           f"多少种不同的排法？（提示：先排男生，再把女生插进男生之间"
           f"的空隙里）")
    lines = [
        f"男生全排列 = {b} × {b - 1} = {b * (b - 1)}",
        f"男生排完 = {b * (b - 1)} × {b - 2} = {b_fact}种",
        f"空隙数 = {b} + 1 = {gaps}个",
        f"选{g}个空隙 = {gaps} × {gaps - 1} = {gaps * (gaps - 1)}",
        f"空隙选法 = {gaps * (gaps - 1)} ÷ 2 = {choose}种",
        f"女生排列 = 2 = 2种",
        f"总排法 = {b_fact} × {choose} = {b_fact * choose}",
        f"总排法 = {b_fact * choose} × 2 = {ans}种",
    ]
    return ins, lines, ans


_reg("queue_gap_nonadj", queue_gap_nonadj)


# 隔板法：数字和为s的三位数 = C(s+1, 2)
def stars_bars_digitsum(rng):
    s = rng.choice([4, 5, 6])
    total = s + 1
    p = total * s
    ans = p // 2
    ins = (f"在所有的三位数中，各个数位上的数字之和等于{s}的三位数一共有"
           f"多少个？（提示：百位至少为1，把百位减1后转化为隔板问题）")
    lines = [
        f"百位减1后总和 = {s} - 1 = {s - 1}",
        f"份数 = 3 = 3个",
        f"隔板数 = 3 - 1 = 2",
        f"加隔板的总数 = {s - 1} + 2 = {total}",
        f"选2个位置 = {total} × {s} = {p}",
        f"三位数个数 = {p} ÷ 2 = {ans}个",
    ]
    return ins, lines, ans


_reg("stars_bars_digitsum", stars_bars_digitsum)


# 公式：封闭图形上植树，棵数 = 周长 ÷ 间距
def closed_loop_planting(rng):
    P = rng.choice([120, 180, 240, 360])
    d = rng.choice([x for x in [3, 4, 5, 6] if P % x == 0])
    ans = P // d
    ins = (f"一个圆形池塘的周长是{P}米，沿池塘边每隔{d}米栽一棵树"
           f"（首尾重合处只栽一棵），一共要栽多少棵树？"
           f"（提示：封闭图形上，棵数 = 段数）")
    lines = [
        f"棵数 = {P} ÷ {d} = {ans}棵",
    ]
    return ins, lines, ans


_reg("closed_loop_planting", closed_loop_planting)


# 配对：1到n的总和减去错算的和，差就是漏加的页码
def page_sum_missing(rng):
    n = rng.choice([20, 30, 40, 50, 60, 63, 64])
    total = n * (n + 1) // 2
    m = rng.randint(10, n - 5)
    wrong = total - m
    ins = (f"一本书的页码从1到{n}，小明把所有页码加起来时，漏加了一页，"
           f"得到的和是{wrong}。他漏加的页码是几？"
           f"（提示：先算出1到{n}的正确总和）")
    lines = [
        f"{n}乘{n + 1} = {n} × {n + 1} = {n * (n + 1)}",
        f"总和 = {n * (n + 1)} ÷ 2 = {total}",
        f"漏加的页码 = {total} - {wrong} = {m}",
    ]
    return ins, lines, m


_reg("page_sum_missing", page_sum_missing)


# 整体法：(甲+乙)+(甲+丙)-(乙+丙) = 2甲
def pairwise_sums_find(rng):
    jia = rng.randint(20, 60)
    yi = rng.randint(20, 60)
    bing = rng.randint(20, 60)
    a = jia + yi
    b = jia + bing
    c = yi + bing
    ins = (f"甲、乙、丙三个数，甲与乙的和是{a}，甲与丙的和是{b}，"
           f"乙与丙的和是{c}。甲数是多少？（提示：把前两个和加起来，"
           f"乙数、丙数各出现一次，甲数出现两次）")
    lines = [
        f"甲乙与甲丙的和 = {a} + {b} = {a + b}",
        f"减去乙丙 = {a + b} - {c} = {a + b - c}",
        f"甲数 = {a + b - c} ÷ 2 = {jia}",
    ]
    return ins, lines, jia


_reg("pairwise_sums_find", pairwise_sums_find)


# 重叠：前m个与后m个的和相加，重叠项被算两次
def overlap_average_mid(rng):
    a, b, mid, c = rng.choice([
        (45, 55, 50, 50), (45, 55, 43, 51), (45, 55, 57, 49),
        (42, 58, 50, 50), (48, 52, 50, 50), (40, 60, 50, 50),
        (45, 55, 36, 52), (45, 55, 64, 48),
    ])
    n, m = 7, 4
    ins = (f"有{n}个数，它们的平均数是{c}；前{m}个数的平均数是{a}，"
           f"后{m}个数的平均数是{b}。第{m}个数是多少？"
           f"（提示：前{m}个与后{m}个的和相加，第{m}个数被算了两次）")
    lines = [
        f"前{m}个的和 = {m} × {a} = {m * a}",
        f"后{m}个的和 = {m} × {b} = {m * b}",
        f"{n}个数的和 = {n} × {c} = {n * c}",
        f"第{m}个数 = {m * a} + {m * b} = {m * a + m * b}",
        f"第{m}个数 = {m * a + m * b} - {n * c} = {mid}",
    ]
    return ins, lines, mid


_reg("overlap_average_mid", overlap_average_mid)


# 中项：奇数个等差数列的平均数 = 中间项，末项 = 2×中项 - 首项
def ap_midterm_average(rng):
    n = rng.choice([5, 7, 9, 11])
    mid = rng.randint(20, 40)
    first = rng.randint(10, mid - 5)
    last = 2 * mid - first
    ins = (f"{n}个数排成一列，相邻两个数的差都相等，它们的平均数是{mid}，"
           f"第一个数是{first}。最后一个数是多少？"
           f"（提示：奇数个等差数列的平均数就是中间项）")
    lines = [
        f"{n}个数的中间项 = {mid} = {mid}",
        f"中间项的2倍 = {mid} × 2 = {2 * mid}",
        f"末项 = {2 * mid} - {first} = {last}",
    ]
    return ins, lines, last


_reg("ap_midterm_average", ap_midterm_average)


# 不变量：分子加x、分母减x，分子分母的和不变
def frac_sum_invariant(rng):
    a, b = rng.choice([
        (2, 3), (2, 5), (2, 7), (2, 9), (3, 4), (3, 5), (3, 7),
        (3, 8), (4, 5), (4, 7), (4, 9), (5, 6), (5, 7), (5, 8), (5, 9),
    ])
    k = rng.randint(2, 5)
    new_num = a * k
    new_den = b * k
    S = new_num + new_den
    x = rng.randint(1, new_num - 1)
    p = new_num - x
    q = new_den + x
    ins = (f"分数{p}/{q}的分子加上一个数、分母减去同一个数后，约分得到"
           f"{a}/{b}。这个数是多少？（提示：分子与分母的和不变）")
    lines = [
        f"原分子分母的和 = {p} + {q} = {S}",
        f"新分数{a}/{b}的份数和 = {a} + {b} = {a + b}",
        f"每份 = {S} ÷ {a + b} = {k}",
        f"新分子 = {a} × {k} = {new_num}",
        f"加上的数 = {new_num} - {p} = {x}",
    ]
    return ins, lines, x


_reg("frac_sum_invariant", frac_sum_invariant)


# 容斥：分母n的最简真分数 = n - n/p - n/q + 1
def euler_phi_count(rng):
    p, q = rng.choice([(3, 5), (3, 7), (5, 7)])
    n = p * q
    ans = n - n // p - n // q + 1
    ins = (f"分母是{n}的最简真分数一共有多少个？"
           f"（提示：分子不能是{n}的质因数{p}或{q}的倍数，注意{p}和{q}"
           f"的公倍数被减了两次）")
    lines = [
        f"{p}的倍数 = {n} ÷ {p} = {n // p}个",
        f"{q}的倍数 = {n} ÷ {q} = {n // q}个",
        f"{n}的倍数 = {n} ÷ {n} = 1个",
        f"最简真分数 = {n} - {n // p} = {n - n // p}",
        f"继续减 = {n - n // p} - {n // q} = {n - n // p - n // q}",
        f"加回重复 = {n - n // p - n // q} + 1 = {ans}个",
    ]
    return ins, lines, ans


_reg("euler_phi_count", euler_phi_count)


# 不变量：年龄差不变，(大年龄-小年龄) = 3个年龄差
def teacher_student_age(rng):
    d = rng.choice([8, 10, 12, 14, 15])
    young = 3
    old = young + 3 * d
    student = young + d
    teacher = young + 2 * d
    ins = (f"老师对学生说：我像你这么大时，你才{young}岁；你像我这么大时，"
           f"我已经{old}岁了。老师现在多少岁？"
           f"（提示：两人的年龄差不变，{old}减{young}正好是3个年龄差）")
    lines = [
        f"年龄差的3倍 = {old} - {young} = {3 * d}",
        f"年龄差 = {3 * d} ÷ 3 = {d}岁",
        f"学生现在 = {young} + {d} = {student}岁",
        f"老师现在 = {student} + {d} = {teacher}岁",
    ]
    return ins, lines, teacher


_reg("teacher_student_age", teacher_student_age)


# 不变量：长增加、宽减少，面积不变
def rect_area_invariant(rng):
    L, W, inc = rng.choice([
        (20, 15, 5), (18, 12, 6), (24, 15, 6), (12, 8, 4),
        (18, 10, 2), (20, 10, 5), (16, 10, 4), (15, 12, 5),
    ])
    area = L * W
    new_L = L + inc
    new_W = area // new_L
    ans = W - new_W
    ins = (f"一个长方形长{L}米、宽{W}米。如果长增加{inc}米，要使面积不变，"
           f"宽应该减少多少米？（提示：面积不变，先求出新的宽）")
    lines = [
        f"原面积 = {L} × {W} = {area}",
        f"新长 = {L} + {inc} = {new_L}",
        f"新宽 = {area} ÷ {new_L} = {new_W}",
        f"宽减少 = {W} - {new_W} = {ans}",
    ]
    return ins, lines, ans


_reg("rect_area_invariant", rect_area_invariant)


# 不变量：除数与余数的差相同，数+差 = 各除数的公倍数
def crt_same_gap(rng):
    ds, gap = rng.choice([
        ([3, 4, 5], 2), ([3, 4, 5], 1), ([5, 6, 7], 1),
        ([5, 6, 7], 2), ([3, 5, 7], 1), ([2, 3, 5], 1),
    ])
    d1, d2, d3 = ds
    r1, r2, r3 = d1 - gap, d2 - gap, d3 - gap
    lcm_pair = d1 * d3
    lcm_all = lcm_pair * d2
    ans = lcm_all - gap
    ins = (f"一个数除以{d1}余{r1}，除以{d2}余{r2}，除以{d3}余{r3}。"
           f"这个数最小是多少？（提示：除数与余数的差都是{gap}，这个数加上"
           f"{gap}后就是{d1}、{d2}、{d3}的公倍数）")
    lines = [
        f"余数{r1}、{r2}、{r3}同差，{d1}和{d3}互质，最小公倍数 = {d1} × {d3} = {lcm_pair}",
        f"三个数的最小公倍数 = {lcm_pair} × {d2} = {lcm_all}",
        f"所求最小数 = {lcm_all} - {gap} = {ans}",
    ]
    return ins, lines, ans


_reg("crt_same_gap", crt_same_gap)


# 平方差：x+A=a²、x+B=b²，(b-a)(b+a) = B-A
def perfect_square_between(rng):
    a, A = rng.choice([
        (16, 100), (12, 100), (15, 100), (20, 200), (10, 50), (18, 150),
    ])
    b = a + 2
    B = A + (b * b - a * a)
    x = a * a - A
    s = a + b
    ins = (f"一个数加上{A}后是一个完全平方数，加上{B}后也是一个完全平方数。"
           f"这个数是多少？（提示：两个平方数的差可以分解成两个因数的积）")
    lines = [
        f"两个平方数的差 = {B} - {A} = {B - A}",
        f"分解成两因数 = 2 × {s} = {B - A}",
        f"较大底数 = {s} + 2 = {s + 2}",
        f"较大底数 = {s + 2} ÷ 2 = {b}",
        f"较小底数 = {s} - 2 = {s - 2}",
        f"较小底数 = {s - 2} ÷ 2 = {a}",
        f"所求数 = {a} × {a} = {a * a}",
        f"所求数 = {a * a} - {A} = {x}",
    ]
    return ins, lines, x


_reg("perfect_square_between", perfect_square_between)


# 配对：(n-1)/n 与 n/(n+1) 的差 = 1/[n(n+1)]
def frac_compare_gap(rng):
    n = rng.choice([100, 500, 1000, 2026])
    den = n * (n + 1)
    ans = Fraction(1, den)
    ins = (f"比较{n - 1}/{n}和{n}/{n + 1}的大小，大数减小数的差是多少？"
           f"（提示：两个分数都接近1，分别写成1减一个分数单位）")
    lines = [
        f"{n - 1}/{n}与{n}/{n + 1}的分母相乘 = {n} × {n + 1} = {den}",
        f"交叉相乘的差 = {n + 1} - {n} = 1",
        f"两分数之差 = 1 ÷ {den} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("frac_compare_gap", frac_compare_gap)


# 整体法：两次购买相加，正好5件上衣+5条裤子
def whole_substitution_clothes(rng):
    a = rng.randint(40, 80)
    b = rng.randint(30, 70)
    while b == a:
        b = rng.randint(30, 70)
    eq1 = 3 * a + 2 * b
    eq2 = 2 * a + 3 * b
    ans = a + b
    ins = (f"买3件上衣和2条裤子共付{eq1}元，买2件上衣和3条裤子共付"
           f"{eq2}元。买1件上衣和1条裤子一共要付多少元？"
           f"（提示：把两次买的合起来看，正好买了5件上衣和5条裤子）")
    lines = [
        f"两次共付钱 = {eq1} + {eq2} = {eq1 + eq2}元",
        f"共买上衣 = 3 + 2 = 5件",
        f"共买裤子 = 2 + 3 = 5条",
        f"一套的价格 = {eq1 + eq2} ÷ 5 = {ans}元",
    ]
    return ins, lines, ans


_reg("whole_substitution_clothes", whole_substitution_clothes)


# 裂项：1×2+2×3+...+n×(n+1) = n(n+1)(n+2)/3
def consec_product_sum(rng):
    n = rng.choice([5, 6, 7, 8, 9])
    p1 = n * (n + 1)
    p2 = p1 * (n + 2)
    ans = p2 // 3
    ins = (f"计算 1×2+2×3+3×4+……+{n}×{n + 1} 的和。"
           f"（提示：用裂项法，每一项拆成两个相邻乘积之差的三分之一）")
    lines = [
        "1 × 2 = 2",
        "2 × 3 = 6",
        "第一项 = 6 ÷ 3 = 2",
        "2 × 3 = 6",
        "6 × 4 = 24",
        "前两项的和 = 24 ÷ 3 = 8",
        f"{n} × {n + 1} = {p1}",
        f"{p1} × {n + 2} = {p2}",
        f"总和 = {p2} ÷ 3 = {ans}",
    ]
    return ins, lines, ans


_reg("consec_product_sum", consec_product_sum)


# 公式：1²+2²+...+n² = n(n+1)(2n+1)/6
def square_sum_formula(rng):
    n = rng.choice([5, 6, 7, 8, 9, 10])
    p1 = n * (n + 1)
    odd = 2 * n + 1
    p2 = p1 * odd
    ans = p2 // 6
    ins = (f"计算 1²+2²+3²+……+{n}² 的和。"
           f"（提示：平方和公式 = n×(n+1)×(2n+1)÷6）")
    lines = [
        f"1²+2²+3²+……+{n}²，n×(n+1) = {n} × {n + 1} = {p1}",
        f"再乘{odd} = {p1} × {odd} = {p2}",
        f"总和 = {p2} ÷ 6 = {ans}",
    ]
    return ins, lines, ans


_reg("square_sum_formula", square_sum_formula)


# 公式：1³+2³+...+n³ = [n(n+1)/2]²
def cube_sum_formula(rng):
    n = rng.choice([4, 5, 6, 7, 8])
    p = n * (n + 1)
    half = p // 2
    ans = half * half
    ins = (f"计算 1³+2³+3³+……+{n}³ 的和。"
           f"（提示：立方和等于和的平方，先算1+2+……+{n}的和，再平方）")
    lines = [
        f"1³+2³+3³+……+{n}³，1+2+……+{n}的和 = {n} × {n + 1} = {p}",
        f"一半 = {p} ÷ 2 = {half}",
        f"总和 = {half} × {half} = {ans}",
    ]
    return ins, lines, ans


_reg("cube_sum_formula", cube_sum_formula)


# 容斥：1到N中既不是3也不是5的倍数 = N - N/3 - N/5 + N/15
def count_not_multiple(rng):
    bound = rng.choice([60, 90, 120, 150])
    m3 = bound // 3
    m5 = bound // 5
    m15 = bound // 15
    sub = bound - m3 - m5
    ans = sub + m15
    ins = (f"从1到{bound}的自然数中，既不是3的倍数也不是5的倍数的数"
           f"一共有多少个？（提示：先减去3的倍数和5的倍数，再把重复"
           f"减去的15的倍数加回来）")
    lines = [
        f"3的倍数 = {bound} ÷ 3 = {m3}个",
        f"5的倍数 = {bound} ÷ 5 = {m5}个",
        f"15的倍数 = {bound} ÷ 15 = {m15}个",
        f"总数 = {bound} - {m3} = {bound - m3}",
        f"继续减 = {bound - m3} - {m5} = {sub}",
        f"加回重复 = {sub} + {m15} = {ans}个",
    ]
    return ins, lines, ans


_reg("count_not_multiple", count_not_multiple)


# 配对：分母n的最简真分数，先数个数再首尾配对，每对和为1
def reduced_frac_sum(rng):
    n, p, q, phi = rng.choice([
        (12, 2, 3, 4), (15, 3, 5, 8), (20, 2, 5, 8), (10, 2, 5, 4),
    ])
    pairs = phi // 2
    ans = pairs
    ins = (f"分母是{n}的所有最简真分数（分子小于分母，且分子与分母互质）"
           f"的和是多少？（提示：最小的和最大的配成一对，每对的和都是1）")
    lines = [
        f"{p}的倍数 = {n} ÷ {p} = {n // p}个",
        f"{q}的倍数 = {n} ÷ {q} = {n // q}个",
        f"{p * q}的倍数 = {n} ÷ {p * q} = {n // (p * q)}个",
        f"最简真分数个数 = {n} - {n // p} = {n - n // p}",
        f"继续减 = {n - n // p} - {n // q} = {n - n // p - n // q}",
        f"加回重复 = {n - n // p - n // q} + {n // (p * q)} = {phi}个",
        f"首尾配对分子 = 1 + {n - 1} = {n}",
        f"每对的和 = {n} ÷ {n} = 1",
        f"对数 = {phi} ÷ 2 = {pairs}对",
        f"总和 = {pairs} × 1 = {ans}",
    ]
    return ins, lines, ans


_reg("reduced_frac_sum", reduced_frac_sum)


if __name__ == "__main__":
    rng = random.Random(3)
    from run_math_short import verify
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines}"
            ok += 1
    print(f"short_l3 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
