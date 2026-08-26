#!/usr/bin/env python3
"""L4 short-solution programs: minimal-but-complete clever solutions.

Every arithmetic step is present as a labeled two-operand equation; what is cut
is restatement and meta-commentary. Techniques: 巧算速算, 数列求和, 计数原理,
几何, 数论, 分数比例, 假设法/盈亏, 概率, 年龄轴.
"""
import random
from fractions import Fraction
from mathcommon import num

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L4", name, fn))


# ---------- 巧算速算 ----------

# 凑整加法: 9999+999+99+9
def round_up_add(rng):
    a, b, c, d = 9999, 999, 99, 9
    ins = (f"用简便方法计算：{a} + {b} + {c} + {d}。"
           f"把每个数先凑成整十、整百、整千或整万，再把多算的零头减去。")
    lines = [
        f"{a}的补数 = 10000 - {a} = 1",
        f"{b}的补数 = 1000 - {b} = 1",
        f"{c}的补数 = 100 - {c} = 1",
        f"{d}的补数 = 10 - {d} = 1",
        f"整万加整千 = 10000 + 1000 = 11000",
        f"再加整百 = 11000 + 100 = 11100",
        f"再加整十 = 11100 + 10 = 11110",
        f"补数和 = 1 + 1 = 2",
        f"补数和 = 2 + 1 = 3",
        f"补数和 = 3 + 1 = 4",
        f"原式 = 11110 - 4 = 11106",
    ]
    return ins, lines, 11106


_reg("round_up_add", round_up_add)


# 凑整减法: 321-98
def round_subtract(rng):
    a = rng.randint(300, 900)
    b = rng.choice([98, 99, 199, 299, 399])
    while b >= a:
        a = rng.randint(300, 900)
    ans = a - b
    ins = (f"用简便方法计算：{a} - {b}。"
           f"把减数看成与它接近的整百数，先减整百，再把多减的部分加回来。")
    lines = [
        f"{b}的补数 = 100 - {b} = {100 - b}" if b < 100
        else f"{b}的补数 = {((b // 100) + 1) * 100} - {b} = {((b // 100) + 1) * 100 - b}",
        f"先减整百 = {a} - {((b // 100) + 1) * 100} = {a - ((b // 100) + 1) * 100}",
        f"多减补回 = {a - ((b // 100) + 1) * 100} + {((b // 100) + 1) * 100 - b} = {ans}",
    ]
    return ins, lines, ans


_reg("round_subtract", round_subtract)


# 基准数求和: 102+103+98+97
def base_number_sum(rng):
    base = rng.choice([100, 200, 500, 1000])
    p = rng.choice([2, 3, 4, 5])
    q = rng.choice([x for x in [2, 3, 4, 5] if x != p])
    n1, n2, n3, n4 = base + p, base + q, base - p, base - q
    ins = (f"用简便方法计算：{n1} + {n2} + {n3} + {n4}。"
           f"选整百数{base}作基准，把每个数与基准的差累计起来再调整。")
    lines = [
        f"{n1}超出基准 = {n1} - {base} = {p}",
        f"{n2}超出基准 = {n2} - {base} = {q}",
        f"{n3}低于基准 = {base} - {n3} = {p}",
        f"{n4}低于基准 = {base} - {n4} = {q}",
        f"超出和 = {p} + {q} = {p + q}",
        f"低于和 = {p} + {q} = {p + q}",
        f"零头和 = {p + q} - {p + q} = 0",
        f"基准总和 = {base} × 4 = {base * 4}",
        f"原式 = {base * 4} + 0 = {base * 4}",
    ]
    return ins, lines, base * 4


_reg("base_number_sum", base_number_sum)


# 分组求和: 100-99+98-97+...+2-1
def group_alternating_sum(rng):
    n = rng.choice([50, 80, 100, 200])
    ins = (f"用简便方法计算：{n} - {n - 1} + {n - 2} - {n - 3} + …… + 2 - 1。"
           f"从{n}到1的连续自然数，加一个减一个交替求总和，把相邻两个数配成一对。")
    lines = [
        f"第一对 = {n} - {n - 1} = 1",
        f"第二对 = {n - 2} - {n - 3} = 1",
        f"对数 = {n} ÷ 2 = {n // 2}",
        f"总和 = {n // 2} × 1 = {n // 2}",
    ]
    return ins, lines, n // 2


_reg("group_alternating_sum", group_alternating_sum)


# 提取公因数: 25×17+25×3
def extract_common_factor(rng):
    k = rng.choice([25, 125])
    if k == 25:
        s = rng.choice([20, 40, 100])
    else:
        s = rng.choice([8, 40, 80])
    a = rng.randint(2, s - 1)
    b = s - a
    ans = k * s
    ins = (f"用简便方法计算：{k} × {a} + {k} × {b}。"
           f"两个乘法项中有相同的因数{k}，先把它提取出来，再算括号里的和。")
    lines = [
        f"括号和 = {a} + {b} = {s}",
        f"原式 = {k} × {s} = {ans}",
    ]
    return ins, lines, ans


_reg("extract_common_factor", extract_common_factor)


# 拆数凑整: 25×32
def split_for_round(rng):
    k = rng.choice([25, 125])
    if k == 25:
        n = rng.choice([32, 36, 44, 48])
        m = 4
    else:
        n = rng.choice([56, 64, 72])
        m = 8
    other = n // m
    ans = k * n
    ins = (f"用简便方法计算：{k} × {n}。"
           f"把{n}拆成{m}乘{other}，让{k}先乘{m}得到整百或整千，再乘{other}。")
    lines = [
        f"拆{n} = {m} × {other} = {n}",
        f"{k}乘{m} = {k} × {m} = {k * m}",
        f"原式 = {k * m} × {other} = {ans}",
    ]
    return ins, lines, ans


_reg("split_for_round", split_for_round)


# 平方差公式: 99²-98²
def diff_squares(rng):
    a = rng.randint(50, 99)
    b = a - 1
    ans = a * a - b * b
    ins = (f"用简便方法计算：{a}² - {b}²。"
           f"利用平方差公式：两个相邻数的平方差等于这两个数的和乘它们的差。")
    lines = [
        f"两数和 = {a} + {b} = {a + b}",
        f"两数差 = {a} - {b} = 1",
        f"平方差 = {a + b} × 1 = {ans}",
    ]
    return ins, lines, ans


_reg("diff_squares", diff_squares)


# 完全平方公式: 99²
def near_hundred_square(rng):
    a = rng.choice([97, 98, 99, 101, 102, 103])
    d = abs(a - 100)
    ans = a * a
    ins = (f"用简便方法计算：{a}²。"
           f"把{a}看成整百数100与一个较小数的{'差' if a < 100 else '和'}，用完全平方公式展开。")
    cross = 200 * d
    if a < 100:
        mid = 10000 - cross
        lines = [
            f"与整百差 = 100 - {a} = {d}",
            f"整百平方 = 100 × 100 = 10000",
            f"两倍整百 = 2 × 100 = 200",
            f"交叉项 = 200 × {d} = {cross}",
            f"差的平方 = {d} × {d} = {d * d}",
            f"先减交叉项 = 10000 - {cross} = {mid}",
            f"补回差平方 = {mid} + {d * d} = {ans}",
        ]
    else:
        mid = 10000 + cross
        lines = [
            f"与整百差 = {a} - 100 = {d}",
            f"整百平方 = 100 × 100 = 10000",
            f"两倍整百 = 2 × 100 = 200",
            f"交叉项 = 200 × {d} = {cross}",
            f"差的平方 = {d} × {d} = {d * d}",
            f"先加交叉项 = 10000 + {cross} = {mid}",
            f"补回差平方 = {mid} + {d * d} = {ans}",
        ]
    return ins, lines, ans


_reg("near_hundred_square", near_hundred_square)


# 末位5的平方: 35²
def square_end_five(rng):
    t = rng.choice([15, 25, 35, 45, 55, 65, 75, 85, 95])
    h = t // 10
    ans = t * t
    ins = (f"用简便方法计算：{t}²。"
           f"个位是5的两位数平方，末两位总是25，前面的数是十位数字乘十位数字加一。")
    lines = [
        f"十位加一 = {h} + 1 = {h + 1}",
        f"头乘头加一 = {h} × {h + 1} = {h * (h + 1)}",
        f"前积移位 = {h * (h + 1)} × 100 = {h * (h + 1) * 100}",
        f"接上25 = {h * (h + 1) * 100} + 25 = {ans}",
        f"原式 = {t} × {t} = {ans}",
    ]
    return ins, lines, ans


_reg("square_end_five", square_end_five)


# 头同尾合十: 23×27
def same_head_ten_tail(rng):
    h = rng.randint(1, 9)
    t = rng.choice([1, 2, 3, 4])
    a = 10 * h + t
    b = 10 * h + (10 - t)
    ans = a * b
    ins = (f"用简便方法计算：{a} × {b}。"
           f"这两个数十位数字相同、个位数字相加等于10，用头同尾合十的方法："
           f"头乘头加一放前面，尾乘尾放后面。")
    lines = [
        f"十位加一 = {h} + 1 = {h + 1}",
        f"头乘头加一 = {h} × {h + 1} = {h * (h + 1)}",
        f"尾乘尾 = {t} × {10 - t} = {t * (10 - t)}",
        f"前积移位 = {h * (h + 1)} × 100 = {h * (h + 1) * 100}",
        f"原式 = {h * (h + 1) * 100} + {t * (10 - t)} = {ans}",
        f"验算 = {a} × {b} = {ans}",
    ]
    return ins, lines, ans


_reg("same_head_ten_tail", same_head_ten_tail)


# 尾同头合十: 34×74
def same_ten_head_tail(rng):
    t = rng.randint(1, 9)
    h = rng.choice([1, 2, 3, 4])
    a = 10 * h + t
    b = 10 * (10 - h) + t
    ans = a * b
    ins = (f"用简便方法计算：{a} × {b}。"
           f"这两个数个位数字相同、十位数字相加等于10，用尾同头合十的方法："
           f"头乘头再加尾放前面，尾乘尾放后面。")
    front = h * (10 - h) + t
    lines = [
        f"头乘头 = {h} × {10 - h} = {h * (10 - h)}",
        f"再加尾 = {h * (10 - h)} + {t} = {front}",
        f"尾乘尾 = {t} × {t} = {t * t}",
        f"前积移位 = {front} × 100 = {front * 100}",
        f"原式 = {front * 100} + {t * t} = {ans}",
        f"验算 = {a} × {b} = {ans}",
    ]
    return ins, lines, ans


_reg("same_ten_head_tail", same_ten_head_tail)


# 几十一乘几十一: 21×31
def teen_one_multiply(rng):
    h = rng.randint(2, 9)
    k = rng.choice([x for x in range(2, 10) if x != h])
    a = 10 * h + 1
    b = 10 * k + 1
    ans = a * b
    ins = (f"用简便方法计算：{a} × {b}。"
           f"这两个数个位都是1，用几十一乘几十一的口诀：头乘头、头相加、末尾添1。")
    lines = [
        f"头乘头 = {h} × {k} = {h * k}",
        f"头相加 = {h} + {k} = {h + k}",
        f"头积移位 = {h * k} × 100 = {h * k * 100}",
        f"头和移位 = {h + k} × 10 = {(h + k) * 10}",
        f"合并 = {h * k * 100} + {(h + k) * 10} = {h * k * 100 + (h + k) * 10}",
        f"末尾添1 = {h * k * 100 + (h + k) * 10} + 1 = {ans}",
        f"原式 = {a} × {b} = {ans}",
    ]
    return ins, lines, ans


_reg("teen_one_multiply", teen_one_multiply)


# 101乘两位数: 101×35
def times_101(rng):
    n = rng.randint(11, 99)
    ans = 101 * n
    ins = (f"用简便方法计算：101 × {n}。"
           f"把101看成100加1，分别与{n}相乘，再把两个积相加。")
    lines = [
        f"拆101 = 100 + 1 = 101",
        f"{n}乘100 = {n} × 100 = {n * 100}",
        f"{n}乘1 = {n} × 1 = {n}",
        f"原式 = {n * 100} + {n} = {ans}",
    ]
    return ins, lines, ans


_reg("times_101", times_101)


# 99乘两位数: 99×35
def times_99(rng):
    n = rng.randint(11, 99)
    ans = 99 * n
    ins = (f"用简便方法计算：99 × {n}。"
           f"把99看成100减1，先算{n}乘100，再减去多算的一份{n}。")
    lines = [
        f"99的补数 = 100 - 99 = 1",
        f"{n}乘100 = {n} × 100 = {n * 100}",
        f"多算一份 = {n} × 1 = {n}",
        f"原式 = {n * 100} - {n} = {ans}",
    ]
    return ins, lines, ans


_reg("times_99", times_99)


# 小数凑整: 0.25×44
def decimal_quarter_round(rng):
    k = rng.choice([Fraction(1, 4), Fraction(1, 8), Fraction(1, 2)])
    if k == Fraction(1, 4):
        n = rng.choice([36, 44, 48])
        m = 4
    elif k == Fraction(1, 8):
        n = rng.choice([56, 64, 72])
        m = 8
    else:
        n = rng.choice([36, 48, 64])
        m = 2
    other = n // m
    ans = k * n
    ins = (f"用简便方法计算：{num(k)} × {n}。"
           f"把{n}拆成{m}乘{other}，让{num(k)}先乘{m}得到整数1，再乘{other}。")
    lines = [
        f"拆{n} = {m} × {other} = {n}",
        f"{num(k)}乘{m} = {num(k)} × {m} = 1",
        f"原式 = 1 × {other} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("decimal_quarter_round", decimal_quarter_round)


# 连续偶数和: 2+4+...+20
def even_sum_formula(rng):
    n = rng.choice([20, 30, 40, 50, 100])
    cnt = n // 2
    ans = cnt * (cnt + 1)
    ins = (f"用简便方法计算从2加到{n}的所有连续偶数之和。"
           f"偶数的个数是{n}除以2，连续偶数的和等于个数乘个数加一。")
    lines = [
        f"偶数个数 = {n} ÷ 2 = {cnt}个",
        f"连续偶数和 = {cnt} × {cnt + 1} = {ans}",
    ]
    return ins, lines, ans


_reg("even_sum_formula", even_sum_formula)


# 补集求和: 51+52+...+100
def sum_range_complement(rng):
    hi = rng.choice([100, 200])
    lo = hi // 2
    total_hi = hi * (hi + 1) // 2
    total_lo = lo * (lo + 1) // 2
    ans = total_hi - total_lo
    ins = (f"用简便方法计算从{lo + 1}加到{hi}的连续自然数之和。"
           f"先算1到{hi}的和，再减去1到{lo}的和。")
    lines = [
        f"1到{hi}和 = {hi} × {hi + 1} = {hi * (hi + 1)}",
        f"1到{hi}和 = {hi * (hi + 1)} ÷ 2 = {total_hi}",
        f"1到{lo}和 = {lo} × {lo + 1} = {lo * (lo + 1)}",
        f"1到{lo}和 = {lo * (lo + 1)} ÷ 2 = {total_lo}",
        f"{lo + 1}到{hi}和 = {total_hi} - {total_lo} = {ans}",
    ]
    return ins, lines, ans


_reg("sum_range_complement", sum_range_complement)


# 相邻数乘积和: 1×2+2×3+...+10×11
def sum_n_n_plus_1(rng):
    n = rng.choice([9, 10, 11, 12])
    prod = n * (n + 1) * (n + 2)
    ans = prod // 3
    ins = (f"用简便方法计算相邻两个自然数乘积的和：1×2 + 2×3 + …… + {n}×{n + 1}。"
           f"公式是{n}乘{n + 1}乘{n + 2}再除以3。")
    lines = [
        f"前两数 = {n} × {n + 1} = {n * (n + 1)}",
        f"再乘{n + 2} = {n * (n + 1)} × {n + 2} = {prod}",
        f"除以3 = {prod} ÷ 3 = {ans}",
    ]
    return ins, lines, ans


_reg("sum_n_n_plus_1", sum_n_n_plus_1)


# 平方交错和: 20²-19²+18²-17²+...+2²-1²
def alt_square_diff_sum(rng):
    n = rng.choice([20, 40, 60, 100])
    m = n // 2
    first = n + (n - 1)
    second = (n - 2) + (n - 3)
    last = 2 + 1
    pair_sum = first + last
    ans = m * (n + 1)
    ins = (f"用简便方法计算从{n}到1的平方交错和：{n}² - {n - 1}² + {n - 2}² - {n - 3}² + …… + 2² - 1²。"
           f"每对相邻数的平方差等于两数之和，这些和组成等差数列。")
    lines = [
        f"第一组平方差 = {n} + {n - 1} = {first}",
        f"第二组平方差 = {n - 2} + {n - 3} = {second}",
        f"末组平方差 = 2 + 1 = {last}",
        f"首末和 = {first} + {last} = {pair_sum}",
        f"组数 = {n} ÷ 2 = {m}",
        f"等差和 = {pair_sum} × {m} = {pair_sum * m}",
        f"除以2 = {pair_sum * m} ÷ 2 = {ans}",
    ]
    return ins, lines, ans


_reg("alt_square_diff_sum", alt_square_diff_sum)


# 加法原理: 先分类后分步
def addition_principle(rng):
    top = rng.randint(2, 5)
    pants = rng.randint(2, 5)
    skirts = rng.randint(2, 5)
    bottoms = pants + skirts
    ans = top * bottoms
    ins = (f"小红有{top}件上衣、{pants}条裤子、{skirts}条裙子。"
           f"她任意选一件上衣和一件下装（裤子或裙子），一共有多少种不同的穿法？")
    lines = [
        f"下装总数 = {pants} + {skirts} = {bottoms}件",
        f"搭配总数 = {top} × {bottoms} = {ans}种",
    ]
    return ins, lines, ans


_reg("addition_principle", addition_principle)


# 乘法原理: 三段路线
def multiplication_principle(rng):
    a = rng.randint(2, 5)
    b = rng.randint(2, 5)
    c = rng.randint(2, 5)
    ab = a * b
    ans = ab * c
    ins = (f"从小明家到学校有{a}条路，从学校到公园有{b}条路，从公园到图书馆有{c}条路。"
           f"小明从家经过学校、公园到图书馆，一共有多少种不同的走法？")
    lines = [
        f"家到公园走法 = {a} × {b} = {ab}种",
        f"家到图书馆走法 = {ab} × {c} = {ans}种",
    ]
    return ins, lines, ans


_reg("multiplication_principle", multiplication_principle)


# 捆绑法: 相邻排列
def bundle_adjacent_perm(rng):
    n = rng.choice([4, 5])
    ins = (f"{n}个小朋友排成一排照相，其中小明和小红必须相邻。"
           f"一共有多少种不同的排法？（用捆绑法：把相邻两人看成一个整体）")
    if n == 4:
        lines = [
            f"捆绑后元素 = 4 - 1 = 3个",
            f"整体排法 = 3 × 2 = 6种",
            f"两人内部排法 = 2 × 1 = 2种",
            f"总排法 = 6 × 2 = 12种",
        ]
        ans = 12
    else:
        lines = [
            f"捆绑后元素 = 5 - 1 = 4个",
            f"整体排法 = 4 × 3 = 12种",
            f"整体排法 = 12 × 2 = 24种",
            f"两人内部排法 = 2 × 1 = 2种",
            f"总排法 = 24 × 2 = 48种",
        ]
        ans = 48
    return ins, lines, ans


_reg("bundle_adjacent_perm", bundle_adjacent_perm)


# 插空法: 不相邻排列
def gap_nonadjacent_perm(rng):
    n = rng.choice([4, 5])
    ins = (f"{n}个小朋友排成一排照相，其中小明和小红不能相邻。"
           f"一共有多少种不同的排法？（用插空法：先排其余人，再把两人插进空隙）")
    if n == 4:
        lines = [
            f"先排其余两人 = 4 - 2 = 2种",
            f"空隙数 = 2 + 1 = 3个",
            f"两人插入 = 3 × 2 = 6种",
            f"总排法 = 2 × 6 = 12种",
        ]
        ans = 12
    else:
        lines = [
            f"先排其余三人 = 5 - 2 = 3种",
            f"三人排法 = 3 × 2 = 6种",
            f"三人排法 = 6 × 1 = 6种",
            f"空隙数 = 3 + 1 = 4个",
            f"两人插入 = 4 × 3 = 12种",
            f"总排法 = 6 × 12 = 72种",
        ]
        ans = 72
    return ins, lines, ans


_reg("gap_nonadjacent_perm", gap_nonadjacent_perm)


# 0不在首位: 组成三位数
def zero_first_digit_count(rng):
    m = rng.choice([4, 5])
    ins = (f"用0到{m - 1}这{m}个数字可以组成多少个没有重复数字的三位数？"
           f"注意百位不能是0。")
    bai = m - 1
    shi = m - 1
    ge = m - 2
    hou = shi * ge
    ans = bai * hou
    lines = [
        f"百位选择 = {m} - 1 = {bai}种",
        f"十位选择 = {m} - 1 = {shi}种",
        f"个位选择 = {shi} - 1 = {ge}种",
        f"后两位排法 = {shi} × {ge} = {hou}种",
        f"三位数总数 = {bai} × {hou} = {ans}个",
    ]
    return ins, lines, ans


_reg("zero_first_digit_count", zero_first_digit_count)


# 折叠求角
def fold_angle(rng):
    a = rng.choice([20, 25, 30, 35, 40])
    ans = 180 - 2 * a
    ins = (f"把一张长方形纸的一角折起来，已知∠1 = {a}°，求∠2的度数。"
           f"折叠后重合的两个角相等，∠1、与它重合的角和∠2共同组成一个平角。")
    lines = [
        f"两个折叠角 = {a} × 2 = {2 * a}度",
        f"∠2 = 180 - {2 * a} = {ans}度",
    ]
    return ins, lines, ans


_reg("fold_angle", fold_angle)


# 切饼最多块数
def max_pieces_cuts(rng):
    n = rng.choice([4, 5, 6, 8])
    prod = n * (n + 1)
    half = prod // 2
    ans = half + 1
    ins = (f"一张圆饼切{n}刀，每刀都沿直线穿过整个圆饼，最多能切成多少块？"
           f"切n刀最多的块数等于1加n乘n加一除以2。")
    lines = [
        f"刀数加一 = {n} + 1 = {n + 1}",
        f"刀数乘加一 = {n} × {n + 1} = {prod}",
        f"除以2 = {prod} ÷ 2 = {half}",
        f"加1 = {half} + 1 = {ans}块",
    ]
    return ins, lines, ans


_reg("max_pieces_cuts", max_pieces_cuts)


# 圆柱侧面展开
def cylinder_lateral_unfold(rng):
    r = rng.choice([5, 10, 15, 20])
    h = rng.choice([10, 20, 30])
    pi = Fraction(314, 100)
    circ = pi * 2 * r
    ans = circ * h
    ins = (f"一个圆柱的底面半径是{r}厘米，高是{h}厘米。"
           f"把它的侧面沿高剪开，展开成一个长方形，这个长方形的面积是多少平方厘米？")
    lines = [
        f"底面直径 = {r} × 2 = {2 * r}厘米",
        f"底面周长 = 3.14 × {2 * r} = {num(circ)}厘米",
        f"侧面积 = {num(circ)} × {h} = {num(ans)}平方厘米",
    ]
    return ins, lines, ans


_reg("cylinder_lateral_unfold", cylinder_lateral_unfold)


# 正方形对角线求面积
def square_diagonal_area(rng):
    d = rng.choice([6, 8, 10, 12, 20])
    ans = d * d // 2
    ins = (f"正方形的对角线长{d}厘米，它的面积是多少平方厘米？"
           f"正方形沿对角线分成两个三角形，面积等于对角线乘对角线再除以2。")
    lines = [
        f"对角线乘积 = {d} × {d} = {d * d}",
        f"面积 = {d * d} ÷ 2 = {ans}平方厘米",
    ]
    return ins, lines, ans


_reg("square_diagonal_area", square_diagonal_area)


# 组合面积: 长方形+三角形
def composite_area(rng):
    a = rng.choice([8, 10, 12])
    b = rng.choice([6, 8, 10])
    h = rng.choice([4, 5, 6])
    while (b * h) % 2 != 0:
        h = rng.choice([4, 5, 6])
    rect = a * b
    tri = b * h // 2
    ans = rect + tri
    ins = (f"一块菜地由一个长方形和一个三角形组成：长方形长{a}米、宽{b}米，"
           f"三角形的底是{b}米、高是{h}米。这块菜地的总面积是多少平方米？")
    lines = [
        f"长方形面积 = {a} × {b} = {rect}平方米",
        f"三角形底乘高 = {b} × {h} = {b * h}",
        f"三角形面积 = {b * h} ÷ 2 = {tri}平方米",
        f"总面积 = {rect} + {tri} = {ans}平方米",
    ]
    return ins, lines, ans


_reg("composite_area", composite_area)


# 阶梯形周长平移
def perimeter_translation(rng):
    a = rng.choice([12, 15, 20])
    b = rng.choice([8, 10, 12])
    s = a + b
    ans = 2 * s
    ins = (f"一块阶梯形的地，横向总长度是{a}米，纵向总宽度是{b}米。"
           f"把所有横边上下平移、竖边左右平移后，周长与一个长{a}米宽{b}米的长方形相等。这块地的周长是多少米？")
    lines = [
        f"长加宽 = {a} + {b} = {s}米",
        f"周长 = {s} × 2 = {ans}米",
    ]
    return ins, lines, ans


_reg("perimeter_translation", perimeter_translation)


# 11的余数特征
def divisibility_11(rng):
    even = rng.choice([3, 4, 5])
    rem = rng.choice([1, 2, 3, 4])
    while even + rem > 7:
        rem = rng.choice([1, 2, 3, 4])
    diff = 11 + rem
    odd = even + diff
    a = rng.choice([1, 2])
    c = even - a
    b = rng.choice([8, 9])
    d = odd - b
    while d > 9 or d < 0:
        b = rng.choice([8, 9])
        d = odd - b
    number = a * 1000 + b * 100 + c * 10 + d
    ins = (f"一个四位数，千位是{a}、百位是{b}、十位是{c}、个位是{d}，它除以11的余数是多少？"
           f"用11的余数特征：奇数位数字和与偶数位数字和的差，减去11的倍数，剩下的就是余数。")
    lines = [
        f"奇数位数字和 = {b} + {d} = {odd}",
        f"偶数位数字和 = {a} + {c} = {even}",
        f"两位和之差 = {odd} - {even} = {diff}",
        f"余数 = {diff} - 11 = {rem}",
    ]
    return ins, lines, rem


_reg("divisibility_11", divisibility_11)


# 奇偶配对差
def pair_even_odd_diff(rng):
    n = rng.choice([10, 20, 50, 100])
    ins = (f"用简便方法计算：从2到{2 * n}的连续偶数之和，减去从1开始的同样多个连续奇数之和。"
           f"偶数和奇数可以依次配对，每对的差都是1。")
    lines = [
        f"每对差 = 2 - 1 = 1",
        f"对数 = {2 * n} ÷ 2 = {n}",
        f"总和差 = {n} × 1 = {n}",
    ]
    return ins, lines, n


_reg("pair_even_odd_diff", pair_even_odd_diff)


# 冰水体积变化
def ice_water(rng):
    ins = ("水结成冰后体积增加，冰化成水后体积减少1/11。"
           "那么水结成冰后，体积比原来增加几分之几？")
    lines = [
        f"水的份数 = 11 × 10/11 = 10份",
        f"增加份数 = 11 - 10 = 1份",
        f"增加分率 = 1 ÷ 10 = 1/10",
    ]
    return ins, lines, Fraction(1, 10)


_reg("ice_water", ice_water)


# 等时间平均速度
def avg_speed_equal_time(rng):
    v1 = rng.choice([40, 50, 60])
    v2 = rng.choice([70, 80, 90])
    t = rng.choice([2, 3])
    s = v1 + v2
    ans = Fraction(s, 2)
    ins = (f"一辆汽车从甲地到乙地，前{t}小时每小时行{v1}千米，后{t}小时每小时行{v2}千米。"
           f"这辆汽车全程的平均速度是多少千米/时？")
    lines = [
        f"前{t}小时路程 = {v1} × {t} = {v1 * t}千米",
        f"后{t}小时路程 = {v2} × {t} = {v2 * t}千米",
        f"总路程 = {v1 * t} + {v2 * t} = {v1 * t + v2 * t}千米",
        f"总时间 = {t} × 2 = {t * 2}小时",
        f"平均速度 = {v1 * t + v2 * t} ÷ {t * 2} = {num(ans)}千米/时",
    ]
    return ins, lines, ans


_reg("avg_speed_equal_time", avg_speed_equal_time)


# 连比统一
def unify_ratio(rng):
    from math import gcd
    sets = [((2, 3), (4, 5)), ((1, 2), (3, 4)), ((2, 5), (3, 4))]
    (p, q), (r, s) = rng.choice(sets)
    L = q * r // gcd(q, r)
    jia = p * (L // q)
    bing = s * (L // r)
    g = gcd(jia, bing)
    ans = Fraction(jia, bing)
    ins = (f"甲数与乙数的比是{p}:{q}，乙数与丙数的比是{r}:{s}。"
           f"甲数是丙数的几分之几？（先把两个比中乙数的份数统一）")
    lines = [
        f"乙统一为 = {q} × {L // q} = {L}份",
        f"甲扩大 = {p} × {L // q} = {jia}份",
        f"丙扩大 = {s} × {L // r} = {bing}份",
        f"甲是丙的 = {jia} ÷ {bing} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("unify_ratio", unify_ratio)


# 给后成倍数
def transfer_becomes_ratio(rng):
    total = rng.choice([72, 84, 96])
    b = rng.choice([18, 22])
    a = total - b
    later_b = total // 3
    give = later_b - b
    ins = (f"甲有{a}元，乙有{b}元。甲给乙多少元后，甲的钱正好是乙的2倍？"
           f"两人总钱数不变，把后来乙的钱看成1份、甲的钱看成2份。")
    lines = [
        f"总钱数 = {a} + {b} = {total}元",
        f"总份数 = 2 + 1 = 3份",
        f"后来乙的钱 = {total} ÷ 3 = {later_b}元",
        f"甲给乙 = {later_b} - {b} = {give}元",
    ]
    return ins, lines, give


_reg("transfer_becomes_ratio", transfer_becomes_ratio)


# 假设法: 采松子
def assume_sun_rain(rng):
    sets = [(6, 2, 12, 20), (5, 3, 12, 20), (4, 4, 12, 20), (4, 4, 15, 25)]
    rain, sun, rainy, sunny = rng.choice(sets)
    days = rain + sun
    total = sun * sunny + rain * rainy
    avg = total // days
    diff = sunny - rainy
    over = days * sunny - total
    ins = (f"松鼠妈妈采松子，晴天每天采{sunny}个，雨天每天采{rainy}个。"
           f"它一连几天共采了{total}个，平均每天采{avg}个。这几天中有几天是雨天？")
    lines = [
        f"总天数 = {total} ÷ {avg} = {days}天",
        f"假设全是晴天 = {days} × {sunny} = {days * sunny}个",
        f"多算的松子 = {days * sunny} - {total} = {over}个",
        f"每天差 = {sunny} - {rainy} = {diff}个",
        f"雨天天数 = {over} ÷ {diff} = {rain}天",
    ]
    return ins, lines, rain


_reg("assume_sun_rain", assume_sun_rain)


# 两盈问题
def two_surplus(rng):
    people = rng.choice([3, 4, 5])
    p1 = rng.choice([6, 7, 8])
    p2 = p1 + 2
    s2 = rng.choice([2, 4])
    s1 = s2 + people * 2
    apples = p1 * people + s1
    ins = (f"老师把一些苹果分给小朋友，每人分{p1}个还多{s1}个，每人分{p2}个还多{s2}个。"
           f"有多少个小朋友？多少个苹果？")
    lines = [
        f"每人多分 = {p2} - {p1} = {p2 - p1}个",
        f"共多分 = {s1} - {s2} = {s1 - s2}个",
        f"小朋友人数 = {s1 - s2} ÷ {p2 - p1} = {people}人",
        f"苹果总数 = {p1} × {people} = {p1 * people}个",
        f"苹果总数 = {p1 * people} + {s1} = {apples}个",
    ]
    return ins, lines, apples


_reg("two_surplus", two_surplus)


# 一盈一尽
def surplus_exact(rng):
    people = rng.choice([3, 4, 5, 6])
    p1 = rng.choice([6, 7, 8])
    p2 = p1 + rng.choice([2, 3])
    s1 = people * (p2 - p1)
    apples = p2 * people
    ins = (f"老师把一些苹果分给小朋友，每人分{p1}个还多{s1}个，每人分{p2}个正好分完。"
           f"有多少个小朋友？多少个苹果？")
    lines = [
        f"每人多分 = {p2} - {p1} = {p2 - p1}个",
        f"小朋友人数 = {s1} ÷ {p2 - p1} = {people}人",
        f"苹果总数 = {p2} × {people} = {apples}个",
    ]
    return ins, lines, apples


_reg("surplus_exact", surplus_exact)


# 一亏一尽
def shortage_exact(rng):
    people = rng.choice([3, 4, 5, 6])
    p2 = rng.choice([6, 7, 8])
    p1 = p2 + rng.choice([2, 3])
    s1 = people * (p1 - p2)
    apples = p2 * people
    ins = (f"老师把一些苹果分给小朋友，每人分{p1}个还少{s1}个，每人分{p2}个正好分完。"
           f"有多少个小朋友？多少个苹果？")
    lines = [
        f"每人少分 = {p1} - {p2} = {p1 - p2}个",
        f"小朋友人数 = {s1} ÷ {p1 - p2} = {people}人",
        f"苹果总数 = {p2} × {people} = {apples}个",
    ]
    return ins, lines, apples


_reg("shortage_exact", shortage_exact)


# 扑克牌概率
def card_probability(rng):
    pair = rng.choice([("红桃", "方块"), ("红桃", "黑桃"), ("方块", "梅花")])
    s1, s2 = pair
    ins = (f"一副扑克牌去掉大小王后共52张，从中任意抽出一张。"
           f"抽到{s1}或{s2}的概率是多少？")
    lines = [
        f"{s1}加{s2} = 13 + 13 = 26张",
        f"概率 = 26 ÷ 52 = 1/2",
    ]
    return ins, lines, Fraction(1, 2)


_reg("card_probability", card_probability)


# 年龄轴: 师徒
def age_axis_puzzle(rng):
    past = rng.choice([4, 5, 6])
    future = rng.choice([f for f in range(58, 70) if (f - past) % 3 == 0])
    d = (future - past) // 3
    tu = past + d
    shi = tu + d
    ins = (f"师傅对徒弟说：我像你现在这么大时，你才{past}岁。"
           f"徒弟说：我像你现在这么大时，你就{future}岁了。师傅和徒弟现在各多少岁？")
    lines = [
        f"年龄差 = {future} - {past} = {future - past}岁",
        f"三个年龄差 = {future - past} ÷ 3 = {d}岁",
        f"徒弟现在 = {past} + {d} = {tu}岁",
        f"师傅现在 = {tu} + {d} = {shi}岁",
    ]
    return ins, lines, shi


_reg("age_axis_puzzle", age_axis_puzzle)


if __name__ == "__main__":
    rng = random.Random(1234)
    from run_math_short import verify
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines}"
            ok += 1
    print(f"short_l4 OK: {len(PROGRAMS)} programs, {ok} instances verified")
