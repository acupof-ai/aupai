#!/usr/bin/env python3
"""L4 ext3: reverse reasoning, number theory, multi-constraint, rates — 65 families.

Every program: fn(rng) -> (instruction, lines, ans). Lines solve FORWARD from
givens to the asked value (no x variable). All exact arithmetic via Fraction.
Every equation line is chained: 中文标签 = 表达式 = 值[单位].
"""
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


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# 1. 某数的k1倍加上c等于它的k2倍 → 某数
def rev_k1x_plus_c_eq_k2x(rng):
    k1 = rng.randint(2, 6)
    diff = rng.randint(1, 4)
    x = rng.randint(3, 30)
    k2 = k1 + diff
    c = diff * x
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数的{k1}倍加上{c}，正好等于这个数的{k2}倍。这个数是多少？",
        f"{name}在练习册上看到一道题：某数的{k1}倍与{c}的和，等于这个数的{k2}倍。请你帮他算出这个数。",
    ])
    lines = [
        f"倍数差 = {k2} - {k1} = {diff}",
        f"这个数 = {c} ÷ {diff} = {x}",
    ]
    return ins, lines, x


_reg("rev_k1x_plus_c_eq_k2x", rev_k1x_plus_c_eq_k2x)


# 2. 某数的k倍加上它的1/n等于s → 某数
def rev_coeff_plus_frac(rng):
    k = rng.randint(2, 5)
    n = rng.randint(2, 4)
    t = rng.randint(3, 20)
    x = n * t
    s = t * (k * n + 1)
    coef = Fraction(k * n + 1, n)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数的{k}倍加上这个数的1/{n}，和是{s}。这个数是多少？",
        f"{name}遇到一道思考题：某数的{k}倍与它的1/{n}相加得{s}。请你帮他算出这个数。",
    ])
    lines = [
        f"一共的倍数 = {k} + 1 ÷ {n} = {num(coef)}",
        f"这个数 = {s} ÷ ({num(coef)}) = {x}",
    ]
    return ins, lines, x


_reg("rev_coeff_plus_frac", rev_coeff_plus_frac)


# 3. 某数加上它的1/n等于r → 某数
def rev_x_plus_own_fraction(rng):
    n = rng.randint(2, 5)
    k = rng.randint(2, 15)
    x = n * k
    r = k * (n + 1)
    coef = Fraction(n + 1, n)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数加上它的1/{n}，结果是{r}。这个数是多少？",
        f"{name}在作业本上看到一道题：某数与它的1/{n}的和是{r}。请你列式算一算这个数。",
    ])
    lines = [
        f"一共的倍数 = 1 + 1 ÷ {n} = {num(coef)}",
        f"这个数 = {r} ÷ ({num(coef)}) = {x}",
    ]
    return ins, lines, x


_reg("rev_x_plus_own_fraction", rev_x_plus_own_fraction)


# 4. 某数的1/n1比它的1/n2多d → 某数
def rev_frac_diff(rng):
    n1, n2 = rng.choice([(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)])
    k = rng.randint(3, 20)
    x = n1 * n2 * k
    d = k * (n2 - n1)
    diff = Fraction(n2 - n1, n1 * n2)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数的1/{n1}比它的1/{n2}多{d}。这个数是多少？",
        f"{name}遇到一道思考题：某数的1/{n1}减去它的1/{n2}，差是{d}。请你帮他算出这个数。",
    ])
    lines = [
        f"分率差 = 1 ÷ {n1} - 1 ÷ {n2} = {num(diff)}",
        f"这个数 = {d} ÷ ({num(diff)}) = {x}",
    ]
    return ins, lines, x


_reg("rev_frac_diff", rev_frac_diff)


# 5. 末尾添一个0后比原数多d → 原数
def append_zero_diff(rng):
    x = rng.randint(3, 60)
    d = 9 * x
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数的末尾添上一个0后，比原数多{d}。这个数是多少？",
        f"{name}在练习册上看到一道题：把一个数的末尾添上一个0，所得的数比原数大{d}。请你帮他算出原数。",
    ])
    lines = [
        f"倍数差 = 10 - 1 = 9",
        f"原数 = {d} ÷ 9 = {x}",
    ]
    return ins, lines, x


_reg("append_zero_diff", append_zero_diff)


# 6. 分子分母和s，分子加a后分数等于1 → 原分数
def fraction_num_den_sum(rng):
    n = rng.randint(3, 20)
    a = rng.randint(2, n - 1)
    d = n + a
    s = 2 * n + a
    ans = Fraction(n, d)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个分数的分子与分母的和是{s}，如果分子加上{a}，这个分数就等于1。这个分数是多少？",
        f"{name}遇到一道思考题：某分数的分子和分母相加得{s}，分子加上{a}后分数等于1。请你帮他算出这个分数。",
    ])
    lines = [
        f"分子的2倍 = {s} - {a} = {2 * n}",
        f"分子 = {2 * n} ÷ 2 = {n}",
        f"分母的2倍 = {s} + {a} = {2 * d}",
        f"分母 = {2 * d} ÷ 2 = {d}",
        f"这个分数 = {n} ÷ {d} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("fraction_num_den_sum", fraction_num_den_sum)


# 7. 约分后是p/q，分子分母和s → 原分数
def fraction_simplify_sum(rng):
    p, q = rng.choice([(2, 3), (3, 4), (2, 5), (3, 5), (4, 5)])
    t = rng.randint(2, 12)
    n = p * t
    d = q * t
    s = (p + q) * t
    ans = Fraction(n, d)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个分数约分后是{p}/{q}，原分数的分子与分母的和是{s}。这个分数是多少？",
        f"{name}在练习册上看到一道题：某分数约分后等于{p}/{q}，且分子和分母的和是{s}。请你帮他算出这个分数。",
    ])
    lines = [
        f"份数 = {p} + {q} = {p + q}",
        f"每份 = {s} ÷ {p + q} = {t}",
        f"分子 = {t} × {p} = {n}",
        f"分母 = {t} × {q} = {d}",
        f"这个分数 = {n} ÷ {d} = {num(ans)}",
    ]
    return ins, lines, ans


_reg("fraction_simplify_sum", fraction_simplify_sum)


# 8. 两位数，数字差d，交换后与原数和s → 原数
def digit_swap_sum(rng):
    name = rng.choice(NAMES)
    if rng.random() < 0.5:
        a = rng.randint(2, 9)
        b = rng.randint(1, a - 1)
        d = a - b
        s = 11 * (a + b)
        num_ = 10 * a + b
        ins = rng.choice([
            f"数学课上老师出了一道题：一个两位数，十位数字比个位数字大{d}，把十位和个位交换位置后，新数与原数的和是{s}。这个两位数是多少？",
            f"{name}遇到一道思考题：一个两位数的十位数字比个位数字大{d}，交换两个数字后所得的数与原数相加得{s}。请你帮他算出这个两位数。",
        ])
        lines = [
            f"数字和 = {s} ÷ 11 = {a + b}",
            f"十位数字的2倍 = {a + b} + {d} = {2 * a}",
            f"十位数字 = {2 * a} ÷ 2 = {a}",
            f"个位数字 = {a} - {d} = {b}",
            f"这个两位数 = {a} × 10 + {b} = {num_}",
        ]
        return ins, lines, num_
    a = rng.randint(1, 8)
    b = rng.randint(a + 1, 9)
    d = b - a
    s = 11 * (a + b)
    num_ = 10 * a + b
    ins = rng.choice([
        f"数学课上老师出了一道题：一个两位数，个位数字比十位数字大{d}，把十位和个位交换位置后，新数与原数的和是{s}。这个两位数是多少？",
        f"{name}遇到一道思考题：一个两位数的个位数字比十位数字大{d}，交换两个数字后所得的数与原数相加得{s}。请你帮他算出这个两位数。",
    ])
    lines = [
        f"数字和 = {s} ÷ 11 = {a + b}",
        f"个位数字的2倍 = {a + b} + {d} = {2 * b}",
        f"个位数字 = {2 * b} ÷ 2 = {b}",
        f"十位数字 = {b} - {d} = {a}",
        f"这个两位数 = {a} × 10 + {b} = {num_}",
    ]
    return ins, lines, num_


_reg("digit_swap_sum", digit_swap_sum)


# 9. 百位是十位的2倍，个位比十位多d，数字和s → 数/百位/个位
def digit_relations(rng):
    t = rng.randint(1, 4)
    d = rng.randint(1, 9 - t)
    h = 2 * t
    u = t + d
    s = h + t + u
    ask = rng.choice(["数", "百", "个"])
    ans = {"数": 100 * h + 10 * t + u, "百": h, "个": u}[ask]
    name = rng.choice(NAMES)
    target = {"数": "这个数", "百": "百位数字", "个": "个位数字"}[ask]
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数的百位数字是十位数字的2倍，个位数字比十位数字多{d}，各位数字之和是{s}。{target}是多少？",
        f"{name}遇到一道思考题：某数的百位数字是十位数字的2倍，个位数字比十位数字多{d}，各位数字相加得{s}。请你算出{target}。",
    ])
    base = [
        f"十位数字的4倍 = {s} - {d} = {4 * t}",
        f"十位数字 = {4 * t} ÷ 4 = {t}",
    ]
    if ask == "数":
        lines = base + [
            f"百位数字 = {t} × 2 = {h}",
            f"个位数字 = {t} + {d} = {u}",
            f"这个数 = {h} × 100 + {t} × 10 + {u} = {100 * h + 10 * t + u}",
        ]
    elif ask == "百":
        lines = base + [
            f"个位数字 = {t} + {d} = {u}",
            f"这个数 = {h} × 100 + {t} × 10 + {u} = {100 * h + 10 * t + u}",
            f"百位数字 = {t} × 2 = {h}",
        ]
    else:
        lines = base + [
            f"百位数字 = {t} × 2 = {h}",
            f"这个数 = {h} × 100 + {t} × 10 + {u} = {100 * h + 10 * t + u}",
            f"个位数字 = {t} + {d} = {u}",
        ]
    return ins, lines, ans


_reg("digit_relations", digit_relations)


# 10. 两位数中间插入0后比原数大d，数字和s → 原数
def insert_zero_diff(rng):
    a = rng.randint(1, 9)
    b = rng.randint(0, 9)
    s = a + b
    d = 90 * a
    num_ = 10 * a + b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个两位数，十位数字与个位数字的和是{s}。在这个数的中间插入数字0，得到的新数比原数大{d}。这个两位数是多少？",
        f"{name}遇到一道思考题：一个两位数的各位数字之和是{s}，在它的十位和个位之间插入一个0，所得的数比原数大{d}。请你算出这个两位数。",
    ])
    lines = [
        f"十位数字 = {d} ÷ 90 = {a}",
        f"个位数字 = {s} - {a} = {b}",
        f"这个两位数 = {a} × 10 + {b} = {num_}",
    ]
    return ins, lines, num_


_reg("insert_zero_diff", insert_zero_diff)


# 11. 等差数列首项a末项l共n项 → 和
def ap_sum_first_last(rng):
    a = rng.randint(2, 15)
    n = rng.randint(4, 10)
    d = rng.randint(1, 6)
    if n % 2 == 0 and d % 2 == 1:
        d += 1
    l = a + (n - 1) * d
    s = (a + l) * n // 2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个等差数列的首项是{a}，末项是{l}，共有{n}项。这个数列的和是多少？",
        f"一列数按规律排列：首项是{a}，以后每一项都比前一项多{d}，最后一项是{l}，一共有{n}项。{name}想知道这{n}个数的和是多少，请你帮他算一算。",
    ])
    lines = [
        f"末项 = {a} + {n - 1} × {d} = {l}",
        f"首项与末项的和 = {a} + {l} = {a + l}",
        f"总和的2倍 = {a + l} × {n} = {(a + l) * n}",
        f"数列的和 = {(a + l) * n} ÷ 2 = {s}",
    ]
    return ins, lines, s


_reg("ap_sum_first_last", ap_sum_first_last)


# 12. 等差数列首项a公差d → 第n项
def ap_nth_term(rng):
    a = rng.randint(3, 20)
    d = rng.randint(2, 8)
    n = rng.randint(5, 15)
    term = a + (n - 1) * d
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个等差数列的首项是{a}，公差是{d}。这个数列的第{n}项是多少？",
        f"一列数的第一个数是{a}，从第二个数起每个数都比前一个数大{d}。{name}想知道第{n}个数是多少，请你帮他算一算。",
    ])
    lines = [
        f"间隔数 = {n} - 1 = {n - 1}",
        f"一共增加 = {n - 1} × {d} = {(n - 1) * d}",
        f"第{n}项 = {a} + {(n - 1) * d} = {term}",
    ]
    return ins, lines, term


_reg("ap_nth_term", ap_nth_term)


# 13. 最大公约数g最小公倍数l，甲数a → 乙数
def gcd_lcm_partner(rng):
    g = rng.randint(2, 9)
    m = rng.randint(2, 9)
    n = rng.randint(2, 9)
    for _ in range(50):
        if n != m:
            break
        n = rng.randint(2, 9)
    else:
        n = m + 1 if m < 9 else m - 1
    a = g * m
    l = g * m * n
    b = g * n
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：甲、乙两个数的最大公约数是{g}，最小公倍数是{l}。已知甲数是{a}，乙数是多少？",
        f"{name}遇到一道思考题：两个数的最大公约数是{g}，最小公倍数是{l}，其中一个数是{a}。请你帮他算出另一个数。",
    ])
    lines = [
        f"两数的乘积 = {g} × {l} = {g * l}",
        f"乙数 = {g * l} ÷ {a} = {b}",
    ]
    return ins, lines, b


_reg("gcd_lcm_partner", gcd_lcm_partner)


# 14. 三人分别每a、b、c天去一次，至少再过几天同去
def lcm_three_people(rng):
    a = b = c = None
    for _ in range(50):
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        c = rng.randint(2, 9)
        if a != b and b != c and a != c and _gcd(a, b) == 1 and _gcd(b, c) == 1 and _gcd(a, c) == 1:
            break
    else:
        a, b, c = 2, 3, 5
    x = a * b * c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}每{a}天去一次图书馆，小红每{b}天去一次，小华每{c}天去一次。某天他们同时去图书馆，至少再过多少天他们又同时去？",
        f"甲每{a}天去一次公园，乙每{b}天去一次，丙每{c}天去一次。今天他们正好同时去，至少再过多少天他们又同时去？",
    ])
    lines = [
        f"甲乙天数的积 = {a} × {b} = {a * b}",
        f"最少相隔天数 = {a * b} × {c} = {x}天",
    ]
    return ins, lines, x


_reg("lcm_three_people", lcm_three_people)


# 15. 除以a、b（、c）都差1整除 → 最小数
def crt_minus_one(rng):
    name = rng.choice(NAMES)
    if rng.random() < 0.5:
        a = b = None
        for _ in range(50):
            a = rng.randint(3, 9)
            b = rng.randint(a + 1, 12)
            if _gcd(a, b) == 1:
                break
        else:
            a, b = 3, 4
        x = a * b - 1
        ins = rng.choice([
            f"数学课上老师出了一道题：一个数除以{a}余{a - 1}，除以{b}余{b - 1}。这个数最小是多少？",
            f"{name}遇到一道思考题：某数除以{a}差1就能整除，除以{b}也差1就能整除。这个数最小是多少？",
        ])
        lines = [
            f"除以{a}的差 = {a} - 1 = {a - 1}",
            f"除以{b}的差 = {b} - 1 = {b - 1}",
            f"两除数的积 = {a} × {b} = {a * b}",
            f"最小的数 = {a * b} - 1 = {x}",
        ]
        return ins, lines, x
    a = b = c = None
    for _ in range(50):
        a = rng.randint(2, 5)
        b = rng.randint(a + 1, 7)
        c = rng.randint(b + 1, 9)
        if _gcd(a, b) == 1 and _gcd(b, c) == 1 and _gcd(a, c) == 1:
            break
    else:
        a, b, c = 2, 3, 5
    x = a * b * c - 1
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数除以{a}余{a - 1}，除以{b}余{b - 1}，除以{c}余{c - 1}。这个数最小是多少？",
        f"{name}遇到一道思考题：某数除以{a}、除以{b}、除以{c}都差1就能整除。这个数最小是多少？",
    ])
    lines = [
        f"除以{a}的差 = {a} - 1 = {a - 1}",
        f"除以{b}的差 = {b} - 1 = {b - 1}",
        f"除以{c}的差 = {c} - 1 = {c - 1}",
        f"甲乙两数的积 = {a} × {b} = {a * b}",
        f"三个数的积 = {a * b} × {c} = {a * b * c}",
        f"最小的数 = {a * b * c} - 1 = {x}",
    ]
    return ins, lines, x


_reg("crt_minus_one", crt_minus_one)


# 16. 两根铁丝截成每段g米 → 总段数
def wires_cut_segments(rng):
    g = rng.randint(2, 9)
    m = rng.randint(2, 9)
    n = rng.randint(2, 9)
    a = g * m
    b = g * n
    total = m + n
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"工人把两根长{a}米和{b}米的铁丝都截成每段{g}米的小段，一共可以截成多少段？",
        f"两根铁丝分别长{a}米和{b}米，要把它们截成同样长的小段，每段长{g}米。{name}想知道一共能截成多少段，请你帮他算一算。",
    ])
    lines = [
        f"第一根的段数 = {a} ÷ {g} = {m}段",
        f"第二根的段数 = {b} ÷ {g} = {n}段",
        f"总段数 = {m} + {n} = {total}段",
    ]
    return ins, lines, total


_reg("wires_cut_segments", wires_cut_segments)


# 17. 甲乙和a、乙丙和b、甲丙和c → 某数
def pairwise_sums(rng):
    a = b = c = None
    for _ in range(50):
        a = rng.randint(15, 60)
        b = rng.randint(15, 60)
        c = rng.randint(15, 60)
        if (a + b + c) % 2 == 0:
            break
    else:
        a, b, c = 20, 30, 40
    who = rng.choice(["甲", "乙", "丙"])
    ans = {"甲": (a + c - b) // 2, "乙": (a + b - c) // 2, "丙": (b + c - a) // 2}[who]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：甲、乙两数的和是{a}，乙、丙两数的和是{b}，甲、丙两数的和是{c}。{who}数是多少？",
        f"{name}遇到一道思考题：甲数加乙数等于{a}，乙数加丙数等于{b}，甲数加丙数等于{c}。请你帮他算出{who}数。",
    ])
    if who == "甲":
        lines = [
            f"两倍的甲 = {a} + {c} - {b} = {a + c - b}",
            f"甲数 = {a + c - b} ÷ 2 = {ans}",
        ]
    elif who == "乙":
        lines = [
            f"两倍的乙 = {a} + {b} - {c} = {a + b - c}",
            f"乙数 = {a + b - c} ÷ 2 = {ans}",
        ]
    else:
        lines = [
            f"两倍的丙 = {b} + {c} - {a} = {b + c - a}",
            f"丙数 = {b + c - a} ÷ 2 = {ans}",
        ]
    return ins, lines, ans


_reg("pairwise_sums", pairwise_sums)


# 18. 共有s元，甲给乙a元后甲是乙的k倍 → 甲原
def sum_transfer_multiple(rng):
    k = rng.randint(2, 5)
    per = rng.randint(5, 20)
    s = (k + 1) * per
    a = rng.randint(3, per - 1)
    jia = k * per + a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两人共有{s}元钱。如果甲给乙{a}元，甲的钱正好是乙的{k}倍。甲原来有多少元？",
        f"{name}和弟弟共有{s}元零花钱，{name}给弟弟{a}元后，{name}的钱正好是弟弟的{k}倍。{name}原来有多少元？",
    ])
    lines = [
        f"总份数 = {k} + 1 = {k + 1}",
        f"乙后来的钱 = {s} ÷ {k + 1} = {per}元",
        f"甲后来的钱 = {per} × {k} = {k * per}元",
        f"甲原来的钱 = {k * per} + {a} = {jia}元",
    ]
    return ins, lines, jia


_reg("sum_transfer_multiple", sum_transfer_multiple)


# 19. 甲缸是乙缸的k倍，甲捞f条到乙后甲是乙的m倍 → 某缸原
def fish_tank_two_constraints(rng):
    k = rng.randint(3, 6)
    m = rng.randint(2, k - 1)
    t = rng.randint(2, 8)
    b0 = (m + 1) * t
    f = (k - m) * t
    a0 = k * b0
    who = rng.choice(["甲", "乙"])
    ans = a0 if who == "甲" else b0
    fish = rng.choice(ANIMALS)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲缸{fish}的条数是乙缸的{k}倍。如果从甲缸捞出{f}条放入乙缸，甲缸的{fish}正好是乙缸的{m}倍。{who}缸原来有多少条？",
        f"两个鱼缸里养着{fish}，甲缸的条数是乙缸的{k}倍。从甲缸捞出{f}条放进乙缸后，甲缸的条数变成乙缸的{m}倍。{name}想知道{who}缸原来有多少条，请你帮他算一算。",
    ])
    lines = [
        f"倍数差 = {k} - {m} = {k - m}",
        f"份数和 = {m} + 1 = {m + 1}",
        f"乙缸条数的{m + 1}倍 = {f} × {m + 1} = {f * (m + 1)}",
        f"乙缸原来的条数 = {f * (m + 1)} ÷ {k - m} = {b0}条",
    ]
    if who == "甲":
        lines.append(f"甲缸原来的条数 = {b0} × {k} = {a0}条")
    return ins, lines, ans


_reg("fish_tank_two_constraints", fish_tank_two_constraints)


# 20. 甲绳a米乙绳b米，剪去同样长后甲是乙的k倍 → 剪去多少
def two_ropes_equal_after_cut(rng):
    b = rng.randint(5, 20)
    k = rng.randint(2, 5)
    c = rng.randint(2, b - 1)
    a = k * b - (k - 1) * c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲绳长{a}米，乙绳长{b}米。两根绳子剪去同样长的一段后，甲绳剩下的长度正好是乙绳剩下的{k}倍。剪去了多少米？",
        f"两根绳子分别长{a}米和{b}米，剪去同样长的一段后，长绳剩下的长度是短绳剩下的{k}倍。{name}想知道剪去了多少米，请你帮他算一算。",
    ])
    lines = [
        f"乙绳的{k}倍 = {k} × {b} = {k * b}",
        f"多出的部分 = {k * b} - {a} = {k * b - a}",
        f"倍数差 = {k} - 1 = {k - 1}",
        f"剪去的长度 = {k * b - a} ÷ {k - 1} = {c}米",
    ]
    return ins, lines, c


_reg("two_ropes_equal_after_cut", two_ropes_equal_after_cut)


# 21. 两绳共长s，甲减a、乙减b后相等 → 某绳原长
def two_ropes_diff_cuts_equal(rng):
    a = rng.randint(3, 15)
    b = rng.randint(3, 15)
    if (a - b) % 2 != 0:
        b += 1
    s = rng.randint(20, 60) * 2
    jia = (s + a - b) // 2
    yi = (s - a + b) // 2
    who = rng.choice(["甲", "乙"])
    ans = jia if who == "甲" else yi
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲、乙两根绳子共长{s}米。甲绳剪去{a}米、乙绳剪去{b}米后，两根绳子剩下的长度相等。{who}绳原来长多少米？",
        f"两根绳子一共长{s}米，第一根剪去{a}米，第二根剪去{b}米后，剩下的部分一样长。{name}想知道{('第一根' if who == '甲' else '第二根')}原来长多少米，请你帮他算一算。",
    ])
    if who == "甲":
        lines = [
            f"两倍的甲 = {s} + {a} - {b} = {s + a - b}",
            f"甲绳原长 = {s + a - b} ÷ 2 = {jia}米",
        ]
    else:
        lines = [
            f"两倍的乙 = {s} + {b} - {a} = {s + b - a}",
            f"乙绳原长 = {s + b - a} ÷ 2 = {yi}米",
        ]
    return ins, lines, ans


_reg("two_ropes_diff_cuts_equal", two_ropes_diff_cuts_equal)


# 22. 今年哥是妹的k倍，a年后年龄和s → 妹今年
def age_brother_sister_future_sum(rng):
    k = rng.randint(2, 5)
    x = rng.randint(4, 15)
    a = rng.randint(3, 10)
    s = (k + 1) * x + 2 * a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"今年哥哥的年龄是妹妹的{k}倍，{a}年后兄妹俩的年龄和是{s}岁。妹妹今年多少岁？",
        f"{name}今年的年龄是妹妹的{k}倍，再过{a}年，两人的年龄加起来是{s}岁。妹妹今年多少岁？",
    ])
    lines = [
        f"总份数 = {k} + 1 = {k + 1}",
        f"今年的年龄和 = {s} - 2 × {a} = {s - 2 * a}岁",
        f"妹妹今年的年龄 = {s - 2 * a} ÷ {k + 1} = {x}岁",
    ]
    return ins, lines, x


_reg("age_brother_sister_future_sum", age_brother_sister_future_sum)


# 23. 三代年龄和s，爸是子的a倍，爷是爸的b倍 → 孙今年
def age_three_generations(rng):
    a = rng.randint(2, 4)
    b = rng.randint(2, 4)
    x = rng.randint(3, 10)
    s = x * (1 + a + a * b)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"爷爷、爸爸和孙子的年龄和是{s}岁。爸爸的年龄是孙子的{a}倍，爷爷的年龄是爸爸的{b}倍。孙子今年多少岁？",
        f"爷爷、爸爸和孙子的年龄加起来是{s}岁，爸爸的岁数是孙子的{a}倍，爷爷的岁数是爸爸的{b}倍。{name}想知道孙子今年多少岁，请你帮他算一算。",
    ])
    lines = [
        f"爷爷是孙子的倍数 = {a} × {b} = {a * b}",
        f"总份数 = 1 + {a} + {a * b} = {1 + a + a * b}",
        f"孙子今年的年龄 = {s} ÷ {1 + a + a * b} = {x}岁",
    ]
    return ins, lines, x


_reg("age_three_generations", age_three_generations)


# 24. 父子年龄差d，a年后父是子的k倍 → 子今年
def age_diff_future_ratio(rng):
    k = rng.randint(2, 5)
    son_a = rng.randint(5, 15)
    d = (k - 1) * son_a
    a = rng.randint(2, son_a - 2)
    x = son_a - a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"今年父子俩的年龄差是{d}岁，{a}年后父亲的年龄正好是儿子的{k}倍。儿子今年多少岁？",
        f"父亲比儿子大{d}岁，再过{a}年，父亲的岁数是儿子的{k}倍。{name}想知道儿子今年多少岁，请你帮他算一算。",
    ])
    lines = [
        f"倍数差 = {k} - 1 = {k - 1}",
        f"{a}年后儿子的年龄 = {d} ÷ {k - 1} = {son_a}岁",
        f"儿子今年的年龄 = {son_a} - {a} = {x}岁",
    ]
    return ins, lines, x


_reg("age_diff_future_ratio", age_diff_future_ratio)


# 25. 今年父子年龄和s，a年后父是子的k倍 → 子今年
def age_sum_future_ratio(rng):
    k = rng.randint(2, 5)
    son_a = rng.randint(6, 15)
    a = rng.randint(2, son_a - 3)
    s = (k + 1) * son_a - 2 * a
    x = son_a - a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"今年父子俩的年龄和是{s}岁，{a}年后父亲的年龄正好是儿子的{k}倍。儿子今年多少岁？",
        f"父亲和儿子今年的年龄加起来是{s}岁，再过{a}年，父亲的岁数是儿子的{k}倍。{name}想知道儿子今年多少岁，请你帮他算一算。",
    ])
    lines = [
        f"总份数 = {k} + 1 = {k + 1}",
        f"{a}年后的年龄和 = {s} + 2 × {a} = {s + 2 * a}岁",
        f"{a}年后儿子的年龄 = {s + 2 * a} ÷ {k + 1} = {son_a}岁",
        f"儿子今年的年龄 = {son_a} - {a} = {x}岁",
    ]
    return ins, lines, x


_reg("age_sum_future_ratio", age_sum_future_ratio)


# 26. 两层共s本，上层拿a本到下层后上层仍多b本 → 某层原
def books_two_shelves_transfer(rng):
    s = a = b = upper = lower = None
    for _ in range(50):
        s = rng.randint(50, 150)
        a = rng.randint(3, 12)
        b = rng.randint(3, 12)
        if (s + b) % 2 == 0 and (s - b) // 2 > a + 1:
            upper = (s + b) // 2 + a
            lower = (s - b) // 2 - a
            break
    else:
        s, a, b = 80, 5, 6
        upper, lower = 48, 32
    who = rng.choice(["上层", "下层"])
    ans = upper if who == "上层" else lower
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"书架上、下两层共有{s}本书。从上层拿{a}本放到下层后，上层还比下层多{b}本。{who}原来有多少本书？",
        f"一个书架的两层共放了{s}本书，如果从上层取出{a}本放到下层，上层仍比下层多{b}本。{name}想知道{who}原来有多少本书，请你帮他算一算。",
    ])
    if who == "上层":
        lines = [
            f"后来上层的2倍 = {s} + {b} = {s + b}",
            f"后来上层的本数 = {s + b} ÷ 2 = {(s + b) // 2}本",
            f"上层原来的本数 = {(s + b) // 2} + {a} = {upper}本",
        ]
    else:
        lines = [
            f"后来下层的2倍 = {s} - {b} = {s - b}",
            f"后来下层的本数 = {s - b} ÷ 2 = {(s - b) // 2}本",
            f"下层原来的本数 = {(s - b) // 2} - {a} = {lower}本",
        ]
    return ins, lines, ans


_reg("books_two_shelves_transfer", books_two_shelves_transfer)


# 27. 圆形池塘周长L，每隔d米栽树，每棵price元 → 总价
def circular_trees_cost(rng):
    d = rng.randint(3, 8)
    k = rng.randint(5, 20)
    L = d * k
    price = rng.randint(5, 30)
    total = k * price
    place = rng.choice(["池塘", "花坛", "操场", "草坪"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个圆形{place}的周长是{L}米，沿周围每隔{d}米栽一棵树，每棵树苗{price}元。买树苗一共要多少元？",
        f"园林工人沿圆形{place}周围栽树，{place}周长{L}米，每隔{d}米栽一棵，每棵树苗{price}元。{name}想知道买树苗共需多少元，请你帮他算一算。",
    ])
    lines = [
        f"树苗棵数 = {L} ÷ {d} = {k}棵",
        f"总价钱 = {k} × {price} = {total}元",
    ]
    return ins, lines, total


_reg("circular_trees_cost", circular_trees_cost)


# 28. 每分钟v1米迟到t1，每分钟v2米早到t2 → 距离
def speed_late_early(rng):
    d = rng.randint(5, 15)
    k = rng.randint(4, 10)
    v1 = d * k
    v2 = d * (k + 1)
    t1 = rng.randint(3, 10)
    t2 = rng.randint(3, 10)
    s = (t1 + t2) * v1 * v2 // d
    name = rng.choice(NAMES)
    place = rng.choice(PLACE)
    ins = rng.choice([
        f"{name}从家出发去{place}，如果每分钟走{v1}米，就要迟到{t1}分钟；如果每分钟走{v2}米，就会早到{t2}分钟。家到{place}有多少米？",
        f"小华上学，每分钟走{v1}米会迟到{t1}分钟，每分钟走{v2}米会早到{t2}分钟。他家到学校有多少米？",
    ])
    lines = [
        f"速度差 = {v2} - {v1} = {d}",
        f"时间差 = {t1} + {t2} = {t1 + t2}",
        f"速度积 = {v1} × {v2} = {v1 * v2}",
        f"家到{place}的距离 = {t1 + t2} × {v1 * v2} ÷ {d} = {s}米",
    ]
    return ins, lines, s


_reg("speed_late_early", speed_late_early)


# 29. 甲每天a页乙每天b页，甲早t天看完 → 书的页数
def reading_days_diff(rng):
    d = rng.randint(2, 8)
    k = rng.randint(3, 10)
    a = d * (k + 1)
    b = d * k
    t = rng.randint(2, 8)
    pages = t * a * b // d
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"看同一本书，甲每天看{a}页，乙每天看{b}页，甲比乙早{t}天看完。这本书有多少页？",
        f"{name}和小红看同一本故事书，{name}每天看{a}页，小红每天看{b}页，结果{name}比小红早{t}天看完。这本书共有多少页？",
    ])
    lines = [
        f"每天看的页数差 = {a} - {b} = {d}",
        f"每天看的页数积 = {a} × {b} = {a * b}",
        f"这本书的页数 = {a * b} × {t} ÷ {d} = {pages}页",
    ]
    return ins, lines, pages


_reg("reading_days_diff", reading_days_diff)


# 30. 甲池a吨乙池b吨，甲每小时注c乙每小时注d → 几小时后相等
def pools_equalize(rng):
    e = rng.randint(2, 8)
    k = rng.randint(3, 10)
    diff = e * k
    a = rng.randint(diff + 20, diff + 120)
    b = a - diff
    c = rng.randint(3, 12)
    d = c + e
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲池有水{a}吨，乙池有水{b}吨。甲池每小时注入{c}吨水，乙池每小时注入{d}吨水。多少小时后两池的水一样多？",
        f"两个水池，甲池原有{a}吨水，乙池原有{b}吨水。甲池每小时进水{c}吨，乙池每小时进水{d}吨。{name}想知道几小时后两池水一样多，请你帮他算一算。",
    ])
    lines = [
        f"原来的水量差 = {a} - {b} = {diff}吨",
        f"每小时进水量差 = {d} - {c} = {e}吨",
        f"需要的时间 = {diff} ÷ {e} = {k}小时",
    ]
    return ins, lines, k


_reg("pools_equalize", pools_equalize)


# 31. 池原有w吨，进a排b，几小时后有c吨
def pool_net_to_target(rng):
    e = rng.randint(2, 8)
    k = rng.randint(3, 10)
    need = e * k
    w = rng.randint(20, 80)
    c = w + need
    a = rng.randint(10, 25)
    b = a - e
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个水池原有水{w}吨，进水管每小时进水{a}吨，排水管每小时排水{b}吨。两管同时打开，多少小时后水池有水{c}吨？",
        f"水池里已有{w}吨水，管理员同时打开进水管和排水管，进水管每小时进{a}吨，排水管每小时排{b}吨。{name}想知道几小时后水池有水{c}吨，请你帮他算一算。",
    ])
    lines = [
        f"需要注入的水量 = {c} - {w} = {need}吨",
        f"每小时净进水量 = {a} - {b} = {e}吨",
        f"需要的时间 = {need} ÷ {e} = {k}小时",
    ]
    return ins, lines, k


_reg("pool_net_to_target", pool_net_to_target)


# 32. 进水管a小时满，排水管b小时空，先进t小时再齐开 → 还需几小时
def pool_fill_then_drain(rng):
    a = rng.randint(3, 8)
    b = a + rng.randint(3, 10)
    t = rng.randint(1, a - 1)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    net = ra - rb
    rest = 1 - t * ra
    t2 = rest / net
    total = t + t2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个空水池，单开进水管{a}小时注满，单开排水管{b}小时排空。先开进水管{t}小时后，再打开排水管，还要多少小时才能注满？",
        f"水池有一个进水管和一个排水管，进水管单开{a}小时注满空池，排水管单开{b}小时排空满池。工人先开进水管{t}小时，然后打开排水管，{name}想知道还要几小时注满，请你帮他算一算。",
    ])
    lines = [
        f"进水管效率 = 1 ÷ {a} = {num(ra)}",
        f"排水管效率 = 1 ÷ {b} = {num(rb)}",
        f"净效率 = {num(ra)} - {num(rb)} = {num(net)}",
        f"先注入的水量 = {t} × {num(ra)} = {num(t * ra)}",
        f"剩余的水量 = 1 - {num(t * ra)} = {num(rest)}",
        f"还需要的时间 = {num(rest)} ÷ ({num(net)}) = {num(t2)}时",
        f"一共用的时间 = {t} + {num(t2)} = {num(total)}时",
    ]
    return ins, lines, total


_reg("pool_fill_then_drain", pool_fill_then_drain)


# 33. 甲a天乙b天，合做期间甲休息c天 → 一共用几天
def work_rest_days(rng):
    a = rng.randint(8, 15)
    b = rng.randint(6, 12)
    c = rng.randint(1, 4)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    rate = ra + rb
    done = c * rb
    rest = 1 - done
    t_rest = rest / rate
    total = c + t_rest
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程，甲队单独做{a}天完成，乙队单独做{b}天完成。两队合做期间甲队休息了{c}天，完成这项工程一共用了多少天？",
        f"修一条路，甲队单独修{a}天完成，乙队单独修{b}天完成。两队合修时甲队休息了{c}天，{name}想知道从开工到完成共用了多少天，请你帮他算一算。",
    ])
    lines = [
        f"甲队效率 = 1 ÷ {a} = {num(ra)}",
        f"乙队效率 = 1 ÷ {b} = {num(rb)}",
        f"合作效率 = {num(ra)} + {num(rb)} = {num(rate)}",
        f"甲休息时乙做的 = {c} × {num(rb)} = {num(done)}",
        f"剩余的工程 = 1 - {num(done)} = {num(rest)}",
        f"合作的天数 = {num(rest)} ÷ ({num(rate)}) = {num(t_rest)}天",
        f"一共用的天数 = {c} + {num(t_rest)} = {num(total)}天",
    ]
    return ins, lines, total


_reg("work_rest_days", work_rest_days)


# 34. 甲乙效率比a:b，合做t天完成 → 甲单独几天
def work_ratio_solo(rng):
    a = b = t = None
    for _ in range(50):
        a = rng.randint(2, 6)
        b = rng.randint(2, 6)
        if b == a:
            continue
        t = rng.randint(3, 8)
        if (t * (a + b)) % a == 0:
            break
    else:
        a, b, t = 2, 3, 4
    solo = t * (a + b) // a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一项工程，甲、乙两队的工作效率比是{a}比{b}，两队合做{t}天完成。甲队单独做需要多少天完成？",
        f"加工一批零件，甲、乙两人的效率比是{a}:{b}，两人合做{t}天完成。{name}想知道甲单独做需要几天，请你帮他算一算。",
    ])
    lines = [
        f"效率份数和 = {a} + {b} = {a + b}",
        f"总工作量 = {t} × {a + b} = {t * (a + b)}",
        f"甲单独做的天数 = {t * (a + b)} ÷ {a} = {solo}天",
    ]
    return ins, lines, solo


_reg("work_ratio_solo", work_ratio_solo)


# 35. 甲a小时满乙b小时满，排水管c小时空，三管齐开 → 几小时满
def pipes_two_in_one_out(rng):
    a = rng.randint(3, 6)
    b = rng.randint(3, 6)
    c = b + rng.randint(3, 8)
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    rc = Fraction(1, c)
    net = ra + rb - rc
    t = Fraction(1, net)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个水池有甲、乙两个进水管和一个排水管。单开甲管{a}小时注满，单开乙管{b}小时注满，单开排水管{c}小时排空。甲、乙两管和排水管同时打开，多少小时注满？",
        f"水池的甲管单开{a}小时注满，乙管单开{b}小时注满，排水管单开{c}小时把满池水排空。{name}把甲、乙两管和排水管同时打开，想知道几小时注满，请你帮他算一算。",
    ])
    lines = [
        f"甲管效率 = 1 ÷ {a} = {num(ra)}",
        f"乙管效率 = 1 ÷ {b} = {num(rb)}",
        f"排水管效率 = 1 ÷ {c} = {num(rc)}",
        f"净效率 = {num(ra)} + {num(rb)} - {num(rc)} = {num(net)}",
        f"注满时间 = 1 ÷ ({num(net)}) = {num(t)}时",
    ]
    return ins, lines, t


_reg("pipes_two_in_one_out", pipes_two_in_one_out)


# 36. 长增a米面积增s1，宽增b米面积增s2 → 原面积
def rect_area_change(rng):
    w = rng.randint(4, 15)
    l = rng.randint(6, 20)
    a = rng.randint(2, 8)
    b = rng.randint(2, 8)
    s1 = a * w
    s2 = b * l
    area = w * l
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个长方形，如果长增加{a}米，面积就增加{s1}平方米；如果宽增加{b}米，面积就增加{s2}平方米。这个长方形原来的面积是多少平方米？",
        f"一块长方形菜地，长增加{a}米后面积增加{s1}平方米，宽增加{b}米后面积增加{s2}平方米。{name}想知道这块菜地原来的面积，请你帮他算一算。",
    ])
    lines = [
        f"原来的宽 = {s1} ÷ {a} = {w}米",
        f"原来的长 = {s2} ÷ {b} = {l}米",
        f"原来的面积 = {w} × {l} = {area}平方米",
    ]
    return ins, lines, area


_reg("rect_area_change", rect_area_change)


# 37. 正方形边长增a米，面积增s平方米 → 原边长
def square_side_increase(rng):
    x = rng.randint(3, 15)
    a = rng.randint(2, 8)
    s = 2 * a * x + a * a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个正方形，如果边长增加{a}米，面积就增加{s}平方米。这个正方形原来的边长是多少米？",
        f"一块正方形菜地，边长增加{a}米后，面积增加了{s}平方米。{name}想知道原来的边长，请你帮他算一算。",
    ])
    lines = [
        f"增加的小正方形面积 = {a} × {a} = {a * a}",
        f"两个长方形的面积 = {s} - {a * a} = {s - a * a}",
        f"边长增加的2倍 = 2 × {a} = {2 * a}",
        f"原来的边长 = {s - a * a} ÷ {2 * a} = {x}米",
    ]
    return ins, lines, x


_reg("square_side_increase", square_side_increase)


# 38. 正方体棱长之和s → 表面积
def cube_edge_sum_surface(rng):
    e = rng.randint(3, 15)
    s = 12 * e
    surf = 6 * e * e
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个正方体的棱长之和是{s}厘米，它的表面积是多少平方厘米？",
        f"用一根长{s}厘米的铁丝正好做成一个正方体框架，{name}想知道这个正方体的表面积是多少平方厘米，请你帮他算一算。",
    ])
    lines = [
        f"棱长 = {s} ÷ 12 = {e}厘米",
        f"一个面的面积 = {e} × {e} = {e * e}",
        f"表面积 = {e * e} × 6 = {surf}平方厘米",
    ]
    return ins, lines, surf


_reg("cube_edge_sum_surface", cube_edge_sum_surface)


# 39. 长方体长宽高 → 表面积
def cuboid_surface(rng):
    a = rng.randint(3, 15)
    b = rng.randint(3, 15)
    c = rng.randint(3, 15)
    surf = 2 * (a * b + b * c + a * c)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个长方体的长是{a}厘米，宽是{b}厘米，高是{c}厘米。它的表面积是多少平方厘米？",
        f"一个长方体礼盒，长{a}厘米、宽{b}厘米、高{c}厘米。{name}想知道包装这个礼盒至少需要多少平方厘米的彩纸，请你帮他算一算。",
    ])
    lines = [
        f"上下两个面 = {a} × {b} = {a * b}",
        f"前后两个面 = {b} × {c} = {b * c}",
        f"左右两个面 = {a} × {c} = {a * c}",
        f"三个面的面积和 = {a * b} + {b * c} + {a * c} = {a * b + b * c + a * c}",
        f"表面积 = {a * b + b * c + a * c} × 2 = {surf}平方厘米",
    ]
    return ins, lines, surf


_reg("cuboid_surface", cuboid_surface)


# 40. 玻璃缸长a宽b水深h，放石头后水面到c → 石头体积
def rock_displacement(rng):
    a = rng.randint(3, 12)
    b = rng.randint(3, 12)
    h = rng.randint(3, 10)
    rise = rng.randint(1, h - 1)
    c = h + rise
    vol = a * b * rise
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一个长方体玻璃缸，长{a}分米，宽{b}分米，里面水深{h}分米。放入一块石头后，水面正好上升到缸口，缸高{c}分米。这块石头的体积是多少立方分米？",
        f"在长{a}分米、宽{b}分米的长方体水缸里，水深{h}分米。把一块石头放入水中，水面上升到{c}分米（水未溢出）。{name}想知道这块石头的体积，请你帮他算一算。",
    ])
    lines = [
        f"水面上升的高度 = {c} - {h} = {rise}分米",
        f"水缸底面积 = {a} × {b} = {a * b}",
        f"石头的体积 = {a * b} × {rise} = {vol}立方分米",
    ]
    return ins, lines, vol


_reg("rock_displacement", rock_displacement)


# 41. 竹竿h米影长a米，大树影长比竹竿多d米 → 树高
def shadow_height(rng):
    h = rng.choice([2, 3, 4])
    a = rng.randint(2, 6)
    k = rng.randint(2, 8)
    d = a * k
    tree = h * (k + 1)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"同一时刻，一根{h}米长的竹竿影子长{a}米，一棵大树的影子比竹竿的影子长{d}米。这棵大树高多少米？",
        f"阳光下，{h}米高的竹竿影长{a}米，旁边一棵大树的影长比竹竿多{d}米。{name}想知道大树的高度，请你帮他算一算。",
    ])
    lines = [
        f"大树的影长 = {a} + {d} = {a + d}米",
        f"大树的高度 = {h} × {a + d} ÷ {a} = {tree}米",
    ]
    return ins, lines, tree


_reg("shadow_height", shadow_height)


# 42. 出租车起步价a元3千米，超出每千米b元，行s千米 → 车费
def taxi_fare(rng):
    a = rng.randint(8, 15)
    b = rng.randint(2, 5)
    s = rng.randint(5, 30)
    fare = a + (s - 3) * b
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"出租车起步价{a}元（3千米以内），超过3千米后每千米{b}元。{name}乘出租车行了{s}千米，应付车费多少元？",
        f"某市出租车起步价为{a}元，3千米内只收起步价，超出部分每千米{b}元。{name}坐车行驶{s}千米，要付多少元？",
    ])
    lines = [
        f"超出起步价的路程 = {s} - 3 = {s - 3}千米",
        f"超出部分的车费 = {s - 3} × {b} = {(s - 3) * b}元",
        f"应付车费 = {a} + {(s - 3) * b} = {fare}元",
    ]
    return ins, lines, fare


_reg("taxi_fare", taxi_fare)


# 43. 出租车起步价a元3千米，超出每千米b元，付f元 → 路程
def taxi_fare_reverse(rng):
    a = rng.randint(8, 15)
    b = rng.randint(2, 5)
    km = rng.randint(3, 20)
    f = a + km * b
    s = km + 3
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"出租车起步价{a}元（3千米以内），超过3千米后每千米{b}元。{name}乘出租车付车费{f}元，他行了多少千米？",
        f"某市出租车起步价{a}元，3千米内只收起步价，超出部分每千米{b}元。{name}下车时付了{f}元，出租车行驶了多少千米？",
    ])
    lines = [
        f"超出起步价的车费 = {f} - {a} = {f - a}元",
        f"超出起步价的路程 = {f - a} ÷ {b} = {km}千米",
        f"一共行驶的路程 = {km} + 3 = {s}千米",
    ]
    return ins, lines, s


_reg("taxi_fare_reverse", taxi_fare_reverse)


# 44. 存p元年利率r%存n年，利息税5% → 取回多少
def interest_tax(rng):
    k = rng.randint(2, 10) * 2
    p = 1000 * k
    r = rng.choice([2, 3, 4, 5])
    n = rng.randint(2, 5)
    interest = p * r // 100 * n
    tax = interest * 5 // 100
    total = p + interest - tax
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}把{p}元压岁钱存入银行，年利率是{r}%，存满{n}年。按规定利息要缴纳5%的利息税，到期时一共可以取回多少元？",
        f"银行一年期年利率是{r}%，{name}把{p}元存了{n}年，利息税为5%。到期后扣除利息税，一共能取回多少元？",
    ])
    lines = [
        f"一年的利息 = {p} × {r}/100 = {p * r // 100}元",
        f"{n}年的利息 = {p * r // 100} × {n} = {interest}元",
        f"利息税 = {interest} × 5/100 = {tax}元",
        f"一共取回的钱 = {p} + {interest} - {tax} = {total}元",
    ]
    return ins, lines, total


_reg("interest_tax", interest_tax)


# 45. 存p元年利率r%，复利2年 → 取回多少
def compound_interest(rng):
    r = rng.choice([5, 10, 20])
    if r == 5:
        k = rng.randint(2, 10)
        p = 400 * k
    else:
        k = rng.randint(2, 20)
        p = 100 * k
    y1 = p * (100 + r) // 100
    i2 = y1 * r // 100
    total = y1 + i2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}把{p}元存入银行，年利率是{r}%，按复利计算，存满2年后一共可以取回多少元？",
        f"银行年利率为{r}%，{name}存入{p}元，每年的利息计入下一年的本金。2年后一共能取回多少元？",
    ])
    lines = [
        f"第一年的利息 = {p} × {r}/100 = {p * r // 100}元",
        f"第一年到期的本息 = {p} + {p * r // 100} = {y1}元",
        f"第二年的利息 = {y1} × {r}/100 = {i2}元",
        f"两年到期的本息 = {y1} + {i2} = {total}元",
    ]
    return ins, lines, total


_reg("compound_interest", compound_interest)


# 46. 售价a元，首付b元，余c个月每月还d元 → 多花多少
def installment(rng):
    a = b = c = d = extra = None
    for _ in range(50):
        a = rng.randint(20, 80) * 100
        b = rng.randint(5, 20) * 100
        c = rng.randint(6, 12)
        d = rng.randint(20, 60) * 10
        if c * d > a - b:
            extra = c * d - (a - b)
            break
    else:
        a, b, c, d = 3000, 500, 10, 300
        extra = 500
    name = rng.choice(NAMES)
    obj = rng.choice(["手机", "电脑", "电动车", "冰箱", "洗衣机"])
    ins = rng.choice([
        f"一台{obj}售价{a}元，{name}先付{b}元，余下的分{c}个月付清，每月还{d}元。分期付款比一次性付款多花多少元？",
        f"商店里一台{obj}卖{a}元，可以先付{b}元，剩下的分{c}个月每月还{d}元。{name}想知道分期付款比原价多花多少元，请你帮他算一算。",
    ])
    lines = [
        f"分期付款总额 = {c} × {d} = {c * d}元",
        f"需要分期的金额 = {a} - {b} = {a - b}元",
        f"多花的钱 = {c * d} - {a - b} = {extra}元",
    ]
    return ins, lines, extra


_reg("installment", installment)


# 47. 甲存a元年利率r1%，乙存b元年利率r2% → 利息差
def two_deposits_interest(rng):
    x = rng.randint(2, 20)
    r1 = rng.choice([2, 3, 4, 5])
    y = rng.randint(2, 20)
    r2 = rng.choice([2, 3, 4, 5])
    for _ in range(50):
        if x * r1 != y * r2:
            break
        y = rng.randint(2, 20)
    else:
        y = x + 1
    a = 100 * x
    b = 100 * y
    i1 = x * r1
    i2 = y * r2
    diff = abs(i1 - i2)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲把{a}元存入银行，年利率{r1}%；乙把{b}元存入银行，年利率{r2}%。一年后两人的利息相差多少元？",
        f"{name}把{a}元按年利率{r1}%存入银行，小红把{b}元按年利率{r2}%存入银行。一年后两人的利息相差多少元？",
    ])
    lines = [
        f"甲的利息 = {a} × {r1}/100 = {i1}元",
        f"乙的利息 = {b} × {r2}/100 = {i2}元",
        f"利息差 = {max(i1, i2)} - {min(i1, i2)} = {diff}元",
    ]
    return ins, lines, diff


_reg("two_deposits_interest", two_deposits_interest)


# 48. 两件商品各卖a元，一件赚p%一件亏p% → 合计盈亏
def profit_loss_pair(rng):
    p = rng.choice([20, 25])
    if p == 20:
        k = rng.randint(2, 20)
        a = 12 * k
        c1 = 10 * k
        c2 = 15 * k
    else:
        k = rng.randint(2, 20)
        a = 15 * k
        c1 = 12 * k
        c2 = 20 * k
    sold = 2 * a
    cost = c1 + c2
    loss = cost - sold
    name = rng.choice(NAMES)
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店卖出两件{obj}，每件都卖{a}元，其中一件赚{p}%，另一件亏{p}%。两件合起来亏了多少元？",
        f"老板把两件{obj}都卖了{a}元，一件赚了{p}%，另一件亏了{p}%。{name}想知道两件合计亏了多少元，请你帮他算一算。",
    ])
    lines = [
        f"两件卖的总钱数 = {a} × 2 = {sold}元",
        f"赚的那件成本 = {a} ÷ ({100 + p}/100) = {c1}元",
        f"亏的那件成本 = {a} ÷ ({100 - p}/100) = {c2}元",
        f"两件的总成本 = {c1} + {c2} = {cost}元",
        f"亏的钱数 = {cost} - {sold} = {loss}元",
    ]
    return ins, lines, loss


_reg("profit_loss_pair", profit_loss_pair)


# 49. 甲做总数的1/n1，乙做总数的1/n2，甲比乙多c个 → 总数
def fraction_diff_total(rng):
    n1, n2 = rng.choice([(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)])
    k = rng.randint(3, 20)
    c = k * (n2 - n1)
    total = k * n1 * n2
    diff = Fraction(n2 - n1, n1 * n2)
    obj = rng.choice(["零件", "花", "树", "书"])
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"加工一批{obj}，甲完成了总数的1/{n1}，乙完成了总数的1/{n2}，甲比乙多做{c}个。这批{obj}共有多少个？",
        f"一批{obj}，甲做了全部的1/{n1}，乙做了全部的1/{n2}，甲比乙多做{c}个。{name}想知道这批{obj}的总数，请你帮他算一算。",
    ])
    lines = [
        f"分率差 = 1 ÷ {n1} - 1 ÷ {n2} = {num(diff)}",
        f"总数 = {c} ÷ ({num(diff)}) = {total}个",
    ]
    return ins, lines, total


_reg("fraction_diff_total", fraction_diff_total)


# 50. 第一次剪1/n1，第二次剪1/n2，最后一次比第二次多a米 → 原长
def rope_three_cuts(rng):
    n1, n2, mult, div = rng.choice([(3, 4, 6, 1), (3, 5, 15, 4), (4, 5, 20, 7),
                                    (2, 5, 10, 1), (3, 6, 3, 1), (4, 6, 12, 5)])
    k = rng.randint(2, 12)
    a = div * k
    total = mult * k
    f1 = Fraction(1, n1)
    f2 = Fraction(1, n2)
    f3 = 1 - f1 - f2
    diff = f3 - f2
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一根绳子，第一次剪去全长的1/{n1}，第二次剪去全长的1/{n2}，最后一次比第二次多剪{a}米，正好剪完。这根绳子原来长多少米？",
        f"一根绳子分几次剪完，第一次剪全长的1/{n1}，第二次剪全长的1/{n2}，最后一次比第二次多剪{a}米。{name}想知道绳子原来的长度，请你帮他算一算。",
    ])
    lines = [
        f"前两次共剪的分率 = 1 ÷ {n1} + 1 ÷ {n2} = {num(f1 + f2)}",
        f"最后一次剪的分率 = 1 - {num(f1 + f2)} = {num(f3)}",
        f"最后一次比第二次多的分率 = {num(f3)} - 1 ÷ {n2} = {num(diff)}",
        f"绳子原长 = {a} ÷ ({num(diff)}) = {total}米",
    ]
    return ins, lines, total


_reg("rope_three_cuts", rope_three_cuts)


# 51. 第一天看a页，第二天看全书的1/n2，两天共看全书的1/n1 → 全书页数
def pages_two_days_fraction(rng):
    n1, n2 = rng.choice([(3, 4), (3, 5), (4, 5), (2, 5), (3, 6), (4, 6)])
    k = rng.randint(2, 12)
    a = k * (n2 - n1)
    total = k * n1 * n2
    diff = Fraction(1, n1) - Fraction(1, n2)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"看一本书，第一天看了{a}页，第二天看了全书的1/{n2}，两天正好看了全书的1/{n1}。这本书共有多少页？",
        f"{name}读一本故事书，第一天读{a}页，第二天读了全书的1/{n2}，两天共读了全书的1/{n1}。这本书共有多少页？",
    ])
    lines = [
        f"两天的分率差 = 1 ÷ {n1} - 1 ÷ {n2} = {num(diff)}",
        f"全书页数 = {a} ÷ ({num(diff)}) = {total}页",
    ]
    return ins, lines, total


_reg("pages_two_days_fraction", pages_two_days_fraction)


# 52. 两个连续页码和s → 较小页码
def consecutive_pages(rng):
    x = rng.randint(3, 40)
    s = 2 * x + 1
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}翻开一本书，看到两个连续的页码，它们的和是{s}。较小的页码是几？",
        f"小明打开一本书，发现两个连续页码相加得{s}。{name}想知道较小的页码是多少，请你帮他算一算。",
    ])
    lines = [
        f"较小页码的2倍 = {s} - 1 = {s - 1}",
        f"较小的页码 = {s - 1} ÷ 2 = {x}",
    ]
    return ins, lines, x


_reg("consecutive_pages", consecutive_pages)


# 53. n边形 → 对角线总数 / 一个顶点的对角线 / 三角形个数
def polygon_diagonals(rng):
    n = rng.randint(5, 20)
    ask = rng.choice(["对角线", "一个顶点", "三角形"])
    name = rng.choice(NAMES)
    if ask == "对角线":
        ans = n * (n - 3) // 2
        q = f"一个{n}边形一共有多少条对角线？"
        lines = [
            f"每个顶点出发的对角线 = {n} - 3 = {n - 3}条",
            f"所有顶点的对角线总数 = {n} × {n - 3} = {n * (n - 3)}条",
            f"对角线总数 = {n * (n - 3)} ÷ 2 = {ans}条",
        ]
    elif ask == "一个顶点":
        ans = n - 3
        q = f"从一个{n}边形的一个顶点出发，可以画出多少条对角线？"
        lines = [f"对角线数 = {n} - 3 = {ans}条"]
    else:
        ans = n - 2
        q = f"从一个{n}边形的一个顶点出发画对角线，可以把这个{n}边形分成多少个三角形？"
        lines = [f"三角形个数 = {n} - 2 = {ans}个"]
    ins = rng.choice([
        f"数学课上老师出了一道题：{q}",
        f"{name}在练习册上看到一道题：{q}请你帮他算一算。",
    ])
    return ins, lines, ans


_reg("polygon_diagonals", polygon_diagonals)


# 54. n边形 → 内角和 / 正n边形每个内角
def polygon_interior_angles(rng):
    ask = rng.choice(["内角和", "每个内角"])
    if ask == "每个内角":
        n = rng.randint(3, 12)
        ans = Fraction((n - 2) * 180, n)
        q = f"一个正{n}边形的每个内角是多少度？"
        lines = [
            f"三角形的个数 = {n} - 2 = {n - 2}",
            f"内角和 = {n - 2} × 180 = {(n - 2) * 180}度",
            f"每个内角 = {(n - 2) * 180} ÷ {n} = {num(ans)}度",
        ]
    else:
        n = rng.randint(3, 30)
        ans = (n - 2) * 180
        q = f"一个{n}边形的内角和是多少度？"
        lines = [
            f"三角形的个数 = {n} - 2 = {n - 2}",
            f"内角和 = {n - 2} × 180 = {ans}度",
        ]
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：{q}",
        f"{name}在练习册上看到一道题：{q}请你帮他算一算。",
    ])
    return ins, lines, ans


_reg("polygon_interior_angles", polygon_interior_angles)


# 55. 两数和s积p → 平方和
def sum_product_squares(rng):
    a = rng.randint(3, 20)
    b = rng.randint(3, 20)
    s = a + b
    p = a * b
    ans = s * s - 2 * p
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：甲、乙两个数的和是{s}，积是{p}。这两个数的平方和是多少？",
        f"{name}遇到一道思考题：两个数相加得{s}，相乘得{p}。请你帮他算出这两个数的平方和。",
    ])
    lines = [
        f"和的平方 = {s} × {s} = {s * s}",
        f"积的2倍 = {p} × 2 = {2 * p}",
        f"平方和 = {s * s} - {2 * p} = {ans}",
    ]
    return ins, lines, ans


_reg("sum_product_squares", sum_product_squares)


# 56. 两种套餐，通话t分钟 → 费用差
def phone_plans(rng):
    a = rng.randint(10, 30)
    b = rng.randint(50, 150)
    c = rng.randint(1, 3)
    d = rng.randint(10, 30)
    e = rng.randint(50, 150)
    for _ in range(50):
        if e != b:
            break
        e = rng.randint(50, 150)
    else:
        e = b + 10
    f = rng.randint(1, 3)
    t = rng.randint(max(b, e) + 10, max(b, e) + 100)
    A = a + (t - b) * c
    B = d + (t - e) * f
    diff = abs(A - B)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"通信公司有两种套餐：A套餐月租{a}元，含{b}分钟通话，超出每分钟{c}元；B套餐月租{d}元，含{e}分钟通话，超出每分钟{f}元。{name}每月通话{t}分钟，两种套餐的费用相差多少元？",
        f"手机套餐A：月租{a}元，免费通话{b}分钟，超出部分每分钟{c}元；套餐B：月租{d}元，免费通话{e}分钟，超出部分每分钟{f}元。{name}每月打{t}分钟电话，两种套餐费用相差多少元？",
    ])
    lines = [
        f"A套餐超出的分钟 = {t} - {b} = {t - b}分钟",
        f"A套餐超出的费用 = {t - b} × {c} = {(t - b) * c}元",
        f"A套餐总费用 = {a} + {(t - b) * c} = {A}元",
        f"B套餐超出的分钟 = {t} - {e} = {t - e}分钟",
        f"B套餐超出的费用 = {t - e} × {f} = {(t - e) * f}元",
        f"B套餐总费用 = {d} + {(t - e) * f} = {B}元",
        f"费用差 = {max(A, B)} - {min(A, B)} = {diff}元",
    ]
    return ins, lines, diff


_reg("phone_plans", phone_plans)


# 57. 甲b小时做a个，乙d小时做c个 → 每小时差几个
def efficiency_diff(rng):
    b = rng.randint(3, 8)
    d = rng.randint(3, 8)
    for _ in range(50):
        if d != b:
            break
        d = rng.randint(3, 8)
    else:
        d = b + 1
    m = rng.randint(2, 9)
    n = rng.randint(2, 9)
    a = b * m
    c = d * n
    r1 = Fraction(a, b)
    r2 = Fraction(c, d)
    diff = abs(r1 - r2)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"甲工人{b}小时做{a}个零件，乙工人{d}小时做{c}个零件。谁做得快？每小时多做多少个？",
        f"师傅{b}小时加工{a}个零件，徒弟{d}小时加工{c}个零件。{name}想知道谁的效率高，每小时多做多少个，请你帮他算一算。",
    ])
    lines = [
        f"甲每小时做的 = {a} ÷ {b} = {num(r1)}个/时",
        f"乙每小时做的 = {c} ÷ {d} = {num(r2)}个/时",
        f"每小时的差 = {num(max(r1, r2))} - {num(min(r1, r2))} = {num(diff)}个/时",
    ]
    return ins, lines, diff


_reg("efficiency_diff", efficiency_diff)


# 58. 打d折亏c元，打e折赚f元 → 成本价
def discount_loss_reverse(rng):
    d = rng.randint(5, 8)
    e = d + rng.randint(1, 10 - d)
    h = e - d
    c = h * rng.randint(2, 8)
    f = h * rng.randint(2, 8)
    price = 10 * (c + f) // h
    sell_d = price * d // 10
    cost = sell_d + c
    name = rng.choice(NAMES)
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}按定价打{d}折出售亏{c}元，打{e}折出售赚{f}元。这件{obj}的成本是多少元？",
        f"商店把一件{obj}打{d}折卖亏{c}元，打{e}折卖赚{f}元。{name}想知道这件{obj}的成本价，请你帮他算一算。",
    ])
    lines = [
        f"折扣差 = {e} - {d} = {h}",
        f"盈亏和 = {c} + {f} = {c + f}元",
        f"定价 = {c + f} × 10 ÷ {h} = {price}元",
        f"打{d}折的售价 = {price} × {d}/10 = {sell_d}元",
        f"成本价 = {sell_d} + {c} = {cost}元",
    ]
    return ins, lines, cost


_reg("discount_loss_reverse", discount_loss_reverse)


# 59. 去时v1回时v2，往返共用t分钟 → 距离
def round_trip_total_time(rng):
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    for _ in range(50):
        if b != a:
            break
        b = rng.randint(2, 6)
    else:
        b = a + 1
    m = rng.randint(2, 10)
    t = (a + b) * m
    v1, v2 = a, b
    dist = m * a * b
    name = rng.choice(NAMES)
    place = rng.choice(PLACE)
    ins = rng.choice([
        f"{name}从家出发去{place}，去时每分钟走{v1}米，返回时每分钟走{v2}米，往返一共用了{t}分钟。家到{place}有多少米？",
        f"小明从家到{place}，去时每分钟行{v1}米，原路返回时每分钟行{v2}米，往返共用{t}分钟。{name}想知道家到{place}的距离，请你帮他算一算。",
    ])
    lines = [
        f"速度积 = {v1} × {v2} = {v1 * v2}",
        f"速度和 = {v1} + {v2} = {v1 + v2}",
        f"家到{place}的距离 = {t} × {v1 * v2} ÷ {v1 + v2} = {dist}米",
    ]
    return ins, lines, dist


_reg("round_trip_total_time", round_trip_total_time)


# 60. 某数的一半减去c等于r → 某数
def rev_half_minus_c(rng):
    r = rng.randint(5, 40)
    c = rng.randint(3, r - 1)
    x = 2 * (r + c)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数的一半减去{c}，结果是{r}。这个数是多少？",
        f"{name}在练习册上看到一道题：某数的一半比{r}多{c}。请你帮他算出这个数。",
    ])
    lines = [
        f"这个数的一半 = {r} + {c} = {r + c}",
        f"这个数 = {r + c} × 2 = {x}",
    ]
    return ins, lines, x


_reg("rev_half_minus_c", rev_half_minus_c)


# 61. 某数除以a再减c等于b → 某数
def rev_div_minus_c(rng):
    a = rng.randint(2, 9)
    b = rng.randint(4, 25)
    c = rng.randint(2, b - 1)
    x = a * (b + c)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数除以{a}，再减去{c}，正好等于{b}。这个数是多少？",
        f"{name}在作业本上看到一道题：某数除以{a}的商比{b}多{c}。请你帮他算出这个数。",
    ])
    lines = [
        f"这个数除以{a}的商 = {b} + {c} = {b + c}",
        f"这个数 = {b + c} × {a} = {x}",
    ]
    return ins, lines, x


_reg("rev_div_minus_c", rev_div_minus_c)


# 62. 某数减去c后再乘a等于r → 某数
def rev_sub_then_mult(rng):
    a = rng.randint(3, 9)
    q = rng.randint(5, 25)
    c = rng.randint(3, 20)
    r = a * q
    x = q + c
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：一个数减去{c}后，再乘{a}，结果是{r}。这个数是多少？",
        f"{name}遇到一道思考题：某数减去{c}的差乘{a}，积是{r}。请你帮他算出这个数。",
    ])
    lines = [
        f"这个数减去{c}的差 = {r} ÷ {a} = {q}",
        f"这个数 = {q} + {c} = {x}",
    ]
    return ins, lines, x


_reg("rev_sub_then_mult", rev_sub_then_mult)


# 63. 4个数平均m，去掉一个后3个数平均n → 去掉的数
def avg_four_remove_one(rng):
    m = rng.randint(25, 60)
    n = rng.randint(20, m - 1)
    removed = 4 * m - 3 * n
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"数学课上老师出了一道题：4个数的平均数是{m}，去掉其中一个数后，剩下3个数的平均数是{n}。去掉的数是多少？",
        f"{name}在练习册上看到一道题：四个数的平均数是{m}，去掉一个数后，余下三个数的平均数是{n}。请你帮他算出去掉的数。",
    ])
    lines = [
        f"4个数的总和 = {m} × 4 = {4 * m}",
        f"剩下3个数的总和 = {n} × 3 = {3 * n}",
        f"去掉的数 = {4 * m} - {3 * n} = {removed}",
    ]
    return ins, lines, removed


_reg("avg_four_remove_one", avg_four_remove_one)


# 64. 果汁与水按a:b，现有果汁m克 → 加水多少克
def weighted_blend(rng):
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    for _ in range(50):
        if b != a:
            break
        b = rng.randint(2, 6)
    else:
        b = a + 1
    k = rng.randint(10, 60)
    m = a * k
    water = b * k
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一种饮料由果汁和水按{a}比{b}的比例配制。现有果汁{m}克，需要加水多少克？",
        f"配制一种饮料，果汁与水的比是{a}:{b}。{name}拿来{m}克果汁，需要加入多少克水？",
    ])
    lines = [
        f"每份的质量 = {m} ÷ {a} = {k}克",
        f"需要加水的质量 = {k} × {b} = {water}克",
    ]
    return ins, lines, water


_reg("weighted_blend", weighted_blend)


# 65. 绳子对折n次后每段a米 → 原长
def rope_folded(rng):
    n = rng.randint(2, 4)
    a = rng.randint(3, 30)
    layers = 2 ** n
    L = layers * a
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"一根绳子对折{n}次后，每段长{a}米。这根绳子原来长多少米？",
        f"{name}把一根绳子对折{n}次，量得每段长{a}米。这根绳子原来长多少米？",
    ])
    lines = []
    cur = 2
    for j in range(2, n + 1):
        nxt = cur * 2
        lines.append(f"第{j}次对折后的段数 = {cur} × 2 = {nxt}")
        cur = nxt
    lines.append(f"绳子原长 = {cur} × {a} = {L}米")
    return ins, lines, L


_reg("rope_folded", rope_folded)


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
    print(f"L4 ext3 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
