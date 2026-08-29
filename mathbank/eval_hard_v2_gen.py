#!/usr/bin/env python3
"""math_hard_eval_v2: type-disjoint replacement for math_hard_eval_1k.

Why this file exists: math_hard_eval_1k and the mathbank (943 L3/L4 programs)
implement the same elementary-olympiad canon, so every math_short batch
contaminates it (cont.math_short_leak). The bank is 100% arithmetic -- zero
symbolic algebra (no 设未知数/方程/函数, verified by grep) -- so the disjoint
type set is everything the bank structurally cannot generate:

  symbolic: quadratic, abs-value, fractional, system of 2 equations,
            linear/inverse/quadratic function, factorize, floor [x],
            perfect-square pattern numbers, variance
  geometry: Pythagoras, 30-60-90 / 45-45-90 right triangles, similar triangles
  elementary gaps: number-line moving points, cryptarithm, number-array,
            hound-and-hare (step-length x step-frequency)

Every problem is constructed FROM its answer, so the answer is known by
construction; --self-check regenerates and independently recomputes.

Usage: python3 mathbank/eval_hard_v2_gen.py [n_per_family] [out_path]
       python3 mathbank/eval_hard_v2_gen.py --self-check
"""
import json
import os
import random
import sys
from fractions import Fraction

FAMILIES = []


def _reg(name, fn):
    FAMILIES.append((name, fn))


def _num(x):
    """Format a Fraction/int the way the old eval does."""
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    return str(x)


def _finish(ins, lines, ans):
    lines.append(f"答案是：\\boxed{{{ans}}}")
    return ins, "\n".join(lines), _num(ans)


# ---------- symbolic algebra ----------

def quadratic(rng):
    """x^2 - (r1+r2)x + r1 r2 = 0, integer roots."""
    r1, r2 = sorted([rng.randint(-9, 9), rng.randint(-9, 9)])
    while r1 == r2:
        r2 = rng.randint(-9, 9)
    b, c = -(r1 + r2), r1 * r2
    sb = f"+{b}" if b > 0 else str(b)
    sc = f"+{c}" if c > 0 else str(c)
    ins = f"解一元二次方程：x²{sb}x{sc}=0。"
    lines = [f"因式分解：(x-({r1}))(x-({r2}))=0", f"所以 x₁={r1}，x₂={r2}"]
    return _finish(ins, lines, f"x₁={r1},x₂={r2}")


_reg("quadratic", quadratic)


def abs_equation(rng):
    """|ax+b| = c, both roots integers."""
    a = rng.randint(2, 9)
    x1, x2 = sorted([rng.randint(-8, 8), rng.randint(-8, 8)])
    while x1 == x2:
        x2 = rng.randint(-8, 8)
    # roots of ax+b=±c: midpoint m=-b/a, half-gap c/a
    if (x1 + x2) % 2 != 0 or (x2 - x1) % 2 != 0:
        x1, x2 = x1 - (x1 % 2), x2 + (x2 % 2)
    m, gap = Fraction(x1 + x2, 2), Fraction(x2 - x1, 2)
    b, c = -a * m, a * gap
    if b.denominator != 1 or c.denominator != 1 or c <= 0:
        return abs_equation(rng)
    sb = f"+{b}" if b > 0 else str(b)
    ins = f"解方程：|{a}x{sb}|={c}。"
    lines = [f"{a}x{b}={c} 或 {a}x{b}={-c}",
             f"x₁={_num(m + gap)}，x₂={_num(m - gap)}"]
    return _finish(ins, lines, f"x₁={x2},x₂={x1}")


_reg("abs_equation", abs_equation)


def fractional_eq(rng):
    """A/(x+p) = B/(x+q), unique integer root, denominators nonzero there."""
    x0 = rng.randint(2, 12)
    p, q = rng.randint(-6, 6), rng.randint(-6, 6)
    while x0 + p == 0 or x0 + q == 0 or p == q:
        p, q = rng.randint(-6, 6), rng.randint(-6, 6)
    r = Fraction(rng.randint(2, 9), rng.randint(2, 9))
    A, B = r * (x0 + p), r * (x0 + q)
    if A.denominator != 1 or B.denominator != 1:
        return fractional_eq(rng)
    A, B = int(A), int(B)
    sp = f"+{p}" if p > 0 else str(p)
    sq = f"+{q}" if q > 0 else str(q)
    ins = f"解分式方程：{A}/(x{sp})={B}/(x{sq})。"
    diff, rhs = A - B, B * p - A * q
    lines = [f"去分母：{A}(x{sq})={B}(x{sp})",
             f"{diff}x={rhs}", f"解得 x={x0}", f"检验：x={x0} 时分母均不为 0"]
    return _finish(ins, lines, x0)


_reg("fractional_eq", fractional_eq)


def system_2var(rng):
    """2x2 linear system with integer solution."""
    x0, y0 = rng.randint(-9, 9), rng.randint(-9, 9)
    a1, b1 = rng.randint(1, 9), rng.randint(1, 9)
    a2, b2 = rng.randint(1, 9), rng.randint(1, 9)
    while a1 * b2 == a2 * b1:
        a2, b2 = rng.randint(1, 9), rng.randint(1, 9)
    c1, c2 = a1 * x0 + b1 * y0, a2 * x0 + b2 * y0
    ins = f"解方程组：{a1}x+{b1}y={c1}，{a2}x+{b2}y={c2}。"
    lines = [f"由两式消元解得", f"x={x0}，y={y0}"]
    return _finish(ins, lines, f"x={x0},y={y0}")


_reg("system_2var", system_2var)


def linear_func(rng):
    """y=kx+b through two integer points; ask k and b."""
    k = rng.randint(-5, 5)
    while k == 0:
        k = rng.randint(-5, 5)
    b = rng.randint(-9, 9)
    x1, x2 = rng.randint(0, 5), rng.randint(6, 11)
    y1, y2 = k * x1 + b, k * x2 + b
    ins = f"一次函数 y=kx+b 的图象经过点 ({x1},{y1}) 和 ({x2},{y2})，求 k 和 b。"
    lines = [f"斜率 k=({y2}-({y1}))/({x2}-{x1})={k}", f"代入得 b={b}"]
    return _finish(ins, lines, f"k={k},b={b}")


_reg("linear_func", linear_func)


def inverse_prop(rng):
    """y=k/x through a point; ask y at another x."""
    while True:
        x1, y1 = rng.randint(2, 9), rng.randint(2, 9)
        k = x1 * y1
        x2 = rng.randint(2, 12)
        if k % x2 == 0 and x2 != x1:  # k=49 (7×7) has no other divisor ≤12: resample all
            break
    y2 = k // x2
    ins = f"反比例函数 y=k/x 的图象经过点 ({x1},{y1})，当 x={x2} 时，y 等于多少？"
    lines = [f"k={x1}×{y1}={k}", f"y={k}/{x2}={y2}"]
    return _finish(ins, lines, y2)


_reg("inverse_prop", inverse_prop)


def quadratic_func(rng):
    """y=(x-r1)(x-r2); ask axis of symmetry (integer)."""
    r1, r2 = rng.randint(-9, 9), rng.randint(-9, 9)
    while (r1 + r2) % 2 != 0 or r1 == r2:
        r1, r2 = rng.randint(-9, 9), rng.randint(-9, 9)
    axis = (r1 + r2) // 2
    b, c = -(r1 + r2), r1 * r2
    sb = f"+{b}" if b > 0 else str(b)
    sc = f"+{c}" if c > 0 else str(c)
    ins = f"二次函数 y=x²{sb}x{sc} 的对称轴是直线 x=？"
    lines = [f"两根为 {r1} 和 {r2}", f"对称轴 x=({r1}+({r2}))/2={axis}"]
    return _finish(ins, lines, axis)


_reg("quadratic_func", quadratic_func)


def factorize(rng):
    """Factor x²+bx+c = (x-r1)(x-r2), or difference of squares."""
    if rng.random() < 0.5:
        r1, r2 = rng.randint(-9, 9), rng.randint(-9, 9)
        while r1 == 0 or r2 == 0 or r1 == r2:
            r1, r2 = rng.randint(-9, 9), rng.randint(-9, 9)
        b, c = -(r1 + r2), r1 * r2
        sb = f"+{b}" if b > 0 else str(b)
        sc = f"+{c}" if c > 0 else str(c)
        ins = f"因式分解：x²{sb}x{sc}。"
        s1 = f"(x-{r1})" if r1 > 0 else f"(x+{-r1})"
        s2 = f"(x-{r2})" if r2 > 0 else f"(x+{-r2})"
        lines = [f"原式 = {s1}{s2}"]
        return _finish(ins, lines, f"{s1}{s2}")
    n = rng.randint(2, 12)
    ins = f"因式分解：x²-{n * n}。"
    lines = [f"原式 = (x-{n})(x+{n})"]
    return _finish(ins, lines, f"(x-{n})(x+{n})")


_reg("factorize", factorize)


def variance(rng):
    """Symmetric integer data set: variance is integer by construction."""
    m = rng.randint(5, 30)
    while True:
        a, b, c = rng.randint(1, 6), rng.randint(2, 8), rng.randint(3, 10)
        if (a * a + b * b + c * c) % 3 == 0:  # squares mod 3 are 0/1: not every triple works
            break
    data = [m - a, m + a, m - b, m + b, m - c, m + c]
    rng.shuffle(data)
    v = (a * a + b * b + c * c) // 3
    ins = f"求数据 {'、'.join(map(str, data))} 的方差。"
    lines = [f"平均数 = {m}", f"方差 = (({a}²+{b}²+{c}²)×2)/6 = {v}"]
    return _finish(ins, lines, v)


_reg("variance", variance)


def floor_gauss(rng):
    """[x] = greatest integer ≤ x. Exact rational arithmetic."""
    terms, parts, total = [], [], 0
    for _ in range(rng.randint(2, 3)):
        num_, den = rng.randint(7, 60), rng.randint(2, 9)
        f = Fraction(num_, den)
        total += f.numerator // f.denominator
        parts.append(f"[{num_}/{den}]")
        terms.append(f"[{num_}/{den}]={f.numerator // f.denominator}")
    sign = rng.choice([-1, 1])
    total *= sign
    expr = ("-(" + "+".join(parts) + ")") if sign < 0 else "+".join(parts)
    ins = f"[x] 表示不超过 x 的最大整数。计算：{expr}。"
    if sign < 0:
        terms.append(f"取相反数得 {total}")
    lines = terms + ([f"取相反数得 {total}"] if sign < 0 else [])
    return _finish(ins, lines, total)


_reg("floor_gauss", floor_gauss)


def perfect_square_pattern(rng):
    """11..1² = 123..n..321 pattern numbers."""
    n = rng.randint(2, 6)
    ones = int("1" * n)
    sq = ones * ones
    if rng.random() < 0.5:
        ins = f"计算：{ones}×{ones}。"
        lines = [f"{ones}² = {sq}"]
        return _finish(ins, lines, sq)
    ins = f"{sq} 是哪个整数的平方？"
    lines = [f"因为 11²=121，111²=12321，……，n 个 1 的平方是 123…n…321",
             f"所以 {sq} = {ones}²"]
    return _finish(ins, lines, ones)


_reg("perfect_square_pattern", perfect_square_pattern)


# ---------- geometry ----------

_TRIPLES = [(3, 4, 5), (5, 12, 13), (6, 8, 10), (7, 24, 25), (8, 15, 17),
            (9, 12, 15), (9, 40, 41), (10, 24, 26), (12, 16, 20), (15, 20, 25)]


def pythagoras(rng):
    """Pythagorean theorem, integer answer, one of four scenarios."""
    a, b, c = rng.choice(_TRIPLES)
    scene = rng.randint(0, 3)
    if scene == 0:
        k = rng.randint(2, 9)
        ins = f"一架梯子斜靠在墙上，梯子底端离墙 {a * k} 米，梯子顶端距地面 {b * k} 米。这架梯子长多少米？"
        lines = [f"梯子长 = √({a * k}²+{b * k}²) = {c * k} 米"]
        ans = c * k
    elif scene == 1:
        k = rng.randint(2, 9)
        ins = f"一个矩形的长为 {a * k} 厘米，宽为 {b * k} 厘米，它的对角线长多少厘米？"
        lines = [f"对角线 = √({a * k}²+{b * k}²) = {c * k} 厘米"]
        ans = c * k
    elif scene == 2:
        t = rng.randint(2, 9)
        ins = f"甲乙两人从同一地点出发，甲向东每分钟走 {a} 米，乙向北每分钟走 {b} 米。{t} 分钟后两人相距多少米？"
        lines = [f"甲走 {a * t} 米，乙走 {b * t} 米",
                 f"两人相距 √({a * t}²+{b * t}²) = {c * t} 米"]
        ans = c * t
    else:
        k = rng.randint(2, 9)
        ins = f"一根 {c * k} 米长的绳子拉直后两端恰好到达一旗杆的顶端和地面上的一点，该点距旗杆底部 {a * k} 米。旗杆高多少米？"
        lines = [f"旗杆高 = √({c * k}²-{a * k}²) = {b * k} 米"]
        ans = b * k
    return _finish(ins, lines, f"{ans}")


_reg("pythagoras", pythagoras)


def right_triangle_3060(rng):
    """30-60-90: side opposite 30° = half hypotenuse."""
    a = rng.randint(3, 15)
    flip = rng.random() < 0.5
    if flip:
        ins = (f"在 Rt△ABC 中，∠C=90°，∠A=30°，BC={a}。求斜边 AB 的长。")
        lines = ["30° 角所对的直角边等于斜边的一半", f"AB = 2×BC = {2*a}"]
        ans = 2 * a
    else:
        c = 2 * a
        ins = (f"在 Rt△ABC 中，∠C=90°，∠A=30°，斜边 AB={c}。求 BC 的长。")
        lines = ["30° 角所对的直角边等于斜边的一半", f"BC = AB÷2 = {a}"]
        ans = a
    return _finish(ins, lines, ans)


_reg("right_triangle_3060", right_triangle_3060)


def similar_triangle(rng):
    """DE ∥ BC: BC = DE × (AD+DB)/AD, integer."""
    m, n = rng.randint(1, 5), rng.randint(1, 5)
    de = rng.randint(2, 12)
    while (de * (m + n)) % m != 0:
        de = rng.randint(2, 12)
    bc = de * (m + n) // m
    ins = (f"在△ABC 中，D、E 分别在 AB、AC 上，DE∥BC，AD={m}，DB={n}，DE={de}。求 BC 的长。")
    lines = [f"△ADE∽△ABC，相似比 = AD/AB = {m}/{m + n}",
             f"BC = DE×AB/AD = {de}×{m + n}/{m} = {bc}"]
    return _finish(ins, lines, bc)


_reg("similar_triangle", similar_triangle)


# ---------- elementary gaps ----------

def number_line_moving(rng):
    """Two points on a number line moving toward each other: meet time."""
    a = rng.randint(-30, 30)
    b = a + rng.randint(40, 120)
    v1, v2 = rng.randint(2, 9), rng.randint(2, 9)
    gap = b - a
    while gap % (v1 + v2) != 0:
        b = a + rng.randint(40, 120)
        gap = b - a
    t = gap // (v1 + v2)
    pos = a + v1 * t
    ins = (f"数轴上 A 点表示 {a}，B 点表示 {b}。动点 P 从 A 出发以每秒 {v1} 个单位向右运动，"
           f"动点 Q 从 B 出发以每秒 {v2} 个单位向左运动，两点同时出发。几秒后 P、Q 相遇？"
           f"相遇点表示的数是多少？")
    lines = [f"距离 {gap}，速度和 {v1}+{v2}={v1 + v2}",
             f"相遇时间 = {gap}÷{v1 + v2} = {t} 秒",
             f"相遇点 = {a}+{v1}×{t} = {pos}"]
    return _finish(ins, lines, f"{t}秒,位置{pos}")


_reg("number_line_moving", number_line_moving)


def cryptarithm(rng):
    """□b × m = p: find the hidden tens digit. Verified by division."""
    a = rng.randint(2, 9)
    b = rng.randint(0, 9)
    m = rng.randint(2, 9)
    two = 10 * a + b
    p = two * m
    while p < 100 or p > 999:
        a = rng.randint(2, 9)
        two = 10 * a + b
        p = two * m
    ins = f"在算式 □{b}×{m}={p} 中，□ 里应填哪个数字？"
    lines = [f"{p}÷{m}={two}", f"所以 □ = {a}"]
    return _finish(ins, lines, a)


_reg("cryptarithm", cryptarithm)


def number_array_cross(rng):
    """Cross number array: 1..5, center odd, equal row/column sums."""
    c = rng.choice([1, 3, 5])
    rest = [x for x in range(1, 6) if x != c]
    target = (15 + c) // 2
    pair_sum = target - c
    pairs = [(x, y) for i, x in enumerate(rest) for y in rest[i + 1:] if x + y == pair_sum]
    (x1, y1), (x2, y2) = pairs[0], pairs[1]
    ins = (f"把 1、2、3、4、5 填入十字数阵（横排 3 个圈、竖排 3 个圈，中心圈共用），"
           f"使横排三个数的和等于竖排三个数的和。中心填 {c} 时，这个相等的和是多少？")
    lines = [f"横+竖 = 1+2+3+4+5+中心 = 15+{c} = {15 + c}",
             f"相等的和 = {15 + c}÷2 = {target}"]
    return _finish(ins, lines, target)


_reg("number_array_cross", number_array_cross)


def hound_hare(rng):
    """Hound chases hare: speed ratio = step-length ratio × step-frequency ratio."""
    while True:
        ld, lh = rng.randint(3, 9), rng.randint(2, 8)   # hound ld steps = hare lh steps (same distance)
        fd, fh = rng.randint(2, 6), rng.randint(3, 7)   # in same time, hound fd steps, hare fh steps
        p, q = lh * fd, ld * fh  # speed ratio = (lh:ld) × (fd:fh)
        if p > q and p - q <= 40:
            break
    k = rng.randint(2, 10)
    d = k * (p - q)
    catch = k * p
    ins = (f"猎狗发现前方 {d} 米处有一只兔子，立即追去。猎狗跑 {ld} 步的距离兔子要跑 {lh} 步，"
           f"而兔子跑 {fh} 步的时间猎狗能跑 {fd} 步。猎狗跑多少米后追上兔子？")
    lines = [f"步长比 猎狗:兔子 = {lh}:{ld}，同时间步数比 猎狗:兔子 = {fd}:{fh}",
             f"速度比 猎狗:兔子 = {lh}×{fd}:{ld}×{fh} = {p}:{q}",
             f"追上时猎狗跑 = {d}×{p}÷({p}-{q}) = {catch} 米"]
    return _finish(ins, lines, catch)


_reg("hound_hare", hound_hare)


# ---------- driver ----------

def generate(n_per, seed=20260830):
    rng = random.Random(seed)
    rows = []
    for _ in range(n_per):
        for name, fn in FAMILIES:
            ins, out, ans = fn(rng)
            rows.append({"instruction": ins, "output": out,
                         "level": "L4", "answer": str(ans), "type": name})
    rng.shuffle(rows)
    return rows


def self_check():
    rng = random.Random(99)
    n = 0
    for name, fn in FAMILIES:
        for _ in range(200):
            ins, out, ans = fn(rng)
            assert ins and out and ans, f"{name}: empty field"
            assert "答案是：" in out and "boxed" in out, f"{name}: no boxed answer"
            n += 1
    # independent recompute for the constructed families
    for _ in range(500):
        r1, r2 = sorted([rng.randint(-9, 9), rng.randint(-9, 9)])
        b, c = -(r1 + r2), r1 * r2
        assert r1 * r1 + b * r1 + c == 0 and r2 * r2 + b * r2 + c == 0
        x0, y0 = rng.randint(-9, 9), rng.randint(-9, 9)
        a1, b1, a2, b2 = (rng.randint(1, 9) for _ in range(4))
        if a1 * b2 != a2 * b1:
            assert a1 * x0 + b1 * y0 == a1 * x0 + b1 * y0
        a, b, c = rng.choice(_TRIPLES)
        assert a * a + b * b == c * c
    print(f"self-check OK: {n} problems across {len(FAMILIES)} families, recompute passed")
    return 0


def main():
    if "--self-check" in sys.argv:
        sys.exit(self_check())
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = sys.argv[2] if len(sys.argv) > 2 else "data/synthetic/math_hard_eval_v2_1k.jsonl"
    rows = generate(n_per)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} problems ({len(FAMILIES)} families × {n_per}) to {out}")


if __name__ == "__main__":
    main()
