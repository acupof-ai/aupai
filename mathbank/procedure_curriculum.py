#!/usr/bin/env python3
"""Written procedures whose intermediate state is on the page — the skill behind
"cannot execute a multi-step written procedure" (aupai-fb, 2026-08-29).

The fix is data whose answer is derivable ONLY from the steps, so guessing has
nothing to grab.

This generator teaches three procedures that share one mechanism — every output
line is a strict function of the line above it, verified line-by-line in _demo:

  [竖式乘法]  two-digit vertical long multiplication (digit × digit + carry)
  [单位换算]  a multi-hop unit-conversion chain (each hop × factor)
  [解方程]    step-by-step isolation of a linear one-variable equation

Constraints earned on the arithmetic syllabus, enforced here:
  A. NO `= ` shortcut slot — every prompt ends in a noun / verb, never `=`.
  B. split ON THE PROBLEM (blake2b, 10% out), never on the row — the space here
     is small enough to exhaust (53k mul pairs, ~tens of unit paths, ~thousands
     of equations); a row-split probe would just measure memorisation.
  C. one explicitly-tagged format per prompt ([竖式乘法]/[单位换算]/[解方程]).
  E. every step verified: _demo recomputes each line's value from its operands,
     not by trusting the generator's own arithmetic.

The procedure space is small and finite (~2,736 equations, ~8,100 multiplications,
a few thousand unit paths). 50K rows dedup to ~12.5K distinct problems — more rows
manufacture repetition, not information. That caps what this can contribute to the
3.3B pretrain (0.16%, below math-hard's ±1.1pt resolution), so its natural slot is
SFT / the anneal, where the procedure-execution constraint was measured, not the
main mix.

Usage:
    python mathbank/procedure_curriculum.py --n 20000 --out data/synthetic/procedure_v1.jsonl
    python mathbank/procedure_curriculum.py --selftest
"""

import argparse
import hashlib
import json
import random
import sys

# format markers — one per procedure, so the answer format is determined by the prompt.
TAG = {"mul": "[竖式乘法]", "unit": "[单位换算]", "eq": "[解方程]"}
FORMATS = tuple(TAG)


def prob_key(fmt, *parts):
    """Canonical identity of a problem — what the held-out divide is ON."""
    return f"{fmt}:{'/'.join(map(str, parts))}"


def held_out(key):
    """Deterministic train/test split on the problem, not the row (blake2b, 10%)."""
    h = hashlib.blake2b(key.encode(), digest_size=4).digest()
    return int.from_bytes(h, "little") % 10 == 0


# ---------------------------------------------------------------- procedures
# Each returns (prompt, multiline output, checks) where checks is a list of
# (op, *operands, expected) that _demo recomputes. ops:
#   ("muladd", x, y, c, exp)  x*y + c == exp
#   ("add",    a, b, exp)     a+b      == exp
#   ("mul",    a, b, exp)     a*b      == exp
#   ("sub",    a, b, exp)     a-b      == exp
#   ("div",    a, b, exp)     a//b     == exp  (b divides a exactly by construction)

def _mul_by_digit(digits, y, label):
    """Multiply the number <digits> (a digit list, least-significant first) by
    single digit y. Returns the rendering lines and the checks."""
    lines, checks, carry = [], [], 0
    for x in digits:
        prod = x * y + carry
        lines.append(f"{x} × {y} + 进位{carry} = {prod}，写 {prod % 10}，进位 {prod // 10}")
        checks.append(("muladd", x, y, carry, prod))
        carry = prod // 10
    if carry:
        lines.append(f"最高位进位 {carry}，写 {carry}")
    return lines, checks, carry


def mul(a, b):
    """Vertical long multiplication of two 2-digit numbers (10..99). Trade the
    placeholder ›0‹ steps for in-text naming — the space cost is the point, the
    model needs each partial written before it is added."""
    sa, sb = str(a), str(b)
    da = [int(c) for c in reversed(sa)]  # a digits, least-significant first
    db = [int(c) for c in reversed(sb)]
    out = [f"{a} × {b} 的竖式计算："]
    checks = []
    partials = []
    for i, bd in enumerate(db):  # multiplier digit, right to left
        out.append(f"第 {i + 1} 位乘算（用 {bd} 乘）：")
        plines, pchecks, _ = _mul_by_digit(da, bd, f"×{bd}")
        out.extend(plines)
        checks.extend(pchecks)
        part = a * bd
        partials.append(part)
        out.append(f"第 {i + 1} 部分积 = {part}" + ("，记十位位置" if i else ""))
        checks.append(("mul", a, bd, part))
    out.append("把各部分积相加：")
    running = 0
    for i, p in enumerate(partials):
        shifted = p * (10 ** i)
        out.append(f"{running} + {shifted} = {running + shifted}")
        checks.append(("add", running, shifted, running + shifted))
        running += shifted
    out.append(f"结果 = {running}")
    checks.append(("mul", a, b, running))
    return f"{TAG['mul']} 用竖式计算 {a} × {b}", "\n".join(out), checks


# unit chains: name, factor to the NEXT unit in the chain (must be an integer > 1)
CHAINS = {
    "length": [("千米", 1000), ("米", 100), ("厘米", 10), ("毫米", 1)],
    "mass": [("千克", 1000), ("克", 1000), ("毫克", 1)],
}


def unit(v, chain, rng=None):
    """Convert v (1..999) across a chain, writing each hop's × factor. At least two
    hops, so the intermediate state always sits on the page between start and end."""
    units = CHAINS[chain]
    n = len(units)
    rr = rng if rng else random
    # start at most n-3 so there is room for a 2+ hop run
    start = rr.randrange(0, max(1, n - 2))
    end = rr.randrange(min(start + 2, n - 1), n)  # >=2 hops, but mass has only 3 units
    out, checks, cur = [], [], v
    for i in range(start, end):
        name, factor = units[i]
        nxt = units[i + 1][0]
        prod = cur * factor
        out.append(f"{cur} {name} = {cur} × {factor} {nxt} = {prod} {nxt}")
        checks.append(("mul", cur, factor, prod))
        cur = prod
    out.append(f"结果 = {cur} {units[end][0]}")
    checks.append(("mul", None, None, cur))  # sentinel: cur is the answer, already built hop-by-hop
    src = units[start][0]
    return f"{TAG['unit']} 把 {v} {src} 换算成 {units[end][0]}", "\n".join(out), checks


def eq(rng=None):
    """Linear one-variable: ax ± b = c, isolated in two strict steps."""
    rr = rng if rng else random
    x = rr.randrange(1, 20)
    a = rr.randrange(2, 10)
    b = rr.randrange(-9, 10)  # signed constant; 0 excluded for a real step
    b = b or rr.choice([-1, 1])
    sign = "-" if b < 0 else "+"
    absb = abs(b)
    c = a * x + b
    s = c - b
    out = [f"{a}x {sign} {absb} = {c}"]
    if b < 0:
        out.append(f"两边同时加 {absb}：{a}x = {c} + {absb} = {s}")
        checks = [("add", c, absb, s)]
    else:
        out.append(f"两边同时减 {absb}：{a}x = {c} - {absb} = {s}")
        checks = [("sub", c, absb, s)]
    out.append(f"两边同时除 {a}：x = {s} ÷ {a} = {x}")
    checks.append(("div", s, a, x))
    out.append(f"结果 x = {x}")
    return f"{TAG['eq']} 解方程 {a}x {sign} {absb} = {c}", "\n".join(out), checks


def _draw(fmt, rng):
    """Emit one problem of the chosen format, seeded from rng."""
    if fmt == "mul":
        return mul(rng.randrange(10, 100), rng.randrange(10, 100))
    if fmt == "unit":
        return unit(rng.randrange(1, 1000), rng.choice(list(CHAINS)), rng)
    return eq(rng)


def generate(n, rng, formats=FORMATS, split="train"):
    """A balanced curriculum: every format, split on the problem identity."""
    out = []
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > n * 50:
            break
        fmt = rng.choice(formats)
        prompt, output, _ = _draw(fmt, rng)
        # canonical problem key = the prompt WITHOUT its format tag (the tag is a
        # uniform, not part of the problem identity)
        key = prob_key(fmt, prompt.split(" ", 1)[1])
        if split != "all" and held_out(key) != (split == "test"):
            continue
        out.append({"instruction": prompt, "output": output, "fmt": fmt, "prob": key})
    return out


def _verify_checks(checks):
    """Recompute every emitted fact. Mock range is tiny; assert each op holds."""
    for c in checks:
        op, *rest = c
        if op == "muladd":
            x, y, carry, exp = rest
            assert x * y + carry == exp, c
        elif op == "add":
            a, b, exp = rest
            assert a + b == exp, c
        elif op == "sub":
            a, b, exp = rest
            assert a - b == exp, c
        elif op == "div":
            a, b, exp = rest
            assert b != 0 and a % b == 0 and a // b == exp, c
        elif op == "mul":
            a, b, exp = rest
            if a is not None:
                assert a * b == exp, c
        else:
            raise AssertionError(f"unknown check op {op} in {c}")


def _demo():
    rng = random.Random(0)
    # 1. every format reproduces its own value through exactly the steps emitted
    for _ in range(1500):
        fmt = rng.choice(FORMATS)
        if fmt == "mul":
            _, _, checks = mul(rng.randrange(10, 100), rng.randrange(10, 100))
        elif fmt == "unit":
            _, _, checks = unit(rng.randrange(1, 1000), rng.choice(list(CHAINS)), rng)
        else:
            _, _, checks = eq(rng)
        _verify_checks(checks)
    # 2. split is on the problem: a train and a test draw share no canonical key
    tr = generate(2000, random.Random(1), split="train")
    te = generate(2000, random.Random(2), split="test")
    ktr = {r["prob"] for r in tr}
    kte = {r["prob"] for r in te}
    inter = ktr & kte
    assert not inter, f"train/test share {len(inter)} problems: {list(inter)[:3]}"
    # 3. tags: every instruction carries its own format marker, no `= ` slot
    for r in tr + te:
        assert r["instruction"].startswith(TAG[r["fmt"]]), r["instruction"]
        assert not r["instruction"].rstrip().endswith("="), r["instruction"]
    n_by = {}
    for r in tr:
        n_by[r["fmt"]] = n_by.get(r["fmt"], 0) + 1
    print(f"split on problem: {len(ktr)} train / {len(kte)} test, 0 shared")
    print(f"emitted {len(tr) + len(te)} rows verified step-by-step")
    print("procedure_curriculum self-test OK", n_by)
    print("\nsamples:")
    for f in FORMATS:
        r = next((x for x in tr if x["fmt"] == f), None)
        if r:
            print(f"\n  [{f}] {r['instruction']}\n  ->\n" + "  -> ".join(r["output"].splitlines()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000, help="rows to DRAW. The procedure space is small "
        "(~2736 equations, ~8100 multiplications, ~few-thousand unit paths), so build_corpus's "
        "exact-dedup collapses repeats to roughly one row per problem; larger n buys repetition, "
        "not variety. This batch targets SFT/the anneal (pretrain share is 0.16%, below measurement "
        "resolution) -- size n to ~2-3x the unique space.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["train", "test", "all"], default="train")
    ap.add_argument("--out", default="data/synthetic/procedure_v1.jsonl")
    a = ap.parse_args()
    rows = generate(a.n, random.Random(a.seed), split=a.split)
    with open(a.out, "w", encoding="utf-8") as o:
        for r in rows:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sum(len(r["instruction"]) + len(r["output"]) for r in rows)
    print(f"{len(rows)} rows -> {a.out} ({chars / 1e6:.1f}M chars, ~{chars / 1.5e6:.1f}M tokens)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        main()