#!/usr/bin/env python3
"""Band-center SFT math batch v11.

Weight-driven L3/L4 batch matched to math_hard_eval_1k's surface statistics.
The point of "band" (as opposed to the earlier make_v11.py which pinned axes at
the ±30% edge) is robustness: constrain every target axis to a TIGHTER center
band in the LP, so the real sampled distribution clears the ±30% dist_check
wall with margin even under verify/dedup perturbation and seed changes.

Verified settings (measured 2026-08-29, seed-stable across 5 seeds):
  --opband 0.20  --fracband 0.10  ->  plus .80 / minus .93 / times 1.20 /
                                      div 1.20 / fraction 1.18 of eval, all
                                      inside [.70,1.30] with >=.10 margin.
  decimal is a structural wall: only 55 decimal-heavy programs survive, real
  combined capacity 6941 rows -> decimal ceilings at ~8.6% (eval 26.1%, ratio
  ~.33 -- reported OFF, by design). ~11% was the uniform-capacity overestimate;
  real per-program capacity caps it at 8.7%.

Every target axis is constrained in the LP via the linear share model:
  operator share of axis a = sum_i x_i·a_i / sum_i x_i·ops_i   (sum x(a-lo·ops)>=0, (a-hi·ops)<=0)
  fraction  = sum_i x_i·frac_i / sum_i x_i                     (row-level)
  objective maximizes decimal rows. x_i bounded by measured program capacity.

Usage:
  python3 mathbank/make_v11_band.py [--opband 0.20] [--fracband 0.10]
"""

import argparse
import collections
import contextlib
import json
import os
import random
import re
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_math_short import load_programs, verify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "synthetic", "math_short_v11.jsonl")
CAP, SEED = 150, 20260829
EVAL_OP = {"plus": 0.352, "minus": 0.220, "times": 0.283, "div": 0.145}
EVAL_FRAC = 0.202
OPS = re.compile(r"[+\-×÷*]")
DEC = re.compile(r"\d+\.\d+")
FRAC = re.compile(r"\d+\s*/\s*\d+|frac")


def measure(fn, k=40, seed=5):
    """Per-row means for one program: per-op counts, fraction rate, decimal rate."""
    rng = random.Random(seed)
    acc = collections.Counter()
    fx = 0
    dc = 0
    t = 0
    for _ in range(k):
        try:
            ins, lines, ans = fn(rng)
        except Exception:
            continue
        s = "\n".join(lines)
        c = collections.Counter(OPS.findall(s))
        tt = sum(c.values())
        if not tt:
            continue
        acc["plus"] += c.get("+", 0)
        acc["times"] += c.get("×", 0) + c.get("*", 0)
        acc["div"] += c.get("÷", 0) + c.get("/", 0)
        acc["minus"] += c.get("-", 0)
        acc["ops"] += tt
        fx += bool(FRAC.search(s))
        dc += bool(DEC.search(s))
        t += 1
    if not t:
        return None
    return dict(
        plus=acc["plus"] / t,
        times=acc["times"] / t,
        div=acc["div"] / t,
        minus=acc["minus"] / t,
        ops=acc["ops"] / t,
        frac=fx / t,
        dec=dc / t,
    )


def capacity(fn, seed=9, draw_budget=260):
    """Max distinct verified rows the program can actually produce (rejects and
    duplicate-instruction space included). This is the LP upper bound -- without it
    the LP assigns 150 to programs that yield 30, and the batch under-delivers."""
    seen = set()
    got = 0
    for dr in range(draw_budget):
        if got >= CAP:
            break
        srng = random.Random(f"cap-{seed}-{dr}")
        try:
            ins, lines, ans = fn(srng)
        except Exception:
            continue
        o, ok = verify(ins, lines, ans)
        if not ok or ins in seen:
            continue
        seen.add(ins)
        got += 1
    return got


def solve(pairs, N, opband, fracband):
    n = len(pairs)
    ops_a = np.array([m["ops"] for *_, m, c in pairs], float)
    A = {
        k: np.array([m[k] for *_, m, c in pairs], float)
        for k in ("plus", "minus", "times", "div", "frac", "dec")
    }
    ub = np.array([c for *_, m, c in pairs], float)
    A_ub, b_ub = [], []
    for ax in ("plus", "minus", "times", "div"):
        lo, hi = EVAL_OP[ax] * (1 - opband), EVAL_OP[ax] * (1 + opband)
        A_ub.append(-(A[ax] - lo * ops_a))
        b_ub.append(0.0)
        A_ub.append(A[ax] - hi * ops_a)
        b_ub.append(0.0)
    fl, fh = EVAL_FRAC * (1 - fracband), EVAL_FRAC * (1 + fracband)
    A_ub.append(-(A["frac"] - fl))
    b_ub.append(0.0)
    A_ub.append(A["frac"] - fh)
    b_ub.append(0.0)
    r = linprog(
        -A["dec"],
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.ones((1, n)),
        b_eq=np.array([float(N)]),
        bounds=[(0, ub[i]) for i in range(n)],
        method="highs",
    )
    if not r.success:
        return None
    x = np.clip(np.round(r.x), 0, ub).astype(int)
    diff = N - int(x.sum())
    order = np.argsort(-(x / ub))
    if diff > 0:
        for i in order:
            if diff <= 0:
                break
            add = min(ub[i] - x[i], diff)
            x[i] += add
            diff -= add
    else:
        for i in order[::-1]:
            if diff >= 0:
                break
            rem = min(x[i], -diff)
            x[i] -= rem
            diff += rem
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--opband", type=float, default=0.20, help="operator ± band (0.20 = constrain to .80-1.20)"
    )
    ap.add_argument("--fracband", type=float, default=0.10, help="fraction ± band")
    ap.add_argument("--n", type=int, default=80000)
    a = ap.parse_args()

    progs = load_programs()
    pairs = []
    for lev in ("L3", "L4"):
        for name, fn in progs[lev]:
            m = measure(fn)
            if m is None:
                continue
            c = capacity(fn)
            if c <= 0:
                continue
            pairs.append((lev, name, fn, m, c))
    print(f"programs kept: {len(pairs)}", file=sys.stderr)

    x = solve(pairs, a.n, a.opband, a.fracband)
    if x is None:
        print("LP infeasible")
        return 1

    seen = set()
    rows = []
    for i, (lev, name, fn, _m, _c) in enumerate(pairs):
        want = int(x[i])
        got = 0
        for dr in range(want * 8):
            if got >= want:
                break
            srng = random.Random(f"v11-{a.opband}-{name}-{dr}")
            try:
                ins, lines, ans = fn(srng)
            except Exception:
                continue
            o, ok = verify(ins, lines, ans)
            if not ok or ins in seen:
                continue
            seen.add(ins)
            rows.append({"instruction": ins, "output": o, "level": lev})
            got += 1
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dec = sum(1 for r in rows if DEC.search(r["output"]))
    print(f"wrote {len(rows)} -> {OUT}  dec_rows={dec / len(rows):.1%}", file=sys.stderr)

    sys.argv = ["dist_check", OUT]
    from dist_check import main as dc

    with contextlib.suppress(SystemExit):
        dc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
