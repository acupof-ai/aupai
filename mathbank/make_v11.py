#!/usr/bin/env python3
"""Weight-driven math batch v11.

aupai-fb's ask: an 80K-row L3/L4 batch at cap 150, matched to the eval's
operator mix by *weighting program draws*, never deleting rows; decimal taken
at its reachable ceiling. The LP (measured 2026-08-29) showed the 4 operator
axes are each reachable via weights, while decimal caps at ~11.3% (bank has
only 61/943 dec-heavy programs). This scripts the LP -> integer instance
counts -> sample -> verify -> dist_check loop.

Usage:
  python3 mathbank/make_v11.py
"""
import collections, json, os, random, re, sys
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_math_short import load_programs, verify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "synthetic", "math_short_v11.jsonl")
CAP, N, SEED = 150, 80000, 20260829
EVAL_OP = {"plus": .352, "times": .283, "div": .145, "minus": .220}
OPS = re.compile(r"[+\-×÷*]"); DEC = re.compile(r"\d+\.\d+")


def measure(fn, k=10, seed=5):
    """Dist metrics for one program: mean per-op counts and decimal rate."""
    rng = random.Random(seed)
    acc = collections.Counter(); t = 0
    for _ in range(k):
        try:
            ins, lines, ans = fn(rng); s = "\n".join(lines)
            c = collections.Counter(OPS.findall(s)); tt = sum(c.values())
            if not tt:
                continue
            acc["plus"] += c.get("+", 0); acc["times"] += c.get("×", 0) + c.get("*", 0)
            acc["div"] += c.get("÷", 0) + c.get("/", 0); acc["minus"] += c.get("-", 0)
            acc["ops"] += tt; acc["dec"] += 1.0 if DEC.search(s) else 0.0
            t += 1
        except Exception:
            pass
    if not t:
        return None
    return {"plus": acc["plus"] / t, "times": acc["times"] / t, "div": acc["div"] / t,
            "minus": acc["minus"] / t, "ops": acc["ops"] / t, "dec": acc["dec"] / t}


def lp_weights(metrics):
    n = len(metrics)
    if n == 0:
        return None
    # each metric field is a PER-ROW MEAN COUNT (handed down from measure()).
    # overall per-op share of axis a = sum_i x_i·a_i / sum_i x_i·ops_i. For target t the
    # in-band condition is linear: sum_i x_i (a_i - lo·ops_i) >= 0 and (a_i - hi·ops_i) <= 0.
    ops = np.array([m["ops"] for m in metrics])
    plus = np.array([m["plus"] for m in metrics]); times = np.array([m["times"] for m in metrics])
    div = np.array([m["div"] for m in metrics]); minus = np.array([m["minus"] for m in metrics])
    dec = np.array([m["dec"] for m in metrics])
    ab, bh = [], []
    for ax, v in (("plus", plus), ("times", times), ("div", div), ("minus", minus)):
        lo, hi = EVAL_OP[ax] * .7, EVAL_OP[ax] * 1.3
        ab.append(-(v - lo * ops)); bh.append(0.0)   # sum x(v - lo·ops) >= 0
        ab.append(v - hi * ops); bh.append(0.0)      # sum x(v - hi·ops) <= 0
    r = linprog(-dec, A_ub=np.array(ab), b_ub=np.array(bh),
                A_eq=np.ones((1, n)), b_eq=np.array([float(N)]),
                bounds=[(0, CAP)] * n, method="highs")
    if not r.success:
        print("LP infeasible on operator axes"); return None
    x = np.clip(np.round(r.x), 0, CAP).astype(int)
    diff = N - int(x.sum())
    order = np.argsort(-x)
    if diff > 0:
        for i in order:
            if diff <= 0: break
            add = min(CAP - x[i], diff); x[i] += add; diff -= add
    else:
        for i in order[::-1]:
            if diff >= 0: break
            rem = min(x[i], -diff); x[i] -= rem; diff += rem
    return x


def main():
    progs = load_programs()
    progs34 = [(lev, name, fn) for lev in ("L3", "L4") for name, fn in progs[lev]]
    print(f"L3/L4 programs: {len(progs34)}")
    metrics = [measure(fn) for _, _, fn in progs34]
    # drop programs that measure failed; pair back with their (lev,name,fn)
    keep = [(a[0], a[1], a[2], m) for a, m in zip(progs34, metrics) if m]
    x = lp_weights([k[3] for k in keep])
    if x is None:
        x = np.array([N // len(keep)] * len(keep))
    rng = random.Random(SEED)
    seen = set(); n_out = 0; dec_rows = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for i, (lev, name, fn, m) in enumerate(keep):
            want = int(x[i]); got = 0
            for dr in range(want * 3):
                if got >= want:
                    break
                srng = random.Random(f"v11-{name}-{dr}")
                try:
                    ins, lines, ans = fn(srng)
                except Exception:
                    continue
                out, ok = verify(ins, lines, ans)
                if not ok or ins in seen:
                    continue
                seen.add(ins)
                f.write(json.dumps({"instruction": ins, "output": out, "level": lev},
                                   ensure_ascii=False) + "\n")
                if DEC.search(out): dec_rows += 1
                got += 1; n_out += 1
    print(f"wrote {n_out} rows -> {OUT} (decimal-row {dec_rows / max(1, n_out):.1%})")
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    from dist_check import main as dc
    sys.argv = ["dist_check", OUT]
    try:
        dc()
    except SystemExit:
        pass


if __name__ == "__main__":
    main()