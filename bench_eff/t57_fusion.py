#!/usr/bin/env python3
"""Steady-state fusion analysis for t57: which fused kernels are worth hand-writing in TileLang.

    python3 bench_eff/t57_fusion.py <trace.json> <n_active_steps>

Three deliverables fb asked for, from one trace:
  1. steady idle ms/step, measured in ONE time base (settles 95 vs 187)
  2. per-kernel P50/P95 on the top fusions -- a mean hides whether a group is a few bad
     kernels or many mediocre ones, which is the whole TileLang-vs-leave question
  3. memory-vs-compute classification per fusion, so a kernel already near its roof is LEFT

The classification is a bound, not a measurement: inductor fusion names list the ops absorbed
but not the shapes, so bytes moved cannot be read off the trace. What IS available per kernel is
grid x block (threads), registers/thread and shared memory, which bound occupancy and say whether
a kernel is latency-limited. A kernel at high occupancy and high duration is doing real work and
is a poor TileLang target; a long kernel at low occupancy is the opposite.
"""

import json
import statistics as st
import sys
from collections import defaultdict

# H20
PEAK_BF16_TFLOPS = 148.0
HBM_TB_S = 4.0
BALANCE_FLOP_PER_BYTE = PEAK_BF16_TFLOPS / HBM_TB_S  # 37


def load(path):
    with open(path) as f:
        return json.load(f)["traceEvents"]


def gpu_kernels(ev):
    return [e for e in ev if e.get("cat") == "kernel" and "dur" in e]


def idle_one_base(ev, n):
    """Kernel busy, wall span and idle for the compute stream, all from the same window.

    The 95-vs-187 dispute came from dividing a warmup window's kernel time by a steady step
    derived from throughput. Both numbers here come from this trace's own timestamps.
    """
    k = [e for e in gpu_kernels(ev) if e["args"].get("stream") == 7]
    if not k:
        return None
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in k)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append([a, b])
    busy = sum(b - a for a, b in merged)
    span = merged[-1][1] - merged[0][0]
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    return {
        "busy_ms_per_step": busy / n / 1000,
        "span_ms_per_step": span / n / 1000,
        "idle_ms_per_step": (span - busy) / n / 1000,
        "busy_pct": busy / span * 100,
        "n_gaps_per_step": len(gaps) / n,
        "gap_sum_ms_per_step": sum(gaps) / n / 1000,
        "gaps_over_1ms_ms_per_step": sum(g for g in gaps if g > 1000) / n / 1000,
    }


def big_gaps(ev, n, thresh_us=1000, top=12):
    """Every gap over thresh_us with the kernels on either side, plus what the CPU was doing.

    fb/e1's question: the warmup window put ~165 of its 186.6 ms idle into 20 gaps, all at the
    rms_norm -> flash seam, and those gaps contained compile_attempt_1. If they are compile
    events they are gone past step 50 and the process side collapses to ~10 ms. If they persist
    at the same seam, "fix one seam" outranks KDA. This function answers which.
    """
    k = sorted((e for e in gpu_kernels(ev) if e["args"].get("stream") == 7), key=lambda e: e["ts"])
    cpu = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation", "python_function") and "dur" in e]
    out = []
    for i in range(len(k) - 1):
        gap = k[i + 1]["ts"] - (k[i]["ts"] + k[i]["dur"])
        if gap <= thresh_us:
            continue
        lo, hi = k[i]["ts"] + k[i]["dur"], k[i + 1]["ts"]
        inside = sorted((c for c in cpu if lo <= c["ts"] <= hi), key=lambda c: -c["dur"])
        out.append({
            "gap_ms": gap / 1000,
            "after": k[i]["name"],
            "before": k[i + 1]["name"],
            "top_cpu": [(c["name"], c["dur"] / 1000) for c in inside[:3]],
        })
    out.sort(key=lambda g: -g["gap_ms"])
    return out, sum(g["gap_ms"] for g in out) / n, len(out) / n


def fusion_table(ev, n, top=12):
    """Per-fusion total, P50/P95 duration, occupancy and a latency verdict."""
    by = defaultdict(list)
    meta = {}
    for e in gpu_kernels(ev):
        nm = e["name"]
        if not nm.startswith("triton_"):
            continue
        by[nm].append(e["dur"])
        a = e.get("args", {})
        if nm not in meta:
            grid = a.get("grid") or [0, 0, 0]
            blk = a.get("block") or [0, 0, 0]
            meta[nm] = {
                "occ": a.get("est. achieved occupancy %", 0) or 0,
                "reg": a.get("registers per thread", 0) or 0,
                "smem": a.get("shared memory", 0) or 0,
                "threads": (grid[0] * grid[1] * grid[2]) * (blk[0] * blk[1] * blk[2]),
            }
    rows = []
    for nm, ds in by.items():
        tot = sum(ds)
        m = meta[nm]
        rows.append({
            "name": nm,
            "ms_per_step": tot / n / 1000,
            "launches_per_step": len(ds) / n,
            "p50_us": st.median(ds),
            "p95_us": sorted(ds)[int(len(ds) * 0.95)] if len(ds) > 1 else ds[0],
            "max_us": max(ds),
            **m,
        })
    rows.sort(key=lambda r: -r["ms_per_step"])
    return rows[:top], sum(r["ms_per_step"] for r in rows), len(rows)


def verdict(r):
    """TileLang target or leave. Occupancy is the discriminator we can actually read.

    A long kernel at low occupancy is latency-limited: too few threads resident to hide memory
    latency, which is what a hand-written schedule fixes. A long kernel at high occupancy is
    already saturating and a rewrite buys little -- leave it.
    """
    if r["occ"] >= 70:
        return "LEAVE (saturated, occ >= 70%)"
    if r["occ"] >= 40:
        return "marginal (occ 40-70%)"
    if r["ms_per_step"] < 5:
        return "LEAVE (too small to matter)"
    return "TILELANG TARGET (low occ, big ms)"


if __name__ == "__main__":
    path, n = sys.argv[1], int(sys.argv[2])
    ev = load(path)
    print(f"trace {path}, {n} active steps\n")

    idl = idle_one_base(ev, n)
    print("== idle, single time base (settles 95 vs 187) ==")
    for k, v in idl.items():
        print(f"  {k:28} {v:10.2f}")
    print()

    gaps, gap_ms_step, gap_n_step = big_gaps(ev, n)
    print("== gaps > 1 ms (fb/e1: do the compile-seam gaps persist past step 50?) ==")
    print(f"  count/step {gap_n_step:.2f} | sum {gap_ms_step:.1f} ms/step")
    if not gaps:
        print("  NONE. The warmup window's ~165 ms in 20 large gaps was compile and it is gone;")
        print("  the process side is whatever idle_ms_per_step says above, all in small gaps.")
    else:
        seam = sum(1 for g in gaps if "rms_norm" in g["after"] and ("flash" in g["before"] or "cute" in g["before"]))
        print(f"  at the rms_norm -> flash seam: {seam} of {len(gaps)}")
        compile_ms = sum(d for g in gaps for nm, d in g["top_cpu"]
                         if "compile" in nm or "guard" in nm or "bytecode" in nm)
        print(f"  cpu compile/guard time inside them: {compile_ms / n:.1f} ms/step")
        for g in gaps[:6]:
            print(f"    {g['gap_ms']:8.1f} ms  after {g['after'][:40]} | before {g['before'][:34]}")
            for nm, d in g["top_cpu"]:
                print(f"        cpu {d:7.1f} ms {nm[:56]}")
    print()

    rows, tot, ndistinct = fusion_table(ev, n)
    print(f"== inductor fusions: {tot:.1f} ms/step over {ndistinct} distinct kernels ==")
    print(f"{'ms/step':>8} {'n/step':>7} {'p50us':>8} {'p95us':>8} {'occ%':>5} {'reg':>4}  verdict")
    for r in rows:
        print(f"{r['ms_per_step']:8.1f} {r['launches_per_step']:7.1f} {r['p50_us']:8.1f} "
              f"{r['p95_us']:8.1f} {r['occ']:5.0f} {r['reg']:4.0f}  {verdict(r)}")
    print()
    tgt = [r for r in rows if verdict(r).startswith("TILELANG")]
    print(f"TileLang candidates: {len(tgt)}, {sum(r['ms_per_step'] for r in tgt):.1f} ms/step")
    print(f"  ideal 2x on those = {sum(r['ms_per_step'] for r in tgt) / 2:.1f} ms/step saved")
    for r in tgt:
        print(f"    {r['ms_per_step']:6.1f} ms  occ {r['occ']:3.0f}%  {r['name'][:72]}")
