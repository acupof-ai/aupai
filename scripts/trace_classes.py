#!/usr/bin/env python3
"""Chrome-trace -> per-class GPU time, with a roofline ratio per class (tilerl).

Consumes b0's `runs/trace_p200m_3step.json` (torch.profiler, CUDA+CPU, 3 steps)
and answers the question fb actually asked: **how much room is there in each
stage**. A share alone does not say that -- 40% in GEMM is healthy if those
GEMMs are near peak and terrible if they are at 15% of it. So every class gets
an ideal time from its own FLOPs or bytes, and the ratio is the room.

WHAT IS MEASURED VS ASSUMED, because the ratio is only as good as its ideal:
  * measured   kernel name, device duration, input shapes -- all from the trace
  * derived    FLOPs for a GEMM of known shapes = 2*M*N*K, bytes for an
               elementwise op = sum(input) + sum(output) at its dtype width
  * assumed    the peak it is divided by: H20 SXM FP8 296 TFLOPS dense / BF16
               148 / 4.0 TB/s HBM (facts/efficiency.json#eff.h20_peak)

A kernel whose shapes the trace does not carry gets NO ideal time and is
reported as unknown rather than given a guessed one -- an ideal computed from a
shape nobody recorded is a number without a basis.

CLASSES follow the block structure, not the kernel vendor: fla KDA, flash/MLA,
GEMM (the FFN and projections), elementwise+cast, NCCL, optimizer, other.
Block class is decided by the `A_log` key when a state dict is supplied
(`blocks.<i>.mixer.A_log` present => KDA), never by a hardcoded layer index:
that ratio is 3:1 at L12 and L32 but 3.5:1 at L18, so an index pattern silently
mislabels the 300M.

    python3 scripts/trace_classes.py runs/trace_p200m_3step.json --steps 3
    python3 scripts/trace_classes.py --selftest      # no trace, no card
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

#: facts/efficiency.json#eff.h20_peak. Dense, not sparse.
PEAK = {"fp8": 296e12, "bf16": 148e12, "fp16": 148e12, "fp32": 44e12}
HBM_BYTES_PER_S = 4.0e12

DTYPE_BYTES = {"float": 4, "float32": 4, "f32": 4, "double": 8,
               "bfloat16": 2, "bf16": 2, "half": 2, "float16": 2, "f16": 2,
               "char": 1, "int8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
               "long": 8, "int64": 8, "int": 4, "int32": 4, "bool": 1}

#: Matched in order; first hit wins. Names are substrings of the CUDA kernel.
CLASS_PATTERNS = [
    ("nccl", ("nccl", "allreduce", "all_reduce", "reduce_scatter", "all_gather")),
    ("fla_kda", ("chunk_kda", "fused_recurrent", "chunk_gla", "fla_", "chunk_delta")),
    ("flash_mla", ("flash", "fmha", "attention_kernel", "mha_")),
    ("optimizer", ("adam", "muon", "newton_schulz", "foreach_", "zero_grad", "clip_")),
    ("gemm", ("gemm", "cutlass", "sgemm", "hgemm", "s16816", "nvjet", "matmul",
              "ampere_", "hopper_", "cublas", "scaled_mm")),
    ("elementwise_cast", ("elementwise", "vectorized_", "cast", "copy", "fill",
                          "convert", "quantize", "dequant", "silu", "gelu", "mul",
                          "add", "norm", "softmax", "cross_entropy", "index_")),
]


def classify(name: str) -> str:
    low = name.lower()
    for cls, pats in CLASS_PATTERNS:
        if any(p in low for p in pats):
            return cls
    return "other"


def kda_layers(state_dict_keys: list[str]) -> tuple[set[int], set[int]]:
    """(kda, attn) layer indices, decided by whether `A_log` exists in a block.

    Works at any depth; an index pattern does not (3:1 at L12/L32, 3.5:1 at L18).
    """
    kda, seen = set(), set()
    pat = re.compile(r"blocks\.(\d+)\.")
    for k in state_dict_keys:
        m = pat.search(k)
        if not m:
            continue
        i = int(m.group(1))
        seen.add(i)
        if k.endswith("mixer.A_log"):
            kda.add(i)
    return kda, seen - kda


def _shape_flops(shapes: list) -> int | None:
    """2*M*N*K for a matmul of recorded shapes, else None.

    None is the honest answer for a kernel whose shapes the trace does not
    carry: an ideal time from an invented shape is worse than no ideal time.
    """
    dims = [s for s in shapes if isinstance(s, list) and len(s) >= 2]
    if len(dims) < 2:
        return None
    a, b = dims[0], dims[1]
    m, k = a[-2], a[-1]
    if b[-2] == k:
        n = b[-1]
    elif b[-1] == k:
        n = b[-2]
    else:
        return None
    batch = 1
    for d in a[:-2]:
        batch *= d
    return 2 * batch * m * n * k


def _shape_bytes(shapes: list, width: int) -> int | None:
    total = 0
    for s in shapes:
        if not isinstance(s, list):
            continue
        n = 1
        for d in s:
            n *= d
        total += n
    return total * width if total else None


def _shape_index(events: list[dict]) -> dict:
    """External id -> the cpu_op that launched it, for its recorded shapes.

    A CUDA kernel event carries no Input Dims of its own; the shapes live on the
    aten op that launched it, joined by "External id". Reading them off the
    kernel returns nothing, which is why the first run of this script reported
    every ideal time as unknown.
    """
    ops = {}
    for e in events:
        if e.get("cat") != "cpu_op":
            continue
        a = e.get("args") or {}
        x = a.get("External id")
        if x is not None and a.get("Input Dims") and x not in ops:
            ops[x] = e
    return ops


def _busy_us(spans: list[tuple[float, float]]) -> float:
    """Union of the spans, not their sum.

    Kernels on different streams overlap, so summing durations double-counts
    and can exceed the wall clock -- the first run reported a NEGATIVE idle,
    which is the arithmetic saying so.
    """
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(spans):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0.0)


def analyse(events: list[dict], steps: int, precision: str = "fp8") -> dict:
    """Per-class device time, share, ideal time and the ratio."""
    per = defaultdict(lambda: {"us": 0.0, "n": 0, "ideal_us": 0.0, "unknown": 0})
    ops = _shape_index(events)
    spans: list[tuple[float, float]] = []
    sum_us = 0.0
    tmin, tmax = None, None
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(e.get("dur", 0.0))
        name = e.get("name", "")
        cls = classify(name)
        args = e.get("args") or {}
        op = (ops.get(args.get("External id")) or {}).get("args") or {}
        shapes = op.get("Input Dims") or []
        types = op.get("Input type") or []
        width = next((DTYPE_BYTES[t] for t in types if t in DTYPE_BYTES), None)
        rec = per[cls]
        rec["us"] += dur
        rec["n"] += 1
        sum_us += dur
        ts = float(e.get("ts", 0.0))
        spans.append((ts, ts + dur))
        tmin = ts if tmin is None else min(tmin, ts)
        tmax = ts + dur if tmax is None else max(tmax, ts + dur)
        ideal = None
        if cls in ("gemm", "fla_kda", "flash_mla"):
            fl = _shape_flops(shapes)
            if fl is not None:
                ideal = fl / PEAK.get(precision, PEAK["bf16"]) * 1e6
        elif width is not None:
            by = _shape_bytes(shapes, width)
            if by is not None:
                ideal = by / HBM_BYTES_PER_S * 1e6
        if ideal is None:
            rec["unknown"] += 1
        else:
            rec["ideal_us"] += ideal
    wall_us = (tmax - tmin) if (tmin is not None and tmax is not None) else 0.0
    busy_us = _busy_us(spans)
    rows = []
    for cls, r in sorted(per.items(), key=lambda kv: -kv[1]["us"]):
        covered = r["n"] - r["unknown"]
        rows.append({
            "class": cls,
            "gpu_ms_per_step": r["us"] / 1000.0 / steps,
            "share": r["us"] / sum_us if sum_us else 0.0,
            "kernels": r["n"],
            "ideal_ms_per_step": (r["ideal_us"] / 1000.0 / steps) if covered else None,
            "ratio": (r["us"] / r["ideal_us"]) if r["ideal_us"] > 0 else None,
            "shapes_missing": r["unknown"],
        })
    return {
        "rows": rows,
        "busy_ms_per_step": busy_us / 1000.0 / steps,
        "wall_ms_per_step": wall_us / 1000.0 / steps,
        # Everything the GPU was not running a kernel: launch gaps, host work,
        # unoverlapped H2D. This line is why the table is not just shares.
        "idle_ms_per_step": (wall_us - busy_us) / 1000.0 / steps,
        "precision_assumed": precision,
    }


def render(rep: dict) -> str:
    out = [f"{'class':<18}{'GPU ms/step':>12}{'share':>8}{'ideal ms':>10}"
           f"{'ratio':>8}{'kernels':>9}{'no-shape':>10}"]
    for r in rep["rows"]:
        ideal = f"{r['ideal_ms_per_step']:.2f}" if r["ideal_ms_per_step"] is not None else "-"
        ratio = f"{r['ratio']:.1f}x" if r["ratio"] is not None else "-"
        out.append(f"{r['class']:<18}{r['gpu_ms_per_step']:>12.2f}{r['share']:>7.1%}"
                   f"{ideal:>10}{ratio:>8}{r['kernels']:>9}{r['shapes_missing']:>10}")
    out.append(f"{'':<18}{'-'*12}")
    out.append(f"{'busy':<18}{rep['busy_ms_per_step']:>12.2f}")
    out.append(f"{'idle (wall-busy)':<18}{rep['idle_ms_per_step']:>12.2f}")
    out.append(f"{'wall':<18}{rep['wall_ms_per_step']:>12.2f}")
    out.append(f"\nratio = measured / ideal at {rep['precision_assumed']} peak; "
               "'-' = the trace carried no shapes, so no ideal was invented.")
    return "\n".join(out)


def _selftest() -> None:
    assert classify("nccl:all_reduce") == "nccl"
    assert classify("chunk_kda_fwd_kernel") == "fla_kda"
    assert classify("flash_fwd_kernel") == "flash_mla"
    assert classify("nvjet_hsh_128x128") == "gemm"
    assert classify("vectorized_elementwise_kernel") == "elementwise_cast"
    assert classify("some_mystery_kernel") == "other"
    # NCCL must win over gemm for a fused name -- order matters in the table.
    assert classify("ncclDevKernel_AllReduce_Sum_gemm") == "nccl"
    # A_log decides the block class at any depth, including 3.5:1.
    keys = [f"blocks.{i}.mixer.A_log" for i in (0, 1, 3, 4)] + \
           [f"blocks.{i}.mixer.kv_down.weight" for i in (2, 5)]
    kda, attn = kda_layers(keys)
    assert kda == {0, 1, 3, 4} and attn == {2, 5}, (kda, attn)
    assert _shape_flops([[4, 8], [8, 16]]) == 2 * 4 * 8 * 16
    assert _shape_flops([[2, 4, 8], [8, 16]]) == 2 * 2 * 4 * 8 * 16
    assert _shape_flops([[4, 8]]) is None, "one operand is not a matmul"
    assert _shape_flops([[4, 8], [16, 32]]) is None, "no shared dim -> no guess"
    # One 1 TFLOP-ish GEMM at exactly peak must read 1.0x, and a kernel with no
    # shapes must land in shapes_missing rather than distorting the ratio.
    flops = 2 * 100 * 100 * 100
    us = flops / PEAK["fp8"] * 1e6
    ev = [{"ph": "X", "cat": "cpu_op", "name": "aten::mm", "dur": 1.0, "ts": 0,
           "args": {"External id": 1, "Input Dims": [[100, 100], [100, 100]]}},
          {"ph": "X", "cat": "kernel", "name": "nvjet_gemm", "dur": us, "ts": 0,
           "args": {"External id": 1}},
          {"ph": "X", "cat": "kernel", "name": "mystery", "dur": 5.0, "ts": us,
           "args": {}}]
    rep = analyse(ev, steps=1)
    gemm = next(r for r in rep["rows"] if r["class"] == "gemm")
    assert abs(gemm["ratio"] - 1.0) < 1e-6, gemm
    other = next(r for r in rep["rows"] if r["class"] == "other")
    assert other["ratio"] is None and other["shapes_missing"] == 1, other
    # idle is wall minus busy, and the two kernels here are back to back.
    assert abs(rep["idle_ms_per_step"]) < 1e-9, rep
    # Overlapping kernels on two streams: busy is the UNION, so idle never goes
    # negative. Summing durations here would give 3.0 busy against 2.0 wall.
    assert _busy_us([(0.0, 2.0), (1.0, 2.0)]) == 2.0
    assert _busy_us([(0.0, 1.0), (2.0, 3.0)]) == 2.0
    over = [{"ph": "X", "cat": "kernel", "name": "a", "dur": 2.0, "ts": 0.0, "args": {}},
            {"ph": "X", "cat": "kernel", "name": "b", "dur": 1.0, "ts": 1.0, "args": {}}]
    r2 = analyse(over, steps=1)
    assert r2["busy_ms_per_step"] <= r2["wall_ms_per_step"], r2
    assert r2["idle_ms_per_step"] >= 0.0, r2
    # Shapes come from the launching cpu_op, joined by External id -- a kernel
    # carries none of its own, and reading them off it yields no ideal at all.
    joined = [{"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0.0, "dur": 1.0,
               "args": {"External id": 9, "Input Dims": [[4, 8], [8, 16]]}},
              {"ph": "X", "cat": "kernel", "name": "nvjet_gemm", "ts": 0.0, "dur": 5.0,
               "args": {"External id": 9}}]
    r3 = analyse(joined, steps=1)
    g = next(r for r in r3["rows"] if r["class"] == "gemm")
    assert g["shapes_missing"] == 0 and g["ratio"] is not None, g
    print("trace_classes selftest OK: classes, A_log split, roofline, idle, join")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--precision", default="fp8", choices=sorted(PEAK))
    ap.add_argument("--state-dict-keys", help="newline-separated keys, for the KDA/MLA split")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.trace:
        ap.error("trace required (or --selftest)")
    with open(a.trace, encoding="utf-8") as f:
        doc = json.load(f)
    events = doc.get("traceEvents", doc if isinstance(doc, list) else [])
    rep = analyse(events, steps=a.steps, precision=a.precision)
    if a.state_dict_keys:
        kda, attn = kda_layers(open(a.state_dict_keys, encoding="utf-8").read().split())
        rep["kda_layers"], rep["attn_layers"] = sorted(kda), sorted(attn)
    print(json.dumps(rep, indent=1) if a.json else render(rep))
    if not events:
        sys.exit("no traceEvents in that file")


if __name__ == "__main__":
    main()
