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
  * measured   the FLOP peak it is divided by: THIS pod's, 279.6 TFLOPS fp8 and
               136.7 bf16 (facts/efficiency.json#eff.fp8_gemm_at_realizable_peak,
               #eff.gpu4_peak_flops). NOT the H20 sheet's 296/148, which this
               silicon reaches 94.5% and 92.4% of -- dividing by the sheet is how
               a GEMM already at 100% of the machine reads as "93%, 7% to gain".
  * assumed    the BANDWIDTH peak: 4.0 TB/s is the sheet figure and nothing here
               has measured achievable HBM, so an elementwise ratio against it is
               a lower bound on how close to bandwidth-bound a kernel already is.

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

#: MEASURED on this pod, not the vendor sheet. facts/efficiency.json#eff.fp8_gemm_at_realizable_peak
#: (probes/t59_fp8_peak.py: 8192^3, 30 iters, 279.6 TFLOPS fp8 / 136.7 bf16) and
#: #eff.gpu4_peak_flops (137.0 and 137.3 on two cards). The sheet says 296 / 148, which this
#: silicon does not reach: 94.5% and 92.4% of it. Pricing against the sheet is what produced the
#: familiar "93% of peak" and its "7% headroom" -- headroom against a number no kernel can hit.
#: The fp8 linears' 274.5 TFLOPS is 98.2% of the measured peak, so the real headroom is 1.8%.
#: fp32 stays a sheet figure, labelled: nothing here measured it, and no kernel in these traces
#: runs in fp32.
PEAK = {"fp8": 279.6e12, "bf16": 136.7e12, "fp16": 136.7e12, "fp32": 44e12}
#: Also measured, same file: 4.0 TB/s is the sheet. Kept as the sheet value and named as such --
#: nothing in this repo has measured achievable HBM bandwidth, so an elementwise ratio computed
#: against it is a bound, not a measurement.
HBM_BYTES_PER_S = 4.0e12

#: Every key is a string the profiler ACTUALLY writes, verified against the p200m trace's own
#: Input type values, not guessed from torch's dtype repr. The trace says "c10::BFloat16" and
#: "c10::Float8_e4m3fn" -- neither was in this table, so 3,720 of 15,426 elementwise kernels got
#: no width and therefore no ideal at all. The short aliases below are kept because other
#: profiler versions emit them.
#:
#: NOT-A-TENSOR entries are deliberately absent: "Scalar", "TensorList", "" and the like carry no
#: bytes, and _width_of skips them rather than mapping them to a size.
DTYPE_BYTES = {"float": 4, "float32": 4, "f32": 4, "double": 8,
               "bfloat16": 2, "bf16": 2, "half": 2, "float16": 2, "f16": 2,
               "char": 1, "int8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
               "long": 8, "long int": 8, "int64": 8, "int": 4, "int32": 4, "bool": 1,
               "c10::BFloat16": 2, "c10::Half": 2, "c10::Float8_e4m3fn": 1,
               "c10::Float8_e5m2": 1, "c10::complex<float>": 8}

#: Input type entries that are not tensors. A Scalar has no bytes, and it is usually FIRST in the
#: list for ops like add_(t, alpha=1) -- so "the first entry with a known width" picked the
#: scalar's `float` (4 bytes) for 1,794 bf16 kernels and `double` (8) for 531, pricing 2-byte
#: traffic at 2x and 4x. The width must come from the first entry that is a TENSOR.
NON_TENSOR_TYPES = {"Scalar", "ScalarList", "TensorList", "GenericList", "Device", "Layout",
                    "MemoryFormat", "Generator", "Storage", "", "None"}


def _width_of(types: list) -> int | None:
    """Bytes per element of the first TENSOR operand, or None.

    The dtype of the data, not of whatever happens to be listed first: an aten op's Input type
    interleaves tensors with Scalars, and a Scalar records as `float` or `double`. Taking the
    first *recognised* string therefore priced bf16 traffic at 4 or 8 bytes per element on 2,325
    of p200m's elementwise kernels. None when no operand is a tensor of known dtype -- an
    unpriced kernel is reported as unpriced, never given a guessed width.
    """
    for t in types:
        if t in NON_TENSOR_TYPES:
            continue
        if t in DTYPE_BYTES:
            return DTYPE_BYTES[t]
        return None
    return None

#: Matched in order; first hit wins. Names are substrings of the CUDA kernel.
CLASS_PATTERNS = [
    ("nccl", ("nccl", "allreduce", "all_reduce", "reduce_scatter", "all_gather")),
    # kda / delta_rule / recompute_w_u added from p200m's real trace: chunk_gated_delta_rule_*
    # (25 ms/step), recompute_w_u_fwd_kda_* (9), kda_gate_* (6) all matched none of the five
    # patterns below and sat in `other`, so fla_kda read 61 ms/step against a real 172.
    ("fla_kda", ("chunk_kda", "kda", "fused_recurrent", "chunk_gla", "fla_", "chunk_delta",
                 "delta_rule", "recompute_w_u")),
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
    """Bytes an elementwise op moves: every input READ, plus one output WRITTEN.

    The write was missing, and the docstring above already claimed it was counted (44 found the
    disagreement by reading the two against each other). Its absence halves the ideal for a
    two-input op and understates it by a third for the common in-place case, so the
    elementwise ratio printed 1.6x -- "room to fuse" -- when the corrected figure is at or near
    the bandwidth roofline and there is none.

    The output shape is INFERRED as the largest input, not read: this profiler's traces carry
    Input Dims and Input Strides and no Output Dims at all (checked across all 55,512 shaped
    cpu_ops in the p200m trace). An elementwise op writes one tensor whose shape is the
    broadcast of its inputs, and the broadcast of a set of shapes has the element count of the
    largest -- so the inference is exact for elementwise and is why this function is only ever
    called for that class. It is wrong for a reduction, which writes less: those land in `other`
    or `fla_kda`, where this is not called.

    In-place is counted correctly by the same rule without a special case: aten::add_(a, b)
    records both operands as inputs and writes a, so 2 reads + 1 write = 3 tensor-sized
    transfers, which is what sum(inputs) + max(input) gives.
    """
    sizes = []
    for s in shapes:
        if not isinstance(s, list) or not s:
            continue
        n = 1
        for d in s:
            n *= d
        sizes.append(n)
    if not sizes:
        return None
    return (sum(sizes) + max(sizes)) * width


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


#: aten ops whose Input Dims ARE a matmul's operands. A GEMM-classified kernel launched by
#: anything else -- a fused triton epilogue, a cast -- has dims describing its inputs but not a
#: multiply, and 2*M*N*K over them is a number with no basis. MEASURED on p200m:
#: triton_poi_fused__scaled_mm reports [[16,4096,6144],[65536,6144],[6144,65536]] and yields an
#: ideal of 59.43 ms/step against 0.374 ms measured, and a handful of those ids outweighed every
#: real GEMM combined -- which is why gemm still read 0.1x after the per-op pricing fix. They go
#: to shapes_missing, which is what that column is for.
MATMUL_OPS = {"aten::mm", "aten::bmm", "aten::addmm", "aten::baddbmm", "aten::matmul",
              "aten::_scaled_mm", "aten::linear", "aten::_int_mm"}

#: fp8 only when the op says so. One trace holds both: p200m ran aten::_scaled_mm for the
#: converted linears and aten::mm for everything else, and pricing the bf16 half at the fp8 peak
#: halves its apparent MFU.
FP8_OPS = {"aten::_scaled_mm"}


def _op_class(op_name: str, kernel_cls: str) -> str:
    """The class that should carry a launching op's ideal time.

    A matmul op's FLOPs belong to its GEMM kernel, never to the memset or copy
    that shares its External id.
    """
    return "gemm" if op_name in MATMUL_OPS else kernel_cls


def analyse(events: list[dict], steps: int, precision: str = "fp8") -> dict:
    """Per-class device time, share, ideal time and the ratio."""
    per = defaultdict(lambda: {"us": 0.0, "n": 0, "ideal_us": 0.0, "unknown": 0, "priced_us": 0.0})
    ops = _shape_index(events)
    counted: set = set()
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
        args = e.get("args") or {}
        op_ev = ops.get(args.get("External id")) or {}
        op = op_ev.get("args") or {}
        op_name = op_ev.get("name") or ""
        shapes = op.get("Input Dims") or []
        types = op.get("Input type") or []
        # The launching op names the work when the kernel name does not: `kernel_kernel` is
        # 58 ms/step of ChunkKDAFunctionBackward, and with kda_gate_*, chunk_gated_delta_rule_*
        # and recompute_w_u_* also landing in `other`, fla_kda read 61 against a real 172 --
        # understating by 64% the block whose cost the 300M A/B is about.
        #
        # KDA only, NOT a general "classify by the op when the kernel is opaque". MEASURED on
        # this trace: a generic fallback moves 2.72 ms/step and what it moves is wrong -- a
        # Memset launched by aten::_scaled_mm becomes gemm, a DtoD memcpy becomes elementwise. A
        # rule earning 0.2% of a step by labelling memory traffic as compute is worse than none.
        cls = classify(name)
        if op_name and ("kda" in op_name.lower() or "delta_rule" in op_name.lower()):
            cls = "fla_kda"
        width = _width_of(types)
        rec = per[cls]
        rec["us"] += dur
        rec["n"] += 1
        sum_us += dur
        # One op can launch many kernels (up to 32 here: split-k, epilogues).
        # Its FLOPs are the op's, not each kernel's -- charging them per kernel
        # inflated the GEMM ideal 12x and produced a 0.1x ratio, i.e. measured
        # "faster than peak", which is the arithmetic reporting its own error.
        ext = args.get("External id")
        # The ideal is charged once per launching op, to the class of the
        # kernel that does the WORK -- not to whichever event happens to come
        # first. MEASURED on p200m: of 2,070 matmul ids, the first event is a
        # memset (class `other`) for 1,818 of them, so 88% of the GEMM ideal
        # was landing in `other`. That is what made gemm read 37.8x and other
        # 0.1x: mirror images of one misattribution, not two findings.
        first = ext is None or ext not in counted
        if ext is not None and cls == _op_class(op_name, cls):
            first = ext not in counted
            counted.add(ext)
        elif ext is not None:
            first = False
        ts = float(e.get("ts", 0.0))
        spans.append((ts, ts + dur))
        tmin = ts if tmin is None else min(tmin, ts)
        tmax = ts + dur if tmax is None else max(tmax, ts + dur)
        ideal = None
        if op_name in MATMUL_OPS:
            fl = _shape_flops(shapes)
            if fl is not None:
                peak = PEAK["fp8"] if op_name in FP8_OPS else PEAK.get(precision, PEAK["bf16"])
                ideal = fl / peak * 1e6
        elif cls not in ("gemm", "fla_kda", "flash_mla") and width is not None:
            by = _shape_bytes(shapes, width)
            if by is not None:
                ideal = by / HBM_BYTES_PER_S * 1e6
        if ideal is None:
            rec["unknown"] += 1
        elif first:
            rec["ideal_us"] += ideal
        # The ratio's NUMERATOR must cover the same kernels as its denominator. Measured time was
        # accumulated for every kernel while ideal time only for those an op priced, so a class
        # with unpriced kernels divided ALL of its time by PART of its ideal. MEASURED: gemm
        # printed 1.36x that way and 1.17x over its priced kernels alone, because 43% of its 3,918
        # kernels carry no matmul op. That 0.19x is bookkeeping, not room.
        #
        # `ideal is not None` is exactly "this kernel's time is covered by a pricing basis": an
        # op's ideal covers every kernel it launched, and the flag is recomputed per kernel from
        # the same op, so all of a priced op's kernels count here while its ideal is added once.
        if ideal is not None:
            rec["priced_us"] += dur
    wall_us = (tmax - tmin) if (tmin is not None and tmax is not None) else 0.0
    busy_us = _busy_us(spans)
    rows = []
    for cls, r in sorted(per.items(), key=lambda kv: -kv[1]["us"]):
        rows.append({
            "class": cls,
            "gpu_ms_per_step": r["us"] / 1000.0 / steps,
            "share": r["us"] / sum_us if sum_us else 0.0,
            "kernels": r["n"],
            "ideal_ms_per_step": (r["ideal_us"] / 1000.0 / steps) if r["ideal_us"] > 0 else None,
            # measured/ideal over THE SAME kernels. `priced_ms_per_step` is that numerator, printed
            # beside the ratio so a class whose ratio covers half its time says so.
            "priced_ms_per_step": r["priced_us"] / 1000.0 / steps,
            "ratio": (r["priced_us"] / r["ideal_us"]) if r["ideal_us"] > 0 else None,
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
    out = [f"{'class':<18}{'GPU ms/step':>12}{'share':>8}{'priced ms':>11}{'ideal ms':>10}"
           f"{'ratio':>8}{'kernels':>9}{'no-shape':>10}"]
    for r in rep["rows"]:
        ideal = f"{r['ideal_ms_per_step']:.2f}" if r["ideal_ms_per_step"] is not None else "-"
        ratio = f"{r['ratio']:.2f}x" if r["ratio"] is not None else "-"
        priced = f"{r['priced_ms_per_step']:.2f}" if r["ideal_ms_per_step"] is not None else "-"
        out.append(f"{r['class']:<18}{r['gpu_ms_per_step']:>12.2f}{r['share']:>7.1%}"
                   f"{priced:>11}{ideal:>10}{ratio:>8}{r['kernels']:>9}"
                   f"{r['shapes_missing']:>10}")
    out.append(f"{'':<18}{'-'*12}")
    out.append(f"{'busy':<18}{rep['busy_ms_per_step']:>12.2f}")
    out.append(f"{'idle (wall-busy)':<18}{rep['idle_ms_per_step']:>12.2f}")
    out.append(f"{'wall':<18}{rep['wall_ms_per_step']:>12.2f}")
    out.append(
        f"\nratio = priced ms / ideal ms at {rep['precision_assumed']} peak (aten::_scaled_mm "
        "always at the fp8 peak). BOTH cover the same kernels: 'priced ms' is the measured time of\n"
        "the kernels an op could price, which is what the ideal is computed over -- dividing a "
        "class's WHOLE time by a partial ideal read gemm as 1.36x where its priced\nkernels are "
        "1.17x. 'GPU ms/step' minus 'priced ms' is time with no roofline basis, not time at ratio "
        "1. 'no-shape' counts the kernels in that gap: no cpu_op\nshapes, or a launching op that "
        "is not a matmul, whose dims do not describe a multiply."
    )
    return "\n".join(out)


def _selftest() -> None:
    # PEAK must be THIS pod's measured peak, asserted against literals. Every other case here
    # computes its expectation FROM PEAK, so all of them move together when PEAK moves and none
    # can see a wrong peak -- the mutation that swaps in the vendor sheet's 296/148 stayed green
    # across the whole file. A constant that only ever appears on both sides of an equation is
    # unguarded by construction. Sources: facts/efficiency.json#eff.fp8_gemm_at_realizable_peak
    # (279.6 fp8, probes/t59_fp8_peak.py) and #eff.gpu4_peak_flops (137.0/137.3 bf16).
    assert PEAK["fp8"] == 279.6e12, (
        f"fp8 peak is {PEAK['fp8'] / 1e12:.1f} TFLOPS; the MEASURED value is 279.6. The H20 sheet's "
        "296 is 5.5% higher than this silicon reaches, and dividing by it is how a GEMM already at "
        "100% of the machine reads as 93% with 7% to gain."
    )
    assert PEAK["bf16"] == PEAK["fp16"] == 136.7e12, (
        f"bf16 peak is {PEAK['bf16'] / 1e12:.1f} TFLOPS; the MEASURED value is 136.7, against the "
        "sheet's 148."
    )
    assert HBM_BYTES_PER_S == 4.0e12, "bandwidth is the sheet's 4.0 TB/s, labelled assumed"
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

    # _shape_bytes had NO known answer at all, which is how its docstring ("sum(input) +
    # sum(output)") and its code (inputs only) disagreed unmeasured until 44 read them against
    # each other. Every case below is one tensor of 100 elements at 2 bytes = 200 B.
    assert _shape_bytes([[10, 10]], 2) == 400, "one input: read it, write one output"
    assert _shape_bytes([[10, 10], [10, 10]], 2) == 600, "two inputs read, one output written"
    # A broadcast writes the LARGER shape, and the output is inferred as the largest input --
    # this trace format carries no Output Dims, checked over all 55,512 shaped cpu_ops.
    assert _shape_bytes([[10, 10], [10, 1]], 2) == (100 + 10 + 100) * 2
    # aten::add_(a, b) records both operands and writes a: 3 tensor-sized transfers.
    assert _shape_bytes([[10, 10], [10, 10]], 2) == 3 * 200
    assert _shape_bytes([], 2) is None and _shape_bytes([[]], 2) is None, \
        "no shapes means no ideal, never zero bytes"
    # The width matters and is the dtype's, not a constant: bf16 and fp32 differ 2x.
    assert _shape_bytes([[10, 10]], 4) == 800

    # _width_of: the dtype of the first TENSOR, and the strings the profiler really writes.
    # Both halves were wrong and each is worth a case. Counted on p200m's 15,426 elementwise
    # kernels: 3,720 got NO width because "c10::BFloat16" was not in the table, and 2,325 got
    # 4 or 8 bytes for 2-byte data because a leading Scalar records as float or double.
    assert _width_of(["c10::BFloat16"]) == 2, "the trace writes c10::BFloat16, not bfloat16"
    assert _width_of(["c10::Float8_e4m3fn"]) == 1
    assert _width_of(["Scalar", "c10::BFloat16"]) == 2, "a Scalar is not the data's dtype"
    assert _width_of(["float", "c10::BFloat16"]) == 4, \
        "a leading float TENSOR is the data's dtype -- only non-tensors are skipped"
    assert _width_of(["Scalar", "Scalar"]) is None, "no tensor operand means no width"
    assert _width_of([]) is None
    assert _width_of(["SomeFutureType"]) is None, "an unknown tensor type is unpriced, not guessed"
    # bf16 at 2 bytes, priced as float, is 2x the real traffic -- the ratio it produces is HALF.
    assert _shape_bytes([[10, 10]], _width_of(["Scalar", "c10::BFloat16"])) == 400
    assert _shape_bytes([[10, 10]], 4) == 800
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
    # The ideal must EXIST before the ratio is compared. Comparing None to 1.0 raises TypeError,
    # and a crash is not a caught defect: the first mutation run of the join fix exited nonzero
    # with a TypeError and read as "the guard fired" while naming nothing.
    assert gemm["ratio"] is not None, f"a priced aten::mm produced no ideal: {gemm}"
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

    # A GEMM-classified kernel whose op is NOT a matmul gets no ideal. p200m's
    # triton_poi_fused__scaled_mm reports [[16,4096,6144],[65536,6144],[6144,65536]]: 2*M*N*K
    # over that is 59.43 ms/step against 0.374 measured, and a few such ids outweighed every
    # real GEMM, which is why gemm read 0.1x -- an MFU over 1000% -- even after per-op pricing.
    fused = [{"ph": "X", "cat": "cpu_op", "name": "triton_poi_fused__scaled_mm", "ts": 0.0,
              "dur": 1.0, "args": {"External id": 11,
                                   "Input Dims": [[16, 4096, 6144], [65536, 6144],
                                                  [6144, 65536]],
                                   "Input type": ["bfloat16", "bfloat16", "bfloat16"]}},
             {"ph": "X", "cat": "kernel", "name": "nvjet_fused", "ts": 0.0, "dur": 3.0,
              "args": {"External id": 11}}]
    f = next(r for r in analyse(fused, steps=1)["rows"])
    assert f["class"] == "gemm" and f["ratio"] is None and f["shapes_missing"] == 1, f

    # Each op at its own peak: one trace holds fp8 and bf16 matmuls, and pricing aten::mm at the
    # fp8 peak halves its MFU while pricing _scaled_mm at bf16 doubles its ideal.
    def _one(op, dtype, prec):
        e = [{"ph": "X", "cat": "cpu_op", "name": op, "ts": 0.0, "dur": 1.0,
              "args": {"External id": 12, "Input Dims": [[100, 100], [100, 100]],
                       "Input type": [dtype, dtype]}},
             {"ph": "X", "cat": "kernel", "name": "nvjet_gemm", "ts": 0.0, "dur": 1.0,
              "args": {"External id": 12}}]
        r = next(x for x in analyse(e, steps=1, precision=prec)["rows"])
        assert r["ideal_ms_per_step"] is not None, f"the join is broken: {r}"
        return r["ideal_ms_per_step"]

    assert abs(_one("aten::mm", "bfloat16", "bf16") - flops / PEAK["bf16"] * 1e6 / 1e3) < 1e-9
    assert abs(_one("aten::_scaled_mm", "float8_e4m3fn", "bf16")
               - flops / PEAK["fp8"] * 1e6 / 1e3) < 1e-9, \
        "aten::_scaled_mm must keep the fp8 peak even under --precision bf16"

    # The launching op names the class when the kernel name cannot: `kernel_kernel` was 58 ms/step
    # of KDA backward sitting in `other`.
    opaque = [{"ph": "X", "cat": "cpu_op", "name": "ChunkKDAFunctionBackward", "ts": 0.0,
               "dur": 1.0, "args": {"External id": 13, "Input Dims": [[8, 8]],
                                    "Input type": ["bfloat16"]}},
              {"ph": "X", "cat": "kernel", "name": "kernel_kernel", "ts": 0.0, "dur": 4.0,
               "args": {"External id": 13}}]
    assert next(r for r in analyse(opaque, steps=1)["rows"])["class"] == "fla_kda"
    gdr = [{"ph": "X", "cat": "kernel", "name": "chunk_gated_delta_rule_bwd_kernel_dhu",
            "ts": 0.0, "dur": 4.0, "args": {"External id": 14}}]
    assert next(r for r in analyse(gdr, steps=1)["rows"])["class"] == "fla_kda"
    # A matmul id whose FIRST event is a memset: the ideal must land on the
    # GEMM kernel, not on the memset's class. This is the 88% misattribution
    # that made gemm read 37.8x and other 0.1x on the real trace.
    mix = [{"ph": "X", "cat": "cpu_op", "ts": 0.0, "dur": 1.0, "name": "aten::mm",
            "args": {"External id": 11, "Input Dims": [[4, 8], [8, 16]]}},
           {"ph": "X", "cat": "gpu_memset", "name": "Memset", "ts": 0.0, "dur": 1.0,
            "args": {"External id": 11}},
           {"ph": "X", "cat": "kernel", "name": "nvjet_gemm", "ts": 1.0, "dur": 4.0,
            "args": {"External id": 11}}]
    rows = {r["class"]: r for r in analyse(mix, steps=1)["rows"]}
    assert rows["gemm"]["ratio"] is not None, rows

    # The ratio's numerator and denominator must cover THE SAME kernels. A class with some
    # unpriced kernels divided ALL its time by PART of its ideal: gemm printed 1.36x that way
    # where its priced kernels are 1.17x, because 43% of its 3,918 kernels carry no matmul op.
    # Two GEMM-classified kernels here, one priced at exactly peak and one launched by nothing:
    # the ratio must read 1.0x over the priced one, not 3.0x over both.
    part = [{"ph": "X", "cat": "cpu_op", "ts": 0.0, "dur": 1.0, "name": "aten::_scaled_mm",
             "args": {"External id": 21, "Input Dims": [[100, 100], [100, 100]],
                      "Input type": ["c10::Float8_e4m3fn", "c10::Float8_e4m3fn"]}},
            {"ph": "X", "cat": "kernel", "name": "nvjet_gemm", "ts": 0.0, "dur": us,
             "args": {"External id": 21}},
            {"ph": "X", "cat": "kernel", "name": "nvjet_gemm", "ts": us, "dur": 2 * us,
             "args": {"External id": 22}}]
    g = next(r for r in analyse(part, steps=1)["rows"] if r["class"] == "gemm")
    assert g["ratio"] is not None, f"the priced kernel produced no ideal: {g}"
    assert abs(g["ratio"] - 1.0) < 1e-6, \
        f"ratio must be priced/ideal, not total/ideal: {g['ratio']:.2f}x over {g}"
    assert abs(g["priced_ms_per_step"] - us / 1000.0) < 1e-9, g
    assert g["gpu_ms_per_step"] > g["priced_ms_per_step"], \
        "the unpriced kernel's time must still be reported in the class total"
    assert g["shapes_missing"] == 1, g

    # analyse() must go through _width_of, not pick the first recognised type itself. The unit
    # cases above pass while the call site does the old thing -- a function tested in isolation
    # and never reached is the same defect as an import with no call site. End to end: an
    # in-place bf16 add with a leading Scalar, 100 elements, 2 reads + 1 write at 2 bytes.
    ip = [{"ph": "X", "cat": "cpu_op", "ts": 0.0, "dur": 1.0, "name": "aten::add_",
           "args": {"External id": 31, "Input Dims": [[10, 10], [10, 10], []],
                    "Input type": ["c10::BFloat16", "c10::BFloat16", "Scalar"]}},
          {"ph": "X", "cat": "kernel", "name": "vectorized_elementwise_kernel", "ts": 0.0,
           "dur": 1.0, "args": {"External id": 31}}]
    e_row = next(r for r in analyse(ip, steps=1)["rows"])
    want_ms = 3 * 100 * 2 / HBM_BYTES_PER_S * 1e6 / 1000.0
    assert e_row["ideal_ms_per_step"] is not None, f"the bf16 width did not resolve: {e_row}"
    assert abs(e_row["ideal_ms_per_step"] - want_ms) < 1e-12, \
        f"expected {want_ms} ms for 3x100 bf16 elements, got {e_row['ideal_ms_per_step']}"
    # A Scalar LEADING the list must not set the width. Same op, Scalar first: same answer.
    ip[0]["args"]["Input type"] = ["Scalar", "c10::BFloat16", "c10::BFloat16"]
    e_row2 = next(r for r in analyse(ip, steps=1)["rows"])
    assert e_row2["ideal_ms_per_step"] == e_row["ideal_ms_per_step"], \
        f"a leading Scalar changed the width: {e_row2} vs {e_row}"

    # analyse() must REFUSE when the first tensor's dtype is unknown, rather than walking on to a
    # later operand's. That walk is the general form of the real failure: p200m has 525
    # aten::mul with ["c10::BFloat16", "double", ...] and 525 aten::div_ with
    # ["c10::BFloat16", "long int"], where skipping the unrecognised leading name landed on the
    # 8-byte one and priced 2-byte traffic at 4x. Adding the c10:: names fixes those two cases;
    # this assertion is what stops the next unrecognised dtype from doing it again.
    ip[0]["args"]["Input type"] = ["c10::SomeFutureDtype", "double"]
    e_row3 = next(r for r in analyse(ip, steps=1)["rows"])
    assert e_row3["ideal_ms_per_step"] is None, \
        f"an unknown leading dtype borrowed a later operand's width: {e_row3}"
    assert e_row3["shapes_missing"] == 1, e_row3
    assert rows["other"]["ideal_ms_per_step"] in (None, 0.0), rows
    print("trace_classes selftest OK: classes, A_log split, roofline, idle, join, "
          "matmul-only pricing, per-op precision, opaque KDA names")


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
