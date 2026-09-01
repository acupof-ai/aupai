"""fp8 head: how much of the -3.9% A/B regression is per-tensor quantisation, not the GEMM?

eff.fp8_head_ab_noship measured the fp8 head 3.9% SLOWER (66.4 ms of a 1702 ms step) where the
roofline predicted +5.6%. The GEMM is not the suspect: `patch_liger_flce_fp8` rewrites Liger's
three call sites to `_fp8_mm` -> `torch._scaled_mm` and train.py:2155 is fail-closed, so fp8
tensor cores WERE on the path. What the roofline never counted is the traffic `_fp8_mm` adds
around the GEMM: an amax reduction per operand, a divide-and-cast, and a `.contiguous()`.

Arithmetic for the head's shape (M=2048 K=1024 N=32784, 64 chunks, 3 call sites) puts that at
~22 ms of streaming plus ~7.5 ms of fp32-accumulator traffic plus a strided transpose -- about
34 ms of the 66.4. This measures it instead of predicting it, and the residual is the finding:
near 66 means the 320 per-step amax launches are the rest; near 34 means something unnamed
costs 32 ms and the lever stays no-ship until it has a name.

Arms, interleaved in one process (fb's condition, so drift and clock state hit both equally):
    quant    _fp8_mm as production runs it -- amax, cast, contiguous, then _scaled_mm
    pre      the same _scaled_mm on operands quantised OUTSIDE the timed window
    bf16     the baseline GEMM the fp8 path replaced
The tax is quant - pre. The lever's ceiling is bf16 - pre.

    CUDA_VISIBLE_DEVICES=7 python -u probes/t58_quant_tax.py
"""
import json
import statistics
import sys

import torch

sys.argv = ["t58"]
sys.path.insert(0, "/work/aupai/scripts")
sys.path.insert(0, "/work/aupai")
import train  # noqa: E402

M, K, N = 2048, 1024, 32784
CHUNKS, ITERS, WARMUP = 64, 120, 20
E4M3_MAX = 448.0


def _time(fn, iters=ITERS, warmup=WARMUP):
    """Median ms over `iters`, cuda-synchronised. Median, not mean: one eval or clock
    excursion in a shared container moves a mean and not a median (eff.ab_throughput_statistic
    found the raw cv 6.1% against a 3% gate for exactly this reason)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        ts.append(start.elapsed_time(end))
    return statistics.median(ts), statistics.pstdev(ts)


def _q(t):
    s = (t.detach().abs().amax().clamp(min=1e-12) / E4M3_MAX).float()
    return (t / s).to(torch.float8_e4m3fn).contiguous(), s


def main():
    dev = "cuda"
    torch.manual_seed(0)
    # The three head GEMMs, same shapes Liger FLCE emits per chunk.
    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16).normal_(0, 0.5)
    G = torch.empty(M, N, device=dev, dtype=torch.bfloat16).normal_(0, 0.02)

    # Liger's three sites, verbatim from patch_liger_flce_fp8's substitutions:
    #   logits      = _fp8_mm(_input_chunk, weight.t(), ...)   A  @ Wt   [M,K]x[K,N]
    #   grad_input  = _fp8_mm(grad_logits, weight, ...)        G  @ W ... but Liger's `weight`
    #                                                          is [N,K], so W here is Wn
    #   grad_weight = _fp8_mm(grad_logits.t(), _input_chunk)   Gt @ A    [N,M]x[M,K]
    # `weight` in Liger is [vocab, hidden] = [N, K]; weight.t() is [K, N].
    Wn = torch.empty(N, K, device=dev, dtype=torch.bfloat16).normal_(0, 0.02)
    Wt = Wn.t().contiguous()  # [K, N], the forward's b operand
    Gt = G.t().contiguous()  # [N, M], the grad_weight site's a operand

    qA, sA = _q(A)
    qWt, sWt = _q(Wt)
    qWn, sWn = _q(Wn)
    qG, sG = _q(G)
    qGt, sGt = _q(Gt)

    def sm(a, sa, b, sb, out):
        return torch._scaled_mm(a, b.t().contiguous().t(), scale_a=sa, scale_b=sb, out_dtype=out)

    ARMS = {
        # production shape: train.py:_fp8_mm, quantisation inside the timed window
        "quant": lambda: (
            train._fp8_mm(A, Wt, A.dtype, cache_b=True),
            train._fp8_mm(G, Wn, torch.bfloat16),
            train._fp8_mm(Gt, A, torch.float32),
        ),
        # same GEMMs, operands already fp8: the quantisation is outside the window
        "pre": lambda: (
            sm(qA, sA, qWt, sWt, torch.bfloat16),
            sm(qG, sG, qWn, sWn, torch.bfloat16),
            sm(qGt, sGt, qA, sA, torch.float32),
        ),
        # the baseline the fp8 path replaced
        "bf16": lambda: (
            A @ Wt,
            G @ Wn,
            torch.mm(Gt, A).float(),
        ),
    }

    # Interleaved, not sequential: a thermal or clock drift over the run then lands on
    # every arm rather than on whichever ran last.
    samples = {k: [] for k in ARMS}
    for _ in range(3):
        for name, fn in ARMS.items():
            med, sd = _time(fn, iters=ITERS // 3, warmup=WARMUP // 3)
            samples[name].append(med)
    per_chunk = {k: statistics.median(v) for k, v in samples.items()}
    per_step = {k: v * CHUNKS for k, v in per_chunk.items()}

    tax = per_step["quant"] - per_step["pre"]
    ceiling = per_step["bf16"] - per_step["pre"]
    net = per_step["bf16"] - per_step["quant"]

    print(f"\nhead GEMMs M={M} K={K} N={N}, {CHUNKS} chunks/step, 3 call sites")
    print(f"{'arm':6s} {'ms/chunk':>10s} {'ms/step':>10s}   spread over 3 interleaved blocks")
    for k in ARMS:
        sp = max(samples[k]) - min(samples[k])
        print(f"{k:6s} {per_chunk[k]:10.3f} {per_step[k]:10.1f}   {sp:.3f} ms/chunk")

    print(f"\nquantisation tax   = quant - pre  = {tax:6.1f} ms/step")
    print(f"fp8 GEMM ceiling   = bf16  - pre  = {ceiling:6.1f} ms/step   (what EVT could keep)")
    print(f"net as shipped     = bf16  - quant= {net:+6.1f} ms/step   (negative = fp8 slower)")

    ACCOUNTED, MEASURED_REGRESSION = 34.0, 66.4
    resid = MEASURED_REGRESSION - tax
    print(f"\nA/B regression {MEASURED_REGRESSION} ms; arithmetic accounted for ~{ACCOUNTED} ms.")
    print(f"measured tax {tax:.1f} ms leaves a residual of {resid:+.1f} ms.")
    if abs(resid) < 12:
        print("  -> the tax IS the regression. The 320 amax launches per step are the")
        print("     remainder (inference, not measured here).")
    else:
        print(f"  -> {resid:+.1f} ms is unnamed. The lever stays no-ship until it has a name.")

    out = {
        "probe": "t58_quant_tax", "shape": {"M": M, "K": K, "N": N, "chunks": CHUNKS},
        "statistic": "median of 3 interleaved blocks, cuda events, pre-declared",
        "ms_per_step": per_step, "ms_per_chunk": per_chunk,
        "quant_tax_ms": tax, "fp8_gemm_ceiling_ms": ceiling, "net_as_shipped_ms": net,
        "ab_regression_ms": MEASURED_REGRESSION, "residual_ms": resid,
    }
    with open("/work/aupai/runs/t58_quant_tax.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote runs/t58_quant_tax.json")


if __name__ == "__main__":
    main()
