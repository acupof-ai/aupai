"""Does fp8 realize the same fraction of its vendor peak as bf16 does, on this pod?

eff.fp8_gemm_at_realizable_peak argues the fp8 GEMMs are at ~100% of realizable peak, not 93%,
and so a faster GEMM library buys ~0. That argument carries eff.gpu4_peak_flops' MEASURED bf16
ratio (137.0 against a vendor 148 = 92.6%) over to fp8 by ASSUMPTION. Its own uncertainty field
names this probe as the one measurement that would settle it.

Same shape and method as eff.gpu4_peak_flops (8192^3, 30 iters) so the two ratios are comparable:
if fp8 also lands near 92.6% of its 296 vendor number, the entry stands as written; if fp8
realizes a materially higher fraction, there is more nominal headroom than the entry claims and
it must be amended.

    CUDA_VISIBLE_DEVICES=7 python -u probes/t59_fp8_peak.py
"""
import json
import statistics

import torch

N, ITERS, WARMUP = 8192, 30, 10
VENDOR_BF16, VENDOR_FP8 = 148.0, 296.0
E4M3_MAX = 448.0


def _tflops(fn, flop):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(ITERS):
        start.record()
        fn()
        end.record()
        end.synchronize()
        ts.append(start.elapsed_time(end))
    ms = statistics.median(ts)
    return flop / (ms / 1e3) / 1e12, ms


def main():
    dev = "cuda"
    torch.manual_seed(0)
    flop = 2.0 * N * N * N

    a = torch.empty(N, N, device=dev, dtype=torch.bfloat16).normal_(0, 0.5)
    b = torch.empty(N, N, device=dev, dtype=torch.bfloat16).normal_(0, 0.5)

    def q(t):
        s = (t.detach().abs().amax().clamp(min=1e-12) / E4M3_MAX).float()
        return (t / s).to(torch.float8_e4m3fn), s

    qa, sa = q(a)
    qb, sb = q(b)
    # TN is the layout fp8 tensor cores want; .t().contiguous().t() gives a column-major b
    # without a timed copy, matching how train.py:_fp8_mm hands operands to _scaled_mm.
    qbt = qb.t().contiguous().t()

    bf16_tflops, bf16_ms = _tflops(lambda: a @ b, flop)
    fp8_tflops, fp8_ms = _tflops(
        lambda: torch._scaled_mm(qa, qbt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16), flop
    )

    bf16_pct = bf16_tflops / VENDOR_BF16 * 100
    fp8_pct = fp8_tflops / VENDOR_FP8 * 100

    print(f"\n{N}^3 GEMM, median of {ITERS} iters")
    print(f"  bf16  {bf16_ms:7.2f} ms  {bf16_tflops:7.1f} TFLOPS  = {bf16_pct:5.1f}% of vendor {VENDOR_BF16:.0f}")
    print(f"  fp8   {fp8_ms:7.2f} ms  {fp8_tflops:7.1f} TFLOPS  = {fp8_pct:5.1f}% of vendor {VENDOR_FP8:.0f}")
    print("\n  eff.gpu4_peak_flops measured bf16 at 137.0 TFLOPS = 92.6% of vendor")
    print(f"  this run's bf16: {bf16_tflops:.1f} ({bf16_pct:.1f}%) -- {'consistent' if abs(bf16_tflops - 137.0) < 6 else 'INCONSISTENT, investigate'}")

    # The entry assumed fp8 realizes the same 92.6%. Does it?
    assumed = VENDOR_FP8 * (bf16_tflops / VENDOR_BF16)
    print(f"\n  entry's assumption: fp8 realizable = 296 x {bf16_pct / 100:.3f} = {assumed:.1f} TFLOPS")
    print(f"  measured fp8      : {fp8_tflops:.1f} TFLOPS = {fp8_tflops / assumed * 100:.1f}% of that")
    if fp8_pct > bf16_pct + 3:
        print("  -> fp8 realizes a HIGHER fraction than bf16. The entry understates nominal")
        print("     headroom and must be amended with this number.")
    elif fp8_pct < bf16_pct - 3:
        print("  -> fp8 realizes a LOWER fraction than bf16. The entry's conclusion holds a")
        print("     fortiori: even less headroom than it claims.")
    else:
        print("  -> fp8 realizes the same fraction as bf16, within 3 points. The entry's")
        print("     assumption is confirmed and its uncertainty can be closed.")

    # What the head's 274.5 TFLOPS looks like against a MEASURED fp8 peak rather than a carried one.
    HEAD_ACHIEVED = 274.5
    print(f"\n  the fp8 linears' 274.5 TFLOPS is {HEAD_ACHIEVED / fp8_tflops * 100:.1f}% of this measured fp8 peak")

    out = {
        "probe": "t59_fp8_peak", "n": N, "iters": ITERS,
        "bf16_tflops": round(bf16_tflops, 1), "bf16_pct_of_vendor": round(bf16_pct, 1),
        "fp8_tflops": round(fp8_tflops, 1), "fp8_pct_of_vendor": round(fp8_pct, 1),
        "vendor_bf16": VENDOR_BF16, "vendor_fp8": VENDOR_FP8,
        "fp8_linears_achieved_tflops": HEAD_ACHIEVED,
        "fp8_linears_pct_of_measured_peak": round(HEAD_ACHIEVED / fp8_tflops * 100, 1),
    }
    with open("/work/aupai/runs/t59_fp8_peak.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote runs/t59_fp8_peak.json")


if __name__ == "__main__":
    main()
