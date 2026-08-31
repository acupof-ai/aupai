#!/usr/bin/env python3
"""The LM-head GEMMs Liger FLCE emits, priced against H20 peak, as a function of vocab size.

At batch 16 / seq 4096 / d 1024 / vocab 32773 Liger picks (fused_linear_cross_entropy.py):
    BT = 65536, H = 1024, V = 32773
    inc_factor = cdiv(V, H)              = 33
    chunk_size = next_pow2(cdiv(BT, 33)) = 2048
    num_chunks = cdiv(BT, 2048)          = 32
so every step runs 32 chunks x 3 GEMMs of M=2048, K=1024, N=V:
    logits   = x @ w.T          (nt)
    grad_x   = dlogits @ w      (nn)
    grad_w  += dlogits.T @ x    (tn)

The question this answers: how much of that time is the *odd* vocab. 32773 is not a multiple of
8, so the [M, V] logits have a 2-byte-aligned leading dimension and cuBLAS falls back to an
alignment-1 kernel. Sweeping V tells us what alignment costs, separately from bf16-vs-fp8.

    CUDA_VISIBLE_DEVICES=7 python -u scripts/lm_head_gemm.py
"""

import argparse

import torch
from torch.autograd import DeviceType

PEAK_BF16, PEAK_FP8 = 148.0, 296.0


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def kernel_name(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    ks = [e for e in p.key_averages() if e.device_type == DeviceType.CUDA]
    return max(ks, key=lambda e: e.self_device_time_total).key if ks else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2048, help="Liger chunk size at BT=65536, H=1024, V=32773")
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--chunks", type=int, default=32)
    ap.add_argument("--vocabs", default="32773,32776,32784,32832,33024")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()

    dev = "cuda"
    M, K = a.M, a.K
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    print(f"H20, bf16, M={M} K={K}, {a.chunks} chunks/step, peak {PEAK_BF16} TFLOPS bf16\n", flush=True)
    hdr = f"{'V':>7} {'V%8':>4} {'nt fwd':>9} {'nn dx':>9} {'tn dw':>9} {'3 GEMM':>9} {'x32':>9} {'TFLOPS':>8} {'%pk':>6}"
    print(hdr, flush=True)

    names = {}
    for V in [int(v) for v in a.vocabs.split(",")]:
        w = torch.randn(V, K, device=dev, dtype=torch.bfloat16)
        g = torch.randn(M, V, device=dev, dtype=torch.bfloat16)
        nt, nn, tn = (lambda w=w: x @ w.t()), (lambda g=g, w=w: g @ w), (lambda g=g: g.t() @ x)
        t_nt, t_nn, t_tn = timeit(nt, a.iters), timeit(nn, a.iters), timeit(tn, a.iters)
        tot = t_nt + t_nn + t_tn
        flops = 3 * 2 * M * K * V
        tf = flops / (tot / 1000) / 1e12
        print(
            f"{V:>7} {V % 8:>4} {t_nt:>8.3f}m {t_nn:>8.3f}m {t_tn:>8.3f}m {tot:>8.3f}m"
            f" {tot * a.chunks:>8.2f}m {tf:>8.1f} {tf / PEAK_BF16:>5.0%}",
            flush=True,
        )
        names[V] = (kernel_name(nt), kernel_name(nn), kernel_name(tn))
        del w, g
        torch.cuda.empty_cache()

    print("\ncuBLAS kernel picked (nt / nn / tn):", flush=True)
    for V, ns in names.items():
        for tag, n in zip(("nt", "nn", "tn"), ns, strict=True):
            print(f"  V={V} {tag}: {n[:96]}", flush=True)

    # A slice of a padded weight, which is what train.py hands FLCE: head.weight[:vocab].
    wp = torch.randn(32832, K, device=dev, dtype=torch.bfloat16)
    ws = wp[:32773]

    def sliced():
        return x @ ws.t()

    t = timeit(sliced, a.iters)
    print(f"\nsliced head.weight[:32773] of [32832,{K}]: nt {t:.3f} ms -> {kernel_name(sliced)[:90]}")

    # fp8 reference at an aligned V, forward GEMM only (_scaled_mm needs both dims % 16).
    V8 = 32832
    xf = x.to(torch.float8_e4m3fn)
    wf = torch.randn(V8, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    s = torch.ones((), device=dev)

    def mm8():
        return torch._scaled_mm(xf, wf.t(), scale_a=s, scale_b=s, out_dtype=torch.bfloat16)

    t8 = timeit(mm8, a.iters)
    tf8 = 2 * M * K * V8 / (t8 / 1000) / 1e12
    print(
        f"fp8 _scaled_mm nt at V={V8}: {t8:.3f} ms, {tf8:.1f} TFLOPS ({tf8 / PEAK_FP8:.0%} of {PEAK_FP8})",
        flush=True,
    )


if __name__ == "__main__":
    main()
