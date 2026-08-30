#!/usr/bin/env python3
"""Where the KDA chunk kernels spend their time: dependency chain, occupancy, or launch overhead.

The discriminating experiment is the scaling sweep. KDA's state scan is sequential along T, but
every (batch, head) pair is an independent scan. So:
    time ~ constant as B*H grows  -> the GPU was idle, the limit is parallelism (recoverable)
    time ~ linear in B*H          -> the GPU is fed, the limit is the chain or bandwidth (not)
Sweeping T instead isolates the sequential axis: that one MUST be linear either way.

Calls fla's chunk_kda directly with train.py's exact arguments, so no model, no compile, no fp8.

    CUDA_VISIBLE_DEVICES=7 python -u scripts/kda_probe.py --mode sweep
    CUDA_VISIBLE_DEVICES=7 python -u scripts/kda_probe.py --mode kernels --batch 16
"""

import argparse
import math

import torch
from fla.ops.kda import chunk_kda
from torch.autograd import DeviceType

H20_SMS = 78


def make(B, T, h, hd, dev="cuda", dtype=torch.bfloat16):
    def rq(*s):
        return torch.randn(*s, device=dev, dtype=dtype, requires_grad=True)

    q, k, v, g = (rq(B, T, h, hd) for _ in range(4))
    beta = rq(B, T, h)
    A_log = torch.zeros(h, device=dev, requires_grad=True)
    dt = torch.exp(torch.rand(h * hd, device=dev) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
    dt_bias = (dt + torch.log(-torch.expm1(-dt))).requires_grad_()
    return q, k, v, g, beta, A_log, dt_bias


def call(args):
    q, k, v, g, beta, A_log, dt_bias = args
    out, _ = chunk_kda(
        q,
        k,
        v,
        g=g,
        beta=beta,
        cu_seqlens=None,
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=True,
        disable_recompute=True,
    )
    return out


def timeit(args, iters, backward):
    grad = None

    def once():
        nonlocal grad
        out = call(args)
        if backward:
            if grad is None or grad.shape != out.shape:
                grad = torch.randn_like(out)
            torch.autograd.grad(out, [t for t in args if t.requires_grad], grad, retain_graph=False)

    for _ in range(3):
        once()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        once()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def sweep(a):
    print(f"H20 GPU7, bf16, hd={a.hd}, no compile, no fp8, {a.iters} iters, fwd+bwd\n", flush=True)
    print("axis           B     T    h    B*h      ms   ms/(B*h)  vs prev  work-scaled", flush=True)
    for label, cases in (
        ("batch", [(b, a.seq, a.heads) for b in (1, 2, 4, 8, 16, 32)]),
        ("heads", [(a.batch, a.seq, h) for h in (1, 2, 4, 8, 16, 32)]),
        ("seqlen", [(a.batch, t, a.heads) for t in (512, 1024, 2048, 4096, 8192)]),
    ):
        if label not in a.axes.split(","):
            continue
        prev_t = prev_w = None
        for B, T, h in cases:
            args = make(B, T, h, a.hd)
            try:
                ms = timeit(args, a.iters, True)
            except torch.OutOfMemoryError:
                print(f"{label:<10} {B:>5} {T:>5} {h:>4}   OOM", flush=True)
                break
            w = B * h * T
            ratio = f"{ms / prev_t:.2f}x" if prev_t else "   -"
            wr = f"{(ms / prev_t) / (w / prev_w):.2f}" if prev_t else "   -"
            print(
                f"{label:<10} {B:>5} {T:>5} {h:>4} {B * h:>6} {ms:>7.2f} {ms / (B * h):>10.3f}"
                f" {ratio:>8} {wr:>12}",
                flush=True,
            )
            prev_t, prev_w = ms, w
            del args
            torch.cuda.empty_cache()
        print(flush=True)


def kernels(a):
    args = make(a.batch, a.seq, a.heads, a.hd)
    for _ in range(3):
        out = call(args)
        torch.autograd.grad(out, [t for t in args if t.requires_grad], torch.randn_like(out))
    torch.cuda.synchronize()
    n = 3
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            out = call(args)
            torch.autograd.grad(out, [t for t in args if t.requires_grad], torch.randn_like(out))
        torch.cuda.synchronize()
    ev = [e for e in p.key_averages() if e.device_type == DeviceType.CUDA]
    tot = sum(e.self_device_time_total for e in ev) / n / 1000
    print(
        f"\nH20 GPU7, bf16, B={a.batch} T={a.seq} h={a.heads} hd={a.hd}, no compile, fwd+bwd,"
        f" {n} iters\ntotal {tot:.3f} ms per fwd+bwd of ONE layer ({tot * 9:.1f} ms for 9 layers)\n",
        flush=True,
    )
    print(f"{'ms':>8} {'calls':>6} {'us/call':>9} {'%':>6}  kernel", flush=True)
    for e in sorted(ev, key=lambda e: -e.self_device_time_total):
        ms = e.self_device_time_total / n / 1000
        c = e.count / n
        print(f"{ms:>8.3f} {c:>6.1f} {ms * 1000 / c:>9.1f} {ms / tot:>5.1%}  {e.key[:74]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "kernels"], default="sweep")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--hd", type=int, default=128)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--axes", default="batch,heads,seqlen")
    a = ap.parse_args()
    torch.manual_seed(0)
    (sweep if a.mode == "sweep" else kernels)(a)


if __name__ == "__main__":
    main()
