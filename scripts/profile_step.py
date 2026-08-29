#!/usr/bin/env python3
"""Where the step time actually goes: CUDA kernels ranked by total time, grouped by what emitted them.

Runs on one GPU, so it answers where the time goes without the whole DDP job.

    python scripts/profile_step.py --steps 12 --batch 24 --attn_res --attn_res_blocks 4
    python scripts/profile_step.py --steps 12 --batch 32            # the AttnRes-free baseline
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402

# Kernel-name fragments -> the thing in the model that emits them. First match wins, so the specific
# patterns come before the generic ones.
BUCKETS = [
    ("attnres", r"attn_?res|rms_scale|_body"),
    ("kda", r"kda|chunk_(fwd|bwd|intra)|wy_fast|short_conv|conv1d"),
    ("attention", r"flash|attn_fwd|attn_bwd|_fwd_kernel|_bwd_kernel"),
    ("loss", r"cross_entropy|flce|logsumexp"),
    ("fp8", r"float8|scaled_mm|fp8|amax"),
    ("matmul", r"gemm|cutlass|sm90|nvjet|ampere|_mm|addmm|bmm"),
    ("optimizer", r"muon|newton|adam|polar|orthogonal|clip|foreach"),
    ("norm", r"rms|norm|rsqrt"),
    ("elementwise", r"elementwise|vectorized|pointwise|triton_poi|copy|cat|stack|silu|sigmoid|tanh|mul|add"),
    ("reduction", r"reduce|triton_red|sum|mean|softmax"),
    ("comm", r"nccl|allreduce|all_reduce|broadcast"),
]


def bucket(name):
    low = name.lower()
    for label, pat in BUCKETS:
        if re.search(pat, low):
            return label
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12, help="profiled steps after warmup")
    ap.add_argument("--warmup", type=int, default=8, help="steps before profiling (compile + autotune)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--attn_res", action="store_true")
    ap.add_argument("--attn_res_blocks", type=int, default=None)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--no_fp8", action="store_true")
    ap.add_argument("--top", type=int, default=18, help="individual kernels to list")
    a = ap.parse_args()

    for k in ("batch", "attn_res", "attn_res_blocks"):
        v = getattr(a, k)
        if v:
            setattr(train.Cfg, k, v)
    if a.no_compile:
        train.Cfg.compile = False
    cfg = train.Cfg
    dev = "cuda"

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    model = train.HybridLM(cfg).to(dev).to(torch.bfloat16)
    if not a.no_fp8:
        train.convert_to_fp8_compute(model)
    if cfg.compile:
        model = torch.compile(model, dynamic=False)
    opts = train.build_optimizers(model, cfg)
    flce = train.LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=train.SOFTCAP)
    raw = getattr(model, "_orig_mod", model)

    x = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq), device=dev)
    y = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq), device=dev)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, _ = model(x, y)
        w = raw.head.weight[: cfg.vocab]
        loss = flce(w, hidden.to(w.dtype).reshape(-1, hidden.shape[-1]), y.reshape(-1))
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)

    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    peak_before = torch.cuda.max_memory_allocated() / 2**30

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(a.steps):
            step()
        torch.cuda.synchronize()

    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total = sum(e.self_device_time_total for e in events)
    per_step_ms = total / a.steps / 1000
    tokens = cfg.batch * cfg.seq
    print(
        f"\nbatch {cfg.batch} seq {cfg.seq} attn_res {cfg.attn_res}/{cfg.attn_res_blocks} "
        f"compile {cfg.compile}\n"
        f"{per_step_ms:.1f} ms/step of GPU time, {tokens / (per_step_ms / 1000) / 1e3:.0f}K tok/s/gpu, "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB (warmup {peak_before:.1f})\n"
    )

    groups = defaultdict(float)
    counts = defaultdict(int)
    for e in events:
        groups[bucket(e.key)] += e.self_device_time_total
        counts[bucket(e.key)] += e.count
    print(f"{'group':<13}{'% of step':>10}{'ms/step':>10}{'launches/step':>15}")
    for g, t in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{g:<13}{t / total:>9.1%}{t / a.steps / 1000:>10.2f}{counts[g] / a.steps:>15.0f}")

    print(f"\ntop {a.top} kernels:")
    for e in sorted(events, key=lambda e: -e.self_device_time_total)[: a.top]:
        print(
            f"  {e.self_device_time_total / total:>6.1%} {e.self_device_time_total / a.steps / 1000:>8.2f}ms"
            f" {e.count / a.steps:>6.0f}x  {e.key[:88]}"
        )


if __name__ == "__main__":
    main()
