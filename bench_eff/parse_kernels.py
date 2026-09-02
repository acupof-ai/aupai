#!/usr/bin/env python3
"""Per-kernel raw output: name, self CUDA time/step, count, grid, grouped by op semantics.

Reads the 6-GPU DDP trace (GPU 1-6, exclusive, batch16/seq4096/bf16+fp8/compiled).
"""
import json
from collections import defaultdict

path = "/work/aupai/bench_eff/ddp_trace_rank0.json"
tr = json.load(open(path))["traceEvents"]

def categorize(name):
    n = name.lower()
    if "kda" in n or "chunk_gated" in n or "chunk_gla" in n or "conv_depthwise" in n:
        return "KDA (fla chunk_kda + shortconv)"
    if "kernel_kernel" in n:
        return "KDA (fla chunk_kda + shortconv)"
    if "flash" in n or "fmha" in n:
        return "GatedMLA attention (flash)"
    if "liger" in n or "flce" in n:
        return "FLCE (liger)"
    if "nccl" in n or "allreduce" in n:
        return "DDP allreduce"
    if "memcpy" in n or "memset" in n:
        return "data load (memcpy)"
    if "nvjet" in n:
        return "FP8 linear (cuBLASLt scaled_mm)"
    if "scaled_mm" in n:
        return "FP8 linear (quant fusions)"
    if "cutlass" in n and "bf16" in n:
        return "bf16 GEMM (cutlass sm75) [LM head]"
    if "softmax" in n and ("stack" in n or "select" in n or "unsqueeze" in n):
        return "AttnRes (softmax over sources)"
    if "tanh" in n and ("abs" in n or "max" in n):
        return "SwiGLU (bounded activation)"
    if "triton" in n:
        return "inductor fusions (norms/elementwise)"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "elementwise (residual adds etc.)"
    return "other"

kernels = defaultdict(lambda: [0.0, 0, None])
for ev in tr:
    if ev.get("cat") == "kernel" and "dur" in ev:
        n = ev["name"]
        k = kernels[n]
        k[0] += ev["dur"]
        k[1] += 1
        if k[2] is None:
            k[2] = ev.get("args", {}).get("grid", "?")

n_steps = 20
total = sum(v[0] for v in kernels.values())
print(f"=== total GPU kernel time: {total/n_steps/1000:.1f} ms/step ({n_steps} active steps) ===")
print("config: 6xGPU1-6 DDP, batch16/seq4096, bf16+fp8, compiled, grad_ckpt=False, attn_res=ON\n")

groups = defaultdict(list)
for n, (dur, cnt, grid) in kernels.items():
    groups[categorize(n)].append((n, dur, cnt, grid))

for cat in sorted(groups, key=lambda c: -sum(k[1] for k in groups[c])):
    cat_dur = sum(k[1] for k in groups[cat])
    print(f"=== {cat}: {cat_dur/n_steps/1000:.1f} ms/step ({100*cat_dur/total:.1f}%) ===")
    for n, dur, cnt, grid in sorted(groups[cat], key=lambda x: -x[1])[:15]:
        print(f"  {dur/n_steps/1000:>8.1f} ms  x{cnt:>5}  grid={grid}  {n[:90]}")
    print()
