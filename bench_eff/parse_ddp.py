#!/usr/bin/env python3
"""Parse the 7-GPU DDP trace: kernel breakdown + cutlass GEMM attribution + grid dims."""
import json, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/work/aupai/bench_eff/ddp_trace_rank0.json"
tr = json.load(open(path))["traceEvents"]

# 1. kernel categorization
def categorize(name):
    n = name.lower()
    if "kda" in n or "chunk_gated" in n or "chunk_gla" in n or "conv_depthwise" in n:
        return "KDA (fla chunk_kda + shortconv)"
    if "kernel_kernel" in n:
        return "KDA (fla chunk_kda + shortconv)"  # confirmed ChunkKDAFunctionBackward
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
        return "bf16 GEMM (cutlass sm75) [LM head?]"
    if "softmax" in n and ("stack" in n or "select" in n or "unsqueeze" in n):
        return "AttnRes (softmax over sources)"
    if "tanh" in n and ("abs" in n or "max" in n):
        return "SwiGLU (bounded activation)"
    if "triton" in n:
        return "inductor fusions (norms/elementwise)"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "elementwise (residual adds etc.)"
    return "other"

cats = defaultdict(float)
total = 0.0
for ev in tr:
    if ev.get("cat") == "kernel" and "dur" in ev:
        cats[categorize(ev["name"])] += ev["dur"]
        total += ev["dur"]

n_steps = 20
print(f"=== GPU kernel time: {total/1e6:.1f} ms over {n_steps} active steps = {total/n_steps/1000:.1f} ms/step ===\n")
print(f"{'category':<45} {'ms/step':>10} {'pct':>7}")
for c, us in sorted(cats.items(), key=lambda kv: -kv[1]):
    print(f"{c:<45} {us/n_steps/1000:>10.1f} {100*us/total:>6.1f}%")

# 2. cutlass bf16 GEMM: grid dims + CPU correlation
print("\n=== cutlass bf16 GEMM details (grid dims -> shape) ===")
cutlass_kernels = defaultdict(lambda: [0.0, None, 0])
for ev in tr:
    if ev.get("cat") == "kernel" and "cutlass" in ev["name"] and "bf16" in ev["name"]:
        key = ev["name"][:60]
        cutlass_kernels[key][0] += ev["dur"]
        if cutlass_kernels[key][1] is None:
            cutlass_kernels[key][1] = ev.get("args", {}).get("grid", "?")
        cutlass_kernels[key][2] += 1
for name, (us, grid, cnt) in sorted(cutlass_kernels.items(), key=lambda kv: -kv[1][0]):
    print(f"  {us/n_steps/1000:>8.1f} ms/step  x{cnt:>4}  grid={grid}  {name}")

# 3. CPU correlation for the biggest cutlass kernel
print("\n=== CPU op correlation for top cutlass kernel ===")
top_cutlass = max(
    (ev for ev in tr if ev.get("cat") == "kernel" and "cutlass" in ev["name"] and "bf16" in ev["name"]),
    key=lambda e: e["dur"],
    default=None,
)
if top_cutlass:
    ext_id = top_cutlass.get("args", {}).get("External id")
    corr = top_cutlass.get("args", {}).get("correlation")
    print(f"  kernel: {top_cutlass['name'][:70]}  ext_id={ext_id} corr={corr}")
    for ev in tr:
        if ev.get("cat") == "cpu_op" and ev.get("args", {}).get("External id") == ext_id:
            print(f"  CPU op: {ev['name'][:80]}")
            break
    # also check runtime correlation
    for ev in tr:
        if ev.get("cat") == "runtime" and ev.get("args", {}).get("correlation") == corr:
            print(f"  runtime: {ev['name'][:80]}")
            break

# 4. DDP allreduce
print("\n=== DDP allreduce ===")
nccl = sum(ev["dur"] for ev in tr if ev.get("cat") == "kernel" and ("nccl" in ev["name"].lower() or "allreduce" in ev["name"].lower()))
print(f"  NCCL kernel time: {nccl/n_steps/1000:.1f} ms/step ({100*nccl/total:.1f}%)")

# 5. data loading (HtoD memcpy)
print("\n=== data loading (HtoD) ===")
htod = sum(ev["dur"] for ev in tr if ev.get("cat") == "kernel" and "memcpy" in ev["name"].lower() and "HtoD" in ev.get("args", {}).get("memory copy kind", ""))
print(f"  HtoD memcpy: {htod/n_steps/1000:.2f} ms/step ({100*htod/total:.2f}%)")
