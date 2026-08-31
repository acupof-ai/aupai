"""Attribute every GPU kernel in a chrome trace to a named group. t56/tilerl-4.

Groups are matched on kernel name, most specific first. The point of the exercise is
coverage: whatever does not match a named group lands in 'UNATTRIBUTED', which is the
number the task exists to shrink below 5%.
"""
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
n_active = int(sys.argv[2]) if len(sys.argv) > 2 else 20

with open(path) as fh:
    ev = json.load(fh)["traceEvents"]
# GPU kernels only: cat kernel/gpu_memcpy/gpu_memset carry device duration in us.
gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]

RULES = [
    ("fp8 GEMM (_scaled_mm)",        r"nvjet_qqtst|_scaled_mm|cutlass.*f8|sm90.*fp8"),
    ("bf16 GEMM (LM head/FLCE)",     r"nvjet_tst|cutlass.*bf16|sm90.*bf16.*gemm"),
    ("flash attention",              r"flash|fmha|attn_fwd|attn_bwd|cute"),
    # KDA/gated-delta kernels. `kernel_kernel` is a Triton kernel with a generic name;
    # its launcher is ChunkKDAFunctionBackward (correlation -> cpu_op, verified in this trace).
    ("KDA / gated-delta (triton)",   r"chunk_kda|chunk_delta|chunk_gated_delta|chunk_gla|chunk_local_cumsum|l2norm_|fused_beta_sigmoid|kda|fused_recurrent|triton_.*chunk|^kernel_kernel$|^element_mul_kernel$"),
    ("liger FLCE",                   r"liger|fused_linear_cross|_flce"),
    ("fp8 quant / scale",            r"to_float8|abs_max|amax|_quant|scaled_cast|convert_fp8"),
    ("inductor fusion (triton)",     r"triton_poi|triton_red|triton_per|triton_tem|triton_unk"),
    ("optimizer (Muon/AdamW)",       r"muon|adam|newton|zeropower|orthogonal|_foreach|multi_tensor"),
    ("NCCL allreduce",               r"nccl|allreduce|all_reduce|reduce_scatter|all_gather"),
    ("memcpy / memset",              r"^Memcpy|^Memset|memcpy|memset"),
    ("elementwise / copy (aten)",    r"elementwise|vectorized_|copy_|CatArrayBatched|index_select|fill_|unrolled"),
    ("reduction / norm / softmax",   r"reduce|softmax|layer_norm|rms_norm|welford|norm_kernel"),
]

tot = defaultdict(float)
counts = defaultdict(int)
unattr = defaultdict(float)
for e in gpu:
    nm = e["name"]
    for label, pat in RULES:
        if re.search(pat, nm, re.I):
            tot[label] += e["dur"]
            counts[label] += 1
            break
    else:
        tot["UNATTRIBUTED"] += e["dur"]
        counts["UNATTRIBUTED"] += 1
        unattr[nm] += e["dur"]

grand = sum(tot.values())
per_step_total = grand / n_active / 1000.0
print(f"trace {path}")
print(f"{len(gpu)} GPU kernels over {n_active} active steps")
print(f"total GPU kernel time {grand/1e6:.3f} s -> {per_step_total:.1f} ms/step\n")
print(f"{'kernel group':32} {'ms/step':>9} {'% GPU':>7} {'launches/step':>14}")
print("-" * 66)
for label in sorted(tot, key=tot.get, reverse=True):
    ms = tot[label] / n_active / 1000.0
    print(f"{label:32} {ms:9.1f} {tot[label]/grand*100:7.1f} {counts[label]/n_active:14.0f}")
print("-" * 66)
named = 100 - tot.get("UNATTRIBUTED", 0) / grand * 100
print(f"named coverage {named:.1f}%  (target >= 95%)")
if unattr:
    print("\ntop unattributed kernels:")
    for nm, d in sorted(unattr.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {d/n_active/1000.0:7.2f} ms/step  {nm[:88]}")
