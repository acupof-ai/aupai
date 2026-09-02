#!/usr/bin/env python3
"""Parse a chrome trace into the step-time breakdown table."""
import json, sys

def categorize(name):
    n = name.lower()
    if "kda" in n: return "kda_kernel"
    if "flash" in n or "fmha" in n: return "attention"
    if "liger" in n or "flce" in n: return "flce"
    if "nccl" in n or "allreduce" in n: return "ddp_comm"
    if "memcpy" in n or "memset" in n: return "memcpy_memset"
    if any(k in n for k in ("gemm", "cutlass", "ampere", "sgemm", "dgrad", "wgrad", "dot_")): return "gemm_linear"
    if "triton" in n: return "triton_compiled"
    if "elementwise" in n or "reduce" in n or "norm" in n: return "elementwise_norm"
    return "other"

path = sys.argv[1] if len(sys.argv) > 1 else "/work/aupai/bench_eff/trace.json"
cats = {}
total_us = 0.0
kernel_time = {}
with open(path) as f:
    for ev in json.load(f)["traceEvents"]:
        if ev.get("cat") == "kernel" and "dur" in ev:
            c = categorize(ev["name"])
            cats[c] = cats.get(c, 0.0) + ev["dur"]
            total_us += ev["dur"]
            kernel_time[ev["name"]] = kernel_time.get(ev["name"], 0.0) + ev["dur"]

print(f"total GPU kernel time: {total_us/1e6:.1f} ms over 20 steps = {total_us/20/1000:.2f} ms/step")
print(f"{'category':<20} {'ms/step':>10} {'pct':>8}")
for c, us in sorted(cats.items(), key=lambda kv: -kv[1]):
    print(f"{c:<20} {us/20/1000:>10.2f} {100*us/total_us:>7.1f}%")
print("\ntop 25 kernels by total time:")
for name, us in sorted(kernel_time.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  {us/20/1000:>8.2f} ms/step  {name[:100]}")
