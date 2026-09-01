"""Who actually launches the 107 ms of 'elementwise / copy (aten)' kernels?

t56's group is one regex over kernel NAMES (bench_eff/t56_attr.py:39), matched after the
triton rules, so it means "every non-triton kernel with a pointwise word in its name". Those
names are ATen's op-agnostic launchers -- vectorized_elementwise_kernel says a pointwise op ran
eagerly, never which one. eff.step_remainder_attribution's own correction (d) says the
elementwise and copy rules have no correlation evidence and "their true ownership is
unverified", so the 107 ms cannot support a fusion claim as it stands: you cannot fuse a set
whose members you cannot name.

This follows each such kernel's correlation id back to the cpu_op that launched it and
histograms by launcher, which is the evidence rule (d) says is missing. Two outcomes, both
publishable: the launchers resolve and the 107 ms gets an owner list, or they do not and the
finding is that this trace cannot name it, with the reason.

Offline over the existing trace. No GPU, no run.

    python -u bench_eff/t56_elementwise_owner.py /work/aupai/bench_eff/ddp_trace_rank0.json 20
"""
import json
import re
import sys
from collections import defaultdict

ELEMENTWISE = r"elementwise|vectorized_|copy_|CatArrayBatched|index_select|fill_|unrolled"
TRITON_FIRST = r"^triton_(poi|red|per|tem|unk)"

path = sys.argv[1]
n_active = int(sys.argv[2]) if len(sys.argv) > 2 else 20

with open(path) as fh:
    ev = json.load(fh)["traceEvents"]

# Same selection as t56_attr: device-side events carrying a duration.
gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
# cpu_op events are the launcher side; ac2g / correlation links the two.
cpu = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation") and "dur" in e]

# Reproduce the group exactly: triton rules win first, so a triton fusion with a pointwise
# word in its name is NOT in this bucket.
members = [e for e in gpu
           if not re.search(TRITON_FIRST, e["name"], re.I)
           and re.search(ELEMENTWISE, e["name"], re.I)]

group_ms = sum(e["dur"] for e in members) / n_active / 1000.0
print(f"trace {path}")
print(f"elementwise/copy group: {len(members)} kernels, {group_ms:.1f} ms/step over {n_active} steps\n")

# correlation id -> the cpu_op that launched it. Torch puts it in args.correlation on both sides.
def corr(e):
    a = e.get("args") or {}
    return a.get("correlation", a.get("External id"))


cpu_by_corr = {}
for e in cpu:
    c = corr(e)
    if c is None:
        continue
    # Innermost op wins: the deepest cpu_op is the one that actually issued the kernel.
    prev = cpu_by_corr.get(c)
    if prev is None or e["dur"] < prev["dur"]:
        cpu_by_corr[c] = e

by_owner = defaultdict(float)
by_owner_n = defaultdict(int)
by_kernel = defaultdict(float)
unresolved = defaultdict(float)
for e in members:
    by_kernel[e["name"].split("(")[0][:60]] += e["dur"]
    c = corr(e)
    owner = cpu_by_corr.get(c, {}).get("name") if c is not None else None
    if owner is None:
        unresolved[e["name"].split("(")[0][:60]] += e["dur"]
    else:
        by_owner[owner] += e["dur"]
        by_owner_n[owner] += 1

res_ms = sum(by_owner.values()) / n_active / 1000.0
unres_ms = sum(unresolved.values()) / n_active / 1000.0
print(f"resolved to a cpu_op : {res_ms:6.1f} ms/step ({res_ms / group_ms * 100:4.1f}%)")
print(f"UNRESOLVED           : {unres_ms:6.1f} ms/step ({unres_ms / group_ms * 100:4.1f}%)")
print("  -> unresolved means this trace cannot name the owner; that is the finding, not a gap\n")

if by_owner:
    print("by launching cpu_op (ms/step, count):")
    for name, us in sorted(by_owner.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {us / n_active / 1000.0:7.2f} ms  n={by_owner_n[name]:6d}  {name[:88]}")

print("\nby kernel name (ms/step):")
for name, us in sorted(by_kernel.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {us / n_active / 1000.0:7.2f} ms  {name}")

if unresolved:
    print("\nunresolved kernels (ms/step):")
    for name, us in sorted(unresolved.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {us / n_active / 1000.0:7.2f} ms  {name}")

# The fusion claim needs producer-consumer adjacency, not just a name list. A launcher that
# appears once per step cannot be fused with itself; one appearing 64x is a chunk loop.
print("\nfusability read: a launcher with a high count is a loop (fusable in principle);")
print("a one-shot launcher is a single eager op (fusable only into its neighbour).")

out = {
    "probe": "t56_elementwise_owner", "trace": path, "n_active": n_active,
    "group_ms_per_step": round(group_ms, 2),
    "resolved_ms_per_step": round(res_ms, 2),
    "unresolved_ms_per_step": round(unres_ms, 2),
    "by_owner_ms_per_step": {k: round(v / n_active / 1000.0, 3)
                             for k, v in sorted(by_owner.items(), key=lambda kv: -kv[1])[:20]},
    "by_owner_count": {k: by_owner_n[k] for k, _ in sorted(by_owner.items(), key=lambda kv: -kv[1])[:20]},
    "by_kernel_ms_per_step": {k: round(v / n_active / 1000.0, 3)
                              for k, v in sorted(by_kernel.items(), key=lambda kv: -kv[1])[:12]},
}
with open("/work/aupai/runs/t56_elementwise_owner.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote runs/t56_elementwise_owner.json")
