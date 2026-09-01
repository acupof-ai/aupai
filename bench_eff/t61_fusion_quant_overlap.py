"""Is the inductor-fusion group partly the same quantisation work as the fp8 tax? (03's question)

Inductor names a fused kernel after every op it absorbed, so a fusion whose name mentions `abs`
or `_scaled_mm` may be quantisation, or may be model arithmetic that merely touched a quantised
tensor. Names cannot separate those, and that exact ambiguity already inverted one ranking today.

Two findings, and the second is the one that matters:

1. THE GROUPS ARE DISJOINT AS KERNEL SETS. t56's group excludes anything matching `^triton_`
   before the pointwise rules run, so no kernel is in both. 511.88 + 250.61 double-counts nothing.

2. THE TRACE HAS THE FP8 HEAD ON. 181 aten::_scaled_mm/step sit inside a Liger FLCE region, and
   the only code that puts _scaled_mm there is patch_liger_flce_fp8 (train.py:488), reachable
   only under FP8_HEAD=1 (train.py:2143) -- which is no-ship at -3.91% and absent from the live
   run. So 156.9 of the elementwise group's 250.6 ms is head work that does not run in
   production, including 79.2 of aten::div's 84.0 and 60.4 of copy_'s 67.2.

The join is NOT the one that resolved the elementwise group: aten kernels launch through
`cuda_runtime` (cudaLaunchKernel), triton kernels through `cuda_driver` (cuLaunchKernel). Keying
only on cuda_runtime resolves 0.00% of triton -- a null that reads like a real negative. Build
the correlation map from both categories; then it is 99.98%.

    python3 t61_fusion_quant_overlap.py <trace.json> <n_active>
"""
import bisect
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TRITON = r"^triton_(poi|red|per|tem|unk)"
ELEMENTWISE = r"elementwise|vectorized_|copy_|CatArrayBatched|index_select|fill_|unrolled"
# _fp8_mm's signature: abs().amax().clamp() computes the scale, then (x/s).to(fp8) materialises.
# A fused region implementing it must absorb BOTH halves; one alone is ambiguous.
SCALE_OP, CAST_OP = r"(abs|amax|clamp|reciprocal)", r"(_to_copy|_scaled_mm|to_float8)"

with open(path) as fh:
    ev = json.load(fh)["traceEvents"]
gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
cpu = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation") and "dur" in e]

c2x, c2l = {}, {}
for e in ev:
    if e.get("cat") in ("cuda_runtime", "cuda_driver"):
        a = e.get("args") or {}
        c, x = a.get("correlation"), a.get("External id")
        if c is None:
            continue
        c2l[c] = e
        if x is not None:
            c2x[c] = x

x2op = {}
for e in cpu:
    x = (e.get("args") or {}).get("External id")
    if x is None:
        continue
    prev = x2op.get(x)
    if prev is None or e["dur"] < prev["dur"]:
        x2op[x] = e

# Ancestry by timestamp containment on the launching thread: the External-id map gives the
# innermost op, and "is this under Liger FLCE" needs the enclosing region, not the leaf.
bytid = defaultdict(list)
for e in cpu:
    bytid[e["tid"]].append(e)
for v in bytid.values():
    v.sort(key=lambda o: o["ts"])
starts = {t: [o["ts"] for o in v] for t, v in bytid.items()}


def under_liger(launch):
    v = bytid.get(launch["tid"])
    if not v:
        return False
    i = bisect.bisect_right(starts[launch["tid"]], launch["ts"])
    return any(o["ts"] <= launch["ts"] <= o["ts"] + o["dur"] and "Liger" in o["name"]
               for o in v[max(0, i - 6000):i])


def group_of(name):
    if re.match(TRITON, name):
        return "fusion-quant" if (re.search(SCALE_OP, name) and re.search(CAST_OP, name)) else "fusion-other"
    if re.search(ELEMENTWISE, name, re.I):
        return "elementwise"
    if re.search(r"nvjet_qqtst|_scaled_mm|cutlass.*f8", name, re.I):
        return "fp8 GEMM"
    return None


head, body, owners = defaultdict(float), defaultdict(float), defaultdict(lambda: [0.0, 0.0])
tri_ms = ew_ms = 0.0
for e in gpu:
    nm = e["name"].split("(")[0]
    g = group_of(nm)
    if g is None:
        continue
    if g.startswith("fusion"):
        tri_ms += e["dur"]
    elif g == "elementwise":
        ew_ms += e["dur"]
    c = (e.get("args") or {}).get("correlation")
    launch = c2l.get(c)
    is_head = bool(launch and under_liger(launch))
    (head if is_head else body)[g] += e["dur"]
    if g == "elementwise":
        op = x2op.get(c2x.get(c)) if c is not None else None
        if op:
            owners[op["name"]][0 if is_head else 1] += e["dur"]

print(f"trace {path}, n_active={N}")
print(f"fusion group {tri_ms / N / 1000:.2f} ms/step   elementwise group {ew_ms / N / 1000:.2f} ms/step")
print("the two groups share 0 kernels: t56 matches ^triton_ first, so no double count\n")

print(f"{'group':16s} {'HEAD (FP8_HEAD=1)':>19s} {'BODY (live)':>14s}")
for g in ("fusion-quant", "fusion-other", "elementwise", "fp8 GEMM"):
    print(f"{g:16s} {head[g] / N / 1000:16.2f} ms {body[g] / N / 1000:11.2f} ms")
H, B = sum(head.values()), sum(body.values())
print(f"{'TOTAL':16s} {H / N / 1000:16.2f} ms {B / N / 1000:11.2f} ms")
print(f"\nhead is {H / (H + B) * 100:.1f}% of this work, and FP8_HEAD is absent from the live run")

print("\nelementwise owners, split by whether they sit under Liger FLCE:")
for k, (h, b) in sorted(owners.items(), key=lambda kv: -sum(kv[1]))[:8]:
    print(f"  {k:16s} head {h / N / 1000:7.2f} ms   body {b / N / 1000:7.2f} ms")

out = {"probe": "t61_fusion_quant_overlap", "trace": path, "n_active": N,
       "fusion_ms_per_step": round(tri_ms / N / 1000, 2),
       "elementwise_ms_per_step": round(ew_ms / N / 1000, 2),
       "groups_share_kernels": 0,
       "head_ms_per_step": {k: round(v / N / 1000, 2) for k, v in head.items()},
       "body_ms_per_step": {k: round(v / N / 1000, 2) for k, v in body.items()},
       "elementwise_owner_head_body": {k: [round(h / N / 1000, 2), round(b / N / 1000, 2)]
                                       for k, (h, b) in sorted(owners.items(), key=lambda kv: -sum(kv[1]))[:8]}}
with open("/work/aupai/runs/t61_fusion_quant_overlap.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("\nwrote runs/t61_fusion_quant_overlap.json")
