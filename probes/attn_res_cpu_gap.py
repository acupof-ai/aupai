#!/usr/bin/env python3
"""Is the AttnRes end-to-end cost GPU kernel time, or CPU / launch-gap time?

Both arms in ONE process, ONE session, ONE config, so nothing is stitched across runs:
    1. wall clock per step, from CUDA events (includes every gap the GPU sits idle)
    2. sum of CUDA kernel self-time, from torch.profiler
    3. GPU idle = wall - kernel time.  If the arms differ in wall but not in kernel time,
       the cost is CPU-side; if they differ in both, it is on the GPU.
    4. CPU self-time by op, so the Python/dispatch side is visible either way.

    CUDA_VISIBLE_DEVICES=7 python -u scripts/attn_res_cpu_gap.py --batch 16
"""

import argparse
import gc
import os
import sys
from collections import defaultdict

import torch
from torch.autograd import DeviceType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402


def build(cfg, attn_res, blocks, fp8, compile_):
    cfg.attn_res = attn_res
    cfg.attn_res_blocks = blocks
    model = train.HybridLM(cfg).to("cuda").to(torch.bfloat16)
    if fp8:
        train.convert_to_fp8_compute(model)
    return torch.compile(model, dynamic=False) if compile_ else model


def run_arm(name, cfg, model, x, y, warmup, steps, top):
    flce = train.LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=train.SOFTCAP)
    raw = getattr(model, "_orig_mod", model)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

    def step():
        ev[0].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, _ = model(x, y)
        w = raw.head.weight[: cfg.vocab]
        loss = flce(w, hidden.to(w.dtype).reshape(-1, hidden.shape[-1]), y.reshape(-1))
        ev[1].record()
        loss.backward()
        ev[2].record()
        for p in model.parameters():
            p.grad = None
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2])

    for _ in range(warmup):
        step()
    fwd = bwd = 0.0
    for _ in range(steps):
        f, b = step()
        fwd += f
        bwd += b
    fwd, bwd = fwd / steps, bwd / steps
    print(f"[{name}] wall fwd {fwd:.2f}  bwd {bwd:.2f}  fwd+bwd {fwd + bwd:.2f} ms/step", flush=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    # Only DeviceType.CUDA rows are real kernels. An `aten::mm` row also carries device time, but
    # that is the SAME time attributed to its parent op -- summing both double-counts every
    # aten-dispatched kernel while leaving inductor's triton kernels counted once.
    kern_ev = [e for e in ka if e.device_type == DeviceType.CUDA]
    gpu = sum(e.self_device_time_total for e in kern_ev) / steps / 1000
    cpu = sum(e.self_cpu_time_total for e in ka) / steps / 1000
    launches = sum(e.count for e in kern_ev) / steps
    print(
        f"[{name}] kernel {gpu:.2f} ms/step   cpu-self {cpu:.2f} ms/step   "
        f"launches {launches:.0f}/step   idle(wall-kernel) {fwd + bwd - gpu:.2f} ms",
        flush=True,
    )
    kern = defaultdict(float)
    kcount = defaultdict(float)
    for e in kern_ev:
        if e.self_device_time_total > 0:
            kern[e.key] += e.self_device_time_total / steps / 1000
            kcount[e.key] += e.count / steps
    print(f"[{name}] top {top} kernels (ms/step, launches/step):", flush=True)
    for k, v in sorted(kern.items(), key=lambda kv: -kv[1])[:top]:
        print(f"    {v:8.2f} {kcount[k]:7.0f}x  {k[:92]}", flush=True)
    cops = sorted(ka, key=lambda e: -e.self_cpu_time_total)[:8]
    print(f"[{name}] top CPU ops (self ms/step):", flush=True)
    for e in cops:
        print(
            f"    {e.self_cpu_time_total / steps / 1000:8.2f} {e.count / steps:7.0f}x  {e.key[:70]}",
            flush=True,
        )
    return dict(fwd=fwd, bwd=bwd, gpu=gpu, cpu=cpu, launches=launches, kern=kern, kcount=kcount)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--no_fp8", action="store_true")
    ap.add_argument("--arms", default="off,full", help="comma list of off|n2|noscale|full")
    a = ap.parse_args()

    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = 256
    cfg = train.Cfg
    cfg.batch = a.batch
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    x = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq), device="cuda")
    y = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq), device="cuda")

    res = {}
    for arm in a.arms.split(","):
        if arm == "noscale":
            one = {}

            def of(v, _one=one):
                k = (v.shape[0], v.shape[1], v.dtype, v.device)
                if k not in _one:
                    _one[k] = torch.ones(v.shape[0], v.shape[1], 1, dtype=v.dtype, device=v.device)
                return train.Source(v, _one[k])

            train.Source.of = staticmethod(of)
        model = build(cfg, arm != "off", 1 if arm == "n2" else 0, not a.no_fp8, True)
        res[arm] = run_arm(arm, cfg, model, x, y, a.warmup, a.steps, a.top)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    arms = list(res)
    base = res[arms[0]]
    for arm in arms[1:]:
        r = res[arm]
        print(
            f"\nDELTA {arm}-{arms[0]}: wall {r['fwd'] + r['bwd'] - base['fwd'] - base['bwd']:+.2f} ms "
            f"(fwd {r['fwd'] - base['fwd']:+.2f}, bwd {r['bwd'] - base['bwd']:+.2f})  "
            f"kernel {r['gpu'] - base['gpu']:+.2f} ms  cpu-self {r['cpu'] - base['cpu']:+.2f} ms  "
            f"launches {r['launches'] - base['launches']:+.0f}",
            flush=True,
        )
        d = defaultdict(float)
        for k, v in r["kern"].items():
            d[k] += v
        for k, v in base["kern"].items():
            d[k] -= v
        print(f"kernels that grew most in {arm}:", flush=True)
        for k, v in sorted(d.items(), key=lambda kv: -kv[1])[: a.top]:
            print(
                f"    {v:+8.2f} ms  {r['kcount'].get(k, 0) - base['kcount'].get(k, 0):+7.0f}x  {k[:88]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
