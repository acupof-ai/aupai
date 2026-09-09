#!/usr/bin/env python3
"""CSA step-speed probe vs gated-MLA: is the v2 first arm launchable on the reference?

4c GO, 2026-09-09. One number each, same card, same shape:
  control:   GatedMLA, csa=False (the latent-MLA path every shipped arm runs)
  treatment: GatedMLA, csa=True, csa_compress=4, csa_topk=512, csa_window=128
             (the v2 spec's CSA-with-SWA attention, prereg v2_loop_moe_csa_0908)

Shape: d=1024 heads=8 (hd=128, latent=256), B=16 T=4096, bf16, packed cu path
with one document per row (cu = [0, T, 2T, ...]). Forward+backward, alternating
treatment/control, median of 20 after 3 warmup, torch.cuda.synchronize around
every timed region. On OOM at B=16 the probe falls back to B=4 then B=1 and
reports the batch it actually ran -- a different batch is a different number,
not a rescue.

Decision rule is in the exp row's hypothesis, written before this ran.
"""
import statistics
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, "/work/aupai")
from model import GatedMLA

D, H, T = 1024, 8, 4096
N_WARM, N_TIMED = 3, 20
DEV = "cuda:0"


def cfg(csa):
    return SimpleNamespace(
        d=D, heads=H, csa=csa, csa_compress=4, csa_topk=512, csa_window=128,
    )


def run(batch):
    torch.manual_seed(0)
    x = torch.randn(batch, T, D, device=DEV, dtype=torch.bfloat16)
    cu = torch.arange(0, (batch + 1) * T, T, device=DEV, dtype=torch.int32)
    mods = {
        "control": GatedMLA(cfg(False)).to(DEV).bfloat16(),
        "csa": GatedMLA(cfg(True)).to(DEV).bfloat16(),
    }
    for m in mods.values():
        m.train()

    def step(m):
        for p in m.parameters():
            p.grad = None
        y = m(x, cu)
        y.sum().backward()

    for _ in range(N_WARM):
        step(mods["control"])
        step(mods["csa"])
    torch.cuda.synchronize()

    times = {k: [] for k in mods}
    for _ in range(N_TIMED):
        for k, m in mods.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step(m)
            torch.cuda.synchronize()
            times[k].append((time.perf_counter() - t0) * 1000)
    return times


def main():
    for batch in (16, 4, 1):
        try:
            times = run(batch)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at B={batch}, falling back", flush=True)
            torch.cuda.empty_cache()
    else:
        print("OOM at B=1; nothing measured")
        sys.exit(1)

    print(f"shape: B={batch} T={T} D={D} heads={H} bf16, packed cu, fwd+bwd, "
          f"median of {N_TIMED} after {N_WARM} warmup, alternating")
    med = {}
    for k, ts in times.items():
        med[k] = statistics.median(ts)
        print(f"{k:8s} median {med[k]:9.2f} ms  mean {statistics.mean(ts):9.2f} "
              f"min {min(ts):9.2f}  tok/s {batch * T / (med[k] / 1000):,.0f}")
    ratio = med["csa"] / med["control"]
    print(f"ratio csa/control = {ratio:.3f}")


if __name__ == "__main__":
    main()
