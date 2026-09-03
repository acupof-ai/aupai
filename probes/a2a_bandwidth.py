"""Measured effective bandwidth of dist.all_to_all_single at MoE dispatch size, 2 cards.

The one number facts/smelt_deeploop.json#repo.moe_a2a_cost_h20 lacks: every figure there is
bytes/link-bandwidth with no efficiency factor, which makes it an upper bound on throughput
and a lower bound on cost. At L32 that bound spans 4.35% to 9.9% of a step depending on the
factor, and the decision falls inside the span.

Scope is deliberately one measurement: this size, this collective, 2 ranks. It does not sweep
world size, does not model MoE, and does not touch allreduce.

Two ranks is the minimum that crosses a link, and the reason there is no single-card version:
two ranks on one device is a device-to-device copy and measures no interconnect at all.
"""

import json
import os
import sys
import time

import torch
import torch.distributed as dist

TOKENS = 65536  # tokens per micro-batch per card (131072 tok/step / accum 2)
HIDDEN = 836  # d 1024 / sqrt(1.5), the width matched FLOPs force (repo.smelt_shape_correction)
LINK_GBPS = 450.0  # NV18 measured: 18 lanes x 25 GB/s per direction (repo.nv18_topology_measured)
ITERS = 20
WARMUP = 5


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    if TOKENS % world:
        raise SystemExit(f"TOKENS {TOKENS} must divide by world {world}")
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)

    # Bytes that actually LEAVE this card: each rank keeps its own 1/world share.
    out_bytes = (TOKENS // world) * (world - 1) * HIDDEN * x.element_size()

    for _ in range(WARMUP):
        dist.all_to_all_single(out, x)
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_to_all_single(out, x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    med = times[len(times) // 2]
    gbps = out_bytes / (med / 1e3) / 1e9
    pct = gbps / LINK_GBPS * 100

    if rank == 0:
        print(
            f"all_to_all_single  world={world}  {TOKENS}x{HIDDEN} bf16 "
            f"({x.numel() * x.element_size() / 1e6:.1f} MB tensor, {out_bytes / 1e6:.1f} MB off-card)"
        )
        print(f"  median {med:.3f} ms   min {times[0]:.3f}   max {times[-1]:.3f}")
        print(f"  achieved {gbps:.1f} GB/s = {pct:.1f}% of the {LINK_GBPS:.0f} GB/s link")
        print(
            json.dumps(
                {
                    "world": world,
                    "tokens": TOKENS,
                    "hidden": HIDDEN,
                    "off_card_mb": round(out_bytes / 1e6, 1),
                    "median_ms": round(med, 3),
                    "min_ms": round(times[0], 3),
                    "max_ms": round(times[-1], 3),
                    "achieved_gbps": round(gbps, 1),
                    "pct_of_link": round(pct, 1),
                }
            )
        )
    dist.destroy_process_group()


def selftest():
    """The measurement is a division; assert it on a known answer, since a wrong constant
    here reports a wrong efficiency and nothing else would notice.

    A second vacuity, found the same way: the shape constants were re-typed as locals here, so
    editing HIDDEN from 836 to 1024 -- the constant the measurement actually uses -- left this
    green. Every value under test is read from the module, never restated."""
    world = 2
    assert abs(LINK_GBPS - 450.0) < 1e-9, f"LINK_GBPS is {LINK_GBPS}, not the measured NV18 450"
    assert HIDDEN == 836, f"HIDDEN is {HIDDEN}, not the 836 matched FLOPs force at d1024"
    assert TOKENS == 65536, f"TOKENS is {TOKENS}, not 131072/accum 2"
    off = (TOKENS // world) * (world - 1) * HIDDEN * 2
    assert off == 54788096, off
    # LITERAL times, so LINK_GBPS is on one side of the comparison only.
    # 54788096 B / 450e9 B/s = 0.12175 ms, and NV18's 450 GB/s is what the fact records.
    for ms, want in ((0.121751, 100.0), (0.243502, 50.0), (0.608756, 20.0)):
        gbps = off / (ms / 1e3) / 1e9
        pct = gbps / LINK_GBPS * 100
        assert abs(pct - want) < 0.05, f"{ms} ms read as {pct:.2f}%, expected {want}%"
    # at world=2 exactly half the tensor leaves the card; at world=4, three quarters
    assert (TOKENS // 4) * 3 * HIDDEN * 2 == off * 1.5
    print(
        f"selftest OK: {off / 1e6:.1f} MB off-card at world=2; 0.1218/0.2435/0.6088 ms "
        f"read as 100/50/20% of the 450 GB/s link (literal times, so the constant is tested)"
    )


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif os.environ.get("RANK") is None and "LOCAL_RANK" not in os.environ:
        raise SystemExit("launch with torchrun --nproc_per_node=2 (or --selftest)")
    else:
        main()
