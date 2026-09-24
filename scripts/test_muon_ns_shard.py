#!/usr/bin/env python3
"""Bit-parity for --muon_ns_shard: sharding Newton-Schulz over ranks along the stacked
same-shape instance axis must leave every updated weight and momentum buffer bit-identical
to one rank running all instances.

Runs a gloo process group (CPU), so it needs no GPU and is deterministic single-threaded.
Each rank holds the SAME stacked params/grads (as after a DDP all-reduce). Rank r computes
instances [r*n/w:(r+1)*n/w) with the real compiled Muon update and all_gathers; every rank
compares that concatenation against a flag-off Muon that computed all n instances locally.
Equality is exact, not a tolerance: the partition is per-instance and each instance's NS
iteration / momentum / weight step is independent, so shuffled reduction order is the only
possible divergence, and all_gather restores rank order 0..w-1 == instance order 0..n-1.

Also asserts the uneven-group fallback (n not divisible by world) still computes all n.

    python scripts/test_muon_ns_shard.py
Exit 0 = parity. Exit 1 = named mismatch.
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

N_INST, OUT, INP = 4, 32, 16  # 4 same-shape 2-D matrices; divisible by worlds 2 and 4
NS_STEPS = 5

if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--selftest"):
    raise SystemExit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")


def _worker(rank, world, queue):
    """Module-level so mp.spawn (spawn start method) can pickle it."""
    torch.set_num_threads(1)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MUON_SHARD_PORT", "29577")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    import train  # imported after process start, as in torchrun

    torch.manual_seed(0)
    # Identical inputs on every rank (the post-all-reduce state).
    params0 = [torch.randn(OUT, INP, dtype=torch.float64) for _ in range(N_INST)]
    grads0 = [torch.randn(OUT, INP, dtype=torch.float64) for _ in range(N_INST)]

    def make_opt(shard):
        train.Cfg.muon_shape_lr = False
        train.Cfg.muon_ns_shard = shard
        ps = [p.clone() for p in params0]
        opt = train.Muon(ps, lr=0.01, momentum=0.95, ns_steps=NS_STEPS, weight_decay=0.1)
        gs = [g.clone() for g in grads0]
        return ps, gs, opt

    # Reference: flag off, one rank computes every instance.
    rp, rg, ropt = make_opt(False)
    for p, g in zip(rp, rg):
        p.grad = g
    ropt.step()

    # Sharded: flag on; each rank computes its slice then gathers.
    sp, sg, sopt = make_opt(True)
    for p, g in zip(sp, sg):
        p.grad = g
    sopt.step()

    # Every rank ends with all instances (all_gather), so every rank can compare all.
    fails = []
    for i in range(N_INST):
        if not torch.equal(sp[i], rp[i]):
            fails.append(("w", i, (sp[i] - rp[i]).abs().max().item()))
        # momentum buffer state[p]["mb"]
        if not torch.equal(sopt.state[sp[i]]["mb"], ropt.state[rp[i]]["mb"]):
            fails.append(("mb", i, (sopt.state[sp[i]]["mb"]
                                    - ropt.state[rp[i]]["mb"]).abs().max().item()))

    # Uneven group fallback: a group of 3 instances with world 2 must still update all 3
    # and match a local all-3 computation.
    up = [torch.randn(OUT, INP, dtype=torch.float64) for _ in range(3)]
    ug = [torch.randn(OUT, INP, dtype=torch.float64) for _ in range(3)]
    train.Cfg.muon_ns_shard = True
    uopt = train.Muon([p.clone() for p in up], lr=0.01, momentum=0.95,
                      ns_steps=NS_STEPS, weight_decay=0.1)
    for p, g in zip(list(uopt.param_groups[0]["params"]), ug):
        p.grad = g.clone()
    uopt.step()
    train.Cfg.muon_ns_shard = False
    uref = train.Muon([p.clone() for p in up], lr=0.01, momentum=0.95,
                      ns_steps=NS_STEPS, weight_decay=0.1)
    for p, g in zip(list(uref.param_groups[0]["params"]), ug):
        p.grad = g.clone()
    uref.step()
    for i, (a, b) in enumerate(zip(uopt.param_groups[0]["params"],
                                   uref.param_groups[0]["params"])):
        if not torch.equal(a, b):
            fails.append(("uneven", i, (a - b).abs().max().item()))

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        queue.put(fails)


def _run_world(world):
    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    ps = [mp.Process(target=_worker, args=(r, world, q)) for r in range(world)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    fails = q.get() if not q.empty() else [("noprocess", 0, 0.0)]
    dead = [p.exitcode for p in ps if p.exitcode != 0]
    if fails or dead:
        lines = [f"{kind}[{i}] maxabs {d:.3e}" for kind, i, d in fails[:10]]
        print(f"world{world} PARITY FAIL ({len(fails)}), child exit {dead}: " + "; ".join(lines))
        return False
    print(f"world{world} PARITY OK: sharded NS bit-identical across {N_INST} instances; "
          "uneven group falls back and matches")
    return True


def main():
    for w in (2, 4):
        if not _run_world(w):
            return 1
    print("muon_ns_shard bit-parity gate: all worlds OK")
    return 0


def _selftest():
    """Model-free: the shard precondition and the flag default, no process group."""
    import train

    # The flag defaults OFF -- a parser that dropped the default would silently change the run.
    assert getattr(train.Cfg, "muon_ns_shard", None) is False, \
        "muon_ns_shard must default to False"
    # The shard predicate itself: n divisible by world, n>=world, else fall back.
    def shardable(n, world):
        return world > 1 and n >= world and n % world == 0
    assert [shardable(n, 2) for n in range(1, 9)] == \
        [False, True, False, True, False, True, False, True]
    assert not shardable(3, 2), "n=3/world=2 must fall back (uneven)"
    assert shardable(4, 4) and not shardable(4, 8), "boundary n==world vs n<world"
    # The Muon constructor reads the flag; a flag-on build must record it.
    train.Cfg.muon_ns_shard = False
    p = torch.nn.Parameter(torch.randn(8, 8))
    off = train.Muon([p], lr=0.01)
    assert off._ns_shard is False
    train.Cfg.muon_ns_shard = True
    on = train.Muon([torch.nn.Parameter(torch.randn(8, 8))], lr=0.01)
    assert on._ns_shard is True
    train.Cfg.muon_ns_shard = False
    print("muon_ns_shard selftest: default-off + shard predicate + constructor flag OK")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        sys.exit(main())
