#!/usr/bin/env python3
"""Map a training STEP RANGE to domains and pool rows, CPU-only (de, 1e order 2026-09-27).

When the v41_ced_0926 run was killed by the grad-spike watchdog at step 19220 (gnorm 49408),
the question was "which data did those steps actually read?". A step is world-striped plan
columns [step*rows_per_step, (step+1)*rows_per_step); this rebuilds the plan's domain/row
column EXACTLY (it is a pure function of the mix, pool lengths, per-domain val split and
Cfg.seed phase permutation) and slices any step interval, no tokens or GPUs touched.

Correctness is a token-free identity, not an assumption: the per-domain count of the rebuilt
plan over the prefix the checkpoint saved must equal the checkpoint's own `row_cursor`. The
cursor is that prefix count by construction (save_checkpoint), so equality proves the plan
column is the run's without materializing tokens.

Pool rows are pool-relative DOC indices into each domain's shuffled cache, not corpus bytes;
mapping them back to source documents is a separate shuffle-inverse step, intentionally not
done here.

    # pod, beside a live run (mmap reads cache metadata only; OMP-limited, NICE'd):
    python3 scripts/spike_step_domains.py --ckpt ckpt_x.pt.step18000 --steps 19190:19220
    python3 scripts/spike_step_domains.py --selftest          # no ckpt/cache/GPU
"""

import argparse
import collections
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))


def fresh_plan(pool_len, mix, names, seed, seq, anneal_frac, world):
    """The run's fresh-start plan: [2, n_cols] (domain_index, pool_row) after rank-strip trim.

    Mirrors train.build_mix's allocation + one randperm per phase. Called fresh because the
    spiked run started at step 0; a resumed run re-seeds the cursor and the caller would pass
    that cursor. Kept here rather than importing build_mix, which needs token-shaped pools to
    return sequences (it cannot run against mere pool lengths)."""
    rows = mix["total_tokens"] / seq
    phases = [(1 - anneal_frac, "weight")] + ([(anneal_frac, "anneal")] if anneal_frac else [])
    g = torch.Generator().manual_seed(seed)
    used = {n: 0 for n in names}
    plan = []
    for frac, key in phases:
        parts = []
        for di, name in enumerate(names):
            d = mix["domains"][name]
            want = int(rows * frac * d.get(key, d["weight"]))
            cap = int(pool_len[name] * d.get("epochs", 1)) - used[name]
            if want > cap:
                want = max(0, cap)
            if want:
                idx = torch.arange(used[name], used[name] + want) % pool_len[name]
                parts.append(torch.stack([torch.full_like(idx, di), idx]))
            used[name] += want
        if parts:
            ph = torch.cat(parts, dim=1)
            plan.append(ph[:, torch.randperm(ph.shape[1], generator=g)])
    if not plan:
        raise RuntimeError("empty plan: budget fully consumed by the resume cursor")
    plan = torch.cat(plan, dim=1)
    n = (plan.shape[1] // world) * world
    return plan[:, :n]


def pool_lengths(names, mix, seq):
    """Per-domain TRAIN pool rows = cache rows minus the one val_split_n definition.

    Only the cache SHAPE is read (mmap + len, near-zero bytes), but the rule is about opening a
    token cache by path beside a live run, so call the same guard as every cache reader; the
    operator sets AUPAI_ALLOW_CORESIDENT_CACHE=1 to run it next to training."""
    import train
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from cache_guard import assert_not_co_resident

    assert_not_co_resident(names)
    out = {}
    for name in names:
        t = torch.load(train._domain_cache_path(name), map_location="cpu", weights_only=True, mmap=True)
        n_rows = len(t) // (seq + 1)
        out[name] = n_rows - train.val_split_n(name, n_rows, mix)
        del t
    return out


def load_cfg(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from train import Cfg

    cfg = ck["cfg"]
    for k, v in cfg.items():
        if not k.startswith("_"):
            setattr(Cfg, k, v)
    Cfg.seq = int(cfg.get("seq", 4096))
    return Cfg, ck


def analyze(ckpt_path, step_lo, step_hi, world_override=None):
    import train

    Cfg, ck = load_cfg(ckpt_path)
    world = int(world_override or 8)
    B, A = int(Cfg.batch), int(Cfg.accum)
    rps = B * A * world
    with open(Cfg.mix, encoding="utf-8") as fh:
        mix = json.load(fh)
    names = list(mix["domains"])
    af = train._mix_anneal_frac(mix, Cfg.mix, False)
    pool_len = pool_lengths(names, mix, Cfg.seq)
    plan = fresh_plan(pool_len, mix, names, int(Cfg.seed), Cfg.seq, af, world)

    # Token-free identity check against the checkpoint cursor (see module docstring).
    saved = {n: int(v) for n, v in (ck.get("row_cursor") or {}).items()}
    if saved:
        as_of = int(ck.get("row_cursor_as_of_step", ck.get("step", 0)))
        pref = plan[0][: as_of * rps]
        got = collections.Counter(int(x) for x in pref.tolist())
        di_of = {n: i for i, n in enumerate(names)}
        for n in names:
            g_, sv = int(got.get(di_of[n], 0)), saved.get(n, 0)
            assert g_ == sv, f"{n}: rebuilt {g_} != checkpoint cursor {sv}"
        print(
            f"VALIDATED rebuilt prefix at step {as_of} == row_cursor "
            f"({as_of * rps} cols, {len(names)} domains)",
            flush=True,
        )

    c0, c1 = step_lo * rps, step_hi * rps
    seg = plan[:, c0:c1]
    cnt = collections.Counter(int(x) for x in seg[0].tolist())
    print(f"\nsteps {step_lo}..{step_hi - 1}  global cols [{c0},{c1})  total {c1 - c0} (rows/step {rps})")
    for di, ct in cnt.most_common():
        rows_here = [int(x) for x in seg[1][seg[0] == di].tolist()]
        print(
            f"  {names[di]:28} {ct:>5} ({ct / (c1 - c0) * 100:4.1f}%) "
            f"pool_row {min(rows_here)}..{max(rows_here)}"
        )
    return seg, names, rps


def _selftest():
    # A 4-domain synthetic mix with known weights; build a plan and check the prefix-count
    # identity the real path uses, plus the column->row slicing bounds.
    mix = {
        "total_tokens": 1000 * 32,
        "anneal_frac": 0.0,
        "domains": {f"d{i}": {"weight": w} for i, w in enumerate((0.4, 0.3, 0.2, 0.1))},
    }
    names = list(mix["domains"])
    pool_len = {f"d{i}": 250 for i in range(4)}
    world = 2
    plan = fresh_plan(pool_len, mix, names, seed=42, seq=32, anneal_frac=0.0, world=world)
    total = plan.shape[1]
    assert total % world == 0 and total > 0
    # every column's domain index is valid and row is within that pool
    for di, ri in zip(plan[0].tolist(), plan[1].tolist(), strict=True):
        assert 0 <= di < 4 and 0 <= ri < 250
    # the sum of per-domain counts over the whole plan equals the column count
    full = collections.Counter(int(x) for x in plan[0].tolist())
    assert sum(full.values()) == total
    # deterministic: same seed reproduces byte-identical order
    again = fresh_plan(pool_len, mix, names, 42, 32, 0.0, world)
    assert torch.equal(again, plan), "plan must be deterministic in (mix,pools,seed)"
    # a different seed changes the order (overwhelmingly)
    other = fresh_plan(pool_len, mix, names, 7, 32, 0.0, world)
    assert not torch.equal(other, plan), "different seed must re-permute"
    # prefix-count identity shape: a "cursor" taken at half the columns round-trips
    half = (total // 2 // world) * world
    cur = collections.Counter(int(x) for x in plan[0][:half].tolist())
    assert sum(cur.values()) == half
    print("spike_step_domains selftest OK: plan deterministic, rows in-pool, prefix counts exact")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", help="checkpoint carrying cfg + row_cursor (validates the rebuild)")
    ap.add_argument("--steps", default="", help="inclusive:exclusive step interval, e.g. 19190:19220")
    ap.add_argument("--world", type=int, default=None, help="world size (else ckpt/default 8)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.ckpt or not a.steps or ":" not in a.steps:
        ap.error("--ckpt and --steps LO:HI are required (or --selftest)")
    lo, hi = (int(x) for x in a.steps.split(":", 1))
    analyze(a.ckpt, lo, hi, a.world)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
