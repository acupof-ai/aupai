#!/usr/bin/env python3
"""Dry-run the data schedule against the token caches that exist, before burning GPU hours on it.

Reads the size of each tokens_<domain>.pt (from its file size, not by loading 36GB) and prints what
build_mix will actually do: rows per phase, epochs per domain, which domains get capped, the anneal
composition, and the resulting step count. Run this after scripts/pretokenize.py and before launching.

    python scripts/check_mix.py [--mix data/mix_scale_3.24b.json] [--batch 32] [--world 8]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import harness  # single source of truth for the configured mix

import train  # noqa: E402
from data_overview import cache_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    # The configured mix (harness.cfg_default), not a hardcoded name: a hardcoded path
    # goes stale when the mix file is replaced.
    ap.add_argument("--mix", default=os.path.join(ROOT, harness.cfg_default("mix")))
    ap.add_argument("--batch", type=int, default=train.Cfg.batch)
    ap.add_argument("--world", type=int, default=8)
    a = ap.parse_args()

    mix = json.load(open(a.mix, encoding="utf-8"))
    seq, af = train.Cfg.seq, train.Cfg.anneal_frac
    total_rows = mix["total_tokens"] / seq
    missing = [d for d in mix["domains"] if cache_tokens(d) is None]
    if missing:
        print(f"MISSING token caches: {missing} -- run scripts/pretokenize.py first")
        return 1

    pools, used, got = {}, {}, {}
    print(f"{'domain':<7}{'pool tok':>11}{'pool rows':>11}")
    for d in mix["domains"]:
        rows = cache_tokens(d) // (seq + 1)
        n_val = min(max(1, int(rows * train.Cfg.val_frac)), train.Cfg.val_rows_max)
        pools[d] = rows - n_val
        used[d] = 0
        got[d] = {"weight": 0, "anneal": 0}
        print(f"{d:<7}{cache_tokens(d) / 1e9:>10.2f}B{pools[d]:>11}")

    capped = []
    for frac, key in ((1 - af, "weight"), (af, "anneal")):
        for d, c in mix["domains"].items():
            want = int(total_rows * frac * c.get(key, c["weight"]))
            cap = int(pools[d] * c.get("epochs", 1)) - used[d]
            if want > cap:
                capped.append(f"{d}/{key}: wanted {want} rows, cap left {cap}")
                want = max(0, cap)
            used[d] += want
            got[d][key] = want

    print(
        f"\n{'domain':<7}{'main':>9}{'anneal':>9}{'total':>9}{'epochs':>8}{'cap':>5}"
        f"{'main %':>8}{'anneal %':>9}"
    )
    main_tot = sum(g["weight"] for g in got.values())
    ann_tot = sum(g["anneal"] for g in got.values())
    for d in mix["domains"]:
        g = got[d]
        n = g["weight"] + g["anneal"]
        print(
            f"{d:<7}{g['weight'] * seq / 1e9:>8.2f}B{g['anneal'] * seq / 1e9:>8.2f}B"
            f"{n * seq / 1e9:>8.2f}B{n / max(pools[d], 1):>8.2f}{mix['domains'][d]['epochs']:>5}"
            f"{g['weight'] / max(main_tot, 1):>7.1%}{g['anneal'] / max(ann_tot, 1):>8.1%}"
        )

    sched = main_tot + ann_tot
    per_rank = (sched // a.world * a.world) // a.world
    steps = per_rank // a.batch
    print(
        f"\nscheduled {sched * seq / 1e9:.2f}B tokens of the {mix['total_tokens'] / 1e9:.2f}B asked for"
        f"  ({sched} rows, {per_rank} per rank, {steps} steps at batch {a.batch} x {a.world})"
    )
    if capped:
        print("capped by the epoch limit:")
        for c in capped:
            print("  " + c)
    if sched * seq < 0.9 * mix["total_tokens"]:
        print(
            "WARNING: the schedule is more than 10% short of total_tokens -- raise an epoch cap "
            "or lower total_tokens so the LR schedule matches the data actually delivered"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
