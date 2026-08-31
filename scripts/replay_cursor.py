#!/usr/bin/env python3
"""Reconstruct a row cursor for a checkpoint that predates the field (de-7).

The stage-1 checkpoint carries no `row_cursor`, so a stage-2 resume would restart
every domain at row 0 and leave the tail unread -- 26% of code_rp1t, 34% of en_c4,
92% of zh_web. This recovers the cursor by REPLAYING the plan that produced the
checkpoint.

Replay rather than proportional consumption, because proportional is wrong exactly
where it matters. build_mix caps each domain at `pool_rows * epochs`, so a capped
domain draws fewer rows than its weight asks for: stage-1 cot wanted 310,546 rows
and drew 295,512. Weights do not predict consumption for any capped domain, and the
capped domains are the ones whose tails matter.

The plan is a pure function of (mix, total rows, sample seed, val split), so replaying
it reproduces the exact counts, not an estimate. What replay cannot verify is that the
corpus is unchanged since -- so the corpus fingerprint is recorded at reconstruction
time and a later mismatch invalidates the result (fb's attachment 2).

    python3 scripts/replay_cursor.py --ckpt ckpt_pretrain_15b_s1.pt [--write]

Without --write it prints the reconstruction and changes nothing.

# restartable: read-only without --write, and idempotent with it -- the replay is a
# pure function of (mix, step, pool sizes), so re-running reproduces the same counts.
# An interrupt costs one re-read; the only mutation is a single torch.save at the end,
# and a checkpoint that already carries a row_cursor is skipped rather than rewritten.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def replay(mix_path, steps_done, batch, accum, world, seq, val_frac, val_rows_max, pool_rows):
    """Per-domain rows consumed by the plan, replayed phase by phase.

    Mirrors build_mix: for each phase, want = int(total_rows * frac * weight), capped at
    pool*epochs minus what earlier phases took. `pool_rows` maps domain -> pool size, which
    the caller reads from the token caches (authoritative) rather than from a log.
    """
    mix = json.load(open(mix_path, encoding="utf-8"))
    total_rows = mix["total_tokens"] / seq
    anneal = mix.get("anneal_frac", 0.0)
    phases = [(1 - anneal, "weight")] + ([(anneal, "anneal")] if anneal else [])
    used = {}
    capped = []
    for name, d in mix["domains"].items():
        pool = pool_rows.get(name)
        if not pool:
            continue
        u = 0
        for frac, key in phases:
            want = int(total_rows * frac * d.get(key, d["weight"]))
            cap = int(pool * d.get("epochs", 1)) - u
            if want > cap:
                capped.append((name, key, want, max(0, cap)))
                want = max(0, cap)
            u += want
        used[name] = u
    # The run stopped short of the whole plan: 16,281 steps consume 3,646,944 rows of a
    # 3,662,109-row plan. Which domains those missing rows belonged to is NOT
    # proportional -- that is the approximation this tool exists to avoid. The plan is
    # built phase by phase and then SHUFFLED within each phase, so a prefix of it draws
    # from each domain in proportion to that phase's composition, and a run that stopped
    # inside the final phase never reached the phase after it at all.
    planned_rows = sum(used.values())
    consumed_rows = steps_done * batch * accum * world
    if planned_rows and consumed_rows < planned_rows:
        # Walk the phases in order and stop where the run stopped. Within one phase the
        # draw IS proportional to that phase's weights, because the shuffle is uniform
        # over the phase's rows -- so this is exact per phase, not an estimate over the
        # whole plan.
        remaining = consumed_rows
        used = {k: 0 for k in used}
        for frac, key in phases:
            phase_rows = {}
            for name, d in mix["domains"].items():
                pool = pool_rows.get(name)
                if not pool:
                    continue
                want = int(total_rows * frac * d.get(key, d["weight"]))
                cap = int(pool * d.get("epochs", 1)) - used[name]
                phase_rows[name] = max(0, min(want, cap))
            phase_total = sum(phase_rows.values())
            if remaining >= phase_total:
                for k, v in phase_rows.items():
                    used[k] += v
                remaining -= phase_total
            else:
                share = remaining / max(phase_total, 1)
                for k, v in phase_rows.items():
                    used[k] += int(v * share)
                remaining = 0
                break
    return used, capped, planned_rows, consumed_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="/data00")
    ap.add_argument("--world", type=int, default=7)
    ap.add_argument("--write", action="store_true",
                    help="write row_cursor into the checkpoint (default: print only)")
    a = ap.parse_args()

    import torch

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if ck.get("row_cursor"):
        print(f"{a.ckpt} already carries a row_cursor; nothing to reconstruct")
        return 0
    cfg = ck.get("cfg", {})
    step = ck.get("step")
    if step is None:
        print("FAIL: checkpoint carries no step, so the plan cannot be replayed",
              file=sys.stderr)
        return 1
    mix_rel = cfg.get("mix")
    mix_path = mix_rel if os.path.isabs(mix_rel) else os.path.join(ROOT, mix_rel)
    seq, batch, accum = cfg.get("seq"), cfg.get("batch"), cfg.get("accum")

    # Pool sizes from the caches: authoritative, unlike a log-rounded weight (44).
    sys.path.insert(0, ROOT)
    from train import DATA, _corpus_fp  # noqa: E402

    pool_rows = {}
    mix = json.load(open(mix_path, encoding="utf-8"))
    for name in mix["domains"]:
        cache = os.path.join(a.cache_dir, f"tokens_{name}.pt")
        if not os.path.exists(cache):
            continue
        # The cache is a FLAT token stream, not [N, seq+1]: tokens_code_rp1t.pt is
        # torch.Size([7569081553]), one dimension. train.py:1468 reshapes it with
        # data[: n * (seq+1)].view(-1, seq+1), so rows = len // (seq+1). Reading
        # .shape[0] as a row count gave pools ~4000x too large and every epoch figure
        # as 0.00 -- and since the pool feeds the CAP, a capped domain would have been
        # silently reconstructed as uncapped.
        stream = torch.load(cache, map_location="cpu", weights_only=False, mmap=True)
        n = stream.shape[0] // (seq + 1) if stream.dim() == 1 else stream.shape[0]
        n_val = min(max(1, int(n * cfg.get("val_frac", 0.01))), cfg.get("val_rows_max", 2000))
        pool_rows[name] = n - n_val

    used, capped, planned, consumed = replay(
        mix_path, step, batch, accum, a.world, seq,
        cfg.get("val_frac", 0.01), cfg.get("val_rows_max", 2000), pool_rows)

    print(f"replaying {os.path.basename(a.ckpt)}: step {step}, mix {os.path.basename(mix_path)}")
    print(f"  plan {planned:,} rows, consumed {consumed:,} rows "
          f"({consumed / max(planned, 1):.1%} of the plan)")
    for name, key, want, cap in capped:
        print(f"  CAPPED {name} {key}: wanted {want:,}, cap {cap:,} -- weights would have "
              f"over-predicted this domain by {want - cap:,} rows")
    print("  reconstructed cursor:")
    for k in sorted(used):
        print(f"    {k:15s} {used[k]:>9,} rows  ({used[k] / max(pool_rows.get(k, 1), 1):.2f} epochs)")

    # fb's attachment 2: the corpus fingerprint AT RECONSTRUCTION TIME. A corpus that
    # changed between the run and this replay invalidates the counts, and the exp row is
    # where that is recorded so a later reader can tell.
    srcfp = {}
    for name in used:
        ddir = os.path.join(DATA, "corpus", name)
        if os.path.isdir(ddir):
            srcfp[name] = _corpus_fp(ddir)
    print(f"  srcfp at reconstruction: {json.dumps({k: v[:8] for k, v in srcfp.items()}, sort_keys=True)}")

    if not a.write:
        print("\n(--write not given: nothing changed)")
        return 0
    ck["row_cursor"] = used
    ck["row_cursor_srcfp"] = srcfp
    # The shuffle seed the reconstruction is measured against. Without it the rehearsal's
    # seed check has nothing to compare and reads as unverified -- and a reconstruction
    # is exactly the case where the seed matters, since the counts were derived from a
    # pool order this field is the only record of. cfg.sample_seed if the run set one,
    # else cfg.seed, which is _sample_seed()'s rule.
    _ss = cfg.get("sample_seed")
    ck["row_cursor_seed"] = cfg.get("seed") if _ss is None else _ss
    ck["row_cursor_reconstructed"] = {
        "method": "replay of the stage-1 plan (de-7)",
        "step": step,
        "mix": mix_rel,
        "note": "counts-only assertion; a corpus change since invalidates this",
    }
    torch.save(ck, a.ckpt)
    print(f"\nwrote row_cursor into {a.ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
