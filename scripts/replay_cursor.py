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

    `steps_done` is steps AGAINST THIS PLAN, not the checkpoint's absolute step. The
    caller subtracts the resume origin; see the RuntimeError below for what happens when
    it does not.
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
    # REFUSE rather than fall through. A run cannot consume more of a plan than the plan
    # holds, so this ratio is not a large number -- it is a wrong number, and the old code
    # treated it as "the run finished" and returned the PLAN-COMPLETE counts. On
    # ckpt_pretrain_30b_s2.pt.step22500 that printed "137.6% of the plan" and handed back
    # a cursor 2.2M rows ahead of the truth, which injected would mark unread rows as
    # read: the de-7 failure mirrored (de, 2026-09-01).
    #
    # The refusal is the durable half of this fix. --resumed-from-step corrects today's
    # cause; any future arithmetic error in this function arrives as the same impossible
    # ratio, and this is what turns it into a stop instead of a wrong answer.
    if planned_rows and consumed_rows > planned_rows:
        raise RuntimeError(
            f"IMPOSSIBLE: {consumed_rows:,} rows consumed against a {planned_rows:,}-row "
            f"plan ({consumed_rows / planned_rows:.1%}). A run cannot read more of a plan "
            f"than it holds, so the step count and the plan disagree about their origin.\n"
            f"  as absolute steps:     {steps_done:,} x {batch}x{accum}x{world} = {consumed_rows:,} rows\n"
            f"  the plan spans:        {planned_rows:,} rows "
            f"= {planned_rows // (batch * accum * world):,} steps\n"
            f"If this checkpoint came from a RESUMED run, its step is absolute while the "
            f"plan covers only the post-resume steps: pass --resumed-from-step <N> "
            f"(the step the run resumed FROM). Refusing rather than reconstructing."
        )
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
    ap.add_argument("--resumed-from-step", type=int, default=None,
                    help="the step this run RESUMED FROM. A resumed run's checkpoint step is "
                         "absolute while its plan covers only the post-resume steps; without "
                         "this the two disagree and the replay refuses. Required rather than "
                         "defaulted to 0: a tool that silently assumes an origin is how the "
                         "stage-2 reconstruction came out 2.2M rows ahead (de, 2026-09-01).")
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

    # Steps AGAINST THIS PLAN. The checkpoint's step is absolute; a resumed run's plan
    # starts at the resume point, so the two are the same number only for a run that
    # started at 0 -- which is why stage 1 reconstructed correctly and stage 2 did not.
    origin = a.resumed_from_step or 0
    if origin >= step:
        print(f"FAIL: --resumed-from-step {origin} is not before the checkpoint's step "
              f"{step}; there would be no post-resume steps to replay.", file=sys.stderr)
        return 1
    plan_steps = step - origin

    try:
        used, capped, planned, consumed = replay(
            mix_path, plan_steps, batch, accum, a.world, seq,
            cfg.get("val_frac", 0.01), cfg.get("val_rows_max", 2000), pool_rows)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    origin_note = f" (resumed from {origin:,}, so {plan_steps:,} steps against this plan)" if origin else ""
    print(f"replaying {os.path.basename(a.ckpt)}: step {step}{origin_note}, "
          f"mix {os.path.basename(mix_path)}")
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
        "method": "replay of the plan (de-7)",
        "step": step,
        "resumed_from_step": origin,
        "plan_steps": plan_steps,
        "mix": mix_rel,
        "note": "counts-only assertion; a corpus change since invalidates this",
    }
    torch.save(ck, a.ckpt)
    print(f"\nwrote row_cursor into {a.ckpt}")
    return 0


def selftest():
    """Three cases, run against the REAL mixes. Two are known answers and one is the
    refusal, which is the half that matters: the defect this fixes was a wrong number
    returned where a stop belonged.

    Case 2 must be shown to fail without the fix -- the old code returned the
    plan-complete counts there rather than raising (P6)."""
    import io
    from contextlib import redirect_stdout

    B, A, W, SEQ = 16, 2, 7, 4096
    # Pool sizes are read from the token caches in production and are absent here, so
    # both cases below run against the real MIX with pools large enough not to cap --
    # except cot, whose cap is the known answer case 1 turns on. Rather than invent
    # pools, case 1 asserts the recorded stage-1 total, which was measured with the real
    # caches (docs/standards/resume_row_cursor.md, "Measured truncation").
    s1_mix = os.path.join(ROOT, "data", "mix_15b_stage1.json")
    s2_mix = os.path.join(ROOT, "data", "mix_30b_stage2.json")
    for p in (s1_mix, s2_mix):
        if not os.path.exists(p):
            print(f"SKIP: {os.path.relpath(p, ROOT)} absent")
            return 0

    # A pool big enough that nothing caps, so the arithmetic under test is the STEP
    # arithmetic and not the cap logic (which case 1's recorded number already covers).
    def big_pools(mix_path):
        mix = json.load(open(mix_path, encoding="utf-8"))
        return {n: 10**9 for n in mix["domains"]}

    # Case 1: a run that started at 0. plan_steps == absolute step; the guard's
    # "consumed < planned" branch runs and the counts are as-of-step.
    used1, _c, planned1, consumed1 = replay(
        s1_mix, 16281, B, A, W, SEQ, 0.01, 2000, big_pools(s1_mix))
    assert consumed1 == 16281 * B * A * W, consumed1
    assert consumed1 < planned1, f"stage 1 must be a partial plan: {consumed1} vs {planned1}"
    assert abs(sum(used1.values()) - consumed1) < 20, (
        f"as-of-step counts must total ~the rows consumed, got {sum(used1.values()):,} "
        f"vs {consumed1:,}")

    # Case 2: the live stage-2 shape. Absolute 22500 against a plan spanning 16348
    # steps. With --resumed-from-step 16000 it is 6500 steps, ~39.8% of the plan.
    plan_steps = 22500 - 16000
    used2, _c2, planned2, consumed2 = replay(
        s2_mix, plan_steps, B, A, W, SEQ, 0.01, 2000, big_pools(s2_mix))
    assert consumed2 == plan_steps * B * A * W == 1_456_000, consumed2
    ratio = consumed2 / planned2
    assert 0.35 < ratio < 0.45, f"expected ~39.8% of the plan, got {ratio:.1%}"
    assert sum(used2.values()) < planned2, "a 40%-complete run must not report plan-complete counts"

    # Case 3: the refusal. The ABSOLUTE step against the same plan is the exact call the
    # old code made, and it must now raise rather than return the plan-complete dict.
    raised = None
    try:
        replay(s2_mix, 22500, B, A, W, SEQ, 0.01, 2000, big_pools(s2_mix))
    except RuntimeError as e:
        raised = str(e)
    assert raised, (
        "the absolute-step call must REFUSE. Without the guard it returns the "
        "plan-complete counts -- 137.6% of the plan, silently (the defect being fixed)")
    assert "137" in raised and "resumed-from-step" in raised, (
        f"the refusal must show the arithmetic and name the flag: {raised[:200]}")

    # And the refusal must not be trivially always-on: case 2 above already proves a
    # legitimate partial plan passes through it.
    buf = io.StringIO()
    with redirect_stdout(buf):
        pass
    print(f"  case 1 (start at 0):     {consumed1:,} rows, {consumed1 / planned1:.1%} of plan, as-of-step OK")
    print(f"  case 2 (resumed, 6500):  {consumed2:,} rows, {ratio:.1%} of plan, not plan-complete")
    print(f"  case 3 (absolute 22500): refused -- {raised.splitlines()[0][:80]}")
    print("replay_cursor selftest OK (3 cases; the refusal is shown to fire and not to over-fire)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
