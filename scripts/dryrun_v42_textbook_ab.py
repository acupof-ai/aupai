#!/usr/bin/env python3
"""CPU dry-run for the v42 textbook continuation A/B (de, fb 2026-09-14).

Builds BOTH stage-2 mixes through train.build_mix(tok=None) up to the plan tensor and prints
the schedule/epoch/verdict structure WITHOUT training a step and WITHOUT a card:
  - total_steps (segment only; build_mix computes plan length, this script does NOT add the
    resume origin -- train.py main does `total += resume_step` when cursor-seeded)
  - per-domain rows drawn and the effective epoch over that domain's pool
  - the textbook domain must draw EXACTLY its 4-epoch budget (not get epoch-capped)
  - _assert_mix_domains (corpus fingerprints) and srcfp/seed checks passing
  - _assert_mix_derived_against against a real r3 checkpoint cursor: the six r3 domains must
    match the ckpt triple, the textbook domain has no cursor and must legally start at row 0.

Run ON THE POD (caches live there), CPU only:
    python3 scripts/dryrun_v42_textbook_ab.py \
        --t   data/mix_textbook_cont.json \
        --c   data/mix_cont_ctrl.json \
        --ckpt ckpt_v41_r3_0914.pt.step23000 \
        --resume-step 23000

It writes no data and trains nothing. Any non-zero exit is a launch blocker to record.
restartable: read-only over caches/mixes/ckpt; re-running just rebuilds the plans.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402

import train  # noqa: E402


def load_ckpt_cursor(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {
        "step": int(ck.get("step", 0)),
        "row_cursor": dict(ck.get("row_cursor") or {}),
        "row_cursor_srcfp": dict(ck.get("row_cursor_srcfp") or {}),
        "row_cursor_seed": ck.get("row_cursor_seed"),
        "total_steps": ck.get("total_steps"),
        "vocab_id": ck.get("vocab_id"),
    }


def build_one(mix_path, cursor, is_t):
    """Call the real builder with tok=None (plan only, no encode/gpu). Capture its build log."""
    # build_mix reads Cfg.mix for some paths; set the bare minimum.
    train.Cfg.mix = os.path.basename(mix_path)
    # The continuation is a single phase; the real launch passes --anneal_frac 0. The mix
    # declares 0.0 and _mix_anneal_frac refuses the Cfg default 0.1, so mirror the flag here.
    train.Cfg.anneal_frac = 0.0
    # With tok=None nothing is encoded, but _domain_seqs(is_main=True) still gates on the cache
    # being FRESH, and same_vocab requires VOCAB_ID -- only the tokenizer path sets it. A real
    # launch sets it from the live tokenizer; the ckpt's vocab_id IS that stamp, so set it here
    # or every domain reads as "another vocabulary" and falls into the tok=None encode crash.
    train.VOCAB_ID = cursor["vocab_id"]
    # Cache shuffle seed must match the .seed sidecar (42) for same_seed to mmap, not rebuild.
    train.Cfg.seed = cursor["row_cursor_seed"]
    # Mirror the launch geometry (run_ddp.sh --batch 4 --accum 6, world 8). A standalone import
    # leaves Cfg at batch 32/accum 1, which would divide the plan by the wrong rows/step. The
    # plan is built at world=1 here so tr holds ALL global rows (the real run stripes rank::8);
    # hence expected global rows = SEG * 8 * batch * accum = 26,304.
    train.Cfg.batch = 4
    train.Cfg.accum = 6
    import contextlib
    import io
    buf = io.StringIO()
    # tok=None: skips the VOCAB_ID encode path but still mmaps caches and builds the plan.
    with contextlib.redirect_stdout(buf):
        tr, _va = train.build_mix(
            mix_path, None, is_main=True, ddp=False, rank=0, world=1,
            row_cursor=cursor["row_cursor"],
            cursor_srcfp=cursor["row_cursor_srcfp"],
            cursor_seed=cursor["row_cursor_seed"],
        )
    return tr, buf.getvalue()


# Strings the stage-2 plan build must NEVER print (fb checklist 'known must-check'):
#  - 'absent or stale': a cache_exclude/sidecar contract the plain *_dc binding must avoid.
#  - '-> capped' on the textbook domain specifically: it must draw its full 4-epoch budget.
FORBIDDEN_LOG = "absent or stale"


def published_plan_report():
    """Read what build_mix published onto Cfg for the just-built plan."""
    names = list(getattr(train.Cfg, "_plan_names", []) or [])
    full = getattr(train.Cfg, "_plan_domains_full", None)
    used = dict(getattr(train.Cfg, "_row_cursor", {}) or {})
    base = dict(getattr(train.Cfg, "_row_cursor_base", {}) or {})
    fps = dict(getattr(train.Cfg, "_row_cursor_srcfp", {}) or {})
    return names, full, used, base, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", required=True, help="treatment mix")
    ap.add_argument("--c", required=True, help="control mix")
    ap.add_argument("--ckpt", required=True, help="r3 checkpoint to take the cursor from")
    ap.add_argument("--resume-step", type=int, default=None)
    ap.add_argument("--require-final-step", type=int, default=None,
                    help="if set (38070 for the r3 FINAL), a cursor at any other step is a "
                         "BLOCKER, not just a validation run. Use this to certify launch mixes.")
    args = ap.parse_args()

    cursor = load_ckpt_cursor(args.ckpt)
    step = args.resume_step if args.resume_step is not None else cursor["step"]
    R3_FINAL_STEP = 38070
    print("=== v42 stage-2 CPU dry-run (no train, no gpu) ===")
    print(f"ckpt={args.ckpt} step={step} seed={cursor['row_cursor_seed']} "
          f"ckpt_total_steps={cursor['total_steps']}")
    print(f"cursor domains: {sorted(cursor['row_cursor_srcfp'])}")
    # fb condition 2: the step23000 mixes validate the PLAN only and are NOT the launch mixes.
    # A launch must regenerate against the r3-final cursor. Without --require-final-step this is
    # a validation run (banner, still exit 0 on a sound plan); with it, a non-final cursor blocks.
    launch_ready = (step == R3_FINAL_STEP and cursor["total_steps"] == R3_FINAL_STEP)
    if not launch_ready:
        msg = (f"VALIDATION ONLY: cursor is step {step}, not the r3-final {R3_FINAL_STEP}. These "
               f"mixes are NOT launch-ready; regenerate with write_mix_v42_stage2.py --ckpt <r3 "
               f"final> and re-run with --require-final-step {R3_FINAL_STEP}.")
        if args.require_final_step == R3_FINAL_STEP:
            print("LAUNCH GATE: " + msg)
        else:
            print(msg)

    blockers = []
    if args.require_final_step == R3_FINAL_STEP and not launch_ready:
        blockers.append(f"--require-final-step {R3_FINAL_STEP}: ckpt cursor step {step}/"
                        f"total {cursor['total_steps']} is not the r3 final; refusing to certify "
                        "these mixes as launch-ready")
    arm_rows = {}
    for label, path, expect_textbook in [("T", args.t, True), ("C", args.c, False)]:
        print(f"\n----- arm {label}: {path} -----")
        with open(path, encoding="utf-8") as fh:
            mix = json.load(fh)
        tr, buildlog = build_one(path, cursor, label == "T")
        names, full, used_after, _base, fps_after = published_plan_report()
        if FORBIDDEN_LOG in buildlog:
            blockers.append(f"{label}: build log contains {FORBIDDEN_LOG!r} (a cache_exclude/"
                           f"sidecar contract leaked; stage-2 must bind plain *_dc caches)")
        # surface the captured build log so the human/CI reads the real builder output
        print("  --- build_mix log ---")
        for ln in buildlog.splitlines():
            print("   ", ln)
        seg_rows = tr.shape[0]
        seg_steps = seg_rows // (train.Cfg.batch * train.Cfg.accum)
        # GLOBAL PLAN LENGTH. world=1 here returns rank 0's 1/world slice = every plan row; the
        # real world8 launch stripes these same SEG*8*b*a = 26,304 rows. Any epoch cap truncates
        # a domain and shortens this below 26,304, which floors total_steps to 136 and moves the
        # warmdown ratio -- so a cap on ANY domain, not just the textbook, is a blocker.
        EXPECT_PLAN_ROWS = 137 * 8 * train.Cfg.batch * train.Cfg.accum
        any_capped = [ln.split("mix: ", 1)[1].split(" ", 1)[0]
                      for ln in buildlog.splitlines() if "-> capped" in ln]
        if any_capped:
            blockers.append(f"{label}: epoch cap fired on {sorted(set(any_capped))}; every domain "
                            f"must draw its full weight so the segment is exactly 137 steps")
        if seg_rows != EXPECT_PLAN_ROWS:
            blockers.append(f"{label}: plan rows {seg_rows} != {EXPECT_PLAN_ROWS} "
                            f"(137 steps x 8 x b{train.Cfg.batch} x a{train.Cfg.accum}); the "
                            f"warmdown ratio 0.003586 would be wrong")
        # NOTE: world=1 here. Real launch world=8 stripes but total plan rows are identical;
        # train prints total_steps = plan_rows // (8*batch*accum) then += resume_step.
        print(f"plan rows={seg_rows} (expect {EXPECT_PLAN_ROWS})  capped={sorted(set(any_capped))}  "
              f"world1_steps={seg_steps}; world8 steps = {seg_rows // (8*train.Cfg.batch*train.Cfg.accum)}")
        # per-domain rows drawn in this segment = bincount over the full domain row
        counts = torch.bincount(full.long(), minlength=len(names)).tolist()
        seg_by = dict(zip(names, counts, strict=True))
        print("per-domain segment rows:")
        for n in names:
            d = mix["domains"][n]
            print(f"  {n:32s} seg_rows={seg_by.get(n,0):8d}  mix_epochs={d.get('epochs')} "
                  f"weight={d.get('weight')} anneal={d.get('anneal')}")
        if expect_textbook:
            tb = "textbook_claude_v41_dc"
            if tb not in names:
                blockers.append(f"{label}: textbook domain {tb} missing from plan")
            else:
                # Hard assertion (not just eyeballing): the builder's cap warning names the domain
                # as "mix: <name> ... -> capped". The textbook must draw its full 4-epoch budget.
                tb_capped = any(
                    ln.lstrip().startswith(f"mix: {tb} ") and "-> capped" in ln
                    for ln in buildlog.splitlines())
                if tb_capped:
                    blockers.append(f"{label}: textbook domain {tb} was epoch-capped -- it did not "
                                   f"draw its full 4-epoch budget; raise pool or epochs in the mix")
                print(f"  textbook seg_rows={seg_by.get(tb)} capped={tb_capped} "
                      f"(must be False; exactly 4 epochs of the pool)")
            # textbook MUST have no inherited cursor (legit row-0)
            if tb in cursor["row_cursor"]:
                blockers.append(f"{label}: textbook unexpectedly carries an r3 cursor entry; "
                                f"stage-2 domain must start row 0")
            else:
                print("  textbook cursor: absent in r3 ckpt -> starts row 0 (correct stage-2 form)")
        arm_rows[label] = seg_by

    # fb condition 3: the T arm is the C arm with the six domains scaled by ONE common factor
    # (the 0.7001 residual) and the freed 0.2999 given wholly to the textbook. So for every six
    # domain, T_rows/C_rows must be the SAME ratio; no domain zeroed, no relative weight changed.
    tb = "textbook_claude_v41_dc"
    six = [n for n in arm_rows["C"] if n != tb]
    assert six and set(six) == set(arm_rows["T"]) - {tb}, (
        f"T and C must name the same six continuation domains: T={sorted(arm_rows['T'])} "
        f"C={sorted(arm_rows['C'])}")
    ratios = {n: arm_rows["T"][n] / arm_rows["C"][n] for n in six}
    r0 = ratios[six[0]]
    # integer apportionment tolerates a one-row largest-remainder wobble, so compare on rows,
    # not on exact float equality: each T row must be within 1 of round(C*r0).
    bad_ratio = []
    for n in six:
        expect = round(arm_rows["C"][n] * r0)
        if arm_rows["T"][n] == 0 or abs(arm_rows["T"][n] - expect) > 1:
            bad_ratio.append(f"{n}: T {arm_rows['T'][n]} vs C {arm_rows['C'][n]} "
                             f"(ratio {ratios[n]:.5f}, common {r0:.5f})")
    print("\n=== six-domain T/C proportionality (must be one common scale) ===")
    for n in six:
        print(f"  {n:28s} T={arm_rows['T'][n]:6d} C={arm_rows['C'][n]:6d} "
              f"T/C={ratios[n]:.5f}")
    print(f"  common ratio ~ {r0:.5f} (the 0.7001 residual / 1.0; largest-remainder +/-1 row)")
    if bad_ratio:
        blockers.append("six-domain T/C ratios are not a common scale: " + "; ".join(bad_ratio)
                        + " -- the A/B would change more than the textbook share")
    # the textbook share must equal exactly what the six gave up: C_six_total - T_six_total.
    tb_rows = arm_rows["T"].get(tb, 0)
    gave_up = sum(arm_rows["C"].values()) - sum(arm_rows["T"][n] for n in six)
    if tb_rows != gave_up:
        blockers.append(f"textbook rows {tb_rows} != rows the six gave up {gave_up}; the arms "
                        "must differ ONLY by the textbook substitution")

    print("\n=== _derived_against triple (built into each mix at write time) ===")
    for path in (args.t, args.c):
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        da = m.get("_derived_against")
        print(f"{os.path.basename(path)}: _derived_against present={bool(da)}")
        if da:
            want_rows = {k: int(v) for k, v in (da.get("row_cursor") or {}).items()}
            got_rows = {k: int(v) for k, v in cursor["row_cursor"].items()}
            # every six-domain the mix names must equal the ckpt cursor rows
            miss = [n for n in want_rows if want_rows[n] != got_rows.get(n)]
            if miss:
                blockers.append(f"{os.path.basename(path)}: cursor row mismatch on {miss}")
            if da.get("row_cursor_seed") != cursor["row_cursor_seed"]:
                blockers.append(f"{os.path.basename(path)}: seed mismatch")
            print(f"  named cursor domains={sorted(want_rows)} mismatches={miss} "
                  f"seed_ok={da.get('row_cursor_seed') == cursor['row_cursor_seed']}")

    print("\n=== RESULT ===")
    if blockers:
        print("BLOCKERS (must fix before launch):")
        for b in blockers:
            print("  -", b)
        return 1
    if launch_ready:
        print("OK and LAUNCH-READY: cursor is the r3 final (step 38070); both plans are 137 "
              "steps, six-domain T/C is one common scale, triple matched, no cap/stale.")
    else:
        print("OK as a PLAN VALIDATION (cursor step "
              f"{step}, not r3 final): schedule/mix arithmetic is sound, but these mixes are "
              "NOT launch-ready. Regenerate against the r3-final ckpt and re-run with "
              "--require-final-step 38070 before any launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
