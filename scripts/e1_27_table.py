#!/usr/bin/env python3
"""The nine-point table: our arm's five lr points beside the control's five, on one population.

# restartable: reads small JSONs and prints a table; writes nothing. An interrupt costs
# milliseconds. The expensive parts are the training runs and the scoring, both already done by
# the time this runs.

WHAT THIS SWEEP CAN AND CANNOT SETTLE, written before the numbers exist.

Section 5.0 of docs/audits/control_pythia160m_vs_ours.md records the control reducing its own
floor by 61.0% against our 34.8%, and gives three reasons that cannot be read as "the control's
SFT recipe is better". This sweep removes exactly ONE of them -- the second, that the control
swept five lrs while our arm ran one fixed recipe and never swept.

It does NOT remove the first: relative reductions are not comparable across different floors.
Even if our arm finds a point below 0.293989, "fell X% from 0.450964" and "fell 61.0% from
0.903758" still are not one comparison, because the distance a first epoch can close depends on
where it starts. Separating floor effect from recipe difference needs both arms starting from
the same floor, which no lr sweep can arrange.

So the output is "how far our fixed recipe sits from its own optimum", not "now the reductions
are comparable". 1e's ruling, and the reason the reading is written here rather than inferred
from whatever the table turns out to look like.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HV2 = os.path.join(ROOT, "runs", "heldout_v2")
WANT_SHA = "cae4daf7ad59388c"
WANT_BYTES = 10_554_038

# lr_scale 0.1 is ckpt_control_ours.pt, scored through the current guarded evaluator. It is a
# point in this sweep by reuse, not by retraining -- measured legitimate in runs/e1_27_step0/
# (40 steps reproduce runs/control_ours.log to four deltas of 0.000, while lr_scale 1.0 diverges
# to -0.224). Rescoring it would make a second implementation of one number.
OURS = {
    "0.01": "ours_lr0.01.json",
    "0.03": "ours_lr0.03.json",
    "0.1": "ours_sft_reguarded.json",   # reused, not retrained
    "0.3": "ours_lr0.3.json",
    "1.0": "ours_lr1.0.json",
}
FLOOR_OURS = "floor_ours.json"
FLOOR_CTRL = "floor_control.json"


def load(name):
    p = os.path.join(HV2, name)
    if not os.path.isfile(p):
        return None, f"{name} absent"
    d = json.load(open(p))
    if d.get("REFUSED"):
        return None, f"{name} REFUSED: {d['REFUSED']}"
    if d.get("evaluated_ids_sha256") != WANT_SHA:
        return None, (f"{name} evaluated a different population "
                      f"({d.get('evaluated_ids_sha256')} != {WANT_SHA})")
    if d.get("supervised_bytes") != WANT_BYTES:
        return None, (f"{name} used denominator {d.get('supervised_bytes')} != {WANT_BYTES}")
    return d, None


def main():
    errs, rows = [], []
    for lr, name in OURS.items():
        d, e = load(name)
        if e:
            errs.append(e)
            continue
        rows.append((float(lr), lr, d, name))
    rows.sort()

    fo, e1 = load(FLOOR_OURS)
    fc, e2 = load(FLOOR_CTRL)
    errs += [x for x in (e1, e2) if x]

    ctrl = {}
    for p in sorted(glob.glob(os.path.join(HV2, "ctrl_lr*.json"))):
        d, e = load(os.path.basename(p))
        if e:
            errs.append(e)
            continue
        ctrl[os.path.basename(p)[7:-5]] = d

    if errs:
        for e in errs:
            print(f"EXCLUDED: {e}", file=sys.stderr)

    # A PARTIAL TABLE MUST NOT READ AS COMPLETE. Run before the sweep finished, this printed a
    # one-point table, a BEST, and a verdict, and exited 0 -- every number correct and the
    # conclusion unearned. Missing points are named in the output and change the exit code, so a
    # driver cannot mistake "four arms did not finish" for "the sweep says this".
    missing = [lr for lr in OURS if lr not in [r[1] for r in rows]]
    complete = not missing
    if missing:
        print(f"INCOMPLETE: {len(missing)} of {len(OURS)} of our points are not usable "
              f"({', '.join(sorted(missing))}). Everything below describes only the points that "
              f"are here -- BEST, bracketing and any comparison to the control are provisional.\n")

    print(f"population {WANT_SHA}, denominator {WANT_BYTES:,} supervised bytes, "
          f"one evaluator (scripts/eval_heldout.py)\n")
    print("OUR ARM -- SFT lr sweep (this is the new measurement)")
    print(f"{'lr_scale':>9} {'nat/byte':>10} {'nat/tok':>9} {'next vs skip1':>14} "
          f"{'vs 0.1':>8}  source")
    base = next((d for f, lr, d, n in rows if lr == "0.1"), None)
    best = None
    for _, lr, d, name in rows:
        b = d["nll_per_supervised_byte"]
        r = d["skip_one_nll_per_token"] / d["next_token_nll_per_token"]
        rel = f"{100 * (b - base['nll_per_supervised_byte']) / base['nll_per_supervised_byte']:+.2f}%" if base else "--"
        tag = "reused (ckpt_control_ours.pt)" if lr == "0.1" else name
        print(f"{lr:>9} {b:>10.6f} {d['nll_per_supervised_token']:>9.4f} {r:>13.2f}x "
              f"{rel:>8}  {tag}")
        if best is None or b < best[1]:
            best = (lr, b)
    if not rows:
        print("  (no usable points)")
        return 1

    print()
    lrs = [lr for _, lr, _, _ in rows]
    interior = best[0] not in (lrs[0], lrs[-1])
    print(("PROVISIONAL " if not complete else "") + f"BEST {best[0]} at {best[1]:.6f} nat/byte -- "
          + (f"BRACKETED (interior of {lrs})" if interior
             else f"AT AN ENDPOINT of {lrs}: best of what was tried, not a minimum"))
    if base:
        gap = 100 * (base["nll_per_supervised_byte"] - best[1]) / base["nll_per_supervised_byte"]
        print(f"the fixed recipe (0.1) sits {gap:+.2f}% from the swept optimum")
        if abs(gap) < 1e-9:
            print("  -- i.e. the fixed recipe IS the best swept point")

    if fo and base:
        print(f"\nfloor {fo['nll_per_supervised_byte']:.6f} -> best {best[1]:.6f} = "
              f"{100 * (fo['nll_per_supervised_byte'] - best[1]) / fo['nll_per_supervised_byte']:.1f}% "
              f"reduction from OUR floor")
    if fc and ctrl:
        cb = min(d["nll_per_supervised_byte"] for d in ctrl.values())
        print(f"control floor {fc['nll_per_supervised_byte']:.6f} -> best {cb:.6f} = "
              f"{100 * (fc['nll_per_supervised_byte'] - cb) / fc['nll_per_supervised_byte']:.1f}% "
              f"reduction from THE CONTROL's floor")
        print("\nTHESE TWO PERCENTAGES ARE STILL NOT COMPARABLE. Different floors, and the")
        print("distance a first epoch closes depends on where it starts. This sweep removed the")
        print("'we never tuned' objection; it did not make the reductions one comparison.")
        print("The pre-registered verdict rule stays on nat/byte, unchanged:")
        lead = 100 * (cb - best[1]) / cb
        print(f"  lead = (ctrl-ours)/ctrl = {lead:.2f}% vs threshold 14.1% -> "
              f"{'ours better' if lead > 14.1 else 'INDISTINGUISHABLE'}")
        if base:
            old = 100 * (cb - base["nll_per_supervised_byte"]) / cb
            print(f"  (with the FIXED recipe it was {old:.2f}%; tuning our arm moves the "
                  f"published lead by {lead - old:+.2f} points)")
            print("  NOTE the published verdict used the fixed recipe and stays as published --")
            print("  swapping in a tuned point after seeing it would be choosing the number.")
    print(f"\n{len(rows)} of our points, {len(ctrl)} control points, "
          f"{sum(1 for x in (fo, fc) if x)} floors, all on {WANT_SHA}")
    if not complete:
        print(f"EXIT 1: table incomplete ({', '.join(sorted(missing))} missing) -- do not quote "
              f"BEST or the bracketing line as the sweep's result")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
