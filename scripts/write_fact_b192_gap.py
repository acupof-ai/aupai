#!/usr/bin/env python3
"""Append moe.equal_token_gap_vs_dense_b192 entry 1 to facts/moe.json.

Ruled by fb 2026-09-06 20:0xZ: write once, after the dense comparator reaches step 10000,
with all nine in-training val pairs. A measured fact's value is never rewritten -- a later
change is a new entry or a retraction -- which is why this waits for the ninth point instead
of publishing seven and revising.

REFUSES rather than writing a partial series: both arms must have a val row at every
thousand from 2000 to 10000, read from the pod's logs, or nothing is written.

What this entry must NOT say, each because it was wrong when tried:
  - "the gap widens from 0.030 to 0.096": true at the endpoints, false as a direction. Step
    7000 narrows to 0.077 against 6000's 0.085.
  - "the maximum is at step 6000": that was true at n=7 and false at n=8 (step 9000 is
    0.096). An extremum's POSITION migrates with sample count, so this reports the range and
    the shape, never a peak location.
  - a one-line verdict on MoE vs dense: the per-domain split is what decides anything, and
    that is entry 2 after the profile-control rows.

THE NINTH PAIR IS NOT THE SAME QUANTITY AS THE FIRST EIGHT. Warmdown starts at step 9155 on
both arms -- wd_steps = max(1, int(cfg.warmdown * total)), wd_start = total - wd_steps, with
total 10172 and warmdown 0.1 -- so pairs through 9000 sit at constant lr while the 10000
pair is 845 steps into annealing. MoE falls 1.896 -> 1.795 across that boundary, 0.101 of
which is lr rather than tokens. The pair is still valid, because both arms share total and
warmdown and therefore anneal identically from the same step, but a trend extended from the
eight constant-lr pairs into it mixes two states.
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POD = os.path.expanduser("~/bin/pod")
MOE_LOG = "runs/1.5b-a0.2b-e48_8b.log"
DEN_LOG = "runs/0.2b_8b_b192.log"
STEPS = list(range(2000, 10001, 1000))
WARMDOWN_START = 9155


def _pod_vals(rel):
    """{step: val} from a log ON THE POD. The logs are not in the repo."""
    out = subprocess.run(
        [POD, f"cat /work/aupai/{rel}"], capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        sys.exit(f"REFUSING: cannot read /work/aupai/{rel} on the pod: {out.stderr[:200]}")
    vals = {}
    for ln in out.stdout.splitlines():
        m = re.match(r"^step (\d+)/(\d+) val ([0-9.]+)", ln)
        if m:
            vals[int(m.group(1))] = float(m.group(3))
    if not vals:
        sys.exit(f"REFUSING: no `step N/M val X` rows in {rel}")
    return vals


def main():
    moe = _pod_vals(MOE_LOG)
    den = _pod_vals(DEN_LOG)

    missing = [(s, side) for s in STEPS
               for side, d in (("MoE", moe), ("dense", den)) if s not in d]
    if missing:
        sys.exit("REFUSING to write a partial series; absent val rows: "
                 + ", ".join(f"{side}@{s}" for s, side in missing))

    gaps = {s: round(moe[s] - den[s], 4) for s in STEPS}
    # The shape, computed rather than asserted.
    reversals = [s for i, s in enumerate(STEPS[1:], 1)
                 if abs(gaps[s]) < abs(gaps[STEPS[i - 1]])]
    lo, hi = min(abs(g) for g in gaps.values()), max(abs(g) for g in gaps.values())
    if not reversals:
        sys.exit("REFUSING: no reversal found, so the 'not monotone' claim this entry makes "
                 "is not supported by the data it just read -- re-read the series")

    series = "; ".join(f"{s}: {moe[s]:.3f} vs {den[s]:.3f} = {gaps[s]:+.3f}" for s in STEPS)
    rev = ", ".join(str(s) for s in reversals)

    entry = {
        "id": "moe.equal_token_gap_vs_dense_b192",
        "value": (
            f"At matched steps and matched global batch the 8B-token MoE-48 arm is BELOW the "
            f"batch-matched dense arm at all nine in-training val points, by {lo:.3f} to "
            f"{hi:.3f} nat. The sign is opposite to moe.equal_token_gap_vs_dense_0p2b, whose "
            f"comparator ran at batch 64. NO DIRECTION IS CLAIMED: the series is not "
            f"monotone -- it reverses at step(s) {rev} -- so the range above is the claim and "
            f"a peak location is not, an extremum's position having moved once already "
            f"between the 7-point and 8-point reads of this same series. Pairs: {series}."
        ),
        "measured": "2026-09-06",
        "status": "measured",
        "source": (
            f"Both arms' own run logs ON THE POD, read through ~/bin/pod, not present in the "
            f"repo: /work/aupai/{MOE_LOG} (MoE-48, 48 routed experts, top_k 3, expert_ffn "
            f"768, 1 shared, moe_layers 0-11, moe_bias_gamma 0.001) and /work/aupai/{DEN_LOG} "
            f"(dense, no moe_* keys in its cfg line). Rows matched on the literal "
            f"`^step N/M val X` form at every thousand from {STEPS[0]} to {STEPS[-1]}; this "
            f"entry's writer refuses to emit a partial series. THE MoE ARM'S LOG IS THE "
            f"SECOND SEGMENT of its run: the first is /work/aupai/runs/b0_moe48_8b.log, which "
            f"ends at step 1000 with signal 15, and the resume is named in the second "
            f"segment's WSD JOIN line. The pairs above all lie past step 1000, so they are "
            f"entirely inside the second segment and the resume does not enter them."
        ),
        "config": {
            "pairing_rule": (
                "MATCHED STEPS, because both arms run the same global batch: batch 8 x accum "
                "4 x 6 cards = 786,432 tokens/step on each side, ratio 1. This is what "
                "distinguishes this fact from moe.equal_token_gap_vs_dense_0p2b, whose "
                "comparator ran 262,144 tokens/step at ratio 3 with step N against 3N. That "
                "comparator ALSO cannot reach this token count at all: it totals 3815 steps "
                "and anneals past 3434, so a pair beyond it compares an annealing model to a "
                "stable-lr one. A batch-matched comparator is not a different pairing "
                "convention here, it is the only pairing that exists at 7.86B tokens."
            ),
            "both_arms": "total 10172 steps, warmdown 0.1, anneal_frac 0.0, warmup 300, "
                         "val_every 200, seed 42, mix data/mix_200m_8b.json, seq 4096",
            "warmup_share": "300 of 10172 = 2.9% on both arms, so warmup fraction is not a "
                            "difference between them (it IS one against the batch-64 "
                            "comparator, which carries 300 of 3815 = 7.9%)",
            "lr_state": (
                f"Warmdown begins at step {WARMDOWN_START} on BOTH arms, from train.py's "
                f"wd_steps = max(1, int(cfg.warmdown * total)) and wd_start = total - "
                f"wd_steps; the run logs print it as `warmdown starts at step "
                f"{WARMDOWN_START}`. Pairs {STEPS[0]}-9000 therefore sit at constant lr and "
                f"the {STEPS[-1]} pair is {STEPS[-1] - WARMDOWN_START} steps into annealing. "
                f"MoE's val falls {moe[9000]:.3f} -> {moe[STEPS[-1]]:.3f} across that "
                f"boundary, a drop driven by lr and not by tokens. The pair remains valid "
                f"because the two arms share total and warmdown and so anneal identically "
                f"from the same step, but it is NOT the same quantity as the eight "
                f"constant-lr pairs and a trend must not be extended through it."
            ),
        },
        "uncertainty": (
            "Single seed (42) on each side, so no seed spread is available and none is "
            "claimed. val is the in-training figure at val_every 200, one number per point, "
            "not a re-scored evaluation. NEITHER ARM'S HOST WAS CONTROLLED: the MoE arm ran "
            "12:xx-13:09Z with card 6's occupancy unrecorded, and the dense arm's window "
            "spans a 95 GB backup (14:31:57-15:24:10Z on /mnt/data02). That affects step "
            "TIME, measured separately, and no mechanism is claimed by which it would move "
            "val -- but it is recorded here because it was not held fixed."
        ),
        "boundary": (
            "PER-DOMAIN ATTRIBUTION IS NOT IN THIS ENTRY. A single val number cannot say "
            "whether the gap is code, math, zh_web, or spread evenly, and that is what "
            "decides where the extra parameters are spent; it is entry 2, from the "
            "--profile control rows at step 10000. ALSO NOT SEPARABLE HERE: batch and "
            "architecture are held together with the token axis only in the sense that both "
            "arms match on batch -- what this entry does not isolate is the 1.28B extra "
            "parameters against the routing itself, since the arms differ in both at once. "
            "moe.equal_token_gap_vs_moe48_small_batch is the pair that isolates batch and "
            "warmup share with architecture fixed, and it should be read beside this one "
            "rather than combined with it."
        ),
    }

    p = os.path.join(ROOT, "facts", "moe.json")
    with open(p, encoding="utf-8") as fh:
        doc = json.load(fh)
    if any(f.get("id") == entry["id"] for f in doc["facts"]):
        sys.exit(f"REFUSING: {entry['id']} already exists; a measured fact's value is not "
                 f"rewritten -- add a new entry or a retraction instead")
    doc["facts"].append(entry)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"appended {entry['id']}: {len(STEPS)} pairs, |gap| {lo:.3f}-{hi:.3f}, "
          f"reversal at {rev}")
    print(f"  facts in file: {len(doc['facts'])}")


if __name__ == "__main__":
    main()
