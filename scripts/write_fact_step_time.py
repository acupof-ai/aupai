#!/usr/bin/env python3
"""Write facts/moe.json#moe.step_time_vs_dense_b192 from both arms' pod logs.

A WRITER RATHER THAN A HAND-EDIT, for the reason the b192 per-domain entry needed one: every
number in the entry is recomputed here from the logs at write time, so the entry cannot disagree
with its source. It refuses rather than emitting a partial or unsupported claim.

WHAT THIS MEASURES AND WHAT IT DOES NOT. s/step at matched global batch, MoE-48 against the
batch-matched dense arm. It is NOT an efficiency claim: the MoE arm is 1.5B total parameters
against the dense arm's 0.2B, so a 1.31x step time at equal tokens/step says what this pair of
configurations costs per step and nothing about MoE throughput in general.

THE HOST CONFOUND IS RESOLVED HERE, NOT NOTED. facts/moe.json#moe.equal_token_gap_vs_dense_b192's
uncertainty field records a 95 GB backup inside the dense arm's window and says it "affects step
TIME, measured separately" -- this is that separate measurement, so leaving the confound as a
caveat would be leaving the job undone. The dense log shows it directly: 49 of 809 steady lines
sit at MFU 42-48% against the arm's own 49-51%, median 2.4950 against 2.1545. The resolution is
that the ratio is computed at six estimators from the full median to each arm's own minimum, and
the fastest decile of each arm cannot contain a step that was waiting on host I/O. The spread
across all six IS the uncertainty, and it is reported as a band rather than a point.
"""
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS = os.path.join(ROOT, "facts", "moe.json")
FACT_ID = "moe.step_time_vs_dense_b192"

MOE_LOG = "/work/aupai/runs/1.5b-a0.2b-e48_8b.log"
DENSE_LOG = "/work/aupai/runs/0.2b_8b_b192.log"

# THE RULE IS LIFTED FROM scripts/time_bias_projection.py:_step_seconds_from_log, not re-invented:
# skip "this interval" lines (they include the val pass and read 3.5-5.2 s/step), require
# step >= 2000 (allocator and cache warmup persist ~1000 steps past a resume), take the MEDIAN.
STEP_RE = re.compile(r"^step (\d+)/\d+ .*s/step ([0-9.]+)")
MIN_STEP = 2000
MIN_LINES = 30  # the same floor time_bias_projection refuses below

# restartable: reads two logs over ~/bin/pod and rewrites one JSON file in place after every
# number is in hand. An interrupt before the write leaves facts/moe.json untouched; an interrupt
# during it is a single json.dump of a file measured in kilobytes. Re-running recomputes
# everything from the logs, so there is no state to resume.


def pod_read(path):
    """The log's text from the pod. These logs are NOT in the repo -- both arms wrote them to
    /work/aupai/runs and neither is tracked, which is why the source field names the pod path."""
    p = subprocess.run([os.path.expanduser("~/bin/pod"), f"cat {path}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"REFUSING: could not read {path} from the pod: {p.stderr.strip()[:300]}")
    return p.stdout


def steady(text, path):
    vals = []
    for ln in text.splitlines():
        if "this interval" in ln:
            continue
        m = STEP_RE.match(ln)
        if m and int(m.group(1)) >= MIN_STEP:
            vals.append((int(m.group(1)), float(m.group(2))))
    if len(vals) < MIN_LINES:
        sys.exit(f"REFUSING: {path} has {len(vals)} steady-state lines with step >= {MIN_STEP}; "
                 f"a median over fewer than {MIN_LINES} is not the run's rate")
    return vals


def cfg_line(text, path):
    for ln in text.splitlines():
        if ln.startswith("cfg batch "):
            return ln.strip()
    sys.exit(f"REFUSING: {path} has no `cfg batch ` line, so the batch match cannot be verified "
             f"from the log and would have to be assumed")


def main():
    moe_txt, dense_txt = pod_read(MOE_LOG), pod_read(DENSE_LOG)
    moe, dense = steady(moe_txt, MOE_LOG), steady(dense_txt, DENSE_LOG)

    # THE BATCH MATCH IS READ OFF BOTH LOGS, not assumed. The whole comparison rests on both arms
    # running the same tokens/step; if they did not, this is a different fact.
    mc, dc = cfg_line(moe_txt, MOE_LOG), cfg_line(dense_txt, DENSE_LOG)
    for key in ("batch 8", "accum 4", "seq 4096", "grad_ckpt False"):
        if key not in mc or key not in dc:
            sys.exit(f"REFUSING: `{key}` is not in both cfg lines, so the arms are not matched "
                     f"the way this entry claims.\n  MoE:   {mc}\n  dense: {dc}")
    # Everything except the MoE keys must agree, or some other difference is in play.
    strip = re.compile(r" moe_router [0-9.e-]+")
    if strip.sub("", mc) != strip.sub("", dc):
        sys.exit(f"REFUSING: the cfg lines differ beyond the MoE keys.\n  MoE:   {mc}\n"
                 f"  dense: {dc}")

    a = sorted(v for _, v in moe)
    b = sorted(v for _, v in dense)
    n_eq = min(len(a), len(b))

    def est(v, how):
        return {
            "full median": statistics.median(v),
            "fastest 50% median": statistics.median(v[: len(v) // 2]),
            "fastest 10% median": statistics.median(v[: max(1, len(v) // 10)]),
            "fastest 1% median": statistics.median(v[: max(1, len(v) // 100)]),
            "minimum": min(v),
            "full mean": statistics.fmean(v),
        }[how]

    order = ["full median", "fastest 50% median", "fastest 10% median", "fastest 1% median",
             "minimum", "full mean"]
    table = [(h, est(a, h), est(b, h)) for h in order]
    ratios = [x / y for _, x, y in table]
    lo, hi = min(ratios), max(ratios)
    med_ratio = table[0][1] / table[0][2]

    # EQUAL n IN TWO WINDOWS. The arms have unequal steady counts (769 vs 809), and n and window
    # are separate variables: if the shorter arm's number were window-sensitive, the first-n and
    # last-n reads would disagree. Both are computed so the entry can state that they do not.
    moe_ord = [v for _, v in moe]
    den_ord = [v for _, v in dense]
    first_r = statistics.median(moe_ord[:n_eq]) / statistics.median(den_ord[:n_eq])
    last_r = statistics.median(moe_ord[-n_eq:]) / statistics.median(den_ord[-n_eq:])
    if abs(first_r - last_r) > 0.01:
        sys.exit(f"REFUSING: the equal-n ratio depends on the window ({first_r:.4f} first vs "
                 f"{last_r:.4f} last). This entry's claim is that it does not, so the claim is "
                 f"unsupported and the entry must say something else")

    # The contended fraction of the dense arm, from its own printed MFU, so the confound is a
    # measured quantity in the entry rather than a reference to a backup's clock.
    mfu = re.compile(r"^step (\d+)/\d+ .*MFU (\d+)%.*s/step ([0-9.]+)")
    drows = [(int(m.group(1)), int(m.group(2)), float(m.group(3)))
             for m in (mfu.match(ln) for ln in dense_txt.splitlines()
                       if "this interval" not in ln) if m and int(m.group(1)) >= MIN_STEP]
    if not drows:
        sys.exit("REFUSING: no MFU could be parsed from the dense log, so the contended fraction "
                 "cannot be measured and the host confound would be an assertion")
    top = max(x[1] for x in drows)
    slow = [x for x in drows if x[1] < top - 2]
    slow_med = statistics.median([x[2] for x in slow]) if slow else None
    fast_med = statistics.median([x[2] for x in drows if x[1] >= top - 2])

    rows = "; ".join(f"{h} MoE {x:.4f} dense {y:.4f} ratio {x / y:.4f}" for h, x, y in table)

    fact = {
        "id": FACT_ID,
        "value": (
            f"At matched global batch the MoE-48 arm's step time is {lo:.4f}x to {hi:.4f}x the "
            f"batch-matched dense arm's -- +{100 * (lo - 1):.1f}% to +{100 * (hi - 1):.1f}% "
            f"wall-clock per step. The single figure, if one is needed, is the full-log median "
            f"ratio {med_ratio:.4f}: MoE {table[0][1]:.4f} s/step against dense "
            f"{table[0][2]:.4f} s/step. THE BAND IS THE CLAIM AND THE POINT IS NOT: six "
            f"estimators from the full median to each arm's own fastest step span "
            f"{100 * (hi - lo) / lo:.1f}%, so a four-digit ratio would be reporting the "
            f"estimator rather than the arms."
        ),
        "measured": "2026-09-07",
        "status": "measured",
        "source": (
            f"Both arms' own run logs ON THE POD, read through ~/bin/pod, neither tracked in the "
            f"repo: {MOE_LOG} (MoE-48) and {DENSE_LOG} (dense, no moe_* keys in its cfg line "
            f"beyond moe_router's lr). Steady-state lines only, and the two logs are the same "
            f"files facts/moe.json#moe.equal_token_gap_vs_dense_b192 reads for val. Written by "
            f"scripts/write_fact_step_time.py, which recomputes every number here from the logs "
            f"and refuses on a partial read, an unmatched cfg, or a window-dependent ratio."
        ),
        "config": {
            "rule": (
                f"Lifted from scripts/time_bias_projection.py's _step_seconds_from_log, not "
                f"re-invented: skip any line containing `this interval` (validation intervals "
                f"include the val pass and read 3.5-5.2 s/step), require step >= {MIN_STEP} "
                f"(allocator and cache warmup persist about 1000 steps past a resume), take the "
                f"MEDIAN not the mean. Refuses below {MIN_LINES} lines, the same floor that "
                f"script uses."
            ),
            "estimators": rows,
            "equal_n": (
                f"The arms have unequal steady-state counts: MoE n={len(moe)}, dense "
                f"n={len(dense)}. Equal n={n_eq} trims the DENSE side. Computed in two windows "
                f"because n and window are separate variables: first-{n_eq} ratio "
                f"{first_r:.4f}, last-{n_eq} ratio {last_r:.4f}, and the full unequal-n ratio "
                f"{med_ratio:.4f}. They agree to "
                f"{max(abs(first_r - med_ratio), abs(last_r - med_ratio)):.4f}, so neither n nor "
                f"the window moves this number -- which is what makes the estimator spread, not "
                f"the sample size, the thing to report."
            ),
            "batch_match": (
                f"Verified from both cfg lines rather than assumed, and they are identical apart "
                f"from the MoE keys: batch 8 x accum 4 x 6 cards x seq 4096 = 786,432 "
                f"tokens/step on each side, ratio 1:1, grad_ckpt False on both. MoE cfg: {mc}"
            ),
        },
        "uncertainty": (
            f"THE HOST CONFOUND IS MEASURED, NOT ASSUMED AWAY. "
            f"moe.equal_token_gap_vs_dense_b192 records a 95 GB backup (14:31:57-15:24:10Z on "
            f"/mnt/data02) inside the dense arm's window and defers step time to a separate "
            f"measurement; this is it. The dense log carries the contention itself: "
            f"{len(slow)} of {len(drows)} steady lines "
            f"({100 * len(slow) / len(drows):.1f}%) sit at MFU {min(x[1] for x in slow)}-"
            f"{max(x[1] for x in slow)}% against the arm's own {top - 2}-{top}%, with median "
            f"{slow_med:.4f} s/step against {fast_med:.4f} for the rest. The median is robust to "
            f"it -- {table[0][2]:.4f} over all lines against {fast_med:.4f} over the uncontended "
            f"ones, a difference of {abs(table[0][2] - fast_med):.4f} s.\n"
            f"THE CONTENTION SETS NEITHER END OF THE BAND, and an earlier version of this entry "
            f"said it set the lower one. Recomputed against the same MoE median: as-observed "
            f"{table[0][1] / table[0][2]:.4f}, contention removed "
            f"{table[0][1] / fast_med:.4f}. Removing the confound RAISES the ratio by "
            f"{table[0][1] / fast_med - table[0][1] / table[0][2]:.4f}; it does not lower it by "
            f"the band's width. The band's ends are min/min ({lo:.4f}) and median/median "
            f"({hi:.4f}) -- a DIFFERENT ESTIMATOR applied to both arms, so its width measures "
            f"estimator choice, exactly as `value` says. Calling the low end confound-free would "
            f"invite 'the architecture-only cost is near {lo:.4f}', which this entry's boundary "
            f"forbids. The fastest-decile argument is kept as what it actually supports: a "
            f"fast-tail step cannot have been waiting on host I/O, so the contention is absent "
            f"from the tail -- which is evidence about the tail, not a confound-free version of "
            f"the median. Corrected 2026-09-07 by tilerl, who recomputed both ratios rather than "
            f"reading the sentence; the robustness clause above already contained the right "
            f"reading, so the entry held two conclusions that contradicted each other.\n"
            f"Single seed (42) per side, so no seed spread exists and none is "
            f"claimed. NEITHER ARM'S HOST WAS CONTROLLED: card occupancy during the MoE arm's "
            f"window is unrecorded, so its own step time may carry contention that no MFU split "
            f"can separate -- its printed MFU is 252-283%, computed on TOTAL rather than active "
            f"parameters, so it is not a utilisation and its spread is not comparable to the "
            f"dense arm's. So the SAME method that found the dense arm's {len(slow)} contended "
            f"lines cannot be run on the MoE side at all, which is why 'neither arm's host was "
            f"controlled' stands as written rather than being softened."
        ),
        "boundary": (
            "NOT AN EFFICIENCY OR ARCHITECTURE CLAIM. (1) The MoE arm is ~1.5B total parameters "
            "against the dense arm's ~0.2B, so a 1.3x step time at equal tokens/step is what "
            "THIS pair of configurations costs per step; it says nothing about MoE throughput in "
            "general and nothing per active parameter. (2) One card-set and one shape: 6 cards, "
            "batch 8 accum 4 seq 4096, 48 routed experts top_k 3 expert_ffn 768 with 1 shared, "
            "moe_layers 0-11. A different expert count, top_k, or card count is a different "
            "number. (3) HOST STATE UNCONTROLLED ON BOTH ARMS, so the difference is NOT "
            "attributable to architecture: it is what the two arms cost on this machine in these "
            "two windows. Attribution needs both arms run back to back on the same cards with "
            "nothing else on the host, which has not been done. (4) s/step, not tokens/s: at "
            "matched batch these are reciprocal, but only because the batch match was verified "
            "above -- the reciprocal reading breaks if anyone reuses this entry for a pair whose "
            "cfg lines differ."
        ),
    }

    with open(FACTS, encoding="utf-8") as fh:
        doc = json.load(fh)
    if any(f.get("id") == FACT_ID for f in doc["facts"]):
        sys.exit(f"REFUSING: {FACT_ID} already exists in facts/moe.json. This writer creates the "
                 f"entry; amending one is a different operation and must not be a silent "
                 f"overwrite of someone else's read.")
    doc["facts"].append(fact)
    with open(FACTS, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"{'estimator':28}{'MoE':>10}{'dense':>10}{'ratio':>9}")
    for h, x, y in table:
        print(f"{h:28}{x:10.4f}{y:10.4f}{x / y:9.4f}")
    print(f"\nband {lo:.4f}-{hi:.4f}  full-median {med_ratio:.4f}")
    print(f"equal n={n_eq}: first {first_r:.4f}  last {last_r:.4f}")
    print(f"dense contended: {len(slow)}/{len(drows)} lines, median {slow_med:.4f} vs "
          f"{fast_med:.4f}")
    print(f"wrote {FACT_ID} to facts/moe.json ({len(doc['facts'])} facts)")


if __name__ == "__main__":
    main()
