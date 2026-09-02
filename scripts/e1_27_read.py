#!/usr/bin/env python3
"""Read e1-27 step 0: does lr_scale 0.1 reproduce ckpt_control_ours.pt's first 40 steps, and
does lr_scale 1.0 fail to?

The reading rule was fixed before the numbers existed (1e's ruling): the reuse of 0.293989 as
the sweep's x1 point is legitimate only if 0.1 matches AND 1.0 separates. A test where every
outcome reads as confirmation is not a test.
"""
# restartable: reads two text logs, prints a table and writes one small verdict.json. An
# interrupt costs milliseconds and loses nothing -- the logs it reads are produced by the GPU
# arms, which are the expensive part, and this reader can be rerun against them any number of
# times. Nothing here is sharded because there is nothing to shard.
import argparse
import json
import os
import re
import sys

# From runs/control_ours.log. The driver re-greps the file before running, so these having
# drifted is a refusal, not a mismatch.
ANCHORS = {10: 1.606, 20: 1.629, 30: 1.606, 40: 1.663}

# TOLERANCE, set before seeing the reruns. The log prints 3 decimals, so quantisation alone is
# +/-0.0005. The larger term is bf16 accumulation order on a rerun, possibly on another card:
# the pod's only hard evidence of determinism (b0_16 arm A reproducing a scored row to
# max|diff| 0.000000) covers INFERENCE, where there is no optimizer and no atomics.
#
# 0.02 nat is ~1.2% of the 1.6 anchors: wide enough that bf16 noise cannot fail a true match,
# narrow enough that a 10x lr cannot pass -- at steps 10-40 of an SFT, 10x the lr is a visibly
# different curve, and if it lands inside this band that IS the finding (40 steps cannot see
# lr_scale) rather than a pass.
TOL = 0.02


def parse(text):
    """Regex, never a slice: a hand-counted index ate a digit twice in this experiment, once
    loudly (e-4) and once silently (wrong lr labels, nearly into the report)."""
    out = {}
    for m in re.finditer(r"^step (\d+)/(\d+) loss ([0-9.]+)", text, re.M):
        out[int(m.group(1))] = (float(m.group(3)), int(m.group(2)))
    return out


def read_arm(path):
    if not os.path.isfile(path):
        return None, f"{path} absent -- the arm never ran"
    txt = open(path, errors="replace").read()
    got = parse(txt)
    if not got:
        tail = " / ".join(txt.strip().splitlines()[-2:])[:160]
        return None, f"no step lines in {os.path.basename(path)}; tail: {tail}"
    totals = {t for _, t in got.values()}
    if totals != {1024}:
        return None, (f"total_steps reads {totals}, not 1024 -- the schedule is not the "
                      f"original's, so the comparison is void (did --stop_after get replaced "
                      f"by --max_steps?)")
    return got, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/e1_27_step0")
    a = ap.parse_args()

    print(f"anchors (runs/control_ours.log): {ANCHORS}")
    print(f"tolerance: {TOL} nat\n")

    verdicts, tables = {}, {}
    for scale in ("0.1", "1.0"):
        got, err = read_arm(os.path.join(a.dir, f"repro_lr{scale}.log"))
        print(f"lr_scale {scale}:")
        if err:
            print(f"  UNREADABLE: {err}")
            verdicts[scale] = None
            continue
        rows, worst = [], 0.0
        for s, want in ANCHORS.items():
            if s not in got:
                print(f"  step {s:3d}: MISSING")
                rows.append(None)
                continue
            d = got[s][0] - want
            worst = max(worst, abs(d))
            rows.append(d)
            print(f"  step {s:3d}: got {got[s][0]:.3f}  want {want:.3f}  "
                  f"delta {d:+.3f}  {'within' if abs(d) <= TOL else 'OUTSIDE'}")
        ok = all(r is not None and abs(r) <= TOL for r in rows)
        verdicts[scale] = ok
        tables[scale] = worst
        print(f"  -> {'MATCHES' if ok else 'DOES NOT MATCH'} (worst |delta| {worst:.3f})\n")

    m01, m10 = verdicts.get("0.1"), verdicts.get("1.0")
    print("=== VERDICT (rule fixed before the numbers) ===")
    if m01 is None or m10 is None:
        v, msg = "unreadable", ("an arm did not produce a readable log -- nothing is concluded, "
                                "and in particular this is NOT a refutation")
    elif m01 and not m10:
        v, msg = "confirmed", (
            f"x1 == lr_scale 0.1 CONFIRMED and the negative world separates "
            f"(0.1 worst |delta| {tables['0.1']:.3f} <= {TOL} < {tables['1.0']:.3f} for 1.0). "
            f"Reusing 0.293989 as the fifth point is LEGITIMATE.")
    elif m01 and m10:
        v, msg = "no_resolution", (
            "BOTH scales reproduce the anchors, so 40 steps cannot see a 10x lr difference: "
            "this test has no discriminating power and the fifth point must be trained for "
            "real. (Do not read the 0.1 match as confirmation -- an instrument that cannot "
            "separate a 10x change said nothing about a 1x one.)")
    elif not m01 and not m10:
        v, msg = "refuted", (
            "NEITHER scale reproduces the anchors, so the difference is not lr_scale at all: "
            "pack, seq, seed or base checkpoint differ from the original run. Do NOT reuse "
            "0.293989, and do not assume x1 is any particular value.")
    else:
        v, msg = "inverted", (
            "lr_scale 1.0 matches and 0.1 does not -- x1 is 1.0, not the argparse default. "
            "The sweep's grid multipliers all shift by 10x; report before launching.")
    print(msg)

    out = {"anchors": ANCHORS, "tolerance": TOL, "match": verdicts,
           "worst_abs_delta": tables, "verdict": v, "reading": msg}
    with open(os.path.join(a.dir, "verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.dir}/verdict.json")
    return 0 if v == "confirmed" else 1


if __name__ == "__main__":
    sys.exit(main())
