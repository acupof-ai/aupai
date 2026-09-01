#!/usr/bin/env python3
"""Read the lr_scale probe by fb-1's PRE-REGISTERED rule, and report its resolution.

THE RULE (runs/tasks.jsonl#fb-1, frozen 2026-09-01 15:31, before arm 2 existed):
  1. NaN, a loss spike above 2x the arm's own trailing-100 mean, or gnorm > 5
     -> that arm is refuted and the other wins outright.
  2. mean training loss over the LAST 100 steps of each arm.
  3. max gnorm over steps 250-499.
  DECISION: a gap below 0.05 is NOT DECIDABLE and the recipe keeps 0.85 on b0's
  t71 derivation. Above 0.05 favours the lower arm only if its second-half max
  gnorm is not higher.

  Amendment (board 15:33): the last-100 window spans the [main]->[anneal] change at
  step 450. The rule STANDS -- both arms run the identical schedule on the identical
  mix, so the phase boundary cancels in the difference.

WHY THIS PRINTS A SEM. train.py:2518 syncs the loss only on `step % 10 == 9`, and
logs the SINGLE-STEP value, not a mean over the ten. So "mean over the last 100
steps" is a mean of TEN samples. On arm 0.85 that is sd 0.135 -> SEM 0.043, and the
difference of two arms carries sqrt(2) x that = 0.060. The 0.05 threshold is 0.83
sigma: two arms with no true difference clear it about 41% of the time.

The rule is NOT changed here -- it was frozen before the data, which is the whole
point of it. This prints the gap AND the sigma so the reader cannot mistake a
coin-flip for a signal. Reported before arm 2 finished, so it cannot be a number
chosen after seeing the answer.

    python3 scripts/read_lr_probe.py --selftest
    python3 scripts/read_lr_probe.py runs/lrprobe_0.85.log runs/lrprobe_1.2.log
"""
import re
import statistics as st
import sys

STEP_RE = re.compile(r"step (\d+)/\d+.*?loss ([\d.]+).*?gnorm ([\d.naN]+)")


def parse(path_or_lines):
    lines = (open(path_or_lines, encoding="utf-8", errors="replace")
             if isinstance(path_or_lines, str) else path_or_lines)
    rows = {}
    for line in lines:
        m = STEP_RE.search(line)
        if m:
            try:
                rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            except ValueError:  # a literal nan in either field
                rows[int(m.group(1))] = (float("nan"), float("nan"))
    return sorted((s, v[0], v[1]) for s, v in rows.items())


def read_arm(rows):
    """The three quantities, in fb-1's order. Returns a dict; never a verdict."""
    if not rows:
        return {"refuted": "no step lines"}
    last = [r for r in rows if r[0] >= rows[-1][0] - 99]
    L = [r[1] for r in last]
    trailing = st.mean(L) if L else float("nan")
    bad = []
    if any(v != v for _, v, _ in rows):
        bad.append("NaN")
    if any(g == g and g > 5 for _, _, g in rows):
        bad.append("gnorm>5")
    if L and max(L) > 2 * trailing:
        bad.append(f"loss spike {max(L):.3f} > 2x{trailing:.3f}")
    half = [g for s, _, g in rows if 250 <= s <= 499 and g == g]
    return {
        "refuted": "; ".join(bad) or None,
        "n": len(L),
        "mean_loss": trailing,
        "sd": st.stdev(L) if len(L) > 1 else float("nan"),
        "sem": st.stdev(L) / len(L) ** 0.5 if len(L) > 1 else float("nan"),
        "max_gnorm_250_499": max(half) if half else float("nan"),
        "last_step": rows[-1][0],
    }


def verdict(a, b, name_a="0.85", name_b="1.2"):
    """fb-1's decision rule, applied verbatim. 'not-decidable' is a real result."""
    if a["refuted"] and not b["refuted"]:
        return name_b, f"{name_a} refuted ({a['refuted']})"
    if b["refuted"] and not a["refuted"]:
        return name_a, f"{name_b} refuted ({b['refuted']})"
    if a["refuted"] and b["refuted"]:
        return "not-decidable", "both arms refuted"
    gap = abs(a["mean_loss"] - b["mean_loss"])
    if gap < 0.05:
        return "not-decidable", f"gap {gap:.4f} < 0.05"
    lower, lname, other = ((a, name_a, b) if a["mean_loss"] < b["mean_loss"]
                           else (b, name_b, a))
    if lower["max_gnorm_250_499"] > other["max_gnorm_250_499"]:
        return "not-decidable", (
            f"gap {gap:.4f} favours {lname} but its second-half max gnorm "
            f"{lower['max_gnorm_250_499']:.3f} is higher")
    return lname, f"gap {gap:.4f} >= 0.05, second-half gnorm not higher"


def report(a, b):
    out = []
    for nm, r in (("0.85", a), ("1.2", b)):
        out.append(f"  arm {nm}: last step {r['last_step']}, n={r['n']} samples, "
                   f"mean loss {r['mean_loss']:.4f} (sd {r['sd']:.4f}, sem {r['sem']:.4f}), "
                   f"max gnorm 250-499 {r['max_gnorm_250_499']:.3f}"
                   + (f", REFUTED: {r['refuted']}" if r["refuted"] else ""))
    v, why = verdict(a, b)
    gap = abs(a["mean_loss"] - b["mean_loss"])
    sem_d = (a["sem"] ** 2 + b["sem"] ** 2) ** 0.5
    out.append(f"\n  gap {gap:.4f}, SEM of the difference {sem_d:.4f} "
               f"= {gap / sem_d:.2f} sigma"
               + ("  -- the 0.05 threshold is 0.83 sigma here, so a gap near it is "
                  "not a signal" if sem_d > 0.03 else ""))
    out.append(f"\n  VERDICT: {v}  ({why})")
    return "\n".join(out)


def _selftest():
    def synth(loss, gnorm=0.5, n=500, nan_at=None):
        for s in range(10, n, 10):
            g = 99.0 if nan_at == s else gnorm
            yield f"step {s}/499 | loss {loss:.3f} | lr 1e-3 | gnorm {g:.2f} |"
    a = read_arm(parse(list(synth(2.30))))
    b = read_arm(parse(list(synth(2.31))))
    v, why = verdict(a, b)
    assert v == "not-decidable", f"a 0.01 gap must be not-decidable, got {v} ({why})"
    b2 = read_arm(parse(list(synth(2.50))))
    v2, _ = verdict(a, b2)
    assert v2 == "0.85", f"a 0.20 gap favouring 0.85 must name it, got {v2}"
    # gnorm > 5 refutes outright, and the OTHER arm wins even with a worse loss.
    b3 = read_arm(parse(list(synth(2.00, nan_at=200))))
    v3, why3 = verdict(a, b3)
    assert v3 == "0.85" and "refuted" in why3, f"gnorm>5 must refute: {v3} {why3}"
    # The lower arm does NOT win if its second-half gnorm is higher.
    a4 = read_arm(parse(list(synth(2.00, gnorm=3.0))))
    b4 = read_arm(parse(list(synth(2.50, gnorm=0.5))))
    v4, _ = verdict(a4, b4)
    assert v4 == "not-decidable", f"lower arm with higher gnorm must not win: {v4}"
    print("read_lr_probe selftest OK: 4 cases (not-decidable, clear gap, refuted, "
          "lower-but-noisier)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(report(read_arm(parse(sys.argv[1])), read_arm(parse(sys.argv[2]))))
