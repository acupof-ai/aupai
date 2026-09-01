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

WHY THIS PRINTS A SEM. Two layers, and the name is wrong before the number is.

train.py:2518 syncs the loss only on `step % 10 == 9` and logs that SINGLE step, so
"mean over the last 100 steps" is a mean of TEN samples. And each of those ten is
rank 0's micro-batch alone: 2513-2516 all-reduces only the FINITENESS flag, never the
loss, so at world 7 batch 32 a sample covers 32 of 224 sequences (b0's second read).

So the quantity is "the mean of ten single-rank micro-batch losses sampled from the
last 100 steps", not "the mean training loss over the last 100 steps". The number is
unaffected -- the empirical sd already contains the 1/7 sampling variance -- but the
NAME is what the next person quotes. On arm 0.85 that is sd 0.135 -> SEM 0.043, and the
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
        # Named for what it is: see the module docstring. Ten samples, each one rank.
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
    # A gap of EXACTLY 0.05 must count as reaching the threshold, and binary floats do
    # not give you that: 2.3968 - 2.3468 is 0.04999999999999982, so a bare `< 0.05`
    # reports "gap 0.0500 < 0.05" -- a printed line that contradicts itself, and a
    # verdict decided by representation error rather than by the rule. The rule says
    # BELOW 0.05 is not decidable, so the comparison carries a tolerance well under the
    # 4-decimal precision the report prints.
    if gap < 0.05 - 1e-9:
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
    out.append(f"\n  quantity: mean of {a['n']} single-rank micro-batch losses "
               f"sampled from the last 100 steps (not a 100-step mean)")
    sig = gap / sem_d if sem_d else float("inf")
    # de: 0.83 sigma does not translate itself into "four times in ten" for a reader
    # who stops at the verdict line, so print the probability too. This is the chance
    # two arms with NO true difference produce a gap this large or larger.
    from statistics import NormalDist
    p_null = 2 * (1 - NormalDist().cdf(sig)) if sig < 40 else 0.0
    out.append(f"  gap {gap:.4f}, SEM of the difference {sem_d:.4f} = {sig:.2f} sigma"
               f"  -- two arms with NO true difference land here or higher "
               f"{p_null * 100:.0f}% of the time")
    # fb: the warning goes ON the verdict line, not beside it. A reader who stops at
    # the verdict is exactly the reader this protects, and a number that lives only in
    # the ruling does not exist. Likewise "not-decidable" must say WHY -- resolution,
    # not sample count -- because the two imply opposite next moves: run longer, or fix
    # the logging (the two layers that together take SEM down by about sqrt(70)).
    if v == "not-decidable" and not (a["refuted"] or b["refuted"]):
        line = (f"VERDICT: not-decidable (gap {gap:.4f} < 0.05, and that threshold is "
                f"{0.05 / sem_d:.2f} sigma on this instrument -- the limit is "
                f"RESOLUTION, not run length)")
    elif v == "not-decidable":
        line = f"VERDICT: not-decidable  ({why})"
    elif p_null >= 0.05:
        line = (f"VERDICT: {v} (gap {gap:.4f}) -- but two arms with no true difference "
                f"produce a gap this large or larger about {p_null * 100:.0f}% of the "
                f"time. Not enough on its own to change the recipe.")
    else:
        line = f"VERDICT: {v} (gap {gap:.4f}, {sig:.2f} sigma)  ({why})"
    out.append("\n  " + line)
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
    # THE BOUNDARY, in binary floats. An exact 0.05 gap subtracts to 0.049999...982, so
    # a bare `< 0.05` calls it not-decidable AND prints "gap 0.0500 < 0.05", which reads
    # as a typo and is actually the verdict turning on representation error.
    a5 = read_arm(parse(list(synth(2.3468))))
    b5 = read_arm(parse(list(synth(2.3968))))
    v5, why5 = verdict(a5, b5)
    assert v5 != "not-decidable", (
        f"a gap of exactly 0.05 reaches the threshold and must decide, got {v5} ({why5})")
    print("read_lr_probe selftest OK: 5 cases (not-decidable, clear gap, refuted, "
          "lower-but-noisier, exact-0.05 boundary)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(report(read_arm(parse(sys.argv[1])), read_arm(parse(sys.argv[2]))))
