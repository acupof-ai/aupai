#!/usr/bin/env python3
"""The rate printed on a val step is not a training rate (e1-44).

WHAT THIS EXISTS FOR. train.py prints one progress line per interval, and on a val_every
boundary that line's tok/s/gpu and s/step cover the validation too. Reading a throughput off
such a line understates it, and reading a RATIO off two of them does not cancel: validation
costs two arms different absolute amounts, so the depression is asymmetric. MEASURED, and this
is the episode that produced the module: sampling steps 3000/3200/3400 of the two batch-64
comparators gave 51K and 35K tok/s/gpu, ratio 1.457x, which I sent to the controller as the
basis for a six-card arm's ETA. Ten steps later the same files read 68K and 53K, ratio 1.283x
-- which agrees with the 1.263x already recorded from a CUDA-event profiler. The whole 0.174
of ratio was validation time, and it looked plausible because BOTH sides were contaminated.

    python3 scripts/val_step_rates.py --selftest    # the guard, no pod access
    python3 scripts/val_step_rates.py --log NAME    # split one pod log's rates and report

WHY A MODULE AND NOT A NOTE IN THE FACT. A note says "do not read val steps"; this fails when
someone does. `steady_rate` refuses a series it cannot separate rather than returning a number
that averages the two populations.
"""

import argparse
import os
import re
import subprocess
import sys

POD = os.path.expanduser("~/bin/pod")
POD_RUNS = "/work/aupai/runs"

# `step N/TOTAL ... 47K tok/s/gpu ... s/step 2.8143`. The rate is optional on some lines and
# s/step is absent from older runs, so both are matched separately against the same step.
STEP_RE = re.compile(r"^step (\d+)/(\d+)\b")
RATE_RE = re.compile(r"(\d+)K tok/s/gpu")
SSTEP_RE = re.compile(r"s/step (\d+\.\d+)")
VAL_EVERY_RE = re.compile(r"val_every (\d+)")


def parse_val_every(text):
    """val_every off the cfg line. None when absent -- then nothing can be split."""
    m = VAL_EVERY_RE.search(text)
    return int(m.group(1)) if m else None


def split_rates(text, val_every):
    """(clean, on_val) as {step: rate_K}, split by the val_every boundary.

    A step is a val step when step % val_every == 0 and step > 0. Step 0 prints no val.
    """
    clean, on_val = {}, {}
    for line in text.splitlines():
        m = STEP_RE.match(line.strip())
        if not m:
            continue
        r = RATE_RE.search(line)
        if not r:
            continue
        step, rate = int(m.group(1)), int(r.group(1))
        is_val = val_every and step > 0 and step % val_every == 0
        (on_val if is_val else clean)[step] = rate
    return clean, on_val


def steady_rate(text, tail=40, spread_max=0.15):
    """The training rate: median of the last `tail` CLEAN intervals.

    REFUSES rather than guessing, in three cases, because each would return a number that
    reads like a measurement:
      - no val_every on the cfg line: val steps cannot be identified, so no series is clean
      - no clean intervals at all: every line sampled was a boundary
      - the clean series itself spans more than `spread_max`: the run is not at steady state
        (warmup, contention, or a resume), and a median over it describes no regime
    """
    val_every = parse_val_every(text)
    if val_every is None:
        raise RuntimeError("no val_every on the cfg line -- val steps cannot be identified, "
                           "so no rate read from this log is known to be a training rate")
    clean, on_val = split_rates(text, val_every)
    if not clean:
        raise RuntimeError(f"every one of {len(on_val)} sampled step(s) is a val_every-"
                           f"{val_every} boundary; no clean interval to read")
    steps = sorted(clean)[-tail:]
    vals = sorted(clean[s] for s in steps)
    lo, hi = vals[0], vals[-1]
    if hi and (hi - lo) / hi > spread_max:
        raise RuntimeError(f"the clean series spans {lo}K-{hi}K over the last {len(steps)} "
                           f"interval(s), {(hi - lo) / hi:.0%} > {spread_max:.0%} -- not a "
                           f"steady state, so its median describes no regime")
    mid = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1]
                                                     + vals[len(vals) // 2]) / 2
    return mid, clean, on_val, val_every


def pod_read(name):
    """Progress lines and the cfg line of <name>.log. One call, plain grep (pod argv
    cannot carry prose)."""
    cmd = (f"grep -E '^step [0-9]+/[0-9]+ ' {POD_RUNS}/{name}.log; "
           f"grep -m1 'cfg batch' {POD_RUNS}/{name}.log")
    r = subprocess.run([POD, cmd], capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and not r.stdout.strip():
        raise RuntimeError(f"pod read of {name}.log failed: {(r.stderr or '')[:200]}")
    return r.stdout


def _selftest():
    """The guard: a fixture whose two populations differ, and the ratio that moves.

    The fixtures reproduce the measured shape of b0_e1p_dense and b0_e1p_moe48 -- clean 68K
    and 53K, val-step 45-51K and 30-35K -- so this is a regression test on the split, not on
    invented numbers.
    """
    def mk(clean, val, val_every=200, first=2600, last=3800):
        out = [f"cfg batch 16 accum 2 seq 4096 val_every {val_every}"]
        for s in range(first, last + 10, 10):
            k = val if (s % val_every == 0) else clean
            out.append(f"step {s}/3815 42% | loss 1.9 | {k}K tok/s/gpu | ETA 0.3h")
        return "\n".join(out)

    dense, moe = mk(68, 48), mk(53, 32)

    # THE SPLIT ITSELF. Both populations must be found, and they must not overlap: if the
    # split cannot separate them this guard proves nothing about the readings it protects.
    r_d, clean_d, val_d, ve = steady_rate(dense)
    r_m, clean_m, val_m, _ = steady_rate(moe)
    assert ve == 200, ve
    assert (r_d, r_m) == (68, 53), (r_d, r_m)
    assert set(val_d) == set(range(2600, 3801, 200)), sorted(val_d)
    assert max(val_d.values()) < min(clean_d.values()), (val_d, clean_d)
    print(f"  ok   split found {len(clean_d)} clean and {len(val_d)} val-step interval(s), "
          f"disjoint")

    # THE FAILURE THIS MODULE EXISTS FOR, asserted as a number rather than described: the
    # ratio taken off val steps must differ from the true one by more than the spread between
    # the two instruments that measure it (1.263x vs 1.283x, 1.6%).
    true = r_d / r_m
    contaminated = max(val_d.values()) / max(val_m.values())
    assert abs(true - 1.283) < 0.01, true
    assert contaminated - true > 0.10, (contaminated, true)
    print(f"  ok   val-step ratio {contaminated:.3f}x against the true {true:.3f}x, "
          f"{contaminated - true:+.3f} -- far outside the 1.6% inter-instrument spread")

    # AND THE MUTATION THAT MATTERS, which is not the one I first wrote. My first version
    # asserted that medianing clean and val-step rates together gives a wrong number; the
    # assertion FAILED, and it was right to. Val steps are 1 line in 20, so a median over the
    # whole series still lands on the clean rate -- a non-splitting reader who takes a median
    # gets the right answer. The trap is not aggregation, it is SINGLE-POINT SAMPLING AT ROUND
    # STEPS: a reader picks 3000, 3200, 3400 because they are round, and round step numbers ARE
    # the val_every boundaries. Every round step in this fixture is contaminated, and that is
    # what the guard has to assert.
    every = {**clean_d, **val_d}
    naive_median = sorted(every.values())[len(every) // 2]
    assert naive_median == r_d, (
        f"median over the whole series is {naive_median}K, not the clean {r_d}K -- if this ever "
        f"changes, the framing in this comment is wrong and aggregation is a trap too")
    round_steps = [s for s in sorted(every) if s % 200 == 0]
    assert round_steps and all(every[s] == 48 for s in round_steps), \
        f"expected every round step contaminated; got {[(s, every[s]) for s in round_steps]}"
    print(f"  ok   a median over everything is fine ({naive_median}K); it is the {len(round_steps)}"
          f" round steps that are all val steps, and picking those is the trap")

    # THREE REFUSALS, each returning no number rather than a wrong one.
    for text, want in (
        ("step 100/3815 | 68K tok/s/gpu", "no val_every"),
        ("cfg batch 16 accum 2 seq 4096 val_every 200\nstep 200/3815 | 48K tok/s/gpu\n"
         "step 400/3815 | 47K tok/s/gpu", "boundary"),
        (mk(68, 48, first=2600, last=2700) + "\nstep 2710/3815 | 20K tok/s/gpu", "steady"),
    ):
        try:
            steady_rate(text)
            raise AssertionError(f"expected a refusal naming {want!r}, got a number")
        except RuntimeError as e:
            assert want in str(e), f"refusal for {want!r} said: {e}"
    print("  ok   refuses on no val_every, on all-boundary samples, and on a non-steady series")

    # A CLEAN RUN MUST NOT BE FLAGGED. val_every 500 over 100 steps never prints a val line,
    # which is why eff.bf16_master_dense_200m_tps and eff.bf16_vs_fp32_master_dense_200m are
    # unaffected: their logs (b0_p5_ctrl_bf16, ...v2) have no boundary to land on. A guard
    # that flags them too would make the real finding unreadable.
    short = mk(62, 0, val_every=500, first=50, last=100)
    rate, clean, on_val, _ = steady_rate(short)
    assert rate == 62 and not on_val, (rate, on_val)
    print("  ok   a run whose val_every exceeds its length has no val step and is not flagged")

    print("\nval_step_rates selftest OK: the two populations split disjointly, the val-step "
          "ratio\nis off by more than the inter-instrument spread, a non-splitting reader is "
          "caught,\nand the three unreadable cases refuse instead of returning a number")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--log", help="a run name under /work/aupai/runs, without .log")
    ap.add_argument("--tail", type=int, default=40)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.log:
        ap.error("--log NAME or --selftest")

    text = pod_read(a.log)
    try:
        rate, clean, on_val, ve = steady_rate(text, tail=a.tail)
    except RuntimeError as e:
        print(f"REFUSING: {e}", flush=True)
        return 1
    print(f"{a.log}: val_every {ve}, {len(clean)} clean and {len(on_val)} val-step interval(s)")
    print(f"  training rate (median of the last {min(a.tail, len(clean))} clean): {rate}K "
          f"tok/s/gpu")
    if on_val:
        vv = sorted(on_val.values())
        print(f"  val-step rates span {vv[0]}K-{vv[-1]}K and are NOT training rates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
