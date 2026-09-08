"""StepProfiler's arithmetic and its labels, without a card.

The class is timing-only, but its reduce differential is real arithmetic and its LABELS make
claims that the arithmetic cannot check. Both kinds are asserted here, because the defect this
file exists for was a label: the first version printed "unmeasurable at accum 1" whenever no
no-sync arm ran, which is also true at accum 4 with DDP off -- correct behaviour, false stated
cause, and every behavioural assertion passed. So the cases below assert on the printed string.

Mutants this catches: summing instead of averaging the no-sync arm turns +15.0 into -5.0 (a
negative reduce cost, which reads as the reduce making the step FASTER); dropping the spread
verdict lets a differential smaller than its own arm's noise print as a measurement; dropping
the cadence rounding lets --profile_step_every 25 arm on steps that never reach line().

Execs the class body rather than importing train.py, which builds its parser at import.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeEvent:
    """A cuda event whose elapsed_time is exactly the gap the test scheduled."""

    def __init__(self, t):
        self.t = t

    def record(self):
        pass

    def elapsed_time(self, other):
        return other.t - self.t


class FakeCuda:
    def __init__(self):
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


def _load():
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as fh:
        src = fh.read()
    body = src[src.index("class StepProfiler:"):src.index("def opt_snapshot(")]
    ns = {}
    exec(body, ns)  # noqa: S102 -- the class under test, isolated from argparse at import
    return ns["StepProfiler"]


def main():
    SP = _load()
    torch = FakeTorch()
    clock = [0.0]

    def mk():
        clock[0] += 1.0
        return FakeEvent(clock[0])

    def run(nolast_ms, last_ms, tm=None, **kw):
        """One profiled step: a no_sync backward per entry in nolast_ms, then the reduce one."""
        p = SP(10, tm or FakeTorch(), **kw)
        p._mk = mk
        p.step_begins(10)
        for each in nolast_ms:
            p._ev["bwd_nolast"] = mk()
            clock[0] += each - 1
            p.stop("bwd_nolast")
        p._ev["bwd_last"] = mk()
        clock[0] += last_ms - 1
        p.stop("bwd_last")
        return p

    # OFF BY DEFAULT means recording nothing, not recording and discarding.
    off = SP(0, torch)
    off.step_begins(10)
    assert off.active is False, "every=0 must never arm"
    assert off.line(2.8) is None
    off.start("fwd")
    off.stop("fwd")
    assert off._acc == {}, "an off profiler must record nothing"

    armed = SP(10, torch)
    armed.step_begins(9)
    assert armed.active is False, "must not arm off-cadence"
    armed.step_begins(10)
    assert armed.active is True

    # CADENCE SNAPS TO THE LOG'S OWN. line() is reached only when step % 10 == 0, so an every
    # that is not a multiple of 10 would arm on steps that never print -- events recorded, sync
    # paid, nothing emitted.
    assert SP(25, torch).every == 30, "25 must round up to 30, not profile into a void"
    assert SP(3, torch).every == 10, "below the log cadence, snap to it"
    assert SP(20, torch).every == 20, "an exact multiple is left alone"
    assert SP(0, torch).every == 0, "off stays off -- rounding must not switch it on"
    # The requested value survives beside the effective one, so the caller can say it rounded
    # rather than leaving a user who asked for 25 to work out why the log says 30.
    assert SP(25, torch).requested == 25
    assert SP(0, torch).requested == 0

    # A PROFILED STEP CAN END WITHOUT A PRINT: the non-finite-grad path does `continue` before
    # the logging block, so fwd/bwd sit in _acc and active stays True. The next profiled step's
    # step_begins must clear them, or its line reports two steps of work as one.
    leak = SP(10, FakeTorch())
    leak._mk = mk
    leak.step_begins(10)
    for r in ("fwd", "bwd_nolast", "bwd_last"):
        leak.start(r)
        clock[0] += 9
        leak.stop(r)
    for s in range(11, 20):
        leak.step_begins(s)
    leak.step_begins(20)
    assert leak._acc == {}, "a skipped profiled step must not carry into the next one"
    assert leak._ev == {}, "no start() may outlive its step"

    # ACCUM 4: three suppressed backwards at 10 ms, one with the reduce at 25.
    # bwd_nolast prints the SUM (30) and the differential uses the MEAN (10), so 25-10 = +15.
    p = run([10.0, 10.0, 10.0], 25.0, torch)
    line = p.line(0.100)
    assert "bwd_last_excess +15.0" in line, line
    assert "n=3 mean 10.0 spread 0.0" in line, line
    assert "bwd_nolast 30.0" in line, line
    assert "bwd_last 25.0" in line, line
    # NOT "reduce": the excess contains the reduce and is not only the reduce, and a field name
    # is what survives into a spreadsheet. The docstring is not.
    assert "reduce" not in line, line
    assert torch.cuda.syncs == 1, f"one sync per print, got {torch.cuda.syncs}"

    # NO PER-STEP VERDICT, and n/mean/spread printed instead. A range grows with n on its own, so
    # a spread-vs-excess test fires more often at accum 8 than accum 4 on identical hardware.
    noisy = run([5.0, 15.0, 10.0], 12.0).line(0.100)
    assert "unresolvable" not in noisy, noisy
    assert "spread 10.0" in noisy, noisy
    assert "+2.0" in noisy, noisy

    # NO NO-SYNC ARM: the label must name the missing arm, not assert a value of accum. This is
    # the accum-4-DDP-off case as much as the accum-1 case, and the first version said "accum 1".
    solo = run([], 25.0).line(0.100)
    assert "no no-sync backward ran" in solo, solo
    assert "accum 1, or DDP off" in solo, solo
    assert "excess +" not in solo and "excess -" not in solo, solo

    # --no_bucket_view MAKES THE FIELD A DIFFERENT QUANTITY (the excess plus a full grad->bucket
    # copy), so the name changes with it rather than meaning two things.
    bv = run([10.0, 10.0, 10.0], 25.0, bucket_view=False).line(0.100)
    assert "bwd_last_excess+bucket +15.0" in bv, bv

    # THE DENOMINATOR IS NOT THIS STEP: step_s is a ten-step wall mean carrying val passes and
    # checkpoint writes. The name says so, and the field is signed.
    rest = run([10.0, 10.0, 10.0], 25.0)
    line3 = rest.line(0.100)
    assert "rest_vs_10step_mean 45.0" in line3, line3
    assert "| rest " not in line3, "must not claim to be unattributed step time"
    assert rest.line(0.100) is None, "line() is one-shot per step"
    fast = run([10.0, 10.0, 10.0], 25.0).line(0.010)
    assert "rest_vs_10step_mean -45.0" in fast, fast

    print("ok  off-by-default, arming, cadence, excess, inputs-not-verdict, missing arm,")
    print("    bucket_view label, signed residual, one-shot, one sync")
    for x in (line, noisy, solo, fast):
        print("   ", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
