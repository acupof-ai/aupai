#!/usr/bin/env python3
"""The warmdown start has ONE implementation, and the WSD JOIN print reads it.

WHY THIS EXISTS. train.py computed the warmdown start twice: `lr_mult` derived it from
`wd_steps`, and the WSD JOIN print recomputed `total_steps - max(1, int(Cfg.warmdown *
total_steps))` inline. They agreed at every value anyone had looked at, which is why the
duplication survived -- two copies of a formula are not a bug until one of them is edited, and
then the print reports a step the scheduler does not use. The print exists so that a two-stage
launch can be verified at a glance; a print that can drift from the schedule it describes is
worse than no print, because it is read as evidence.

WHAT IS CHECKED, and the first assertion is the one with teeth:

  1. warmdown_start agrees with lr_mult's OWN behaviour -- the returned step is the first step
     at which lr_mult departs from 1.0, and the step before it is still exactly 1.0. This is
     measured from the scheduler rather than from a second copy of the formula, so it stays
     true only while there is one implementation. Swap in any off-by-one and it fires.
  2. The function reads the cfg it is HANDED, not a global. Two configs differing only in
     warmdown must give different starts, and each must match what lr_mult does under that
     same config. 4c asked for this world as a class-vs-instance divergence; see the note
     below on why it is written as two configs instead.
  3. train.py's source holds no second copy of the formula. A behavioural test cannot see a
     duplicate that happens to agree today, and the whole defect was a duplicate that agreed.

ON 4c's REQUESTED WORLD, and a correction to what I reported. I told 4c the two sites used
different objects -- the scheduler an instance `cfg`, the print the class `Cfg`. That is WRONG:
`set_schedule(optimizers, step, total_steps, Cfg, args.lr_scale)` passes the class too, so both
sites always read the same object and no class-vs-instance divergence was ever reachable. The
real defect was only the duplicated formula. Assertion 2 keeps the spirit of the requested world
-- two configs whose warmdown differs must not collapse to one answer -- which is what proves the
function is parameterised rather than reading a global, and that IS a live hazard: a plain
`Cfg.warmdown` inside the helper would pass assertions 1 and 3 and fail this one.

    python3 scripts/test_warmdown_start.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

_fails = []


def check(ok, msg):
    if not ok:
        _fails.append(msg)


def _first_below_one(total, cfg, lr_mult):
    """The first step at or after warmup where lr_mult drops BELOW 1.0.

    Walks rather than solves, on purpose: solving would re-derive the formula under test.

    NOTE ON THE OFF-BY-ONE, which this test got wrong on its first run and the code did not:
    lr_mult returns exactly 1.0 AT wd_start, because progress is then 0, cosine is 1, and
    final + (1-final)*1 == 1. So the first step strictly below 1.0 is wd_start + 1, and an
    assertion of `departure == wd_start` fires on correct code. wd_start is the last step of the
    stable phase, not the first of the tail.
    """
    for step in range(int(cfg.warmup), total + 1):
        if lr_mult(step, total, cfg) < 1.0:
            return step
    return None


def main():
    import train

    lr_mult, warmdown_start, Cfg = train.lr_mult, train.warmdown_start, train.Cfg

    # (1) THE SCHEDULER'S OWN BEHAVIOUR IS THE REFERENCE. Several totals, including the two the
    # live runs use (10172 and 19151) and one small enough that max(1, ...) can bind.
    #
    # THE TOTALS ARE CONSTRAINED, and the constraint is a fact about the schedule rather than a
    # convenience: the reference is "the step at which lr_mult drops below 1.0", which exists only
    # when the warmdown start is at or after warmup. Two ways it does not: a total below warmup
    # (lr_mult never leaves the warmup branch at all -- total=3 against warmup=20), and a total
    # where warmdown*total lands inside warmup, so warmup SHADOWS the tail (total=40 warmdown=0.65
    # gives start 14 against warmup 20, and lr_mult is still ramping there). Both are real
    # pathologies of the CONFIG, not of the formula, and asserting on them reports 12 failures
    # against correct code -- which this test did on its first two runs. Assertion (1) therefore
    # covers the region where the schedule has a stable phase, and every (total, warmdown) below
    # is verified to be in it -- measured, not assumed: the smallest start any pair here produces
    # is 35 (total=100, warmdown=0.65) against warmup 20.
    for total in (10172, 19151, 100):
        for warmdown in (0.65, 0.1, 0.5, 0.01):
            cfg = _cfg_with(Cfg, warmdown=warmdown)
            start = warmdown_start(total, cfg)
            check(
                int(cfg.warmup) <= start < total,
                f"warmdown_start({total}, warmdown={warmdown}) = {start}, outside "
                f"[{int(cfg.warmup)}, {total}): the tail would be empty, or start inside warmup",
            )
            dep = _first_below_one(total, cfg, lr_mult)
            check(
                dep == start + 1,
                f"THE PRINT AND THE SCHEDULE DISAGREE at total={total} warmdown={warmdown}: "
                f"warmdown_start says {start}, so lr_mult must be 1.0 at {start} and below 1.0 "
                f"from {start + 1}; it actually first drops at {dep}. That is two "
                f"implementations of one formula, which is the defect this test exists for -- "
                f"the WSD JOIN line would report a step the schedule does not use",
            )
            check(
                lr_mult(start, total, cfg) == 1.0,
                f"lr_mult had already left the stable phase AT the reported start {start} "
                f"(total={total}, warmdown={warmdown}); wd_start is the last stable step",
            )

    # (2) THE FUNCTION READS THE CFG IT IS HANDED. A helper hardcoding Cfg.warmdown passes every
    # assertion above and fails here, which is why this world is separate.
    total = 10172
    a = _cfg_with(Cfg, warmdown=0.65)
    b = _cfg_with(Cfg, warmdown=0.10)
    sa, sb = warmdown_start(total, a), warmdown_start(total, b)
    check(
        sa != sb,
        f"two configs differing ONLY in warmdown (0.65 vs 0.10) gave the same start {sa}: "
        f"warmdown_start is not reading the cfg it was handed, so the WSD JOIN line would "
        f"describe a schedule other than the one running",
    )
    check(
        _first_below_one(total, a, lr_mult) == sa + 1 and _first_below_one(total, b, lr_mult) == sb + 1,
        f"the per-cfg starts {sa}/{sb} do not match lr_mult's behaviour under those same "
        f"configs -- the two functions disagree about which cfg they are reading",
    )

    # (3) NO SECOND COPY OF THE FORMULA IN THE SOURCE. Assertions 1-2 cannot see a duplicate that
    # agrees today, and a duplicate that agreed today is exactly what was here.
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as fh:
        src = fh.read()
    body = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    dupes = re.findall(r"max\(\s*1\s*,\s*int\(\s*\w+\.warmdown\s*\*", body)
    check(
        len(dupes) == 1,
        f"the warmdown-start formula `max(1, int(cfg.warmdown * total))` appears {len(dupes)} "
        f"times in train.py's code (comments stripped). It must appear exactly once, inside "
        f"warmdown_start; every other site calls that. Two copies agreed on every value "
        f"anyone checked until one was edited",
    )
    check(
        "warmdown_start(total_steps, Cfg)" in body,
        "the WSD JOIN line does not call warmdown_start(total_steps, Cfg). It is the reason "
        "this function exists: the print must report the scheduler's number, not its own",
    )

    if _fails:
        for f in _fails:
            print(f"FAIL: {f}")
        print(f"\n{len(_fails)} failure(s)")
        return 1
    print("ok: warmdown_start is the single implementation, and the WSD JOIN line reads it")
    return 0


def _cfg_with(base, **over):
    """A cfg object with `over` applied, without mutating the real Cfg.

    A plain subclass: Cfg is a class of class-level attributes and every reader uses attribute
    access, so a subclass overriding one attribute is exactly the shape the callers see.
    """
    return type("CfgUnder", (base,), dict(over))


if __name__ == "__main__":
    sys.exit(main())
