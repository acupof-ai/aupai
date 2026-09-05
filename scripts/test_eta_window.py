#!/usr/bin/env python3
"""The ETA is a window mean, so one slow interval does not move it by hours.

    python3 scripts/test_eta_window.py

WHY THIS EXISTS (AGENTS.md line 57). train.py printed `eta = (total_steps - step) * dt / 10`,
extrapolating the LAST 10-step interval over every remaining step. At 19,151 steps one interval
54 s slow printed 29 lost hours, and every checkpoint save -- which lands inside one interval and
costs seconds -- printed ~99 h. Consecutive lines then disagreed by tens of hours and the field
could not be used to decide anything.

WHAT THE ASSERTION HAS TO BE. Not "the ETA is smaller": a mean over a trailing window is smaller
than a spike by construction, so that passes on any smoothing including a wrong one. The load-
bearing property is a BOUND on how far one interval can move the estimate -- with a window of N,
a single slow interval moves the ETA by at most 1/N of what it moved before -- and that the
overrun is still REPORTED rather than absorbed. A smoother that silently ate the spike would be
worse than the spike: the save would become invisible.

EVERY NUMBER HERE COMES FROM CALLING train._eta_fields. The first draft read the window cap out of
the source text and then recomputed the mean locally, and two mutations of the real expression
SURVIVED that: widening the cap to 100000 moved the fixture's own window with it, and moving the
`{_overrun}` field ahead of the MFU group was invisible because the log line was a hand copy. A
test that reimplements the arithmetic it checks passes whenever both copies are wrong the same
way, which is why _eta_fields was lifted to module level (mutant-survived-because-worlds-shared-
the-error, 2026-09-05).
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import train  # noqa: E402

FAILS = []
TOTAL, STEP, NORMAL, SLOW = 19151, 1200, 6.0, 60.0
REMAINING = TOTAL - STEP


def _eta_h(win, dt):
    """Hours the real function projects, given a window and this interval."""
    eta, _ = train._eta_fields(win, dt, REMAINING)
    return eta / 3600


def _fill(dt, n):
    """A window the real function has filled with n intervals of dt, at its own cap."""
    win = []
    for _ in range(n):
        train._eta_fields(win, dt, REMAINING)
    return win


def main():
    cap = train.ETA_WINDOW

    # THE OLD BEHAVIOUR, so a green run means the fix is present rather than that the world is
    # easy: the defect reproduced from the formula it used, one 60 s interval where 6 s is normal.
    old_jump = REMAINING * SLOW / 10 / 3600 - REMAINING * NORMAL / 10 / 3600
    if old_jump < 20:
        FAILS.append(f"fixture has no power: the OLD formula moves the ETA by only {old_jump:.1f}h "
                     f"on a 10x interval, so there is nothing to fix")

    # 1. THE BOUND, which is the property and not just "smaller". Fed through the real function:
    #    a saturated window of normal intervals, then the same window plus one slow one. Widening
    #    the cap must NOT relax this -- the bound is stated against the cap the function reports.
    win = _fill(NORMAL, cap * 2)          # 2x the cap, so the window is genuinely trailing
    if len(win) != cap:
        FAILS.append(f"the window grew to {len(win)} entries against ETA_WINDOW={cap}; it is not "
                     f"being trimmed, so the ETA is a whole-run mean and never recovers from a "
                     f"slow patch")
    quiet = _eta_h(_fill(NORMAL, cap * 2), NORMAL)
    spiked = _eta_h(_fill(NORMAL, cap * 2), SLOW)
    new_jump = spiked - quiet
    if new_jump > old_jump / cap * 1.001:
        FAILS.append(f"one slow interval still moves the ETA by {new_jump:.2f}h; a window of {cap} "
                     f"bounds it at {old_jump / cap:.2f}h (was {old_jump:.2f}h unsmoothed)")

    # 2. AND IT STILL MOVES. A smoother that ignored the interval would pass 1 and be wrong the
    #    other way: the run really is slower and the estimate should say so.
    if new_jump <= 0:
        FAILS.append(f"a 10x slow interval moved the ETA by {new_jump:.3f}h, i.e. not at all -- the "
                     f"window must include the new interval, not discard it")

    # 3. THE OVERRUN IS REPORTED, not absorbed. This is what keeps a checkpoint save visible: the
    #    ETA barely moves (1) and the interval's own cost is printed as itself.
    _, overrun = train._eta_fields(_fill(NORMAL, cap * 2), SLOW, REMAINING)
    if not re.fullmatch(r" \| \+\d+s this interval", overrun):
        FAILS.append(f"a {SLOW:.0f}s interval against a {NORMAL:.0f}s window printed {overrun!r}; "
                     f"want a `+NNs this interval` field. Nothing printed means the save is "
                     f"invisible -- the failure a silent smoother introduces")

    # 4. AND A NORMAL INTERVAL PRINTS NOTHING, or the field is noise on every line.
    _, quiet_field = train._eta_fields(_fill(NORMAL, cap * 2), NORMAL, REMAINING)
    if quiet_field != "":
        FAILS.append(f"a normal interval printed {quiet_field!r}; the threshold is too tight")

    # 5. A COLD WINDOW DOES NOT DIVIDE BY ZERO and prints no overrun: the first interval of every
    #    segment is its own mean, so `over` is 0 and the guard on win_mean > 0 is what keeps the
    #    very first call (dt could be ~0 on a stub) off a ZeroDivisionError.
    eta0, field0 = train._eta_fields([], 0.0, REMAINING)
    if eta0 != 0.0 or field0 != "":
        FAILS.append(f"an empty window with dt=0 returned ({eta0}, {field0!r}), want (0.0, '')")

    # 6. THE TRACKIO PARSER STILL MATCHES, checked against the LINE THE CODE BUILDS. RunLog._STEP_RE
    #    (train.py:46) ends at `MFU (\d+)%`, so a field inserted ahead of MFU breaks the parse
    #    silently -- every metric on the line stops while the log looks richer, the reason the bare
    #    `lr` field is pinned. The f-string is extracted from train.main and evaluated here, so
    #    moving {_overrun} ahead of MFU is visible; a hand-copied line made that mutant survive
    #    (and made me write "after the ETA field" in train.py when the real bound is "after MFU").
    import inspect
    src = inspect.getsource(train.main)
    m = re.search(r"runlog\(\n(\s+f\"step \{step\}/\{total_steps\}.*?)\n\s+\)\n", src, re.S)
    if not m:
        FAILS.append("could not extract the step-line f-string from train.main")
    else:
        pieces = re.findall(r'f"(.*?)"\n', m.group(1) + "\n")
        tmpl = "".join(pieces)
        # THE TEMPLATE MUST ACTUALLY INTERPOLATE {_overrun}, asserted on the EXTRACTED SOURCE
        # before anything is rendered. The two renders below bind `_overrun` in their own env, so
        # they succeed identically whether or not the line uses it -- MEASURED: deleting
        # `{_overrun}` from train.py's f-string left this whole world green (mutant M3 survived a
        # 5-mutant run; the other four were killed by their own named assertions). A save that
        # overruns then prints nothing and the field is silently gone, which is the defect the
        # field exists to prevent.
        if "{_overrun}" not in tmpl:
            FAILS.append("train.main's step line does not interpolate {_overrun} -- the overrun "
                         "field is computed and thrown away, so a slow interval prints nothing. "
                         "The renders below cannot catch this: they bind _overrun themselves.")
        # AND IT MUST COME AFTER `MFU (\d+)%`, which is where RunLog._STEP_RE stops matching.
        # The M4 mutant (moving it ahead of MFU) is caught by the regex check below, but only
        # for the non-empty case; asserting the ORDER here holds for both.
        i_mfu, i_ov = tmpl.find("MFU {mfu"), tmpl.find("{_overrun}")
        if i_mfu >= 0 and 0 <= i_ov < i_mfu:
            FAILS.append(f"{{_overrun}} is at offset {i_ov}, ahead of MFU at {i_mfu} -- "
                         f"RunLog._STEP_RE ends at `MFU (\\d+)%` and matches on adjacency, so a "
                         f"field before it silently stops every trackio metric on the line")
        env = {"step": STEP, "total_steps": TOTAL, "phase": " [main]", "last": 2.451,
               "lrs": "muon 7.00e-03 embed 1.00e-01", "mfu": 0.41, "peak_gib": 61.01,
               "tps": 82000.0, "eta": 12.3 * 3600, "world": 8,
               # `dt` is 62's field (a65f595e): the step line now ends `| s/step {dt / 10:.4f}`.
               # Bound HERE rather than stubbed away, because the template is EXTRACTED from
               # train.main -- so every name the real line interpolates has to be bound or the
               # render raises NameError and worlds 6-7 report a fixture failure as a defect.
               # That is what happened when a65f595e landed: two BUGs reading "could not render
               # the step line: NameError: name 'dt' is not defined", which is the test correctly
               # noticing the line changed and incorrectly describing why.
               "dt": 22.824,
               "grad_norm": type("G", (), {"item": lambda self: 0.42})(),
               "optimizers": [type("O", (), {"param_groups": [{"lr": 7.0e-3}]})()],
               "Cfg": train.Cfg}
        for label, ov in (("without the overrun field", ""), ("with the overrun field",
                                                              " | +54s this interval")):
            try:
                line = eval('f"""' + tmpl.replace('"""', '') + '"""',
                            {"__builtins__": {}}, dict(env, _overrun=ov))
            except Exception as e:
                FAILS.append(f"could not render the step line {label}: {type(e).__name__}: {e}")
                continue
            g = train.RunLog._STEP_RE.search(line)
            if not g:
                FAILS.append(f"RunLog._STEP_RE does not match the step line as train.main BUILDS it "
                             f"{label} -- trackio would silently record nothing. Line: {line}")
            elif g.groups() != ("1200", "2.451", "7.00e-03", "0.42", "82", "41"):
                FAILS.append(f"RunLog._STEP_RE captured {g.groups()} {label}, want the six metrics "
                             f"unchanged -- the new field shifted a capture group")

    # 7. THE WINDOW IS RESET PER SEGMENT. `_eta_win = []` must sit beside t_log's initialisation
    #    inside main, not at module scope: carried across a resume it would average this segment's
    #    intervals with the previous one's, at a different shape and world size. An absent
    #    initialisation is a NameError at the first log step, which is how this was caught before
    #    it shipped.
    i_win, i_tlog = src.find("_eta_win = []"), src.find("t_log = time.time()")
    if i_win < 0:
        FAILS.append("no `_eta_win = []` in train.main -- an uninitialised window raises NameError "
                     "at the first log step")
    elif not (i_tlog >= 0 and abs(src[:i_win].count("\n") - src[:i_tlog].count("\n")) <= 6):
        FAILS.append("`_eta_win = []` is not beside `t_log = time.time()`; it must be reset with "
                     "the clock it measures")
    if "_eta_fields(_eta_win, dt," not in src:
        FAILS.append("train.main does not call _eta_fields(_eta_win, dt, ...) -- worlds 1-5 then "
                     "assert a function the training loop does not use")

    for f in FAILS:
        print(f"BUG {f}", file=sys.stderr)
    if not FAILS:
        print(f"eta window: the old formula moves the ETA {old_jump:.1f}h on one 10x interval, a "
              f"{cap}-interval window moves it {new_jump:.2f}h, and the overrun still prints as"
              f"{overrun}")
    print(f"eta window test: {'PASS (7 worlds)' if not FAILS else f'{len(FAILS)} BUG(S)'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
