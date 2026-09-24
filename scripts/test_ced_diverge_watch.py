#!/usr/bin/env python3
"""Known-answer selftest for scripts/ced_diverge_watch.py (the CED divergence watchdog).

Synthetic known-answer pairs for the pure parse/decide layer. The REAL segment-1 log
known-answer test (12630-12700 must fire, 12000-12600 must stay silent) is not portable
to a checkout -- that log is pod-only -- so it runs on the pod with --once over the real
file; this selftest covers everything that is decidable without that log.

Run: python scripts/test_ced_diverge_watch.py --selftest
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WATCH_PY = os.path.join(ROOT, "scripts", "ced_diverge_watch.py")

from scripts.ced_diverge_watch import (  # noqa: E402
    DivergeDetector,
    LogParseError,
    cmdline_matches,
    parse_step_line,
    scan_lines,
)

HEALTHY = (
    "step {s}/38146 31% [main] | loss 1.100 | lr 1.00e-02 | gnorm 0.70 | "
    "9.9B tok | 27K tok/s/gpu | peak 43.16GiB | s/step 3.60"
)
VAL = "step 12000/38146 val 1.948 val_s 3.55 val_s_total 112.6"


def feed(det, pairs):
    for p in pairs:
        r = det.feed(*p)
        if r:
            return r
    return None


def t_parse_progress():
    assert parse_step_line(HEALTHY.format(s=12010)) == (12010, 1.1, 0.70)
    assert parse_step_line(VAL) is None  # val line shares the step prefix but is not progress
    assert parse_step_line("loading token cache domain x") is None
    for broken in (
        HEALTHY.format(s=1).replace("gnorm 0.70", "gnorm NaNx"),
        "step 5/38146 1% [main] | lr 1.00e-02",
    ):
        try:
            parse_step_line(broken)
        except LogParseError:
            pass
        else:
            raise AssertionError("shaped-but-broken line did not raise")


def t_healthy_silent():
    det = DivergeDetector()
    assert feed(det, [(s, 1.1, 0.7) for s in range(12000, 12100, 10)]) is None


def t_gnorm_consecutive():
    det = DivergeDetector(g_thresh=10.0, g_consec=3)
    # fires only on the THIRD consecutive high line
    assert feed(det, [(12640, 3.9, 239.0), (12650, 3.9, 1184.0)]) is None
    assert "consecutive" in det.feed(12660, 4.0, 5000.0)
    # one low line resets the run
    d2 = DivergeDetector(g_thresh=10.0, g_consec=3)
    assert feed(d2, [(1, 3, 20), (2, 3, 20), (3, 1, 0.5), (4, 3, 20)]) is None
    assert d2.feed(5, 3, 20) is None
    assert d2.feed(6, 3, 20) is not None


def t_val_line_does_not_break_count():
    det = DivergeDetector(g_thresh=10.0, g_consec=3)

    def hi(s, l, g):
        return HEALTHY.format(s=s).replace("loss 1.100", f"loss {l}").replace("gnorm 0.70", f"gnorm {g}")

    lines = [hi(12640, 3.9, "239.00"), VAL, hi(12650, 3.9, "1184.00"), hi(12660, 4.0, "5000.00")]
    reason, n = scan_lines(iter(lines), det)
    assert "consecutive" in (reason or "")
    assert n == 3  # val line skipped, not counted


def t_loss_mean_window():
    det = DivergeDetector(loss_window=50, loss_thresh=3.0)
    assert feed(det, [(s, 4.0, 0.5) for s in range(49)]) is None  # window not full
    assert "mean loss" in det.feed(49, 4.0, 0.5)  # 50th, mean 4.0 > 3
    # recoverable: 30 high + 20 low -> first full-window mean 2.8 < 3, stays silent
    d2 = DivergeDetector(loss_window=50, loss_thresh=3.0)
    pairs = [(s, 4.0, 0.5) for s in range(30)] + [(s, 1.0, 0.5) for s in range(30, 200)]
    assert feed(d2, pairs) is None


def t_nonfinite_fires():
    det = DivergeDetector()
    assert "nonfinite" in det.feed(1, float("nan"), 0.5)
    assert "nonfinite" in det.feed(2, 1.0, math.inf)


def t_scan_broken_raises():
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(HEALTHY.format(s=1) + "\n")
        f.write("step 2/38146 1% [main] | gnorm 0.5\n")
        f.write(HEALTHY.format(s=3) + "\n")
        path = f.name
    try:
        raised = False
        with open(path) as fh:
            try:
                scan_lines(fh, DivergeDetector())
            except LogParseError:
                raised = True
        assert raised
    finally:
        os.unlink(path)


def t_cmdline_nul_separated():
    # /proc cmdline joins argv with NUL; --name and the run name are separate tokens.
    # This is the mutation the live SIGTERM test caught: matching "--name X" with a space
    # matches nothing real and would have returned rc5 instead of killing a runaway.
    real = (
        "\x00".join(["torchrun", "--nnodes=1", "train.py", "--name", "v41_ced_0923", "--mix", "m.json"])
        + "\x00"
    )
    assert cmdline_matches(real, "v41_ced_0923")
    assert not cmdline_matches(real, "v41_ced_other")
    assert not cmdline_matches(real.replace("v41_ced_0923", "v41_ced_09234"), "v41_ced_0923")
    # worker process, not the launcher
    worker = "\x00".join(["python", "train.py", "--name", "v41_ced_0923"]) + "\x00"
    assert not cmdline_matches(worker, "v41_ced_0923")
    # the space-joined wrong representation must NOT count
    assert not cmdline_matches("torchrun train.py --name v41_ced_0923 ", "v41_ced_0923")


def _run_live(log, max_stale, out_path, interval=0.1):
    return subprocess.Popen(
        [
            sys.executable,
            WATCH_PY,
            "--name",
            "x",
            "--log",
            log,
            "--interval",
            str(interval),
            "--ready_wait",
            "5",
            "--max_stale",
            str(max_stale),
            "--dry_run",
        ],
        stdout=out_path,
        stderr=subprocess.STDOUT,
    )


def _wait_exit(p, deadline=5.0):
    end = time.time() + deadline
    while time.time() < end:
        rc = p.poll()
        if rc is not None:
            return rc
        time.sleep(0.05)
    p.kill()
    return None


def t_live_history_is_ignored_new_lines_watch():
    # The blocking defect: live mode on a pre-existing log replayed from offset 0 and would
    # SIGTERM a healthy resumed run for a PRIOR segment's divergence. Now it seeks to EOF.
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "l.log")
        with open(log, "w") as f:  # history already contains a full trigger
            for g in (239.0, 1184.0, 5000.0):
                f.write(f"step 12640/38146 33% [main] | loss 4.0 | gnorm {g}\n")
        with open(os.path.join(d, "o.txt"), "w") as out:
            p = _run_live(log, max_stale=20.0, out_path=out)
            time.sleep(0.6)
            assert p.poll() is None, "watchdog fired on pre-attach history (must seek EOF)"
            # append a NEW diverged sequence -> must trigger on the third new line
            with open(log, "a") as a:
                for g in (15.0, 16.0, 17.0):
                    a.write(f"step 14010/38146 40% [main] | loss 3.5 | gnorm {g}\n")
                    a.flush()
                    time.sleep(0.15)
            assert _wait_exit(p) == 2
        with open(os.path.join(d, "o.txt")) as got:
            assert "TRIGGER" in got.read()


def t_live_history_only_goes_stale_not_trigger():
    # history-only: with nothing appended after attach, outcome is STALE (4), never TRIGGER(2)
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "l.log")
        with open(log, "w") as f:
            for g in (239.0, 1184.0, 5000.0):
                f.write(f"step 12640/38146 33% [main] | loss 4.0 | gnorm {g}\n")
        with open(os.path.join(d, "o.txt"), "w") as out:
            p = _run_live(log, max_stale=0.6, out_path=out)
            assert _wait_exit(p) == 4


def t_live_chatter_does_not_reset_staleness():
    # non-progress lines must NOT refresh the progress staleness timer. Chatter is appended
    # CONTINUOUSLY, past max_stale, until the watchdog exits: a regression that refreshes the
    # timer on any line never goes stale and blows the 2.5s cap, so this test is discriminating
    # (the prior version stopped chattering before the wait, which a line-refreshing build also
    # survived by then going idle).
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "l.log")
        with open(log, "w") as f:
            f.write("step 12000/38146 31% [main] | loss 1.0 | gnorm 0.5\n")
        with open(os.path.join(d, "o.txt"), "w") as out:
            p = _run_live(log, max_stale=0.6, out_path=out)
            t0 = time.time()
            rc = None
            while time.time() - t0 < 2.5:
                rc = p.poll()
                if rc is not None:
                    break
                with open(log, "a") as a:  # continuous chatter, never a progress shape
                    a.write("saving checkpoint / warming compile / val noise\n")
                time.sleep(0.1)
            elapsed = time.time() - t0
            if rc is None:
                p.kill()
                raise AssertionError("continuous non-progress chatter kept staleness from firing")
            assert rc == 4
            # fired on the 0.6s PROGRESS timer while chatter was still flowing, not after it stopped
            assert elapsed < 1.5, f"stale fired late ({elapsed:.2f}s), timer may track chatter"


TESTS = [
    t_parse_progress,
    t_healthy_silent,
    t_gnorm_consecutive,
    t_val_line_does_not_break_count,
    t_loss_mean_window,
    t_nonfinite_fires,
    t_scan_broken_raises,
    t_cmdline_nul_separated,
    t_live_history_is_ignored_new_lines_watch,
    t_live_history_only_goes_stale_not_trigger,
    t_live_chatter_does_not_reset_staleness,
]


def selftest():
    for t in TESTS:
        t()
        print(f"ok {t.__name__}")
    print(f"selftest pass: {len(TESTS)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest:
        ap.error("this file runs only with --selftest")
    selftest()
