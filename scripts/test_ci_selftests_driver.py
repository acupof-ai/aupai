#!/usr/bin/env python3
"""The ci-selftests DRIVER's own contract: loud failures, streamed evidence, group timeout.

The driver (harness._run_ci_targets / _run_one_selftest) is the thing CI trusts to run every
registered selftest. 3b measured §298 against it: a child that ran past the timeout was killed
but the driver exited ~115s later with 0 buffered bytes and no clear nonzero signal, so the
authoritative failure list was unreachable. These three known answers pin that shape:
  1. a passing target -> rc 0;
  2. a failing target -> rc 1 and the FAIL line names it (continue-all, not silent);
  3. a target that prints a marker then hangs -> rc 1, a machine-readable
     "CI-SELFTESTS: TIMEOUT after Ns" line, and the marker printed BEFORE the hang is present
     (streaming means a killed child still leaves evidence). The timeout must be roughly the
     bound, not the child's full sleep.
Run: python3 scripts/test_ci_selftests_driver.py --selftest
"""
import io
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import harness as H  # noqa: E402

PASS_PY = "import sys; print('pass-marker'); sys.exit(0)\n"
FAIL_PY = "import sys; print('fail-marker'); sys.exit(7)\n"
HANG_PY = (
    "import time, sys\n"
    "print('pre-hang-marker', flush=True)\n"
    "while True:\n"
    "    time.sleep(1)\n"
)


def _write(d, name, body):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(body)
    return name


def _drive(d, targets, timeout, fail_fast=False):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        rc = H._run_ci_targets(list(targets), {}, timeout, fail_fast,
                               {"CUDA_VISIBLE_DEVICES": "", "PATH": os.environ.get("PATH", "")}, d)
    finally:
        sys.stdout = old
    return rc, buf.getvalue()


def _selftest():
    bad = 0
    n = 0

    def check(cond, msg):
        nonlocal bad, n
        n += 1
        if not cond:
            print("  FAIL " + msg)
            bad += 1

    with tempfile.TemporaryDirectory() as d:
        ok = _write(d, "ok_case.py", PASS_PY)
        badf = _write(d, "bad_case.py", FAIL_PY)
        hang = _write(d, "hang_case.py", HANG_PY)

        rc, out = _drive(d, [ok], 30)
        check(rc == 0 and "target(s) passed" in out, f"pass case: rc={rc}, no pass line")

        # continue-all: a failing target is named and the run is nonzero.
        rc, out = _drive(d, [badf, ok], 30)
        check(rc == 1 and "FAIL bad_case.py" in out and "fail-marker" in out,
              f"fail case: rc={rc}, failure not named/streamed")
        check("ok_case.py" in out, "continue-all: the pass target never ran after the failure")

        # the load-bearing one: hang -> TIMEOUT line, nonzero, pre-hang evidence retained.
        t0 = time.time()
        rc, out = _drive(d, [hang], 2)
        dt = time.time() - t0
        check(rc == 1, f"timeout case: rc={rc}, must be 1")
        check("CI-SELFTESTS: TIMEOUT after 2s running hang_case.py" in out,
              "timeout case: missing machine-readable TIMEOUT line")
        check("pre-hang-marker" in out,
              "timeout case: pre-hang output lost (buffered until exit, §298)")
        check(dt <= 15, f"timeout case: driver waited {dt:.0f}s, the group kill did not bound it")

        # the hung child's whole group is actually dead, not reparented and still sleeping.
        # os.killpg on the returned pid is impossible here (pid not exposed), so assert the
        # driver RETURNED promptly -- the bound above -- which wait() on a live group would deny.
    print(f"ci-selftests driver selftest: {n - bad}/{n} pass")
    return bad


if __name__ == "__main__":
    sys.exit(1 if _selftest() else 0)
