#!/usr/bin/env python3
"""_drop_zombies must FIRE on a zombie, not merely run clean (tilerl, 2026-09-01).

`pgrep -f X` matches on argv and a zombie keeps its argv, so a name match is not
evidence of a live process. On 2026-09-01 `pgrep -f compile_worker | wc -l` returned
1577 where 1570 were zombies; the miscount was reported as CPU saturation on a machine
whose loadavg was 9.84 across 180 cores. The same substitution on a kill path is worse
than a miscount -- it acts on processes that already exited.

The stat codes below are captured from that pod: Z and Zs were the compile-worker
zombies, Zl a zombie with a thread, Ssl/Rsl the live ranks. A live-pod run cannot
stand in for this test, because when it was first attempted the zombies had been
reaped and the filter dropped nothing -- passing for the wrong reason.

    python3 scripts/test_drop_zombies.py --selftest
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import harness  # noqa: E402

CAPTURED = """1382919 Ssl
1367615 Z
1370843 Zs
1373210 Rsl
 999001 Zl
1382917 Ssl
"""
PIDS = ["1382919", "1367615", "1370843", "1373210", "999001", "1382917"]
LIVE = ["1382919", "1373210", "1382917"]


def with_table(table, pids):
    real = harness.subprocess.run
    harness.subprocess.run = lambda cmd, **kw: types.SimpleNamespace(stdout=table, returncode=0)
    try:
        return harness._drop_zombies(pids)
    finally:
        harness.subprocess.run = real


def main():
    got = with_table(CAPTURED, PIDS)
    assert got == LIVE, f"zombies not dropped: {got} != {LIVE}"
    # A table where nothing is a zombie must keep everything: the filter has to be
    # capable of a non-empty answer, or "drops everything" would pass the case above.
    allive = "1382919 Ssl\n1373210 Rsl\n"
    got2 = with_table(allive, ["1382919", "1373210"])
    assert got2 == ["1382919", "1373210"], f"live pids wrongly dropped: {got2}"
    assert harness._drop_zombies([]) == [], "empty input must not shell out"
    print("selftest ok: drops Z/Zs/Zl, keeps Ssl/Rsl, empty input short-circuits")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv else main())
