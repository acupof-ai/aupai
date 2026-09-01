#!/usr/bin/env python3
"""spawned_scripts_exist must still catch c3a47e8 after being made fast.

It timed out eight runs in a row at a 5s budget and blocked every commit in the repo
while reporting "has not actually run since" -- a red no fix could clear. The cost was
one import: fla.ops.kda at 6.07s of 7.6s, reached transitively by `import train` in
datagen/pretokenize.py. torch itself is 0.92s, and threading the three subprocesses made
it worse (11.4s) because they contend rather than overlap.

So it resolves imports instead of executing them. THAT TRADE IS ONLY WORTH ANYTHING IF
IT STILL FAILS ON THE DEFECT, and three versions of the fast check did not:

  find_spec + prepended sys.path  -> PASS. find_spec reads the CALLING process's path,
                                     and harness.py runs with the real scripts/ on it.
  find_spec + replaced sys.path   -> PASS. find_spec consults sys.modules first, and
                                     harness.py has already imported these modules.
  PathFinder + own-path ownership -> PASS. harness.py lives in scripts/, which the
                                     broken script never adds, so it read as
                                     third-party and was skipped.

Each looked correct and tested nothing. This file builds the c3a47e8 tree and asserts
FAIL, so a fourth such version cannot land quietly.

    python3 scripts/test_spawned_fast.py --selftest
"""
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _c3a47e8_tree():
    """harness.py in scripts/, the spawned script in datagen/, only ROOT inserted.

    This is the tree as c3a47e8 left it: the file is at its path and cannot import.
    """
    d = tempfile.mkdtemp(prefix="c3a47e8")
    os.makedirs(os.path.join(d, "scripts"))
    os.makedirs(os.path.join(d, "datagen"))
    open(os.path.join(d, "scripts", "harness.py"), "w").write("x = 1\n")
    open(os.path.join(d, "datagen", "pretokenize.py"), "w").write(
        "import os, sys\n"
        "ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
        "sys.path.insert(0, ROOT)\n"
        "import harness\n")
    return d


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    sys.argv = ["harness"]
    import harness

    orig = harness._SPAWNED_SCRIPTS
    d = _c3a47e8_tree()
    try:
        harness._SPAWNED_SCRIPTS = [("datagen/pretokenize.py", "the c3a47e8 case")]
        status, why = harness.check_spawned_scripts_exist(d)[:2]
        assert status == harness.FAIL, (
            f"the c3a47e8 tree PASSED: {why}. The script is at its path and cannot "
            f"import harness -- if this check cannot see that, it is fast and useless.")
        assert "pretokenize" in why, f"the refusal does not name the file: {why}"
    finally:
        harness._SPAWNED_SCRIPTS = orig
        shutil.rmtree(d, ignore_errors=True)

    # And the real tree must pass, or the check is a refusal wearing a reason.
    t0 = time.time()
    status, why = harness.check_spawned_scripts_exist(ROOT)[:2]
    dt = time.time() - t0
    assert status == harness.PASS, f"the real tree FAILs: {why}"

    # The budget it kept blowing is 5s. Assert well inside it: at 4.9s this would be
    # green while one added import puts it back to timing out every run.
    assert dt < 2.0, (
        f"{dt:.1f}s -- the 5s budget is met by a margin thin enough that the next "
        f"import re-breaks it; the point of resolving instead of executing was headroom")

    print(f"selftest OK (FAILs on the c3a47e8 tree, real tree PASS in {dt * 1000:.0f}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv else (print(__doc__) or 0))
