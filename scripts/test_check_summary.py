#!/usr/bin/env python3
"""`harness check`'s summary must state what it did NOT run.

Same sha 2dfe207a, same 56 checks: the Mac printed 0 FAIL and exited 0, the pod FAILed
9. The 15 that skip on a dev box are exactly where those FAILs live, and the last line
a reader acts on never mentioned them -- so reading only the Mac reports green.

The banner has said "a check that cannot run is a FAILURE, never a pass" all along, and
the TIMEOUT branch honours it. SKIP did not.

    python3 scripts/test_check_summary.py --selftest
"""
import os
import re
import subprocess
import sys

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                      text=True).stdout.strip() or os.path.dirname(
                          os.path.dirname(os.path.abspath(__file__)))


def main():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "harness.py"), "check"],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    out = r.stdout
    n_skip = len(re.findall(r"\[SKIP\]", out))
    if not n_skip:
        print("selftest OK (nothing skipped on this machine; nothing to state)")
        return 0

    # The count must be in the summary, not merely inferable by counting [SKIP] lines
    # in the body. A reader acts on the last line.
    tail = out.strip().splitlines()[-6:]
    joined = "\n".join(tail)
    m = re.search(r"(\d+) did NOT run here", joined)
    assert m, (
        f"{n_skip} check(s) were skipped and the summary does not say so. A reader sees "
        f"'0 FAIL' and concludes green, while the checks that fail on the pod were "
        f"never attempted. Tail was:\n{joined}")
    assert int(m.group(1)) == n_skip, \
        f"summary says {m.group(1)} skipped, body has {n_skip}"

    # The denominator too: "0 FAIL" alone is the claim that misled; "0 FAIL of N run"
    # cannot be read as "everything passed".
    assert re.search(r"\d+ FAIL of \d+ run", joined), \
        f"the summary states no denominator, so 0 FAIL still reads as all-clear:\n{joined}"

    # THE ORDERING CASE, ON A TEXT FIXTURE -- NOT ON A LIVE `harness check`. The assertions
    # above can only run when THIS machine has skips and no failing invariant, so they cannot
    # reach the case that matters: a run where `bad` is non-empty. That is exactly where the
    # old print order dropped the denominator line behind an early `return 1`, and the live
    # dependency is also what made this file's own failure mode a strike-2 timeout under the
    # CI driver. Asserted on the source and on a fixture instead, so it cannot depend on the
    # timing or the environment of the run that happens to be executing it.
    #
    # MUTATION: put the `if bad: return 1` back above the `if skipped:` print and this reds.
    src = open(os.path.join(ROOT, "scripts", "harness.py"), encoding="utf-8").read()
    i_bad = src.find('if bad:\n        print(f"\\n{len(bad)} invariant(s) FAILED')
    i_sum = src.find('NOT run here: {\', \'.join(skipped)}')
    assert i_bad != -1, "cannot locate the invariant-FAIL print in harness.py"
    assert i_sum != -1, "cannot locate the skip/denominator print in harness.py"
    # the early return must not sit between them
    between = src[i_bad:i_sum]
    assert "return 1" not in between, (
        "harness.py returns 1 before printing the skip/denominator line, so a run with a "
        "failing invariant reports the FAIL list and never states what did not run -- the "
        "reader most in need of that line is the one who cannot see it. Move the return "
        "below the summary prints.")
    # and the summary must still be emitted before any return at all: the FAIL must not be
    # the last thing on stdout
    i_auth = src.find("authority: {len(EVIDENCE)")
    assert i_sum < i_auth, "the skip/denominator line must print before the authority line"
    print("selftest OK: ordering case -- the denominator line is not behind the FAIL return")

    # THE DEADLINE IS ASSERTED TOO, because this file's own failure in the #567 ci-selftests
    # job was not an assertion at all: no_hardcoded_cache_path scans every .py/.sh in the tree,
    # takes 2.2s by hand, and exceeded the 5s default on a 2-core runner -- strike 1 in CI's
    # explicit `harness check` step, strike 2 inside THIS file's selftest, so the step FAILed
    # with "a second consecutive timeout". main's CI never ran that driver, so the second
    # strike was unreachable there and the missing budget went unseen.
    #
    # WHAT IS ASSERTED IS THE PROPERTY, NOT THE CONSTANT: the check must not sit on the
    # default deadline. That reds both ways it can regress -- the entry deleted, or pulled
    # back down to the default -- without restating the number itself.
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import harness as _H  # noqa: E402
    _t = _H._CHECK_TIMEOUTS.get("no_hardcoded_cache_path")
    assert _t is not None, (
        "no_hardcoded_cache_path is back on the default _CHECK_TIMEOUT: it is a repo-scope "
        "AST scan of every .py/.sh and exceeds 5s on a 2-core runner, where it times out to "
        "a FAIL by strike 2 rather than by any assertion failing")
    assert _t > _H._CHECK_TIMEOUT, (
        f"no_hardcoded_cache_path's budget ({_t}) is not above the default "
        f"({_H._CHECK_TIMEOUT}), so the entry no longer buys it anything")
    print(f"selftest OK: no_hardcoded_cache_path budget {_t}s > default {_H._CHECK_TIMEOUT}s")

    print(f"selftest OK ({n_skip} skipped, summary states both count and denominator)")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv else (print(__doc__) or 0))
