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
    # EVERY index is located and asserted non-negative BEFORE any comparison. The first
    # version of this case found only i_bad and i_sum and then sliced `src[i_bad:i_sum]`:
    # when i_sum < i_bad that slice is EMPTY and `"return 1" not in ""` is True, so the
    # assertion passed vacuously. de measured it (2026-09-19) on four worlds -- swapping the
    # two prints (A) and, the pure form of the original defect, putting `if bad: return 1`
    # ABOVE both prints (C) -- and the old assertion missed both. An ordering assertion that
    # only looks at one gap cannot see an early return placed before the gap.
    # An ANCHORED WINDOW WAS TRIED TWICE AND ESCAPED TWICE. The first version anchored on the
    # FAIL print and sliced `src[i_bad:i_sum]`; when `i_sum < i_bad` that slice is EMPTY and
    # `"return 1" not in ""` is True, so it passed vacuously (de, four worlds). The second
    # anchored on the summary banner comment and de stepped over it by one line (a return
    # immediately ABOVE a comment that starts the window is outside it) -- on that tree the
    # selftest passed while a real `harness check` with a failing invariant printed NOTHING.
    # No anchor is used below; see the early-return assertion.
    _anchor = "stages(res)"
    assert src.count(_anchor) == 1, (
        f"{_anchor!r} appears {src.count(_anchor)} times in harness.py, so the window has no "
        f"single start and this case cannot check anything")
    i_stage = src.find(_anchor)
    i_fail = src.find('if bad:\n        print(f"\\n{len(bad)} invariant(s) FAILED')
    i_sum = src.find('NOT run here: {\', \'.join(skipped)}')
    i_auth = src.find("authority: {len(EVIDENCE)")
    for _name, _i in (("the summary block's anchor", i_stage),
                      ("the invariant-FAIL print", i_fail),
                      ("the skip/denominator print", i_sum),
                      ("the authority line", i_auth)):
        assert _i != -1, (
            f"cannot locate {_name} in harness.py, so this ordering case cannot check "
            f"anything -- a rename must red here, not pass")
    # NO `if bad:` BLOCK IN main() RETURNS BEFORE THE SUMMARY -- ASSERTED WITHOUT AN ANCHOR.
    # Two earlier versions anchored a window on a fixed position inside the block (first the
    # FAIL print, then the banner comment) and de escaped each by moving the return one line
    # further up: the banner is the block's first COMMENT, so a return immediately above it
    # sits outside a window that starts at it. On that tree the selftest passed while a real
    # `harness check` with a failing invariant printed NOTHING -- no FAIL list, no denominator,
    # no authority line. Any fixed anchor can be stepped over by one line.
    #
    # THE PREDICATE IS THE DEFECT ITSELF, SCANNED AS A BLOCK. It is not "any return before the
    # summary" (main holds 22 legitimate `return cmd_*(...)` dispatches there, so that reds on
    # the correct tree) and not a text shape: the pre-fix code was
    # `if bad: / print(...) / return 1`, three lines, so a regex for `if bad:` immediately
    # followed by `return` finds the mutation but MISSES the real thing it exists to prevent.
    # What is asserted is the property: an `if bad:` guard in main() whose body returns. Its
    # block is walked by indentation, so the return may sit anywhere inside the guard.
    # MEASURED: 0 on the correct tree, 1 on the pre-fix revision (a27c50b0^), and 1 in each of
    # de's worlds -- return above the FAIL print / above the banner / above the block's stage
    # call / between the banner and the FAIL print.
    i_main = src.rfind("\ndef main(", 0, i_auth)
    assert i_main != -1, "cannot locate `def main(` above the authority line in harness.py"
    early = []
    for _m in re.finditer(r"^([ \t]*)if bad:[^\n]*\n", src[i_main:i_auth], re.M):
        _ind = len(_m.group(1))
        for _line in src[i_main + _m.end():i_auth].split("\n"):
            if not _line.strip():
                continue
            if len(_line) - len(_line.lstrip()) <= _ind:
                break
            if re.search(r"\breturn\b", _line):
                early.append(_m.group(0).strip() + " -> " + _line.strip())
                break
    assert not early, (
        f"main() has an `if bad:` guard that returns before its summary ({early}), so a run "
        f"with a failing invariant reports the FAIL list -- or nothing at all -- and never "
        f"states what did not run. The reader most in need of that line is the one who cannot "
        f"see it. Move the return below the summary prints, or collapse it into the single "
        f"`return 1 if bad else 0` that follows them.")
    assert i_stage < i_fail < i_sum < i_auth, (
        f"harness.py's summary prints are out of order or outside the block "
        f"(anchor={i_stage}, FAIL={i_fail}, skip={i_sum}, authority={i_auth}): the "
        f"reader-facing summary must follow the FAIL line and precede the authority line")
    assert "\n        return" not in src[i_stage:i_auth], (
        "harness.py returns between the invariant-FAIL line and the authority line, so a run "
        "with a failing invariant reports the FAIL list and never states what did not run -- "
        "the reader most in need of that line is the one who cannot see it. Move the return "
        "below the summary prints.")
    print("selftest OK: ordering case -- no return between the FAIL line and the authority line")

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
