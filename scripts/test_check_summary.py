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

    print(f"selftest OK ({n_skip} skipped, summary states both count and denominator)")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv else (print(__doc__) or 0))
