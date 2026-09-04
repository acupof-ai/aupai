"""Does each check FAIL on an UNMUTATED world?

The rule the harness selftest enforces is "broken() must produce FAIL". It does not
enforce that the FAIL is caused by the mutation. AGENTS.md records the consequence
already: three worlds built on the bare `_tmp_repo()` were green because the check FAILs
on any tree where the paths it reads are absent, mutation or not.

This measures the confound directly. For every check it runs run(root) against two
UNMUTATED worlds:

  bare    `_tmp_repo()`         -- empty tree with data/corpus + runs/
  shaped  `_tmp_repo_shaped()`  -- symlinks the real code/docs/data/runs

A check that returns FAIL on the unmutated world matching its own broken()'s base is a
check whose selftest green is not evidence about its mutation. That is the finding; the
check itself may still be correct in the real tree.

  python3 runs/audit_0904/unmutated_fail.py
  python3 runs/audit_0904/unmutated_fail.py --selftest
"""

import os
import sys

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness as H  # noqa: E402
from scan_broken_worlds import scan  # noqa: E402


def run_all():
    bare = H._tmp_repo()
    shaped = H._tmp_repo_shaped()
    base = {name: tags for name, _, _, tags in scan()}
    out = []
    for name, _asserts, _incident, fn, _broken in H.CHECKS:
        tags = base.get(name, set())
        which = "shaped" if "shaped" in tags else ("bare" if "bare" in tags else "?")
        res = {}
        for label, root in (("bare", bare), ("shaped", shaped)):
            try:
                st, ev = fn(root)
            except H.SelftestSkip as e:
                st, ev = "SELFTEST-SKIP", str(e)
            except Exception as e:
                st, ev = "RAISE", f"{type(e).__name__}: {e}"
            res[label] = (st, str(ev)[:120])
        out.append((name, which, res))
    return out


def _selftest():
    rows = run_all()
    assert len(rows) > 60, f"only {len(rows)} checks run"
    states = {st for _, _, r in rows for st, _ in r.values()}
    # If every check reported the same thing on both worlds, the instrument is inert.
    assert len(states) >= 2, f"every check returned {states} -- this scan cannot distinguish"
    # Known answer, read by hand from harness.py:9107-9112: no_shared_stash reads
    # `root`/.git, so a bare world (no .git) must SKIP and a shaped world (`git init`,
    # empty stack) must PASS. A scan that reports anything else is not calling the check
    # with the world it thinks it is.
    got = dict((n, r) for n, _, r in rows)
    b, s = got["no_shared_stash"]["bare"][0], got["no_shared_stash"]["shaped"][0]
    assert (b, s) == ("SKIP", "PASS"), (
        f"no_shared_stash read ({b}, {s}) on (bare, shaped); harness.py:9107 SKIPs without "
        ".git and _tmp_repo_shaped git-inits, so ('SKIP', 'PASS') is the only correct pair"
    )
    # And the negative half: a check that reads a real doc must NOT pass on the bare world.
    b2 = got["docs_root_clean"]["bare"][0]
    assert b2 != "PASS", f"docs_root_clean PASSed a tree with no docs/ ({b2} expected non-PASS)"
    print(f"unmutated_fail selftest ok ({len(rows)} checks, states {sorted(states)})")


if __name__ == "__main__":
    st = "--selftest" in sys.argv
    rows = run_all()
    if st:
        _selftest()
        raise SystemExit(0)
    n_bad = 0
    print(f"{'check':40s} {'base':7s} {'on bare':16s} {'on shaped':16s}")
    for name, which, res in sorted(rows):
        b, s = res["bare"][0], res["shaped"][0]
        own = res.get(which, ("-", ""))[0]
        flag = "  <-- FAILs unmutated on its own base" if own == "FAIL" else ""
        if flag:
            n_bad += 1
        print(f"{name:40s} {which:7s} {b:16s} {s:16s}{flag}")
    print(f"\n{n_bad} of {len(rows)} checks FAIL on an unmutated world of their own base type")
    for name, which, res in sorted(rows):
        if res.get(which, ("-", ""))[0] == "FAIL":
            print(f"  {name} ({which}): {res[which][1]}")
