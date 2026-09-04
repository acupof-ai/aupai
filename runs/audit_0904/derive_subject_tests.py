#!/usr/bin/env python3
"""The derived subject->test map, restricted to tests the pre-commit hook can actually run.

    python3 runs/audit_0904/derive_subject_tests.py            # subjects by test count
    python3 runs/audit_0904/derive_subject_tests.py --pairs    # subject<TAB>tests
    python3 runs/audit_0904/derive_subject_tests.py --runnable # only SELFTEST_FILES|PARTIAL

WHY. The hook's TESTS_FOR_SUBJECT is three hand-written entries, and 6e's item is to derive it
from what each test imports or opens. The raw derivation is NOT the map: it says train.py has 29
tests, which would put 29 selftests on every train.py commit. This prints both so the gap between
"reads it" and "worth running" is a measured number rather than a judgement.

Two sources, both syntactic: imports (`import X`, `from X.Y import Z`) and string literals naming
a tracked repo file (a test that opens or execs its subject). A test that merely mentions a name in
a comment is not counted -- ast, not grep.
"""
import ast
import os
import re
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOOK = os.path.join(ROOT, "scripts", "hooks", "pre-commit")
tracked = set(subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True,
                             text=True).stdout.split())
py = {t for t in tracked if t.endswith(".py")}
by_mod = defaultdict(set)
for t in py:
    by_mod[os.path.basename(t)[:-3]].add(t)


def _read(rel):
    try:
        return open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def hook_runnable():
    """The set the hook can invoke here: SELFTEST_FILES plus PARTIAL's keys.

    Parsed out of the hook source with ast, from the assignments themselves -- a regex over the
    file would also match the names in its comment paragraphs, which discuss both sets at length.
    """
    src = _read("scripts/hooks/pre-commit")
    out = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if not names & {"SELFTEST_FILES", "PARTIAL"}:
            continue
        if isinstance(node.value, ast.Set):
            out |= {e.value for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        elif isinstance(node.value, ast.Dict):
            out |= {k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return out


tests = sorted(t for t in py
               if os.path.basename(t).startswith("test_")
               or "--selftest" in _read(t)
               or "selftest" in os.path.basename(t))
# THE LOOSE SET IS NOT A SET OF TESTS. "--selftest" appears in most tools in this repo, so the
# above also calls scripts/profile_step_cost.py, eval/cache_guard.py and scripts/card_claim.py
# tests of train.py -- they import it. Measured 2026-09-04: 167 files match, and three of the top
# twelve subject->test pairs by usage depth were tools rather than tests. A hook that ran those on
# a train.py commit would be running the pipeline, not a check. `test_*.py` is the honest set for
# the pairing question; the loose set stays available because it answers a different one (who
# reads this file at all).
strict_tests = [t for t in tests if os.path.basename(t).startswith("test_")]

subj_to_tests = defaultdict(set)
for t in (strict_tests if "--strict" in sys.argv else tests):
    try:
        tree = ast.parse(_read(t))
    except SyntaxError:
        continue
    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                hits |= by_mod.get(al.name.split(".")[0], set())
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            hits |= by_mod.get(parts[0], set())
            if "/".join(parts) + ".py" in py:
                hits.add("/".join(parts) + ".py")
            for al in node.names:
                if "/".join(parts + [al.name]) + ".py" in py:
                    hits.add("/".join(parts + [al.name]) + ".py")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = re.sub(r"^\./", "", node.value)
            if v.endswith(".py") and "/" in v and v in py:
                hits.add(v)
    hits.discard(t)
    for h in hits:
        subj_to_tests[h].add(t)

runnable = hook_runnable()


def _depth(test, subj):
    """How hard `test` leans on `subj`: (distinct names taken from it, total uses of those names).

    AN IMPORT EDGE IS NOT EVIDENCE THE TEST EXERCISES THE SUBJECT, and that is the whole reason
    this function exists. 23 hook-runnable tests import train.py; nearly all of them want `Cfg` and
    nothing else, because Cfg is where every shape constant lives. Running 23 selftests on a
    train.py commit would cost minutes and most of them could not go red for the edit.

    So: count the names actually taken from the subject and how often they are USED. A test that
    binds one name and touches it once has a dependency; one that binds several and touches them
    repeatedly is testing the thing. The threshold is a judgement, so this prints the numbers and
    does not pick one.
    """
    try:
        tree = ast.parse(_read(test))
    except SyntaxError:
        return 0, 0
    mod = os.path.basename(subj)[:-3]
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and \
                node.module.split(".")[-1] == mod:
            names |= {al.asname or al.name for al in node.names}
        elif isinstance(node, ast.Import):
            for al in node.names:
                if al.name.split(".")[-1] == mod:
                    names.add(al.asname or al.name.split(".")[-1])
    if not names:
        return 0, 0
    uses = sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Name) and n.id in names
               and not isinstance(getattr(n, "ctx", None), ast.Store))
    return len(names), uses


if "--depth" in sys.argv:
    # subject, test, distinct names, uses -- the evidence for whether a pair is worth running
    for s in sorted(subj_to_tests):
        for t in sorted(subj_to_tests[s] & runnable):
            k, u = _depth(t, s)
            print(f"{s}\t{t}\t{k}\t{u}")
elif "--runnable" in sys.argv or "--pairs" in sys.argv:
    only = runnable if "--runnable" in sys.argv else None
    for s in sorted(subj_to_tests):
        ts = sorted(subj_to_tests[s] & only) if only is not None else sorted(subj_to_tests[s])
        if ts:
            print(f"{s}\t{','.join(ts)}")
else:
    n_r = {s: (v & runnable) for s, v in subj_to_tests.items()}
    n_r = {s: v for s, v in n_r.items() if v}
    print(f"{len(tests)} test-carrying file(s); {len(runnable)} the hook can run here")
    print(f"subjects with >=1 test: {len(subj_to_tests)} derived, {len(n_r)} after restricting "
          f"to what the hook can run")
    print("\nderived / runnable, by subject (top 14):")
    for s, ts in sorted(subj_to_tests.items(), key=lambda kv: -len(kv[1]))[:14]:
        print(f"  {len(ts):3d} / {len(ts & runnable):2d}  {s}")
