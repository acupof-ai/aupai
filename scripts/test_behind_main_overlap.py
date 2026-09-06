#!/usr/bin/env python3
"""_main_touched_staged's seven worlds, including the committed-and-behind carry (tilerl-31).

    python3 scripts/test_behind_main_overlap.py --selftest

# restartable: builds temp git repos. Costs ~3s.

06ed3b2a replaced "N commits behind" with the intersection of the staged paths and what main
changed since the branch diverged, and shipped with NO test of the predicate: it appears once in
scripts/hooks/pre-commit and nowhere else in the tree. The world tilerl asked for is W3 -- the
file is COMMITTED on a branch behind main and main has also moved it -- because that is the case
the count predicate and the overlap predicate answer differently, and the one an operator meets.

The predicate is imported from the hook file, never reimplemented: a reimplementation shares the
original's assumptions and its agreement is not evidence (gate_failure_shapes §231). The hook has
no .py extension and calls sys.exit() at import, so it is loaded via importlib with a source
loader and a __name__ that is not "__main__".

THE WORLDS, and what each one alone would let through:

  W1  main moved on OUR staged path        -> the path, refuse.  Without it the predicate could
                                              return [] always and pass everything else.
  W2  main moved on a DIFFERENT path       -> [], allow.  This is 06ed3b2a's whole point: the old
                                              count refuses here and the overlap must not.
  W3  our path is COMMITTED and behind,    -> the path, refuse.  Committed-vs-staged is not the
      main moved it too                       question the predicate asks; a version keying on
                                              dirtiness would answer [] here.
  W3c same, main moved the OTHER file      -> [], allow.  Without it a count predicate passes W3
                                              for the wrong reason: it refuses on behind-ness.
  W4  no main at all                       -> [], allow.  A question with no answer refuses
                                              nothing; the intersection must not throw.
  W5  no staged paths                      -> [], allow.  The short-circuit before any git call.
  W6  we COMMITTED an edit and stage a     -> [], allow.  Pins the three-dot base: `git diff HEAD
      second one; main moved another file     main` lists our own commit too and would refuse a
                                              commit because of its own edits.

W2 IS THE LOAD-BEARING NEGATIVE. A predicate that returns every staged path passes W1 and W3 and
is exactly the behaviour 06ed3b2a removed, so a fixture without W2 would certify the old bug.

ACCEPTANCE, measured 2026-09-07 -- three mutants, each killing a DIFFERENT world set:
  count-not-overlap (return every staged path)  -> W2, W3c, W6
  two-dot (HEAD..main instead of the base)      -> W3c, W6
  always-empty (never refuse)                   -> W1, W3
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(ROOT, "scripts", "hooks", "pre-commit")
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
ENV.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")


def _load_hook():
    """The hook module, imported for its predicate. Not exec'd as a program.

    It has no .py extension, so spec_from_file_location needs an explicit SourceFileLoader.
    The module name is deliberately not "__main__": the hook's `if __name__ == "__main__"` block
    runs the whole gate and exits.
    """
    loader = importlib.machinery.SourceFileLoader("_hook_under_test", HOOK)
    spec = importlib.util.spec_from_file_location("_hook_under_test", HOOK, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(d, *a, **kw):
    return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True, env=ENV, **kw)


def _repo():
    d = tempfile.mkdtemp(prefix="behind_overlap_")
    _git(d, "init", "-q", "-b", "main", ".")
    _git(d, "config", "user.email", "t@example.invalid")
    _git(d, "config", "user.name", "t")
    return d


def _write(d, rel, text):
    p = os.path.join(d, rel)
    os.makedirs(os.path.dirname(p) or d, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def _base(d):
    """main holding two files, then a branch `work` diverging from it."""
    _write(d, "ours.py", "base\n")
    _write(d, "theirs.py", "base\n")
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "base")
    _git(d, "checkout", "-q", "-b", "work")


def _main_moves(d, rel):
    """main advances on `rel` while `work` stays behind."""
    head = _git(d, "rev-parse", "HEAD").stdout.strip()
    _git(d, "checkout", "-q", "main")
    _write(d, rel, "main's version\n")
    _git(d, "add", rel)
    _git(d, "commit", "-qm", f"main edits {rel}")
    _git(d, "checkout", "-q", "work")
    assert _git(d, "rev-parse", "HEAD").stdout.strip() == head, "work must not have moved"


def _in(d, fn, paths):
    """Run the predicate with cwd inside the world: it calls git with no -C."""
    cwd = os.getcwd()
    try:
        os.chdir(d)
        return fn(paths)
    finally:
        os.chdir(cwd)


def selftest():
    mod = _load_hook()
    fn = mod._main_touched_staged
    fails = []

    def case(label, d, paths, want):
        got = _in(d, fn, paths)
        if got == want:
            print(f"  ok   {label} -> {got}")
        else:
            print(f"  FAIL {label}: want {want}, got {got}", file=sys.stderr)
            fails.append(label)

    worlds = []
    try:
        # W1: main moved the very path we stage.
        d = _repo()
        worlds.append(d)
        _base(d)
        _main_moves(d, "ours.py")
        _write(d, "ours.py", "my staged edit\n")
        _git(d, "add", "ours.py")
        case("W1 main moved our staged path", d, ["ours.py"], ["ours.py"])

        # W2: main moved a DIFFERENT path. The count refuses here; the overlap must not.
        d = _repo()
        worlds.append(d)
        _base(d)
        _main_moves(d, "theirs.py")
        _write(d, "ours.py", "my staged edit\n")
        _git(d, "add", "ours.py")
        behind = _git(d, "rev-list", "--count", "HEAD..main").stdout.strip()
        assert behind == "1", f"W2's premise is stale: work must be behind main, got {behind}"
        case("W2 main moved a different path (behind=1)", d, ["ours.py"], [])

        # W3, THE ONE tilerl ASKED FOR: our change is COMMITTED on a branch behind main, and main
        # moved the same file. `git diff` sees nothing dirty; the predicate must still refuse,
        # because what it answers is "did main move this path", not "is this path dirty".
        d = _repo()
        worlds.append(d)
        _base(d)
        _write(d, "ours.py", "my committed edit\n")
        _git(d, "add", "ours.py")
        _git(d, "commit", "-qm", "work commits ours.py")
        _main_moves(d, "ours.py")
        assert not _git(d, "status", "--porcelain").stdout.strip(), "W3's tree must be clean"
        behind = _git(d, "rev-list", "--count", "HEAD..main").stdout.strip()
        assert behind == "1", f"W3's premise is stale: work must be behind main, got {behind}"
        case("W3 committed and behind, main moved it too", d, ["ours.py"], ["ours.py"])

        # W3-CONTROL: the same committed-and-behind shape where main moved the OTHER file. Without
        # it, a predicate that refuses whenever HEAD is behind main would pass W3 for the wrong
        # reason -- the count predicate 06ed3b2a removed does exactly that.
        d = _repo()
        worlds.append(d)
        _base(d)
        _write(d, "ours.py", "my committed edit\n")
        _git(d, "add", "ours.py")
        _git(d, "commit", "-qm", "work commits ours.py")
        _main_moves(d, "theirs.py")
        case("W3-control committed and behind, main moved another path", d, ["ours.py"], [])

        # W4: no main. merge-base has no answer, and no answer refuses nothing.
        d = _repo()
        worlds.append(d)
        _write(d, "ours.py", "base\n")
        _git(d, "add", "-A")
        _git(d, "commit", "-qm", "base")
        _git(d, "branch", "-m", "main", "solo")
        assert not _git(d, "rev-parse", "--verify", "main").stdout.strip(), "W4 must have no main"
        case("W4 no main at all", d, ["ours.py"], [])

        # W5: empty input short-circuits before any git call.
        case("W5 no staged paths", worlds[0], [], [])

        # THREE-DOT, NOT TWO. `git diff HEAD main` would also list what THIS branch changed,
        # refusing a commit because of its own edits. World: main moved theirs.py, work moved
        # ours.py and COMMITTED it, and we stage a second edit to ours.py. Two-dot lists both
        # files; the merge-base three-dot lists only theirs.py, so the answer must be [].
        d = _repo()
        worlds.append(d)
        _base(d)
        _write(d, "ours.py", "work's first edit\n")
        _git(d, "add", "ours.py")
        _git(d, "commit", "-qm", "work edits ours.py")
        _main_moves(d, "theirs.py")
        _write(d, "ours.py", "work's second edit\n")
        _git(d, "add", "ours.py")
        two_dot = set(_git(d, "diff", "--name-only", "HEAD", "main").stdout.split())
        assert two_dot == {"ours.py", "theirs.py"}, (
            f"W6's premise is stale: two-dot must list both files, got {sorted(two_dot)}"
        )
        case("W6 two-dot would list our own commit; three-dot must not", d, ["ours.py"], [])
    finally:
        for d in worlds:
            shutil.rmtree(d, ignore_errors=True)

    if fails:
        print(f"test_behind_main_overlap: {len(fails)} failure(s): {', '.join(fails)}", file=sys.stderr)
        return 1
    print(
        "test_behind_main_overlap OK: the predicate answers 'did main move this path', not "
        "'is this path dirty' and not 'is HEAD behind' -- W1/W3 refuse, W2/W3-control/W4/W5/W6 "
        "allow, and W6 pins the three-dot base."
    )
    return 0


if __name__ == "__main__":
    if "--selftest" not in sys.argv:
        print(__doc__.strip().split("\n")[0])
        print(f"usage: python3 {os.path.relpath(__file__, ROOT)} --selftest")
        sys.exit(2)
    sys.exit(selftest())
