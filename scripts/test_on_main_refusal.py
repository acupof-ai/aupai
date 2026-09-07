#!/usr/bin/env python3
"""_refuse_committing_on_main's five worlds, including the unborn HEAD that kills the swap mutant.

    python3 scripts/test_on_main_refusal.py --selftest

# restartable: builds temp git repos and runs the real hook in each. Costs ~7s.

2026-09-08 05:40:09Z, b0 committed 484a9528 onto refs/heads/main from aupai-b0, which had main
checked out. At 05:45:45 they ran `branch: Reset to origin/main` and main moved off that commit --
a sideways move, which main_advances_by_ancestry caught and which then refused every commit in the
repository until PR #54 recorded the pair by hand. The commit survived only because b0 created
refs/heads/b0 from HEAD in the same second (both reflog entries read 05:45:45): a recovered loss,
not a near miss.

THE PREDICATE IS "HEAD IS main", NOT "main != origin/main", and this test is what settles it.
4c proposed the latter. It would not have fired: at 05:40:09 local main was 7049750c and
origin/main had been pushed to 7049750c at 05:39:05, so the two were EQUAL and the divergence was
created BY the commit the hook runs before. M2 below is that proposal, and it goes red on the
world it was designed for.

THE HOOK IS INSTALLED LAST IN EVERY WORLD. An earlier version of this file installed it during
setup, so the merge world's own `git commit -am "main side"` on main was refused, the conflicted
state was never built, and the merge exemption read as broken when it was not. Setup must not pass
through the gate under test.

THE WORLDS, and what each one alone would let through:

  W1  HEAD is main, repo has commits   -> refuse.  The incident. Without it the gate can be a
                                         no-op and every other world still passes.
  W2  HEAD is main, UNBORN (no commit) -> refuse.  The repo's first commit, made on main. This is
                                         the ONLY world separating `symbolic-ref --short -q HEAD`
                                         from `rev-parse --abbrev-ref HEAD`: abbrev-ref exits 128
                                         with an unborn HEAD and prints "HEAD", so a gate built on
                                         it returns early and PERMITS. Measured -- the swap mutant
                                         survived all four other worlds. Detached is NOT that
                                         world: both spellings refuse there identically, because
                                         "HEAD" != "main".
  W3  HEAD is a branch                 -> allow.  Every session's own worktree. A gate matching
                                         any branch name WARNs on all ten of them.
  W4  HEAD detached                    -> allow.  The integration tree's normal state.
  W5  finishing a CONFLICTED merge     -> allow.  A session resolving a conflict on a branch that
      while main is checked out           legitimately tracks main must be able to finish; the
                                         merge exemption is the only thing permitting it.

The hook is executed as git executes it, never reimplemented: a reimplementation shares the
original's assumptions and its agreement is not evidence (gate_failure_shapes §231).
"""

import os
import subprocess
import sys
import tempfile

HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hooks", "pre-commit")
MARK = "HEAD is the branch `main`"


def _write(path, text, mode="w"):
    with open(path, mode) as f:
        f.write(text)


def _read(path):
    with open(path) as f:
        return f.read()


def run(args, cwd):
    """Run git with the override flags stripped -- an inherited AUPAI_BEHIND_MAIN_OK or
    AUPAI_CONTROLLER from the caller's shell would make some worlds pass for the wrong reason."""
    e = dict(os.environ)
    e.pop("AUPAI_BEHIND_MAIN_OK", None)
    e.pop("AUPAI_CONTROLLER", None)
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, env=e)


def _init(prefix):
    d = tempfile.mkdtemp(prefix=prefix)
    run(["git", "init", "-q", "-b", "main"], d)
    run(["git", "config", "user.email", "t@t"], d)
    run(["git", "config", "user.name", "t"], d)
    return d


def _with_commit(prefix):
    d = _init(prefix)
    _write(os.path.join(d, "f.txt"), "one\n")
    run(["git", "add", "f.txt"], d)
    run(["git", "commit", "-q", "-m", "base"], d)
    return d


def install(d, src):
    hd = os.path.join(d, ".git", "hooks")
    os.makedirs(hd, exist_ok=True)
    dst = os.path.join(hd, "pre-commit")
    _write(dst, src)
    os.chmod(dst, 0o755)


def _stage(d, text):
    _write(os.path.join(d, "f.txt"), text + "\n", "a")
    run(["git", "add", "f.txt"], d)


def w1_on_main(src):
    d = _with_commit("de_onmain_w1_")
    install(d, src)
    _stage(d, "x")
    return run(["git", "commit", "-m", "on main"], d)


def w2_unborn_on_main(src):
    d = _init("de_onmain_w2_")
    _write(os.path.join(d, "f.txt"), "first\n")
    run(["git", "add", "f.txt"], d)
    install(d, src)
    return run(["git", "commit", "-m", "first commit on main"], d)


def w3_on_branch(src):
    d = _with_commit("de_onmain_w3_")
    run(["git", "checkout", "-q", "-b", "sidebranch"], d)
    install(d, src)
    _stage(d, "x")
    return run(["git", "commit", "-m", "on branch"], d)


def w4_detached(src):
    d = _with_commit("de_onmain_w4_")
    run(["git", "checkout", "-q", "--detach"], d)
    install(d, src)
    _stage(d, "x")
    return run(["git", "commit", "-m", "detached"], d)


def w5_merging_on_main(src):
    d = _with_commit("de_onmain_w5_")
    run(["git", "checkout", "-q", "-b", "other"], d)
    _write(os.path.join(d, "f.txt"), "other\n")
    run(["git", "commit", "-q", "-am", "other side"], d)
    run(["git", "checkout", "-q", "main"], d)
    _write(os.path.join(d, "f.txt"), "mainside\n")
    run(["git", "commit", "-q", "-am", "main side"], d)
    r = run(["git", "merge", "other"], d)
    assert "CONFLICT" in (r.stdout + r.stderr), (
        f"W5 is not a conflicted merge, so it does not test the exemption: {r.stdout} {r.stderr}"
    )
    _write(os.path.join(d, "f.txt"), "resolved\n")
    run(["git", "add", "f.txt"], d)
    install(d, src)
    return run(["git", "commit", "--no-edit"], d)


WORLDS = [
    ("W1 HEAD is main", w1_on_main, True),
    ("W2 HEAD is main, unborn", w2_unborn_on_main, True),
    ("W3 HEAD is a branch", w3_on_branch, False),
    ("W4 HEAD detached", w4_detached, False),
    ("W5 merging on main", w5_merging_on_main, False),
]


def _probe(src):
    """Returns the list of world names whose outcome is wrong.

    Checks the REFUSAL STRING, not just the exit code: the hook runs a dozen other gates, so a
    non-zero rc collapses "this gate fired" with "some other gate fired" and with "the hook died
    before reaching it" (gate_failure_shapes -- an exit code is not a diagnosis)."""
    bad = []
    for name, build, want_refused in WORLDS:
        r = build(src)
        refused = MARK in (r.stderr or "")
        if refused != want_refused:
            bad.append(f"{name}: refused={refused} (want {want_refused}) rc={r.returncode}")
        elif refused and r.returncode == 0:
            bad.append(f"{name}: printed the refusal but exited 0 -- the commit was made anyway")
    return bad


def _demo():
    src = _read(HOOK)
    assert "_refuse_committing_on_main" in src, (
        f"{HOOK} does not define the gate under test -- it was renamed or removed"
    )

    bad = _probe(src)
    assert not bad, "the real hook fails its own worlds: " + "; ".join(bad)
    print(
        f"  on-main refusal: {len(WORLDS)} worlds correct "
        f"(refuses on main incl. unborn; allows branch, detached, merging)"
    )

    # Each mutant must go red, and the world set it kills is the evidence the worlds are not
    # redundant. Mutating the SOURCE and re-running the real git hook, never a copy of the logic.
    mutants = {
        "swap symbolic-ref for rev-parse --abbrev-ref": src.replace(
            '["git", "symbolic-ref", "--short", "-q", "HEAD"]', '["git", "rev-parse", "--abbrev-ref", "HEAD"]'
        ),
        "refuse only when main != origin/main (4c's proposal)": src.replace(
            'if r.returncode != 0 or r.stdout.strip() != "main":\n        return',
            'if r.returncode != 0 or r.stdout.strip() != "main":\n        return\n'
            '    _o = subprocess.run(["git", "rev-parse", "-q", "--verify",\n'
            '                         "refs/remotes/origin/main"],\n'
            "                        capture_output=True, text=True)\n"
            '    _l = subprocess.run(["git", "rev-parse", "refs/heads/main"],\n'
            "                        capture_output=True, text=True)\n"
            "    if _o.returncode != 0 or _o.stdout.strip() == _l.stdout.strip():\n"
            "        return",
        ),
        "never call the gate": src.replace("        _refuse_committing_on_main(merging)\n", ""),
        "drop the merge exemption": src.replace("    if merging:\n        return\n", "", 1),
    }
    for label, msrc in mutants.items():
        assert msrc != src, f"mutant {label!r} did not apply -- the source it edits moved"
        killed = _probe(msrc)
        assert killed, (
            f"MUTANT SURVIVED: {label} -- every world still reads correct, so the "
            f"worlds do not test what this gate claims to do"
        )
        print(f"  mutant red: {label} -> {', '.join(k.split(':')[0] for k in killed)}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
        print("test_on_main_refusal: ok")
    else:
        print(__doc__)
        print("run with --selftest")
