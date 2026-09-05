#!/usr/bin/env python3
"""The pre-commit hook must refuse a commit from a COPY of a linked worktree.

cf3dbaea "probe 2": b0's `cp -r` of a linked worktree kept the `.git` FILE, which is only a
`gitdir:` pointer into .git/worktrees/<name> in the real repository. The copy therefore shared the
original's HEAD, index and branch ref, so a commit made in /tmp/reg_moe landed on the real branch
and merged to main carrying a SwiGLU mutant.

THE WORLD IS A REAL `cp -r` OF A REAL WORKTREE. A hand-written .git file would share the check's
own assumption about what a copy looks like -- the failure mode AGENTS names for broken worlds. So
this builds a repo, adds a worktree with `git worktree add`, copies it with shutil.copytree
(symlinks=False, the same bytes `cp -r` produces), and commits in both.

FOUR WORLDS, and the negative controls are the load-bearing ones. A hook that refuses every
directory with a `.git` file would refuse the ORIGINAL worktree and every plain checkout -- which
is every commit anyone makes.

restartable: yes -- every world is a fresh temp dir, removed at the end.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(ROOT, "scripts", "hooks", "pre-commit")

CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
CLEAN_ENV["AUPAI_INTEGRATION_TREE"] = "/nonexistent-so-the-integration-rule-never-fires"


def g(cwd, *a):
    return subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=True, env=CLEAN_ENV)


def run_hook(cwd):
    """Only the copied-worktree gate, not the whole hook: the rest needs this repo's ledgers,
    manifests and 90-odd checks, none of which exist in a two-commit temp repo. Importing the
    function and calling it is the SUBJECT here; a full hook run would fail for twenty reasons
    that have nothing to do with the property under test."""
    code = (
        "import sys, os, importlib.machinery, importlib.util\n"
        f"loader = importlib.machinery.SourceFileLoader('pc', {HOOK!r})\n"
        "spec = importlib.util.spec_from_loader('pc', loader)\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "loader.exec_module(m)\n"
        "import subprocess\n"
        "top = subprocess.run(['git','rev-parse','--show-toplevel'],capture_output=True,text=True).stdout.strip()\n"
        "m._refuse_copied_worktree(top)\n"
        "print('PASSED_GATE')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=cwd, env=CLEAN_ENV)
    return r.returncode, r.stdout + r.stderr


def main():
    fails = []
    d = tempfile.mkdtemp(prefix="wtcopy_")
    try:
        # A real repository with a real linked worktree.
        repo = os.path.join(d, "repo")
        os.makedirs(repo)
        g(repo, "init", "-q")
        g(repo, "config", "user.email", "t@t.t")
        g(repo, "config", "user.name", "t")
        with open(os.path.join(repo, "train.py"), "w") as f:
            f.write("x = 1\n")
        g(repo, "add", "train.py")
        g(repo, "commit", "-q", "-m", "base", "--no-verify")

        wt = os.path.join(d, "wt_feature")
        r = g(repo, "worktree", "add", "-q", wt, "-b", "feature")
        if not os.path.isdir(wt):
            print(f"SKIP: git worktree add failed: {r.stderr.strip()[:200]}")
            return 0
        assert os.path.isfile(os.path.join(wt, ".git")), "a linked worktree must have a .git FILE"

        # THE COPY: same bytes cp -r produces, including the .git pointer file.
        copy = os.path.join(d, "wt_feature_copy")
        shutil.copytree(wt, copy, symlinks=False)
        assert os.path.isfile(os.path.join(copy, ".git")), "the copy must carry the .git pointer"

        # WORLD 1: commit from the COPY -> must refuse.
        rc, out = run_hook(copy)
        if rc == 0 or "REFUSING" not in out:
            fails.append(f"1: a commit from the copied worktree must refuse (rc={rc}, "
                         f"out={out.strip()[:200]})")
        elif os.path.realpath(wt) not in out:
            fails.append("1: the refusal must name the REGISTERED path so the operator knows "
                         f"where to go; it said: {out.strip()[:200]}")

        # WORLD 2, negative control: the ORIGINAL worktree must pass. Without this, world 1 would
        # also pass for a hook that refuses every linked worktree, i.e. every session's commits.
        rc, out = run_hook(wt)
        if rc != 0 or "PASSED_GATE" not in out:
            fails.append(f"2: the original worktree must pass (rc={rc}, out={out.strip()[:200]})")

        # WORLD 3, negative control: a plain checkout has no back-pointer and must pass.
        rc, out = run_hook(repo)
        if rc != 0 or "PASSED_GATE" not in out:
            fails.append(f"3: a plain checkout must pass (rc={rc}, out={out.strip()[:200]})")

        # WORLD 4: `git worktree move` rewrites the back-pointer, so the MOVED worktree must pass
        # at its new path. This separates "is this directory the registered one" from "is this
        # directory where it was created", and only the first is the property.
        moved = os.path.join(d, "wt_moved")
        r = g(repo, "worktree", "move", wt, moved)
        if r.returncode == 0 and os.path.isdir(moved):
            rc, out = run_hook(moved)
            if rc != 0 or "PASSED_GATE" not in out:
                fails.append(f"4: a worktree MOVED with `git worktree move` must pass at its new "
                             f"path -- the criterion is the registered path, not the original one "
                             f"(rc={rc}, out={out.strip()[:200]})")
        else:
            print(f"  note: `git worktree move` unavailable here ({r.stderr.strip()[:80]}); "
                  f"world 4 not run")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    if fails:
        print("test_worktree_copy_refused FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("test_worktree_copy_refused ok: a real cp -r of a real linked worktree is refused and "
          "the refusal names the registered path; the original worktree, a plain checkout, and a "
          "worktree moved with `git worktree move` all still pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
