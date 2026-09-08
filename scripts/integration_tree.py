#!/usr/bin/env python3
"""Is this working tree the integration tree -- the one every other worktree branches from?

ONE IMPLEMENTATION, IMPORTED BY BOTH SIDES. harness.py's ledger writers and
scripts/hooks/pre-commit both need this question answered, and two spellings of a similar idea
is how FRICTION_KINDS came to reject a kind this repo's own merge_main.sh emits. 4c's ruling
2026-09-05: one importable module, and the hook calls it rather than restating it.

WHY NOT THE BRANCH. `git symbolic-ref --short HEAD == "main"` was the first predicate and it
was correct for one day. The integration tree is being detached deliberately (tilerl's flip,
2026-09-05), which makes symbolic-ref answer "" exactly where the guard is needed, and the
same hole was already open at pre-commit:378. A branch is a label someone can change; being
the tree other worktrees hang off is structural.

WHY NOT A PATH. /Users/bytedance/code/aupai is one laptop's layout. It is wrong on the pod,
wrong in CI, and wrong for anyone who clones this repo somewhere else.

THREE CLAUSES, ALL FROM GIT, MEASURED IN SEVEN WORLDS (/tmp/de51_pred2.py, 2026-09-05):

  realpath(--git-dir) == realpath(--git-common-dir)   the main worktree. True in a normal
                                                      clone and in the tree worktrees hang
                                                      off; False in every linked worktree.
                                                      Branch-independent, so a detached
                                                      integration tree still reads True.
  <common>/worktrees exists and is non-empty          linked worktrees exist.
  otherwise                                           False.

THE SECOND CLAUSE IS NOT DECORATION. Without it a standalone clone is its own main worktree,
so a CI checkout and all 27 of harness.py's `git init` fixtures read True and every write in
them refuses -- measured, world A: main=True linked=False. The integration tree's defining
property is that other worktrees branch from it, which is also exactly why a dirty ledger
there blocks everyone; a standalone checkout has no siblings and blocks nobody.

FAILS OPEN, DELIBERATELY. No repository, git absent, git erroring: False. Such a tree cannot
be the integration tree, and a guard whose broken state blocks the write is a guard someone
disables. The refusal has to survive being wrong in the safe direction.

Measured (2026-09-05), the seven worlds:

  standalone repo on main                       False   CI, and every fixture
  same repo once a worktree is added            True    it became an integration tree
  the linked worktree itself                    False   every session's normal write
  integration tree DETACHED, with siblings      True    tilerl's flip; symbolic-ref reads ""
  a subdirectory of either                      same    --path-format=absolute, not relative
  no repository                                 False
  git unavailable                               False

restartable: pure read, no state.
"""

import os
import subprocess


def _git(root, *args):
    try:
        return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                              timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None


def is_integration_tree(root="."):
    """True if `root` is the main worktree of a common git dir that has linked worktrees.

    Read the module docstring before changing this: each clause is there because dropping it
    was measured to break a real world.
    """
    gd = _git(root, "rev-parse", "--path-format=absolute", "--git-dir")
    gc = _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if gd is None or gc is None or gd.returncode != 0 or gc.returncode != 0:
        return False
    common = gc.stdout.strip()
    if not common or not gd.stdout.strip():
        return False
    common = os.path.realpath(common)
    if os.path.realpath(gd.stdout.strip()) != common:
        return False
    wt = os.path.join(common, "worktrees")
    try:
        return os.path.isdir(wt) and bool(os.listdir(wt))
    except OSError:
        return False


def selftest():
    """The seven worlds from the docstring, built rather than described.

    Each world is a real `git init` or `git worktree add`, because the question is about git's
    own directory layout and a hand-written fixture would share this module's assumptions about
    it -- the shape that left three harness checks dead while their selftest was green.
    """
    import shutil
    import tempfile

    fails = []
    tmp = tempfile.mkdtemp(prefix="itree_pred_")
    try:
        def g(d, *a):
            return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True,
                                  timeout=60)

        solo = os.path.join(tmp, "solo")
        os.makedirs(solo)
        g(solo, "init", "-q", ".")
        g(solo, "config", "user.email", "t@example.invalid")
        g(solo, "config", "user.name", "t")
        g(solo, "commit", "-q", "--allow-empty", "-m", "base")
        g(solo, "branch", "-M", "main")

        # W1: a standalone clone on main is NOT the integration tree. The clause that exists
        # only for this world; without it CI and every fixture refuse.
        if is_integration_tree(solo):
            fails.append("W1 a standalone repo on main read as the integration tree -- CI and "
                         "all 27 of harness.py's git-init fixtures are this shape")

        # W2: the same tree, once a worktree hangs off it.
        wt = os.path.join(tmp, "wt")
        g(solo, "worktree", "add", "-q", "-b", "sidebranch", wt)
        if not is_integration_tree(solo):
            fails.append("W2 a main worktree with linked worktrees did NOT read as the "
                         "integration tree -- this is the tree everyone merges through")

        # W3: the linked worktree is where every session writes.
        if is_integration_tree(wt):
            fails.append("W3 a linked worktree read as the integration tree -- that is every "
                         "session's normal write and all of them would refuse")

        # W4: DETACHED integration tree. The world tilerl's flip creates, and the reason the
        # branch predicate was replaced: symbolic-ref answers "" here.
        g(solo, "checkout", "-q", "--detach", "HEAD")
        if not is_integration_tree(solo):
            fails.append("W4 a DETACHED integration tree did not read as one -- this is the "
                         "world the flip creates and the whole reason the branch test was "
                         "dropped")
        sr = g(solo, "symbolic-ref", "--short", "HEAD")
        if sr.returncode == 0 and sr.stdout.strip() == "main":
            fails.append("W4 precondition: the tree is not actually detached, so this world "
                         "proves nothing about the flip")

        # W5/W6: a SUBDIRECTORY answers the same as its tree. --git-dir prints a relative path
        # without --path-format=absolute, so a naive compare disagrees with itself one level down.
        sub_main = os.path.join(solo, "sub")
        os.makedirs(sub_main, exist_ok=True)
        if is_integration_tree(sub_main) is not True:
            fails.append("W5 a subdirectory of the integration tree answered differently from "
                         "the tree -- --git-dir is relative unless --path-format=absolute")
        sub_wt = os.path.join(wt, "sub")
        os.makedirs(sub_wt, exist_ok=True)
        if is_integration_tree(sub_wt) is not False:
            fails.append("W6 a subdirectory of a linked worktree answered differently from the "
                         "worktree")

        # W7: no repository -> fail open.
        bare = os.path.join(tmp, "bare")
        os.makedirs(bare)
        if is_integration_tree(bare):
            fails.append("W7 a non-repository read as the integration tree -- the guard must "
                         "fail open where git cannot answer")

        # W8: git unavailable -> fail open. A different code path from W7: the subprocess
        # raises rather than returning 128.
        saved = os.environ.get("PATH", "")
        try:
            empty = os.path.join(tmp, "empty-bin")
            os.makedirs(empty, exist_ok=True)
            os.environ["PATH"] = empty
            if is_integration_tree(solo):
                fails.append("W8 refused with git unavailable -- an unanswerable predicate "
                             "must not block the write")
        finally:
            os.environ["PATH"] = saved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        print(f"\n{len(fails)} failure(s)")
        return 1
    print("  is_integration_tree: 8 worlds -- standalone no, +worktrees yes, linked no, "
          "DETACHED integration tree yes, subdirs agree, no-git and no-git-binary fail open")
    return 0


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(is_integration_tree(os.getcwd()))
