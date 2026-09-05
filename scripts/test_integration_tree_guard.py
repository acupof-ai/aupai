#!/usr/bin/env python3
"""A ledger writer refuses in the integration tree, and nowhere else.

4c's ruling 2026-09-05, after two rows landed in the integration tree ten minutes apart: b0's
task row, then 44's board row. AGENTS.md already said to run these in your worktree, and the
rule's coverage row explained why prose was all it could be -- "the invoking directory is a
shell fact no artifact records". It is recoverable from the tree the writer is about to append
to, which is what this makes checkable.

THE PREDICATE IS NOT THE BRANCH, AND THIS FILE'S FIRST VERSION GOT THAT WRONG. It tested
`branch == "main"` and its worlds asserted branch semantics; hours later the integration tree
was detached on purpose (tilerl's flip, main 0425accb) and all three guards silently turned
OFF in the one tree they exist for. The predicate now lives in scripts/integration_tree.py --
main worktree of a common git dir that HAS linked worktrees -- and these worlds BUILD that
property with `git worktree add` instead of naming a branch.

tilerl found the same defect in the hook's own selftest from the other side: worlds 1-4 there
built a bare `git init`, which is not an integration tree by this definition, so world 1 was
asserting "a commit here is refused" against a repo that never qualified. It passed for years
because the branch test did not care. A world that does not hold the property proves nothing
about a guard that reads it.

WHAT THE REFUSAL PREVENTS is not the row -- the row is valid content -- it is the DIRTY LEDGER.
The integration tree's pre-commit hook refuses the commit, so the append sits uncommitted in the
tree every other session merges through, and the next merge aborts on it. That is why the guard
is at the write and not at the commit: by the time the hook speaks, the file is already dirty.

THE FAIL-OPEN WORLDS ARE LOAD-BEARING, and they are the half someone tightening the guard would
drop. Where git cannot answer -- no repository, no git binary -- the tree is not the integration
tree, and refusing would break every one of harness.py's 27 mkdtemp worlds and every CI runner.
A guard whose broken state blocks the write is a guard someone deletes. An UNIMPORTABLE
predicate is the one case that is loud instead of silent: with the tree detached there is no
branch test left to fall back to, so a swallowed ImportError would leave the integration tree
wholly unguarded rather than merely degraded.

AND THE WRITERS MUST CALL IT. A guard nothing calls satisfies every world above (§233), so the
last worlds read the three call sites: harness._append_task (tasks.jsonl and friction.jsonl both
reach disk through it), harness._write_tasks (a rewrite, which dirties the whole file rather
than one line), and board.append (the third ledger, and the one 44 wrote). board.py imports the
guard from harness rather than copying it -- a second implementation of one rule is how
FRICTION_KINDS came to reject a kind this repo's own merge_main.sh emits.

restartable: yes -- every world is a fresh temp git repo removed in a finally. Nothing reads or
writes the repository's real ledgers.
"""
import inspect
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))


def _git(d, *a):
    return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True, timeout=60)


def _integration_tree(parent):
    """A tree that IS the integration tree: the main worktree, with a linked worktree hanging
    off it. Built, not asserted -- that is the whole lesson of this file's first version."""
    d = tempfile.mkdtemp(dir=parent)
    _git(d, "init", "-q", ".")
    _git(d, "config", "user.email", "t@example.invalid")
    _git(d, "config", "user.name", "t")
    _git(d, "commit", "-q", "--allow-empty", "-m", "base")
    _git(d, "branch", "-M", "main")
    wt = tempfile.mkdtemp(dir=parent)
    os.rmdir(wt)  # git worktree add wants the path absent
    r = _git(d, "worktree", "add", "-q", "-b", "sidebranch", wt)
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    os.makedirs(os.path.join(wt, "runs"), exist_ok=True)
    return d, os.path.join(d, "runs", "tasks.jsonl"), wt, os.path.join(wt, "runs", "tasks.jsonl"), r


def _report(fails):
    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        print(f"\n{len(fails)} failure(s)")
        return 1
    print("  ledger writers: refuse in the integration tree (detached too), not in a linked "
          "worktree, fail open on no-git/no-git-binary; all three writers call the guard")
    return 0


def main():
    import harness

    fails = []
    tmp = tempfile.mkdtemp(prefix="itree_")
    try:
        d_int, p_int, d_wt, p_wt, r_add = _integration_tree(tmp)
        if r_add.returncode != 0:
            return _report([f"the world could not be built: `git worktree add` failed "
                            f"({r_add.stderr.strip()[:120]}) -- a world that does not hold the "
                            f"property proves nothing about a guard that reads it"])

        # W1: the integration tree REFUSES. The whole point.
        if not harness.refuse_in_integration_tree("w1", path=p_int):
            fails.append("W1: the integration tree did not refuse -- this is the tree whose "
                         "pre-commit hook cannot commit the row, so the append would sit dirty "
                         "in the tree every session merges through")

        # W2: THE SAME TREE, DETACHED, still refuses. The world tilerl's flip created, and the
        # one that turned all three guards off when the predicate read the branch. Asserts the
        # tree really is detached, so it cannot pass for the wrong reason.
        _git(d_int, "checkout", "-q", "--detach", "HEAD")
        sr = _git(d_int, "symbolic-ref", "--short", "HEAD")
        if sr.returncode == 0 and sr.stdout.strip() == "main":
            fails.append("W2 precondition: the tree is not actually detached, so this world says "
                         "nothing about the flip")
        if not harness.refuse_in_integration_tree("w2", path=p_int):
            fails.append("W2: a DETACHED integration tree did not refuse -- this is exactly the "
                         "state the flip created (main 0425accb), where a branch-name predicate "
                         "reads 'HEAD' and every guard silently turns off")

        # W3: a LINKED WORKTREE of the same repo does not refuse. Every session's normal write,
        # and the direction a too-broad guard breaks -- it would block all of them.
        if harness.refuse_in_integration_tree("w3", path=p_wt):
            fails.append("W3: a linked worktree refused -- that is every session's normal write")

        # W4: a standalone repo does not refuse. Not decoration: without the has-linked-worktrees
        # clause, a plain clone is its own main worktree, so CI and all 27 of harness.py's
        # git-init fixtures would refuse every write.
        d_solo = tempfile.mkdtemp(dir=tmp)
        _git(d_solo, "init", "-q", ".")
        _git(d_solo, "config", "user.email", "t@example.invalid")
        _git(d_solo, "config", "user.name", "t")
        _git(d_solo, "commit", "-q", "--allow-empty", "-m", "base")
        _git(d_solo, "branch", "-M", "main")
        os.makedirs(os.path.join(d_solo, "runs"), exist_ok=True)
        if harness.refuse_in_integration_tree("w4", path=os.path.join(d_solo, "runs", "tasks.jsonl")):
            fails.append("W4: a standalone clone on main refused -- it is its own main worktree "
                         "but nothing hangs off it, so it blocks nobody; CI is this shape")

        # W5: no repository at all -> fail open.
        d_bare = tempfile.mkdtemp(dir=tmp)
        os.makedirs(os.path.join(d_bare, "runs"), exist_ok=True)
        if harness.refuse_in_integration_tree("w5", path=os.path.join(d_bare, "runs", "tasks.jsonl")):
            fails.append("W5: a non-repository refused -- every _tmp_repo() world is this shape "
                         "and the selftest would refuse to write its own fixtures")

        # W6: git itself unavailable -> fail open. PATH emptied, so the subprocess raises rather
        # than returning nonzero: a different code path from W5.
        saved_path = os.environ.get("PATH", "")
        try:
            empty = os.path.join(tmp, "empty-bin")
            os.makedirs(empty, exist_ok=True)
            os.environ["PATH"] = empty
            refused = harness.refuse_in_integration_tree("w6", path=p_int)
        finally:
            os.environ["PATH"] = saved_path
        if refused:
            fails.append("W6: refused when git was unavailable -- an unanswerable predicate must "
                         "not block the write")

        # W7-W9: THE THREE WRITERS CALL IT. A guard nobody calls passes W1-W6 (§233).
        for fn in (harness._append_task, harness._write_tasks):
            if "refuse_in_integration_tree" not in inspect.getsource(fn):
                fails.append(f"W7: harness.{fn.__name__} does not call the guard -- both ledgers "
                             f"reach disk through these two functions, so a guard in the CLI ops "
                             f"instead would miss the next op someone adds")
        bsrc = open(os.path.join(ROOT, "scripts", "board.py"), encoding="utf-8").read()
        i = bsrc.find("def append(")
        if i < 0:
            fails.append("W9: scripts/board.py has no append() -- the writer moved")
        elif "refuse_in_integration_tree" not in bsrc[i:i + 2000]:
            fails.append("W9: board.append does not call the guard -- board.jsonl is the third "
                         "ledger and the one 44 wrote into the integration tree")
        elif "from harness import" not in bsrc[i:i + 2000]:
            fails.append("W9: board.append does not IMPORT the guard -- a local copy is a second "
                         "implementation of one rule, which is how FRICTION_KINDS came to reject "
                         "a kind merge_main.sh emits")

        # W10: the predicate is the SHARED one, not a private copy. tilerl's hook calls the same
        # function; two spellings of a similar idea is the defect this file exists downstream of.
        #
        # READ THE BODY, NOT THE WHOLE SOURCE. The first version grepped the function's source
        # for "symbolic-ref" and went red on its own docstring, which explains why that predicate
        # was abandoned -- a substring test cannot tell code from the prose recording why the
        # code is not there any more. Same shape as de-55's "signalled" grep, one hour apart.
        gsrc = inspect.getsource(harness.refuse_in_integration_tree)
        body = gsrc.split('"""')[2] if gsrc.count('"""') >= 2 else gsrc
        if "from integration_tree import is_integration_tree" not in body:
            fails.append("W10: the guard does not import integration_tree.is_integration_tree -- "
                         "the hook and the writers must read one implementation, or they drift")
        if "symbolic-ref" in body or "abbrev-ref" in body:
            fails.append("W10: the guard's BODY still reads a branch name -- that predicate went "
                         "inert the moment the integration tree was detached (2026-09-05)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return _report(fails)


if __name__ == "__main__":
    sys.exit(main())
