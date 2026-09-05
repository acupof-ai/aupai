#!/usr/bin/env python3
"""A ledger writer refuses in the integration tree unless AUPAI_CONTROLLER=1.

4c's ruling 2026-09-05, after two rows landed in the integration tree ten minutes apart: b0's
task row, then 44's board row. AGENTS.md already said to run these in your worktree, and the
rule's own coverage row explains why prose was all it could be -- "the invoking directory is a
shell fact no artifact records". It is recoverable from the tree the writer is about to append
to, which is what this makes checkable.

THE PREDICATE IS THE CHECKED-OUT BRANCH, NOT A PATH. A path test would hardcode one laptop's
/Users/bytedance/code/aupai and be wrong on the pod and in CI. `git symbolic-ref --short HEAD`
== "main" is the property, because the controller is the only session that commits to main
directly, so main IS the integration tree here.

WHAT THE REFUSAL PREVENTS is not the row -- the row is valid content -- it is the DIRTY LEDGER.
The integration tree's pre-commit hook refuses a non-controller commit, so the append sits
uncommitted in the tree every other session merges through, and the next merge aborts on it.
That is why the guard is at the write and not at the commit: by the time the hook speaks, the
file is already dirty.

THREE FAIL-OPEN WORLDS ARE LOAD-BEARING, and they are the half that would be dropped by
someone tightening the guard. If git cannot answer -- no repository, a detached HEAD, git
missing -- the tree cannot be the integration tree, which is by definition a checkout of main.
Refusing there breaks every one of harness.py's 27 mkdtemp worlds and every detached CI
checkout, and a guard whose broken state blocks the write is a guard someone deletes.

AND THE WRITERS MUST CALL IT. A guard nothing calls satisfies every predicate assertion above,
which is §233's shape, so the last three worlds read the three call sites: harness._append_task
(tasks.jsonl and friction.jsonl both reach disk through it), harness._write_tasks (a rewrite,
which dirties the whole file rather than one line), and board.append (the third ledger, and the
one 44 wrote). board.py imports the guard from harness rather than copying it -- a second
implementation of one rule is how FRICTION_KINDS came to reject a kind this repo's own
merge_main.sh emits.

restartable: yes -- every world is a fresh temp git repo removed in a finally, and
AUPAI_CONTROLLER is restored. Nothing reads or writes the repository's real ledgers.
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


def _repo(parent, branch):
    """A real git repo on `branch`, with runs/ -- the shape a worktree has."""
    d = tempfile.mkdtemp(dir=parent)
    _git(d, "init", "-q", ".")
    _git(d, "config", "user.email", "t@example.invalid")
    _git(d, "config", "user.name", "t")
    _git(d, "commit", "-q", "--allow-empty", "-m", "base")
    _git(d, "branch", "-M", branch)
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    return d, os.path.join(d, "runs", "tasks.jsonl")


def _report(fails):
    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        print(f"\n{len(fails)} failure(s)")
        return 1
    print("  ledger writers: refuse on main, lift under AUPAI_CONTROLLER=1, fail open on "
          "no-git/detached/no-git-binary; all three writers call the guard")
    return 0


def main():
    import harness

    fails = []
    saved = os.environ.get("AUPAI_CONTROLLER")
    tmp = tempfile.mkdtemp(prefix="itree_")
    try:
        os.environ.pop("AUPAI_CONTROLLER", None)

        # W1: a tree on main REFUSES. The whole point.
        d_main, p_main = _repo(tmp, "main")
        if not harness.refuse_in_integration_tree("w1", path=p_main):
            fails.append("W1: a tree checked out on main did not refuse -- this is the "
                         "integration tree, where the row cannot be committed")

        # W2: AUPAI_CONTROLLER=1 lifts it, on the SAME tree. Same world as W1, so a pass here
        # cannot be an artifact of a differently-built repo.
        os.environ["AUPAI_CONTROLLER"] = "1"
        if harness.refuse_in_integration_tree("w2", path=p_main):
            fails.append("W2: AUPAI_CONTROLLER=1 did not lift the refusal -- the controller is "
                         "the one session that does commit on main, and merge_main.sh's own "
                         "friction append runs there")
        os.environ.pop("AUPAI_CONTROLLER", None)

        # W3: a branch does NOT refuse. The normal case, and the one a too-broad guard breaks.
        d_de, p_de = _repo(tmp, "de")
        if harness.refuse_in_integration_tree("w3", path=p_de):
            fails.append("W3: a worktree on its own branch refused -- that is every session's "
                         "normal write and the guard would block all of them")

        # W4: no repository at all -> fail open. harness.py builds 27 worlds of this shape.
        d_bare = tempfile.mkdtemp(dir=tmp)
        os.makedirs(os.path.join(d_bare, "runs"), exist_ok=True)
        if harness.refuse_in_integration_tree("w4", path=os.path.join(d_bare, "runs", "tasks.jsonl")):
            fails.append("W4: a non-repository refused -- every _tmp_repo() world is this shape "
                         "and the selftest would refuse to write its own fixtures")

        # W5: detached HEAD -> fail open. symbolic-ref exits nonzero here, a third code path.
        _git(d_main, "checkout", "-q", "--detach", "HEAD")
        if harness.refuse_in_integration_tree("w5", path=p_main):
            fails.append("W5: a detached HEAD refused -- a detached checkout is not the "
                         "integration tree, and CI lands detached")

        # W6: git itself unavailable -> fail open. PATH emptied, so the subprocess raises
        # rather than returning nonzero: a different code path from W4/W5.
        d_ok, p_ok = _repo(tmp, "main")
        saved_path = os.environ.get("PATH", "")
        try:
            os.environ["PATH"] = os.path.join(tmp, "empty-bin")
            os.makedirs(os.environ["PATH"], exist_ok=True)
            refused = harness.refuse_in_integration_tree("w6", path=p_ok)
        finally:
            os.environ["PATH"] = saved_path
        if refused:
            fails.append("W6: refused when git was unavailable -- an unanswerable predicate "
                         "must not block the write")

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
    finally:
        if saved is None:
            os.environ.pop("AUPAI_CONTROLLER", None)
        else:
            os.environ["AUPAI_CONTROLLER"] = saved
        shutil.rmtree(tmp, ignore_errors=True)

    return _report(fails)


if __name__ == "__main__":
    sys.exit(main())
