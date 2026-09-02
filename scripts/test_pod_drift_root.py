#!/usr/bin/env python3
"""selftest for 44-15: pod_drift.py run from a gitdir location -- the hook's
staged-copy path in a linked worktree, <common>.git/worktrees/<name>/ -- must
write the manifest to the WORKTREE, not to a phantom path under the gitdir.
The hook passes the real root via POD_DRIFT_ROOT. Without that override,
__file__-relative ROOT resolves to the gitdir and the manifest lands where the
hook's consistency check never reads it -- a false REFUSAL on every worktree
commit that deletes a scoped file (measured 2026-09-02: the 44-13 probe
deletion, refused and reproduced in a throwaway worktree).

    python3 scripts/test_pod_drift_root.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

CLEAN = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=CLEAN)


def main():
    tmp = tempfile.mkdtemp()
    try:
        repo = os.path.join(tmp, "repo")
        os.makedirs(repo)
        _git(["init", "-q"], repo)
        _git(["config", "user.email", "t@t.t"], repo)
        _git(["config", "user.name", "t"], repo)
        with open(os.path.join(repo, "probe.py"), "w") as f:
            f.write("# scoped file\n")
        _git(["add", "probe.py"], repo)
        _git(["commit", "-qm", "init"], repo)

        # Mimic the hook's staged-copy location in a linked worktree:
        # <common>.git/worktrees/<name>/pod_drift_staged.py
        gitdir = os.path.join(tmp, "gitdir", "worktrees", "wt")
        os.makedirs(gitdir)
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        staged = os.path.join(gitdir, "pod_drift_staged.py")
        shutil.copy(os.path.join(here, "scripts", "pod_drift.py"), staged)

        real = os.path.join(repo, "data", "pod_head_manifest.txt")

        # Without the override: the manifest must NOT land in the repo (ROOT
        # resolves to the gitdir -- the bug's shape).
        subprocess.run([sys.executable, staged, "--write-index"],
                       cwd=repo, env=CLEAN, capture_output=True, text=True)
        assert not os.path.exists(real), \
            "manifest landed in repo without POD_DRIFT_ROOT -- the env override is dead code"

        # With the override: the manifest lands in the repo and names the scoped file.
        env = dict(CLEAN, POD_DRIFT_ROOT=repo)
        r = subprocess.run([sys.executable, staged, "--write-index"],
                           cwd=repo, env=env, capture_output=True, text=True)
        assert r.returncode == 0, f"--write-index failed: {r.stderr[-400:]}"
        assert os.path.exists(real), f"manifest not written to {real}"
        assert "probe.py" in open(real).read(), "manifest does not name the scoped file"
        phantom = os.path.join(gitdir, "data", "pod_head_manifest.txt")
        assert not os.path.exists(phantom), f"phantom manifest created at {phantom}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("selftest OK: pod_drift honors POD_DRIFT_ROOT (worktree manifest lands in the worktree)")


if __name__ == "__main__":
    main()
