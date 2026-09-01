#!/usr/bin/env python3
"""run_ddp.sh's pre-launch sync gate: the four states, driven through the real script.

Written the way de's ruling requires -- each case is a world where the defect is
PRESENT, and the assertion is that the guard goes red there. A test that only ever
sees the fixed world proves the guard runs, not that it catches anything.

The gate exists because train.py's manifest check cannot see this failure: it compares
the pod against the pod's OWN manifest, so a whole-tree push of an OLD sha is
internally consistent and passes. Nothing else reads which commit the pod is at.

torchrun is stubbed on PATH: these cases assert on the REFUSAL, and a case that
reached torchrun would otherwise try to start training.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

# The repo, not this file's parent: the pre-commit hook runs the STAGED copy of this
# test out of .git/worktrees/<wt>/hook-staged/scripts/, where a __file__-relative root
# names a directory holding only the staged file. git answers it correctly from any cwd.
ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                      capture_output=True, text=True).stdout.strip() \
    or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _world(stamp, drift=False):
    """A pod-shaped tree: run_ddp.sh, no .git, and whatever stamp the case wants.

    drift=True gives it a manifest naming a file whose content does not match, i.e.
    the pod was edited after the push -- the case the stamp alone cannot see.
    """
    d = tempfile.mkdtemp(prefix="podsync")
    shutil.copy(os.path.join(ROOT, "run_ddp.sh"), os.path.join(d, "run_ddp.sh"))
    os.chmod(os.path.join(d, "run_ddp.sh"), 0o755)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "scripts", "pod_drift.py"),
                os.path.join(d, "scripts", "pod_drift.py"))
    # A manifest the drift check can actually evaluate: one real file, and a sha that
    # either matches it or does not.
    body = b"print('x')\n"
    with open(os.path.join(d, "scripts", "probe_file.py"), "wb") as f:
        f.write(body)
    want = hashlib.sha256(b"EDITED-ON-THE-POD" if drift else body).hexdigest()
    with open(os.path.join(d, "data", "pod_head_manifest.txt"), "w") as f:
        f.write(f"{want}  scripts/probe_file.py  training\n")
    if stamp is not None:
        with open(os.path.join(d, "data", "pod_synced_head"), "w") as f:
            f.write(stamp)
    # torchrun must exist but must never really run: a case that gets past the gate
    # would otherwise launch training out of a temp dir.
    bin_ = os.path.join(d, "bin")
    os.makedirs(bin_, exist_ok=True)
    with open(os.path.join(bin_, "torchrun"), "w") as f:
        f.write("#!/bin/bash\necho TORCHRUN_REACHED\n")
    os.chmod(os.path.join(bin_, "torchrun"), 0o755)
    return d, bin_


def _run(d, bin_, env_extra=None):
    env = dict(os.environ, PATH=bin_ + os.pathsep + os.environ["PATH"])
    env.update(env_extra or {})
    r = subprocess.run([os.path.join(d, "run_ddp.sh"), "--name", "probe"],
                       capture_output=True, text=True, env=env, cwd=d)
    return r.returncode, r.stdout + r.stderr


def main():
    sha = "a" * 40
    cases = [
        # (name, stamp, env, want_rc0, must_appear, drift)
        ("no stamp at all -- nobody ran --all, or a partial push cleared it",
         None, {}, False, "no data/pod_synced_head", False),
        ("stamp from a DIRTY tree -- that code is on no commit",
         f"{sha} 3 2026-09-01T00:00:00Z\n", {}, False, "uncommitted manifest file", False),
        ("clean stamp -- the only state that may launch",
         f"{sha} 0 2026-09-01T00:00:00Z\n", {}, True, "TORCHRUN_REACHED", False),
        # The real one: build_starcoder_py.py was edited on the pod 99 seconds after a
        # clean stamp was written, and the stamp still read clean. Stamp says WHICH sha;
        # only the manifest says the tree is STILL that sha.
        ("clean stamp but the pod was edited after the push",
         f"{sha} 0 2026-09-01T00:00:00Z\n", {}, False, "drifted from the manifest", True),
        ("ALLOW_UNSYNCED=1 with no stamp -- the deliberate escape still works",
         None, {"ALLOW_UNSYNCED": "1"}, True, "TORCHRUN_REACHED", False),
    ]
    for name, stamp, env, want_rc0, needle, drift in cases:
        d, bin_ = _world(stamp, drift)
        try:
            rc, out = _run(d, bin_, env)
            if want_rc0:
                assert rc == 0, f"{name}: expected launch, got rc={rc}\n{out}"
            else:
                assert rc != 0, f"{name}: expected REFUSAL, got rc=0\n{out}"
                assert "TORCHRUN_REACHED" not in out, f"{name}: refused but still ran torchrun\n{out}"
            assert needle in out, f"{name}: missing {needle!r} in:\n{out}"
        finally:
            shutil.rmtree(d, ignore_errors=True)

    # The guard must not fire in a git checkout: developers run this on a laptop, where
    # no stamp exists and none should be required.
    d, bin_ = _world(None)
    try:
        os.makedirs(os.path.join(d, ".git"), exist_ok=True)
        rc, out = _run(d, bin_)
        assert rc == 0 and "TORCHRUN_REACHED" in out, f"git tree must not need a stamp:\n{out}"
        assert "REFUSING" not in out, f"git tree was refused:\n{out}"
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print("selftest OK")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(main())
    print(__doc__)
