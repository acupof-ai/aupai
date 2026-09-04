#!/usr/bin/env python3
"""One temp root per selftest process, removed when the process ends.

The producers are 27 `mkdtemp` sites in harness.py and 7 in pod_drift.py, none of
which can clean up: a `_broken_*` returns its world for the caller to inspect, so
the world must outlive the function that built it. Fixing that per-site means 34
edits and a 35th leak the next time someone adds a world.

`scoped()` redirects `tempfile` at the process level instead, so every site --
including ones that do not exist yet, and including subprocesses, which inherit
TMPDIR -- lands under one directory that is removed on the way out.

MEASURED 2026-09-05, before this existed: `harness.py --selftest` left 12
directories and 7.5 MB behind per run, 5 of them from the `pod_drift --selftest`
subprocess it spawns. The user has cleared this laptop's disk three times.
"""

import contextlib
import os
import shutil
import sys
import tempfile


@contextlib.contextmanager
def scoped(prefix="aupai_selftest_"):
    """Run the body with every mkdtemp landing under one root, removed at exit.

    Restores tempfile.tempdir and TMPDIR afterwards so a caller that keeps
    running (a test importing this) is not left pointing at a deleted directory.
    """
    root = tempfile.mkdtemp(prefix=prefix)
    saved_tempdir = tempfile.tempdir
    saved_env = {k: os.environ.get(k) for k in ("TMPDIR", "TEMP", "TMP")}
    tempfile.tempdir = root
    for k in saved_env:
        os.environ[k] = root
    try:
        yield root
    finally:
        tempfile.tempdir = saved_tempdir
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(root, ignore_errors=True)


def _selftest():
    sysroot = tempfile.gettempdir()
    before = set(os.listdir(sysroot))

    # 1. Everything a body creates lands inside the root, and the root is gone after.
    made = []
    with scoped() as root:
        for _ in range(3):
            made.append(tempfile.mkdtemp())
        assert all(p.startswith(root) for p in made), made
        assert all(os.path.isdir(p) for p in made), "worlds must exist INSIDE the body"
        # A subprocess inherits TMPDIR, which is how the pod_drift --selftest leak dies.
        import subprocess

        sub = subprocess.run(
            [sys.executable, "-c", "import tempfile;print(tempfile.mkdtemp())"],
            capture_output=True, text=True,
        ).stdout.strip()
        assert sub.startswith(root), f"subprocess escaped the root: {sub} not under {root}"
    assert not os.path.exists(root), f"the root survived: {root}"
    assert not any(os.path.exists(p) for p in made), "a world survived its root"

    # 2. The system temp dir is left exactly as it was found -- the property the
    #    whole module exists for. Asserting only (1) would pass a scoped() that
    #    cleaned its root and leaked a sibling.
    after = set(os.listdir(sysroot))
    assert after == before, f"leaked into {sysroot}: {sorted(after - before)}"

    # 3. tempfile and the environment are restored, so a long-lived caller is not
    #    left writing into a deleted directory.
    assert tempfile.gettempdir() == sysroot, tempfile.gettempdir()
    assert tempfile.mkdtemp not in (None,) and not tempfile.gettempdir().startswith("/nonexistent")
    probe = tempfile.mkdtemp()
    try:
        assert probe.startswith(sysroot), f"tempdir not restored: {probe}"
    finally:
        shutil.rmtree(probe, ignore_errors=True)

    # 4. An exception in the body still cleans up.
    caught = None
    try:
        with scoped() as root2:
            tempfile.mkdtemp()
            raise ValueError("boom")
    except ValueError as e:
        caught = e
    assert caught is not None, "scoped() swallowed the exception"
    assert not os.path.exists(root2), f"the root survived a raising body: {root2}"

    print("tmpworld selftest OK (4 cases: containment, no sibling leak, restore, raise)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else 0)
