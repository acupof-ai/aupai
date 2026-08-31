#!/usr/bin/env python3
"""Opening an eval artifact for writing, with an overwrite refusal.

An eval's predictions are the only copy of what a checkpoint actually produced. A
rerun with the same name silently replaces them: on 2026-08-31 a 477-row preds file
that a fact cited was overwritten and the cited rows no longer existed. Scores can be
recomputed from a checkpoint; the generations that produced a number cannot, once the
checkpoint is gone or the sampler moves.

    with open_artifact(path) as f:        # refuses if path exists
    with open_artifact(path, force=True) as f:   # overwrite, deliberately
    p = versioned_path(path, run="ms_ckpt_p324")  # …/preds_x.ms_ckpt_p324.jsonl

Selftest: python3 scripts/eval_artifacts.py --selftest
"""
import json
import os
import sys


class ArtifactExists(Exception):
    """Raised instead of overwriting. Carries what to do about it."""


def versioned_path(path, run):
    """`<stem>.<run><ext>` -- one artifact per run, so a rerun never collides."""
    stem, ext = os.path.splitext(path)
    return f"{stem}.{run}{ext}"


def open_artifact(path, force=False, run=None, mode="w"):
    """Open an eval artifact for writing. Refuses an existing path unless `force`.

    With `run`, writes to the versioned path instead, which cannot collide and so
    never needs the refusal. Appending is not overwriting and is always allowed.
    """
    if run:
        path = versioned_path(path, run)
    if "a" not in mode and not force and os.path.exists(path):
        raise ArtifactExists(
            f"{path} exists ({os.path.getsize(path)} bytes). An eval artifact is the only "
            f"copy of what a checkpoint generated -- a 477-row preds file that a fact cited "
            f"was overwritten this way (2026-08-31). Pass --force to replace it, or name the "
            f"run so it versions: {versioned_path(path, '<run>')}"
        )
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return open(path, mode, encoding="utf-8")


def seal(path, expected, written=None):
    """Record how many rows an artifact should hold, next to it.

    A killed run leaves a partial file at the path a complete one belongs at:
    hard_ckpt_p324.pt.jsonl stopped at 512 of 1032 rows and nothing distinguished
    it from a finished run (e1, 2026-08-31). The refusal cannot see this -- the
    file does not exist yet when the doomed run opens it. Counting lines against
    the eval set is the only test, so the expected count is written where a reader
    will find it. Call after the writer closes; returns (ok, message)."""
    n = written
    if n is None:
        n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
    with open(path + ".rows", "w", encoding="utf-8") as f:
        json.dump({"expected": expected, "written": n, "complete": n == expected}, f)
    if n != expected:
        return False, f"{path}: {n} rows, expected {expected} -- truncated, do not cite it"
    return True, f"{path}: {n} rows, complete"


def verify_sealed(path):
    """(ok, message) for an artifact's sidecar. An unsealed artifact is unknown, not
    complete: it predates the seal or was written by something that does not seal."""
    side = path + ".rows"
    if not os.path.exists(side):
        return None, f"{path}: no .rows sidecar; completeness unknown"
    with open(side, encoding="utf-8") as f:
        d = json.load(f)
    n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
    if n != d.get("expected"):
        return False, f"{path}: {n} rows on disk, sidecar expected {d.get('expected')}"
    return True, f"{path}: {n} rows, matches its seal"


def _selftest():
    import tempfile

    d = tempfile.mkdtemp()
    p = os.path.join(d, "preds_x.jsonl")
    with open_artifact(p) as f:
        f.write('{"a":1}\n')
    assert os.path.exists(p)

    try:
        open_artifact(p)
    except ArtifactExists as e:
        assert "--force" in str(e) and "preds_x" in str(e), str(e)
    else:
        raise AssertionError("an existing artifact must refuse")

    # force replaces; the refusal must not be the only way through
    with open_artifact(p, force=True) as f:
        f.write('{"a":2}\n')
    assert open(p).read().strip() == '{"a":2}'

    # a versioned write never collides with the base path or another run
    v = versioned_path(p, "run1")
    assert v.endswith("preds_x.run1.jsonl"), v
    with open_artifact(p, run="run1") as f:
        f.write("x\n")
    with open_artifact(p, run="run2") as f:
        f.write("y\n")
    assert os.path.exists(v) and os.path.exists(versioned_path(p, "run2"))
    # the same run twice still refuses: versioning is not a licence to overwrite
    try:
        open_artifact(p, run="run1")
    except ArtifactExists:
        pass
    else:
        raise AssertionError("a repeated run must refuse too")

    # append is not overwrite
    with open_artifact(p, mode="a") as f:
        f.write('{"a":3}\n')
    assert len(open(p).read().strip().split("\n")) == 2
    # seal: a complete artifact verifies, a truncated one is caught
    q = os.path.join(d, "hard_x.jsonl")
    with open_artifact(q) as f:
        for i in range(1032):
            f.write('{"i":%d}\n' % i)
    ok, msg = seal(q, 1032)
    assert ok, msg
    ok, msg = verify_sealed(q)
    assert ok, msg
    # the real case: a killed run, 512 of 1032
    with open_artifact(q, force=True) as f:
        for i in range(512):
            f.write('{"i":%d}\n' % i)
    ok, msg = seal(q, 1032)
    assert ok is False and "truncated" in msg, msg
    ok, msg = verify_sealed(q)
    assert ok is False, msg
    assert verify_sealed(os.path.join(d, "preds_x.jsonl"))[0] is None, "unsealed must be unknown, not complete"
    print("eval_artifacts selftest OK: refuses, forces, versions, appends, seals, catches truncation")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else 0)
