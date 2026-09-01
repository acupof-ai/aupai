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
import hashlib
import json
import os
import sys
import time


class ArtifactExists(Exception):
    """Raised instead of overwriting. Carries what to do about it."""


def versioned_path(path, run):
    """`<stem>.<run><ext>` -- one artifact per run, so a rerun never collides."""
    stem, ext = os.path.splitext(path)
    return f"{stem}.{run}{ext}"


def _write_refusal(path, msg):
    """Leave the refusal AT the artifact path it protected: `<path>.REFUSED`.

    A guard that refuses correctly but reports somewhere the reader is not looking
    produces a symptom of absence, and absence is ambiguous. Three instances in one
    day, all correct guards (fb, 2026-09-01):

      - pod_push.sh exited 128 with no output at all, because `set -euo pipefail`
        killed it before its own refusal printed
      - the dynamo cache assert refused at startup; the observable was idle cards
      - THIS refusal landed in the wrapper's log while each cell's own log stayed
        zero bytes, so two 16B sampled cells looked like "still running" for an hour

    So the reader who goes looking for the output finds the reason instead of nothing.
    Best-effort by construction: a refusal that cannot write its sidecar must still
    raise, and never mask the ArtifactExists with an IOError from the sidecar.
    """
    try:
        with open(path + ".REFUSED", "w", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{msg}\n")
    except OSError:
        pass


def open_artifact(path, force=False, run=None, mode="w"):
    """Open an eval artifact for writing. Refuses an existing path unless `force`.

    With `run`, writes to the versioned path instead, which cannot collide and so
    never needs the refusal. Appending is not overwriting and is always allowed.

    The returned handle's `.name` is the path actually written. Attest THAT, never
    the path you passed in: with `run` they differ, and attesting the input meant a
    versioned run recorded a hash for the unversioned file it did not touch --
    l1_15b_final attested preds_l1_d3.jsonl, the 477-row overwrite it was written
    specifically to avoid (e1, 2026-09-01).
    """
    if run:
        path = versioned_path(path, run)
    if "a" not in mode and not force and os.path.exists(path):
        msg = (
            f"{path} exists ({os.path.getsize(path)} bytes). An eval artifact is the only "
            f"copy of what a checkpoint generated -- a 477-row preds file that a fact cited "
            f"was overwritten this way (2026-08-31). Pass --force to replace it, or name the "
            f"run so it versions: {versioned_path(path, '<run>')}"
        )
        _write_refusal(path, msg)
        raise ArtifactExists(msg)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    # A successful write clears any refusal left by an earlier attempt. Otherwise the
    # sidecar outlives the problem and the next reader finds a REFUSED note beside a
    # complete artifact -- a stale explanation is worse than none, because it explains
    # something that is no longer true.
    try:
        os.remove(path + ".REFUSED")
    except OSError:
        pass
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


def attest(path, root=None):
    """Record that this artifact existed with these bytes, in runs/artifact_refs.jsonl.

    A fact may cite a gitignored preds file, so no check can resolve the path on a
    machine that does not hold it -- five facts cited preds_l1_d3.jsonl after an
    unlogged rerun overwrote it, and nothing noticed for hours (e1, 2026-08-31).

    The writer attests at write time; the citation carries path + sha256; the check
    matches the two. What that proves is HISTORICAL: the cited bytes existed when the
    citation was made. It deliberately does not prove the current file matches, because
    preds are regenerated every run and a current-state check would false-alarm on
    every legitimate rerun (44's contract).
    """
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    rows = sum(1 for line in open(path, encoding="utf-8", errors="replace") if line.strip())
    row = {
        "path": os.path.relpath(path, root) if path.startswith(root) else path,
        "sha256": h.hexdigest(),
        "bytes": n,
        "rows": rows,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = os.path.join(root, "runs", "artifact_refs.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)  # one atomic append: concurrent evals must not interleave
    finally:
        os.close(fd)
    return row


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
    # the refusal must be findable AT the artifact, not only in whatever log the
    # caller happened to be writing. This is the assertion that would have saved an
    # hour: two sampled cells refused into a wrapper's log while their own logs stayed
    # zero bytes, and silence reads as "still running".
    assert os.path.exists(p + ".REFUSED"), (
        "a refusal must leave a sidecar at the path it protected; without it the "
        "reader finds an absence, and an absence is ambiguous")
    note = open(p + ".REFUSED", encoding="utf-8").read()
    assert "--force" in note and "exists" in note, note

    # force replaces; the refusal must not be the only way through
    with open_artifact(p, force=True) as f:
        f.write('{"a":2}\n')
    assert open(p).read().strip() == '{"a":2}'
    # and a successful write CLEARS the stale refusal: an explanation that outlives
    # the problem it explains is worse than none.
    assert not os.path.exists(p + ".REFUSED"), (
        "a successful write must clear the REFUSED sidecar, or the next reader finds "
        "a refusal note beside a complete artifact")

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
