"""Fixed-position task sharding for multi-card stage-2 evals (E0/ET/EC).

The split is by LIST POSITION in the data file (index % shard_n), never by dict
order, task_id hash, or a random draw: the same file must reproduce the same
partition, and N cards with shard_i=0..N-1 must cover every task exactly once.
Position -- not task_id -- is the key because MBPP ids are non-contiguous and
HumanEval ids are strings; hashing either would make coverage uncheckable by eye
and could silently skip or duplicate a task.
"""

import os


def validate(shard_i, shard_n):
    """Raise unless shard flags are both unset or a valid pair."""
    if (shard_i is None) != (shard_n is None):
        raise ValueError("--shard_i and --shard_n must be passed together")
    if shard_n is not None:
        if shard_n < 1:
            raise ValueError(f"--shard_n must be >= 1, got {shard_n}")
        if not 0 <= shard_i < shard_n:
            raise ValueError(f"--shard_i must be in [0, {shard_n}), got {shard_i}")


def select(items, shard_i, shard_n):
    """The shard's (original_index, item) pairs, preserving file order."""
    validate(shard_i, shard_n)
    if shard_n is None:
        return list(enumerate(items))
    return [(i, x) for i, x in enumerate(items) if i % shard_n == shard_i]


def runs_full_control(shard_i, shard_n):
    """Whether THIS process runs the full-dataset canonical/known-answer control.

    The control is a whole-set judge gate that hard-indexes a specific task
    (e.g. HumanEval/0), so it cannot run on a subset. It runs on an unsharded
    process and on shard 0 only; shards 1..n-1 skip it. Shard 0 covers the gate
    once, before any model load (the control is pure execution).
    """
    return shard_n is None or shard_i == 0


def claim_next(queue_dir, total):
    """Atomically claim the next task index, or None when the queue is exhausted.

    A dynamic queue exists because per-task cost here varies by ~2x (a long generation runs to
    max_new; a stop_at_0 task ends at token 0), so fixed-position sharding leaves workers idle
    while a slow peer finishes. Measured on step18000: fastest shard 27.9 min, slowest 51.3.

    The cursor is a FILE plus flock, not an in-process counter: the claimers are separate
    processes. `a+` so a missing file is created without a race between two openers, and the
    read-modify-write happens entirely inside the lock. Returns the pre-increment value, so
    callers get each index exactly once."""
    import fcntl  # noqa: PLC0415
    path = os.path.join(queue_dir, "cursor")
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            try:
                n = int((f.read() or "0").strip())
            except ValueError:
                n = 0
            if n >= total:
                return None
            f.seek(0)
            f.truncate()
            f.write(str(n + 1))
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return n


def label(shard_i, shard_n):
    """Filename tag for a shard run; '' for an unsharded run."""
    return f".shard{shard_i}of{shard_n}" if shard_n is not None else ""


def _selftest():
    # The real stage-2 populations: 164 HE and 427 MBPP over 8 cards must be an
    # exact, disjoint cover -- one missing or doubled task changes the denominator.
    for total in (156, 164, 338, 427):
        covered = []
        for si in range(8):
            part = [i for i, _ in select(list(range(total)), si, 8)]
            assert part == list(range(si, total, 8)), (total, si)
            covered += part
        assert len(covered) == total == len(set(covered)) == len(set(covered)), total
        assert sorted(covered) == list(range(total)), total
    # Unsharded returns everything; order preserved.
    assert select(["a", "b", "c"], None, None) == [(0, "a"), (1, "b"), (2, "c")]
    # Validation refuses half-set, bad range, and shard_n<1.
    for bad in ((0, None), (None, 8), (8, 8), (-1, 8), (0, 0)):
        try:
            validate(*bad)
        except ValueError:
            continue
        raise AssertionError(f"validate accepted {bad}")
    assert label(3, 8) == ".shard3of8" and label(None, None) == ""
    # Full-set control runs unsharded and on shard 0 only; every other shard skips,
    # so a hard-indexed full-set control never sees a subset without its key task.
    assert runs_full_control(None, None)
    assert runs_full_control(0, 8)
    assert not runs_full_control(1, 8)
    assert not runs_full_control(7, 8)
    # --- claim_next: a SHARED queue must hand out every index exactly once.
    # Real processes, not threads: the claimers are separate python processes, and a
    # threading.Barrier test would pass against a lock that only works in-process.
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    # os.fork, not multiprocessing: the spawn start method cannot pickle a locally
    # defined target, and fork is the truer model anyway -- the real claimers are
    # independent processes that share nothing but the file.
    d = tempfile.mkdtemp()
    try:
        total, nproc = 164, 11
        pids, files = [], []
        for k in range(nproc):
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:                      # child
                os.close(r)
                got = []
                # BOUNDED, and the bound is the point: an unfixed-cursor defect makes this
                # loop never terminate (a stale cursor re-hands indices forever), which hangs
                # the selftest instead of failing it -- measured on the off-by-one mutant.
                # 4x total is far above any honest claim count and still terminates.
                while len(got) < 4 * total:
                    i = claim_next(d, total)
                    if i is None:
                        break
                    got.append(i)
                os.write(w, (" ".join(map(str, got))).encode())
                os.close(w)
                os._exit(0)
            os.close(w)
            pids.append(pid)
            files.append(r)
        collected = []
        for pid, r in zip(pids, files):
            buf = b""
            while True:
                chunk = os.read(r, 65536)
                if not chunk:
                    break
                buf += chunk
            os.close(r)
            os.waitpid(pid, 0)
            if buf.strip():
                collected += [int(x) for x in buf.split()]
        assert sorted(collected) == list(range(total)), (
            f"queue is not an exact cover: got {len(collected)} claims, "
            f"{len(set(collected))} distinct, over {nproc} processes")
        # Exhausted queue keeps returning None rather than restarting at 0.
        assert claim_next(d, total) is None, "an exhausted queue handed out another index"
        # A fresh queue dir starts at 0, and an EMPTY cursor file is not a crash.
        d2 = tempfile.mkdtemp()
        try:
            assert claim_next(d2, 3) == 0
            open(os.path.join(d2, "cursor"), "w").close()   # truncated mid-flight
            assert claim_next(d2, 3) == 0, "an empty cursor file did not restart at 0"
            assert claim_next(d2, 3) == 1
        finally:
            shutil.rmtree(d2, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("shard selftest OK: 156/164/338/427 over 8 are an exact disjoint cover; "
          f"{nproc} processes claim {total} tasks exactly once")


if __name__ == "__main__":
    _selftest()
