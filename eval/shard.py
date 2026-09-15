"""Fixed-position task sharding for multi-card stage-2 evals (E0/ET/EC).

The split is by LIST POSITION in the data file (index % shard_n), never by dict
order, task_id hash, or a random draw: the same file must reproduce the same
partition, and N cards with shard_i=0..N-1 must cover every task exactly once.
Position -- not task_id -- is the key because MBPP ids are non-contiguous and
HumanEval ids are strings; hashing either would make coverage uncheckable by eye
and could silently skip or duplicate a task.
"""


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
    print("shard selftest OK: 156/164/338/427 over 8 are an exact disjoint cover")


if __name__ == "__main__":
    _selftest()
