#!/usr/bin/env python3
# restartable: pure aggregation over an already-merged jsonl; writes nothing, seconds to rerun.
"""Per-task HIT RATE over an e0-merged HumanEval preds file: fraction of problems with >=1
passing sample. This is the project's RL-readiness gate quantity (eval/math_hard.py's
pass@k is the same empirical any() indicator, k=8, temperature 0.8):

    gap = hitrate(n=8, T=0.8) - greedy_pass@1

It is NOT the unbiased pass@k estimator (that needs n > k). Named "hit rate" everywhere so
the two are never read as the same number. Input is the merged jsonl from
eval/e0_merge_score.py (one _header row, then one row per (task_id, sample_idx)).

    python eval/humaneval_hitrate.py preds_merged.jsonl
    python eval/humaneval_hitrate.py --selftest
"""

import json
import sys


def hitrate(path):
    by_task = {}
    n = None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("_header"):
            n = r.get("n")
            continue
        by_task.setdefault(r["task_id"], []).append(bool(r["ok"]))
    if not by_task:
        sys.exit(f"{path}: no task rows")
    counts = {t: len(v) for t, v in by_task.items()}
    bad = sorted(set(counts.values()))
    if len(bad) != 1:
        sys.exit(f"{path}: unequal sample counts across tasks (got {bad[:3]} ...)")
    n = n or next(iter(counts.values()))
    hits = sum(any(v) for v in by_task.values())
    n_tasks = len(by_task)
    return hits, n_tasks, n


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <merged_preds.jsonl> | --selftest")
    if sys.argv[1] == "--selftest":
        _selftest()
        return
    hits, n_tasks, n = hitrate(sys.argv[1])
    print(
        f"HUMANEVAL hitrate(n={n}, T from header) = {hits}/{n_tasks} = "
        f"{100 * hits / n_tasks:.2f}%  (fraction of problems with >=1 passing sample)"
    )


def _selftest():
    import os
    import tempfile

    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.jsonl")
    rows = [{"_header": 1, "n": 2}]
    # a: 0/2 miss, b: 1/2 hit, c: 2/2 hit -> 2/3
    for tid, oks in (("a", [False, False]), ("b", [True, False]), ("c", [False, True])):
        for si, ok in enumerate(oks):
            rows.append({"task_id": tid, "sample_idx": si, "ok": ok})
    open(p, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
    assert hitrate(p) == (2, 3, 2), hitrate(p)
    # Unequal counts must refuse, not silently average.
    p2 = os.path.join(d, "b.jsonl")
    open(p2, "w").write(
        json.dumps({"_header": 1, "n": 2})
        + "\n"
        + json.dumps({"task_id": "a", "sample_idx": 0, "ok": True})
        + "\n"
        + json.dumps({"task_id": "b", "sample_idx": 0, "ok": True})
        + "\n"
        + json.dumps({"task_id": "b", "sample_idx": 1, "ok": True})
        + "\n"
    )
    try:
        hitrate(p2)
    except SystemExit:
        pass
    else:
        raise AssertionError("unequal sample counts must refuse")
    print("humaneval_hitrate selftest OK: any() hit rate 2/3; unequal counts refuse")


if __name__ == "__main__":
    main()
