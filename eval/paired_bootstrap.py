"""Paired bootstrap for per-task c/n sampling evals (stage-2 prereg, fb 2026-09-14).

Reads two n-sample preds files (e.g. ET and EC checkpoints on the same benchmark),
aligns them by task_id, builds per-task success fraction c_i/n, and bootstrap-
resamples TASKS (not samples) B times to give a one-sided 95% CI for the mean
within-checkpoint rate and for the paired difference mean(c_T - c_C)/n.

Resampling the task unit is the point: the n draws within a task share one
problem difficulty and are not independent; the independent experimental units
are the tasks. Paired resampling draws the same task indices for both files so
the within-task difference is preserved.

Outputs JSON: per-file rates, mean difference, and the one-sided lower 95%
bound on d (the value above which T beats C with 95% confidence).

  python3 eval/paired_bootstrap.py --a preds_T....n10temp02.jsonl \
      --b preds_C....n10temp02.jsonl --label_a ET --label_b EC
"""
import argparse
import json

import numpy as np


def load_task_rates(path):
    """task_id -> (successes, n), plus task_id -> set(sample_idx).

    The sample-index set is the pairing contract: a valid paired stage-2
    comparison draws BOTH files' sample si for task t from the same seeded
    stream (eval/sampling.py). If the sets differ -- one file missing si, or
    indices shuffled between independent runs -- the per-task rates look
    pairable but the draws behind them are not, so the caller must refuse
    rather than emit a mislabeled CI.
    """
    agg = {}
    sis = {}
    n_seen = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "task_id" not in r:
                continue
            s, k = agg.get(r["task_id"], (0, 0))
            agg[r["task_id"]] = (s + int(bool(r.get("ok"))), k + 1)
            if "sample_idx" in r:
                sis.setdefault(r["task_id"], set()).add(r["sample_idx"])
            n_seen = r.get("n", n_seen)
    rates, ns = {}, {}
    for tid, (s, k) in agg.items():
        rates[tid] = s / k
        ns[tid] = k
    return rates, ns, sis


def assert_sample_aligned(sa, sb, tasks):
    """Refuse if the two files' per-task sample_idx sets disagree.

    Either both carry sample_idx (n>1 stage-2 preds) or neither does (greedy
    n=1 files, pairable by task by construction). One carrying it and the other
    not, or sets differing for any shared task, means the draws are unpaired.
    """
    a_has = bool(sa)
    b_has = bool(sb)
    if a_has != b_has:
        raise SystemExit(
            f"unpaired preds: one file carries sample_idx and the other does not "
            f"(a={a_has}, b={b_has}); compare files produced under one sampling "
            "contract (both n>1 seeded, or both greedy n=1)")
    if not a_has:
        return
    bad = [t for t in tasks if sa.get(t, set()) != sb.get(t, set())]
    if bad:
        ex = bad[0]
        raise SystemExit(
            f"unpaired preds: {len(bad)} task(s) have differing sample_idx sets "
            f"between A and B, e.g. {ex}: {sorted(sa.get(ex, ()))} vs "
            f"{sorted(sb.get(ex, ()))}. Paired bootstrap requires the same "
            "(task_id, sample_idx) draws in both files (eval/sampling.py seed "
            "contract)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="treatment preds (e.g. ET)")
    ap.add_argument("--b", required=True, help="control preds (e.g. EC)")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--alpha", type=float, default=0.05, help="one-sided tail")
    args = ap.parse_args()

    ra, na, sa = load_task_rates(args.a)
    rb, nb, sb = load_task_rates(args.b)
    tasks = sorted(set(ra).intersection(rb))
    only_a = set(ra) - set(rb)
    only_b = set(rb) - set(ra)
    if only_a or only_b:
        print(f"WARN task sets differ: only_a={len(only_a)} only_b={len(only_b)}; "
              f"pairing over {len(tasks)} shared tasks")
    if not tasks:
        raise SystemExit("no shared task_ids between the two files")
    assert_sample_aligned(sa, sb, tasks)
    assert na[tasks[0]] == nb[tasks[0]], (na[tasks[0]], nb[tasks[0]])
    n = na[tasks[0]]

    a = np.array([ra[t] for t in tasks])
    b = np.array([rb[t] for t in tasks])
    d = a - b
    rng = np.random.default_rng(args.seed)
    T = len(tasks)
    boot_a = np.empty(args.boot)
    boot_b = np.empty(args.boot)
    boot_d = np.empty(args.boot)
    for k in range(args.boot):
        idx = rng.integers(0, T, T)
        boot_a[k] = a[idx].mean()
        boot_b[k] = b[idx].mean()
        boot_d[k] = d[idx].mean()

    q = args.alpha
    result = {
        "tasks": T, "samples_per_task": n, "boot": args.boot, "seed": args.seed,
        "rate_a_observed": float(a.mean()),
        "rate_b_observed": float(b.mean()),
        "mean_diff_observed": float(d.mean()),
        # one-sided lower 95% bound on d: if > 0, A beats B at 95% confidence
        "diff_one_sided_lower_95": float(np.quantile(boot_d, q)),
        "diff_ci95_two_sided": [float(np.quantile(boot_d, q / 2)),
                                float(np.quantile(boot_d, 1 - q / 2))],
        "rate_a_ci95_two_sided": [float(np.quantile(boot_a, q / 2)),
                                  float(np.quantile(boot_a, 1 - q / 2))],
        "rate_b_ci95_two_sided": [float(np.quantile(boot_b, q / 2)),
                                  float(np.quantile(boot_b, 1 - q / 2))],
        "a_beats_b_fraction": float((boot_d > 0).mean()),
    }
    print(json.dumps(result, indent=2))
    return result


def _selftest():
    """Misalignment counterexamples: unpaired index sets must be refused.

    World A (missing si): both files have 3 rows per task but B lost sample 1;
    its rates look pairable by task, yet the draws are not the same stream.
    World B (one file n=1 without sample_idx, the other n=3 with it): contracts
    differ and the comparison cannot mean what its CI label says.
    Positive control: identical index sets pass the gate.
    """
    import os
    import tempfile

    def write(path, rows):
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "a.jsonl")
        b = os.path.join(td, "b.jsonl")
        def rows(ids):
            return [{"task_id": "t0", "sample_idx": i, "ok": 0, "n": 3}
                    for i in ids]
        write(a, rows([0, 1, 2]))
        write(b, rows([0, 2, 3]))
        _, _, sa = load_task_rates(a)
        _, _, sb = load_task_rates(b)
        try:
            assert_sample_aligned(sa, sb, ["t0"])
            raise AssertionError(
                "differing sample_idx sets were accepted -- the unpaired-draw "
                "defect (reseed-once stream offset) would produce a mislabeled CI")
        except SystemExit as e:
            assert "differing sample_idx" in str(e), str(e)

        c = os.path.join(td, "c.jsonl")
        d = os.path.join(td, "d.jsonl")
        write(c, [{"task_id": "t0", "ok": 1, "n": 1}])
        write(d, rows([0, 1, 2]))
        _, _, sc = load_task_rates(c)
        _, _, sd = load_task_rates(d)
        try:
            assert_sample_aligned(sc, sd, ["t0"])
            raise AssertionError("greedy-vs-n3 files were accepted as paired")
        except SystemExit as e:
            assert "sample_idx" in str(e), str(e)

        e = os.path.join(td, "e.jsonl")
        write(e, rows([0, 1, 2]))
        _, _, se = load_task_rates(e)
        assert_sample_aligned(sa, se, ["t0"])
    print("paired_bootstrap selftest OK: missing/shifted si and greedy-vs-n "
          "refused; identical sets accepted")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    main()
