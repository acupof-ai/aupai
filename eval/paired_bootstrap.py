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
    """task_id -> (successes, n). Header rows skipped. Requires constant n."""
    agg = {}
    n_seen = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "task_id" not in r:
                continue
            s, k = agg.get(r["task_id"], (0, 0))
            agg[r["task_id"]] = (s + int(bool(r.get("ok"))), k + 1)
            n_seen = r.get("n", n_seen)
    rates, ns = {}, {}
    for tid, (s, k) in agg.items():
        rates[tid] = s / k
        ns[tid] = k
    return rates, ns


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

    ra, na = load_task_rates(args.a)
    rb, nb = load_task_rates(args.b)
    tasks = sorted(set(ra).intersection(rb))
    only_a = set(ra) - set(rb)
    only_b = set(rb) - set(ra)
    if only_a or only_b:
        print(f"WARN task sets differ: only_a={len(only_a)} only_b={len(only_b)}; "
              f"pairing over {len(tasks)} shared tasks")
    if not tasks:
        raise SystemExit("no shared task_ids between the two files")
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


if __name__ == "__main__":
    main()
