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


def load_clean_task_ids(path):
    """Int task_ids from an MBPP contam manifest's r3_mbpp_clean list.

    Preds carry the bare sanitized int task_id; the manifest names them
    mbpp427:<num>. Refuses a missing file/key or a non-MBPP id.
    """
    import re
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        u = json.load(fh)
    v = u.get("r3_mbpp_clean")
    if not isinstance(v, list) or not v:
        raise SystemExit(f"{path}: r3_mbpp_clean missing or not a non-empty list")
    out = set()
    for s in v:
        m = re.fullmatch(r"mbpp427:(\d+)", str(s))
        if not m:
            raise SystemExit(f"{path}: clean id {s!r} is not mbpp427:<num>")
        out.add(int(m.group(1)))
    return out


def load_he_exclude(path):
    """String task_ids in an HE contam manifest's r3_humaneval_union (the 8).

    HE CLEAN = 164 - union = 156, i.e. EXCLUDE these, opposite of MBPP's
    r3_mbpp_clean keep-list. Preds carry ids like "HumanEval/21".
    """
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        u = json.load(fh)
    v = u.get("r3_humaneval_union")
    if not isinstance(v, list) or not v:
        raise SystemExit(f"{path}: r3_humaneval_union missing or not a non-empty list")
    return {str(s) for s in v}


def apply_task_filters(tasks, clean_ids=None, he_exclude=None):
    """The single filter path main and the selftest share. Returns the filtered list.

    The two flags are OPPOSITE filters on DIFFERENT benchmark id spaces (MBPP int keep vs
    HE string exclude); passing both is always a caller error, never an intersection.
    """
    if clean_ids is not None and he_exclude is not None:
        raise SystemExit("--clean (MBPP keep-list) and --he_union (HE exclude-list) "
                         "are opposite filters for different benchmarks; pass one")
    if clean_ids is not None:
        before = len(tasks)
        tasks = [t for t in tasks if t in clean_ids]
        if not tasks:
            raise SystemExit("no CLEAN task_ids survive the --clean filter")
        if len(tasks) != before:
            print(f"CLEAN: pairing restricted to {len(tasks)} of {before} tasks "
                  f"(r3 six-domain union excluded)")
    if he_exclude is not None:
        before = len(tasks)
        tasks = [t for t in tasks if t not in he_exclude]
        if not tasks:
            raise SystemExit("no tasks survive the --he_union exclusion")
        if len(tasks) != before:
            print(f"HE CLEAN: {len(tasks)} of {before} tasks "
                  f"(r3_humaneval_union excluded)")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="treatment preds (e.g. ET)")
    ap.add_argument("--b", required=True, help="control preds (e.g. EC)")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--alpha", type=float, default=0.05, help="one-sided tail")
    ap.add_argument("--clean", default=None,
                    help="MBPP contam manifest; restrict the paired unit to r3_mbpp_clean")
    ap.add_argument("--he_union", default=None,
                    help="HE contam manifest; exclude r3_humaneval_union (164->156)")
    args = ap.parse_args()

    clean_ids = load_clean_task_ids(args.clean)
    he_exclude = load_he_exclude(args.he_union)

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
    tasks = apply_task_filters(tasks, clean_ids=clean_ids, he_exclude=he_exclude)
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

        # HE union loader: an EXCLUDE list of string ids; empty/missing refuses.
        he = os.path.join(td, "he.json")
        with open(he, "w", encoding="utf-8") as fh:
            json.dump({"r3_humaneval_union": ["HumanEval/21", "HumanEval/7"]}, fh)
        assert load_he_exclude(he) == {"HumanEval/21", "HumanEval/7"}
        bad_he = os.path.join(td, "bad_he.json")
        with open(bad_he, "w", encoding="utf-8") as fh:
            json.dump({"r3_humaneval_union": []}, fh)
        try:
            load_he_exclude(bad_he)
            raise AssertionError("empty HE union was accepted")
        except SystemExit as e:
            assert "r3_humaneval_union" in str(e), str(e)

        # END-TO-END through apply_task_filters -- the exact list main() bootstraps.
        # (1) The full 164 HE tasks minus the real-shaped 8-id union is EXACTLY 156.
        he_all = [f"HumanEval/{i}" for i in range(164)]
        he8 = {"HumanEval/19", "HumanEval/66", "HumanEval/71", "HumanEval/78",
               "HumanEval/105", "HumanEval/123", "HumanEval/129", "HumanEval/156"}
        kept = apply_task_filters(list(he_all), he_exclude=he8)
        assert len(kept) == 156, len(kept)
        assert not (he8 & set(kept)), "an excluded union task entered the bootstrap"
        assert set(kept) == set(he_all) - he8  # exact complement, nothing else dropped
        # (2) the per-task stats vectors the bootstrap indexes have the 156 denominator;
        # an excluded id cannot be indexed into the kept list.
        rates = {t: 0.0 for t in he_all}
        import numpy as np
        arr = np.array([rates[t] for t in kept])
        assert arr.shape == (156,)

        # (3) MBPP --clean is a KEEP-list on INT ids and is unaffected by the HE path:
        # pass only clean, no he set, and it restricts normally; passing BOTH refuses.
        mbpp_all = list(range(427))
        mbpp_keep = {2, 9, 42, 400}
        mkept = apply_task_filters(list(mbpp_all), clean_ids=mbpp_keep)
        assert set(mkept) == mbpp_keep, mkept
        try:
            apply_task_filters(list(mbpp_all), clean_ids=mbpp_keep, he_exclude=he8)
            raise AssertionError("--clean and --he_union together were accepted")
        except SystemExit as e:
            assert "opposite filters" in str(e), str(e)
    print("paired_bootstrap selftest OK: missing/shifted si and greedy-vs-n "
          "refused; identical sets accepted; HE 164->156 exclude and MBPP keep "
          "filters verified end-to-end (both-together refused)")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    main()
