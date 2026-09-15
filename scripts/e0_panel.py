#!/usr/bin/env python3
# restartable: read-only single-pass aggregation over a merged preds file; an interrupt
# costs only a CPU re-read (seconds), and --json_out is one atomic final write.
"""Read-only E0 readout panel: per-task c_i/n distribution, empties, FULL/CLEAN,
and a task-level flip comparison against an n=1 greedy preds file.

Consumes the merged preds written by eval/e0_merge_score.py (header + rows with
task_id/sample_idx/ok/empty/n). Does not score, generate, or touch eval/. The
greedy comparison is NOT paired_bootstrap: greedy files carry no sample_idx, so
eval/paired_bootstrap.py correctly refuses n=1-vs-n=10; this panel only aligns
task_ids and reports categories.
"""
import argparse
import json
import os
import sys
from collections import Counter


def _root():
    # script lives in <ROOT>/scripts on the pod; allow a /tmp scratch copy when CWD is the repo
    if os.path.isdir(os.path.join(os.getcwd(), "runs")):
        return os.getcwd()
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = _root()
HE_UNION = os.path.join(ROOT, "runs", "contam_r3_he_union.json")
MBPP_UNION = os.path.join(ROOT, "runs", "contam_r3_mbpp_union.json")


def read_rows(path):
    header, rows = None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_header"):
                header = obj
            else:
                rows.append(obj)
    return header, rows


def excluded_set(bench):
    if bench == "humaneval":
        with open(HE_UNION, encoding="utf-8") as fh:
            return set(json.load(fh)["r3_humaneval_union"])
    import re
    with open(MBPP_UNION, encoding="utf-8") as fh:
        u = json.load(fh)
    return {int(re.fullmatch(r"mbpp427:(\d+)", s).group(1)) for s in u["r3_mbpp_union"]}


def bench_of(rows):
    return "humaneval" if str(rows[0]["task_id"]).startswith("HumanEval") else "mbpp"


def per_task(rows):
    c, empt, n_seen = Counter(), Counter(), {}
    for r in rows:
        t = r["task_id"]
        c[t] += int(bool(r.get("ok")))
        empt[t] += int(bool(r.get("empty")))
        n_seen[t] = n_seen.get(t, 0) + 1
    n = n_seen[next(iter(n_seen))]
    return c, empt, n, n_seen


def rate(c, n, tasks):
    s = sum(c[t] for t in tasks)
    d = len(tasks) * n
    return s, d, s / d if d else float("nan")


def panel(merged_path, greedy_path):
    header, rows = read_rows(merged_path)
    bench = bench_of(rows)
    excl = excluded_set(bench)
    c, empt, n, n_seen = per_task(rows)
    bad_n = sorted(t for t, k in n_seen.items() if k != n)
    if bad_n:
        raise SystemExit(f"{len(bad_n)} tasks with n!={n}, e.g. {bad_n[:3]}")
    tasks = sorted(c, key=str)
    full = set(tasks)
    clean = [t for t in tasks if t not in excl]
    fs, fd, fr = rate(c, n, list(full))
    cs, cd, cr = rate(c, n, clean)
    total_samples = len(rows)
    empty_n = sum(int(bool(r.get("empty"))) for r in rows)

    # pass@1-style TASK fraction = tasks with c_i >= 1. At n=10,temp0.2 this estimates
    # pass@10-at-least-one, NOT unbiased pass@1 (unbiased pass@1 is the sample rate fr).
    f1 = sum(1 for t in tasks if c[t] >= 1)
    c1 = sum(1 for t in clean if c[t] >= 1)
    in_excl_pass = sorted(map(str, (t for t in tasks if t in excl and c[t] >= 1)))

    hist = Counter(c[t] for t in tasks)
    out = {
        "benchmark": bench, "merged_header": header, "n": n, "tasks": len(tasks),
        "FULL": {"pass": fs, "samples": fd, "rate": round(fr, 4),
                 "task_c_ge_1": f1, "task_c_ge_1_rate": round(f1 / len(tasks), 4)},
        "CLEAN": {"pass": cs, "samples": cd, "rate": round(cr, 4),
                  "excluded_tasks": len(full) - len(clean),
                  "task_c_ge_1": c1, "task_c_ge_1_rate": round(c1 / len(clean), 4),
                  "passing_tasks_in_excluded_set": in_excl_pass},
        "empty_samples": empty_n, "empty_rate": round(empty_n / total_samples, 4),
        "all_empty_tasks": sum(1 for t in tasks if empt[t] == n),
        "ci_histogram": {k: hist.get(k, 0) for k in range(n + 1)},
    }

    if greedy_path:
        _, grows = read_rows(greedy_path)
        g = {r["task_id"]: bool(r.get("ok")) for r in grows}
        rescued, rescued_all = [], []
        dropped, strong_greedy_fail = [], []
        for t in tasks:
            if t not in g:
                continue
            if not g[t] and c[t] >= 1:
                rescued.append(t)
                if c[t] == n:
                    rescued_all.append(t)
            if g[t] and c[t] == 0:
                dropped.append(t)
            if not g[t] and c[t] == n:
                strong_greedy_fail.append(t)
        out["vs_greedy"] = {
            "greedy_file": os.path.basename(greedy_path), "paired_tasks": len(set(tasks) & set(g)),
            "fail_greedy_pass_ge1_n10": sorted(map(str, rescued)),
            "fail_greedy_all10_pass": sorted(map(str, strong_greedy_fail)),
            "pass_greedy_zero_n10": sorted(map(str, dropped)),
        }
    return out


def render(p):
    L = [f"benchmark={p['benchmark']} n={p['n']} tasks={p['tasks']}",
         f"FULL sample solve {p['FULL']['pass']}/{p['FULL']['samples']} = {p['FULL']['rate']:.4f} "
         f"| tasks c>=1: {p['FULL']['task_c_ge_1']}/{p['tasks']} = {p['FULL']['task_c_ge_1_rate']:.4f}",
         f"CLEAN sample solve {p['CLEAN']['pass']}/{p['CLEAN']['samples']} = {p['CLEAN']['rate']:.4f} "
         f"({p['CLEAN']['excluded_tasks']} excluded) | tasks c>=1: "
         f"{p['CLEAN']['task_c_ge_1']}/{p['tasks']-p['CLEAN']['excluded_tasks']} = "
         f"{p['CLEAN']['task_c_ge_1_rate']:.4f}",
         ("NOTE n=1: tasks c>=1 IS greedy pass@1, identical to sample rate."
          if p["n"] == 1 else
          f"NOTE tasks c>=1 at n=10 is at-least-one-of-10 (pass@10-like), NOT unbiased "
          f"pass@1 (that is the sample rate). CLEAN passers inside excluded set: "
          f"{p['CLEAN']['passing_tasks_in_excluded_set']}"),
         f"empty {p['empty_samples']} samples ({p['empty_rate']:.3f}), "
         f"{p['all_empty_tasks']} tasks all-empty",
         "c_i/n histogram (tasks):"]
    for k, v in p["ci_histogram"].items():
        L.append(f"  c={k:>2}: {'#' * v} {v}")
    if "vs_greedy" in p:
        q = p["vs_greedy"]
        L += [f"vs greedy {q['greedy_file']} over {q['paired_tasks']} tasks:",
              f"  greedy FAIL -> n10 c>=1 (rescued): {len(q['fail_greedy_pass_ge1_n10'])} {q['fail_greedy_pass_ge1_n10']}",
              f"  greedy FAIL -> n10 c==n (all10):   {len(q['fail_greedy_all10_pass'])} {q['fail_greedy_all10_pass']}",
              f"  greedy PASS -> n10 c==0 (dropped): {len(q['pass_greedy_zero_n10'])} {q['pass_greedy_zero_n10']}"]
    return "\n".join(L)


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        m = os.path.join(td, "m.jsonl")
        with open(m, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_header": 1, "n": 3}) + "\n")
            # task 0: c=2/3; task 1: c=0/3 all empty; task 2: c=3/3
            for si, ok in enumerate([1, 1, 0]):
                fh.write(json.dumps({"task_id": "HumanEval/0", "sample_idx": si, "ok": ok,
                                     "empty": False, "n": 3}) + "\n")
            for si in range(3):
                fh.write(json.dumps({"task_id": "HumanEval/1", "sample_idx": si, "ok": 0,
                                     "empty": True, "n": 3}) + "\n")
            for si in range(3):
                fh.write(json.dumps({"task_id": "HumanEval/2", "sample_idx": si, "ok": 1,
                                     "empty": False, "n": 3}) + "\n")
        g = os.path.join(td, "g.jsonl")
        with open(g, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_header": 1, "n_problems": 3}) + "\n")
            for tid, ok in [("HumanEval/0", 0), ("HumanEval/1", 0), ("HumanEval/2", 0)]:
                fh.write(json.dumps({"task_id": tid, "ok": ok, "empty": False}) + "\n")
        p = panel(m, g)
        assert p["FULL"]["pass"] == 5 and p["FULL"]["samples"] == 9, p
        assert p["FULL"]["task_c_ge_1"] == 2 and p["FULL"]["task_c_ge_1_rate"] == round(2 / 3, 4), p
        assert p["empty_samples"] == 3 and p["all_empty_tasks"] == 1, p
        assert p["ci_histogram"][0] == 1 and p["ci_histogram"][2] == 1 and p["ci_histogram"][3] == 1
        q = p["vs_greedy"]
        assert q["fail_greedy_pass_ge1_n10"] == ["HumanEval/0", "HumanEval/2"], q
        assert q["fail_greedy_all10_pass"] == ["HumanEval/2"], q
        assert q["pass_greedy_zero_n10"] == [], q
        txt = render(p)
        assert "sample solve 5/9" in txt and "c= 3" in txt and "pass@10-like" in txt
    print("e0_panel selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="e0_merge_score output preds jsonl")
    ap.add_argument("--greedy", default=None, help="n=1 greedy preds for task flip comparison")
    ap.add_argument("--json_out", default=None)
    if "--selftest" in sys.argv:
        _selftest()
        return
    a = ap.parse_args()
    p = panel(a.merged, a.greedy)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(p, fh, ensure_ascii=False, indent=1)
    print(render(p))


if __name__ == "__main__":
    main()
