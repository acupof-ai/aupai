#!/usr/bin/env python3
# restartable: pure aggregation over finished per-task preds files; writes one small JSON at the end;
# a rerun takes seconds and overwrites the same output.
"""Paired, statistically honest comparison of checkpoints on one task set.

WHY THIS EXISTS. HumanEval has 164 tasks. A single pass@1 number carries about +/-3.5pt of
binomial error at p=0.27, so "44 vs 45 solved" is not evidence of anything. The plateau
analysis (2026-10-13) compared four checkpoints by headline percentages, and one of them was
scored on 129/164 tasks -- an easier subset that read as +5pt. This tool refuses both errors:

  * it only compares tasks present in EVERY input (the common set), and says how many;
  * it reports a paired bootstrap CI over TASKS and an exact McNemar test on discordant pairs,
    so a difference is called real only when its interval excludes zero.

Pairing is the point: the same task appears in both checkpoints, so per-task difficulty
cancels and the variance of the difference is far below the variance of either rate.

INPUTS are prediction jsonl files in any of the repo's shapes (humaneval_gen, mbpp_gen,
gsm8k, sharded): one row per task with `task_id` and a boolean `ok`; `_header` rows are
skipped. Multiple files per label are merged (shards). A task scored twice for one label is
refused, not averaged -- that is a pass@k file and needs eval/paired_bootstrap.py.

    python3 eval/paired_compare.py \
        --run base=data/eval/preds_humaneval_ckpt_v42s2.pt.rstripnl.comp_v42s2.jsonl \
        --run s62269=data/eval/preds_humaneval_...step62269...jsonl \
        --exclude runs/contam_r3_he_union.json:r3_humaneval_union \
        --json runs/paired_he.json
    python3 eval/paired_compare.py --selftest

Pure CPU, read-only on its inputs.
"""
import argparse
import glob
import json
import math
import random
import sys

DEFAULT_B = 10000
DEFAULT_SEED = 20261013


def load_run(paths):
    """task_id -> bool for one label, merged over shards. Refuses duplicate task_ids."""
    rows = {}
    for pat in paths:
        hits = sorted(glob.glob(pat)) or [pat]
        for path in hits:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if "_header" in r or "task_id" not in r:
                        continue
                    tid = str(r["task_id"])
                    if tid in rows:
                        raise SystemExit(
                            f"{path}: task {tid} scored twice for one label. This tool takes "
                            f"one greedy row per task; use eval/paired_bootstrap.py for n>1.")
                    rows[tid] = bool(r.get("ok"))
    if not rows:
        raise SystemExit(f"no scored rows in {paths}")
    return rows


def load_exclude(spec):
    """'path:key' -> set of task ids. A bare path means the file is a JSON list."""
    if not spec:
        return set()
    path, _, key = spec.partition(":")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    ids = d[key] if key else d
    return {str(x) for x in ids}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value on discordant counts b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(p_hat, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    den = 1 + z * z / n
    mid = (p_hat + z * z / (2 * n)) / den
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def paired_bootstrap(a, b, n_boot, seed):
    """95% percentile CI of mean(b - a) over tasks, resampling task indices jointly."""
    n = len(a)
    rng = random.Random(seed)
    diffs = [bb - aa for aa, bb in zip(a, b)]
    stats = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[min(n_boot - 1, int(0.975 * n_boot))]
    return lo, hi


def compare(runs, exclude, ref, n_boot=DEFAULT_B, seed=DEFAULT_SEED):
    labels = list(runs)
    common = set.intersection(*[set(v) for v in runs.values()]) - exclude
    union = set.union(*[set(v) for v in runs.values()])
    ids = sorted(common)
    out = {
        "n_common": len(ids),
        "n_union": len(union),
        "excluded": len(exclude & union),
        "partial_labels": {k: len(v) for k, v in runs.items() if set(v) - exclude != common
                           and len(set(v) - exclude) != len(common)},
        "per_run": {},
        "vs_ref": {},
        "ref": ref,
    }
    if not ids:
        raise SystemExit("no task is common to every run after exclusion")
    for k in labels:
        x = [int(runs[k][t]) for t in ids]
        p = sum(x) / len(x)
        lo, hi = wilson(p, len(x))
        out["per_run"][k] = {"pass": sum(x), "n": len(x), "rate": round(p, 4),
                             "wilson95": [round(lo, 4), round(hi, 4)],
                             "n_scored_total": len(runs[k])}
    a = [int(runs[ref][t]) for t in ids]
    for k in labels:
        if k == ref:
            continue
        b = [int(runs[k][t]) for t in ids]
        only_ref = [t for t in ids if runs[ref][t] and not runs[k][t]]
        only_new = [t for t in ids if runs[k][t] and not runs[ref][t]]
        lo, hi = paired_bootstrap(a, b, n_boot, seed)
        delta = (sum(b) - sum(a)) / len(ids)
        out["vs_ref"][k] = {
            "delta": round(delta, 4),
            "paired_boot95": [round(lo, 4), round(hi, 4)],
            "mcnemar_p": round(mcnemar_exact(len(only_ref), len(only_new)), 4),
            "lost": only_ref, "gained": only_new,
            "verdict": ("better" if lo > 0 else "worse" if hi < 0 else "no detectable difference"),
        }
    # Smallest true difference detectable at alpha .05 two-sided, power .80: (1.96+0.84)*SE.
    # Paired SE uses the OBSERVED discordance d=(gained+lost)/n (largest over runs, so the bound
    # is conservative): Var(delta) ~= d/n for paired binary outcomes. The independent-sample
    # bound sqrt(2p(1-p)/n) is shown beside it: it is what an unpaired reading of the same
    # numbers would need, and it shows how much the paired design buys.
    n = len(ids)
    p = sum(a) / n
    disc = max(((len(v["gained"]) + len(v["lost"])) / n for v in out["vs_ref"].values()), default=0.0)
    z = 1.96 + 0.84
    out["discordance_max"] = round(disc, 4)
    out["mde80_paired"] = round(z * math.sqrt(max(disc, 1.0 / n) / n), 4)
    out["mde80_independent"] = round(z * math.sqrt(2 * p * (1 - p) / n), 4)
    return out


def render(res):
    lines = [f"common tasks: {res['n_common']} (union {res['n_union']}, excluded {res['excluded']})"]
    if res["partial_labels"]:
        lines.append(f"PARTIAL runs restricted to the common set: {res['partial_labels']}")
    for k, v in res["per_run"].items():
        lines.append(f"  {k:<14} {v['pass']:>4}/{v['n']:<4} {100 * v['rate']:6.2f}%  "
                     f"wilson95 [{100 * v['wilson95'][0]:.1f}, {100 * v['wilson95'][1]:.1f}]")
    for k, v in res["vs_ref"].items():
        lines.append(f"  {k} - {res['ref']}: {100 * v['delta']:+.2f}pt  paired95 "
                     f"[{100 * v['paired_boot95'][0]:+.2f}, {100 * v['paired_boot95'][1]:+.2f}]  "
                     f"McNemar p={v['mcnemar_p']}  -> {v['verdict']}  "
                     f"(+{len(v['gained'])}/-{len(v['lost'])})")
    lines.append(f"min detectable true difference (alpha .05, power .80): paired "
                 f"{100 * res['mde80_paired']:.1f}pt at discordance {100 * res['discordance_max']:.1f}%; "
                 f"unpaired reading would need {100 * res['mde80_independent']:.1f}pt")
    return "\n".join(lines)


def _selftest():
    # 1) identical runs: delta 0, verdict no difference, p=1.
    base = {f"T/{i}": i % 3 == 0 for i in range(150)}
    r = compare({"a": base, "b": dict(base)}, set(), "a", n_boot=500)
    assert r["vs_ref"]["b"]["delta"] == 0 and r["vs_ref"]["b"]["mcnemar_p"] == 1.0
    assert r["vs_ref"]["b"]["verdict"] == "no detectable difference"
    # 2) a large real gain is detected.
    better = {t: True if int(t.split("/")[1]) < 120 else v for t, v in base.items()}
    r = compare({"a": base, "b": better}, set(), "a", n_boot=500)
    assert r["vs_ref"]["b"]["verdict"] == "better", r["vs_ref"]["b"]
    # 3) a +1 task change on 150 is NOT called real.
    plus1 = dict(base); plus1["T/1"] = True
    r = compare({"a": base, "b": plus1}, set(), "a", n_boot=500)
    assert r["vs_ref"]["b"]["verdict"] == "no detectable difference"
    # 4) a partial run restricts every label to the common set.
    part = {t: v for t, v in base.items() if int(t.split("/")[1]) < 100}
    r = compare({"a": base, "b": part}, set(), "a", n_boot=200)
    assert r["n_common"] == 100 and r["per_run"]["a"]["n"] == 100
    # 5) exclusion removes tasks from the denominator.
    r = compare({"a": base, "b": base}, {"T/0", "T/3"}, "a", n_boot=200)
    assert r["n_common"] == 148
    # 6) McNemar matches a known value: b=0, c=6 -> p = 2/64 = 0.03125.
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-12
    print("paired_compare selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", default=[], help="label=path[,path...] (globs ok)")
    ap.add_argument("--ref", help="reference label (default: first --run)")
    ap.add_argument("--exclude", help="path:key of task ids to drop (e.g. contamination union)")
    ap.add_argument("--boot", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if len(args.run) < 2:
        ap.error("need at least two --run label=path")
    runs = {}
    for spec in args.run:
        label, _, paths = spec.partition("=")
        runs[label] = load_run(paths.split(","))
    ref = args.ref or next(iter(runs))
    res = compare(runs, load_exclude(args.exclude), ref, args.boot, args.seed)
    print(render(res))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
