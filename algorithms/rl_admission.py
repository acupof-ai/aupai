#!/usr/bin/env python3
# restartable: pure aggregation over an already-merged predictions jsonl; writes nothing.
"""Is a checkpoint worth RL cards? Report the quantity a group-relative method actually needs.

THE DOCUMENTED GATE IS THE WRONG QUANTITY. AGENTS.md's RL entry point is
`eval/math_hard.py --ckpt X --k 8 --temperature 0.8` with "needs pass@8 - pass@1 >= 15pt". It
is wrong twice:

  1. math-hard v1 is void as a metric of record -- this repo's own generators contaminated it
     (AGENTS.md, Eval table). Continuity only.
  2. pass@k - pass@1 is not what GRPO/GSPO consumes. The advantage is computed WITHIN a
     sampled group of G rollouts, so a group teaches nothing unless it contains both a success
     and a failure. For per-sample success rate p and group size G that probability is

         P_mixed(p, G) = 1 - p^G - (1-p)^G

     The two disagree sharply near saturation. At p = 0.95, G = k = 8:
         old bar:  hit@8 - p = (1 - 0.05^8) - 0.95 = 5.000 pt   -> REJECTED at 15pt
         P_mixed:  1 - 0.95^8 - 0.05^8           = 33.658 %     -> a third of groups train
     The old bar rejects a checkpoint whose groups carry gradient two steps out of six.

WHAT THE TRAINER ACTUALLY DOES, verified rather than assumed. algorithms/rl_code_trainer.py:468
calls group_advantage(rewards, normalize_std=False): advantage is MEAN-SUBTRACTED ONLY, no std
division (the math arm, rlvr_trainer.py:130, does divide). Two consequences reported below:

  - the mixed indicator is the gate, but not the whole signal. With c successes in G, the
    advantages are 1-c/G on successes and -c/G on failures, so the group's total advantage
    mass is sum|adv| = 2c(G-c)/G, and over c ~ Binomial(G, p),
        E[sum|adv|] = 2 (G-1) p (1-p)
    which is maximal at p = 0.5 and falls off toward either end. A std-normalised trainer
    would divide that back out to roughly a constant; this one does not.
  - so mean-only DOWNWEIGHTS near-solved tasks automatically (p=0.95 gives 0.665 against
    p=0.5's 3.500 at G=8). That is an argument for a LOOSER admission bar than a
    std-normalised trainer would need, not a tighter one: a saturated task is not wasting an
    optimizer step, it is contributing a small step.

Input is the merged predictions jsonl eval/humaneval_hitrate.py reads: one `_header` row, then
one {task_id, sample_idx, ok} row per sample. p-hat per task is passes/n from that file; every
derived number below is a plug-in at p-hat and says so.

    python3 algorithms/rl_admission.py preds_merged.jsonl [--group_size 8] [--json out.json]
    python3 algorithms/rl_admission.py --selftest

THE THRESHOLD IS NOT PICKED HERE. See `derivation` in the output and the commit message: the
bar is a function of the trainer's prompts-per-rank-per-step, and choosing it is the user's
decision.
"""

import argparse
import json
import sys

#: the trainer's default --group_size (algorithms/rl_code_trainer.py:208 and
#: algorithms/rlvr_trainer.py:161); the reported numbers are meaningless at a different G, so
#: it is an explicit flag rather than a constant read from a distance.
DEFAULT_G = 8
#: the trainer's default --batch, prompts per GPU per step (rl_code_trainer.py:207). The
#: derivation below is stated per rank at this B.
DEFAULT_B = 4


def p_mixed(p, g):
    """P(a group of g i.i.d. Bernoulli(p) rollouts contains both a success and a failure)."""
    return 1.0 - p**g - (1.0 - p) ** g


def hit_at_k(p, k):
    """P(at least one success in k i.i.d. rollouts) -- the empirical any() indicator
    eval/math_hard.py and eval/humaneval_hitrate.py report, as a plug-in at p. NOT the
    unbiased pass@k estimator."""
    return 1.0 - (1.0 - p) ** k


def advantage_mass(p, g):
    """E[sum|advantage|] over a group under MEAN-ONLY advantages: 2 (g-1) p (1-p).

    Derivation: with c successes, sum|adv| = c(1 - c/g) + (g-c)(c/g) = 2c(g-c)/g. For
    c ~ Binomial(g, p), E[c(g-c)] = g E[c] - E[c^2] = g(g-1) p (1-p), so E[sum|adv|] =
    2 (g-1) p (1-p). Checked against a brute-force enumeration in the selftest.
    """
    return 2.0 * (g - 1) * p * (1.0 - p)


def read_preds(path):
    """{task_id: [bool, ...]} plus the per-task sample count, refusing an unequal file.

    Same reader contract as eval/humaneval_hitrate.py: unequal sample counts across tasks
    refuse rather than silently average, because p-hat's resolution is 1/n and mixing two n's
    makes the reported distribution uninterpretable.
    """
    by_task = {}
    header_n = None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("_header"):
            header_n = r.get("n")
            continue
        by_task.setdefault(r["task_id"], []).append(bool(r["ok"]))
    if not by_task:
        sys.exit(f"{path}: no task rows")
    counts = sorted({len(v) for v in by_task.values()})
    if len(counts) != 1:
        sys.exit(f"{path}: unequal sample counts across tasks (got {counts[:3]} ...)")
    return by_task, header_n or counts[0]


def _quantile(sorted_vals, q):
    """Nearest-rank quantile: an order statistic of the sample, never an interpolation
    between two tasks (there is no task at p=0.37 just because two neighbours bracket it)."""
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def report(by_task, n, g=DEFAULT_G, b=DEFAULT_B):
    """Every admission number for one predictions file, as a dict. No thresholds applied."""
    ps = sorted(sum(v) / len(v) for v in by_task.values())
    t = len(ps)
    mean_p = sum(ps) / t
    f = sum(p_mixed(p, g) for p in ps) / t
    mass = sum(advantage_mass(p, g) for p in ps) / t
    # The assumption-free second reading, available only when the file's own block size is the
    # group size: the fraction of tasks whose OBSERVED n samples were themselves mixed. It
    # makes no Bernoulli assumption at all, and it is one draw per task, so it is noisier than
    # the plug-in and is reported beside it rather than instead of it.
    observed_mixed = (
        sum(1 for v in by_task.values() if 0 < sum(v) < len(v)) / t if n == g else None
    )
    return {
        "tasks": t,
        "samples_per_task": n,
        "group_size": g,
        "batch_prompts_per_rank_per_step": b,
        "p_mean": mean_p,
        "p_min": ps[0],
        "p_p10": _quantile(ps, 0.10),
        "p_median": _quantile(ps, 0.50),
        "p_p90": _quantile(ps, 0.90),
        "p_max": ps[-1],
        "tasks_p_eq_0": sum(1 for p in ps if p == 0.0),
        "tasks_p_eq_1": sum(1 for p in ps if p == 1.0),
        # THE ADMISSION QUANTITY: expected fraction of sampled groups that carry a gradient.
        "mixed_fraction": f,
        "mixed_fraction_observed": observed_mixed,
        # Per unit of generation: a group costs g rollouts, so this is how many
        # gradient-carrying groups 1,000 generated rollouts buy.
        "mixed_groups_per_1k_rollouts": 1000.0 * f / g,
        # Expected gradient-carrying groups per rank per optimizer step at the trainer's batch.
        "kept_groups_per_rank_step": b * f,
        # Fraction of steps in which at least one of the b groups is mixed, i.e. the step is
        # not skipped outright (rl_code_trainer.py:388 `continue`s when none is kept).
        "steps_with_a_gradient": 1.0 - (1.0 - f) ** b,
        # Signal quality, not just presence; mean-only advantage (rl_code_trainer.py:468).
        "mean_advantage_mass": mass,
        # THE OLD BAR, computed from the SAME p-hat so the two are comparable. hit@k is the
        # any() indicator eval/math_hard.py reports; pass@1 is p-hat itself.
        "old_bar_hit_at_k_minus_p_pt": 100.0 * (sum(hit_at_k(p, g) - p for p in ps) / t),
        "old_bar_threshold_pt": 15.0,
    }


def fmt(r):
    obs = ("n/a (samples_per_task != group_size)" if r["mixed_fraction_observed"] is None
           else f"{100 * r['mixed_fraction_observed']:.2f}%")
    return "\n".join([
        f"tasks {r['tasks']}  samples/task {r['samples_per_task']}  G {r['group_size']}  "
        f"B {r['batch_prompts_per_rank_per_step']}",
        f"p-hat  min {r['p_min']:.3f}  p10 {r['p_p10']:.3f}  median {r['p_median']:.3f}  "
        f"p90 {r['p_p90']:.3f}  max {r['p_max']:.3f}  mean {r['p_mean']:.3f}",
        f"       p==0 on {r['tasks_p_eq_0']} tasks, p==1 on {r['tasks_p_eq_1']} tasks",
        f"MIXED GROUP FRACTION (plug-in at p-hat)  {100 * r['mixed_fraction']:.2f}%"
        f"   [observed, assumption-free: {obs}]",
        f"  gradient-carrying groups per 1k rollouts  {r['mixed_groups_per_1k_rollouts']:.1f}",
        f"  kept groups per rank per step at B={r['batch_prompts_per_rank_per_step']}  "
        f"{r['kept_groups_per_rank_step']:.2f}",
        f"  steps that produce any gradient            {100 * r['steps_with_a_gradient']:.2f}%",
        f"  mean advantage mass E[sum|adv|] (mean-only advantages)  {r['mean_advantage_mass']:.3f}"
        f"  (max at p=0.5 is {advantage_mass(0.5, r['group_size']):.3f})",
        f"OLD BAR hit@{r['group_size']} - pass@1 = {r['old_bar_hit_at_k_minus_p_pt']:.3f} pt "
        f"against its {r['old_bar_threshold_pt']:.0f} pt -- reported for comparison only; "
        f"it is not the quantity a group-relative method consumes.",
        "THRESHOLD: not applied here. The bar is a function of B, not a constant; see the "
        "module docstring and the PR. User's decision.",
    ])


def _selftest():
    import os
    import tempfile

    g = 8
    # 1. Closed forms against brute force, over the whole simplex of group compositions.
    for p in (0.0, 0.05, 0.5, 0.95, 1.0):
        # P_mixed and E[sum|adv|] by enumerating c = 0..g with binomial weights.
        from math import comb
        pm = sum(comb(g, c) * p**c * (1 - p) ** (g - c) for c in range(1, g))
        am = sum(comb(g, c) * p**c * (1 - p) ** (g - c) * 2 * c * (g - c) / g for c in range(g + 1))
        assert abs(p_mixed(p, g) - pm) < 1e-12, (p, p_mixed(p, g), pm)
        assert abs(advantage_mass(p, g) - am) < 1e-12, (p, advantage_mass(p, g), am)
    print("  closed forms == brute-force enumeration at p in {0, .05, .5, .95, 1}")

    # 2. The four known-answer predictions files.
    d = tempfile.mkdtemp()

    def write(name, n, per_task_passes):
        path = os.path.join(d, name)
        rows = [{"_header": 1, "n": n}]
        for ti, k in enumerate(per_task_passes):
            for si in range(n):
                rows.append({"task_id": f"t{ti}", "sample_idx": si, "ok": si < k})
        open(path, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
        return report(*read_preds(path), g=g)

    allpass = write("allpass.jsonl", 8, [8] * 5)
    assert allpass["mixed_fraction"] == 0.0, allpass["mixed_fraction"]
    assert allpass["mixed_fraction_observed"] == 0.0
    assert allpass["mean_advantage_mass"] == 0.0
    allfail = write("allfail.jsonl", 8, [0] * 5)
    assert allfail["mixed_fraction"] == 0.0, allfail["mixed_fraction"]
    assert allfail["p_max"] == 0.0
    print("  all-pass and all-fail: mixed fraction exactly 0, advantage mass exactly 0")

    half = write("half.jsonl", 8, [4] * 5)
    assert abs(half["mixed_fraction"] - (1 - 2 * 0.5**8)) < 1e-12, half["mixed_fraction"]
    assert abs(half["mixed_fraction"] - 0.9921875) < 1e-12
    assert abs(half["mean_advantage_mass"] - 2 * 7 * 0.25) < 1e-12  # 3.5
    assert abs(half["mixed_groups_per_1k_rollouts"] - 1000 * 0.9921875 / 8) < 1e-9  # 124.02
    print("  p=0.5, G=8: mixed 99.21875%, advantage mass 3.500, 124.0 groups per 1k rollouts")

    # THE CASE THAT SEPARATES THE TWO CRITERIA: p = 0.95. 20 tasks, 19 of 20 samples pass.
    sat = write("p095.jsonl", 20, [19] * 20)
    assert abs(sat["p_mean"] - 0.95) < 1e-12
    want = 1 - 0.95**8 - 0.05**8
    assert abs(sat["mixed_fraction"] - want) < 1e-12, (sat["mixed_fraction"], want)
    assert abs(sat["mixed_fraction"] - 0.33657956867218018) < 1e-12, sat["mixed_fraction"]
    old = sat["old_bar_hit_at_k_minus_p_pt"]
    assert abs(old - 5.0) < 1e-6, old  # (1 - 0.05^8) - 0.95 = 0.05000000000
    assert old < sat["old_bar_threshold_pt"], "the old bar must REJECT this checkpoint"
    assert sat["mixed_fraction"] > 0.20, "P(mixed) must ACCEPT it at the pilot 20%"
    # and at the derivation's own bar, 1/B with the trainer's B=4:
    assert sat["mixed_fraction"] > 1.0 / DEFAULT_B, "P(mixed) must clear 1/B = 25% too"
    assert abs(sat["kept_groups_per_rank_step"] - 4 * want) < 1e-12  # 1.346 groups/rank/step
    print(f"  p=0.95, G=8: old bar {old:.3f} pt < 15 pt REJECTS; "
          f"P(mixed) {100 * want:.3f}% > 25% = 1/B ACCEPTS; "
          f"{sat['kept_groups_per_rank_step']:.3f} kept groups per rank per step")

    # 3. An unequal file must refuse, not average (same contract as humaneval_hitrate).
    bad = os.path.join(d, "bad.jsonl")
    open(bad, "w").write("\n".join(json.dumps(r) for r in [
        {"_header": 1, "n": 2},
        {"task_id": "a", "sample_idx": 0, "ok": True},
        {"task_id": "b", "sample_idx": 0, "ok": True},
        {"task_id": "b", "sample_idx": 1, "ok": False},
    ]) + "\n")
    try:
        read_preds(bad)
    except SystemExit:
        pass
    else:
        raise AssertionError("unequal sample counts must refuse")
    print("  unequal sample counts refuse")
    print("rl_admission selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("preds", nargs="?", help="merged predictions jsonl (eval/e0_merge_score.py)")
    ap.add_argument("--group_size", type=int, default=DEFAULT_G,
                    help=f"the trainer's --group_size G (default {DEFAULT_G})")
    ap.add_argument("--batch", type=int, default=DEFAULT_B,
                    help=f"the trainer's --batch, prompts per rank per step (default {DEFAULT_B})")
    ap.add_argument("--json", help="also write the report dict here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.preds:
        ap.error("a predictions jsonl or --selftest")
    r = report(*read_preds(a.preds), g=a.group_size, b=a.batch)
    print(fmt(r))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2)


if __name__ == "__main__":
    main()
