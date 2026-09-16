# E0 / ET / EC readout guide

E0 is the r3 final checkpoint's n=10 gate reading; ET/EC are later arm
comparisons. All consume preds produced by `runs/e0_n10.sh` (8-card shards,
HE→MBPP serial, ~5,910 generations/ckpt) and merged by `eval/e0_merge_score.py`.

## 1. Merge (already done by the runner / fb, writes two files per bench)

```bash
python3 eval/e0_merge_score.py --bench humaneval \
  --glob 'data/eval/preds_*e0_n10*humaneval*.jsonl' \
  --n 10 --out runs/e0_he_merged.jsonl --result runs/e0_summary.json
python3 eval/e0_merge_score.py --bench mbpp \
  --glob 'data/eval/preds_*e0_n10*mbpp*.jsonl' \
  --n 10 --out runs/e0_mbpp_merged.jsonl --result runs/e0_summary.json
```

The merger verifies 164/427 tasks, exactly n samples/task, contiguous shards,
and prints FULL vs CLEAN (HE 156 / MBPP 338) sample-level pass rates.

## 2. Panel (read-only, CPU; scripts/e0_panel.py)

Per-task c_i/n histogram, empty rate, FULL/CLEAN, and task flips vs any n=1
greedy preds file (37k/36k/…).

```bash
# HE, compared with the 37k greedy point already on the pod
python3 scripts/e0_panel.py \
  --merged runs/e0_he_merged.jsonl \
  --greedy data/eval/preds_humaneval_ckpt_v41_r3_0914.milestone_he37k_step37000.pt.v41r3_he37000_rstrip_cpu.jsonl
# MBPP (no greedy counterpart file exists yet; drop --greedy)
python3 scripts/e0_panel.py --merged runs/e0_mbpp_merged.jsonl
```

Output: FULL/CLEAN pass and samples, empty samples + all-empty tasks,
`c=0..n` task histogram, and against greedy: **rescued** (greedy fail → n10
c≥1, with the all-10 subset) and **dropped** (greedy pass → n10 zero).

Validated on real preds 2026-09-15: 37k greedy vs 36k greedy (n=1 both) —
histogram 136/28, 0 empty, rescued 49/55, dropped 1/114/120/14/25/27/38/57.

## 3. Paired bootstrap — exact commands (eval/paired_bootstrap.py)

For arm T vs control C on HumanEval (task ids are the contam strings):

```bash
python3 eval/paired_bootstrap.py \
  --a runs/et_he_merged.jsonl --b runs/ec_he_merged.jsonl \
  --label_a ET --label_b EC --boot 10000 --seed 20260914
```

MBPP CLEAN only (preds carry bare int ids; the tool takes the manifest):

```bash
python3 eval/paired_bootstrap.py \
  --a runs/et_mbpp_merged.jsonl --b runs/ec_mbpp_merged.jsonl \
  --label_a ET --label_b EC --clean runs/contam_r3_mbpp_union.json
```

Interpretation of the JSON:
- `mean_diff_observed` — raw T−C pass-rate delta over paired tasks.
- **One-sided 95% decision rule**: T beats C at one-sided 95% iff
  `diff_one_sided_lower_95 > 0` (equivalently `a_beats_b_fraction ≥ 0.95`).
- `diff_ci95_two_sided` / `rate_*_ci95_two_sided` for the two-sided report.
- The tool refuses unpaired files (differing sample_idx sets, or n=1 vs n>1),
  so ET/EC must come from one seeded sampling contract.

## 4. Minimum detectable difference

`--boot 10000` gives the empirical CI directly; an a-priori normal
approximation for a paired rate delta over T tasks, per-task fraction in
[0,1] with SD of the paired difference σ_d:

  MDE(one-sided 95%, ~80% power) ≈ 2.49 σ_d / √T

Read σ_d from the data: `std(c_i^T/n − c_i^T/n … c_i^C/n)` over tasks. With
T=156 CLEAN HE tasks and a typical paired-fraction SD 0.35-0.45 (n=10
binomial spread), MDE ≈ 2.49×0.40/√156 ≈ **0.080** — deltas under ~8 points
on HE CLEAN are not resolvable at 80% power; MBPP CLEAN T=338 gives ≈ **0.054**.
Bootstrap CIs supersede this approximation; it is only for pre-reading power.

## 5. What NOT to compare

- n=1 greedy (36k/37k) vs n=10 E0: different sample contracts; paired_bootstrap
  refuses. Use the panel's flip categories for orientation only.
- FULL vs CLEAN within a checkpoint: a population difference, not an arm effect;
  report both numbers but no CI between them.
