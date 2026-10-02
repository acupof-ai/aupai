---
question: What did the S0/S2/S3 CED ladder measure, and did each stage pass before the gate launch?
status: measured
source: runs/experiments.jsonl v41_ced_smoke_0922/v41_ced_w8smoke_0923; runs/prereg.jsonl#v41_ced_0923@amended_1; PR #648
---

# 03 — The CED ladder

Three stages, each answered before the next launched.

| stage | shape | result | verdict |
|---|---|---|---|
| S0 build_only | `train.py --build_only` at gate shape, meta device, CPU | total 3,221,975,040 / active 355,430,400, exact | pass |
| S2 single-card smoke | v41_ced_smoke_0922, 1×H20, B4/accum6, 300 steps, val every 50 | train loss 7.539→3.629; val 5.706→4.062 monotone over 6 points; 0 NaN; gnorm 4.12→0.58; peak 43.16 GiB; steady 3.64 s/step, MFU 19%; no abnormal recompile | pass |
| S3 world-8 smoke | v41_ced_w8smoke_0923, 8 ranks, 60 steps, val every 30 | loss 7.524→4.385; val 5.227 (step 30) / 4.686 (step 60), end val 4.757; 0 NaN; peak 43.14 GiB; ~21K tok/s/gpu inside warmup; 8 ranks stepped together | pass |

Sources: S0 count in `runs/prereg.jsonl#v41_ced_0923@amended_1`; S2 row
`runs/experiments.jsonl` name=v41_ced_smoke_0922 (commit 2adbab50, status ok,
2026-09-22); S3 row name=v41_ced_w8smoke_0923 (commit a32d325f, 2026-09-23). S3's
experiment row is marked score-blocked: training passed and the rc=1 came only from the
chained scoring stage, which could not get a lane card because the grant is the full
8-card block with lane_card null. That is an allocation outcome, not a training signal;
the smoke checkpoint is unscored.

S3's 21K tok/s/gpu was read inside warmup; the gate run's post-warmup steady value is
27K (see [`04_main_run.md`](04_main_run.md)). The floor in the stop rule is 12K over a
500-step window, so neither reading is near it.

## The one code fix the ladder found

S2's first launches crashed in the CED global-KV projection with an fp8 alignment error:
the fp8 linear requires the M axis divisible by 16, and the projected doc-block M was
2,072. PR #648 (merged 2026-09-22) pads the decoder W_KV/W_Z projection M-axis to 16.
After the fix the smoke ran 300 steps with 0 NaN. No other model change was made
between the smokes and the gate launch; the gate flags are the S3 flags with
`--max_steps`/`--val_every` removed and `--save_every 2000` restored.

S2's history also includes one refused launch for training-scope drift (model.py on the
pod ahead of the pushed stamp) and one startup-gate timeout probe; both are recorded
rows, not model defects.
