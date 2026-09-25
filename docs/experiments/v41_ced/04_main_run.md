---
question: What is the v41_ced_0923 gate run's recipe, its stop rules, its resource use, and its full validation curve so far?
status: measured
source: runs/prereg.jsonl#v41_ced_0923@amended_1; runs/ced_w8_launch.sh; runs/experiments.jsonl; pod runs/v41_ced_0923.log read 2026-09-23
---

# 04 — Main run v41_ced_0923

## Recipe

| item | value |
|---|---|
| launch | 2026-09-23 03:52, `bash runs/ced_w8_launch.sh` (world 8, block 0-7, no lane) |
| tokens | 30.0B budget, 786,432 tok/step (B4 × accum6 × seq4096 × 8 ranks), 38,146 steps |
| schedule | warmup 500 absolute steps; warmdown 0.65; anneal_frac 0.10 (last 3,815 steps use anneal mix) |
| checkpoint | `--save_every 2000`, `ckpt_v41_ced_0923.pt` + `.stepN` rollers |
| model flags | `--ced --ced_enc_layers 6 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912` |
| MoE flags | `--moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41ced` |
| other | `--fp8`, compiled, `--no-grad_ckpt`, lr_scale 1.0 |

Source of truth: `runs/ced_w8_launch.sh` (committed in PR #659; the pod copy was
sha256 4c7b3a37… at launch) and the registered row
`runs/prereg.jsonl#v41_ced_0923@amended_1`.

## Stop rules (registered, not tightened after launch)

1. NaN or non-finite loss at any step → stop and report.
2. Peak > 80 GiB → stop.
3. tok/s/gpu below 12K over 500 consecutive steps at step ≥ 500 with no other job on
   the host → stop and report. 12K is the 0911 controller floor; steady is 27K.
4. HumanEval at a decision checkpoint inconsistent with reaching 30% → escalate to the
   user, no unilateral stop.

State at the last read: none has fired.

## Resources

| metric | value | config |
|---|---|---|
| peak per-rank memory | 43.14 GiB (S3 and gate run; S2 43.16) | fp8, compiled, B4, no grad ckpt |
| steady throughput | 27K tok/s/gpu, 3.58 s/step, MFU 20% | step 10940 log line |
| NaN count | 0 | grep over training log |
| ETA | ~28 h remaining as of step 10940 on 2026-09-23 | log's own window estimate; checkpoint saves perturb it |

The flat 0911 world-8 resumed segment read 26.5K; CED at 27K is within ordinary jitter
of it, and the two runs are different stages (the registered will_not_claim bars a
comparison claim).

## Validation curve, every 500-step point

Val is a fixed held-out pass printed every 500 steps. All 21 points to date, read from
the pod log with `grep -aoE 'step [0-9]+/38146 val [0-9.]+'`; machine-readable copy
[`data/val.csv`](data/val.csv).

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 |
|---|---|---|---|---|---|---|---|
| val | 2.539 | 2.317 | 2.249 | 2.207 | 2.178 | 2.143 | 2.125 |
| step | 4000 | 4500 | 5000 | 5500 | 6000 | 6500 | 7000 |
| val | 2.098 | 2.079 | 2.065 | 2.068 | 2.043 | 2.032 | 2.020 |
| step | 7500 | 8000 | 8500 | 9000 | 9500 | 10000 | 10500 |
| val | 2.011 | 2.024 | 2.010 | 1.999 | 1.981 | 1.974 | 1.984 |

Val falls overall with small reversals (5500 vs 5000, 8000 vs 7500, 10500 vs 10000),
each 0.003–0.013. No matched flat arm exists at this data and seed; the curve is read
against the gate trajectory alone. Reference points from the other runs, not controls:
flat 0922 val 2.001 at its stop step 8000; flat 0911 val 1.950 at 6000 on a different
mix build.

## Training log fields

Each step line carries train loss, learning rates per param group (muon, embed, scalar,
arq, moe_router), gnorm, cumulative tokens, tok/s/gpu, MFU, peak GiB, ETA and s/step.
gnorm moved from 4.12 early (S2) to ~0.58 around step 10940.
