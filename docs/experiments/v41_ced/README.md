# V4.1 / CED experiment line, 2026-09-10 → present

One-page index. Reader: the user. Every number here has its source in the linked section.

## Goal

A ~355M-active coding/math model that clears **HumanEval pass@1 ≥ 30% at 30B tokens**
(`runs/prereg.jsonl#v41_ced_0923@amended_1`, gate unchanged from
`docs/standards/p1_data_recipe.md:256`). Target architecture: DeepSeek-V4.1-Flash
(`docs/standards/v41_pivot.md`). User order 2026-09-10 stopped V2/KDA; user order
2026-09-22 stopped the flat stack and made CED the single arm.

## Current state (read 2026-09-23 ~15:40Z on the pod)

| item | value | source |
|---|---|---|
| run | `v41_ced_0923`, world 8, block 0-7, no lane | `runs/experiments.jsonl` |
| progress | step ~11,300 / 38,146 (30%), 8.6B tok | pod log |
| val | 1.984 at step 10500, 21 points monotone overall | [`data/val.csv`](data/val.csv) |
| HumanEval rstrip | 13/164 at 8k, 8/164 at 10k (n=1 greedy; ±3–4 tasks noise) | [`data/humaneval.csv`](data/humaneval.csv) |
| throughput | 27K tok/s/gpu, s/step 3.58, MFU 20% | pod log |
| peak memory | 43.14 GiB / H20 (limit 80) | pod log |
| NaN | 0 | pod log |
| ETA | ~28 h to step 38,146 as of the last read | pod log |

The gate number is the **rstrip** arm (trailing prompt newline stripped), user ruling
2026-09-23; the standard arm is reported beside it, not used for the gate. See
[`05_eval.md`](05_eval.md).

## Key curves (full tables in the linked sections)

| step | 4000 | 6000 | 8000 | 10000 |
|---|---|---|---|---|
| val | 2.098 | 2.043 | 2.024 | 1.974 |
| HE rstrip FULL | 4/164 | 6/164 | 13/164 | 8/164 |
| HE rstrip CLEAN | 4/156 | 6/156 | 13/156 | 8/156 |

## Sections

| file | content |
|---|---|
| [`01_data.md`](01_data.md) | gate mix: 6 domains, weights, anneal, scheduled tokens, supply, ratios; decontamination; 32,768 tokenizer |
| [`02_architecture.md`](02_architecture.md) | flat line and its stop; CED structure; exact param counts |
| [`03_ladder.md`](03_ladder.md) | S0 build_only, S2 single-card smoke, S3 world-8 smoke |
| [`04_main_run.md`](04_main_run.md) | recipe, stop rules, throughput/memory, full val curve |
| [`05_eval.md`](05_eval.md) | rstrip protocol, CPU 8-shard loop, full HumanEval curve |
| [`06_incidents.md`](06_incidents.md) | fp8 M%16, resume-gate reds, merge_main stale reads |
| [`07_next.md`](07_next.md) | decision points and next measurements |

Filed with this line: [`flat_refactor_sequence_0923.md`](flat_refactor_sequence_0923.md),
[`non_ced_surface_analysis_0923.md`](non_ced_surface_analysis_0923.md),
[`v41_gate_0911_end.md`](v41_gate_0911_end.md) (historical runbook).
