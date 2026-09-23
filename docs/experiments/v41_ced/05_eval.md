---
question: How is HumanEval scored for the CED gate, why the rstrip protocol, and what does the pass curve look like so far?
status: measured
source: runs/prereg.jsonl#v41_ced_0923@amended_1; PR #662; facts/base_eval.json#be.humaneval_rstrip_tokenizer_artifact; pod heval_merged_step*_result.json read 2026-09-23
---

# 05 — HumanEval protocol and curve

## The gate column: rstrip

The acceptance number is greedy pass@1 with the prompt's trailing newline stripped
before generation (`eval/humaneval_gen.py --rstrip_nl`). User ruling 2026-09-23,
`runs/prereg.jsonl#v41_ced_0923@amended_1`. The unstripped standard arm stays in the
report for continuity but is not used for the ≥30% gate; n=164 and the 30% threshold
did not change.

Why. The canonical HumanEval prompt ends in a bare newline. After a docstring line the
tokenizer merges newline+indent into one token, and a bare trailing newline is the
training-context predecessor of a column-0 line, so the model dedents and starts a new
top-level def with an empty function body. Measured on ckpt_v41_r3_0914, CPU greedy:
standard gave 0/164 at step 6000 and 1/164 at 12k/13k, with 82/164 and 63/164 empty
generations; 40/40 probed empties were column-0 def cuts. The same checkpoints under
rstrip gave 15/164, 11/164, 12/164 with zero empties. Appending four spaces is equally
out of distribution (1/164, 129/164 empty). Source:
facts/base_eval.json#be.humaneval_rstrip_tokenizer_artifact; the cross-harness protocol
audit (bigcode-evaluation-harness and DeepSeek-Coder strip the trailing newline;
Qwen2.5-Coder strips then re-adds one) is PR #662,
`docs/lessons/humaneval_base_protocol_audit_0923.md`.

This explains the flat line's zeros. v41_gate_0911's standard-arm HumanEval was
genuinely 0/164 at steps 2000/4000/6000/12000 under that prompt
(facts/v41.json#v41.gate_run_v41_gate_0911_summary and its per-step facts); the rstrip
read on a step-12k checkpoint of the same run was 12/164. The zero measured the model's
reaction to an unseen prompt shape, not coding capability. Base-model generative evals
must not introduce a token sequence absent from pretraining; the pretrain corpus has
effectively no ChatML and the continuation rstrip prompt is the in-distribution shape.

## How the CED curve is produced

All eight cards are held by training and the grant has lane_card null, so GPU scoring is
refused by the claim system. Scoring runs on CPU:

- 8 shards × 8 threads, each shard scores 1/8 of the 164 tasks
  (`eval/humaneval_gen.py --device cpu --rstrip_nl --shard_i i --shard_n 8`), pinned to
  cores away from the training dataloader;
- merged by `eval/e0_merge_score.py`, which writes
  `runs/heval_merged_step<N>_result.json` with FULL (denominator 164) and CLEAN
  (denominator 156; the same 8-task contamination union excluded) counts;
- a pod-only `runs/heval_auto_loop.sh` (flock-single, iteration-capped, MIN_STEP 6000)
  waits for each new `.stepN` checkpoint and runs the next sharded point.

One full sharded point takes 41–64 min; the first single-process attempt on step 2000
took ~109 s/problem and is still running (140/164, 5 passing, at the last read), so
step 2000 has no completed point. The machinery is PR #670 (open); step 4000 was the
first sharded read.

## Pass curve, rstrip arm, n=1 greedy

| step | FULL | CLEAN | empty | run window (UTC) |
|---|---|---|---|---|
| 2000 | incomplete: 5/140 in the single-process run | — | 0 | running |
| 4000 | 4/164 = 2.44% | 4/156 = 2.56% | 0 | ~12:0x |
| 6000 | 6/164 = 3.66% | 6/156 = 3.85% | 0 | 12:03–13:05 |
| 8000 | 13/164 = 7.93% | 13/156 = 8.33% | 1 | 13:07–14:11 |
| 10000 | 8/164 = 4.88% | 8/156 = 5.13% | 0 | 14:33–15:27 |

Machine-readable copy: [`data/humaneval.csv`](data/humaneval.csv). Sources: pod
`runs/heval_merged_step{4000,6000,8000,10000}_result.json`; the step-2000 partial is the
live log `runs/humaneval_rstrip_ced_s2000_cpu_0923.log`. Per-task resolution at n=164 is
0.61pp, so ±3–4 tasks is the noise band (uncertainty field of
facts/base_eval.json#be.humaneval_rstrip_tokenizer_artifact): the 8000→10000 drop of 5
tasks is not a trend. At step 10000 the run is 26% through the schedule, and the
anneal phase (last 3,815 steps) has not started.

## Before the final read

The registered amendment requires the sampled arm's truncate to pass `entry_point`
(`eval/humaneval_sample.py` had a truncation bug; fix owned by genA). The gate read at
step 38,146 uses the same rstrip arm and n=164; the standard arm is reported beside it.
The final read happens after training ends, on GPU, per the stop-rule escalation path.
