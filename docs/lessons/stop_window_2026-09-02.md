---
question: What enters the first stop window of p500m_20b_0902, in what order, and what decides the relaunch configuration?
status: recorded
source: user ruling 2026-09-02 09:44Z (stop after the step-3000 save) and 09:50Z (throughput is the core); runs/experiments.jsonl p500m_20b_0902
---

# Stop window 1: after the step-3000 save

Timeline (UTC): step 2600 at 09:44Z, 11.0 s/step, save at step 3000 ≈ 10:57Z. Stop: pin step 3000 first (e1, `_pin_milestone` only, no scoring: `ckpt_p500m_20b_0902.milestone_stopwindow1_step3000.pt`, a hardlink the `.pt.step*` pruner cannot see; without it step 4500 deletes the resume target), then kill `supervise_run.sh` (it would auto-resume once), then SIGTERM the torchrun leader (interrupt checkpoint), then `nvidia-smi` shows zero processes. Executor: tilerl. Downtime target: 2.5 h.

## Throughput, the main item

| quantity | value | source |
|---|---|---|
| current | 12K tok/s/gpu, 12% MFU, batch 32, accum 1, grad_ckpt on, FP8, 8 cards | run log |
| 200M run | 73K tok/s/gpu, 31% MFU, batch 32, no grad_ckpt | `facts/efficiency.json#eff.fb_mfu` |
| grad_ckpt cost | ~25% wall-clock | `facts/efficiency.json#eff.batch_ceiling` |
| H20 FP8 dense peak | ~296 TFLOPS | `facts/efficiency.json#eff.h20_specs` |
| 500M FLOPs per token | ~3.0 GFLOPs (6N) plus attention | arithmetic |
| ceiling at 100% MFU | ~99K tok/s/gpu → 20B tokens ≥ 7 h on 8 cards | arithmetic |
| reachable | 30–40K tok/s/gpu (3×) → ~15–18 h total | 200M precedent |
| not reachable | 10× (120K tok/s/gpu) exceeds the FP8 ceiling; a 3 h run of 20B tokens needs 231K tok/s/gpu | arithmetic |

In the window: torch.profiler over 3 steps at the live shape; A/B of 40 steps each: `--batch 16 --accum 2` without grad_ckpt, current shape with torch.compile, `--batch 32` without grad_ckpt; relaunch with any configuration ≥16K tok/s/gpu that keeps 1,048,576 tokens/step; ≥30K is the target. A wall-clock cut beyond 3× needs a different problem: fewer tokens, a smaller model, or more cards.

## Code that merges, in order, each behind `test_arch_compat`

de-13 cursor (the first resume is this window; the second re-reads without it) → de-23 train half → de-20 → d57273f (domain_loss reads the checkpoint's mix) → e1-23 required flags → e1-22 dispatch and continuation prompts → e1-16 → tilerl-14/15 → b0-8 model split. Then `prove_resume`, `harness check` 0 FAIL, `pod_push --all`, relaunch with `--resume ckpt_p500m_20b_0902.milestone_stopwindow1_step3000.pt` through `supervise_run.sh`, first 50 steps read against the pre-stop loss.

Excluded: de-2 (changes data), 44-12 (startup path, run end), any corpus change. `eval/score_matrix.py:765` still defaults `--mix` to the ladder mix (44's challenge on d57273f, 09:52Z): correct as a fact, deferred to de-26 because the file is in the frozen set and `cache_guard` turns the defect into a refusal, not a wrong number.

## Token budget (user, 09:52Z: model unchanged, mix unchanged, fewer tokens)

Candidate: `total_tokens` 19,999,997,952 → 9,999,998,976 in `data/mix_500m.json` at the relaunch, tokens/step unchanged at 1,048,576, steps 19,151 → 9,537; step 3000 = 3.1B is then 31% in and still in the constant-LR phase (warmdown 0.1 × total starts at step 8,583). Gated on de answering from `build_mix`: the cursor continues every domain without re-read under the halved plan, and the LR at step 3000 is bitwise the old value. Remaining time at 10B: 6.9B tokens / (8 × 16K) = 15 h, / (8 × 30K) = 8 h. The user's "3 hours" equals 2.6B tokens at 30K tok/s/gpu, less than the 3.1B already trained; it is not a budget the run can meet, and a Chinchilla-scale 10B is the smallest budget with a literature basis.

## Decision (user, 10:06Z): 200M first, then 300M; the 500M is not resumed

The 500M stops at step 3000 and stays pinned as `ckpt_p500m_20b_0902.milestone_stopwindow1_step3000.pt`; the resume line above is void. The token-budget section above is superseded by this one. Composition unchanged: `data/mix_200m_4b.json` and `data/mix_300m_6b.json` carry the 500M weights at 4B and 6B (`write_mix_500m.py --total`, generated on the pod, 863143b).

| run | config | steps | measured or estimated tok/s/gpu | wall |
|---|---|---|---|---|
| p200m_4b_0902 | d1024 L12 (3 MLA + 9 KDA), 206.13M built, batch 16 accum 2, `--no-grad_ckpt`. The first launch (11:57Z) used batch 32 accum 1 and OOM'd in the first backward at 95.1 GiB on every rank, exactly as `facts/efficiency.json#eff.microbatch_32_oom` (2026-08-31, 200M shape, 93.8/95.2 GB) recorded; the 72-73K baseline was batch 16 accum 2. The line had been checked against argparse, not against the facts | 3,815 | 73K measured (`facts/efficiency.json#eff.fb_mfu`) | 1.9 h |
| p300m_6b_0902 | d1024 L18 (4 MLA + 14 KDA), 293.05M built, batch 32 `--no-grad_ckpt` **does not fit**: 93.7-93.96 GiB allocated of 95.22 at fp8/bf16 (b0's probe, 11:46Z, real peak). Decided by an A/B in the 200M/300M gap: `--grad_ckpt` b16 a2 vs `--no-grad_ckpt` b8 a4, 20 steps each, tokens/step unchanged (b16 without grad_ckpt is not expected to fit at L18 since b32 does not fit at L12) | 5,722 | unmeasured, ~50K | ~4 h |

Order after the stop: merges as listed → a 60-minute throughput sprint on all eight cards with the 200M config (user, 10:07Z: every idle owner works on training speed) → launch the 200M with the sprint's best config → the 300M after it ends. Launch lines, every knob explicit (row 173's omission):

```
setsid nohup bash scripts/supervise_run.sh p200m_4b_0902 -- env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 ./run_ddp.sh --mix data/mix_200m_4b.json --name p200m_4b_0902 --dim 1024 --layers 12 --heads 8 --ffn_hidden 3072 --batch 16 --accum 2 --no-grad_ckpt --lr_scale 1.0 --warmdown 0.1 --anneal_frac 0 --warmup 300 --save_every 500
setsid nohup bash scripts/supervise_run.sh p300m_6b_0902 -- env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 ./run_ddp.sh --mix data/mix_300m_6b.json --name p300m_6b_0902 --dim 1024 --layers 18 --heads 8 --ffn_hidden 3072 <batch/accum/grad_ckpt from the gap A/B> --lr_scale 1.0 --warmdown 0.1 --anneal_frac 0 --warmup 300 --save_every 500
```

Sprint split: tilerl profiles (3-step trace, compile on/off, batch 32/64, fla chunk size); b0 reads the trace per block kind and lists the matmuls not on FP8; de measures host side (loader wait, save, val, NCCL share); 3b measures the startup cost of `build_mix` and the token-cache read path for the 4B mix; e1 surveys industry methods with an expected gain and cost per item for this stack (`docs/lessons/throughput_survey.md`); 44 reviews every number's measurement config. Every probe is a `harness launch` row with a hypothesis.

## Card split (user, 12:35Z): cards 4-7 belong to the user, aupai works on 0-3, performance first

p200m_4b_0902 was stopped by fb at 12:35:57Z with SIGTERM to the torchrun leader (verified cmdline); `ckpt_p200m_4b_0902.pt.interrupt.step832` (959,435,257 bytes) is the resume point, later on four cards as `--batch 16 --accum 4` (tokens/step unchanged). The supervisor had already been removed at 12:29Z so that run_ddp's end-of-run scoring, which used the ladder default `--mix`, could not trigger a spurious resume; 3415e9e now reads the mix from the checkpoint. Cards 4, 5, 6, 7 are the user's until further notice: no aupai process touches them. On 0-3, one job at a time, announced to fb first, in this order: de's 3-step trace at b16a2 (busy vs idle, per-class measured vs roofline) → 300M A/B (`--grad_ckpt` b16a2 vs `--no-grad_ckpt` b8a4) → kernel and computation A/Bs (tilerl: kernels, b0: model-level), each 20 steps with loss parity ≤ 1e-3 per step. The user allows replacing or hand-writing kernels and changing the computation.

## Stop window open for engineering (user, 12:57Z): idle owners take the train-path items and the surveys with demand

No run lives on the cards: the 500M is stopped and the 200M is paused at step 832. The user's
word was that whoever is idle does the engineering work, and that surveys with a stated demand
are done too. So the stop-window list is released, under three conditions that keep the
performance sprint's numbers comparable:

| condition | reason |
|---|---|
| train-path edits are staged on the owner's branch with tests, merged in batches between card jobs, never while a trace or A/B is on a card | an A/B whose two arms ran under different sync stamps measures the merge, not the variable |
| every A/B row records the pod sync stamp it ran under | the stamp is the only identity of "the code that ran" |
| the step-832 interrupt checkpoint of p200m_4b_0902 must still load after every merge; `test_arch_compat`'s legacy round-trip is the gate | the resume is the next training job and a checkpoint that stops loading is a lost 832 steps |

| owner | engineering, off-card, now | survey with demand |
|---|---|---|
| de | trace (on 0-3), then the 300M A/B, then de-13 cursor, domain_loss CLI default + `APPLIES["sft"]`, de-20 cache knob | none new: de-26 eval matrix v2 |
| tilerl | tilerl-14 one writer per log, tilerl-15 resume total_steps, roofline table when the trace lands | kernel inventory: every GPU kernel class the trace names, its source (fla / torch / torchao / Triton), replacement candidate, effort, and the parity test; tilerl-10 |
| b0 | b0-10 offline checkpoint health read on the step-832 interrupt (CPU), facts-join re-report of the computation list | MLA layer count and latent at 200M, selective FFN recompute: what the literature measured, joined with `facts/efficiency.json` before any run is proposed |
| e1 | e1-23 then e1-16 required flags (train.py, staged on e1), e1-22 base-prompt audit, v11 diff | none new: e1-21 |
| 3b | 3b-8 near-dedup pass, code and tests on the sample now, full pass after de's trace; 3b-9 fetch and admission | task-set survey stays in 3b-9 |
| 44 | 44-20 launch-line check, reviews as they land | 44-17 post-pretrain plan |
| 98 | 98-1 per-owner queue on the progress page | — |

## No pod push while a card job runs (13:34Z)

44 pushed a9c5952 while arm 1 of de's 300M A/B was on cards 0-3 under stamp 7660c00. The
diff (AGENTS.md, the manifest, facts/efficiency.json, harness.py, trace_classes.py) lies
outside the profiler's import path, so the A/B stands, and its report carries each arm's
stamp. The rule that the launch-window freeze already implied: while any job holds a card,
nobody pushes unless every changed file is outside the set of files the job reads at runtime (code it imports, the mix and config it loads, the stamp and manifest, its shell wrapper) and the push is
named in the job's report; during a two-arm A/B nobody pushes at all.
