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
