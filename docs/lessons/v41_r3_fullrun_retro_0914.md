---
question: v41_r3 0914 全量 30B 跑到退火末段的实测底表：val 曲线、HE/MBPP 轨迹与 gate 读数、E0 n=10、非代码 eval、吞吐、事件时间线是什么？
status: measured
source: 98 于 2026-09-14/15 从 pod 日志/产物读数（runs/v41_r3_0914.log、data/eval/e0_*_merged.n10temp0.2.jsonl、runs/noncode_eval_psft.json、runs/score_matrix.jsonl）
---

# v41_r3_0914 full-run retrospective base table

> 幸存底表（98 归档 2026-09-18）：r3 final checkpoint 与 pod 上原始日志/预测产物随 sglang-test pod 于
> 2026-09-16 12:44Z 被撤销而全部丢失（emptyDir 未持久化，见 runs/deletion 与 pod 销毁记录）。
> runs/retro.jsonl 无 r3 行；本文件是该 30B 全量跑唯一留存的实测事实表，逐数为当日同分钟读数。
> ckpt sha256 f76ddeb9…（表内）供未来产物对指纹；凡仅存于已毁 pod 的源，引用时按"无原始可复算"对待。

## 1. Val curve (every 500 steps; runs/v41_r3_0914.log)

| step | val | | step | val | | step | val |
|---|---|---|---|---|---|---|---|
| 500 | 2.640 | | 9500 | 2.117 | | 18500 | 1.990 |
| 1000 | 2.427 | | 10000 | 2.111 | | 19000 | 1.974 |
| 1500 | 2.347 | | 10500 | 2.105 | | 19500 | 1.961 |
| 2000 | 2.290 | | 11000 | 2.098 | | 20000 | 1.954 |
| 2500 | 2.253 | | 11500 | 2.123 | | 20500 | 1.934 |
| 3000 | 2.222 | | 12000 | 2.162 | | 21000 | 1.922 |
| 3500 | 2.200 | | 12500 | 2.089 | | 21500 | 1.915 |
| 4000 | 2.195 | | 13000 | 2.067 | | 22000 | 1.902 |
| 4500 | 2.176 | | 13500 | 2.084 | | 22500 | 1.884 |
| 5000 | 2.162 | | 14000 | 2.065 | | 23000 | 1.880 |
| 5500 | 2.165 | | 14500 | 2.045 | | 23500 | 1.868 |
| 6000 | 2.160 | | 15000 | 2.046 | | 24000 | 1.852 |
| 6500 | 2.147 | | 15500 | 2.031 | | 24500 | 1.839 |
| 7000 | 2.158 | | 16000 | 2.035 | | 25000 | 1.829 |
| 7500 | 2.141 | | 16500 | 2.024 | | 25500 | 1.816 |
| 8000 | 2.136 | | 17000 | 2.013 | | 26000 | 1.802 |
| 8500 | 2.141 | | 17500 | 2.000 | | ... | ... |
| 9000 | 2.129 | | 18000 | 1.998 | | 34000 | 1.683 |
| — | — | | — | — | | 34500 | 1.682 (anneal start 34264) |
| — | — | | — | — | | 35000 | 1.680 |
| — | — | | — | — | | 35500 | 1.679 |
| — | — | | — | — | | 36000 | 1.679 |
| — | — | | — | — | | 36500 | 1.678 |
| — | — | | — | — | | 37000 | 1.678 |
| — | — | | — | — | | 37500 | 1.677 |

Schedule markers: lr-warmdown start 13325 (val around 2.06); anneal start 34264 (val 1.683@34000 pre-entry, 1.682@34500, 1.680@35000 — −0.003 across the first anneal quarter, within noise; anneal benefit, if any, is expected in the second half: 36000/37000/final).
Min as of 35000: 1.680. One non-monotonic window: 2.098@11000 -> 2.162@12000 (HE/save window, one-off; recovered 2.089@12500).

## 2. HumanEval gate trajectory (rstrip, 8 contaminated problems removed → /156) + bpb

| ckpt | FULL rstrip /164 | CLEAN /156 | gold_bpb per-task-mean | gold_bpb byte-weighted |
|---|---|---|---|---|
| step6000 | 15/164 9.15% | 15/156 9.62% | 0.6052 | 0.4869 |
| step12000 | 11/164 6.71% | 11/156 7.05% | 0.6377 | 0.4995 |
| step13000 | 12/164 7.32% | 12/156 7.69% | 0.5778 | 0.4645 |
| step18000 | 18/164 10.98% | 18/156 11.54% | 0.5599 | 0.4475 |
| step24000 | 25/164 15.24% | 25/156 16.03% | 0.4920 | 0.3929 |
| step36000 (anneal mid) | 34/164 20.73% | 34/156 21.79% | 0.4735 | 0.3708 |
| step37000 (anneal late) | 28/164 17.07% | 28/156 17.95% | 0.4697 | 0.3705 |

Resolution note: greedy /156 ≈ ±3 problems; 36k→37k CLEAN −6 problems (21.79→17.95), a 2σ greedy-sample dip — bpb was flat (0.4735→0.4697 per-task), so likelihood did not regress; the final 38070 gate reading is n=10 GPU (66) for exactly this noise. Earlier: 24k→36k +9 problems across the anneal first half. None of the r3 passes land in the 8-problem contamination union (19/66/71/78/105/123/129/156), so FULL numerator = CLEAN numerator at all seven measured points. Anneal half moved HumanEval up +9 then the next greedy draw gave 6 back while aggregate val stayed flat (1.680@35000 → 1.677@37500): single-sample generation is the noisy channel here.
Standard column is OOD (prompt ends in lone Ċ) and not the gate; 24k standard 1/164 0.61%, 1/156 0.64%, kept for history only.

Points pending: final step38070 (GPU, n=10 per 66); last val 1.677@37500, final val at 38000/38070.

E0 final n=10 temp0.2 (8-card shards, merged data/eval/e0_he_merged.n10temp0.2.jsonl; prereg textbook_continuation_ab_0914 metric = mean c_i/10):
- HE metric of record: CLEAN **0.1872** (292/1560), FULL 0.1787 (293/1640) — the 30% gate receives 18.72, below gate.
- Auxiliary coverage c_i>=1 (pass@10@0.2-like, NOT pass@1): CLEAN 48/156 = 0.3077 (FULL 49/164 0.2988). Bimodal: 115 tasks c=0, 15 c=10, 34 in between; 0 empty; HumanEval/66 passes only into FULL.
- Flips vs 37k greedy: rescued 21, dropped 0; vs 36k greedy: rescued 17, dropped 2 (HumanEval/1,/38). No task went all-10 from a greedy fail.
- Reading (fb): coverage ~31% but one-shot reliability 18.7% — the gap is stability, the SFT target space.

E0 MBPP n=10 (merged data/eval/e0_mbpp_merged.n10temp0.2.jsonl, 06:00Z):
- Metric of record: CLEAN **0.2689** (909/3380), FULL 0.2623 (1120/4270).
- Coverage c_i>=1: CLEAN 136/338 = 0.4024, FULL 170/427 = 0.3981. 45 empty samples (1.1%), one all-empty task. Bimodal: 257 c=0 / 58 c=10, 112 in between.
- 34 passing tasks fall in the 89-task contam exclusion.
- Next GPU job (06:36Z, 8×77.5 GiB/99%): sft_math.py --resume ckpt_v41_r3_0914.pt, sft_phi_codeexercises_v42_65m_0914, epochs 6 lr_scale 0.1, out ckpt_v42_phisft_n6_*.pt — the SFT stage on the r3 final.

## Appendix B — non-code eval, r3 final vs psft (2026-09-15, runs/noncode_eval_psft.json)

Same read-only drivers, identical screened subsets and order; n identical per benchmark.

| metric | n | r3 | psft | Δ psft−r3 |
|---|---|---|---|---|
| MMLU 4-choice LL | 13564 | 0.2334 | 0.2370 | +0.0035 |
| ARC-Easy LL | 2221 | 0.3908 | 0.3588 | −0.0320 |
| ARC-Challenge | — | unmeasured (no local screened data, pod offline) |
| Lambada EN acc1 | 5153 | 0.2193 | 0.1923 | −0.0270 (~5σ, ci95hw ~0.011) |
| Lambada ZH acc1 | 1000 | 0.0 | 0.0 | 0 (acc5 0→0.001) |
| Lambada ZH 2-way | 1000 | 0.6482 | 0.6305 | −0.0177 |
| math_v2_like 2-way | 3134 | 0.9799 | 0.9742 | −0.0057 (near ceiling) |

MMLU 13564 = 14042 physical rows − 478 contam_mmlu_r3 union row ids; ARC-E 2221 = local data/arc_easy.jsonl 2241 − 20 contam row ids (HF cache is a different 2376-row population). eval/mmlu.py itself has two bugs (splitlines on embedded-newline questions; never applies the 478 exclusion) — bypassed read-only in the driver; a fix PR is queued post-round. Reading: narrow code SFT holds MMLU within noise but costs ~2-3pt on every other general language/reasoning channel while the code gate stayed below 30 — evidence against repeating this narrow SFT.

## Appendix C — harness score_matrix panel, same two ckpts (2026-09-15, runs/score_matrix.jsonl)

Different metric sets by ckpt type; this is the harness panel, NOT the gate (gate = E0 n10 HE 0.1872 / MBPP 0.2689).

r3 (base full panel): domain_loss 1.5817; domain_bpb 0.6104 across 6 domains (per-domain bpb: l3_stub 0.176, keep_p1 0.556, l2 0.480, math_owm 0.806, en_c4 1.189, cot 0.456); minimal_pairs 0.4713 (261 pairs, margin -0.239); C-Eval 22.0; LambEN 0.2193; LambZH 2-way 0.6482; math_v2 0.9799 (3134); HE gold bpb 0.4750/0.3725.

psft (SFT-type panel): domain_loss **1.8879** (worse on the same mix, +0.306 — corroborates narrow-SFT general-likelihood cost); minimal_pairs 0.5134 (margin -0.692); mc_full C-Eval 21.0 / ARC-Easy 39.3 (UNCLEANED internal set — not comparable to Appendix B's 35.88 on the 2221 screened set) / Average 30.1.

Tool-missing, recorded unmeasured NOT zero: math_hard ERROR (no preds file hard_...phisft...0.jsonl), pass_at_k ERROR (ArtifactExists guard; n=1 pass@k is not the gate) — no rerun per fb. code_500/code_500_v2/math_500 report 0 but that is the known ChatML-SFT generation-form artifact, not a scored zero. One card (GPU0) used, released after.

## 3. MBPP (sig-rstrip): FULL /427 and CLEAN /338 (r3_mbpp_clean union, runs/contam_r3_mbpp_union.json)

| point | FULL /427 | CLEAN /338 | empty (FULL) | source preds |
|---|---|---|---|---|
| run1 step12000 | 46/427 10.77% | 40/338 11.83% | 11 | v41gate_0911 …v41gate_mbpp12k_sigrstrip |
| r3 step12000 | 37/427 8.67% | 31/338 9.17% | 17 | v41r3 …v41r3_mbpp12k_sigrstrip |
| r3 step18000 | 51/427 11.94% | 39/338 11.54% | 9 | v41r3 …v41r3_mbpp18k_sigrstrip |
| r3 step18000 max-1024b | STOPPED by fb at 15:21Z, no result (491-byte log) | — | — | v41r3_mbpp18k_max1024b_cpu |

The earlier max1024 attempt (v41r3_mbpp18k_max1024_cpu, 25 rows then killed) and the `_b` attempt both end without an artifact; neither number enters the table. Job-running status is pgrep evidence only, not log presence (fb correction 16:5xZ).

CLEAN denominator 338 = 427 − 89 problems in the r3 domains' 13-gram union (union fp 0aefe6a2, full scan 2026-09-14). Rescored 2026-09-14 from preds 'ok' field.

## 4. Throughput stability (parsed 2,639 step lines)

- tok/s/gpu: median 26K in all three 10k-step segments (1-10000 26, 10001-20000 26, 20001+ 26). Slow intervals coincide with CPU HE/MBPP eval + checkpoint/val windows (19-27K readings), not training drift.
- MFU median 18%; peak mem median 43.44 GiB, range 43.40-43.59 — flat across the run.

## 5. Event timeline (UTC)

- 09-13 07:54: v41_r3_0914 launched world 8 (step 210 healthy at 11:0x local).
- 09-13 ~15:3x: step6000 HE 0/164 standard; later rstrip shows 15/156.
- 09-13 20:5x: step12000 HE standard 0/164 (standard column later found OOD; rstrip 11/156).
- 09-13 23:1x: laptop reboot #1 (~11:12); pod training unaffected; /tmp pipeline lost, moved to ~/aupai-textgen.
- 09-14 00:3x: gate column switched to rstrip CLEAN /156 (8 contaminated problems excluded).
- 09-14 03-06: 13k/18k rstrip evals; 18k 18/156 = first material generative signal.
- 09-14 13:57 local: laptop reboot #2 (180 GB memory; 3b mutation_gate orphans/RLIMIT); safe_exec pgroup+2GB fix; option B halts textbook generation.
- checkpoint pins: he6k/he12k/he13k/he18k milestones + step10000 + valmin_step11000; he24k symlink pending (66 script).
- stop rules not triggered: plateau rule was the run1 shape; r3 continued by user decision (phi-1); val kept declining; no NaN.
- 09-14 09:36Z: read-only anneal watcher pod pid 3634293 armed (triggers at 34264).
- 09-14 15:19-15:21Z: HE24k chain finished — rstrip CLEAN 25/156 16.03%, bpb 0.4920/0.3929; 66's waiter touched runs/HE24K_ALL_DONE.flag and MBPP max1024b briefly started, then fb stopped it at 15:21Z (no result).
- 09-14 14:55Z: 98 final-ckpt waiter pod pid 3785389 armed (runs/r3_final_wait.sh, waits the no-suffix ckpt_v41_r3_0914.pt, logs FINAL_READY + sha256 to runs/r3_final_wait.log).
- 09-14 20:48Z: anneal segment entered (step 34264).
- 09-14 20:50-20:59Z: four jump ALERTs, all ruled benign (fb; de confirmed process health): step34300 +0.991 / 34310 −1.029 / 34420 +0.792 / 34440 −0.727. Raw phase tags show ALL FOUR pairs were same-phase [anneal]-[anneal] alternating domain batches (step34280 [anneal] 0.377 → 34290/34300 [anneal] 1.368; losses hop 0.34↔1.58 routinely, gnorm flat 0.13-0.16, zero non-finite/rollback). The real main→anneal pair (34260 [main] 1.573 → 34270 [anneal] 0.397, −1.176) never alerted because 34270 is the window's first point with prev_loss unset. PR #355's phase filter therefore did NOT fix these four: the 0.5 fixed-threshold-on-adjacent-batches rule still false-positives on within-phase batch alternation inside the 200-step head window; annotated in runs/r3_anneal_watch.log with correction, CRASH-grep count 0. Follow-up if it recurs: phase-aware baseline instead of 0.5 — registered as issue #356 (fix before next training, grouped with #354). Pod watcher pid 3843446 from 21:38Z (see Appendix A).

## Appendix A — anneal watcher verification (pre-anneal, 2026-09-14 19:47Z)

- Live watcher: /work/aupai/runs/r3_anneal_watch.py, sha256 `e110a2a4db013f0f2de04421269e5443564f3fcacb40bfe1b7570b0f17b74c4c` (mtime 09:34Z), sole pod copy, run by pid 3634293 since 09:36Z.
- fb suspected the "running minimum" was set once and never updated. Refuted against the running file: the not-above branch executes `ann_min=min(ann_min,val)` and clears `last_above`; two consecutive points above the current min are required for an `anneal val worsening` ALERT.
- Pod selftest with a synthetic anneal segment 1.7→1.6→1.65→1.7→1.5→1.55→1.45→1.46→1.47 (60 s poll shortened to 1 s): exactly two ALERTs, at the 1.65/1.7 pair (min 1.6) and the 1.46/1.47 pair (min 1.45); single-point bounces do not alert; a new min resets state. No PR, no restart (decision: zero-value change at the anneal head).
- Crash channel is separate: a local persistent Monitor pages on NaN / anneal-head loss-jump ALERT lines and on watcher pid 3634293 disappearing; worsening lines stay trend-only in the 2 h poll. The monitor exits when the no-suffix final ckpt exists.
- Superseded 21:38Z after the step-34300 false pages: PR #355 (merged 8995b88) moved the canonical watcher to scripts/r3_anneal_watch.py, sha256 `e9865d7b16e499996817207bb8fcdb64809a189f317f3210f07b53e0cfcc597c`. New live pid 3843446 (PPID 1, scripts/ copy, no runs/ copy). Changes: jump compares same-phase adjacent lines only; crash signals match train.py's real runlog literals ("non-finite grad — step skipped" WARN, ≥10 per rolling 500-step window CRASH; "20 skips in a row — rolled back to snapshot" CRASH) plus gnorm >5x rolling median and a lowercase-nan fallback; val worsening gained a 0.005 epsilon; startup rebuilds history. 11-case --selftest, rc 0 on the pod. The local crash relay pages CRASH-class lines and pid 3843446 death only.

## Pending last rows (fill after final + E0)

- step 34264-38070 anneal val (watcher), final val, final ckpt step/sha. → DONE 00:52Z 2026-09-15: end-of-epoch line `ep 1/1 train 1.582 val 1.609 147194s` (the 1.609 is the FULL end-of-epoch val set, not the 500-step 4 s sample series; never plot the two on one axis); final ckpt_v41_r3_0914.pt size 12,906,895,957, sha256 `f76ddeb9f9607ba9b9280501f73b8054663fcbc2231cf2404b12d1bb2ba49b2f`. Watcher pid 3843446 reaped by fb at 02:4xZ (Zs, init-collected, no children; cards 0 MiB).
- HE final step38070 n=10 GPU (66) — the statistical gate reading.
- MBPP CLEAN rescoring.
- POST-FINAL GAP (02:41Z): the wrapper exited rc=1 — post-train score_matrix waited for a lane card 30 min and died (`FATAL: no free lane card in 30min ... NO metrics`), so the final ckpt carries NO automatic metrics despite 8 cards being 0 MiB/0% afterward. Manual re-score required: `CUDA_VISIBLE_DEVICES=<lane> python eval/score_matrix.py --ckpt ckpt_v41_r3_0914.pt --json runs/score_matrix.jsonl`. Watcher pid 3843446 was still alive at 02:41Z (fb reaper owns it). stage2_timing_calib.sh runs on CPU (log 0 bytes since 00:55Z).
- Stage-2 decision numbers (textbook_claude_v41_dc: 1,938 rows / 8.12M sum_n at 09:52Z rebuild, decon_fp 0aefe6a2; manifest net raw 8,454,300 tok at 14:51Z).
