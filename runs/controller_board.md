# Controller board (fb) — 2026-09-11 18:3xZ

Goal: HumanEval pass@1 >= 30% on the V4.1 gate run (`runs/prereg.jsonl#v41_gate_0911@amended_7`).
Everything below is ordered by distance from that number. Task ids are `runs/tasks.jsonl` rows.

## P0 — launch critical path: DONE, gate is up 15:11Z

v41_gate_0911 step 10/38146 at 15:11Z, loss 7.36, peak 43.4 GiB, ranks 0-5 at 69.5 GB, stamp 544532a6, launcher sha 46f7d105. 66 GREEN on all five enforcement items at 14:5xZ; go given 15:0xZ. L3 cache 26.697B (vocab f1f86097, srcfp 63a3b0e6). Lost ~4h to the tunnel outage (friction row 3028e6ae) and ~15 min to the old-driver root path.

Historical P0 table (all rows closed):

| step | owner | task | state | acceptance | ETA |
|---|---|---|---|---|---|
| L3 static conversion 147 shards | 0e | 0e-3 | 90/147 SHARD_OK at 09:41Z, 30 parents, 2.4 shard/min | conservation assert per shard, aggregate stamp with fingerprint | ~10:20Z |
| L3 cache at gate vocab | 98 | 98-5 | waiting on stamp | tokens_code_ultra_l3_noexec_dc.pt stamped f1f86097 + srcfp + seed; count within 10% of ~26.8B | ~11:05Z |
| enforcement-path GREEN | 66 | 66-8 (report) | 7/8 green at stamp d17db480 | pod_drift rc 0, 8/8 shards+stats, 8/8 caches fresh, build_only, OOM ladder | ~11:10Z |
| cards to block 0-5 | fb | fb-8 | file at block 0-3 / lane 5 | card_assignment.json on pod: block 0-5, lane null; cards 6,7 tilerl | at go |
| launch | 66 | 66-9 | script on pod md5 f619eed7, `--csa2_win_flash` | first step line; peak <= 80 GiB; tok/s >= 15K; no NaN to step 100 | ~11:15Z |

Waived (amendment 7): launch_gate.py's two inline-mix advisories. Retired by user order: L3 sandbox exec.

## Run state 19:0xZ and the switch plan

- v41_gate_0911 step 2710/38146 at 18:5xZ, loss 1.403, val 2.444/2.208/2.108 at 500/1000/2000, 28K tok/s/gpu, peak 43.4 GiB, 0 NaN. .step4000 save ~20:32Z, .step6000 ~23:00Z. Disk 73%.
- HumanEval .step2000 with the corrected instrument (#278, controls 154/154 self-redeclare PASS, wrong body FAIL): pass@1 0/164, empty 51/164 (eos_first 21, stop_at_0 30), 113 bodies scored and all fail. This is the real number at 1.57B tokens; the earlier 0/164 with 143 empty was the instrument. Runtime 1578 s on card 6 (24 min, full-function decode). Next: .step4000 at its save on card 6 (66), decision at .step6000 per amended_8.
- USER RULING: formal run on 8 cards. Path C: at the .step6000 save stop the w6 job by PID, resume the same run at world 8 accum 6 (786,432 tok/step, LR schedule unchanged, total_steps 38146 kept). #275 launcher merged a8d367d3 (de second read: six points hold at 72c5a34; #277 locks the W6->W8 cursor re-stripe in CI). Launch only after ckpt_v41_gate_0911.pt.step6000 exists. Prereg amendment 9 records the switch.
- Eval card after the switch: world 8 leaves no card, and world 7 cannot keep 786,432 tok/step (192 seqs/step is not divisible by 7; accum 7 gives 917,504 tok/step and a recomputed schedule), so option A as stated is not schedule-neutral. Options: B = world 8 + CPU HumanEval fallback in model.py for win_flash checkpoints (de; 3.4 h per eval at 6000-step spacing); D = world 8 and no in-run HumanEval after step 6000 until the final checkpoint on a card handed back at run end. fb recommends B. User decision pending; default at 23:00Z is B with de owning the CPU path.
- Cards: 6 lent to aupai for eval windows through 23:59Z (grant string note); tilerl-a3 gives 7 outright at 22:45Z (fidelity + two sparse points + MMLU pair run until then); 6 outright after 66's evals. Grant file rewrite to block 0-7 at the switch.
- Pod stamp: cleared twice today by partial pushes with drift (pod_push.sh:133 by design); restored with --all at 208e0f56 then a8d367d3. run_ddp.sh:15 refuses a launch without it, so the stamp is a launch precondition at 23:00Z.
- Merged today: #260 #266 #268 #269 #270 #271 #273 #274 #275 #276 #277 #278. Open: #272 (98 cache facts, 3b merges).

## P1 — during the run (2-3 days, 38.1K steps)

| owner | task | acceptance |
|---|---|---|
| 66 | 66-10 HumanEval per 2000-step checkpoint | pass@1 rows in facts/v41.json; plateau < 10% by step 20000 is the early-stop signal |
| 3b | 3b-21 merge #260 (de-108, default off) and #266 (docs) after "gate is up"; 5/5 docs review rows | shas on main, pod stamp advanced |
| fb | hourly progress to user; deletion of six pre-_dc originals + 7 old caches (~150G) after first ckpt | named list, DELETED lines in runs/deletion_0911b.txt |
| 66 | 66-7 launch_gate reads per-dir stamps | all-GO with 0 waived on the gate mix |
| ae | ae-6 inline fingerprints in mix json | same |
| de | de-108 check_default_identical --ref + de-109-flag step-cost breakdown | per-component share of step at gate shape |
| 66 | 66-12 (ex b0-36) score rows carry cache srcfp | 100% of rows |

## P2 — after the run

| owner | tasks |
|---|---|
| fb | fb-8 end half: notify agent-infer-be and tilerl-58; cards 6,7 revert |
| de | de-102 REVIEW_PAIRS from roster; de-66/71/74 train.py hygiene; de-78/80/81/83/85/86/99 harness+merge_main; de-109 (ex b0-28), de-110..113 (ex b0-29/30/31, e1-54) |
| 66 | 66-8 six global FAILs triage (zh_web/Cfg.mix fix is de's); 66-11 (ex b0-32) grant expiry |
| 3b | reviewer of record for every P1/P2 PR |
| de | CED topology (de-105, dropped until the gate number exists) |

## Ledger reconciliation this tick

Closed as delivered: de-106, de-107, 66-1, 66-2, fb-7. Dropped as retired by the pivot or owner departed
(b0/e1/44): 22 rows. Reassigned from departed owners to live members as new rows: 7. Open rows went 49 -> 29,
all owned by live roster members.

## Open user decisions

Eval card after the world-8 switch: B (CPU fallback, fb recommends, default at 23:00Z) or D (no in-run HumanEval after step 6000). Standing: Standing: tileRL keeps 6,7 through the run; L3 exec dropped; 390G reference weights kept.
