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

## Run state 2026-09-12 02:2xZ: world-8 continuation up

- v41_gate_0911 resumed at step 6000 on world 8 accum 6 at 02:1xZ (torchrun pid 3078695 in the container, claim 0-1-2-3-4-5-6-7): step 6060 loss 1.315, 24K tok/s/gpu and climbing, peak 43.40 GiB, 0 NaN, lr 1e-2, warmdown starts at step 13352. Launcher runs/v41_gate_0911_resume_w8.sh at f5e9851f (#275 + #284), pod stamp d5643fd3/a44bed84 == main.
- HumanEval trajectory (corrected instrument #278): step2000 0/164 (113 judged), step4000 0/164 (118), step6000 0/164 (118 judged, CPU path #280, 6602 s). Facts #279 #283 #287. Val 2.444 (500) -> 1.950 (6000); amended_8 stop rule needs 0/164 AND val not below smoke-g 2.829, so the run continues on the val arm alone. Next HumanEval on CPU at step 12000, then every 6000; 66-10 plateau rule (<10% by 20000) stands.
- Switch cost: the first world-8 launch (23:27Z) was killed by the explicit --gate-timeout 300 in the launcher (cold load of the 228 GiB cache takes ~5 min); the fix #284 needed two CI reds cleared first (card 6 lend expired 23:59Z -> re-extended to 09-12; harness selftest pinned card 7 to tileRL -> #286). Cards idle 23:24Z-02:1xZ, ~2.8 h lost. Friction rows: gate-timeout (66), controller override (fb).
- Log truncation: harness launch opens runs/<name>.log for writing, so the 23:27Z launch erased the world-6 log (steps 0-6000). Recovered val lines are runs/v41_gate_0911_w6_val_excerpt.txt (#288); the failed-attempt log is runs/v41_gate_0911.log.w8attempt1_2327Z on the pod. de owns the rotation fix (PR pending).
- Cards: block 0-7, no lane; tileRL off all H20s (tilerl-a3 22:3xZ). Card 6 is a string lend re-extended daily (theirs_baseline pin [0,6]) until de's pin PR lands; card 7 {owner: aupai}.
- Merged since 19:0xZ: #279 #280 #281 #283 #284 #286 #287 #288. Open: de pin PR, de log-rotation PR, de-102.

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

Eval card: closed, B in effect (#280 CPU path on the pod). None pending. Standing: Standing: tileRL keeps 6,7 through the run; L3 exec dropped; 390G reference weights kept.

## Run state 2026-09-12 05:5xZ: step 9200, val 1.892@9000

- Run: step 9200/38146, 26K tok/s/gpu steady, peak 43.4 GiB, ETA 31.3 h; val 1.911@8000 1.897@8500 1.892@9000; .step8000 saved 04:27Z. Next save .step10000; next HumanEval (both columns, CPU) at .step12000 ~08:5xZ.
- HumanEval no-doctest arm, full 164 on he6k: 2/164 (HumanEval/23, /60) vs standard 0/164; empty 62/164; Fisher ~0.25. Ruling: second column from step12000, gate number stays the standard column (66 writes the fact after #290).
- Merged by fb this tick: #285 (theirs_baseline [], card 6 aupai outright; daily lend re-extension ENDS), #289 (harness launch rotates runs/<name>.log), #282 (review pairs from roster.json). Pod stamp dbf721e2.
- Chinese block: zh_wiki_dc 0.271B + zh_c4_dc 0.501B = 0.771B (2.57%), 0 HE/MBPP hits, held for a post-gate phase; gate mix frozen (ruling to ae/0e). PR #291 to 3b.
- 3b-22 SFT pack built: 39.37M tokens, 9610 rows, 72/18/10 code/en/zh; PR #293 blocked on the no-comment rule, approve on resubmit.
- Open: user's named deletion go (runs/deletion_0912_candidates.txt); 3b merges #290 #291 #292 #293.

## Run state 2026-09-12 11:0xZ: step 14230, step12000 HumanEval 0/164 both columns

- Run: step 14230/38146, 27K tok/s/gpu, peak 43.4 GiB, no NaN; warmdown since 13352; val 1.840@12000 1.835@12500 1.830@13000 1.820@13500 1.818@14000; .step14000 saved 10:52Z.
- HumanEval on he12k pin (CPU, controls green): standard 0/164 (empty 62, repetitive 10/102); no-doctest 0/164 (was 2/164 at step6000; empty 80). Trajectory standard 0/0/0/0 at 2000/4000/6000/12000. The step6000 no-doctest 2/164 reads as noise at n=164.
- amended_8 joint stop stays off (val far below 2.829). Plateau rule (<10% by step 20000, ~17:1xZ) is the next controller call; options drafted for the user: A stop at 20000 and SFT; B continue to 30B per prereg; C pause at 20000, 3 h SFT probe on one card (runs/v41_sft_0913.sh, by-name ChatML read), resume or stop on its number.
- Merged this tick by fb: #293 (SFT pack builder), #296 (SFT launcher + prereg v41_sft_0913 + --chatml by-name arm + --check_pack), #297 (run-end sequence runs/v41_gate_0911_end.md); 3b merged #290 #291 #292 #294 #295. Pod stamp 3e0511cd. 3b-21/22/23/24 closed.

## 2026-09-12 14:3xZ: v41_gate_0911 STOPPED by user order; cards to tileRL

- Stopped at step 17411/38146 (13.7B tokens) by SIGTERM to torchrun 3078695; .interrupt.step17411 saved 14:28Z; last periodic .step16000; nvidia-smi 0 MiB on 0-7 after. exp row reclassified stopped (reason logged); prereg amendment 10.
- Why: HumanEval 0/164 in both columns through step12000; the 164 step12000 generations show 62 empty (23 EOS-first, 39 stop-first), 101/102 non-empty re-declare the function, 73 reach a body and the bodies are wrong. Read as the L3 problem-to-solution shape (30% of mix, 96% of L3 rows) teaching "write a new def", not "continue this body".
- Cards: USER ORDER all eight to tileRL (runs/card_assignment.json 27cd6831 on main and pod); tileRL took 0,1,2 (+3 reserved) at 14:4xZ. aupai gives a one-hour notice before its next launch.
- Next round plan (user agreed in principle): reshape code data -- L3 as stub+docstring -> body continuation or weight 30% -> 10-15%; FIM in the packer; doctest slice generated from L3 solutions; docstring-without-body filter; CoT 1.5% -> ~5%. SFT probe (#296 launcher) on .step16000 when a card is available. Owners assigned next tick.
- USER ORDER: 0e is the standing disk scavenger (清道夫): tmp local+pod, deletion_0912 groups A-D, daily sweep, manifest per deletion.
- Dispatched 14:5xZ for round 2 (no card, no launch): ae-8 data/mix_v41_r2.json draft (L3 30->10-15%, CoT 5%, placeholders l3_stub/doctest_gen, anneal restated) -> de reviews; 0e-8 code_ultra_l3_stub_dc (stub+docstring->body, drop docstring-without-body rows) -> 3b; de-114 FIM in the packer (sentinel-slot question first), de-115 doctest generator slice -> fb; 3b-25 SFT pack v2 with 10% CoT slice -> fb; SFT probe on .step16000 waits on a tileRL card + user go; 66 run-summary fact; 98 page shows stopped.
