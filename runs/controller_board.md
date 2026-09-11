# Controller board (fb) — 2026-09-11 09:5xZ

Goal: HumanEval pass@1 >= 30% on the V4.1 gate run (`runs/prereg.jsonl#v41_gate_0911@amended_7`).
Everything below is ordered by distance from that number. Task ids are `runs/tasks.jsonl` rows.

## P0 — launch critical path (blocks the number)

| step | owner | task | state | acceptance | ETA |
|---|---|---|---|---|---|
| L3 static conversion 147 shards | 0e | 0e-3 | 90/147 SHARD_OK at 09:41Z, 30 parents, 2.4 shard/min | conservation assert per shard, aggregate stamp with fingerprint | ~10:20Z |
| L3 cache at gate vocab | 98 | 98-5 | waiting on stamp | tokens_code_ultra_l3_noexec_dc.pt stamped f1f86097 + srcfp + seed; count within 10% of ~26.8B | ~11:05Z |
| enforcement-path GREEN | 66 | 66-8 (report) | 7/8 green at stamp d17db480 | pod_drift rc 0, 8/8 shards+stats, 8/8 caches fresh, build_only, OOM ladder | ~11:10Z |
| cards to block 0-5 | fb | fb-8 | file at block 0-3 / lane 5 | card_assignment.json on pod: block 0-5, lane null; cards 6,7 tilerl | at go |
| launch | 66 | 66-9 | script on pod md5 f619eed7, `--csa2_win_flash` | first step line; peak <= 80 GiB; tok/s >= 15K; no NaN to step 100 | ~11:15Z |

Waived (amendment 7): launch_gate.py's two inline-mix advisories. Retired by user order: L3 sandbox exec.

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

None pending. Standing: tileRL keeps 6,7 through the run; L3 exec dropped; 390G reference weights kept.
