# Controller board (fb) — updated 2026-09-04T18:26Z, rewritten every tick

Percent = share of the deliverable landed on main and verified by a second reader; "quality" names what the reviewer opened, or the defect the owner caught in their own work.

## Memory layers program (user order 2026-09-05; charter docs/standards/memory_layers_0905.md, prereg #memory_layers_0905 + amendments 1-3)

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| Memory layer in model.py (shared pool 3/6/9, top-k 32, sparse values) | b0 | 85 | model.ProductKeyMemory on main (72bbad2b/09ff42a5): both _body paths, real Cfg fields + flags, frozen keys; train.py wiring 4d0319cf (keys/values out of Muon and FP8 by full fqn, own Adagrad group); probes/mem_toy.py | attn_res path silently omitted the memory (§177, now raises); three false-green assertions caught by mutation and replaced; 44 review e20ae279 re-applied a mutant, opened the legacy case | lane-5 smoke: step-30 tok/s, peak GiB at M1 and M3 shapes, pool touched; memory_diag call wired |
| Kernel/throughput + gradient exchange | tilerl | 70 | item 1: lookup 51.6 ms of 799 ms step, 77K tok/s/gpu = 93.9% of control, embedding_bag 1.38x over index_select; item 2: sparse exchange loses at every arm (1.5-2.3x dense; table 69-100% touched per step, n/M = 2 and 8) — RULED dense for M1/M2, M3 (n/M 0.5) measured both ways on the smoke; 47bfc95a | own prediction reversed and stated; two tooling defects (per-rank device, false DDP_DONE) found by running | item 3 on the pod: compile backend with the real module, FP8 exclusion verified by listing converted modules; embedding_bag diff to b0 |
| Readout 2 fact probe (seen-vs-unseen DiD, API-name cloze over code_py_starcoder) | e1 | 75 | 5,000 items written to /work/aupai/data/eval/api_cloze.jsonl (2,500 seen from 1,359 rows, 2,500 tail from 1,386 rows; ~150K candidates dropped for unequal option token length); score_mc_items with SE; contiguity proven from train.py:1783 + cursor-sum identity | own test caught two return-type bugs; integration-tree write self-corrected | pin provenance (cap 85,380, seed, N off the cache) into the file; commit; 3b reviews; control floor on lane 5 |
| Data parity + region boundary | 3b | 100 | 9/9 fps live == .srcfp == control ckpt; boundary measured; review row ff39ab70; zh_web supply confirmed exactly (9d3d16db) | two self-caught counter errors ([:16], eos convention) → §178 | review e1's item file |
| Prereg row + amendments | fb | 100 | row 30f3f29b; amendments bcc86d18, 87960318, 6fe808a5, wording 25e0ca58 | 44 review 8319002d: dates PASS, superseded-marker fixed, stale charter sha fixed | — |
| memory_diag ledger + schema, launch monitors with stop rules | de | 60 | scripts/memory_diag.py + data/ledger_schema.json + check memory_diag_fresh (95579a06): schema-validated append, run-log as independent freshness reader (300/301 boundary), 11 mutants | two blind mutants found and fixed | monitors: stop_rules() as WARN conditions, message to controller, never kill |
| score_matrix_present sees #cu rows | de | 100 | 00:56 commit | found by b0 on armB | — |
| Progress page + experiments doc section | 98 | 100 | 67b2ac04, 2040f009 | follows amended prereg, no unledgered numbers | curves when runs print |
| Card grant | fb | 100 | 77a9fb1f: block 1,2,4,6; lane 5 | — | write M3 grant (0+3) + lane 7 when card 3 reads 0 MiB (still 96 GB / 29% at this tick) |

## Global items carried from 2026-09-04

| item | owner | % | state | next |
|---|---|---|---|---|
| CI green | de | 100 | four consecutive greens (bb876bd7, 27b9ea15, bd82cad6, 95579a06) | 25e0ca58 and 10acde74 green; 6fe808a5 in progress; four reds fixed (launch_tests, score_matrix_failpath, holdout regenerate, pod_drift world, free-card fixture) | CI-red WARN 2 h / FAIL 8 h check still owed |
| Harness: pod-reaching checks time out when tunnel down | de | 60 | UnboundLocalError in strike handling fixed (8319002d); reachability SKIP written, not committed | deadlines raised (stopgap); pod_stamp_is_main → auth=pod with reachability SKIP written, not committed | commit with the dead-host broken world |
| Harness package split (core/checks/cmds/infra) | de | 0 | design accepted | plan doc after the memory program's monitors |
| Funnel table (T0-T3 per rule) + gates_inventory.tsv | de | 0 | tiers ruled | table |
| temp-world leak (12 dirs / 7.5 MB per harness selftest; 22 dirs / 12 MB per test_exp_done_started) | de | 100 | scripts/tmpworld.py process-level redirect + rmtree at exit, four call sites, 4 → 0 leaked through the hook (bd82cad6); 349 MB freed | mutation-tested 4 ways; test_attest_cases.py still leaks by hand (not in SELFTEST_FILES) |
| Infra decoupling (aupai-infra) | tilerl | 40 | inventory + contract 326831f1; wrappers vendored 1ab666f6 with cd-into-background refusal; two inventory misfiles returned | subtree cut after the memory program's kernel work |
| Prose rules → layers (user direction 2026-09-05: less control, brief before the work) | 44+de | 50 | 31 manual rules reviewed against 5 sources: KEEP 15 / BRIEF 14 / MOVE 2 / DELETE 0 (ec64087f, 3d11225f); rule→kind table handed to de; `harness brief <kind>` + two L1 gates (fixture-only checks; owned output dirs) + `harness ruling` queued at de behind the monitors | Shared-files T0 (3b, file_claim + owner scoping ed3eba36), Language T3 deleted, rule 1 (e1), rule 2 T1 (e1 38af3d47), rule 3 (e1, unblocked, after the probe) | rule 3 train.py ETA |
| gate_failure_shapes restructure + 33 folded incidents | 44 | 100 | 141 = 59 + 82; 33/33 located, §162 re-cited, file::function rule; §175 §176 landed | §177 (attn_res) on b0's write-up |
| Friction summary + fixes | 44 | 100 | 2fa5c710; post-merge hook warn-only; minutes_required check | daily |
| Credential in transcripts (ANTHROPIC_AUTH_TOKEN) | fb | 100 | 83 occurrences in 6 transcripts + 2 zsh_history lines redacted; repo/pod clean across 18 branch tips and 4149 commits | user rotates the relay token; GitHub secret scanning is off on the repo (user setting) |
| Laptop disk | user | 100 | 5 GB → 112 GB free | /private/tmp/e1_v11_snapshot (2.8 GB, 8 projects' transcripts) left in place per user |
| Head-hybrid A/B | b0 | 100 | B loses 0.087 nat; closes + prereg outcome 15de1e0e; checkpoints pinned | user: keep or prune both arm checkpoints |

## Open user decisions (unchanged)
/data01 backup refresh schedule; code_tests fetch placement on /data02; head-hybrid and Stage E checkpoints keep/prune; e1_v11_snapshot.
