# Controller board (fb) — updated 2026-09-04T17:23Z, rewritten every tick

Percent = share of the deliverable landed on main and verified by a second reader; "quality" names what the reviewer opened, or the defect the owner caught in their own work.

## Memory layers program (user order 2026-09-05; charter docs/standards/memory_layers_0905.md, prereg #memory_layers_0905 + amendments 1-3)

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| Memory layer in model.py (shared pool 3/6/9, top-k 32, sparse values) | b0 | 40 | CPU fwd+bwd runnable, sparse COO grad (243/4096 rows), pool registered once | caught attn_res path silently omitting the memory (would have trained the control under memory flags); now raises | read control ck cfg attn_res; arch_compat cases; train.py Adagrad wiring; 50-step smoke + M3 --peak-only on lane 5 |
| Kernel/throughput + sparse DDP grad path | tilerl | 10 | acknowledged; charter merged into worktree | none yet | microbench items 1-3 on lane 5 (<20 min card time): tok/s for lookup+gather, bytes/s per step sparse vs dense |
| Readout 2 fact probe (seen-vs-unseen DiD, API-name cloze over code_py_starcoder) | e1 | 40 | design ruled and pinned (cap = n_val + row_cursor = 85,380; contiguity proven from train.py:1783 + cursor-sum identity); score_mc_items with SE landed in worktree | own test caught two return-type bugs in the refactor; one integration-tree write self-corrected | item builder reading the 35 GB cache BEFORE any arm claims a card |
| Data parity + region boundary | 3b | 100 | 9/9 corpus fps live == .srcfp == control ckpt; boundary measured | self-corrected a [:16] truncation bug by importing the producer's fp function | review e1's pinned item file |
| Prereg row + amendments | fb | 100 | row 30f3f29b; amendments bcc86d18, 87960318, 6fe808a5, wording 25e0ca58 | 44 review 8319002d: dates PASS, superseded-marker fixed, stale charter sha fixed | — |
| memory_diag ledger + schema, launch monitors with stop rules | de | 0 | — | — | schema in data/ledger_schema.json; harness launch for M1/M2/M3 |
| score_matrix_present sees #cu rows | de | 100 | 00:56 commit | found by b0 on armB | — |
| Progress page + experiments doc section | 98 | 100 | 67b2ac04, 2040f009 | follows amended prereg, no unledgered numbers | curves when runs print |
| Card grant | fb | 100 | 77a9fb1f: block 1,2,4,6; lane 5 | — | write M3 grant (0+3) + lane 7 when card 3 reads 0 MiB (user: ~1 h from 00:45 local) |

## Global items carried from 2026-09-04

| item | owner | % | state | next |
|---|---|---|---|---|
| CI green | de | 90 | 25e0ca58 and 10acde74 green; 6fe808a5 in progress; four reds fixed (launch_tests, score_matrix_failpath, holdout regenerate, pod_drift world, free-card fixture) | CI-red WARN 2 h / FAIL 8 h check still owed |
| Harness: pod-reaching checks time out when tunnel down | de | 50 | deadlines raised (stopgap); pod_stamp_is_main → auth=pod with reachability SKIP written, not committed | commit with the dead-host broken world |
| Harness package split (core/checks/cmds/infra) | de | 0 | design accepted | plan doc after the memory program's monitors |
| Funnel table (T0-T3 per rule) + gates_inventory.tsv | de | 0 | tiers ruled | table |
| _tmp_repo() never cleans up (~80 worlds per selftest) | de | 0 | ruled: fix the producer now | — |
| Infra decoupling (aupai-infra) | tilerl | 40 | inventory + contract 326831f1; wrappers vendored 1ab666f6 with cd-into-background refusal; two inventory misfiles returned | subtree cut after the memory program's kernel work |
| Prose rules → machine gates (36 → 32 manual) | all | 100 for this round | Shared-files T0 (3b, file_claim + owner scoping ed3eba36), Language T3 deleted, rule 1 (e1), rule 2 T1 (e1 38af3d47), rule 3 (e1, unblocked, after the probe) | rule 3 train.py ETA |
| gate_failure_shapes restructure + 33 folded incidents | 44 | 100 | 141 = 59 + 82; 33/33 located, §162 re-cited, file::function rule; §175 §176 landed | §177 (attn_res) on b0's write-up |
| Friction summary + fixes | 44 | 100 | 2fa5c710; post-merge hook warn-only; minutes_required check | daily |
| Credential in transcripts (ANTHROPIC_AUTH_TOKEN) | fb | 100 | 83 occurrences in 6 transcripts + 2 zsh_history lines redacted; repo/pod clean across 18 branch tips and 4149 commits | user rotates the relay token; GitHub secret scanning is off on the repo (user setting) |
| Laptop disk | user | 100 | 5 GB → 112 GB free | /private/tmp/e1_v11_snapshot (2.8 GB, 8 projects' transcripts) left in place per user |
| Head-hybrid A/B | b0 | 100 | B loses 0.087 nat; closes + prereg outcome 15de1e0e; checkpoints pinned | user: keep or prune both arm checkpoints |

## Open user decisions (unchanged)
/data01 backup refresh schedule; code_tests fetch placement on /data02; head-hybrid and Stage E checkpoints keep/prune; e1_v11_snapshot.
