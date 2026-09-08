# Controller board (fb) — updated 2026-09-07T22:20Z (dates in this file corrected 2026-09-08 00:3xZ: the 09-07 evening had been written as 09-08), rewritten every tick

Percent = share of the deliverable landed on main and verified by a second reader; "quality" names what the reviewer opened, or the defect the owner caught in the process. The memory-layers program (2026-09-05) is closed: all three arms stopped on readout 4 (key-usage collapse), facts in facts/memory_layers.json, every entry from a STOPPED arm.

## 1.5b-a0.2b-e48_30b: resume 1 (prereg runs/prereg.jsonl#moe48_30b_0907, amendments 1-12)

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| Resume 1 on cards 1,2,3,4,5,7 | b0 | running | launched 18:33Z 09-07 at step 22500; step ~27,000/38,146 at 22:20Z; 2.83 s/step, 45-46K tok/s/gpu, peak 44.1 GiB; log runs/1.5b-a0.2b-e48_30b_resume1.log | de measured 1,179 steps/h over the segment; ends ~07:1xZ 09-08 | stop trigger: two consecutive vals > 1.91 (amendment 2), max so far 1.846 |
| Val rise after the cot/chat cap | b0 | 100 | amendments 9-12; runs/b0_domain_loss_resume1.jsonl, runs/b0_domain_loss_cap.jsonl (main 99f5f226) | registered discriminator FLATTENS: cot +0.0093/1k vs +0.0729 window 1 (|d|/se 4.70); chatml 0.17x, chat_qa 0.19x; control floor flat | cap on cot/chatml/chat_qa at 4.0 epochs stays for resume 1; fresh cot supply is a resume-2 prereg question (user decision) |
| MFU print 202-279% (dense-priced denominator) | de | 90 | PR #40 approved by 3b (review row 80993127); latent 3-D short_conv false-match filed de-74 | six mutants red | merges at the freeze lift ~07:50Z, order #24, #23, #40, de-74; merger pushes the pod |
| Resume 2 corpus: code_rp1t_dd09_full | 3b | 100 | 9,837,521,903 tok, 5,537,807 docs, 387 shards, fp 6bfa756cc1b7a965 (PR #39) | zero new bytes, hardlink union | user decision: flip CODE_RP1T_DOMAIN to dd09_full for resume 2 |
| Stale row code_rp1t_fetch8_0907 | fb | 100 | closed ok 20:1xZ: 8/8 files, 19 GB, b2v2 4,891,752,636 tok | it had blocked every pre-commit hook for 40 h | 3b may amend |

## Cards and guard

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| Card guard was inert (theirs=[]) | b0 | 100 | PR #45 f145ce4c; live file ours [1,2,3,4,5,7] theirs [0,6]; pod `harness launch --cards 0` RC=2 | tilerl-0a case #45#issuecomment-5575429861, four mutants incl. the original regex | — |
| Card 6 lend for the cap read | fb | 100 | grant note 21:30-21:45Z (main 38c8e935), used 21:32-21:34Z 09-07, reverted c58e693d, verified by tilerl-0a on main and pod | guard exercised end to end on a real job | — |
| Grants carry an expiry the classifier reads | b0 | 0 | task b0-32 | rulings: theirs_baseline pinned {0,6} in the check citing the 09-06 order; agreement property (baseline-theirs = theirs or valid lend; baseline-ours = not theirs); not-yet-open lend reads theirs; unparseable expiry refuses; now= injectable; paired mutations | reviewer tilerl-0a; after resume 1 |
| AGENTS.md:50 said all 8 cards are ours | fb | 90 | PR #53 (one line + ledger rows), approved by tilerl-0a, MERGEABLE at 3bd3cf15 | tilerl-0a verified the four claims against artifacts; old line from 08471eec superseded not corrected | merge + pod push by tilerl-0a |
| AGENTS.md:52-56 lane doctrine for a six-card world | fb | 0 | task fb-7 | — | after resume 1, reviewer 44 |

## Repository mechanics (tonight's incidents)

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| main checked out in aupai-b0; hand write of refs/heads/main | b0 / de | 100 | PR #52 main_in_no_worktree (WARN); PR #54 records the pair 484a9528 -> b8b39618; b0's commit recovered via merge_main b0 (99f5f226) | de: check FAIL on main, WARN at PR head, broken world still FAILs; b0 branch created in the same second as the reset (a recovered loss) | de: behind-main hook refuses a commit when HEAD is main and main != origin/main (separate PR) |
| GitHub merge ignores merge=union | fb | 100 | friction row 22:1xZ; fix: code PRs from a branch with no ledger files | tilerl-0a three-way without attributes: tasks.jsonl 1 marker | — |
| Hook runs the committing worktree's harness.py | de | 0 | b0 friction row | a fix on main does nothing for a branch until merged there | de owns |
| runs/ scripts have no route to the pod | 44 | 0 | task #45; friction row | restamp ran as inline heredoc | one-off pod scripts live under scripts/ |
| pod/local ledger classifier: monitor rows | de | 100 | PR #47 f4cacde6 monitor_state_only + _MONITOR_FAILED split | 44 two rounds, four wiring mutants red | — |
| pod ledger rows home | de | 100 | de-72, 23 rows; 5 differing keys are process-state vs measurement | — | — |
| doc_flags_parse check | 3b | 100 | PR #41 bfb5e9ee, 308/428 | broken world was green twice; _tracked_py empty-git fix | task #90 subparser walking |
| DeepSeek-V4 facts + loop design doc | 44 | 100 | PR #44 67d32d96: facts/deepseek_v4.json (9), docs/lessons/next_version_v4_loop.md (open) | tilerl-48 two rounds, 7 numeric errors caught in round one | user: next-version architecture after resume 1 |
| cot_open_thoughts restamp | 44 | 100 | PR #43 7aa4c0b2, tokens_kept 776,084,377 | — | — |
| Unified token counter | fb/de | 100 | PR #36; eight cs.* facts restated | — | — |

## Open user decisions

1. cot/chatml/chat_qa 4.0-epoch cap: keep for resume 1 (controller recommendation, forgetting saturates: cot rate 1/8 of window 1); fresh cot supply for resume 2?
2. Resume 2: flip CODE_RP1T_DOMAIN to code_rp1t_dd09_full (9.84B; starcoder share of code rows 95.4% -> 58.4%)?
3. /mnt/data02/aupai_backup holds 99 GB of backed-up checkpoints; a backup is a copy, not a move, and facts cite the files. Not deleted.
4. Next version: loop (SMELT) + sparse MoE + CSA-with-SWA + HCA (DeepSeek-V4); NoPE->RoPE breaks checkpoint compatibility. Design doc open.

## Standing rules restated

Cards 0 and 6 are tileRL's (user order 2026-09-06); lends are a GRANTED note with a window and are reverted by hand until b0-32. train.py/model.py/hooks frozen while the block is held; window opens ~07:1xZ 09-08. Code goes through a PR from a branch holding no ledger files; ledger rows through merge_main. No attribution trailers; commit subjects end with the session marker.
