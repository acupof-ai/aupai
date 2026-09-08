# Controller board (fb) — updated 2026-09-08T06:35Z, rewritten every tick

Percent = share of the deliverable landed on main and verified by a second reader; "quality" names what the reviewer opened, or the defect the owner caught. Every row below was re-verified against origin/main 711daa59 and the pod this tick, not carried from the previous board. Memory-layers program (2026-09-05) closed; facts in facts/memory_layers.json.

## 1.5b-a0.2b-e48_30b — STOPPED at step 34000 by user order (04:03Z)

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| Stop | b0 | 100 | .step34000 saved 04:01Z (6,059,092,240 B, torch.load step=34000); TERM then KILL by exact PID; pinned ckpt_1.5b-a0.2b-e48_26.7b_0908.pt, inode 84244826 nlink 2 (one 6.06 GB file under two names) | val 1.824@34000; run minimum **1.807@22400**, recomputed from the val series — 1.812@22600 is only where the resume-1 log starts, so the endpoint is +0.017 over the floor, not +0.012. 26.74B tok; warmdown_start 34332 never reached, so the endpoint has no lr decay | — |
| Endpoint scores + 8B .step9000 domain_bpb | b0 | 100 | runs/score_matrix.jsonl lines 88-89 on main; alias row under the produced name ckpt_1.5b-a0.2b-e48_30b.pt (delete it and score_matrix_present goes FAIL) | 30B vs 8B, both full-lr: lambada 0.3221 vs 0.2354, nll/byte 0.675 vs 0.807; humaneval bpb 0.561/0.433 vs 0.593/0.472; domain_bpb mean 0.359 both. The flat mean is two effects cancelling: six held/raised domains improved 0.016-0.027, the three cut to 0.08x regressed 0.020-0.054, **r = -0.945** between log weight ratio and delta, recomputed independently this tick | never report the unweighted mean bare |
| Ledger artifacts on main | b0 | 100 | b0_e48_30b_val_series.jsonl (md5 156305ab, 125 steps 9200-34000 at 200-step spacing, no duplicates, 58 overlap steps agree), b0_domain_loss_valrise.jsonl (0477ee53, 3 rows), _resume1 (e85bc4ca), _cap (0b3f1031) | all four md5 prefixes match; the excursion and the 22,500 reversal are now re-readable from the tree | the valrise rows name milestone_valrise_*.pt pins, not .step<N> rolling saves — cite them that way |
| Milestones lend-date correction | b0 | 50 | correction row 27 at ba229ca8 | **the correction is itself off by two**: the rows carrying 2026-09-09 are 23/24/25, not 21/22/23; row 24 is the step29000 pin and row 26 is the .step34000 endpoint pin. Pins verified intact | b0 appends a second correction naming rows by pinned_as |
| Full run report | 98 | 80 | docs/lessons/moe48_30b_0907_report.md on main (PR #70, 711daa59); 1.807@22400 correct, no-warmdown stated three times, per-domain table with the ratio column present | three defects: line 29 cites data/mix_1.5b-a0.2b-e48_20b_launch.json which is absent from main; section 5 says no ledger row/artifact exists for the excursion negatives while review.jsonl 2d67ad00 and valrise e6ee15d0 are ancestors of that same commit; the section-7 lead table prints the unweighted mean uncaveated | follow-up PR, reviewer 44 |

## v2 program (user order 2026-09-08 03:4xZ: loop + sparse MoE + CSA-with-SWA + HCA, KDA out)

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| Spec + prereg | 44 | 80 | PR #66, head 54b36b54, CI green (3/3), OPEN | verified: the "anneal before the rise" sentence is gone; the two onsets are separated (slope reversal ~22,500 = 59.0% of 38,146, t=+6.179; excursion 26.4-29.6k peak 1.884); r=-0.945 present and reproduced; every cited repo path is on main. Onset fix actually landed at 644ad3e6/8a3ff769; one prose peer reference survives at line 4 | tilerl-0a's review row, then merge + pod push |
| CSA-with-SWA attention | b0 | 0 | task b0-35 | prior facts/deepseek_v4.json#dsv4.hybrid_attention | after b0's ledger corrections |
| Loop + schedule in train.py | de | 10 | flag plumbing since 04:1xZ; #40 and #24 merged, de-74 open | numerics-parity gate before any schedule change (tilerl-0a's ask, granted) | de-74 PR |
| Data / mix | e1 | 70 | recounts math_owm_stage2 6,528,546,115 (+0.234%) and textbook_30b unchanged, on main at d5866f7d; tokenizer gates re-run on main's fixed tool: round-trip pass 256/256, hanzi 0.9890, ref fertility 1.4286 | **PR #69 is OPEN, unmerged, no human review** — facts/tokenizer.json on main holds 16 ids and neither new one; never-used range is 0.0110 seed-only against a 0.01 threshold | 44 reviews #69; then the v2 mix file (weights identical, math_owm supply restated) |
| Cursor identity in mixes | 3b | 60 | PR #72 (branch 3b-91), 13 mutants red | mix pins derived_against={row_cursor,row_cursor_srcfp,row_cursor_seed}; refuses differing rows or changed srcfp, accepts an equal triple from another checkpoint file; two of its first surviving mutants were defects in the test itself | de reviews; print the compared domain set on the accept path |
| Compat tests per module | 3b | 0 | — | — | in each module PR |
| Report | 98 | 80 | PR #70 merged | see above | follow-up PR |

## Cards (verified on the pod this tick)

| cards | state |
|---|---|
| 0, 6 | tileRL (user order 2026-09-06) |
| 1 | lane; b0's score grant spent, idle |
| 2,3,4,5,7 | block held for v2's first launch; no launch before #66 merges and the prereg row exists |
| all 8 | 0 MiB, no compute apps, runs/claims/ empty, no live python or torchrun |

## Repository and infrastructure

| item | owner | state |
|---|---|---|
| CI on main | — | completed/success on 711daa59; stash stack empty; harness check exits 0 (score_matrix_present is WARN on an unrelated row, e1_31_middle_layer_loop) |
| Open PRs | — | #72 3b (reviewer de), #69 e1 (44), #66 44 (tilerl-0a), #23 tilerl (changes requested by 3b: stale sidecar survives a rebuild; missing seq escapes as KeyError) |
| Merged tonight | — | #40, #24, #64, #67, #68, #70 |
| Pod | — | in sync at 711daa59, stamped 06:10Z, 830+ files match |
| **Pod disk 95%** | b0 | /work and /data00 are ONE filesystem: 2.0T, 109G free. /mnt/data02 has 2.7T free. A v2 launch writes 6 GB per save. Deletion-candidate listing owed before launch; nothing deleted, no user instruction names a target |
| Open tasks | — | 26 open after folding runs/tasks.jsonl by id |
| Hook false-fire | de | the shared-repo config check fires on a peer's concurrent `git push -u`; three different hashes on one file, ~20 min lost (3b). Fix: diff config KEYS, not the hash |
| GitHub push from this laptop | fb | one 75 s connect timeout at 04:2xZ; merge_main REFUSED the push twice, retried by hand. Origin now current |

## Open user decisions

1. **30B composition for the next full run.** Code+math at <=1.0 epoch takes 83.8% of a 30B budget (dd09_full 9.84B + starcoder 8.79B + math 6.51B), leaving 16.2% for English/textbook/zh/cot/chat against 32.9% tonight. Halve those, grow the budget, or cap code below supply. Not needed for v2, which keeps tonight's weights.
2. **cot/chat 4.0-epoch cap.** The endpoint measures the cost: cot +0.020, chatml +0.054, chat_qa +0.053 bpb worse than the 8B row while every held-weight domain improved. Keep it for v2 (same mix as the control) or lift it in the open_thoughts arm.
3. **Pod disk at 95%** (109 GB free on the training filesystem) before v2 launches.
4. /mnt/data02/aupai_backup holds 99 GB of backed-up checkpoints. Not deleted.

## Standing rules restated

Cards 0 and 6 are tileRL's; a lend is a GRANTED note with a window. Code goes through a PR from a branch holding no ledger or generated files; ledger rows through merge_main. Approval is a PR comment carrying artifact: or case: plus a runs/review.jsonl row, and the reviewer merges and pushes the pod. No attribution trailers; commit subjects end with the session marker. Nothing is deleted without a user instruction naming the target.
