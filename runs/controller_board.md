# Controller board (fb) — updated 2026-09-08T05:25Z, rewritten every tick

Percent = share of the deliverable landed on main and verified by a second reader; "quality" names what the reviewer opened, or the defect the owner caught in the process. Memory-layers program (2026-09-05) closed; facts in facts/memory_layers.json.

## 1.5b-a0.2b-e48_30b: resume 1 — STOPPED at step 34000 by user order (04:03Z)

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| Stop at .step34000 | b0 | 100 | save 04:01Z 6,059,092,240 B, torch.load step=34000; TERM then KILL by exact PID, eight cards 0 MiB; row re-closed ok (monitor had written fail/vanished; exp.py logged the reclassification); pinned ckpt_1.5b-a0.2b-e48_26.7b_0908.pt (same inode) | val 1.824@34000, run min 1.812@22600, 26.74B tok; never warmed down (warmdown_start 34332) | — |
| Endpoint score_matrix + 8B .step9000 domain_bpb on card 1 | b0 | 100 | both rows in runs/score_matrix.jsonl (pod, 05:1xZ); card 1 released, eight cards 0 MiB | 30B vs 8B (both full-lr): lambada 0.3221 vs 0.2354, nll/byte 0.675 vs 0.807; humaneval bpb 0.561/0.433 vs 0.593/0.472; domain_bpb mean 0.359 vs 0.359 — 30B better on six domains (math 0.338/0.365, en_c4 0.521/0.544, textbook 0.240/0.258, zh 0.643/0.671, starcoder 0.229/0.246, rp1t 0.193/0.209), worse on the three capped domains (cot 0.247/0.226, chatml 0.402/0.347, chat_qa 0.416/0.363); mc_ceval 26.1 (chance); lambada_zh open_acc1 0.314 | rows to main via pod_push --all; 98 cites them |
| Resume-1 mix in the tree | b0 | 50 | PR #65 open, md5 c12ba979 both sides; not regenerable (.step22500 deleted, deriver refuses to guess a cursor) | — | reviewer 44 |
| Full report | 98 | 50 | docs/lessons/moe48_30b_0907_report.md on branch 98 (3aa21736); score cells pending | fact_refs_resolve green | PR after scores; reviewer 44 |
| Excursion 26-29k negatives (composition, epoch wrap, repetition) | b0 | 0 | exist only as messages | b0 declined to back-file prereg amendments (correct) | rows to runs/review.jsonl tonight |

## v2 program (user order 2026-09-08 ~03:4xZ: loop + sparse MoE + CSA/HCA + partial RoPE, KDA out)

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| Spec + prereg | 44 | 30 | PR #66 open | rulings: loop builds by user order; repo.loop_not_adopted_equal_compute stays measured with a boundary line; prereg names what retires looping (equal-FLOP arm loses on val nats and humaneval bpb) | review #67 first; cite tokenizer_eval only after it |
| CSA-with-SWA attention | b0 | 0 | task b0-35 | — | after score_matrix |
| Loop + schedule in train.py | de | 5 | flag plumbing started 04:1xZ; no model.py until 44's spec | numerics-parity gate before any schedule change (tilerl-0a ask, granted) | #24 merged bf2cd0ef; #23 open; de-74 after #40 (merged 9e69ea34) |
| Data / mix | e1 | 60 | recounts: math_owm_stage2 6,528,546,115 (+0.234% vs extrapolated), textbook_30b unchanged (e1 d5866f7d); tokenizer_eval: round-trip pass, hanzi 0.9890 (after #67), ref fertility 1.4286; never-used gate not decidable (seed range 0.0042 vs threshold 0.01) → e1-52 | ruling: v2 mix = resume-1 weights byte-for-byte except math_owm supply restated; open_thoughts is a later arm | PR #67 (tokenizer_report per-domain sampling) reviewer 44; facts/tokenizer.json after #67 |
| Deriver: shrink branch discards cursor; cursor_used_rows unread | 3b | 10 | task #91: refuse --total<default with --resume-cursor; mix records derived_against={row_cursor,srcfp,seed}; launcher refuses a differing triple; stage2 writer too | 24 committed mixes scanned, none self-contradictory | one PR, reviewer de; pod run arbitrates the reading |
| Compat tests per module | 3b | 0 | — | — | in each module PR |
| Reviewer | tilerl-0a | — | — | — | — |

## Cards

| item | state |
|---|---|
| 0, 6 | tileRL (user order 2026-09-06) |
| 1 | lane; b0 grant spent 05:1xZ, idle |
| 2,3,4,5,7 | block held for v2's first launch; no launch before 44's prereg row and #67 |

## Repository mechanics

| item | owner | % | delivered | next gate |
|---|---|---|---|---|
| Grants carry an expiry | b0 | 100 | PRs #58, #61 | — |
| coresident_cache_refusal dead world | b0 | 100 | PR #64 | — |
| Approval form (PR comment artifact:/case: + review row) | fb | 100 | PR #57 | — |
| AGENTS.md:50 card split; board in ledger set; dates corrected | fb | 100 | PRs #53, #60 | — |
| GitHub push from this laptop | fb | — | 04:2xZ: connect timeout 75 s to github.com, then ok; merge_main REFUSING push twice, retried by hand | — |
| Merge queue reviews: #24 no row, #23 no row, #40 by 3b | 3b | 50 | ruled: 3b merges #40 (done 9e69ea34), reviews+merges #24 (done bf2cd0ef) and #23 | #23 |
| Hook runs the committing worktree's harness.py | de | 0 | friction row | de |
| runs/ scripts have no route to the pod | 44 | 0 | task #45 | — |

## Open user decisions

1. 30B composition: code+math at ≤1.0 epoch takes 83.8% of 30B (dd09_full 9.84B + starcoder 8.79B + math 6.51B), leaving 16.2% for en/textbook/zh/cot/chat vs 32.9% tonight. Halve those, grow the budget, or cap code below supply. Not needed for v2 (keeps tonight's weights).
2. cot/chat 4.0-epoch cap: endpoint domain_bpb says cot +0.030, chatml +0.085, chat_qa +0.084 worse than step 20000 while every other domain improved. Keep for v2 (same mix as control) or lift in the open_thoughts arm.
3. Resume 2 / dd09_full flip: moot for the stopped run; part of decision 1.
4. /mnt/data02/aupai_backup 99 GB of checkpoints: not deleted.

## Standing rules restated

Cards 0 and 6 are tileRL's; lends are a GRANTED note with a window. Code through a PR from a branch holding no ledger files; ledger rows through merge_main. Approval = PR comment with artifact:/case: plus review.jsonl row; the reviewer merges and pushes the pod. No attribution trailers; commit subjects end with the session marker.
