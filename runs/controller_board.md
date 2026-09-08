# Controller board (fb) — updated 2026-09-08T08:0xZ, rewritten every tick

Percent = share landed on main and verified by a second reader. Rows re-verified against origin/main and the pod this tick.

## The 30B leg: CLOSED by user ruling

The user ruled 07:4xZ that the warmdown is NOT completed. The stop at step 34,000 of 38,146 stands and this version is final as it is. It is recorded as an incomplete schedule, not claimed as an advance.

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| **What the anneal was worth, measured** | fb | 100 | the 8B leg's own pair, same run and same held-out mix (mix_200m_8b.json) | `ckpt_..._8b.pt.step9000` 0.35897 -> `ckpt_..._8b.pt` 0.33424 on unweighted mean domain_bpb, **-6.89%**, all nine domains down, range -4.0% (en_c4) to -12.4% (chat_qa). Both terms share a denominator, so the divisor defect cancels exactly per domain and the ratio survives it | re-score the pair on b0's fixed code; the level moves, the ratio should not |
| **domain_bpb divisor defect** | b0 | 0 | eval/domain_bpb.py:117 text_bpb truncates to max_ctx=2048 (line 127) and divides by the whole row's bytes (line 135) | every published domain_bpb and the prereg bar 0.334243 are ~2x low. Known-answer test: uniform-256 text, true 8.000 bits/byte, reported 5.460. Per-domain factors 1.9671-2.0201, 64 of 64 rows truncated in every domain. Deltas, signs and ratios unaffected | PR #79, reviewer 3b. **Ruling: no rescaled level is published** — a factor from today's rows on a level from other rows is §277's shape. Re-score with fixed code; cards are idle now |
| **Held-out row sets are not frozen** | b0 | 0 | found while cross-checking the factor | `val_seqs` takes a seed-42 PREFIX sized `min(max(1,int(pool*0.05)),5000)`, so any domain under the cap changes its held-out rows when its pool grows. chatml moved 5.20% (1,106,294 -> 1,051,866 bytes), chat_qa likewise; four domains at the 5,000 cap are stable | outranks the divisor: it means two runs' chatml and chat_qa numbers were never scored on the same bytes, and no divisor fix repairs it. Shape + freeze the val row ids to a committed list |
| Verdict on the leg | fb | 100 | reported to the user, corrected twice under review | across versions on one vocabulary: **third** on HumanEval gold bpb/task (0.5609 behind ckpt_0.2b_8b_b192 0.5559 and its own annealed 8B sibling 0.5590), below two dense models on minimal pairs (0.7653 vs 0.8014), only clear win LAMBADA-en (0.3221, +0.0340 vs the annealed 8B, ~3.9 sigma) | b0 writes it into the ledger row; 98 into the report, naming the missing decay as why the comparison cannot be resolved |
| Token efficiency | b0 | 0 | measured by review | on the six weight-stable domains 18.7B constant-LR tokens bought ~0.0216 bpb; the 8B leg's 1,172-step decay (~0.92B tokens) bought ~0.0198 on the same six. Constant-LR tokens are worth about a twentieth of decay tokens | facts rows, after the re-score |
| Token-only contrast EXISTS | b0 | 0 | steps 9,000-22,500 all ran mix_..._20b_launch.json with bit-identical weights; .step9000/.step14000/.step20000 are comparable and .step20000 is scored | the cut to cot/chatml/chat_qa happened at 22,500, not at the resume | write up |
| Forgetting cost | b0 | 100 | b0_domain_loss_{resume1,cap,valrise}.jsonl on main | the RATE saturates (half-life 400-1000 steps) but the LEVEL is permanent: cot, chatml, chat_qa end 9-16% worse | — |
| Report | 98 | 100 | PR #70 + fixes in #74 | three defects filed and fixed | needs the divisor restatement and the closing verdict |

## v2 program

| item | owner | % | delivered | next gate |
|---|---|---|---|---|
| Spec + prereg | 44 | 100 | **PR #66 merged** by tilerl-0a, merge 1d6718e9, pod stamp 9522269e, 836 files | on main and on the pod; launch path unblocked |
| CSA attention | b0 | 60 | PR #78, **approved by 3b** (review row f761e0b9): flag-off torch.equal at 15,360 params max delta 0.0, flag-on differs 0.154, 16 positions perturbing k and v in float64 with no leaks, three mutants red | 3b merges and pushes the pod. Two divergences from facts/deepseek_v4.json#dsv4.hybrid_attention go in the b0-35 prereg row, not the docstring: non-overlapping grouping vs the reference's 2m neighbours, and coarse-score selection reuse instead of a Lightning Indexer. Gate is three summed sigmoids, not a mixture: 1.21x into the residual at step 0 |
| CSA cost | b0 | 0 | — | 60-step smoke + per-step cost in facts/efficiency.json. The O(T^2) select branch is what decides whether a kernel is worth writing; this number gates the arm, not the parity check |
| Loop + schedule | de | 10 | flag plumbing | de-74, then the loop blocks |
| Data / mix | e1 | 80 | total 30B, weights = resume-1 byte-for-byte except math_owm supply +0.234%; gates pass (hanzi 0.9793, ref fertility 1.4286) | write the mix; PR #77 with 3b |
| Cursor identity | 3b | 80 | PR #72, 15 mutants red | de reviews |
| Answer-format arm | e1 | 0 | deprioritised on e1's own measurement: marker exposure falls 1.42x, not 13x, because math_owm's 28.9% weight at 3.75% marker density carries 84.5% of exposure | preregister at the 1.42x effect size with the math-side lever as the competing alternative, both below v2 |

## Cards, repo, infrastructure

| item | state |
|---|---|
| Cards | 0 and 6 are tileRL's. 2,3,4,5,7 are the block, reserved for v2's first launch; 1 is the lane and free for b0's re-score. The warmdown grant was withdrawn and then ruled against by the user |
| CI / main | green; stash empty |
| Pod | pushed this tick, `pod_sync_check` verified |
| Deletion listing | #76 merged. 44 files / 161 GB free, 19 family-annotated. May PROPOSE; owner confirms each; deletion still needs a user instruction naming targets. Disk 95%, 109 GB free |
| Open PRs | #79 b0 (3b), #78 b0 (3b, approved), #77 e1 (3b), #75 44 (3b), #72 3b (de), #23 tilerl (changes requested) |
| Shape of the night | a predicate set answers the question it ENUMERATES, not the one it is named for: §275, then its own fix, then the tokenizer samplers, then domain_bpb's divisor, and now val_seqs' moving prefix |

## Open user decisions

1. ~~Finish the warmdown~~ — **ruled 2026-09-08: do not run it.** The leg is final at step 34,000.
2. 30B composition for the next full run: code+math at <=1.0 epoch takes 83.8% of a 30B budget. Not needed for v2, which keeps the control's weights.
3. cot/chat 4.0-epoch cap: keep for v2 as the control.
4. Pod disk at 95%; nothing deleted.
5. User has opened SFT and RL as allowed next work. No arm is preregistered for either yet.

## Standing rules restated

Cards 0 and 6 are tileRL's. Code through a PR from a branch carrying no ledger files; ledger rows through merge_main. Approval is a PR comment with `artifact:` or `case:` plus a review row, and the REVIEWER merges (`--merge`, never `--squash`) and pushes the pod in the same step. No attribution trailers; subjects end with the session marker. Nothing is deleted without a user instruction naming the target.
