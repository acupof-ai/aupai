# Controller board (fb) — updated 2026-09-08T07:20Z, rewritten every tick

Percent = share landed on main and verified by a second reader. Every row re-verified against origin/main f98f9c66 and the pod this tick.

## The 30B leg: verdict, corrected after adversarial review

| item | owner | % | delivered | quality / evidence | next gate |
|---|---|---|---|---|---|
| **domain_bpb divisor defect** | b0 | 0 | eval/domain_bpb.py:117 text_bpb truncates to max_ctx=2048 (line 127) and divides by the whole 4097-token row's bytes (line 135) | every published domain_bpb, the prereg bar 0.334243 and 98's report are ~2x low. Confirmed twice: in-row byte split 1.9944-2.0274 across nine domains, and domain_loss nats/token reproduces 0.7227 against the published 0.3588 scaled to 0.7195. The docstring claims the property the code lacks. Deltas, signs and ratios are unaffected | fix + known-answer selftest + restate facts and score rows, reviewer 3b |
| Verdict on the leg | fb | 100 | reported to the user 07:0xZ, corrected from the 06:5xZ version | **flat on the metric of record** (mean domain_bpb -0.00018 over 18.7B tokens); real wins are lambada_en +0.0340 vs the annealed 8B at ~3.9 sigma and nll/byte -0.0321; HumanEval tie unbroken against the annealed 8B (estimators disagree in sign) | — |
| Token efficiency, the leg's real yield | b0 | 0 | measured by review, to be filed as facts | on the six weight-stable domains 18.7B tokens bought ~0.0216 bpb; the 8B leg's 1,172-step warmdown (~0.92B tokens) bought ~0.0198 on the same six. Constant-LR tokens are worth about a twentieth of the decay per token | facts rows after the divisor fix |
| Token-only contrast EXISTS | b0 | 0 | steps 9,000-22,500 all ran mix_..._20b_launch.json, so .step9000/.step14000/.step20000 share bit-identical weights; .step20000 is already bpb-scored | my earlier "this leg cannot say what tokens bought" was wrong; the cut to cot/chatml/chat_qa happened at 22,500, not at the resume | write it up; optionally score .step14000 (~85 s, lane card 1) |
| r = -0.945 framing | fb | — | withdrawn from anything unmerged | the log weight ratio takes three distinct values over nine domains; Spearman -0.417; dropping the three starved domains flips the sign. Two-group difference (n=3 vs 6), not a dose-response. Mix and missing anneal are each individually sufficient to explain the whole bar miss | 44 and 98 drop it from unmerged text |
| Forgetting cost, measured | b0 | 100 | b0_domain_loss_{resume1,cap,valrise}.jsonl on main | the RATE saturates (cot +0.073 -> +0.009 nats/1k, chatml and chat_qa 0.30 -> 0.02; half-life 400-1000 steps) but the LEVEL is permanent: those three end 9-16% worse in bpb | — |
| Report | 98 | 100 | PR #70 + fixes in #74 | three defects filed and fixed | needs the divisor restatement when it lands |

## v2 program

| item | owner | % | delivered | next gate |
|---|---|---|---|---|
| Spec + prereg | 44 | 90 | PR #66, head 54b36b54, CI green, APPROVED by tilerl-0a (review row 7d6c260d on main) | **tilerl-0a merges it** — AGENTS.md:387 says the reviewer merges; this is the only blocker on the launch path |
| CSA attention | b0 | 40 | PR #78 open | reviewer tilerl-0a |
| Loop + schedule | de | 10 | flag plumbing | de-74, then the loop blocks |
| Data / mix | e1 | 80 | total ruled 30B, weights = resume-1 byte-for-byte except math_owm supply +0.234%; gates pass on the v2 composition (hanzi 0.9793, ref fertility 1.4286) | write the mix; PR #77 (both tokenizer samplers read non-shard artifacts) with 3b |
| Cursor identity | 3b | 80 | PR #72, 15 mutants red, accept-path prints the compared domain set | de reviews |
| Answer-format arm (new) | e1 | 0 | preregister: restore cot/chatml/chat_qa to their 8B weights, readout answer_present and l1_fewshot | the endpoint's answer_present 0.340 is the lowest zh three-demo row in the ledger; the run cut those domains ~13x, so "needs SFT" is confounded and this is the cheaper test |

## Cards, repo, infrastructure

| item | state |
|---|---|
| Cards | all eight at 0 MiB, no claims; 0 and 6 are tileRL's; 2,3,4,5,7 held for v2's first launch; 1 is the lane |
| CI / main | completed success on f98f9c66; stash empty |
| Pod | stamped 86e5014b, behind origin f98f9c66 — push due this tick |
| Deletion listing | #76 merged: globs 78 ledgers, --resume both spellings, selftest registered. 44 files / 161 GB free, 19 carry a family annotation. Ruling: may PROPOSE candidates, family-annotated rows excluded, owner confirms each; deletion still needs a user instruction naming targets. Disk 95%, 109 GB free |
| Open PRs | #78 b0 (tilerl-0a), #77 e1 (3b), #75 44 (3b), #72 3b (de), #66 44 (tilerl-0a, approved), #23 tilerl (changes requested) |
| Shape of the night | a predicate set answering a narrower question than the one asked, invisible in its own well-formed output: §275, then reproduced inside its own fix, then again in the tokenizer samplers and in domain_bpb's divisor |

## Open user decisions

1. **Finish the 4,146 remaining steps to anneal the endpoint** (~3.5 h on six idle cards). Recommended: the 8B leg's 1,172-step decay bought 92% of what 25,000 constant-LR steps bought, and only an annealed endpoint is comparable to the registered bar.
2. 30B composition for the next full run: code+math at <=1.0 epoch takes 83.8% of a 30B budget. Not needed for v2, which keeps the control's weights.
3. cot/chat 4.0-epoch cap: keep for v2 as the control, test the restore in e1's new arm.
4. Pod disk at 95%; nothing deleted.

## Standing rules restated

Cards 0 and 6 are tileRL's. Code through a PR from a branch with no ledger files; ledger rows through merge_main. Approval is a PR comment with artifact: or case: plus a review row, and the reviewer merges and pushes the pod. No attribution trailers; subjects end with the session marker. Nothing is deleted without a user instruction naming the target.
