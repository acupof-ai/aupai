---
question: What are the nodes between today and a scored 30B-recipe decision, who owns each, what number closes it, and what gets deleted on the way?
status: recorded
source: fb ruling 2026-09-03 12:20Z, user order the same hour ("每一个节点那都要拿到结果"; "删除式成功化，仓库内只保留唯一的"). Inputs: runs/experiments.jsonl (data_leg_206m_8b), facts/corpus_supply.json, docs/lessons/eval_resolution_200m.md, data/mix_30b.json, runs/tasks.jsonl (46 open of 205)
---

# Roadmap from 2026-09-03

Two rules govern every row below. A node has one owner, one exit number, one date; a node
without its number on its date is closed as failed and the ledger says so. Each concern has
one canonical artifact; a second artifact saying the same thing is deleted, and the survivor
is named in the deleting commit.

## 1. Measured state, 12:20Z

| item | value | basis |
|---|---|---|
| data leg `data_leg_206m_8b` | step 14270/15258, ETA 0.4 h, 82K tok/s/gpu, peak 49.52 GiB | pod log |
| pre-registered val check (e3b06e89) | step 9000 read 2.023 < 2.032: print resolution, no action; monotone since to 1.956 at 14000 | pod log |
| the one metric with resolution | HumanEval gold BPB 0.5451 → 0.5199 (step 7000 → 10000, +43% tokens), 2021 → 1928 bytes for 164 solutions | runs/score_matrix.jsonl |
| its anchors on the same 29,662 bytes | gzip -9 2.0961, bzip2 1.8974, Pythia-160M @4.2B tok 0.918 bits/byte; math_test_500 scored 379,651 bytes: gzip -9 1.9815, bzip2 1.4236 | `eval/n3_report.py` `corpus_bytes` + `anchors`, e1 2026-09-03; Pythia from `humaneval_bpb.py --hf` |
| public models on the same 29,662 bytes (`humaneval_bpb.py --hf`, e1 2026-09-03; tokens as stated by model cards) | Qwen2.5-0.5B 494M/18T 0.2640; SmolLM2-360M 362M/4T 0.3624; SmolLM2-135M 134M/2T 0.4463; **ours 206M/8B 0.4567**; Pythia-160M 162M/300B 0.6024. Different points on different data curves: says the number is good for the budget, not that the architecture is better | runs/n7_hf_*.json |
| N7 result (e1-31, 2026-09-03) | 2x2, each arm scored in its own topology: humaneval BPB unlooped-trained 0.4635 vs looped-trained 0.4658 (+0.0023, SE 0.0006, z 3.6, 103/164); domain_loss 1.9858 vs 1.9951 (+0.0093, 9/9). Mismatch cells are 8-11x larger (+0.0193, +0.0264): weights and topology must match, and Stage A (+0.0273) measured that mismatch, not the loop. Forward latency ~1.3x (1.22-1.34 over four 60-iter reps; the earlier 1.64x was one 20-iter run and is retracted). 500-step rerun (fresh arms, e1 2026-09-04): 0.4643 vs 0.4676 (+0.0033, SE 0.0006, z 5.7, 112/164); mismatch cells +0.0240 / +0.0209; training loss identical at every checkpoint (final 1.057 both) while BPB diverged, so the cost grows with steps rather than closing. **Not adopted.** Stage C (user proposal 2026-09-04: post-loop layers see the prompt bidirectionally): the only post-loop MLA layer is block 11, the last, and a prefix mask there is a null by construction (prompt positions are unsupervised and nothing above the head reads them; measured 0 of 16384 supervised logits change, e1). Rescoped to two arms with existing twins: P3 = unlooped, prefix on all three MLA layers vs ckpt_n7c_unlooped; P7 = looped 4-7, prefix on layer 7 vs ckpt_n7c_looped; 500 steps, same pack and seed. Results (e1 2026-09-04, eval path verified: eval vs train construction differ by 0.0004 BPB on the same bytes; prompt-length binning flat, so not a prompt-distribution effect): P7 null (+0.0003 BPB, SE 0.0003, 73/164); P3 scored under its trained prefix mask 0.4871 vs twin 0.4643 (+0.0228, z −13.7, 151/164 worse) while training loss matched the twin (1.058 vs 1.057); P3 weights scored causally 0.4612, 126/164 better than the twin (−0.0031, z 5.5), same size as the loop gap, opposite sign; n=1 seed, directional only. Verdict: bidirectional prompt at inference costs 7x what prefix training gains; not adopted. Side finding to probe: packing six documents in one row moves logits by 3-4 even under causal document masking, so packed training was never scored in its own regime. Stage D (user order 2026-09-04, b0-24): from-scratch A/B at 122M (d=768 h=6 ffn=2304), 1B tokens, two arms differing only in the loop, cards 0-3 after N2 and Stage C; pre-registered thresholds in the task row. RESULT (b0, 2026-09-04): both arms 3815 steps / 1.00B tokens, seed 42; humaneval task-mean BPB looped 0.7138 vs unlooped 0.7070, looped − unlooped = +0.0068, paired SE 0.0042, t +1.62, 93/164 worse, 95% CI [−0.0015, +0.0150]: excludes the "worth a follow-up" band, a null with a positive point estimate; wall clock 1.244× (FLOPs 1.333×). Mismatch cells: unlooped scored looped +0.073 (159/164), looped scored unlooped +0.211 (164/164), so the looped model depends on its extra visits without gaining net. domain_loss block pairing (b0): looped BETTER by 0.0309 nat, SE 0.0011, t −28.5, 531/576 blocks, 9/9 domains; training-path val agrees (2.173 vs 2.192, cu passed). The two rulers disagree in sign; domain_loss was scored on the cu=None eval path (N8 leak), HumanEval one task per row. cu-passed rescore (b0, 2026-09-04): looped − unlooped = −0.022325 nat, SE 0.000665, t −33.55, 537/576, 9/9 domains, against −0.030937 on the cu=None path: 28% of the advantage was leak-mediated, 72% survives. Per domain the shrink tracks the leak (corr −0.951): chatml −0.0625→−0.0311, chat_qa −0.0646→−0.0312, the seven low-leak domains moved ≤0.007. The reading "the loop helps dialogue most" is RETRACTED; what survives is a uniform −0.012..−0.033 across nine domains. Scorecard at equal tokens: corpus loss LOOP WINS, HumanEval code likelihood n.s. (+0.0068, SE 0.0042): two different questions, not one question with two answers. Fact: facts/smelt_deeploop.json#repo.loop_from_scratch_stage_d (both retractions in its uncertainty); eval-path artifact: facts/efficiency.json#eff.eval_path_cu_artifact_ce. Both Stage D arms pinned (milestone_keep_b0_stagedarm, 2026-09-04). ADOPTION VERDICT WITHHELD for the equal-compute arm; the equal-compute arm (unlooped, 4824 steps = 1.2646B tokens on the 6N active-parameter accounting, ratio 1.2646 bracketed by the arms' measured 1.243 wall / 1.272 tok/s; the 16/12 = 1.333 counted the head and embedding twice; same seed, warmup 300 absolute) launched 2026-09-04 on cards 0+1, ~1.6 h, scored in the same six cells. Stages A-C stay not adopted. Boundary of A-C: 250 and 500 SFT steps on a 206M checkpoint rules out cheap adoption, not SMELT's from-scratch claim | runs/n7b_*.json, e1-31 |
| N7 Stage A (inference-only loop, blocks 4-7) | humaneval BPB 0.4567 -> 0.4840 (145/164 worse, z 13.4); domain_loss 1.9443 -> 2.0609 (9/9); ~1.3x ms/token forward-only (**the 1.64x first published here is RETRACTED** — one 20-iter run; four 60-iter reps read 1.223/1.284/1.329/1.336 and 1.640 is outside that range), 1.002x in SFT training; parity 0. Stage B done, see the row above; the +0.0273 here is the topology mismatch, not the loop | runs/n7_domain.jsonl, e1-31 |
| curve so far | humaneval byte-weighted BPB 0.5451 / 0.5227 / 0.5199 at steps 7000 / 7500 / 10000; lambada_en 22.9 / 21.0 / 23.9 % | runs/score_matrix.jsonl |
| compute-matched control | held-out nat/byte floors ours 0.451 vs Pythia-160M 0.904; after SFT 0.294 vs 0.353 | docs/audits/control_pythia160m_vs_ours.md |
| 30B mix | `domains` is empty; all 8 domains sit in `_blocked` | data/mix_30b.json |
| backlog | 46 open tasks of 205 | runs/tasks.jsonl |

Supply against the 30B contracts (tokens, `facts/corpus_supply.json`):

| domain | contract | landed | gap |
|---|---|---|---|
| code_rp1t | 8.0B | 7.57B, unstamped | re-stamp |
| code_tests | 2.0B | 0. Mined set is a subset of code_py_starcoder by construction (same 59 shards, pairing is a repo property applied after ast.parse): kept 97.1%, content-hash overlap 99.999%, 15 novel rows of 1,474,440, about 28K new tokens (tilerl-23, runs/ct_overlap.json). NOT BUILT: a build would double-count starcoder rows under a second domain name. Real supply needs repo-shaped fetching (pair rate rises with files/repo: 1.7→0.10%, 10.2→3.81%; ~52M rows, 150-200 GB, does not fit the 150 GB emptyDir; tilerl f6e90bfa, runs/cs_probe.json). USER DECISION 2026-09-04 (option A): the 2.0B moves into the existing code domain's weight; code_tests leaves `_blocked`; no fetch (tilerl-24) | 0, resolved by weight move |
| math_owm | 5.5B | 6.51B | none |
| cot | 4.5B | 0.42B | 4.1B |
| en_c4 | 5.5B | 7.844B exact across three directories: en_c4/ 4.801B, en_c4_stage2/ 2.397B, en_c4_30b/ 0.646B (tilerl 2026-09-04, frozen tokenizer full pass; build_corpus's own 1.51B was chars/1.5, ratio 2.34, §155 second instance) | surplus 2.34B; en_c4_30b stats stamp pending (3b) |
| textbook | 1.0B | 1.61B | surplus (the 3.3B written here earlier was zh_web's contract) |
| code_py_starcoder | — | 8.78B | surplus |
| arxiv | stand-in | 3.10B | role undecided |

## 2. Nodes

| # | node | owner / reviewer | exit number | date (UTC) |
|---|---|---|---|---|
| N1 | params-leg launch gate | fb / tilerl | exp row holds PEAK, STARTUP, `code_fp 6925ce02 <HEAD>` = NONE, both under 95.22 GiB; then launched | 09-03 |
| N2 | params-vs-data verdict | fb / b0 | one delta: domain_loss(params leg) − domain_loss(data leg) on the shared val cache, in nat, beside the paired per-item SE on the shared val cache (eval/domain_loss.py per-block paired mode, b0-23; b0-18 is the per-domain predecessor); seed σ is labelled unmeasured, not estimated; decision line names the 30B shape | DONE 2026-09-04 (b0-23): params leg − data leg = −0.010770 nat/token, paired SE 0.001655, t −6.51, 349/576 blocks down, params×tokens matched 1.0000×; gain in chatml/cot/chat_qa, code and en_c4 within noise, zh_web and code_py_rp1t reversed; seed σ unmeasured. Decision: 30B shape leans larger-parameter at fixed compute; code is a data lever. runs/b0_23_n2_verdict.json |
| N3 | benchmark v2 | e1 / b0 | three metrics only: humaneval_bpb, math_bpb (math_test_500 golds, build per eval_resolution_200m §ranked 2), lambada_en. Delta between two checkpoints carries the paired per-item SE (`eval/n3_report.py paired_se`, same ids, same bytes); seed σ stays unmeasured since N7 trains nothing. Per-checkpoint report is one line: bits/byte, bytes saved vs the previous point, gzip and Pythia beside it | 09-05 |
| N4 | 30B domains stamped | 3b / b0 | `mix_30b.json` `domains` non-empty for all 8; each domain has a data-auditor KEEP row, a 13-gram scan against eval golds (e1-28's scanner), and MinHash 0.8 cross-source dedup before its stamp | all 09-06 (code_tests dropped by the user's 09-04 weight-move decision) |
| N5 | post-pretrain ready | e1 / 44 | `post_pretrain_plan.md` §5 has zero open items: SFT pack path and row count, RL-gate script runs on a checkpoint, card-hour numbers from measured tok/s | 09-06 |
| N6 | deletion pass | de / 44, each owner in their area | §3 targets met; `reachability.py` unreferenced count and open-task count printed in the closing row; harness check `roadmap_pyramid`: every N row has a task row and every open task names an N row | 09-05 |
| N7 | middle-layer loop (user proposition 2026-09-03; SMELT arXiv 2609.01343, DeepLoop 2607.13491, `facts/smelt_deeploop.json`) | e1 / 44 | No pretraining arm: the 200M@1B delta the papers report, 0.019 nat, is inside seed noise at 3.6e19 FLOPs (SMELT's own 1e20 interval reaches 1%). Stage A, inference only, on the newest data-leg checkpoint: middle 4 of 12 layers visited twice, KV and KDA state from the last visit (user ruling), implemented as a forward wrapper so model.py stays untouched; ruler humaneval_bpb and domain_loss against the unlooped checkpoint, plus ms per token. Stage B, only if A regresses: the same SFT pack with and without the loop, ~500 steps each, same ruler. Enters the 30B shape only if B's looped arm beats its unlooped twin on both rulers | A 09-03 on the lane card, B 09-04 |
| N8 | document isolation in packed rows (found 2026-09-04 by e1-32 while verifying Stage C) | e1 / b0 | First divergence between a document scored alone and inside a packed row is block 0 (KDA), max 48.9 vs tolerance 0.93, two-document control 35.75, largest at each document's start and decaying inward (forget-gate signature), not growing with row position; doc 0 clean. Exit: the leaking line named (kernel vs model.py plumbing), fact eff.kda_document_isolation_violated with the table, then a fix ruling and a fixed-vs-current A/B at Stage D scale. Scope: every packed training run in the repo shares it; within-repo A/Bs stay valid, the phrase "document-masked" does not | line NAMED 09-04: model.py:109-113 short_conv left-pads the whole row and never sees cu; the kernel (fla 0.5.2 chunk_kda) honours cu_seqlens exactly (0.0000 on random inputs). Fix ruled (revised on e1's throughput reading: the multiply-add form exists for inductor fusion, 3.44x): keep the K shifted multiply-adds and zero each shift's tap at the first j positions of every document from cu, exact and fused, behind Cfg.conv_doc_isolated, old checkpoints auto-off (e1). Second finding (b0): eval/domain_loss.py:229 passes no cu at all, so every published per-domain loss was scored full-row causal across documents; measured (b0, 9 domains x 32 rows): pooled −0.0818 nat/token, 275/288 rows, 7.6x N2's delta, dose-response in documents per row (2 eos: −0.016..−0.034; 18 eos: −0.20..−0.28), so no constant correction. RULED: domain_loss.py passes doc cu (e1, after the model.py fix); every published per-domain value is re-scored on that path; old rows stay marked path=cu_none; then fixed-vs-current A/B at Stage D scale (b0). Fact eff.kda_document_isolation_violated |

Task rows: fb-5 fb-6 e1-29 3b-11 e1-30 de-43 e1-31 e1-32 (N7, reassigned from b0-22 13:4xZ), one per N row.

Out of scope until N2 lands: structure experiments (b0-12) other than N7, contrastive HumanEval (proposed
2026-09-03; BPB has resolution, so a second code instrument waits for BPB to saturate), RL
(after N5).

Metrics that leave the decision loop now, kept as tripwires and never reported beside the
three above: ceval (both formats), math_v2_like, lambada_zh two_way, the MC suite, code-500
pass@k on a base checkpoint. Basis: eval_resolution_200m.md §admission (chance+10pp at 154M).

## 3. Deletion pass, by area

| area | owner | delete | survivor |
|---|---|---|---|
| plans | 44 | the domain-contract tables in scale_36b_plan.md §1 and readout_30b_prereg.md that repeat `mix_30b.json._blocked` | `data/mix_30b.json`; docs cite it |
| tasks | every owner | every open task not on an N-row: state `dropped`, one line why | ≤ 15 open |
| runs/ | tilerl (tilerl-22, from b0) | measured 2026-09-04: 0 duplicate (ckpt, profile) keys (milestone and full are different metric sets); 0 exp rows deletable (later rows are corrections or attributed revisions, not supersets); 63 log snapshots KEPT since /work is an emptyDir and the commit is the only durable copy; 7 diverged snapshots refreshed | runs/tilerl_22_prune_list.md; nothing deleted |
| corpus | 3b | the second name of each of the 2,010 hardlinked shards; the 5 domains with no build_corpus_stats.json get a stamp or a deletion-list entry | one path per shard; every domain stamped |
| code | de | scripts/ (151) and eval/ (47) entries unreferenced by `reachability.py` AND absent from the AGENTS.md entry-point table, after the per-directory `glob`/`importlib` grep | entry-point table |
| facts | de | retracted entries with no live citation (de-5) | cited facts only |
| docs | 44 | lessons whose question is answered by a fact and cited nowhere | 66 → the cited set |

One commit per area. The commit names each deleted path and the survivor. A deletion that a
loader could reach at runtime needs the grep in the commit message.
