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
| N7 result (e1-31, 2026-09-03) | 2x2, each arm scored in its own topology: humaneval BPB unlooped-trained 0.4635 vs looped-trained 0.4658 (+0.0023, SE 0.0006, z 3.6, 103/164); domain_loss 1.9858 vs 1.9951 (+0.0093, 9/9). Mismatch cells are 8-11x larger (+0.0193, +0.0264): weights and topology must match, and Stage A (+0.0273) measured that mismatch, not the loop. Forward latency 1.64x. **Not adopted.** Boundary: 250 SFT steps on a 206M checkpoint rules out cheap adoption, not SMELT's from-scratch claim | runs/n7b_*.json, e1-31 |
| public models on the same 29,662 bytes (`humaneval_bpb.py --hf`, e1 2026-09-03; tokens as stated by model cards) | Qwen2.5-0.5B 494M/18T 0.2640; SmolLM2-360M 362M/4T 0.3624; SmolLM2-135M 134M/2T 0.4463; **ours 206M/8B 0.4567**; Pythia-160M 162M/300B 0.6024. Different points on different data curves: the number is good for the budget, it says nothing about which architecture is better | runs/n7_hf_*.json |
| N7 Stage A (inference-only loop, blocks 4-7) | humaneval BPB 0.4567 -> 0.4840 (145/164 worse, z 13.4); domain_loss 1.9443 -> 2.0609 (9/9); 1.64x ms/token forward-only, 1.002x in SFT training; parity 0. Stage B: SFT twin on control_sft_ours.pt, 250 steps per arm, running | runs/n7_domain.jsonl, e1-31 |
| curve so far | humaneval byte-weighted BPB 0.5451 / 0.5227 / 0.5199 at steps 7000 / 7500 / 10000; lambada_en 22.9 / 21.0 / 23.9 % | runs/score_matrix.jsonl |
| compute-matched control | held-out nat/byte floors ours 0.451 vs Pythia-160M 0.904; after SFT 0.294 vs 0.353 | docs/audits/control_pythia160m_vs_ours.md |
| 30B mix | `domains` is empty; all 8 domains sit in `_blocked` | data/mix_30b.json |
| backlog | 46 open tasks of 205 | runs/tasks.jsonl |

Supply against the 30B contracts (tokens, `facts/corpus_supply.json`):

| domain | contract | landed | gap |
|---|---|---|---|
| code_rp1t | 8.0B | 7.57B, unstamped | re-stamp |
| code_tests | 2.0B | 0 (Phase A mining running) | 2.0B |
| math_owm | 5.5B | 6.51B | none |
| cot | 4.5B | 0.42B | 4.1B |
| en_c4 | 5.5B | 4.801B exact in en_c4/ (frozen tokenizer, 3b 2026-09-03); the 2.40B was en_c4_stage2/'s 3-shard extrapolation | 0.70B, tilerl fetching ~5 files |
| textbook | 1.0B | 1.61B | surplus (the 3.3B written here earlier was zh_web's contract) |
| code_py_starcoder | — | 8.78B | surplus |
| arxiv | stand-in | 3.10B | role undecided |

## 2. Nodes

| # | node | owner / reviewer | exit number | date (UTC) |
|---|---|---|---|---|
| N1 | params-leg launch gate | fb / tilerl | exp row holds PEAK, STARTUP, `code_fp 6925ce02 <HEAD>` = NONE, both under 95.22 GiB; then launched | 09-03 |
| N2 | params-vs-data verdict | fb / b0 | one delta: domain_loss(params leg) − domain_loss(data leg) on the shared val cache, in nat, beside the paired per-item SE on the shared val cache (eval/domain_loss.py per-block paired mode, b0-23; b0-18 is the per-domain predecessor); seed σ is labelled unmeasured, not estimated; decision line names the 30B shape | params leg launched 12:52Z 09-03, ~6.8 h, ends ~19:45Z |
| N3 | benchmark v2 | e1 / b0 | three metrics only: humaneval_bpb, math_bpb (math_test_500 golds, build per eval_resolution_200m §ranked 2), lambada_en. Delta between two checkpoints carries the paired per-item SE (`eval/n3_report.py paired_se`, same ids, same bytes); seed σ stays unmeasured since N7 trains nothing. Per-checkpoint report is one line: bits/byte, bytes saved vs the previous point, gzip and Pythia beside it | 09-05 |
| N4 | 30B domains stamped | 3b / b0 | `mix_30b.json` `domains` non-empty for all 8; each domain has a data-auditor KEEP row, a 13-gram scan against eval golds (e1-28's scanner), and MinHash 0.8 cross-source dedup before its stamp | code_tests 09-05, all 09-06 |
| N5 | post-pretrain ready | e1 / 44 | `post_pretrain_plan.md` §5 has zero open items: SFT pack path and row count, RL-gate script runs on a checkpoint, card-hour numbers from measured tok/s | 09-06 |
| N6 | deletion pass | de / 44, each owner in their area | §3 targets met; `reachability.py` unreferenced count and open-task count printed in the closing row; harness check `roadmap_pyramid`: every N row has a task row and every open task names an N row | 09-05 |
| N7 | middle-layer loop (user proposition 2026-09-03; SMELT arXiv 2609.01343, DeepLoop 2607.13491, `facts/smelt_deeploop.json`) | e1 / 44 | No pretraining arm: the 200M@1B delta the papers report, 0.019 nat, is inside seed noise at 3.6e19 FLOPs (SMELT's own 1e20 interval reaches 1%). Stage A, inference only, on the newest data-leg checkpoint: middle 4 of 12 layers visited twice, KV and KDA state from the last visit (user ruling), implemented as a forward wrapper so model.py stays untouched; ruler humaneval_bpb and domain_loss against the unlooped checkpoint, plus ms per token. Stage B, only if A regresses: the same SFT pack with and without the loop, ~500 steps each, same ruler. Enters the 30B shape only if B's looped arm beats its unlooped twin on both rulers | A 09-03 on the lane card, B 09-04 |

Task rows: fb-5 fb-6 e1-29 3b-11 e1-30 de-43 e1-31 (N7, reassigned from b0-22 13:4xZ), one per N row.

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
| runs/ | b0 | score_matrix duplicate (ckpt, profile) keys (b0-15); killed-duplicate exp rows; committed log snapshots | one row per key; live logs stay on the pod |
| corpus | 3b | the second name of each of the 2,010 hardlinked shards; the 5 domains with no build_corpus_stats.json get a stamp or a deletion-list entry | one path per shard; every domain stamped |
| code | de | scripts/ (151) and eval/ (47) entries unreferenced by `reachability.py` AND absent from the AGENTS.md entry-point table, after the per-directory `glob`/`importlib` grep | entry-point table |
| facts | de | retracted entries with no live citation (de-5) | cited facts only |
| docs | 44 | lessons whose question is answered by a fact and cited nowhere | 66 → the cited set |

One commit per area. The commit names each deleted path and the survivor. A deletion that a
loader could reach at runtime needs the grep in the commit message.
