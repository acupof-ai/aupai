---
question: After the 2026-09-04 audit, what is true now — what stands, what is retracted or qualified, what is unmeasured?
status: measured
source: joined from the seven reports under runs/audit_0904/ and the area owners' lists (3b, 98, b0 received verbatim; other four areas derived from their reports, no new claims); cleanup state per runs/audit_0904/cleanup.jsonl
---

# State of record, 2026-09-04

One line per item. `stands` = survives with its fact id or ledger row; `qualified` = survives only with the named boundary (finding id); `unmeasured` = named, not silence.

## Decisions

- N7 middle-layer loop: **NOT ADOPTED** — the equal-compute competitor beats it on three rulers (`repo.loop_not_adopted_equal_compute`).
- Stage C prefix mask: **NOT ADOPTED** — bidirectional prompt at inference costs 7x what prefix training gains (roadmap_0903.md N7 row; EXPERIMENTS.md n7c_p3/p7; no fact id exists).
- N8 `conv_doc_isolated` fix: **ENTERS** the recipe for correctness, HumanEval disagreement recorded (`eff.conv_doc_isolation_ab_200m`).
- N2 "larger params at fixed compute": **SUSPENDED** until the doc_cu re-score (C11) (`ds.n2_params_vs_data_matched_compute`).
- code_tests: **folded into code_rp1t** — 100.00 % of mined pairs already in code_py_starcoder (`cs.code_tests_supply`); the retired 2.0B target had been justified by `ds.code_exec_supply`, a fact id that never existed anywhere in the tree (40e907af; the dangling citation is recorded in `cs.code_tests_supply`'s note).

## model and training code (b0, model_training.md + owner list)

- **stands**: conv document isolation arithmetically sound, bitwise gate 576/576 (`scripts/b0_n8_reuse_gate.py`); the flag-off branch is deliberately not the masked form, so pre-flag checkpoints score bitwise as they trained; `eval/domain_loss.py` cu path fixed, `cu_none` default deliberate; sft_math head-slice aliasing safe — no `.data` rebind anywhere, rollback copies in place, `data_ptr()` matched after rollback; `train.py:2059` head_dim=128 guard makes both Stage E shapes legal by the guard, not luck; 40 of 124 facts sampled, **0 contradicted values** (6 HELD, 21 PARTIAL each hand-classified as a number the cited artifact cannot carry, 13 NO-FILE); `eff.run_end_cursor_overstates_under_max_steps` reproduces exactly (976,384 / 1,189,548); `eff.seam_dynamo_disable` 70→0 and `eff.ckpt_resume_16h_interval` 5.874/5.841 reproduce on main; Stage E arm 1 completed (train 1.860 val 2.218, 669,593,333 B, n_params 154.665288M matching its ledger prediction; score_matrix 8/9 metrics) — its ledger row still reads "running" though it finished 04:13Z (C4).
- **qualified**: MT-1 `eff.kv_pool_undersized_for_serving` — describes code superseded 2026-09-02 (48ae458), refuted by its own criterion; **C7 done**: retracted (the 2026-08-30 reading preserved verbatim as correct-when-measured) and superseded by `eff.kv_pool_follows_context` — 8.0x the declared budget at every context tested (ff035f77); MT-2 15 facts cite pod-only artifacts, 8 uncommittable (>5 MB) — pod-citation convention needed; MT-3 `CANONICAL_STATS_KEYS` governs 1 of 4 stamp writers; MT-7 recipe_provenance gap — four Stage D runs ran with no recipe group, twelve argv values sourced only after the fact (group added fd6a382a); MT-9 30/40 sampled facts not recomputable by a reader on main (9 need the pod, 11 a card, 4 literature-only, 6 cite a script in no commit); MT-4/MT-5 launch_tests last-write-wins + pass-after-assert — **fixed since** (shape-keyed rows); any "arch tests passed" claim before C10 needs its shape named.
- **unmeasured**: domain_bpb for every checkpoint (MT-12, see evaluation area); MT-10 sft pack vocab_id assert never fired in either state, zero SFT runs since 09-03, correctness rests on a code reading; MT-11 NaN rollback discards up to 199 steps with nothing recording the distance (`grep -c good_step` = 0); MT-13 `loader.py:136` vocab_id warning fires on every `load_tokenizer(path, None)` and has never carried information; Stage E arms 2-3 parked, no card, no numbers — the gap-widens question is unanswered; fp8 numerics (conversion population and scale plumbing checked, not the numerics).

## evaluation and held-out (e1, eval_heldout.md)

- **stands**: no checkpoint trained on any agentic pack — four whole-population readings (E16); the cu fix landed 2026-09-03T22:41:12Z; N2 verdict survives its timestamp correction by a wider margin (E3 pair check); contamination scan populations as enumerated.
- **qualified**: N2 verdict (`ds.n2_params_vs_data_matched_compute`) measured on **cu_none**, doc_cu pending (E3, C8.1); `eff.eval_path_cu_artifact_ce` cancellation is first-order — 28% residual on the one cross-path rescore (E2, C8.2); four measured contamination facts instrument-lost (E14, C8.4); `dq.agentic_credential_split`'s 3 is what find_secrets could see, recount 62 on v14 (E16, C8.3); MT-12/E18 `domain_bpb` has never produced a published number (0 of 60 score_matrix rows; 6 shadowed by the stderr-or-stdout capture, 3 refused legibly and unactioned, 1 truncated), so the control-arm cross-tokenizer comparison is UNMEASURED until C6 — the same capture defect touches humaneval_bpb (1 path), minimal_pairs (all failures), l1_fewshot (SIGTERM rows).
- **unmeasured**: doc_cu re-score of the checkpoints behind published numbers (C11, the one card job, runs last); per-number derivation inside docs (facts area blind spot).

## instruments and ledgers (de, instruments_ledgers.md)

- **stands**: harness check states at 6d250424 (laptop 0 FAIL of 60, pod 0 FAIL of 55); 79 CHECKS enumerated; ledger census (38 local / 54 pod jsonl; 233 task ids); DL-4/5/6 fixes verified on main.
- **qualified**: DL-1 `no_conflict_markers` SKIP on the pod 20 consecutive runs (16.4 s vs 5 s deadline) — C5; DL-2 `walk_tracked` yields 489 of ~850 (`.jsonl` 196/0) — C5; DL-3 `tasks_stale` blind to prose `blocked_on` (11/62 open) — C4; DL-7 three hook selftests ungated; DL-8 `_ckpt_names` reads source+config only — C5; DL-9 zombie pids keep claims alive; DL-10 `lane_respected` card-set membership wrong; DL-11 four ledgers have no transport either direction — C4; DL-12 `artifact_refs` two schemas, no row identity; DL-17 `gate_cards` cannot see a stale grant; DL-20 `check_test_record_after_last_stage` unregistered — C5.
- **unmeasured**: population-vs-rule for the 49 unsampled checks; 19 checks run nowhere on the laptop, 24 not on the pod.

## corpus and data (3b, corpus_data.md + owner list)

- **stands**: 11-domain supply matches `facts/corpus_supply#cs.*_landed` (two 0.3–0.4% gaps are 3-shard sampling); zh_web / code_rp1t fingerprints MATCH live shards; no undersupply on either mix at row level; data/raw healthy 263 GB; en_c4 + math_owm ws-13 scans clean (`cont.en_c4_ws13`, `cont.math_owm_ws13`).
- **qualified**: CD-1 zh_web (21.29B) / textbook_30b / wiki_chat supply not reproducible from stamps — extrapolated, basis missing; CD-2 no stamp answers which holdout a corpus was built against; CD-4 `filters_fp` cannot identify the filter profile; CD-7 `CANONICAL_STATS_KEYS` covers 1 of 4 writers.
- **unmeasured**: who wrote the zh_web 7-key stamp; fingerprints for the other 9 domains; zh_web raw provenance; re-tokenize for the 3 minimal-schema domains.

## pod and repository state (tilerl, pod_repo_state.md)

- **stands**: pre-prune baseline 403 checkpoints / 317.3 GB (runs/audit_0904/ckpt_pre_prune_0352Z.tsv); 22 milestone KEEP pins; PR-9 153 rolling `.stepN`, 12 inode-pinned (the one live-fact source now pinned); Stage E arm 1 finished normally into its scoring chain; **C1 done** — 314 processes killed by exact PID (306 orphaned `tail -F` + 3 wait-loops + 5 stale tails), 0 remaining in all three classes, 8 cards 0 MiB at kill time (ff035f77; pair-checked 05:42Z); **C7 done** — `eff.kv_pool_undersized_for_serving` retracted and restated as `eff.kv_pool_follows_context` (8.0x, tileRL 48ae458); PR-6/7 drift counts reconciled — the manifest is rewritten by every push (469 at 03:50Z → 481 → 482), so "N outside the manifest" is checkable only with the population rule beside the count: PR-6 counts files tracked in main (197 outside 469), tilerl counts everything on disk the manifest does not name (130 outside 481).
- **qualified**: PR-1 and PR-3 were drafted S1 and restated S2 — the eleven findings are six S2 / five S3, no S1; PR-1 "foreign card 7" — in-container, tilerl's own checkout, job ended on its own; PR-2 foreign/ours decided from prose markers, not cgroup; PR-4 /work at 96 %, 90 GB free; PR-8 8 absent fact checkpoints overlap the existing harness WARN; PR-11 — the "not ours" classification of the 307 tails is **retracted** (C1.5: cwd `/sgl-workspace/sglang` is the container default and carries no ownership; positive evidence is ours — fd 3 → our events file, reader-less stdout, PPid 0); the stale tails were FIVE, not six (the sixth was a grep self-match); 3,958 zombies (~300/day, ppid 1 = `sleep infinity` never reaps; 3,971 at 05:42Z) are unreapable from inside the container — a reaping PID 1 or a container restart is the only fix, recorded as a post-cleanup user decision, none applied.
- **unmeasured**: the 168 untracked pod files individually; `/work/tl-ab` repository state; nlink≥2 verification of the 12 pinned inodes.

## facts and documents (44, facts_docs.md)

- **stands**: 421 facts, configs 100 % non-empty; 3/3 `artifact_sha256` byte-exact on the pod; 618/634 source tokens resolved; seed-904 sample 15/30 exact (incl. β=0.5909 under the project's own model); 74/74 docs declare a frontmatter source.
- **qualified**: F1 `eff.run_end_cursor_overstates_under_max_steps` prose-source dependency (pinned; checker blind — C5); F2 `adversarial_review.md` citation now sha-anchored (C8.5); F4 `ds.mde` 0.1021 vs computed 0.1022; F6 `attn_every_1` base vals in no artifact; F7 nvidia-smi readings only in fact config; F8 `cache_load_gates` 386 s unverifiable; F9 ~17 /tmp instrument citations (the 4 measured ones are E14); F10 credential split — see evaluation area.
- **unmeasured**: 11/30 sampled items (GPU reruns, 2.6 GB scan, external APIs); per-number derivation inside docs; de's reverse pair check of this report.

## user-facing statements (98, user_facing.md + owner list)

- **stands**: 10/30 sampled store lines trace to the digit (mix weights, 69.63 GiB gate, lr probe, step40 ckpt bytes, ab_untie_head, data-leg timing, launch NO-GO, 1.84×/2.7×, domain_bpb crash); board rule/block hard numbers (94.7→35.5 TFLOPS; constant-answer 9.78/8.13/5.69 %, z=8.42; format rates 33.5/80.5 %); EXPERIMENTS.md render faithful to exp.py; UF-2's "333 分片" (127+206).
- **qualified**: UF-1 94.4 %/0.3 % ChatML pair — retracted, no source, AGENTS.md removal note; UF-2 humaneval 15614/GB, lambada_en 18106/GB — void, unit no instrument produces; UF-3 N7 "全面变差" causal claim + 1.64× — retracted; Stage A figures stand, "不采用" survives; UF-7 roadmap cites pod-only `runs/n7_domain.jsonl` — figure stands, source unreachable from the repo.
- **unmeasured**: 560/590 numeric store lines (retracted-fingerprint grep only); page history before 2026-09-01 16:30 unrecoverable; store day assignment inferred (UF-4, C9 forward-only); 71 board find/done/note rows; status cards.
