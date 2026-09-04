---
question: After the 2026-09-04 audit, what is true now — what stands, what is retracted or qualified, what is unmeasured?
status: measured
source: joined from the seven reports under runs/audit_0904/ and the area owners' lists (3b, 98 received; other five areas derived from their reports, no new claims); cleanup state per runs/audit_0904/cleanup.jsonl
---

# State of record, 2026-09-04

One line per item. `stands` = survives with its fact id or ledger row; `qualified` = survives only with the named boundary (finding id); `unmeasured` = named, not silence.

## model and training code (b0, model_training.md)

- **stands**: conv document isolation arithmetically correct, bitwise gate 576/576 identical; fp8 conversion population exact at both shapes (64 linears/63 converted, 66/3); `eval/domain_loss.py` cu path fixed, `cu_none` default deliberate; `eff.run_end_cursor_overstates_under_max_steps` reproduces (976,384 / 1,189,548, pinned); 40 seed-904 facts recomputed, **0 contradicted values**; `eff.seam_dynamo_disable` 70→0 and `eff.ckpt_resume_16h_interval` 5.874/5.841 reproduce from logs.
- **qualified**: MT-1 `eff.kv_pool_undersized_for_serving` — describes code superseded 2026-09-02 (48ae458), refuted by its own criterion (C7 retraction); MT-2 15 facts cite pod-only artifacts, 8 uncommittable (>5 MB) — pod-citation convention needed; MT-3 `CANONICAL_STATS_KEYS` governs 1 of 4 stamp writers; MT-4/MT-5 launch_tests last-write-wins + pass-after-assert — **fixed since** (shape-keyed rows); MT-7 recipe_provenance gap (12 values, 4 runs); MT-9 30/40 sampled facts not recomputable by a reader on main; MT-10 sft vocab_id guard never fired in either state; MT-11 NaN rollback discards up to 199 steps with no record.
- **unmeasured**: sft_math masking internals; fp8 numerics (needs a card); train.py cursor arithmetic beyond cited facts; the 11 named needs-card facts; filter equivalence vs `filters_fp` hash.

## evaluation and held-out (e1, eval_heldout.md)

- **stands**: no checkpoint trained on any agentic pack — four whole-population readings (E16); the cu fix landed 2026-09-03T22:41:12Z; N2 verdict survives its timestamp correction by a wider margin (E3 pair check); contamination scan populations as enumerated.
- **qualified**: N2 verdict (`ds.n2_params_vs_data_matched_compute`) measured on **cu_none**, doc_cu pending (E3, C8.1); `eff.eval_path_cu_artifact_ce` cancellation is first-order — 28% residual on the one cross-path rescore (E2, C8.2); four measured contamination facts instrument-lost (E14, C8.4); `dq.agentic_credential_split`'s 3 is what find_secrets could see, recount 62 on v14 (E16, C8.3).
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

- **stands**: pre-prune baseline 403 checkpoints / 317.3 GB (runs/audit_0904/ckpt_pre_prune_0352Z.tsv); 22 milestone KEEP pins; PR-9 153 rolling `.stepN`, 12 inode-pinned (the one live-fact source now pinned); Stage E arm 1 finished normally into its scoring chain; PR-6/7 drift counts reconciled (population rule named: 197 = 135 + data/ + ext-set).
- **qualified**: PR-1 "foreign card 7" — in-container, tilerl's own checkout, job ended on its own; PR-2 foreign/ours decided from prose markers, not cgroup; PR-4 /work at 96 %, 90 GB free; PR-8 8 absent fact checkpoints overlap the existing harness WARN; PR-11 3,958 zombies (~300/day, ppid 1 never reaps), 307 foreign `tail -F` holding our events FD, 3 non-terminating wait-loops — C1.
- **unmeasured**: the 168 untracked pod files individually; `/work/tl-ab` repository state; nlink≥2 verification of the 12 pinned inodes.

## facts and documents (44, facts_docs.md)

- **stands**: 421 facts, configs 100 % non-empty; 3/3 `artifact_sha256` byte-exact on the pod; 618/634 source tokens resolved; seed-904 sample 15/30 exact (incl. β=0.5909 under the project's own model); 74/74 docs declare a frontmatter source.
- **qualified**: F1 `eff.run_end_cursor_overstates_under_max_steps` prose-source dependency (pinned; checker blind — C5); F2 `adversarial_review.md` citation now sha-anchored (C8.5); F4 `ds.mde` 0.1021 vs computed 0.1022; F6 `attn_every_1` base vals in no artifact; F7 nvidia-smi readings only in fact config; F8 `cache_load_gates` 386 s unverifiable; F9 ~17 /tmp instrument citations (the 4 measured ones are E14); F10 credential split — see evaluation area.
- **unmeasured**: 11/30 sampled items (GPU reruns, 2.6 GB scan, external APIs); per-number derivation inside docs; de's reverse pair check of this report.

## user-facing statements (98, user_facing.md + owner list)

- **stands**: 10/30 sampled store lines trace to the digit (mix weights, 69.63 GiB gate, lr probe, step40 ckpt bytes, ab_untie_head, data-leg timing, launch NO-GO, 1.84×/2.7×, domain_bpb crash); board rule/block hard numbers (94.7→35.5 TFLOPS; constant-answer 9.78/8.13/5.69 %, z=8.42; format rates 33.5/80.5 %); EXPERIMENTS.md render faithful to exp.py; UF-2's "333 分片" (127+206).
- **qualified**: UF-1 94.4 %/0.3 % ChatML pair — retracted, no source, AGENTS.md removal note; UF-2 humaneval 15614/GB, lambada_en 18106/GB — void, unit no instrument produces; UF-3 N7 "全面变差" causal claim + 1.64× — retracted; Stage A figures stand, "不采用" survives; UF-7 roadmap cites pod-only `runs/n7_domain.jsonl` — figure stands, source unreachable from the repo.
- **unmeasured**: 560/590 numeric store lines (retracted-fingerprint grep only); page history before 2026-09-01 16:30 unrecoverable; store day assignment inferred (UF-4, C9 forward-only); 71 board find/done/note rows; status cards.
