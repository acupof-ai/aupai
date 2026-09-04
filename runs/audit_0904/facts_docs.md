# Audit 2026-09-04 — facts and documents (area owner: 44)

View: host worktree /Users/bytedance/code/aupai-44 (branch 44) + pod via ~/bin/pod. Report committed 2026-09-04 03:49Z (afe2c0a0; the draft's "~12:40Z" was laptop local, UTC+8 — same zone class as the controller's return). All pod stats from this session.

## 1. Scope

Covered: all 11 `facts/*.json` (9 facts files = 421 entries; `facts/corpus_filters_baseline.json` and `facts/source_baseline.json` are dict-shaped auxiliaries, not facts — read, not censused), `docs/lessons/*.md` (63), `docs/audits/*.md` (10), `docs/standards/roadmap_0903.md`. Checkpoint citations joined against `runs/pod_ckpt_candidates_2026-09-04.txt` (311 candidates, 62 KEEP-claimed).

Excluded: the v14 hand-read (stopped per order), friction summary (deferred per order), eval code and score_matrix (e1's area), corpus supply counts (3b's area).

## 2. Method

- Census: entry/field survey over all 421 entries (status set, config presence, measured-field presence, retracted_value/refuted_by consistency).
- Source existence: b0's `runs/audit_0904/audit_fact_sources.py` (745c6a09) over all 9 facts files — 634 path-shaped source tokens. Its broken-world selftest passes (catches a missing citation, ignores prose, fails when a present artifact is removed). Pod paths pod-statted; repo paths checked on disk; `@sha` citations checked at commit.
- Checkpoint join: harness's own `_parse_ckpt_listing` / `_ckpt_names` (scripts/harness.py:2759) so the population matches the guard's exactly; second pass over all string fields.
- Sample: `random.Random(904).sample` over the 259 `status=measured` entries, n=30. Each recomputed from the cited artifact or marked not-recomputed with the reason.
- Docs: per-doc census of numeric-claim sentences vs inline fact-id/artifact citations; frontmatter source-field check over all 74 docs; hand-check of the 0%-inline docs.

## 3. Population counts

- 421 facts entries across 9 files; 259 measured; 421/421 config non-empty; 259/259 measured entries carry a `measured` field.
- 634 path-shaped source tokens: 618 resolved (repo, pod, KEEP-claimed checkpoint, @sha, deleted-in-reset recoverable by sha, or URL); 16 unresolved — the /tmp cluster (F9).
- Checkpoints: every fact-cited checkpoint is KEEP-claimed or pinned after today's pin (see S2-1).
- Docs: 74 files, 16,254 lines, 14,064 numeric-claim sentences; 74/74 declare a frontmatter `source:`.
- Sample: 30 drawn; 15 recomputed exactly, 4 partial, 11 not recomputed (named in §5).

## 4. Findings

| id | sev | claim as published | evidence | what contradicts / qualifies it |
|---|---|---|---|---|
| F1 | S2 | `eff.run_end_cursor_overstates_under_max_steps` depends on `ckpt_p200m_4b_0902.pt` and `.ep1` | fact's `source` is prose ("torch.load(mmap=True) of both checkpoints on the pod"), names neither file exactly; exact names + the 976,384/1,189,548 measurements live in `value`/`config`; `_ckpt_names(source+config)` is empty | `check_ckpt_facts_sources_present` has never seen this fact — its population (source+config, exact names) is narrower than the property (a fact's recomputable sources exist). The bare .pt is KEEP-claimed (fb); .ep1 was an unkept deletion candidate. I recomputed both sides from the pod at ~03:40Z (row_cursor dict sums: ep1 976,384 / as_of_step 3814; bare 1,189,548 / absent — exact match to config; the draft's "11:40Z" was laptop local, anchored between the v13 review row @03:03Z and this report's commit @03:49Z) and b0 pinned .ep1 as `ckpt_p200m_4b_0902.milestone_keep_44_runendcursor.pt`; the pin was verified on the pod at 04:01Z (nlink=2, hardlink of .ep1, identical size and mtime) ahead of the scheduled prune (the charter's "12:03Z" label carries the same zone class). Routed to 6e → de. |
| F2 | S3 | `ds.mathhard_resolution` source "aupai EXPERIMENTS.md + adversarial_review.md scoreboard" | `docs/lessons/adversarial_review.md` was added at 59d751b3 and deleted in the 0830v1 reset at 7359a56f (`git log --all`); recoverable by sha (`git show 59d751b3:docs/lessons/adversarial_review.md`) | The source is recoverable, not lost — citation hygiene: the fact should carry the sha so a reader does not have to know the reset deleted it. (Controller return 1: my "never in git" was wrong; fixed.) |
| F3 | S2 | `check_ckpt_facts_sources_present` is the guard for fact-cited checkpoints | scripts/harness.py:2759-2829; its docstring defends scanning source+config only ("a ckpt mentioned in a value … is prose") | F1 is the counterexample: the value mention IS the source claim because source is prose. Same shape as the §162 finding (a KEEP line invisible to the deleter) one level up — the dependency claim is invisible to the checker. de's area; recorded here because F1 surfaced it. |
| F4 | S3 | `ds.mde_recomputed_from_measured_sigma` = 0.1021 | `2.8 * 0.0516 * sqrt(1/4+1/4)` = 0.102164 → 0.1022 at 4dp | 1e-4 rounding slip. Not load-bearing. |
| F5 | — | 5 retracted entries carry empty `retracted_value` (`[]`): be.l1_3shot_retracted, cont.eval_surface_matching, cont.sft_all_code_holdout_leak, ds.second_resume_rereads_one_segment, eff.layer9_branch_scale_is_architectural | AGENTS.md:369 documents exactly these five as the deliberate common case — "`[]` is a real answer": the conclusion was retracted while the numbers stand | Not a defect; consistent with the documented rule. Withdrawn as a finding. (Controller return 2.) |
| F6 | S3 | `eff.attn_every_1_has_no_position_information` = 1.086, from "p02_a1_s2 val 4.765 vs p02_s2 val 3.679" | arithmetic 4.765−3.679 = 1.086 ✓; but neither val number is in any artifact — p02 training logs absent from pod and repo, experiments.jsonl rows carry no val, score_matrix has different instrument readings (5.2433/3.7433, 7-domain eval) | The value is internally consistent; its two base numbers rest on the source string alone. |
| F7 | S3 | `eff.nvidia_smi_minus_torch_is_not_a_transient` = 34.31 | config carries b8 torch 36.56 / smi 70.87 / gap 34.31, b16 torch 72.12 / smi 73.31 / gap 1.19; probe logs `runs/peak_b{8,16}_d1536*.log` confirm torch reserved (36.56, 72.12 ✓) but record NO nvidia-smi readings | The torch side reproduces; the nvidia-smi side exists only in the fact's config, not in the cited logs. |
| F8 | S3 | `eff.cache_load_gates_startup` = 386s from launch to first step | `runs/pretrain_15b_s1.log`: launch W0831 12:45:05; training step lines (line 218+) carry no timestamps; no timestamped line between | Unverifiable from the cited artifact. |
| F9 | S3 | `mlm.fertility.traditional` = 1.161, source `/tmp/fert_pairs.json` | /tmp is ephemeral; the file is gone | The measurement rests on the fact alone. Same class as the ~17 /tmp instrument citations across contamination/data_quality/multilingual facts. The four `status=measured` contamination facts in this class — `cont.scanner_idf_weighting`, `cont.gsm8k_zh_webhq_scan`, `cont.math500_webhq_fp_explained`, `cont.code_holdout_carved` — are e1's E14 (eval_heldout.md @6f96da60), rated S2 there because a measured fact cannot be re-derived; this row covers the remaining (recorded-status / non-contamination) citations and points at E14 rather than duplicating it. |
| F10 | S2 | `dq.agentic_credential_split` (measured 2026-09-04): "of the 866, 863 are opaque-ONLY and **3** also carry a REAL_CREDENTIAL (GitHub Token 1, IBM Cloud IAM Key 1, Private Key 1)" | The fact's own `config.real_credential` names the instrument: "find_secrets (detect_secrets, chunk=1) types intersected with REAL_CREDENTIAL's 22 provider rules". e1's E16 (eval_heldout.md): the v14 build over a comparable population (9,060 admitted rows vs the fact's 10,000-episode cache) found **62** REAL_CREDENTIAL episodes, **59 of them invisible to find_secrets** — caught only by the new CLI/env tier — and the 3 legacy types E16's build found via the legacy detector are exactly the fact's 3. | The published 3 is what the named instrument could see, not the credential count (direction certain, ratio approximate per E16). The fact's `overlap_scope` already concedes the 3 is an overlap count, and its `boundary` names the non-opaque-episode silence — but not the instrument's CLI-flag/env-assignment blind spot, which is the one that bit. Post-audit (no edit now, audit only): the fact's boundary gains "instrument blind to CLI-flag and env-assignment forms; recount 62 on v14 population". |

What held (the point of the audit): 421/421 configs non-empty; 3/3 artifact_sha256 verified byte-exact on the pod (be.l1_3shot_15b, be.l1_3shot_24k, be.l1_8demo_format_collapse); 618/634 source tokens resolved; the seed-904 sample's one apparent disagreement (ds.beta_data_leg_206m_early: my naive log-log fit gave 0.1010 vs published 0.5909) was MY model being wrong — the project's 3-param model `E+(B/D)^β` on the same 12 val points gives β=0.5909 exactly. `refuted_by` is a falsification condition, not a refutation (37 entries, convention verified). The checkpoint join found zero unkept fact-cited checkpoints after the F1 pin; the 3 section-A `[zeroed]` overlaps (be.known_answer_panel_3_4, be.lambada_zh_heldout_3p24b, be.math_v2_5k_known_answer → ckpt_0830v1_3.24b.pt.ep1) are WARN-tier by design (`_noted_gone`=True) — the check is consistent with its green state.

Docs: no doc is source-less (74/74 frontmatter). Traceability is doc-level (one source declaration per doc), not sentence-level — 14,064 numeric sentences, 4% carry an inline citation, and the 0%-inline docs hand-checked all name facts or artifacts in their frontmatter (base_eval_panel, kernel_vs_process, throughput_survey, instrument_not_system). Whether every number in a 500-line doc actually comes from its declared source was not verifiable in 3h.

## 5. Blind spots

- 11 of 30 sample items not recomputed: GPU reruns (be.ctx_length_p324), 2.6GB parquet scan (cs.code_supply_pair_yield), 166-shard token count (cs.en_c4_landed), external APIs (mlm.src.skypile, fineweb_edu_zh_v21), sft_all.pt span analysis (be.agentic_loop_tokens), the cont.union aggregate (per-batch scan logs exist, not summed), chi2 pairing (be.l1_demo_lang_null), qualitative drift (be.preamble_is_not_the_cause), chatml supply resample, and F8's timestamp gap.
- Pod-only artifacts were stat-verified; content was re-read only for the sample's 15 recomputes and the 3 sha256.
- de was unreachable at audit time (no de session in ListAgents); the pair check (de recomputes 3 of these, I recompute 3 of instruments_ledgers.md) is pending. 6e routed F1/F3 to de at uds:/tmp/cc-socks/39861.sock.
- The two dict-shaped auxiliary files were read but not censused as facts; `source_baseline.json` self-declares 3 non-durable sources (runs/p02_seeds, runs/warmup_smoke, data/mix_scale_ prefix) — all 3 confirmed absent from the repo (pod-only or prefix).

## 6. Process breach (self-reported)

2026-09-04 ~04:1xZ: while merging de's updated report I ran `git stash push -- <file>` once, against the no-shared-stash rule (stack is shared across worktrees); dropped it within the same command and re-applied the edit via Edit instead. Stack verified empty afterwards (controller). No ledger row during the freeze.

## 7. Open questions for the controller

1. Should a fact whose `value`/`config` names a checkpoint be required to name it exactly in `source` or `config` (F1/F3 class)? A lint on `_ckpt_names` over all fields would close it.
2. Ban `/tmp` instruments as fact sources (F9), or accept value-in-fact as the record?
3. Should training logs timestamp their step lines (F8)? The 386s claim cannot be checked without it.
4. Is doc-level (frontmatter) traceability accepted, or do decision-driving numbers need sentence-level citations?
