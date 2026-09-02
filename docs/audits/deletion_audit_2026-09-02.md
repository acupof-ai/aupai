---
question: What in this repository can be deleted or merged, with per-file proof, so it gets smaller and reuses more?
status: recorded
source: read-only audit by a subagent, 2026-09-02, main 0622878; greps G1/G2/G3 per file are in the tables
---

# Deletion / merge audit — aupai @ main 0622878, 2026-09-02 (read-only)

Baseline: 645 tracked files; 91,142 py lines; `scripts/harness.py` 10,172; `mathbank/` 40,563; `scripts/` 22,657 (83 files); `datagen/` 10,163; `eval/` 7,467; `probes/` 4,311.

Method. (1) `python scripts/reachability.py` live run, worktree noise (`.claude/worktrees/wf_1c60d38d-c68-12/`, an untracked stale worktree that the scanner walks) filtered out → `scratchpad/reach_now.txt`: 527 files, 94 entry points, 271 reachable, **39 code files with REACHED FROM = none (4,836 lines)**. (2) Per-file citation matrix over every tracked `.py/.sh/.yml/.json` (code), `.md` (docs), `facts/*.json`, `runs/*` and `scripts/hooks/pre-commit`, matching basename and stem → `scratchpad/citation_matrix.tsv`. (3) Runtime loaders: `grep -rn -E 'glob\(|glob\.glob|iglob|rglob|importlib|__import__|import_module|spec_from_file' --include='*.py' --include='*.sh' .` (worktree excluded). Every loader that names a directory:

| loader | file:line | pattern | files it makes live |
|---|---|---|---|
| mathbank registry | `mathbank/run_math_short.py:37-39`, `mathbank/vet_programs.py:34-36` | hardcoded `math_programs_l{1,2,3,4}` + `glob("math_programs_l*_ext*.py")` → `importlib.import_module` | 4 base + **21** ext (not 23: `git log --diff-filter=D -- mathbank/math_programs_*` is empty, count was 21 at the commit that wrote 23) |
| short-solution registry | `mathbank/run_short_sol.py:25,32` | hardcoded list → importlib | `math_programs_short_l3.py`, `math_programs_short_l4.py` |
| RL registry | `algorithms/__init__.py:46-48` | `import_module(f".{mod_name}")` | `algorithms/rlvr_*.py` |
| eval registry | `eval/run_eval.py:18,49,137-153` | `import_module(f"eval.{name}")`, table `MC_BENCHMARKS` | `eval/{arc,boolq,ceval,gsm8k,hellaswag,mmlu,openbookqa,piqa,winogrande}.py` |
| train.py | `train.py:1632,1750,2635,2646` | globs corpus shards and `ckpt*.step*` | data, not code |
| probes | `probes/t71_depth_lr_rule.py:75` (`spec_from_file_location(... "train.py")`), `probes/{chatml_in_corpus,t62,t63,t64,t65,t70}` | glob over `data/corpus/*` | data, not code |
| infer_local.py:42 | `glob("ckpt*.pt")` | data |

Stale committed listing: `runs/reachability.txt` (197 lines) says 187 files / 0 unreachable and `scripts/harness.py` = 5,940 lines; live = 527 files / 253 unreachable rows / harness 10,172. It is cited by `docs_commands_exist`-style rules as "the committed listing with fate rulings" (AGENTS.md entry-point table) and has not been regenerated since 2026-08-31. Regenerate or drop the committed copy.

---

## 1. Deletable now — not reachable, not cited, not globbed

Proof per file = the three greps below returned only the file itself (or nothing). `G1` = stem grep over the whole tree, `G2` = basename grep, `G3` = the loader grep above (no loader names `scripts/`, `probes/`, `bench_eff/`, `datagen/` by directory). Commands (run from repo root, `X` = path, `S` = stem):

```
G1: grep -rlw --exclude-dir=.claude --exclude-dir=.git --exclude-dir=data --exclude-dir=runs "S" .
G2: grep -rl  --exclude-dir=.claude --exclude-dir=.git "X" .        # includes runs/ facts/ docs/ hooks
G3: grep -rn -E 'glob\(|importlib|__import__|import_module' --include='*.py' --include='*.sh' <dir>
```

### 1a. Zero references in code, docs, facts, runs, hook (39 candidates → 24 after manual exclusion)

| file | lines | last commit | G1/G2 hits besides self | note |
|---|---:|---|---|---|
| `datagen/measure_duplication.py` | 160 | b40ac28 09-01 "measure: corpus duplication rates … P0" | none | one-shot P0 measurement; result not in any fact (facts cite `probes/t62/t63` for dup rate) |
| `datagen/executable_yield.py` | 112 | cb3fd36 09-01 | none | one-shot count on code_rp1t; result not in facts |
| `datagen/ast_parse_survivors.py` | 74 | 0f16bf3 09-01 | none | superseded by `executable_yield.py` (same measurement, "catch parser-stack-overflow giants") |
| `datagen/test_near_dedup_identity.py` | 87 | 0db14ad 09-01 | `runs/review.jsonl` only | test of `build_corpus` near-dedup, run by nothing (not CI, not hook) |
| `scripts/test_eval_rescore.sh` | 66 | 616205b 09-01 | none | test of `eval_math.sh`/`eval_code.sh` rescoring; run by nothing |
| `scripts/test_monitor_exit.py` | 145 | f292f53 09-01 | none | harness monitor test; not in CI, not in hook `SELFTEST_FILES` |
| `scripts/test_domain_loss_val.py` | 137 | a153681 09-01 | none | same |
| `scripts/test_math_passk.py` | 94 | 683496f 09-01 | none | same |
| `scripts/test_resume_cursor_pod.py` | 181 | 6a456cb 09-02 | none | pod-only test; run by nothing |
| `scripts/test_cursor_save.py` | 191 | 1c8f0c4 09-01 | none | same |
| `scripts/test_rep_stop_flag.py` | 92 | bd1e005 09-01 | none | same |
| `scripts/test_arch_L32.py` | 325 | e83fe5d 09-01 | none (`launch_tests.py` reads a `test_file` field from `runs/launch_tests.json`, does not name it) | run by nothing |
| `scripts/rehearse_cursor.py` | 111 | 0e4c525 09-01 | `runs/review.jsonl` only | rehearsal superseded by `scripts/replay_cursor.py` (same commit) |
| `scripts/stamp_cache_seeds.py` | 63 | 1a439ee 09-01 | none | one-shot migration ("seed stamps written explicitly") |
| `scripts/sweep_stale_rows.py` | 60 | 1eb5660 09-01 | none | one-shot sweep, already applied ("12 stale rows swept") |
| `scripts/probe_gradckpt_sources.py` | 94 | f78f22b 09-01 | none | one-shot probe; conclusion in commit message, not in facts |
| `scripts/ab_attnres.sh` | 52 | dec5bcb 09-01 | none | launcher for an A/B superseded by `scripts/run_ablation.sh` (AGENTS.md row) |
| `scripts/lr_probe.sh` | 52 | 4407c20 09-01 | none (`data/mix_probe_lr.json` cited by `write_mix_500m.py`) | 500-step lr A/B launcher; its log reader `read_lr_probe.py` is hook-only (1b) |
| `scripts/watch_pod_log.sh` | 19 | f24669b 09-01 | none | |
| `scripts/test_orphan_kill.sh` | 27 | d87feb4 08-31 | none | tests `run_ddp.sh` kill path — **post-run** (frozen file) |
| `scripts/test_kill_pairing.sh` | 37 | 2aa35c9 08-31 | `runs/retro.jsonl` only | same — **post-run** |
| `probes/t57_outlier.py` | 33 | 2faa10f 09-01 | none | |
| `probes/t70_external_loss.py` | 180 | be8e94a 09-01 | `docs/lessons/copy_hypothesis_prereg.md` names it; output `runs/external_loss_step24000.json` | one-shot probe, measured; doc citation is prose not a command block → `doc_commands_exist` unaffected only if the mention is not in a code fence (it is prose: keep the doc line, delete the probe, or move the number into a fact) |
| `scripts/test_parallel_encode.py` | 38 | 8e7c2d4 08-31 | `facts/efficiency.json` source field | see 2 |

Subtotal 1a (excluding the two post-run and the fact-cited one): **21 files, 2,376 lines**, zero risk to `harness check` (none is in AGENTS.md, CI, or the hook map). One check to re-run after: `python scripts/harness.py check` (`entrypoints_ran`, `doc_commands_exist`, `selftests_are_gated`).

Excluded from 1a after the second grep (NOT deletable): `datagen/t2s_corpus.py` (imported by `datagen/clean_web.py:67,135`), `datagen/data_overview.py` (imported by `datagen/check_mix.py:24`), `scripts/run_sampled_cell.sh` (named in `scripts/harness.py`), `scripts/progress_feed.py` (controller's progress page writer, operational, uncited), `datagen/build_chat_qa.py` + `datagen/build_chatml.py` (144 lines, uncited, but they built the live `chat_qa`/`chatml` domains of `data/mix_500m.json` — record them in `data/PROVENANCE.md` rather than delete), `scripts/bootstrap_pod.sh` (self-cited only, but it is the pod bootstrap).

### 1b. Cited only by the pre-commit hook's `SELFTEST_FILES` map (run only when the file itself is staged)

`scripts/hooks/pre-commit:218-249`. `check_selftests_are_gated` (`scripts/harness.py:562`) asserts every file *carrying* `--selftest` is in the map; it does not assert map entries exist, so deleting a file plus its map line is clean. These are tests nothing schedules: not CI (`.github/workflows/ci.yml`), not `harness check`.

| file | lines | last commit | tests what |
|---|---:|---|---|
| `scripts/test_shard_glob.py` | 178 | cbaa12d 09-01 | train.py shard glob — **post-run** |
| `scripts/test_pod_sync_stamp.py` | 121 | 8479815 09-01 | hook/pod_drift stamp |
| `scripts/test_check_summary.py` | 54 | aded6c5 09-01 | harness check summary line |
| `scripts/test_drop_zombies.py` | 59 | 79dfda7 09-01 | harness kill path |
| `scripts/test_num_id_resolve.py` | 62 | aef7dbb 09-01 | `[NUM]` id derivation |
| `scripts/test_spawned_fast.py` | 87 | 8548ac8 09-01 | harness spawned-import resolver |
| `scripts/read_lr_probe.py` | 200 | 81a53be 09-02 | reads `runs/lrprobe_*.log`; pairs with `lr_probe.sh` (1a) |
| `scripts/stale_claims.py` | 111 | df80b6b 09-01 | "claims older than the code they describe, as a reading queue" — a report generator nobody cites |
| `scripts/audit_population_universals.py` | 131 | f588f2b 09-01 | one-shot audit of 11 harness checks |
| `eval/score_code_exec.py` | 410 | 01fdc08 09-01 | cited in `docs/audits/harness_review_0901_de.md` prose; not called by `eval_code.sh`, `score_matrix.py`, or any runner (G1) |
| `eval/code_l0prime.py` | 281 | 27c8a7f 09-01 | L0' hard layer; no runner |

Decision split: the six `test_*.py` (561 lines) either go into CI (`ci.yml` one line each) or go — a test run only when it changes is not a regression test. `score_code_exec.py` + `code_l0prime.py` (691 lines) are eval capability with no runner: either wire into `eval/score_matrix.py` or delete. Subtotal 1b: **1,694 lines**, decision by owner (de for scripts, 44 for eval).

### 1c. Tracked non-code with zero citations

| path | lines | last commit | evidence |
|---|---:|---|---|
| `data/synthetic/hard_eval/{shard_1..4,verify_0..3}.jsonl` | 2,080 (8 files) | d535674 08-31 — the `git add -A` sweep commit | `grep -rn 'synthetic/hard_eval\|hard_eval/(shard\|verify)'` → 0 hits outside the directory |
| `data/eval/corpus/math/newsrc2.jsonl` | 95 | d535674 | `grep -rn newsrc2` → 0 hits |
| `runs/logs/*.log` (13 files: train_1b, train_2b, train_k3, train_k3_v2, k4_11b_lr05, train_fp8, sft_run, sft_v3/v4/v5, sft_math, sft_short, rlvr2) | 2,337 | pre-reset | `grep -rn 'runs/logs'` outside `runs/` → 0; every one is a pre-0830v1 run the reset zeroed |
| `runs/p47_s1.log`, `runs/p47_s2.log`, `runs/clean_dry.txt`, `runs/t14_truncation_scan.txt`, `runs/t47_readout.md` | 1,472 | | cited_by=0 outside `runs/` (`grep -rl <basename>`) |
| `docs/standards/REBUILD_PLAN_2026-08-29.md` | 71 | pre-reset plan "for aupai-fb approval" | cited only by `data/pod_head_manifest.txt` and the stale `runs/reachability.txt`; superseded by `docs/standards/0830v1_gates.md` |
| `data/CORPUS_V3_CARD.md` | 145 | 2026-08-29 | cites `docs/standards/data_recipe_v3.md`, which does not exist; only reference is the hook allow-list `scripts/hooks/pre-commit:41` (remove that line too) |

Subtotal 1c: **6,200 lines**, no check reads any of them (`no_oversized_blob`, `docs_root_clean` unaffected). Owner: data/runs → controller; docs → 44.

---

## 2. Deletable after a fact or doc citation is retired

Facts cite scripts in `source`; `fact_refs_resolve`/`entrypoints_ran` only FAIL when a *doc* cites a missing path in a command block or when AGENTS.md cites it — a fact's `source` naming a deleted script does not fail a check today (`facts_well_formed` checks fields, not paths). Still: retire the source to `<script>@<sha>` before deleting, so the provenance stays resolvable.

### 2a. Whole directory `bench_eff/` — 4 files, 541 lines, reached by facts only

| file | lines | citing fact ids (`facts/efficiency.json`) |
|---|---:|---|
| `bench_eff/t56_attr.py` | 74 | `eff.step_remainder_attribution` [measured], `eff.dynamo_recompile_not_a_lever` **[retracted]** |
| `bench_eff/t56_elementwise_owner.py` | 141 | `eff.quant_tax_is_the_elementwise_group` |
| `bench_eff/t57_fusion.py` | 188 | `eff.steady_state_composition`, `eff.pad_dynamic_shapes_surface` |
| `bench_eff/t61_fusion_quant_overlap.py` | 138 | `eff.fusion_and_elementwise_are_disjoint_but_the_trace_is_off_config` |

Trace analyzers over `runs/t57_*.log`/`*.json`; the numbers are in facts. Retire sources to `bench_eff/<f>@<sha>`, delete the directory. Also cited in prose by `docs/lessons/arch_efficiency_plan.md` ("profile traces in bench_eff/"), not in a command block.

### 2b. Probes cited only by facts (`probes/`, 31 files, 4,311 lines)

| file | lines | citing fact(s) | status of fact |
|---|---:|---|---|
| `probes/padshim.py` | 10 | `eff.pad_dynamic_shapes_ab` | **retracted** → delete now, retire source |
| `probes/t7_attest_path.py` | 21 | none (prose in `docs/audits/harness_review_0901_tilerl.md`) | delete now |
| `probes/t57_absmax.py` | 71 | `eff.fp8_head_activation_range` | hardcodes `/work/aupai/ckpt_pretrain_15b_s1.pt.step*` (pre-reset checkpoint, line 25) |
| `probes/t57_seam.py` | 30 | `eff.seam_dynamo_disable` | |
| `probes/t58_quant_tax.py`, `t59_fp8_peak.py`, `t60_weight_cache.py` | 391 | `eff.quant_tax…`, `eff.fp8_gemm_at_realizable_peak`, `eff.fp8_weight_byte_cache` | |
| `probes/t66_depth_shape.py` | 303 | `eff.grad_ckpt_inverts_with_depth`, `eff.depth_shape_matched_pair`, `eff.depth_is_not_the_mfu_gap` **[retracted]** | in hook `SELFTEST_FILES` |
| `probes/bench_gpu_probe.py`, `lm_head_gemm.py`, `bf16_update_loss.py`, `ab_vocab_loss.py`, `attn_res_cpu_gap.py` | 556 | `eff.gpu4_peak_flops`, `eff.lm_head_is_compute_bound`, `eff.bf16_updates_discarded`, `eff.vocab_align_parity`, — | |
| `probes/attn_res_bench.py`, `bench_gated_mla.py`, `mem_account.py` | 630 | **none**; only edge is `scripts/restartability_baseline.json` (see 2d) | delete now after removing their baseline rows |
| `probes/t62..t65, t66_gold_reachability, t67, t68, t69` | 1,382 | `dq.corpus_exact_dup_is_zero`, `dq.near_dup_rate…`, `be.gold_bpb_falls…`, `be.gold_is_ranked_high…`, `be.rep_stop_truncates…`, `be.self_repetition…`, `be.free_running_agreement…` | measured 09-01; the copy-hypothesis chain — one round of probes, each read once |
| `probes/chatml_in_corpus.py`, `fone_digit_acc.py`, `profile_step.py`, `t71_depth_lr_rule.py` | 687 | AGENTS.md (`fone_digit_acc`), `0830v1_gates.md` (`profile_step`), hook map | keep |

Rule for the directory: a `t<NN>_*.py` probe is a one-shot whose number lives in a fact. 27 of 31 probes fit that description (3,624 lines). Retire the fact `source` to `probes/<f>@<sha>` and delete; keep the four in the last row. Owner 44.

### 2c. `eval/` and `datagen/` files cited by facts only

| file | lines | citing fact | note |
|---|---:|---|---|
| `eval/compare_fewshot_arms.py` | 205 | `be.l1_3shot_24k`, `be.l1_8demo_format_collapse`, `be.l1_below_constant_guess` | hook map; no runner |
| `eval/ctx_probe.py` | 218 | `be.ctx_length_p324` | pre-reset checkpoint p324 |
| `eval/measure_sft_termination.py` | 213 | `dq.sft_termination_underdetermined` | |
| `eval/kda_probe.py` | 181 | `eff.kda_*` ×3 | |
| `eval/l1_fewshot.py` | 136 | `be.l1_3shot_*` ×5 incl. one retracted | ENTRY per reachability (eval_all.sh?) — verify before deleting |
| `datagen/fasttext_junk.py` | 147 | `dq.fasttext.cci3_failed` | also named in `train.py` (prose) — **post-run** check |
| `datagen/scan_scm_tests.py` | 100 | `ds.code_mining_feasibility` | |
| `scripts/test_parallel_encode.py` | 38 | `eff.…` (t50) | |
| `scripts/w7_driver.sh`, `scripts/w7_peak.sh` | — | `eff.w7_peak_memory_b32_fits` | |
| `scripts/sample_code_rp1t.py` | 48 | `dq.code_rp1t_handread_50` | |

Facts citing a path that does not exist (fix the fact, nothing to delete): `facts/base_eval.json#be.gold_bpb_method` → `eval/configs/task_suites.py` (no `eval/configs/` directory).

### 2d. Reached only through `scripts/restartability_baseline.json` — 21 files, 4,502 lines

The baseline is a ratchet of *offenders* (`check_restartability`, `scripts/harness.py:2553-2568`: "only a NEW offender fails"). A row there is not a citation; removing a row and its file passes. `reachability.py` counts it as a `facts:` edge, which is why these 21 read as reachable.

| file | lines | any other citation |
|---|---:|---|
| `datagen/gen_knowledge.py` | 575 | none (pre-reset zh knowledge generator) |
| `datagen/gen_knowledge2.py` | 433 | `data/PROVENANCE.md`, `data/MANIFEST.tsv` (frozen-tier provenance) |
| `datagen/gen_code.py` | 340 | `docs/lessons/code500_v2.md` prose, `datagen/gen_code_v2.py` docstring |
| `datagen/score_web_27b.py` | 280 | `facts/data_quality.json` ×1, hook map |
| `datagen/score_27b_zero.py` | 220 | none |
| `datagen/distil_traces.py` | 168 | hook map |
| `datagen/annotate_quality.py` | 139 | none |
| `datagen/gen_0094.py` | 133 | none |
| `datagen/augment_data.py` | 114 | none |
| `datagen/sample_pattern_discovery.py` | 79 | none |
| `datagen/make_mixed.py` | 39 | `data/PROVENANCE.md` |
| `datagen/prepare_sft.py` | 348 | `sft_math.py`, `scripts/test_e2e.py`, `datagen/prepare_sft_math.py` → **live, keep** |
| `mathbank/procedure_curriculum.py` | 277 | none; output `data/synthetic/procedure_v1.jsonl` unreferenced |
| `mathbank/arith_curriculum.py` | 220 | none; output absent |
| `mathbank/program_probe.py` | 158 | `eval/probe_band.sh:47` ← `scripts/run_pipeline.sh:56` (pre-reset SFT/RL chain) |
| `mathbank/make_v11.py` | 127 | `make_v11_band.py` |
| `mathbank/run_short_sol.py` | 95 | `datagen/build_math_expand.sh` comment |
| `probes/attn_res_bench.py`, `probes/bench_gated_mla.py`, `probes/mem_account.py` | 630 | none |
| `scripts/ckpt_sweep.py` | 127 | none |

Delete-now subset (no citation at all besides the ratchet): `gen_knowledge.py`, `score_27b_zero.py`, `annotate_quality.py`, `gen_0094.py`, `augment_data.py`, `sample_pattern_discovery.py`, `procedure_curriculum.py`, `arith_curriculum.py`, `attn_res_bench.py`, `bench_gated_mla.py`, `mem_account.py`, `ckpt_sweep.py` = **12 files, 2,905 lines**; remove their rows from `scripts/restartability_baseline.json` in the same commit; `harness check` `restartability` must stay PASS. Owner 3b (datagen/mathbank), 44 (probes), de (scripts).

### 2e. Pre-reset datagen/mathbank cited by `docs/audits/audit_math_corpus.md`, `facts/contamination.json`, `data/PROVENANCE.md`

`mathbank/run_math_short.py` (252), `make_v11_band.py` (225), `split_bank.py` (65), `dist_check.py` (117, reads retired `math_hard_eval_1k.jsonl`, `dist_check.py:20`), `eval_hard_v2_gen.py` (467, **live**: produced `math_hard_eval_v2_1k.jsonl` read by `eval/math_v2_like.py:43`), `datagen/build_math_expand.sh` (39), `eval/gate_math_short.sh` (28). Their output feeds only the `math` domain of the frozen `mix_scale_*.json` ladder; no live mix (`mix_500m`, `mix_15b_stage1`, `mix_30b_stage2`) names `math` (they name `math_owm*`), and `build_math_expand.sh:21-28` gates on `gate_math_short.sh` which rejects every batch v3–v11 — the pipeline cannot produce a new shard. See §5 for the 38K-line consequence.

---

## 3. Duplicated helpers across `scripts/ eval/ datagen/ probes/`

Method: `grep -n 'def …'` per category over 241 tracked `.py`, bodies read, plus an AST pass hashing function bodies with docstrings stripped and identifiers normalised (`scratchpad/dupes.py`), groups ≥ 2 sites, ≥ 3 lines.

### 3a. AST-identical function bodies (renamed or verbatim)

| group | sites | lines each | keep |
|---|---|---|---|
| `log_likelihood` | `eval/arc.py:30`, `piqa.py:28`, `hellaswag.py:17`, `winogrande.py:22` | 9 | none — `eval/run_eval.score_mc` (:159) is the live scorer (see 3k) |
| `load_model` | `eval/arc.py:23`, `piqa.py:21`, `boolq.py:19`, `openbookqa.py:19` | 4 | `scripts/loader.load_checkpoint` |
| `evaluate` + `_DummyTok/_DummyLM` smoke block | `hellaswag.py:28-64`, `winogrande.py:33-69` | 9 + 26 | delete both |
| `_reg`/`wrapped` decorator | `mathbank/math_programs_l3_ext{3,4,5}.py:17-30`; `l4_ext{2,3,4,5,6}.py:23-42` | 14+12 ×3; 5+3 ×5 | `mathbank/mathcommon.py` |
| `_d` | `mathbank/math_programs_l3_ext{5,7,8}.py:34-54` | 13 ×3 | `mathcommon.py` |
| `load_pairs` | `probes/t68_free_running.py:36`, `probes/t66_gold_reachability.py:44` | 19 | one of the two (or delete both, §2b) |
| `main` | `filters/pass2_garbage.py:49`, `pass3_garbage.py:55` | 14 | one `filters/_cli.py` |
| `_weight_for_rows` | `scripts/write_mix_500m.py:305`, `scripts/write_mix_stage2.py:118` | 12 | `write_mix_500m.py` (stage2 mix is frozen) |
| `config_stamp`, `err` | `probes/bench_gated_mla.py:176/110`, `probes/attn_res_bench.py:159/81` | 7+6 | delete (§2d) |
| `run` | `datagen/gen_code.py:235`, `gen_code_v2.py:218` | 8 | `gen_code_v2.py` (`gen_code.py` is §2d) |
| `find` | `eval/measure_sft_termination.py:99`, `datagen/build_corpus.py:860` | 5 | `build_corpus.py` |
| `want` | `scripts/test_score_matrix_failpath.py:70`, `scripts/test_free_card.py:73` | 4 | — |
| `encode` | `probes/t66_gold_reachability.py:148`, `probes/t65_gold_bpb.py:128` | 4 | — |
| `git` | `scripts/harness.py:1967` and `:2043` | 3 | one `_git(root, *a)` (8 wrappers total, §4) |
| `stmt_text` | `mathbank/math_programs_l4_ext8.py:1299` and `:1342` | 6 | same file, twice |
| `__init__`/`forward`/`num_logits` ×6 | `train.py` vs `infer_local.py:100-272` | 4–10 | deliberate mirror; `scripts/test_arch_compat.py:438` asserts parity — not a target |

### 3b. Per category

| cat | definition sites (file:line, fn, L) | inline copies | keep | L saved |
|---|---|---|---|---|
| a. jsonl reader | `datagen/build_corpus.py:192 iter_jsonl` (16, gz-aware, bad-line count); `scripts/board.py:35 rows` (13); `scripts/exp.py:33 rows` (12, fold by (name,started)); `datagen/measure_duplication.py:32 _read_rows` (8) | `for line in open(): json.loads(line)` — 146 lines in 79 files: `harness.py` 30, `build_corpus.py` 8, `score_matrix.py` 5, `fasttext_junk.py` 5, `code_l0prime.py` 4, `rlvr_data.py` 4, … | `iter_jsonl` moved to `scripts/loader.py` (torch-free, imported by 43 files) | 320 |
| b. mix json | `scripts/harness.py:1286 read_mix` (13, → (domains, err)); `scripts/launch_gate.py:83 _mix` (3); `eval/readout_30b.py:134 _mix_weights` (5) | `json.load(open(mix))["domains"]` at `check_mix:36`, `corpus_fingerprint:89`, `data_overview:85`, `pretokenize:51`, `domain_loss:78`, `ppl:49`, `score_matrix:140`, `t70_external_loss:72`, `build_tokenizer:80`, `harness:6582`, `replay_cursor:48/163/270`, `test_e2e:77/91`, `launch_gate:642` | `datagen/corpus_fingerprint.load_mix` (split out of `fp_mix` :87-90); `harness.read_mix` wraps it | 17 (gain is one schema owner) |
| c. tracked-file walk | `scripts/harness.py:5585 _repo_owned_files` (18, ls-files else pod manifest); `scripts/pod_drift.py:133 scoped_paths` (5) | `git ls-files` inline: `launch_gate:319,609`, `stale_claims:105`, `test_shard_glob:162`; `os.walk` repo scans: `harness:1109,1139,7817`, `pod_drift:309,341`, `reachability:60`, `restartability_audit:160` | `pod_drift.scoped_paths` with `_repo_owned_files`'s body (harness already imports pod_drift) | 25 |
| d. content hash | chunked file sha256: `datagen/data_verify.py:87 sha256_file`, `datagen/fetch_chat_data.py:36 sha256`, `scripts/launch_gate.py:194 _sha256`, `scripts/launch_tests.py:31 _sha256`, `scripts/eval_artifacts.py:141` inline, `scripts/pod_drift.py:139 _sha_bytes`; whole-read `[:16]`: `datagen/prepare_sft.py:67`, `harness.py:2753`, `sft_math.py:112`, `probes/t65_gold_bpb.py:172`; head+tail 64KB: `corpus_fingerprint:35 _shard_line` vs `harness.py:3018-3029` re-implementation; sorted-dir `name\0sha`: `corpus_fingerprint:49 fp_filters` vs `prepare_sft.py:75 _fp_sources`, `harness.py:3010-3015`, `datagen/holdout.py:38` (sha1) | — | `datagen/corpus_fingerprint.py` (+`sha256_file`); `train.py:1712` keeps its copy by design (imports nothing from datagen, `--self-check` asserts parity) | 55 |
| e. step-line parser | `train.py:47-51 RunLog._STEP_RE` (emitter); `scripts/plot_curves.py:27-45` (19); `scripts/read_lr_probe.py:43-56` (13); `harness.py:9308,9652,9833,9979` `\.step(\d+)$` ×4 | — | `scripts/exp.parse_step`; train's regex stays with a test that the two agree | 20 |
| f. UTC time | `scripts/exp.py:64 now()` only def | `time.strftime(..., time.gmtime())` 23 sites, 4 formats: `harness.py` ×15 (4497 … 10012), `board.py:107,113`, `launch_tests.py:85`, `eval_artifacts.py:52,153`, `progress_feed.py:99,145`, `train.py:2212` | `exp.now()` | 5 (gain: one ledger timestamp format) |
| g. pod / tn call | no def anywhere | `subprocess.run([~/bin/pod, cmd])` ×7 (`harness.py:953,978,8695,8732,9509,9597,9676`); `["tn","exec",cmd]` ×15 (`harness.py:9439-9583`); timeout present at 4 of 22 | `pod_drift.pod(cmd, timeout)` / `tn(cmd, timeout)` | 30 |
| h. ckpt → model | `scripts/loader.py:34 load_checkpoint` (43, canonical); `infer_local.py:292 load_model` (22, standalone by design); wrappers `eval/arc.py:23`, `piqa.py:21`, `boolq.py:19`, `openbookqa.py:19`, `run_eval.py:263`; cfg-only `torch.load`: `score_matrix.py:121 read_cfg`, `harness.py:9408,9418`, `ckpt_info.py:24`; `setattr(Cfg,k,v)` loop: `sft.py:56-62,89`, `sft_math.py:74-80,141`, `probes/bf16_update_loss.py:69-72,85`; `type("C",(),ck["cfg"])`: `probes/fone_digit_acc.py:36-41` | — | `loader.load_checkpoint` + new `loader.read_cfg` + `loader.apply_cfg(ck, Cfg)` | 30 |
| i. tokenizer | `scripts/loader.load_tokenizer` (verifies size + `vocab_id`) | bare `Tokenizer.from_file` 34 sites / 24 files; `TOK_PATH = os.path.join(ROOT,"data","tokenizer.json")` defined 20×; hardcoded `/work/aupai/data/tokenizer.json` in `datagen/ast_parse_survivors.py:25`, `build_starcoder_py.py:37`, `executable_yield.py:66`; `vocab_fingerprint` twice (`loader:20`, `train:1414`) by design | `loader.load_tokenizer(path=None)`; `train.TOK_PATH` | 30 |
| j. ChatML | `serve.py:56 format_history` (18): docstring says 问：/答：, body calls `loader.format_prompt` (ChatML) then appends `答：` | — | `loader.format_history` (:138) — behaviour change, flag it | 16 |
| k. MC suite | 8 files, 590 L. `run_eval.MC_BENCHMARKS` (:136) is already the task table and `score_mc` (:159) the scorer; run_eval imports only `load_dataset/load_items` from each module — nothing imports a module's `evaluate`/`log_likelihood`/`load_model`. Dead standalone scaffolding: `arc.py` 48/82, `piqa.py` 44/74, `boolq.py` 31/60, `openbookqa.py` 35/67, `hellaswag.py` 44/64, `winogrande.py` 44/69, `mmlu.py` 42/67; `ceval.py` `_demo` selftest (31) keep. Scorer variants: 4× `log_likelihood` (separate encode), `eval/__init__.log_likelihood_joint` (:28, joint encode — boolq/openbookqa standalone numbers differ from run_eval's for the same items), mmlu inline, `score_mc` | run_eval adapters `:52-134` (~80) exist only because 6 of 8 modules return raw HF rows | each module = `load_items()` (~10 L) + the table | 350 |
| l. argparse `--ckpt/--tokenizer/--device` | 30 / 22 / 27 files; 16 carry all three. Defaults disagree: `--device` "cuda" (18) vs "cuda:0" (8) vs int; `--tokenizer` spelled 5 ways incl. `data/tokenizer_k5.json`; `--ckpt` default `ckpt_k5_clean_0827.pt` at `datagen/audit_source_score.py:47`, `train_quality_head.py:68` (pre-reset) | `ROOT = dirname(dirname(abspath(__file__)))` in 96 files, `sys.path.insert` in 71 | `loader.add_model_args(parser)` + `loader.load_from_args(args)` | 40 |
| AST extras (3a) | 12 groups | — | `mathcommon.py` and per-pair owner | 165 |
| **total** | | **~1,600 duplicated** | | **~1,100** |

Three defects found on the way, not line counts: (1) standalone `boolq/openbookqa` score with joint tokenisation, `run_eval` with separate encode — two numbers for one checkpoint; (2) `serve.format_history` emits ChatML then `答：`; (3) `probes/t65_gold_bpb.py:172` records a tokenizer *file* hash under the name `tok_fp`, which is not `vocab_id`.

---

## 4. `scripts/harness.py` — 10,172 lines, 257 commits since 08-29

### 4a. Where the lines are

| bucket | lines | % |
|---|---:|---:|
| `check_*` (59 fns) | 2,469 | 24.3 |
| `_broken_*` (59 fns) | 1,125 | 11.1 |
| check-private helpers (46 fns: `_exp_events`, `_read_tasks`, `merge_took_one_side`, `_template_scan`, `_busy_training_cards`, …) | 1,018 | 10.0 |
| CLI subcommands (`launch` 388, `kill` 285, `milestone` 395, `task` 290, `run` 321, `board` 300, `clean` 102, `sync` 88, `measure` 78, `free-card` 63, `gaps` 49, `install-hooks` 41, `ledger` 18, `stages` 20) | 2,427 | 23.9 |
| `_selftest_*` (19) + `_demo` | 1,288 | 12.7 |
| `CHECKS` tuple list 5792–6211 | 420 | 4.1 |
| docstrings + `#` comments (inside the above) | 2,197 | 21.6 |
| world builders `_tmp_repo` 1411, `_tmp_repo_shaped` 1426, `_tiny_tokenizer_json` 1465 | 67 | 0.7 |

Dead code: `_broken_agents_rules_unmapped` (`:931`, 13 L) is referenced only by a docstring at `:917`, never called; `_broken_spawned_scripts_importable` (`:1880`) is called only from `_demo`.

### 4b. Checks that overlap (same artifact, same assertion shape)

| artifact | checks (line, L) | shared shape | merge | L saved |
|---|---|---|---|---:|
| `ckpt_*.pt` header | `env_fp_present` 4297(25), `opt_state_present` 4341(26) | byte-identical loop: glob, `_read_ckpt_dict`, key-in-dict | `ckpt_fields_present` with 2 (key, scope) rules | 20 (+12 broken) |
| git working tree | `untracked_aged` 5689(24), `dirty_aged` 5734(31) | same `.git` guard, subprocess, cutoff/mtime loop; `git status --porcelain -uall` yields both | `aged_uncommitted` | 25 (+20 broken) |
| tracked `.py/.sh` walk | `curl_ipv4` 1089(37), `timestamps_are_utc` 1128(36) — lines 1109–1120 ≡ 1139–1150 | identical 20-line walk + `_SKIP_DIRS` + docstring strip. **Five hand-written populations for "repo python files"**: 1109, 562 (`scripts,datagen,eval`), 712 (+root), 2311, 1835 (+`probes,algorithms`) — the bug class `curl_ipv4`'s own comment 1103–1108 records | one `_source_files(root, exts)` used by all five | 30 |
| `data/tokenizer.json` | `tokenizer_roundtrip` 1675(15), `pinned_ids` 1692(19) | same exists+`Tokenizer.from_file` preamble; broken world already shared (`_broken_tokenizer(eos_id=5)` 5850) | `tokenizer_contract` | 10 (+7 tuple) |
| default mix | `mix_not_unfiltered` 1524(8), `mix_shards_present` 1595(13) | same `read_mix` preamble, same `_broken_mix` world used twice (5805/5812) | one | 15 |
| `AGENTS.md` | `entrypoints_table_present` 3745(11) is lines 3711–3712 of `entrypoints_ran` 3689(54); `doc_commands_exist` 4015–4018 ≡ `readme_current` 4084–4087 (`CMD_BLOCK_RE`/`CMD_PATH_RE` loop over a different file) | fold table_present into entrypoints_ran; parametrize the cmd-block loop over (AGENTS.md, README.md) | 21 + 8 |
| `runs/experiments.jsonl` running rows | `no_stale_running` 2436(20), `no_ghost_running` 2458(41) | ghost re-implements the fold inline 2475–2482; strptime block 2487–2489 ≡ 2447–2449 | `running_rows` (pod branch adds pgrep) | 25 |
| `runs/tasks.jsonl` rows | `tasks_paired_and_prior` 3294(31), `tasks_closed_by_commit` 3340(24), `tasks_well_formed` 5303(27) | "per row in scope, field X present/valid"; each carries its own `_read_tasks` + SKIP + `bad[:3]` scaffolding (~8 L ×3) | `tasks_rows_valid` | 25 |
| corpus stamp vs live value | `corpus_filters_fp` 2571(72), `score_input_fresh` 2661(49), `corpus_fp_matches` 4772(44) | "per default-mix domain, stamp field == live fn" ×3; SKIP rules differ (`CORPUS_FILTERS_BASELINE` debt, PROVENANCE tier, pod-only) | one check parametrized by (stamp file, field, live_fn) with per-rule SKIP branches; one world carrying all three defects (~30 vs 58) | 75 (+28 broken) |
| `docs/**` + `facts/*.json` | `lessons_have_frontmatter` 3861–3868 ≡ `fact_refs_resolve` 3897–3903 walk `DOCS_SUBDIRS`; facts globbed+parsed 4× (411, 3564, 3893, 4098) | `_facts_index(root)`, `_lesson_files(root)` | 12 + 8 |
| `train.py` ast | `guard_on_path` 2501, `gemm_dims` 2523 | same 4-line exists+`ast.parse` preamble | `_train_ast(root)` | 6 |
| nvidia-smi | `_busy_training_cards` 5444(36) ≈ `_busy_once` 8565(17) | same two nvidia-smi calls + uuid→index map | one | 17 |

(a) merging overlapping checks: 59 → 47 checks, **≈ 330 lines** including their tuples and worlds. Every merged entry keeps one `broken()` that FAILs; all pairs admit a single world carrying both defects. Renames propagate to `EVIDENCE` 6222, `_CHECK_TIMEOUTS` 55, `_RULE_CHECKS` 149, `STAGES`, and the AGENTS.md coverage table — `agents_rules_covered` FAILs until done, which is the guard working.

**4b correction, measured 2026-09-02 (de). Not done: the merge is ≈94 lines at best, not 330, and three of the reasons are structural rather than arithmetic.**

1. **The base predates de-12, so one row is already collected.** This audit reads main
   `0622878`; `git merge-base --is-ancestor 34acbb9 0622878` is false, and `34acbb9` is
   "harness: one tree walker for curl_ipv4 and timestamps_are_utc". The 20-line duplicate
   walk at 1109–1120 ≡ 1139–1150 is gone: both call `walk_tracked` (`:1122`, `:1145`), and
   the whole file now holds **2** `os.walk` sites — that helper and the selftest's
   repo-real-path guard. The "five hand-written populations for repo python files" row is
   resolved; counting its 30 lines again would book one deletion twice.

2. **Two candidate merges straddle the repo/pod authority split and cannot be merged at
   all.** `CHECK_AUTHORITY` (`:6280`) separates checks whose evidence is in git (answers on
   main) from those whose evidence exists only on the training box:

   | candidate | authorities |
   |---|---|
   | default mix | `mix_not_unfiltered`=repo, `mix_shards_present`=**pod** |
   | running rows | `no_stale_running`=repo, `no_ghost_running`=**pod** |

   A merged check carries one authority. Give it repo and a pod-only artifact gets judged on
   a Mac; give it pod and a rule whose evidence is in git stops answering on main. The
   summary line "green here is not green on the pod" is exactly this split, so these two
   pairs are not a size question.

3. **Four candidates cannot be proved on a dev box, and proving them is the acceptance
   condition.** The rule (fb) is that each folded rule must still be shown to FAIL on its
   own after the merge. Here `ckpt header` and `tokenizer` both SKIP (no checkpoint;
   `data/tokenizer.json` is gitignored, and `_broken_tokenizer` raises `SelftestSkip`), and
   `corpus stamp` is pod-authority. Merging them here would record "cannot show" as
   "verified".

   Ceiling after removing the three: **≈94 lines** across `git tree`, `tasks rows` and
   `AGENTS.md` — and 237, the all-ten figure, assumes a merged function costs nothing, while
   `tasks rows` needs a (row-scope, field, validator) table and this audit's own text notes
   the `corpus stamp` trio's SKIP rules differ three ways.

   The direction is also wrong. `CHECKS` holds one `broken()` per row, so a check with two
   independent failure modes cannot register both — `agents_rules_covered` needed 20 lines
   of explicit post-loop wiring for its second world (`72ebbce`). Merging two checks makes
   two rules share one world, tightening the constraint that just had to be worked around.

**4a correction (de).** Both "dead code" calls in 4a are wrong, and the shape is worth
naming: reachability is not usefulness. `_broken_agents_rules_unmapped` (:931) FAILs the
check with "1 rule(s) map to neither a check nor a manual reason", a defect the registered
world cannot produce; it was unreachable because CHECKS holds one world per row, not because
it was dead, and it is now wired in. `_broken_spawned_scripts_importable` is called at
`:7897`, not "only from `_demo`". **An unwired broken world is a check somebody finished
writing and nobody ran; under grep it is identical to dead code.** Before deleting anything
shaped like `test_*` or `_broken_*` from 1a's 21 files, run it and see whether it FAILs.

### 4c. `_broken_*` worlds — 1,125 lines

| setup pattern | n | worlds (line) |
|---|---:|---|
| write synthetic file(s) at a repo-real path | 14 | 685 760 793 1228 1484 1534 1610 1863 2348 2645 2712 3163 3207 4107 |
| copy one real file, edit a field/line | 13 | 462 543 912 931 1166 1180 1584 1880 3192 3366 4132 4591 5665 |
| copy real file(s), append a row | 7 | 3177 3327 3435 3496 4830 4842 4882 |
| fabricate a `.pt` (torch.save / zipfile+pickle) | 6 | 2759 2912 4324 4369 4476 4536 |
| env-var injection (`HARNESS_*`) | 4 | 1060 4726 4986 5538 |
| `git init` temp repo + commits | 4 | 1927 2228 5715 5767 |
| real `exp.py start --root d` | 3 | 1495 3758 4230 |
| `copytree(docs|facts)` + append/strip | 3 | 3921 3932 3948 |
| `_tmp_repo_shaped()` + copy + mutate | 2 | 345 3633 |
| `_read_tasks`/`_write_tasks` round-trip | 2 | 5332 5393 |
| delete lines from real AGENTS.md | 1 | 3806 |

Composition: 59 `def` + 188 docstring + 45 comment + 39 blank + **331 setup boilerplate** (`_tmp_repo()`, `makedirs`, `shutil.copy`, `open().read()`, `return d`) + 463 mutation. Verbatim repeats: the 7-line symlink loop of `_tmp_repo_shaped` 1441–1447 copied at 3785–3790 and 4119–4125; the stub `open(join(d,"scripts","harness.py"),"w").close()` as the "repo-real path" token 6× (1236, 2720, 2768, 2926, 3167, 4331, 4375).

(b) one builder `_world(copy=[rel…], edit={rel: fn}, append={rel: text}, link=True)`: 13×~8 + 7×~6 + 3×4 + 14×3 ≈ **180 of 331 boilerplate lines**, +14 for the symlink copies, +13 for the dead world = **≈ 205**. Docstrings (188) stay — they are the incident record — or move into the CHECKS `incident` field. Every world still names a repo-real path, so `_demo`'s detector (7800–7821) holds by construction.

### 4d. Helpers duplicated inside the file and against `exp.py`/`pod_drift.py`/`corpus_fingerprint.py`

| harness.py | sites | duplicate of |
|---|---|---|
| jsonl read + fold | `experiments()` 1302 (weak last-wins fold), `_exp_events()` 2365 (terminal-wins), `_read_tasks()` 4997; 38 `json.loads(` sites incl. inline folds at 390, 503, 2475, 3399, 3443, 3700, 4187, 4199, 4432, 6719, 6727, 7399, 7772, 8483, 8519, 9621, 9839 | `exp.rows()` exp.py:33–49; harness shells out to `exp.py` 12× (1504, 3791, 4241, 6520, 8240, 9190, 9210, 9271, 9380, 9598) instead of importing, though `scripts/` is already on `sys.path` (:36–37). `experiments()` keeps the fold `_exp_events`'s docstring 2372–2377 says reopened closed runs; `ledger`/`gaps`/`recorded_scores` still read through it |
| `json.load(open(p))` | 27 sites; facts parsed 4× | — |
| local `git(*a)` | 1968, 2044, 2237, 5078, 6309, 6928, 7248, 7345 | one `_git(root, *a)` |
| `time.strftime(gmtime)` | 12 literals (4497 … 8110) | `exp.now()` exp.py:64 |
| age from `"%Y-%m-%d %H:%M"` | 2447, 2488, 3422, 5379 | one `_age_min(ts)` |
| `_merge_jsonl` 8641 (union, byte-dedupe) | — | `exp.py:213–226 merge` (field-merge by (name,started)) — two merge semantics for one ledger |
| `_template_inputs_key` 3003–3031 (head+tail 64KB sha) | — | `cfp._shard_line` corpus_fingerprint.py:35–46 (already imported as `cfp`) |
| `sha256(f.read())` | 2753, 3015 | `pod_drift.sha_disk` :162 (used at 4876) |
| manifest parse `parts[1]` | 5598–5602, 4862–4870 | `pod_drift.read_manifest` :224 |
| `read_mix` 1286 + inline 1643, 4683, 6582 | — | `cfp.fp_mix` :87–90 |

≈ **110 lines** by one `_jsonl`/`_fold`, deleting `experiments()` for `_exp_events`, `exp.now()`, one `_git`.

**4d correction, measured 2026-09-02 (de). Not done: ≈4 lines, not 110.** The duplication is
real and it is almost all *inline expressions*, which cost tokens rather than lines:

| claim | measured | lines a fix saves |
|---|---|---:|
| 12 `strftime(gmtime)` literals duplicating `exp.now()` | 11 exact matches, **every one inline** inside a dict literal or call argument | 0 |
| harness shells out to `exp.py` 12× instead of importing | 10 sites, all `start`/`done` **writes** | 0 (importing changes failure semantics, not size) |
| local `git(*a)` at 8 sites → one `_git(root, *a)` | 4 definitions; exactly 2 byte-identical (`:1965`, `:2041`, 4 lines each, inside `merge_reverted_content` and `merge_took_one_side`); `:6365` returns `None` not `""`, `:6984` takes `cwd=` and returns the CompletedProcess | 4 |

Replacing an inline `time.strftime(...)` with `exp.now()` shortens a line and removes none:
zero of the 11 sites is a standalone statement. Deleting `experiments()` in favour of
`_exp_events` is a behaviour change (weak last-wins vs terminal-wins fold), which the entry
itself notes `ledger`/`gaps`/`recorded_scores` still depend on — a fold swap under three
readers is not a line-count item. The two identical `git` helpers net 4 lines and sit inside
the two merge-safety functions; that is not a trade worth making on this code.

The general point for the other sections' figures: a repeated *expression* and a repeated
*block* both read as duplication in a grep, and only the second is lines.

### 4e. Subcommands that are not checks

| verb | lines | wraps | citations outside harness.py (`grep -rn -E 'harness(\.py)? +<verb>\b'`) | ruling |
|---|---:|---|---|---|
| `stages` + `STAGES` | 20 | — | 0; `STAGES["eval"]` gates = `[]` | delete |
| `gaps` | 49 | — | 0 | delete |
| `clean` | 102 | `~/bin/pod ls`, `--dry` only, never deletes | 0 | delete |
| `ledger` | 18 (+57 shared `recorded_scores`/`checkpoint_names`) | — | 1 (a test docstring) | delete |
| `measure` | 78 | `eval/eval_hard.sh` + `exp.py start/done` | AGENTS.md row only | `launch <n> -- eval/eval_hard.sh <ckpt>` per unscored ckpt is the same; delete with the doc row |
| `board` + `_30b_readiness` | 300 | ledgers → `runs/board.html` | 1 (`reachability.py`) | owner decision |
| `sync` | 88 | pod base64 + union | 2 | `exp.py merge` exists |
| `run <step>` | 321 | datagen/*, run_ddp.sh, eval_all.sh | 10 | = `launch` + `_strip_frozen` ladder logic (45) |
| `launch` 388, `kill` 285, `milestone` 395, `task` 290, `free-card` 63, `install-hooks` 41, `check`, `--selftest` | 1,462 | — | 28 / 10 / 4 / 5 / 4 / 2 / many | keep |

`main` 10083–10168 also carries 9 `if sys.argv[1] == …` branches before argparse (10088–10105).

### 4f. Verdict on a package split

**Does not reduce lines; adds ≈ 150.** Per module `import os, json, re, subprocess, time, glob` + `from harness._common import ROOT, HERE, PASS, FAIL, WARN, SKIP, SelftestSkip, cfg_default, read_mix, _tmp_repo, _tmp_repo_shaped` ≈ 8–12 L × ~12 artifact modules = 100–140, plus a registry assembling `CHECKS`/`EVIDENCE`/`_CHECK_TIMEOUTS` (~20). The sharing problem is the 46 private helpers, which a split spreads, not removes. It also breaks 24 literal `scripts/harness.py` references inside the file (`_broken_no_duplicate_defs` 793 copies it; 6 worlds stub it; `check_spawned_scripts_exist` 1757 reads its own source), the hook `SELFTEST_FILES` entry, `data/pod_head_manifest.txt`, `_SPAWNED_SCRIPTS`, `ci.yml:33–34`. The one real argument is merge-conflict surface (257 commits in 4 days; `merge_complete` exists because of it) — coordination, not size.

### 4g. Sum

| change | lines | selftest constraint |
|---|---:|---|
| delete `clean` only (**not** `stages`/`gaps`/`ledger` — see 4g note) | 104 measured | none (not a check); remove its `sys.argv[1]` dispatch, keep the `clean` PIPELINE step at `:8668`, which is `clean_corpus.py` and unrelated |
| (b) world builder + dead world | 47 measured (not 205) | skipped by ruling: 47 lines against each world's self-evidence |
| (a) merge 12 → 6 checks + stamp trio 3 → 1 | ≈94 measured ceiling (not 330); **not done** | see 4b correction: one row already collected by de-12, two pairs straddle repo/pod authority, four unprovable on a dev box |
| one `_jsonl`/`_fold`, `_git`, `exp.now`, `cfp._shard_line`, `pod_drift.sha_disk` | 4 measured (not 140); **not done** | see 4d correction: 11 of the 12 duplicates are inline expressions, which cost no lines |

**4g note, measured 2026-09-02 (de, corrects this table's first two rows).**
`stages`, `gaps` and `ledger` are NOT dead surface and were not deleted. They run under
`cmd == "all"`, and `all` is the argparse DEFAULT (`harness.py:10239`), so bare
`python scripts/harness.py` calls all three (`:10264`, `:10267`, `:10274`). Deleting them
changes the default invocation's behaviour, which is a product decision, not cleanup.
Three live runtime references besides that: `harness.py:3644` (`facts_well_formed`'s own
evidence string says "see `harness gaps`"), `:6583` (`measure`'s closing line), and
AGENTS.md:117/118/120, which `entrypoints_table_present` reads. "0 external references"
was wrong.

Line counts, measured rather than estimated: `ledger` 20, `gaps` 51, `stages` 14,
`cmd_clean` 104 — 189 together, not 215. The world-builder saving is 47, not 205: 53 of
the repeats are `d = _tmp_repo()`, which is already the shared builder's call site rather
than removable duplication, and a factored builder still needs one call per world.

One caution for whoever reads the other sections' numbers. A first pass measured 1,896
lines across the 59 `_broken_` functions and a 466-line outlier; both were artifacts of a
span walker that ended a function only at the next `def`/`class`, so trailing
module-level constants and nested defs were absorbed. Ending a span at the next
zero-indent non-blank non-comment line gives 1,290 lines, mean 21.9, largest 50. The
audit's own figures were produced by a read-only agent and were not re-derived here
except where a task touched them.
| **total without dropping a check** | **≈ 890 (8.7 %)** | `python scripts/harness.py --selftest && python scripts/harness.py check` |

Past ~10 % means retiring checks or CLI verbs (`board` 300, `run` 321, `sync` 88), not refactoring. 2,197 lines (21.6 %) are docstrings and comments, against the user's 2026-09-01 no-comments order — the incident text belongs in the CHECKS `incident` field or the commit that added the check.

---

## 5. `mathbank/` — 39 files, 40,563 lines

Loaded how: §0 table. 21 `math_programs_l*_ext*.py` = 35,569 lines; all 27 `math_programs_*` = 38,480 lines.

| measure (21 ext files) | value |
|---|---|
| nonblank lines | 30,281 |
| unique stripped nonblank lines | 19,867 |
| redundant occurrences | 10,414 = 34.4 % of nonblank (29.3 % of total) |
| distinct lines in ≥ 10 files | 32 lines, 5,644 occurrences (`]`, `lines = [`, `ins = rng.choice([`, `])`, `return ins, lines, ans`, …) |
| AST-identical function bodies across files | 7 groups / 24 functions / 92 lines — all `_reg` (6 groups) and `_d` (`l3_ext5.py:34`, `l3_ext6.py:38`, `l3_ext7.py:41`, `l3_ext8.py:42`) |
| per-file scaffolding: header+imports+`PROGRAMS=[]`+`_reg` def | 732 lines (16–50 per file) |
| `_reg("name", fn)` registration calls | 1,257 lines |
| `if __name__ == "__main__"` self-check tails | 223 lines in 18 files, all textually different, same function as `vet_programs.py` |
| total scaffolding | ≈ 2,300 = 6.5 % |

Same-named generators with different bodies (not mergeable without a semantic check): `taxi_fare` ×4 (`l1_ext3:769`, `l2_ext3:344`, `l3_ext5:370`, `l4_ext3:1147`), `age_sum_future` ×3, `chicken_rabbit` ×3.

Shared base verdict: a base module removes the `_reg` defs (−90), `_d` (−52), self-check tails (−223, `vet_programs.py` already does it), duplicate headers (−100); registration by module scan removes the 1,257 `_reg(...)` lines. **Total ≈ 1.6–1.7K lines (4.7 %) with module-scan registration, ≈ 450 (1.3 %) without.** The remaining 28 % redundancy is the program-body idiom, not scaffolding — a shared base does not cut it. `mathcommon.py` (40 lines) already holds `NAMES/GOODS/…`, `num`, `frac`, `pct`, `eval_lhs`; only `_d`, `_gcd` (`l4_ext3.py:31`, stdlib `math.gcd`), `_dec`, `_factor` are reimplemented locally.

The larger question is §2e: the generators' only consumer is the frozen 0830v1 `math` domain, which no live mix names and which the gate forbids rebuilding. Deleting `mathbank/math_programs_*` + `run_math_short.py` + `run_short_sol.py` + `make_v11*.py` + `split_bank.py` + `dist_check.py` + `vet_programs.py` + `program_probe.py` + `*_curriculum.py` (keeping `eval_hard_v2_gen.py` 467 + `mathcommon.py` 40, which `eval_hard_v2_gen.py` imports — verify) removes **≈ 39,900 lines, 44 % of all Python in the repo**, at the cost of reproducibility of frozen bytes whose provenance already records the generator + seed (`data/PROVENANCE.md:55,113,127`). Citations to retire first: AGENTS.md:60,76,271 (`vet_programs.py` entry-point row), `scripts/harness.py:213`, `scripts/reachability.py:50,219-230,343`, `docs/audits/audit_math_corpus.md:102-145`, `facts/contamination.json:133-141,447-464` (source → `@sha`). Checks that must pass after: `harness check` (`entrypoints_ran`, `doc_commands_exist`, `entrypoints_table_present`), CI `py_compile mathbank/*.py` (edit `ci.yml:13` if the directory is emptied). Owner 3b; ruling is the user's, since it deletes the last generator of a frozen corpus.

---

## 6. Docs

### 6a. Frontmatter status

76 docs; 0 `retracted`. Every `docs/lessons` and `docs/audits` file has frontmatter (`lessons_have_frontmatter` PASS). Six `docs/standards` files have none (`0830v1_gates`, `corpus_rebuild`, `incremental_batch_jobs`, `REBUILD_PLAN_2026-08-29`, `synthetic_data_standard`, `writing`) — the check does not cover `standards/`.

Retracted / unmeasured fact ids (14) → docs that cite them: `cont.same_numbers` [unmeasured] ← `docs/audits/audit_math_corpus.md`; `eff.h20_mfu_200m` [unmeasured] ← `docs/lessons/architecture_efficiency.md`; `mlm.ratio.sub1b_optimum`, `mlm.transfer.enzh_math_200m` [unmeasured] ← `docs/lessons/multilingual_mix.md`; `mlm.ratio.30b_measured` [unmeasured] ← `docs/lessons/scale_36b_plan.md`. **No doc cites a retracted id** (`be.l1_3shot_retracted`, `cont.template_share_0_3pct`, `cont.eval_surface_matching`, `cont.sft_all_code_holdout_leak`, `eff.dynamo_recompile_not_a_lever`, `eff.pad_dynamic_shapes_ab`, `eff.depth_is_not_the_mfu_gap`); their only remaining consumers are the scripts in §2a/2b.

### 6b. Status rot: `open` pre-registrations whose measurement exists

| doc (status open) | artifact exists | fact recorded |
|---|---|---|
| `copy_hypothesis_prereg.md` (125) | `runs/copy_arms_step24000.json` | `be.self_repetition_not_context_copying` |
| `free_running_prereg.md` (75) | `runs/free_running.json` | `be.free_running_agreement_collapses_below_teacher_forced` |
| `gold_bpb_prereg.md` (83) | `runs/gold_bpb.json` | `be.gold_bpb_falls_while_generation_scores_zero` |
| `gold_reachability_prereg.md` (87) | `runs/gold_reachability.json` | `be.gold_is_ranked_high_but_not_reachable_by_greedy` |
| `preamble_refutation_prereg.md` (84) | `runs/preamble_refutation.json` | `be.preamble_is_not_the_cause_of_the_collapse` |
| `coverage_anchor_prereg.md` (80), `positional_anchored_prereg.md` (70), `positional_profile_prereg.md` (74) | `runs/*.json` present | no fact |

Eight e1 docs, 678 lines, one hypothesis chain, one day (2026-09-01), identical frontmatter template. Merge into one `docs/lessons/copy_hypothesis_chain.md` with eight sections and `status: measured`: −≈160 lines of repeated preamble, and the eight `open` flags stop lying. `fone_ab_prereg.md` (`runs/fone_ab.json` absent) stays open.

### 6c. Duplicate / superseded standards and lessons

| set | lines | overlap | action |
|---|---:|---|---|
| `docs/standards/REBUILD_PLAN_2026-08-29.md` vs `corpus_rebuild.md` vs `0830v1_gates.md` | 71 / 55 / 1,045 | REBUILD_PLAN is the pre-reset plan awaiting approval; `0830v1_gates.md` is what was approved | delete REBUILD_PLAN (§1c) |
| `docs/lessons/architecture_efficiency.md` (open, e1 lit review 08-30) + `arch_efficiency_plan.md` (08-30) vs `arch_efficiency_2x.md` (b0, measured, 08-31) + `two_x_plan.md` (L32 restatement) | 94 + 140 vs 304 + 132 | same question (MFU ceiling / where the step time goes), the first two predate t56/t57 measurement | fold the two 08-30 docs into `arch_efficiency_2x.md` §6 as a "prior estimate" table: −≈200 lines |
| `docs/standards/incremental_batch_jobs.md` | 18 | one rule + one incident; cited only by the manifest | move the rule into `sop.md` (which owns work rules) or `0830v1_gates.md`; −18 |
| `docs/audits/harness_review_0901_{3b,44,b0,de,e1,tilerl}.md` | 1,002 | six reviewers, six distinct scopes, same frontmatter | not duplicates; keep |
| `data/CORPUS_V3_CARD.md` | 145 | pre-reset card, dangling `data_recipe_v3.md` link | delete (§1c) |
| `eval/README.md` (101) vs AGENTS.md eval table | 101 | MC-suite table duplicated in `eval/run_eval.py:137-153` `MC_BENCHMARKS` | keep one: README table or the code table |

---

## Frozen files (report only, post-run)

| item | evidence | lines |
|---|---|---|
| `train.py` ↔ `infer_local.py:100-272` model mirror; `train.py:1414` ↔ `loader.py:20` `vocab_fingerprint`; `train.py:1712` ↔ `corpus_fingerprint` hash | deliberate: train imports nothing from `scripts/`/`datagen/`; `test_arch_compat.py:438` asserts parity | 0 to delete; add an equality test for the two hashes |
| `scripts/test_orphan_kill.sh` 27, `scripts/test_kill_pairing.sh` 37, `scripts/test_shard_glob.py` 178 | test `run_ddp.sh`/`train.py` kill and shard paths; run by nothing | 242, after the run |
| `scripts/supervise_run.sh` 110 | cited only by `runs/roster.json`, `runs/tasks.jsonl`; reads the live run | keep until the run ends |
| `datagen/fasttext_junk.py` 147 | named in `train.py` prose + one fact | after the run |

## Top 10 (lines removed / risk)

| # | item | lines | owner | check that must still pass |
|---|---|---:|---|---|
| 1 | Tracked non-code with zero citations (§1c): `data/synthetic/hard_eval/*` 2,080, `runs/logs/*.log` 2,337, five uncited `runs/*` 1,472, `newsrc2.jsonl` 95, `REBUILD_PLAN_2026-08-29.md` 71, `data/CORPUS_V3_CARD.md` 145 (+ hook allow-list line) | 6,200 | controller (data/runs), 44 (docs) | `harness check` (no check reads them; `no_oversized_blob`, `docs_root_clean` unaffected) |
| 2 | Zero-reference code (§1a, 21 files: one-shot measures, one-shot sweeps, tests no runner schedules, superseded launchers) | 2,376 | de (scripts 13), 3b (datagen 4), 44 (probes 2) | `harness check` (`entrypoints_ran`, `doc_commands_exist`), CI `ruff` |
| 3 | Reached only via the restartability ratchet (§2d, 12 files) — remove their rows from `scripts/restartability_baseline.json` in the same commit | 2,905 | 3b (datagen 6, mathbank 2), 44 (probes 3), de (`ckpt_sweep.py`) | `harness check` `restartability` PASS (fewer offenders is fine) |
| 4 | One-shot probes whose number is already a fact (§2b, 24 more probes after item 3) — retire each fact `source` to `probes/<f>@<sha>`, delete `padshim.py` (retracted fact) and `t7_attest_path.py` first | 2,994 | 44 | `harness check` (`probe_numbers_unique`, `selftests_are_gated` after editing the hook map, `fact_refs_resolve`) |
| 5 | Hook-only tests + eval with no runner (§1b): six `scripts/test_*.py` into CI or gone (561); `score_code_exec.py` + `code_l0prime.py` wired into `score_matrix.py` or gone (691); `stale_claims.py`, `audit_population_universals.py`, `read_lr_probe.py` (442) | 1,694 | de (scripts), 44 (eval) | `harness check` `selftests_are_gated` after removing the map lines; CI if tests are added |
| 6 | Helper dedup (§3): MC suite → `load_items` + `run_eval` table (350), `iter_jsonl` in `loader.py` (320), hash → `corpus_fingerprint` (55), argparse → `loader.add_model_args` (40), 12 AST-identical pairs (165), rest (170) | 1,100 | 44 (eval 400), de (scripts/loader 400), 3b (datagen/mathbank 300) | CI `ruff` + `python scripts/loader.py selftest` + `scripts/test_arch_compat.py`; `eval/score_matrix.py --selftest` for the MC change |
| 7 | `scripts/harness.py` (§4): delete `clean` (104 measured; `stages`/`gaps`/`ledger` KEPT — they run under the argparse default, see 4g note), 3 redundant `import subprocess` in `_broken_` fns; check merges and helper dedup NOT done, see the 4b and 4d corrections | **−86 landed** of ~890 claimed; the remaining ~800 is not there at these prices | de | `python scripts/harness.py --selftest && python scripts/harness.py check`; `agents_rules_covered` after renaming; AGENTS.md Harness table rows edited |
| 8 | `bench_eff/` (§2a, 4 trace analyzers, facts-only) — retire `source` to `@sha` | 541 | 44 | `harness check` `facts_well_formed`, `fact_refs_resolve` |
| 9 | Docs (§6): eight `*_prereg.md` → one `copy_hypothesis_chain.md` with `status: measured` (−160); `architecture_efficiency.md` + `arch_efficiency_plan.md` folded into `arch_efficiency_2x.md` (−200); `incremental_batch_jobs.md` into `sop.md` (−18) | 380 | 44 | `harness check` (`lessons_have_frontmatter`, `fact_refs_resolve`, `doc_commands_exist`); `python scripts/reachability.py > runs/reachability.txt` to un-stale the listing |
| 10 | `mathbank/` generators (§5, §2e): 27 `math_programs_*` + `run_math_short`/`run_short_sol`/`make_v11*`/`split_bank`/`dist_check`/`vet_programs`/`program_probe`/`*_curriculum`; keep `eval_hard_v2_gen.py` (live: `math_hard_eval_v2_1k.jsonl` ← `eval/math_v2_like.py:43`) | 39,900 (44 % of all Python) | 3b; **user ruling** — deletes the last generator of a frozen corpus whose provenance already records generator+seed | `harness check` (`entrypoints_ran`, `entrypoints_table_present` after editing AGENTS.md:60,76,271; `doc_commands_exist` for `audit_math_corpus.md:102-145`), CI `py_compile mathbank/*.py` (edit `ci.yml:13`), `facts/contamination.json` sources → `@sha`. Fallback if refused: shared `_reg`/`_d`/self-check base = −1,650 (4.7 %) |

Items 1–9 sum to **≈ 19,100 lines (21 % of the repo)** with no ruling needed beyond the owner's; item 10 is the single decision worth more than all nine.

Housekeeping found on the way: an untracked stale worktree at `.claude/worktrees/wf_1c60d38d-c68-12/` (branch `worktree-wf_1c60d38d-c68-12`, a full pre-09-01 copy of the tree) doubles every repo-wide grep and pollutes `reachability.py` output — `git worktree remove` it. `runs/reachability.txt` is two days and 5,000 harness lines stale.
