---
question: Do model.py / train.py / sft_math.py and the efficiency + smelt_deeploop facts say what
  their artifacts say, on 2026-09-04?
status: pass 1 (source existence) and pass 2 (40 values recomputed, seed 904) complete;
  named gaps in §5
source: user order 2026-09-04 (whole-team audit); charter docs/standards/audit_0904.md
---

# Audit: model and training code (b0, 2026-09-04)

## 1. Scope

Covered: `facts/efficiency.json` (106 facts) and `facts/smelt_deeploop.json` (18 facts) against
their cited source artifacts; `model.py` document isolation (conv, attention, KDA state);
`eval/domain_loss.py`'s cu path as it reaches the model; the four writers of
`data/corpus/*/build_corpus_stats.json` (found while pair-checking 3b, kept here because it is a
code finding); `runs/launch_tests.json` and `runs/recipe_provenance.json` as launch-gate inputs.

Deliberately NOT covered in this partial report, and not by silence — these are the named gaps:
`train.py`'s cursor/resume arithmetic beyond the two facts that cite it, the fp8 path,
checkpoint save/load beyond the `step`/`opt` fields, the `Cfg` CLI whitelist, `sft_math.py`
entirely, and the loop seam's own code (`loop_wrapper.patch_body`). Section 5 says what that
costs.

## 2. Method

`runs/audit_0904/audit_fact_sources.py`, written for this audit, with the broken-world test the
charter's principle 4 requires:

    python3 runs/audit_0904/audit_fact_sources.py --selftest
    → selftest OK: catches a missing citation, ignores prose, and fails when a
      previously-present artifact is removed

Three fixtures: a real path, a missing path, and prose (`A/B at batch16/seq4096`). It asserts one
missing, asserts the prose contributes ZERO tokens, then deletes the real artifact and asserts the
count rises to two. The third assertion is the one that matters — an instrument that reports
"nothing missing" because its tokeniser found nothing is indistinguishable from a clean result.

**The tokeniser is the finding, not a detail.** My first pass used `'/' in token` and reported 94
of 238 tokens missing, because it read `A/B`, `batch16/seq4096` and `batch16/accum2/seq4096` as
paths. A real absence in that list is invisible. A token now counts only if it carries a known
file extension or its leading directory exists in the repo; `@sha` citations are resolved with
`git cat-file -e <sha>:<path>`, because a probe deleted after it ran is honestly cited by sha and
demanding it on disk would report an honest citation as broken.

Every unresolved citation was then classified by opening it in three places: this checkout, the
pod (`~/bin/pod`, one batched `ls` per path), and `git ls-files` for bare basenames.

Pod artifacts were opened, not inferred: `torch.load(..., mmap=True)` for checkpoint fields,
`json.load` for stamps, `grep -n` for source lines.

## 3. Population counts

| population | exists | read | sampled |
|---|---|---|---|
| facts in the two files | 124 | 124 (every `source` field) | 8 read in full |
| path-shaped source tokens | 252 | 252 | — |
| unresolved from this checkout | 57 | 57 classified | — |
| fact ids citing a pod-only path | 15 | 15 | 2 verified on the pod |
| fact ids citing a path absent everywhere | 12 | 12 | 4 chased through git history |
| corpus stamps opened on the pod | 47 | 47 (`filters_fp`) | 6 opened in full |
| facts whose VALUE was recomputed (pass 2) | 124 | 40 sampled at seed 904 | 21 PARTIAL classified by hand, 4 re-derived arithmetically |
| `runs/score_matrix.jsonl` rows | 60 | 60 (`metrics`/`skipped` keys) | 10 `domain_bpb` entries read in full |

## 4. Findings

| id | sev | claim as published | evidence (path/cmd) | what contradicts it |
|---|---|---|---|---|
| MT-1 | S1 | `eff.kv_pool_undersized_for_serving`: "REFUTED… the pool holds 256 x 16 = 4096 tokens — HALF the declared budget", `boundary`: "Raising num_blocks is the fix… **Not done**: the line was paused before restart." | The two cited files are not in this repo and not in `git log --all`; they live in a DIFFERENT repository, `/work/tilerl/src/tilerl/{cli,kv_cache}.py` (found by `find /work/tilerl -name kv_cache.py`). Opened there: `kv_cache.py:19 BLOCK_TOKENS = 16` still holds, but `cli.py:57` now reads `num_blocks=max(256, (ctx * 8) // BLOCK_TOKENS)` — the hardcoded 256 is GONE. At ctx=8192 that is 4096 blocks = 65,536 tokens against a declared 8192: **8× the budget, not half.** `cmd_serve:75` → `_build_engine` → that line, so it is the serve path the fact is about. cli.py mtime 2026-09-02 13:08, three days AFTER the fact's `measured: 2026-08-30`. | The fact's `value` describes code that no longer exists and its `boundary` asserts the fix is "not done" when it is done. A reader deciding serving concurrency today would read "safe concurrency is 1, not 4" off a file provisioning 8× headroom. |
| MT-2 | S2 | 15 fact ids cite artifacts as their evidence. | `audit_fact_sources.py` + a batched pod `ls`: all 15 cite at least one path **present on the pod and absent from this checkout**. Full list in §4a. Every one stat'd on the pod — see the §4a table. `bench_eff/ddp_trace_rank0.json` is cited by five facts and is **415,798,937 bytes**; `runs/trace_p200m_3step.json` by four at 59,446,282; `runs/warmup_smoke*.log` globs to zero here and to three files totalling 17,781 bytes there. | A fact's source is a promise a later reader can open it. These cannot be opened from where the facts are read, and `facts_well_formed` runs on main, so nothing reports it. The 57 MB trace explains why one was never pulled — this is a real constraint, not an oversight, which is why it needs a decision rather than a fix. |
| MT-3 | S2 | `data/corpus/*/build_corpus_stats.json` conforms to `CANONICAL_STATS_KEYS` (15 keys), enforced by `_assert_canonical_stats`. | Four writers of that filename: `datagen/build_corpus.py` (asserts, :687), `datagen/build_cot.py:97`, `datagen/code_dedup_build.py:162`, `datagen/build_code_tests_v1.py:408`. `grep -c _assert_canonical_stats` → **0, 0, 0** for the latter three. `build_cot.py:90-96` writes `srcfp`/`criterion`/`schema`/`docs_in`/`docs_kept`/`reject_checks`/`tokens_kept`/`check4`, of which only `n_shards` and `filters` are canonical keys. | The schema governs one writer in four. A reader consulting it to learn what a stamp guarantees learns nothing about three quarters of the writers. Cross-referenced: this corrects 3b's CD-1 mechanism in `corpus_data.md` (which cited `build_corpus.py:664` as *requiring* `tokens_config`; :678-681 writes tokens/tokens_status/tokens_config in ONE branch, so zh_web's stamp — `tokens` + `"measured"`, no `tokens_config` — cannot have come from the conforming writer at all). |
| MT-4 | S2 | `runs/launch_tests.json` records that the arch tests passed on the shape being launched. | One row per test FILE, last write wins (`scripts/launch_tests.py:105`, `rows[key] = {...}` keyed on the test's repo path). Measured live, three writes in sequence: `test_arch_L32` at d768 **L12** (03:0xZ), then at d768 **L16** (recorded 03:04Z) — which ERASED the L12 row, verified by reading the file and finding only `{'d':768,'ffn_hidden':2304,'heads':6,'layers':16}` — then at **L12** again (03:14Z) immediately before launching arm 2, which erased the L16 row in turn. | Two arms at two depths cannot both hold a certificate; certifying one erases the other, and whichever launches second launches on a green describing the other model. Stage E runs exactly that shape pair. |
| MT-5 | S2 | Same file: a `pass` row means the run passed. | `scripts/test_e2e.py` calls `record_launch_test` at the end of its `try` block (:402) while stage 11 runs in the `finally` (:423). Observed: `runs/b0_se_e2e_L12.log` ends `AssertionError: resume from step-less … did not refuse (rc=0)`, and `launch_tests.json` holds `result: pass` for that same run. | The gate's record asserts green for a run that exited on an AssertionError — the class of thing the gate exists to prevent. Would clear `arch_tests` for the next launcher who does not read the log. |
| MT-6 | S3 | `test_e2e` stage 11: "resume from the step-less final save refuses". | `de-31` (`c9011022`, 2026-09-03) changed train.py's run-end save to pass `opt_snapshot(optimizers), step` — deliberately, because the bare save carried no `row_cursor` and silently restored a cold optimizer. `torch.load` of `ckpt_e2e_tmp.pt`: `step=6`, `opt` present. `train.py:2229` refuses only on `missing = [k for k in ("step","opt") if k not in ck]`. | The test asserts a postcondition its own repo removed on purpose. Not a code defect — a test behind the code, which fails every e2e run at every shape. |
| MT-7 | S2 | `runs/recipe_provenance.json` gives every launched value a source. | No group matched `d768 L12 h6 ffn2304` until I added one (`fd6a382a`), and **four completed Stage D runs had already used that shape** (`b0_sd_unlooped`, `b0_sd_looped`, `b0_sd_equalcompute`, `b0_n8_fixed`). `gate_launch_command` reconciles argv against these keys. | Twelve values were passed on the command line across four runs with no recorded source, and the gate read GO because `_recipe_for_shape` found *a* recipe. Same defect the shape partition was built to remove, one shape later. |
| MT-8 | S3 | 12 fact ids cite a path resolvable nowhere. | `git log --all --diff-filter=D` for each: `ckpt_cost.py`, `bench_short_conv2.py` exist in **no commit, on no machine, and never did**. `kv_cache.py`/`cli.py` are MT-1 (another repo). `bench_eff2.py`→`bench_eff/bench_eff2.py`, `gate_failure_shapes.md`→`docs/lessons/…`, `score_matrix.jsonl`→`runs/…` resolve as bare basenames. Brace-expansion residue (`ckpt_…pt.step{2500`) accounts for 7 of the 12 — same class as the `runs/b0_sd_he_{…}` defect I hit on 2026-09-03. | Severity is S3, not higher, BECAUSE the two facts whose scripts vanished keep their logs: `eff.short_conv_shifted_madd` cites `runs/p02_sc_{base,patched}.log` and `eff.ckpt_resume_16h_interval` cites `runs/t38_{ref,resume}.log`, all four present on main AND the pod. The measurement survives; only re-derivation is lost. |
| MT-9 | S2 | The two fact files are the repo's record of what it measured. | §4c: of 40 facts sampled at seed 904, **30 cannot be recomputed by a reader on main** — 9 need the pod, 11 need a card, 4 are literature-only (`smelt.*`), 6 cite a script in no commit. Instrument `runs/audit_0904/audit_fact_values.py`. | Nothing is CONTRADICTED — zero wrong values in 40 — but the repo cannot demonstrate its own numbers to anyone who was not there. Pass 1 found 15 facts with an unopenable source; pass 2 puts the sampled rate at 75%, so 15 was a floor and not a measure. |
| MT-10 | S2 | `sft_math.py:190` asserts a pack's `vocab_id` equals the checkpoint's — "a pack from another vocabulary trains silently at ~4x the loss: every id is wrong and in range, and the sizes match." | The file's OWN comment (:181-187) records that the condition read `"vocab" in d` while `prepare_sft.pack_and_save` writes `"vocab_id"`, so for every pack built by the current packer the assert was skipped and the run took the WARNING branch — "the pack predates vocabulary fingerprinting" printed about a pack that carries the fingerprint. Fixed 2026-09-03. **Nothing has exercised the fix.** `grep -rln pack_vocab --include=*.py` returns `sft_math.py` and `scripts/check_sft_ready.py` and no test; `runs/experiments.jsonl` has **zero** SFT runs after 2026-09-03. | The guard has never fired in either state — not once while broken (the whole point of the comment), and not once since the repair. Its correctness rests on a code reading alone, which is the same shape as `guards-dont-backfill`: the fix is present, the demonstration that it works is not. |
| MT-11 | S2 | `sft_math.py:496` on a NaN step: `step {step}/{total} NaN — restored last good state`, and the run continues. | `good_state` is initialised at `:413` before the loop and refreshed ONLY at `if step % args.save_every == 0` (`:510`), and `--save_every` defaults to `SAVE_INTERVAL = 200` (`:45`). So the restore target is the last multiple of 200, not the previous step. Nothing records the distance: `grep -c good_step` → **0**, and the log line names neither the step the state came from nor how many steps were discarded. | A NaN at step 399 silently discards 199 optimizer steps and the run reports one line about it. Two runs that both "recovered from a NaN" can differ by 200 steps of training with nothing in the artifact to distinguish them, and the token count the run reports is unaffected because `step` is incremented on the rollback path too (`:501`). Not a correctness bug — the rollback is real and the state is consistent — but the run's own record cannot say what it lost. |
| MT-12 | S1 | `eval/domain_bpb.py`'s failures are recorded in `runs/score_matrix.jsonl` as `domain_bpb.py exited 1: /work/aupai/eval/domain_bpb.py:221: UserWarning: checkpoint has no vocab_id (old format); cannot cross-check tokenizer \| ours_tok = load_tokenizer(a.tokenizer, None)`, seven rows of it (rows 55-60 plus arm 1's live run). | **The quoted text is a WARNING and a source line; neither is the cause, and the cause is in no artifact.** Three separate facts, each checked: (1) The warning is unconditional. `eval/domain_bpb.py:221` passes `cfg=None` DELIBERATELY (`:222-235` explains it: both arms must reach `val_seqs` and the `--hf` control has no cfg), and `scripts/loader.py:126` reads `getattr(cfg,"vocab_id",None)`, which for `None` is always `None`, so `:136` warns on every call. Arm 1's checkpoint, written minutes before its row, carries `vocab_id: '0bce3584bc24f255'` at top level — I loaded it. (2) `main()` has exactly ONE nonzero return, `:293`, reached from `:291 if not out:` after printing `REFUSING: no domain produced a number` — **to stdout**. The captured stderr for these rows is exactly two non-blank lines (warning + its `stacklevel` source echo) with no traceback and no `*Error` line, so no exception was raised: every one of these rows is that `REFUSING`, and the per-domain reason is in `skipped[name]` at `:280`, discarded with the rest of stdout. (3) `eval/score_matrix.py:304` reads `(r.stderr or r.stdout)`. stderr is non-empty — the unconditional warning alone fills it — so **stdout is never read**. Reproduced with a fixture reproducing the stream layout: captured the warning, discarded `REFUSING: no domain produced a number`. | `domain_bpb` has produced a number in **zero of 60 rows** of `runs/score_matrix.jsonl`: ten rows carry `metrics.domain_bpb`, all ten an `{"error": …}` object. It failed in two eras — rows 51-53 on the `eval/cache_guard.py:140` retokenise refusal, rows 54-60 after `2d37eede` (2026-09-03 17:04) replaced that with the present state — and `2d37eede`'s own message says "all three of its rows … are that error", written when three was the count; seven more have failed since under a cause its record cannot name. The stream-preference defect is what sustains it: `57a1177c` (2026-09-03 18:48) was committed as "record WHAT failed, not just where" and its comment states the old capture "recorded domain_bpb's failure as the SOURCE LINE … which names where, never what" — that is still exactly what rows 59-60, measured 2026-09-04 after the fix, contain. Two fixes have been aimed at this metric and neither reached it. |
| MT-13 | S3 | `scripts/loader.py:136`'s warning tells a reader their checkpoint predates vocabulary fingerprinting. | It fires on every `load_tokenizer(path, None)` call, of which `eval/domain_bpb.py:221` is one by design. It cannot distinguish "no cfg was supplied" from "the cfg has no vocab_id". | A warning that fires whether or not its condition holds carries no information, and here it does active harm: it keeps stderr non-empty, which is the precondition for MT-12's `stderr or stdout` masking. The `None` call site is deliberate and should stay; the warning needs to say which of the two cases it is in. |

### 4a. Every pod-only source, with size and committability (MT-2)

Sizes are `stat -c %s` on the pod in this pass. "committable" = under 5 MB, the threshold the
controller asked for; it is a size judgement only, not a ruling that the artifact belongs in git.

| path (pod, repo-relative) | bytes | <5 MB | cited by |
|---|---:|---|---|
| `bench_eff/ddp_trace_rank0.json` | 415,798,937 | no | `eff.dynamo_recompile_not_a_lever`, `eff.fusion_and_elementwise_are_disjoint_but_the_trace_is_off_config`, `eff.quant_tax_is_the_elementwise_group`, `eff.steady_state_composition`, `eff.step_remainder_attribution` |
| `ckpt_ab_untiehead_untiehead.pt.ep1` | 2,187,661,555 | no | `eff.tied_head_does_not_inflate_tok_200m` |
| `ckpt_ab_untieheadlr_untieheadlr.pt.ep1` | 2,187,664,147 | no | `eff.tied_head_does_not_inflate_tok_200m` |
| `ckpt_ab_valueembed_valueembed.pt.ep1` | 2,053,519,923 | no | `eff.padded_vocab_table_no_pay_200m` |
| `ckpt_ab_shapelr_base.pt.ep1` | 1,784,216,707 | no | `eff.padded_vocab_table_no_pay_200m`, `eff.tied_head_does_not_inflate_tok_200m` |
| `ckpt_p200m_4b_0902.pt.interrupt.step832` | 959,435,257 | no | `eff.kda_mla_growth_ratio_l12` |
| `ckpt_p324.pt` | 412,319,307 | no | `eff.bf16_updates_discarded` |
| `runs/trace_p200m_3step.json` | 59,446,282 | no | `eff.clip_and_sync_cost_p200m`, `eff.optimizer_step_gpu_cost_p200m`, `eff.step_class_breakdown_p200m_4card`, `eff.step_roofline_p200m_4card` |
| `runs/n7_domain.jsonl` | 55,804 | **YES** | `docs/standards/roadmap_0903.md:24` (Stage A) — a DOC citation, not a fact; supplied by e1 |
| `runs/warmup_smoke*.log` (3 files) | 17,781 | **YES** | `eff.warmup_absolute_not_fractional` |
| `runs/milestone_p324_v2.jsonl` | 1,882 | **YES** | `eff.light_profile_wall` |
| `runs/p500m_20b_0902.log` | 65,990 | no | `eff.p500m_20b_throughput_and_dips` (config) — from 58's sweep |
| `runs/eval_p500m_step1500_base.log` | 5,247 | **YES** | `eff.p500m_20b_throughput_and_dips` (config) — from 58's sweep |
| `runs/eval_p500m_step1500_l1.log` | 40 | **YES** | `eff.p500m_20b_throughput_and_dips` (value + config) — from 58's sweep |
| `runs/ppl_step1500_v2.log` | **0** | **YES** | `eff.p500m_20b_throughput_and_dips` (value + config) — from 58's sweep |

**Six of the fifteen rows are committable, and they are 80,754 bytes together.** The other eight are
checkpoints and profiler traces — 9.9 GB in total, one of them 415 MB — so "pull them into the
repo" is not available for the paths that matter most. That asymmetry is the finding: the
citations that CANNOT be committed are exactly the ones carrying five, four and two facts each,
because a big artifact is what an expensive measurement produces.

What the small three cost to fix is one commit. What the large eight need is a decision: a hash
and a size recorded beside the path so a later reader can verify the artifact they open is the one
that was measured, or an explicit convention that a pod path is a legitimate citation and
`facts_well_formed` reports it as pod-resident rather than staying silent. Right now a reader on
main cannot distinguish "this artifact is on the pod" from "this artifact is gone", and MT-8 shows
both cases exist.


**MT-2b (S3, and it corrects a stronger claim).** 58's sweep — over `source`, `config`,
`uncertainty` AND `boundary`, where mine read `source` only — found four paths mine missed, all
four confirmed by my own `stat` on the pod. Two are near-empty: `runs/ppl_step1500_v2.log` is
**0 bytes** and `runs/eval_p500m_step1500_l1.log` is **40 bytes** (one header line, no result).
58 reads the zero-byte one as a citation that "cannot be satisfied anywhere". **It can, and this
is worth stating precisely because the opposite reading would retract a sound fact.**
`eff.p500m_20b_throughput_and_dips` cites these two files for their **mtime**, not their contents:
it attributes throughput dips to eval jobs contending for the cards, and the evidence is *that a
job wrote a file at that instant*. I read both: `ppl_step1500_v2.log` mtime **07:25:52** and
`eval_p500m_step1500_l1.log` mtime **06:29:12**, exactly the two timestamps the fact's `value`
names, and the l1 log's single line is verbatim the string the fact quotes
(`L1 few-shot: 3 demos, 497 eval problems`). An empty file with a trustworthy mtime is sufficient
evidence for a contention claim. So the finding here is narrower than 58's: the two files are
**pod-only**, which is MT-2, and separately the l1 eval **produced no result** — a fact about that
eval, not about this fact's support.

Also recording 58's instrument defect because it is the same class as my own tokeniser problem: a
regex alternating `json|jsonl` matches `.json` first and truncates every `.jsonl` path, which
reported 21 absent paths of which 15 were ledgers sitting in the repo. Their published 7 is the
corrected count. Two independent sweeps, two tokeniser defects, both over-reporting — the
enumeration step is where this class of audit fails, not the checking step.

### 4b. Checked and SOUND — recorded so absence of a finding is not read as absence of a check

- **`model.py` conv document isolation** (`DeltaRecurrence.forward`, :116-146). The masked branch
  is arithmetically right: tap `i` at output `t` reads input `t-(K-1-i)`, and the mask
  `((pos - (K-1-i)) >= doc_start)` is exactly the condition that read lands in the same document.
  `cu` indexes the flat `B*T` stream while `h` is `[B,D,T]`, and the mask is built flat then viewed
  `[B,1,T]` to broadcast over D — correct. `doc_start` via `bucketize(pos, cu[1:], right=True)`
  indexes `cu[:-1]`, which is the document start for every position.
- **The flag-off branch is deliberately NOT the masked form with an all-ones tap** (:147-154), so a
  pre-flag checkpoint scores bitwise as it trained. I verified this by gate rather than by reading:
  `scripts/b0_n8_reuse_gate.py` measured 576/576 bitwise identical, worst |diff| 0.000000e+00.
- **`eval/domain_loss.py` now passes cu.** The missing-`cu` defect I found on 2026-09-03
  (`model(x[i:i+bs])` with `HybridLM.forward`'s `cu=None` default) is fixed: `_ce` takes
  `cu_path`, and `doc_cu` routes through `train.doc_cu_seqlens` (:245-253). The DEFAULT is still
  `cu_none` at three call sites (:174, :194, :215) — correct, since changing it would rescore every
  published number against a different reference, and that is a ruling, not a cleanup.
- **`ckpt_p200m_4b_0902.pt.ep1`** — 44's pin request. Both sides recomputed independently before
  pinning: ep1 `row_cursor` sum **976384** over 9 domains at `as_of_step 3814`; bare `.pt`
  **1189548** with `as_of_step` **ABSENT**. Matches
  `eff.run_end_cursor_overstates_under_max_steps` exactly — the fact reproduces. Pinned as
  `ckpt_p200m_4b_0902.milestone_keep_44_runendcursor.pt`, inode **84199064**, nlink 2.


## 4c. Pass 2: 40 fact values recomputed at seed 904 (controller's Q5 ruling)

Instrument: `runs/audit_0904/audit_fact_values.py --selftest` then no flag. Sample drawn
`random.Random(904).sample(sorted(ids), 40)` from all 124 facts — seed 904 matches 44's so the
two samples can be checked for overlap.

**HEADLINE: not one contradicted number in 40 facts.** Every discrepancy the instrument raised
resolved, on inspection, to a number the cited artifact could not carry. That is a real result and
it is also a weak one — §5.2 below says why.

    HELD 6   PARTIAL 21   NO-FILE 13

**The instrument was wrong three times before it was right, and each defect over-reported.**
Recorded because the pattern is now three-for-three across two auditors: in this class of audit
the enumeration step fails, not the checking step (58's `json|jsonl` alternation truncated every
`.jsonl` path; my pass-1 tokeniser read `A/B` as a path).

1. A trailing `\b` on the number regex killed every K/M/G-suffixed figure — `11.87K` matched
   nothing, and most throughput numbers in this repo are suffixed.
2. `\b\d{3,}(?:,\d{3})*` matched `189,548` INSIDE `1,189,548`, so a 7-digit grouped number
   truncated and the search then looked for a number no artifact contains.
3. Exact string matching called precision disagreement. First run: 22 of 40 PARTIAL, and the first
   I checked by hand was `repo.loop_from_scratch_stage_d` "missing" 122.30 while its log prints
   `params 122.3M`. Same number. Replaced with numeric comparison at the ARTIFACT's precision —
   and the fixture that forced that (`11.9` claimed against a log reading `11.87`) caught my first
   fix, which truncated the wanted string to `11.` and accepted it.

**All 21 PARTIALs classified by hand.** None is a wrong value:

| what the missing number is | facts |
|---|---|
| explicitly `derived:` or a computed ratio/delta across two runs | `repo.moe_a2a_cost_h20`, `eff.bucket_cap_mb_7gpu_ab`, `eff.ddp_5k_not_identified`, `eff.attn_every_1_has_no_position_information` |
| stdout of a probe cited BY SHA and since deleted | `eff.bf16_updates_discarded`, `eff.fp8_head_activation_range`, `eff.kda_chunk_size_32` |
| in a pod-only artifact (MT-2) | `eff.clip_and_sync_cost_p200m`, `eff.quant_tax_is_the_elementwise_group`, `eff.p500m_20b_throughput_and_dips`, `eff.tied_head_does_not_inflate_tok_200m` |
| computed from checkpoint tensors, never printed | `eff.layer9_mixer_o_lags`, `eff.resume_inflates_total_steps`, `repo.loop_from_scratch_stage_d` |
| a microbenchmark whose script was never committed (MT-8) | `eff.short_conv_shifted_madd`, `eff.ckpt_resume_16h_interval` |
| a PID from a live pod reproduction in `/tmp` | `eff.wrapper_orphans_torchrun` (6173) |
| a LINE NUMBER, not a measurement | `repo.tpp_and_step_profile` (2361 = `facts/efficiency.json:2361`) |
| host tool output not committed | `repo.nv18_topology_measured` (nvidia-smi topo) |
| a breakdown computed from a logged total | `eff.w7_peak_memory_b32_fits`, `eff.seam_dynamo_disable` |

**Two facts I had provisionally called unrecoverable DO reproduce, and I checked before writing
it down.** `eff.seam_dynamo_disable`: `runs/t57_seam.log` is present on main and carries `218` and
`70 flash` and a final `--- flash recompiles --- 0`, which is the fact's whole claim (70 → 0).
`eff.ckpt_resume_16h_interval`: the resume-equivalence numbers are exactly there — `5.874` in
`runs/t38_ref.log`, `5.841` in `runs/t38_resume.log`. Only their secondary write-cost figures
(959 MB, 4.91 s) came from the uncommitted `ckpt_cost.py`. So MT-8's severity holds at S3 for a
second, independently-checked reason.

**Where the 40 can be recomputed, which is the finding that matters more than the tally:**

| recomputable | count | note |
|---|---:|---|
| on main today | 10 | logs and code present here |
| needs the pod | 9 | MT-2's artifacts |
| needs a card | 11 | a forward pass or a kernel benchmark; NOT sampled, per the ruling |
| literature only | 4 | `smelt.*` — arXiv 2609.01343, nothing local to check |
| neither, script gone | 6 | but see the two above that reproduce from logs anyway |

Per the ruling the 11 "needs card" facts are listed rather than sampled: `eff.fp8_head_activation_range`,
`eff.kda_chunk_size_32`, `eff.attnres_internal`, `eff.gpu4_peak_flops`, `eff.vocab_alignment_2x_lm_head`,
`eff.attn_every_1_has_no_position_information`, `eff.launch_overhead_is_not_a_cost`,
`eff.launch_reduction_not_a_lever`, `eff.throughput_quality_exchange_rate`, `eff.fb_data_curve`,
`eff.w7_peak_memory_b32_fits`.

**MT-9 in the table above is what pass 2 produced.** 30 of 40 sampled facts cannot be recomputed
by a reader on main: 9 need the pod, 11 need a card, 4 are literature, 6 cite a vanished script.
The published numbers are not in doubt — nothing contradicted — but the repo cannot demonstrate
that to anyone who was not there. Pass 1 found this for source EXISTENCE at 15 facts; pass 2 finds
it for VALUE at 30 of 40, so the sampled rate is 75% and the pass-1 count was a floor, not a
measure.


- **`conv_doc_isolated`'s checkpoint round-trip, and every path that could bypass its guard.**
  `scripts/loader.py:67` pins `cfg.conv_doc_isolated = False` when the key is absent, rather than
  letting the generic live-default backfill at `:57-59` supply it — correct, and its comment states
  why: the flag is NOT numerically neutral, so backfilling it would score a checkpoint in a topology
  it never trained in. The population question is whether every model build goes through that
  loader. Enumerated over `git ls-files '*.py'`: five files construct `HybridLM` from a
  checkpoint's own cfg, three via the loader (`scripts/eval_heldout.py`, `scripts/test_e2e.py`,
  the loader itself) and **two that bypass it** — `scripts/ve_row_norms.py:302` and
  `probes/fone_digit_acc.py:39`. Both are harmless, for DIFFERENT reasons, and the reasons are the
  point: ve_row_norms builds the model only to read `value_embed.weight` at init and never runs a
  forward, so no mixer executes; fone_digit_acc does forward (`:66`) but passes `cu=None`, and the
  branch condition is `self.conv_doc_isolated and cu is not None`, so the masked path is
  unreachable. **No finding, but the second reason is the missing-`cu` shape again** — the same
  condition that made `eval/domain_loss.py` score packed rows as one undivided sequence for weeks.
  It is benign here because this probe measures per-digit accuracy on FoNE positions rather than a
  likelihood, so document bleed does not enter the statistic. If it is ever repurposed to report a
  loss, it inherits the −0.082 nat/token artifact.


- **The fp8 path: which linears it converts, at two shapes.** `train.py:384` `_fp8_ok` excludes by
  NAME (`head`, `num_proj`, `num_head`) and by `all(d % 16 == 0 for d in mod.weight.shape)`, and
  `convert_to_float8_training`'s filter passes `fqn.rsplit(".", 1)[-1]` — the LAST FQN segment, so a
  nested module whose own name collided with an excluded one would be skipped silently. Enumerated
  by constructing the real model rather than reading the traversal. Stage E shape (d768 L12 h6
  ffn2304): **64 `nn.Linear`, 63 converted, 1 excluded** — `head`, by name, which is
  `eff.lm_head_is_compute_bound`'s deliberate choice — and NO tail-name collision (`head` occurs
  once). FoNE arm (`fone=True`): 66 linears, 3 excluded, all three by name and all three at depth 0.
  So the name filter's population is exactly its intent at both shapes, and the `rsplit` hazard is
  latent rather than live. SOUND.
- **The fp8 forward/backward's scale handling** (`train.py:316-362`, the legacy `FP8LinearFunction`).
  Forward computes per-tensor absmax scales, caches `x_fp8`/`w_fp8` AND both scales for backward
  (`:334`) — the comment's "5 quants → 3" — and backward re-derives only `go_scale` from the
  incoming gradient. `grad_b = go2d.sum(0)` stays in bf16, never quantised, which is right: a bias
  gradient is a reduction, not a matmul. What I did NOT verify is the numerics, and `:375-377`
  records why that matters: Inductor's min-cut partitioner recomputes the saved fp8 tensors,
  re-dividing already-scaled values into NaN grads at step 1 without `grad_ckpt`, which is why this
  module is kept out of the compiled graph. That failure and its guard are load-bearing and
  untested by me — it needs a card. **Note this is not the live path**: `convert_to_fp8_compute`
  prefers torchao's `Float8Linear` and reaches the legacy class only when torchao is missing or
  `FP8_RECIPE=legacy`. Arm 1's log says `FP8 compute enabled` with torchao present, so every Stage D
  and Stage E number was measured on the torchao path, not this one.

## 5. Blind spots of this audit

1. **`sft_math.py`: load path, vocab_id refusal, training loop and rollback now read
   (MT-10, MT-11); the masking internals are not.** Read: `:150` the checkpoint load, `:175` the
   pack load, `:180-212` the vocab_id comparison and the holdout-stamp refusal, `:403-411` the DDP
   and compile wrap, `:413` the rollback buffer's initialisation, `:445-470` the loss path,
   `:471-481` the fone auxiliary loss and clip, `:484-501` the NaN health check and restore,
   `:503-530` the schedule, the optimizer step and the mid-run save.

   Two things checked by RUNNING rather than by reading, because both are the kind of aliasing
   question a code read gets wrong. `weight = raw_model.head.weight[: cfg.vocab]` is sliced ONCE at
   `:445` and used every step at `:470`; a slice of a Parameter tracks in-place optimizer updates
   but NOT a `.data` rebind, so I checked both — the repo rebinds nowhere (`grep '\.data = '` over
   sft_math.py and train.py: no hits) and the `:491` rollback uses `load_state_dict`, which copies
   in place. Verified on a toy module: after a rollback the slice sees the restored values and
   `data_ptr()` still matches. So the loss always uses current weights. SOUND.

   NOT read: `format_agentic`'s per-turn boundary construction in the packer (it is
   `datagen/prepare_sft_math.py`, outside this file and outside my area's named scope), the
   prefix-LM mask's `_plens` implementation, and the distributed init. train.py:2228 notes SFT is
   "the legitimate step-0 case and has its own loader", so the save/load findings above do not
   automatically transfer and I have not checked whether they do.
2. **Pass 2 closes the count and not the depth.** 40 values are now checked (§4c), above the
   charter's 30, but the check is a TEXT SEARCH for the number in the artifact. It proves a number
   appears where the fact says it does; it does not prove the artifact's own arithmetic. A median
   quoted faithfully from a log whose step lines are wrong reads as HELD. Of the 40 I re-derived
   arithmetic for exactly four (MT-1's block product, the two cursor sums in §4b, and the
   1.264604 ratio 44 independently confirmed). That is now the largest gap.
3. **`filters_fp` uniformity was measured, filter EQUIVALENCE was not.** 47 stamps share 3 values;
   whether two domains with the same `filters_fp` and different `filters` descriptions actually got
   the same filtering is not answered by the hash and not by me.
4. **`train.py`'s cursor arithmetic** is checked only where a fact cites it. The `row_cursor`
   as_of_step semantics that MT-4b touches are one field of a larger mechanism.
5. **fp8: the conversion population and the scale plumbing were read (§4b), the NUMERICS were
   not.** I did not run an fp8 step, so the claim I cannot check is the one that matters — that
   e4m3 tensorwise scaling with grad_output in e4m3 is stable where stock `tensorwise` (e5m2
   backward) was not. `train.py:485-487` asserts both that and a torchao failure mode
   (`aten.clone.default with axiswise scaling`) from experience; neither is reproducible without a
   card. Nor did I verify the NaN-at-step-1 partitioner interaction at `:375-377`, which is the
   reason the legacy module is dynamo-disabled.
6. **MT-1's history is unreconstructible.** `/work/tilerl` has no `.git`, so I cannot date the
   `num_blocks` change or see whether the 256 was ever there — only that the file post-dates the
   fact by three days and does not contain it now.

7. **MT-12 is one metric of `score_matrix.py`'s several; the masking is in shared code.**
   `(r.stderr or r.stdout)` at `eval/score_matrix.py:304` is the capture for EVERY script the
   matrix drives, not just `domain_bpb`. Any of them that writes its reason to stdout while
   anything at all — a `UserWarning`, a torch deprecation, a NCCL banner — reaches stderr will be
   recorded by the wrong stream. I checked this for `domain_bpb` only, and only because its rows
   were the ones in front of me. I did not enumerate the other scripts the matrix runs, or check
   which of them print their refusal to stdout. That enumeration is the natural next step and I
   did not do it, so the count of affected metrics is unknown, not one.

## 6. Open questions for the controller

1. **MT-1**: retract `eff.kv_pool_undersized_for_serving`, or re-measure against the current
   `/work/tilerl` and rewrite it? Its `refuted_by` names "a run at 2-way concurrency that does not
   exhaust the pool", which the present code would pass — so by its own criterion it is refuted.
2. **MT-2**: should a fact be allowed to cite a pod-only artifact at all? Options are pull it
   (59 MB for the trace), record a hash and size beside the path, or declare pod-only citations
   legitimate and teach `facts_well_formed` to say so rather than staying silent.
3. **MT-4/MT-5** are de's per your ruling — confirming they are out of my hands, not dropped.
4. Does a fact whose derivation script never existed in any commit (MT-8:
   `ckpt_cost.py`, `bench_short_conv2.py`) need a status change, given its measurement logs survive?
5. Should §5.2 be closed before this report is accepted — i.e. do you want the ≥30-value
   recomputation the charter asks for as a second pass, or is the source-existence sweep the
   deliverable?
6. **MT-12** is a live gate hole, not a historical one: rows 59-60 were measured 2026-09-04 and
   `domain_bpb` has produced a number in 0 of 60 `score_matrix.jsonl` rows. Two commits have
   already been aimed at it (`2d37eede`, `57a1177c`) and neither reached it, because each was
   written from the record's text and the record's text is the defect. Whose is the fix — the
   `stderr or stdout` preference at `eval/score_matrix.py:304` is score_matrix's, the unconditional
   warning at `scripts/loader.py:136` is the loader's, and the `REFUSING`-to-stdout at
   `eval/domain_bpb.py:292` is domain_bpb's. Any ONE of the three would have made the cause
   visible. I have not touched them: audit only.
