---
question: Do model.py / train.py / sft_math.py and the efficiency + smelt_deeploop facts say what
  their artifacts say, on 2026-09-04?
status: partial — first report at the 3h mark, per the charter's "partial beats late"
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

**Three of eleven are committable, and they are 75,467 bytes together.** The other eight are
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

## 5. Blind spots of this audit

1. **`sft_math.py` was not read at all.** It has its own checkpoint loader (train.py:2228 notes SFT
   is "the legitimate step-0 case and has its own loader"), so every save/load finding above may or
   may not transfer, and I did not check.
2. **The instrument reads `source` fields only.** A fact whose *number* is wrong but whose artifact
   exists passes it silently. Of 124 facts I recomputed the value of exactly one (MT-1's, plus the
   two cursor sums in 4b) — the charter asks for a fixed-seed sample of ≥30 and this report has 3.
   That is the largest gap here.
3. **`filters_fp` uniformity was measured, filter EQUIVALENCE was not.** 47 stamps share 3 values;
   whether two domains with the same `filters_fp` and different `filters` descriptions actually got
   the same filtering is not answered by the hash and not by me.
4. **`train.py`'s cursor arithmetic** is checked only where a fact cites it. The `row_cursor`
   as_of_step semantics that MT-4b touches are one field of a larger mechanism.
5. **fp8 was not exercised.** `eff.fp8_transpose_cast_no_config_lever` cites a torchao path
   (third-party, correctly outside the repo), and I neither ran nor read the fp8 code.
6. **MT-1's history is unreconstructible.** `/work/tilerl` has no `.git`, so I cannot date the
   `num_blocks` change or see whether the 256 was ever there — only that the file post-dates the
   fact by three days and does not contain it now.

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
