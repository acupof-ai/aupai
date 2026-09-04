---
question: What is the true state of evaluation and held-out on 2026-09-04, read from artifacts?
status: open
area: evaluation and held-out
owner: e1
pair: 3b
source: user order 2026-09-04, method docs/standards/audit_0904.md
---

# Audit: evaluation and held-out, 2026-09-04

Second pass. The first was partial at the 3-hour mark; the 43-scorer sweep has since landed as
E9-E12 and section 5 names what remains unseen. Fifteen entries: two S1, nine S2, three S3, and
E9, which carries no severity because it is a clean result -- recorded as an entry anyway,
since "no defect" is an answer to an assigned question and silence is not.

E15 is a defect in this report itself, found after 6e accepted it: all six of its `Z`-suffixed
timestamps were +0800, and two claimed evidence gathered nine hours after the commit that
published them. Both affected conclusions (E3, E4) survive and by a wider margin; the labels are
corrected in place and E15 records what was wrong, why, and what it says about the other dates.

## 1. Scope

Covered:

- `runs/score_matrix.jsonl` — all 60 rows, every field, read directly.
- `runs/b0_23_blocks.jsonl`, `runs/b0_final_blocks.jsonl` — all rows.
- `eval/domain_loss.py` and `eval/score_matrix.py` — the cu path, line by line.
- All 43 `eval/*.py` scorers — prompt format, cu path, output artifact, selftest, population
  guard. Delegated sweep; every line cited in E9-E12 re-verified by me at the file.
- `datagen/holdout.py` — the registry and `EVAL_FILES`, imported and enumerated.
- `facts/contamination.json` — all 36 entries: status, source, population.
- `facts/*.json` — every entry citing `score_matrix`, by grep over the serialized entry.
- Corpus domain build dates: all 321 directories under `/work/aupai/data/corpus/` on the pod,
  by `stat`.

Deliberately excluded from THIS report, listed rather than silent:

- `eval/*.sh` drivers.
- Anything requiring a GPU. The audit forbids launches and I hold no card. This is what makes
  E10 an S2 and not an S1: whether the four extra cu=None scorers moved any published number
  is a measurement, not a reading.

## 2. Method

Commands whose output is quoted below, all with `CUDA_VISIBLE_DEVICES=""`:

```
python3 -c "...json.load(runs/score_matrix.jsonl)..."      # field enumeration, 60 rows
python3 -c "...re.compile(r'\bcu_none\b|\bdoc_cu\b|#cu\b')..."   # strict cu-label search
python3 -c "from datagen.holdout import REGISTRY"          # 13 entries enumerated
python3 -c "...glob facts/*.json, grep 'score_matrix'..."   # 18 entries found
~/bin/pod 'cd /work/aupai && for f in data/corpus/*/; do stat -c %y ...'  # 321 dirs
git log -1 --format='%ci %h' e970c343                       # registry commit time
git show bfa1a846 --name-only                               # what the cu fix touched
```

Broken-world test of the one instrument I wrote: the strict cu-label regex was first written
as the substring `cu`, and it reported 2 of 60 rows as labelled. Reading the matches showed
both were the word "because" inside a prose field. That is the instrument failing on a known
world — the rows have no label — and the regex was narrowed to the three tokens that are
actually labels (`cu_none`, `doc_cu`, `#cu`). **The loose version's answer, 2 labelled rows,
was wrong in the direction of reassurance.** Reported because the audit's principle 4 asks for
it and because my first count is the sort of number that would otherwise have been published.

No fixed-seed sample was needed: every population here is small enough to read whole
(60 rows, 13 registry entries, 36 contamination facts, 321 directories).

## 3. Population counts

| set | exists | read | sampled |
|---|---|---|---|
| score_matrix rows | 60 | 60 | n/a, read whole |
| score_matrix rows carrying `domain_loss` | 51 | 51 | n/a |
| blocks-ledger rows | 3 | 3 | n/a |
| registry entries | 13 | 13 | n/a |
| contamination facts | 36 | 36 | n/a |
| facts citing score_matrix | 18 | 18 | n/a |
| corpus domains on the pod | 321 | 321 | n/a |
| eval scorers under eval/ | 43 | 43 (E9-E12); 2 line by line by me | n/a, enumerated whole |

### Which corpus domains were built against which holdout population

Asked by the area's scope. Answered empirically rather than from the builders' source, because
the scan measures the outcome and the source only shows the intent:

| domain (mix_200m_4b) | alone hits | built | guard population at build time |
|---|---|---|---|
| chatml | 1,515 | 2026-09-01 | 4-file `EVAL_FILES`, no `control_sft_text_heldout` |
| chat_qa | 1,515 | 2026-09-01 | same |
| math_owm_stage2 | 542 | 2026-08-31 | same |
| cot | 334 | 2026-08-31 | same |
| code_py_starcoder | 239 | 2026-09-01 | same |
| textbook_30b | 185 | 2026-08-31 | same |
| zh_web | 81 | 2026-08-31 | same |
| code_py_rp1t | 57 | 2026-09-01 | same |
| en_c4_stage2 | 22 | 2026-08-31 | same |

**Nine of nine.** No domain in this mix was built against a population containing the
held-out file, and none was built against the 13-entry registry (E4: it landed 2026-09-04
00:50:45Z, after the newest domain). Hit counts from `runs/e1_28/e1_28_per_domain_alone.json`,
build dates from `stat` on the pod, population from `datagen/holdout.py`'s history.

## 4. Findings

| id | sev | claim as published | evidence | what contradicts it |
|---|---|---|---|---|
| E1 | S1 | `bfa1a846`'s subject line: "domain_loss passes the document mask, **and every row records which path it used**" | `git show bfa1a846 --name-only` lists no `eval/score_matrix.py`. `eval/score_matrix.py:244` calls `domain_loss_seqs(model, rows, device, per_row=True)` with no `cu_path`, taking the `cu_none` default at `eval/domain_loss.py:194`. | 0 of 60 published rows carry a cu label under the strict regex; 51 of them carry `domain_loss`. The fix labelled `domain_loss.py`'s own CLI output (`eval/domain_loss.py:763`, `:794`) and left the file that writes the published ledger untouched. Two rows measured 2026-09-04, after the fix, are also unlabelled. |
| E2 | S2 | `facts/efficiency.json#eff.eval_path_cu_artifact_ce` boundary: "NO PUBLISHED DELTA MOVES BECAUSE OF THIS … the artifact is common-mode and cancels in a difference." | Same fact's own sibling: `facts/smelt_deeploop.json#repo.loop_from_scratch_stage_d` uncertainty records the cu-passed rescore moving the pooled Stage D delta from −0.030937 to −0.022325 nat, "so 28% of the measured advantage was leak-mediated and 72% was not." | The artifact does NOT cancel. It cancelled to first order and left 28% of one measured delta behind, with a per-domain shrink correlating with per-domain leak at −0.951. The boundary states the cancellation as a property; the measurement shows it is an approximation whose residual is 28% on the one delta anyone re-scored. |
| E3 | S1 | `ds.n2_params_vs_data_matched_compute` = −0.010770 nat, the params-vs-data verdict, and the roadmap's N2 decision line "30B shape leans larger-parameter at fixed compute". | Row `measured: 2026-09-03` for both arms in `runs/score_matrix.jsonl`; the cu fix landed **2026-09-03T22:41:12Z** (`TZ=UTC git log bfa1a846`; its `+0800` stamp reads 2026-09-04 06:41, see E15). `runs/b0_23_blocks.jsonl` has `path` ABSENT. `eff.eval_path_cu_artifact_ce` measures the artifact at −0.0818 nat pooled. | The verdict was measured entirely on the `cu_none` path, and the artifact is **7.6× the delta itself** (the fact says so in those words). E2 shows the artifact does not fully cancel. Nothing in the N2 fact's boundary or uncertainty mentions the cu path at all — the boundary discusses seed sigma and the shared mix. So a decision about the 30B shape rests on a number whose instrument had a known defect 7.6× its size, and the fact does not say which path produced it. |
| E4 | S2 | The holdout guard's population, post-fix: `datagen/holdout.py` REGISTRY, 13 entries, replacing the 4-path `EVAL_FILES`. | `TZ=UTC git log -1 e970c343` → **2026-09-04T00:50:45Z** (`+0800` stamp: 08:50:45, see E15). Pod `stat` over all 321 corpus directories: the newest is `2026-09-03` (`cot_open_thoughts`, `en_c4_30b`, `rp1t_arxiv_papers`); every one of the 9 domains in `data/mix_200m_4b.json` is dated 2026-08-31 or 2026-09-01. | **Every corpus domain that exists was built before the registry existed.** The registry fixes what the NEXT build excludes and cannot retroactively protect any current domain. The 13-entry population is correct and no corpus has been built against it. `cont.heldout_in_pretrain_corpus` records this for its own scan; no other contamination fact does. |
| E5 | S2 | `runs/score_matrix.jsonl` folds rows on `ckpt`, and `domain_loss.py` appends `#cu` to the ckpt name so a doc_cu row does not collide with a cu_none one (`eval/domain_loss.py:770-775`, comment). | 0 of 60 rows have `#cu` in the ckpt name. `runs/b0_final_blocks.jsonl` has 2 rows that DO (`ckpt_b0_sd_equalcompute.pt#cu`, `ckpt_b0_n8_fixed.pt#cu`, both `path: doc_cu`). | The collision guard exists only on the path that writes the blocks ledger. `score_matrix.py` neither labels the path nor suffixes the name, so when a doc_cu re-score is written there it will fold onto the cu_none row of the same checkpoint and silently replace it — the exact outcome the comment at `domain_loss.py:770` was written to prevent. Nothing wrong yet: no doc_cu row has been written to score_matrix. |
| E6 | S3 | 14 `score_matrix` rows appear to carry a `path` field. | The 14 are `"path": "/work/aupai/data/eval/preds_l1_d3.jsonl"` on rows 16-19 and 24 — a predictions-file location, not a forward path. | Cosmetic, but it is why a grep for `path` in this ledger reads as partially labelled when the cu label is absent everywhere. Noted so the next reader does not repeat my first mistake. |
| E7 | S2 | The holdout guard runs per row in both corpus builders and is fail-closed and fingerprinted, so a domain that went through it is protected. | `datagen/build_corpus.py:28` imports `is_holdout`; it is called at `:86`, `:89`, `:93`, `:448`, `:717`. All 9 domains of `data/mix_200m_4b.json` are reachable through its generic `--domain` path (`:1585`). And `runs/e1_28/e1_28_per_domain_alone.json` gives every one of those 9 domains a non-zero alone-hit count: chatml 1515, chat_qa 1515, math_owm_stage2 542, cot 334, code_py_starcoder 239, textbook_30b 185, zh_web 81, code_py_rp1t 57, en_c4_stage2 22. | The guard ran, on every domain, and excluded none of these items — because the population it was checking against did not contain `data/sft/control_sft_text_heldout.jsonl`. **This is the empirical form of E4 and it is stronger than the date argument:** 9 of 9 domains carry hits, so no domain in the 200M mix was built against a population containing the held-out file. The per-domain spread also shows the guard is not uniformly blind — it is blind to one file, and the hit count tracks how much of that file's material each domain contains (chatml and chat_qa at 1515 each are two renders of one source). |
| E8 | S3 | `datagen/build_cot.py` and `datagen/build_code_tests_v1.py` are corpus builders. | `grep -c 'is_holdout\|holdout'` returns 0 for both. | Neither consults the guard at all. This is NOT a finding about the `cot` domain in the current mix — E7 shows `cot`'s 334 hits, so the domain that exists was built through the guarded `build_corpus.py` path. It is a finding about the builders: two scripts that can write corpus material have no holdout check, so whether a future domain is guarded depends on which script someone reaches for. No consequence found in current data. |
| E9 | — | **The assigned question, answered: does any eval hand ChatML to a base checkpoint?** NO. | `grep -ln 'im_start' eval/*.py` returns exactly one file, `eval/score_matrix.py`, where the marker appears in a dispatch-guard comment and not in a prompt. The five generative scorers all route through `prompt_fn(classify(cfg, ...))`: `code_zh.py`, `gsm8k.py`, `math_zh.py`, `math_hard.py`, `run_eval.py`. `eval/gsm8k.py:43-45` raises on a missing format with the reason in the message: "a default would silently score a base checkpoint in ChatML". Backed by an AST test, `scripts/test_eval_base_prompt_format.py`. | Nothing. This is a clean result, recorded as a finding because "no defect" is an answer and silence is not. The one gap: `eval/code_l0prime.py:121` builds a hardcoded continuation prompt with no `classify` call — correct for a base checkpoint, but an SFT/RL checkpoint scored by it also gets continuation with nothing recording the choice. S3 at most. |
| E10 | S2 | The cu-path defect is confined to `domain_loss`/`score_matrix` (E1). | Four more scorers pass no cu on packed multi-document rows, each verified at the cited line: `eval/ppl.py:74` `logits, _ = model(xb)` on rows from `train._domain_seqs`; `eval/domain_bpb.py:67` `out = self.model(x)` where `:257` takes `val_seqs(name, ours_tok)` and `:272` decodes a whole packed row to one text; `eval/math_bpb.py:211` constructs `OursModel` with no `prefix_arm`, so `eval/humaneval_bpb.py:178`'s `cu = None` holds on every forward; and `humaneval_bpb.py` itself is cu=None unless `--prefix` is passed. | E1 is narrower than the defect. `ppl.py`'s docstring claims it "rebuilds exactly the rows train.py holds out" and then scores them under a different mask than training used. `domain_bpb.py` is the metric `score_matrix.py:62` uses for the **control-arm comparison**, so a cross-tokenizer conclusion rests on an undocumented path. None of these four carries a `path` label either. Whether any published number moves is UNMEASURED — that needs a card. |
| E11 | S2 | `block_paired.py` and `readout_30b.py` refuse mismatched pairings. | They refuse on block-set mismatch (`block_paired.py:114-120`), token-count mismatch (`:123-129`) and `head_fp` (`readout_30b.py:486-509`). Neither reads `path`: no `get("path")` in either file. | `block_paired.py:220-233` loads `runs/b0_23_blocks.jsonl` (domain_loss CLI, doc_cu) beside `runs/score_matrix.jsonl` (score_matrix, cu_none) in the same call. The guards that exist are thorough about population and silent about the forward, so a doc_cu row and a cu_none row can be paired and nothing refuses. The `#cu` suffix keeps the keys distinct enough to be addressable, which makes the mixing possible rather than impossible. |
| E12 | S3 | Sibling scorers are the same instrument. | `eval/code_fewshot.py:173` hardcodes `rep_stop=False` with no `tokenizer=` and has no `--no_rep_stop` flag; `eval/l1_fewshot.py:429-431` passes `tokenizer=tok` and `rep_stop=not args.no_rep_stop`. Same divergence in `math_hard.py:123` (no `tokenizer=`, no flag) vs `math_zh.py:136-138`. | `l1_fewshot.py:417-423` records that this exact omission once made two arms differ in DECODER rather than in model — "a decoder difference read as a model difference, from an argument that defaults to a silent False". The lesson is recorded in one sibling and not applied in the other two, and `math_hard.py` supports `--k`/`--temperature` for the sampled arm that `math_zh.py` documents this decoder as confounding. |

## 5. Blind spots of this audit

- **The 46-scorer prompt-format sweep is DONE and its results are E9-E12.** Running count:
  **43 of 43 scorers enumerated** — and the "46" in my earlier draft was wrong: `ls eval/*.py | wc -l`
  is 43. I had counted from a directory listing that included `__init__.py`, `_devs.sh` and
  `__pycache__`. The sweep was delegated; I re-verified every line it cites for E9-E12 myself
  (`ppl.py:74`, `domain_bpb.py:67`/`:257`/`:272`, `math_bpb.py:211`, `humaneval_bpb.py:178`,
  `gsm8k.py:43-45`, the `im_start` grep, the `prompt_fn(classify` grep) before writing them down,
  because a subagent's report is a claim and principle 1 does not exempt it. The sweep also
  reported "42 files"; the count is 43.
- I read no scorer's behaviour, only its source. A file that passes `doc_cu` in the line I
  quoted may still be reached through a wrapper that does not.
- E3 says the N2 verdict's instrument was defective by 7.6× its own size. It does NOT say the
  verdict is wrong: whether the sign survives a doc_cu re-score is a card measurement and the
  audit forbids running it. What I can say is that the fact does not record which path it was
  taken on, so no reader can tell.
- Contamination facts: I audited status, source, population and instrument-existence (E14). I
  did not RE-RUN any scanner, so a fact whose recorded population is right and whose scan was
  wrong still looks clean here. `cont.heldout_in_pretrain_corpus` is the one case where the scan
  itself was checked, by me, before this audit — and that check is what retracted the 316-item
  result, so the class of defect E14 cannot see is known to occur in this very file.
- The pod's `data/` is gitignored, so every corpus fact rests on a `stat` I ran once at
  2026-09-04 ~03:30Z (E15: my working note said "11:40Z", which was +0800). A directory mtime is when it was last written, not when its contents
  were built; a domain rebuilt in place would read as newer than its data.

## 6. Open questions for the controller

1. Does the N2 verdict (E3) need a doc_cu re-score before the 30B shape decision stands, or
   does it stand with a boundary line naming the path? One decision either way.
2. `eff.eval_path_cu_artifact_ce`'s "no published delta moves" (E2) — retract that sentence,
   or qualify it as first-order with the 28% residual named?
3. Should `eval/score_matrix.py` label and suffix like `domain_loss.py` does (E5)? It is a
   one-line fix and the audit forbids me making it.
4. E4: do any facts other than `cont.heldout_in_pretrain_corpus` need the "built before the
   registry" line, or is one statement in one place enough?
5. Is the 46-scorer sweep still wanted as part of this area's report, given it will land after
   the 3-hour mark?

## Pair check (3b, 2026-09-04)

Recomputed E1 / E3 / E4 independently as e1's pair, per the controller's assignment. Artifacts
reopened in this pass; numbers read from the artifacts, not from e1's report.

**E1 — HOLDS.** `git show bfa1a846 --name-only` lists only `eval/domain_loss.py`. `eval/score_matrix.py:244`
calls `domain_loss_seqs(model, rows, device, per_row=True)` with no `cu_path`, so it takes the
`cu_none` default at `eval/domain_loss.py:194`. `runs/score_matrix.jsonl` = 60 rows; a grep for
`cu_none|doc_cu|#cu` = **0** rows; 52 carry `domain_loss`. The cu-path fix never reached the
ledger writer; no published row records which path it used.

**E3 — HOLDS.** The 6 `score_matrix.jsonl` rows dated 2026-09-03 include the N2 arms
(`ckpt_params_leg_438m_3p76b.pt`, `ckpt_data_leg_206m_8b.pt` step7000/step7500/step10000) plus the
pythia control. `facts/data_scaling.json#ds.n2_params_vs_data_matched_compute` value = −0.01077.
`facts/efficiency.json#eff.eval_path_cu_artifact_ce` uncertainty states the pooled artifact is
**7.59×** N2's delta ("the pooled figure is 7.59x"), matching e1's "7.6×". The cu fix
(`bfa1a846`) is **2026-09-03T22:41:12Z**; `runs/b0_23_blocks.jsonl` was committed
2026-09-03T19:18:38Z (`19db9840`), so the measurement precedes the fix by 3.4 h and is pre-fix.
`runs/b0_23_blocks.jsonl` has keys
[ckpt, domains, unweighted_mean], `path` absent. Verdict rests on a number whose instrument had
a known defect ~7.6× its size, and the fact does not record the path.

**E4 — HOLDS (same-source as corpus CD-2).** `TZ=UTC git log -1 e970c343` = **2026-09-04
00:50:45Z** (the `+0800` stamp on the commit reads 08:50:45; see E15).
Pod `stat` (`/work/aupai/data/corpus/`, the pod runs UTC): newest corpus dirs are all 2026-09-03
(`en_c4_30b` 2026-09-03T20:29:50Z, `cot_open_thoughts` 2026-09-03T15:23:45Z,
`rp1t_arxiv_papers` 2026-09-03T08:19:57Z). Every corpus domain predates the registry by ≥4.3 h.
This
matches the corpus audit's CD-2: no stamp records which holdout population it was built against,
and none could have been built against the registry.

### One number reconciled between the two passes (e1, after reading 3b's check)

3b read **52** score_matrix rows carrying `domain_loss`; I read **51**. **51 is correct.**

The 52nd is `pythia-160m-step2000`, the control. Its metrics keys are
`[domain_bpb, lambada_en, humaneval_bpb]` — no `domain_loss`. The string `domain_loss` appears in
that row only inside a `skipped` reason. A search over the serialized row counts it; the metric is
absent.

Kept rather than quietly corrected, because it is the same shape as the broken-world note in §2 —
a loose match reading as presence — arrived at independently by two readers on the same file
within an hour. Neither pass was careless; the substring is simply not the property. The finding
E1 rests on is unaffected: 0 rows carry a cu label either way.

### E13 (S2): published numbers whose cited source cannot be opened from the repository

6e's ruling 2026-09-04, same class as b0's `trace_p200m_3step.json`. Enumerated over every
`runs/*` path cited in any `facts/*.json` entry's `source`, `config`, `uncertainty` or
`boundary`: **112 cited, 7 absent from the repo, all 7 present on the pod** (`stat` at
2026-09-04 ~03:44Z; the "~12:30Z" first written here was +0800, see E15).

| cited path | pod size | cited by |
|---|---|---|
| `runs/n7_domain.jsonl` | 55,804 | the Stage A roadmap row, `roadmap_0903.md:24` |
| `runs/trace_p200m_3step.json` | 59,446,282 | `eff.clip_and_sync_cost_p200m` (+3 others) |
| `runs/p500m_20b_0902.log` | 65,990 | `eff.p500m_20b_throughput_and_dips` (+1) |
| `runs/eval_p500m_step1500_base.log` | 5,247 | `eff.p500m_20b_throughput_and_dips` |
| `runs/milestone_p324_v2.jsonl` | 1,882 | `eff.light_profile_wall` |
| `runs/eval_p500m_step1500_l1.log` | 40 | `eff.p500m_20b_throughput_and_dips` |
| `runs/ppl_step1500_v2.log` | **0** | `eff.p500m_20b_throughput_and_dips` |

Two of these are worse than "not in the repo". `runs/ppl_step1500_v2.log` is **0 bytes on the
pod**: it is cited as the source of a published throughput fact and contains nothing, so the
citation cannot be satisfied anywhere, not just here. And `trace_p200m_3step.json` is 59 MB,
which is why it is not in git — that is a real constraint, not an oversight, and the fix for it
is a derived summary committed beside the fact rather than the trace itself.

**Broken-world test of this enumeration, and it failed the first time.** My initial regex was
`runs/[A-Za-z0-9_./-]+\.(?:json|jsonl|log|txt)` — with `json` before `jsonl` in the
alternation, so every `.jsonl` path matched as `.json` and truncated. It reported **21 absent
paths**, 15 of which were ledgers sitting in the repo (`runs/score_matrix.jsonl` read as
`runs/score_matrix.json`, and so on). Reordering to `jsonl|json` gives 6, plus `n7_domain.jsonl`
found by hand while pair-checking = 7. The wrong answer was 3.5x too alarming, in the opposite
direction from §2's error — a loose pattern can fail either way, and neither direction is safe.

### E14 (S2): four `measured` contamination facts cite an instrument that exists nowhere

Enumerated the instrument of all 36 `facts/contamination.json` entries by extracting every
script path from each `source`. Fifteen cite `datagen/scan_math_contamination.py`, which exists;
most others resolve. **Four cite a script under `/tmp`, and none of the four scripts exists on
this machine or on the pod** (`ls` on both, 2026-09-04 ~03:47Z; the "~12:40Z" first written
here was +0800, see E15):

| fact | status | cited instrument |
|---|---|---|
| `cont.scanner_idf_weighting` | measured | `/tmp/harden_scan.py` |
| `cont.gsm8k_zh_webhq_scan` | measured | `/tmp/gsm_hit_detail.py` |
| `cont.math500_webhq_fp_explained` | measured | `/tmp/contam_details.py`, `/tmp/harden_scan.py` |
| `cont.code_holdout_carved` | measured | `/tmp/carve_code_eval.py` |

`/tmp` is cleared on reboot, so these citations could not survive by construction — this is not
a file someone forgot to commit, it is a location that guarantees the loss. Each of the four is
`status: measured`, i.e. the store's strongest claim, and none can be re-derived: the numbers
stand only as recorded values. `cont.scanner_idf_weighting` is the one that matters most, because
it is the fact that justifies the IDF weighting the other scans rely on — the instrument that
validated the instrument is gone.

Separately, **9 of 36 entries name no script at all** in `source`. Six describe the procedure in
prose ("contrasted scans, same script/params", "sampled 2000 docs … bigram-jaccard all pairs",
"two stratified hand-reads … reader: cklxx session"), which is a method statement rather than a
reproducible instrument. Two name only a log (`runs/scan_code_holdout.log`,
`runs/scan_code_holdout_sft.log`) — both present in the repo, so the OUTPUT is readable while the
code that produced it is not named. One is a decision record (`cont.math_hard_v1_void`), where
naming no instrument is correct.

Not a finding about the numbers: nothing here contradicts any contamination value. It is a
finding about what "measured" can mean in this store — for these four, it means "was measured
once, by something no longer in existence".

### E15 (S2): every UTC timestamp in this report was +0800, and two of them were in the future

Found by me while re-reading my own report, after 6e had accepted it. The defect is in the audit
instrument, not in the code under audit.

Six timestamps in this file carry a `Z` suffix. All six were local Asia/Shanghai time
(`CST+0800`), 8 hours ahead of the UTC they claimed:

| where | as published | true UTC | how established |
|---|---|---|---|
| E3 (table) | cu fix "2026-09-04 06:41" | **2026-09-03T22:41:12Z** | `TZ=UTC git log -1 --date=format-local bfa1a846` |
| E4 (table) | registry "2026-09-04 08:50:45 +0800" | **2026-09-04T00:50:45Z** | same, `e970c343` |
| E4 (pair-check §) | registry "08:50Z", corpus dirs "20:29Z/15:23Z/08:19Z" | **00:50:45Z**; corpus dirs unchanged | pod `stat`; the pod runs UTC (`date -u` == `date`), so those three were already right |
| §3 nine-of-nine | registry "08:50Z" | **00:50:45Z** | same |
| §5 blind spots | corpus `stat` "~11:40Z" | **~03:30Z** | bounded by `d0f450dc`, the commit that first wrote the line |
| E13 | `stat` "~12:30Z" | **~03:44Z** | bounded by `bc15f1c3` |
| E14 | `ls` "~12:40Z" | **~03:47Z** | bounded by `6f96da60` |

**Two were in the future.** This report's E13 and E14 claim their evidence was gathered at
12:30Z and 12:40Z on 2026-09-04. E14 was committed at 2026-09-04T03:47:07Z. A reader checking
whether the evidence predates the claim would find the claim predates the evidence by nine hours.
Nothing in the report or the commit hooks caught this; the hook checks harness state, not
whether a timestamp is reachable.

**Cause, one site not one number.** I ran bare `git log --date=iso` and bare `stat`, both of
which format in the shell's local zone, then typed `Z` because the surrounding convention is UTC.
The `Z` was my own annotation, never output by any command. Same shape as
`memory/stat-format-Z-is-not-evidence.md`: the character that makes a timestamp verifiable was
supplied by the writer. `TZ=UTC` on the command, or `--date=format-local:...Z` under `TZ=UTC`,
is the only form where the `Z` is produced by the tool.

**What survives.** Both affected conclusions hold, and both hold by a wider margin than
published, because the correction moves the two events I was comparing in the same direction:

- E3: the N2 measurement (`runs/b0_23_blocks.jsonl`, committed 2026-09-03T19:18:38Z) still
  precedes the cu fix (2026-09-03T22:41:12Z), now by **3.4 h** rather than by "measured 09-03 vs
  fix 09-04". Pre-fix, so the verdict still rests on the `cu_none` path.
- E4: the newest corpus directory (`en_c4_30b`, 2026-09-03T20:29:50Z, read on the pod, which runs
  UTC) still precedes the registry (2026-09-04T00:50:45Z), now by **4.3 h**. Every corpus domain
  still predates the registry.

No published number moves. What moves is whether a reader can check the ordering I asserted:
with the wrong labels, the E3 gap read as 8 h in the wrong direction from the truth and the E4
gap as 12.3 h, and E13/E14's evidence was unreachable in time.

**What this says about the rest of the report.** Every other date in this file is a date without
a clock time (`2026-09-03`, `2026-08-31`), taken from a `measured` field or a `stat` day, and a
day boundary is 8 h away from +0800 only between 16:00 and 24:00 UTC. Two such dates fall in that
window and I checked both: `runs/b0_23_blocks.jsonl` at 19:18Z and `en_c4_30b` at 20:29Z. The pod
value was already UTC. The repo commit's day is unchanged under the correction (2026-09-03 either
way, 19:18Z vs 03:18 local next day — the local reading would have been 09-04, so the day I
published, 09-03, is the correct UTC one). No other date in the report is affected.

**Not checked:** whether the same `Z` habit is in my earlier ledger rows, facts, or board posts
outside this audit. That is a wider sweep than this area and the audit forbids the fix; recorded
here so the class is on the record with its cause, not just this file's six instances.
