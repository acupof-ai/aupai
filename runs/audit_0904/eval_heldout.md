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
E9-E12 and section 5 names what remains unseen. Twelve entries: two S1, six S2, three S3, and
E9, which carries no severity because it is a clean result -- recorded as an entry anyway,
since "no defect" is an answer to an assigned question and silence is not.

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
08:50Z, after the newest domain). Hit counts from `runs/e1_28/e1_28_per_domain_alone.json`,
build dates from `stat` on the pod, population from `datagen/holdout.py`'s history.

## 4. Findings

| id | sev | claim as published | evidence | what contradicts it |
|---|---|---|---|---|
| E1 | S1 | `bfa1a846`'s subject line: "domain_loss passes the document mask, **and every row records which path it used**" | `git show bfa1a846 --name-only` lists no `eval/score_matrix.py`. `eval/score_matrix.py:244` calls `domain_loss_seqs(model, rows, device, per_row=True)` with no `cu_path`, taking the `cu_none` default at `eval/domain_loss.py:194`. | 0 of 60 published rows carry a cu label under the strict regex; 51 of them carry `domain_loss`. The fix labelled `domain_loss.py`'s own CLI output (`eval/domain_loss.py:763`, `:794`) and left the file that writes the published ledger untouched. Two rows measured 2026-09-04, after the fix, are also unlabelled. |
| E2 | S2 | `facts/efficiency.json#eff.eval_path_cu_artifact_ce` boundary: "NO PUBLISHED DELTA MOVES BECAUSE OF THIS … the artifact is common-mode and cancels in a difference." | Same fact's own sibling: `facts/smelt_deeploop.json#repo.loop_from_scratch_stage_d` uncertainty records the cu-passed rescore moving the pooled Stage D delta from −0.030937 to −0.022325 nat, "so 28% of the measured advantage was leak-mediated and 72% was not." | The artifact does NOT cancel. It cancelled to first order and left 28% of one measured delta behind, with a per-domain shrink correlating with per-domain leak at −0.951. The boundary states the cancellation as a property; the measurement shows it is an approximation whose residual is 28% on the one delta anyone re-scored. |
| E3 | S1 | `ds.n2_params_vs_data_matched_compute` = −0.010770 nat, the params-vs-data verdict, and the roadmap's N2 decision line "30B shape leans larger-parameter at fixed compute". | Row `measured: 2026-09-03` for both arms in `runs/score_matrix.jsonl`; the cu fix landed `2026-09-04 06:41` (`git log bfa1a846`). `runs/b0_23_blocks.jsonl` has `path` ABSENT. `eff.eval_path_cu_artifact_ce` measures the artifact at −0.0818 nat pooled. | The verdict was measured entirely on the `cu_none` path, and the artifact is **7.6× the delta itself** (the fact says so in those words). E2 shows the artifact does not fully cancel. Nothing in the N2 fact's boundary or uncertainty mentions the cu path at all — the boundary discusses seed sigma and the shared mix. So a decision about the 30B shape rests on a number whose instrument had a known defect 7.6× its size, and the fact does not say which path produced it. |
| E4 | S2 | The holdout guard's population, post-fix: `datagen/holdout.py` REGISTRY, 13 entries, replacing the 4-path `EVAL_FILES`. | `git log -1 e970c343` → `2026-09-04 08:50:45 +0800`. Pod `stat` over all 321 corpus directories: the newest is `2026-09-03` (`cot_open_thoughts`, `en_c4_30b`, `rp1t_arxiv_papers`); every one of the 9 domains in `data/mix_200m_4b.json` is dated 2026-08-31 or 2026-09-01. | **Every corpus domain that exists was built before the registry existed.** The registry fixes what the NEXT build excludes and cannot retroactively protect any current domain. The 13-entry population is correct and no corpus has been built against it. `cont.heldout_in_pretrain_corpus` records this for its own scan; no other contamination fact does. |
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
- Contamination facts: I audited status/source/population as recorded. I did not re-run any
  scanner, so a fact whose recorded population is right and whose scan was wrong looks clean
  here. `cont.heldout_in_pretrain_corpus` is the one case where that was checked, by me, before
  this audit.
- The pod's `data/` is gitignored, so every corpus fact rests on a `stat` I ran once at
  2026-09-04 ~11:40Z. A directory mtime is when it was last written, not when its contents
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
