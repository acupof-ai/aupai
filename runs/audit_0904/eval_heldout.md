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
E9-E12 and section 5 names what remains unseen. Twenty-two entries: four S1, fourteen S2, three S3, and
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
| E13 | S2 | 112 `runs/*` paths cited by `facts/*.json` are readable sources. | Enumerated every path in every entry's `source`/`config`/`uncertainty`/`boundary`; `stat` locally and on the pod, ~03:44Z. | 7 of 112 are absent from the repo and present only on the pod, so a reader with the repository alone cannot open the source of a published number. Same class as b0's `trace_p200m_3step.json`. |
| E14 | S2 | Four `facts/contamination.json` entries carry status `measured`, the store's strongest claim. | `cont.scanner_idf_weighting`, `cont.gsm8k_zh_webhq_scan`, `cont.math500_webhq_fp_explained`, `cont.code_holdout_carved` cite scripts under `/tmp`; `ls` on this machine and the pod, ~03:47Z. | None of the four scripts exists in either place, and `/tmp` is cleared on reboot, so the loss was guaranteed by construction. No value is contradicted; `measured` here means measured once by something no longer in existence. `cont.scanner_idf_weighting` is the fact that justifies the IDF weighting the other scans rely on. Also: 9 of 36 entries name no script at all. |
| E15 | S2 | Six timestamps in THIS report carried a `Z` suffix asserting UTC. | `TZ=UTC git log` on `bfa1a846`/`e970c343`; the commits that first wrote each `~HH:MMZ` note; `date -u` == `date` on the pod. | All six were local +0800. E13's and E14's stated collection times (12:30Z, 12:40Z) fall NINE HOURS AFTER the commit that published them, so the claim preceded its own evidence. The `Z` was typed by me; no command emitted it. E3 and E4 survive by a wider margin after correction (3.4 h and 4.3 h gaps). |
| E16 | S2 | 59 of 62 credential-bearing episodes in the v14 build were invisible to the old detector, so they were in v13's admission set and every pack before it — which checkpoints trained on them? | 297 ledger lines folded by `name` to 152 distinct, every `--sft_path` in every `cmd` (11 SFT runs, 4 packs); all 60 `score_matrix` rows (55 distinct `ckpt`); `ls` on the pod's `data/sft/` (41 entries, 21 `.pt`); `torch.load` on all 21 reading `sources`/`source`/`meta.sources`. | NONE. Zero agentic packs in all four populations, and the training path cannot reach one: the builder writes JSONL and nothing converts a JSONL to the `.pt` that `sft_math.py --sft_path` consumes. Held at S2 because 20 of 21 `.pt` packs carry NO `sources` field, so provenance came from the producers' code, not the packs' own metadata. |
| E17 | S2 | The v14 build's log certifies that the secret scan gated the rename. | `runs/e1_v14_agentic_build_2026-09-04.log` lines 95-102 and 122; the three background-task output files. | The log holds a `FileNotFoundError` on `os.replace` AND `build exit=0`, and LACKS the `wrote 4823 rows` line both Monitors reported: two processes wrote it concurrently, one won the rename, the other died on it, and the loser's traceback interleaves mid-report. The gate verdict is sound but the log cannot certify it — the pack was verified independently (4,823 rows, 0 unparseable, 0 wrong-shape, 0 byte-identical duplicates). The two-writer origin is UNRESOLVED, not guessed. |
| E18 | S2 | `eval/score_matrix.py`'s failure records name the cause of a scorer's refusal. | Three runners read: `_run_eval_json:304` and `metric_minimal_pairs:281` use `(r.stderr or r.stdout)`; `_run:387` uses `(r.stdout + r.stderr)`. Refusal stream read per script for all 7. Mechanism reproduced with a 4-line program through line 304's expression verbatim. | Two of three runners discard stdout entirely whenever stderr is non-empty, and `domain_bpb.py` is the one script whose every refusal is on stdout, so a single `vocab_id` UserWarning replaces the cause. **6 of 10** `domain_bpb` rows are blinded this way (not 10 as MT-12 reports: 3 carry a real stderr cause, 1 carries a bare source line from the truncation shape the code comment calls fixed). `l1_fewshot` adds 8 rows of a second shape: exit -15 SIGTERM, cause recorded as a progress line. `_run` already holds the correct form eleven lines below the second defective site. |
| E19 | S1 | `domain_bpb` is a published metric of this project: it is the control arm's cross-tokenizer reading, `runs/score_matrix.jsonl` carries it as a field on 60 rows, and `facts/*.json` cite it. | Ran domain_bpb's whole pre-forward half on the pod with `CUDA_VISIBLE_DEVICES=""` (no model, no card): `val_seqs` + `roundtrip_fraction` over all 9 domains of `data/mix_200m_4b.json`. Round-trip fractions math_owm_stage2 0.1094, en_c4_stage2 0.0156, cot 0.0000, textbook_30b 0.0156, chatml 0.0000, chat_qa 0.0000, zh_web 0.1094, code_py_starcoder 0.2188, code_py_rp1t 0.3594 — every one below `MIN_ROUNDTRIP = 0.98`. | **All 9 domains are skipped, `out` is empty, and `domain_bpb.py:317` prints `REFUSING: no domain produced a number` and returns 1. The metric has NEVER produced a number for any checkpoint and cannot on the current data path** — the 10 error rows in E18 are this one cause, not a per-checkpoint problem. Cause is one line: `tok.decode([EOS_ID])` returns `''`, and every val row from `train._domain_seqs` is packed and EOS-delimited (cot row 0 holds 8), so decode drops the delimiters and re-encode cannot reproduce the ids. The gate measures the tokenizer's special-token handling, not whether the two arms score the same bytes. |
| E20 | S1 | `ds.n2_params_vs_data_matched_compute` = −0.010770 nat, block-paired over 576 blocks, t −6.51, sign test 227 up / 349 down p 2.10e-07 — the params-vs-data verdict behind the 30B shape decision. | `eval/block_paired.py --from runs/score_matrix.jsonl --arms ckpt_data_leg_206m_8b.pt#cu ckpt_params_leg_438m_3p76b.pt#cu`, the same instrument the fact's `source` names, same 576 blocks / 2,359,296 tokens, same orientation. | On doc_cu the mean is **−0.000920** (t **−0.55**, 1/43 of its own SE) and **the sign test REVERSES: 329 up / 247 down, p 3.62e-04** — a significant majority of blocks where params is WORSE, opposite to the mean, with median **+0.003204** also opposite. On cu_none both statistics agreed. The surviving negative mean rides on chatml/chat_qa/textbook_30b, and chatml+chat_qa are the two domains the mask moved most (−0.2364, −0.1695), so it is E2's non-uniformity inside the pair rather than an advantage. Domain count is 9 in all four rows, unchanged between the published number and the re-score. |
| E21 | S2 | E18's own numbers: `l1_fewshot` "has 8 error rows, all of the form `l1_fewshot.py exited -15`", over "the 18 score_matrix error entries". | All 8 `l1_fewshot` error values read verbatim from `runs/score_matrix.jsonl`; the population recounted at E18's own commit `2c87a493` under both counting rules; `TZ=UTC git log` against bare `--date=format-local:...Z`. | **1 of 8 is the SIGTERM shape, not 8** — the other 7 name `eval_artifacts.ArtifactExists` and are readable as published. I read one row and wrote "all of the form". The population is 19 by E18's own rule or 25 by the plainer one, never 18: "18" was 10 `domain_bpb` + 8 `l1_fewshot`, the two metrics I had looked at, stated as the population. And dating the E18 commit I printed `12:38:11Z` from `--date=format-local:'...Z'` — +0800 with a `Z` I typed into the format string, the exact defect E15 records, reproduced while checking E15's neighbour; true UTC is `04:38:11Z`. E18's MECHANISM survives intact, including the 6-of-10 count. |
| E22 | S2 | E18 called the `l1_fewshot` error rows unreadable failures; E21 corrected the count and called the seven `ArtifactExists` rows "readable as published". Both stopped at the exception NAME. | All 7 error values read in full; `eval/l1_fewshot.py:371` and its fix commit `5a989647` (2026-09-02T07:00:52Z, `TZ=UTC`); the artifact read on the pod. | These are not failures — `open_artifact` REFUSED TO OVERWRITE an existing result, which is what it exists for (`be.l1_3shot_retracted` records the 477-row file that was overwritten before the guard). Six are the pre-fix bare `preds_l1_d3.jsonl` colliding across checkpoints. **The seventh, `ckpt_b0_sd_equalcompute.pt` measured 2026-09-04, names the FIXED path and collides with ITSELF: the artifact holds a complete 497-row result at accuracy 0.0181** (9 ok, 423,307 B, Sep 4 00:39). A real L1 measurement of a Stage D arm exists on disk while its score_matrix row says ERROR. Cause: `metric_l1_fewshot` passes neither `--force` nor `--run`, so any re-score of an already-scored checkpoint can only refuse. |

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
supplied by the writer. **`TZ=UTC` on the command is the only form where the `Z` is produced by the
tool.**

**A SECOND FORM OF THE SAME DEFECT, found 2026-09-04 while checking E18 (E21, 6e's ruling that it
belongs here).** The first version of this paragraph offered two fixes: `TZ=UTC`, or
"`--date=format-local:...Z` under `TZ=UTC`". The second half is a trap and I then fell into it.
Dating E18's commit I ran

```
git log -1 --format='%ad' --date=format-local:'%Y-%m-%dT%H:%M:%SZ' 2c87a493   ->  2026-09-04T12:38:11Z
TZ=UTC git log -1 --format='%ad' --date=format-local:'%Y-%m-%dT%H:%M:%SZ' 2c87a493 -> 2026-09-04T04:38:11Z
```

`format-local` means "the local zone", and the `Z` inside the format string is a literal character
git copies out. So a `format-local:...Z` string WITHOUT `TZ=UTC` produces a timestamp that looks
UTC-stamped, is +0800, and carries a `Z` that no tool asserted — identical in effect to typing the
`Z` by hand, but harder to spot because the `Z` now sits in a format string and reads as machinery.
The distinction that matters is not where the `Z` appears; it is whether `TZ=UTC` is on the command.
The two forms of this defect are: typing `Z` after a bare `stat`/`git log`, and putting `Z` in a
`format-local` string. Both were mine, three hours apart, the second while auditing the first.

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

### E16 (S2): which checkpoints trained on a credential-bearing agentic pack — none, and the packs are unreachable from training

6e's question, 2026-09-04, turning the v14 build's count into the question a user asks: 59 of 62
credential-bearing episodes were caught only by the new CLI/env tier, so they were in v13's
admission set and every pack before it. Which checkpoints trained on them?

**Answer: no checkpoint in this repository trained on any agentic pack, and no published number
rests on one.** Four independent readings, each over a whole population:

| population | how enumerated | agentic packs found |
|---|---|---|
| SFT runs in `runs/experiments.jsonl` | 297 lines folded by `name` → 152 distinct; every `--sft_path` in every `cmd` | 0 of 11 |
| `runs/score_matrix.jsonl` | all 60 rows, 55 distinct `ckpt`, whole-row grep | 0 |
| SFT packs on the pod | `ls /work/aupai/data/sft/` — 41 entries, 21 `.pt` | 0 |
| `.pt` pack provenance | `torch.load` on all 21, read `sources`/`source`/`meta.sources` | 0 (no pack has an agentic source; see the caveat below) |

The 11 SFT runs name exactly four packs: `control_sft_ours.pt` (5 runs), `sft_all.pt` (4),
`proc_v1.pt` (1), `sft_all_family_clean.pt` (1). Their producers are
`datagen/pack_control_sft_ours.py` (reads `control_sft_text_train.jsonl`) and
`datagen/prepare_sft.py`/`make_mixed.py`/`fetch_sft_data.py` (a 10-entry literal `SOURCES` list of
instruction datasets). None of the three reads `~/.claude/projects/` or any `agentic_v*` path.

**The mechanism, which is stronger than the count.** `build_agentic_sft.py` writes JSONL and
nothing converts a JSONL to the `.pt` that `sft_math.py --sft_path` consumes: the only
`format_agentic` callers are the builder itself, its own selftest, `scripts/test_sft_pack.py`, and
three files that merely mention it in comments (`n7c_gates.py:452`, `eval/prefix_mask.py:177`,
`:288`). `prepare_sft.py`'s `SOURCES` is a literal list and does not include an agentic path. So
the agentic packs are not one commit away from training — the conversion step does not exist.

**S2 not S1, and the caveat is the reason.** Rated S2 because no published number is affected.
The caveat that keeps it from being S1-clean in the other direction: **20 of 21 `.pt` packs carry
no `sources` field at all** — keys are `input_ids`/`labels`/`vocab_id`, sometimes `holdout_fp` and
`sources_fp`, and `sources_fp` is a fingerprint, not a list. So pack provenance was established
from the producers' source code, not from the packs' own metadata. A pack built from an
undocumented path would look exactly like these. `control_sft_ours.pt` and
`sft_all_family_clean.pt`/`sft_all_v5.pt` are the only ones carrying `sources_fp`.

**What this does NOT clear.** The 59 episodes are still real, still in
`~/.claude/projects/*/*.jsonl`, and still in whatever v13 pack existed before 6e deleted it. E16
says the training path never reached them; it says nothing about the source sessions, which is
where a rotation has to act (list delivered separately, no values).

**A published fact this contradicts, and it is not mine.**
`facts/data_quality.json#dq.agentic_credential_split`, status `measured`, states: "of the 866, 863
are opaque-ONLY and **3** also carry a REAL_CREDENTIAL (GitHub Token 1, IBM Cloud IAM Key 1,
Private Key 1)". Those three types are exactly the three my build found via the legacy detector —
and my build found **62** REAL_CREDENTIAL episodes over a comparable population, 59 of them
invisible to the instrument that fact used. The fact's `config` names that instrument in its own
words: "find_secrets (detect_secrets, chunk=1) types intersected with REAL_CREDENTIAL's 22
provider rules". So the number 3 is what that instrument could see, not the credential count. The
fact's `boundary` names a different silence ("silent on the 9,134 non-opaque episodes'
provider-credential rate") and does not name this one. **The 3 is an undercount by a factor of
~20, and the fact does not say so.** Not corrected here (audit rule 5); the population is not
byte-identical to mine (its 10,000-episode cache vs my 9,060 admitted rows), so the ratio is
approximate while the direction is not. Owner: 44's area (facts), same class as E14.

### E17 (S2): the v14 build's own log contradicts itself, and the log is the only record of the gate

Found while reading the v14 log to report the gate verdict. Not a defect in the pack — the
artifact validates — but the log that certifies it cannot be read as a sequence.

`runs/e1_v14_agentic_build_2026-09-04.log`, 122 lines, contains BOTH of these:

```
 95 secret scan gate: 0 real-credential row(s), 2 allowed type(s) [...] over 16,288,781 chars
 96 Traceback (most recent call last):
100     os.replace(staged, a.out)
102 FileNotFoundError: ... 'data/sft/agentic_v14.jsonl.unscanned' -> 'data/sft/agentic_v14.jsonl'
...
122 build exit=0
```

A `FileNotFoundError` on the rename and `exit=0` in the same file, and the `wrote 4823 rows` line
that both Monitors reported is not in the file at all. Cause, established from the three
background-task output files rather than guessed: **two processes wrote this log concurrently.**
`tasks/blhm5f4k4.output` (the background command) ends at the traceback and reports
`[exited with code 0]` — the wrapper's own exit, not python's. `tasks/bsp31waas.output` and
`tasks/b8pod829l.output` both carry `wrote 4823 rows -> data/sft/agentic_v14.jsonl`. One process
renamed the staged file; the second reached `os.replace` after the name was already gone and died
on it. Both were appending to the same path with `>>`-style shared descriptors, so the surviving
file interleaves them and the loser's traceback lands mid-report (line 102 is followed by line 103
`251 opener the model cannot act on`, part of the winner's dropped-rows table).

**Consequence for the gate.** The gate verdict is `0 real-credential row(s), 2 allowed type(s)
['Base64 High Entropy String', 'Secret Keyword'] over 16,288,781 chars`, and it is trustworthy on
its own terms — but not because the log says so. The log is a record two writers can garble, and
the line that says the pack was written is missing from it. I verified the pack independently:
`data/sft/agentic_v14.jsonl`, 21,602,682 B, **4,823 rows, 0 unparseable, 0 wrong-shape, 0
byte-identical duplicates**, every row a dict carrying `messages`.

**Why this is E17 and not a fix.** Audit rule 5. The defect is that a build whose whole discipline
is "the scan gates the rename" writes its certificate to a file that a second concurrent process
can corrupt, and nothing in the build detects a second writer. The `.unscanned` staging survives a
kill; it does not survive a race. Two one-line changes would close it (an exclusive-create lock on
the out path, and `os.replace` guarded by the staged file still existing) and neither belongs in
this audit.

**How the second process arose is not established.** I launched the build once, as a background
command, and armed two Monitors on the log — Monitors read, they do not run the builder. The
second writer's identity is unresolved and I am not going to guess it. What is established:
two processes, one log, one surviving rename, and a pack that validates.

### E18 (S2): the stream expression that discards a refusal — three runner paths, two defective, and b0's count is 6 of 10 not 10 of 10

6e's assignment from b0's MT-12, read independently. b0's finding is real and the mechanism
reproduces; two of its specifics do not survive an independent read, and the population is wider
than one call site.

**score_matrix drives its scorers through THREE runners, not one.** Enumerated from
`eval/score_matrix.py`, every `subprocess.run` in the file:

| runner | line | expression | scripts driven | defective? |
|---|---|---|---|---|
| `_run_eval_json` | 304 | `(r.stderr or r.stdout)` | `domain_bpb.py`, `lambada_en.py`, `humaneval_bpb.py`, `lambada_zh.py`, `math_v2_like.py`, `l1_fewshot.py` | **yes** — stderr shadows stdout entirely |
| `metric_minimal_pairs` | 281 | `(r.stderr or r.stdout)`, `[-1:]` | `base_matrix.py` | **yes**, and worse: one line, no exception-line search |
| `_run` | 387 | `(r.stdout + r.stderr)` | `run_eval.py`, `eval_hard.sh`, `eval_math.sh`, `eval_code.sh` | no — concatenates, keeps both |

So `_run` is the correct form and it is already in the same file, eleven lines below the second
defective site. The fix is not a design question; two call sites disagree with a third.

**Which stream each driven script uses for its refusal**, read per file (`sys.exit(str)` and
`raise SystemExit(str)` write to stderr; `print()` writes to stdout):

| script | refusal on stderr | refusal on stdout | affected by the shadowing |
|---|---|---|---|
| `domain_bpb.py` | 0 | 2 (`REFUSING: no domain produced a number` :292, `SKIPPED (round-trip …)` :267) | **YES — every refusal it has** |
| `humaneval_bpb.py` | 11 (6 `sys.exit`, 5 `raise SystemExit`) | 1 | mostly safe; the 1 stdout refusal is shadowed |
| `lambada_en.py` | 1 | 0 | no |
| `lambada_zh.py` | 1 | 0 | no |
| `l1_fewshot.py` | 1 | 0 | no |
| `math_v2_like.py` | 0 | 0 | n/a — has no refusal path at all |
| `base_matrix.py` | 0 | 0 | n/a, but its runner truncates to one line |

`domain_bpb.py` is the confirmed case for the reason b0 gives: it is the ONE script whose refusals
are all on stdout, driven by the ONE runner that discards stdout whenever stderr is non-empty.

**Broken-world test of the mechanism, run rather than reasoned.** A 4-line program that warns on
stderr and prints `REFUSING: no domain produced a number` on stdout, then exits 1, passed through
line 304's expression verbatim:

```
RECORDED  -> domain_bpb.py exited 1: <string>:2: UserWarning: checkpoint has no vocab_id (old
             format); cannot cross-check tokenizer
REFUSAL PRESENT IN RECORD? False
```

The refusal is gone and the record names a warning as the cause.

**Where I differ from b0, both directions.** b0 reports 10 of 10 `domain_bpb` rows as error
objects carrying only the vocab_id warning. Reading all 10 in `runs/score_matrix.jsonl`:

- **6 of 10** carry the warning as their whole recorded cause (`ckpt_data_leg_206m_8b.pt` and its
  step7000/step7500, `ckpt_params_leg_438m_3p76b.pt`, `ckpt_b0_n8_fixed.pt`,
  `ckpt_b0_sd_equalcompute.pt`).
- **3 of 10** carry a real cause that survived: "Retokenizing here would overwrite caches the live
  run reads…" (`ckpt_ab_untiehead_untiehead.pt.ep1`, `ckpt_ab_untieheadlr_untieheadlr.pt.ep1`,
  `pythia-160m-step2000`). Those failures wrote their cause to stderr, so the shadowing cost
  nothing.
- **1 of 10** is a third shape b0's description does not cover: `ckpt_data_leg_206m_8b.pt.step10000`
  records `ours_tok = load_tokenizer(a.tokenizer, None)` — `eval/domain_bpb.py:221`, a SOURCE LINE,
  no exception, no warning. That is the truncation failure the code's own comment at :299-303
  describes as already fixed. It is not fixed for this row; the row predates the fix.

So the count is 6, the mechanism is confirmed, and the defect has a shape neither b0's summary nor
the code comment names.

**A second affected metric b0's list omits — and my first count of it was wrong; see the
correction under E21.** `l1_fewshot` has 8 error rows. **ONE** is
`l1_fewshot.py exited -15:   448/497 acc=0.0%`: exit −15 is SIGTERM, killed not crashed, and what
got recorded is a progress line, because a killed process writes no exception anywhere and the
tail-3 fallback grabs whatever stdout last held. That is not the stream-shadowing bug but the same
consequence, a record naming progress as a cause. The other **7** name a real exception
(`eval_artifacts.ArtifactExists`) and are readable as published — I wrote "all of the form exited
-15" from reading one row and generalising, which is the aggregate-adjective error, in a finding
whose subject is records that misname their cause.

**Affected metrics, stated as the assignment asks:** `domain_bpb` (6 rows blinded by shadowing,
1 by truncation), `humaneval_bpb` (1 refusal path of 12), `base_matrix`/`minimal_pairs` (all
failures, one-line truncation), `l1_fewshot` (**1** row of the SIGTERM shape, not 8 — see E21).
Unaffected: `lambada_en`, `lambada_zh`, `math_v2_like`, and everything on the `_run` path.

**What normally writes to stderr on a clean run** — the precondition that makes the shadowing
fire. Verified: importing `scripts/loader.py` alone writes **0 bytes** to both streams, so the
warning is not unconditional at import; `loader.py:136` fires it per checkpoint via
`warnings.warn` when `vocab_id` is absent. Every old-format checkpoint therefore puts one line on
stderr and shadows all of stdout. Not verified, and named rather than assumed: whether torch/NCCL
banners reach stderr on the pod's GPU path. That needs a GPU run, which the audit forbids.

### E19 (S1): domain_bpb has never produced a number for any checkpoint, and the cause is one line

6e's ruling 2026-09-04 after the C11 launch prep. Found while building C11's launch line rather
than by auditing the metric: the plan said "domain_bpb first", so I ran its CPU half before asking
for a card, and it cannot produce a number at all.

**The measurement.** `domain_bpb.py` splits cleanly at the forward pass: everything before it —
`val_seqs` reading shards, `roundtrip_fraction` tokenizing — is CPU. Ran on the pod with
`CUDA_VISIBLE_DEVICES=""`, no model loaded, all 9 domains of `data/mix_200m_4b.json`, 64 rows each:

| domain | round-trip | verdict at `MIN_ROUNDTRIP = 0.98` |
|---|---|---|
| code_py_rp1t | 0.3594 | skipped |
| code_py_starcoder | 0.2188 | skipped |
| math_owm_stage2 | 0.1094 | skipped |
| zh_web | 0.1094 | skipped |
| en_c4_stage2 | 0.0156 | skipped |
| textbook_30b | 0.0156 | skipped |
| cot | 0.0000 | skipped |
| chatml | 0.0000 | skipped |
| chat_qa | 0.0000 | skipped |

Nine of nine. `out` is empty, so `domain_bpb.py:317` prints `REFUSING: no domain produced a
number` and returns 1. No GPU, no checkpoint and no re-score can change this.

**The 10 E18 error rows are this, not ten problems.** E18 read them as a stream-capture defect and
that reading holds for the RECORD — C6a makes the cause visible where it was replaced by a
`UserWarning`. But the metric itself was failing identically every time. C6a fixed what the ledger
says; it did not make the metric work. Stated plainly because my own E18 could be read as implying
the numbers were merely unrecorded.

**Cause, one line.** `tok.decode([EOS_ID])` returns `''`. Every val row from
`train._domain_seqs` is packed and EOS-delimited — `cot` row 0 holds 8 EOS in 4,097 ids — so
decode drops the delimiters and the re-encode is 4,089 ids, first differing at position 243 exactly
where an EOS was. `roundtrip_fraction` demands `encode(decode(ids)) == ids`, which no packed row
can satisfy while decode is lossy on the delimiter.

So the gate measures the tokenizer's special-token handling, not the property it claims: whether
the two arms would score the same bytes. Its docstring is right about why the check exists ("a
lossy merge or a normalisation step ... would otherwise turn into a silent difference") and the
lossiness it caught is its own decode call's.

**Measured fix direction, not a guess** — `decode(ids, skip_special_tokens=False)`:

| domain | before | after |
|---|---|---|
| cot | 0.0000 | 0.9688 |
| code_py_rp1t | 0.3594 | 1.0000 |
| zh_web | 0.1094 | 0.9375 |

One clears 0.98, two do not, so this is not a one-line fix that opens the gate. That is C12: keep
the special tokens, then derive the threshold from a known answer (unpacked plain text must
round-trip 1.0) and give the residual on packed rows a named cause before any number is chosen.

**What this costs downstream.** `domain_bpb` is the control arm's only cross-tokenizer metric, so
every our-vs-Pythia comparison in byte terms is unmeasured — E6 recorded the three ERROR panels as
a systematic gap without knowing the gap is total. Nothing published is WRONG because of this; a
metric that always refused never entered a number anywhere. The S1 is for a published metric that
does not exist.

### E20 (S1): on doc_cu the N2 verdict's block-paired sign test REVERSES, and the mean is 1/43 of its own SE

6e's assignment after C11 landed: run the block-paired delta on the two `#cu` rows with the same
instrument the published exit number used, and state the domain count. The answer is stronger than
the shrinking mean I reported: the paired test does not merely weaken, it changes direction.

**The instrument is the published one.** `eval/block_paired.py`, 576 blocks (9 domains x 64), the
same command the fact's `source` names. Arms: B = params leg, A = data leg, so a NEGATIVE delta
favours the params leg, which is the fact's own orientation.

| | published (cu_none) | doc_cu (`#cu` rows) |
|---|---|---|
| paired mean | **−0.010770** | **−0.000920** |
| sd | 0.039713 | 0.039866 |
| SE | 0.001655 | 0.001661 |
| t | **−6.51** | **−0.55** |
| sign test | 227 up, 349 down, p 2.10e-07 | **329 up, 247 down, p 3.62e-04** |
| n tokens | 2,359,296 | 2,359,296 |

**Two independent statistics now disagree in direction, which neither did before.** The mean stays
negative (params better) at −0.000920 while the sign test's majority flips to 329 of 576 blocks
where params is WORSE, at p 3.62e-04 — a significant majority pointing the opposite way from the
mean. The median confirms it: **+0.003204**, opposite in sign to the mean. On cu_none both agreed
(mean negative, 349 of 576 blocks negative). So the doc_cu mean is carried by a minority of
large-magnitude blocks, not by a consistent advantage: the extremes are symmetric (most negative
−0.2023, most positive +0.2187) and the typical block favours the data leg.

|t| = 0.55 is the compact statement: the delta is **1/43 of its own sampling SE** where it was
6.51 SEs before. Against `ds.seed_variance_0p2b`'s 0.0516 sd it is 1/56 of one seed sd, and against
the 0.24 nat `readable_move` it is 1/261.

**Per-domain, the split is by format, not noise.** Six of nine domains have a majority of blocks
where params is worse:

| domain | mean | median | blocks params-worse / better |
|---|---|---|---|
| cot | +0.0091 | +0.0113 | 52 / 12 |
| code_py_rp1t | +0.0031 | +0.0066 | 44 / 20 |
| en_c4_stage2 | +0.0044 | +0.0081 | 43 / 21 |
| code_py_starcoder | +0.0025 | +0.0031 | 42 / 22 |
| math_owm_stage2 | −0.0001 | +0.0034 | 36 / 28 |
| zh_web | −0.0042 | −0.0021 | 30 / 34 |
| chat_qa | −0.0084 | −0.0055 | 29 / 35 |
| chatml | −0.0119 | −0.0124 | 28 / 36 |
| textbook_30b | −0.0027 | −0.0022 | 25 / 39 |

The three domains that still favour the params leg by mean AND median are chatml, chat_qa and
textbook_30b — and chatml/chat_qa are exactly the two whose cu_none-to-doc_cu gain was largest
(−0.2364 and −0.1695, against −0.0157 for code_py_rp1t). The verdict's surviving margin comes from
the domains most affected by the mask, which is E2's non-uniformity showing up inside the pair
rather than cancelling.

**The domain count is 9, not 11.** 6e's message reported 11 and that is not what the rows carry:
all four rows (`ckpt_data_leg_206m_8b.pt`, its `#cu`, `ckpt_params_leg_438m_3p76b.pt`, its `#cu`)
carry exactly 9 named domains — chat_qa, chatml, code_py_rp1t, code_py_starcoder, cot,
en_c4_stage2, math_owm_stage2, textbook_30b, zh_web — enumerated from the `domain_loss` dict with
`unweighted_mean` and `_`-prefixed keys excluded. The N2 fact's `config` says "9 domains x 64
blocks", block_paired reports "576 blocks over 9 domain(s)", and 9 x 64 = 576 with no room for two
more. **The denominator is unchanged between the published number and the re-score**, so the
comparison above is like-for-like. Nothing in the repo produces an 11-domain count for these rows;
`data/mix_200m_4b.json` has 9 domains.

**A limit on this comparison, and it is mine to state.** The cu_none block-paired figures in the
table come from the fact's `config`, not from a re-run: `block_paired.py --arms
ckpt_data_leg_206m_8b.pt ckpt_params_leg_438m_3p76b.pt` REFUSES on the published rows with "no
per-block data ... A record scored before --per-block existed carries only the domain mean". The
cu_none per-block data lives in `runs/b0_23_blocks.jsonl` (present, 27,910 B), which is what the
fact's source pairs. I did not re-derive the published sd/SE/t from it; they are quoted. The doc_cu
column is measured by me end to end.

**What this does and does not say.** It does not say the params leg is worse. It says that on the
path that matches training, the two statistics that agreed on cu_none now point in opposite
directions and the mean is inside its own noise — so the N2 result is not a measurement of a
direction at this resolution. E3 recorded that the verdict rested on an instrument with a defect
7.6x the delta; this is that defect removed, and what remains is unresolved rather than reversed.
The decision belongs to 6e and b0.

### E21 (S2): E18's own `l1_fewshot` count was one row generalised to eight, and the E15 defect recurred while I was checking it

Found by starting e1-38, whose first item was "re-read the 18 score_matrix error entries and record
which now have a readable cause, since C6a changed the record and nobody has checked what it
produced". The re-read's first result was that two of E18's published numbers do not survive it.

**Correction 1: `l1_fewshot` is 1 SIGTERM row, not 8.** E18 says the metric "has 8 error rows, all
of the form `l1_fewshot.py exited -15: 448/497 acc=0.0%`". Read verbatim, all 8:

| ckpt | recorded cause |
|---|---|
| `ckpt_p02_fp32m_s0.pt` | `exited -15:   448/497 acc=0.0%` — the SIGTERM shape |
| `ckpt_pretrain_15b_s1.pt` | `exited 1: eval_artifacts.ArtifactExists: …preds_l1_d3.jsonl exists (454238 bytes)` |
| `ckpt_rehearse_resume.pt` | same `ArtifactExists` |
| `ckpt_rehearse_join.pt` | same |
| `ckpt_lrprobe_1.2.pt` | same |
| `ckpt_proberesume.pt` | same |
| `ckpt_p500m_20b_0902.pt.step1500` | same |
| `ckpt_b0_sd_equalcompute.pt` | `exited 1: File ".../eval_artifacts.py", line 79 … eval_artifacts.ArtifactExists: …` |

Seven of eight name a real exception and are readable exactly as published; one is the SIGTERM
shape. I read one row, saw `exited -15`, and wrote "all of the form" — the aggregate-adjective
error (`memory/aggregate-adjective-fakes-replication.md`), committed inside a finding whose whole
subject is failure records that misname their cause. E18's affected-metrics line said "`l1_fewshot`
(8 rows, SIGTERM shape)"; corrected in place to 1.

The 7 also change what e1-38 should fix. A SIGTERM cause line is worth adding, but it would improve
**one** row, not eight. The `ArtifactExists` seven point somewhere else entirely: the same
`preds_l1_d3.jsonl` path collides across six different checkpoints, so the artifact name does not
carry the checkpoint — the 8th row's path (`preds_l1_d3_ckpt_b0_sd_equalcompute.pt.zh.jsonl`) shows
the naming was later fixed. That is a real defect and it is not E18's.

**Correction 2: the population is 19 entries, not 18, on two different counting rules.** E18 says
"the 18 score_matrix error entries". Counting at E18's own commit (`2c87a493`, 60 rows): 19 by the
rule I used then (`'exited' or 'ERROR' in the serialized value`) and 25 by the plainer rule (an
`error` key exists). The 25 includes entries whose value carries an error key without the word
"exited" — `math_500`, `mc_ceval`, `code_500`, a degeneration entry. So "18" was neither rule; it
was the sum of 10 `domain_bpb` and 8 `l1_fewshot`, which is the two metrics I had looked at, stated
as if it were the population. Two rules, both defensible, neither giving 18.

**Correction 3, and it is the same defect E15 documents, made while investigating E15's neighbour.**
To date the E18 commit I ran `git log -1 --format='%ad' --date=format-local:'%Y-%m-%dT%H:%M:%SZ'`
and it printed **2026-09-04T12:38:11Z**. That is +0800 with a `Z` I supplied in the format string
myself: `--date=format-local` formats in the SHELL's zone, and `...Z` inside the format is a literal
character, not an assertion. Under `TZ=UTC` the same command gives **2026-09-04T04:38:11Z**. E15
recorded exactly this — "the `Z` was my own annotation, never output by any command" — and the fix
E15 named (`TZ=UTC` on the command) is the one I did not apply, because I put the `Z` in the format
string instead, which looks like the fix and is not. The correct form is `TZ=UTC git log ...`; a
`format-local` string ending in `Z` is a way to reproduce the defect while appearing to have fixed
it.

**What survives of E18.** Everything about the mechanism: three runners, two defective, the
`(r.stderr or r.stdout)` shadowing, `domain_bpb` as the one script whose refusals are all on stdout,
the broken-world test, and the 6-of-10 count (verified again here: at E18's commit exactly 10
`domain_bpb` error entries, 6 warning-as-cause; now 11 and 7, the growth being
`ckpt_b0_se_16lnew_1b.pt`). What does not survive is the `l1_fewshot` count and the population size —
both mine, both stated with more confidence than the reading behind them supported.

### E22 (S2): the seven `l1_fewshot` "failures" are a guard working, and one of them hides a real measurement nobody read

e1-38's third item, and it inverts what E18 and E21 both assumed. E18 called these failure records
unreadable; E21 corrected the count and said the seven "name a real exception and are readable as
published". Both readings stopped at the exception NAME. Reading what the exception says changes the
finding: `l1_fewshot` did not fail in these seven runs — it **refused to overwrite an existing
result**, which is what `scripts/eval_artifacts.open_artifact` exists to do.

**The seven, by artifact name and date:**

| ckpt | measured | artifact the run refused to overwrite |
|---|---|---|
| `ckpt_pretrain_15b_s1.pt` | 2026-08-31 | bare `preds_l1_d3.jsonl` |
| `ckpt_rehearse_resume.pt` | 2026-08-31 | bare `preds_l1_d3.jsonl` |
| `ckpt_rehearse_join.pt` | 2026-08-31 | bare `preds_l1_d3.jsonl` |
| `ckpt_lrprobe_1.2.pt` | 2026-09-01 | bare `preds_l1_d3.jsonl` |
| `ckpt_proberesume.pt` | 2026-09-01 | bare `preds_l1_d3.jsonl` |
| `ckpt_p500m_20b_0902.pt.step1500` | 2026-09-02 | bare `preds_l1_d3.jsonl` |
| `ckpt_b0_sd_equalcompute.pt` | **2026-09-04** | `preds_l1_d3_ckpt_b0_sd_equalcompute.pt.zh.jsonl` |

**Six of the seven are historical and explained.** The bare `preds_l1_d3.jsonl` name carried no
checkpoint, so every checkpoint's run collided on one path and `open_artifact` refused all but the
first. The name was fixed at `eval/l1_fewshot.py:371` on 2026-09-02T07:00:52Z (`5a989647`) — it now
interpolates `os.path.basename(args.ckpt)`, the demo language, and the arm flags. All six predate or
coincide with that commit. The guard did its job: `be.l1_3shot_retracted` records a 477-row preds
file that WAS overwritten this way on 2026-08-31, which is why the refusal was added.

**The seventh does not fit that story and is the finding.** `ckpt_b0_sd_equalcompute.pt` was measured
**2026-09-04**, two days after the naming fix, and its error names the FIXED, checkpoint-specific
path. So the collision is not with another checkpoint — it is with itself. I read the file on the
pod rather than inferring: `/work/aupai/data/eval/preds_l1_d3_ckpt_b0_sd_equalcompute.pt.zh.jsonl`,
423,307 B, Sep 4 00:39, **497 rows**, and it is a complete result — `ok=True` on 9 of 497,
**accuracy 0.0181**. The generations are real Chinese arithmetic attempts, not empty.

So a valid L1 measurement of a Stage D arm exists, was produced at 00:39Z, and the score_matrix row
for that checkpoint records `l1_fewshot` as an ERROR. Whoever reads the row sees a failure; the
number is on disk. Nothing published cites it, which is why it went unnoticed — but "the metric
errored" and "the metric ran and nobody transcribed it" are different states and the ledger says the
wrong one.

**Why this is S2 and not S1.** No published number is wrong: `l1_fewshot` for
`ckpt_b0_sd_equalcompute.pt` appears nowhere in `facts/*.json`. The defect is that the ledger's
error entry contradicts an artifact in the same run's output directory, and the reason it does is
that `metric_l1_fewshot` passes neither `--force` nor `--run`, so a re-score of the same checkpoint
can only ever refuse. A second scoring pass over any already-scored checkpoint produces this row.

**Not fixed, per audit rule 5 and because the right fix is a decision, not an edit.** Three options
and they differ in what they cost: pass `--run` so each attempt versions its own artifact (keeps
every generation, grows the directory); pass `--force` (loses the prior generations, which is exactly
what `be.l1_3shot_retracted` was about); or have `metric_l1_fewshot` READ the existing artifact when
one is present and record its accuracy rather than re-generating. The third is the only one that
turns these rows into numbers without either discarding data or re-running a card job, and it is a
change to what "scoring" means, so it belongs to whoever owns the metric.
