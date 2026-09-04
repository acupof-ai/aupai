---
question: What is the true state of evaluation and held-out on 2026-09-04, read from artifacts?
status: open
area: evaluation and held-out
owner: e1
pair: 3b
source: user order 2026-09-04, method docs/standards/audit_0904.md
---

# Audit: evaluation and held-out, 2026-09-04

Partial report at the 3-hour mark, per the method's "a partial report at 3 hours beats a
complete one at 6". Section 5 names what is not yet covered.

## 1. Scope

Covered:

- `runs/score_matrix.jsonl` — all 60 rows, every field, read directly.
- `runs/b0_23_blocks.jsonl`, `runs/b0_final_blocks.jsonl` — all rows.
- `eval/domain_loss.py` and `eval/score_matrix.py` — the cu path, line by line.
- `datagen/holdout.py` — the registry and `EVAL_FILES`, imported and enumerated.
- `facts/contamination.json` — all 36 entries: status, source, population.
- `facts/*.json` — every entry citing `score_matrix`, by grep over the serialized entry.
- Corpus domain build dates: all 321 directories under `/work/aupai/data/corpus/` on the pod,
  by `stat`.

Deliberately excluded from THIS report, listed rather than silent:

- The per-file prompt-format sweep over all 46 `eval/*.py` scorers. Delegated and still
  running when this was written; it is the largest remaining piece and section 5 says so.
- `eval/*.sh` drivers.
- Anything requiring a GPU. The audit forbids launches and I hold no card.

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
| eval scorers under eval/ | 46 | 4 in this report | see §5 |

## 4. Findings

| id | sev | claim as published | evidence | what contradicts it |
|---|---|---|---|---|
| E1 | S1 | `bfa1a846`'s subject line: "domain_loss passes the document mask, **and every row records which path it used**" | `git show bfa1a846 --name-only` lists no `eval/score_matrix.py`. `eval/score_matrix.py:244` calls `domain_loss_seqs(model, rows, device, per_row=True)` with no `cu_path`, taking the `cu_none` default at `eval/domain_loss.py:194`. | 0 of 60 published rows carry a cu label under the strict regex; 51 of them carry `domain_loss`. The fix labelled `domain_loss.py`'s own CLI output (`eval/domain_loss.py:763`, `:794`) and left the file that writes the published ledger untouched. Two rows measured 2026-09-04, after the fix, are also unlabelled. |
| E2 | S2 | `facts/efficiency.json#eff.eval_path_cu_artifact_ce` boundary: "NO PUBLISHED DELTA MOVES BECAUSE OF THIS … the artifact is common-mode and cancels in a difference." | Same fact's own sibling: `facts/smelt_deeploop.json#repo.loop_from_scratch_stage_d` uncertainty records the cu-passed rescore moving the pooled Stage D delta from −0.030937 to −0.022325 nat, "so 28% of the measured advantage was leak-mediated and 72% was not." | The artifact does NOT cancel. It cancelled to first order and left 28% of one measured delta behind, with a per-domain shrink correlating with per-domain leak at −0.951. The boundary states the cancellation as a property; the measurement shows it is an approximation whose residual is 28% on the one delta anyone re-scored. |
| E3 | S1 | `ds.n2_params_vs_data_matched_compute` = −0.010770 nat, the params-vs-data verdict, and the roadmap's N2 decision line "30B shape leans larger-parameter at fixed compute". | Row `measured: 2026-09-03` for both arms in `runs/score_matrix.jsonl`; the cu fix landed `2026-09-04 06:41` (`git log bfa1a846`). `runs/b0_23_blocks.jsonl` has `path` ABSENT. `eff.eval_path_cu_artifact_ce` measures the artifact at −0.0818 nat pooled. | The verdict was measured entirely on the `cu_none` path, and the artifact is **7.6× the delta itself** (the fact says so in those words). E2 shows the artifact does not fully cancel. Nothing in the N2 fact's boundary or uncertainty mentions the cu path at all — the boundary discusses seed sigma and the shared mix. So a decision about the 30B shape rests on a number whose instrument had a known defect 7.6× its size, and the fact does not say which path produced it. |
| E4 | S2 | The holdout guard's population, post-fix: `datagen/holdout.py` REGISTRY, 13 entries, replacing the 4-path `EVAL_FILES`. | `git log -1 e970c343` → `2026-09-04 08:50:45 +0800`. Pod `stat` over all 321 corpus directories: the newest is `2026-09-03` (`cot_open_thoughts`, `en_c4_30b`, `rp1t_arxiv_papers`); every one of the 9 domains in `data/mix_200m_4b.json` is dated 2026-08-31 or 2026-09-01. | **Every corpus domain that exists was built before the registry existed.** The registry fixes what the NEXT build excludes and cannot retroactively protect any current domain. The 13-entry population is correct and no corpus has been built against it. `cont.heldout_in_pretrain_corpus` records this for its own scan; no other contamination fact does. |
| E5 | S2 | `runs/score_matrix.jsonl` folds rows on `ckpt`, and `domain_loss.py` appends `#cu` to the ckpt name so a doc_cu row does not collide with a cu_none one (`eval/domain_loss.py:770-775`, comment). | 0 of 60 rows have `#cu` in the ckpt name. `runs/b0_final_blocks.jsonl` has 2 rows that DO (`ckpt_b0_sd_equalcompute.pt#cu`, `ckpt_b0_n8_fixed.pt#cu`, both `path: doc_cu`). | The collision guard exists only on the path that writes the blocks ledger. `score_matrix.py` neither labels the path nor suffixes the name, so when a doc_cu re-score is written there it will fold onto the cu_none row of the same checkpoint and silently replace it — the exact outcome the comment at `domain_loss.py:770` was written to prevent. Nothing wrong yet: no doc_cu row has been written to score_matrix. |
| E6 | S3 | 14 `score_matrix` rows appear to carry a `path` field. | The 14 are `"path": "/work/aupai/data/eval/preds_l1_d3.jsonl"` on rows 16-19 and 24 — a predictions-file location, not a forward path. | Cosmetic, but it is why a grep for `path` in this ledger reads as partially labelled when the cu label is absent everywhere. Noted so the next reader does not repeat my first mistake. |

## 5. Blind spots of this audit

- **The 46-scorer prompt-format sweep is not in this report.** Whether any eval hands ChatML
  to a base checkpoint is therefore UNANSWERED here, and it is one of the three questions the
  area was assigned. It is running; a follow-up section will carry it.
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
