# Corpus v3 — what it is, how it was made, what is wrong with it

Built 2026-08-29 to replace the 11.5B-token corpus behind `ckpt_k5_clean_0827`.
Recipe and reasoning: `docs/data_recipe_v3.md`. Weights: `data/mix_v3.json`.

## Contents

| domain | directory | documents | tokens | origin | human or synthetic |
|---|---|---:|---:|---|---|
| web_hq | `data/corpus/web_hq` | 1,126,846 | **1.12B** | fineweb-2 cmn_Hani, rebuilt from `/data00/fw2raw`, t2s, filtered | human |
| textbook | `data/corpus/textbook` | ~800K | 1.74B | opencsg/chinese-cosmopedia, 8 of 62 parquets | **synthetic** |
| wiki | `data/corpus/wiki` | 212,413 | 0.23B | wikimedia/wikipedia 20231101.zh | human |
| math | `data/corpus/math` | 34,887 | 0.15B | mathbank + school_math_r1_zh | mixed |
| en | `data/corpus/en` | 37,642 | 0.16B | cosmopedia_extra | synthetic |
| code | `data/corpus/code` | 13,370 | 0.06B | code_filtered | human |
| chat | `data/corpus/chat` | 9,255 | 0.04B | coig, in ChatML | human |

The unfiltered `data/corpus/web` (2,991,648 documents, 3.55B tokens) is kept so a
different quality threshold can be cut without rebuilding. **The mix must name
`web_hq`**: train.py globs `data/corpus/<domain>/`, so the unfiltered corpus would
train perfectly well and silently discard every filter.

## How web_hq was filtered

2,991,648 → 1,126,846 (37.7%).

| rejected by | share |
|---|---:|
| quality classifier, below the 60th percentile | 52.0% |
| gambling / adult / contact spam by keyword | 10.3% |
| within-document repetition | 0.0% |
| fragment splicing | 0.0% |

Repetition and splicing near zero because the rebuild's dedup already removed
them; the filters stay because a future source will not be deduped.

The classifier is a logistic head on the frozen 200M's mean hidden state, trained
on 4,887 documents judged by Qwen3.8-27B (MMLU 76.3%). Against 180 documents read
and labelled by hand it reaches **AUC 0.825**, above the 27B teacher's own 0.739 —
the teacher's hard yes/no ties cap its AUC, the student's continuous score
recovers the ordering. Keeping the top 20% gives 52.8% hand-labelled-keep against
a base rate of 18.3%.

## What is wrong with it

- **The base rate is 18.3%.** Even after filtering, by the same hand labels
  roughly half of what survives is not worth training on. The filter is a 2.6x
  enrichment, not a cleaner.
- **The hand labels are one person's.** 180 documents, one reader, and they also
  steered earlier choices in the pipeline (feature dimension, rubric). Not a clean
  held-out set. n=180 puts the interval on that AUC near ±0.07.
- **textbook is model-generated** and could supply the whole corpus. It is capped
  below web on purpose; SmolLM2 uses Cosmopedia at ~11% against real web, and no
  benchmark we own could detect an overdose — every multiple-choice eval sits at
  the 25% chance line (C-Eval: 24.8 / 23.7 / 23.0, chance 25%, ±1.34pt).
- **Published quality scores in the sources are not usable as thresholds.**
  Measured: cosmopedia's own `score` column correlates with ours at Spearman
  +0.198 and is non-monotonic across its own bands. A 120-document hand audit of
  opencsg/Fineweb-Edu-Chinese found the same shape — bands 52% / 66% / 59% usable,
  top band dirtiest.
- **Our classifier cannot rank cosmopedia**, and this was measured rather than
  assumed. It scores cosmopedia below raw web (median −1.67 against −1.33). Judged
  instead by the 27B itself on one rubric:

  | | judged educational |
  |---|---:|
  | our web, unfiltered | 21.8% |
  | cosmopedia | **59.3%** (383/400 answered) |

  An independent 120-document hand audit of a sibling opencsg corpus landed on
  **59%**. Two methods, different corpora, one number. So the classifier's contrary
  signal is an out-of-distribution artifact: it was trained on the 27B's judgements
  of *web pages* and textbook prose is outside that. **Cross-source comparison needs
  one judge on one rubric.**

  This does not raise textbook's weight. The cap is a policy about synthetic data
  inheriting a generator's distribution, with no instrument here able to detect an
  overdose; better quality does not lift it.
- **t2s covers single-codepoint 1:1 mappings only** (3,553 entries). Zero
  convertible characters remain, but vocabulary-level differences (軟體/软件) were
  never in scope.
- **Nothing here is validated by a trained model yet.** Every number is a property
  of the data. Whether this corpus produces a better model than v2 is unmeasured.

## Provenance and contamination

`scripts/scan_contamination.py`, 60,000 documents per source, whole document and
every line up to 500 characters against `scripts/holdout.py`: **zero eval questions**
in cosmopedia, wiki-zh and the R1 distill set.

`data/pretrain_full.jsonl` was dropped from the rebuild. `data/PROVENANCE.md`
records its origin as unknown; that is why web fell from 8.11B to 3.55B, and
dropping it is a gain.
