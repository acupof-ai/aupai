# What counts as good synthetic pretraining data

Written 2026-08-29, before we buy or build any. Every criterion below is checkable, because
"high quality" is what a dataset card says and this project has already been burned by
believing one: a published quality score correlates with our own measurement at Spearman
**+0.198**, and in `opencsg/Fineweb-Edu-Chinese` the **top** score band was the dirtiest
(52/66/59% usable across three bands).

## The distinction that decides everything

Two things get called synthetic and only one of them works at our size.

| | anchored REPHRASING | from-scratch GENERATION |
|---|---|---|
| what the source contributes | the document | a topic |
| can you check the output against it | **yes** — facts, numbers, entities | no |
| measured, 1B model | **+6.7pp** over web, 7.7x training speedup | 47.1%, ties naive summarisation at 46.7% |
| safe share of the mix | ~30% | **under 5% for sub-1B**, collapse above |

Measured on the two candidates we downloaded, 2026-08-29:

```
opencsg/chinese-fineweb-edu-v2      cols [text, score, __index__, source]
   source: CCI3 / IndustryCorpus2               <- real web, this is FILTERING not synthesis
   text:   "生理盐水的成分是什么？…[编辑本段]性质"   <- raw encyclopedia, edit artefacts intact
   300,271 rows/file, ~420B tokens total, score mean 0.690

opencsg/chinese-cosmopedia          cols [text, score, source, data_format]
   source: baike / blog / knowledge qa          <- only the TOPIC came from here
   data_format: middle-school textbook / normal story / college textbook / wikihow
   text:   "### 课程单元：速写概述 … 本单元将深入探讨速写的概念"
   269,551 rows/file, score mean 0.839
```

**cosmopedia's output has no checkable relationship to its own `source`.** Its samples — "正丙基
亚砜", "速写概述" — could have been written from any source with the same title. That is
from-scratch generation, and `data/mix_v3.json` gives it **36%** of a 3.3B-token budget.

## The six criteria

1. **Anchored to one source document, verifiably.** Numbers and named entities in the output
   must be a subset of the source's. This is the only hard test that separates interpretation
   from invention, and it is cheap to run.
2. **It adds something the source lacks, and you can name what.** Only three things count:
   format (web -> QA), register (-> pedagogical), and **explicit intermediate steps**. If you
   cannot name the addition it is a paraphrase, and paraphrase saturates — naive summarisation
   scores 46.7% against rephrasing's 50.4%.
3. **One source, several styles.** Single-strategy generation shows diminishing returns from
   "lack of stylistic diversity". Measure the distribution over strategies and audiences, and
   n-gram diversity against the source corpus.
4. **No eval leakage.** `scripts/scan_contamination.py` on every new source, before it enters
   a mix. This is finding #1 of docs/review_2026-08-26.md and it recurs whenever skipped.
5. **A minority of the mix.** ~30% for rephrased, **under 5% for from-scratch at 200M**.
6. **The verdict is held-out loss on the OTHER domains.** Ours, not borrowed: a model trained
   on 36% textbook will of course score well on textbook. Synthetic data that improves loss on
   its own kind proves nothing. The falsifying measurement is two pretrains differing only in
   the synthetic share, compared on web / wiki / math.

Criterion 6 is running as `runs/ab_tb36.log` vs `runs/ab_tb05.log` — 500 steps each, identical
seed and card count, `data/mix_v3.json` (textbook 36%) against `data/mix_v3_lowtb.json` (5%).

## Where the open data actually is

- **Scale, real text**: `opencsg/chinese-fineweb-edu-v2`, ~420B tokens from CCI3 and
  IndustryCorpus2. Filtering, not synthesis. **Deprecated in favour of Fineweb-Edu-Chinese-V2.1.**
  Do not trust its own `score` column as a cut — run `datagen/audit_source_score.py` first, and
  our own filters after.
- **Rephrased and CoT synthesis, Chinese**: `BAAI/CCI4.0-M2-Base-v1` (10,867 files, 1,975 of
  them `zh_cc-*`, plus `Nemotron-CC-high-synthetic-{extract_knowledge,wrap_medium,
  diverse_qa_pairs,knowledge_list}`) and `BAAI/CCI4.0-M2-CoT-v1` (1,349 files,
  `cot_synthesis_{CC,math,arxiv,wiki,code}-{low,mid,high}`). **Both are GATED** — they need an
  HF account to accept the terms, which is why they are not downloaded here.
- **`nvidia/Nemotron-CC-v2`** — 2.1T rephrased tokens from Qwen3-30B-A3B over 110 CC snapshots,
  five prompts. `Diverse-QA`, `High-Quality-Synthetic`, `Translated-Diverse-QA`. **Also gated.**

CCI4.0-M2-CoT-v1 is, subset for subset, the thing we were about to spend days building with the
27B: explicit-reasoning synthesis over Chinese CC, math, arxiv, wiki and code, banded by
quality. Accepting its terms is cheaper than generating it.

## On building it ourselves

Still worth doing for the one thing no open set contains: **procedure execution in our own
format**, the failure measured in `docs/exp_procedure_sft.md`. But not before criterion 6
returns, and not with a 27B generator by default — measured, rephraser quality saturates at
about 3B (1B -> 3B is +1.5pp, 3B -> 8B is +0.4pp, and a 70B often loses to an 8B).
