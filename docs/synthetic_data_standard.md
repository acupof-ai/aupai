# What counts as good synthetic pretraining data

Written 2026-08-29, before we buy or build any. Every criterion below is checkable, because
"high quality" is what a dataset card says and this project has already been burned by
believing one: a published quality score correlates with our own measurement at Spearman
**+0.198**, and in `opencsg/Fineweb-Edu-Chinese` the **top** score band was the dirtiest
(52/66/59% usable across three bands).

## The classification that sets the weight

The literature measures two categories. **Our first real candidate fits neither**, so the
classification a source gets is a measurement, not a lookup.

| | anchored REPHRASING | seed-anchored, FACTS STRIPPED | from-scratch GENERATION |
|---|---|---|---|
| what the source contributes | the document | the document | a topic |
| output checkable against it | **yes** — facts, numbers, entities | the seed is provable, the facts are gone | no |
| what the literature measured | 5–10x speedup to the same Pile PPL, at 33% and 67% alike | **nothing — not a studied category** | 33% needs ~20B tokens to catch plain CC; 67% never does |
| safe share of the mix | ~30% | **prior <5%, moved by criterion 6** | **under 5% for sub-1B** |
| our example | CCI4.0-M2-CoT (98% number subset, n=200, CC-high only) | `chinese-cosmopedia` | none admitted |

The two outer columns come from arXiv 2510.01631, whose grid runs 200M–1B params × 1B–50B
tokens — the smallest cell is 200M/1B, so **30% is an interpolation at our size**. Its metric is
**Pile perplexity over 14 domains plus Wikitext, not benchmark accuracy**; no downstream point
difference is reported at any scale.

The middle column is ours. It exists because a source can pass the "was it handed a document"
test and still fail the thing that test is a proxy for — cosmopedia's seed is provable in 2.97%
of documents and its dates survive at 0.18x of real web. Neither bound was measured on anything
like it, so criterion 6 sets the weight. **Until it reports, the prior is the from-scratch
bound**, because 0.12–0.22x fact survival is measured information and an unconstrained prior
throws it away; criterion 6 can move it in either direction.

Earlier versions of this file gave "+6.7pp at 1B", "7.7x", and "47.1 / 46.7 / 50.4". Audited
2026-08-29 against 2510.01631v1 (both HTML renderings), 2506.04689 and 2508.10975v2: **none of
those numbers appears in any of them, and no source was found.** They are removed rather than
replaced. Two sessions searching independently also failed to locate the BeyondWeb paper itself
— cited here as 2508.10975v2, which one session did read. Treat any claim in this file that
rests only on it as unverified.

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

**cosmopedia is seeded on a real document and then rewritten until most of the document is
gone.** It is neither of the two categories above, and two hand reads called it from-scratch
before a census showed otherwise (`docs/audit_cosmopedia.md`):

- **2.97% of documents (8,003 / 269,551) name the seed in their own text** — 「网页摘录」
  「上述文本」 — and the content beside those references is the seed's: a chipset model, a
  county bureau's duties, 三重县's 5,776.56 km². That is a prompt leak, so it identifies the
  pipeline for **all** of it, not the anchored share.
- **The anchoring is diluted, and that is what to measure.** Checkable-fact markers per 1,000
  chars against `web_hq`: dates **0.18x**, date+month **0.11x**, percentages **0.12x**, number
  +unit **0.22x**, any number 0.52x, pedagogical framing (本单元/我们将) **11.4x**. Most of its
  digits are section numbering.
- Its own `score` cannot separate anything: **0.828 vs 0.836**, so no `score >= X` cut exists.

`data/mix_v3.json` gives it **36%** of a 3.3B-token budget. Neither the 30% nor the 5% bound
applies; the weight is set by criterion 6, not read off the literature.

**The general lesson: "is it anchored" is not a document-level judgement call.** Reading whether
prose *feels* generic classifies register, not provenance. Use a marker whose presence proves the
pipeline (a seed reference, a `metadata.raw` field, a number subset test) and then measure how
much of the seed survives, as a rate against a real-text control.

## The six criteria

1. **Anchored to one source document, verifiably.** Numbers and named entities in the output
   must be a subset of the source's. This is the only hard test that separates interpretation
   from invention, and it is cheap to run.
2. **It adds something the source lacks, and you can name what.** BeyondWeb (2508.10975v2)
   defines three strategies and only three: format transformation (web -> QA), style
   modification (-> pedagogical register), and content restructuring. If you cannot name the
   addition it is a paraphrase.
   An earlier version listed **explicit intermediate steps** as a fourth. It is not in that
   paper — "explicit reasoning" returns zero hits in the full text — and no other source was
   found for it. Kept out until one is. This matters for CCI4.0-M2-CoT: added reasoning steps
   are not evidence of anchoring, so the anchoring test in criterion 1 still decides whether it
   takes the ~30% bound or the 5% one.
3. **One source, several styles.** Multi-strategy beats single-strategy in BeyondWeb, as a
   saturation curve with no per-strategy ablation behind it. Measure the distribution over
   strategies and audiences, and n-gram diversity against the source corpus.
   Rephraser size saturates at ~3B: 1B -> 3B is +1.5pp, 3B -> 8B is +0.4pp. Using the 27B to
   generate is spending 9x for the +0.4.
4. **No eval leakage.** `scripts/scan_contamination.py` on every new source, before it enters
   a mix. This is finding #1 of docs/review_2026-08-26.md and it recurs whenever skipped.
5. **A minority of the mix.** ~30% for rephrased, **under 5% for from-scratch at 200M**.
6. **The verdict is held-out loss on the OTHER domains.** Ours, not borrowed: a model trained
   on 36% textbook will of course score well on textbook. Synthetic data that improves loss on
   its own kind proves nothing. The falsifying measurement is two pretrains differing only in
   the synthetic share, compared on web / wiki / math.

Criterion 6 is NOT cleanly answered: the prepared comparison is confounded and cannot support a
"from-scratch synthetic is harmful" claim.

`ckpt_tb36` (textbook 36%, `mix_v3`) vs `ckpt_tb05` (textbook 5%, `mix_v3_lowtb`), each 500 steps,
same seed 42, same val split, same vocab_id 0bce3584bc24f255, both fone. Held-out loss on the
same fixed web_hq/wiki shards:

| domain | tb36 (36%) | tb05 (5%) | Δ (36%−5%) |
|---|---|---|---|
| web_hq | 5.2296 | **5.1330** | +0.097 |
| wiki | 4.6598 | **4.5509** | +0.109 |

**The comparison is confounded.** `mix_v3_lowtb`'s own comment says the freed weight goes to the
real-text domains in their existing proportions — so tb05 has 31% less textbook AND ~31% more
web/wiki. Scoring the two arms on the web_hq/wiki holdout then measures an arm trained on more web
against one trained on less. A model trained on more web scoring better on web holdout is a
tautology, the mirror image of the in-kind-training trap criterion 6 was written to guard against.

So this answers **"at a fixed token budget, is replacing textbook with web worth more?"** — yes,
and that supports **down-weighting** the seed-diluted synthetic (0.36 → 0.12). It does NOT answer
"is from-scratch synthetic harmful to the representation"; that claim has no evidence here and
needs an equal-exposure design (arms with identical web/wiki/math token counts, differing only in
an added textbook slab) before it can be asserted. No paired test was run; Δ≈0.10 (relative ~2%)
is the magnitude a 31% same-domain data difference explains, so treat both point estimates as
unverified beyond the confound.

Both arms are fone; the comparison holds on the fone channel only and does not transfer to a
non-fone mix_v4 target without re-measuring.

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
