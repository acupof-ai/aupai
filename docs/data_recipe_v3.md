# Pretrain corpus v3 — the recipe, and the measurements behind each choice

Written 2026-08-29, for the rebuild that replaces the 11.5B-token corpus behind
`ckpt_k5_clean_0827` and `ckpt_k6_fone`.

## Why rebuild

Three measurements, each on the existing corpus.

| | measured | how |
|---|---|---|
| web share of the pretrain | **88%** | `data/mix.json` |
| web documents worth training on | **18%** | 180 random documents, read by hand |
| fineweb2 Chinese slice that is traditional, never converted | **59.4%** | `scripts/t2s_corpus.py` on the source |

The 82% is not one problem. Reading the 180 gives six distinct kinds: gambling
and adult SEO pages (with brand names injected mid-sentence into otherwise
ordinary text), enterprise product sheets with phone numbers, hospital ads,
serialized web novels, machine translation, forum fragments spliced together,
and synonym-substituted plagiarism that teaches wrong Chinese — `曩昔五年` where
a person would write `过去五年`.

For scale: 11.5B tokens over 200M parameters is **57 tokens per parameter**.
SmolLM2-135M trained on 2T, about 15,000. We are not going to close that, and we
do not have to: FineWeb-Edu's result is that filtering to their threshold 3
removes 92% of the corpus and still reaches the same MMLU with **10x fewer
tokens**. That is the licence to train on less.

## Sources

| source | unique tokens | provenance | human or synthetic |
|---|---:|---|---|
| our fineweb2 web, cleaned + filtered | ~1.8B | `data/corpus/web`, 8.11B before filtering | human |
| opencsg/chinese-cosmopedia | 4.69B available | 8 of 62 parquets, 2.16M docs | **synthetic** |
| wikipedia zh | ~0.35B | wikimedia/wikipedia 20231101.zh | human |
| existing math / chat / code / en | 0.41B | `data/corpus/*` | mixed |
| Chinese-DeepSeek-R1-Distill 110k | ~0.2B | reasoning traces | synthetic |

Cosmopedia's own metadata: 30% middle-school textbook, 20% college textbook, 18%
wikihow, 32% story; 71% sourced from baike. Its `score` column runs 0.75–0.95, so
what is published is already a high-score slice.

## The one rule that shapes the mix: synthetic data is capped

Cosmopedia alone could supply the whole corpus, and it must not. SmolLM2 uses
Cosmopedia v2 as roughly 11% of its mix (28B against 220B of FineWeb-Edu); the
bulk is real web. A corpus that is mostly model-generated text inherits that
model's distribution, and at 200M parameters we have no way to measure the
damage — every multiple-choice benchmark we own sits at the 25% chance line
(C-Eval: k5 24.8%, k5+SFT 23.7%, k6 23.0%, against +/-1.34pt).

So the target keeps real human text as the plurality and treats cosmopedia as
the textbook supplement it is.

## Target mix, ~4.5B tokens, one epoch

| domain | tokens | share | note |
|---|---:|---:|---|
| web (filtered) | 1.8B | 40% | real text, the anchor |
| textbook (cosmopedia) | 1.4B | 31% | capped below web on purpose |
| wiki | 0.35B | 8% | human, factual |
| math | 0.30B | 7% | existing + mathbank |
| chat | 0.25B | 6% | existing + R1 distill, in ChatML |
| en | 0.16B | 4% | |
| code | 0.06B | 1% | |

Roughly one epoch over unique data, against a schedule that currently repeats the
small domains 4–6 times. 11.5B -> 4.5B is a 2.6x cut in compute.

## The filters, in order of how certain each one is

1. **Traditional -> Simplified.** Deterministic. `datagen/clean_web.py` applies
   the opencc-derived table; running it after `scripts/t2s_corpus.py` is a no-op.
2. **Keyword spam.** Gambling, adult, contact details. Unambiguous per hit.
3. **Within-document repetition and fragment splicing.** Structural, no model.
4. **Educational-quality classifier.** Last and softest, and the only one whose
   threshold is a judgement call.

### What it took to get a usable classifier, including what failed

Every cheap route was measured against the same 180 hand labels before reaching
for a 27B. AUC, 5-fold where applicable:

| | AUC |
|---|---:|
| gambling/contact spam regex alone | 0.50 |
| hashed character 2–4 grams | 0.60 |
| structural features (tables, phones, quote density, repeated segments) | 0.62 |
| Qwen3-0.6B, 0–5 rubric | 0.539 |
| Qwen3-0.6B, binary yes/no | 0.647 |
| **Qwen3.8-27B, binary yes/no** | **0.739** |

Two findings worth carrying forward:

- **Character n-grams rank by topic; the labels split on register.** A page about
  air conditioners is a technical explainer or a product sheet and its n-grams
  barely differ. This is why the cheap route has a ceiling, not a tuning problem.
- **Rubric design beat model size.** The same 0.6B went 0.539 -> 0.647 when six
  levels became one yes/no question. Do not assume a bigger model can take a
  finer scale; measure it.

The 27B's operating point, on the hand labels:

| | |
|---|---:|
| base rate (hand-labelled keep) | 18.3% |
| documents the model says yes to | 21.8% |
| of those, hand-labelled keep | **52.8%** |
| enrichment | **2.9x** |

**Its honest limits.** n=165 with about 30 positives puts the 95% interval on
that AUC near +/-0.08, so "clearly above 0.62" holds but not overwhelmingly. And
52.8% precision means nearly half of what survives is still junk by the same
labels — the deterministic filters above have to carry the rest. The labels are
one person's judgement; they want a second reader.

**One earlier run of this number was invalid and is recorded so it is not
repeated.** `max_tokens` was set to 24 from a toy prompt where the model's
`<think>` block came back empty. Real documents make it think for a few hundred
tokens, 24 truncated it mid-reasoning, and the parser picked a stray digit out of
the reasoning text — that is the whole of the five-point score, AUC 0.407,
anti-correlated with the labels. Worse than the wrong number: truncation
correlates with length and length correlates with quality, so the documents that
failed were the good ones (13.7% hand-keep among answered against 28.6% among
failed). Any run of this scorer now prints the answer rate and that split, and
says outright that nothing is trustworthy below 90%.

## Format

Everything is ChatML (`scripts/loader.format_example`), including the
**pretraining** chat domain — so the format is seen during pretraining and SFT
does not have to teach it from nothing in a few hundred steps. The eval
contamination filter in `datagen/build_corpus.py` matches ChatML *and* the old
问：/答：, because the corpus still holds documents written the old way.

## Open questions, not yet answered

- Is opencsg's own filtering better than ours on the same axes? Being SOTA is not
  evidence; belle passed four automated checkers and was 38.7% defective by hand.
  Audit in progress.
- Contamination: every new source needs `scripts/holdout.py` run over it. Not yet
  done for cosmopedia or wiki.
- 32% of cosmopedia is story format. Useful for fluency, but the right weight for
  a 200M model is not measured.
