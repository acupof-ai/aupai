# Pretraining data recipe

Replaces `data_recipe_v2.md` (1.27B tokens) and `data_recipe_v3.md` (a 4B-token
SkyPile / OpenWebMath / The-Stack plan that was never implemented). Both described
targets rather than what was built; this describes what runs.

## The mix

11.33B tokens scheduled from 8.52B of corpus, so most domains repeat. Weights and
epoch caps live in `data/mix_v3.json`; `python scripts/check_mix.py` dry-runs the
schedule before a launch.

| domain | corpus | main weight | anneal weight | epoch cap |
|---|---|---|---|---|
| web | 8.11B | 88% | 44% | 2 |
| en | 162M | 5% | 10% | 4 |
| math | 150M | 4% | 28% | 5 |
| code | 58M | 2% | 8% | 4 |
| chat | 40M | 1% | 10% | 6 |

Training runs the main phase, then gives the last `Cfg.anneal_frac` of tokens the
anneal weights. Math goes from 4% to 28% there.

**The epoch cap is shared across both phases**, which is what sets the main weights
so low for the small domains. An earlier version gave math 8% of the main phase and
28% of the anneal; `check_mix` showed the anneal capped to zero rows, because the
main phase had already spent math's entire epoch budget. Each small domain now
spends about half its budget in the main phase.

## What the corpus is worth

The filters are the product, not the sources. `datagen/build_corpus.py` runs every
domain through the same cleaning, cross-domain dedup, near-duplicate removal for
math, and the eval-holdout filter; sources are interchangeable behind it.

Measured on this project rather than assumed:

- A clean-corpus pretrain (k5) matched the older k4 on both math metrics — math-500
  51.2% against 51.6%, p=0.899; math-hard 1.9% against 2.9%, p=0.152 — while holding
  a better validation loss, 2.020 against 2.086. Cleaning buys language modelling at
  equal math. The math ceiling was a number-representation problem instead, which is
  what `--fone` addresses.
- `data/synthetic/math_short_v8.jsonl` matches `math_hard_eval_1k` on answer length
  (median 88 against 85) and level mix (100% L3/L4 both). Earlier batches did not,
  and the SFT built on them measured harmful: k5 51.2% dropped to 44.8%, p=0.043.

## Findings carried over from v2

Applied, and still the basis for the numbers above:

1. Quality filtering beats ratio optimization at this scale (DCLM).
2. Two or three epochs over a filtered subset beat one epoch of the superset
   (arXiv 2503.07879).
3. No curriculum is needed below 1B tokens; anneal on quality at the end instead.
4. Long chain-of-thought belongs in SFT, not pretraining (arXiv 2506.07712).
5. Cap any single dataset at four or five epochs (SmolLM2 degradation threshold).
