# Synthetic self-repetition metric (3b, 2026-09-09)

## Problem

The ~1B-token synthetic corpus (0.8B textbooks + 0.18B exercises,
p1_data_recipe.md:68-69; an earlier 20B sizing was retracted at :53) will
self-repeat. The repetition is not file-level (exact dedup catches that) but
**content-mode level**: same explanation skeleton, same worked-example
pattern, same exercise template with swapped names and numbers. 4c's
question: how much of the synthetic set is "reskinned same content"?

## Metric: three numbers, one index

All three are measured on the synthetic corpus AND on a real-code baseline
(the 6B filtered code) and a web baseline (web_hq sample). The index is the
ratio — absolute numbers mean nothing without the baseline.

### R1. Distinct-8gram rate (vocabulary collapse)

Streaming. Hash every 8-gram of the content field; keep hashes divisible by 2^p
(the "kth hash" estimator — bounded memory, unbiased for the distinct count).
distinct_rate = distinct_8grams / total_8grams.

Collapsed synthetic text reuses the same phrasing → low distinct rate.
phi-1-quality text should approach the web baseline.

### R2. Near-duplicate rate at low threshold (reskin band)

MinHash signature per doc on a 200K-doc sample; Jaccard estimate via signature
equality (reuses near_dedup_scale.py machinery). Report the pair-rate in the
**reskin band J in [0.3, 0.7)** — exact dedup removes J>=0.85; the [0.3, 0.7)
band is what "same template, different skin" looks like. Baseline: same-band
rate on real code. The synthetic rate should not exceed the baseline's.

### R3. Cluster mass (semantic concentration)

TF-IDF (char 2-3gram) + k-means (k=200) on a 100K-doc sample. Report the mass
of the top-10 clusters. Reskinned content concentrates in few clusters.
Baseline: top-10 mass on the same-size real-code sample.

### Index

reskin_index = (R1_base / R1_synth)          # >1 means synth repeats more
             + (R2_synth / R2_base)          # >1 means more reskin-band pairs
             + (R3_synth / R3_base)          # >1 means more concentrated

All three components reported separately with their baselines; the index is a
summary, not a substitute. Pre-registered verdict rule (proposal, needs fb/4c sign-off):
**reject the generation run if reskin_index > 1.5 on any component**, i.e. any
single dimension is 50% worse than the real-code baseline.

## What this does NOT measure

- Factual correctness of the synthetic content (separate QA gate).
- Diversity ACROSS topics — that is the 20K seed table's job (measured there).
- Within-topic pedagogical variety (needs human/LLM rubric, future work).

## Implementation

`datagen/synth_selfrep.py` (this directory's sibling): streaming R1 over any
jsonl corpus; R2/R3 on a sample. Run after the first 1B-token generation
milestone, not at the end — regeneration is days, not hours.
