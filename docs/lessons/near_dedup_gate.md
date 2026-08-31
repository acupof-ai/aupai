---
question: stage-2 重盖章的近重复硬门限怎么定——shingle 大小、Jaccard 阈值、期望移除比例?
status: recorded
source: fb tasking 2026-08-31 (44-4); arXiv:2107.06499, 2406.17557, 2402.00159, 2303.09540, 2308.12284; data/corpus/sample/code_rp1t_handread50.jsonl (t24)
---

# Near-duplicate hard gate for the stage-2 re-stamp

**The literature MinHash standard (word 5-grams, Jaccard 0.75-0.8) is the wrong gate for this corpus. On the 50 hand-read docs it passes every template-family near-duplicate (max raw Jaccard 0.18). The calibrated gate is: normalize code (strip comments, strings, numbers, identifiers), 3-word shingles, Jaccard >= 0.5. It catches 5 of the 6-8 hand-read cluster docs with 0 false positives in the sample, implying a 10% removed fraction (Wilson 95% CI 3.4-21%).**

## 1. The literature

| source | scale | shingle | threshold | removed | downstream |
|---|---|---|---|---|---|
| Lee et al. 2021 (ACL 2022) | C4 360M docs; RealNews 31M; Wiki-40B 2.9M; LM1B 30M sent | word 5-gram | Jaccard 0.8, edit-sim 0.8 confirmation; 9000 perms (b20 r450) | 3.04% C4; 13.63% RealNews; 0.39% Wiki-40B; 4.86% LM1B | memorization ~10x lower; perplexity equal or better (up to -10%) |
| FineWeb (Penedo 2024) | 15T tokens, 96 CC snapshots | word 5-gram | Jaccard 0.75; 112 perms (14x8) | 44% per-snapshot (36T to 20T tokens); global iterative up to 90%, rejected | per-snapshot matched RefinedWeb; global gave no improvement |
| Dolma (Soldaini 2024) | 3T tokens, 4B+ docs | exact URL/doc/paragraph (Bloom); code (the Stack) MinHash, settings unreported | n/a | URL 53.2%; doc 14.9%; paragraph 18.7% | positive compounding, no isolated ablation |
| SemDeDup (Abbas 2023) | LAION-440M; C4 | embedding cosine (OPT/CLIP), not MinHash | eps 0.03 tight | up to 37% no perf drop; 50% under 0.5% | 2x faster at equal perf; C4: full-dataset perf with 10-15% less compute |
| D4 (Tirumala 2023) | CC-dedup 600M docs | MinHash preprocessing; shingle and threshold unreported (Spark defaults, 20 perms) | unreported | not stated; cites 3.9% fixed-ratio on C4 | +2% avg 0-shot; 20% efficiency at 6.7B |

## 2. Adoption rule outcome

The rule: adopt if two independent sources at <=1B scale agree within 0.1 Jaccard; otherwise state the disagreement and take the stricter.

Only one <=1B-scale source reports a Jaccard threshold: Lee, 0.8. FineWeb's 0.75 is a 15T-scale setting. The condition is not met, and the stricter of the two is 0.75. The calibration in section 3 then shows the premise fails: neither threshold catches the observed class, so no literature threshold is adopted.

## 3. Calibration on the 50 hand-read docs

The t24 hand-read found 2 template clusters touching 6/50 docs (12%): Yii CRUD views and Java POJOs. Post-hoc reconstruction identifies 8 candidate cluster docs: {3,24,30,38} Yii, {5,7,19,35} POJO. The sampler's language detector mislabeled the PHP docs as rust/python because its head-600-char regex matched `use `.

| measure | best 0-FP threshold | cluster docs caught | false pairs |
|---|---|---|---|
| raw word 5-gram Jaccard | 0.10 | 3 {3,30,38} | 0 |
| normalized 3-gram Jaccard | 0.50 | 5 {5,7,19,30,38} | 0 |
| normalized 3-gram containment | 0.77 | 5 {5,7,19,30,38} | 0 |
| normalized 5-gram Jaccard | 0.40 | 2 {7,19} | 0 |

At the literature threshold (raw 5-gram, 0.75-0.8): 0 of 8 caught. The Yii cluster tops out at raw J 0.181, normalized 0.511, containment 0.787; the POJO cluster at raw under 0.01, normalized 0.717, containment 0.846. Whole-document Jaccard dilutes the shared skeleton with doc-specific content; smaller shingles after normalization surface it.

Chosen setting: **normalized 3-word shingles, Jaccard >= 0.5, union-find clusters, keep one doc per cluster.** Margin to the nearest false pair: 0.511 vs 0.462 (a python-rust boilerplate pair). Containment's margin is thinner (0.778 vs 0.739) with cross-language boilerplate as the false mode.

Normalization: strip comments and string literals, map numbers to #, map identifiers to a placeholder (70-word keyword stoplist kept), collapse whitespace. Same family as the harness `eval_sft_template_contamination` check.

## 4. Expected removed fraction

On the hand-read sample: 5/50 = 10% (Wilson 95% CI 3.4-21%), all from the two template clusters, 0 false positives among the other 42. The sample is stratified by language (cap 8 per language, PHP 32%), so the corpus-wide fraction is not directly 10%; 3b measures it at re-stamp on the full corpus. The hand-read's 12% (6/50) is the upper reference; the gate catches 5 of the 6-8 flagged docs and misses the most divergent cluster members (docs 24, 35).

## 5. What this gate does not catch

- Semantic near-duplicates without lexical overlap: SemDeDup territory (embedding cosine, eps 0.03, 37% removable at no perf loss on LAION). A future layer if the lexical gate's residual is material.
- Cross-language boilerplate below the threshold: the python-rust pair at normalized J 0.462 is the warning; 0.5 stays clear of it.
- Exact duplicates: an exact-substring pass (Lee ExactSubstr style) runs alongside; the literature removes up to 19% of tokens with it.

## 6. Contamination-scan standard

The code-500 v1 failure was a template family shared between SFT and eval, invisible to verbatim matching. The standard for contamination scans: normalize literals (numbers, strings, ordinals) before matching; use either normalized-prefix containment (the harness `eval_sft_template_contamination` check, 200-char prefix) or n-gram overlap with n >= 3 on normalized text. Verbatim-only matching is insufficient for generator-template contamination. Lee's ExactSubstr remains the right layer for verbatim.

## Sources

- Lee et al. 2021, arXiv:2107.06499 (ACL 2022)
- Penedo et al. 2024, FineWeb, arXiv:2406.17557
- Soldaini et al. 2024, Dolma, arXiv:2402.00159 (ACL 2024)
- Abbas et al. 2023, SemDeDup, arXiv:2303.09540
- Tirumala et al. 2023, D4, arXiv:2308.12284
