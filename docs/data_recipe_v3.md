> **SUPERSEDED (2026-08-27).** This 4B-token target was never built as written.
> The live mix is **11.5B tokens** (`data/mix.json`: web 8.31B / en 0.20B / math
> 0.16B / code 0.08B / chat 0.04B; main phase + anneal, 1 epoch forced by mix
> mode). The KenLM-perplexity / OpenWebMath / SkyPile / The-Stack pipeline below
> was **never implemented** in `datagen/build_corpus.py` — the filters there are
> the actual product. For the current pipeline see `data/mix.json` and `CLAUDE.md`
> ("Corpus", "Mix"). Kept only for the design rationale below.

# Pretraining Data Recipe v3 — 4B Tokens (Chinchilla Optimal for 200M — superseded)

## Target
- **4B tokens** (20 tok/param × 200M params = Chinchilla optimal)
- Current: ~1.27B tokens → need +2.7B

## Data Sources & Target Mix

| Source | Target Tokens | Share | Method |
|--------|--------------|-------|--------|
| SkyPile-150B (Chinese web) | 1.6B | 40% | Quality filter + dedup |
| Cosmopedia Chinese (textbook) | 600M | 15% | Already have, reuse |
| Cosmopedia English (all subsets) | 500M | 12.5% | Add remaining subsets |
| The Stack (code, Python+JS+Go) | 500M | 12.5% | Official dedup + license filter |
| minimind (conversational) | 400M | 10% | Already have, reuse |
| Existing corpus | 200M | 5% | Already have, reuse |
| Math (OpenWebMath) | 200M | 5% | Quality filter |
| **Total** | **4B** | | |

## Quality Filtering Pipeline

### SkyPile (Chinese web)
1. Language detection: keep only Chinese (langdetect, zh probability > 0.8)
2. Length filter: 200-50,000 chars
3. Deduplication: MinHash (n-gram=5, threshold 0.8)
3. Perplexity filter: KenLM 5-gram on Chinese Wikipedia, keep bottom 80% perplexity
4. Remove template/boilerplate: regex for headers, nav bars, cookie banners
5. Deduplicate against existing corpus

### The Stack (code)
1. Use official deduplicated subset
2. Filter by language: Python, JavaScript, Go, Rust
3. Length filter: 100-100K chars
4. Remove auto-generated files (minified, generated)
5. License: only MIT/Apware/BSD

### Cosmopedia
- Already filtered (textbook/wikihow format)
- Add remaining subsets: stories, web_samples

### Math (OpenWebMath)
- Already filtered by source
- Length filter: 200-20K chars

## Deduplication
- MinHash LSH (5-gram, 128 permutations, Jaccard threshold 0.8)
- Cross-source dedup: check new data against existing corpus
- Within-source dedup: remove near-duplicates

## Mixing Strategy
1. Tokenize all sources separately
2. Sample by target ratio with random shuffle
3. 2 epochs (SmolLM2 evidence: 2-3 epochs optimal)
4. Quality annealing: reserve cleanest 5% for final 10% of training

## Estimated Compute
- 4B tokens / 800K tok/s = 5,000s ≈ 1.4 hours (8×H20, optimized)
- Data prep: ~2-4 hours (download + filter + dedup)
