---
question: Stage-3 near-dedup post-pass: how to run 44's calibrated gate over stage-2 corpora, parallel, byte-identical, with a fact per domain
status: recorded
source: fb ruling 2026-08-31; docs/lessons/near_dedup_gate.md (44, main); build_corpus.py (parallel exact pass, bytes-identity test datagen/test_parallel_exact_identity.py)
---

# Near-dedup post-pass over stage-2 corpora

The stage-2 stamps are `near_dedup:false` by the 2026-08-31 dedup ruling. The in-stream MinHash was the wrong instrument (44: the literature raw-5-gram Jaccard 0.75-0.8 caught 0/8 of the hand-read template clusters; max raw J 0.181). The corrected gate is a separate pass that runs AFTER stage 2 lands on the un-near-deduped stamps. This is its design: parallel, byte-identical, domain-specific normalisation, removed fraction as a fact.

## Gate (44, near_dedup_gate.md)

- Normalise: strip comments and string literals, map numbers to `#`, map identifiers to a placeholder (70-word keyword stoplist kept), collapse whitespace.
- Shingles: word 3-grams.
- Similarity: Jaccard >= 0.5.
- Clustering: union-find; keep one doc per cluster (first in stream order).
- Calibration: on the 50-doc hand-read sample, 5/6-8 flagged caught, 0 false positives among the other 42; expected removal ~10% (Wilson 95% CI 3.4-21%).
- Contamination standard: normalise literals before matching; verbatim-only matching misses generator-template families.

## Two-stage scheme (the parallel shape, fb + 44)

Reuse the `_parallel_exact_pass` skeleton from build_corpus.py (byte-identity test at datagen/test_parallel_exact_identity.py). Exact-dedup has no Jaccard; near-dedup does. The shared structure:

1. **Signatures, parallel** (map over the domain shards, `a.workers`): for each doc, apply the domain normaliser, shingle into word 3-grams, emit a compact signature. Emit `(doc_ordinal, nearest_cluster_key)` where the key is the set of 3-gram shingles (hashed band). No cross-doc state: each worker signs its slice independently.
2. **Adjudication, serial over compact signatures**: load the per-doc signature list (ordinals + shingle sets). Build union-find over docs whose shingle-sets have Jaccard >= 0.5 (only near-sets need the pairwise check; a locality-sensitive band pass finds candidates in O(size) rather than pair-all). Keep one doc per cluster — the earliest `doc_ordinal` — so the survivor set is deterministic and matches a serial execution.
3. **Rewrite, parallel**: re-read the shards in the same global order, keep only survivor ordinals, write merged `{domain}_*.jsonl` through ShardWriter, remove input shards, stamp with the removed fraction.

Byte-identity: the serial reference is the same three steps run single-worker. Parallel output must equal serial byte-for-byte. Verify on two shards (same input, serial and parallel), like the exact-identity test.

**Correctness rule for the near-dedup survivor set**: a doc survives iff it is the lowest-`doc_ordinal` member of its near-cluster. Deterministic given a fixed serial ordering (shard order, then line). This matches the serial keeper ("first stream occurrence"), so byte-identity holds by construction.

## Domain-specific normalisation

The gate's normalisation is CODE-specific (strips comments/strings/identifiers). It does not generalise to every stage-2 domain. Each gets its own, with a hand-read false-positive margin check before its removed fraction means anything (fb condition 1; the hand-read check on code_rp1t was 50 docs stratified by language).

| domain | normaliser | hand-read margin |
|---|---|---|
| code_rp1t | the code gate as calibrated: strip comments+strings, numbers->#, identifiers->placeholder, 70-word keyword stoplist | done (t24, 0 FP on 42 non-template docs) |
| en_c4 | English prose: no comments/identifiers. Normalise: numbers->#, URLs/emails/hex->placeholder, collapse whitespace. Remove repeated boilerplate lines (license headers). | pending (50-doc read, target 0 FP, margin below nearest false pair) |
| math_owm | math/LaTeX: map inline/diplays to a token, numbers->#, collapse whitespace; fragment-level (a Q+A are near-dups if skeleton + answer skeleton match). | pending (50-doc read) |

The margin check: 50-doc hand-read per domain, compute the nearest false-positive pair's normalised Jaccard; the gate's 0.5 must sit above it (the code gate's margin was 0.511 vs 0.462). Land the margin as a fact per domain.

## Output: the removed fraction as a fact

The post-pass writes, per domain, a fact: removed fraction (docs removed / docs in), with the domain, normaliser, and the measurement config (skinger, Jaccard threshold, Wilson CI from the calibration). Land it in facts/ (a near_dedup fact file). If removed fraction > 21% (Wilson upper on the 50-doc calibration), ping fb before any readout is interpreted (fb condition 2).

## What the pass does NOT do, and where the byte-identity test sits

- Not the stage-2 launch path: it runs after the stamps, so a failure cannot block a run.
- Not exact-dedup replacement: exact-substring dedup runs alongside (Lee ExactSubstr; up to 19% tokens on some corpora), the near pass only drops near-dups.
- The acceptance is the same one 44 gave stage 1: byte-identity on two shards, recorded in the stamp.

## Implementation order

1. Salalise the shared near-dedup core (signatures -> adjudication -> rewrite) in build_corpus as `_near_dedup_postpass`.
2. Per-domain normaliser functions (code, en_c4, math_owm) + the hand-read margin checks.
3. Byte-identity test (serial vs parallel on two shards) for each domain.
4. Run on each stage-2 domain, land the removed-fraction fact, ping fb above 21%.