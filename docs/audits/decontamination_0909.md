---
question: What did the 2026-09-09/10 decontamination pass delete, what does it not cover, and which clean corpus does p1 read?
status: measured
source: datagen/scan_code_contamination.py, datagen/cleanup_pass.py, datagen/near_overlap.py (PR #172, #177); pod artifacts under data/decontam/
---

# Decontamination pass, 2026-09-09/10

Two passes, two manifests, executed as clean copies under
`data/corpus_clean/<domain>/` (pod; source shards untouched until the swap).

| pass | criterion | rows removed | manifest |
|---|---|---|---|
| decontamination | IDF-weighted bigram containment >= 0.5 vs HumanEval-164 + MBPP-974 holdouts (long holdouts only) | 11,745 (dd09 1,563, b2v2 802, dedup08 9,380) | `data/decontam/decontam_manifest_0909.jsonl` |
| exact cross-domain overlap | normalized SHA1 in dd09|b2v2's 5,537,807-hash set | 169,561, all in dedup08 | `data/decontam/overlap_manifest_0909.jsonl` |

The verdict scan of record is `data/scan_ledger.jsonl` (11,744 hit rows + 1
exact = 11,745). The full pass reproduced the LIST_ONLY manifest byte-for-byte
(determinism). 243 docs hit both channels; the overlap channel skips rows the
decontamination channel already marked.

## The exact channel is blind to near-duplicates

dd09 <-> b2v2 exact overlap is 0 while MinHash near-overlap participation
(est J>=0.5) is 23.27%/26.21% — same source (code_rp1t), different
versions/formatting, so bytes differ and content largely matches. "Exact
overlap handled" must not be read as "cross-domain duplication handled".

## Why the deletion concentrated on the 15 rp1t shards

dedup08 is a union build: 283 starcoder shards + 15 `code_py_rp1t` shards.
The 15 are by construction a third copy of dd09/b2v2 content, so the 169,561
exact-overlap docs landed almost entirely on them: 138,564 (82%) on the 15
rp1t shards, 30,997 (18%) on the 283 starcoder shards — an ~84x per-shard
rate difference (9,238 vs 110 docs/shard, b0 pod count 2026-09-10). Seven
shards ended with 4-81 rows in the clean copies (both channels; the
exact-overlap channel alone leaves six — the seventh was cut from 99 to 81
by the decontamination channel). This also explains the dd09<->b2v2
near-overlap: exact overlap between them is 0 (different versions/
formatting), while both overlap dedup08 exactly because dedup08 holds their
third copy.

## Near-overlap: re-measured 2026-09-10 (PR #177)

Participation (50K-doc sample per domain vs the full 11.78M-doc index,
96-perm MinHash char 5-gram, LSH 24x4, est J>=0.5):

| pair | participation |
|---|---|
| dd09 <-> b2v2 | 23.27% / 26.21% |
| dd09 <-> dedup08 | 6.68% / 5.84% |
| b2v2 <-> dedup08 | 4.02% / 3.14% |

Calibration: 100 pairs sampled from the flagged hits, exact char-5gram
Jaccard — MAE 0.003, bias +0.000, 100/100 exact J>=0.5. **The calibration
measures precision only** (the sample is drawn from flagged pairs), so recall
is unmeasured and the participation numbers are **lower bounds**, not point
estimates. All 5,972,131 unique hit pairs are persisted to
`data/decontam/near_overlap_hits_0909.jsonl`; the keep-set participation cut
is a separate join over that file by doc id, not a statistic the instrument
emits.

The 2026-09-09 run reported 22.9%/26.3%, 6.7%/5.8%, 4.2%/3.1% with a
loc/sig order misalignment (~85% of rows). The statistic is order-independent
(sampling is over sig row positions), and the fixed run confirms it.

**Near-dedup deletion is NOT approved, and the numbers above are likely not a
defect.** dd09 was deduped at MinHash-J 0.9 at build time; b2v2 was
cross-deduped against dd09 at the same 0.9 threshold (228,283 cross-domain
edges >= 0.9 removed then). A 0.9 threshold by construction keeps pairs with
0.5 <= J < 0.9, so the 23-26% participation is very likely the band that
threshold deliberately leaves, not leakage. Deleting at J>=0.5 would
re-choose the threshold — a different decision needing its own argument
(whether 0.5-0.9 near-duplicates harm training), which nobody has measured.

The calibrated 0.5 post-pass (`_near_dedup_postpass` in build_corpus.py,
128-perm, exact normalized word-3-gram Jaccard 0.5) exists as a separate
stage but never ran on these three domains: their stats stamps are
stage-2-shaped (no `near_dedup` key; the post-pass adds one when it runs), so
the [0.5, 0.9) band is intact by construction, not missed by a 0.5 pass. The post-pass also decides on
word-3-gram Jaccard, a different overlap notion than this instrument's
char 5-gram.

## Source of truth for the two clean corpora

Tracked authority: `docs/standards/p1_data_recipe.md` ("Which clean corpus p1
reads"). In short: `data/corpus_clean/<domain>/` serves non-classifier uses;
p1 training reads the classifier keep set minus the deleted doc ids. The
corpus swap (plan b) **executed 2026-09-10**: old dirs renamed aside as
`data/corpus/{code_rp1t_dd09,code_rp1t_b2v2_dd,code_dedup08}_predecontam`
(kept, not deleted), clean copies renamed into place.

## Shard-count sizing is not used

The seven near-empty dedup08 shards are harmless to the training path:
build_mix (`train.py:2639`) sizes by rows — `_domain_seqs` concatenates every
shard's rows and the want/cap arithmetic is row counts — never by shard
count. `datagen/count_cleaned_code.py` counts tokens per row.

## Manifest field naming

`decontam_hits_0909.jsonl` records the exact-overlap channel as
`kind="overlap"` (169,561 rows); `kind="exact"` is a different mechanism
(1 row). An audit grepping `kind="exact"` for the exact-overlap channel will
mismatch — read `kind="overlap"`.

## Post-scoring sequence (2026-09-10)

e1's scorer finished (cut -0.258355; keep doc 0.2599, byte 0.1536). The
keep-set join (`datagen/keep_set_join.py`, PR #192; result
`data/decontam/keep_set_join_0910.json`) over the 5,972,131 hit pairs:

| pair | both in keep | exactly one in | neither |
|---|---|---|---|
| dd09 <-> b2v2 | 7,198 | 18,041 | 4,643,788 |
| dd09 <-> dedup08 | 1,979 | 4,773 | 678,931 |
| b2v2 <-> dedup08 | 1,220 | 5,444 | 610,757 |

Both-ends-in-keep totals 10,397 pairs = 0.34% of the 3,060,432-doc keep set.
Deleting near-duplicates at J>=0.5 removes at most one doc per pair, so the
upper bound on what such a pass could remove from the keep set is 0.34% —
the no-deletion ruling (4c) now has a measured bound. (The participation
rates are lower bounds — recall unmeasured — so 0.34% is a lower bound too.)

Of the 169,561 exact-overlap deletions, **43,341 (25.56%) are in the keep
set** — far below the ~62,000 independence expectation, so overlap docs
score low. Both channels together: 48,283 of 178,941 dedup08 deletions in
the keep set (9,380 of the 11,745 decontamination-channel deletions are
dedup08's; 178,941 is all dedup08). Post-deletion doc keep, consistent
two-channel basis: (2,281,811 − 48,283) / (6,239,038 − 178,941) = **36.86%**,
above the pre-deletion 36.57%.

Token counts (frozen tokenizer, full counts, `datagen/count_domain_tokens.py`):

| what | tokens | docs |
|---|---|---|
| dedup08 clean copy (post-deletion, in place) | 8.509B | 6,060,097 |
| keep set, post-deletion, exact (2.8828B scored minus the 71.21M deleted-in-keep) | **2.8116B** | 3,011,677 |
| — dd09 | 0.5113B | 479,502 |
| — b2v2 | 0.3026B | 298,647 |
| — dedup08 | 1.9977B | 2,233,528 |

The 48,283 dedup08 deleted-in-keep docs average 1,283 tok/doc vs the keep
mean of 903 — the exact-overlap docs are longer than average, not shorter
(`data/decontam/deleted_in_keep_tokens_0910.json`). The 8.41B figure the
8.509B replaces was an extrapolation from 30.16 GB of shards, mislabeled
"measured" in two docs (corrected, PR #195); it was 4.0% below the implied
pre-deletion count.
