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

## Why the deletion concentrated on seven shards

dedup08 is a union build: 283 starcoder shards + 15 `code_py_rp1t` shards.
The 15 are by construction a third copy of dd09/b2v2 content, so the 169,561
exact-overlap docs landed almost entirely on them: 138,564 (82%) on the 15
rp1t shards, 30,997 (18%) on the 283 starcoder shards — an ~84x per-shard
rate difference (9,238 vs 110 docs/shard, b0 pod count 2026-09-10). Seven
shards were left with 4-81 rows. This also explains the dd09<->b2v2
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

## Source of truth for the two clean corpora

Tracked authority: `docs/standards/p1_data_recipe.md` ("Which clean corpus p1
reads"). In short: `data/corpus_clean/<domain>/` serves non-classifier uses;
p1 training reads the classifier keep set minus the deleted doc ids; the
corpus swap (old dirs renamed aside, clean renamed into place, old kept)
happens after e1's scoring run, per 4c's plan (b).

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
