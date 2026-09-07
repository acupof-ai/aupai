---
question: Is the 33.02% rp1t deletion in code_dedup08 duplicate removal or supply loss?
status: open
source: datagen/code_dedup_handread.py; facts/data_quality.json#dq.handread_excerpt_sufficient
---

# code_dedup08 rp1t-vs-starcoder hand-read criterion (0908)

Written 2026-09-08 **before reading any cluster** (aupai-6e: criterion before reading). #70's
question is whether `code_dedup08` deleted 33.02% of `code_py_rp1t` because those documents were
already in `code_py_starcoder`, or because a threshold at 0.8 Jaccard over char 5-grams also
catches different programs that share boilerplate. The first is duplicate removal and the supply
number stands; the second is supply loss and the deletion rate is an overcount.

The number under test is not derived from this read. The read decides whether it means what its
name says.

## What is being judged

One row per cluster. `rep_text_excerpt` is the kept `code_py_starcoder` representative (minimum
ordinal, which is what the build kept); `member_text_excerpt` is the dropped `code_py_rp1t`
member. `same_file` is blank in the sheet by construction and is the only field a reader writes.

## The criterion, verbatim from aupai-6e's 2026-09-03 ruling

> near-duplicate = same code modulo whitespace, identifiers, or comments; different program = not.

Three verdicts, and the third is not a hedge:

| verdict | meaning |
|---|---|
| `dup` | Same program. Differences confined to whitespace, identifier names, comments, import order, or a license header. The deletion was correct. |
| `diff` | Different program. Shared text is boilerplate — license, imports, generated scaffold, a common framework skeleton — and the executable logic differs. The deletion removed supply. |
| `undecidable` | The excerpts do not contain enough of either program to apply the criterion. |

`undecidable` exists because the sheet shows 400 characters, and a verdict invented past the
evidence is worse than a gap that is counted. It is reported as its own number, never folded into
either side and never dropped from the denominator.

## What the instrument can resolve, measured before the read

`facts/data_quality.json#dq.handread_excerpt_sufficient`: 15 of 4200 different starcoder documents
share a whitespace-normalised 400-char prefix, **0.36%** (95% CI 0.20–0.59%). Median document is
1883 chars; 14.5% are under 400 and are shown whole. So a false `dup` from two different files
having identical openings is a sub-1% effect, well below the 80% decision threshold below.

The boundary is in that entry and repeated here because it bites this read specifically: the
measurement does not bound the opposite error, a `diff` verdict on one file whose two excerpts
differ inside the first 400 chars. A cluster whose excerpts differ only in a header is
`undecidable`, not `diff`.

## The decision rule, fixed before the read

Over the mixed-stratum clusters, let `p = dup / (dup + diff)`, with `undecidable` reported
separately.

- `p >= 0.80` — `code_dedup08` stands. The 33.02% is duplicate removal.
- `p < 0.80` — rerun the dedup with a domain-fair representative (prefer the rp1t member), because
  the deletion is then partly supply loss and the kept side was chosen by ordinal, not by merit.

80% is aupai-6e's threshold, not mine, and it is recorded here so the read cannot move it
afterwards. If `undecidable` exceeds 20% of the mixed stratum, `p` is not reportable at n=40 and
the answer is a larger excerpt, not a verdict.

## Sampling, and why this doc names it

The mixed stratum is drawn by `draw_mixed` (`datagen/code_dedup_handread.py`), which refuses when
its draw equals the pool's lowest-ordinal n. Until 2026-09-08 the draw was
`rng.shuffle(list(mixed.keys()))` — a no-op, because shuffle mutates in place and the list was
discarded on the same line, so the sample was the 40 lowest-ordinal clusters: a contiguous slice
of the first shards. Ordinal is corpus position, and position correlates with source file, so that
sample could not speak for the stratum. Fixed at `796fec85`.

Any sheet produced before that commit is discarded rather than read with a caveat. The 60
pure-starcoder clusters are blinding: they carry no rp1t member, so a reader who is inferring the
answer from the sheet's shape rather than the code will produce `dup` verdicts on rows where the
correct verdict is undefined.

## What this read cannot answer

- Whether 0.8 is the right threshold. This judges the clusters the threshold produced, not the
  threshold. A `p` near 1.0 says the deletions were duplicates; it does not say a higher threshold
  would have kept genuine supply.
- Anything about the starcoder side's own internal duplication.
- The rp1t documents deleted by the *exact* content-hash pass, which never entered a cluster.
