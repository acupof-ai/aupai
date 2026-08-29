---
question: Which methods and failure modes from pre-0830v1 work survive the reset?
status: recorded
source: extracted 2026-08-30 from docs/audits/audit_cosmopedia.md, docs/audits/audit_fineweb_edu_v2.md, docs/standards/synthetic_data_standard.md (deleted in the reset)
---

# Kept methods

## Table of contents

1. [Published quality scores](#1-published-quality-scores)
2. [Score bands as directories](#2-score-bands-as-directories)
3. [Cross-source quality comparison](#3-cross-source-quality-comparison)
4. [Two-stage quality filter](#4-two-stage-quality-filter)
5. [Sampling from blocked corpora](#5-sampling-from-blocked-corpora)
6. [Synthetic data: classification and anchoring](#6-synthetic-data-classification-and-anchoring)
7. [Synthetic overdose and mix caps](#7-synthetic-overdose-and-mix-caps)
8. [A/B and falsifying experiment design](#8-ab-and-falsifying-experiment-design)
9. [Contamination scanning](#9-contamination-scanning)
10. [Near-duplication](#10-near-duplication)
11. [Hand reading](#11-hand-reading)
12. [Filter portability and silent skips](#12-filter-portability-and-silent-skips)
13. [Traditional-to-Simplified conversion](#13-traditional-to-simplified-conversion)
14. [ASCII punctuation preprocessing](#14-ascii-punctuation-preprocessing)
15. [Metadata and dataset cards are claims](#15-metadata-and-dataset-cards-are-claims)
16. [Token counting](#16-token-counting)
17. [Using literature bounds](#17-using-literature-bounds)
18. [Failed and unmeasured cells](#18-failed-and-unmeasured-cells)

## 1. Published quality scores

- A published quality score is a claim, not a measurement.
- Run `datagen/audit_source_score.py` before using any source's score as a cut.
- Observed failure shape, repeated across three sources in the same family: near-zero rank correlation with our own judgement, non-monotonic across deciles, top decile among the worst. Treat that shape as the family default until a source passes the audit.
- A source's score can fail to separate anything inside the source too — if two populations we know differ score the same, no `score >= X` cut exists.
- A score that cannot separate formats or quality bands is still allowed to be monotonic with nothing — check separation, not just range.

## 2. Score bands as directories

- A score BAND that is a directory can work where the score NUMBER does not.
- Cut on the directory, never on the column inside it.
- Do not assume the two carry the same information: the band ordering can be strongly monotonic by our own judge while the continuous score inside a single band is worthless or inverted.
- Validate the band ordering with your own judge on a stratified sample before trusting it.

## 3. Cross-source quality comparison

- Cross-source quality comparisons need ONE judge on ONE rubric.
- A judge trained on web pages cannot rank textbook prose: it scores Wikipedia below a filtered web crawl and rates synthetic exposition as raw web. Use it within its training domain only.
- A continuous student score recovers ordering the binary teacher's hard labels cap — but only inside the domain the teacher labelled.
- When a judge is out of domain, say so and run the in-domain judge (a larger model on the same rubric, or a hand read) instead of quoting the out-of-domain number.

## 4. Two-stage quality filter

Architecture (FineWeb-Edu's):

1. A large teacher annotates a stratified sample of documents with binary labels.
2. A logistic head on the frozen small model's mean hidden state learns the labels.
3. The head scores every document; the continuous score is the cut.

Properties and rules:

- The student's continuous score can out-rank the teacher's own hard yes/no labels, because hard ties cap the teacher's AUC while the student recovers the ordering.
- Everything cheaper was measured first and failed before this was built: spam regex, character n-grams, structural features, a small open model. Character n-grams rank by TOPIC; the quality labels split on REGISTER. Do not retry topic features for a register judgement.
- The head is a within-source threshold and a cross-source trap — see section 3.
- A known out-of-domain population run as a CONTROL reproduces the artefact exactly; keep that control in every audit run.

## 5. Sampling from blocked corpora

- Sources are stored in blocks: rows are blocked by source inside a file, and a source can be a shard-level constant. Reading rows in order reads one source; a prefix read is a single-source read.
- Draw row groups at random, not a prefix. A brief that names two sources can be wrong — a full row-group census found eight.
- When source is a shard-level constant, a per-band source table has effective n = shard count, not row count. State that n.
- Sampling shards in sorted order until the quota is met reads a biased slice; stratify over shards.
- Sample the population that enters the mix: if only some shards were ingested, rates measured on them do not extend to the rest. Say which shards were not sampled.
- Never truncate documents in a quality sample; clipping changes per-character and per-word ratios.

## 6. Synthetic data: classification and anchoring

Two categories the literature measures, and the test that assigns a source:

| | anchored rephrasing | from-scratch generation |
|---|---|---|
| what the source contributes | the document | a topic |
| output checkable against it | yes | no |
| safe share of the mix at sub-1B | ~30% | under 5% |

- The subset test: numbers and named entities in the output must be a subset of the declared source's. It is the only hard test that separates interpretation from invention, and it is cheap to run.
- "Is it anchored" is not a document-level judgement call. Prose that FEELS generic classifies register, not provenance. Use a marker whose presence proves the pipeline — a seed reference in the text, a `metadata.raw` field, a number-subset test.
- A prompt leak (the seed named in the output) identifies the pipeline for ALL of it; it is not the anchored share. Story formats hide the leak because narrative has nowhere to put the reference.
- Measure anchoring as a RATE against a real-text control: checkable-fact markers per character (dates, percentages, number+unit, 《title》, any number) on candidate and control with the same regexes. A source can be seeded on a document and still discard most of its specifics — digits that survive can be mostly section numbering.
- A source can fit neither literature category. The classification is a measurement, not a lookup; a middle category needs its own weight-setting evidence.
- Added reasoning steps are not evidence of anchoring; the subset test still decides the weight.
- Name what the synthesis adds. Three strategies and three: format transformation, style modification, content restructuring. If you cannot name the addition, it is a paraphrase.
- One source, several styles: multi-strategy beats single-strategy. Measure the distribution over strategies and audiences, and n-gram diversity against the source corpus.
- Generator size saturates well before the largest available model; using the largest to generate pays a large multiple for a small gain. Do not default to the biggest generator.

## 7. Synthetic overdose and mix caps

- Synthetic overdose is undetectable by chance-level benchmarks: if every multiple-choice eval sits at the chance line, no benchmark you own can see an overdose. Cap synthetic share by external evidence, not by eval scores.
- External anchors that exist: another project's published mix ratio, and the falsifying experiment in section 8.
- In-kind training proves nothing: a model trained on a synthetic domain will of course score well on that domain. The verdict is held-out loss on the OTHER domains.
- Pool binding: freed mix weight cannot always go to the domain you prefer. If that domain's filtered pool is already repeated past one epoch, spending the budget means a larger filtered pull, not a larger multiplier on the existing pool.

## 8. A/B and falsifying experiment design

- Before running a two-arm test, name what ELSE changed with the variable, and ask whether it alone could produce the result you expect. Then either hold it fixed or add the arm that isolates it.
- The confound that actually happened: the low-synthetic arm gave the freed weight to the real-text domains, so it also trained on much more web/wiki — and the verdict was its held-out web/wiki loss. A model trained on more web scoring better on web holdout is a tautology, the mirror image of the in-kind-training trap.
- That design answers "at a fixed token budget, is replacing A with B worth more?" — not "is A harmful?". The harmful claim needs an equal-exposure design: arms with identical real-text token counts, differing only in an added slab of A.
- The falsifying measurement for a synthetic-share decision: two pretrains differing only in the synthetic share, compared on held-out web/wiki/math shards in neither mix. If the higher share wins there, the down-weight decision is wrong.
- The falsifying measurement for new tokens at scale: two pretrains at equal token budget, identical seed and card count, differing only in whether the new tokens come from the new source or from repeating the existing pool. New tokens that do not beat the repeat buy epochs, not information.
- Cost discipline: a scoring pass over fixed holdout shards with two existing checkpoints is hours, not a retrain. Run it before claiming a comparison does not exist.
- A null landing in a pre-registered cell does not certify that cell; a confounded null certifies nothing.

## 9. Contamination scanning

- Run `scripts/scan_contamination.py` on every new source, before it enters a mix. This rule recurs whenever skipped.
- A census (every row of a shard) beats a sample: a zero in a census gives a rule-of-three upper bound on the rate; a zero in a sample bounds only the sampled slice.
- Scan whole document plus the first lines; contamination can sit past the head.
- Seed pages are a blind spot: clean outputs do not certify clean seeds. If you do not hold the seeds, say the seed contamination is unmeasured.

## 10. Near-duplication

- MinHash LSH with a bottom-k sketch and a max-Jaccard per candidate is the standing method; exact dedup uses a whitespace-normalised hash.
- Assert a self-find control in-run: a document must find itself at J=1.000 and its own truncation at the expected lower J. A run that reported all-zero once was silently reading zero documents because the two corpora keyed on different column names (`content` vs `text`). Without the control, zero collisions and zero reads are indistinguishable.
- A sample-against-sample zero bounds the collision rate with the SAMPLE, not with the whole corpus. State the bound's denominator.
- Cross-shard duplication is the number that decides whether a token count is real, and a within-shard measurement does not bound it. List it as unmeasured when only some shards are held.
- A dedup stat printed by a corpus builder's dry run is meaningless when its MinHash set is seeded only with the sample itself.
- Two overlapping sources must not both be ingested; a weak overlap probe (a Poisson interval too wide to quote) is still strong enough to say that.

## 11. Hand reading

- Hand-read a stratified sample, stratified over the variables that matter (format × score band), not a random draw.
- Register-level garbage is invisible to topic models and to quality scores: gambling/adult SEO with brand names injected mid-sentence, synonym-substituted plagiarism, machine-translation garble, spliced unrelated fragments, boilerplate. Only a human read sees it.
- Confidently stated false claims in fluent, on-register prose are invisible to the filter chain and to both quality scores. The worst documents read were well-formed, correctly punctuated, and high-scoring.
- A generator does not remove the seed's commercial intent; it can give the advertising copy a narrator. Hand reading is where this shows up.
- Plan the n from the target half-width. A small read resolves order-of-magnitude gaps (one source's usable share is plainly not another's); it does not resolve adjacent rates. State what the n can and cannot separate.
- Classify each document on what it IS (expository with checkable content / generic exposition / fiction / false claim / label mismatch), and note that categories overlap.

## 12. Filter portability and silent skips

- Garbage patterns written against one domain's SEO false-positive on another: patterns for gambling/adult web content fire on ordinary sports and games prose. Check every pattern's hits on the new domain before enabling the chain.
- A filter loaded with `os.path.exists` silently becomes no-op when the file is absent: local and remote runs of the SAME command then return different pass rates and nothing raises. Either ship the filter file with the checkout or make its absence an error.
- A "100% pass through our filters" claim must name WHICH filter chain; a length/bytes/holdout-only chain and the full web chain are different instruments.
- Published-score decile has no monotone relation to a real filter's pass rate; do not use one as a proxy for the other.

## 13. Traditional-to-Simplified conversion

- Traditional-to-Simplified conversion changes chars/token materially; measure the corpus both ways before deciding.
- The opencc table is single-codepoint 1:1 only. Vocabulary-level differences (a word that is a different word, not a different character) are not covered and never were.
- Record the traditional-detection definition (threshold, table, codepoint set) with the rate. A rate measured under an unrecorded definition is not comparable to a later rate, however large the gap looks.
- Measure conversion as a preprocessing decision per source; some sources need no pass at all.

## 14. ASCII punctuation preprocessing

- A corpus can be mostly ASCII-punctuated while your vocabulary trained on mostly full-width; the sentence-boundary token is the casualty, not the content.
- Normalise punctuation with `str.translate` BEFORE the corpus builder, not after: several filters count punctuation, so order changes the pass rates.

## 15. Metadata and dataset cards are claims

- Every metadata field describes provenance as claimed, not content as found: `source`, `score`, `data_format`. Check conclusions against each other.
- A format label can be wrong about half the time; verify label-vs-content on a sample before weighting a domain by its labels.
- A brief's premise about a dataset (its sources, its category) is a claim too; the audit that corrected the source count also corrected the premise.
- A dataset card's deprecation/supersession note is actionable: audit the source that supersedes, and do not ingest both.

## 16. Token counting

- Count tokens with your own frozen tokenizer; a card's token figure is measured with theirs.
- A card figure that is a consistent multiple of yours across every row is a tokenizer difference, not a data discrepancy — the consistency is the diagnostic.
- Vocabulary identity: every checkpoint is scored with the vocabulary it was trained on; a size check passes while the scores are noise. Carry the fingerprint and refuse mismatches.

## 17. Using literature bounds

- Before applying a paper's bound, check its grid: the smallest param/token cell may still be larger than your run, making the bound an interpolation, not a measurement.
- Check the metric: a bound measured as perplexity over held-out domains is not a benchmark-accuracy bound, and no downstream point difference may be reported at any scale.
- A number that does not appear in the paper it is cited to is removed, not replaced. Two independent searches failing to find a cited paper leaves every claim resting on it unverified — label them so.
- An earlier draft's "fourth strategy" that returns zero hits in the cited full text stays out until a source is found.

## 18. Failed and unmeasured cells

- Report a failed measurement as failed. A title-matching subset test that pairs mostly spurious matches measures the matcher, not the data; publishing its overlap figure would be quoting the instrument.
- Every audit ends with "what could not be measured": the unheld seed corpora, the unsampled shards, the cross-shard number, the rate that needs a larger hand read. A rate measured on four shards of sixty-two is labelled as such.
- A permanent red CI is the same as no signal; a check that cannot fail is not a check. (Standing rule, restated because these audits' selftests depend on it.)
