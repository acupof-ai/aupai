# AST code dedup — build-vs-YAGNI decision memo (2026-09-16)

Read-only CPU measurement. No implementation. Question: how much duplication does an
AST-canonical key add over the existing path-3 lexical key, and at what false-merge risk?

## Baseline already normalises most surface edits

`near_dedup_postpass.normalise_code` strips comments and string literals, maps numbers to
`#`, maps every non-keyword identifier to `@`, collapses whitespace; then word-3gram
Jaccard ≥ 0.5. So identifier rename, literal/number change, whitespace and comments are
already covered. The only surface edits an AST key can additionally see are **top-level
definition reorder** and **file = subset of another's definitions**.

## Measurement (bounded sample, whole-doc python only)

4 strided shards × first 1,500 parse-OK docs/domain; AST key = set of top-level
def/class subtree hashes (docstrings dropped, constants/attribute names kept, no local
renaming), document Jaccard ≥ 0.8; compared against path-3 Jaccard ≥ 0.5.

| domain | docs with top-level defs | AST pairs | also caught by path3 | AST-only pairs | AST-only docs |
|---|---:|---:|---:|---:|---:|
| starcoder | 4,712 | 7 | 4 | 3 | 4 (0.085%) |
| rp1t | 5,013 | 8 | 6 | 2 | 4 (0.080%) |
| ultra_l2 | 4,946 | 11 | 8 | 3 | 6 (0.121%) |

Marginal extra coverage: **~0.08–0.12% of parse-OK documents** (~2–3 pairs per 5,000).

## Hand-read of all 8 AST-only pairs (precision)

6 of 8 are **false over-merges of shared boilerplate**: two unrelated `setup.py` packages
matched on an identical `readme()` helper; two unrelated FastAPI apps matched on a shared
`app = FastAPI()` / CORS scaffold; a Steam mod script matched two r/dailyprogrammer
solutions on one trivial shared def shape. 2 of 8 are real but **ubiquitous templates**, not
corpus-internal duplicates worth a deletion: a Holberton `inherits_from` exercise solution,
and a union-find template appearing once with English comments and once with Korean.

Estimated precision of the marginal clusters: **~25%, and the two true hits are canonical
algorithm templates that arguably should stay for diversity.**

## Decision: YAGNI

The added rate is an order of magnitude below the shipped autogen-banner rule (starcoder
2.55%), and most of what it catches it catches wrongly because the key hashes only
top-level def/class shape and so collapses on common `setup.py`/FastAPI/CRUD scaffolds.
Building a correct canonicaliser (binding-aware local renaming, boilerplate-subtree
down-weighting, subset-aware threshold, per-domain parse coverage, C-family parser) is a
large surface for ≤0.1% mostly-wrong removal. Path-3 lexical dedup plus the autogen banner
already cover the measured duplication. **Do not build.** Reopen only if a downstream PPL /
English-locked hand-read set shows a concrete loss from template or reorder duplicates.

Scope not measured: C/JS/Java/C++ (no parser on the pod), `code_keep_p1` (multilingual),
`l3_noexec` (prose-wrapped). The conclusion is for the whole-doc python domains only.
