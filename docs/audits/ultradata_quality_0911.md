---
question: "What fraction of openbmb/UltraData-Code L2/L3 python is top-tier pretraining material, and does a mechanical filter reach it?"
status: measured
source: "3b-21 stratified hand-read, 2026-09-11, parquet shard 1 per level (L2 612k rows, L3 141k)"
---

# UltraData-Code L2/L3 quality audit (3b, 2026-09-11)

V4.1 trains on openbmb/UltraData-Code python (user order 2026-09-10, teacher synthesis
stopped). The user keeps only the **top tier**. This audit measures what that is and tests
whether a cheap mechanical rule reaches it.

## Method

- **Stratified sample, shard 1 per level**: L2 n=420 (70 per `category`, 6 categories);
  L3 n=400 (100 per `full_content_format`, 4 formats). Seed 20260910.
- **Hand read by 48 independent LLM readers**, each doc KEEP (phi-1-tier instructional
  value) or CUT with a reason code; a 10% blind double-read measured agreement **88.9%**.
- **Objective anchors, computed not judged**: L2 `ast.parse`; L3 every solution executed
  against its bundled test in one sandbox (the production predicate).
- A first L3 pass capped documents at 6000 chars and mislabeled 89 long docs
  `broken_truncated`; a full-content re-judge corrected this. The 39.2% below is the
  corrected number (28.5% was the artifact).

## L2 (natural GitHub python)

**Top-tier yield 6.4%** (26/409, 95% CI 4.4–9.2%).

| category | KEEP/n | rate | shard-1 weight |
|---|---|---|---|
| DATA | 14/68 | 20.6% | 9.2% |
| ALGO | 7/70 | 10.0% | 25.1% |
| TOOL | 3/68 | 4.4% | 30.9% |
| WEB | 2/66 | 3.0% | 13.7% |
| CONFIG | 0/68 | 0% | 7.3% |
| TEST | 0/69 | 0% | 13.8% |

- **`quality_score` is not a filter**: KEEP mean 5.49 vs CUT 5.44; rates flat 4–9% over
  score bands; TEST has the highest mean (6.23) with 0 KEEP. Same lesson as
  `datagen/audit_source_score.py` — never cut on an unmeasured published score.
- Cuts are vendored/generated (pip/_vendor, bundled CPython stdlib/tests), CRUD/glue
  boilerplate, trivial files, and non-algorithmic test fixtures.
- **Mechanical rule (fb ruling)**: drop `category ∈ {CONFIG,TEST}` — pooled 0/137 KEEP
  (upper 95% CI 2.7%), removes 21.1% of rows at ~zero top-tier loss. This is a pre-filter,
  not a per-doc guarantee; even DATA/ALGO are mostly CUT.
- ast.parse 96.9% (13 syntax errors, 6 in TOOL).

## L3 (synthetic task/analysis/solution/test)

**Top-tier yield 39.2%** (153/390, CI 34.5–44.2%), flat across the four ordering formats
(34–42%).

The production predicate **"solution passes its own bundled test"** yields **45.0%**
(180/400) but is not, on its own, a quality filter. Joined on the same 390 rows:

| exec predicate \ hand | KEEP | CUT |
|---|---|---|
| **pass** | 75 (TP) | 100 (FP) |
| **fail/timeout** | 78 (FN) | 137 (TN) |

**Precision 42.9%** (CI 35.8–50.3), **recall 49.0%** (CI 41.2–56.9).

- **False positives (100)**: 98 are *trivial passing exercises* — print loops, single
  builtin (`abs`/`len`/`str`), hello-world — whose tests pass trivially. Executability
  does not imply algorithmic content.
- **False negatives (78)**: 52 are substantive real algorithms and ~13 are faithful
  chardet/prober/textbook structures (EUC-JP/Hebrew probers, N-queens, lazy segment tree,
  spectral modularity) where the **bundled test contradicts the stated spec** while the
  solution is correct. A same-spec execution run can never rescue good-code/bad-test.
- Good-code/wrong-test is ~13/390 = 3.3%.

### Implication for the filter

Exec-pass is a sound **necessary / de-risking stage** (it removes 137 genuinely broken
rows cheaply) but is insufficient for "top tier":

- Stage 1 (landed, PR #231): decontam + exact-dedup + exec-pass; L2 also drops
  CONFIG/TEST.
- Stage 2 (fb ruling 2026-09-11, `datagen/ud_solution_exec.nontrivial` / `keep_l3`):
  a static AST non-triviality floor calibrated on these 390 rows —
  **node count ≥ 90 OR (≥3 branch/loop control points with ≥2 loops)**. The OR keeps
  compact-but-real algorithms a size cutoff loses (calibration lost only 3/75
  substantive pass-set docs).

| rule | precision | recall | yield |
|---|---|---|---|
| exec-pass only | 42.9% | 49.0% | 44.9% |
| **exec AND nontrivial (keep_l3)** | **67.9%** (CI 59–77) | **47.1%** (CI 39–55) | **27.2%** (CI 22.6–31.5) |

The floor lifts precision +25 points at −2 recall, removing 66 trivial-pass rows.
Recall caps near 47% regardless: the good-code/wrong-test false negatives cannot be
recovered by any content floor, and that ~3.3% loss is accepted for the gate run
(fb ruling; no spec-aware classifier is worth it here).
- The good-code/wrong-test FN class needs either a spec-aware judge or accepting the
  ~3.3% loss; exec alone cannot recover it.

## Boundaries

One parquet shard per level (L2 1/119, L3 1/147), within-shard stratification; full-set
re-measure after conversion bounds the rates. Python only; non-python levels unmeasured.
Hand labels carry an 11% double-read disagreement, concentrated at the good-code/wrong-test
rescue boundary. No downstream model-quality claim follows from execution alone.
