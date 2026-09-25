---
question: With every code dataset measured, which HumanEval types are short of data and which are supplied but still failing?
status: measured
source: runs/code_type_coverage_0925.jsonl (8 datasets, one classifier); supersedes the coverage section of docs/lessons/humaneval_problem_types_0925.md
---

# Code-type coverage, all datasets

Follow-up to `docs/lessons/humaneval_problem_types_0925.md`, which measured five datasets and
recorded two limitations: the pretrain column was a single shard of one domain, and the RL
column covered only the finished part of the pool. Both are closed here. Every domain in the
gate mix (`data/mix_v41_gate.json`) now has a measured column, plus the full post-dedup RL
pool. **One verdict moved, and the exposure analysis below weakens the framework it sat in.**

Same classifier, same rule as before: **absolute rate per 1000 sampled rows**, not
share-of-classified (whose denominator is whatever the classifier could label, 55%..100%
depending on dataset, so it partly measures classifier drift). Sampling is a fixed stride,
reproducible from the file length.

## 1. The table

`*` = more than 25% below HumanEval's rate for that type.

| type | HE pass | HE | code_if | sc2 | apps | **RL (dedup)** | starcoder | ultra_l2 | **ultra_l3** |
|---|---|---|---|---|---|---|---|---|---|
| scenario_simulation | 0% | 61.0 | 8.5\* | 9.5\* | 32.5\* | 29.9\* | 17.5\* | 13.5\* | 2.0\* |
| counting_histogram | 0% | 54.9 | 24.0\* | 27.0\* | 44.5 | **70.5** | 32.5\* | 42.0 | **93.0** |
| arithmetic_basics | 13% | 91.5 | 35.5\* | 45.5\* | 104.0 | **302.0** | 57.5\* | 64.0\* | 72.5 |
| string_parse | 14% | 42.7 | 38.0 | 73.5 | 29.5\* | 6.0\* | 66.0 | 46.5 | 24.0\* |
| sorting_select | 15% | 146.3 | 36.0\* | 50.0\* | 106.0\* | **170.1** | 58.0\* | 77.0\* | 139.0 |
| float_geometry | 17% | 73.2 | 106.5 | 75.5 | 64.5 | 63.1 | 111.0 | 112.0 | 98.0 |
| number_theory | 20% | 109.8 | 14.5\* | 21.0\* | 74.0\* | **101.7** | 14.5\* | 14.5\* | 46.0\* |
| string_transform | 21% | 146.3 | 71.0\* | 191.0 | 192.0 | 99.2\* | 85.5\* | 116.5 | 54.5\* |
| dynamic_programming | 21% | 48.8 | 14.5\* | 8.5\* | 38.5 | 14.2\* | 18.0\* | 13.5\* | 8.5\* |
| base_and_bits | 25% | 24.4 | 10.5\* | 32.5 | 31.5 | 27.8 | 20.5 | 18.0\* | 37.0 |
| collection_transform | 27% | 164.6 | 191.5 | 366.5 | 197.5 | 108.0\* | 170.0 | 209.5 | 413.5 |
| bracket_nesting | 28% | 36.6 | 1.0\* | 4.0\* | 15.0\* | 7.4\* | 1.0\* | 2.0\* | 10.0\* |
| **unclassified** | — | 0.0 | 448.5 | 95.5 | 70.5 | 0.0 | 348.0 | 271.0 | **2.0** |

Columns: `code_if` / `sc2` / `apps` are the three SFT-A code sources; `RL (dedup)` is all
9,535 survivors of the pool's global dedup; `starcoder` / `ultra_l2` / `ultra_l3` are three
pretrain code domains.

## 1b. The whole gate mix, with weights, and what exposure does and does not predict

The authoritative domain list is `data/mix_v41_gate.json` (1e: scan exactly this list). The
gate mix is **six domains**, and all six now have a measured column:

| domain | weight | note |
|---|---|---|
| `code_ultra_l2_dc` | 0.4720 | whole-repository code |
| `code_ultra_l3_noexec_dc` | 0.3146 | synthetic task-shaped text |
| `math_owm_stage2_dc` | 0.0800 | math word problems / exposition |
| `code_py_starcoder_dc` | 0.0734 | whole-repository code |
| `en_c4_stage2_dc` | 0.0450 | English web prose |
| `cot_dc` | 0.0150 | math reasoning chains with worked solutions |

Code is 0.860 of the mix across three domains. `code_keep_p1_dc` and `code_py_rp1t_dc` were
removed from the mix at the 0920 rebalance and are **not scanned**.

**Weighted exposure** — the sum of `weight x rate/1000` over the mix — is the closest thing
this data can give to "has the model seen this shape". It is an indicator, not a measurement:
every mix domain is measured in `files` mode, so a rate is *topics per document*, and a
document that mentions a count is not a problem that requires counting.

| type | HE pass | HE rate | weighted exposure | code-only | ratio | reading |
|---|---|---|---|---|---|---|
| scenario_simulation | 0% | 61.0 | 21.8 | 12.7 | **36%** | thin |
| counting_histogram | 0% | 54.9 | 54.7 | 54.3 | **100%** | **at parity, still 0%** |
| arithmetic_basics | 13% | 91.5 | 77.6 | 75.0 | 85% | ~parity |
| string_parse | 14% | 42.7 | 34.8 | 34.7 | 82% | ~parity |
| sorting_select | 15% | 146.3 | 93.8 | 90.4 | 64% | thin |
| float_geometry | 17% | 73.2 | 119.2 | 113.4 | **163%** | supplied, still 17% |
| number_theory | 20% | 109.8 | 36.6 | 34.8 | 33% | thin |
| string_transform | 21% | 146.3 | 79.8 | 79.2 | 55% | thin |
| dynamic_programming | 21% | 48.8 | 13.9 | 11.9 | **28%** | thin |
| base_and_bits | 25% | 24.4 | 24.3 | 23.5 | 100% | at parity |
| collection_transform | 27% | 164.6 | 253.3 | 251.9 | **154%** | supplied |
| bracket_nesting | 28% | 36.6 | 4.9 | 4.7 | **13%** | **thinnest exposure, best pass rate** |

**Exposure does not explain the pass rates, and the table is the evidence.** Read the two
ends:

- The **thinnest-exposure type is the best-passing one.** `bracket_nesting` sits at 13% of
  HumanEval's rate and passes 28% — the highest of the twelve. It is a two-line depth counter
  with an obvious invariant.
- The type with the **highest exposure fails below the median.** `float_geometry` at 163% of
  HumanEval's rate passes 17%.
- Of the two types that never pass, one is **at parity** (`counting_histogram`, 100%) and the
  other is thin (`scenario_simulation`, 36%). Exposure separates them; pass rate does not.

So "the model has not seen this shape" is **not** a sufficient explanation for the 0% rows,
and for `counting_histogram` it is now a poor one. What survives is the pairing: exposure
tells you about the supply of *topics*, and the pass rate is governed by something the
topical rate does not capture — most plausibly whether the model has had to *produce* the
shape under supervision. That is a claim about the SFT pack, not about the corpus, and this
table cannot test it. Stated as a question to answer rather than a conclusion.

**A caveat that has to travel with the exposure column:** `en_c4_stage2_dc` is English prose
(a BBQ-class advertisement is a real row), so its non-prose type counts are the classifier
firing on ordinary English rather than evidence of code tasks. It is 4.5% of the mix, and the
`code-only` column excludes it; the two columns differ by at most 9.1 per 1000 across the
twelve types (that maximum is counting_histogram, whose en_c4 rate is only 8.5), so the
qualitative reading above holds either way. Its `scenario_simulation` rate of 201.5 is the
single largest number in the raw table and is the most misleading one.

## 2. Two verdicts move

`docs/lessons/humaneval_problem_types_0925.md` §4a put four types in **缺数据 (short of
data)**. Two of them do not survive the fuller coverage.

### counting_histogram: short of data → supplied upstream → **and at parity by weight (third and final revision)**

| source | rate |
|---|---|
| HumanEval | 54.9 |
| **ultra_l3_noexec** | **93.0** |
| **RL pool** | **70.5** |
| apps | 44.5 |
| ultra_l2 | 42.0 |
| starcoder | 32.5 |
| sc2 | 27.0 |
| code_if | 24.0 |

Two independent sources exceed HumanEval's rate, one of them by 1.7x. So this type is
**abundant in the corpus and under-represented in SFT-A** — a sampling decision in the mix,
not a corpus gap. The earlier verdict was reached on the three SFT-A sources plus a partial
RL scan; both of the sources that clear it were missing from that picture.

**And with the mix weights applied it goes further than that** (§1b): counting's weighted
exposure is 54.7 against HumanEval's own 54.9, i.e. **100% — at parity**, not merely
"present upstream". The two revisions should be read in order: the raw per-source table made
it look absent; the weights show it is supplied at exactly the rate HumanEval tests it. So
the under-supplied half of the original explanation does not survive, and **§5.1 of the type
study is superseded: counting's 0% is not explained by the corpus.**

This is the clearest single entry in the table for the report's overall reading: a type at
parity exposure and 0% pass rate. What remains is a question about the pack — whether the
model had to *produce* counting tasks under supervision — which this data cannot test.

### sorting_select: short of data → **SFT-A under-samples it**

| source | rate |
|---|---|
| HumanEval | 146.3 |
| **RL pool** | **170.1** |
| ultra_l3 | 139.0 |
| apps | 106.0 |
| ultra_l2 | 77.0 |
| starcoder | 58.0 |
| sc2 | 50.0 |
| code_if | 36.0 |

The RL pool and ultra_l3 both exceed HumanEval, and the three SFT-A sources run 36–106, i.e.
the SFT-A side is 1.4–4x thinner than every other source. This was already flagged as
"mixed" in §4b; it now resolves to under-sampling.

### The two that stay

| type | best non-SFT-A source | HumanEval | verdict |
|---|---|---|---|
| `scenario_simulation` | 29.9 (RL) | 61.0 | **below HumanEval everywhere**, across 7 datasets. Stays 缺数据, and is the strongest such case. |
| `number_theory` | 101.7 (RL) | 109.8 | **at parity, below the 25% bar everywhere else** — 14.5–46.0 in every non-RL dataset. Keep as 缺数据, but note it is parity-with-RL, not absent. |

## 3. A new observation: the pretrain domains are not one population

`code_py_starcoder_dc` and `code_ultra_l2_dc` behave almost identically (unclassified 348 and
271 per 1000; both whole-repository code). `code_ultra_l3_noexec_dc` is a **different
population**: unclassified falls to **2.0 per 1000**, and its content is synthetic
task-shaped text — `"Write a Python function `draw_star_square(n)` that takes a positive
integer `n` and returns a list of strings…"`, BNF-parser specs, queue-simulation specs.
Roughly 99.8% of it is a function-writing prompt of the same shape as HumanEval.

The consequence for §5 of the type study: the claim that SFT-A differs from HumanEval
because code_if is real-world repository code holds, but it is **not true of all our code
data**. A pretrain/practice domain already exists in the HumanEval format, and it is by far
the least ambiguous to classify. Whether the model's post-training ever sees this shape is a
separate question about the mix, and this table does not answer it.

## 4. Method and its limits

- **Read-only throughout.** sc2 and apps were rebuilt into `/tmp` by importing the production
  renderer (`scripts/sfta_render_external.py`) as a module with `OUT_DIR` rebound; rebuilt
  sizes match production exactly (49,599 / 10,776), which is what makes them comparable to
  the pack. The RL pool was **filtered in memory** by the builder's own rule
  (`sha1(norm_text(prompt))`, survivor = max test count, `scripts/rl_code_pool.py`) because
  the builder consumes the files it reads; the reproduction is verified against the
  committed stats — 9,535 survivors and per-file counts (27 / 182 / 9,326) match exactly.
  Nothing was written under the pod repo.
- **RL pool = 9,535 after dedup, and the dedup is real**: 131 within-file and 378
  cross-file duplicates dropped. The earlier 27 + 539 partial scan is superseded.
- **Row ≠ problem**, unchanged: pretrain rows are whole FILES and one repository contributes
  many code_if pairs. Fine rankings are not supported; 5–8x supply differences are.
- **apps is measured on the original question text.** The renderer keeps only the starter
  signature, so measuring the rendered prompt reads 990/2000 unclassified — a signature-only
  string against HumanEval's docstring — which would report a measurement artifact as a data
  gap. On the question text it is 141/2000.
- **Single shard per pretrain domain** (1 of 283 / 1304 / 940). Shard-to-shard variation is
  not measured.
