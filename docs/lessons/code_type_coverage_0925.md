---
question: With every code dataset measured, which HumanEval types are short of data and which are supplied but still failing?
status: measured
source: runs/code_type_coverage_0925.jsonl (8 datasets, one classifier); supersedes the coverage section of docs/lessons/humaneval_problem_types_0925.md
---

# Code-type coverage, all datasets

Follow-up to `docs/lessons/humaneval_problem_types_0925.md`, which measured five datasets and
recorded two limitations: the pretrain column was a single shard of one domain, and the RL
column covered only the finished part of the pool. Both are closed here — and closing them
**moves two verdicts**.

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

## 2. Two verdicts move

`docs/lessons/humaneval_problem_types_0925.md` §4a put four types in **缺数据 (short of
data)**. Two of them do not survive the fuller coverage.

### counting_histogram: short of data → **supplied by some sources, thin in SFT-A**

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

This also sharpens §5.1 of the type study, which said counting's 0% had two live
explanations (cannot-count vs under-supplied) and that the data did not separate them. The
data still does not separate them — but the "under-supplied" half is now weaker, because the
supply exists upstream. **The 0% is more likely a capability or prompting effect than a
corpus gap, and it is now the leading candidate for that reading.** Not yet a conclusion:
the RL pool is not SFT-A, and whether the model has seen this type is a question about the
pack, not about the corpus.

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
