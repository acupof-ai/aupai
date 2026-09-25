---
question: What problem types does HumanEval test, which types does the model fail, and is each failing type short of data or short of capability?
status: measured
source: scripts/problem_types.py (one classifier) applied to HumanEval, the four SFT-A sources, the RL pool and a pretrain code shard; per-type pass rates from runs/heval_fail_classes_30_32_34k.jsonl
---

# Problem types: where HumanEval fails, and whether the data reaches them

One classifier (`scripts/problem_types.py`) is used on every dataset below — HumanEval, each
SFT-A source, the RL pool, and a pretrain code shard. It is mechanical: each type is a weighted
set of literal signal patterns, the score is a match count, and a row is assigned its top type
(plus a second when the two are within 0.6 of each other). No model in the loop, so the same
text always gives the same type, and a reader can see which signal fired.

`python3 scripts/problem_types.py --selftest` asserts all 12 types are reachable and all carry
signals — a rubric whose rules never fire is a rubric nobody follows.

## 1. The taxonomy

Defined by **what the solution must do**, not by what the prompt talks about. A task about a
fruit basket solved by a subtraction is `arithmetic_basics`; a task about planets solved by an
index lookup is `scenario_simulation`.

| type | criterion |
|---|---|
| `bracket_nesting` | balance/depth of parentheses or brackets; the answer is a property of the nesting structure |
| `string_parse` | input is a formatted string; the work is extracting structured values or validating it |
| `dynamic_programming` | a sequence or optimum defined by a recurrence or overlapping subproblems (incl. path/subarray) |
| `number_theory` | divisibility, primality, factors, or digit properties of a number |
| `base_and_bits` | changing or reading a number's representation: base conversion, binary, roman, bit patterns |
| `float_geometry` | real-number arithmetic: rounding, areas, distances, rescaling, float decomposition |
| `sorting_select` | the answer is an ordering or order statistic: sort by key, k-th, min/max/median, top-k |
| `counting_histogram` | the answer is a count or frequency aggregate over a collection |
| `string_transform` | character-level work: case, reverse, replace, remove, encode/decode, shift |
| `collection_transform` | a collection (list/array/dict) in and out by a position-wise or membership rule |
| `scenario_simulation` | a described world or process that must be modelled |
| `arithmetic_basics` | short numeric/boolean arithmetic fitting none of the above |

## 2. Audit — 21 rows read twice, before any number was reported

Two independent deterministic samples (every 8th row starting at 0, and at 4), read by hand
against the criteria above.

**Sample 1: 16/21. Sample 2: 20/21.**

Five disagreements in sample 1, and **three of them were my error, not the classifier's**:
HumanEval/120 (top-k maximum), /136 (largest negative / smallest positive), /128 (scalar
arithmetic) all match their written criteria exactly — I had judged them by the prompt's
narrative, which is precisely what the criteria exist to override. I was applying the
taxonomy's stated rule to some rows and my own reading to others.

The two real defects were both signal coverage, and both were fixed:

| row | classifier said | should be | cause | fix |
|---|---|---|---|---|
| HumanEval/24 `largest_divisor` | `sorting_select` | `number_theory` | prompt says "divides n evenly"; I had `divisor` but not `divides` | added `divides` |
| HumanEval/124 `valid_date` | `string_transform` | `string_parse` | pattern was the literal `valid date`; prompt says "the date is valid" | widened to `valid date\|date is valid\|validat` |

The audit also found a weighting cause: generic words (`a list of`, `elements`, `largest`) were
weighted high enough that several of them out-accumulated the one specific signal that named
the real type. Generic weights were dropped to 1. After both fixes the classifier agrees with
my hand-read on **both** samples (20/21 → 21/21 in sample 2; the single miss was /124).

## 3. Type × pass rate

From the 492-row classification table (`runs/heval_fail_classes_30_32_34k.jsonl`), pass@1
greedy at steps 30000/32000/34000.

| type | n | 30k | 32k | 34k | all-3-wrong |
|---|---|---|---|---|---|
| scenario_simulation | 10 | **0%** | **0%** | **0%** | 10 |
| counting_histogram | 9 | **0%** | **0%** | **0%** | 9 |
| arithmetic_basics | 15 | 13% | 13% | 13% | 13 |
| string_parse | 7 | 14% | 14% | 14% | 6 |
| sorting_select | 24 | 17% | 17% | 12% | **20** |
| float_geometry | 12 | 17% | 17% | 17% | 10 |
| number_theory | 18 | 17% | 22% | 22% | 12 |
| string_transform | 24 | 25% | 21% | 17% | 17 |
| dynamic_programming | 8 | 25% | 12% | 25% | 6 |
| base_and_bits | 4 | 25% | 25% | 25% | 3 |
| collection_transform | 27 | 30% | 26% | 26% | 18 |
| bracket_nesting | 6 | 33% | 33% | 17% | 4 |
| **total** | 164 | 19% | 18% | 16% | 128 |

**`scenario_simulation` (10) and `counting_histogram` (9) never pass at any of the three
steps.** They are 19 of the 128 always-wrong tasks (15%). `sorting_select` has the largest
absolute short-board: 20 of 24 tasks wrong at all three.

## 4. Coverage — is the type short of data, or short of capability?

Absolute rate per 1000 sampled rows, **not** share-of-classified. Share-of-classified was
computed first and rejected: its denominator is whatever the classifier could label on that
dataset, which ranges from 55% (code_if) to 100% (HumanEval), so a type's share partly
measures the classifier's own coverage drift instead of the data. Absolute rates do not move
with it. Sampling is a fixed stride (reproducible from the file length), n=2000 per dataset,
except the RL pool which is measured whole.

| type | HE pass | HE | code_if | sc2 | apps | pretrain | RL stdin |
|---|---|---|---|---|---|---|---|
| scenario_simulation | 0% | 61.0 | **8.5** | **9.5** | **32.5** | **17.5** | **26.0** |
| counting_histogram | 0% | 54.9 | **24.0** | **27.0** | 44.5 | **32.5** | 46.4 |
| arithmetic_basics | 13% | 91.5 | 35.5 | 45.5 | **104.0** | 57.5 | **315.4** |
| string_parse | 14% | 42.7 | 38.0 | **73.5** | 29.5 | **66.0** | 5.6 |
| sorting_select | 15% | 146.3 | 36.0 | 50.0 | 106.0 | 58.0 | **168.8** |
| float_geometry | 17% | 73.2 | **106.5** | 75.5 | 64.5 | **111.0** | 51.9 |
| number_theory | 20% | 109.8 | **14.5** | **21.0** | 74.0 | **14.5** | 81.6 |
| string_transform | 21% | 146.3 | 71.0 | **191.0** | **192.0** | 85.5 | 96.5 |
| dynamic_programming | 21% | 48.8 | **14.5** | **8.5** | 38.5 | **18.0** | **9.3** |
| base_and_bits | 25% | 24.4 | 10.5 | 32.5 | 31.5 | 20.5 | 44.5 |
| collection_transform | 27% | 164.6 | **191.5** | **366.5** | **197.5** | **170.0** | 144.7 |
| bracket_nesting | 28% | 36.6 | **1.0** | **4.0** | 15.0 | **1.0** | 9.3 |
| unclassified | — | 0.0 | 448.5 | 95.5 | 70.5 | 348.0 | 0.0 |

Bold = more than 25% below HumanEval's rate for that type.

### 4a. Short of data — the type is thin everywhere, and the model fails it

| type | HE | every dataset | verdict |
|---|---|---|---|
| `scenario_simulation` | 61.0 | 8.5–32.5 (all below) | **缺数据.** The worst-supplied type that still fails 10/10. Word-problem modelling is in effect absent from all four SFT-A sources and from the pretrain shard. |
| `number_theory` | 109.8 | 14.5–21.0 in SFT-A | **缺数据.** 5–8x thinner than HumanEval in the data the model actually trains on. |
| `dynamic_programming` | 48.8 | 8.5–18.0 | **缺数据.** Structurally so: Sc2/APPS-style short single-function problems rarely contain a recurrence. |
| `counting_histogram` | 54.9 | 24.0–46.4 (all below) | **缺数据,** milder — apps and the RL stdin pool come within ~20%. |

### 4b. Supplied, but still wrong — more of the same data is not the obvious fix

| type | HE pass | supply | verdict |
|---|---|---|---|
| `arithmetic_basics` | 13% | apps 104, **RL stdin 315** — 3.4x HumanEval | **数据多照样错.** This is the clearest case: the type is abundant and still fails 13/15 tasks at all three steps. |
| `sorting_select` | 15% | RL 169, apps 106 (but SFT-A 36–50) | **Mixed.** Supply exists upstream; the SFT-A sources under-sample it by 3–4x. Worth checking whether the under-sampling is the cause before assuming a capability gap. |
| `string_parse` | 14% | sc2 73.5, pretrain 66 — above HE | **数据多照样错,** at 7 tasks the sample is small. |
| `float_geometry` | 17% | code_if 106.5, pretrain 111 — above HE | **数据多照样错.** |

### 4c. Counterexample worth keeping: supply is not the whole story

`bracket_nesting` is **the best-passing type (28%) with the worst supply** (1.0–15.0 per 1000
outside HumanEval, i.e. 2.4–37x below HE). It is a two-line depth counter with an obvious
invariant. Conversely `collection_transform` is the best-supplied type in every dataset and
only mid-table on pass rate (27%). Neither direction is reliable on its own — which is why
§4a and §4b are separated by supply, not by pass rate alone.

## 5. What the numbers do not support

1. **`counting_histogram`'s 0% should not be read as "the model cannot count".** The dominant
   `logic_wrong` shape found earlier is a short plausible library one-liner answering a
   neighbouring question (`len(set(string))`, `s[::-1]`, `sorted(set(x))`). Counting tasks
   invite exactly that shape. The coverage table says counting is under-supplied, so both
   explanations are live; this data does not separate them.
2. **Row ≠ problem, and this bounds §4.** One HumanEval row is one distinct problem. One
   code_if/sc2/apps row is an extracted function pair, and a single repository contributes many.
   One pretrain row is a whole FILE. So "per 1000 rows" is not "per 1000 problems" — the table
   is sound for order-of-magnitude supply differences (5–8x) and not for fine rankings.
3. **The pretrain shard is one shard of 283**, measured in `files` mode where no problem
   boundary exists. Its numbers are topic rates over whole files, not per-problem rates, and
   are not comparable to the pairs-mode columns. It is included because 3b could not reach the
   other five code domains from the digest box; those remain unmeasured.
4. **The APPS column is measured on the original question text, not the rendered prompt.** The
   renderer keeps only the starter signature (plus up to 4 solutions per problem), which is
   right for training but withholds the statement the classifier needs: measured on the
   rendered form, apps reads 990/2000 unclassified, i.e. a signature-only string versus
   HumanEval's full docstring. On the question text it is 141/2000. Both numbers are in the
   table's source; the question-text one is used.
5. **code_if is 45% unclassified and that is a finding, not a defect.** Reading the residue:
   `readHatebase`, `load_model` (gzip+pickle), an async framework predicate, numpy cubic
   interpolation, `__setitem__`, a Django 403 handler. code_if is real-world repository code —
   library, IO, framework and method bodies — not algorithmic puzzles, so an algorithmic
   taxonomy legitimately has no label for it. It also means code_if and HumanEval are largely
   different distributions, and the model must acquire the puzzle format, not only the skill.

## 6. Reproduction

```
python3 scripts/problem_types.py --selftest
python3 scripts/problem_types.py --humaneval data/eval/humaneval/humaneval_164.jsonl
```

Coverage scans ran read-only on the digest box (3b's paths, `taskset 8-15 nice 19`). sc2 and
apps were rebuilt into `/tmp` by importing the production renderer
(`scripts/sfta_render_external.py`) as a module with its `OUT_DIR` rebound to `/tmp`; nothing
was written under the repo. The rebuilt sizes match production exactly (sc2 49,599 /
apps 10,776 pairs), which is what makes the sample comparable to the pack.
