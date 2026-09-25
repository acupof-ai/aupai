---
question: After SFT-A, how do we turn the 9,531 verified problems into SFT-B supervised pairs vs RL problems, what is the function-form vs stdin-form gap, and what does ultra_l3 add?
status: measured
source: facts/v41.json#v41.rl_code_pool_stdin_supply_0925; runs/heval_fail_classes_30_32_34k.jsonl; PR #701/#703 problem-type classifier; digest wrap500 probe; pod ultra_l3 probe (both 2026-09-25)
---

# Post-SFT-A code data: SFT-B vs RL candidates, and the form gap

This is a CANDIDATES document for 1e/the user. The recommendation column is intentionally
empty pending (a) the 500-problem wrap experiment below and (b) the SFT-A before/after
HumanEval delta. No HumanEval lift is promised: HumanEval has 4-27 tasks per type, so per-type
pass rates are noisy.

## 1. What fails, restated (66, PR #701/#703)

- 128 of 164 HumanEval tasks are wrong at all three checkpoints (30k/32k/34k); 15 are
  flaky across steps, 21 always pass.
- Weighted gate-mix exposure does NOT predict pass rate: the least-supplied type
  (bracket_nesting) passes best; counting_histogram is at parity with HumanEval exposure
  and passes 0%. Therefore the fix is not "add more corpus of the thin type"; it is
  problem+verified-solution pairs (SFT-B) or test-bearing problems (RL) that force the
  model to produce an answer and receive binary feedback.
- `scenario_simulation` is the one type thin in every measured dataset; external-source
  work on it stays the lowest priority.

## 2. The verified pool and the SFT-B/RL split

The pool (fact `v41.rl_code_pool_stdin_supply_0925`) is **9,531 unique, deterministic,
reference-verified, HE/MBPP-decontaminated, SFT-A-disjoint** problems:

| file | form | rows |
|---|---|---|
| rl_code_taco.jsonl | call-style (`def f`, pytest) | 27 |
| rl_code_apps_stdin.jsonl | stdin whole-program | 182 |
| rl_code_taco_stdin.jsonl | stdin whole-program | 9,322 |

SFT-B and RL are mutually exclusive by the same normalised-prompt hash. Default split
(1e): low difficulty to SFT-B, medium+ to RL, UNKNOWN to RL.

| stage | call-style function | stdin whole-program | total |
|---|---|---|---|
| SFT-B (APPS introductory, TACO EASY) | **8** | **3,199** | 3,207 |
| RL (everything else, incl. UNKNOWN) | 19 | 6,305 | 6,324 |

**The decisive fact for form:** only 27 of 9,531 problems are HumanEval-shaped function
continuations; 9,504 are stdin whole-programs. The SFT-B subset is 99.75% stdin form.
HumanEval scores a function body, so a naïve SFT-B built from these pairs teaches
stdin→stdout scripting, not function completion.

### Per-type counts under the default split

| type | SFT-B call | SFT-B stdin | RL call | RL stdin |
|---|---:|---:|---:|---:|
| arithmetic_basics | 1 | 1,052 | 4 | 1,822 |
| sorting_select | 0 | 514 | 2 | 1,106 |
| collection_transform | 3 | 356 | 8 | 663 |
| number_theory | 1 | 210 | 0 | 758 |
| string_transform | 0 | 363 | 0 | 583 |
| counting_histogram | 0 | 263 | 1 | 407 |
| float_geometry | 1 | 164 | 1 | 436 |
| scenario_simulation | 0 | 122 | 0 | 163 |
| base_and_bits | 2 | 80 | 1 | 182 |
| dynamic_programming | 0 | 39 | 0 | 96 |
| bracket_nesting | 0 | 19 | 1 | 50 |
| string_parse | 0 | 17 | 1 | 39 |

Labels are one mechanical classifier (PR #703), audited 21/21 on its HumanEval sample; use
for order-of-magnitude supply, not ground truth.

## 3. The form options (A/B/C) — recommendation left blank

### Option A: keep stdin form, train algorithmic reasoning as-is

Use the 3,199 low-difficulty stdin pairs for SFT-B unchanged (statement docstring → whole
program). Zero reference rewriting; every pair already passes its tests and the two-run
determinism screen. Risk: surface form differs from HumanEval's single-function prompt.

### Option B: wrap stdin references into `def solve(input_str) -> str`

A source transform indents each verified reference into a function, rebinding
`sys.stdin`/`sys.stdout` to StringIO, hoisting the reference's own top-level
`from x import *` lines (which are illegal inside a function), then re-running the
ORIGINAL cases through isolate.

500-problem fixed-seed experiment (seed 20260925, SFT-B low-difficulty rows, 30 s
per case):

| wrapper version | transform parses | post-wrap pass | note |
|---|---:|---:|---|
| v1 (naive indent) | 500/500 | 477/500 = 95.4% | 7 failures were the wrapper's own fault (`import *` indented into a function) |
| v2 (hoist ALL imports) | 500/500 | 473/500 = 94.6% | over-broad: lifted local imports too, introduced regressions |
| **v3 (hoist only col-0 `import *`)** | **500/500** | **482/500 = 96.4%** | minimal correct transform |

v3 failure buckets (18): 10 stdout mismatch (scripts that read stdin/print in a way the
rebind changes, or genuinely different output), 2 unpack errors, 2 int-parse errors, 1
undefined name, 1 bad fd, 1 malformed drive output, 1 StringIO attribute (script calls a
StringIO method the rebind lacks). None are repaired heuristically; wrapping keeps only the
482 that run their own original cases green.

**Function-form supply this unlocks (estimate from the 96.4% rate):** of the 3,199
SFT-B-split stdin problems, roughly ~3,080 could become HumanEval-shaped
`def solve(input_str)->str` pairs that pass their original tests (rate applied to the sampled
low-difficulty population; a full conversion would produce the exact count). This closes the
form gap that the 27 native call-style problems leave open.

### Option C: source function-form pairs elsewhere

Only 27 function-form problems survive in the pool because APPS function problems were
removed by SFT-A dedup (SFT-A trained them) and TACO contributes almost no novel
function-form problems. Additional function-form supply must come from a new source or from
ultra_l3 (§4); candidates and counts are [PENDING the SFT-A delta and wrap result].

## 4. ultra_l3_noexec as a function-form SFT-B source (pod, measured)

`data/corpus/code_ultra_l3_noexec_dc` is in the pretrain mix (15,309,833 kept rows /
26.697B packed tokens across 940 shards; mix weight 0.314634, anneal 0.421333, epochs 1).
Each content is an English statement followed by a reference `def` and inline `assert`s —
HumanEval-shaped. Two independent probes (fixed-stride sample of 1,000 over 5 shards, seed
20260925):

### 4a. Reference/assert quality — the gates are stricter than they first looked

| fact | count | rate of 1000 |
|---|---:|---:|
| statement/code separable at first code line | 1,000 | 100% |
| carry a top-level def | 998 | 99.8% |
| carry >=1 assert | 890 | 89.0% |
| runs exit 0 at all (the loose "green" first measured) | 432 | 43.2% |
| **runs green AND carries >=1 assert (real self-test pass)** | **323** | **32.3%** |
| fails its own asserts | 407 | 40.7% |
| runtime error / timeout | 162 | 16.2% |

The 109 exit-0 rows with NO assert are not verified — they must not count as candidates.

**Hand classification (1e order, seed 20260925): 20 random assert-fail rows read end to end,
each disputed expectation independently recomputed.**
- **12/20 (60%) the ASSERT/expected value is wrong, the solution is a correct/standard
  implementation.** Examples: `look_and_say("9",3)` truth 3119, assert wants 132112;
  Armstrong `find(5000)` assert lists 8208/9474 which exceed 5000; `667^937 mod 2537`
  = 1808 by `pow`, assert writes 916 against a textbook binary-exponentiation solution;
  longest-consecutive run of 4 asserted as 3; a Floyd-Warshall test whose own comment says
  "==3 … so 2".
- **2/20 (10%) the SOLUTION is genuinely wrong** (a float confidence returned
  0.010000000000000009 against an exact `==0.01`, needs rounding; a multi-source BFS
  returning unreachable for a reachable cell).
- **6/20 (30%) not independently settled in the sample** (large backtracking/state-machine/
  NumPy-SLSQP problems); left unclassified rather than forced.
- An objective batch signal: **77/407 (19%) of all assert-fail files carry a
  self-correction/contradiction phrase directly in an assert line** ("Let's fix",
  "actually", "correct is", …) — a lower bound on wrong-expectation rate.

**10 random error/timeout rows:** 6 are test-only files with NO solution `def` generated at
all; 1 is missing a fixture class (TreeNode); 3 carry a `def` that is an empty stub/real bug
(one literally comments "Assume the solution function is defined here").

Conclusion: this domain's dominant defect is **unreliable test expectations and missing
solutions, not wrong algorithms taught at scale**. The earlier hypothesis "~18% of pretrain
tokens teach wrong code" (0.3146 × (40.7+16.2%)) is an UPPER bound before removing harness
artifacts, and the hand read does not support wrong-solution as the main component. Any
SFT-B use must keep only rows that (a) carry >=1 assert, (b) run every assert green, and
(c) pass an assert-consistency filter that drops self-contradictory test lines — a wrong
solution can pass a wrong assert, so "runs green" alone is insufficient.

### 4b. How much pretrain actually consumed (checkpoint row cursor)

CED ran to completion (step 38146/38146). `ckpt_v41_ced_0923.pt` row_cursor basis
`full_plan_prefix`, seed 42: the domain rows are `random.Random(42).shuffle(concat(shards))`
and consumed in that order. ultra_l3 consumed **2,320,255 / 15,309,833 rows = 15.16%**, so
the unconsumed shuffle tail is **12,989,578 rows (84.84%)** — exactly reproducible with the
same seed/index permutation. (This is a row-count identity; rows vary in token length.)

### 4c. REAL 200k-tail funnel (measured)

First 200,000 rows of the unconsumed tail, reproduced in the exact seed-42 order, on pod
node1 (numactl 114-127, nice 19, 16 workers). Gates in order: def+>=1 assert → every assert
runs green → assert self-correction markers removed → HE/MBPP 13-gram → prompt/code hash
disjoint from the SFT-A two layers AND the RL pool (15,465 dedup hashes).

| funnel stage | rows |
|---|---:|
| tail rows read | 200,000 |
| no parseable code | 59 |
| no top-level def | 19,972 |
| def but no assert | 23,725 |
| has def AND >=1 assert (sent to execute) | 156,244 |
| reference fails its own asserts | 102,184 |
| reference times out (10 s) | 1,005 |
| **runs green with asserts** | **53,055** |
| dropped by assert self-correction markers | 2,423 |
| dropped by HE/MBPP 13-gram | 0 |
| dropped by SFT-A/RL prompt-hash overlap | 0 |
| **kept** | **50,632** |

The 25.3% kept rate matches the 1,000-row 32.3% gate within sampling (the executed-pool green
rate is 53,055/156,244 = 34.0% on shape-ok rows; the difference vs 32.3% is which denominator
includes the no-def/no-assert rows). HE/MBPP and SFT-A/RL drops are exactly zero, consistent
with the domain already being HE-decontaminated at build and the tail never having been used
for SFT-A/RL. Products (pod only, not in git): `/data00/aupai_work/u3sftb_0925/`,
`ultra_l3_sftb_candidates.jsonl` (50,632 rows, sha256
`5dc616cfdc8a392a46551c413a95f…b13a95f`) and `ultra_l3_sftb_funnel.json` (sha256
`190d7c7fca938f6f…10ee894`); producer `scripts/sftb_ultra3_tail_funnel.py`.

**Marker finding during the hand audit.** The barber row revises its expected value in a
trailing assert comment (`# … So N=4 ->1`) that none of the keyword markers matched. A bare
"actually" marker was tried and rejected: it fires on **3,342 / 50,633** survivors, almost all
benign reasoning comments that agree with the assert (`# Actually best: 4+8=12`). The targeted
marker is the arrow-correction regex `so\s+n\s*=\s*\d+\s*(?:->|=>)\s*\d+`, which matches
**exactly 1 row in 50,633** — the barber row — so the kept set drops to 50,632 and the barber
function (57 lines) is outside the ≤280 subset anyway.

The 30-row survivor hand audit and body-token lengths are reported in §6.


## 5. code_if rebalance (measured)

code_if_clean is 76,644 SFT-A pairs; only **1,562 (2.0%)** outputs contain a `def` — the rest
are real repository method/library/IO bodies (PR #701: 45% unclassified as algorithmic). It
is real-world code, not puzzle form. Concrete down-weight options with numbers are [PENDING
the wrap result, since the chosen form decides what replaces its share].

## 6. Indicative SFT-B mix (1e preliminary direction; final after SFT-A delta)

Two complementary function-form sources, different difficulty bands:

### Component 1 — ultra_l3 unconsumed tail, function body ≤ 280 gate tokens

Restricting the 50,632 verified 200k-tail candidates to reference FUNCTIONS at most 280 gate
tokens (aligned with SFT-A's 256 body gate and the HumanEval 280 generation budget):

- **19,654 rows** (the dropped barber row is 57 lines and outside this gate); reference-function
  body token total **4.05M**, median 214 / p90 267.
- These are **4.6× the HumanEval canonical-body median (~46 tokens)** at the median and ~5.8×
  at p90; even the ≤280-gated ultra functions are substantially longer bodies than HumanEval's.
  Function-only length is measured with the gate tokenizer (the `ast` top-level FunctionDef,
  excluding asserts/fixtures/trailing prose — the same object as a HumanEval canonical body).
- Per type (66 #703 classifier): collection_transform 6,083; sorting_select 3,773;
  counting_histogram 1,625; float_geometry 1,482; number_theory 1,434; base_and_bits 1,361;
  arithmetic_basics 1,276; string_transform 1,235; string_parse 509; dynamic_programming 400;
  bracket_nesting 205; scenario_simulation 75; unclassified 196.

### Component 2 — TACO/APPS low-difficulty stdin wrapped to function form

- 3,199 low-difficulty stdin problems (3,132 TACO EASY + 67 APPS introductory); at the
  measured v3 wrap pass rate 96.4%, **~3,084 become verified function-form pairs** (estimate;
  a full conversion gives the exact number). impl token total 0.69M across all 3,199 (~0.67M
  for the wrapped subset).
- Per type: arithmetic_basics 1,052; sorting_select 514; string_transform 363;
  collection_transform 356; counting_histogram 263; number_theory 210; float_geometry 164;
  scenario_simulation 122; base_and_bits 80; dynamic_programming 39; bracket_nesting 19;
  string_parse 17.

### Combined picture (body tokens; prompts/eos add overhead at pack time)

| component | estimated rows | body tokens (approx) | difficulty |
|---|---:|---:|---|
| ultra_l3 ≤280 | 19,654 (measured) | 4.05M | medium-hard algorithmic |
| TACO/APPS wrapped | ~3,084 (96.4% estimate) | ~0.67M | EASY/introductory |
| **sum** | **~22,738** | **~4.7M** | complementary bands |

Type complementarity: TACO adds a disproportionate share of arithmetic/sorting/counting at
the easy end; ultra_l3 supplies collection/sorting/number_theory/float depth. scenario remains
the thinnest (75 + 122 = 197). These are supply numbers only — the proportion is a user
decision, and no HumanEval lift is promised.

### Residual bad-pair rate that a keyword filter cannot remove

The assert-hygiene filter drops explicit self-correction phrases. The barber-shop audit row
showed a WRONG solution passing a WRONG assert using phrasing the original keyword list did not
catch (`assert barber(4, [2,5])==2` while the assert's own trailing comment derives customer 4
→ barber 1, and an independent queue simulation agrees with 1). The producer now drops this one
row with the `So N=k ->v` arrow regex (§4c), but **a semantic contradiction between a wrong
solution and its wrong expectation cannot be caught by keywords in general** — a row with no
self-revealing comment is invisible. In the 30-row survivor hand audit the confirmed
wrong-solution-passes-wrong-assert rate was **1/30 = 3.3%** (the barber row); by rule of three
the 95% upper bound on the residual rate after marker removal is ~12%. The
SFT-B builder should treat the candidates as ~97%-precision and consider an independent
reference cross-check for the final shipped subset rather than trusting bundled asserts alone.

## 7. Open measurements before a mix is set

1. ~~500-row wrap rate (option B go/no-go)~~ — DONE: 96.4% post-wrap pass (v3), so wrapping is
   viable; ~3.1k of the 3,199 low-difficulty stdin problems can become function-form pairs.
2. ~~Full-domain ultra_l3 green-pair count~~ — DONE, replaced by the REAL 200k unconsumed-tail
   funnel (§4c): 50,632 kept; the ≤280 subset is 19,654 rows / 4.05M body tokens, median 214 /
   p90 267; the 30-row survivor hand audit found one wrong-pair (barber, now dropped), residual
   3.3% / upper ~12%.
3. SFT-A before/after HumanEval delta (1e, incoming) — the one measurement still open; it sets
   the form/proportion decision.
4. scenario_simulation external source supply — lowest priority; no number yet. The pool
   contains 285 scenario rows (122 SFT-B / 163 RL under the difficulty split), so it is thin
   but not zero in-house.

Recommendation column: intentionally left blank. The 200k funnel, body-token lengths and
30-row audit are in; the only remaining input is the SFT-A before/after HumanEval delta, after
which 1e chooses between TACO-wrapped pairs and ultra_l3 function pairs for SFT-B form and
proportion.
