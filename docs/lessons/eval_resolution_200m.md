---
question: "which evals resolve at 200M / 15-30B tokens for code and math, and what should enter the milestone profile"
status: recorded
source: "e1-2 2026-08-31; OLMES arXiv 2406.08446, DataDecide arXiv 2504.11393 + allenai/DataDecide-eval-results, AI2 olmes repo; our facts/base_eval.json"
---

# Eval resolution at 200M

Research, 2026-08-31 (e1-2). The milestone profile carries five metrics: three of
five MC sets sit at chance and code-500-v2 reads 0/500 on every checkpoint. The
question is which additions would actually move.

This extends `base_eval_at_200m.md` (2026-08-30) rather than replacing it. That
doc established the likelihood-first principle from Pythia/OLMo/Schaeffer/CLiMP
and it holds. What follows is what OLMES, DataDecide and the task-ladder line add
or contradict, and it is mostly **quantitative confirmation plus one method we
did not have**.

## What is new

| source | adds | contradicts |
|---|---|---|
| OLMES | the cloze-vs-MCF split is a *format* rule, not a size rule; MCF is learned at ~400B tokens | nothing in our doc |
| DataDecide | measured accuracy and signal-to-noise at 150M/15B and 300M/30B, our exact rungs | our assumption that near-chance means unusable |
| AI2 `olmo3:base_easy` | bits-per-byte for code and math at small scale | our treatment of code as unmeasurable |

### DataDecide lands on our rungs exactly

DataDecide's 150M rung is 151.9M params / 15.0B tokens and its 300M rung is
320.0M / 30.0B. Our stage-1 targets are 15B and 30B at ~200M. This is the closest
scale match in the literature to what we are doing, and it is the reason its
numbers carry more weight here than Pythia's.

The paper reports per-task results only as figures, so the table below was
computed from the released `allenai/DataDecide-eval-results` (1,410,750 rows),
final checkpoint, mean over 25 data recipes, `seed=default`, metric
`acc_per_char`. **These are computed from their data, not quoted from the paper.**

| task | chance | 150M | 300M | headroom @150M | SNR |
|---|---|---|---|---|---|
| ARC-Easy | 25 | 52.18 | 58.97 | +27.2 | 9.73 |
| CSQA | 20 | 42.10 | 49.07 | +22.1 | 2.46 |
| PIQA | 50 | 65.14 | 68.35 | +15.1 | 2.59 |
| HellaSwag | 25 | 35.83 | 43.92 | +10.8 | 3.11 |
| SocialIQA | 33.3 | 43.11 | 45.65 | +9.8 | 1.09 |
| BoolQ | 50 | 53.56 | 55.95 | +3.6 | 1.41 |
| MMLU | 25 | 27.92 | 29.89 | +2.9 | — |
| OpenBookQA | 25 | 26.50 | 29.52 | +1.5 | 2.17 |
| ARC-Challenge | 25 | 26.28 | 29.53 | +1.3 | 4.27 |
| WinoGrande | 50 | 51.00 | 53.10 | +1.0 | 0.76 |

SNR = spread across 25 recipes ÷ std-dev across 3 seeds, at 150M.

**This contradicts one of our working assumptions.** We treated near-chance as
unusable. ARC-Challenge sits 1.3 points above chance and still separates data
recipes at SNR 4.27 — higher than PIQA, which has 15 points of headroom. Absolute
headroom and discriminating power are different properties. What kills a metric is
SNR below 1, which is WinoGrande at 0.76: its recipe spread is smaller than its own
seed noise.

The corollary is that **every one of these numbers is meaningless without a seed
estimate**, and seed noise at this scale runs 0.3–4.0 points depending on task. We
have this discipline already (`be.panel_expressive_seed_variance`, 4 seeds at 0.2b).

### The English numbers do not transfer to us

Every row above is an English benchmark on English-pretrained models. Our measured
English share is 0.75% of tokens, and `be.mc_language_confound` isolated benchmark
language as the cause of our MC floor: ceval moves while four English sets do not,
same model, same scale. So the table is evidence about **method**, not a prediction
of our scores. Read it as: at 150M with 15B tokens, cloze-scored MC is alive in the
pretraining language. Ours is Chinese, so the Chinese equivalent is what would move.

### OLMES: cloze vs MCF is a format threshold, not a size threshold

OLMES has no minimum-parameter rule; its smallest model is Pythia-1B, and nothing
at 150–300M was tested. What it does state is stronger for us:

> the model starts learning the MCF task format after about **400 billion training
> tokens**, so in early training CF provides a better signal — Figure 1 caption

400B is 13–27× our budget. Their three weakest models show the split cleanly
(MCF / CF, Tables 6–7):

| model | ARC-E | HellaSwag | CSQA |
|---|---|---|---|
| Pythia-1B | 24.0 / **63.4** | 23.6 / **48.0** | 21.0 / **50.9** |
| OLMo-1B | 25.4 / **68.3** | 24.6 / **65.2** | 20.2 / **62.2** |
| TinyLlama-1.1B | 24.3 / **69.5** | 26.2 / — | 17.9 / **61.1** |

MCF at chance, cloze alive, at 1B — five times our size. Normalisation is
per-character (`acc_norm`), not per-token.

**Action item independent of any new metric: check how `mc_full` scores ceval.**
If it is MCF-style (presenting lettered options and reading the letter), the
Chinese tripwire is being read in the format that is dead below 400B tokens, and
switching to cloze costs nothing but a rescore. This is the single cheapest thing
in this document.

### Bits-per-byte is the method we were missing

AI2's `olmes` repo ships a suite commented "for evaluating small-scale
experiments" (`oe_eval/configs/task_suites.py:931-1005`). Its composition is the
answer to our code problem: all QA is cloze (`:rc`, never `:mc`), and code and
math are **bits-per-byte** — `codex_humaneval:3shot:bpb::none`,
`mbpp:3shot:bpb::none`, `minerva_math_*:bpb`.

DataDecide reports the same conclusion from the other direction:

> a change of proxy metric makes predictable two code tasks that are otherwise too
> challenging for our small models… decision accuracy goes from trivial to ~80%
> when using Correct Prob… **allows small models to get above the noise floor**.
> Notably, two math benchmarks [MATH, GSM8K] **do not see such a benefit**. — §3.4

Read that last sentence carefully, because it is the one finding here that goes
against what we would like to be true: **the metric change rescues code and does
not rescue math.**

## Where our panel actually stands

Math already has its likelihood twin (`eval/math_v2_like.py`, gold-win 94.92% vs
wrong-win 5.08%, 3012 pairs, swap-stable). Code has nothing — every code metric we
own is generative and reads zero.

That asymmetry is the gap, and it is the opposite of what the literature says is
achievable: code is the one that responds to a proxy metric, math is the one that
does not.

## Ranked additions

Reading rule from fb: an eval enters the profile only with a cited non-chance
signal at ≤300M and a contamination-clean copy we hold.

### 1. code_bpb — bits-per-byte on `reference_code`

- **Cited signal at ≤300M**: DataDecide §3.4 (code becomes predictable under a
  likelihood proxy at 150M); AI2 `base_easy` uses bpb for HumanEval and MBPP at
  small scale.
- **Clean copy**: yes, `data/eval/code_holdout_v2_500.jsonl`, carved before ingest
  and scanned (`cont.code_holdout_carved`, `cont.code_crawl_scan`). 500 problems,
  30 families, 48,446 bytes of reference code.
- **n and resolution**: n = 500 problems / 48,446 bytes. BPB is continuous with no
  floor and no chance level, so it has no binomial MDE — **its resolution must be
  measured as seed variance, not derived**. That measurement does not exist yet and
  is the first thing to run.
- **Cost**: one forward pass per problem, no generation. Cheaper than any current
  generative metric by an order of magnitude.
- **Why first**: it converts our only permanently-zero metric into a continuous one,
  on data we already hold clean, using the construction AI2 uses at this scale.

### 2. ceval scored as cloze rather than MCF

- **Cited signal at ≤300M**: OLMES §3.4 and Tables 6–7 — cloze is above chance
  where MCF is at chance, at 1B; the MCF format itself is learned at ~400B tokens.
- **Clean copy**: already in the profile.
- **n and resolution**: ceval as configured; MDE unchanged since n does not change.
  This is a scoring change, not a new eval.
- **Why second**: near-zero cost, and if `mc_full` is currently MCF it may explain
  part of our MC floor. It is second only because the win is contingent on how
  `mc_full` is implemented, which I have not yet read.

### 3. code_holdout_v2 as a likelihood twin (in-family distractors)

- **Cited signal at ≤300M**: same DataDecide result as (1); the construction is our
  own `math_v2_like`, which is swap-validated and works.
- **Clean copy**: yes, same file as (1).
- **n and resolution**: 30 families, all with ≥2 members, giving 7,848 in-family
  distractor pairs. At the panel's frozen formula MDE = 1.4/√n: n=500 → 6.26pt,
  n=1500 (3 distractors per problem) → 3.61pt, n=2000 → 3.13pt.
- **Why third**: it duplicates (1)'s signal at higher cost and needs a distractor
  design review — an in-family distractor is another problem's correct code, which
  is plausible but not a controlled edit the way `math_v2_like`'s digit
  perturbation is.

**Not recommended:** scoring `expected_output` given instruction + reference code.
It is the closest analogue to `math_v2_like` and it is easy to build, but it
measures execution tracing rather than code generation. A model that cannot write
the code may still predict its printed output.

## Ceilings

- The DataDecide table is my computation over their released data, not paper text.
  The paper publishes those per-task numbers only as figures.
- MBPP, HumanEval, GSM8K and MATH are absent from the released dataset; their §3.4
  claim is figure-only, so no numeric threshold for code-under-bpb exists to quote.
- Every DataDecide number is English-on-English. It is method evidence for us, not
  a score prediction.
- BPB's resolution at our scale is unmeasured. Until a 2-seed run exists, no BPB
  movement should be called readable.

## Sources

- OLMES, arXiv 2406.08446v2 (Findings of NAACL 2025) — Table 2, §3.1, §3.4,
  Figure 1 caption, Tables 6–7
- DataDecide, arXiv 2504.11393v2 (ICML 2025) — Table 2, §2.4, §3.1, §3.3, §3.4;
  `allenai/DataDecide-eval-results`
- AI2 `olmes` — `oe_eval/configs/task_suites.py:931-1005`
- Our prior: `docs/lessons/base_eval_at_200m.md`, `facts/base_eval.json`
