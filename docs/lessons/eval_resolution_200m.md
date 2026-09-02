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

### Gold-answer bits-per-byte is the method we were missing

AI2's `olmes` repo ships a suite commented "for evaluating small-scale
experiments" (upstream olmes `oe_eval/configs/task_suites.py:931-1005`, URL in
facts/base_eval.json). Its composition is the
answer to our code problem: all QA is cloze (`:rc`, never `:mc`), and code and
math are **bits-per-byte** — `codex_humaneval:3shot:bpb::none`,
`mbpp:3shot:bpb::none`, `minerva_math_*:bpb`.

OLMo 3 (arXiv 2512.13961 §3.3.2) states the construction exactly, and the
definition is the load-bearing part:

> Base Easy task suite which measures **bits-per-byte (BPB)** over tasks from the
> Base Main suite **that have gold labels or human-written answers**, calculated
> as the **negative log-likelihood of the answer divided by the number of UTF-8
> bytes in the answer string**.

Its stated motivation is our exact problem: small-compute models "exhibit
random-chance performance on math, code, and multiple-choice question answering
(MCQA) tasks". The gold strings are `canonical_solution` for HumanEval
(`codex_humaneval.py:149`), the `code` field for MBPP (`codex_mbpp.py:213`), and
the human-written `solution` for Minerva MATH (`minerva_math.py:198`).

**The distinction that decides the whole recommendation: gold-answer BPB is not
corpus BPB.** Conditional NLL over a gold answer string works. Held-out corpus
perplexity as a downstream proxy does not — Sun et al. (arXiv 2504.12491, Table 1)
measure pairwise accuracy .332/.380/.354 across 50 1B variants, *below* the 0.500
random baseline, and Gadre names code the worst extrapolation domain. Our
`domain_loss` is corpus-level and stays what it is: a data-pricing instrument, not
a capability proxy.

DataDecide reports the same conclusion from the other direction:

> a change of proxy metric makes predictable two code tasks that are otherwise too
> challenging for our small models… decision accuracy goes from trivial to ~80%
> when using Correct Prob… **allows small models to get above the noise floor**.
> Notably, two math benchmarks [MATH, GSM8K] **do not see such a benefit**. — §3.4

### Two corrections to what I first wrote

**pass@k is not physically dead at 300M.** I had argued from SmolLM2-135M's
HumanEval 0.0 that exact-match must floor at our scale. The Codex paper (arXiv
2107.03374, Table 1) refutes it: 12M reads 2.00%, 85M reads 8.22%, **300M reads
13.17%**. §3.4 says the models scoring near 0% are the ones *not trained on code*,
not the small ones. So the reason to keep pass@k out of the profile is variance,
not impossibility — at our Chinese-majority mix with a modest code share the signal
would drown, but **raising the code share could bring pass@1 back**. That makes it
a design variable, not a wall, and it is worth knowing before anyone concludes code
capability is unmeasurable here.

**GSM8K-style gold is the risk, and our data does not have it.** The literature
routes math BPB through Minerva's human-written solutions and skips GSM8K; the
likely cause is that GSM8K's gold CoT carries calculator annotations (`<<48/2=24>>`)
that are out-of-distribution for a base model. I checked ours: `math_test_500` has
**0 of 500** golds with such annotations — it is human-written prose ending in
`\boxed{}`, Minerva-shaped. `math_hard_v2` golds are constructed-from-answer and
equally clean. **The objection that excludes GSM8K does not apply to our math
data**, which moves math BPB from "the literature says no" to "worth measuring".

## Where our panel actually stands

Math already has its likelihood twin (`eval/math_v2_like.py`, gold-win 94.92% vs
wrong-win 5.08%, 3012 pairs, swap-stable). Code has nothing — every code metric we
own is generative and reads zero.

That asymmetry is the gap, and the literature says code is the side that responds
to a proxy metric.

## An admission threshold, measured at our configuration

Gadre et al. give the only quantitative rule in this literature for whether a task
is worth measuring at all (§3.4): a task qualifies if

> at least one **0.154B** scale model — trained with as many as **99B tokens** —
> gets **10 percentage points above chance** accuracy

That is 154M params, which is our scale. Of 46 tasks, 17 qualified; MMLU,
ARC-Challenge, MathQA and OpenBookQA were excluded, MMLU named as "close to random
chance". Adopt it directly: **a task that cannot clear chance+10pp at our scale
does not enter the decision loop.** Note this also retires the temptation to read
ARC-Challenge's SNR 4.27 as a reason to add it — high discriminating power on data
recipes is not the same as a readable capability signal.

## Ranked additions

Reading rule from fb: an eval enters the profile only with a cited non-chance
signal at ≤300M and a contamination-clean copy we hold.

All three below are **gold-answer BPB or cloze**, never corpus BPB and never
exact-match. Normalisation is per **UTF-8 byte**, not per character: byte
normalisation is tokenizer-invariant, which matters more for us than for anyone in
the cited work because our 32K vocab is Chinese-optimised and character counts do
not translate across tokenizers.

### 1. code_bpb — gold-answer BPB on `reference_code`

- **Cited signal at ≤300M**: DataDecide §3.4 (code becomes predictable under a
  likelihood proxy at 150M); OLMo 3 §3.3.2 ships exactly this construction for
  HumanEval and MBPP in its small-scale suite.
- **Clean copy**: yes, `data/eval/code_holdout_v2_500.jsonl`, carved before ingest
  and scanned (`cont.code_holdout_carved`, `cont.code_crawl_scan`). 500 problems,
  30 families, 48,446 bytes of reference code, mean 96 bytes.
- **n and resolution**: n = 500 problems / 48,446 bytes. BPB is continuous with no
  floor and no chance level, so it has no binomial MDE — **its resolution must be
  measured as seed variance, not derived**. That measurement does not exist yet and
  is the first thing to run.
- **Cost**: one forward pass per problem, no generation. Cheaper than any current
  generative metric by an order of magnitude.
- **Why first**: it converts our only permanently-zero metric into a continuous one,
  on data we already hold clean, using the construction AI2 uses at this scale.

### 2. math_bpb — gold-answer BPB on the math solutions

- **Cited signal at ≤300M**: OLMo 3 §3.3.2 routes math BPB through Minerva's
  human-written solutions. The Signal-and-Noise separation is stark — Minerva BPB
  SNR 88.6 against GSM8K BPB 7.0 — and the likely cause is GSM8K's calculator
  annotations being out-of-distribution for a base model.
- **Clean copy**: yes, and this is the finding that promoted it. `math_test_500`
  golds are human-written prose ending in `\boxed{}` with **0 of 500** carrying
  `<<...>>` annotations; `math_hard_v2` golds are constructed-from-answer. Both are
  Minerva-shaped, so the objection that excludes GSM8K does not apply to us.
  `math_test_500` is 381,669 gold bytes; `math_hard_v2` is 1,080 problems / 86,886
  bytes and already carries a contamination verdict (`cont.math_hard_v2`).
- **n and resolution**: same as (1) — continuous, resolution must be measured.
- **Why second and not excluded**: DataDecide says a proxy metric does not rescue
  MATH/GSM8K *accuracy*. That is a claim about their gold and their tasks. Ours is
  the shape the literature says works, so this is a measurement worth making rather
  than a conclusion to inherit. If it shows no seed-separable movement, it drops.

### 3. ceval scored as cloze rather than MCF

- **Cited signal at ≤300M**: OLMES §3.4 and Tables 6–7 — cloze is above chance
  where MCF is at chance at 1B, and MCF format acquisition sits at ~400B tokens.
- **Clean copy**: already in the profile.
- **Status: built and verified.** `eval/ceval.py:38` was scoring the bare letters
  A/B/C/D, which is MCF. `load_items(cloze=True)` now scores the four option texts
  as continuations with per-character normalisation; selftest passes on the pod,
  1,050 items both ways (commit 89de381, selftest fix edc5c70).
- **n and resolution**: n unchanged at 1,050, so MDE is unchanged. This is a
  scoring change, not a new eval.
- **Why third**: it is done and it costs nothing, but it is a tripwire on the
  Chinese MC axis rather than a capability instrument, and the MC floor is partly a
  size effect that cloze does not remove.

**Not recommended:** scoring `expected_output` given instruction + reference code.
It is the closest analogue to `math_v2_like` and easy to build, but it measures
execution tracing rather than code generation — a model that cannot write the code
may still predict its printed output.

**Dropped from my first draft:** a code likelihood twin with in-family distractors.
It duplicates (1)'s signal at higher cost, and an in-family distractor is another
problem's *correct* code, not a controlled edit the way `math_v2_like`'s digit
perturbation is. Gold-answer BPB gets the same information without the distractor
design question.

**Explicitly not dropped, but reframed:** pass@k stays out of the profile on
variance grounds, not impossibility. Codex-300M reads 13.17% pass@1 (arXiv
2107.03374 Table 1), so if the code share of the mix rises, this decision should be
revisited rather than treated as settled.

## Free variance reduction

Averaging the last k checkpoints cuts scaling-fit error substantially (Signal-and-
Noise intervention 2: GSM8K scaling error 7.46 → 3.85). We keep the newest three
checkpoints by default, so this costs nothing but a change in how the milestone
readout aggregates. Worth doing before adding any new metric, because it improves
every metric already in the profile.

## Ceilings

- The DataDecide table is my computation over their released data, not paper text.
  The paper publishes those per-task numbers only as figures.
- MBPP, HumanEval, GSM8K and MATH are absent from the released dataset; their §3.4
  claim is figure-only, so no numeric threshold for code-under-bpb exists to quote.
- Every DataDecide number is English-on-English. It is method evidence for us, not
  a score prediction.
- BPB's resolution at our scale is unmeasured. Until a 2-seed run exists, no BPB
  movement should be called readable.
- OLMo 3's appendix claim of "signal even at 190M" could not be independently
  verified — the HTML full text truncates before the appendix. Only §3.3.2's
  definition and the three gold-string sources are confirmed. Treat the 190M figure
  as unverified.
- No peer-reviewed source publishes concrete small-model code/math numbers of the
  "Pythia-410M GSM8K = 0.0%" kind. The literature asserts random-chance
  qualitatively and consistently omits the zeros. That is a real gap in the
  evidence, not a gap in the search.
- DataDecide does not say how MBPP/HumanEval candidate sets are built for Correct
  Prob, and its 300M decision accuracy is not reported.

## Sources

- OLMES, arXiv 2406.08446v2 (Findings of NAACL 2025) — Table 2, §3.1, §3.4,
  Figure 1 caption, Tables 6–7
- DataDecide, arXiv 2504.11393v2 (ICML 2025) — Table 2, §2.4, §3.1, §3.3, §3.4;
  `allenai/DataDecide-eval-results`
- OLMo 3, arXiv 2512.13961 — §3.3, §3.3.2
- Gadre et al., scaling laws over-trained / downstream — §3.4 (the chance+10pp
  admission rule at 0.154B / 99B tokens), Table 2
- Codex, arXiv 2107.03374 — Table 1 (pass@1 by model size), §3.4
- Sun et al., arXiv 2504.12491 — Table 1 (corpus perplexity below random as a
  downstream proxy)
- Schaeffer et al., arXiv 2304.15004 — discontinuous metrics manufacture emergence
- AI2 `olmes` (upstream) — `oe_eval/configs/task_suites.py:931-1005`;
  `codex_humaneval.py:149`, `codex_mbpp.py:213`, `minerva_math.py:198`
- Our prior: `docs/lessons/base_eval_at_200m.md`, `facts/base_eval.json`
