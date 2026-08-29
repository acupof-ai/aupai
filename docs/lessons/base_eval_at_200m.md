# How to measure capability change at 200M scale

Research, 2026-08-30. Question (aupai-fb): at 200M parameters, domain loss moves
(1.66–1.99 nat over 16× data) but every capability metric sits at its floor —
what actually measures capability change at this scale, and what should the
base-model eval matrix be.

## TL;DR

1. **Generative evals are dead at 200M base. Don't repair them — replace them.**
   Our own measurements: 0/500 math-500, 0.8% boxed rate, degenerate output
   (165-colon loops). A base model has no instruction format to comply with;
   generation exact-match has a floor near zero and leaves it last.
2. **What moves at sub-1B, in order of appearance: domain loss → token-level
   target logprob / acceptability minimal pairs → LAMBADA-style constrained
   prediction → MC commonsense → few-shot ICL → generation.** The first three
   are likelihood-only and are the standard base-model instruments in the
   Pythia/OLMo/SmolLM lineage.
3. **The "loss moves but nothing else does" gap is mostly metric granularity,
   not a real waiting period.** Schaeffer et al. (2023): with smooth, token-level
   metrics, improvement is continuous from early training; apparent emergence is
   produced by discontinuous metrics (exact match, pass@1). Measure the
   token-level likelihood version of the capability you care about and it moves
   while accuracy is at floor.
4. **Minimal pairs are the right core for Chinese. CLiMP (Xiang et al. 2021)
   is the prior art: 16 syntactic contrasts, 1000 pairs each, 95.8% human
   agreement.** Construction rules and sample-size math below; the floor is 50%,
   so the signal density is an order of magnitude above 4-choice MC.
5. **The MC floor is mostly a suite problem, not a model problem.** ceval (Chinese)
   moves; the four English benchmarks don't. Same model, same scale, only the
   benchmark language changes. Drop English MC from the base matrix; keep ceval
   as a tripwire with z-values.
6. **Sample size is a hard constraint: at a 50% floor, resolving 2 points needs
   ≈4900 pairs per dimension. BLiMP/CLiMP's 1000/dimension resolves ≈4.4 points.**
   Our 277-pair prototype (eval/base_matrix.py) is a construction de-risking,
   not a measuring instrument.

## 1. Which metrics have resolution at sub-1B, and why

### What the big suites actually do

Pythia (Biderman et al. 2023, arXiv 2304.01373) scores base checkpoints on
**LAMBADA, PIQA, WinoGrande, WSC, ARC-easy/challenge, SciQ, LogiQA — all
zero-shot likelihood** (continuation log-probability, no generation), plus
perplexity. OLMo and SmolLM use the same family: zero-shot likelihood
commonsense MC + perplexity, SmolLM evaluating every 2B tokens of training.
Nobody in this lineage scores a base model with generation. That is not an
accident: generation measures instruction compliance, which a base model does
not have.

### What still moves below 400M

- **Perplexity / domain loss.** Continuous everywhere. Ours: 1.66–1.99 nat
  across the 0.2B→3.24B ladder.
- **LAMBADA-style last-word prediction.** The target is the actual next word;
  scoring is pure likelihood over a constrained target. Pythia-70M already
  reads 0.24–0.27 on LAMBADA (Figure 6) — real signal at 70M, where its MC
  commonsense is at chance. LAMBADA's floor is near zero (open vocab), so even
  small absolute readings are above floor.
- **Minimal-pair acceptability (BLiMP/CLiMP).** Binary comparison, floor 50%.
  BLiMP's finding (Warstadt et al. 2020): linguistic knowledge is detectable
  via acceptability judgments at sizes and training steps where task accuracy
  is at floor — syntax comes early. CLiMP: LSTMs sit moderately above chance,
  Chinese BERT reaches 81.8%.
- **Token-level logprob on the target answer** (Schaeffer et al. 2023,
  arXiv 2304.15004): score the log-probability of the correct answer string
  against matched wrong ones instead of exact-matching generated output. This
  turns a discontinuous metric continuous and is the single most transferable
  trick in the paper.

### Why these and not generation/MC

Schaeffer et al. show "emergent abilities" are largely a metric artifact:
smooth, dense metrics (token logprob, acceptability) improve continuously with
scale and training; discontinuous metrics (exact match, pass@1) create the
appearance of a phase transition. Two corollaries for us:

- **Floor height sets when a metric leaves the floor.** Generation exact-match
  floors at ~0%; 4-choice MC at 25%; minimal pairs at 50%. At 200M, only the
  last is within reach. Our MC record (3/5 pinned at 25%) is the expected
  outcome, not a model defect.
- **A zero from a generative eval carries no information** about whether the
  capability exists. We proved this on ourselves: math-hard v2 read 0/1080 on
  ckpt_k8_proc_replay while the same checkpoint scored 2.8% on math-500 — the
  zero was degeneration, not measured failure (facts/base_eval.json#be.probe_degeneration).

## 2. Chinese minimal pairs: how to build them so they hold

### Prior art

- **CLiMP** (Xiang et al. 2021, arXiv 2101.11131, github.com/beileixiang/CLiMP):
  16 syntactic contrasts across 9 Mandarin phenomena, **1000 minimal pairs per
  contrast**, 95.8% human agreement. Semi-automatic generation from grammar
  templates over an annotated vocabulary (3,456 words, 84 features), with
  lexical/syntactic/semantic constraints per paradigm and **matched sentence
  length within each pair**. One paradigm (coverb-direction) was *discarded*
  because human agreement fell below 85%.
- **ZhoBLiMP** (2024, arXiv 2411.06096): a newer systematic Chinese BLiMP
  extension. **MultiBLiMP** (2025, arXiv 2504.02768): massively multilingual
  minimal pairs. de should pull CLiMP's 16 paradigms as the backbone rather
  than hand-rolling dimensions — the human validation is already done.
- CLiMP's 9 phenomena: classifier–noun agreement, verb complement selection,
  bǎ construction, coverbs, NP head finality (relative-clause order), binding,
  filler-gap dependencies, anaphora/agreement, plus one more family in the
  repo (16 paradigms total). Models do best on classifier–noun agreement and
  verb complement selection (local selectional restrictions); worst on bǎ,
  binding, and filler-gap (hierarchical/long-distance syntax).

### Construction rules (de's spec)

1. **One controlled edit per pair.** CLiMP enforces matched character length;
   we additionally enforce **matched tokenization** — BPE merges across the
   edit boundary break the comparison. In our prototype, 22% of first-draft
   pairs failed this: `他在` merges into one token while `他再` does not
   (在/再 dimension is unbuildable with pronoun subjects); `吃了` merges in
   `我吃了饭` but not in the permuted order. Rule: encode both sentences,
   assert equal length; for substitutions, the differing positions must form
   one contiguous span; for permutations, the two token multisets must be
   identical. Skip failures and report the skip rate — a high skip rate means
   the dimension's surface forms fight the tokenizer and needs redesign, not
   silent scoring.
2. **Score = sum log-probability of the full sentence, teacher-forced.** Pairs
   are length-matched, so raw sums are comparable — no length normalization
   (it dilutes the one-point difference). Correct = p(well-formed) > p(ill-formed).
3. **Known-answer validation, non-negotiable (repo rule, commit 38af944):**
   swap the labels and the score must invert — target ≥60 points between the
   two readings on the strongest checkpoint. A metric that cannot produce both
   a high and a low reading is uncalibrated.
4. **Human spot-check per dimension.** CLiMP discarded a paradigm below 85%
   human agreement; our factual/numeric dimensions need the same discipline —
   a "fact" the annotator is unsure about is a broken pair.
5. **Dimension catalog for zh** (CLiMP backbone + our additions): word order
   (SVO/SOV), classifier–noun agreement, function words (的/地/得, 把/被),
   aspect (了/着/过), bǎ construction, binding (自己), filler-gap/relativization,
   coverbs, verb complements, negation (不/没), comparatives (比/没有/不如),
   numeric consistency, factual consistency. The last two are ours, not
   CLiMP's — they need the human validation CLiMP's paradigms already have.

### Sample size — the number that decides the design

At a 50% floor, SE = √(0.25/n):

| n per dimension | SE    | resolves (80% power) |
|---|---|---|
| 100 | 5.0% | ~14 pt |
| 277 (our prototype) | 3.0% | ~8.4 pt |
| 1000 (BLiMP/CLiMP) | 1.6% | ~4.4 pt |
| 4900 | 0.7% | **2 pt** |

To resolve 2 points at 80% power you need ≈4900 pairs per dimension
(n = ((z_α + z_β)·0.5/0.02)²). CLiMP's 1000/contrast is the proven operating
point — it resolves ~4.4 points, which is enough to track a 16× data ladder.
Our 277-pair prototype (eval/base_matrix.py, 5 dimensions, self-test passing,
pushed to the pod) de-risks construction and tokenization only; it is not a
measuring instrument. Either scale to ~1000+/dimension or report at the
resolution the n supports — never report a 2-point movement on n=100.

## 3. What lights up between "loss moves" and "capability appears"

The literature's answer is uncomfortable for the question as posed: **the gap
is mostly the metric, not the model.**

- Schaeffer et al. 2023: re-score "emergent" tasks with smooth metrics
  (token-level logprob, BLiMP-style) and the phase transition disappears —
  improvement is continuous from early training. The middle period is filled
  by token-level likelihood on the target capability, moving the whole time.
- BLiMP: syntactic knowledge is present well below task-performance thresholds.
- Pythia: LAMBADA moves at 70M; MC commonsense needs ~410M; in-context
  learning later still.

The synthesized ordering: **domain loss → token-level target logprob /
acceptability → LAMBADA-style constrained prediction → MC commonsense →
few-shot ICL → generation.** Each step is a real capability (recognition
before production), not just a softer metric — but the metric is what makes it
visible.

**Practical consequence for us:** for every capability we eventually want
generation for (math reasoning), build its likelihood twin now — score the
log-probability of the correct solution/answer against matched wrong ones,
rather than generating. math-hard v2 can be rescored this way without touching
the data: for each problem, compare p(correct solution) vs p(matched wrong
solution) under the base model. That metric should move across the 16× ladder
at 200M, giving a continuous math-capability signal years (in model-time)
before generation works. The caveat to record: likelihood measures
recognition, not production — necessary, not sufficient; a model can prefer
the right answer and never produce it.

## 4. Is the MC floor the model or the suite?

Mostly the suite, for the three English benchmarks.

- **Same model, same scale, only the benchmark language changes: ceval (Chinese)
  moves, the four English ones don't.** That isolates the language of the
  benchmark as the cause, not model scale.
- A Chinese-pretrained model on English MC measures English exposure, which is
  ~0 by corpus design. Two confounds stack: size (MC is near chance below
  ~400M even in-language — Pythia/SmolLM) and language (English text fragments
  into byte-pieces under a zh-optimized 32K vocab; the model has minimal
  English data).
- Supporting literature: Pythia's BLOOM analysis — the "curse of
  multilinguality" is benchmark-dependent (BLOOM underperforms on LAMBADA,
  PIQA, WSC but not on WinoGrande, ARC, SciQ, LogiQA). Multilingual models
  underperform on English-heavy benchmarks; the benchmark language is part of
  the measurement.
- **Recommendation:** drop English MC from the base-model matrix — it measures
  a training-data decision (no English) we already made. Keep ceval as a
  Chinese regression tripwire, reported as z over chance, not as capability.
  And note even ceval is near floor at 200M for the size reason: the
  likelihood metrics in §1 are the primary instrument; MC is the tripwire.

## What changes (the delta)

1. The matrix de builds is **likelihood-only**: CLiMP-backbone minimal pairs
   (≥1000/dimension), LAMBADA-zh if available, domain loss, token-level
   target-logprob for math. No generation for base checkpoints.
2. **English MC leaves the base matrix**; ceval stays as a z-value tripwire.
3. **math-hard v2 gets a likelihood twin** (p(correct solution) vs matched
   wrong) as the continuous math signal at 200M.
4. **Known-answer swap validation and human spot-checks are gates**, not
   nice-to-haves (tokenizer_report lesson, CLiMP's discarded paradigm).
5. Our eval/base_matrix.py (277 pairs, 5 dims, tokenization-alignment filter,
   swap mode, self-test) is a construction reference for de — the tokenizer
   merge failure modes (他在, 吃了, 这本书) are already found and coded around.

## Sources

- Xiang et al. 2021, CLiMP, arXiv 2101.11131 — github.com/beileixiang/CLiMP
- Warstadt et al. 2020, BLiMP
- ZhoBLiMP 2024, arXiv 2411.06096; MultiBLiMP 2025, arXiv 2504.02768
- Biderman et al. 2023, Pythia, arXiv 2304.01373
- Schaeffer et al. 2023, Are Emergent Abilities a Mirage?, arXiv 2304.15004
- SmolLM (HuggingFace, 2024) — eval every 2B tokens; 135M/360M numbers in
  Figure 12/14 of the paper/blog, not transcribable from the sources fetched
- Our measurements: facts/base_eval.json
