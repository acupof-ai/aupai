---
question: What decides corpus quality at 36B, given that both current instruments — the regex chain and the quality head — are measured not to?
status: recorded
source: fb tasking 2026-08-30; facts/data_quality.json (positive_control_protocol, register.style_flip, cost.zero_shot_vs_distill), facts/multilingual.json#mlm.corpus.step01_hit_rate
---

# Quality at 36B: pre-registered decision (2026-08-30)

Question from fb: what decides quality at 36B, given both current instruments are measured not to? Mandate: pre-register the kill rule **before anything is fetched**.

## What the facts settle

- **Regex chain = format/encoding pass.** Precision 1.0, quality recall 3.4% home / 1.7% CCI3 (`dq.criterion.positive_control_protocol`, `dq.criterion.cci3_measured`). Near-perfect precision, near-zero quality recall.
- **Quality head = register proxy.** AUC 0.823 vs hand on web, but scores cosmopedia below raw web (`dq.quality_head.auc_vs_hand`); Wikipedia-style rewrite with facts preserved flips ~7% of FineWeb-Edu decisions, all 26 domains score higher after rewrite (`dq.register.style_flip`). Resemblance instruments measure register, not quality.
- **No universal criterion exists.** Three mainstream classifiers agree on 10.1% of docs as HQ (`dq.criterion.hq_intersection`); every criterion is a choice of target distribution (`dq.universal.criterion_absent`).
- **The corpus arrived clean without a working quality filter.** Census: Step 0/1 marginal capture 0.326% on web_hq, old chain already cleaned it (`mlm.corpus.step01_hit_rate`). Source selection was doing the work.
- **Corpus-independent rules (encoding, bytes, MinHash dedup) always valid; corpus-dependent rules necessarily fail on a new corpus** (`dq.criterion.rule_split`).
- **Zero-shot full-corpus scoring is not an option:** 11,000 H100-hours at 36B vs ~60 for distillation (`dq.cost.zero_shot_vs_distill`).

## Decision: source + format + dedup is the default; a scorer must earn its way in

Source selection is not the fallback, it is the null — the census is evidence it did the work. The scorer direction is justified only by a measured prevalence number on the new sources, decided by a rule written before the fetch:

**Pre-registered kill rule (measured on the FIRST fetched batch of each new source, before full fetch):**

1. Hand-read n=385 docs (stratified, post-format-chain; CCI3 usable/junk rubric — already validated, 3b knows it). n=385 gives ±0.05 at p=0.5, tighter at low p; n=30 flips (`dq.handread.small_sample_flip`).
2. Format-chain drop rate on the full batch (CPU, cheap).
3. Decision by CI:
   - **Upper 95% bound < 2% junk → NO SCORER BUILT (for the val-NLL question).** Physics, same derivation as the W/F kill: a scorer's maximum capture is the junk prevalence p; its val-NLL benefit is bounded by p × Δ_marginal (second-order, "not seen in training" effect), and at p=2% that bound is <0.02 nat < any MDE these seeds reach. A scorer that can touch <2% of the corpus has an effect ceiling below NLL resolution. Quality = source selection + format chain + MinHash dedup.
   - **Denominator caveat (fb 2026-08-30, accepted):** this rule is denominated in val NLL. The panel shows discrimination and generation coming apart by 74 points (lambada 90.4% vs open-acc@1 16.0%) under a single val NLL of 3.691 — a metric that cannot distinguish those two states is not obviously the right denominator for whether junk matters. 2% of 36B is ~720M tokens of junk; the derivation says that cannot move val NLL past noise, and says NOTHING about generation. If anything the correction runs stricter: NLL averages over all tokens while generation samples from the tail, so junk may shift production patterns more than average loss — the NLL rule is not a safe harbor for "junk doesn't matter." The generation-denominated question is therefore a SEPARATE decision (below), not covered by this rule.
   - **Lower bound > 5% → scorer funnel below.**
   - **CI overlaps [2%, 5%] → fb adjudicates** (pre-agreed band, not ad hoc). Stated now so it is not decided under pressure: no splitting the difference — the question is which side the panel's own uncertainty falls on; if genuinely ambiguous, default is **no scorer plus a second panel at larger n** (a scorer we cannot justify costs more than 385 more hand-reads).
4. Every subsequent batch of the same source gets the same measurement (standing duty: never assume a new batch matches the last). A batch that crosses 5% reopens the scorer question.

## The generation-denominated question (separate decision, open)

The NLL kill rule does not cover generation, and the W/F-style physics argument does not transfer: NLL's bound works because loss is a per-token average that dilutes a 2% shift, while generation samples from the tail and may amplify it. The generation-denominated scorer question has **no derived kill rule yet** and stays open under the same default (no scorer; the burden of proof is on the scorer side).

Pre-registered shape, numbers to be filled when b0 delivers:

1. **Resolution gate (b0):** does open-acc@1 have seed-level resolution at 200M? If not, generation cannot adjudicate anything at this scale and the NLL rule is all we have — clean outcome, rule stands as written.
2. **If it resolves:** the scorer question re-opens with generation's own σ and its own Δ_marginal. The 2% line may land elsewhere — direction unknown a priori (stricter if junk hurts generation more, looser if generation is noisier). Do not import the NLL number.
3. **Δ_marginal for generation has no nat-scale loose bound** — the a-fortiori trick from the W/F derivation is unavailable. The honest options when the time comes: a small junk-ablation training run to bound it directly, or accept that the generation case cannot be pre-registered with a number and decide by expert judgment with the default unchanged.

## Scorer funnel (only if prevalence justifies it)

- **Stage 1 — teacher (zero-shot 70B-class, ~1000 inferences, hours not 11,000):** score (a) style-flip pairs rewritten FROM the new source (facts preserved, register changed — the `style_flip` protocol applied to our own data) and (b) the hand-read panel. **Kill: labels flip >5% across register (teacher measures register) OR panel AUC < 0.70 (teacher cannot separate A_d/B_d at all — our 27B teacher sat at 0.739, so this bar is real).** Either kill → scorer direction dead; quality stays source+format+dedup.
- **Stage 2 — distillation (~60 H100-hours), per-register prompts and per-register validation** (the transferable practice from `dq.universal.criterion_absent`). **Kill: student flips >5% on HELD-OUT style-flip pairs, or student-teacher agreement < 0.8 on a held-out panel.** Never re-test on the calibration set (`dq.criterion.recall_ratio`).
- **Zero-shot scoring of the full corpus: never, at any prevalence.** 11,000 hours is not a slow option.
- The scorer, if built, inherits the ratio criterion WITH the absolute recall floor ≥0.5 (`dq.criterion.ratio_needs_floor`) — a do-nothing scorer must not read VALID.

## Synthetic-share hypothesis (my owned piece; b0 owns measurability)

Claim handed to 3b as *hypothesis, do not schedule against*: 10× more of the same mix moves knowledge but not 表达能力 — 53.8% chinese-cosmopedia (lowest domain loss, 2.79 vs web_hq 5.05) teaches pattern-matching over production.

What is measured: the 53.8% share (`mlm.corpus.en_domain_is_mostly_chinese` chain) and the loss gap. What is NOT measured: that templating causes either the loss gap or the discrimination/generation asymmetry (lambada 90.4% vs open-acc@1 16.0%).

**My CPU pre-registration (the premise, runnable now):** measure template concentration of cosmopedia vs web_hq — share of docs matching the top-100 sentence-frame patterns, and n-gram diversity. **If cosmopedia's concentration is <2× web_hq's, the "templated synthetic" mechanism is wrong and the hypothesis's premise fails** — the loss gap needs another explanation (cleanliness, domain, register) and the 表达能力 claim loses its mechanism. If ≥2×, the premise holds; the training-outcome claim still needs b0's runs.

**MEASURED 2026-08-30 (`mlm.corpus.cosmopedia_template_concentration`): PREMISE HOLDS at 6.38×.** Top-100 frame coverage 48.2% (cosmopedia) vs 7.6% (web_hq); df≥10 coverage 56.2% vs 7.6%; 8-gram diversity 0.853 vs 0.947. Half of cosmopedia docs are built from the same 100 sentence frames. The necessary condition for the mechanism is real; the consequence (knowledge vs 表达) is still b0's to measure.

**Prediction pre-registered for b0's design (if the premise holds):** at fixed token budget, reducing synthetic share should move generation metrics (open-acc@1) MORE than discrimination metrics (lambada). A mix ablation that moves both equally, or discrimination more, falsifies the mechanism.

## What I need

- Nothing until the first batch of a new source lands. Hand-read is 3b's shop (or mine if fb assigns).
- GPU: none for this design. Stage 1 needs zero-shot judge access (API-class, hours), Stage 2 ~60 H100-hours only if Stage 1 passes — I'll ask when the prevalence number justifies it.
