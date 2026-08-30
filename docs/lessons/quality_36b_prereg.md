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

## Reasoning-model direction (2026-08-30, fb): execution is the filter

**Objective change: the deliverable is a reasoning model (coding + math at ~30B tokens), not a scaling law.** For math/code the no-universal-criterion constraint does not apply — answers are verifiable by execution, which has perfect precision and recall for the property that matters. The scorer funnel (Stage 1/2) and the 2%/5% prevalence bands are **superseded** for math/code: they priced a scorer against a general corpus. They survive unchanged for the general-text remnant (problem statements), which will be a much smaller share (3b re-deriving).

**Verification-as-pipeline, pre-registered before first measurement:**

The interesting question is not "which documents are good" but "which generated solutions survive execution." Protocol (first target: `blossom-math-v4-10k`, the synthetic math batch on disk; checker: `scripts/eqcheck.py`, the repo's own arithmetic-step verifier):

- **coverage** = docs with ≥1 verifiable equation / total docs
- **survival** = docs with ≥1 checked equation AND zero bad steps / total docs
- **unverifiable** = docs with zero checked equations — reported separately, NOT counted as survived (a vacuous pass is not a pass)
- bad-step rate per checked equation, steps per doc

**Decision shape (economics, costs from de):** the generation-filter loop costs `generation_cost_per_doc / survival` per kept doc. It is viable iff that beats the alternative cost per kept doc. I own the survival number; de owns the cost inputs — the economic call is not made without both. A survival under ~5% means the loop needs >20× generation overshoot and is likely infeasible at 30B regardless of cost; that band is a judgment, not a derived line, and is flagged as such.

**MEASURED 2026-08-30 (blossom-math-v4-10k, the synthetic batch on disk): the instrument is the result.**
- `scripts/eqcheck.py` (arithmetic-step, the repo's own checker): coverage 21.6%, and its flagged equations are ~all false positives — the chain detector is broken by Chinese units/variables between operands ("1600元 × 3/10 = 480" mis-read as "3/10 = 480"). Conditional "survival" 91.65% is instrument noise, not batch quality. **Arithmetic-step checking is the wrong instrument for Chinese synthetic math** (`dq.verification.eqcheck_blind_spot`).
- Answer-level checking (output final answer vs the record's gold `answer` field, extraction pre-registered): **99.85% raw agreement, 99.91-99.97% after manual review of the 15 disagreements** (~6 extractor artifacts, ~9 genuine — mostly integer rounding of exact fractions, ≥1 real misread). The batch is clean (`dq.verification.blossom_gold_agreement`).
- **Pipeline implication:** the verification instrument must be answer-level, and synthetic batches must retain the gold answer; free-text generations need an extraction contract — `\boxed{}` (math_short_v10's format) is the existing extractable target. The survival number is only meaningful once the checker is named — "survives its own checker" has no single answer.

**Contamination (fb's item 2, mine to run):** math/code benchmarks leak, and we already retired math-hard v1 for self-contamination. Protocol: MinHash the eval sets (GSM8K-zh on disk; the math-hard successor once named) against the math/code corpus AND against any generation prompt set, BEFORE generation at scale. Zero-tolerance rule pre-registered: **a batch that overlaps the eval set at all is rebuilt** — one leaked problem is one inflated eval point. The MinHash instrument (`cont.minhash_known_answer`, 296/296 recall, 0/72 FP) is reused as-is.

**Ordering requirement (fb 2026-08-30, stronger than rebuild):** the eval slice is carved **before the corpus fingerprint is stamped** — the same ordering that made the LAMBADA-zh holdout verifiable by two independent scans. Retrofitting a holdout after a fingerprint means the fingerprint describes a corpus that no longer exists. Instrument: `scripts/scan_math_contamination.py` (containment = |holdout bigrams ∩ row bigrams| / |holdout bigrams|, zh char-bigrams, threshold 0.8, min 20 bigrams; scale-free per-GB / per-million-doc rates with a same-scale in-training baseline — absolute per-shard counts are meaningless across unequal sizes, `cont.cci3_scale_failure`). Scan targets: **math-500 first** (L1's eval, decides whether reasoning output is measurable at 200M), GSM8K-zh next (on disk). The math-hard successor is not decided; b0's panel names it.

**MEASURED 2026-08-30 — math-500 vs web_hq: NOT contaminated, L1 stands.** 11 unweighted hits at 0.80-0.86, **0 exact** — all 11 are common-vocabulary false positives: short template word problems (24-28 bigrams) whose bigrams are generic education vocabulary, embedded in unrelated education-adjacent web docs. Signature: hits cluster on 6 of the shortest, most generic holdouts (439×2, 227×3, 299×4, 208, 192, 50) — real contamination spreads across holdouts; an instrument artefact clusters where the instrument is weakest (`cont.math500_webhq_fp_explained`). fb's verdict accepted.

**Scanner hardened (IDF-weighted containment, `cont.scanner_idf_weighting`):** the 0.8 threshold meant different quantities on a 25-bigram generic holdout and a distinctive one. Pass 1 accumulates document frequency per holdout bigram over the scanned corpus; pass 2 scores with idf = log((N+1)/(df+1))+1. Validated: 12/12 known FP pairs drop below 0.8 (max 0.789), verbatim injections stay at 1.000. Pre-registered 80%-prefix control failed (5/6 below 0.8) — the criterion was mis-specified: a doc missing the distinctive tail of a problem is not the problem. Known limitation: char-bigram format sensitivity (no-punct injections 0.49-0.89) — pre-existing, unweighted fails identically, separate upgrade queued. Residual FP class: mojibake docs share rare bigrams by coincidence and IDF upweights them (see GSM8K-zh).

**MEASURED 2026-08-30 — GSM8K-zh (train, 7,473 holdouts) vs web_hq: clean.** 1 hit / 1,366,324 docs = 0.2/GB, 0 exact. The 1 hit is a mojibake stock-analysis doc (Big5-as-GBK) vs a triangle-angle problem — false positive of the residual class above; the doc is a **format-chain miss** (mojibake should have been filtered at encoding), not leakage. Unweighted score 0.854 — the old scanner would have flagged it too. Action: format chain needs a mojibake/Big5-as-GBK detector (3b's data territory); scanner-side defense-in-depth option is an IDF cap, fb's call (`cont.gsm8k_zh_webhq_scan`). Test split not on disk — train serves as the canary; fetch test before GSM8K-zh becomes an eval.

**Template inversion (fb's item 3):** the 6.38× finding concerned prose templates (cosmopedia tutorials). Templated worked solutions (given/goal/steps/answer) are deliberately pedagogical — whether they help or hurt a reasoning model is open and is b0's training-outcome territory. Descriptive extension only: measure frame concentration in solution corpora the same way, no causal claim without a run.

## Scorer funnel (only if prevalence justifies it) — SUPERSEDED for math/code by the verification section above; stands for the general-text remnant

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
