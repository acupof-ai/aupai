---
question: "200M 模型上,一条 data scaling 曲线要怎么设计才成立"
status: open
source: "e1 literature review 2026-08-30; all numerical claims in facts/data_scaling.json (referenced as #<id>)"
---

# Data scaling curve design at 200M

Architecture fixed (KDA + MLA + AttnRes, blocks=4). Corpus 3.24B unique tokens, 166M non-embedding params = 19.5 tok/param (facts/data_scaling.json#ds.project_corpus). Every rule below is a number or a procedure; every number carries its measurement config. Literature measured at 1B+ is labeled **[extrapolation]**.

## P0 — Repetition in the data-constrained regime (decisive)

**Rule: repeat to 4 epochs. It is nearly free, and it is the only literature-measured way to multiply your corpus without buying data.**

- The fitted law (facts/data_scaling.json#ds.muennighoff_formula): each repeated epoch is worth ~94% of the previous pass; at the limit (R_D*≈15.4 epochs) repeated tokens average ~63% of fresh; **hard ceiling: effective data plateaus at ~16.4× the unique corpus** — 52B tokens, ~315 tok/param, no matter how many more epochs (facts/data_scaling.json#ds.muennighoff_per_epoch_decay).
- The 4-epoch point is directly measured: **+0.5% validation loss vs single-epoch** at 8.7B; downstream insignificant to ~4 epochs (facts/data_scaling.json#ds.muennighoff_four_epoch). 200M sits inside the fit range (7M–9B) but the +0.5% itself is an 8.7B measurement — **[extrapolation]** in the strict sense.
- Size interaction favors you: data-constrained frontier prefers *smaller models + more epochs* over larger models (R_N*=5.3 < R_D*=15.4; a 27%-smaller model beat Chinchilla's pick at 25B unique tokens) (facts/data_scaling.json#ds.muennighoff_size_epoch_interaction).
- Mitigations: no dropout test exists; reshuffling per epoch is standard (not ablated); code-mixing ~2× effective tokens for natural-language eval (facts/data_scaling.json#ds.muennighoff_mitigations).

**Your numbers:** 3.24B × 4 epochs = 13B seen ≈ 3.24 × 3.66 ≈ **11.8B effective ≈ 71 tok/param** at ~0.5% loss cost — vs 19.5 tok/param at 1 epoch. That is the single biggest lever in this document. Beyond 4 epochs, only continue while the curve is still buying loss; stop well before 16.

**Falsifier:** run 1/2/4/8 epochs at the 3.24B point. If 4 epochs costs >2% held-out loss (vs the fitted ~0.5%), the repetition prior is dead at 200M and buying data beats repeating. Signal at the end of one continuous WSD run (~26B tokens, ~16h on 6×H20 at 440K tok/s).

## P1 — Must every point run a full cosine anneal? (cost structure of the whole curve)

**fb's judgment (full cosine per point) is the Chinchilla/Kaplan protocol and the safe default. The literature supports a cheaper design, but only at 1B+ — it must be calibrated at 200M before adoption.**

| Option | Cost for a 5-point curve | Evidence | Scale |
|---|---|---|---|
| Full cosine per point | Σ budgets = 7.7B tokens | Chinchilla Approach 1: 4 horizons/N, 16× span, own cosine per run (facts/data_scaling.json#ds.chinchilla_approach1) | 70M–16B ✓ |
| WSD: one stable run + short anneal tails | max run + Σ tails ≈ 4.0B | MiniCPM WSD (decay 1.4% of total, exp 0.5^) (facts/data_scaling.json#ds.wsd_minicpm); SmolLM2 anneal 9% (facts/data_scaling.json#ds.smollm2_anneal) | 1.2–2.4B **[extrapolation]** |
| WSM: one constant-LR run + checkpoint merges | max run only, merges free | one run's merges at 2T–10T milestones "closely mirror" full 100B decay runs (facts/data_scaling.json#ds.wsm_proxy, facts/data_scaling.json#ds.wsm_mechanism) | 1.4B active MoE **[extrapolation]** |

Two supporting facts: WSD tails should use **moderate decay (ending LR ~1/3 of peak)** — aggressive decay wastes the tail data (facts/data_scaling.json#ds.lr_decay_waste, measured 1.5B **[extrapolation]**); and WSqD makes the peak LR horizon-independent so one peak LR serves all points (facts/data_scaling.json#ds.wsqd).

**Rule: calibrate before committing.** At two budgets (1B and 3.24B), run all three ways and compare held-out loss. Seed noise is ~0.05 loss (facts/data_scaling.json#ds.kaplan_noise). If WSD-tail and WSM-merge land within 2× seed noise of cosine, adopt WSM for the full curve (it is free) and the curve costs one run. If not, fall back to per-point cosine. Cost of the calibration itself: ~2 extra runs.

## P2 — Budget points

**Rule: geometric, ≥5 points, nested prefixes, smallest ≥ 250 steps.**

- Chinchilla's fixed-N protocol used 4 horizons over a 16× span (facts/data_scaling.json#ds.chinchilla_approach1). Three points (your current 0.3/1/3.24B) exactly identifies L(D)=E+B/D^β with zero residual degrees of freedom — it cannot show misfit. Five points can.
- Lower bound: Kaplan's fits broke at ~22M tokens / 40 updates per epoch (facts/data_scaling.json#ds.kaplan_min_regime). Your warmup is 20 steps = 15.7M tokens (facts/data_scaling.json#ds.trainpy_steps). A 0.2B point = 254 steps = 12.7× warmup and 6.3× Kaplan's breakdown floor. **0.2B is the smallest defensible point; do not go below 0.15B.**
- Proposed: **0.2 / 0.4 / 0.8 / 1.6 / 3.24B** (16.2× span, geometric). With WSD/WSM, extra points cost only checkpoints — if the calibration passes, add 0.15 and 2.4B for free.
- Fit target: L(D) = E + B/D^β with Chinchilla's β=0.28 as the prior (facts/data_scaling.json#ds.chinchilla_exponents), not a fixed value.

## P3 — Readout

**Rule: held-out loss, per-domain, ≥2 seeds at the extremes; math-hard is sanity-only.**

- Chinchilla fit on smoothed *training* loss under an infinite-data assumption (facts/data_scaling.json#ds.chinchilla_readout). That assumption is void the moment you repeat data — training loss is then biased by memorization. **Held-out loss is mandatory for the repetition axis.**
- Read per-domain (web / wiki / math), not merged: the domains have different weights in your mix and a merged loss hides which domain drives the curve. (Precedent: 14 Pile domains + Wikitext in the synthetic-data grid.)
- Seeds: run-to-run noise is ~0.05 loss (facts/data_scaling.json#ds.kaplan_noise). ≥2 seeds at min and max points anchors the noise floor; middle points single-seed is acceptable if the fit's residuals stay below it.
- math-hard resolves ±1.1pt at 2–3% (facts/data_scaling.json#ds.mathhard_resolution) — it cannot see the ~0.01–0.1 deltas a β-fit is made of. Record it at the final model only.

## Recommended design (one paragraph)

One WSD stable run on the 3.24B corpus to 8 epochs (26B tokens, ~16h on 6×H20), checkpoints at every epoch boundary and at 0.2/0.4/0.8/1.6/3.24B within epoch 1; per-point readout = WSM mean-merge of the last 12 checkpoints (calibration permitting, else 10%-of-budget anneal tails with ending LR 1/3 peak); held-out loss read per domain (web/wiki/math), 2 seeds at the extremes; fit L(D)=E+B/D^β on the epoch-1 points, and read the repetition axis as loss-vs-epoch at 3.24B with the Muennighoff formula as the fitted prior (facts/data_scaling.json#ds.muennighoff_formula). Buy data only if the 4-epoch cost exceeds 2% (facts/data_scaling.json#ds.muennighoff_four_epoch's falsifier).

## Open items this design cannot close

- WSD-tail and WSM fidelity at 200M (P1 calibration) — no sub-1B measurement exists for either.
- The repetition δ at 200M on Chinese/KDA+MLA — the fit is GPT-2/English (facts/data_scaling.json#ds.muennighoff_formula's boundary).
- Per-domain anneal weights — no controlled study anywhere (facts/data_scaling.json#ds.smollm2_anneal's boundary); inherited-and-unmeasured.
