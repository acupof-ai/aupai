---
question: "does FoNE improve a 500M-class run enough to justify its cache, measured at 1B tokens"
status: open
source: "e1, 2026-09-01, written BEFORE runs/fone_ab.json existed; fb ruling: FoNE off in the main run, A/B on the lane"
---

# Pre-registration: the FoNE A/B

> **`status: open` means pre-registered.** `runs/fone_ab.json` is absent as of
> this commit; `git log` on this file against that path is the ordering proof.
>
> **Not started.** Cards 4/5/6 belong to the launch path until fb releases them.
> This document is written before the hardware exists, not before the result.

## What is being decided

FoNE is fully implemented in the pretraining path and **has never been used by
any experiment** — no recorded decision anywhere. The mechanism is real: BPE
splits `1640` into `16|40`, which is the exact failure mode for arithmetic.
`Cfg.fone=False` is a default, not a decision, and this converts it into one.

**Single variable.** Two runs at 1B tokens, `--fone` the only difference: same
seed, same data order, same mix, same commit, same cards. The `_fone` cache
suffix means the arms cannot share a cache, which is the one asymmetry and it is
in the data pipeline rather than the model.

## The noise bound, and why it is an upper bound

From `be.one_seed_family_misstates_its_own_noise_by_2x`: two independent
four-seed families at `mix_scale_0.2b` give **sd 0.0444 / spread 0.1076** and
**sd 0.0239 / spread 0.0495** on mean domain loss.

**I use the larger: spread 0.11.** Understating noise ships a null as a finding.

**The caveat is inside the rule, not beside it: those families are 0.2B-token
runs and this A/B is 1B.** Seed noise shrinks with tokens, so 0.11 is an
**upper bound** on the noise at 1B, not a measurement of it. That direction is
the safe one — a threshold calibrated against 0.2B noise is conservative at 1B,
so an effect that clears it clears it honestly — but **no one may quote 0.11 as
the 1B seed spread**, including me, in this document or downstream of it.

## The rule, in three bands

Let **Δ** = (no-FoNE mean domain loss) − (FoNE mean domain loss), positive
meaning FoNE is better.

| Δ | decision |
|---|---|
| **≥ 0.33 nat** (3 × 0.11) | **FoNE ships.** The effect clears a conservative bound on seed noise. |
| **≤ 0.11 nat** | **No detectable effect at this budget.** |
| **0.11 < Δ < 0.33** | **NOT DECIDABLE at this budget.** |

**The middle band is a result, not a failure to get one.** It means FoNE needs a
larger run to decide, and that is information: it bounds the effect from both
sides and tells the next person what a decisive experiment would cost.

> **To whoever reads this outcome later, including me: do not round the middle
> band to "no effect."** A null is easier to write and it is a different claim.
> If Δ lands between 0.11 and 0.33, the sentence is "not decidable at 1B", and
> the honest follow-up is a token count, not a verdict.

I expect the middle band. Saying so now so that landing there cannot be
presented as either a disappointment or a discovery.

## Secondary readouts, reported but not decisive

Digit-position accuracy on held-out arithmetic is the mechanism-specific signal
and it is **why** FoNE would work if it does. It is reported alongside Δ. It does
not override the rule: a digit-accuracy gain with Δ inside the noise band is
"the mechanism engages and the effect is below resolution", not a ship.

## Known ways this A/B misleads

- **The cache asymmetry.** `--fone` needs its own tokenised cache, so the arms
  differ in cache identity as well as model config. If the two caches differ in
  anything beyond `[NUM]` substitution, the comparison is not single-variable. I
  will verify token counts match between caches before scoring.
- **One answer format per prompt**, per the documented failure mode: multiple
  answer formats on identical prompts leave termination underdetermined.
- **`num_id` must be derived, not assumed.** Fixed in `aef7dbb` —
  `resolve_num_id()` reads `[NUM]` from the tokenizer and raises if absent. This
  A/B is the first thing that would ever have exercised the broken path.

## What I will report

Δ, both arms' domain losses, digit-position accuracy, the token-count check on
the two caches, and the band it lands in. **Not a verdict** — fb rules.
