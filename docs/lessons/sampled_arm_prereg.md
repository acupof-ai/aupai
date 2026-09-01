---
question: Is the 0.0% on math-500 and code-500 absent capability, or capability hidden behind a greedy decoding pathology?
status: open
source: pre-registered 2026-09-01 before any sampled score was run; e1's self-repeat measurement (0.747 median greedy vs 0.088 sampled); fb's assignment
---

# Pre-registration: the sampled arm of math-500 and code-500

Written before the runs. Nothing below is adjusted after seeing a number; an amendment
carries its own date and says what it revises.

## The question

Every generative 0.0% this project has recorded was measured under greedy decoding.
e1 measured why that might matter: median self-repeat 0.747 greedy against 0.088
sampled, max repeated run 175 tokens against 9, and 74% of greedy generations loop
while the remaining 26% are fluent and on topic. A model that loops on three quarters
of its greedy generations scores near zero whether or not it can solve the problems.

So the 0.0 is compatible with two states of the world, and they call for opposite
decisions:

| | capability absent | capability hidden |
|---|---|---|
| what 0.0 means | the model cannot do the task | the decoder destroys answers the model has |
| what to do | change the data or the objective | change the decode, then re-measure everything |

Nothing on the board separates them today.

## What is being run

`ckpt_pretrain_30b_s2.pt.step24000` (the 22B pin) and
`ckpt_pretrain_30b_s2.milestone_16b_step17500.pt`, on math-500 and code-500:

- temperature 0.8, k=8, reporting pass@1 (greedy) and pass@8 (any-of-8 sampled)
- `rep_stop` OFF, so the metric is not computed over a prefix the harness truncated
- `--force`/`--run` so the rescore is possible at all (fixed in 616205b)

Two arms per checkpoint, because pass@1 and pass@8 answer different halves. pass@1 is
still greedy by construction and is the control; pass@8 is the treatment.

`rep_stop` deserves its own line. It stops a generation when a whitespace 8-gram
repeats three times, which is exactly the degenerate case — so under greedy it fires
on the majority of generations and the recorded text is a truncated prefix. Every
degeneration rate quoted today was measured over that truncation (e1). Leaving it on
would confound the treatment with a decode-time intervention that fires at different
rates in the two arms.

## Pre-registered readings

Thresholds first, in absolute points on 500 problems. Binomial se at p≈0 is under
1pt, and the readable-move convention for generative metrics here is 12.6pt
(`NOISE_THRESHOLDS`, code_500). I use two levels because the interesting result is
not near the noise floor:

**A. Capability was hidden.** pass@8 ≥ 5.0% on either metric, at either checkpoint.

Thirty or more correct answers out of 500 cannot be produced by a model with no
capability, whatever the decoder does. If this lands, the 0.0 was a decoding artifact,
every generative number on the board is uninformative rather than negative, and the
first thing after is a decode fix, not a data change.

**B. Capability is absent.** pass@8 < 1.0% on both metrics at both checkpoints.

Five or fewer of 500, with eight independent samples per problem and no repetition
stop. Eight draws is a large amount of slack; a model holding a solvable distribution
would find one. If this lands, the greedy 0.0 was reporting the truth and the
duplication and data hypotheses keep their full weight.

**C. Neither.** 1.0% ≤ pass@8 < 5.0%.

Real but marginal. This is the outcome I expect to be hardest to act on and I am
naming it in advance so it does not get rounded to whichever neighbour is convenient.
It would mean capability exists at the edge of resolution: 500 problems cannot
separate 2% from 4%, so the next step would be more problems, not more interpretation.

## Pre-registered secondary readings

- **pass@8 − pass@1 ≥ 15pt** is the project's existing RL gate. If it fires, the
  sampled/greedy gap is not merely a measurement artifact — it is a usable training
  signal, and the RL path becomes live rather than blocked.
- **Degeneration rate with `rep_stop` off** is a different number from the one on
  record and must not be compared to it. Recording both arms so the truncation's size
  is measured rather than assumed.
- **16B vs 22B.** If pass@8 moves between them, capability is accumulating and the
  greedy metric was blind to it. If it does not, the checkpoints are equivalent on
  this axis regardless of which reading above holds.

## What this cannot answer

The sampled arm cannot distinguish "the model knows the answer" from "eight draws
found it by chance". At 4-way-equivalent guessing on free-form math the chance floor
is not zero but is very small; on code-500 a sampled program that passes tests is
strong evidence. pass@8 is an upper bound on capability, not an estimate of it — the
number that matters for deployment is pass@1 under a fixed decode, and that is a
separate measurement once a decode is chosen.

It also cannot say anything about the duplication question. A model that memorised its
training data samples memorised text; pass@8 on held-out problems is not contaminated
by that, but neither does it speak to it.

## Falsification

If pass@8 comes back exactly 0.0% on all four cells, that is reading B at its
strongest, and it also falsifies the hypothesis that motivated this run. Recording
that in advance: a null here is a real answer, not a failed experiment.
