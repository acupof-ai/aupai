---
question: "is the gold answer reachable under sampling, or is rising gold probability trapped below the sampling threshold"
status: open
source: "e1, 2026-09-01, written BEFORE runs/gold_reachability.json existed; fb's last question of the day"
---

# Pre-registration: is the gold reachable?

> **`status: open` means pre-registered.** `runs/gold_reachability.json` is absent
> as of this commit; `git log` on this file against that path is the ordering
> proof.

## The question, and why the previous answer forces it

`be.gold_bpb_falls_while_generation_scores_zero` established that the model
assigns rising probability to correct answers — code BPB down 15.6% across the
ladder while generative `code_500` sits at 0.0. It also carries the boundary
that makes this measurement necessary: **BPB scores a fixed gold string, so a
model could assign rising probability to one phrasing and still produce nothing
usable.**

So: **does the rising probability convert into anything reachable?**

Two deficits with the same symptom and opposite fixes:

| if | the deficit is | the fix is |
|---|---|---|
| gold probability rises but the model never samples it | **decoding and search** | temperature, top-p, beam, best-of-n, a sampler change |
| gold is sampled but wrong answers dominate | **knowledge** | more tokens, better data, SFT |

These are not adjacent conclusions. One says the model has it and we cannot get
it out; the other says it does not have it.

## What is measured

For each of 200 code-500 and 200 math-500 problems, on
`ckpt_pretrain_30b_s2.pt.step24000`:

- **gold rank** — where the gold's per-token probability sits against the
  model's own distribution at each gold position, summarised as the fraction of
  gold tokens in the model's top-1, top-10, top-100.
- **gold mass** — the sequence probability of the gold, and the same for the
  model's greedy continuation, so the two are directly comparable on one scale.
- **sampled reachability** — over `k=32` samples at t=0.8, whether the gold
  string is ever produced verbatim, and the best per-token agreement any sample
  achieves with the gold.

All arms `rep_stop=False`, per `be.rep_stop_truncates_the_thing_it_measures`.

## Falsification, fixed now

| observation | reading |
|---|---|
| gold top-1 fraction ≥ 0.50 **and** gold never sampled in k=32 | **decoding deficit** — the model ranks gold tokens first and sampling still cannot assemble the string |
| gold top-1 fraction ≤ 0.15 | **knowledge deficit** — the gold is not near the top of the distribution and no sampler will find it |
| gold sequence probability ≥ greedy's on ≥ 30% of problems | the model *prefers* gold to what it emits, which is a decoding/search failure by definition |
| gold sequence probability < greedy's on ≥ 90% of problems | the model prefers its own wrong output; knowledge deficit |
| 0.15 < top-1 < 0.50 | **no verdict** — report the distribution; this is the outcome I expect and it is not a failure of the measurement |

**I expect the middle band.** Saying so in advance because a measurement whose
predicted outcome is "no verdict" is still worth running — it bounds the split
even when it does not resolve it — and because predicting the ambiguous outcome
beforehand is the only way to stop myself narrating a clean one afterwards.

## Two ways this measurement misleads

- **A long gold cannot be sampled verbatim at any temperature.** Sequence
  probability falls geometrically in length, so "never sampled in k=32" is close
  to guaranteed for a 200-token gold regardless of the model. The **per-token
  rank** is therefore the load-bearing statistic and the verbatim-sample count is
  a sanity check, not evidence. If I report the verbatim count as the finding, I
  have measured gold length.
- **Teacher forcing.** Ranks are computed with the gold as context, so each
  position is scored given a *correct* prefix the model would not have produced.
  This measures "can it continue a correct answer", which is strictly easier
  than "can it produce one" and **biases the result toward the decoding-deficit
  conclusion**. It is the standard confound of this metric and it is why a high
  top-1 fraction alone cannot settle the split.

The second is serious enough that I will report the free-running agreement
alongside, and if the two disagree the teacher-forced number is the one to
distrust.

## What I will report

Distributions per set, the two confounds' magnitudes where measurable, and which
falsification row the numbers land in. **Not a verdict** — fb rules.
