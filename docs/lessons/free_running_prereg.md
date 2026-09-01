---
question: "how far does per-token gold agreement fall when the model conditions on its own prefix instead of the gold's"
status: open
source: "e1, 2026-09-01, written BEFORE runs/free_running.json existed; fb's order after t66"
---

# Pre-registration: the free-running arm

> **`status: open` means pre-registered.** `runs/free_running.json` is absent as
> of this commit; `git log` on this file against that path is the ordering proof.

## The gap this closes

`be.gold_is_ranked_high_but_not_reachable_by_greedy` measured gold token top-1
at **72.7%** (code) and **69.3%** (math) — teacher-forced. Every position was
scored given a *correct prefix the model would not have produced*, so that
number answers "can it continue a correct answer", not "can it produce one". It
is an **upper bound** on free-running accuracy, and it biases toward the
decoding-deficit conclusion it produced.

This arm measures the same quantity under the model's own prefixes.

## What is measured

`ckpt_pretrain_30b_s2.pt.step24000`, 200 code-500 + 200 math-500 problems,
`rep_stop=False`:

- **free-running per-token agreement** — generate from the prompt at t=0.8, then
  at each position compare the generated token against the gold token at that
  index. Prefix-aligned, so a single insertion desynchronises the rest; that is
  the honest comparison and its weakness is stated below.
- **the teacher-forced number on the same problems**, recomputed here rather
  than quoted, so the two come from one run and one prompt set.
- **the corrected version of my vacuous row**: gold sequence log-prob against
  each *sampled* sequence's log-prob, over k=8 samples. This is the test the
  argmax comparison could never be — a sampled sequence is not the arg-max, so
  gold can genuinely beat it.

## Falsification, fixed now

Let **TF** = teacher-forced top-1 (72.7% code, 69.3% math) and **FR** =
free-running agreement.

| observation | reading |
|---|---|
| FR ≥ 0.60 (within ~0.12 of TF) | the decoding reading **holds**; the model's own prefixes do not destroy its distribution |
| FR ≤ 0.30 (less than half of TF) | **"decoding and search" understates the problem** — the model's own prefixes destroy the distribution, and error compounding is self-reinforcing rather than merely multiplicative |
| 0.30 < FR < 0.60 | partial; report the number and the ratio FR/TF, no verdict |
| gold log-prob beats ≥ 25% of sampled sequences | the model **prefers** gold to what it samples — a real search failure, on the corrected test |
| gold log-prob beats ≤ 5% of sampled sequences | the model prefers its own output; the deficit is not search alone |

**I expect FR well below TF** — the whole reason teacher forcing is a confound
is that it should be optimistic. What I do not know is whether the drop is
modest or catastrophic, and those license different work: modest means better
decoding is worth building, catastrophic means the sequence model is the problem
and no sampler rescues it.

## Two ways this measurement misleads

- **Prefix-aligned comparison punishes a correct answer phrased differently.**
  If the model emits one extra token early, every later position is compared
  against the wrong gold index and agreement collapses to near chance even for a
  good generation. **So a low FR is ambiguous between "wrong tokens" and
  "shifted tokens", and I cannot separate them with this design.** I will report
  the best alignment over small shifts (±4 tokens) alongside the naive one; if
  they differ a lot, the naive number is measuring desynchronisation.
- **t=0.8 is not greedy.** The 0.0 generative scores were greedy; sampling at
  0.8 is a different decoder. Comparing FR-at-0.8 against TF is therefore not
  quite like-for-like, and I run greedy free-running too so at least one arm
  matches the decoder that produced the zeros.

## What I will report

FR and TF from one run, their ratio, the shift-tolerant variant, the corrected
sampled-sequence comparison, per set. **Not a verdict** — fb rules.
