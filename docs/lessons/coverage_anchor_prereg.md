---
question: "is the anchored agreement measured on the best-behaved 19% representative of the other 81%"
status: open
source: "e1, 2026-09-01, written BEFORE runs/coverage_anchor.json existed; fb's coverage order"
---

# Pre-registration: the other 81%

> **`status: open` means pre-registered.** `runs/coverage_anchor.json` is absent
> as of this commit.

## The problem with the number it tests

`be.preamble_is_not_the_cause_of_the_collapse` reports anchored agreement of
0.2609 (shift) / 0.4524 (LCS) — on **19 of 100** generations. Those 19 are the
ones that produced a recognisable code anchor, i.e. **selected for having
produced code at all**. Every anchored figure is therefore computed on the
best-behaved fifth of the set.

This is not a caveat on the number. It is a reason the number cannot be
generalised, and it makes the 81% the population of interest.

## Why a better anchor, rather than more prompts

The current anchor is a regex: a ` ```python ` fence, or a line opening `def `,
`class `, or an assignment. A generation that produces correct code with none of
those markers is invisible to it, so **coverage is a lower bound on "reached
code", not a measurement of it**. Adding prompts multiplies the same blind spot.

So the first question is how much of the 81% is genuinely code-free versus
merely unmatched by my regex.

## What is measured

`ckpt_pretrain_30b_s2.pt.step24000`, the same 100 code prompts, greedy,
`rep_stop=False`:

1. **Coverage under a widened anchor.** Add: indented continuation lines,
   `import `/`from `, `return `, `print(`, `for `/`while `/`if ` at line start,
   and any line containing `=` with balanced brackets. Report new coverage.
2. **Agreement on the newly-anchored generations, reported separately** from the
   original 19. Pooling them would let the well-behaved fifth carry the average
   again, which is the defect being tested.
3. **What the still-unanchored generations contain**, by inspection of a sample
   — the honest answer to "is the remainder code-free or regex-invisible" is
   read, not inferred.

## Falsification, fixed now

Let **A19** = 0.2609 (shift-aligned, original anchored subset) and **Anew** =
the same statistic on generations anchored only by the widened rule.

| observation | reading |
|---|---|
| Anew ≥ 0.40, and coverage rises above 50% | **the drift reading weakens** — the unanchored majority was doing better than the subsample suggested |
| Anew ≤ 0.20 | **bin-1 collapse is understated** — the favourable subsample was flattering the model, and the true picture is worse than reported |
| 0.20 < Anew < 0.40 | the subsample was roughly representative; drift stands as measured |
| coverage stays below 30% even widened | **the remainder is genuinely code-free**, which is itself the answer: the model mostly does not emit code, and agreement on the minority that does is not the interesting statistic |

**The last row is the outcome I expect**, and I am naming it in advance because
it is the one that makes the whole anchored line of work a side quest rather
than a result. If the model simply does not produce code for 80% of code
prompts, then "how well does its code align with gold" was never the question.

## Sizing the instrument, not just naming the confound

The lesson from ±4 applies again, so, concretely:

- the widened anchor is **checked against the raw text of the misses** before
  any agreement number is computed, so I find out whether it is still blind
  rather than assuming coverage equals reality;
- agreement uses the shift-aligned statistic to ±150 **and** the alignment-free
  LCS, since the naive one is known to report near-zero on offsets;
- newly-anchored and originally-anchored are **never pooled**.

## What I will report

Coverage before and after, Anew and A19 side by side, a read of what the misses
actually contain, and the falsification row it lands in. **Not a verdict** — fb
rules.
