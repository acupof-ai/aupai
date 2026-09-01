---
question: "on generations that actually reach code, is the loss still immediate at bin 1"
status: open
source: "e1, 2026-09-01, written BEFORE runs/positional_anchored.json existed; fb's order after the coverage correction"
---

# Pre-registration: bin 1 on the generations that reach code

> **`status: open` means pre-registered.** `runs/positional_anchored.json` is
> absent as of this commit.

## The gap

`be.the_loss_is_immediate_not_gradual` measured bin-1 agreement at **0.250**
(code) over all 100 generations. But `be.the_anchored_subsample_was_flattering_
and_the_anchor_was_blind` established that **31 of those 100 are degenerate
loops with no code at all** — read, not inferred.

So the 0.250 is an average over a population a third of which was never
attempting the task. **A degenerate loop scores near zero at every bin, which
drags bin 1 down for a reason that has nothing to do with distribution drift.**

The mechanism claim deserves the clean comparison: bin 1 on the 69 generations
that actually produce code-shaped output.

## What is measured

`ckpt_pretrain_30b_s2.pt.step24000`, the same 100 code prompts, greedy,
`rep_stop=False`, shift-aligned to ±150. Positional bins 1–8, 9–16, 17–32,
33–64, 65–128, computed on three populations, **reported separately and never
pooled**:

- the **original 19** (old anchor)
- the **newly-caught 50** (widened anchor only)
- the **31 unanchored** — included precisely to confirm they behave as the
  degenerate-loop reading predicts, rather than being assumed to

## Falsification, fixed now

The mechanism test is unchanged in form: bin 1 low **and** the
teacher-forced-to-bin1 drop exceeding the bin1-to-last decay.

| observation on the anchored 69 | reading |
|---|---|
| bin1 ≥ 0.55 | **immediate-loss claim collapses** — it was an artifact of averaging over degenerate loops, and among real attempts the model tracks gold well before diverging |
| bin1 ≤ 0.40 and TF→bin1 drop > bin1→last decay | **immediate loss confirmed on real attempts** — the mechanism survives the correction that killed the coverage denominator |
| 0.40 < bin1 < 0.55 | partial; report all three populations, no verdict |

**This can overturn my own mechanism claim and that is the point of running it.**
If bin 1 on real attempts is high, then "the trajectory is wrong from token one"
is wrong — the trajectory is fine for a while on the generations that try, and
what I measured was the loop population.

I do not expect that, because 0.250 over all 100 with 31 near-zero contributors
implies roughly 0.36 over the remaining 69 even if the loops score exactly zero
— still well inside the immediate band. **Writing the arithmetic down now so I
cannot present a value near 0.36 as a surprise either way.**

## The limit, unchanged and unsoftened

Positional bins are not independent per-position accuracy. Once a generation
diverges, every later bin is scored on an already-off trajectory, so late-bin
figures mean "accuracy given N tokens of prior divergence". Bin 1 is the only
bin conditioned on a mostly-correct prefix and the only one comparable to the
teacher-forced number. That was true of the previous profile and it is true here.

## What I will report

Three bin curves, the TF comparison for bin 1 only, and the falsification row.
**Not a verdict** — fb rules.
