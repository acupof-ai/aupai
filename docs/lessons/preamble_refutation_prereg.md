---
question: "is the free-running collapse preamble structure or distribution drift"
status: open
source: "e1, 2026-09-01, written BEFORE runs/preamble_refutation.json existed; the refutation I named on the free-running fact"
---

# Pre-registration: my own refutation

> **`status: open` means pre-registered.** `runs/preamble_refutation.json` is
> absent as of this commit; `git log` on this file against that path is the
> ordering proof.

## What I am trying to break

`be.free_running_agreement_collapses_below_teacher_forced` reads free-running
agreement at **~0.23** against teacher-forced 0.727 and concludes the model's own
prefix destroys the distribution. I named the refutation on the fact itself:
**if the collapse is preamble structure rather than distribution drift, it is a
different and much cheaper problem.**

## The framing I got wrong, corrected before running

I wrote the refutation as *"prompts whose gold begins immediately with no
preamble."* Inspecting the data first: **the golds already have no preamble.**
`code_holdout_500`'s references start at `def is_prime(x):` or `a = [2, 2, 6]`,
column one, no prose.

**The preamble is the model's, not the dataset's.** The generations open with
`。\n\n### 例子\n\n假设我们有一个整数...` and only later reach a fenced block.
So the measurement is not "select aligned prompts" — there are none to select
against. It is: **strip the model's own preamble and score from where it starts
producing code.**

Recording this because the refutation as I originally phrased it would have been
unrunnable, and discovering that after the run would have looked like a result.

## What is measured

`ckpt_pretrain_30b_s2.pt.step24000`, 100 code prompts, greedy, `rep_stop=False`:

- **anchored agreement** — find the model's first code-looking token (a
  ` ```python ` fence, or a line starting `def `/`class `/an assignment), drop
  everything before it, then score per-token agreement against the gold from
  that point.
- **the same for the gold**, which needs no stripping, as the control that the
  anchor logic is not itself creating the alignment.
- **coverage**: how many generations contain any anchor at all. Measured on 40
  prompts already: **10/40**. That number is load-bearing and is stated below.

## Falsification, fixed now

| observation | reading |
|---|---|
| anchored agreement ≥ 0.60 on the anchored subset | **the collapse is preamble structure** — the reading on the free-running fact is overturned and the problem is cheaper |
| anchored agreement ≤ 0.30 | preamble is not the cause; **distribution drift stands** |
| 0.30–0.60 | partial; report both the figure and the coverage, no verdict |

**Sizing the instrument this time, not just naming the confound** — which is the
lesson from the ±4 window that produced a 0.0 where the truth was 0.23:

- the anchor search scans the **whole** generation, not a fixed window;
- after anchoring I also report the alignment-free LCS ratio, so a residual
  offset cannot masquerade as disagreement;
- and I report agreement over shifts up to **±150**, the full generation length,
  rather than a small window chosen by guess.

## The limit that will probably decide this

**Only 10 of 40 generations contain a code anchor at all.** So the anchored
score is computed on roughly a quarter of the set, and that quarter is
**selected for having produced code** — the best-behaved generations. Any
anchored number is therefore an upper bound on a favourable subsample, and
**a high anchored score does not overturn the collapse for the other three
quarters.**

If the anchored score is high, the honest statement is *"the collapse is
preamble structure for the 25% of prompts that reach code, and unmeasured for
the rest"* — not *"the collapse is preamble structure."* I am writing that
sentence now so I cannot write a stronger one later.

## What I will report

Anchored agreement, coverage, the LCS ratio on the anchored subset, and which
falsification row it lands in. **Not a verdict** — fb rules.
