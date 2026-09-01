---
question: Is the 0.0% on math-500 absent reasoning, or a base model that was never taught the answer format?
status: open
source: pre-registered 2026-09-01 before the few-shot run; measured format gate (0/1439 generations contain \boxed against 494/500 golds)
---

# Pre-registration: the few-shot arm

Written before the run. The sampled arm's pre-registration
(`docs/lessons/sampled_arm_prereg.md`) is not superseded — it is answered, and one of
its assumptions was measured false. That is recorded there and repeated here because it
is the reason this arm exists.

## What the sampled arm settled, and what it could not

pass@8 came back 0.0% on math-500 and code-500. Under the pre-registered bands that is
**B: capability absent**. The number stands. B's conclusion does not follow, and the
diagnostic says why:

| | step24000, t=0.8, k=8, rep_stop off |
|---|---|
| generations | 1439 |
| containing `\boxed` | **0** (0.0%) |
| containing any digit | 1011 (70.3%) |
| gold answers containing `\boxed` | 494/500 (99%) |

`math_zh.score` extracts `\boxed{...}`. The model has never once produced it. Reasoning
sits downstream of a gate that never opens, so no value of k separates *cannot reason*
from *does not know the answer template*. Code-500 is the same shape: 3 of 288
generations carry a code fence, and the grader extracts a fenced block.

The failed clause is named rather than the threshold moved: the pre-registration
assumed the metric could observe capability if capability were present. It cannot.

## What this arm changes, and only this

Three worked examples in a plain-text continuation, then the target problem.
`eval/l1_fewshot.py` already implements exactly this and pins the format: demos are
problems 0–2 of math_test_500 with full gold solutions, N=497 after excluding them, and
**ChatML is deliberately not used** — its own docstring records that the base saw 1.18%
chat-domain data, so a zero-shot ChatML result confounds format with capability. That
docstring anticipated today's finding on 2026-08-30, and `SKIP_REASON` still lists
`l1_fewshot` as a base-panel metric that does not apply. That call was wrong; this is
the evidence.

One variable moves: whether the model has seen the answer format in its context. Same
checkpoint, same problems, same scorer, same greedy decode.

## Pre-registered readings

The threshold is the one L1 already carries, not a new one:
**2·delta = 2·1.4/√497 = 12.6pt**, and for a floor reading what matters is whether the
rate clears zero at all.

**A. The format was the gate.** `\boxed` appears in ≥ 20% of generations AND exact-match
accuracy ≥ 2.0% (≥10 of 497).

Both clauses, because either alone is ambiguous. Format alone means it learned to copy
the demos' shape. Accuracy alone at this scale is within noise of zero. Together they
say the model can produce an answer in the right form and sometimes the right one, and
the 0.0 was a format artifact — which makes every generative number on the board
uninformative rather than negative, and makes the next step a format fix (SFT, or a
prompt convention) rather than a data change.

**B. The format was not the gate.** `\boxed` appears in ≥ 20% of generations AND
accuracy < 1.0%.

The model copies the demonstrated format and still cannot answer. This is the reading
that restores the sampled arm's B on firmer ground: the gate opened and nothing came
through. Data and duplication hypotheses keep full weight.

**C. The demos did not land.** `\boxed` appears in < 20% of generations.

Three examples were not enough to induce the format, so the experiment did not run its
own manipulation. Not a result about capability in either direction — it is a failed
intervention, and the next step is more demos or a different prompt, not a conclusion.
Naming it in advance because it is the outcome most easily written up as B.

## Pre-registered secondary readings

- **Format rate is the manipulation check** and is reported first, before accuracy. If
  the manipulation did not take, the outcome measure is not interpretable — this is the
  discipline whose absence produced the sampled arm's ambiguity.
- **Degeneration with the demos present.** The sampled arm's incidental finding is that
  with `rep_stop` off the generations run long and do *not* loop. Whether few-shot
  changes that is recorded, not interpreted.
- **A number that clears A on format but sits between 1.0% and 2.0%** is reported as
  exactly that. 497 problems cannot separate 1.5% from 2.5%.

## Amendment, written 2026-09-01 before either arm produced output: a second demo count

Reading C — the demos do not induce the format — is the likeliest single outcome, and
with one arm it is indistinguishable from "this model cannot be prompted into the
format at all". Those call for different next steps: more demos, versus abandoning
prompting and fixing the format in training.

So a second arm runs concurrently on the same checkpoint with **8 demos** instead of 3.
One variable, the amount of format evidence in context. Pre-registered readings for the
pair:

| 3-demo format rate | 8-demo format rate | reading |
|---|---|---|
| < 20% | ≥ 20% | the format is inducible; 3 demos was simply too few. C for the 3-demo arm, and the 8-demo accuracy becomes the A/B test |
| < 20% | < 20% | in-context format induction fails at this scale. Not a capability result; the next step is training, not prompting |
| ≥ 20% | ≥ 20% | the manipulation took at both counts; compare accuracy, and 8-vs-3 gives a crude dose-response |

The dose-response row is the one worth naming: if format rate rises with demos and
accuracy does not, that is B with a mechanism — the model learns the shape from context
and still cannot fill it.

This amendment is written before any output exists from either arm, which is the only
thing that makes it a pre-registration rather than a description. Both arms launched at
the same time on GPU2 and GPU3.

## What this cannot answer

It cannot distinguish reasoning from retrieval of a similar training problem. It cannot
speak to code-500, whose gate is a code fence and which needs its own arm. And a
positive result here would not make the *existing* math-500 numbers readable — those
were measured under a format the model does not produce, and no reinterpretation
recovers them; they need rescoring, not rereading.

## Falsification

Reading C falsifies the design rather than the hypothesis: if three demos cannot induce
`\boxed` at all, this arm answered nothing and says so. Recording that in advance so a
null manipulation is not written up as a null result.
