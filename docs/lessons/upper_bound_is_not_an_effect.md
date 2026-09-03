---
question: A numerical derivation said the bf16 logits are 14% wrong. What is that worth in loss?
status: measured
source: runs ab_fp32logits_base / ab_fp32logits_fp32logits, 500 steps each, H20 card 5, 2026-09-03; /tmp/ab_scores.jsonl; scripts/attnres_triton_bf16_gate.py
---

# An upper bound is not an effect, and this is what the gap was worth

`model.py:269` accumulates a `D=1024` dot product in bf16. Measured against fp64,
that puts the logits **0.858 off against a spread of 279.8**, and softmax turns it
into **14% error on the mixing weights**. Every step of that derivation is correct.

The question the derivation cannot answer is what 14% on the weights costs in
loss. Two arms, 500 steps, identical seed/data/shape, differing only in
`--attn_res_fp32_logits`:

| domain | base | fp32 | delta |
|---|---|---|---|
| math_owm_stage2 | 2.7757 | 2.7642 | −0.0115 |
| en_c4_stage2 | 3.8675 | 3.8641 | −0.0034 |
| cot | 1.8131 | 1.7992 | −0.0139 |
| textbook_30b | 3.4999 | 3.4911 | −0.0088 |
| chatml | 4.2268 | 4.1895 | −0.0373 |
| chat_qa | 4.2523 | 4.2296 | −0.0227 |
| zh_web | 5.6424 | 5.6331 | −0.0093 |
| code_py_starcoder | 1.8145 | 1.8005 | −0.0140 |
| code_py_rp1t | 1.8119 | 1.8050 | −0.0069 |
| **unweighted mean** | 3.3005 | 3.2863 | **−0.0142** |

**14% error on the mixing weights is worth 0.0142 nat.** That number did not
exist before this run: the derivation gives a bound on the perturbation, and the
distance between that bound and the loss it causes was simply empty. It is now
one point.

Largest single domain 0.0373 nat = 0.72 seed-sd = **15.5% of the 0.24 nat
readable bar** (`ds.seed_variance_0p2b`, `eval/score_matrix.py:150`). The mean is
0.28 sd. Pre-registered verdict was reject; the reading is reject; `model.py:269`
does not change.

## Quote the mean with its dispersion, not as a constant

The nine deltas are not one number nine times. Between-domain sd is **0.01022
against a mean of 0.0142 — a ratio of 0.72** — and they run from −0.0034 to
−0.0373, an **11x spread**. A single uniform numerical offset would cluster;
these do not. The picture is a small common negative shift plus a
domain-dependent magnitude, so **0.0142 nat/14% is an average over nine domains
and carries that 0.72 with it** wherever it is cited.

## Caveat on a test I ran and should not have

All nine deltas are negative, and I reported a sign test at p = 0.0039. **The
independence assumption is false.** The nine domains are one pair of
checkpoints read nine times, not nine draws: a seed-level fluctuation shifts all
nine the same way, which is the cheapest explanation for the shared sign. The
exponent 9 in `(1/2)^9` is fictional.

The correct statistic was already in hand — **0.28 seed-sd** — and it gives the
same verdict. What looked like two independent pieces of evidence was one
quantity computed two ways, one of them treating correlated readings as
independent.

There is also a prior that fires before independence does: **the sign was
predictable.** bf16 adds rounding noise to the logits, noise through softmax and
cross-entropy raises expected loss, so "fp32 lower" is the direction the
derivation already predicted. Observing a predicted sign weakly confirms the
derivation; it is not new evidence about the size of the effect, and an event
with prior probability well above 1/2 must not be scored against 1/2.

## What did not happen, deliberately

The 0.24 bar was fixed before the run (`eval/score_matrix.py:150`, in the tree
2026-08-31, two days before this A/B), the directional prediction is timestamped
in `169da865`, and no reading for either arm existed anywhere when it was
written. **The interesting pattern turned up after the data did, and it did not
become a criterion.** `docs/lessons/gate_failure_shapes.md` §149 is the same
failure with the sign flipped: there, a threshold that already existed in the
world was called pre-registered; here, a pattern discovered post-hoc was refused
promotion to one.

## Rule

**A numerical bound licenses a hypothesis, never a magnitude.** "The weights are
14% off" and "the loss moves" are different claims, and the second needs an
experiment. When the experiment lands, report the conversion rate — that is the
part that was missing — and report its dispersion with it, because a mean over
domains that vary 11x is not a constant.

And: a statistic whose independence assumption you have not checked can turn one
measurement into an imaginary second one.
