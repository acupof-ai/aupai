# SFT and RL at 200M — what three independent reviews concluded, and what we then measured

Three sessions were briefed separately and deliberately set against each other, 2026-08-29.
The full reports are beside this file. This page is the convergence: what they agree on,
what they still dispute, and — the part that matters — **which of it our own measurements
that day settled.**

| report | brief | file |
|---|---|---|
| SFT recipe | data volume, hyperparameters, loss masking, forgetting | [`sft_at_200m.md`](sft_at_200m.md) |
| RL recipe | smallest viable scale, pass@k premise, process vs outcome reward | [`rl_at_200m.md`](rl_at_200m.md) |
| Adversarial | attack the SFT→RL ladder itself; is the constraint just undertraining? | [`adversarial_review.md`](adversarial_review.md) |

The rule they were given: **where you converge, say so; where you do not, give me both sides
with their evidence — do NOT merge into a consensus that hides the disagreement.** Every
recommendation had to arrive with the experiment that would falsify it.

---

## What all three converged on

1. **The only statistically significant training gain on record is SFT against its own base**
   (1.9% → 3.6% math-hard, p=0.022). RL has no significant positive result at this scale.
2. **`rl_k4`'s 4.1% vs base 2.9% is z=1.44, p=0.15 — it was never a win.** Two reports
   derived this independently. It matters beyond RL: the ledger was silently dropping that
   4.1%, and it is the highest number in the table, so *"which checkpoint is best"* was
   being answered with a noise value. Fixed the same day.
3. **Distillation is the under-weighted arm.** Nobody owned it; all three ended up pointing
   at it.
4. **math-hard cannot resolve its own decisions.** Best-ever gain is z=0.11. Detecting 1pt
   at power 0.8 needs ~4,600 problems per arm against the current 1,032.
5. **The project's own discipline — significance first, instrument second, hypothesis third —
   was still being violated** at the time of review.

## What the measurements settled

Two of the reviews' central claims did not survive contact with the repo. Both corrections
are load-bearing.

**The RL contamination claim was backwards.** The adversarial review found 506 verbatim
math-500 questions in `data/rl/rlvr_math.jsonl` and concluded every post-RL math-500 number
was memorisation-inflated. Measured: every run that produced a number used `rl_band.jsonl`
or `rlvr_clean.jsonl`, **both at 0**. The 515 it counted is `218,468 − 217,953` — the rows
the holdout filter *removed*. It was evidence the filter worked. The raw file is now
quarantined and `data/rl/PROVENANCE.md` records which file each run consumed, because the
dirty file sitting beside the clean one with nothing distinguishing them is what made the
error cheap to make. *A file on disk is not a file that was used.*

**The procedure-SFT null was misread — by our own pre-registration.** Pre-registered in
[`../exp_procedure_sft.md`](../exp_procedure_sft.md), then run:

| | base k8 | + procedure SFT | + SFT and 7.4% replay |
|---|---|---|---|
| `probe_procedure` BOTH | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| digit head, teacher-forced | **21.3%** | **57.2%** | **61.1%** |

McNemar on the same 982 paired positions: base vs SFT **p=5.7e-62**; SFT vs replay
**p=6.6e-05**.

The result landed in the pre-registered cell *"BOTH near zero → coverage was not the
constraint"*, and that reading is wrong. Teacher-forced, the model predicts 57% of the
numbers in gold procedure text and has learned the chain shape and the terminator.
Free-running, every number is wrong and the chain degenerates into `1/1 ≈ 1/1` loops.
**The procedure was learned; it does not survive the model's own rollout.** That is exposure
bias, not missing data.

Replay's dissociation is the positive case for that reading, not a null: one intervention
moved teacher-forced significantly and moved BOTH not at all. Coverage would have moved both.

Mechanism, read from the code rather than inferred: `train.py:577` feeds the number back
into the input embedding from `num_vals`, which is gold on every training step, so **the
digit head has never once trained with a wrong value in its input.** Scheduled sampling has
to be applied to the value channel; the token channel alone cannot reach this gap. That
observation is the adversarial reviewer's, and without it the next experiment would have
been built on the wrong channel.

## Two claims retracted by their authors

- **"There is no middle regime between outcome RL and teacher rollouts."** Both the RL and
  adversarial reviews held this; scheduled sampling *is* that regime, and the null above is
  its first internal evidence. Retracted by both.
- **"Every MC benchmark sits at the 25% chance line."** Ours, and wrong: ARC-E and PIQA are
  significantly above chance (z=9.1 and 3.6). Three of five are not at the line.

## The disagreement that is still open

**Is step-level RL move 3, or never?** The RL review holds it as a contingency once
step-correctness clears 50%; the adversarial review holds that at a 51.2% wrong-equation
rate the step reward goes sparse one level down and the middle collapses into
teacher-rollouts-or-nothing. Teacher-forced step correctness is now **57.2%**, which crosses
the gate as the adversarial review itself wrote it — but the measured failure is *recovery
from the model's own error*, and step-RL samples exactly those rollouts. **Not merged.
Adjudicated by the scheduled-sampling result.**

## The order, by cost — not by which argument reads best

```
procedure SFT                  DONE — diagnosed the failure, did not fix it
self-generated-prefix SFT      NEXT — value channel; cheap mechanism check first
  1. inject wrong values into the prefix with probability p   (no generation needed)
  2. DAgger shape: generate, truncate at the first error, SFT the gold continuation
distillation from the 27B       teacher-yield probe and in-context imitation probe first
RL                              only if a pass@k gap ever appears
30B-token pretrain extension    queued; this round gave it no support, since the failure
                                is dynamics and not coverage
```

## Two rules this exercise produced, now in AGENTS.md

- **A probe asking "did training install X" must measure teacher-forced AND free-running in
  the same run.** The gap between the two numbers is the diagnosis; either alone is
  unreadable, and the free-running number alone would have retired a correct path.
- **A null landing in a pre-registered cell does not certify that cell.** Pre-registration is
  what made the missing branch visible instead of absorbing the result into an existing one.
