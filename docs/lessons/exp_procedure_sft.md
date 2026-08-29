---
question: Does SFT install procedure execution at 200M, and does it survive free-running?
status: measured
source: pre-registered experiment, 2026-08-29; install probe teacher-forced and free-running
---

# Pre-registration: does SFT install procedure execution at 200M?

Written 2026-08-29, **before ckpt_k8_v3_fone exists** (step 1280/3591 at time of writing).
Registered in advance because every interpretation below is one that could be reached after
the fact to fit whichever number came out. Six of one day's conclusions on this project came
from reading a probe backwards; the defence is committing to the reading first.

## The claim under test

`ckpt_k7_v3` scores 2.8% BOTH on `probe_procedure` — it writes procedure-shaped prose around
a guessed answer. Three constraints on arithmetic were isolated with matched controls:
number representation (FIXED by `--fone`, 0% -> 16.7%, p=1.2e-7), format ambiguity (FIXED by
the format tag, computation 16.7% -> 41.7%, p~1e-7), and **procedure execution, not fixed**.

The claim: procedure execution is absent because no SFT run on record used data containing
executed procedures. `data/synthetic/procedure_v1.jsonl` is the first data that does.

Measured, not assumed — the three properties the literature says are load-bearing
(Lee et al. 2307.03381, Nye et al. 2112.00114, rStar-Math 2501.04519):

| property | procedure_v1 | every previous SFT set |
|---|---|---|
| answer AFTER the working | `结果 = ` is always the last line | answer-first |
| every intermediate line machine-verified | `steps_valid` scores gold 100% | 51.2% of equations wrong |
| one format per prompt | 50,000/50,000 tagged, 0 ambiguous | three formats, identical prompts |

## The gate

**`probe_procedure` BOTH on held-out problems.** Not ANSWER (winnable by answer-first
guessing — that is what k6_arith4's 28.9% was) and not math-hard.

## math-hard is NOT the gate, and this is why

Measured 2026-08-29 over `math_hard_eval_1k.jsonl` (1,032) x `procedure_v1.jsonl` (50,000):

```
math-hard rows carrying a procedure tag:        0 / 1032
exact problem-body overlap:                     0 / 1032
max Jaccard, 200x3000 sampled pairs:            0.200   (project threshold 0.8)
```

Zero contamination. But the reason for zero overlap is not a good filter — the two are
different genres:

```
math-hard:      今年老师的年龄比学生的3倍少3岁，9年后老师的年龄正好是学生的2倍…
procedure_v1:   [单位换算] 把 777 千克 换算成 毫克
```

Untagged word problems against tagged bare procedures. They share no template, so a
math-hard null cannot adjudicate whether SFT installed execution. Using it as the falsifier
would kill a correct path for a wrong reason. math-hard is recorded as a **transfer**
readout, separately, gating nothing.

## Committed interpretations

| result | reading | next arm |
|---|---|---|
| BOTH up AND working-before-answer holds on held-out | SFT installs execution. RL is sharpening only. | step-level RL, gated on pass@8-pass@1 >= 15pt |
| BOTH up, answer-first persists out of distribution | SFT installed the FORM, not the dependency | Math-Shepherd-style step reward — the existing line checker makes it free, no PRM to train |
| BOTH near zero | coverage was not the constraint | b0's arm: 30B-token pretrain extension |
| **BOTH up but math-hard flat** | **data-coverage gap, NOT an SFT failure** — nothing in the SFT data bridges word problem to procedure | bridging data, not "SFT does not transfer" |

That last row is the one being pre-registered. It is the reading that will be least
available after the fact.

## AMENDMENT, written AFTER the result — the table above was missing a branch

**Not pre-registered. Added 2026-08-29 after ARM A ran, and labelled so because the whole
point of the section above is that it was written first.**

ARM A landed on "BOTH near zero", whose committed reading is *coverage was not the
constraint, go to the pretrain-extension arm*. **That reading is wrong for this result**,
and taking the null at face value would have retired a correct path.

| | base k8 | + procedure SFT |
|---|---|---|
| BOTH (mul / eq / unit) | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| STEPS, total | 0/180 | 4/180 |
| digit head, teacher-forced on gold procedure text | 209/982 = **21.3%** | 562/982 = **57.2%** |
| FoNE encode -> decode round-trip | 120/120 | — |

Teacher-forced, the model predicts 57.2% of the numbers in gold procedure text, against
21.3% at base. It also learned the unit-conversion chain shape and the `结果 = ` terminator.
Free-running, every number is wrong and the chain degenerates into `1/1 ≈ 1/1` loops.

**The procedure was learned; it does not survive autoregressive rollout.** One wrong number
early and there is no recovery. That is exposure bias, not missing coverage, and SFT loss
falling to 0.144 with zero generalisation says the same thing: it fit
next-token-given-a-gold-prefix, which is not the same skill as executing a procedure.

The missing fourth branch, stated now for the next round:

| result | reading | next arm |
|---|---|---|
| BOTH near zero **but teacher-forced accuracy jumps** | the steps are learned, the rollout is not | train on self-generated prefixes — scheduled sampling / DAgger-style — before any RL, because the failure is recovery from the model's OWN error and step-RL samples exactly those rollouts |

### ARM B's dissociation is positive evidence, not a null

Adding 7.4% general replay moved the teacher-forced digit head **significantly** (57.2% ->
61.1%, McNemar chi2=15.9, p=6.6e-05) and moved BOTH **not at all** (0.0% in all three
checkpoints). If the failure were missing coverage, extra data should have moved both. It
moved only the teacher-forced side.

That dissociation is the positive case for the dynamics reading, and it is stronger than
the ARM A null on its own: one intervention, two measurements, opposite outcomes. (What
replay actually improved is not measured here — it is not the procedure, and attributing it
to "general numeric representation" would be a guess.)

### The mechanism, from the code

`train.py:564` defines the only digit head; `train.py:755` (training) and `train.py:919`
(free generation) both call it, and the teacher-forced measurement above calls the same one.
So the 57.2%-vs-0 gap is not an artifact of a separate head — the hidden state's prefix is
the only variable.

More precisely: `train.py:577` feeds the number back into the input embedding
(`emb = emb + num_proj(feat)`) from `num_vals`, and `num_vals` is gold on every training
step. **The digit head has never once trained with a wrong value in its input.** So
scheduled sampling has to be applied to the VALUE channel; doing it on the token channel
alone cannot reach this gap.

Cheapest first cut (mechanism check, not the real intervention): with probability p, replace
gold `num_vals` in the prefix with a wrong value during training. Off-distribution, but it
tests whether "training under a wrong prefix value" has any signal at all, in one batch and
with no generation. The real intervention is DAgger-shaped: generate, truncate at the first
error, SFT the gold continuation.

Two consequences worth recording:

1. **A null landing in a pre-registered cell does not make that cell correct.** The
   pre-registration is what made the gap visible instead of absorbing the result.
2. 57.2% crosses lessons-b0's own >50% step-correctness gate, so its conditional objection
   to step-level supervision does not bind on this checkpoint. That does not by itself
   argue for step-RL: the failure mode is self-error recovery, and step-RL samples the
   model's own rollouts, which is the thing that is broken.

## What the probe cannot measure

`procedure_curriculum` splits on `prob_key(fmt, body)` — problem level. The held-out 10% is
therefore **same template, unseen numbers**. That is the skill for a procedure (Lee et al.'s
excluded-number robustness is the precedent: models hold at 100% excluding half the 3-digit
numbers; the weak axis is excluded-DIGIT, not excluded-number). But there are only 3
templates and **no held-out template**, so template-level generalisation is not measured. If
the probe succeeds, that is the next question and it needs new templates in the data.

## Arms

Both are ~4 minutes.

```
# A. procedure SFT
python prepare_sft_math.py --sources data/synthetic/procedure_v1.jsonl --out data/sft/proc_v1.pt
scripts/run_sft.sh k8_proc ckpt_k8_v3_fone.pt data/sft/proc_v1.pt

# B. same + 5-10% replay of general/chat
#    Forgetting is measured twice on this project (math-500 51.2->44.8 p=0.043;
#    51.6->39.2 p<0.001) and replay has never once been tried. Flan-T5-250M: 6.2%
#    replay holds NLI 16.5 -> 83.8 with math flat across every ratio tested.

python scripts/probe_procedure.py --ckpt ckpt_k8_proc.pt --tokenizer data/tokenizer.json --fone
```

Score arm B on the procedure probe too, not only on the general evals: the 250M result
predicts math stays flat, but sub-500M full FT overwrites priors (Fine-Tuning Trap 2606.06920)
and dilution is the one risk neither prior covers. If replay dents the probe, that is the
LoRA signal.

## The control is k8, not any earlier number

k8_v3_fone is a NEW base — the first combination of corpus v3 with `--fone`, so it differs
from every earlier checkpoint in both the data and the number representation. **`sft_k5_ctrl`'s
3.6% is therefore not the control for this experiment**, and neither is k7_v3's 2.8% BOTH.
The only valid comparison is k8 base against k8+SFT on the same probe: anything cross-base
mixes the corpus effect with the SFT effect and cannot separate them. Run the probe on the
bare k8 checkpoint first, before any SFT, and record it.

## Statistics, fixed in advance

`probe_procedure --n 60` scores 60 held-out problems per procedure = **n=180**. At a 30% pass
rate SE is about 3.4pt, so a 5pt arm difference is noise.

- **Between arms** (SFT vs SFT+replay): Fisher exact, one-sided. The `--fone` arithmetic
  result on this project used the same test at p=1.2e-7.
- **Within one arm, before vs after**: McNemar (paired) — the same 180 problems are scored
  twice, and a two-proportion z-test throws that pairing away.

## Instrument caveat

math-hard resolves to +-1.1pt at a 2-3% pass rate. `probe_procedure` at n=180 held-out
problems is the same order. Test significance before explaining any gap, and validate the
parser before believing a number — round 4's reverse-parser read a correct `109` as `901`
and reported 5.6% where the truth was 30.0%.
