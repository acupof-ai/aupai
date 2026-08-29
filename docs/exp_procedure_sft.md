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

## Instrument caveat

math-hard resolves to +-1.1pt at a 2-3% pass rate. `probe_procedure` at n=180 held-out
problems is the same order. Test significance before explaining any gap, and validate the
parser before believing a number — round 4's reverse-parser read a correct `109` as `901`
and reported 5.6% where the truth was 30.0%.
