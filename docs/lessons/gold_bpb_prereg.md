---
question: "is knowledge accumulating across checkpoints in a way generation cannot express"
status: open
source: "e1, 2026-09-01, written BEFORE runs/gold_bpb.json existed; fb order after the copy hypothesis died"
---

# Pre-registration: gold-answer BPB across checkpoints

> **`status: open` means pre-registered.** No measurement exists as of this
> commit; `runs/gold_bpb.json` is absent and `git log` against that path is the
> ordering proof.

## Why this metric and not another

Every instrument used today passes through a decoder, and the decoder is broken
in a way that produces zeros: 74–80% of greedy generations loop
(`be.self_repetition_not_context_copying`), and the repetition guard truncates
the very metric that would have measured it
(`be.rep_stop_truncates_the_thing_it_measures`).

**Gold-answer BPB has no decoder.** It is the conditional NLL of the gold answer
string given the prompt, divided by the gold's UTF-8 byte count. No sampling, no
greedy, no repetition, no fence parsing, no stop condition. None of today's
confounds can reach it.

It is also the metric that resolves at our scale, where generative scores sit on
the floor (`be.gold_bpb_method`).

## What is being asked

**Not the level. Whether it falls.** The checkpoints, in token order:

`ckpt_0830v1_3.24b` (p324) → 8B → 15B → the 16B pin → `step24000` (22B)

Scored on code-500 and math-500 golds, the same sets the generative evals score.

## What a falling BPB licenses — and what it does not

This is the part committed in advance, because it is where the result will be
over-read.

| a monotone fall licenses | it does NOT license |
|---|---|
| "the model assigns increasing probability to correct answers" | "the model can do math" |
| "knowledge is accumulating that generation cannot express" | "the generative zeros are only a decoding artifact" |
| "the 0.0 scores are not proof of learning nothing" | "SFT or a decoder change will recover a specific score" |

**The gap between "assigns probability to the right answer" and "can produce it"
is exactly what SFT and decoding are for.** A falling BPB says the first is
happening. It says nothing about how much of the second we get, or when.

## Falsification, fixed now

| observation | reading |
|---|---|
| BPB falls monotonically across all five checkpoints on **both** sets | knowledge is accumulating; the generative zeros are not the whole story |
| BPB is **flat** (total change < 0.02 bits/byte end to end) | **the hypothesis dies** — nothing is accumulating that the golds can see, and "learned nothing" survives as the reading |
| BPB **rises** | worse than flat; something is being lost, and it outranks the decoding question |
| falls on one set, flat on the other | domain-specific; report per set and refuse a single verdict |
| non-monotone but net-falling | report the shape; a dip at one checkpoint is not noise until the seed spread is known |

**A number I cannot yet supply and will state instead of hiding**: the seed
spread of gold BPB at this scale is unmeasured, so "monotone" is a shape claim,
not a significance claim. If the total fall is under ~0.05 bits/byte I will say
the trend is within a range nobody has bounded, rather than calling it a trend.

## Two ways this measurement can be wrong

- **A shorter gold is not a better-predicted gold.** BPB divides by bytes, so a
  set whose golds get shorter across checkpoints would fall for the wrong reason.
  The golds are fixed strings from a fixed file, identical for every checkpoint,
  so this cannot happen here — stated because it is the standard way this metric
  breaks and its absence should be checked, not assumed.
- **Tokenizer drift across checkpoints.** If any checkpoint used a different
  tokenizer, its NLL is not comparable. Byte normalisation absorbs *some* of
  this, which is exactly why BPB is used instead of per-token loss, but I will
  verify the tokenizer is identical across all five and report ABSENT for any
  checkpoint where it is not.

## What I will report

Per checkpoint, per set: BPB, total gold bytes, token count, and the tokenizer
fingerprint. The shape of the curve. **Not a verdict** — fb rules.
