---
question: What corpus does aupai-p1 train on, and how do we know it is good enough before spending five days of GPU?
status: recorded
source: arXiv 2306.11644 (phi-1), arXiv 2309.05463 (phi-1.5), FineWeb-Edu classifier card, StarCoder2/The Stack v2
---

# aupai-p1 data recipe

User order 2026-09-09: the 200M-active MoE line is retired. `ckpt_1.5b-a0.2b-e48_30b.milestone_keep_fb_step34000.pt`
is its final state and nothing resumes from it. p1 is a new model on a new corpus.

## Why the corpus changed, in one measurement

`ckpt_1.5b-a0.2b-e48_30b.milestone_keep_fb_step34000.pt`, 26.7B tokens, 41% Python by mix weight:
HumanEval pass@1 **0/164** under the official protocol, 160/164 empty completions
(`facts/base_eval.json#be.humaneval_pass1_step34000`). Not a capability floor -- a format one:
127/164 are cut at position 0 by a stop string and 33/164 argmax `<eos>`, and four spaces of bare
indentation lift it to 4/164.

A 27.5M-token SFT on `signature + docstring -> body` pairs (`format_sft_0909`) moved it to
**3/164 with the empty rate at 72/164**. The empty rate is the preregistered discriminator and it
moved 55 points; pass@1 moved 3, which at n=164 is Fisher one-sided p=0.124 against 0/164 -- the
format explanation is confirmed, the capability gain is not.

The published comparison says the same thing at a scale where it is not ambiguous:

| model | params | pretraining tokens | HumanEval pass@1 |
|---|---|---|---|
| phi-1-small | 350M | 7B | 45% |
| phi-1 | 1.3B dense | 7B | 50.6% |
| phi-1-base (no CodeExercises) | 1.3B dense | ~6.8B | 29% |
| DeepSeek-Coder-Base-1.3B | 1.3B | 2T | 34.8% |
| ckpt .step34000 | 0.2B active | 26.7B | ~0-2% |

phi-1-small carries 3.7x fewer parameters and 285x fewer tokens than DeepSeek-Coder-Base-1.3B and
scores 10 points higher. The variable is the shape of the data, not its quantity.

## Value per token, which decides the allocation

phi-1's CodeExercises is **under 180M tokens** and is the difference between phi-1-base's 29% and
phi-1's 50.6%.

| artifact | tokens | points bought | points per B token |
|---|---|---|---|
| CodeExercises (synthetic exercises) | <0.18B | +21.6 | ~120 |
| CodeTextbook (6B filtered + synthetic) | ~6.8B | 29 | ~4.3 |

The exercise set is worth 28x per token and is the smallest artifact in the recipe. It gets the
heaviest staffing, not the last slot.

## Composition: phi-1.5's proportions

phi-1.5 is 1.3B parameters on 30B tokens: phi-1's 7B plus roughly 20B of new synthetic
textbook-like data, seeded from **20,000 selected topics**. The only non-synthetic part of the whole
training set is phi-1's 6B filtered code. Web samples enter the *generation prompts* for diversity
and never enter training directly.

p1 follows those proportions:

| part | tokens | source |
|---|---|---|
| synthetic textbooks | ~20B | 27B teacher, seeded from the 20K topic table, **in English** |
| filtered code | ~6B | educational-value classifier over `data/corpus/code_*` |
| synthetic exercises | ~0.18B | 27B teacher, `signature + docstring -> body` with executable tests |

**The synthetic textbooks are English.** HumanEval's docstrings are English, the exercise form is
`signature + docstring -> body` with an English docstring, phi-1 and phi-1.5 are English, and AGENTS.md
already states that this corpus follows capability rather than language at roughly 60:40 English-leaning
because code is written in English. This was left unstated in the first revision of this document and the
omission cost a measurement: the tokenizer proxy named below was `data/corpus/textbook`, which is 76%
Chinese, and it read never_used 0.084 where the English proxy reads 0.63. A recipe that does not name the
language of two thirds of its tokens will be proxied in the wrong one.

**Fully synthetic code is outside the published recipe.** phi-1.5 kept the 6B filtered code. Dropping
it is a legitimate arm but it has no reference score, so it is an ablation, not the plan.

**The web corpus is not deletable.** It is the diversity source for generation prompts. This reverses
the reading under which `data/corpus/web_cci3_p*` was listed as unsuitable.

## Acceptance: one falsifiable gate, not a checklist

A corpus is good enough iff **a 350M dense model trained on it clears HumanEval 30%.** phi-1-small
reports 45% at that size on 7B tokens, so a run landing far below it says the data is wrong and no
increase in scale repairs it. The gate costs about a day; the 1.3B run costs about five. It runs
first.

## Per-line acceptance criteria

| line | owner | acceptance |
|---|---|---|
| teacher serve + synthetic textbooks | b0 | measured tok/s on the tileRL serve BEFORE sizing anything; 50-sample readability judgement; topic coverage cross-table against the exercise set |
| synthetic exercises | 44 | execution pass rate with the discard rate recorded; decontaminated against HumanEval and MBPP; topic distribution table; 50 samples, two readers, agreement recorded |
| educational-value classifier | e1 | held-out AUC against teacher labels; keep rate stated against phi-1's ~17%; **threshold ablation run on our own corpus**; 50 high-scoring and 50 low-scoring samples, two readers |
| topic seeds, dedup, decontamination | 3b | 20K topic table with a coverage measure; decontamination carries a known-positive control; a self-repetition metric for the synthetic set |
| tokenizer + eval harness | d1 | temp 0.2 / top-p 0.95 / 20-sample pass@1 sharing one judge with the greedy path, both reported; tokenizer rebuild decision from `tokenizer_eval` on a sample of the new composition |
| human spot check | 98 | one table, one row per artifact, each with n, two readers, agreement, disagreement count, and a mix/no-mix verdict; a row without an agreement rate does not count |

Two criteria are load-bearing and easy to drop:

- **The classifier threshold is measured, not copied.** FineWeb-Edu reports threshold 3 as the best
  trade-off between knowledge- and reasoning-intensive benchmarks and benchmarks like HellaSwag.
  That trade-off is a property of their corpus. (An earlier revision of this line said higher
  thresholds "significantly degrade HellaSwag and PIQA"; 44 checked the source and the paper's
  text names HellaSwag only, with the per-threshold numbers in a figure. Corrected rather than
  deleted, because the overstatement is the same defect this document exists to prevent.)
- **Decontamination reports a known-positive control.** A HumanEval problem is planted and must be
  caught. `facts/contamination.json#cont.split` records 30% of math-500 questions with a containment
  hit in the math SFT corpus; a decontamination that only reports "0 hits" is indistinguishable from
  one that never ran.

## The open number

Generation, not training, is now the schedule. 20B tokens out of a 27B model is the dominant cost and
its size is unknown until the tileRL serve is measured. Nothing downstream can be scheduled against
an estimate here.

## Reference methods

| method | as published |
|---|---|
| phi-1 filtering | ~100K samples annotated by GPT-4 for "educational value for a student whose goal is to learn basic coding concepts"; random forest over a pretrained codegen model's output embedding; 35M files / >35B tokens down to 6B |
| FineWeb-Edu classifier | 450K Llama-3-70B-Instruct annotations; Snowflake-arctic-embed plus a classification head and one regression output; embedding and encoder frozen; 20 epochs at lr 3e-4 |
| StarCoder2 near-dedup | MinHash LSH at Jaccard 0.5, plus PII redaction and benchmark decontamination |
| corpus spot check | 50 samples per corpus, two independent readers, agreement recorded, disagreements listed rather than averaged |
