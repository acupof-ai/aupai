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

## Composition: phi-1's proportions, not phi-1.5's

**The first revision of this document sized the synthetic set at 20B tokens. That was the wrong
paper's number and it is corrected here rather than quietly replaced, because the 20-fold error
came from reading a composition table without checking which score it produces.**

phi-1.5 is 1.3B parameters on 30B tokens -- phi-1's 7B plus roughly 20B of new synthetic
textbook-like data seeded from 20,000 topics. But that extra 20B targets common-sense reasoning,
and **the HumanEval 50.6% this project is aimed at is phi-1's number, produced by phi-1's 7B.**
Sizing our synthetic set to phi-1.5's 20B buys a capability we are not measuring, at 20x the
generation cost of the one we are.

p1 follows phi-1:

| part | tokens | source | teacher time |
|---|---|---|---|
| filtered code | ~6B | educational-value classifier over `data/corpus/code_*` | **none** -- no generation |
| synthetic textbooks | ~0.8B | 27B teacher, seeded from the 20K topic table, **in English** | ~8 days at 1160 tok/s |
| synthetic exercises | ~0.18B | 27B teacher, `signature + docstring -> body` with executable tests | ~1.8 days |
| classifier labels | ~0.02B | 27B teacher, ~100K educational-value annotations | ~5 hours |

**Generation order is by what blocks the gate, not by size.** The classifier labels are the
smallest artifact and the first one: they unblock the 6B of filtered code, which is 97% of the
gate corpus, and they cost five hours. Exercises second. **The gate corpus is the first two rows
plus the exercises -- 6.18B tokens, about two days of generation.** The textbooks are the
4.3-points-per-token item and do not block it; they generate continuously and land during the
gate run as the second arm.

Dropping the textbooks from the gate corpus is a judgement, not a citation: phi-1-base's 29%
comes from 6B filtered code **and** ~1B synthetic textbooks together, and the paper does not
separate them. It is tested rather than assumed because testing it costs two days instead of ten.

**The web corpus is not deletable.** It is the diversity source for generation prompts. This reverses
the reading under which `data/corpus/web_cci3_p*` was listed as unsuitable.

**Fully synthetic code is outside the published recipe.** phi-1 kept the 6B filtered code. Dropping
it is a legitimate arm but it has no reference score, so it is an ablation, not the plan.

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

## The schedule, measured

Generation, not training, is the schedule. The teacher serve was measured on 2026-09-09 rather
than estimated:

| measurement | value |
|---|---|
| single stream, 1 card | 88 tok/s (tileRL's own B=1 bench reads 92.4, so the kernel is not the variable) |
| aggregate, 3 cards, 32 concurrent, warm | **695 tok/s** |
| aggregate, 5 cards (projected) | ~1160 tok/s |

The first figure taken was 249 tok/s and included JIT warmup; it is recorded here because it is
what a cold measurement of this serve looks like and it was wrong by 2.8x. A `/health` read
showed prefill completing in the first 2 seconds of a 22-second window with `running=11`
throughout, so batching works and 700 tok/s is this stack's real loaded efficiency -- HTTP,
tokenization and prefill against a real request mix, not a synthetic microbenchmark. There is no
cheap optimisation left, and switching inference stacks buys at most 2-3x against a target that
already fits.

At 1160 tok/s the full ~1B synthetic set is about ten days on five cards; the gate corpus is
about two.

## Reference methods

| method | as published |
|---|---|
| phi-1 filtering | ~100K samples annotated by GPT-4 for "educational value for a student whose goal is to learn basic coding concepts"; random forest over a pretrained codegen model's output embedding; 35M files / >35B tokens down to 6B |
| FineWeb-Edu classifier | 450K Llama-3-70B-Instruct annotations; Snowflake-arctic-embed plus a classification head and one regression output; embedding and encoder frozen; 20 epochs at lr 3e-4 |
| StarCoder2 near-dedup | MinHash LSH at Jaccard 0.5, plus PII redaction and benchmark decontamination |
| corpus spot check | 50 samples per corpus, two independent readers, agreement recorded, disagreements listed rather than averaged |
