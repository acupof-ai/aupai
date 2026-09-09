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
| filtered code | **set by the threshold ablation** | educational-value classifier over the three domains named below | **none** -- no generation |
| synthetic textbooks | ~0.8B | 27B teacher, seeded from the 20K topic table, **in English** | ~8 days at 1160 tok/s |
| synthetic exercises | ~0.18B | 27B teacher, `signature + docstring -> body` with executable tests | ~1.8 days |
| classifier labels | ~0.02B | 27B teacher, ~100K educational-value annotations | ~5 hours |

### The classifier's input: three domains, not ten

The pod holds ten `code_*` directories, and **five of them are upstream stages of the other
three.** Feeding a stage and its own descendant passes the same documents through the teacher
twice and inflates the keep rate with duplicates.

| domain | tokens | docs | what it is |
|---|---|---|---|
| `data/corpus/code_rp1t_dd09` | 6.24B | 3.43M | rp1t filter batch 1, MinHash-J 0.9 dedup (3.75M -> 3.43M) |
| `data/corpus/code_rp1t_b2v2_dd` | 3.60B | 2.10M | rp1t filter batch 2 v2, **cross-deduped against dd09**: its stats read `b2v2 against code_rp1t_dd09 AND within b2v2; code_rp1t_dd09 kept whole` |
| `data/corpus/code_dedup08` | ~8.95B (derived) | 6.24M | starcoder-py + py_rp1t union, 0.8 dedup (6.39M -> 6.24M) |
| total | **~18.8B** | **11.78M** | |

Excluded as upstream: `code_rp1t` (7.57B), `code_rp1t_b2` and `code_rp1t_b2v2` (4.89B),
`code_py_starcoder` (8.74B), `code_py_rp1t` (0.42B). `code_rp1t_rest` and `code_rp1t_dd09_full`
are empty shells.

`code_dedup08`'s figure is **derived, not read**: its stats file carries no `tokens` field, so
8.95B is `docs_kept/docs_in = 6239038/6389842 = 97.6%` applied to its 9.17B of inputs. Measure it
before any threshold decision rests on it.

The `code_dedup08` residual overlap, open here as a name-based inference, was
**measured 2026-09-10**: it is a union build of 283 starcoder shards plus 15
`code_py_rp1t` shards, and those 15 are by construction a third copy of
dd09/b2v2 content. The exact-overlap channel deleted 169,561 dedup08 docs, 138.6K (82%) of them
on those 15 rp1t shards and 31.0K (18%) spread over the 283 starcoder shards
(b0, pod count 2026-09-10); per-shard rate differs ~9x, so the starcoder
side was barely touched relative to its size. An earlier draft of this
section bounded the overlap at 0.42B = 2.2% from names and doc counts; that
bound assumed the overlap was diffuse across the domain, and it is in fact
concentrated on the 15 rp1t shards.
The dd09<->b2v2 near-overlap (22.9%/26.3% participation, est J>=0.5) is
PENDING RE-MEASUREMENT after the loc/sig realignment (PR #177); near-dedup
deletion is not approved. Operational detail stays in the pod's
`data/decontam/NOTES.md`.

### Which clean corpus p1 reads (source of truth)

The 2026-09-09/10 decontamination pass produced two artifacts that both read
as "the clean corpus". They serve different uses:

- `data/corpus_clean/<domain>/` (pod, 57G) — clean source copies, for any use
  that does NOT go through the quality classifier.
- the classifier's keep set minus the deleted doc ids — **what p1 training
  reads**. The keep set is decontaminated by doc id after e1's scoring run.

The corpus swap (old source dirs renamed aside, clean copies renamed into
place, old kept) happens after e1's scoring finishes, per 4c's plan (b).
This section is the tracked authority; the manifests and per-pass numbers
live in the pod's `data/decontam/NOTES.md`.

### The 6B is not a target

phi-1 filtered 35B down to 6B, a **17% keep rate**. Our pool is 18.8B; 17% of it is **3.2B**, and
reaching 6B would require a 32% keep rate. Loosening the threshold twofold to hit a token count
copied from another paper inverts that paper's own finding, which is that quality beats quantity.

So the keep rate and the resulting token count are **outputs of the threshold ablation, not inputs
to it**, and both are reported against phi-1's 17% with an explanation either way. If a strict
threshold yields 3B, the gate runs on 3B. **The acceptance criterion is HumanEval 30% at 350M; the
corpus size has never been a criterion.**

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

## The tokenizer is rebuilt at V=20,000

Ruling 2026-09-09 (fb, reviewed by 44 without challenge). The vocabulary frozen 2026-08-29 is
unfrozen for p1. **This invalidates nothing, because p1 has no checkpoints and that is the whole
reason the decision is cheap today and monotonically more expensive from p1's first step.**

**No gate forced it.** Measured by b0 on the real composition (seeds 7/13/21, 143-162 textbook
chapters plus the three code domains at 4M chars, 88:12):

| gate | value | |
|---|---|---|
| round-trip lossless | true, every subset | PASS |
| all 256 bytes | 256/256 | PASS |
| hanzi whole-char >= 0.95 | **undefined** -- the p1 composition has no hanzi | — |
| ref fertility <= 1.55 | **1.4286** | PASS |

The authorisation is **unfreeze condition 2, "the corpus distribution changes materially"**, and
the mechanism is measured rather than asserted: **64.9% of the frozen vocabulary's slots are hanzi
tokens** (`facts/tokenizer.json#tok.minicpm5_slot_budget_vs_ours`), leaving ~11.5K slots serving
English and code where the candidate has 20K. A gate is a guardrail; a condition is an
authorisation, and they are not the same thing.

**What freezing would have cost**, candidate V=20K against the frozen vocabulary, chars/token on
the same sample:

| subset | frozen | candidate | tax |
|---|---|---|---|
| textbooks | 3.263-3.266 | 3.289-3.297 | +0.9% |
| code, three domains | 3.099-3.189 | 3.220-3.298 | **+3.8%** |
| full mix 88:12 | 3.127-3.201 | 3.232-3.298 | **+3.4%** |

The +3.4% is permanent and multiplies across p1's 6-20B tokens and every later run that inherits
the vocabulary. It is also a **lower bound**: the candidate was fitted on a proxy composition
(3:1 prose:code) where the real one is 88:12, so a vocabulary fitted on the real thing would do
better still. That last sentence is an inference, not a measurement.

Freezing carries a second permanent cost that is easy to miss: V=32,773 instead of the 20,000
p1-small is sized for adds **13.1M embedding parameters, +4% on 323M active**, and 64.9% of those
slots are provably dead on this corpus.

**Two conditions on the rebuild.**

1. **Fit on the classifier's keep set, not the raw pool.** The pool is ~18.8B gross; p1 trains on
   the 3-6B that survives filtering, and phi-1's keep rate was 17%. Fitting on the pool feeds the
   vocabulary the statistics of the 70-80% of documents about to be discarded. This costs no extra
   time: tokenization already waits for the corpus to be final.
2. **Measure the tax on held-out text** (44's condition, and **already satisfied**). The repo
   precedent is `facts/tokenizer.json#11` -- fit on a stratified sample, evaluate on held-out text.
   The candidate was fitted on a proxy composition (`en_c4_stage2` + `code_py_starcoder` +
   `code_py_rp1t`, a 62.5M-token sample) while the tax was measured on the three *deduplicated*
   domains plus synthetic textbooks that did not exist when the candidate was fitted; seeds
   7/13/21 are three independent evaluation samples, none of them fitting text. Held-out
   evaluation is now a step inside `scripts/build_p1_tokenizer.py`, so the next rebuild satisfies
   this by construction rather than by remembering. **Held-out and fit overlap at ~0.3%** --
   reported rather than claimed as zero, and negligible against a 3.4% effect, but it belongs in
   the fact's `uncertainty` when the number lands.

## p1 has no math, and that is a decision

`CLAUDE.md` states this project's objective as "a reasoning model targeting coding **and math**
capability". p1's corpus has no math domain. That is a deliberate narrowing to the user's current
instruction -- HumanEval ~60 -- and not an omission.

The consequence, recorded so it is traceable: the reproduction of `math_owm` was stopped on
2026-09-09 because nothing in p1 reads it. **If math comes back, its corpus work restarts from
here.** `code_py_starcoder` was stopped for a different reason -- it is upstream of
`code_dedup08`, which p1 does read, and reproducing an upstream does not validate the downstream
bytes p1 actually consumes. Nothing was deleted in either case.

## Acceptance: one falsifiable gate, not a checklist

A corpus is good enough iff **a 350M dense model trained on it clears HumanEval 30%.** phi-1-small
reports 45% at that size on 7B tokens, so a run landing far below it says the data is wrong and no
increase in scale repairs it. The gate costs about a day; the 1.3B run costs about five. It runs
first.

## Per-line acceptance criteria

| line | owner | socket | acceptance |
|---|---|---|---|
| teacher serve + synthetic textbooks | de | `uds:/tmp/cc-socks/62973.sock` | measured tok/s on the tileRL serve BEFORE sizing anything; 50-sample readability judgement; topic coverage cross-table against the exercise set |
| synthetic exercises | 44 | `uds:/tmp/cc-socks/62780.sock` | execution pass rate with the discard rate recorded; decontaminated against HumanEval and MBPP; topic distribution table; 50 samples, two readers, agreement recorded |
| educational-value classifier | e1 | `uds:/tmp/cc-socks/56034.sock` | held-out AUC against teacher labels; keep rate stated against phi-1's ~17%; **threshold ablation run on our own corpus**; 50 high-scoring and 50 low-scoring samples, two readers |
| topic seeds, dedup, decontamination | 3b | `uds:/tmp/cc-socks/63595.sock` | 20K topic table with a coverage measure; decontamination carries a known-positive control; a self-repetition metric for the synthetic set |
| tokenizer + eval harness | b0 | `uds:/tmp/cc-socks/56758.sock` | temp 0.2 / top-p 0.95 / 20-sample pass@1 sharing one judge with the greedy path, both reported; tokenizer rebuild decision from `tokenizer_eval` on a sample of the new composition |
| human spot check | 98 | `uds:/tmp/cc-socks/34653.sock` | one table, one row per artifact, each with n, two readers, agreement, disagreement count, and a mix/no-mix verdict; a row without an agreement rate does not count |

**The socket column is the point of the table, not decoration.** The first revision named
owners by roster nickname alone, and one of those nicknames -- `d1` -- is not a member of
`runs/roster.json` at all, while `de`, who is actually running the teacher serve, had no row.
Dispatching from this table on 2026-09-09 sent four lines to the wrong sessions: `aupai-dd` is de
and was addressed as b0, `lessons-d1` is b0 and was addressed as d1, e1's rulings went to
`lessons-e1` (whose roster comment reads "lessons-e1 is NOT e1"), and 3b's line went to
`lessons-31`, which is on no roster. Every one of them was caught by a peer, none by the
dispatcher. `runs/roster.json` already carried the rule -- address by socket, never by name --
and it was not read, so the address now sits in the table someone dispatches from rather than in
a second file they have to remember to open.

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
