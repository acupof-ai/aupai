---
question: "has the 22B model learned to copy context rather than predict, and is our val loss blind to it"
status: open
source: "e1, 2026-09-01, written BEFORE any of the three measurements were run; fb P0 from the user"
---

# Pre-registration: the copy hypothesis

> **`status: open` here means pre-registered, not in-progress.** The falsification
> conditions below are fixed and no measurement exists yet. `FRONTMATTER_STATUS`
> has no `preregistered` value and widening a repo-wide enum mid-P0 is not my
> call; flagged to fb.

Written before the artifacts exist. `runs/copy_probe_step24000.json`,
`runs/copy_arms_step24000.json` and `runs/external_loss_step24000.json` are all
absent as of this commit; `git log` on this file against those paths is the
proof of ordering.

**Checkpoint**: `ckpt_pretrain_30b_s2.pt.step24000` (22B tokens), lane card GPU 7,
one job at a time. Confirmed present on the pod at 959 MB.

**The hypothesis under test** (fb, from the user): the model has learned to copy
context rather than predict, and val loss cannot see it because val is drawn from
the same never-deduplicated corpus.

## What each arm must show

### 1. Copy rate

Fraction of generated 8-grams that already appear in the prompt/context.
Reported as a **distribution over prompts**, not a mean — a bimodal mix of
copiers and non-copiers has the same mean as uniform mild copying and means
something completely different.

| observation | reading |
|---|---|
| median copy rate ≥ 0.90 | consistent with copying |
| median ≤ 0.30 | **hypothesis falsified on this arm** |
| 0.30–0.90 | neither; report the distribution and refuse a verdict |

### 2. The null that must not be skipped

**A base model under greedy decoding repeats by construction, so repetition
alone proves nothing.** Three arms, same prompts, same checkpoint:

- greedy (temperature 0)
- temperature 0.8
- 3-shot, where the shots are **unrelated** examples

| observation | reading |
|---|---|
| copy rate stays ≥ 0.90 across all three | copying is **learned behaviour** |
| copy rate collapses under sampling or few-shot | **hypothesis wrong** — this is greedy decoding on an untuned base |

**Pre-committed threshold for "collapses": a drop of ≥ 0.40 absolute in median
copy rate from greedy to either other arm.** A drop of < 0.15 is no collapse. In
between, report both numbers and no verdict.

This arm is the one that can kill the hypothesis, so it is the one most worth
running properly. If I skip it and report only arm 1, a near-1.0 copy rate would
look like a finding and be an artifact of the decoder.

### 3. External-text loss

Token loss on text that is certainly not in our corpus **and not in its sources**,
compared against our val loss on the same domain.

| observation | reading |
|---|---|
| external − val ≥ 0.50 nat/token | val is measuring memorisation |
| external − val ≤ 0.15 nat/token | **hypothesis falsified on this arm** — val generalises |
| 0.15–0.50 | report the gap, no verdict |

**The text I will use, and why I believe it is unseen** — this is the part that
can invalidate the arm, so it is committed here in advance rather than chosen
after seeing a result:

- Text authored **after the corpus cutoff**, so it cannot be in any snapshot our
  fetchers saw. I will state the cutoff, the publication date, and the fetch
  provenance in the result.
- A **domain we never fetched at all**, cross-checked against the source list in
  `datagen/fetch_corpus.py`.
- Before scoring, every candidate document is run through the exact-dup and
  near-dup predicates against the corpus. **A candidate that matches anything is
  discarded, not explained away.**

If I cannot satisfy those conditions for a text, the arm reports **ABSENT with
the reason**, not a number with a caveat. An unseen-text claim I cannot support
is worse than no measurement, because the loss gap is only interpretable if the
text is genuinely unseen.

## Falsification, stated as one line

**The hypothesis dies if:** copy rate collapses under sampling or few-shot (arm
2), **or** external-minus-val is ≤ 0.15 nat/token (arm 3). Either alone is enough
to refute; arm 1 on its own can neither confirm nor refute, because greedy
decoding produces repetition in any untuned base.

**The hypothesis survives if:** copy rate holds ≥ 0.90 across all three decoding
arms **and** external-minus-val ≥ 0.50 nat/token.

Anything else is a partial result and will be reported as one.

## What I will report

Numbers, distributions, and the conditions each was measured under. **Not a
verdict** — fb rules. Each arm names the artifact it wrote and the command that
produced it, and each artifact is attested.

## Known ways this pre-registration could still be wrong

Stated now, so they are not discovered as excuses afterwards:

- **8-gram copy rate is decoder- and tokenizer-sensitive.** An 8-gram in tokens
  is not an 8-gram in words, and a Chinese corpus tokenises very differently from
  an English one. I will report the unit explicitly and, if time allows, both.
- **"Unrelated" 3-shot examples may not be unrelated enough.** If the shots come
  from the same corpus, a copier can copy from the shots. I will pick shots from
  the external text of arm 3 where possible, which makes the two arms share a
  provenance assumption — a coupling worth naming.
- **Prompt selection can manufacture either answer.** Prompts drawn from the
  corpus favour copying; prompts drawn from outside disfavour it. I will run the
  same prompt set across all three arms and state where it came from.
- **The lane card is shared with the stage-2 run's neighbours.** Nothing here is
  a timing measurement, so contention affects wall-clock only, not the numbers.
