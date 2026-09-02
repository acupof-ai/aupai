---
question: "has the 22B model learned to copy context rather than predict — and if not, what is the free-running collapse (eight pre-registrations, one chain)"
status: measured
source: "e1, 2026-09-01 — eight pre-registrations written in one day, each before its artifact existed; merged 2026-09-02 (44-13, audit §6b). fone_ab_prereg.md stays open (runs/fone_ab.json absent)."
---

# The copy-hypothesis chain

One investigation chain, one day (2026-09-01): **has the 22B model learned to
copy context rather than predict — and if it has not, what is the free-running
collapse?** Eight pre-registrations, each fixing its falsification bands before
any artifact existed. The tables below are those bands, kept verbatim; the
result line under each says which row the measurement landed in. The facts
named are the recorded verdicts; the artifacts are the raw numbers.

**Common setup** (every section unless stated): `ckpt_pretrain_30b_s2.pt.step24000`,
code-500 / math-500 golds, `rep_stop=False`
(`be.rep_stop_truncates_the_thing_it_measures`), shift-aligned to ±150 — the ±4
window reported 0.0 where the truth was 0.23, and every instrument here is sized
against that lesson. **Not a verdict anywhere below — fb rules.**

The chain, in order:

| # | question | settled by |
|---|---|---|
| 1 | is the model copying context, and is val blind to it | `be.self_repetition_not_context_copying` |
| 2 | how far does agreement fall under the model's own prefix | `be.free_running_agreement_collapses_below_teacher_forced` |
| 3 | is knowledge accumulating that generation cannot express | `be.gold_bpb_falls_while_generation_scores_zero` |
| 4 | is the gold reachable under sampling | `be.gold_is_ranked_high_but_not_reachable_by_greedy` |
| 5 | is the collapse preamble structure (my own refutation) | `be.preamble_is_not_the_cause_of_the_collapse` |
| 6 | is the anchored 19% representative of the other 81% | `be.the_anchored_subsample_was_flattering_and_the_anchor_was_blind` |
| 7 | is the loss immediate at bin 1 on generations that reach code | `be.the_loss_is_immediate_not_gradual` |
| 8 | immediate or gradual — the mechanism | `be.the_loss_is_immediate_not_gradual` |

---

## 1. The copy hypothesis (P0)

**Question:** has the 22B model learned to copy context rather than predict, and
is our val loss blind to it (val drawn from the same never-deduplicated corpus)?
fb P0, from the user. Checkpoint confirmed on the pod at 959 MB, lane card GPU 7.

### What each arm must show

**Arm 1 — copy rate.** Fraction of generated 8-grams already in the prompt,
reported as a **distribution over prompts, not a mean** — a bimodal mix of
copiers and non-copiers has the same mean as uniform mild copying.

| observation | reading |
|---|---|
| median copy rate ≥ 0.90 | consistent with copying |
| median ≤ 0.30 | **hypothesis falsified on this arm** |
| 0.30–0.90 | neither; report the distribution and refuse a verdict |

**Arm 2 — the null that must not be skipped.** A base model under greedy
decoding repeats by construction, so repetition alone proves nothing. Three
arms, same prompts, same checkpoint: greedy (t=0), t=0.8, 3-shot with
**unrelated** examples.

| observation | reading |
|---|---|
| copy rate stays ≥ 0.90 across all three | copying is **learned behaviour** |
| copy rate collapses under sampling or few-shot | **hypothesis wrong** — greedy decoding on an untuned base |

Pre-committed "collapses": a drop of ≥ 0.40 absolute in median copy rate from
greedy to either other arm. < 0.15 is no collapse. In between, report both, no
verdict. This arm is the one that can kill the hypothesis, so it is the one most
worth running properly.

**Arm 3 — external-text loss.** Token loss on text certainly not in our corpus
nor its sources, vs val loss on the same domain. The unseen-text candidate is
committed in advance: authored after the corpus cutoff, from a domain never
fetched (cross-checked against `datagen/fetch_corpus.py`), run through the
exact-dup and near-dup predicates before scoring — **a candidate that matches
anything is discarded, not explained away.** If the conditions cannot be met,
the arm reports ABSENT with the reason.

| observation | reading |
|---|---|
| external − val ≥ 0.50 nat/token | val is measuring memorisation |
| external − val ≤ 0.15 nat/token | **hypothesis falsified on this arm** — val generalises |
| 0.15–0.50 | report the gap, no verdict |

**Falsification, one line:** the hypothesis dies if copy rate collapses under
sampling or few-shot (arm 2), OR external−val ≤ 0.15 (arm 3). It survives if
copy rate holds ≥ 0.90 across all three arms AND external−val ≥ 0.50.

**Known ways this could still be wrong** (stated now, not discovered as excuses
afterwards): 8-gram copy rate is decoder- and tokenizer-sensitive (an 8-gram in
tokens is not one in words; Chinese tokenises differently); "unrelated" 3-shot
examples may not be unrelated enough (shots from the arm-3 external text where
possible, a coupling worth naming); prompt selection can manufacture either
answer (same prompt set across all three arms, state where it came from); the
lane card is shared with stage-2 neighbours — wall-clock only, not the numbers.

**Result:** `be.self_repetition_not_context_copying` — the model does NOT copy
its context; it repeats ITSELF, and only under greedy decoding. Arm 2 killed
the hypothesis. Artifacts: `runs/copy_arms_step24000.json`,
`runs/copy_probe_step24000.json`, `runs/external_loss_step24000.json`
(probes/t69_copy_rate.py@be8e94a, t67_self_repeat.py@d217bad, GPU7).

---

## 2. The free-running arm

**Question:** how far does per-token gold agreement fall when the model
conditions on its own prefix instead of the gold's?

`be.gold_is_ranked_high_but_not_reachable_by_greedy` measured teacher-forced
gold top-1 at 72.7% (code) / 69.3% (math) — every position scored given a
*correct prefix the model would not have produced*. That answers "can it
continue a correct answer", not "can it produce one": an upper bound, biased
toward the decoding-deficit conclusion. This arm measures the same quantity
under the model's own prefixes (200 code-500 + 200 math-500, t=0.8, greedy too
so at least one arm matches the decoder that produced the zeros). Also the
corrected version of the vacuous argmax row: gold sequence log-prob against each
*sampled* sequence's log-prob over k=8 — a sampled sequence is not the arg-max,
so gold can genuinely beat it.

| observation | reading |
|---|---|
| FR ≥ 0.60 (within ~0.12 of TF) | the decoding reading **holds**; own prefixes do not destroy the distribution |
| FR ≤ 0.30 (less than half of TF) | **"decoding and search" understates the problem** — own prefixes destroy the distribution; error compounding is self-reinforcing |
| 0.30 < FR < 0.60 | partial; report FR and FR/TF, no verdict |
| gold log-prob beats ≥ 25% of sampled sequences | the model **prefers** gold to what it samples — a real search failure |
| gold log-prob beats ≤ 5% of sampled sequences | the model prefers its own output; the deficit is not search alone |

**Two ways this misleads:** prefix-aligned comparison punishes a correct answer
phrased differently — one extra token early desynchronises every later
position, so a low FR is ambiguous between "wrong tokens" and "shifted tokens"
(report best alignment over ±4 shifts alongside the naive one); t=0.8 is not
greedy, so FR-at-0.8 vs TF is not quite like-for-like (the greedy arm covers
this).

**Result:** `be.free_running_agreement_collapses_below_teacher_forced` —
IT COLLAPSES: TF 72.7%/68.8% against free-running ~0.23 alignment-free. Lands
in the pre-registered ≤ 0.30 band: "decoding and search" understates the
problem. Artifact: `runs/free_running.json` (probes/t68_free_running.py@7a77159,
GPU6; shift-window sweep and LCS check on GPU7 after the collapse looked too
clean).

---

## 3. Gold-answer BPB across checkpoints

**Question:** is knowledge accumulating across checkpoints in a way generation
cannot express?

Every instrument in use passes through a decoder, and the decoder is broken in a
way that produces zeros (74–80% of greedy generations loop; the repetition guard
truncates the metric that would have measured it). **Gold-answer BPB has no
decoder**: conditional NLL of the gold string given the prompt, divided by the
gold's UTF-8 byte count. No sampling, no greedy, no fence parsing, no stop
condition — none of today's confounds can reach it. It is also the metric that
resolves at our scale, where generative scores sit on the floor
(`be.gold_bpb_method`).

**Not the level. Whether it falls.** Checkpoints in token order:
`ckpt_0830v1_3.24b` (p324) → 8B → 15B → the 16B pin → `step24000` (22B), scored
on the same code-500 / math-500 golds the generative evals use.

| a monotone fall licenses | it does NOT license |
|---|---|
| "the model assigns increasing probability to correct answers" | "the model can do math" |
| "knowledge is accumulating that generation cannot express" | "the generative zeros are only a decoding artifact" |
| "the 0.0 scores are not proof of learning nothing" | "SFT or a decoder change will recover a specific score" |

The gap between "assigns probability to the right answer" and "can produce it"
is exactly what SFT and decoding are for.

| observation | reading |
|---|---|
| BPB falls monotonically across all five checkpoints on **both** sets | knowledge is accumulating; the generative zeros are not the whole story |
| BPB is **flat** (total change < 0.02 bits/byte end to end) | **the hypothesis dies** — "learned nothing" survives as the reading |
| BPB **rises** | worse than flat; something is being lost, outranks the decoding question |
| falls on one set, flat on the other | domain-specific; report per set, refuse a single verdict |
| non-monotone but net-falling | report the shape; a dip is not noise until the seed spread is known |

A number I cannot yet supply: the seed spread of gold BPB at this scale is
unmeasured, so "monotone" is a shape claim, not a significance claim. If the
total fall is under ~0.05 bits/byte I will say the trend is within a range
nobody has bounded.

**Two ways this can be wrong:** a shorter gold is not a better-predicted gold
(BPB divides by bytes) — the golds are fixed strings from a fixed file,
identical for every checkpoint, so this cannot happen here, stated because it
is the standard way this metric breaks; tokenizer drift across checkpoints —
byte normalisation absorbs some of it, which is why BPB and not per-token loss,
but the tokenizer is verified identical across all five and ABSENT is reported
for any checkpoint where it is not.

**Result:** `be.gold_bpb_falls_while_generation_scores_zero` — YES ON CODE,
YES-WITH-ONE-BUMP ON MATH. Code BPB down 15.6% across the ladder while code_500
generative stays 0.0. Artifact: `runs/gold_bpb.json`
(probes/t65_gold_bpb.py@c0683b7, GPU6; tokenizer file fp verified identical,
recorded as `tokenizer_file_fp`).

---

## 4. Is the gold reachable?

**Question:** is the gold answer reachable under sampling, or is rising gold
probability trapped below the sampling threshold?

`be.gold_bpb_falls_while_generation_scores_zero` established rising probability
on a fixed gold string — a model could assign rising probability to one phrasing
and still produce nothing usable. Two deficits with the same symptom and
opposite fixes:

| if | the deficit is | the fix is |
|---|---|---|
| gold probability rises but the model never samples it | **decoding and search** | temperature, top-p, beam, best-of-n, a sampler change |
| gold is sampled but wrong answers dominate | **knowledge** | more tokens, better data, SFT |

Measured per problem (200 code-500 + 200 math-500): **gold rank** (fraction of
gold tokens in the model's top-1/top-10/top-100), **gold mass** (sequence
probability of the gold vs the model's greedy continuation, one scale),
**sampled reachability** (k=32 at t=0.8: is the gold ever produced verbatim,
and the best per-token agreement any sample achieves).

| observation | reading |
|---|---|
| gold top-1 fraction ≥ 0.50 **and** gold never sampled in k=32 | **decoding deficit** — ranked first, sampling still cannot assemble the string |
| gold top-1 fraction ≤ 0.15 | **knowledge deficit** — no sampler will find it |
| gold sequence probability ≥ greedy's on ≥ 30% of problems | the model *prefers* gold to what it emits — decoding/search failure by definition |
| gold sequence probability < greedy's on ≥ 90% of problems | the model prefers its own wrong output; knowledge deficit |
| 0.15 < top-1 < 0.50 | **no verdict** — report the distribution; the outcome I expect, and not a failure of the measurement |

Predicting the ambiguous outcome beforehand is the only way to stop myself
narrating a clean one afterwards.

**Two ways this misleads:** a long gold cannot be sampled verbatim at any
temperature (sequence probability falls geometrically in length — "never sampled
in k=32" is close to guaranteed for a 200-token gold; the **per-token rank** is
the load-bearing statistic, the verbatim count is a sanity check, not evidence);
teacher forcing biases toward the decoding-deficit conclusion — report
free-running agreement alongside, and if the two disagree, distrust the
teacher-forced number.

**Result:** `be.gold_is_ranked_high_but_not_reachable_by_greedy` — gold ranked
high (72.7% code / 69.3% math top-1) and greedy still does not produce it;
lands in the pre-registered DECODING-AND-SEARCH band, with the boundary that
one of my own falsification rows turned out vacuous. Artifact:
`runs/gold_reachability.json` (probes/t66_gold_reachability.py@0ea4530, GPU6).

---

## 5. Preamble refutation (my own)

**Question:** is the free-running collapse preamble structure or distribution
drift? The refutation I named on `be.free_running_agreement_collapses_below_teacher_forced`:
if the collapse is preamble structure, it is a different and much cheaper
problem.

**The framing I got wrong, corrected before running.** I wrote the refutation
as "prompts whose gold begins immediately with no preamble." Inspecting the
data first: **the golds already have no preamble** — `code_holdout_500`'s
references start at `def is_prime(x):` or `a = [2, 2, 6]`, column one. **The
preamble is the model's, not the dataset's** (generations open with
`。\n\n### 例子\n\n假设我们有一个整数...` and only later reach a fenced block).
So the measurement is not "select aligned prompts" — there are none — it is:
**strip the model's own preamble and score from where it starts producing
code.** Discovering the original phrasing was unrunnable AFTER the run would
have looked like a result.

Measured (100 code prompts, greedy): **anchored agreement** (find the first
code-looking token — a ```python fence, or a line opening `def `/`class `/an
assignment — drop everything before it, score per-token agreement from there);
the same for the gold as the control that the anchor logic is not itself
creating the alignment; **coverage** — how many generations contain any anchor
at all (10/40 on a pre-sample, load-bearing).

| observation | reading |
|---|---|
| anchored agreement ≥ 0.60 on the anchored subset | **the collapse is preamble structure** — the free-running reading is overturned, the problem is cheaper |
| anchored agreement ≤ 0.30 | preamble is not the cause; **distribution drift stands** |
| 0.30–0.60 | partial; report the figure and the coverage, no verdict |

Sizing the instrument this time: the anchor search scans the WHOLE generation;
after anchoring, the alignment-free LCS ratio is reported so a residual offset
cannot masquerade as disagreement; agreement over shifts up to ±150, the full
generation length, not a window chosen by guess.

**The limit that will probably decide this:** only 10 of 40 generations
contain a code anchor at all. The anchored score is computed on roughly a
quarter of the set, **selected for having produced code** — the best-behaved
generations. A high anchored score does not overturn the collapse for the other
three quarters. The honest statement, written now so I cannot write a stronger
one later: *"the collapse is preamble structure for the 25% of prompts that
reach code, and unmeasured for the rest."*

**Result:** `be.preamble_is_not_the_cause_of_the_collapse` — DISTRIBUTION DRIFT
STANDS. Anchored 0.2609 (shift) / 0.4524 (LCS) on 19/100 — lands in the
0.30–0.60 / ≤0.30 boundary region, and the coverage limit binds. Artifact:
`runs/preamble_refutation.json` (GPU7).

---

## 6. The other 81%

**Question:** is the anchored agreement measured on the best-behaved 19%
representative of the other 81%?

`be.preamble_is_not_the_cause_of_the_collapse` reported anchored agreement on
**19 of 100** generations — the ones that produced a recognisable code anchor,
selected for having produced code at all. Every anchored figure is computed on
the best-behaved fifth. This is not a caveat on the number; it is a reason the
number cannot be generalised.

**Why a better anchor, rather than more prompts.** The current anchor is a
regex (```python fence, or a line opening `def `/`class `/assignment). A
generation that produces correct code with none of those markers is invisible,
so **coverage is a lower bound on "reached code", not a measurement of it**.
Adding prompts multiplies the same blind spot. The first question is how much
of the 81% is genuinely code-free versus merely unmatched by my regex.

Measured (same 100 code prompts, greedy): coverage under a WIDENED anchor
(indented continuations, `import `/`from `, `return `, `print(`, `for `/`while `/`if `
at line start, any line containing `=` with balanced brackets); agreement on
the newly-anchored generations **reported separately** from the original 19
(pooling lets the well-behaved fifth carry the average again — the defect being
tested); what the still-unanchored generations contain, by inspection of a
sample — read, not inferred.

Let **A19** = 0.2609 (shift-aligned, original subset) and **Anew** = the same
statistic on generations anchored only by the widened rule.

| observation | reading |
|---|---|
| Anew ≥ 0.40, and coverage rises above 50% | **the drift reading weakens** — the unanchored majority was doing better than the subsample suggested |
| Anew ≤ 0.20 | **bin-1 collapse is understated** — the favourable subsample was flattering the model |
| 0.20 < Anew < 0.40 | the subsample was roughly representative; drift stands as measured |
| coverage stays below 30% even widened | **the remainder is genuinely code-free** — the model mostly does not emit code, and agreement on the minority that does is not the interesting statistic |

The last row is the outcome I expect — naming it in advance because it is the
one that makes the whole anchored line of work a side quest rather than a
result.

**Result:** `be.the_anchored_subsample_was_flattering_and_the_anchor_was_blind`
— TWO ANSWERS, AND THE FIRST IS A DEFECT IN MY OWN INSTRUMENT. (1) The 19%
coverage was a regex artifact, not a property of the model: 31 of the 100 are
degenerate loops with no code at all — read, not inferred. (2) The favourable
subsample was flattering the model. Artifacts: `runs/coverage_anchor.json`,
`runs/positional_anchored.json` (GPU5).

---

## 7. Bin 1 on the generations that reach code

**Question:** on generations that actually reach code, is the loss still
immediate at bin 1?

`be.the_loss_is_immediate_not_gradual` measured bin-1 agreement at 0.250 (code)
over all 100 generations — but 31 of those 100 are degenerate loops with no
code at all (§6). A degenerate loop scores near zero at every bin, dragging bin
1 down for a reason unrelated to distribution drift. The mechanism claim
deserves the clean comparison: bin 1 on the 69 generations that produce
code-shaped output.

Positional bins 1–8, 9–16, 17–32, 33–64, 65–128, shift-aligned to ±150, on
three populations **reported separately and never pooled**: the original 19
(old anchor), the newly-caught 50 (widened anchor only), the 31 unanchored
(included precisely to confirm they behave as the degenerate-loop reading
predicts, rather than being assumed to).

| observation on the anchored 69 | reading |
|---|---|
| bin1 ≥ 0.55 | **immediate-loss claim collapses** — it was an artifact of averaging over degenerate loops |
| bin1 ≤ 0.40 and TF→bin1 drop > bin1→last decay | **immediate loss confirmed on real attempts** — the mechanism survives the correction |
| 0.40 < bin1 < 0.55 | partial; report all three populations, no verdict |

This can overturn my own mechanism claim, and that is the point of running it.
I do not expect it to: 0.250 over all 100 with 31 near-zero contributors implies
roughly 0.36 over the remaining 69 even if the loops score exactly zero —
still well inside the immediate band. **Writing the arithmetic down now so I
cannot present a value near 0.36 as a surprise either way.**

**Result:** `be.the_loss_is_immediate_not_gradual` — IMMEDIATE. The mechanism
survives the correction. Artifact: `runs/positional_anchored.json` (GPU5).

---

## 8. Immediate or gradual — the mechanism

**Question:** is the free-running loss immediate at the first step off a
correct prefix, or gradual accumulation?

Two mechanisms produce the same collapse and imply different work:

| mechanism | signature | what it implies |
|---|---|---|
| **gradual compounding** | agreement decays smoothly with distance from the prompt | a long-horizon problem; better search or shorter generations help |
| **immediate off-distribution** | agreement drops sharply within the first few tokens, then decays slowly from an already-low level | the conditional distribution is calibrated on gold prefixes and wrong on its own from step one; search does not help |

I have been telling the compounding story. It may be wrong.

Agreement binned by distance from the prompt (tokens 1–8, 9–16, 17–32, 33–64,
65–128, 129–192), 100 code + 100 math prompts, greedy, teacher-forced ranks as
the baseline. Let **B1** = free-running agreement in tokens 1–8, **TF** = 0.727
(code).

| observation | reading |
|---|---|
| B1 ≥ 0.55 and later bins decay smoothly toward ~0.1 | **gradual compounding** — my story stands |
| B1 ≤ 0.40, and the drop TF→B1 exceeds the drop B1→last bin | **immediate off-distribution** — the first step off a gold prefix costs more than the entire remaining horizon |
| B1 between 0.40 and 0.55 | partial; report the curve, no verdict |

The second row's second clause is load-bearing: **"immediate" means the initial
drop is larger than the subsequent decay**, not merely that bin 1 is below TF —
bin 1 is below TF under either mechanism.

**Provenance note:** a ruling reached me citing 0.343 in the first eight tokens
decaying to 0.078, attributed to me. I did not produce those numbers and cannot
find them in `facts/`, `runs/`, `docs/`, `probes/` on either branch or on the
pod — flagged separately. This pre-registration deliberately fixes its bands
without reference to them, so that if they turn out to be real this is an
independent replication, and if they do not, nothing here inherits them.

**The limit, sized rather than named:** positional bins are not independent
measurements — once a generation diverges, every later bin is scored on a
trajectory already off the gold, so late-bin agreement is "accuracy given ~100
tokens of prior divergence", not per-position accuracy. The profile shows where
the loss BEGINS. Bin 1 is the only bin conditioned on a mostly-correct prefix
and the only one comparable to the teacher-forced number; comparisons of later
bins against TF are not like-for-like and are not made.

**Result:** `be.the_loss_is_immediate_not_gradual` — IMMEDIATE, and this
REFUTES my own compounding story. Artifact: `runs/positional_profile.json`
(GPU5).
