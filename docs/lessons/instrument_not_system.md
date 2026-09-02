---
question: "how do you notice that a measurement describes the measuring tool rather than the system"
status: recorded
source: "tilerl-10 2026-09-01; six instances found in one day: runs/t56_elementwise_owner.json (the broken join, and its fix breaking the same way on cuda_driver), the pretrain_30b_s2 log's tok/s and ETA format strings, eff.kda_occupancy_bound (config drift), eff.quant_tax_is_the_elementwise_group (off-config trace). Check proposal reviewed by fb, tightened by 62; de-8 carries the seven-sighting class ticket."
span_ms: 1676.63  # t57's own span; the percentages quoted here are against it
---

# The instrument, not the system

## The closing lesson: naming a confound is not sizing the instrument for it

Above the others because it is the failure that survives every practice on this
page. You can pre-register, name the exact confound, commit the falsification
bands in advance — and still ship a number that is wrong in the direction you
expected, because the instrument was not built large enough to see the thing you
named.

Free-running agreement, 2026-09-01. The pre-registration said, in advance:

> *"Prefix-aligned comparison punishes a correct answer phrased differently. If
> the model emits one extra token early, every later position is compared
> against the wrong gold index... I will report the best alignment over small
> shifts (±4 tokens) alongside the naive one."*

The confound is named correctly. The window is ±4. **The model's generations open
with a preamble of forty to eighty tokens.** So the shift-tolerant variant was
almost as desynchronised as the naive one, and both read near zero:

| instrument | reading |
|---|---|
| naive, prefix-aligned | 0.0000 |
| ±4 shift (the pre-registered variant) | 0.0483 |
| ±16 | 0.1026 |
| ±64 | 0.1883 |
| ±150 | 0.2059 |
| alignment-free (LCS ratio) | **0.2317** |

**The pre-registered instrument reported one fifth of the truth**, in the
direction that made the headline stronger.

Three things make this the worst shape on the page:

1. **Every rigour marker was present.** Pre-registered, bands fixed before the
   run, confound named in writing. None of them sizes an instrument.
2. **It errs toward the author's hypothesis.** A too-small window can only
   *lower* agreement, and low agreement was the interesting result. A confound
   named but under-provisioned is not neutral noise; it is a thumb on the scale
   pointing where the author was already looking.
3. **The check is not "did I think of it" but "can the instrument see it".**
   Those feel identical while writing the pre-registration and are not.

It was caught by the oldest tell on this page — **a number too clean for the
thing it claims to measure.** 0.0000 median from a model scoring 72.7%
teacher-forced is not a low score, it is a broken ruler.

The operational form: **for every confound you name, state the magnitude it can
reach and show the instrument covers it.** "I will report a shift-tolerant
variant" is a plan. "Preambles here run 40–80 tokens, so the search must span at
least 150" is an instrument. Where the magnitude is unknown, sweep the parameter
and report the curve rather than picking a value — the sweep above is what
produced the real number, and it cost one extra run.

## Four instruments of mine were the limit, and how each was caught

The empirical case for this page, made by one author against himself in one
afternoon. Not four measurement errors — four occasions where **the tool
reported a property of itself and I read it as a property of the model.**

| # | the instrument | it reported | the truth | caught by |
|---|---|---|---|---|
| 1 | ±4 shift window | agreement 0.0000 | 0.2317 | a number too clean for a model at 72.7% TF |
| 2 | anchor regex | 19/100 reached code | 69/100 | pre-registered expectation falsified |
| 3 | gold-vs-greedy row | 0.0 "never prefers gold" | an identity; can only read 0.0 | deriving what the row could return |
| 4 | bin-1 population split | "loops score ~0, so 69 ≈ 0.36" | loops score 0.25 too | pre-registered arithmetic falsified |

**Start with #3, because it is the only one that needs no run at all.** It was
caught by asking what the test *could* return, not what it did: `greedy_logprob`
sums per-position maxima and `gold_logprob` sums per-position gold values, so
max ≥ gold holds at every position by construction and the row can only ever
read 0.0. **An output space containing one value is an identity in a test's
costume, and that is checkable before any data exists.** Cheapest check on this
page — derive the range of your statistic before you compute it.

**The other three needed a number.** None was caught by reading the code. Every
one was caught by a value that did not fit something already known — and in two
of those three, the thing it did not fit had been written down in advance.

**The ratio is the argument.** Two of four were caught *only* because a
pre-registration made the expectation explicit enough to be falsified. #2 said
"I expect coverage under 30%, which would mean my anchored line of work was a
side quest"; coverage came back 69% and the prediction failed in the direction
that would have demoted my own work. #4 wrote the arithmetic "0.250 over 100
with 31 near-zero loops implies ~0.36 over the remaining 69"; it came back 0.25,
which falsified **the assumption inside the prediction** rather than the
conclusion the prediction was defending — the loops do not score near zero,
because a loop echoing the prompt matches the gold's opening tokens by
coincidence.

Without those two sentences on paper, both results read as bland confirmation.
0.25 is exactly what you would expect if nothing interesting happened; it took a
written prediction of 0.36 to make it a finding.

**That is the case for pre-registration that does not rest on principle.** Not
"it keeps you honest" — it converts an unremarkable number into a detectable
anomaly. A prediction you did not write down cannot fail, and a result that
cannot fail teaches nothing.

## Padding that is indistinguishable from content at the point of comparison

A different family from the rest of this page: not an instrument reporting
itself, but **a shape quantity leaking into a semantic one because both are
integers named `vocab`.**

This repo has three fields that read as "vocabulary size" and **only one is a
count of tokens**:

| field | value | what it is |
|---|---|---|
| `Cfg.vocab_real` | 32773 | the tokenizer. The only token count. |
| `Cfg.vocab` | 32784 | +11, padded to a multiple of 16 so the aligned cuBLAS head kernel is chosen |
| `model.padded_vocab` | 32832 | +48 more, padded to 64 for the embedding and head matrices |

Ids 32773–32831 are addressable and decode to nothing: **59 slots of pure
alignment** that a comparison cannot distinguish from vocabulary.

`build_tokenizer.py` compared its built vocabulary against `Cfg.vocab`. Both
sides were plausible integers, the name matched, and the comparison was wrong:
the trainer targeted `32784 - 5 = 32779` merges, so a rebuild would have emitted
a **32784-token vocabulary** — eleven real merges of what exists only to make a
matrix multiply fast. `load_tokenizer` asserts `size == vocab_real`, so the next
load of **every existing checkpoint** would have failed.

**The tell is that the wrong value is not absurd.** 32784 is a defensible
vocabulary size. Nothing about it looks like padding at the moment of
comparison, which is exactly why the comparison survived review — including
mine, since I read that line while answering a different question and only
caught it by deriving what the build would produce.

Three rules, in cost order:

1. **A padded quantity and its unpadded source must not share a name stem.**
   `vocab` / `vocab_real` / `padded_vocab` differ by a suffix, so at every use
   site the correct choice is one character away from an incorrect one that
   still runs.
2. **Comparisons against a count use the count, and shapes use the shape.** The
   test is not "which variable is in scope" but "is this line asking about
   tokens or about matrix width". `train.py:749` gets this right —
   `head.weight[vocab_real:vocab].zero_()` deliberately spans the padding — and
   it reads almost identically to the line that got it wrong.
3. **Where a docstring names one and the code asserts the other, the docstring
   is a defect.** `scripts/loader.py:84` said "size == cfg.vocab" while the code
   asserted `vocab_real`. The code was correct; the sentence a reader trusts was
   not — **and a comment asserting an invariant is checked by nothing.** That is
   why the rule here is to *assert* the invariant rather than describe it: an
   assert fails when it stops being true, and a sentence goes on being read.

**Why this belongs on this page.** The general form is the same as an
off-config trace or a too-small shift window: *a quantity that is correct for
one purpose being read as if it answered another*, with no error at the moment
of the mistake. Alignment padding is content-shaped, and content-shaped noise is
what defeats a comparison.

## Before anything else: is this code path reached in the live configuration?

One line, and it precedes every other check on this page. It is last here only
because it was found last.

`probes/t60_weight_cache.py@8491325` measured a **39.4 ms/step** saving from caching the
head weight's fp8 bytes. The measurement was correct. But `_fp8_mm` exists only on
the FLCE path installed by `patch_liger_flce_fp8`, which runs **only under
`FP8_HEAD=1`** (train.py:2143) — and `FP8_HEAD=1` is itself no-ship at −3.91%
(`eff.fp8_head_ab_noship`). The live run has no `FP8_HEAD` in its environment and
zero "routed through `_scaled_mm`" lines in its log. The lever saves **0 ms of
what runs**, and re-enabling the path with it is −59.6 + 39.4 = **−20.2 ms**,
still a regression.

**Every check I applied passed**: the statistic was pre-declared, the arms were
interleaved, the spread was 0.003–0.007 ms/chunk, and after review the arms
differed only in the treatment. A parity gate was specified. Those are all checks
on *measurement quality*, and **none of them asks whether the thing measured
runs**. Correctness and relevance fail independently, and the checks for the first
do not detect a failure of the second.

The enforceable version, now proposed as a harness check: **any fact whose value
depends on a code path behind a flag or an env var must name the flag and the
condition under which the path runs, and the check FAILs when a fact claims a
production saving for a path the default configuration does not reach.** The flag
name is in the fact; the default is in the source.

62's tightening, which is the version worth building: *"the fact names the flag"*
is satisfiable by a fact that names the flag **and still claims the production
saving**. The check has to read the flag's default from the source and compare it
against the claim, or it tests whether a field is populated as a proxy for whether
the claim is true — which is the same class it exists to catch.

### And the rung below it: the flag can be in the evidence rather than the claim

Found hours later, answering a different question, and it is the harder half.

`eff.quant_tax_is_the_elementwise_group` attributed a 250.6 ms/step group to the
fp8 quantisation tax at 99.98% resolution. The attribution was right. But the
trace it ran on was **captured with `FP8_HEAD=1`** — 181 `aten::_scaled_mm` per
step inside a Liger FLCE region, and only `patch_liger_flce_fp8` puts them there.
Splitting each kernel by whether its launch is contained in a Liger region: 156.9
of the 250.6 ms is head work that does not run live, and the four quantisation ops
are **92.5% head** (153.24 of 165.68 ms). The live tax in that group is 12.4 ms,
not 165.7. The mechanism claim survived; the scope claim did not.

I first published that share as **92.9%**, and two peers quoted it verbatim
before b0's `doc_numbers_check` recomputed it from my own per-owner table. A
rounded figure typed beside the numbers it summarises, in the very entry arguing
that summaries rot. **Recomputing is cheap enough that there is no case for
asserting a derived number you did not just re-derive.**

The fact **never mentioned `FP8_HEAD`**. Nothing in its text was flag-dependent —
the flag was in the capture conditions of a 415 MB file nobody re-reads. A check
that reads flags out of fact bodies passes this cleanly.

So the general form is one step back from where the first version put it: **a
measurement's provenance is part of the measurement.** A fact must record the
configuration its *evidence* was captured under, not only the configuration its
*claim* asserts. Both failures ask the same question — is this the live
configuration — at two different places, and only one of them is in the text.

A second reproducibility trap in the same investigation, worth its own line
because it produces plausible numbers rather than an error: the backup directory
held **different, earlier traces under identical filenames**. Re-running the probe
there returns 29.6 ms where the real capture returns 250.61. A filename is not a
provenance record either.

---

An instrument returning a confident number that describes the instrument rather
than the system. Six instances on 2026-09-01 — the first four inside about an
hour, all in tools we trusted. The tell is the same every time: **a number too
clean for the thing it claims to measure.**

## The rule

**Before using a logged value as a statistic, read its format string.** A derived
statistic's resolution is bounded by the printf that produced its input. Three of
the six instances below are one `%.1f` away from being real measurements.

The corollary, for attribution and joins: **treat any result of exactly 0% or
exactly 100% as a suspected tooling failure until the join is verified on a case
whose answer is already known.**

## The first four

| # | what was reported | what it actually measured | the tell |
|---|---|---|---|
| 1 | tok/s median 77K | `{tok_s/1000:.0f}K` — quantised to 1K | every block identical |
| 2 | tok/s between-block SD = 0 across 118 blocks | the same quantisation, one step further | SD of exactly 0 |
| 3 | block wall-clock SD = 0.4 s over 119 blocks | `{eta/3600:.1f}h` — one 0.1h tick is ±180 s = ±1.03 s/block, 2.5× the "SD" | 119 blocks landing in three discrete values (min 101.8, median 102.7, max 103.7) |
| 4 | elementwise group "100% unattributable" | a broken join: kernels carry `args.correlation`, cpu_ops carry `args["External id"]`, different keyspaces, and the `cuda_runtime` launch event is the required middle hop | 0% resolved |

A fifth, same family but not a printf: `eff.kda_occupancy_bound`'s 6–12% occupancy
was measured by `eval/kda_probe.py` at its `--chunk 64` default, while
`train.py:153` had shipped `chunk_size=32` the same morning. The number was real;
it just described a configuration that no longer runs. **Also check that a cited
number still describes what currently ships.**

A sixth, and it is #4's own fix breaking the same way in the opposite direction.
The corrected join — kernel → `cuda_runtime` → cpu_op — resolves 99.98% of aten
kernels and **0.00% of triton kernels**, because inductor launches through
`cuda_driver` (`cuLaunchKernel`) and aten through `cuda_runtime`
(`cudaLaunchKernel`). Same shape as #4: a confident, complete null for an entire
250-region group, produced by a join that had *just been verified* on a case whose
answer was known. **Verifying a join on one case licenses it for that keyspace and
no other.** Fix is one line — build the correlation map from both categories —
and the recurrence is the argument for making the join a shared helper instead of
re-deriving it per probe.

One more from the same probe, in the tell family rather than the join family: for
a triton kernel, the "innermost launching cpu_op" is the **inductor wrapper of the
same name**. Grouping by it produces a clean table that says
`triton_poi_fused_… ← triton_poi_fused_…`. A tautology renders exactly like an
attribution. If a resolution step returns its own input, it resolved nothing.

## The pair that tells you how to read a moved number

Two broken worlds in `harness.py` changed behaviour in the same commit, when
`_AGE_HOURS` went from 2 to 6. One was a defect and one was not, and the surface
was identical — which is the useful part (both from de, 2026-09-01).

- **`_broken_dirty_aged` — a defect.** The world hardcoded 2 hours; the check's
  threshold moved to 6. The world's age was real and described the world builder's
  own constant rather than the check's live threshold. A constant and a
  hand-written copy of it are two sources of truth, and only one moved. It went
  green **silently** — nothing in the check changed — and only `--selftest`
  running on every commit turned it red.
- **`review_present` — not a defect.** Its broken world started reporting WARN
  instead of FAIL because the FAIL tier had been deliberately removed. The world
  was still correct; the expectation had legitimately moved.

Same symptom, opposite verdicts. **One number stopped describing the system; the
other started describing a system that had changed.** The way to tell them apart
is to ask which source of truth moved: if the code under test moved and the
instrument did not, it is the first kind.

**But that discriminator has to be captured by the guard, not reconstructed by
the reader afterwards** (b0's point, and it corrects an overstatement in the
first version of this page). At the moment you observe the two cases they are
indistinguishable — both are a world that quietly went green. Asking "which
source of truth moved" is only answerable if the guard recorded *what it was
comparing against at the time it ran*. Where no such record exists the question
cannot be settled from the output at all: b0's `holdout_hashes` empty-set
incident is the same shape and had no record, so it took a downstream failure to
surface.

So the rule is not only "ask which moved" — it is **make the guard record its
own comparand**, i.e. the threshold, the expectation, the set size it checked
against. A guard that prints only pass/fail cannot distinguish a defect from a
legitimate change afterwards, no matter how carefully anyone reads it.

That is also the argument for running the selftest on every commit rather than
before a release. Both changes were invisible in the statistic; only the
mechanism caught them, and it caught them the moment they appeared rather than
after they had been quoted somewhere.

## Why #4 is the dangerous one

The three printf cases produce wrong *positive* numbers, and a wrong positive gets
argued with — someone asks where it came from. #4 produced a wrong *negative*:
"the trace cannot name these kernels." Nobody argues with a null. It reads as a
completed investigation with a disappointing result, and it was one commit from
being written into a fact as a finding. The corrected join resolves **99.98%**, and
the answer reversed a rung: 66% of that group is the fp8 quantisation tax, which
merged the "copies" rung into the fp8-head rung instead of refuting it. (That 66%
is a share of the *traced* group, and #5 later showed 93% of it is head work that
does not run live — the mechanism held, the scope did not.)

The cost asymmetry is the point. A false positive costs a review cycle. A false
negative closes a line of work permanently and silently.

## What each cost

None of these were caught by the statistic itself — only by looking at the
instrument. #1–#3 each produced a proposed kill-criteria threshold; the third
would have set a false-precision floor at median + 4 SD = 104.5 s, inside the
quantisation band, firing on rounding. #4 nearly retired a lever quoted first at
4.44%, then at 3.59% — **both superseded, and neither is its live size**; both
were measured on the off-config trace. And here is the twist worth keeping,
because this sentence was wrong for six hours: that lever turned out to be
**92.5% head work on a path the live run does not reach**, so #4's correction
rescued a rung that #5's class then took away again. The live figure is
**12.4 ms**. Both steps were right. A number can be
rescued from one defect and still be carrying another. #6 would have published
"the fusion group is disjoint from the quantisation tax" as a *null* — the right
answer, reached by a broken join, which is the one way to be correct that teaches
you nothing and cannot be trusted next time.

## The generalisation: a threshold set from the wrong quantity

The printf cases are one instance of something wider, and it was named only after
it had been violated four more times in a single afternoon:

> **A threshold set from a quantity that is not the quantity it will be compared
> against.**

The stage-2 loss kill rule is the clean example. The threshold 0.15 came from a
healthy run's **30-step half-split** (+0.12). The rule compares **60-step block
medians**. Nobody computed the block distribution before setting a number against
it. When someone finally did — ten minutes of work — the block-to-block delta SD
was **0.141**, so the threshold was 1.06× the noise it was meant to clear, and
rises past 0.15 occurred in 18 of 137 blocks against falls past −0.15 in 19.
Symmetric: the signature of noise, not drift. The rule fired on a healthy run.

**Two people approved that threshold — the controller who issued it and the
reviewer who sustained it — and neither had the distribution, because nobody had
computed it. Approval by two parties is not a substitute for one measurement.**

The same shape covers everything above: a spread compared against a resolution
nobody read, a component bounded by an aggregate that did not contain it, a
denominator that did not drive its numerator, two instruments summed without
showing they do not overlap.

### It recurs while you are writing the rule against it

This is the part worth internalising. Having written the retraction, I then
derived a lever at 7.5 ms from a peak-bandwidth model and quoted it before
measuring — it measured 39.4 ms, off by 5.3×. Having written *that* down, I
evaluated the replacement kill rule by comparing the event against **the whole
run's median** when the rule's window is **three blocks either side**; at the
rule's own window the event fires. b0 wrote a usage restriction in §6 of a
document and violated its generalisation in §1 of the same document, minutes
apart.

Two authors, six instances, each within an hour of writing down why not to. **The
discipline does not transfer from the case you derived it on to the next case
with the same shape.** That is the argument for a mechanical test over a
principle: a test is checkable against a specific number, a principle is
checkable against nothing.

**A seventh, and the most instructive:** b0 wrote this very rule into a
document's methods section in one commit, and left two live instances of it four
screens above — summary sentences still reading "the software ceiling is
single-digit percent" after the epilogue had lost 36 ms *and* the EVT row had
gone to zero. **Summary sentences are what rot**, because they restate a
conclusion without restating its inputs, so they survive every input changing
underneath them. Any document-level check has to sweep prose, not only tables.

### Two tests, at two levels (44's refinement)

The skim test below operates on a single number, at writing time, held by the
author. It does not catch the document-level failure, where the restriction and
the number it governs live in different sections with no mechanism connecting
them. That needs a second, reviewer-held test:

> **List every restriction the document states. Apply each, mechanically, to
> every number in every table.**

That is a cross-product, not a judgement, and it is exactly how 44 caught b0's §1
— not by understanding the rule better, but by applying b0's own §6 to b0's own
§1. **An author holding a rule is not the same as the rule being applied.** Both
tests are reviewable artifacts; neither substitutes for the other.

### A guard that passes is only informative where it could have failed

The mirror image of the broken join, and the reason a green board is weaker
evidence than it looks. Today's guard failures were **input-specific**:

- the readout's head guard fired correctly on renamed roles, and would have
  stayed silent on identical ones — it was never exercised where it could not
  have fired;
- the budget gate never fired at all, because both callers omitted
  `--actual-tokens`/`--paired-tokens`. It is silently absent on exactly the
  retention pairs where it is the correct check.

So "the checks passed" means *the checks that could fire on this input found
nothing*. That is real and it is much narrower than "the instrument works". When
reading a clean result, ask which guards were even reachable on that input — and
if the answer is unknown, the clean result is not yet evidence of soundness
(b0, on how to read the 22B milestone table).

### A fourth tell: a baseline that pays work the candidate does not pay

Different enough from the printf and join tells to name separately, and it is the
one that survives careful measurement — because the measurement itself is fine.

`probes/t58_quant_tax.py@b15d348` reported the fp8 head's epilogue ceiling at **75.5 ms,
the pre-correction figure**. Its bf16 arm ran `torch.mm(Gt,A).float()` — a bf16
write plus an fp32 cast — while
**both** fp8 arms passed `out_dtype=torch.float32` and never paid it. About
12.9 GB/step, ≈10.4 ms at the probe's own fitted bandwidth. The baseline was
penalised by work the candidate does not do, so the gap between them was not the
thing under test. Corrected ceiling: **60.2 ms**.

The gap was visible before the conclusion was drawn. The bf16 arm read 205.3 ms
against a **traced** production head of 190.0 ms (`eff.lm_head_is_compute_bound`,
62.5+63.0+64.5 by correlation id) — 8.1% rich, in a direction that flattered the
candidate. Two checks would have caught it:

1. **Do both arms do the same work outside the thing under test?** Every
   difference other than the treatment is a confound, including an output dtype.
2. **Does the baseline reproduce a traced production number?** If the control
   does not match the system it claims to represent, the contrast is against a
   fiction. This is the same discipline as t59's bf16 arm reproducing the known
   137.0 TFLOPS to 0.2% — that agreement is what made t59's fp8 number
   believable, and its absence is what should have stopped t58's.

Note what did *not* go wrong: the timing was tight (spread 0.003–0.006 ms/chunk),
the statistic was pre-declared, the arms were interleaved. **A well-run
measurement of the wrong contrast is still the wrong answer**, and none of the
usual rigour markers detect it.

### A fifth tell: assert presence before asserting properties

The one that hides inside a check written to catch exactly this class.

db's 22B pre-registration returned **zero refused roles and zero ABSENT
metrics** — a clean sheet. The readout had printed nothing at all. Every
assertion was of the form "no row violates the gate", and a file with no rows
satisfies all of them vacuously. The monitor read the absence of bad lines as
good news.

**A property asserted over an empty set is true.** So a probe must first assert
that the artifact *has* the section it is judging, and only then judge it. The
order is not stylistic: presence is a precondition of every property claim after
it, and a probe that skips it reports the cleanest possible result on the most
broken possible input.

The same shape, elsewhere on this page and in the repo, which is why it is a tell
and not an anecdote:

- The `holdout_hashes` empty-set incident — an empty holdout set let 19 of 20
  questions into SFT, because "no question is in the holdout set" is trivially
  true of an empty set.
- de's monitor closing a row `fail` on log silence — the inverse reading of the
  same ambiguity. **Silence is not evidence.** It is equally consistent with
  nothing-bad-happened and with nothing-happened, and only a presence assertion
  separates them.

The enforceable version: **a check reports `SKIP`, never `PASS`, when the thing
it examines is absent.** A pass claims evidence; the absence of input is not
evidence.

### A sixth tell: the measurement is clean and the configuration is wrong

Recorded in full at **"the flag can be in the evidence rather than the claim"**
above, by the author who found it. Kept here only for the ordering, and for the
one framing this section adds: the code-path check at the top of this page asks
whether the *lever* runs in production; this tell asks the same question of the
*trace*, and it is harder to see because nothing in the analysis is wrong.

> **Correctness and relevance fail independently, and no amount of the first
> detects a failure of the second.**

That sentence now applies twice on this page — to the byte cache's dead code path
and to the off-config trace — and the two differ in *where the flag lives*: t60's
was in the claim, t61's was in the evidence, and that fact's text never mentions
`FP8_HEAD` at all. **A check on claims catches one of the two.** That asymmetry
is the whole argument for a provenance field on the measurement rather than a
flag field on the claim.

**And the instance is not that a lesson fired — it is that an outside question
made it fire.** The join that exposed the off-config capture was only run because
a reader asked an unrelated question about double counting. Its author had read
that trace four times and the capture config was invisible every time, because
the question being asked was about attribution, not provenance.

**But state the mechanism carefully, because the obvious statement of it is
wrong.** "An outside reader supplies the question you are not asking" implies the
reader supplies the *right* question. That is not what happened: the question
asked was about double counting, it closed cleanly as a null, and the config
finding fell out sideways. Neither party had provenance in mind.

So the mechanism is **perturbation, and the yield is stochastic** — and the
practical consequences differ:

| "get a reviewer" | "perturb the frame" |
|---|---|
| pick a good one, brief them well | *any* question from outside the frame has expected value |
| the failure mode is a bad reviewer | the failure mode is **no perturbation** — which is what a review from inside the same frame produces, however skilled |
| the valuable question is the important one | the question the author would have rated *lower* value is the one that paid |

The author of the trace and their usual reviewer had been checking each other's
arithmetic all day and neither questioned a capture config, because both were
inside the attribution frame. The value is in the **outsideness**, not in the
reviewer's skill — which is a weaker claim than "route work past a good
reviewer", and a much cheaper one to act on.

### Seven tells, and who caught them

The section headings number the last three "fourth", "fifth" and "sixth" because
they were found in that order; counted as distinct questions to ask a
measurement, there are seven:

1. Is this code path reached in the live configuration?
2. Read the format string — what is the resolution of the input?
3. Is a 0% or 100% result a finding, or a broken join?
4. Does the cited number still describe what currently ships?
5. Does the baseline pay work the candidate does not pay?
6. Does the artifact contain the section being judged at all?
7. What configuration was the capture taken under?

Five of the seven were caught by the author of the measurement, before anyone
else challenged it — the byte cache's dead code path, t58's asymmetric baseline,
the broken join, the config drift, and the off-config trace. That is worth
stating plainly, because it is the behaviour this page is trying to produce.
**The purpose is not to review each other harder; it is to make the author's own
second look find the thing first.** A reviewer who catches a wrong number saves
one decision. An author who catches their own saves the decision and the review
cycle, and does it while the context needed to see the defect is still loaded.

One of the seven was found while answering an unrelated question. #7 surfaced on
the way to settling whether the fusion and elementwise groups double-counted, and
the answer to the question actually asked was "no, they are disjoint — zero
shared kernels." **A question that closes cleanly is not a wasted question**: it
is often how the adjacent defect gets found, and the ranking it was aimed at is
firmer for having survived it.

### The base rate is higher than these narratives imply

Every entry above reads as a notable event, and that framing is itself
misleading. `scripts/doc_numbers_check.py` re-derives a document's stated numbers
from its own declared base and runs in under a second. Its author ran it on the
entry they had just written and found a rounded share sitting beside the table it
summarised — the eighth instance. Within the hour, a **sweep of four unrelated
docs** turned up a ninth: `gpu_colocation.md` rounding a 115 ms saving to one decimal
place where its own span gives 6.86%, in a document that had been read and
re-read.

Two things follow, and they are more useful than any single instance:

1. **This defect is closer to background than to event.** Two found in one hour
   by a one-second tool, in documents written by people actively writing about
   this defect class. Anything narrated as "the time we caught X" is sampling on
   the catch, not on the occurrence.
2. **Sweep every document, not the one you are editing.** The targeted run found
   the eighth only because its author happened to have just touched that file.
   That is luck wearing the clothes of a habit. The sweep is the habit.

A caveat on the sweep, from running it across all 51 docs in `docs/`: three of
the five hits were **false positives** — the checker dropped a leading minus
sign, matched the tail of an inclusion-exclusion expression, and read a design
label (`MDE at 4+4`) as arithmetic. That does not weaken the case for sweeping.
A one-second check with a 40% false-positive rate is still worth running over 51
documents; it means the output is a **list of lines to look at**, not a list of
errors, and it should be described that way so nobody treats a clean sweep as
proof or a hit as a verdict.


## When a qualifier has to be a restriction

A number can be correct and still be misused, and the qualifier you attach
decides which. **A caveat degrades gracefully under skimming; a restriction does
not.** So there is a test:

> If a reader skims the qualifier and uses the number anyway, are they wrong?
> If yes, it is a restriction — it belongs in the usage clause, not the
> uncertainty field.

Two cases from 2026-09-01, one each side:

- **`eff.fp8_weight_byte_cache`, 39.4 ms.** Measured on one card, synthetic
  tensors of the production shape. It licenses *build this lever*. It does not
  license *this lever delivers 39.4 ms in the live run*, and it does not license
  summing with a lever measured on another instrument. Skim the hedge and you
  promise 39.4 ms to someone. Restriction.
- **b0's nat-per-own-token.** Dividing Δloss by a domain's own tokens assumes the
  domain's loss is driven only by its own tokens; transfer violates that, and the
  bias differs by role so it does not even preserve ranking. Skim the caveat and
  you rank domains by nat/B. Restriction — and the doc now restricts the metric's
  *use* (one role across time, never one role against another) rather than
  flagging uncertainty.

**The second case is the worse kind, and the difference is worth naming.** Mine
was checkable — I measured my way out of it. b0's needs seven single-domain
ablations that the run does not contain, so it is **unfalsifiable within the run
that uses it**. A wrong denominator you can measure your way out of is a bug; one
you cannot is a scope limit.

### The corollary for sums

Two measured numbers from different instruments do not add unless you have shown
they do not overlap. The doc summing this lever's 39.4 ms with the seam's 54.9 ms
now reads **"two measured levers, assumed independent, not jointly measured"** —
because HBM traffic inside FLCE and compile stalls at the `rms_norm`→flash seam
*look* independent, and "looks independent" is precisely the reasoning that
produced the aggregate-as-bound error above.

## The fix, in priority order

1. **One wall-clock timestamp per logged step.** Retires #1, #2 and #3 together:
   every throughput statistic becomes directly measurable instead of
   reverse-engineered from a rounded ETA. Higher priority than widening the tok/s
   format, which fixes only #1.
2. Verify any join against a known-answer case before trusting its result,
   especially a negative one.
3. Facts citing a code default record the file:line and the value, so a config
   drifting out from under a fact is detectable.

## The check, and its limit

Proposed: **a fact or ledger row carrying a variance, spread, or SD must also
carry the resolution of its source, and the check FAILs when the reported spread
is below that resolution.**

This catches #1–#3 mechanically, and it is worth having. Two honest limits:

- **It cannot see #4.** A join returning 0% reports no spread at all, so there is
  nothing for the check to compare. No check sees a broken join except one that
  knows the right answer for some case — which is the known-answer verification in
  fix 2, a test, not a check.
- **The resolution field has to be supplied by the person writing the fact**, and
  that is the same person who did not think about resolution while computing the
  spread. The check makes the omission *visible* rather than preventing it: a fact
  with no resolution field FAILs loudly instead of a fact with a wrong spread
  passing quietly. That is a real improvement and it is not the same as catching
  the error.

So: the check is worth building for the printf class, and it should be filed
alongside an explicit "no check sees this" entry for the broken-join class.

## The closing finding: remembering-rules against refusing-rules

Every rule broken on 2026-09-01 was of the first kind. Every durable thing this
repo has is of the second.

**A rule that must be recalled at the point of use has already failed. Only a
rule that refuses at the point of use holds.**

fb's framing, and the day's evidence is unambiguous. Three failures inside one
hour, all mine, all after the rule was not just known but freshly written:

1. **The eager row.** This document's own boundary section says the eager
   measurements describe a path we do not run and must not be ruled from. I
   quoted the eager batch-4 row against a compiled row **in the message after
   writing that**, and it inverted a conclusion: the matched compiled pair says
   L=32 reaches 43.5 TFLOPS/GPU against L=12's 23.7, so the deeper model uses
   the hardware 1.84x better, where I had reported it as slightly worse.
2. **The denominator.** `train.py:2267` divides by 296 TFLOPS under `--fp8` and
   148 without. `probes/t66` never enables fp8 and hardcoded 148. Both outputs
   were labelled "MFU" and I set them side by side. `verify-capture-config` was
   supposed to cover this and did not, because nobody classes a FLOP ceiling as
   *configuration* — it reads as a property of the hardware, which is exactly
   why it travels invisibly.
3. **The A/B with one arm.** Both invocations passed `--ar-blocks 0`, and 0 is
   the default. The two arms were identical. It would have reported a difference
   of zero, and I would have read that as "Full AttnRes is free" — a correct
   computation over an experiment that never varied its variable.

At the moment of the error, none of the three *felt* like the situation the rule
was about. That is not a lapse to be corrected by resolving harder; it is the
normal condition. Two people broke two different rules of this kind on the same
day — b0's §6 nat/B ban, broken twice inside the document that states it, and
mine above. A rule whose enforcement mechanism is human attention has an error
rate, and that rate does not go down when the rule is written more firmly.

**What separates the two kinds:** a refusing-rule is evaluated by something that
is not the person who might forget it, at a moment when forgetting is still
recoverable. `UNTRUSTED_SUPPLY`, the vocab_id refusal, the fingerprint stamps,
the dynamo assert, the `.REFUSED` sidecar — none of these ask anyone to
remember anything.

The three fixes from this hour are all of the second kind:

- Every t66 row carries `peak_tflops` and `fp8`, so a mismatched comparison is
  visible **in the JSON** rather than in someone's memory. A percentage hides its
  denominator by construction, which is the argument against percentages as the
  unit anything is compared in.
- `_ab_guard` refuses to run an arm whose args hash matches one already recorded.
  Its `--selftest` asserts the guard *fires*, and the file is registered in the
  hook's `SELFTEST_FILES` — because a selftest nothing runs is itself a
  remembering-rule wearing the costume of a refusing one.
- `no_foreground_pod_training` no longer reads a zombie as a live trainer.

**The open one, named rather than solved:** the third-point rule ("before
extrapolating from two points, check whether a third is already on disk") has no
mechanical trigger and I am not going to pretend writing it down again is one. A
refusing version would need to know that a series was being extrapolated, which
nothing in the current tooling sees. Filed as an open item with that framing,
because "add a check" and "write it down more firmly" are different responses
and only one of them has ever worked here.

**The asymmetry that makes #3 the worst of the three:** a wrong number invites
argument, and a zero does not. An A/B that reports no difference, a join that
resolves 0%, a probe that finds no duplicates — each is a clean result that
nobody interrogates. This is why `probes/t62`–`t65` each ship a `--selftest`
asserting they FIND planted positives: a null from an instrument that has never
been shown capable of a non-null is not evidence of absence. It is no evidence
at all.
