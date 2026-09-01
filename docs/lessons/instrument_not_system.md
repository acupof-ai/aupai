---
question: "how do you notice that a measurement describes the measuring tool rather than the system"
status: recorded
source: "tilerl-10 2026-09-01; six instances found in one day: runs/t56_elementwise_owner.json (the broken join, and its fix breaking the same way on cuda_driver), the pretrain_30b_s2 log's tok/s and ETA format strings, eff.kda_occupancy_bound (config drift), eff.quant_tax_is_the_elementwise_group (off-config trace). Check proposal reviewed by fb, tightened by 62; de-8 carries the seven-sighting class ticket."
---

# The instrument, not the system

## Before anything else: is this code path reached in the live configuration?

One line, and it precedes every other check on this page. It is last here only
because it was found last.

`probes/t60_weight_cache.py` measured a **39.4 ms/step** saving from caching the
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
quantisation band, firing on rounding. #4 nearly retired a 4.44% lever — and
here is the twist worth keeping, because this sentence was wrong for six hours:
that lever turned out to be **93% head work on a path the live run does not
reach**, so #4's correction rescued a rung that #5's class then took away again.
Both steps were right. A number can be rescued from one defect and still be
carrying another. #6 would have published "the fusion group is disjoint from the
quantisation tax" as a *null* — the right answer, reached by a broken join, which
is the one way to be correct that teaches you nothing and cannot be trusted next
time.

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

`probes/t58_quant_tax.py` reported the fp8 head's epilogue ceiling at **75.5 ms**.
Its bf16 arm ran `torch.mm(Gt,A).float()` — a bf16 write plus an fp32 cast — while
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
