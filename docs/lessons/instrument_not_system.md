---
question: "how do you notice that a measurement describes the measuring tool rather than the system"
status: recorded
source: "tilerl-10 2026-09-01; four instances found in one hour: runs/t56_elementwise_owner.json (the broken join), the pretrain_30b_s2 log's tok/s and ETA format strings, eff.kda_occupancy_bound (config drift). Check proposal reviewed by fb; de-8 carries the seven-sighting class ticket."
---

# The instrument, not the system

An instrument returning a confident number that describes the instrument rather
than the system. Four instances on 2026-09-01, all found within about an hour of
each other, all in tools we trusted. The tell is the same every time: **a number
too clean for the thing it claims to measure.**

## The rule

**Before using a logged value as a statistic, read its format string.** A derived
statistic's resolution is bounded by the printf that produced its input. Three of
the four instances below are one `%.1f` away from being real measurements.

The corollary, for attribution and joins: **treat any result of exactly 0% or
exactly 100% as a suspected tooling failure until the join is verified on a case
whose answer is already known.**

## The four

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
merged the "copies" rung into the fp8-head rung instead of refuting it.

The cost asymmetry is the point. A false positive costs a review cycle. A false
negative closes a line of work permanently and silently.

## What each cost

None of these were caught by the statistic itself — only by looking at the
instrument. #1–#3 each produced a proposed kill-criteria threshold; the third
would have set a false-precision floor at median + 4 SD = 104.5 s, inside the
quantisation band, firing on rounding. #4 nearly retired a live 4.44% lever.

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
