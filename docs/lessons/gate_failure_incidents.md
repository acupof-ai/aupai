---
question: What are the surviving gate-failure incidents, what would close each, and which are already closed?
status: open
source: derived from the 2026-09-04 restructure of gate_failure_shapes.md; 33 closed incidents deleted (see commit), 141 survive here
---

# Gate failure incidents

One entry per surviving §. Grouped by rule. Each entry: id, date, rule/sub-rule, one-line incident, evidence, `open:` what would close it. Closed incidents (33) were deleted in the 2026-09-04 restructure; their closing mechanisms are named in the commit message.

## R1. Verify premises before acting, sources before citing

### §8 (2026-08-30, R1)
A premise was accepted without checking its source; the conclusion was correct but the argument cited a fact that said something else. Evidence: facts/seed_variance.json.
open: a check that flags a citation whose fact value contradicts the claim built on it; none exists.

### §14 (2026-08-31, R1)
A correct conclusion was used to certify an untested argument; the conclusion's truth did not extend to the argument's premises. Evidence: docs/lessons/experiments_0904.md.
open: manual — no check can verify that a true conclusion supports the argument it is attached to.

### §18 (2026-08-31, R1)
A source was cited for a claim it did not make; the citation resolved but the fact's content was adjacent, not supporting. Evidence: facts/data_scaling.json.
open: a check that reads the cited fact's value and compares it to the claim; none exists.

### §37 (2026-09-01, R1)
A premise about the data pipeline was accepted without reading the pipeline code; the premise was wrong and the conclusion inherited the error. Evidence: scripts/build_mix.py.
open: manual — premise verification is a human discipline.

### §38 (2026-09-01, R1)
A number was quoted from memory rather than from the fact store; the remembered number was wrong. Evidence: facts/efficiency.json.
open: a check that flags prose numbers with no fact citation; doc_numbers_check partially covers docs, not chat.

### §46 (2026-09-01, R1)
A self-reported timestamp in a message did not carry its clock source; "UTC hh:mm" was local time, off by 8 hours. Evidence: peer message log.
open: check_timestamps_are_utc covers code-generated timestamps, not self-reported message ones; a message-lint would close it.

### §49 (2026-09-01, R1)
A source was read at the wrong granularity; the fact's population was narrower than the claim's. Evidence: facts/corpus_supply.json.
open: a check that compares a claim's population to the cited fact's population; none exists.

### §52 (2026-09-02, R1)
A conclusion was correct but its argument cited a retracted fact; the retraction had not reached the citation. Evidence: facts/data_scaling.json (retracted entry).
open: a check that flags citations to retracted facts; none exists.

### §57 (2026-09-02, R1)
A premise about GPU availability was accepted without reading card_assignment.json; the card was claimed by another run. Evidence: runs/card_assignment.json.
open: check_card_held_without_claim covers the claim, not the premise that a card is free.

### §66 (2026-09-02, R1)
Saw literal `0` in `blocks=0`, concluded "not the config"; `0 or n_sub` made 0 the sentinel for Full. Evidence: probes/t71_depth_lr_rule.py:130, train.py:219, model.py:330.
open: a check that flags falsy literals used as sentinels in config consumers; none exists.

### §70 (2026-09-02, R1)
A source was cited for a direction it did not support; the fact's sign was opposite to the claim's. Evidence: facts/data_scaling.json.
open: a check that reads the cited fact's sign and compares it to the claim; none exists.

### §96 (2026-09-03, R1)
An adjacent field was inferred as the divisor; the field name suggested the role but the code used a different field. Evidence: scripts/harness.py.
open: a check that verifies field-level role agreement between name and use; none exists.

### §106 (2026-09-03, R1)
A premise about checkpoint contents was accepted without reading the checkpoint; the checkpoint held different weights. Evidence: runs/ckpt_*.pt.
open: manual — checkpoint inspection is a human discipline.

### §131 (2026-09-03, R1)
`tail` read a dead process's `SRCFP CHANGED` line as the current result; the line was a snapshot, not a live signal. Evidence: runs/*.log.
open: a check that verifies a log line's process is still alive before quoting it; none exists.

### §139 (2026-09-03, R1)
A source was cited from a draft that had been superseded; the published version said something different. Evidence: docs/standards/state_0904.md.
open: a check that flags citations to superseded drafts; none exists.

## R2. A criterion must express the property asked

### R2-a No broken world

### §31 (2026-08-31, R2-a)
A verdict was not asserted; the check printed the verdict but did not fail on it. Evidence: scripts/harness.py.
open: a check that every CHECKS entry asserts its verdict (raise/exit, not print); the --selftest contract partially covers this.

### §69 (2026-09-02, R2-a)
A selftest's needle was in its own line; the broken world matched the check's own source, not a real failure. Evidence: scripts/harness.py.
open: a check that selftest needles are not self-referential; none exists.

### §89 (2026-09-02, R2-a)
A selftest "passed" because the world-build step silently failed and the check ran on an empty population. Evidence: scripts/harness.py.
open: a check that asserts the world-build succeeded before running the check; none exists.

### §103 (2026-09-03, R2-a)
A check that cannot fail — its acceptance condition was tautological. Evidence: scripts/harness.py.
open: a check that every CHECKS entry has a negative world that fails; --selftest partially covers this.

### §137 (2026-09-03, R2-a)
A meta-check counted 0 on an empty population; the count was read as "0 violations" rather than "0 items checked." Evidence: scripts/harness.py.
open: a check that meta-checks assert a non-empty population; none exists.

### §153 (2026-09-03, R2-a)
A fixture re-implemented the comparison it was testing; the fixture could only fail if the implementation and the fixture diverged, not if the property was wrong. Evidence: scripts/test_*.py.
open: a check that fixtures are independent of the implementation under test; none exists.

### R2-b Population narrower than the property

### §26 (2026-08-31, R2-b)
A known-answer test lacked the real data's shape; the fixture was clean and the real data was not. Evidence: scripts/test_*.py.
open: a check that fixtures match the real data's shape; none exists.

### §29 (2026-08-31, R2-b)
A criterion could not express the property asked; the criterion's type was wrong for the property. Evidence: scripts/harness.py.
open: manual — criterion design is a human discipline.

### §34 (2026-09-01, R2-b)
A range check covered only one end; the property was two-sided. Evidence: scripts/harness.py.
open: a check that range checks are two-sided where the property is; none exists.

### §35 (2026-09-01, R2-b)
A dynamic acceptance check missed code that stayed; the check only looked at changed lines. Evidence: scripts/harness.py.
open: a check that acceptance checks cover unchanged code where the property applies; none exists.

### §40 (2026-09-01, R2-b)
A test runner could not run `.sh` tests; the shell tests were silently skipped. Evidence: scripts/harness.py.
open: a check that the runner can execute every registered test type; none exists.

### §48 (2026-09-01, R2-b)
A one-sample criterion was used where the property needed a distribution; the single sample could not estimate variance. Evidence: docs/lessons/experiments_0904.md.
open: manual — sample-size design is a human discipline.

### §65 (2026-09-02, R2-b)
A check only rejected one side; the property was symmetric. Evidence: scripts/harness.py.
open: a check that symmetric properties have symmetric checks; none exists.

### §72 (2026-09-02, R2-b)
A conditional stub replaced the real check in one environment; the stub always passed. Evidence: scripts/harness.py.
open: a check that stubs are not registered as the real check; none exists.

### §121 (2026-09-03, R2-b)
A validator used a different definition from the producer; the validator's population was narrower. Evidence: scripts/harness.py.
open: a check that validator and producer share a definition; none exists.

### §134 (2026-09-03, R2-b)
A unit test measured format compliance, not content correctness; the property asked was content. Evidence: scripts/test_*.py.
open: a check that tests assert content, not just format; none exists.

### §146 (2026-09-03, R2-b)
An acceptance check pointed far from the limit; the property was at the limit. Evidence: scripts/harness.py.
open: a check that acceptance checks target the property's boundary; none exists.

### §151 (2026-09-03, R2-b)
`torch.equal` was too fine a criterion; the property allowed tolerance but the check required bitwise equality. Evidence: scripts/test_*.py.
open: a check that equality criteria match the property's tolerance; none exists.

### §169 (2026-09-04, R2-b)
A perturbation was isolated away from the path under test; the perturbation did not reach the code the check exercised. Evidence: scripts/test_*.py.
open: a check that perturbations reach the target path; none exists.

### §171 (2026-09-04, R2-b)
A perturbation was injected at a scale below the instrument's resolution; the property asked (sensitivity) was outside the test's population. Evidence: scripts/test_*.py.
open: a check that perturbation scale exceeds instrument resolution; none exists.

### R2-c Mutation did not take

### §81 (2026-09-02, R2-c)
A mutation check grepped for a keyword instead of reading the exit code; the keyword appeared in a comment. Evidence: scripts/harness.py.
open: a check that mutation verification reads exit codes, not text; none exists.

### §90 (2026-09-02, R2-c)
A mutation was applied but never landed in the running process; the check "passed" because the world was never mutated. Evidence: scripts/harness.py.
open: a check that the mutation reached the live process; none exists.

### §132 (2026-09-03, R2-c)
The mutation test itself was broken — it mutated a copy, not the live object. Evidence: scripts/test_*.py.
open: a check that mutation tests mutate the live object; none exists.

### R2-d Parser reads prose as code

### §56 (2026-09-02, R2-d)
A scanner split the string it was matching; the match read a fragment as the whole. Evidence: scripts/harness.py.
open: a check that scanners match structural boundaries, not substrings; none exists.

### §61 (2026-09-02, R2-d)
A substring/word match read a comment mentioning the symbol as evidence the symbol was used. Evidence: scripts/harness.py.
open: a check that text matches exclude comments and strings; none exists.

### §77 (2026-09-02, R2-d)
A needle was found in the check's own comment; the match was self-referential. Evidence: scripts/harness.py.
open: a check that needles are not self-referential; none exists.

### §94 (2026-09-03, R2-d)
A symbol's name was present in the file, read as "assigned"; the name appeared in a string, not an assignment. Evidence: scripts/harness.py.
open: a check that name-presence is not read as assignment; none exists.

### §141 (2026-09-03, R2-d)
A symbol looked like what it named; the name suggested the role but the code did something different. Evidence: scripts/harness.py.
open: a check that symbol names match their behavior; none exists.

### R2-e Fixture built from the implementation

### §76 (2026-09-02, R2-e)
A fixture was built from the implementation's handled branches; unhandled branches — the ones that fail in production — were absent. Evidence: scripts/test_*.py.
open: a check that fixtures include unhandled branches; none exists.

### §80 (2026-09-02, R2-e)
A fixture was not trained; the fixture's weights were random, not the product of training. Evidence: scripts/test_*.py.
open: a check that fixtures are trained, not random; none exists.

### §97 (2026-09-03, R2-e)
A selftest fed the middle function, not the entry point; the middle function could not fail the way the entry point did. Evidence: scripts/test_*.py.
open: a check that selftests exercise the entry point; none exists.

### §98 (2026-09-03, R2-e)
A fixture had the same form as the formula under test; it could not detect a form error, only a value error. Evidence: scripts/test_*.py.
open: a check that fixtures are independent of the formula under test; none exists.

### R2-f Guard reads the wrong field

### §54 (2026-09-01, R2-f)
A guard's condition was wider than the danger; the guard blocked safe cases and missed the dangerous one. Evidence: scripts/harness.py.
open: a check that guard conditions match the danger surface; none exists.

### §71 (2026-09-02, R2-f)
The guard condition and the assertion body read different keys; the guard blocked on one key while the assertion checked another. Evidence: scripts/harness.py.
open: a check that guard and assertion read the same key; none exists.

### §75 (2026-09-02, R2-f)
A missing init produced the wrong error; the guard read the wrong field and reported a misleading cause. Evidence: scripts/harness.py.
open: a check that init failures are caught before field reads; none exists.

### §85 (2026-09-02, R2-f)
A guard read the wrong key and false-triggered; the key it read was always set, so the guard always fired. Evidence: scripts/harness.py.
open: a check that guard keys are the ones the writer sets; none exists.

### §125 (2026-09-03, R2-f)
A check read a pid file that was never written; the empty read was interpreted as "no process," not "no data." Evidence: scripts/harness.py.
open: a check that distinguishes "no data" from "no process"; none exists.

### §128 (2026-09-03, R2-f)
A check read the wrong key and produced a fake zero; the key it read was always 0. Evidence: scripts/harness.py.
open: a check that check keys are the ones the writer populates; none exists.

### R2-g Criterion answers an adjacent question

### §9 (2026-08-30, R2-g)
A requirement constrained the literal, not the property; the literal was met and the property was not. Evidence: docs/standards/state_0904.md.
open: manual — requirement design is a human discipline.

### §10 (2026-08-30, R2-g)
An unverified pattern was counted; a zero count was not rechecked, and the zero was the population, not the result. Evidence: scripts/harness.py.
open: a check that zero counts are rechecked with a different method; none exists.

### §23 (2026-08-31, R2-g)
A clamp was read as an assignment; the clamped value was the default, not the configured one. Evidence: train.py.
open: a check that clamps are distinguished from assignments; none exists.

### §45 (2026-09-01, R2-g)
A check read stdout text instead of the exit code; the text said "PASS" but the exit code was nonzero. Evidence: scripts/harness.py.
open: a check that checks read exit codes, not stdout; none exists.

### §67 (2026-09-02, R2-g)
A check asserted the value, not the mechanism; the value was right but the mechanism that produced it was wrong. Evidence: scripts/test_*.py.
open: a check that mechanism is asserted, not just value; none exists.

### §73 (2026-09-02, R2-g)
A comment asserted a guarantee no check provided; the comment was read as evidence the guarantee held. Evidence: scripts/harness.py.
open: a check that comments asserting guarantees are backed by checks; none exists.

### §84 (2026-09-02, R2-g)
A wrong metric changed the decision; the metric measured a neighbour property and the decision followed it. Evidence: docs/lessons/experiments_0904.md.
open: manual — metric selection is a human discipline.

### §91 (2026-09-02, R2-g)
A test failure was diagnosed by suspecting the object, not the chain; the chain (fixture, runner, environment) was the failure. Evidence: scripts/test_*.py.
open: manual — diagnosis order is a human discipline.

### §108 (2026-09-03, R2-g)
An off-by-one did not crash; the check passed but the result was off by one. Evidence: scripts/harness.py.
open: a check that boundary values are tested, not just interior; none exists.

### §110 (2026-09-03, R2-g)
A pre-registered branch collapsed two worlds into one; the criterion (branch taken) did not isolate the property (which world). Evidence: runs/prereg.jsonl.
open: a check that pre-registered branches isolate the worlds they name; none exists.

### §112 (2026-09-03, R2-g)
An unchanged arm was not reproduced; the comparison assumed the unchanged arm was the same as last time. Evidence: docs/lessons/experiments_0904.md.
open: a check that unchanged arms are re-run, not assumed; none exists.

### §114 (2026-09-03, R2-g)
A null hypothesis was not isolated; the test rejected a null that was not the property's. Evidence: docs/lessons/experiments_0904.md.
open: manual — null-hypothesis design is a human discipline.

### §135 (2026-09-03, R2-g)
A one-sided fold was used where the property was two-sided; the fold hid the wrong direction. Evidence: scripts/harness.py.
open: a check that fold direction matches the property's sidedness; none exists.

### §140 (2026-09-03, R2-g)
A metric had no "not measured" value; absence was read as zero. Evidence: scripts/harness.py.
open: a check that metrics distinguish "not measured" from zero; none exists.

### §142 (2026-09-03, R2-g)
An alarm fired on the harmless side; the dangerous side was silent. Evidence: scripts/harness.py.
open: a check that alarm conditions target the dangerous side; none exists.

### §147 (2026-09-03, R2-g)
A quantity was measured on the wrong axis; the axis measured was not the property's. Evidence: docs/lessons/experiments_0904.md.
open: manual — axis selection is a human discipline.

### §148 (2026-09-03, R2-g)
A statistic's null-hypothesis value was not the one asked; the statistic answered a different null. Evidence: docs/lessons/experiments_0904.md.
open: manual — statistic selection is a human discipline.

### §149 (2026-09-03, R2-g)
Pre-registration was defined by information order, not by the property; the branch was chosen after seeing the data. Evidence: runs/prereg.jsonl.
open: a check that pre-registration precedes data observation; none exists.

### §150 (2026-09-03, R2-g)
Three paths were all wrong; the criterion chose among them instead of rejecting all. Evidence: scripts/harness.py.
open: a check that the criterion can reject all options, not just choose; none exists.

### §158 (2026-09-03, R2-g)
A parse was read as complete; the parse succeeded but the content was incomplete. Evidence: scripts/harness.py.
open: a check that parse success is not read as content completeness; none exists.

### §165 (2026-09-04, R2-g)
An alive process was read as dead; the liveness check used the wrong signal. Evidence: scripts/harness.py.
open: a check that liveness reads STAT=Z, not kill -0; none exists.

### §170 (2026-09-04, R2-g)
An unresolvable fact reference was used for four days; the criterion (reference present) did not measure the property (reference resolves). Evidence: docs/lessons/gate_failure_shapes.md (old §170).
open: check_fact_refs scans only docs/lessons+audits; widening to data/*.json and bare id forms would close it.

### §173 (2026-09-04, R2-g)
"NOT KEPT" was read as "CLAIMED"; the absence of a KEEP claim was read as a claim. Evidence: runs/pod_ckpt_candidates_*.txt.
open: a check that distinguishes "not kept" from "claimed"; none exists.

## R3. Artifacts carry their producer's identity

### §4 (2026-08-30, R3)
An artifact with no producer identity was silently rebuilt; the rebuild used a different producer, and the artifact's meaning changed. Evidence: scripts/harness.py.
open: check_cache_readers_set_vocab_id covers vocab identity; a general producer-identity check would close it.

### §24 (2026-08-31, R3)
A checkpoint with no recipe provenance was scored; the score was attributed to a recipe the checkpoint did not run. Evidence: runs/ckpt_*.pt.
open: a check that checkpoints carry recipe provenance before scoring; none exists.

### §44 (2026-09-01, R3)
A checkpoint carried an identity that was not the one it ran with; the identity was copied from a template. Evidence: runs/ckpt_*.pt.
open: a check that checkpoint identity is set at production, not copied; none exists.

## R4. Failures must be loud

### §7 (2026-08-30, R4)
A check printed a warning and continued; the warning was lost in the log and the bad data was used. Evidence: scripts/harness.py.
open: a check that failure paths raise, not print; none exists.

### §13 (2026-08-30, R4)
A world-build step silently failed; the check ran on an empty population and passed. Evidence: scripts/harness.py.
open: a check that world-build failures are loud; none exists.

### §25 (2026-08-31, R4)
A check before the write was missing; the bad data was written and the check ran after. Evidence: scripts/harness.py.
open: a check that checks run before the write; none exists.

### §51 (2026-09-01, R4)
An observation channel swallowed the signal; the check read the channel's default, not the observation. Evidence: scripts/harness.py.
open: a check that observation channels propagate errors; none exists.

### §59 (2026-09-02, R4)
An env-gated assertion was skipped in production; the env var was unset and the assertion never ran. Evidence: scripts/harness.py.
open: a check that env-gated assertions have a non-gated default; none exists.

### §136 (2026-09-03, R4)
A failure was printed to stdout and the exit code was zero; the caller read the exit code, not stdout. Evidence: scripts/harness.py.
open: a check that failures set a nonzero exit code; none exists.

### §166 (2026-09-04, R4)
A print-and-continue path swallowed an exception; the exception was logged and the run continued with bad state. Evidence: scripts/harness.py.
open: a check that exceptions in the run path raise, not print; none exists.

## R5. State the vision before the number

### §3 (2026-08-30, R5)
A number was reported without its population; the population (which items, which scale, which seed) determined the number's meaning. Evidence: docs/lessons/experiments_0904.md.
open: manual — population statement is a human discipline.

### §5 (2026-08-30, R5)
A number outside the stated vision was reported without a label; the reader assumed it was inside. Evidence: docs/standards/state_0904.md.
open: manual — vision labeling is a human discipline.

### §6 (2026-08-30, R5)
A measurement was reported as "absent" when it was "unmeasured"; the silence was read as a zero. Evidence: docs/standards/state_0904.md.
open: a check that "unmeasured" is a distinct value from "absent"; none exists.

### §17 (2026-08-31, R5)
A number's vision was stated after the number; the reader could not tell what population the number covered. Evidence: docs/lessons/experiments_0904.md.
open: manual — vision-before-number is a human discipline.

### §19 (2026-08-31, R5)
A sub-population was reported as the whole; the sub-population was the only one measured. Evidence: docs/lessons/experiments_0904.md.
open: manual — population labeling is a human discipline.

### §28 (2026-08-31, R5)
A number was reported with the wrong vision; the vision stated was broader than the measurement. Evidence: docs/standards/state_0904.md.
open: manual — vision accuracy is a human discipline.

### §30 (2026-09-01, R5)
A measurement was extrapolated beyond the vision; the extrapolation was not labeled. Evidence: docs/lessons/experiments_0904.md.
open: a check that extrapolations are labeled; none exists.

### §32 (2026-09-01, R5)
A number's resolution was finer than its vision; the extra digits were noise. Evidence: docs/standards/state_0904.md.
open: manual — resolution matching is a human discipline.

### §36 (2026-09-01, R5)
A vision was stated but not enforced; numbers outside it were reported without a label. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers outside the vision are labeled; none exists.

### §53 (2026-09-02, R5)
A number was reported with no vision at all; the reader could not tell what it measured. Evidence: docs/lessons/experiments_0904.md.
open: manual — vision statement is a human discipline.

### §100 (2026-09-03, R5)
A measurement outside the stated vision was reported as "absent"; the correct label was "unmeasured." Evidence: docs/standards/state_0904.md.
open: a check that "absent" and "unmeasured" are distinct; none exists.

## R6. Every number carries its basis

### §1 (2026-08-30, R6)
A number was quoted without its source type; the source type (measured / extrapolated / inferred) determined comparability. Evidence: docs/standards/state_0904.md.
open: a check that numbers carry a source-type label; none exists.

### §11 (2026-08-30, R6)
A number's resolution was not stated; the reader could not tell the precision. Evidence: docs/lessons/experiments_0904.md.
open: manual — resolution statement is a human discipline.

### §12 (2026-08-30, R6)
A number's algorithm was not named; two numbers with the same name used different algorithms. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their algorithm; none exists.

### §20 (2026-08-31, R6)
A number was extrapolated without a label; the extrapolation was read as a measurement. Evidence: docs/lessons/experiments_0904.md.
open: a check that extrapolations are labeled; none exists.

### §21 (2026-08-31, R6)
A number's basis was a different instrument from the one named; the instrument named was not the one that produced it. Evidence: docs/lessons/experiments_0904.md.
open: a check that the named instrument produced the number; none exists.

### §50 (2026-09-01, R6)
A number's basis was a draft, not a measurement; the draft was read as a result. Evidence: docs/standards/state_0904.md.
open: a check that draft numbers are labeled; none exists.

### §55 (2026-09-02, R6)
A number's resolution was finer than its basis; the extra digits were noise, not precision. Evidence: docs/lessons/experiments_0904.md.
open: manual — resolution matching is a human discipline.

### §62 (2026-09-02, R6)
A number's basis was a single sample; the basis was not labeled as n=1. Evidence: docs/lessons/experiments_0904.md.
open: a check that n=1 numbers are labeled; none exists.

### §63 (2026-09-02, R6)
A number's basis was a different seed; the seed was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their seed; none exists.

### §64 (2026-09-02, R6)
A number's basis was a different checkpoint; the checkpoint was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their checkpoint; none exists.

### §79 (2026-09-02, R6)
A number's basis was a different path; the path was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their path; none exists.

### §86 (2026-09-02, R6)
A number's basis was a different population; the population was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their population; none exists.

### §99 (2026-09-03, R6)
A number's basis was a different metric; the metric was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their metric; none exists.

### §104 (2026-09-03, R6)
A number's basis was a different scale; the scale was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their scale; none exists.

### §105 (2026-09-03, R6)
A number's basis was a different domain; the domain was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their domain; none exists.

### §109 (2026-09-03, R6)
A number's basis was a different arm; the arm was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their arm; none exists.

### §111 (2026-09-03, R6)
A number's basis was a different step; the step was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their step; none exists.

### §115 (2026-09-03, R6)
A number's basis was a different token budget; the budget was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their token budget; none exists.

### §117 (2026-09-03, R6)
A number's basis was a different eval path; the path was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their eval path; none exists.

### §118 (2026-09-03, R6)
A number's basis was a different scorer; the scorer was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their scorer; none exists.

### §124 (2026-09-03, R6)
A number's basis was a different threshold; the threshold was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their threshold; none exists.

### §127 (2026-09-03, R6)
A number's basis was a different window; the window was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their window; none exists.

### §133 (2026-09-03, R6)
A number's basis was a different fold; the fold was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their fold; none exists.

### §143 (2026-09-03, R6)
A number's basis was a different sign convention; the convention was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their sign convention; none exists.

### §152 (2026-09-03, R6)
A number's basis was a different aggregation; the aggregation was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their aggregation; none exists.

### §155 (2026-09-03, R6)
A number's basis was a different baseline; the baseline was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their baseline; none exists.

### §156 (2026-09-03, R6)
A number's basis was a different normalization; the normalization was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their normalization; none exists.

### §157 (2026-09-03, R6)
A number's basis was a different unit; the unit was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their unit; none exists.

### §159 (2026-09-03, R6)
A number's basis was a different confidence level; the level was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their confidence level; none exists.

### §161 (2026-09-04, R6)
A number's basis was a different comparison; the comparison was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their comparison; none exists.

### §164 (2026-09-04, R6)
A number's basis was a different rounding; the rounding was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their rounding; none exists.

### §172 (2026-09-04, R6)
A number's basis was a different timezone; the timezone was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their timezone; none exists.

## R7. Retractions travel as wide as the ruling

### §16 (2026-08-31, R7)
A retraction was narrower than the ruling; the ruling voided a todo that the retraction did not name, and the todo was later acted on. Evidence: docs/standards/state_0904.md.
open: a check that retractions name every todo the ruling voids; none exists.

### §22 (2026-08-31, R7)
A retraction did not reach every consumer; a consumer of the original ruling never saw the retraction. Evidence: docs/standards/state_0904.md.
open: a check that retractions reach every citation of the original; none exists.

### §58 (2026-09-02, R7)
A constraint was stated in prose, not enforced by a machine check; the constraint was violated without a signal. Evidence: AGENTS.md.
open: a check that prose constraints have a machine check; none exists.

### §68 (2026-09-02, R7)
A retraction was published in a different channel from the ruling; the ruling's audience did not see it. Evidence: peer message log.
open: manual — retraction channel is a human discipline.

### §102 (2026-09-03, R7)
A retraction named the fact but not the decisions that used it; the decisions were not revisited. Evidence: docs/standards/state_0904.md.
open: a check that retractions name the decisions that used the fact; none exists.

### §119 (2026-09-03, R7)
A retraction was recorded as a comment, not as a status change; the fact's status still said "stands." Evidence: facts/*.json.
open: a check that retractions change the fact's status; none exists.

## R8. Shared resources are explicitly exclusive

### §15 (2026-08-31, R8)
A shared resource was used without an explicit claim; the co-residency cost was measured against a metric class, not the run's own spend. Evidence: runs/card_assignment.json.
open: check_card_held_without_claim covers cards; a general resource-claim check would close it.

### §126 (2026-09-03, R8)
A resource's exclusivity was inferred from "0 MiB" in nvidia-smi; idle is not a grant. Evidence: runs/card_assignment.json.
open: a check that exclusivity is read from the claim ledger, not from utilization; none exists.

## R9. Run a deletion candidate before judging it

### §39 (2026-09-01, R9)
A deletion candidate was judged without running it; the candidate was a live process, not a stale file. Evidence: runs/pod_ckpt_candidates_*.txt.
open: ckpt_facts_sources_present covers checkpoints; a general liveness check would close it.

### §41 (2026-09-01, R9)
A deletion list was broadcast and deleted within the 24h window; an unclaimed candidate was still in use. Evidence: runs/pod_ckpt_candidates_*.txt.
open: a check that the 24h window elapsed before deletion; none exists.

## R10. What happened only on the pod did not happen

### §2 (2026-08-30, R10)
A measurement existed only on the pod; the pod was recycled, and the measurement was lost. Evidence: pod-only artifact.
open: manual — pod-to-repo transport is a human discipline.

### §116 (2026-09-03, R10)
A pod-only artifact was cited in a decision; the artifact was unreachable from the repo, and the decision rested on an unreadable source. Evidence: docs/standards/state_0904.md.
open: a check that cited artifacts are reachable from the repo; none exists.
