---
question: What are the surviving model-project gate-failure incidents, what would close each, and which are already closed?
status: open
source: derived from the 2026-09-04 restructure of gate_failure_shapes.md; 33 closed incidents removed (33/33 confirmed machine-gated, list in gate_failure_shapes.md), 141 survive here and in infra_incidents.md
---

# Gate failure incidents — model-project layer

One entry per surviving §. Grouped by rule. Each entry: id, date, rule/sub-rule, one-line incident, evidence, `open:` what would close it. Pod/infra incidents live in `infra_incidents.md`. Closed incidents (33) were deleted in the 2026-09-04 restructure; their closing mechanisms are named in the commit message.

## R1. Verify premises before acting, sources before citing

### §8 (2026-08-30, R1)
A premise was accepted without checking its source; the conclusion was correct but the argument cited a fact that said something else. Evidence: facts/seed_variance.json.
open: a check that flags a citation whose fact value contradicts the claim built on it; none exists.

### §37 (2026-09-01, R1)
A premise about the data pipeline was accepted without reading the pipeline code; the premise was wrong and the conclusion inherited the error. Evidence: scripts/build_mix.py.
open: manual — premise verification is a human discipline.

### §49 (2026-09-01, R1)
A source was read at the wrong granularity; the fact's population was narrower than the claim's. Evidence: facts/corpus_supply.json.
open: a check that compares a claim's population to the cited fact's population; none exists.

### §70 (2026-09-02, R1)
A source was cited for a direction it did not support; the fact's sign was opposite to the claim's. Evidence: facts/data_scaling.json.
open: a check that reads the cited fact's sign and compares it to the claim; none exists.

### §96 (2026-09-03, R1)
An adjacent field was inferred as the divisor; the field name suggested the role but the code used a different field. Evidence: scripts/harness.py.
open: a check that verifies field-level role agreement between name and use; none exists.

### §31 (2026-08-31, R2-a)
A verdict was not asserted; the check printed the verdict but did not fail on it. Evidence: scripts/harness.py.
open: a check that every CHECKS entry asserts its verdict (raise/exit, not print); the --selftest contract partially covers this.

### §69 (2026-09-02, R2-a)
A selftest's needle was in its own line; the broken world matched the check's own source, not a real failure. Evidence: scripts/harness.py.
open: a check that selftest needles are not self-referential; none exists.

### §103 (2026-09-03, R2-a)
A check that cannot fail — its acceptance condition was tautological. Evidence: scripts/harness.py.
open: a check that every CHECKS entry has a negative world that fails; --selftest partially covers this.

### §137 (2026-09-03, R2-a)
A meta-check counted 0 on an empty population; the count was read as "0 violations" rather than "0 items checked." Evidence: scripts/harness.py.
open: a check that meta-checks assert a non-empty population; none exists.

### §153 (2026-09-03, R2-a)
A fixture re-implemented the comparison it was testing; the fixture could only fail if the implementation and the fixture diverged, not if the property was wrong. Evidence: scripts/test_*.py.
open: a check that fixtures are independent of the implementation under test; none exists.

### §26 (2026-08-31, R2-b)
A known-answer test lacked the real data's shape; the fixture was clean and the real data was not. Evidence: scripts/test_*.py.
open: a check that fixtures match the real data's shape; none exists.

### §29 (2026-08-31, R2-b)
A criterion could not express the property asked; the criterion's type was wrong for the property. Evidence: scripts/harness.py.
open: manual — criterion design is a human discipline.

### §34 (2026-09-01, R2-b)
A range check covered only one end; the property was two-sided. Evidence: scripts/harness.py.
open: a check that range checks are two-sided where the property is; none exists.

### §40 (2026-09-01, R2-b)
A test runner could not run `.sh` tests; the shell tests were silently skipped. Evidence: scripts/harness.py.
open: a check that the runner can execute every registered test type; none exists.

### §65 (2026-09-02, R2-b)
A check only rejected one side; the property was symmetric. Evidence: scripts/harness.py.
open: a check that symmetric properties have symmetric checks; none exists.

### §81 (2026-09-02, R2-c)
A mutation check grepped for a keyword instead of reading the exit code; the keyword appeared in a comment. Evidence: scripts/harness.py.
open: a check that mutation verification reads exit codes, not text; none exists.

### §90 (2026-09-02, R2-c)
A mutation was applied but never landed in the running process; the check "passed" because the world was never mutated. Evidence: scripts/harness.py.
open: a check that the mutation reached the live process; none exists.

### §132 (2026-09-03, R2-c)
The mutation test itself was broken — it mutated a copy, not the live object. Evidence: scripts/test_*.py.
open: a check that mutation tests mutate the live object; none exists.

### §56 (2026-09-02, R2-d)
A scanner split the string it was matching; the match read a fragment as the whole. Evidence: scripts/harness.py.
open: a check that scanners match structural boundaries, not substrings; none exists.

### §94 (2026-09-03, R2-d)
A symbol's name was present in the file, read as "assigned"; the name appeared in a string, not an assignment. Evidence: scripts/harness.py.
open: a check that name-presence is not read as assignment; none exists.

### §76 (2026-09-02, R2-e)
A fixture was built from the implementation's handled branches; unhandled branches — the ones that fail in production — were absent. Evidence: scripts/test_*.py.
open: a check that fixtures include unhandled branches; none exists.

### §98 (2026-09-03, R2-e)
A fixture had the same form as the formula under test; it could not detect a form error, only a value error. Evidence: scripts/test_*.py.
open: a check that fixtures are independent of the formula under test; none exists.

### §71 (2026-09-02, R2-f)
The guard condition and the assertion body read different keys; the guard blocked on one key while the assertion checked another. Evidence: scripts/harness.py.
open: a check that guard and assertion read the same key; none exists.

### §128 (2026-09-03, R2-f)
A check read the wrong key and produced a fake zero; the key it read was always 0. Evidence: scripts/harness.py.
open: a check that check keys are the ones the writer populates; none exists.

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

### §91 (2026-09-02, R2-g)
A test failure was diagnosed by suspecting the object, not the chain; the chain (fixture, runner, environment) was the failure. Evidence: scripts/test_*.py.
open: manual — diagnosis order is a human discipline.

### §110 (2026-09-03, R2-g)
A pre-registered branch collapsed two worlds into one; the criterion (branch taken) did not isolate the property (which world). Evidence: runs/prereg.jsonl.
open: a check that pre-registered branches isolate the worlds they name; none exists.

### §114 (2026-09-03, R2-g)
A null hypothesis was not isolated; the test rejected a null that was not the property's. Evidence: docs/lessons/experiments_0904.md.
open: manual — null-hypothesis design is a human discipline.

### §148 (2026-09-03, R2-g)
A statistic's null-hypothesis value was not the one asked; the statistic answered a different null. Evidence: docs/lessons/experiments_0904.md.
open: manual — statistic selection is a human discipline.

### §170 (2026-09-04, R2-g)
An unresolvable fact reference was used for four days; the criterion (reference present) did not measure the property (reference resolves). Evidence: docs/lessons/gate_failure_shapes.md (old §170).
open: check_fact_refs scans only docs/lessons+audits; widening to data/*.json and bare id forms would close it.

### §176 (2026-09-05, R2-g)
The expression under test supplied the ground truth it was judged against: an awk `substr($0,16)` off-by-three (correct offset 19) printed `ds/e1`, read as "branches are namespaced ds/<name>" — a false mechanism that was actionable and pointed away from the real bug. The artifact of the bug was indistinguishable from a fact about the repo. One independent reader (`git show-ref`) settles it in one command; it was never run. Evidence: scripts/merge_main.sh awk, 60a56434 (3b fix).
open: a check that findings about an instrument are re-derived from an independent reader; none exists. The general form: the thing being verified cannot also be the source of verification.

## R3. Artifacts carry their producer identity

### §24 (2026-08-31, R3)
A checkpoint with no recipe provenance was scored; the score was attributed to a recipe the checkpoint did not run. Evidence: runs/ckpt_*.pt.
open: a check that checkpoints carry recipe provenance before scoring; none exists.

## R4. Failures must be loud

### §51 (2026-09-01, R4)
An observation channel swallowed the signal; the check read the channel's default, not the observation. Evidence: scripts/harness.py.
open: a check that observation channels propagate errors; none exists.

## R5. State the vision before the number

### §30 (2026-09-01, R5)
A measurement was extrapolated beyond the vision; the extrapolation was not labeled. Evidence: docs/lessons/experiments_0904.md.
open: a check that extrapolations are labeled; none exists.

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

### §21 (2026-08-31, R6)
A number's basis was a different instrument from the one named; the instrument named was not the one that produced it. Evidence: docs/lessons/experiments_0904.md.
open: a check that the named instrument produced the number; none exists.

### §55 (2026-09-02, R6)
A number's resolution was finer than its basis; the extra digits were noise, not precision. Evidence: docs/lessons/experiments_0904.md.
open: manual — resolution matching is a human discipline.

### §63 (2026-09-02, R6)
A number's basis was a different seed; the seed was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their seed; none exists.

### §64 (2026-09-02, R6)
A number's basis was a different checkpoint; the checkpoint was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their checkpoint; none exists.

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

### §133 (2026-09-03, R6)
A number's basis was a different fold; the fold was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their fold; none exists.

### §152 (2026-09-03, R6)
A number's basis was a different aggregation; the aggregation was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their aggregation; none exists.

### §155 (2026-09-03, R6)
A number's basis was a different baseline; the baseline was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their baseline; none exists.

### §161 (2026-09-04, R6)
A number's basis was a different comparison; the comparison was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their comparison; none exists.

### §164 (2026-09-04, R6)
A number's basis was a different rounding; the rounding was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their rounding; none exists.

## R7. Retractions travel as wide as the ruling

### §16 (2026-08-31, R7)
A retraction was narrower than the ruling; the ruling voided a todo that the retraction did not name, and the todo was later acted on. Evidence: docs/standards/state_0904.md.
open: a check that retractions name every todo the ruling voids; none exists.

## R9. Run a deletion candidate before judging it

### §41 (2026-09-01, R9)
A deletion list was broadcast and deleted within the 24h window; an unclaimed candidate was still in use. Evidence: runs/pod_ckpt_candidates_*.txt.
open: a check that the 24h window elapsed before deletion; none exists.

## R10. What happened only on the pod did not happen

