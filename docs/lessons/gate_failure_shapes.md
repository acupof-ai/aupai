---
question: What are the rules that keep gates and measurements honest, and what enforces each?
status: open
source: derived from docs/lessons/gate_failure_incidents.md (141 surviving incidents); 33 closed incidents deleted 2026-09-04 (see commit)
---

# Gate failure rules

Ten rules. Each rule: statement, the check that enforces it (or `manual:` with the reason), one or two canonical incidents, and what the check cannot see. Incidents live in `gate_failure_incidents.md`. A sub-rule with ≥5 incidents and no check is a "check to write" (owner blank).

## Checks to write

- **R2-a No broken world** (6 incidents: §31 §69 §89 §103 §137 §153). A check that was never made to fail is decoration; the broken world must be asserted, not assumed. Owner: blank.
- **R2-b Population narrower than the property** (14 incidents: §26 §29 §34 §35 §40 §48 §65 §72 §121 §134 §146 §151 §169 §171). The check's scope, inputs, or environment do not cover the property asked. Owner: blank.
- **R2-d Parser reads prose as code** (5 incidents: §56 §61 §77 §94 §141). A grep/regex/text match reads comments, strings, or names as behavior. Owner: blank.
- **R2-f Guard reads the wrong field** (6 incidents: §54 §71 §75 §85 §125 §128). Guard and assertion read different keys, or the guard reads a key nobody writes. Owner: blank.
- **R2-g Criterion answers an adjacent question** (23 incidents: §9 §10 §23 §45 §67 §73 §84 §91 §108 §110 §112 §114 §135 §140 §142 §147 §148 §149 §150 §158 §165 §170 §173). The metric measures a neighbour property, not the one asked. Owner: blank.

## R1. Verify premises before acting, sources before citing; a correct conclusion does not certify its argument

`manual:` no check can verify that a human's premise matches the world; `check_fact_refs` (citations resolve) and `ckpt_facts_sources_present` (fact sources exist) cover the citation, not the argument.

- §66: saw literal `0` in `blocks=0`, concluded "not the config"; `0 or n_sub` made 0 the sentinel for Full. Read the default def and the consumer line, not the literal.
- §131: `tail` read a dead process's `SRCFP CHANGED` line as the current result. Read the artifact, not the log tail.

Cannot see: whether a true statement is being used to support an untested conclusion (§14, §18, §37, §38, §49, §52, §57, §70, §96, §106, §139).

## R2. A criterion must express the property asked; test it on known-answer positive and negative worlds before trusting output

Seven mechanism sub-rules. Five are "check to write" (above). Two have partial checks.

### R2-a No broken world

`manual:` the `--selftest` contract requires every CHECKS entry to carry `broken()`, but a selftest that passes on a broken world (the mutation did not take, or the world was never asserted) is invisible to the contract itself.

- §89: a selftest "passed" because the world-build step silently failed and the check ran on an empty population. The world must be asserted before the check.
- §103: a check that cannot fail — its acceptance condition is tautological. A check with no negative world is prose.

Cannot see: whether the selftest's broken world actually exercises the check's logic (§31, §69, §137, §153).

### R2-b Population narrower than the property

`manual:` no check verifies that a test's population matches the property's population; `check_selftests_are_gated` verifies the selftest is registered, not that its inputs cover the asked domain.

- §134: a unit test measured format compliance, not content correctness; the property asked was content. The test's population (format) is narrower than the property (content).
- §171: a perturbation was injected at a scale below the instrument's resolution; the property asked (sensitivity) was outside the test's population (resolution).

Cannot see: whether the test's inputs, environment, or scale match the property's (§26, §29, §34, §35, §40, §48, §65, §72, §121, §146, §151, §169).

### R2-c Mutation did not take

`manual:` mutation testing has no harness support; the verification that the mutation landed and the check caught it is a human discipline.

- §90: a mutation was applied but never landed in the running process; the check "passed" because the world was never mutated.
- §132: the mutation test itself was broken — it mutated a copy, not the live object.

Cannot see: whether the mutation reached the code path the check exercises (§81).

### R2-d Parser reads prose as code

`manual:` no check verifies that a grep/regex match is reading code rather than comments, strings, or names; `check_selftests_are_gated` cross-validates two parsers but only for the selftest registry.

- §61: a substring/word match read a comment mentioning the symbol as evidence the symbol was used. Text match is not structural match.
- §94: a symbol's name was present in the file, read as "assigned"; the name appeared in a string, not an assignment.

Cannot see: whether a text match is reading behavior or prose (§56, §77, §141).

### R2-e Fixture built from the implementation

`manual:` no check verifies that a fixture is independent of the code under test; a fixture derived from the implementation's handled branches or the live file cannot fail.

- §76: a fixture was built from the implementation's handled branches; unhandled branches — the ones that fail in production — were absent.
- §98: a fixture had the same form as the formula under test; it could not detect a form error, only a value error.

Cannot see: whether the fixture's construction is independent of the code it tests (§80, §97).

### R2-f Guard reads the wrong field

`manual:` no check verifies that a guard's condition reads the same key as the assertion it protects; `check_fact_refs` resolves citations, not field-level key agreement.

- §71: the guard condition and the assertion body read different keys; the guard blocked on one key while the assertion checked another.
- §125: a check read a pid file that was never written; the empty read was interpreted as "no process," not "no data."

Cannot see: whether the guard and the assertion agree on the key (§54, §75, §85, §128).

### R2-g Criterion answers an adjacent question

`manual:` no check verifies that a metric measures the property asked; a metric that measures a neighbour property passes while answering a different question.

- §110: a pre-registered branch collapsed two worlds into one; the criterion (branch taken) did not isolate the property (which world).
- §170: an unresolvable fact reference was used for four days; the criterion (reference present) did not measure the property (reference resolves).

Cannot see: whether the metric's null hypothesis is the property's null hypothesis (§9, §10, §23, §45, §67, §73, §84, §91, §108, §112, §114, §135, §140, §142, §147, §148, §149, §150, §158, §165, §173).

## R3. Artifacts carry their producer's identity; missing identity refuses, never rebuilds

`check_cache_readers_set_vocab_id` (registered CHECKS entry, `broken()` by contract) enforces vocab identity on cache readers; `train.py:1472` raises if `VOCAB_ID` is unset. Partial: covers vocab, not all producer identity.

- §4: an artifact with no producer identity was silently rebuilt; the rebuild used a different producer, and the artifact's meaning changed. Missing identity must refuse, not rebuild.
- §24: a checkpoint with no recipe provenance was scored; the score was attributed to a recipe the checkpoint did not run.

Cannot see: whether the identity a checkpoint carries is the identity it ran with (§44).

## R4. Failures must be loud: checks before the write, raise or exit nonzero, never print-and-continue

`manual:` loud-failure is a code-review property; some selftests assert exit codes, but no general check verifies that a failure path raises rather than prints.

- §13: a world-build step silently failed; the check ran on an empty population and passed. A silent failure is indistinguishable from success.
- §51: an observation channel swallowed the signal; the check read the channel's default, not the observation.

Cannot see: whether a print-and-continue path exists in code not covered by a selftest (§7, §25, §59, §136, §166).

## R5. State the vision before the number; outside it, label unmeasured, not absent

`manual:` vision-scope is a design property; no check verifies that a number's population is stated before the number is reported.

- §3: a number was reported without its population; the population (which items, which scale, which seed) was the unmeasured quantity that determined the number's meaning.
- §100: a measurement outside the stated vision was reported as "absent"; the correct label was "unmeasured."

Cannot see: whether a number's population matches the vision it is reported under (§5, §6, §17, §19, §28, §30, §32, §36, §53).

## R6. Every number carries its basis: source type, resolution, algorithm; label extrapolation

`manual:` basis-labeling is a discipline; `doc_numbers_check` partially verifies that docs numbers trace to facts, but does not verify the basis label is correct.

- §1: a number was quoted without its source type; the source type (measured / extrapolated / inferred) determined whether the number could be compared to another.
- §55: a number's resolution was finer than its basis; the extra digits were noise, not precision.

Cannot see: whether the basis a number carries is the basis it was produced with (§11, §12, §20, §21, §50, §62, §63, §64, §79, §86, §99, §104, §105, §109, §111, §115, §117, §118, §124, §127, §133, §143, §152, §155, §156, §157, §159, §161, §164, §172).

## R7. Retractions travel as wide as the ruling and name the todos they void; constraints are machine checks, not prose

`check_frozen_paths` (registered CHECKS entry) enforces frozen-path constraints; partial: covers frozen paths, not retraction width.

- §16: a retraction was narrower than the ruling; the ruling voided a todo that the retraction did not name, and the todo was later acted on.
- §58: a constraint was stated in prose, not enforced by a machine check; the constraint was violated without a signal.

Cannot see: whether a retraction reached every consumer of the original ruling (§22, §68, §102, §119).

## R8. Shared resources are explicitly exclusive; co-residency is judged by each implementation's measured cost in seconds against the run's own spend, never by metric class

`check_card_held_without_claim` + `check_free_card` (registered CHECKS entries) enforce card exclusivity; partial: covers cards, not all shared resources.

- §15: a shared resource was used without an explicit claim; the co-residency cost was measured against a metric class, not the run's own spend.
- §126: a resource's exclusivity was inferred from "0 MiB" in nvidia-smi; idle is not a grant.

Cannot see: whether a non-card shared resource (disk, network, host DRAM) is co-resident with a run it degrades.

## R9. Run a deletion candidate before judging it; broadcast the list, delete after 24h unclaimed

`ckpt_facts_sources_present` + `check_keep_claim_reasons_live` (registered CHECKS entries) enforce checkpoint KEEP claims; partial: covers checkpoints, not all deletion candidates.

- §39: a deletion candidate was judged without running it; the candidate was a live process, not a stale file.
- §41: a deletion list was broadcast and deleted within the 24h window; an unclaimed candidate was still in use.

Cannot see: whether a non-checkpoint deletion candidate (a process, a lease, a temp file) is live.

## R10. What happened only on the pod did not happen; bring it back to the repo the same day

`manual:` pod-vs-repo is a discipline; no check verifies that a pod-only measurement was brought back to the repo the same day.

- §2: a measurement existed only on the pod; the pod was recycled, and the measurement was lost. What happened only on the pod did not happen.
- §116: a pod-only artifact was cited in a decision; the artifact was unreachable from the repo, and the decision rested on an unreadable source.

Cannot see: whether a pod-only measurement was brought back before the pod was recycled.
