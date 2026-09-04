---
question: What are the rules that keep gates and measurements honest, what enforces each, and what does each cost?
status: open
source: derived from docs/lessons/gate_failure_incidents.md (64 model-project incidents) and docs/lessons/infra_incidents.md (85 pod/infra incidents); 33 closed incidents removed 2026-09-04 (149 = 64 + 85); 33/33 confirmed machine-gated (list below)
---

# Gate failure rules

Ten rules, ranked by incidents × cost (rough hours lost per incident). Each rule: statement, count, cost, the check that enforces it (or `manual:` with the reason), one or two canonical incidents, and what the check cannot see. Incidents live in `gate_failure_incidents.md` (model-project) and `infra_incidents.md` (pod/infra). The top 5 by product are checks to write; the rest are rules people read.

Cost is an estimate: R2 (criterion) ~4h/incident (wrong measurements, false greens, some cost days); R1/R3/R4/R10 ~3-4h; R5/R6/R7/R8 ~2h; R9 ~1h.

## Closed incidents (33/33 confirmed machine-gated, 2026-09-04)

10 sampled by 6e, 23 sampled by 44, 5 reviewed by e1 (4c). Each line: §N: gate `file::function`.

§27: launch_tests degrade-string check (6e sample)
§33: `scripts/harness.py::check_no_shared_stash`
§42: `scripts/harness.py::check_frozen_paths`
§43: vocab_id_on_load_path (6e sample)
§47: `scripts/test_vocab_stamp.py` selftest
§60: `scripts/harness.py::check_launch_line_vs_oom_facts`
§74: `scripts/harness.py::_selftest_flagless_test_is_gated` (odd-quote arm)
§78: `eval/domain_loss.py` selftest (0-rows refusal)
§82: `scripts/test_ledger_predicates.py` selftest
§83: `scripts/sft_hf_control.py` source-level pre-shift scan
§87: `scripts/eval_heldout.py::alignment_sentinel`
§88: `scripts/ledger_audit.py` selftest (index-first read)
§92: ledger_audit.duplicates (6e sample)
§93: `scripts/harness.py::check_tasks_closed_by_commit`
§95: `scripts/test_cursor_sum.py::_check_call_sites`
§101: launch_gate mix/UNRECORDED (6e sample)
§107: `scripts/harness.py::check_ckpt_facts_sources_present`
§113: `scripts/harness.py::check_keep_claim_reasons_live`
§120: card_claim basename wait (6e sample)
§122: `scripts/test_sft_lr_provenance.py` case 4 (AST interpolation)
§123: `scripts/test_sft_lr_provenance.py` case 4b (exec shipped block)
§129: fp_dir import (6e sample)
§130: `scripts/head_path_rows.py` selftest case 5
§138: `scripts/test_shard_glob.py` selftest
§144: a2a selftest (6e sample)
§145: `eval/test_l1_fewshot_2x2.py` group 9 (answer_marker disjunction)
§154: `scripts/harness.py::check_card_held_without_claim`
§160: card_claim _cvd selftests (6e sample)
§162: `scripts/gen_ckpt_listing.py::build` (inode pin + refusal on missing claimed file; e1 correction: not check_milestone_ckpt_pinned, which reads milestones.jsonl only)
§163: `scripts/harness.py::_selftest_commit_delivers_fact_ref`
§167: `scripts/test_e1_28_leak_scan.py` selftest (units refusal)
§168: `scripts/harness.py::check_eval_registry_complete`
§174: `scripts/hooks/pre-commit` world 8 (stale __pycache__ fix)

## Checks to write (top 5 by product)

- **R2** (66 incidents, 264h): a criterion must express the property asked; test it on known-answer positive and negative worlds. Split into 7 sub-rules below; each sub-rule is a check target. Owner: blank.
- **R6** (32 incidents, 64h): every number carries its basis. Owner: blank.
- **R1** (17 incidents, 51h): verify premises before acting, sources before citing. Owner: blank.
- **R5** (11 incidents, 22h): state the vision before the number. Owner: blank.
- **R4** (8 incidents, 24h): failures must be loud. Owner: blank.

## R2. A criterion must express the property asked; test it on known-answer positive and negative worlds before trusting output

66 incidents (34 infra, 32 model), ~4h each, 264h. `manual:` no check verifies that a criterion expresses the property asked; `--selftest` requires every CHECKS entry to carry `broken()`, but a selftest that passes on a broken world is invisible to the contract.

Seven mechanism sub-rules. Each is a check target.

### R2-a No broken world (6 incidents)

A check that was never made to fail is decoration; the broken world must be asserted, not assumed.

- §89: a selftest "passed" because the world-build step silently failed and the check ran on an empty population.
- §103: a check that cannot fail — its acceptance condition was tautological.

Cannot see: whether the selftest's broken world actually exercises the check's logic (§31, §69, §137, §153).

### R2-b Population narrower than the property (15 incidents)

The check's scope, inputs, or environment do not cover the property asked.

- §134: a unit test measured format compliance, not content correctness; the property asked was content.
- §171: a perturbation was injected at a scale below the instrument's resolution; the property asked (sensitivity) was outside the test's population.

Cannot see: whether the test's inputs, environment, or scale match the property's (§26, §29, §34, §35, §40, §48, §65, §72, §121, §146, §151, §169, §180).

### R2-c Mutation did not take (3 incidents)

The mutation never landed or its verification reads the wrong signal.

- §90: a mutation was applied but never landed in the running process; the check "passed" because the world was never mutated.
- §132: the mutation test itself was broken — it mutated a copy, not the live object.

Cannot see: whether the mutation reached the code path the check exercises (§81).

### R2-d Parser reads prose as code (5 incidents)

A grep/regex/text match reads comments, strings, or names as behavior.

- §61: a substring/word match read a comment mentioning the symbol as evidence the symbol was used.
- §94: a symbol's name was present in the file, read as "assigned"; the name appeared in a string, not an assignment.

Cannot see: whether a text match is reading behavior or prose (§56, §77, §141).

### R2-e Fixture built from the implementation (4 incidents)

A fixture derived from the implementation's handled branches or the live file cannot fail.

- §76: a fixture was built from the implementation's handled branches; unhandled branches — the ones that fail in production — were absent.
- §98: a fixture had the same form as the formula under test; it could not detect a form error, only a value error.

Cannot see: whether the fixture's construction is independent of the code it tests (§80, §97).

### R2-f Guard reads the wrong field (6 incidents)

Guard and assertion read different keys, or the guard reads a key nobody writes.

- §71: the guard condition and the assertion body read different keys; the guard blocked on one key while the assertion checked another.
- §125: a check read a pid file that was never written; the empty read was interpreted as "no process," not "no data."

Cannot see: whether the guard and the assertion agree on the key (§54, §75, §85, §128).

### R2-g Criterion answers an adjacent question (27 incidents)

The metric measures a neighbour property, not the one asked.

- §110: a pre-registered branch collapsed two worlds into one; the criterion (branch taken) did not isolate the property (which world).
- §170: an unresolvable fact reference was used for four days; the criterion (reference present) did not measure the property (reference resolves).
- §177: an arm's flags said it carried a 1.07B-parameter memory table; the criterion (the flags the run was given) did not measure the property (which of two code paths consumed them), and the arm would have trained as the control and reported a clean null.

Cannot see: whether the metric's null hypothesis is the property's null hypothesis (§9, §10, §23, §45, §67, §73, §84, §91, §108, §112, §114, §135, §140, §142, §147, §148, §149, §150, §158, §165, §173, §174, §176, §177, §178).

## R6. Every number carries its basis: source type, resolution, algorithm; label extrapolation

32 incidents (12 infra, 20 model), ~2h each, 64h. `manual:` basis-labeling is a discipline; `doc_numbers_check` partially verifies that docs numbers trace to facts, but does not verify the basis label is correct.

- §1: a number was quoted without its source type; the source type (measured / extrapolated / inferred) determined whether the number could be compared to another.
- §55: a number's resolution was finer than its basis; the extra digits were noise, not precision.

Cannot see: whether the basis a number carries is the basis it was produced with (§11, §12, §20, §21, §50, §62, §63, §64, §79, §86, §99, §104, §105, §109, §111, §115, §117, §118, §124, §127, §133, §143, §152, §155, §156, §157, §159, §161, §164, §172).

## R1. Verify premises before acting, sources before citing; a correct conclusion does not certify its argument

17 incidents (11 infra, 6 model), ~3h each, 51h. `manual:` no check can verify that a human's premise matches the world; `check_fact_refs` (citations resolve) and `ckpt_facts_sources_present` (fact sources exist) cover the citation, not the argument.

- §66: saw literal `0` in `blocks=0`, concluded "not the config"; `0 or n_sub` made 0 the sentinel for Full. Read the default def and the consumer line, not the literal.
- §131: `tail` read a dead process's `SRCFP CHANGED` line as the current result. Read the artifact, not the log tail.

Cannot see: whether a true statement is being used to support an untested conclusion (§8, §14, §18, §37, §38, §46, §49, §52, §57, §70, §96, §106, §139, §175, §179).

## R5. State the vision before the number; outside it, label unmeasured, not absent

11 incidents (10 infra, 1 model), ~2h each, 22h. `manual:` vision-scope is a design property; no check verifies that a number's population is stated before the number is reported.

- §3: a number was reported without its population; the population (which items, which scale, which seed) was the unmeasured quantity that determined the number's meaning.
- §100: a measurement outside the stated vision was reported as "absent"; the correct label was "unmeasured."

Cannot see: whether a number's population matches the vision it is reported under (§5, §6, §17, §19, §28, §30, §32, §36, §53).

## R4. Failures must be loud: checks before the write, raise or exit nonzero, never print-and-continue

8 incidents (6 infra, 2 model), ~3h each, 24h. `manual:` loud-failure is a code-review property; some selftests assert exit codes, but no general check verifies that a failure path raises rather than prints.

- §13: a world-build step silently failed; the check ran on an empty population and passed. A silent failure is indistinguishable from success.
- §51: an observation channel swallowed the signal; the check read the channel's default, not the observation.

Cannot see: whether a print-and-continue path exists in code not covered by a selftest (§7, §25, §59, §136, §166, §181).

## R7. Retractions travel as wide as the ruling and name the todos they void; constraints are machine checks, not prose

6 incidents (5 infra, 1 model), ~2h each, 12h. `check_frozen_paths` (registered CHECKS entry) enforces frozen-path constraints; partial: covers frozen paths, not retraction width.

- §16: a retraction was narrower than the ruling; the ruling voided a todo that the retraction did not name, and the todo was later acted on.
- §58: a constraint was stated in prose, not enforced by a machine check; the constraint was violated without a signal.

Cannot see: whether a retraction reached every consumer of the original ruling (§22, §68, §102, §119).

## R3. Artifacts carry their producer's identity; missing identity refuses, never rebuilds

3 incidents (2 infra, 1 model), ~4h each, 12h. `check_cache_readers_set_vocab_id` (registered CHECKS entry) enforces vocab identity on cache readers; `train.py:1472` raises if `VOCAB_ID` is unset. Partial: covers vocab, not all producer identity.

- §4: an artifact with no producer identity was silently rebuilt; the rebuild used a different producer, and the artifact's meaning changed. Missing identity must refuse, not rebuild.
- §24: a checkpoint with no recipe provenance was scored; the score was attributed to a recipe the checkpoint did not run.

Cannot see: whether the identity a checkpoint carries is the identity it ran with (§44).

## R10. What happened only on the pod did not happen; bring it back to the repo the same day

2 incidents (2 infra, 0 model), ~4h each, 8h. `manual:` pod-vs-repo is a discipline; no check verifies that a pod-only measurement was brought back to the repo the same day.

- §2: a measurement existed only on the pod; the pod was recycled, and the measurement was lost. What happened only on the pod did not happen.
- §116: a pod-only artifact was cited in a decision; the artifact was unreachable from the repo, and the decision rested on an unreadable source.

Cannot see: whether a pod-only measurement was brought back before the pod was recycled.

## R8. Shared resources are explicitly exclusive; co-residency is judged by each implementation's measured cost in seconds against the run's own spend, never by metric class

2 incidents (2 infra, 0 model), ~2h each, 4h. `check_card_held_without_claim` + `check_free_card` (registered CHECKS entries) enforce card exclusivity; partial: covers cards, not all shared resources.

- §15: a shared resource was used without an explicit claim; the co-residency cost was measured against a metric class, not the run's own spend.
- §126: a resource's exclusivity was inferred from "0 MiB" in nvidia-smi; idle is not a grant.

Cannot see: whether a non-card shared resource (disk, network, host DRAM) is co-resident with a run it degrades.

## R9. Run a deletion candidate before judging it; broadcast the list, delete after 24h unclaimed

2 incidents (1 infra, 1 model), ~1h each, 2h. `ckpt_facts_sources_present` + `check_keep_claim_reasons_live` (registered CHECKS entries) enforce checkpoint KEEP claims; partial: covers checkpoints, not all deletion candidates.

- §39: a deletion candidate was judged without running it; the candidate was a live process, not a stale file.
- §41: a deletion list was broadcast and deleted within the 24h window; an unclaimed candidate was still in use.

Cannot see: whether a non-checkpoint deletion candidate (a process, a lease, a temp file) is live.
