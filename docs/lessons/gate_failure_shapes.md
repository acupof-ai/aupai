---
question: What are the rules that keep gates and measurements honest, what enforces each, and what does each cost?
status: open
source: derived from docs/lessons/gate_failure_incidents.md (113 model-project incidents) and docs/lessons/infra_incidents.md (88 pod/infra incidents); 33 closed incidents removed 2026-09-04 (201 = 113 + 88); 33/33 confirmed machine-gated (list below)
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

- **R2** (101 incidents, 336h): a criterion must express the property asked; test it on known-answer positive and negative worlds. Split into 7 sub-rules below; each sub-rule is a check target. Owner: blank.
- **R6** (34 incidents, 68h): every number carries its basis. Owner: blank.
- **R1** (21 incidents, 63h): verify premises before acting, sources before citing. Owner: blank.
- **R5** (11 incidents, 22h): state the vision before the number. Owner: blank.
- **R4** (12 incidents, 36h): failures must be loud. Owner: blank.

## R2. A criterion must express the property asked; test it on known-answer positive and negative worlds before trusting output

101 incidents (34 infra, 67 model), ~4h each, 336h. `manual:` no check verifies that a criterion expresses the property asked; `--selftest` requires every CHECKS entry to carry `broken()`, but a selftest that passes on a broken world is invisible to the contract.

Seven mechanism sub-rules. Each is a check target.

### R2-a No broken world (11 incidents)

A check that was never made to fail is decoration; the broken world must be asserted, not assumed.

- §89: a selftest "passed" because the world-build step silently failed and the check ran on an empty population.
- §103: a check that cannot fail — its acceptance condition was tautological.
- §218: a fixture sampling "a recent commit" drew a merge commit; `git show --name-only` on a merge prints no files, so the world had no subject and read as broken code.
- §219: the next filter over the same sample rejected the only file the first non-merge commit touched (.jsonl not in the extension list); both were caught only because the world FAILs rather than skips when it cannot find its subject.
- §228: a world copied from real files was too incomplete for the subject to import (only scripts/ on sys.path; `from train import ...` died at module scope); the subject's ModuleNotFoundError read as the subject being broken. An import error in a fixture's verdict is a fixture bug until proven otherwise.
- §231: agreement between two things that share an error is not evidence — a fixture with no power to disagree (one directory, so cwd and $MAIN are the same path) reported agreement and it read as confirmation; a differential fixture must be fed an input where the two sides are known to differ.

Ledger-field semantics (test_ledger_field_writers.py, 315755cc): class/cards ABSENT means unstated and "" is forbidden (indistinguishable from a pre-field row; 243 historical rows stay null, no backfill); 'none' is a STATED cards answer for a CPU or corpus job. defect_caught "" is a REAL clean-review answer; absent means no review reported.

Cannot see: whether the selftest's broken world actually exercises the check's logic (§31, §69, §137, §153, §206, §218, §219, §228, §231).

### R2-b Population narrower than the property (28 incidents)

The check's scope, inputs, or environment do not cover the property asked.

- §134: a unit test measured format compliance, not content correctness; the property asked was content.
- §171: a perturbation was injected at a scale below the instrument's resolution; the property asked (sensitivity) was outside the test's population.
- §201: a device-fd refusal verified only where it cannot fire (macOS, no /proc) reported nothing about where it does (pod, /proc present); all ten claim sites would have been refused on the pod.
- §215: a battery of 19 content-free rules passed while the leak family it samples is unbounded; three closures in one day did not converge, and the battery certifies its sample, not the family.
- §216: a negative control passed on every laptop because the pod mount is absent there — green was a signal about a different world; on the pod it would have tokenized into the live shared cache dir.
- §222: an assertion read claims()'s *.json glob as the claims directory; a duplicate written as <file>.dup survived — a reader-based assertion inherits the reader's blind spot.
- §223: a module test asserted optimizer-group membership and never called .step(); green at 10/10 while the Muon 4-D stack died at the first optimizer step on the card — a new parameter class is a new citizen for every subsystem that dispatches on shape or type.
- §224: a cleanup sweep sat behind 26 sys.exit(1) calls in main(), so it only ran on a commit that had already passed every gate — a cleanup placed after the gates cleans up only when nothing needed cleaning.
- §225: a hook edited on a branch runs main's old copy, so the change appeared to work — a test result attributed to code that did not produce it.
- §229: a gate on main read its evidence ledger from the working tree (bare `open()` where every git call used `-C "$MAIN"`); a branch-only review row satisfied the second-reader gate that exists to certify somebody else signed — a gate's inputs must come from the same namespace as the thing it gates.
- §232: a ledger-diff signature over six fields reported set-equal pairs whose rows differed only in the fields it dropped; a comparison that exists to surface disagreements must sign the whole row, since any excluded field is a disagreement it cannot see.

Cannot see: whether the test's inputs, environment, or scale match the property's (§26, §29, §34, §35, §40, §48, §65, §72, §121, §146, §151, §169, §180, §201, §202, §203, §209, §213, §215, §216, §222, §223, §224, §225, §229, §232).

### R2-c Mutation did not take (7 incidents)

The mutation never landed or its verification reads the wrong signal.

- §90: a mutation was applied but never landed in the running process; the check "passed" because the world was never mutated.
- §132: the mutation test itself was broken — it mutated a copy, not the live object.
- §221: a test recomputed the quantity outside the function and asserted its own arithmetic — the function under test was never called, so no mutant of it can reach the test.
- §227: a refusal raises SystemExit (BaseException, not Exception); the test's `except Exception` let it through, so the mutation was caught by the process dying, not by the assertion.
- §233: three selftest cases passed with the new rule deleted entirely — the fixture routed around it into a pre-existing clause giving the same answer; a case must include a shape where the OLD logic answers differently, enforced by an in-case assertion naming the old answer (a disagreement property, not a coverage property).

Cannot see: whether the mutation reached the code path the check exercises (§81, §207, §221, §227, §233).

### R2-d Parser reads prose as code (11 incidents)

A grep/regex/text match reads comments, strings, or names as behavior.

- §61: a substring/word match read a comment mentioning the symbol as evidence the symbol was used.
- §94: a symbol's name was present in the file, read as "assigned"; the name appeared in a string, not an assignment.
- §196: a scanner located its subject by a delimiter and matched a line carrying that delimiter as a regex STRING, capturing five characters of the pattern itself.
- §200: a guard against an omission, written by substring, omitted itself — the names it searched for appear in its own comment and data table, so it read 3/3 present under a mutant that deleted all three call sites.
- §205: a placeholder-survival guard fired on a correct substitution — the template's own documentation line names the placeholder, and a whole-file scan read that comment as an unsubstituted token; fourth instance of the self-satisfying needle.
- §217: a whole-file substring assertion survived a mutation repointing both executable lines, because the block's own comment named the real path — prose vouching for code that had stopped agreeing with it.
- §226: two regexes over a Python literal were wrong in opposite directions (182 with a revspec from a comment, 149 of 177 one-per-line); ast.literal_eval cannot disagree with the literal by construction.

Cannot see: whether a text match is reading behavior or prose (§56, §77, §141, §205, §212, §217, §226).

### R2-e Fixture built from the implementation (5 incidents)

A fixture derived from the implementation's handled branches or the live file cannot fail.

- §76: a fixture was built from the implementation's handled branches; unhandled branches — the ones that fail in production — were absent.
- §98: a fixture had the same form as the formula under test; it could not detect a form error, only a value error.
- §220: the unclipped baseline was computed by the function under test, so an inverted-ratio mutant inverted both sides and the inequality still held — a test comparing code against itself.

Generalization (e1, 2026-09-05): a differential assertion has power only if its two sides can fail differently. A same-function baseline (§220) and a self-recomputed baseline (§221) are the two ways to lose that, in opposite directions, and both were hit within twenty minutes on one assertion. The operational check is to name where the expected value comes from before writing the comparison: "the function I am testing" and "logic I reimplemented" are both wrong answers; the right one is a property of the fixture with the call under test appearing in the comparison.

Cannot see: whether the fixture's construction is independent of the code it tests (§80, §97, §220).

### R2-f Guard reads the wrong field (6 incidents)

Guard and assertion read different keys, or the guard reads a key nobody writes.

- §71: the guard condition and the assertion body read different keys; the guard blocked on one key while the assertion checked another.
- §125: a check read a pid file that was never written; the empty read was interpreted as "no process," not "no data."

Cannot see: whether the guard and the assertion agree on the key (§54, §75, §85, §128).

### R2-g Criterion answers an adjacent question (31 incidents)

The metric measures a neighbour property, not the one asked.

- §110: a pre-registered branch collapsed two worlds into one; the criterion (branch taken) did not isolate the property (which world).
- §170: an unresolvable fact reference was used for four days; the criterion (reference present) did not measure the property (reference resolves).
- §177: an arm's flags said it carried a 1.07B-parameter memory table; the criterion (the flags the run was given) did not measure the property (which of two code paths consumed them), and the arm would have trained as the control and reported a clean null.
- §184: excluding a parameter from the fp32 master copy would have left it read every forward and never updated; the criterion (is the exclusion correct) did not measure the property (who clears its gradient), and the diagnostics would have shown a healthy pool.

Cannot see: whether the metric's null hypothesis is the property's null hypothesis (§9, §10, §23, §45, §67, §73, §84, §91, §108, §112, §114, §135, §140, §142, §147, §148, §149, §150, §158, §165, §173, §174, §176, §177, §178, §184, §191, §208).

## R6. Every number carries its basis: source type, resolution, algorithm; label extrapolation

35 incidents (12 infra, 23 model), ~2h each, 68h. `manual:` basis-labeling is a discipline; `doc_numbers_check` partially verifies that docs numbers trace to facts, but does not verify the basis label is correct.

- §1: a number was quoted without its source type; the source type (measured / extrapolated / inferred) determined whether the number could be compared to another.
- §55: a number's resolution was finer than its basis; the extra digits were noise, not precision.
- §185: a memory budget was costed at 6 bytes per parameter from a bf16 table nobody had set; the tensors are fp32 and the gradient was omitted, so the real figure is 12 and the 2048^2 arm OOMed after construction succeeded.
- §230: a review reported five checks as MEASURED that had only been READ; the figure then acquired a second independent-looking source when repeated back, with zero executions. A stated basis is itself a claim — ask "when did this command run" of your own claim. Second instance the same day: a derived ratio carried across a rebuild of its inputs, so the digits in the decision document matched neither the old quantity nor the new one.

Cannot see: whether the basis a number carries is the basis it was produced with (§11, §12, §20, §21, §50, §62, §63, §64, §79, §86, §99, §104, §105, §109, §111, §115, §117, §118, §124, §127, §133, §143, §152, §155, §156, §157, §159, §161, §164, §172, §185, §192).

## R1. Verify premises before acting, sources before citing; a correct conclusion does not certify its argument

21 incidents (11 infra, 10 model), ~3h each, 63h. `manual:` no check can verify that a human's premise matches the world; `check_fact_refs` (citations resolve) and `ckpt_facts_sources_present` (fact sources exist) cover the citation, not the argument.

- §66: saw literal `0` in `blocks=0`, concluded "not the config"; `0 or n_sub` made 0 the sentinel for Full. Read the default def and the consumer line, not the literal.
- §131: `tail` read a dead process's `SRCFP CHANGED` line as the current result. Read the artifact, not the log tail.

Cannot see: whether a true statement is being used to support an untested conclusion (§8, §14, §18, §37, §38, §46, §49, §52, §57, §70, §96, §106, §131, §139, §175, §179, §190, §198, §199, §211).

## R5. State the vision before the number; outside it, label unmeasured, not absent

11 incidents (10 infra, 1 model), ~2h each, 22h. `manual:` vision-scope is a design property; no check verifies that a number's population is stated before the number is reported.

- §3: a number was reported without its population; the population (which items, which scale, which seed) was the unmeasured quantity that determined the number's meaning.
- §100: a measurement outside the stated vision was reported as "absent"; the correct label was "unmeasured."

Cannot see: whether a number's population matches the vision it is reported under (§5, §6, §17, §19, §28, §30, §32, §36, §53).

## R4. Failures must be loud: checks before the write, raise or exit nonzero, never print-and-continue

12 incidents (7 infra, 5 model), ~3h each, 36h. `manual:` loud-failure is a code-review property; some selftests assert exit codes, but no general check verifies that a failure path raises rather than prints.

- §13: a world-build step silently failed; the check ran on an empty population and passed. A silent failure is indistinguishable from success.
- §51: an observation channel swallowed the signal; the check read the channel's default, not the observation.

Cannot see: whether a print-and-continue path exists in code not covered by a selftest (§7, §25, §59, §136, §166, §181, §188, §193, §197, §204).

## R7. Retractions travel as wide as the ruling and name the todos they void; constraints are machine checks, not prose

6 incidents (5 infra, 1 model), ~2h each, 12h. `check_frozen_paths` (registered CHECKS entry) enforces frozen-path constraints; partial: covers frozen paths, not retraction width.

- §16: a retraction was narrower than the ruling; the ruling voided a todo that the retraction did not name, and the todo was later acted on.
- §58: a constraint was stated in prose, not enforced by a machine check; the constraint was violated without a signal.

Cannot see: whether a retraction reached every consumer of the original ruling (§22, §68, §102, §119).

## R3. Artifacts carry their producer's identity; missing identity refuses, never rebuilds

6 incidents (2 infra, 4 model), ~4h each, 24h. `check_cache_readers_set_vocab_id` (registered CHECKS entry) enforces vocab identity on cache readers; `train.py:1472` raises if `VOCAB_ID` is unset. Partial: covers vocab, not all producer identity.

- §4: an artifact with no producer identity was silently rebuilt; the rebuild used a different producer, and the artifact's meaning changed. Missing identity must refuse, not rebuild.
- §24: a checkpoint with no recipe provenance was scored; the score was attributed to a recipe the checkpoint did not run.
- §189: a close written without --started stamped the row with its own write time and minted a third identity for a run that never existed; the ledger's fold key is (name, started), so the verdict and the numbers now sit on a phantom row.

Cannot see: whether the identity a checkpoint carries is the identity it ran with (§44, §189, §210).

## R10. What happened only on the pod did not happen; bring it back to the repo the same day

2 incidents (2 infra, 0 model), ~4h each, 8h. `manual:` pod-vs-repo is a discipline; no check verifies that a pod-only measurement was brought back to the repo the same day.

- §2: a measurement existed only on the pod; the pod was recycled, and the measurement was lost. What happened only on the pod did not happen.
- §116: a pod-only artifact was cited in a decision; the artifact was unreachable from the repo, and the decision rested on an unreadable source.

Cannot see: whether a pod-only measurement was brought back before the pod was recycled.

## R8. Shared resources are explicitly exclusive; co-residency is judged by each implementation's measured cost in seconds against the run's own spend, never by metric class

5 incidents (5 infra, 0 model), ~2h each, 10h. `check_card_held_without_claim` + `check_free_card` (registered CHECKS entries) enforce card exclusivity; partial: covers cards, not all shared resources, and WARNs after the launch rather than refusing it.

- §15: a shared resource was used without an explicit claim; the co-residency cost was measured against a metric class, not the run's own spend.
- §126: a resource's exclusivity was inferred from "0 MiB" in nvidia-smi; idle is not a grant.
- §194: a claim held by a live pid was read as evidence the job was progressing; 0% util against 76 GiB held was the signal, the claim status was not.
- §195: a rank-0-only phase (save, 33.6 s) inside a world-2 job desynchronised the ranks; rank 1 entered the next collective with nothing to meet.
- §214: a live job ran unclaimed on card 0 and every reader read it as an orphan; the claim-write is the only thing separating "orphan" from "unclaimed live job", so the unclaimed launch was the defect, not the reading.

Cannot see: whether a non-card shared resource (disk, network, host DRAM) is co-resident with a run it degrades; whether a launch that never wrote a claim is refused before it starts (§214).

## R9. Run a deletion candidate before judging it; broadcast the list, delete after 24h unclaimed

2 incidents (1 infra, 1 model), ~1h each, 2h. `ckpt_facts_sources_present` + `check_keep_claim_reasons_live` (registered CHECKS entries) enforce checkpoint KEEP claims; partial: covers checkpoints, not all deletion candidates.

- §39: a deletion candidate was judged without running it; the candidate was a live process, not a stale file.
- §41: a deletion list was broadcast and deleted within the 24h window; an unclaimed candidate was still in use.

Cannot see: whether a non-checkpoint deletion candidate (a process, a lease, a temp file) is live.

## Design cause: integration happens in a shared writable working tree

User ruling 2026-09-05: analyse to the root, not the surface. The incidents below are ONE cause with surfaces; a shape that names the operator's slip (a timeout wrapper, a cp -r, a stash) as the cause is the surface reading, and this section exists so the doc says so.

**The cause.** main advances by `git merge` run INSIDE a shared writable working tree (merge_main.sh:339), so integration is a non-atomic four-step write — checkout, index, merge commit, hook — to a directory every session reads and writes. Every rule in AGENTS.md's coordination section compensates for this design: the mkdir lock, the index-equals-HEAD rule, the stash rule, the behind-main refusal, the merge-drop restore. None of them is the cause; each is a patch on it.

**The surfaces.**
1. §224 — the .hookstaged sweep sat behind 26 sys.exit(1) calls in pre-commit main(). The sweep exists because a hook killed mid-selftest leaves .hookstaged_* files in the SHARED tree; in a private worktree a leftover is disposable.
2. §225 — a hook edited on a branch runs main's old copy. `.git/hooks/pre-commit` resolves against the shared integration tree's worktree, so the edited hook is not the installed hook until merged; the hoist appeared to work while shipping an UnboundLocalError only ruff caught.
3. The cp -r mutant (cf3dbaea "probe 2", reverted 533a4639, refused since 0375ee1c) — a `cp -r` of a linked worktree kept the `.git` gitdir pointer, so a commit in /tmp/reg_moe landed on the real branch and merged to main carrying a SwiGLU mutant. The workaround exists because the integration tree cannot be experimented in. de's scratch-repo fixture for the hook (§228 instance 2) is the same motion one step more benign: copied scripts/, missed datagen/.
4. The .hookstaged leftovers — the measured instance behind §224: `runs/audit_0904/.hookstaged_dead_worlds.py` sat in the shared tree for hours across a dozen commits.
5. The stash rule (AGENTS.md:369, `no_shared_stash`) — `.git/refs/stash` is one stack shared by every worktree of the one checkout; two sessions stashing in the same window each pop the other's entry.
6. index-equals-HEAD (AGENTS.md:370) — a three-way merge writes the index of the shared working tree, so any staged record is clobbered by another session's merge, related path or not.
7. Stranded merges — de's SIGKILLed merge_main on 2026-09-05 left the shared tree mid-merge; a killed merge in a shared tree leaves a staged tree without MERGE_HEAD for the next session (measured twice the same day).

**The fix landed 2026-09-05 (f6299671, tilerl).** main advances by compare-and-swap `git update-ref`; merges and hooks run in the committer's own worktree; the integration tree is no longer a checkout. The mootness markings below are now operative, not predicted.

**What becomes moot when the fix lands.**
- §224's sweep and the .hookstaged leftovers (surface 4): no shared tree to pollute; the committer's worktree is disposable. MOOT.
- §225: the edited hook IS the installed hook in the committer's own worktree. MOOT.
- The cp -r mutant (surface 3): no shared checkout to copy or experiment around. MOOT.
- index-equals-HEAD (surface 6) as a COORDINATION rule: another session's merge cannot touch your index. The local form — don't merge with a dirty index — survives as ordinary git hygiene. MOOT as coordination.
- Stranded merges (surface 7): a killed merge affects only the committer's own worktree; the integration ref advances atomically. MOOT.
- The stash rule (surface 5): NOT mooted — `.git/refs/stash` is shared across worktrees of the same repo regardless of the integration tree. Survives.
