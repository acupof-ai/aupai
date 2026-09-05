---
question: What are the surviving pod/infra gate-failure incidents, what would close each, and which are already closed?
status: open
source: derived from the 2026-09-04 restructure of gate_failure_shapes.md; 33 closed incidents removed (33/33 confirmed machine-gated, list in gate_failure_shapes.md), 141 survive here and in gate_failure_incidents.md
---

# Gate failure incidents — pod/infra layer

**This file belongs to the infra layer and moves with it when the pod tooling leaves the project.** One entry per surviving §. Grouped by rule. Each entry: id, date, rule/sub-rule, one-line incident, evidence, `open:` what would close it. Model-project incidents live in `gate_failure_incidents.md`. Closed incidents (33) were deleted in the 2026-09-04 restructure; their closing mechanisms are named in the commit message.

## R1. Verify premises before acting, sources before citing

### §14 (2026-08-31, R1)
A correct conclusion was used to certify an untested argument; the conclusion's truth did not extend to the argument's premises. Evidence: docs/lessons/experiments_0904.md.
open: manual — no check can verify that a true conclusion supports the argument it is attached to.

### §18 (2026-08-31, R1)
A source was cited for a claim it did not make; the citation resolved but the fact's content was adjacent, not supporting. Evidence: facts/data_scaling.json.
open: a check that reads the cited fact's value and compares it to the claim; none exists.

### §38 (2026-09-01, R1)
A number was quoted from memory rather than from the fact store; the remembered number was wrong. Evidence: facts/efficiency.json.
open: a check that flags prose numbers with no fact citation; doc_numbers_check partially covers docs, not chat.

### §46 (2026-09-01, R1)
A self-reported timestamp in a message did not carry its clock source; "UTC hh:mm" was local time, off by 8 hours. Evidence: peer message log.
open: check_timestamps_are_utc covers code-generated timestamps, not self-reported message ones; a message-lint would close it.

### §52 (2026-09-02, R1)
A conclusion was correct but its argument cited a retracted fact; the retraction had not reached the citation. Evidence: facts/data_scaling.json (retracted entry).
open: a check that flags citations to retracted facts; none exists.

### §57 (2026-09-02, R1)
A premise about GPU availability was accepted without reading card_assignment.json; the card was claimed by another run. Evidence: runs/card_assignment.json.
open: check_card_held_without_claim covers the claim, not the premise that a card is free.

### §66 (2026-09-02, R1)
Saw literal `0` in `blocks=0`, concluded "not the config"; `0 or n_sub` made 0 the sentinel for Full. Evidence: probes/t71_depth_lr_rule.py:130, train.py:219, model.py:330.
open: a check that flags falsy literals used as sentinels in config consumers; none exists.

### §106 (2026-09-03, R1)
A premise about checkpoint contents was accepted without reading the checkpoint; the checkpoint held different weights. Evidence: runs/ckpt_*.pt.
open: manual — checkpoint inspection is a human discipline.

### §131 (2026-09-03, R1)
`tail` read a dead process's `SRCFP CHANGED` line as the current result; the line was a snapshot, not a live signal. Evidence: runs/*.log.
open: a check that verifies a log line's process is still alive before quoting it; none exists.

### §139 (2026-09-03, R1)
A source was cited from a draft that had been superseded; the published version said something different. Evidence: docs/standards/state_0904.md.
open: a check that flags citations to superseded drafts; none exists.

### §175 (2026-09-04, R1)
A merge was killed by a 2-minute command timeout after resolution; MERGE_HEAD was gone and the staged tree was main's content — 18 files of peers' work. `git status` says "still merging" only while MERGE_HEAD exists; after that the same staged tree looks like ordinary authored work and a commit records it single-parent under the committer's name. Caught by checking parents, not status; discarded and remerged. Evidence: tilerl session 2026-09-04.
open: a pre-commit hook that refuses a commit whose staged tree is byte-identical to main's tree while the commit would be single-parent (`git write-tree` vs `git rev-parse main^{tree}`); broken world = the scenario replayed in a temp repo. Proposed to de.

## R2. A criterion must express the property asked

### §89 (2026-09-02, R2-a)
A selftest "passed" because the world-build step silently failed and the check ran on an empty population. Evidence: scripts/harness.py.
open: a check that asserts the world-build succeeded before running the check; none exists.

### §35 (2026-09-01, R2-b)
A dynamic acceptance check missed code that stayed; the check only looked at changed lines. Evidence: scripts/harness.py.
open: a check that acceptance checks cover unchanged code where the property applies; none exists.

### §48 (2026-09-01, R2-b)
A one-sample criterion was used where the property needed a distribution; the single sample could not estimate variance. Evidence: docs/lessons/experiments_0904.md.
open: manual — sample-size design is a human discipline.

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

### §180 (2026-09-05, R2-b)
A co-residency refusal check's population was narrower than the rule's quantity. `check_coresident_cache_refusal` (38af3d47) guarded `train._domain_seqs` callers, but the rule's quantity is host bytes off /data00 — any file that `torch.load`s `/data00/tokens_<domain>.pt` by path read 35 GB with no refusal. The check's author's own new probe was that file. In the same commit, `_broken_coresident_bypass` was written as the broken world but never wired to a selftest — a check that cannot fail, in the commit that added the check. Evidence: scripts/harness.py:3752,3943, 38af3d47.
open: a check that the population of a refusal check matches the rule's quantity (all readers of a resource, not callers of one function); none exists.

### §61 (2026-09-02, R2-d)
A substring/word match read a comment mentioning the symbol as evidence the symbol was used. Evidence: scripts/harness.py.
open: a check that text matches exclude comments and strings; none exists.

### §77 (2026-09-02, R2-d)
A needle was found in the check's own comment; the match was self-referential. Evidence: scripts/harness.py.
open: a check that needles are not self-referential; none exists.

### §141 (2026-09-03, R2-d)
A symbol looked like what it named; the name suggested the role but the code did something different. Evidence: scripts/harness.py.
open: a check that symbol names match their behavior; none exists.

### §80 (2026-09-02, R2-e)
A fixture was not trained; the fixture's weights were random, not the product of training. Evidence: scripts/test_*.py.
open: a check that fixtures are trained, not random; none exists.

### §97 (2026-09-03, R2-e)
A selftest fed the middle function, not the entry point; the middle function could not fail the way the entry point did. Evidence: scripts/test_*.py.
open: a check that selftests exercise the entry point; none exists.

### §54 (2026-09-01, R2-f)
A guard's condition was wider than the danger; the guard blocked safe cases and missed the dangerous one. Evidence: scripts/harness.py.
open: a check that guard conditions match the danger surface; none exists.

### §75 (2026-09-02, R2-f)
A missing init produced the wrong error; the guard read the wrong field and reported a misleading cause. Evidence: scripts/harness.py.
open: a check that init failures are caught before field reads; none exists.

### §85 (2026-09-02, R2-f)
A guard read the wrong key and false-triggered; the key it read was always set, so the guard always fired. Evidence: scripts/harness.py.
open: a check that guard keys are the ones the writer sets; none exists.

### §125 (2026-09-03, R2-f)
A check read a pid file that was never written; the empty read was interpreted as "no process," not "no data." Evidence: scripts/harness.py.
open: a check that distinguishes "no data" from "no process"; none exists.

### §73 (2026-09-02, R2-g)
A comment asserted a guarantee no check provided; the comment was read as evidence the guarantee held. Evidence: scripts/harness.py.
open: a check that comments asserting guarantees are backed by checks; none exists.

### §84 (2026-09-02, R2-g)
A wrong metric changed the decision; the metric measured a neighbour property and the decision followed it. Evidence: docs/lessons/experiments_0904.md.
open: manual — metric selection is a human discipline.

### §108 (2026-09-03, R2-g)
An off-by-one did not crash; the check passed but the result was off by one. Evidence: scripts/harness.py.
open: a check that boundary values are tested, not just interior; none exists.

### §112 (2026-09-03, R2-g)
An unchanged arm was not reproduced; the comparison assumed the unchanged arm was the same as last time. Evidence: docs/lessons/experiments_0904.md.
open: a check that unchanged arms are re-run, not assumed; none exists.

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

### §173 (2026-09-04, R2-g)
"NOT KEPT" was read as "CLAIMED"; the absence of a KEEP claim was read as a claim. Evidence: runs/pod_ckpt_candidates_*.txt.
open: a check that distinguishes "not kept" from "claimed"; none exists.

### §174 (2026-09-04, R2-g)
A guard was green by hand and red under automation, and three of us looked in the environment. Cause was TIME: a same-length mutation (3565 -> 3565 bytes) landing in the same wall-clock second as the preceding run reuses a stale `.pyc`, because Python invalidates on (whole-second mtime, size) -- so the interpreter ran the pre-mutation code and the mutation test passed on a defect it never executed. Confirmed by the shape of the failure: 6 replicas gave rc 1,1,1,0,0,1, which is a race, not a configuration. Two of my own diagnoses (a TZ artifact, then a resolving symlink) and one of 6e's (GIT_INDEX_FILE) were all refuted. Evidence: de's world-8 replica run.
open: manual -- a mutation test must change the file's SIZE or force a `.pyc` invalidation; "green by hand, red under automation" is a race's signature and the environment is the wrong place to look first.

## R3. Artifacts carry their producer identity

<!-- The two entries below are tagged R10, not R3: they are about a pod-only artifact being
     unreachable, not about producer identity. They sat under a second "R3. Artifacts carry
     their producer's identity" heading that differed from this one only by an apostrophe.
     Headers merged 2026-09-05; the entries are left where they are because reassigning
     someone else's incident to a rule is the file owner's call, not a passing reader's. -->

### §2 (2026-08-30, R10)
A measurement existed only on the pod; the pod was recycled, and the measurement was lost. Evidence: pod-only artifact.
open: manual — pod-to-repo transport is a human discipline.

### §116 (2026-09-03, R10)
A pod-only artifact was cited in a decision; the artifact was unreachable from the repo, and the decision rested on an unreadable source. Evidence: docs/standards/state_0904.md.
open: a check that cited artifacts are reachable from the repo; none exists.

### §4 (2026-08-30, R3)
An artifact with no producer identity was silently rebuilt; the rebuild used a different producer, and the artifact's meaning changed. Evidence: scripts/harness.py.
open: check_cache_readers_set_vocab_id covers vocab identity; a general producer-identity check would close it.

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

### §20 (2026-08-31, R6)
A number was extrapolated without a label; the extrapolation was read as a measurement. Evidence: docs/lessons/experiments_0904.md.
open: a check that extrapolations are labeled; none exists.

### §50 (2026-09-01, R6)
A number's basis was a draft, not a measurement; the draft was read as a result. Evidence: docs/standards/state_0904.md.
open: a check that draft numbers are labeled; none exists.

### §62 (2026-09-02, R6)
A number's basis was a single sample; the basis was not labeled as n=1. Evidence: docs/lessons/experiments_0904.md.
open: a check that n=1 numbers are labeled; none exists.

### §79 (2026-09-02, R6)
A number's basis was a different path; the path was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their path; none exists.

### §86 (2026-09-02, R6)
A number's basis was a different population; the population was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their population; none exists.

### §111 (2026-09-03, R6)
A number's basis was a different step; the step was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their step; none exists.

### §127 (2026-09-03, R6)
A number's basis was a different window; the window was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their window; none exists.

### §143 (2026-09-03, R6)
A number's basis was a different sign convention; the convention was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their sign convention; none exists.

### §156 (2026-09-03, R6)
A number's basis was a different normalization; the normalization was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their normalization; none exists.

### §157 (2026-09-03, R6)
A number's basis was a different unit; the unit was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their unit; none exists.

### §159 (2026-09-03, R6)
A number's basis was a different confidence level; the level was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their confidence level; none exists.

### §172 (2026-09-04, R6)
A number's basis was a different timezone; the timezone was not named. Evidence: docs/lessons/experiments_0904.md.
open: a check that numbers name their timezone; none exists.

## R7. Retractions travel as wide as the ruling

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

### §194 (2026-09-05, R8)
A claim held by a live pid was read as evidence the job was progressing. The m1 decomposition cell hung after writing its row; `card_claim status` named `tilerl_mem_decomp_m1` correctly for 17 minutes while rank 1 held 76 GiB at 0% util. A claim answers who intends to hold a card and carries nothing about whether work is happening — the two questions have different evidence, and only 0% util against a large reservation showed the hang. Evidence: runs/mem_decomp_0905.log, commit acfb67bb.
open: a check that flags a claimed card at ~0% util for N minutes; none exists. `card_claim status` reports claim-vs-memory disagreements and would report this one as agreeing.

### §195 (2026-09-05, R8)
A rank-0-only phase inside a world-2 job desynchronised the ranks: profile_step_cost times save (33.6 s here) and val after the loop, save runs on rank 0 alone, and rank 1 entered the next collective with nothing to meet. The cells' timings were already complete and correct when it hung, so the failure cost card time and no data. Fixed by `--skip-save-val`, which skips both and still writes the JSON row (`--peak-only` skips them but returns before the record is built). Evidence: scripts/profile_step_cost.py, commit acfb67bb.
open: manual — nothing checks that a multi-rank script's post-loop phases are collective or rank-symmetric.

### §214 (2026-09-05, R8)
A live job ran on a shared card with no claim, and every reader read it as an orphan. tilerl-25's re-score held card 0 live; `card_claim status`, the sweep, and nvidia-smi all showed unclaimed memory, which reads identically to a dead process's residue for as long as it takes to ask the owner — and that interval is exactly the window a sweep kills in. The claim-write is the only thing that separates "orphan" from "unclaimed live job", so the defect is the launch that never wrote a claim, not the readings. check_card_held_without_claim WARNs after the fact; nothing refuses the launch. Evidence: tilerl report 2026-09-05, runs/card_assignment.json.
open: a launch path that does not write a claim refuses before it starts; none exists.

## R9. Run a deletion candidate before judging it

### §39 (2026-09-01, R9)
A deletion candidate was judged without running it; the candidate was a live process, not a stale file. Evidence: runs/pod_ckpt_candidates_*.txt.
open: ckpt_facts_sources_present covers checkpoints; a general liveness check would close it.

