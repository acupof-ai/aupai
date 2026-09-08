---
question: What are the rules that keep gates and measurements honest, what enforces each, and what does each cost?
status: open
source: derived from docs/lessons/gate_failure_incidents.md (133 model-project incidents) and docs/lessons/infra_incidents.md (88 pod/infra incidents); 33 closed incidents removed 2026-09-04 (223 = 133 + 88 + 2); 33/33 confirmed machine-gated (list below)
---

# Gate failure rules

Eleven rules, ranked by incidents × cost (rough hours lost per incident). Each rule: statement, count, cost, the check that enforces it (or `manual:` with the reason), one or two canonical incidents, and what the check cannot see. Incidents live in `gate_failure_incidents.md` (model-project) and `infra_incidents.md` (pod/infra). The top 5 by product are checks to write; the rest are rules people read.

Cost is an estimate: R2 (criterion) ~4h/incident (wrong measurements, false greens, some cost days); R1/R3/R4/R10 ~3-4h; R5/R6/R7/R8/R11 ~2h; R9 ~1h.

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

- **R2** (120 incidents, 336h): a criterion must express the property asked; test it on known-answer positive and negative worlds. Split into 7 sub-rules below; each sub-rule is a check target. Owner: blank.
- **R6** (34 incidents, 68h): every number carries its basis. Owner: blank.
- **R1** (21 incidents, 63h): verify premises before acting, sources before citing. Owner: blank.
- **R5** (11 incidents, 22h): state the vision before the number. Owner: blank.
- **R4** (14 incidents, 39h): failures must be loud. Owner: blank.

## R2. A criterion must express the property asked; test it on known-answer positive and negative worlds before trusting output

120 incidents (34 infra, 86 model), ~4h each, 336h. `manual:` no check verifies that a criterion expresses the property asked; `--selftest` requires every CHECKS entry to carry `broken()`, but a selftest that passes on a broken world is invisible to the contract.

Seven mechanism sub-rules. Each is a check target.

### R2-a No broken world (24 incidents)

A check that was never made to fail is decoration; the broken world must be asserted, not assumed.

- §89: a selftest "passed" because the world-build step silently failed and the check ran on an empty population.
- §103: a check that cannot fail — its acceptance condition was tautological.
- §218: a fixture sampling "a recent commit" drew a merge commit; `git show --name-only` on a merge prints no files, so the world had no subject and read as broken code.
- §258: a module extracted so it could be tested got seven worlds, all calling its functions in-process; nothing exec'd its `__main__`, so exit-code and cwd-root mutants — including one that accepts every commit — went 7/7 green.
- §219: the next filter over the same sample rejected the only file the first non-merge commit touched (.jsonl not in the extension list); both were caught only because the world FAILs rather than skips when it cannot find its subject.
- §228: a world copied from real files was too incomplete for the subject to import (only scripts/ on sys.path; `from train import ...` died at module scope); the subject's ModuleNotFoundError read as the subject being broken. An import error in a fixture's verdict is a fixture bug until proven otherwise.
- §231: agreement between two things that share an error is not evidence — a fixture with no power to disagree (one directory, so cwd and $MAIN are the same path) reported agreement and it read as confirmation; a differential fixture must be fed an input where the two sides are known to differ.
- §235: a fixture that copies a module to a temp dir relocates every path it derives from `__file__`; ROOT pointed at a .git-less temp dir, `is_pod(ROOT)` read True, and the branch under test was dead in every mutant while every assertion still ran — a passing mutation run in the wrong world; the discipline is one world-validity assertion before the mutants run (the relocated root must satisfy the property the code keys on).
- §238: `want == pool_rows` passed on five arms and two reviews because both sides were derived from one token count that missed the per-document `<eos>`; the undercount cancelled and the comparison had no way to be unequal. A comparison whose two sides share a producer cannot fail — one side must be replaced by a reading from a different path (the system under test's own output). The cheap arms (20/25 rows) were structurally incapable of showing the defect, and the typed constants had been wrong in all three prior versions.
- §239: a guard was disarmed by the change it protected — the integration-tree refusal tested `branch == "main"` from `rev-parse --abbrev-ref HEAD`, which returns "HEAD" when detached; the flip detached the tree deliberately and the guard went inert at that moment (orphan 8a9dc8a0, a merge that exited 0 and never moved main). The selftest never built a detached integration tree, so test and guard shared the branch-name assumption and their agreement was not evidence (§231). Fixed by a structural predicate (main worktree AND has linked worktrees).
- §240: a fixture that cannot be the thing it tests — the hook's own selftest worlds 1-4 were a bare `git init` standalone repo (no linked worktrees, no scripts/), and world 1 asserted "commit on main is REFUSED" against a repo that was not an integration tree by any structural definition; it passed for five days under the branch-name predicate because that predicate did not care what the tree was. Fixed by building the property (`git worktree add` a sibling, copy the real module in). The mirror image: de's W5 asserted "detached does NOT refuse", correct under the old predicate and exactly backwards under the new one.
- §242: a mutation run can be vacuous end to end — the runner built worlds with `git init` in an empty temp dir, where `git ls-files data runs scripts` is empty, so the selftest SKIPped and exited 0 for every mutant; four "survivors" measuring nothing, reported as a clean run. A SKIP exit 0 and a PASS exit 0 are indistinguishable from outside. Fixed by building worlds as a real worktree of the real repo; the discipline is one world-validity assertion before the mutants run (same as §235).
- §243: a selftest world that passes under both the old and new predicate cannot see its own subject — the behind-main exemption widened from merge=union to any named driver, and the only covering world (5b) stages a merge=union ledger, so it passed under both; fixed by world 5d (named driver, staged alone, from behind main), mutation-verified (reverting the predicate fails 5d by name).
- §244: a single-writer allocator in a multi-writer tree — `harness task add` computes max+1 over the rows it can see, so two sessions independently get the same next-free id; three collisions in one hour, each caught by `tasks_well_formed` at merge time (after both rows were written). The collision check is the backstop, not the allocator.
- §247: a byte-diff over an append-only ledger reported 141 orphans and every one was a superseded row — a temp worktree pinned to an old commit, so byte-equality asked "is this exact historical line still present" instead of "does this row exist"; by the identity its own writer uses ((name, started), (ckpt, type, measured)) it was 0 of 289 and 1 of 58, and that 1 was re-measured a day later. A criterion that reports the whole history as missing cannot tell a lost row from an old one, which is the failure it exists to detect.
- §248: a broken world that SKIPs has proven nothing — `_tmp_repo` makes a directory, not a git repo, so every git call failed silently, no reflog existed, the check SKIPped, and the broken world and a clean control returned the IDENTICAL SKIP string. SKIP is the shape a correct check produces on a machine that legitimately cannot answer, so a SKIP from a broken fixture is indistinguishable from a real absence. `_tmp_repo`'s name is the trap.
- §250: a refusal placed where its condition is unreachable proves nothing — the allocator's refuse-on-collision candidate is max+1 over the very set it tests against, so it is free by construction; a guard-shaped piece of dead code that would have read as protection in review. Same family as §248: a check whose green is structurally guaranteed, found by asking "what would have to be true for this to fire?"
- §252: five defects in one 40-line function, each found only after the previous fix was believed to be the last, all one shape -- the rule keyed on a property ADJACENT to the one that matters. The staged-index carry: trigger on fast-forward-vs-true-merge instead of "does this merge touch a staged path"; index cleared but not the working tree; a new path not resettable from HEAD; restore by overwrite, which for a merge=union ledger is a deletion (it dropped e1-48/e1-49) and for source discards whatever main changed (517 lines of scripts/harness.py, e1's 9a382298 among them). A measurement can be correct and still be generalised past its own conditions, which no check can see. Nine worlds, four controls, each mutant dying on exactly the world that names it.
- §253: a guard tested only where it does not run is untested — `is_mounted` in scripts/pod_backup.sh was green on the laptop and fatal on the pod in OPPOSITE directions: `mountpoint(1)` (absent on macOS) refused every path including `/`, so the negative case passed vacuously; `stat -Lf` (a BSD format that SUCCEEDS on GNU with a filesystem dump) accepted every path, so the backup would have written a GREEN root_durable MANIFEST onto the emptyDir it exists to protect against. The dangerous failure is the one that survives local testing. A negative control alone is satisfied by a guard that refuses everything; the positive control (`/` must pass) is what went red. A `cmd_a || cmd_b` fallback between platforms assumes cmd_a FAILS on platform B; when it succeeds with different semantics, the fallback is dead code and the comparison is garbage-vs-garbage. Fixed by one portable python3 implementation + the positive control + running the selftest on the pod.

Ledger-field semantics (test_ledger_field_writers.py, 315755cc): class/cards ABSENT means unstated and "" is forbidden (indistinguishable from a pre-field row; 243 historical rows stay null, no backfill); 'none' is a STATED cards answer for a CPU or corpus job. defect_caught "" is a REAL clean-review answer; absent means no review reported.

Cannot see: whether the selftest's broken world actually exercises the check's logic (§31, §69, §137, §153, §206, §218, §219, §228, §231, §235, §238, §239, §240, §242, §243).

### R2-b Population narrower than the property (32 incidents)

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
- §236: a check's deepest assertions were dead in every environment — main() never passed the cursor parameter, so cur was {} and both cursor assertions skipped; the off-pod review recorded "plan level SKIPPED, designed behavior", which was correct AND the concealment. A check whose deepest assertion runs in one environment only cannot be reviewed in the others; argument-level refusals must fire before environment gates so the depth is reachable in the review environment.
- §237: a rank-written diagnostic ledger held 2 rows/step (missing `if is_main` guard), and the steps that logged "write FAILED" were the only CLEAN ones — rank 1's crash prevented the duplicate write, so the failure log inverted data quality and a reader averaging it double-weighted 5 of 7 windows. The report also read steps 10-100 and concluded "healthy" while the ledger ran to 400 with entropy already 0.99→0.89: point values against a threshold cannot see direction. A "FAILED" line is never grounds to exclude a step without checking the row count, and a verdict is scoped to the steps read.
- §249: an allocator that reads max+1 over "what I can see" allocates against a partial population — `harness task add` computed the next id from this tree's register, so a peer's committed-but-unmerged allocation was invisible and two sessions got the same id; four collisions plus a phantom id that named nothing for an hour. Fixed (d5ab3d31) by scanning refs/* in the common git dir. The lesson generalizes: any max+1 over a partial population in a shared checkout allocates against peers it cannot see.
- §259: a count added to satisfy "report your population" was a count of HITS, and for a check whose pass condition IS zero hits the two are opposites — `dirty_aged` printed "0 dirty tracked file(s)", all-zero on a clean tree, so T0-2 shipped with the defect it exists to name in one of the eight checks it changed. Population is what was EXAMINED; the hit count is the finding, and its good value is 0.
- §260: a lock file's timestamp and `ps`'s start time live in different timezones — the lock prints UTC (Z), ps prints local. Comparing them by eye is a trap; `etime` answers "how long has this been running" and the lock stamp answers "when was this file written." Reading `01:00` as 1 hour and `00:45` as 45 minutes when etime format is `[[DD-]hh:]mm:ss` produced a false "stuck lock" report; both were 1 minute and 45 seconds, and the lock's UTC stamp and ps's local start agreed within 29 seconds. Read the holder's `since=` and subtract from current UTC before reporting a hang.
- §261: a predicate too narrow to see its own population makes the premise it tests unfalsifiable. A writer regex `torch.save([^)]*\bcache\b` catches `torch.save(data, cache)` but misses `torch.save(dict(a=1), cache)` (the `[^)]*` cannot cross the inner `)`), `torch.save(data, cache_path)` (the trailing `\b` fails on the underscore), and `torch.save(obj, _cache_for(name))` — three ordinary shapes, each defeating the check silently. Green over an unfalsifiable premise is indistinguishable from green over a cleared one; the fix is to match the whole logical line or co-occurrence, not to parse an argument list with a regex.
- §262: a regex-hit is not an anchor. A sha-finder matching loose patterns offers ten candidate shas for one fact; a criterion accepting ten answers identifies none, and a wrong anchor is permanently green. The anchor must be the sha that produced the artifact, verified by content, not by pattern-match.
- §266: a survey for one defect returned five real sites and the four legitimate ones are defended by four DIFFERENT mechanisms, so the greppable shape (`os.kill(pid,0)`, `/proc/<pid>`) is not the criterion — a check on it would be four-fifths false positives. The one finding was a false JUSTIFICATION beside correct code: `sweep.py:167` re-tests `isdir(/proc/<pid>)` because "a dead holder's fd can still appear in the walk", and a zombie measured on the pod has `isdir` true with `/proc/<pid>/fd` at 0 entries — the branch is unreachable for the case its comment names. Survey a defect class by reading each hit's defense, not by counting hits.

Cannot see: whether the test's inputs, environment, or scale match the property's (§26, §29, §34, §35, §40, §48, §65, §72, §121, §146, §151, §169, §180, §201, §202, §203, §209, §213, §215, §216, §222, §223, §224, §225, §229, §232, §236, §237).

### R2-c Mutation did not take (10 incidents)

The mutation never landed or its verification reads the wrong signal.

- §90: a mutation was applied but never landed in the running process; the check "passed" because the world was never mutated.
- §132: the mutation test itself was broken — it mutated a copy, not the live object.
- §221: a test recomputed the quantity outside the function and asserted its own arithmetic — the function under test was never called, so no mutant of it can reach the test.
- §227: a refusal raises SystemExit (BaseException, not Exception); the test's `except Exception` let it through, so the mutation was caught by the process dying, not by the assertion.
- §233: three selftest cases passed with the new rule deleted entirely — the fixture routed around it into a pre-existing clause giving the same answer; a case must include a shape where the OLD logic answers differently, enforced by an in-case assertion naming the old answer (a disagreement property, not a coverage property).
- §234: three mutants were all caught at the SAME assertion with the same message, so the run proved one assertion and exercised none of the others; a mutation run proves N assertions only if the N mutants fail at N DIFFERENT, target-naming assertions — the vacuous-PASS shape moved into the thing that validates the test.
- §241: a test that reimplements its subject tests the copy — the launch_gate selftest re-derived gate 9's partition inline, so mutating the gate's own exclusion changed nothing the test could see; 3 of 4 mutants survived, including "exclude nothing", which is the original bug. Fixed by extracting one function both sides call. Two invalid-mutant lessons from the same run: a mutant that dies of NameError is caught by the interpreter, not the test; and `x = "" or (f"...")` is a mutant identical to the original, whose survival measures nothing.
- §270: a surviving mutant means the mutated clause is dead OR the check is weak, and the two demand opposite fixes. `build_dd09_full`'s new file-set assertion compared `(name, st_ino)` pairs and its selftest summary printed "file set asserted equal to the plan BY INODE"; a mutant zeroing the inodes survived, because the link loop above already refuses any planned name whose `st_ino` differs from its source, so every planned name is inode-verified and present before the comparison runs — `want <= got` always holds, which makes `missing` dead and cardinality equivalent to membership. A second mutant (`len(got) != len(want)`) survived for the same reason. Neither is a hole to plug: the strength was in the loop, and the summary line was claiming a property the comparison could not have. The first mutant of the pair died on the WRONG assertion — zeroing the inodes collapsed `set(inos)` so the duplicate-inode check fired instead, and a red selftest read as the mutation working. A mutation that kills a different assertion than the one it targets proves nothing; read which line raised. The distinct-inode check then had no world at all until one was constructed (two sources already hardlinked to each other: every name present, every link verified, file set == plan, one file counted twice), and its control has to assert the token gate CANNOT catch it — 3 docs over 2 distinct inodes — or the world does not establish that the check is load-bearing.
- §272: a property whose POPULATION is computed with the predicate the defect breaks cannot see the defect. b0-32's lend-expiry property quantified over cards with a PARSEABLE window (`if _parse_lend_window(n) is not None`), so all three unparseable-window defects -- 25:99Z, a backwards window, the timestamps deleted -- removed the card from the set the loop iterates, and 3 of 4 planted defects PASSED with no assertion evaluated. Population from the CLAIM (`_mentions_lend`), assertion from the PROOF. Second instance on one function in two days: §266's `ours=[0..7]`/`theirs=[]` satisfies "every not-ours card refuses" by emptying `theirs`, the same disease at the other end -- the signature is a comprehension whose filter and whose assertion call the same function. Second half: a mutant that does not mutate reads exactly like a surviving defect. Disabling the property by rewriting `_claimed = {...}` as `{} or {...}` changed nothing (a non-empty dict is truthy), and the resulting RED read as the guard failing -- mutate the ITERATION, not the expression feeding it, and assert the mutant's behaviour changed before believing its verdict. The three-way paired prediction is what caught the vacuity: green with the property removed AND green with it present means the pair disagrees with itself; a pass count alone read 8/12 as "mostly working".
- §273: a new refusal makes a NEGATIVE assertion pass by firing first, so the fixture asserting a refusal is the one that goes silently green. `_refuse_committing_on_main` (PR #55) refuses a commit whose HEAD is the branch `main`; three of `harness.py`'s `_demo` fixtures init on main and install the real hook. The one asserting the hook PASSES broke main's CI within the hour; the one asserting a merge is REFUSED kept passing, while the `data/` allow-list it exists to test had stopped being the cause of anything it observes. A positive assertion breaks loudly when a gate fires early and a negative assertion absorbs it, so after adding a refusal the fixtures to audit are the ones that cannot fail. `rc != 0` collapses "the gate under test fired", "another gate fired" and "the process died first"; assert the refusal's own string. The predicate for the sweep is "installs the REAL hook", not "inits on main" — one fixture inits on main deliberately for `--with-tree=main` and installs no hook, and two more write two-line stubs that reach no gate. No check landed: three attempts each failed differently (function-level escape test let one `-b fixture` excuse a 900-line function, per-function hook attribution flagged the legitimate main fixture, per-variable attribution gave 26 hits across 20 hookless functions while the real subject disappeared), and a scan that misses the instance that matters certifies rather than checks. For a hook change, "main is green" and "your commits work" are independent in both directions: the installed hook is a symlink into the integration tree, so the file you run and the file you edited are different.
- §274: a test whose verdict depends on two inputs is deterministic only if BOTH are fixed, and pinning ONE is worse than pinning neither -- it passes for hours, then fails for a reason that reads as a code defect. `_selftest_card_lend_expires` pinned the clock (`21:33Z` inside, `21:00Z` before) and read the note from the live `card_assignment.json`; the controller wrote a second lend at `00:30-00:45Z` and main's CI went red for four merges, blocking every session's ledger merge, over a correct note edit. Derive the second input from the first -- midpoint inside by construction, a day either side outside -- rather than freezing a fixture, which only moves the reword problem one level up. Fixing it exposed two more instances in the same function, NEITHER in the CI log: an assertion requiring "theirs at every clock" was really a property of the note's opening token and would have failed on CORRECT code once the note opened with `GRANTED`; and a `note.replace(stale_literal, bad)` mutation silently returned the note UNCHANGED, so the "unparseable window" world was the live valid note passing correctly -- §272's truthy-`or` reached through a stale literal. Three defects, one shape: a property stated over text another session owns. When a test reads a live artifact, every other input must be a FUNCTION of what it read, never a constant that agreed with it once.
- §268: a shared broken world already reporting 42 errors cannot test a fifth predicate added to it — the verdict is FAIL with or without the new mutation, and only five of the 42 reach the evidence string, so all four mutants survived. The clean tree failed the same way for the opposite reason: its instances of the new class are registered debt. A dedicated one-fact world per predicate separated them, each mutant redding exactly one. Two further green-on-the-wrong-branch defects inside that world: a `path@rev` case must cite a path resolvable ONLY at that rev, or the earlier openable-beside clause answers first and the branch under test is never reached; and a `git init` world has an empty object store so no rev resolves at all — it needs `objects/info/alternates` on the real store. Before adding a mutation to an existing world, ask what that world's verdict and visible evidence are without it.

Cannot see: whether the mutation reached the code path the check exercises (§81, §207, §221, §227, §233, §234, §241, §268); how many errors a world already reports before a new mutation joins it (§268); whether a property's own population is emptied by the defect it is written for, or whether a mutant actually mutated (§272); whether a surviving mutant means a weak check or a dead clause, or whether a dying one died on the assertion it targeted (§270); whether a fixture passes on its own subject or because a newer gate fired before it (§273); whether a pinned constant still agrees with the live artifact the same test reads (§274).

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

### R2-g Criterion answers an adjacent question (33 incidents)

The metric measures a neighbour property, not the one asked.

- §110: a pre-registered branch collapsed two worlds into one; the criterion (branch taken) did not isolate the property (which world).
- §170: an unresolvable fact reference was used for four days; the criterion (reference present) did not measure the property (reference resolves).
- §177: an arm's flags said it carried a 1.07B-parameter memory table; the criterion (the flags the run was given) did not measure the property (which of two code paths consumed them), and the arm would have trained as the control and reported a clean null.
- §184: excluding a parameter from the fp32 master copy would have left it read every forward and never updated; the criterion (is the exclusion correct) did not measure the property (who clears its gradient), and the diagnostics would have shown a healthy pool.
- §254: a signature-based check's criterion (signed/unsigned) measures a neighbour property (written by a new-tool copy) to the one asked (bypassed the CAS); in a shared checkout every session runs its own copy, so the rollout window during which legitimate writers are unsigned closes per-session, not at a timestamp, and a date-based cutoff WARNs on innocent writes.
- §257: a warn_only broken world's criterion is `not PASS` not `== FAIL`; if the real tree is already WARN, with-seed/without-seed/real all WARN and the world tests nothing. The fix's seed-id-in-evidence assertion was itself insufficient — a check that finds the row by accident while miscounting everything else passes it; the discriminating criterion is differential: read the same world twice (with seed / with only that line deleted), require count difference exactly 1. Same world, not two builds — two builds can differ for unrelated reasons, which would make the comparison itself unreliable.

Cannot see: whether the metric's null hypothesis is the property's null hypothesis (§9, §10, §23, §45, §67, §73, §84, §91, §108, §112, §114, §135, §140, §142, §147, §148, §149, §150, §158, §165, §173, §174, §176, §177, §178, §184, §191, §208).

## R6. Every number carries its basis: source type, resolution, algorithm; label extrapolation

35 incidents (12 infra, 23 model), ~2h each, 68h. `manual:` basis-labeling is a discipline; `doc_numbers_check` partially verifies that docs numbers trace to facts, but does not verify the basis label is correct.

- §1: a number was quoted without its source type; the source type (measured / extrapolated / inferred) determined whether the number could be compared to another.
- §55: a number's resolution was finer than its basis; the extra digits were noise, not precision.
- §185: a memory budget was costed at 6 bytes per parameter from a bf16 table nobody had set; the tensors are fp32 and the gradient was omitted, so the real figure is 12 and the 2048^2 arm OOMed after construction succeeded.
- §230: a review reported five checks as MEASURED that had only been READ; the figure then acquired a second independent-looking source when repeated back, with zero executions. A stated basis is itself a claim — ask "when did this command run" of your own claim. Second instance the same day: a derived ratio carried across a rebuild of its inputs, so the digits in the decision document matched neither the old quantity nor the new one.

- §269: a summary grouped by SIZE cannot fail on a difference of KIND. Five recount deltas reported as "small, both directions, consistent with sampling noise"; four were, and the fifth was a definitional error whose +1,029,505 equalled the domain's document count to the unit — an omitted `<eos>` — while its own `tokens_config` claimed a full pass with no extrapolation. +0.159% sits unremarkably among -0.106%, -0.091%, +0.234%, +0.481%, which is the axis the grouping chose. What would have caught it is a per-row predicate the grouping discards: `delta == docs` is true or false of one row and no summary of five can express it. Shares one line with §268 — an aggregate is CHOSEN, so it cannot report what the choice discarded, and nothing in the artifact records that anything was.

Cannot see: whether the basis a number carries is the basis it was produced with (§11, §12, §20, §21, §50, §62, §63, §64, §79, §86, §99, §104, §105, §109, §111, §115, §117, §118, §124, §127, §133, §143, §152, §155, §156, §157, §159, §161, §164, §172, §185, §192, §269); whether a grouped report's members share the KIND its grouping implies (§269).

## R1. Verify premises before acting, sources before citing; a correct conclusion does not certify its argument

24 incidents (14 infra, 10 model), ~3h each, 72h. `manual:` no check can verify that a human's premise matches the world; `check_fact_refs` (citations resolve) and `ckpt_facts_sources_present` (fact sources exist) cover the citation, not the argument.

- §66: saw literal `0` in `blocks=0`, concluded "not the config"; `0 or n_sub` made 0 the sentinel for Full. Read the default def and the consumer line, not the literal.
- §131: `tail` read a dead process's `SRCFP CHANGED` line as the current result. Read the artifact, not the log tail.
- §246: a review row's basis named a sha that later stopped existing (the branch was rebuilt to drop a live-key commit) and stayed auditable only because the reviewer had happened to record a BLOB hash. A basis that is a sha describes something that can be rewritten or garbage-collected; a basis that is content survives its own subject. Same shape as §247 one rule down — the identity of a thing is not its bytes, and here the bytes are the identity that lasts.
- §263: a kill is an input to the parent, not an operation on the child. Two throttled `curl` chunks were killed by exact PID — right PIDs, right intent — and the launcher's `wait "$p" || ok=0` turned that into its whole-file failure path, `rm -f "$out".c*`, which deleted the six chunks that had already completed. Every signal an operator reads before a kill (whose process, what it holds, is it the right one) was read correctly and none of them names the consequence; the parent's response to a non-zero child is the only thing that does. Read the failure branch of whatever launched the process, or kill nothing.

- §271: `gh pr view --json headRefOid` is not a statement about the branch, and its answer is well-formed. PR #47 merged on the API's `d6c4a7c1` while `git ls-remote` reported `d3ef01e3`, so the merge dropped the newest commit with nothing failing. Same shape as `.conclusion // .status` and `[ -d /proc/<pid> ]`: the instrument answers a question adjacent to the one asked, and the field is named exactly what the reader wants. Read a branch's current sha from `git ls-remote origin refs/heads/<branch>`; verify after a merge by grepping main for one symbol that must be PRESENT and one that must be ABSENT, because a sha comparison says "different" while the symbol pair says which half is gone. Second half, §267 again in the same hour: a grep for a removed flag matched the fixed docstring's own sentence recording its removal, so four corrected files read as still rotted -- verify a fix with the checker, never with a grep for the string the fix deleted. Third half, one grep and three wrong readings: "dedup_corpus.py has NO CONSUMER" was reported as measured, relayed by me into a fact's basis, and refuted by 44 naming harness.py's _run_pipeline_step("dedup", "dedup_corpus.py", ...). My account of the miss was wrong too -- the grep searched the ARTIFACT names (data/dedup, dedup_manifest, dedup_stats) with *.py included, and the consumer names none of them because it invokes the SCRIPT. "Who reads these files" and "who runs this thing" are different questions; a pipeline step names the producer, never its outputs. Cite the symbol, not the line: that call site was read at :22336, :22385 and :22560 on one day. Fourth half, and the instrument was innocent: PR #48 lost a commit while both people followed the rule above -- the reviewer merged the tip ls-remote reported, and the author pushed a correction during the review. ls-remote answers "what is the tip", never "has the tip moved since the reviewer opened it", so the author owes the other half: after requesting a review, stop pushing or say the head moved.
Cannot see: whether a true statement is being used to support an untested conclusion (§8, §14, §18, §37, §38, §46, §49, §52, §57, §70, §96, §106, §131, §139, §175, §179, §190, §198, §199, §211, §246); and nothing compares a merge's sha against `ls-remote`, because the window is between push and merge and only the merger is in it (§271).

## R5. State the vision before the number; outside it, label unmeasured, not absent

11 incidents (10 infra, 1 model), ~2h each, 22h. `manual:` vision-scope is a design property; no check verifies that a number's population is stated before the number is reported.

- §3: a number was reported without its population; the population (which items, which scale, which seed) was the unmeasured quantity that determined the number's meaning.
- §100: a measurement outside the stated vision was reported as "absent"; the correct label was "unmeasured."

Cannot see: whether a number's population matches the vision it is reported under (§5, §6, §17, §19, §28, §30, §32, §36, §53).

## R4. Failures must be loud: checks before the write, raise or exit nonzero, never print-and-continue

15 incidents (9 infra, 5 model), ~3h each, 42h. `manual:` loud-failure is a code-review property; some selftests assert exit codes, but no general check verifies that a failure path raises rather than prints.

- §13: a world-build step silently failed; the check ran on an empty population and passed. A silent failure is indistinguishable from success.
- §51: an observation channel swallowed the signal; the check read the channel's default, not the observation.
- §251: a verification command read exit 0 from a program that exited 1, because it piped into `tail` — `$?` is the last stage's status. `set -e` does not catch it, so a script that looks defended is not. Tracked population zero: 39 of 58 `.sh` set `pipefail`, and the failing form was typed at a terminal, where no scan reaches it.
- §256: a broken world was red for a reason other than its mutation — built on a non-git directory, so it failed on absent git and stayed red with its planted row deleted. The selftest counts such a world as coverage while the check it guards is never exercised. `156 of 156` was the second tell: failures equal to the total is an absent comparison side, not N defects.
- §265: a GREEN pull request turned the base RED, and every other open PR then failed on a defect none of them contained. PR CI runs on the merge of the PR into the base as it stands at that moment; nothing re-runs the base's own selftest against the base afterwards. PR #7 was green when merged and carried a `globals()["ROOT"]` patch into main, where the core-reexport guard flags that pattern by name; the fix was on a separate branch, because the guard first fired on CI for a LATER PR. #4 and #5 each burned two rounds on a message naming a file they do not touch, so the reader's first hypothesis is their own change. Until a post-merge job reads the base: a PR failing on code it does not touch is a base failure until proven otherwise, and merging main into it is the test.

Cannot see: whether a print-and-continue path exists in code not covered by a selftest (§7, §25, §59, §136, §166, §181, §188, §193, §197, §204); whether a loud failure was READ correctly by the command that checked for it (§251); whether a broken world is red for its own mutation or for something else (§256) — checkable by running each `_broken_*` twice, priced out at 104 worlds.

## R7. Retractions travel as wide as the ruling and name the todos they void; constraints are machine checks, not prose

6 incidents (5 infra, 1 model), ~2h each, 12h. `check_frozen_paths` (registered CHECKS entry) enforces frozen-path constraints; partial: covers frozen paths, not retraction width.

- §16: a retraction was narrower than the ruling; the ruling voided a todo that the retraction did not name, and the todo was later acted on.
- §58: a constraint was stated in prose, not enforced by a machine check; the constraint was violated without a signal.

Cannot see: whether a retraction reached every consumer of the original ruling (§22, §68, §102, §119).

## R3. Artifacts carry their producer's identity; missing identity refuses, never rebuilds

6 incidents (2 infra, 4 model), ~4h each, 24h. `check_cache_readers_set_vocab_id` (registered CHECKS entry) enforces vocab identity on cache readers; `train.py:1472` raises if `VOCAB_ID` is unset. Partial: covers vocab, not all producer identity.

- §4: an artifact with no producer identity was silently rebuilt; the rebuild used a different producer, and the artifact's meaning changed. Missing identity must refuse, not rebuild.
- §24: a checkpoint with no recipe provenance was scored; the score was attributed to a recipe the checkpoint did not run.
- §189: a close written on an already-CLOSED row stamped a row with its own write time and minted a third identity for a run that never existed; the ledger's fold key is (name, started), so the verdict and the numbers now sit on a phantom row. CLOSED by ac2df315: `done` refuses when the name has a closed row and no open one. The variable is the row being closed, not `--started` being absent, and the outcome forks on the clock -- `now()` is minute resolution, so a close in the same minute collides with the real row's `started` and folds onto it instead of minting anything (and if that row is retracted, `fold()` discards the event: success printed, nothing changed). 29 of 288 folded rows carry the shape.

Cannot see: whether the identity a checkpoint carries is the identity it ran with (§44, §189, §210).

## R10. What happened only on the pod did not happen; bring it back to the repo the same day

2 incidents (2 infra, 0 model), ~4h each, 8h. `manual:` pod-vs-repo is a discipline; no check verifies that a pod-only measurement was brought back to the repo the same day.

- §2: a measurement existed only on the pod; the pod was recycled, and the measurement was lost. What happened only on the pod did not happen.
- §116: a pod-only artifact was cited in a decision; the artifact was unreachable from the repo, and the decision rested on an unreadable source.

Cannot see: whether a pod-only measurement was brought back before the pod was recycled.

## R8. Shared resources are explicitly exclusive; co-residency is judged by each implementation's measured cost in seconds against the run's own spend, never by metric class

7 incidents (6 infra, 1 model), ~2h each (infra), ~1h (model), 13h. `check_card_held_without_claim` + `check_free_card` (registered CHECKS entries) enforce card exclusivity; partial: covers cards, not all shared resources, and WARNs after the launch rather than refusing it.

- §15: a shared resource was used without an explicit claim; the co-residency cost was measured against a metric class, not the run's own spend.
- §126: a resource's exclusivity was inferred from "0 MiB" in nvidia-smi; idle is not a grant.
- §194: a claim held by a live pid was read as evidence the job was progressing; 0% util against 76 GiB held was the signal, the claim status was not.
- §195: a rank-0-only phase (save, 33.6 s) inside a world-2 job desynchronised the ranks; rank 1 entered the next collective with nothing to meet.
- §214: a live job ran unclaimed on card 0 and every reader read it as an orphan; the claim-write is the only thing separating "orphan" from "unclaimed live job", so the unclaimed launch was the defect, not the reading.
- §245: a manual `git update-ref refs/heads/main` (no expected-old-value) bypassed merge_main.sh's CAS and silently overwrote a landed commit; the CAS would have refused. Ruling: nothing but merge_main.sh writes main; a refused push is fixed on the branch and re-merged.
- §264: two readers of one collection in the same function applied different exclusions, and the omitted one waived a FOREIGN claim. `card_claim.acquire` excludes the asker's own claim file from the clash set at :833 (with a comment naming why) and not from the ancestry-exemption candidates at :866; harness launch's PENDING row names the WRAPPER pid and the acquired pid is its child, so the row supplied the ancestor for its own ask unconditionally -- no quiet case. The 30B launch took card 6 while tilerl-gdnfloor held it, acquire returned True and that branch writes nothing, so `runs/claims/` was empty while six ranks ran. Exclude by FILE, not by name: by-name reintroduces §58 (a descendant asking for an overlapping card set under the parent's name IS the same job). The unlink-before-acquire in harness launch was a second defect and the MASK: fixing it alone turns a loud six-ORPHAN symptom into a silent True.

Cannot see: whether a non-card shared resource (disk, network, host DRAM) is co-resident with a run it degrades; whether a launch that never wrote a claim is refused before it starts (§214).

## R9. Run a deletion candidate before judging it; broadcast the list, delete after 24h unclaimed

3 incidents (2 infra, 1 model), ~1h each, 3h. `ckpt_facts_sources_present` + `check_keep_claim_reasons_live` (registered CHECKS entries) enforce checkpoint KEEP claims; partial: covers checkpoints, not all deletion candidates.

- §39: a deletion candidate was judged without running it; the candidate was a live process, not a stale file.
- §41: a deletion list was broadcast and deleted within the 24h window; an unclaimed candidate was still in use.
- §255: a deletion list named 143 tracked files; a filesystem-built list cannot see tracked status, and no pod-side gate would have flagged the removal.
- §275: a deletion listing is a claim about its own PREDICATES, not about the files. Scoring "protected" as pinned OR hardlinked OR named-by-an-open-row put `ckpt_..._8b.pt.step9000` in the candidate pool — the resume source of the whole 30B trajectory and the baseline of an endpoint comparison filed an hour earlier. A resume source carries none of those three marks once the run it seeded finishes, which is exactly when it stops looking needed. Two predicates were missing (cited-by-a-results-artifact; resume-source-of-a-recorded-run, closed rows included) and adding them moved 53 GB from candidate to protected. The listing was not wrong about the file, it was wrong about itself: every row well-formed, the omission with no representation in the output — same disease as §272's emptied population and §270's discarded assertion identity. Put the predicates in the script so the next person inherits the rule rather than the output, and deliver a paired prediction (protected with them, unprotected with exactly those two removed) rather than a green run.

Cannot see: whether a non-checkpoint deletion candidate (a process, a lease, a temp file) is live; whether a listed path is tracked in main; or whether the protection rule that built the list covers the ways a candidate can be depended on (§275 — the narrow machine-checkable half is "no candidate basename may appear in any `runs/*.jsonl`").

## R11. A predicate set answers the question it enumerates, not the one it is named for; assert its population against the filesystem, never against a list

4 incidents (4 model, all 2026-09-08), ~2h each, 8h. `manual:` — no single check fits, because the defect is the absence of a case rather than a wrong case. The nearest machine-checkable halves are per-instance: `deletion_candidates.py::_selftest` asserts that its citation glob subsumes every single-ledger predicate, and `domain_bpb.py::_selftest` asserts a known answer on an input longer than `max_ctx`.

The shape: a function computes over a population it enumerates by hand — a tuple of filenames, a set of predicates, a list of cases — while its name, docstring and output all claim the population the caller means. Both halves are internally correct. Nothing crashes, nothing skips, no row looks odd, and the missing member has no representation anywhere in the output. The listing cannot show the predicate it does not have; the metric cannot show the bytes it did not score.

Distinguishing it from R2: R2 is a criterion that tests the wrong property and can be caught by a known-answer world. This is a criterion that tests the right property over the wrong population, and the known-answer world only catches it if the input is drawn from the part of the population that was omitted — which is precisely the part nobody thought of. So the fix is structural, not another case: take the population from the filesystem or from the data, so it cannot drift from what the caller means.

That all four landed on one day, in one session's work, is the finding. §275 named the shape; §276 is the shape inside the fix for §275; §277 is the same shape in an unrelated file, found by a peer while §276 was being fixed; §278 is the shape in the INSTRUMENT built to measure §277, and it manufactured a defect that had not occurred. Naming a shape does not immunise the next thing you write against it, and the fourth instance says the exposure grows rather than shrinks once you are hunting the shape — each probe you write to chase it is another hand-drawn population.

- §275: a deletion listing scored "protected" as pinned OR hardlinked OR named-by-an-open-row — three predicates against a four-predicate question — and offered the 30B leg's resume source. Also filed under R9.
- §276: the FIX for §275 hardcoded seven `runs/*.jsonl` while the pod holds 78, so the predicate added to catch "a result depends on this checkpoint" was itself narrower than its own name. Four cited checkpoints came out unprotected, one of them cited by `experiments.jsonl` — a file the script already opened for a different predicate and still did not scan for citations. Replaced by a glob; the invariant asserted is that the glob subsumes every single-ledger predicate, since those ledgers are members of the glob's own population. The first selftest written for it asserted the wrong invariant (that every predicate must uniquely protect something) and the tree refuted it: subsumption is necessary, not redundant.
- §277: `domain_bpb.text_bpb` truncated to `max_ctx=2048`, summed loss over the prefix, and divided by the WHOLE text's bytes. Our rows are 4,097 tokens, so every published absolute bpb was its true value over ~2× the bytes actually scored — measured 1.9671 to 2.0201 per domain, 2.0002 overall, 64 of 64 rows truncated in all nine domains. The selftest had five cases and every one was a handful of tokens, so the truncation branch had no input and the disagreement no world. Deltas, signs and ratios were unaffected (both arms carry the same factor), which is why it survived every comparison anyone ran. The docstring asserted the property the code did not have.

Adjacent, filed elsewhere: §272 (R2) computes a property's population with the predicate the defect breaks — the same motion with the population narrowed by the defect's own symptom rather than by a hand-written list.

- §278: the correction factor measured for §277 was computed by a probe that reimplemented what `text_bpb` does, and its population differed from the metric's by one keyword argument and one index — plain `decode()` where the metric passes `skip_special_tokens=False`, and the first token taken from the re-encoded ids rather than the original's. Every intermediate number it printed was well-formed. It produced ratios wrong by up to 2.4% (chatml 2.0145 against the correct 2.0634) and a fabricated 5.20% "row set drift" that sent a peer and me after a defect that had not occurred. Calling the shipped function instead reproduces the recorded byte counts exactly, nine domains, zero difference.

**The operational rule this one adds: to measure a correction for an instrument, CALL the instrument, never reimplement what it does.** A reimplementation is a second population by construction, and its disagreement with the first reads as a finding about the data rather than about itself. Corollary for the check: a probe that recomputes a quantity the artifact already records must reproduce the recorded value on unchanged input before any of its other output is believed — that comparison is free, and it is the one that fails first.

Also the reason a selftest is built from the real tree rather than a fixture wherever it can be: a fixture is a population the author chooses, and the author is the person who already chose wrong once.

Cannot see: whether a hand-enumerated population is complete, in general. What IS checkable, per instance, is that no member of a sub-population is missed by the predicate that should subsume it, and that the output states the size of the population it scanned — 78 ledgers, 64 of 64 rows truncated — so a reader can compare it to the one they meant.

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
