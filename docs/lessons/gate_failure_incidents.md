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

### §179 (2026-09-05, R1)
A probe that needed to know which training rows a checkpoint had seen assumed contiguous consumption. `arange(used[name], used[name] + want) % len(pool)` allocates contiguous, but `randperm(ph.shape[1], generator=g)` shuffles the plan across all domains within each phase and `plan[:, :n][:, rank::world]` stripes it by rank, so an early-stopped run consumes a prefix of the shuffled plan, scattering reads across the whole allocation at ratio (steps consumed / steps planned). Measured: the probe's "seen" region held 12.60% read rows; its "unseen" control held 87% of the training set; the difference-in-differences estimator would have been diluted ~8x into a false null. Two refutations were in hand before the probe was built: (a) cursor/steps = 80380/3815 = 21.069, non-integer — a per-step fractional share is what a shuffled multi-domain plan produces and a contiguous walk cannot, and it was treated as a curiosity; (b) the mix's cap_covers was 6.7x the region size, encoded as a passing selftest asserting the gap was expected — the real allocation is 8.01x, the gap was real, and the selftest converted evidence against the premise into a test that would keep anyone from re-examining it. A selftest that explains away an anomaly is worse than no case: it makes the anomaly look adjudicated. The complementary error: the cursor-sum identity (244,160 == 3815×16×2×2) was cited as confirmation, but it holds under BOTH the right and the wrong picture — a check with no power to discriminate, which feels like verification. Sub-case: predicting the cursor by summing per-rank counts got all nine domains wrong by up to 97 rows, because `counts[i]) * world` writes rank 0's count scaled by world — the read set is the union over ranks, the cursor is rank 0 scaled, and one formula for both fails quietly. The corrected item file ships both fields: n_seen_rows = 80,280 (the union) beside bounds.row_cursor = 80,380 (rank 0 × world), the 100-row overcount visible as two separate fields in a shipped artifact. A second session independently reproduced all nine cursors from train.py and hit the identical rank-sum trap on its first attempt — two sessions, same trap, inside an hour. What caught it: computing the allocation from the mix's own weight and reproducing the plan against something it was not fitted to; nine independent counts is what makes the reconstruction believable where a single matching number (the target's 80,380) was true of both the right and the wrong model. Evidence: train.py `arange(used[name]`, `randperm(ph.shape[1]`, `plan[:, :n][:, rank::world]`, `counts[i]) * world` — code patterns, not line numbers; train.py grew ~90 lines between two readings of this entry, and a line-number citation is a claim that decays.
open: a check that a probe's premise is verified against the code that produces the data, not the line that answers the question; none exists. A mechanical check that greps each shapes-entry `file:line` citation and FAILs when the cited line does not contain the quoted code would catch the citation-decay class.

### §190 (2026-09-05, R1)
A fix for the bf16 table was to remove MasterWeights' table exclusion, but MasterWeights is constructed only under `--fp32_master` — 8 of 113 ledger launches passed it, and neither the arms nor the control did. The fix would have landed in a class the callers never construct and changed nothing on the runs it was meant to fix. The premise "MasterWeights is the active class" was accepted without checking which class the arms actually build. Evidence: train.py `class MasterWeights`, `--fp32_master` flag, prereg amendment_9 (7183e7bb).
open: a check that a fix's target class is constructed on the runs it targets; none exists.

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

### §186 (2026-09-05, R2-c)
A test could not reach the code it was testing, and its "expected" answer was produced by an earlier gate. The change under test was the card-source refusal, narrowed to ladder-mix launches; the test invoked the real `harness launch` for a ladder mix and a non-ladder mix and asserted which one printed the refusal. Neither did: on a laptop with no token caches a startup gate refuses first, for BOTH mixes alike, so the non-ladder world printed "not refused" and read as a pass while never reaching the card-source code at all. The ladder world printed "not refused" too, which is what exposed it -- a test whose negative case passes for the same reason as its positive case has no power to discriminate. Rebuilt at the level the change lives: the condition's inputs, five mixes plus an agreeing-sources control. Evidence: the launch order in `cmd_launch`, cache gate before the allocation gate.
The general form: the assertion was on OUTPUT ABSENCE ("the refusal did not print"), which is satisfied by every path that never runs, including paths that fail earlier for unrelated reasons. An absence-of-output assertion needs proof the code under test executed -- otherwise it measures reachability, not behaviour. Related to §137: 0 violations and 0 items checked read identically.
open: a check that a test asserting output absence also asserts the code under test ran; none exists. The cheap discriminator is the one that worked here -- run the POSITIVE world first and require it to produce the output, so an unreachable code path fails loudly instead of passing quietly.

### §187 (2026-09-05, R2-d)
A document scanner attributed every record to the wrong parent. `gate_failure_incidents.md` is `## Rn` rule headings with `### §N` incidents under them; splitting the text on `"\n### "` puts the NEXT rule's `## Rn` line at the END of the preceding incident's block, so reading the rule from the block that contains the incident credits it to the FOLLOWING rule. Measured: §182, filed under R3, printed as "R4. Failures must be loud" in `harness brief net`, and §126 printed R9 instead of R8. Every incident in the brief was mis-attributed and the output looked entirely plausible, because each incident and each rule were individually real. Fixed by carrying the rule forward and advancing it only after the block that announces it, and by trimming each incident's body at the next `## ` so text belonging to the following section is not read as part of it. Evidence: `scripts/harness.py` `_brief_incidents`.
The general form: splitting on a child delimiter does not partition by parent. The parent of a block that follows a boundary is the boundary before it, not the one inside it -- a scanner that reads the parent from within the child's own block is reading the next parent.
open: a check that a doc scanner's parent attribution is verified against at least one known pair; none exists. One assertion would have caught it: §182 is filed under R3, so any reader that prints R4 for it is wrong.

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

### §183 (2026-09-05, R2-a)
A test that cannot fail, in two shapes, both of which read as coverage in a list of worlds. (1) TAUTOLOGY: a world asserting the undefined-vs-chance rule picks the wider standard error wrote `max(0.0087, 0.0115) == 0.0115` — a test of Python's `max()`, which no mutation of the code under test can turn red. Closed by extracting `is_undefined_vs_chance()` and asserting the predicate instead of the arithmetic. (2) UNREACHED ASSERTION: a world named for group-masking used a single scoreable item, and `cluster_se` returns at its `n < 2` guard before the strict `zip` that the masking protects — so the world exercised the guard, the mutant that deleted the masking stayed green, and the entry in the list said "groups masked with the outcomes". Closed with four items, two scoreable on different rows; the mutant then raises. Both are the same defect as §176's instrument-verifies-itself and §179's cursor-sum citation: evidence with no power against the hypothesis. Four instances landed in one day across two sessions, every one of them inside work whose subject was verification — the density is the finding, not any single case. What makes the unreached-assertion form worse than an absent test is that an absent test is visible in a coverage list and this one is not: it has a name, a comment, an assertion, and no path to the code it names. Evidence: eval/api_cloze.py `cluster_se` n<2 guard and `is_undefined_vs_chance`, b8f308dd.
open: a world must be shown to REACH the line it was written for — an early return, a guard clause or a filtered-empty population silently converts a test into a no-op. Mechanically checkable for the mutation-tested subset: a world whose named mutant does not turn it red is not a world, which is the existing `--selftest` contract applied to each world individually rather than to the check as a whole.

### §178 (2026-09-05, R2-g)
A reimplemented measurement disagrees with the producer by exactly the convention it dropped, and the residual is a clean multiple that reads as a real discrepancy. Twice in one session, both against artifacts that were correct. (1) Checking whether nine token caches were fresh, `_corpus_fp`'s body was copied out of train.py without its closing `[:16]`, so a 40-char hash was compared against a 16-char `.srcfp` sidecar: all 9 domains reported STALE, and the report was written before the shape of the failure — every stored value being exactly a prefix of the live one — was read. (2) Re-deriving zh_web's supply, a fresh counter summed bare `len(ids)`: 21,279,926,542 against a stamped 21,293,403,945, a 0.063% gap. The difference was 13,477,403, exactly the document count — `scripts/count_tokens.py` CONVENTION is `ids + one <eos> per document (train.py encode)`, whose own docstring warns a counter omitting the terminator reads low. Both artifacts were right; both measurements were wrong; in each case the residual named the dropped convention. A near-uniform ratio or an exact multiple of a population count is the signature, and it is easy to publish as a finding about the artifact. Evidence: train.py:1522 `_corpus_fp` return, scripts/count_tokens.py:16 and :22, facts/corpus_supply.json#cs.zh_web_landed (re-derived basis).
open: no check can see a counter that was never written down, so the rule is at the call site — import the producer's function rather than copying its body, and when a re-derivation disagrees, divide the residual by the population before attributing it to the artifact. Related to §176: there the instrument supplied its own ground truth; here the replica silently differs from the original.

### §177 (2026-09-05, R2-g)
An arm's flags, log lines and ledger row would all have said it carried a 1.07B-parameter memory table while it trained as the control, and the primary readout would have printed a clean null. `HybridLM._body` has two paths: the plain one calls `Block.forward`, which adds the memory, and the `attn_res` one iterates `Block.sublayers()`, which returns `(ar1,n1,mixer)` and `(ar2,n2,ffn)` and no memory branch. `Cfg.attn_res` defaults True and the control trained True (`ck["cfg"]` of `ckpt_b0_headmix_armA.pt`: attn_res True, 50 AttnRes tensors in its weights), so every arm would have taken the path that skips the memory. The criterion available at launch — the flags the run was given — cannot see which of two code paths consumed them. Measured after the fix: 966/4096 rows touched on the attn_res path, 0 before it. Found by reading the first CPU run's traceback, whose stack went through `sublayers()`, not by reasoning about the config. Evidence: model.py `_body`, scripts/test_arch_compat.py memory case 1, 72bbad2b.
open: a coverage assertion inside the arm rather than a flag comparison outside it — the arm must assert its own intervention was reached (rows touched > 0) and fail if not. Generalises §142 and gates-blind-to-coverage: a null result is only publishable if the run proves the treatment was applied, because a skipped intervention and an ineffective one produce the same number.

### §184 (2026-09-05, R2-g)
Excluding a parameter from the fp32 master copy would have silently stopped it training. MasterWeights.pull_grads copies p.grad into the master's m.grad and then CLEARS p.grad, because the optimizer holds `m` and its zero_grad() reaches only m.grad -- without the clear, the next backward accumulates onto the old gradient (a bug this codebase already paid for: grad 2.0, 4.0, 6.0 over three steps). The memory's value table is already fp32, so a master copy holds the same numbers at the same precision and costs 4 GiB at M1 and 16 at 2048x2048; excluding it is right. But the obvious exclusion -- drop it from the pairs list -- puts it in a third state neither branch was written for: not mastered, so nothing copies its gradient, and if the loop had also cleared it, nothing would have stepped it either. The table would have been read every forward, its diagnostics would have shown rows touched and a healthy Gini, and its weights would never have moved: the arm trains as the control and every observation says the memory is working. Caught by asking which code clears an unmastered parameter's gradient, and verified at train.py:2785 that opt.zero_grad(set_to_none=True) runs per optimizer after step -- so the exclusion is safe for the reason the comment now states, rather than by luck. Evidence: train.py MasterWeights.__init__ and pull_grads.
open: a check that every parameter reaching an optimizer has exactly one writer of its gradient and one clearer of it; none exists. The general form is §177's: an intervention that is silently not applied produces the control's numbers and the treatment's log.

### §191 (2026-09-05, R2-g)
Readout 4's touched/key_hits/Gini/entropy are all functions of the SELECTION, so a table whose every write rounds to zero in bf16 reads 100% healthy. The instrument is blind to the failure by construction: it measures whether the selection mechanism works, not whether the table is updated, and a dead table produces the same selection metrics as a live one. The fix is readout 6, a per-row checksum that verifies the table's values actually changed. Evidence: prereg amendment_7 (5258a092).
open: a check that a readout's metrics include at least one that is a function of the treatment, not only of the selection; none exists.

## R3. Artifacts carry their producer identity

### §24 (2026-08-31, R3)
A checkpoint with no recipe provenance was scored; the score was attributed to a recipe the checkpoint did not run. Evidence: runs/ckpt_*.pt.
open: a check that checkpoints carry recipe provenance before scoring; none exists.

### §182 (2026-09-05, R3)
A validator and the file it validates against were separated at three different boundaries in one day, and each separation produced a writer whose every call opens a file that is not there. `scripts/memory_diag.py` reads `data/ledger_schema.json` at every `log_diag`. (a) COMMIT: the hook refused the new `data/` path because its allow-list entry sat on the branch, unreachable until merge, so the writer was landed first and the schema second -- an intermediate commit that does not stand on its own, and `merge_main.sh` caught it when the hook's own selftest of the staged copy raised FileNotFoundError. (b) MERGE: the same pair, same cause, from the other direction. (c) POD SCOPE: `pod_drift`'s SCOPE listed `scripts/memory_diag.py` and not `data/ledger_schema.json`, so the writer would have shipped to the training box alone; b0's hook calls it every 100 steps and the FIRST row would have raised there, about two minutes into a run holding cards 1+2, while `memory_diag_fresh` read WARN "no diagnostics row at all" -- which is what a launch before step 100 also reads. Evidence: `scripts/pod_drift.py` SCOPE at 14b9a6a1, `runs/friction.jsonl` rows for 95579a06.
Found by asking which of my files reach the pod, not by a gate: `pod_drift --list-scoped` named the writer and not the schema. Two of the three were caught by existing guards; the pod one had nothing looking at it.
open: a check that a file read at runtime by a scoped script is itself scoped. `pod_drift` can see SCOPE and can parse the open() calls in the scripts it ships, so the join is available; nothing computes it.

### §189 (2026-09-05, R3)
A close written for one run minted a phantom run. `runs/experiments.jsonl` folds on (name, started), and `exp.py done` without `--started` stamps the row with its OWN write time -- so closing a run whose row had already folded shut appends a third identity that no process ever produced. The ledger now holds three `b0_mem_m1` rows where two runs existed: 19:41 (a world-6 launch, killed at ~2 min), 19:45 (the run that produced the throughput data, still folded `fail | vanished` because the harness wrapper shell was inside the kill set so no `.rc` was written), and 19:55 -- not a run: nothing launched, no cards were held, no artifact belongs to it, and it is the row carrying the stop verdict and the numbers. The tool has `--started` for exactly this and did not require it, because its refusal fires only when a name has more than one OPEN row; with zero open rows it silently invented one instead. Both later attempts to attach the verdict to the right row -- `done --started` and `note --started` -- were correctly refused ("no open row ... Open rows: none"), so the correction could not go where the error was: an append-only ledger records the phantom permanently, and the mapping from row to run lives in `runs/b0_mem_m1_vs_control_tps.json` and prereg amendment_6 instead. Evidence: runs/experiments.jsonl rows for b0_mem_m1, runs/b0_mem_m1_vs_control_tps.json (ledger_note).
open: `exp.py done` should require `--started` whenever the name has any CLOSED row and no open one, rather than defaulting to now -- the check it has asks "is the target ambiguous among open rows" when the failure is "there is no open row to target". A reader-side check is available too: a (name, started) whose only event is terminal, with no `running` event ever appended, is a row for a run that never started.

## R4. Failures must be loud

### §188 (2026-09-05, R4)
A launcher silently overrode the caller's card and world request with the grant's, so a two-card arm ran on six cards with every log line still saying `m1`. b0 launched M1 as `CUDA_VISIBLE_DEVICES=1,2 NGPU=2 python3 scripts/harness.py launch b0_mem_m1 --training -- ./run_ddp.sh --mix data/mix_200m_8b.json ...` and `cmd_launch` ran `torchrun --nproc_per_node=6` on cards 0,1,2,3,4,6: `harness.py:16803-16806` does `env["CUDA_VISIBLE_DEVICES"] = cards` and `env["NGPU"] = str(len(cards.split(",")))` from the grant's whole `block_cards`, unconditionally, with nothing comparing them to what the caller asked for. **World 6 changes the data order**, so the arm would have been un-pairable with its world-2 control — readout 1 is block-paired doc_cu val, and the pairing is the measurement. The run would have completed and produced a number that could not be compared with anything.
Two independent defects in one launch line, and each hid a different half. The one I found from the repo, via `no_foreground_pod_training` going red: the launch was a foreground `crictl exec` (`bash -lc ... | tail -40`, no `setsid`), so it dies with the tn tunnel and leaves an orphan holding cards, and the `| tail -40` means there is no log file at all — nothing on disk for anyone, including `memory_diag_fresh`, whose freshness window reads `runs/<name>.log` and degrades to "log step unread" without one. The one only the controller could find, by reading the live process tree: the six-rank torchrun, because the repo cannot see what the caller's environment was.
Evidence: `scripts/harness.py:16803` `env["CUDA_VISIBLE_DEVICES"] = cards`; the killed pids in the container (torchrun 838620 with six ranks, harness 838610, leader 838599); `nvidia-smi` compute-apps empty afterwards, all cards 0 MiB.
open: (a) `harness launch --cards` accepting an explicit list that must be a SUBSET of the grant's block and refusing otherwise, so a multi-arm program does not need the grant rewritten between launches; (b) a refusal when the caller's `CUDA_VISIBLE_DEVICES`/`NGPU` disagree with what the launcher is about to allocate — the silence is the whole incident, and a launcher that knows better than its caller must say so rather than act. Both specced to the controller before they can refuse.

### §51 (2026-09-01, R4)
An observation channel swallowed the signal; the check read the channel's default, not the observation. Evidence: scripts/harness.py.
open: a check that observation channels propagate errors; none exists.

### §181 (2026-09-05, R4)
A DDP benchmark runner reported success while every arm crashed. Two mechanisms in one runner: (a) `echo DDP_DONE rc=$?` after a `for arm in ...; do torchrun ...; done` loop — `$?` after `done` names the loop's last command, and the loop body ended in a pipe to `grep -vE`, whose exit status is 0 whenever it outputs a line; all four arms crashed with ChildFailedError and the log's last line said `rc=0`. (b) the same grep filter (`grep -vE '^\[|Warning|warn'`) deleted precisely the diagnosis: every real error line from a torchrun child is `[rankN]:`-prefixed, so the filter removed the child tracebacks and left a generic `ChildFailedError ... traceback: <N/A>`. Three log reads revealed nothing; the cause (NCCL "invalid usage" — two ranks pinned to one device) found only after re-running with no filter. The two mechanisms compound: the filter hid the traceback AND supplied the zero that hid the crash — even a correctly-captured per-arm exit code would have read as passing through the pipe. The fix (`fail=N` per arm) works only because the pipe was also dropped. General form: a filter written against the success shape silently removes the failure shape, and both leave a log that reads clean. Same family as §136 (e1's `head -1` filter eating the REFUSAL line). A green summary line and a clean log are observations about the reporting, not about the job. Evidence: 47bfc95a (fix commit, immutable — the pod log /work/aupai/runs/ddprun.log is overwritten on each rerun and the crashed-arms version is gone).
open: a check that a runner's summary line is derived from the job's exit codes, not the reporting command's; none exists.

### §193 (2026-09-05, R4)
Two mechanisms where a launch-side guard reported the wrong state. (a) A card claim bound to the harness launch wrapper shell's pid read STALE/ORPHAN on a live arm: the wrapper exits after launching torchrun, but the torchrun process and its python children hold the cards. The claim's liveness signal was about the wrapper, not the job. (b) The pod wrapper's `cd … &` refusal missed a `& ; …` tail: the refusal pattern matched `cd … &` but not `cd … & ; …`, so a malformed launch printed "launched" with no process and no log. Evidence: de queue (poll-then-claim fix), tilerl (pod wrapper tail).
open: a check that a card claim's pid is the torchrun/python process, not the wrapper shell; and that the pod wrapper's refusal pattern covers all shell-backgrounding tails. Neither exists.

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

### §185 (2026-09-05, R6)
A memory budget was computed from a dtype nobody had set. The charter, the controller's arithmetic and mine all costed the product-key value table at 6 bytes per parameter: a bf16 table at 2 plus one fp32 Adagrad moment at 4. Measured on the pod by reading the live tensors instead: the table is torch.float32 (nn.Embedding is constructed in the default dtype and nothing casts it), its dense gradient is fp32 too, and the Adagrad state is fp32 -- 12 bytes per parameter, and the gradient had been omitted from the arithmetic entirely. At M1 the error is 6.00 GiB against 12.00 and went unnoticed because the shape fit anyway; at 2048x2048 it is 24.00 against 48.00, and the arm OOMed in backward allocating the 8.00 GiB gradient after construction had already succeeded. The number that decided the arm's size was an assumption wearing the units of a measurement. It also produced a false intermediate finding: with the wrong 6 bytes I reported a 14.82 GiB "gap over the arithmetic" and attributed it to activations and fragmentation -- a plausible explanation for a discrepancy that was two-thirds arithmetic error. Evidence: probes on card 5, runs/prereg.jsonl#memory_layers_0905 amendment_5.
open: a check that a size computed from a tensor's dtype reads the dtype off the constructed tensor rather than from the code's intent. The construction is available to it -- one forward on a small instance prints every dtype -- and nothing does it. Weaker but cheaper: refuse a memory-budget number in a doc that names no dtype.

### §192 (2026-09-05, R6)
A share carried across denominators: opt_step was 3.4% of M3's 2349 ms step, quoted as ~3 points of the arm's ratio, whose step is 1024 ms — the same figure is 9 points there, and the row count differed 2x on top. The number's basis (which step time it was a share of) was not carried with it, so the same 3.4% became 3 points in one context and 9 in another. Evidence: runs/b0_mem_m3_peak_1448.json, fb error, tilerl correction.
open: a check that a share or ratio quoted in a doc names its denominator; none exists.

## R7. Retractions travel as wide as the ruling

### §16 (2026-08-31, R7)
A retraction was narrower than the ruling; the ruling voided a todo that the retraction did not name, and the todo was later acted on. Evidence: docs/standards/state_0904.md.
open: a check that retractions name every todo the ruling voids; none exists.

## R9. Run a deletion candidate before judging it

### §41 (2026-09-01, R9)
A deletion list was broadcast and deleted within the 24h window; an unclaimed candidate was still in use. Evidence: runs/pod_ckpt_candidates_*.txt.
open: a check that the 24h window elapsed before deletion; none exists.

## R10. What happened only on the pod did not happen

