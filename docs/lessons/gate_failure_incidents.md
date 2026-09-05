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

### §198 (2026-09-05, R1)
A profiling probe measured the wrong world: `scripts/profile_step_cost.py` never constructs `TableMaster` (train.py:2612 does, unconditional on `--fp32_master`), so the 1195 `--peak-only` run measured the without-master table at 6 B/param while the arm runs at 14/16 — 13.62 GiB short. The planned "with master" cell would have measured the without-master world and reported a null as "the master is free." The premise "the probe constructs the same classes as the arm" was accepted without reading the probe's construction code. Caught by b0 by grep after launch, before any decision was taken. Evidence: b0_mem_m3_peak_1195 on the pod, amendments 7-10, train.py `table_master = TableMaster(`.
open: a check that a profiling probe's class construction matches the arm's; none exists.

### §199 (2026-09-05, R1)
A refused commit was reported as landed. A charter edit was refused by the behind-main hook; the worktree looked identical to a success (the change was staged and the tree was correct), and it was reported as on main. "I made the change" and "the change is on main" are different claims; `git status` on one's own tree is not evidence for the second. Verified later with `git show main:<path>` (c82ef127). Evidence: c82ef127.
open: a check that a "landed" claim is verified against the integration tree, not the worktree; none exists.

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

### §206 (2026-09-05, R2-a)
A broken world asked the temp dir for the repo's file list, and it happened twice in one day. `check_no_hardcoded_cache_path`'s broken world is a temp dir holding one mutated copy and no git index; its first version ran `git ls-files` in the world, got nothing, and the check PASSed vacuously — every check does, against an empty population. The same shape had failed to fire that morning in `_broken_pod_reads_are_scoped`. Fixed structurally: the file list comes from `git -C ROOT ls-files` (a property of the repository), the contents from `root` (the property under test). Verified by running: the world now returns a real temp path and the check FAILs on it. Evidence: scripts/harness.py:11450-11455 (the comment names both instances), `_broken_no_hardcoded_cache_path`.
open: a check that a broken world which shells out to git for the file list takes it from ROOT, not from the world; none exists — the second instance in one day is the evidence it needs one.

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

### §201 (2026-09-05, R2-b)
A predicate verified only where it cannot fire reports nothing about where it does. The device-fd refusal in `card_claim.acquire()` was made unconditional at 121a865d, written for a launcher that bound the wrong pid. It was applied to every claim shape without asking which shapes exist: all ten `claim_my_cards` call sites claim the card they are ABOUT to use (the documented contract, and the reason it reads `CUDA_VISIBLE_DEVICES`), so all ten would have been refused on the pod. Every local gate was green because macOS has no /proc and the predicate abstains there; CI (Linux, /proc present, no GPU) caught it in two hours. Fix: `acquire(require_device=False)` with the launcher opting in after `wait_for_device` has already established the fact (4f4e9175). Evidence: 121a865d, 4f4e9175, the ten call sites (probes/fone_digit_acc.py:39, scripts/eval_heldout_ours.py:95, scripts/test_arch_L32.py:53, scripts/test_arch_compat.py:85, scripts/n7c_gates.py:124, scripts/n7c_grad_check.py:63, scripts/test_e2e.py:190, scripts/eval_heldout.py:567, eval/nan_probe.py:24, eval/score_matrix.py:1789).
open: a check that a platform-conditional predicate is tested on every platform where it can fire; none exists.

### §202 (2026-09-05, R2-b)
A check that runs over existing data cannot verify a branch that data does not take, and its pass is not evidence about the untaken branch. `data/ledger_transport_schema.json` is a generated projection of `ledger_audit.KEYS`, with `--verify` comparing the projection's key against each lambda's key on every row of all nine ledgers: 1,249 rows, zero disagreements. As a negative control, `score_matrix`'s `default: "full"` was deleted from the spec — a deliberate break that keys rows differently. `--verify` still reported zero disagreements on all nine. Every row in `runs/score_matrix.jsonl` carries a `profile` field (69/69), so the default is never reached; the check compares on the rows that exist, and a spec form no existing row exercises is unverified while the check reports a clean pass. Not hypothetical: the first score_matrix row written without a profile would key to None in the infra repo reading the JSON and to "full" here, and the transport verb would treat it as a new key rather than a duplicate. Distinct from §89's empty population — the population here is 1,249 rows and healthy; what goes unverified is the default, the fallback, the else-arm no row reaches. A count of rows checked says nothing about which branches those rows took. Closed by six probes comparing each spec form against its own ledger's lambda on a synthetic row built to reach that form — the broken world fails 15/16 where `--verify` sees nothing. Evidence: `scripts/gen_ledger_transport_schema.py` on main, the six probes in `_selftest`, the 0-of-69 measurement in the comment beside them.
open: a check that every branch of a data-driven spec is exercised by at least one case — coverage of the spec's forms, not of the data's rows. None exists; the specific form is "for each spec, one synthetic row per alternative."

### §203 (2026-09-05, R2-b)
A probe task must make the failure both REACHABLE and COSTLY, and both halves must be measured on the task, not assumed. A toy task built to adjudicate why the memory arms collapsed (regression onto tanh(x @ W) with Gaussian inputs) made the collapse CORRECT: one shared low-rank read fits the task, so a memory that distributes is doing unnecessary work and the objective says so. The lowest touched_fraction in a 9-cell grid (BatchNorm at selector lr 0.02, 0.0465) was also the LOWEST LOSS (0.0730), and the two cells that reached touched 1.0000 had the two worst losses. A wrong finding ("BatchNorm makes usage worse") was sent to the controller and withdrawn ~40 minutes later. The mirror: a discrete-recall task (8,192 facts, random targets) made the failure UNREACHABLE — a uniform draw presses every key symmetrically, so every cell read touched 1.0000 and nine identical cells ranked nothing. A Zipf(s=1) task plateaued at touched 0.949 and also never collapsed. NO task built satisfied both properties; the only one that reproduced the collapse was the one where the collapse is correct. The trap inside the trap: a too-early read (steps 1-120) reported the smooth task as "failure does not occur" — the honest selector reads 0.966 in that window and only drops to 0.091 in 121-240. Evidence: probes/mem_usage_toy.py at 8c23ea86 (unmerged, b0-ve-rownorms), selftest at :268, _one_row_forward at :392.
open: a check that a probe task's failure is both reachable and costly on the task itself, not assumed; the selftest now measures both properties per task and asserts none can rank.

### §209 (2026-09-05, R2-b)
A witness tolerance copied from another check sat ABOVE both defect signals it existed to catch. The MoE dispatch bench's tied-weights witness — one expert holding the dense module's own weights must reproduce dense(x) — first used 0.05, copied from the paths-agree check next to it. Measured: same function through two expressions differs by exactly 0.0 at fp32 and bf16 (the noise floor), the activation defect (plain a*sigmoid(b) vs model.SwiGLU's SiTU) reads 0.023-0.025, and a loop with swapped betas reads 0.0172, on outputs of scale 0.75. The 0.05 passed all three broken worlds it was written for. Fixed at 1e-3, between the noise floor and the smallest defect. Rule candidate: a tolerance is set between the measured noise floor and the measured defect signal, never copied. Same shape as §171 (a perturbation injected below the instrument's resolution): the pass band was wider than the defect. Evidence: scripts/moe_dispatch_bench.py, fix at ae149323.
open: a check that a witness tolerance is bracketed by a measured noise floor and a measured defect; none exists.

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

### §207 (2026-09-05, R2-c)
A broken world's injected line carried the check's own exemption marker, so the mutation never reached the check's population. `check_no_hardcoded_cache_path` exempts a line carrying `# cache-path-ok`; the first version of `_broken_no_hardcoded_cache_path` put that marker inside the line it injected, so the world produced the same count as the real tree and the check "passed" its own broken world at WARN. The mutation landed in the file and was neutralised in the same breath — the check read the marker and exempted the very line the world existed to plant. Fixed by assembling the injected line from pieces so the builder's own line does not trip the check while the line it writes does, with a builder assert (`assert "/data00/tokens_" in bad`) that the world's line lacks nothing. Verified by running: the world FAILs (3 hits, the injected line at scripts/stamp_cache_seeds.py:49, against baseline 2). Evidence: scripts/harness.py:11559-11564.
open: a check that a broken world's injected payload cannot be self-exempting — the marker that grants exemption must not be a substring of the line the world injects; none exists.

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

### §208 (2026-09-05, R2-g)
An import-time equal-compute assertion certified top-3+shared while every timed closure ran top-3 with NO shared expert. The MoE dispatch bench computed the grouped_mm-vs-dense ratio with N_SHARED in the formula and the print, but not one line of timed code ran the shared expert, so the grouped path did T*TOP_K rows while being scored against T*(TOP_K+N_SHARED) — exactly 0.7500 of the work it was credited with, and the reported 0.66x was withdrawn. All 7 selftests stayed green: every one interrogated the FORMULA (active_params ratio) and none asked the timed closure what it computed. Same family as the memory-probe objective that rewarded collapse (§203's mirror): the criterion measured a property of the specification, not of the run. Fixed by running the shared expert in the timed path and witnessing both MoE paths against dense separately. Evidence: scripts/moe_dispatch_bench.py, fix at ae149323.
open: a check that a bench's equal-work assertion is exercised by the timed code itself, not only by the formula; none exists.

### §196 (2026-09-05, R2-d)
A scanner located its subject by a delimiter and matched the line that carried the delimiter as a regex STRING. `test_launch_claims` finds the monitor template with `re.search(r"monitor_code = f'''(.*?)'''", src)` over all of `harness.py`; since 913610df a second site — `_selftest_monitor_stop_rules` at `harness.py:13301` — holds that exact text as its own search pattern, and it sits 3,000 lines BEFORE the real template. The search captured five characters, `(.*?)`, then counted `card_claim.py` lines in them and reported "0 release calls in its body" while both releases were present and correct. The failure direction is the harmful one: it fabricated a defect in working code, and the reader's first move is to go add a release that is already there. Same family as §187 — a parser that finds its parent by a boundary token finds whatever mentions the token first — but the earlier case read the NEXT parent while this one read a quotation of the boundary itself. Fixed by taking the function's own source via AST (`ast.get_source_segment` over `_arm_monitor`) and running the regex only inside it, plus a case asserting the template was found at all, because a search that matches nothing must not read as zero releases. Evidence: `scripts/test_launch_claims.py` at 913610df vs the fix; measured 18/19 with the release intact.
open: a check that a regex over source excludes string literals, or that a scanner's captured span is non-degenerate. The second half is cheap and general — a template match of 5 characters is never a 40-line body — and nothing asserts it.

### §200 (2026-09-05, R2-d)
A guard written to catch an omission omitted itself, because it tested for the omitted names by substring and its own comment contains them. The underlying defect first: `train.py:2603` constructs `TableMaster` for any `mem_values > 0`, taking the value table from 6 B/param to 16 at the in-step peak, and `scripts/profile_step_cost.py` — the only tool that measures whether an arm fits a card — constructed none. So `--peak-only` reported 8.17 GiB of table tensors at side 1195 where the arm allocates 21.79: 13.62 GiB missing from the number a launch decision is taken on, against a rule stated in GiB reserved. The probe was measuring a world that no longer launches, and the omission shipped in the same commit that added readout 6's timing to that same file, so the instrument was missing the fix it was measuring. That half is R1's shape and is why the 1195 probe had to be run twice.
The half that belongs here is the guard. Having fixed it, I added a selftest case asserting `TableMaster`, `pull_grads` and `push` appear in BOTH `train.py` and the probe or in neither — a difference between the two files being the actual defect, so a one-sided assertion goes green either way. Written as `frag in mine` over the probe's own source, it reported 3/3 present under a mutant that deleted all three call sites: the strings appear in the case's own comment and in its data table, so the file "contains" them no matter what it does. The check could not fail. Fixed by reading `main()`'s AST — the `calls` set the file already builds for its fp8 assertions — so a name counts only if `main()` really calls it; the mutant then produces three BUG lines at 28/31. This is the THIRD instance of the self-satisfying needle in this one file, which documents the other two at `:133` ("a needle like `tok_step = B * train.Cfg.accum * SEQ` written into this check's own table MATCHES ITSELF"), and I wrote the third one anyway, directly beneath that comment. Evidence: `scripts/profile_step_cost.py` at 440bd579; the 6 B/param and 14/16 B/param readings in `runs/b0_mem_m3_peak_1448.json` and the 1195 pair.
open: a check that a source-scanning assertion cannot be satisfied by its own text — the general form is that a file's self-scan must exclude the scanning code's own span, and nothing asserts it. Cheaper partial: any selftest case whose needle is a bare identifier must read an AST, not a substring; that rule is stated three times in prose in this file and enforced nowhere. Second, smaller: the file's hand-summed case total stayed at 28 while 31 cases ran, so for one commit it under-reported its own coverage — a count that is not derived from the cases cannot witness them.

### §205 (2026-09-05, R2-d)
A placeholder-survival guard fired on a CORRECT substitution. Six probe scripts are generated from `runs/mem_probe_TEMPLATE.sh` by substituting six `__PLACEHOLDER__` tokens; the guard asserted no placeholder survives substitution. The template's own documentation line reads "Copy per cell and set the five __PLACEHOLDER__ values from the table below", and the whole-file scan read that comment as an unsubstituted token — the guard matched its own template's prose, the fourth instance of the self-satisfying needle (after §200's three in one file). Fixed by scanning executable lines only. Evidence: `runs/mem_probe_TEMPLATE.sh` at 8c23ea86 on b0-ve-rownorms (unmerged).
open: same as §200's — a self-scan must exclude the scanning code's own span; nothing enforces it.

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

### §197 (2026-09-05, R4)
A profiling run desynchronised a DDP pair silently: save and val ran after the loop on rank 0 only (33.6 s against a 2.0 s step), so rank 1 waited in the next collective and the pair fell out of step. Rank 1 held 76 GiB on card 2 at 0% util for 17 min with a valid live claim — no error, no timeout, no signal that anything was wrong. A claim answers who intends to hold a card and says nothing about progress; 0% util against held memory was the signal, the claim status was not. Cross-reference the gpu-idle-memory rule (held memory + 0% util can be compile) — here it was the opposite reading, and the distinguishing fact is elapsed time against the known compile length. Evidence: runs/mem_decomp_0905.log on the pod, a788bd0b (claim torchrun pid per cell), dfeadd15 (memory config in row), acfb67bb (--skip-save-val).
open: a check that a held card at 0% util for longer than the known compile window is surfaced as a stall, not left to the claim's liveness; none exists.

### §204 (2026-09-05, R4)
A controller kill was recorded as a crash. 4c ruled M1/M2/M3 stopped under readout 4; the six rank pids were SIGTERMed at 23:11:42Z. `train.py:2786` `_save_on_interrupt` caught it, wrote each checkpoint, then raised KeyboardInterrupt as designed; torchrun turned that into ChildFailedError and the wrapper recorded exit 1. M2's monitor wrote `status=fail, result="exit 1"` — the row said the opposite of what happened, with M2's three measured diag rows nowhere on the record. The monitor cannot distinguish a controller kill from a crash by exit code. A retraction was filed via `exp.py retract` with the numbers (touched 0.9423 → 0.6733 → 0.3471 at steps 10/20/30, throughput 75,634 = 0.919 of control). Related: `settled()` matching on `name` alone released a LIVE arm's card claim and killed its monitor at the first 60s tick (fixed at 9f76da2f). Evidence: runs/experiments.jsonl on the pod (M2's fail row + retraction), runs/memory_diag.jsonl on the pod. R10 applies: the ledger rows are pod-side only and were never pushed (main frozen); if the pod is recycled before the merge, M2's retraction and the diag curves go with it.
open: a check that a monitor's verdict distinguishes controller-kill from crash (signal 15 vs signal 9 vs nonzero exit); none exists.

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

