---
question: Do the train.py guards and the 30B recipe scripts hold under P1-P8, and what single class produced most of 2026-09-01's defects?
status: measured
source: de-8; live artifacts on the pod (ckpt_pretrain_30b_s2.pt.step21500/22500, runs/readout_*.txt); probes /tmp/de_cursor_probe.py, /tmp/de_replay_p6.py; scripts/replay_cursor.py --selftest
---

# de-8: the guards, and the class behind the day

**Verdict: two principle-level defects found and fixed, both in the row-cursor path, both
of them fixes we believed had already shipped.** The first is dead code: `save_checkpoint`
has never written a `row_cursor`, so every checkpoint the live run produces has none. The
second is in the tool that exists to recover from the first: `replay_cursor.py` read an
absolute step against a relative plan and returned plan-complete counts for a 40%-complete
run, silently, 2.2M rows ahead of the truth. Injecting that cursor would have caused the
exact failure the cursor exists to prevent.

Both were found by running the recovery procedure rather than by reading it.

## The class: the test and the code agree, and neither touches the case that matters

Seven sightings on 2026-09-01, across five owners' code. They are one defect class, and
the fix is one sentence in every instance.

| # | site | what it does | what it should do |
|---|---|---|---|
| 1 | `train.py:1289`/`:1315` cursor save | writes nothing; the branch is unreachable | write the cursor, and a test that calls the production writer |
| 2 | `replay_cursor.py` step origin | returns plan-complete counts at 137.6% of the plan | refuse an impossible ratio; take the origin explicitly |
| 3 | `no_stale_running` | FAILs naming 1 of 13 open rows | enumerate the set it detected |
| 4 | `exp.py done` `reversed(rs)` | closes the LAST running row of a name | refuse when a name has >1 running row |
| 5 | `readout_30b.load_score_record` | returns the FIRST match, shadowing a re-score | fold last-wins, like every other reader |
| 6 | `readout_30b` head guard (fixed, b0 8215856) | refused a whole metric for one bad role | refuse per role -- **the reference implementation** |
| 7 | `agents_rules_covered` | asserts the named check EXISTS, not that it enforces the rule | compare the doc's table to the code's map |

Two sub-shapes, and the split matters because the fixes differ:

**Reports one member of a class it detected** (3, 6, 7). The check knows about the whole
set and prints one element. The failure mode is specific: a fixer repairs the named
element, reruns, and the same red returns naming the next one. tilerl hit this live on
`no_stale_running` -- read "1", and had they fixed only that row it would have come back
twelve more times.

**Acts on one of many where it should refuse** (2, 4, 5). Silent-wrong-target. Nobody sees
a red at all; the tool acts on a row the operator did not name and reports success.

**Returns a wrong answer where a stop belonged** is the aggravating factor across 1, 2, 4
and 5. A refusal is recoverable at any hour; a wrong number recorded as a measurement is
not, and it propagates.

### Why the tests did not catch any of them

Every one had a passing test. In each case the test's call shape differs from
production's:

- the de-7 rehearsal exercised the RECONSTRUCTION path (`replay_cursor` writes the dict, a
  resume reads it), which touches neither `:1289` nor `:1315` -- so the production write
  was never executed by any test
- `replay_cursor` was validated on stage 1, which started at step 0, the one case where
  absolute and relative step are the same number
- 3b's T7-2 (already merged, 34c461b) is the same shape twice more: `_build_lock`'s return
  bound in the test and discarded in production, and `_settle_dir(..., 0)` in the test
  versus `SETTLE_S=60` in production, where 0 short-circuits the branch the guard exists
  for

This is not a coincidence of five authors. A test written next to the code inherits the
code's assumptions, and the assumption that failed each time was about the CALLER, which
the test replaced with itself.

### The one check that would have caught the most

**A broken world must contain TWO offenders, and the check's evidence must name both.**

Applied to the seven: it catches 3 directly, and 4 and 5 (a ledger with two rows for one
name is a two-offender world). It does not catch 1 or 2, which need the second rule below.
Cheap to add -- it is a rule about `broken()`, not new machinery -- and mechanically
checkable: for any check whose evidence string is singular, the selftest can assert that a
two-offender world produces evidence naming two.

**Second, narrower, for the other half: a selftest must call what production calls.** Not
mechanically checkable in general. What IS checkable is the special case that produced 1
and 2: a guard whose body is unreachable. A dead-branch scan over the guard functions --
does any `if` in a check or guard have a condition that is constant after the assignments
above it -- would have caught `:1315` at the moment it was written.

## Findings

### D1 -- `save_checkpoint` has never written a row cursor (P3, principle) -- FIXED in the next window

`train.py:1289` rebinds `cfg` to a dict on every path:

```
:1289   cfg = cfg if isinstance(cfg, dict) else vars(cfg)
:1315   cur = getattr(cfg, "_row_cursor", None) if not isinstance(cfg, dict) else None
:1316   fps = getattr(cfg, "_row_cursor_srcfp", None) if not isinstance(cfg, dict) else None
```

By `:1315` `isinstance(cfg, dict)` is always true, so `cur` and `fps` are always `None`
and the whole `if cur:` block -- `row_cursor`, `row_cursor_as_of_step`,
`row_cursor_srcfp`, `row_cursor_seed` -- never executes. Two things hide it: `getattr` on a
dict never finds a data key even if the branch were reached, and `:1301` strips
`_`-prefixed keys from `ck["cfg"]`, so the cursor is absent from the nested config too.
Nothing raises.

Failing case, from the pod rather than from reading:

| checkpoint | top-level keys | `row_cursor` |
|---|---|---|
| `ckpt_pretrain_30b_s2.pt.step21500` (live run) | cfg, vocab_id, corpus_fp, env_fp, step | **absent** |
| `ckpt_pretrain_15b_s1.pt.step16000` | ..., row_cursor, row_cursor_srcfp, `row_cursor_reconstructed` | present |

The stage-1 checkpoint has one only because `replay_cursor.py` injected it; the
`row_cursor_reconstructed` marker is written by that tool, never by `save_checkpoint`.
Branch probe over all three `cfg` shapes (Cfg class, instance, SimpleNamespace):
`cur=None` in all three.

The consequence is not historical. `--auto-resume 2` is armed on the running supervisor,
so a container restart tonight resumes from a cursorless checkpoint and re-reads consumed
rows -- the de-7 failure, live. Fix: capture `cur`/`fps` before `:1289` rebinds `cfg`.
train.py is frozen while PID 3123860 runs; fb ruled this lands first in the between-stages
window, ahead of the other item-7 work, with a test that calls `save_checkpoint` with each
of the three `cfg` shapes and asserts `row_cursor` is in the written file.

### D2 -- `replay_cursor` read an absolute step against a relative plan (P3, principle) -- FIXED, 2645161

Found by running D1's recovery runbook against the live checkpoint before publishing it:

```
plan 3,662,109 rows, consumed 5,040,000 rows (137.6% of the plan)
```

A run cannot read more of a plan than the plan holds, and the printed cursor summed to
exactly 3,662,109 -- the plan-complete counts, which is the figure the as-of-step rule
exists to replace.

| quantity | rows | of the plan |
|---|---|---|
| stage-2 plan | 3,662,109 | 100% |
| absolute step 22500 x 16x2x7 | 5,040,000 | 137.6% (what it used) |
| (22500 - 16000) x 16x2x7 | 1,456,000 | 39.8% (the truth) |
| (32348 - 16000) x 16x2x7 | 3,661,952 | 1.0000 -- the plan is post-resume only |

`consumed_rows` came from `ck["step"]`, which is absolute, while a resumed run's plan
covers only the post-resume steps. With `consumed >= planned` the
`consumed < planned` guard was false, the phase-walk that computes as-of-step counts never
ran, and the function returned the plan-complete dict from its first loop. It errs toward
over-consumption: injecting it marks ~2.2M unread rows as read.

Fixed: `--resumed-from-step`, required rather than defaulted, because the origin is not in
the checkpoint (`cfg` carries no plan/trim/resume/total key) and a tool that assumes an
origin is how this happened. The durable half is the refusal -- `consumed > planned` now
raises and prints both interpretations. `--selftest` covers three cases: start-at-0
reconstructs 3,646,944 as-of-step, resumed-at-16000 gives 1,456,000 = 39.8%, and the
absolute-step call refuses. Case 3 verified to FAIL with the guard neutered.

### D3 -- `agents_rules_covered` checked existence, not enforcement (P2, principle) -- FIXED, a78d666

`AGENTS.md` mapped the `CUDA_VISIBLE_DEVICES` rule to `gemm_dims_aligned` (GEMM shape
alignment); the enforcer is `device_set_honoured`. Both are real checks, so
`harness.py:450`'s "is the named check in CHECKS" test passed and the rule read as covered
while being unenforced in fact. Independently found by 44 and 3b.

Behind it, a larger gap: no check read `AGENTS.md`'s coverage table at all. It was a
hand-maintained copy of `_RULE_CHECKS`, so the doc and the code could drift -- and in this
row were wrong together rather than in disagreement, which is why neither reader caught
it. Fixed both, and `agents_rules_covered` now diffs the table against the map (35 rows,
0 drifting). Second broken world added for the table half.

Stated ceiling: existence is checkable, relevance is not. Nothing can see a pair that
names a real check which does not enforce that rule; only a human re-reading the pair can.

### D4 -- `load_score_record` takes the first match (P4, defect) -- not fixed, not urgent

`eval/readout_30b.py:202` returns the FIRST line matching `(ckpt, profile)`. Every other
fold in the repo takes the last row or the max by key -- `_exp_row_status` explicitly folds
by `(name, started)` then latest, with a comment about the close that landed on the wrong
start. A re-score appended later is therefore shadowed by the older row, silently.

Not biting today: `runs/score_matrix.jsonl` holds exactly one row for
`ckpt_pretrain_30b_s2.pt.step17500`, so the 22B readout loads it unambiguously. The trigger
is any re-score of a checkpoint that already has a record under the same profile -- which
b0's 16B re-run is, one file over. `readout_30b.py` is not frozen, but it is in the 22B
readout's path at ~12:02; changing it during the window is the larger risk. Fix after the
22B milestone scores: last-match, matching `_exp_row_status`.

### D5 -- the 16B ledger row still records a superseded pairing (P5-adjacent, defect) -- not mine to fix

`runs/milestones.jsonl` row 6 reads `paired=ckpt_p324.pt, metrics_moved=0`, and
`runs/readout_ckpt_pretrain_30b_s2.pt.step17500.txt` is still the p324 text: "no metric
moved (all floor/flat/absent)", with `per_role_domain_loss` REFUSING on the whole set and
`code_500`/`math_hard` ABSENT. That is a non-reading, not a null.

b0's re-run against `ckpt_pretrain_15b_s1.pt.step8500` with the per-role head guard is the
result of record, but it overwrote neither artifact, and `facts/base_eval.json` has no
`be.milestone_*` entry for step17500 at all. A reader of the ledger gets the superseded
reading with no marker. Proposed: b0 appends a corrected row so the union fold's
last-row-wins makes the re-run the reading of record while the p324 row stays as history.
The writer attesting their own re-run is the honest shape; flagged to fb rather than done
unilaterally.

## What I verified and found clean

- **The 22B pairing chain.** Simulated the watcher's default pairing against the real
  ledger: 16b already scored and skipped, 22b pairs the 16B pin, 30b pairs the 22B
  checkpoint. Both within-stage, so the per-role head guard does not refuse them. A global
  `--paired` would have pinned 30B's pair to 16B too and skipped 22B as its natural pair --
  I launched with it, killed it 90 seconds later having scored nothing, and relaunched with
  the default chain.
- **The head guard fires.** The 16B readout's refusal names both sides' role sets
  (milestone-only `en_c4_stage2`/`math_owm_stage2`, paired-only `en`/`math`), which is the
  per-role behaviour working as designed, not a failure.
- **`replay_cursor` takes its mix from the checkpoint.** There is no `--mix` flag and there
  should not be one: the run's mix is recorded in the artifact, so the tool cannot be aimed
  at the wrong mix. My first runbook draft told operators to pass `--mix`; corrected before
  publishing.
- **`_pin_milestone` sits outside the pruner's glob.** `train.py:2091` globs
  `ckpt_<run>.pt.step*`; the pin is `ckpt_<run>.milestone_<token>_step<N>.pt`, which has no
  `.pt.step`. The hardlink means the pruner's `os.remove` of the `.step` name drops one
  link and the inode survives.

## Also fixed this round (queue items 2-4, a78d666 / d16c280)

`no_foreground_pod_training` read the pod once instead of once per training process:
6.3 s at 11 ranks to 0.59 s, same verdict, and the 15 s deadline override is deleted.
A deadline hit is now `TIMEOUT`, not `SKIP` -- the hook prints nothing for a SKIP and
`check` exits 0, so a check that timed out on every run was a permanent silent pass
(44's D5); the second consecutive strike FAILs. `cited_artifacts_attested` matches
`(path, sha256)` rather than the hash alone (tilerl T7-1): hash-only accepted an
attestation of a different path carrying the same bytes, which is the normal state during
a versioned rerun rather than a coincidence.

Two of my own broken worlds went green under the 6 h threshold change and `--selftest`
caught both. `_broken_dirty_aged` hardcoded 2 hours -- aged under the old 30-minute
threshold, not aged under 6 -- so its age now derives from `_AGE_HOURS`. That is the
two-offender rule proving itself in miniature: a constant and a hand-written copy of it
are two sources of truth, and only one of them moved.
