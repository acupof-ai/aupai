---
question: What instrumentation localizes the CI-only `gate_resume_equivalent_to_uninterrupted` red, and how is each candidate distinguished?
status: open
source: PR #549 (branch 0e-ci-resume-diag), tests/v41f/diag_resume_bimodal.py, tests/v41f/test_p1_train_ckpt.py (_one_run), v41f/master.py
---

# Resume-gate divergence instrumentation

Spec for 0e to implement on #549. de owns the spec, 0e implements, de second-reads.

## What is already known, and what is not

The required gate is `tests/v41f/test_p1_train_ckpt.py::gate_resume_equivalent_to_uninterrupted`.
It runs three trajectories in **separate `subprocess.run` invocations** (control, restart, fresh),
each seeding `torch.manual_seed(123)` in `_one_run` before `_build(cfg)`, and compares the fp32
master and bf16 run weight of one probed tensor with `torch.equal`.

Line numbers below are the **#549 branch's** (`0e-ci-resume-diag`); they differ by ~15 from `main`
because the branch adds the diagnostic. Functions are named so a rebase does not rot the citation.

The failure signature, identical across every observed red:
`max|delta|=1.334e-2`, `n_diff=506533/524288` (96.6%), `first_flat_idx=0`, `n_nan=0`. The same blob
is green on the digest host (14/14) and flips RED then GREEN across reruns of one sha on GitHub
runners.

**The thread/BLAS-reduction hypothesis is dead.** Measured 2026-09-19, two independent processes per
arm, seed fixed before construction: seeded `OMP_NUM_THREADS` 1 and 2 both give
`n_diff=0/524288, torch.equal=True, sign-flip=0`; only an *unseeded* control differs
(523850/524288, 262320 sign-flips, max 5.393e-4). That earlier number measured init divergence, not
reduction order — `model.py`'s init consumes the global RNG, so an unseeded rebuild gives every
process different weights. The gate seeds correctly; its premise is sound.

So the surviving hypothesis is a **real asymmetry between the two trajectories**, and 0e's existing
`diag_resume_bimodal.py` is the right instrument family. What it does not yet cover is specified
below.

## The red is TWO-STATE, proven on the same run (2026-09-22)

**The same GitHub run, on the same tree, with the same event, both fails and passes.** Run
`35603567150` (sha `5171ad89`, `event=pull_request`) had its `check` job FAIL with the signature
below; after `gh run rerun --failed` the identical run completed **success**, and its log carries the
gate's own PASS line:

```
resume: save/load mid-run bit-identical to control; fresh optim diverges
```

The gate therefore **executed and passed** — it was not skipped, so this is not a missing run.

**This retracts the 2026-09-19 ruling** (recorded as `facts/v41.json#v41.d17_two_state_not_save_load_0922`,
`status: retracted`), which is also the working hypothesis in `.github/workflows/ci.yml`'s diag-resume
header. The case against that ruling is now three independent lines, and this is the first that is a
controlled comparison rather than an inference:

1. the dump shows the save→load round trip bit-exact at all 5 leaves (section above);
2. the dump shows the optimizer and RNG families equal across arms, leaving only the weight family
   untested (§ below);
3. **this rerun shows the outcome is not a function of the tree at all.**

**An earlier attempt at this claim was wrong and is worth keeping as the trap it was.** I first read a
push run FAIL and a pull_request run SUCCESS on one sha as two-state; they had checked out *different
trees* (`raw refs/heads/*` vs `+refs/pull/619/merge`, since `actions/checkout@v4` has no `ref:` and a
PR run takes the synthetic merge ref). `gh run list --commit` reports one snapshot sha for both, which
is what hid it. **A two-state claim needs the checked-out sha per run, not the run-list sha** — and
the way to hold every variable fixed is to rerun the SAME run id, which is what settled it here.

**No mechanism is claimed.** Which runner property flips it is unknown: thread/BLAS order, load,
oneDNN and an unseeded init were each falsified earlier, and the `setsid` and time-to-fork hypotheses
were falsified for the adjacent `card_claim` world control. What is established is only that a green
and a red can be the same code on the same tree.

**Consequence for the gate, and for the launch.** A two-state red is not evidence about the change
under test, so a push whose only failure is this signature is **not** a reason to hold a merge. It is
also **not** grounds to keep rerunning until green: the correct read is that the check is
runner-dependent, and the falsifiable question moves to the **master-weight family**, which no red has
ever recorded — `#624` (merged `c13e9d06`) dumps exactly those two tensors, and **the next red on a sha
carrying it decides that family.** The launch does not wait on it.

### Adjudication: what a code-free PR may do with this red (2026-09-22)

**This is the operative rule, and it exists so that a red with no relation to the diff does not block
unrelated work.** It applies **only** to a failing `check` job whose failing step is `train checkpoint
gates`, whose only failing assertion is `gate_resume_equivalent_to_uninterrupted`, and whose reported
signature is byte-identical to the one above (`max|delta|=1.334e-02`, `n_diff=506533/524288`,
`first_flat_idx=0`, `n_nan=0`).

**What does NOT license a pass.** Two arguments that have been used are both invalid, and the second
is the more dangerous because it cites this very file:

- *"the clean physical host is 14/14 green"* — that is an environment difference, and the whole
  finding here is that the runner and the physical host disagree. Citing it as evidence that a
  runner red is spurious assumes the conclusion.
- *"this job is never a gate"* — `.github/workflows/ci.yml` says that about **`diag-resume`**
  (`:153-166`, a `continue-on-error` job that push/PR skip entirely). Its last two lines say the
  opposite about the job that actually fails: *"The required check job above keeps the real gate
  fail-closed; this job exits 0 by construction."* **The failing job is the required one.**

**The procedure.** A PR that touches no code (docs, ledgers, facts) and whose CI is red *only* under
the paragraph above may proceed after its second reader records, **in the review row**, that they
checked all four conditions in the paragraph above against the run's own log. The row must name the
run id. A PR touching `tests/v41f/`, `v41f/`, `train.py` or the workflow itself does **not** qualify —
there the red is in scope until the weight family is measured.

**The residual, stated as a residual.** This does not say the red is harmless. The master-weight
family is still unmeasured, and until a red uploads `#624`'s two tensors **nobody can say the two
trajectories agree on the weights themselves** — only that the optimizer state, the RNG and the
tree are not the difference. If the weight family later turns out to diverge, this rule was the wrong
call and the PRs it passed need re-examination; that is the cost being accepted, and it is bounded
because the affected merges are docs and ledgers, which no gate re-reads.

**Next step, and it is the only one.** The next red on a sha carrying `c13e9d06` decides the family.
Nothing further should be specified before that artifact exists.

## The first red-after-instrumentation, and what it falsified (2026-09-21)

Run `35603567150` (sha `5171ad89`) failed the required gate on the runner and uploaded
`gate-resume-dump-35603567150` (432,361,251 bytes, 69 files, three arms, 5 leaves). Read on the
laptop with `torch.load(weights_only=False)`. **This is the artifact the `#549` instrumentation was
built to produce, and it removes half the live candidates:**

| comparison | result |
|---|---|
| `restart/loadK.*` vs `restart/optK.*` — the save/load round trip | `max\|delta\|=0.0`, **0 differing elements** on `exp_avg`, `exp_avg_sq`, `step`, at all 5 leaves (l0 `exp_avg` 0/13,107,200; l4 0/524,288) |
| `ckpt_identity.json` | `populated_but_dropped=[]`, `model_missing_keys=[]`, `model_unexpected_keys=[]`, 228 `state_by_name` keys |

**The row above is the whole load-bearing result, and the other two comparisons the dump invites you to
make are NOT evidence.** Each arm is a separate process, but all three run identical code from
`torch.manual_seed(123)` through `_build` and the per-batch generator
(`tests/v41f/test_p1_train_ckpt.py:356-357`, `:346`), and the `optK`/`rng_atK` dump at the `i == k`
loop head (`:374-380`) is reached **before every arm-specific branch** — the save/load for restart and
fresh (`:381-389`), the new-optimizer swap at `:390-391`, restart's `loadK` dump at `:392-395`. So the
cross-arm equalities a reader would naturally tabulate —

- `fresh/optK` vs `restart/optK` → `max|delta|=0.0`
- `rng_atK` identical across control, fresh, restart

— are **construction guarantees**: the same state read three times, not three independent
trajectories agreeing. A *nonzero* value there would have been the finding. Citing them as
corroboration asserts "two independent runs confirm the same value" when there is one run and three
reads, which is the shape `docs/lessons` files under an aggregate that cannot fail on what it hides.
`ckpt_identity.json` is the exception among the auxiliary records: it compares control's *pre-save*
populated set (`:384`) against the **restart** arm's loaded blob, so it is arm-specific and does carry
information.

**Candidate 2 (name re-bind) and candidate 3 (fp32-alias break) are falsified**, and so is the
optimizer arm of candidate 1: after the load, every sampled leaf's AdamW triple equals its
own pre-save value, bit for bit, across the boundary the two arms actually differ at. The round trip
preserved the optimizer exactly.

**The RNG dump found a difference, and it is not yet a cause.** `restart/rng_preSave.pt` vs
`restart/rng_postLoad.pt` differ at 2,488 of 5,056 bytes. The worker writes `preSave` *before*
`save_train_checkpoint` and `postLoad` *after* `load_train_checkpoint`, and the round trip itself
consumes global RNG, so this is expected wherever training consumes none. §4 predicted a no-op and
got a difference; **which of the two readings is right is undetermined.** Do not report it as the
cause. The measurement to settle it is the same one §4 asks for, now with a known-answer value: dump
RNG at a point where *nothing* has touched it between the two reads, and compare.

**What the dump cannot answer, and this is the load-bearing gap.** It records optimizer moments, RNG
bytes, key names and shapes, and **no model or master weight values anywhere** — every `.pt` with
`numel > 1e6` is an `exp_avg`/`exp_avg_sq`. So when the gate reports `n_diff=506533/524288` on the
**fp32 master**, this artifact is blind to it. What is excluded is the optimizer/rebind/RNG family;
the **master-weight family is untested**, not excluded: the fp32→bf16→fp32 path, `refresh_bf16`, and
the save-time dtype truncation assertions at `v41f/master.py:246-249` were never observed by any
artifact from this run.

**`#624` (merged `c13e9d06`) closes exactly that gap** — it dumps the two fp32 master tensors and the
bf16 run weight that actually differed, plus a `call_site`/`values` record. **No red has uploaded one
yet.** The next red on a sha that carries `c13e9d06` is the decisive artifact; nothing further should
be specified until it exists.

## Candidate causes, and the measurement that separates them

Four candidates. Each has a distinct expected signature, so the instrumentation is diagnostic, not
merely descriptive.

| # | Candidate | Predicted signature |
|---|---|---|
| 1 | **Never-persisted state** — something the control trajectory carries that the save/load round-trip drops | restart diverges from control **beginning at the first post-load step**, and the divergence is present in the same leaf at the same step across reruns |
| 2 | **Name re-bind error** — optimizer state attached to the wrong tensor after load | the divergence is **large and structured** (a permutation or whole-tensor swap, not a spread), and `gate_optim_named_roundtrip_and_reorder`'s per-name `torch.equal` checks stay green while the trajectory differs |
| 3 | **fp32-alias break** — an fp32-native master no longer sharing storage with its model Parameter post-load | `alias_*` audit reports a non-empty `broken_alias`; divergence appears in an fp32-native leaf, not a bf16 one |
| 4 | **Non-deterministic kernel on the runner** | divergence does **not** correlate with the load boundary; control and restart differ *before* the checkpoint step |

Candidate 4 is already excluded for the seeded case by the measurement above, and
`diag_resume_bimodal.py --diag-arm` re-tests it per run.

**Status after the 2026-09-21 dump (section above): candidates 2 and 3 are FALSIFIED, candidate 1's
optimizer arm is falsified.** Candidate 1 remains live only in its weight-carrying form —
something about the fp32 master or the bf16 run weight does not survive the round trip. Stated as
the open question rather than as a hypothesis with a mechanism: **no artifact has yet recorded a
weight value from a red run, so nothing is known about which weights differ, by how much, or at
which leaf.** `#624` is the instrument for that question and it has not yet fired on a red.

### One mechanism already excluded, and what the exclusion bought

Tested 2026-09-19 on a detached copy of `main`: make `_save_optim_named` silently `continue` past
one *populated* param (the silent branch in §3), then run the gate and read its own reported
signature.

| leaf whose state was dropped | `max|delta|` | `n_diff` |
|---|---|---|
| `layers.0.…qproj.wq_b.weight` (the probed leaf) | 3.325e-02 | 524288/524288 |
| `layers.0` (other leaves) | 3.300e-02 | 524288/524288 |
| `layers.3` | 1.521e-02 | 523668/524288 |
| `head.weight` | 1.479e-02 | 522632/524288 |
| `embed.weight` | — | **no divergence** |
| CI, for comparison | **1.334e-02** | **506533/524288** |

A single silently-reset leaf is therefore **not** the explanation of the CI numbers: three
informative rows land at 99.7-100% against CI's 96.6%, and `max|delta|` moves by more than 2x with
the leaf choice. The mechanism predicts 0% (a leaf not feeding the probe) or ~100% (one that does);
96.6% is neither. It remains live as a possible cause of a *differently-shaped* red.

Two requirements follow, and they are not optional:

1. **Record per-leaf contribution, not the probed tensor's aggregate.** Each dropped leaf produces a
   measurably different `(max|delta|, n_diff)` pair, so a signature taken from a real red is
   evidence about *which* state diverged. Aggregating over the probed tensor discards exactly that
   inference. Emit, per leaf and per comparison point, its own `max|delta|` and `n_diff`.
2. **`embed.weight` is an open question for the instrumentation, not a footnote.** It carries real
   optimizer state (`exp_avg` nonzero, `step=2.0`, bf16-native, in-group) and skipping it on save
   produced **zero** divergence in the probed leaf, while skipping `head.weight` produced 99.7%.
   The cause is not established. A plausible account is that the probed leaf's gradient path does
   not reach `embed.weight`, but that is a hypothesis to *test*, not to assume: dump whether each
   leaf's restored optimizer state equals its control-side counterpart, so "this leaf's state was
   lost" and "this leaf's loss was invisible to the probe" are separate, recorded facts. Do not
   explain the anomaly away in the write-up; it is a question the dump must answer.

## Required instrumentation

Extend `diag_resume_bimodal.py`. Do not build a second diagnostic.

### 1. Dump the optimizer state, not only the master

`arm()` currently dumps per-step forward logits, backward grads, and post-step master tensors for a
sampled leaf set. It dumps **no optimizer state at all** (`grep -c exp_avg` = 0). Candidate 2 lives
entirely in state the current dump cannot see.

**The dump must be reachable from the required gate's worker, and today it is not.** The required
gate does not call `arm()`: `gate_resume_equivalent_to_uninterrupted` spawns its own worker
`_one_run(kind)` (`test_p1_train_ckpt.py`), which writes the two probe tensors to **stdout and
nothing else** — `grep -c 'OUTDIR|outdir'` on that file is 0, so no directory plumbing exists in the
code path a red actually runs. Instrumentation added only to `arm()` therefore uploads nothing on
the one run the artifact exists for.

Required: `_one_run` writes its dump to a directory named by an **environment variable the parent
sets** (the gate already builds `env = dict(os.environ, OMP_NUM_THREADS="2")` for its subprocesses),
falling back to no-dump when the variable is unset so a plain local run stays cheap. The parent
creates that directory under its own `_scratch` root and the CI step uploads it. Point 1 is
load-bearing for the whole spec: without it the artifact step below uploads an empty directory on
every red.

For every leaf in `_pick_leaves`, dump:

- after `optimizer.step()` at the checkpoint step K **and** the first post-load step only — not every
  step. Measured cost, on 0e's numbers which I confirmed (`embed.weight` and `head.weight` are
  50.0 MiB each as fp32, 13,107,200 elements): per-step dumping of four leaves x 2 arms x 4 steps is
  ~1.7 GiB, against ~0.42 GiB for the two points. The **onset** question is already answered by the
  per-step master tensors `arm()` dumps, so exp_avg at two points loses nothing.
- `exp_avg`, `exp_avg_sq`, `step` (all three, not just `exp_avg`)
- and after the load, the same three **as restored**, under a `loadK.opt.l{j}` name

The load-side dump is what makes candidate 2 separable: a re-bind shows as restored-`exp_avg`
matching a *different* leaf's control value, which is visible only when both sides are dumped.

### 2. Dump the checkpoint's own identity

**Fix the leaf sample to be deliberate, not incidental.** `_pick_leaves` takes `names[0]`,
`names[len//2]`, `names[-1]` from the sorted in-group list, then adds `head.weight` and
`layers.0.attn.qproj.wq_b.weight`. Measured on `v41f_small(indexer_train_mode="off")`,
`names[0]` **is** `embed.weight` — so `embed.weight` is sampled today only because it happens to
sort first alphabetically, and its presence in the dump is an accident of naming. The `embed.weight`
open question below depends on it being covered, so add it to the explicit `want` tuple alongside
`head.weight` and the probed leaf.

Per trajectory, record:

- `sha256` of the checkpoint file bytes (control has none; restart's is the artifact under test)
- the `param_names` list as saved, and the per-name `shape`/`dtype` records from
  `optim_named.state_by_name`
- `state_by_name` **key set** versus `param_names` — the asymmetry check below depends on this

### 3. Assert (not merely print) the save/load set relation

This is the silent branch the spec exists to make loud. `_save_optim_named` skips any param whose
AdamW state dict is falsy (`if not st: continue  # never stepped`), and `_load_optim_named` treats a
name absent from `state_by_name` as `if rec is None: continue` — i.e. **"never stepped"**. So
"the saver dropped this param's state" and "this param never had state" are the same state to the
loader, and a save-side omission resumes silently with a fresh optimizer for that tensor.

**"Silent" means nothing on the load path raises — not that the run stays green.** Measured
2026-09-19 on a copy that drops one populated param: the `--run restart` worker exits **rc=0**
(the loader takes its `rec is None` branch and says nothing), while the full gate exits **rc=1**.
The trajectory diverges because the optimizer really did lose that tensor's state. So this is not a
hypothesis about a failure mode that currently hides; it is an *unattributed* failure — the gate is
already red and nothing names why. That is the whole job of this instrumentation, and it is why the
presence assertion below is a requirement rather than a nicety.

The wording matters because the wrong reading is the opposite of the truth: "nothing raises"
describes the reader, and a reader that stays quiet while the gate goes red is exactly the gap
between a symptom (a red gate) and a cause (this param lost its state).

Required: on the load side, assert that every in-group name whose control trajectory had non-empty
optimizer state at step K is present in `state_by_name`. A name that is absent while its
control-side counterpart was populated is a **FAIL naming that param**, not a silent `continue`.
This is an assertion in the diagnostic only — do not change `v41f/master.py` in this PR; the
diagnostic must first produce the evidence that the loader needs a refusal.

### 4. RNG state across the boundary

Dump `torch.get_rng_state()` before the checkpoint and after the load, both arms, and compare. The
training path currently consumes no RNG (`v41f/train.py` has no `rand`/`dropout`/`manual_seed`,
and `_last_loss_terms` is the only module-level mutable), so this should be a no-op — which is
exactly why it is worth pinning: it converts "I read the code and saw no RNG" into a measurement,
and it catches a future step that adds one.

**Measured 2026-09-21, and the prediction above was wrong as stated.** `rng_atK` is byte-identical
across all three arms — but per the section above, that is a **construction guarantee**, not a
measurement of seeded agreement: the dump sits ahead of every arm-specific branch, so there is one
state read three times. It carries no information about the arms. The informative pair is
`restart/rng_preSave.pt` vs `restart/rng_postLoad.pt`, which differ at **2,488/5,056 bytes** and are
on opposite sides of the save/load boundary within a single arm. The written reading still does not
hold: those two dumps straddle `save_train_checkpoint` + `load_train_checkpoint`, so a difference is
expected even where training consumes no RNG. **"The RNG state does not survive the round trip" and
"the round trip itself draws from the RNG" are both consistent with this pair, and the dump cannot
separate them.** A speculative mechanism (save/load consuming RNG) is recorded as a *possibility*,
not a finding: the competing reading — that the round trip draws nothing — is not excluded by any
measurement. **The discriminating measurement is a third dump with nothing between the two reads**;
until it exists, report this as an undetermined difference and never as the cause.

### 5. Missing / unexpected keys on load

Record `model.load_state_dict(...)`'s `missing_keys` and `unexpected_keys`. `load_train_checkpoint`
currently calls it with `strict=True`, so both must be empty; record them anyway. A strict load that
did not raise is evidence about *names*, not about *values*, and this makes that explicit.

## CI wiring

- The required `check` job keeps the gate **fail-closed**. No change to its semantics.
- The diag step in the required job stays `--diag-selftest` and cheap (it proves the comparator is
  non-empty; that is its whole job there).
- **Add an `upload-artifact` step that runs on failure of the required gate**, uploading the dump
  directory. Today the dump is written into a `tempfile.mkdtemp(prefix="td_eq_")` that nothing
  preserves (`test_p1_train_ckpt.py:319`), and the gate's own failure message carries only the
  scalar signature — so a real red on the runner leaves the runner with nothing to inspect. The
  `diag-gate-tracker` job's `upload-artifact` covers only its own `--gate-once` sample, not the
  required gate's red.

  Gate it on the gate step's outcome and use `if-no-files-found: ignore` so a green run uploads
  nothing. **This step is the deliverable**: the earlier plan was to wait for a natural red, but a
  red that uploads no artifacts cannot distinguish the four candidates, so waiting without this
  step buys only the signature we already have.


- Everything added stays non-blocking: `continue-on-error` on the diag job, no change to the
  required gate's pass/fail.

## Acceptance

1. `python tests/v41f/diag_resume_bimodal.py --diag-selftest` passes, and the diagnostic **reds on a
   deliberately broken input**. Pick the mutation carefully — two obvious ones are non-discriminating:

   | mutation | what happens today |
   |---|---|
   | load-side rebind: re-bind one leaf's `exp_avg` to its neighbour in `_load_optim_named` | **already caught** by `v41f/master.py:177-184`, which asserts per name that restored `exp_avg`/`exp_avg_sq`/`step` equal the saved record (`raise OptimStateError(f"{n}: exp_avg not restored to the same name")`). Red here proves an existing assertion fires, not that the new instrumentation works. |
   | save-side mislabel **across different shapes** (`embed.weight` -> `norm.weight`) | **already caught**, by the shape check at `master.py:155` (`norm.weight: shape (1024,) != saved (12800, 1024)`) — a name check is never reached. |

   Use a save-side mislabel **between two same-shaped leaves** instead: on a copy, relabel
   `layers.0.attn.qproj.wq_b.weight`'s optimizer record as `layers.3.attn.qproj.wq_b.weight`.
   Measured 2026-09-19, and the pair is the point:

   | invocation | rc |
   |---|---|
   | `--run restart` (the worker alone) | **0** |
   | `--gate gate_resume_equivalent_to_uninterrupted` (the whole gate) | **1**, `max|delta|=3.330e-02 n_diff=524288/524288` |
   | the gate on the unmutated tree | 0 |

   The gap is in the reader, not the writer. The worker exits 0 because nothing on the load path
   notices: the label and the state move together, so the shape check (`:155`) sees matching shapes,
   the name-set check sees both names in-group, and the identity loop at `:177-184` compares
   `opt.state[master[n]]` against `sbn[n]` — the *same* relabelled key on both sides, so the
   comparison is self-consistent and cannot disagree. The hole is in `_load_optim_named`'s read
   side: it binds whatever `state_by_name[n]` holds to whatever `param_names` calls `n`, with no
   independent record of which tensor the values came from. The gate nonetheless goes red, because
   the relabelled state does change the trajectory — which is exactly the CI failure being
   investigated.

   **So the acceptance is a PAIR: worker rc=0 AND gate rc=1.** An earlier revision of this spec said
   only "the gate must be rc=0", which was wrong and was wrong for an instructive reason: it was
   written from a run of `--run restart` alone, without ever running the full gate under the mutant.
   The diagnostic's job is to make an *existing* red attributable — not to manufacture one. A spec
   that asks the diagnostic to redden a green gate has mistaken the instrument for the fault.
2. On a green run, the dump contains `exp_avg`/`exp_avg_sq`/`step` for every sampled leaf, both
   trajectories, plus the checkpoint sha and the key-set comparison.
3. The per-param presence assertion fires on a copy where `_save_optim_named`'s `if not st: continue`
   is changed to drop one populated param.
4. **Per-leaf contribution is present and discriminating.** On that same copy, the dump contains a
   per-leaf `(max|delta|, n_diff)` for every sampled leaf, not only the probed tensor's aggregate,
   and dropping *different* leaves produces *different* recorded numbers. Reuse the table above as
   the known-answer set: a build whose per-leaf output is identical for `head.weight` and
   `layers.0` has collapsed the discriminator the spec exists to provide.
5. **The `embed.weight` question is answered by data, not by an assumption.** The dump records, per
   leaf, whether its restored optimizer state equals its control-side counterpart — so "state lost"
   and "state lost but invisible to the probe" are separately readable.
6. Both directions of the ordering are asserted the same way #564 does it: read the fact out of the
   source, do not infer it from printed output.

## What this spec does not claim

It does not claim the cause is candidate 1 or 2. It makes each candidate's signature separable and
makes the red run carry its own evidence. It **excludes** one specific mechanism: a single silently
reset optimizer leaf does not produce CI's 96.6%/1.334e-02 (measured, table above). If the first
red-after-merge shows candidates 1-3 all clean with the divergence present before the load boundary,
the conclusion is that the seeded determinism does not hold on that runner, and the measurement to
run is the seeded cross-process pair from the table above **on that runner**, not on the laptop.

---

## Gate redesign: judge at the reproducible granularity (2026-09-22, D61/K6)

The Adjudication above unblocked code-free PRs but did not fix the gate. The red-run artifact
for `35677058625` (sha `d1d4aef3`, #624 dump) located the cause and named the wrong contract.

**Root cause, measured.** `gate_resume_equivalent_to_uninterrupted` bit-compares two
SEPARATELY LAUNCHED processes' fp32 master. At the save boundary the two processes are
identical (control `optK` == restart `optK` bit-for-bit on all five sampled leaves; restart
`optK` == `loadK` across the save/load; no dropped/missing keys), yet the two post-resume
steps diverge. train_step has no dropout or training RNG (only seeded weight init), so the
only varying input is the runner's bf16 forward/backward kernel numerics (atomic
reductions); bf16 gradients differ process-to-process, are cast into the fp32 master, and
AdamW compounds the difference. The fp32 master deliberately retains sub-bf16 information
that an independent process cannot reproduce.

**The granularity finding (criterion 70).** Per-coordinate equality is NOT reproducible;
aggregate/geometric equality IS. Measured on the failing run's probe
(`layers.0.attn.qproj.wq_b.weight`): fp32 master 96.6% of elements differ; after a bf16 cast
51.5% still differ (ulp gap up to ~6e5, near-zero coordinates random-walk so per-element
relative error is meaningless). But the tensors agree as objects: relative-L2 ~1.6%,
cosine ~0.99987, sign agreement 99.8%. Same training result via a coordinate-divergent
path. Neither fp32 nor bf16 elementwise `torch.equal`/`allclose` is the right predicate.

**The replacement gate — `scripts/resume_equiv_gate.py` (K6 harness).** Two layers:

1. **Deterministic state — bit-exact, hard fail.** What a checkpoint actually promises and
   kernel nondeterminism cannot affect: optimizer triple (`exp_avg`, `exp_avg_sq`, `step`)
   identical `optK`==`loadK` per leaf; no populated leaf dropped from the saved key set;
   no missing/unexpected model keys. Any mismatch is a real save/load bug
   (`rc=1`).
2. **Training trajectory — calibrated whole-tensor bounds, not per-element.** For fp32
   master AND bf16 run weight, compare relative-L2 (`||a-b||/||a||`), cosine, and a
   near-zero-aware RMS floor (RMSE/reference-RMS) (`rc=2` when outside bounds). Bounds come
   from a calibration JSON built from paired HEALTHY control/restart runs on the RUNNER
   (the noisy environment the gate executes in), widened by a documented margin; they are
   never taken from one red run or guessed. Without a calibration the layer reports
   UNCALIBRATED and does not invent a tolerance (`rc=4`, or a loud pass only with
   `--allow-uncalibrated`).

**Calibration is deferred until after CED.** The flat forward is being deleted; its
bf16-reduction shape (and the new W_KV/W_Z path) changes the green-noise distribution, so
bounds calibrated now would be stale. Sequence: CED forward + smoke pass → run several
paired control/restart samples on the runner under 08's GPU authorization → compute bounds
at a ~3-5x margin over observed healthy noise with provenance (runner, sample count,
margin) → wire `resume_equiv_gate.py` into CI in place of the bit-exact assertion, same PR
as this doc's update. The W_KV/W_Z save/load check uses exact-state + bf16-tolerant weight
bounds, never fp32 bit equality.

**Mutation contract (the gate must be lenient but not blind).** Demonstrated in the
harness selftest and against real/synthetic dumps:

- drop a populated leaf, or corrupt one optimizer tensor across the boundary → layer 1
  `rc=1`;
- current runner bf16 noise within calibrated bounds → `rc=0`;
- a fresh re-init, or a localized real regression (one block scaled ×1.1) → layer 2
  `rc=2` — the bound must catch a true localized bug, proving it is not vacuous.

Calibration samples (healthy pairs) and mutation inputs (injected harm) must never be mixed:
the bounds describe observed healthy noise, and the mutations validate them from outside.
