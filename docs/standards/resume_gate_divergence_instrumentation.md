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
`diag_resume_bimodal.py --diag-arm` re-tests it per run. Candidates 1 and 2 are the live ones.

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

For every leaf in `_pick_leaves`, dump after each `optimizer.step()`:

- `exp_avg`, `exp_avg_sq`, `step` (all three, not just `exp_avg`)
- and after the load, the same three **as restored**, under a `loadK.opt.l{j}` name

The load-side dump is what makes candidate 2 separable: a re-bind shows as restored-`exp_avg`
matching a *different* leaf's control value, which is visible only when both sides are dumped.

### 2. Dump the checkpoint's own identity

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
loader, and a save-side omission resumes silently with a fresh optimizer for that tensor. Nothing
raises.

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

1. `python tests/v41f/diag_resume_bimodal.py --diag-selftest` passes, and a **deliberately broken
   optimizer restore** (on a copy: re-bind one leaf's `exp_avg` to its neighbour in
   `_load_optim_named`) makes it red by name. A diagnostic that cannot red on a known-broken input
   is not evidence.
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
