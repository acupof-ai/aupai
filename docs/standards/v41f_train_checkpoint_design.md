# v41f fp32-master + optimizer resume checkpoint — design (P0, design only)

Status: PROPOSAL for fb + de review. No implementation in this change. Scheduled after
engram (de) and MTP (0e) land to avoid editing v41f/model.py while they do. This is the
continuation explicitly deferred by #441 (`v41f/ckpt.py`: "fp32 master/optimizer state is
a training concern and is not part of this model checkpoint").

Scope of THIS checkpoint:
- bf16 model weights (the run/forward dtype),
- an fp32 master copy of every trainable parameter the optimizer updates,
- AdamW state (`exp_avg` m, `exp_avg_sq` v, per-param `step`),
- the V41FConfig, a schema/format version, and a vocab fingerprint.
Out of scope (named, later): LR-scheduler/RNG-state/data-cursor restoration, distributed
sharded checkpoints, fp8 expert weights. It closes the local single-process trainable loop:
a run that saves after k steps and a run that loads and runs step k+1 must be numerically
indistinguishable from one uninterrupted run.

---

## 0. Why a separate design (the #441 lesson)

#441's first resume test check-pointed a model that had been `.float()`-ed (fp32 master
simulated) but `load_checkpoint` builds under the **bf16 default dtype**, so 190 bf16-native
parameters were silently rounded back to bf16 (worst key 3.1e-3, 0.053 logits). The
ratio-only loss assertion (`post < pre*0.2`) passed at 0.0000 on both sides. Two structural
gaps caused it:
1. The on-disk dtype was implicit — it came from whichever default dtype the loader ran in.
2. Nothing paired each saved tensor with the dtype/role it must be restored with, so a
   bf16-build load of fp32-master tensors "succeeded" under strict key matching.

The new checkpoint makes dtype and role **explicit per tensor group**, and refuses any path
that would narrow fp32 master state.

---

## 1. On-disk layout and grouped, atomic, strict save/load

### 1.1 One file, named groups, explicit dtypes

```
{
  "format": "v41f_train_ckpt",
  "version": 1,
  "config": { ...V41FConfig asdict... },
  "vocab_id": "<sha256 of tokenizer, same convention as the v41 gate>",
  "param_meta": { name: {"dtype": "bfloat16"|"float32", "group": "bf16"|"fp32_native"|"master"} },
  "model":        { "<name>": tensor, ... },   # run/forward weights, native run dtype
  "master_fp32":  { "<name>": tensor, ... },   # fp32 master for EVERY updated param
  "optim":        {
      "state": { "<param index>": {"step": int, "exp_avg": t, "exp_avg_sq": t} },
      "param_groups": [ {lr, betas, eps, weight_decay, ...} ],
  },
  "step": <int global update count>,
}
```

Dtype groups (derived from how v41f actually constructs parameters — verified on main):
- **bf16-native** (`group="bf16"`, run+master differ): embed, all attention linears
  (q/kv/wo_b), grouped wo_a, RMSNorm weights, MoE gate.weight, all routed+shared expert
  weights, compressor weights. These are created under the bf16 default dtype.
- **fp32-native** (`group="fp32_native"`, run IS fp32): `head.weight`, the six HyperConn
  tables (`hc_attn_fn/hc_ffn_fn/hc_attn_base/hc_ffn_base/hc_attn_scale/hc_ffn_scale`),
  `attn_sink`. For these `model[name]` and `master_fp32[name]` are the same value/dtype.
- **buffers, not in master/optim**: the persistent fp32 `gate.bias` (selection-only), and
  non-persistent runtime caches (window/compress kv, freqs_cis) which are rebuilt, never
  saved. hard-topk indexer params are real parameters with no CE grad today (see §4 note);
  they are not in the optimizer group (the master criterion, per fb ruling), so they get bf16 weights + param_meta grad=False and NO master/m/v.

`param_meta` is generated from the live model, never hand-written, and is the contract the
loader validates against (§2). Key set must be identical across `model`, `master_fp32`
(except none — master covers all trainable params), and `optim.state` indices.

### 1.2 Atomic group write — no reader ever sees a torn/half group

A single file is one rename unit; the hazard is a crash during write leaving a truncated file
that an older/newer loader half-reads. Use the repo's established staging discipline
(scripts/build_agentic_sft.py:1775-1826, build_p1_tokenizer.py:198-243):
1. write to `path + ".tmp.<pid>"` in the SAME directory (same fs → os.replace atomic);
2. after the last tensor is written and the file is closed, `file.flush(); os.fsync(fd)`
   (fsync is for machine crash — labeled "not covered by selftest", exactly as the repo
   already does, NOT claimed tested);
3. `os.replace(tmp, path)`.
Optionally `fsync(dirfd)` of the parent after replace for directory-entry durability.
On load, a `.tmp.*` sibling is never opened; a failed save leaves only the prior good file.
(Decision needed from de: one `.pt` file vs a directory `ckpt_step_N/{model,optim}.pt` +
manifest. Single file is simpler and matches current `save_checkpoint`; directory makes
sharded/partial writes explicit later. Recommend single file at P0, version field leaves
room to migrate.)

### 1.3 Strict pairing on load

- Rebuild config from the blob; ANY V41FConfig mismatch → fail (same as #441).
- `vocab_id` present and equal to the loader's tokenizer, else refuse (checkpoints and packs
  already fail on vocab_id mismatch: scripts/check_sft_ready.py). Missing vocab_id is a
  loud error for a train ckpt, not a warning.
- `version` must be understood; unknown → loud error with the version seen.
- Rebuild the model in a DTYPE-NEUTRAL way (see §2), then:
  - `model.load_state_dict(blob["model"], strict=True)` — missing/unexpected key raises;
  - verify every `master_fp32` key exists and, after the optimizer owns master (§2), strict;
  - rebuild the optimizer on the SAME parameter list/order, then
    `optimizer.load_state_dict(blob["optim"])` (this restores m/v/step);
  - assert optim state index set == model parameter index set.

---

## 2. Structurally eliminating the silent fp32→bf16 truncation

The truncation is impossible-by-construction, not merely untested:

1. **Never rely on `torch.get_default_dtype()` during training load.** `load_checkpoint`
   today sets bf16 default then builds. Introduce an explicit dtype plan: build the bf16 run
   model (same as today), then create the fp32 master as an EXPLICIT
   `{name: p.detach().float().clone()}` map owned by a small `MasterState`/optimizer wrapper —
   never by casting the whole module (`.float()` is what made every parameter look fp32 and
   hid the grouping).
2. **Master is the source of truth the optimizer updates; bf16 weights are a downcast
   projection for forward.** Concrete training-time object (name TBD by de):
   - `master: dict[str, nn.Parameter(fp32)]`, one entry per trainable param;
   - optimizer is constructed over `master.values()` (AdamW sees only fp32);
   - before forward: `model_param.data.copy_(master[name].to(bfloat16))` for bf16-native
     params; fp32-native params share/alias master directly (no copy);
   - after backward: grads are computed against bf16 weights; cast each grad to fp32 into
     the corresponding master param (`master.grad = model_param.grad.float()`), then
     `optimizer.step()` advances master; bf16 model is refreshed on the next forward.
   This is the standard bf16-mixed-with-fp32-master pattern and matches train.py's stated
   contract ("production runs bf16 forward with fp32 master weights; the optimizer owns the
   fp32 copy").
3. **Save master from `master.values()` (always fp32 by construction), save model weights
   from the bf16 run model.** A save asserts `master_fp32[name].dtype==float32` for every
   entry and `model[name].dtype == param_meta[name].dtype`; violation raises before writing.
4. **Load rejects mixed/narrowing paths explicitly:**
   - there is no API to "load master into a bf16-default model"; the trainer checkpoint
     loader returns `(model, master_state, optimizer, cfg, step)` together, never a bare
     model; the inference-only `load_checkpoint` stays bf16-only and cannot open a train
     blob (different `format` string → refuse with a clear message).
   - `load_state_dict(strict=True)` on the master map with an explicit
     `{key: expected fp32}` assignment (not `load_state_dict` on the module, which would
     follow default dtype) — copy tensor-by-tensor and assert `tensor.dtype is float32`.
   - a mutant that drops the dtype assertion or casts master to bf16 must fail a test (§4).
5. The persistent fp32 `gate.bias` and runtime buffers follow their own explicit handling
   and are never silently promoted into master.

Net: a tensor's restored dtype is decided by `param_meta` + the dedicated master map, never
by an ambient default; the inference and training loaders are disjoint formats, so the #441
cross-path cannot occur.

---

## 3. Resume-equivalence oracle (uninterrupted run vs save/load)

The test must compare against a CONTROL that never checkpointed, at a step boundary, and
prove BOTH weights and the optimizer trajectory (not just one-batch loss) resume exactly.

Construction (CPU, v41f_small, fp32 master path — same harness as test_p1_train_smoke):
- Fix all RNG seeds ONCE at the top; build model A (control) and model B (resume) from the
  identical init (same seed → same weights). Build both master maps identically.
- Deterministic input sequence of B batches `x0..xN` (the data order is fixed; RNG/data
  loader restoration is out of scope, so batches are an explicit list).
- Control A: run steps 0..N continuously with one AdamW over A.master.
- Resume B: run steps 0..k, `save_train_checkpoint`, discard model/optimizer AND force
  Python gc (proves no live object is being compared), `load_train_checkpoint`, then run
  steps k+1..N with the restored optimizer.
- Compare at t=k (immediately after load) and t=N (after more steps):
  - **master params bit-exact**: `A.master[n] == B.master[n]` exact (fp32 deterministic CPU;
    expect 0.0 — AdamW is a fixed pointwise op given identical m/v/step/grad).
  - **bf16 run weights exact** after the refresh copy.
  - **optimizer m/v/step exact** per param.
  - **full-sequence logits at step k+1** for the same batch: fp32 master → bf16 forward is a
    fixed cast, expect bit-exact on CPU; assert atol 0 for logits, with a recorded note if
    any platform-dependent reduction forces a tight tolerance (prefer exact; do not accept
    5e-2 — that is the tolerance that hid #441).
  - **gradients**: run one more identical batch on A and B post-resume WITHOUT stepping;
    compare the fp32 master grads (these prove the restored m/v/step feed AdamW identically,
    not just that weights match).
- Anti-tautology (the oracle must be able to see a wrong resume): assert that resetting B's
  optimizer (fresh AdamW, m=v=0, step=0) while keeping weights makes the trajectory DIVERGE
  at t=k+1 even though step-k weights match — so optimizer-state continuity is actually
  observed, not implied. Also a run re-INITIALISED from seed must differ from resumed B.

A second oracle pins the forward-only inference path: a training checkpoint must still load
in the bf16 inference loader ONLY via its `model` group (explicit conversion helper, format
check), producing the same logits as the bf16 weights in the training blob.

---

## 4. Mutation / gate test list (each must fail for the right reason)

All in a new `tests/v41f/test_p1_train_ckpt.py`, registered in pre-commit SELFTEST_FILES;
direct-runner compatible (no pytest-only fixtures, temp dirs process-private per the
#441 lesson). Numbered so a reviewer can run each mutant.

Correctness gates (green on the real code):
1. round-trip: save then load returns config/vocab_id/version equal; master param set ==
   model trainable-param set; optim index set matches; every master tensor fp32.
2. step-0 save (optimizer never stepped → m/v absent or zero per AdamW semantics) loads.
3. resume-equivalence: §3 control vs resume — master/weights/m/v/step/logits/grads exact at
   t=k and t=N; force gc between save and load.
4. atomic artifact: after save, no `*.tmp.*` remains; target file exists; writing a second
   version replaces atomically and a reader never observes a missing/partial file (the fsync
   itself is labeled not-unit-tested, per repo convention).
5. inference interop: bf16 `model` group loads in the inference loader and reproduces logits.

Mutants (each must turn a NAMED assertion red — target assertion cited, not a generic crash):
M1. drop/zero one master tensor on disk → master bit-exact + strict-set gate fails.
M2. cast a saved master tensor to bf16 before write → save-time dtype assertion (or load-time
    fp32 check) fails; this is the exact #441 regression, now caught.
M3. delete `optim` group (or zero m/v/step) → resume trajectory diverges AND the
    anti-tautology/optimizer-state assertion fails; load-set gate also flags missing group.
M4. bump step counter wrong (or off-by-one in which step is saved) → grad/logits at k+1
    differ (AdamW step-count path), caught by the exact resume oracle.
M5. tamper config (dim*2) → strict load size-mismatch raises (inherited, re-assert).
M6. tamper/remove vocab_id → train loader refuses (missing AND mismatched, two sub-cases).
M7. unknown format/version string → loud unsupported-version error.
M8. remove the master→bf16 refresh before forward (or skip grad cast to fp32) → resumed
    logits/grads diverge from the uninterrupted control.
M9. write non-atomically (os.replace removed / direct write to final path) → the "no tmp
    sibling + atomic replace" assertion fails (fsync itself stays labeled-uncovered).
M10. present an unexpected extra key / omit one model key → strict load raises both ways.
M11. point the inference loader at a train blob (and vice versa) → format guard refuses,
     proving the two paths cannot be crossed (the structural #441 fix).
M12. bypass process-private temp cleanup → leak gate (reuse #441's loud-rm-tree harness;
     injection of rmtree failure must surface).

Non-targets recorded as N/A: hard-topk indexer has no CE gradient today (so its m/v are
moot until STE/aux), but it must still round-trip structurally; LR scheduler/RNG/data cursor
explicitly not restored in version 1 and the docstring/test must not imply they are.

---

## 5. Decisions (ruled by fb 2026-09-17)

1. **Single `.pt` blob** — reuse ckpt.py + build_agentic_sft staging (one os.replace atomic
   point). Directory/sharded layout deferred until a real >single-file need (YAGNI).
2. **Hard-topk indexer params are NOT given a master copy or m/v.** The criterion is
   optimizer-group membership, not a hand-maintained name list: they are not in the AdamW
   param group (#440 already asserts they receive no CE grad), so AdamW state for them is
   naturally empty. Save only their bf16 model weights plus a `param_meta` flag
   `grad=False`; if a future STE/aux loss puts them in the group they get master state with
   no migration of the format.
3. **fp32-native parameters (head, six HC tables, attn_sink) ALIAS master storage** — the
   parameter itself is the fp32 master; no second copy. Only bf16-native params get a
   distinct fp32 master. Avoids two sources of truth for tensors that are already fp32.
4. **New file `v41f/master.py`** owns `TrainState` (master map + bf16 cast hooks) and the
   TRAINING save/load, kept physically separate from the inference-only `ckpt.py`. The two
   format strings must be disjoint so a train blob cannot be opened by the inference loader
   or vice versa (mutant M11).

Implementation is scheduled AFTER de's engram and 0e's MTP land, to avoid editing
v41f/*.py concurrently. This PR is the design + test list only; M1-M12 stand.
