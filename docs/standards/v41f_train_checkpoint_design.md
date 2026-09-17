# v41f fp32-master + optimizer resume checkpoint — design (P0, design only)

Status: revised 2026-09-18 for prereview #487 (G1–G7). The original #447 proposal is
amended: the census is name-keyed over two builds, optimizer state is name-keyed (not integer
index), the loader takes an injected tokenizer and a new blob `vocab_id`, fp32-native master
aliases one storage through save AND load, the model blob carries persistent buffers, and
optimizer-group membership is the `requires_grad` flag. This change is docs-only; code (PR-1)
opens only after #485 and the v41f test-wiring PR land. It is the continuation explicitly
deferred by #441 (`v41f/ckpt.py`: "fp32 master/optimizer state is a training concern and is
not part of this model checkpoint").

Scope of THIS checkpoint:
- the full model state_dict (bf16 run weights, fp32-native weights, persistent gate buffers),
- an fp32 master for every in-group (`requires_grad`) parameter — distinct for bf16-native,
  one-storage alias for fp32-native,
- NAME-keyed AdamW state (`exp_avg` m, `exp_avg_sq` v, 0-d `step`, per-name shape/dtype),
- the V41FConfig, a schema/format version, and a vocab_id fingerprint of the tokenizer.
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
  "vocab_id": "<sha256 of the injected tokenizer; NEW top-level blob key in step D>",
  "param_meta": { name: {"dtype": "bfloat16"|"float32", "group": "bf16"|"fp32_native", "grad": bool} },
  "model":        { "<name>": tensor, ... },   # FULL model.state_dict(): run weights AND persistent buffers
  "master_fp32":  { "<name>": tensor, ... },   # fp32 master for every in-group (requires_grad) param
  "optim_named":  {                           # NAME-keyed, never integer-indexed (#487 G2)
      "param_names":  ["<name>", ...],        # name-sorted in-group master names; doubles as membership census
      "state_by_name": { "<name>": {"step": <0-d tensor>, "exp_avg": t,
                                    "exp_avg_sq": t, "shape": [..], "dtype": "<str>"} },
      "hyper": {lr, betas, eps, weight_decay, amsgrad, maximize},   # constructor-accepted kwargs only
  },
  "step": <int global update count>,
}
```

`vocab_id` is a NEW blob key for v41f: neither `v41f/config.py` nor `v41f/ckpt.py` carries a
vocab field today (#487 G3, 0e evidence). The saver hashes the injected tokenizer (same
convention as the v41 gate / `scripts/check_sft_ready.py`) and writes it; the loader hashes
the tokenizer it is given and refuses on missing OR unequal, which is also the M6 contract.
Without this key the "load a tokenizer consistent with the checkpoint" requirement has
nothing to compare.

`model` is the FULL `model.state_dict()`, not just `named_parameters()`: it carries every
persistent buffer too. In the post-assembly default that is **2153 parameters + 13
persistent buffers = 2166 keys** (#487 G6); the 13 buffers are the 12 backbone
`layers.L.ffn.gate.bias` plus `mtp.0.ffn.gate.bias`, all fp32 selection-only. Buffers never
appear in `master_fp32` or `optim_named`. A strict `load_state_dict` requires these keys, so
saving parameters alone fails on load.

Optimizer state is keyed by NAME, never by AdamW's integer position index (#487 G2).
`AdamW.state_dict()` numbers state by construction order; modules inserted since a
checkpoint (engram, MTP, STE) shift every index, and two models with identical names/shapes
but reversed build order have an EQUAL integer index set while binding `exp_avg/exp_avg_sq`
to the wrong tensor (reproduced end-to-end: A.beta's m lands on B.alpha bit-for-bit while
weights stay name-aligned and bit-exact, so weight/logits assertions stay green).
`param_names` is the canonical name-sorted order and lists ONLY in-group master names, so it
doubles as the optimizer-membership census; `state_by_name` holds m/v/step plus a per-name
shape/dtype record. `step` is a 0-d tensor in torch 2.12, stored verbatim. Hyperparameters
are persisted BY VALUE with only constructor-accepted kwargs
(`decoupled_weight_decay/capturable/foreach/fused` are state_dict-only keys and raise if
passed to `AdamW()`). Params that have never stepped simply have no `state_by_name` entry
(the step-0 save). The concrete name-keyed save/load functions and their five tests are
genA's verified scratch design (`g2_optim_namekey_test_design_2026-09-18.md`, torch 2.12).

Dtype groups are DERIVED FROM EACH PARAMETER'S `.dtype` on the live bf16-build model, never
hand-enumerated. The grouping code is `group = "fp32_native" if p.dtype==float32 else
"bf16"` over `model.named_parameters()`, and a census gate pins the exact NAME SETS so a
silent membership change (including an fp32↔bf16 flip that leaves the total unchanged) goes
red; the count is asserted only as the derived `len(...)` cross-check. There are TWO pinned
builds (#487 G1), both rebuilt 2026-09-18 under the bf16 default dtype on torch 2.12:

**Pre-assembly** — `replace(cfg, engram_layer_ids=(), n_mtp_layers=0)`: **1982 parameters =
87 fp32 + 1895 bf16**, 12 persistent buffers, 1994 state_dict keys. The 87 fp32 names are
exactly:

- `head.weight` (1),
- the six HyperConn tables per layer `layers.L.hc.hc_{attn,ffn}_{fn,base,scale}` (12×6 = 72),
- `layers.L.attn.attn_sink` for L in 0..11 (12),
- the compressor softmax-pool projections `layers.2.attn.compressor.wkv.weight` and
  `…wgate.weight` (2).

**Post-assembly DEFAULT** — `V41FConfig()` with `engram_layer_ids=(1,)`, `n_mtp_layers=1`
(built with the injected tokenizer, §1.3): **2153 parameters = 94 fp32 + 2059 bf16**, 13
persistent buffers, 2166 state_dict keys. The 94 fp32 names are the SAME backbone 87 plus 7
MTP tables: `mtp.0.attn.attn_sink` and the six `mtp.0.hc.hc_{attn,ffn}_{fn,base,scale}`.
Engram adds 4 bf16 and 0 fp32 (`engrams.1.{q_weight,k_weight,embed.weight,wkv.weight}`); the
13th buffer is `mtp.0.ffn.gate.bias`. Increment: 1982+4+167 = 2153, 87+7 = 94, 12+1 = 13.

Two factual corrections over the earlier draft:
- The compressor fp32 pair exists ONLY on the single `kv_source_layers=(2,)` module
  (`v41f/attention.py:96-102`), independent of `compress_ratios`; "layers 2-5 carry a
  compressor" was wrong. The dtype line is `dtype=torch.float32 if compress_ratio > 1 …` but
  the module is instantiated only on the kv-source layer. `compressor.norm` and every
  ratio-1 path are bf16.
- Step B DOES add 7 fp32 params; "adds no fp32" was wrong. And `DSparkBlock` registers
  NEITHER an embed NOR a head — both are passed at the call site (`v41f/model.py`,
  `v41f/train.py`); only the inference-only `DSparkMTP` owns an embed. So there is no tied
  embed/head registration inside the draft block.

For every fp32-native param `model[name]` and `master_fp32[name]` alias one fp32 tensor
(§2.3 handles the refresh hazard; §2.5 reconstructs the alias on load). Every bf16-native
param is `group="bf16"` (run weight and a distinct fp32 master).
- **buffers, not in master/optim**: the persistent fp32 `ffn.gate.bias` set (12 backbone, 13
  with MTP) rides in the `model` blob via the full `state_dict()`; non-persistent runtime
  caches (window/compress kv, freqs_cis, engram primes/offsets/multipliers/token_map) are
  rebuilt, never saved.
- **membership is `requires_grad`, not a name list.** Hard-topk indexer params (6 in
  production: `layers.{2,4,8}.attn.indexer.{wq_b,weights_proj}.weight`) are
  present-dormant: in faithful `off` mode they must be `requires_grad_(False)` at build, so
  they are excluded from the in-group set and get bf16 weights + `param_meta grad=False` and
  NO master/m/v; under `ste` exactly those 6 flip in-group. The other indexer-adjacent params
  (compressor, index_key, non-index-source layers) stay frozen regardless of mode (the STE
  detach seam keeps them indexer-local). `optim_named.param_names` is the census of this set.

`param_meta` is generated from the live model by the dtype rule and the `requires_grad`
membership, never hand-written. The invariant across assembly steps is the RULE (fp32 iff
constructed fp32; in-group iff requires_grad), not the absolute count. The name-keyed
`model` / `master_fp32` / `optim_named.param_names` sets are cross-checked in the gate:
master names == in-group names == optim names, and the buffer names occur only in `model`.

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

The loader takes the tokenizer explicitly:
`load_train_checkpoint(path, *, tokenizer, vocab_id=None, max_batch_size=…)`. A config with
non-empty `engram_layer_ids` (the default) cannot be constructed without it
(`v41f/model.py` raises), so injection is part of the contract, not an option (#487 G3).

- Rebuild config from the blob; ANY V41FConfig mismatch → fail (same as #441).
- Require the injected `tokenizer`; if the config needs one and none is given, refuse by
  name BEFORE building.
- `vocab_id` is a top-level key (new in step D, §1.1); it must be present and equal to the
  hash of the injected tokenizer. Missing on either side OR unequal refuses (M6, both
  sub-cases). This supersedes the old "warn" behaviour; for a train ckpt it is always loud.
- `format`/`version` must be understood; unknown → loud error carrying the value seen.
- Build the bf16 run model under the bf16 default dtype WITH the injected tokenizer, then:
  - `model.load_state_dict(blob["model"], strict=True)` — the full state_dict incl the 13
    persistent `gate.bias`; a missing/unexpected key (parameter OR buffer) raises both ways.
  - Build `TrainState` classification; reconstruct fp32_native master as the SAME model
    Parameter (alias, §2.5/G7), load bf16-distinct master tensor-by-tensor asserting fp32;
    master name set == in-group (`requires_grad`) name set, strict.
  - Build AdamW over the name-sorted in-group master, then resolve `optim_named` BY NAME:
    symmetric name-set diff (missing or extra raises), canonical order match, per-name
    shape/dtype gate, attach m/v/step by name, and post-load `torch.equal` per name. NEVER
    trust integer index equality — it passes under a param reorder (#487 G2).

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
   projection for forward.** Concrete training-time object (`v41f/master.py` `TrainState`):
   - classify once from `named_parameters()`: in-group iff `requires_grad`; within it
     fp32_native iff `.dtype==float32`, else bf16_native. `master: dict[str, nn.Parameter]`
     has one entry per in-group param;
   - optimizer is constructed over name-sorted `master.values()` (AdamW sees only fp32);
   - before forward: `model_param.data.copy_(master[name].to(bfloat16))` for bf16_native
     params ONLY. fp32_native params alias one fp32 tensor shared between `model[name]` and
     `master[name]`, and the refresh loop skips them by group dispatch;
   - after backward: cast each in-group grad to fp32 into its master param
     (`master.grad = model_param.grad.float()`), then `optimizer.step()` advances master;
     bf16 model is refreshed on the next forward.
   This is the standard bf16-mixed-with-fp32-master pattern. Membership is `requires_grad`,
   so the off-mode indexer leaves must be built `requires_grad_(False)` (present-dormant) and
   the STE switch flips exactly those 6 to True — no hand-maintained name list in the
   trainer (#487 G5). IMPORTANT (verified on #485 @920e2504): #485 does NOT own this flip —
   both modes leave the leaves `requires_grad=True` and differ only by grad presence, so
   "in group but grad None" is exactly the G5 defect and the `requires_grad_(False)` at build
   is NEW code, a small standalone change (step D-PRE, de owns) that must merge before the
   step-D code relies on it, not part of the frozen #485.
3. **The bf16 refresh must never touch an aliased fp32-native parameter — this is a named,
   measured hazard.** For the fp32-native params (87 pre-assembly, 94 in the default — §1.1)
   `model[name]` and master share storage (one source of truth). If the §2.2 refresh is applied to them by
   mistake — `model_param.data.copy_(master[name].to(bfloat16))` where model_param IS
   master — the `.to(bfloat16)` rounds in a temporary and `.copy_` writes that rounded value
   back into the SAME fp32 storage: a 32-bit param holding `1.0000305` becomes `1.0` while
   `.dtype` STILL reads float32 (0e reproduced it on torch CPU, 2026-09-18). No dtype
   assertion can see it. The guard is structural, not a dtype check: the refresh dispatches
   on `group=="bf16"`, so an fp32_native/aliased param is unreachable by the copy branch;
   M13 (§4) pins it.
4. **Save master from the live fp32 master map, save model from the bf16 run model's FULL
   `state_dict()` (params + the persistent `gate.bias` buffers).** For fp32_native names the
   saver must place the SAME live Parameter object under both `model[name]` and
   `master_fp32[name]` inside ONE blob — a plain `model.state_dict()` materializes fresh
   tensors and severs the sharing, whereas `torch.save` preserves shared storage when the
   same object sits under both keys in one object graph (0e: one blob keeps data_ptr
   equality after reload; two separate blobs break it). The save asserts
   `master_fp32[name].dtype==float32` for every entry and
   `model[name].dtype == param_meta[name].dtype`; violation raises before writing.
5. **Load reconstructs the alias; it never copies an fp32_native master (#487 G7).**
   - the trainer loader returns `(model, train_state, cfg, step)` together, never a bare
     model; the inference-only `load_checkpoint` keeps its own format and refuses a
     `v41f_train_ckpt` blob by name (M11).
   - fp32_native: `master[name] = model[name]` — the SAME Parameter, one storage, so a master
     step is visible in `model.head` and gate-7 data_ptr equality holds after load. A
     tensor-by-tensor copy here creates a second storage and is the G7 defect (0e measured
     distinct data_ptrs for an `empty_like`+`copy_` build).
   - bf16_native only: a distinct fp32 master copied in tensor-by-tensor with an assert
     `tensor.dtype is float32`. A mutant that copies an fp32_native master, drops the dtype
     assert, or casts master to bf16 fails a named test (M2 / G7 / M13).
6. The persistent fp32 `gate.bias` buffers ride in `model` only; non-persistent runtime
   buffers are rebuilt and never promoted into master.

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

A THIRD oracle is required because the two above build control and resume from the SAME
constructor, so they cannot span a parameter reorder (#487 G2). Build a resume model whose
module construction order is reversed relative to the saver (identical names/shapes), align
weights by name, and load `optim_named`: the m/v saved for a name must reattach under THAT
name (`A.beta.exp_avg == B.beta.exp_avg`, and `!= B.alpha.exp_avg`). A mutant that stores
state by integer index binds `A.beta`'s state onto `B.alpha` and this exact name assertion
goes red while every weight/logits check stays green. This is genA's T-G2-2; the full
five-case G2 list (round-trip, reorder, missing/extra name, shape/dtype tamper, step-0 /
present-dormant census) is in `g2_optim_namekey_test_design_2026-09-18.md`.

---

## 4. Mutation / gate test list (each must fail for the right reason)

All in a new `tests/v41f/test_p1_train_ckpt.py` (auto-discovered by the `p1_selftest.py`
`test_p1_*.py` glob once the v41f test-wiring PR lands); direct-runner compatible (no
pytest-only fixtures, temp dirs process-private per the #441 lesson). Numbered so a reviewer
can run each mutant.

Correctness gates (green on the real code):
1. round-trip: save then load returns config/vocab_id/version equal; `optim_named.param_names`
   == in-group (`requires_grad`) master names == `master_fp32` names; every master tensor
   fp32; every persistent buffer is in `model` and absent from master/optim; hyper by value.
2. step-0 save (optimizer never stepped → the name simply has no `state_by_name` entry) loads;
   present-dormant (off indexer) names are absent from `param_names` but present as bf16 in
   `model` with `param_meta grad=False` (genA T-G2-5).
3. resume-equivalence: §3 control vs resume — master/weights/m/v/step/logits/grads exact at
   t=k and t=N; force gc between save and load.
3b. cross-construction G2 oracle (§3 third oracle): reversed build order, state reattaches by
   name.
4. atomic artifact: after save, no `*.tmp.*` remains; target file exists; writing a second
   version replaces atomically and a reader never observes a missing/partial file (the fsync
   itself is labeled not-unit-tested, per repo convention).
5. inference interop: bf16 `model` group loads in the inference loader and reproduces logits.
6. **dtype census is structural and NAME-keyed, on TWO builds** (#487 G1). Build
   `replace(cfg, engram_layer_ids=(), n_mtp_layers=0)` and assert 1982 params / 87 fp32 /
   1895 bf16 / 12 buffers and the exact 87-name SET; build the DEFAULT
   `V41FConfig()` (synthetic tokenizer) and assert 2153 / 94 / 2059 / 13 and the 94-name SET
   = backbone-87 ∪ {`mtp.0.attn.attn_sink`, six `mtp.0.hc.*`}, with NO `engrams.*` fp32 and
   the extra buffer `mtp.0.ffn.gate.bias`. Counts are the derived `len(...)`; an fp32↔bf16
   reclassification that leaves the total intact must still trip the name-set assertion.
7. **alias identity after a real save→load.** For every fp32_native param
   `model[name].data_ptr() == master[name].data_ptr()` (one storage, G7), and the §2 refresh
   branch is never applied to it. For every bf16 param the two are distinct storages.
8. M1/M3/M8/M9 are each run TWICE — once on a bf16-native param and once on the aliased
   `head.weight` — so the alias class (self-referential model==master) is actually exercised,
   not only the distinct-storage bf16 path.
9. **tokenizer + vocab_id (G3).** A default/engram-on build without an injected tokenizer is
   refused by name before construction; an injected tokenizer whose hash != blob vocab_id,
   and a missing blob vocab_id, are both refused (M6). The inference loader's only caller is
   the v41f test (`tests/v41f/test_p1_ckpt.py`), so adding keyword-only `tokenizer=None`
   (engram-off allows None; the train loader requires it) changes no production call site.
   vocab_id MUST be byte-identical to `scripts/loader.py:20 vocab_fingerprint`: iterate
   `sorted(tok.get_vocab().items(), key=lambda kv: kv[1])` (id order), feed each
   `token.encode()` into one sha256 with NO separator/prefix, take `hexdigest()[:16]`.
   Reimplement this verbatim in a torch-free `v41f/vocab.py` (v41f must not import scripts/);
   a gate cross-checks `v41f/vocab.fingerprint(tok) == scripts.loader.vocab_fingerprint(tok)`
   on the synthetic tokenizer so the two cannot drift.

Mutants (each must turn a NAMED assertion red — target assertion cited, not a generic crash):
M1. drop/zero one master tensor on disk → master bit-exact + strict-set gate fails.
M2. cast a saved master tensor to bf16 before write → save-time dtype assertion (or load-time
    fp32 check) fails; this is the exact #441 regression, now caught.
M3. zero m/v/step for a name, OR drop/swap two `param_names` entries / reorder state (#487
    G2) → the resume trajectory diverges AND the symmetric name-set / per-name shape /
    post-load `torch.equal` gate fails; the cross-construction oracle catches the reorder.
M4. bump step counter wrong (or off-by-one in which step is saved) → grad/logits at k+1
    differ (AdamW step-count path), caught by the exact resume oracle.
M5. tamper config (dim*2) → strict load size-mismatch raises (inherited, re-assert).
M6. tamper/remove vocab_id → train loader refuses (missing AND mismatched, two sub-cases).
M7. unknown format/version string → loud unsupported-version error.
M8. remove the master→bf16 refresh before forward (or skip grad cast to fp32) → resumed
    logits/grads diverge from the uninterrupted control.
M9. write non-atomically (os.replace removed / direct write to final path) → the "no tmp
    sibling + atomic replace" assertion fails (fsync itself stays labeled-uncovered).
M10. present an unexpected extra key / omit one model key → strict load raises both ways;
     includes omitting a persistent `*.gate.bias` buffer (G6: parameters-only save fails).
M11. point the inference loader at a train blob (and vice versa) → format guard refuses,
     proving the two paths cannot be crossed (the structural #441 fix).
M12. bypass process-private temp cleanup → leak gate (reuse #441's loud-rm-tree harness;
     injection of rmtree failure must surface).
M13. **alias×refresh truncation (de, d4f).** Point the pre-forward bf16 refresh at an
     fp32-native ALIASED param (`head.weight`, where model and master are one storage): the
     faulty path does `model["head.weight"].data.copy_(master["head.weight"].to(bfloat16))`,
     rounding 1.0000305 → 1.0 in place while `.dtype` stays float32. Gate 7's data_ptr alias
     holds and no dtype check trips, so ONLY a value oracle catches it — the §3 resume oracle
     must diverge from the uninterrupted control at **atol 0** (fp32 CPU), and a lazy save
     after one forward/step must already carry the truncated value. The correct code dispatches
     the refresh on `group=="bf16"` and leaves the alias untouched (master head.weight stays
     1.0000305). This is the truncation a dtype-only assertion cannot see.
M14. (#487 G5) leave an off-mode indexer leaf `requires_grad=True`, or unfreeze an
     index-adjacent param (compressor/index_key/non-source layer) under ste → the
     `param_names` membership census fails: off must list 0 indexer names; ste must list
     EXACTLY the 6 `layers.{2,4,8}.attn.indexer.{wq_b,weights_proj}.weight` and no other
     indexer/compressor/index_key name.

Non-targets recorded as N/A: in faithful off mode the hard-topk indexer leaves are
present-dormant (built, bf16-saved, `requires_grad=False`, absent from master/optim); STE
moves exactly those 6 in-group. LR scheduler/RNG/data cursor explicitly not restored in
version 1 and the docstring/test must not imply they are.

---

## 5. Decisions

1. **Single `.pt` blob at P0** — one os.replace atomic point. The binding constraint for G7 is
   narrower than "one file": the fp32_native model/master entries must be saved inside ONE
   `torch.save` object graph so the shared Parameter survives (saving them in two separate
   blobs breaks the alias; a future directory layout could still hold them in one nested
   file). The single file is the P0 form that satisfies this; directory/sharded deferred
   (YAGNI).
2. **Membership is `requires_grad`, enforced at build (#487 G5).** Merely being in
   `AdamW(model.parameters())` with a None grad is not dormant — the param is still in the
   group and silently gains m/v when STE later gives it grad. In faithful `off` mode the 6
   hard-topk indexer leaves (`layers.{2,4,8}.attn.indexer.{wq_b,weights_proj}.weight`) are
   built `requires_grad_(False)`: present-dormant, bf16-saved with `param_meta grad=False`,
   excluded from master and `optim_named.param_names`. Under `ste` exactly those 6 flip to
   trainable and get master+m/v with no format migration; compressor/index_key/non-source
   indexer params stay frozen (STE detach seam). No name list in the trainer — membership is
   the structural predicate, and M14 pins the exact 6-name delta.
3. **Every fp32-native parameter ALIASES master storage — 87 pre-assembly, 94 in the default
   (the extra 7 are the MTP sink + six MTP HC tables).** The parameter itself is the fp32
   master, no second copy; only bf16-native params get a distinct fp32 master. On SAVE the
   same live object is placed under both `model` and `master_fp32`; on LOAD fp32_native is
   reconstructed as `master[name] = model[name]` (G7 — a tensor copy breaks gate-7
   data_ptr equality). The bf16 refresh dispatches on group and never touches the alias
   (M13).
4. **New file `v41f/master.py`** owns `TrainState` (requires_grad classification, master
   map, bf16-only refresh, fp32 grad cast) and the TRAINING save/load, physically separate
   from inference-only `v41f/ckpt.py`. The two format strings are disjoint (M11).
5. **Optimizer state is NAME-keyed (#487 G2):** `optim_named.param_names` (name-sorted,
   in-group only), `state_by_name` with per-name shape/dtype, hyper by value
   (constructor-accepted kwargs only); load resolves by name and refuses set/order/shape/dtype
   drift. No integer-index round-trip.
6. **Tokenizer is injected and `vocab_id` is a new top-level blob key (#487 G3).** Both
   loaders take keyword-only `tokenizer` (engram-off may pass None; the train loader and any
   engram-on build require it); vocab_id is the sha256 over the sorted id→token map
   (`scripts/loader.py` convention) in a torch-free `v41f/vocab.py`. Missing or mismatched
   refuses. `v41f/ckpt.py`'s only caller is the v41f test, so this changes no production call.
7. **`model` blob is the full `state_dict()`** including the persistent fp32 `gate.bias`
   buffers (12 backbone, 13 with MTP); buffers never enter master/optim (G6).

This revision is doc-only and addresses prereview #487 (G1–G7). Code (PR-1) opens only
after #485 (indexer STE) and the v41f test-wiring PR land, plus the small standalone
G5 `requires_grad` flip (step D-PRE, de owns — the flip is NOT in #485); it must not edit
v41f/model.py/config.py while de's branch is open. Gates M1–M14 stand.

---

## 6. Parameter lifecycle — three optimizer-group states (shared with #456)

A trainable tensor's state is decided solely by its `requires_grad` flag at build
(optimizer-group membership), never by a name list; the indexer STE design (#456) and this
checkpoint use the same three states so the two docs cannot disagree:

| state | requires_grad / in AdamW | master/m/v | model weights | current example |
|---|---|---|---|---|
| **in-group** | True | fp32 master (distinct for bf16-native, alias for fp32-native) + m/v | bf16, or fp32 alias | default config has 2153 params (2059 bf16 + 94 fp32); under `off` the 6 indexer leaves are dormant so in-group = 2147, under `ste` = 2153 |
| **present-dormant** | False, but the module is built and saved | none | saved bf16 + `param_meta grad=False`, absent from `param_names` | the 6 hard-topk indexer leaves in faithful `off` mode |
| **absent** | module not instantiated on this config | none | not in the blob | level-1 candidate indexer when `candidate_source_layer<0` (v41f-S) |

The STE mode (introduced by #485) is what makes exactly the 6 index-source
`wq_b`/`weights_proj` leaves trainable; a separate, small change (step D-PRE, de owns) reads
that mode and sets `requires_grad` — off→False (present-dormant), ste→True (in-group) — with
no format migration. #485 itself does not set the flag (both modes read True today).
Membership is what turns on the fp32 master and m/v, the census and
`optim_named.param_names` are derived, and M14 pins the exact 6-name delta. A param that is
merely un-stepped (grad None) but still `requires_grad=True` is NOT dormant — that is the
#487 G5 defect. Absent params never appear and a loader must not require them.
