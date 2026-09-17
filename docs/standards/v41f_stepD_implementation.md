# step D decomposition — fp32-master + optimizer resume, driven by prereview #487

Status: implementation plan, doc-only. Inputs are the approved design
`docs/standards/v41f_train_checkpoint_design.md` (#447) and the adversarial prereview issue
**#487** (G1–G7, each with file:line). Every count below I rebuilt independently on this
main under the bf16 default dtype (CPU, torch 2.12); they match #487. No model/config edit
here. Owner: 98. Code opens only after #485 (indexer STE), ae's test-wiring PR, and the
standalone G5 `requires_grad` flip (step D-PRE, de owns — not part of #485) land.

**Sequencing.** #487 changes this from "implement the design" to "amend the design, then
implement". PR-0 is a doc-only revision of `v41f_train_checkpoint_design.md` applying the
G1–G7 language in §1 below (reviewed by genA, the adversarial reader). PR-1 is the code in
§3. G2/G3/G7 are not code-patchable without the design change because the save/load contract
itself is what is wrong.

## 0. Verified facts the plan rests on

| build (v41f_s under bf16 default) | params | fp32 | state_dict | persistent buffers |
|---|---|---|---|---|
| pre-assembly: engram (), n_mtp 0 | 1982 | 87 | 1994 | 12 |
| MTP only: engram (), n_mtp 1 | 2149 | 94 | 2162 | 13 |
| **default: engram (1,), n_mtp 1** | **2153** | **94** | **2166** | **13** |

- Default→MTP-only delta = 4 engram bf16 params (`engrams.1.{q_weight,k_weight,embed.weight,wkv.weight}`),
  0 fp32. Pre→MTP delta = 167 params, +7 fp32: `mtp.0.attn.attn_sink` (1) + six
  `mtp.0.hc.hc_{attn,ffn}_{fn,base,scale}` (6). 87+7 = 94.
- The 87 pre-assembly fp32 names: `head.weight` 1; `attn_sink` 12; six HC tables 72; the
  compressor `wkv`+`wgate` 1+1. The compressor fp32 pair exists ONLY on the single
  `kv_source_layers=(2,)` module (`v41f/attention.py:96-102`), independent of
  `compress_ratios` — the design's "layers 2-5 carry a compressor" sentence is wrong.
- 13 persistent buffers = 12 backbone + 1 MTP `ffn.gate.bias`; 2166 − 2153 = 13.
- Hard-topk indexer: **6** production params, all currently `requires_grad=True`, at
  `index_source_layers=(2,4,8)` → `layers.{2,4,8}.attn.indexer.{wq_b,weights_proj}.weight`
  (4 on v41f_small's `(1,3)`). The current `AdamW(model.parameters())` is unfiltered, so
  they sit in the group and only lack state because the hard path yields grad None.
- G2 reproduced end-to-end: two modules with identical names/shapes and reversed build
  order, weights aligned bit-exact by the name-keyed `state_dict`, bind integer optim-index
  0's `exp_avg/exp_avg_sq` onto the wrong parameter after `load_state_dict`; index-set
  cardinality passes. engram+MTP already shifted every index once; #485 shifts them again.

## 1. Per-gap: which design section changes, then the code

| gap | design edit (PR-0) | code consequence (PR-1) |
|---|---|---|
| **G1** stale census | §1.1 lines 88–101 and gate-6 lines 248–251: stop asserting one frozen total on "the production config". Gate builds TWO named configs — pre-assembly (1982/87) and the post-assembly DEFAULT (2153/94) — and pins the fp32 NAME SET per build plus the dtype rule, never the absolute number alone. Correct lines 92–97: step B DOES add 7 fp32. Correct the tied-embed/head model: `DSparkBlock` registers neither embed nor head (both are passed at the call site); only the inference-only `DSparkMTP` owns embed. Fix the "layers 2-5 compressor" sentence. | census test parametrised over the two configs; expected fp32 suffix sets `{head, sink×L, six-HC×L, compressor×1}` and `+ {mtp.0.sink, mtp.0 six-HC}`. |
| **G2** optim integer index | §1.1 blob and §1.3 line 132: replace the integer-indexed `optim.state` with a NAME-keyed block over the in-group master names (the concrete format is genA's verified scratch design `~/aupai-textgen/genA/g2_optim_namekey_test_design_2026-09-18.md`, torch 2.12): canonical name-sorted `param_names`; `state_by_name{step,exp_avg,exp_avg_sq,shape,dtype}`; hyper by value; symmetric name-set diff + per-name shape/dtype + post-load `torch.equal`; no integer `param_groups.params` round-tripped. `param_names` doubles as the §6 optimizer-membership census, so it lists ONLY requires_grad in-group master names, not all model params. §3 oracle adds a cross-BUILD resume case (reversed construction order) that the same-build oracle structurally cannot see. | TrainState saves/loads by the genA functions; AdamW constructed on name-sorted master; loader refuses set/order/shape/dtype drift and asserts each m/v/step re-attached under the same name (`beta->beta` True, `beta->alpha` False). |
| **G3** tokenizer | §1.3 and §2.5: load is `load_*_checkpoint(path, *, tokenizer, vocab_id, ...)`; an engram-on config without an injected tokenizer raises before build; tokenizer identity is tied to the existing vocab_id refusal. Note the inference `ckpt.py` loader gains the same injection (it cannot open today's default blob either). | loader signature takes tokenizer; `V41FModel(cfg, tokenizer=tokenizer)`; no v41f file read. |
| **G4** optim verbatim | §1.1 lines 50–51: keep AdamW per-param state tensors VERBATIM inside `state_by_name` (`step` is a 0-d tensor in torch 2.12, not int; m/v untouched); persist hyper BY VALUE with the constructor allowlist only `{lr,betas,eps,weight_decay,amsgrad,maximize}` — `decoupled_weight_decay/capturable/foreach/fused` are state_dict-only keys and raise if passed to `AdamW()`. Drop the hand-shaped `{int step, trimmed groups}` and never round-trip integer params. | save/load use genA's `_HYPER` allowlist; 0-d step tensor stored as-is. |
| **G5** dormant not enforced | Decision 2 / §6: membership is `requires_grad`, and the OFF indexer leaves must be `requires_grad_(False)` at build so present-dormant is real, not "in group but grad None". Precise proposition (de): **off → 0 of the 6 in group (in-group 2147); ste → exactly the 6 index-source `wq_b`/`weights_proj` in group (2153)**; every other indexer/compressor/index_key param stays grad-less and census-marked frozen. STE's purpose is to train those 6; the detach seam keeps the rest frozen. | NEW standalone step D-PRE (de owns, §4): the mode sets the 6 leaves' `requires_grad`; #485 does not (both modes True). TrainState filters on it; M14 census asserts the exact 6-name in-group delta. PR-1 blocks on D-PRE. |
| **G6** buffers | §1.1 line 47: the `model` group is the FULL `model.state_dict()` — parameters AND the 13 persistent `gate.bias`; buffers never enter master/optim. §1.3 strict load then sees them. M10 gains buffer missing/extra sub-cases. | save `model` from `state_dict()`, master from the live fp32 params only. |
| **G7** alias on load | §2.5 lines 182–184: by-tensor copy contradicts the single-storage alias. Loader special-cases fp32_native: `master[name] = model[name]` (same Parameter, one storage); only bf16 params get a distinct cloned fp32 master. Gate 7 data_ptr equality then holds and a master step is visible in `model.head`. Also specify the SAVE side: a plain `model.state_dict()` materializes fresh tensors and severs cross-group sharing, so for fp32_native names save must place the SAME live parameter object under both `model[name]` and `master_fp32[name]` (torch.save preserves shared storage within one object graph); buffers and bf16 use the state_dict copies. | loader branch on group; save builds the model group from live params for the alias set. |

## 2. Corrected save/load contract (post-amendment summary)

Blob, one `.pt`, atomic same-dir `.tmp.<pid>` + `os.replace` (fsync labelled untested):

```
format="v41f_train_ckpt", version=1, config=asdict, vocab_id, step,
param_meta: {name: {dtype, group: bf16|fp32_native, grad: bool}},
model:             full state_dict (params + persistent gate.bias); alias names = live object
master_fp32:       name -> fp32 (bf16: distinct clone; fp32_native: SAME object as model[name])
optim_named: {
  param_names:  [name] name-sorted, in-group master only (the membership census),
  state_by_name:{name: {step(0-d tensor), exp_avg, exp_avg_sq, shape, dtype}},  # step-0 -> absent
  hyper:        {lr, betas, eps, weight_decay, amsgrad, maximize} by value,
}
```

Load order: format/version guard → rebuild cfg → tokenizer present + vocab_id equal →
bf16-build model with injected tokenizer → `load_state_dict(model, strict=True)` →
build TrainState classification → fp32_native `master=model param` (share), bf16 distinct
master loaded tensor-by-tensor with fp32 assert → build AdamW on name-sorted in-group master
→ load `optim_named`: symmetric name-set diff, per-name shape/dtype gate, attach m/v/step by
name, post-load `torch.equal` per name → restore step → return `(model, state, cfg, step)`.
Inference `ckpt.py` keeps a disjoint format and refuses this blob (M11).

## 3. Code change list (PR-1)

| file | change |
|---|---|
| `v41f/master.py` (new) | `TrainState` (requires_grad classification, fp32 master map, fp32_native alias set, bf16-only refresh, fp32 grad cast); `save_train_checkpoint` / `load_train_checkpoint` per §2. |
| `v41f/train.py` | `train_step(..., state=None)`: given state, refresh→forward→backward→cast grads→`state.optimizer.step()` over master; `None` keeps the present direct-optimizer path so `test_p1_train_smoke.py` stays green. |
| `v41f/ckpt.py` | add `tokenizer=` to `load_checkpoint`; refuse a `v41f_train_ckpt` format by name (~3 lines). |
| `tests/v41f/test_p1_train_ckpt.py` (new) | §4 gates; auto-discovered by the `p1_selftest.py` glob. |

No `SELFTEST_FILES` edit (glob; ae's PR makes the glob run in the gate).

## 4. G5 prerequisite: a separate `requires_grad` flip (step D-PRE, de owns)

Step D derives the group from `requires_grad`; it is correct only if the indexer leaves'
flag actually reflects the mode. Verified on #485 @920e2504: **#485 does NOT set the flag**
— both `off` and `ste` leave `wq_b`/`weights_proj` at `requires_grad=True` (grep over the
five #485 files is empty); the modes differ only by grad presence. That is exactly the G5
"in group but grad None" defect, so the fix is NEW code cut as its own small change, not
added to the frozen #485:

- **step D-PRE** (de owns): at build, `off` → the 6 index-source
  `layers.{2,4,8}.attn.indexer.{wq_b,weights_proj}.weight` get `requires_grad_(False)`;
  `ste` → exactly those 6 are trainable. compressor/index_key/non-source-layer indexer
  params stay frozen (the STE detach seam already keeps them grad-local).
- TrainState filters on `requires_grad`; it must NOT hand-list names.
- The step-D census under each mode asserts the in-group set is exactly 2147 (off) / 2153
  (ste), i.e. a delta of exactly those 6 names (M14).

Step-D code (PR-1) blocks on D-PRE merging; the design amendment (this PR-0) does not.

## 5. Minimal CPU-acceptable first increment (after PR-0)

Green gates on v41f_small, process-private temp dirs, direct-runner:
- G-rd round-trip: config/vocab_id/version/step; `optim_named.param_names` == in-group
  master names (the membership census); every master fp32; state_by_name step-0 absent.
- G-alias: fp32_native data_ptr equality (incl `head.weight`, one HC table, the MTP sink)
  after a real save→load; bf16 distinct; two-build census (pre 1982/87, default 2153/94).
- G-resume: uninterrupted control vs save→gc→load→resume, atol 0 on master / bf16 weights /
  m,v,step / k+1 logits+grads; anti-tautology fresh-optimizer diverges. Plus the #487 G2
  cross-build case (genA T-G2-2): reversed-construction-order model, name-keyed load puts
  `A.beta.m` on `B.beta` (True) never `B.alpha` (False); a mutant that drops name-keying and
  keeps integer indices binds it onto `B.alpha` and must fail that exact assertion.
- G-atomic: no `*.tmp.*` after save. G-format: inference loader refuses train blob.

Mutants run red by name in PR-1: M1 drop a master tensor; M2 saved master→bf16 (#441);
M3 zero optim m/v/step AND drop/permute a `param_names` entry or swap two names (G2,
T-G2-3/4: missing/extra name and shape/dtype tamper each fail their named gate); M8 remove
refresh or grad cast; M11 cross loaders; M13 refresh aliased `head.weight` → atol-0
divergence; + G3 missing tokenizer refuses, G6 missing `gate.bias` strict-fails, G7
loader-copy breaks data_ptr. T-G2-5: indexer leaves are absent from `param_names` in off and
exactly the 6 names appear in ste (the §6 state census).

## 6. Non-targets v1

LR scheduler / RNG / data-cursor not restored (fixed batch list in oracle); no sharded or
fp8 master; indexer round-trips present-dormant until ste flips its group.
