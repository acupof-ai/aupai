---
question: After the new persistent node is provisioned, what is the FIRST sequence of GPU checks that must pass before any real training — and what does each one prove that the CPU gates could not?
status: draft (provisioning block 3 of 3, 2026-09-17, 66). Pure doc; no GPU run yet. Every GPU-only value is marked prereg, never guessed. Second reader from fb.
source: docs/standards/moe_grouped_mm_dispatch.md (§4 CPU parity, §5 op contract); docs/standards/infra_persistent_rebuild_0916.md §7 (provisioning done-criteria, the precondition); tests/v41f/p0_selftest.py + p1_selftest.py (the CPU reference gates); facts/v41.json#v41.smoke_compiled_flash_h_i_0911 and #v41.smoke_first_steps_f_0910 (the r3-era smoke record convention)
---

# New-node GPU smoke gate — the first numerical gates before training

Third and last provisioning block. The other two gate the data floor and the node:
`docs/standards/infra_persistent_rebuild_0916.md` (mounts, backup, the node exists) and
`docs/standards/data_pipeline_rebuild_0916.md` (corpora, tokenizer, pools). This one is the
**GPU verification** block: the node exists and the data floor is up, and these checks must
pass before a training launch is even drafted.

None of these has been run. The node is down. Every line below is a recipe, and every number
that only the GPU can produce is tagged `prereg` — not estimated. The CPU half of each gate
already exists and passes on `main`; this document is the transplant.

## 0. Preconditions (from the other two blocks)

Do not start §1 until: `infra_persistent_rebuild_0916.md` §7 all hold (persistent root,
real backup mount, `root_durable` PASS, 8 H20 visible, `/data00/models` accounted); and the
gate tokenizer + one smoke token-cache domain exist from the data block. This block assumes
a node that satisfies them.

## 1. GPU visibility and the grouped_mm fast path

Two independent facts, in order.

**1a. Eight H20 visible, and the grouped op is real.** `nvidia-smi` shows 8 devices; and on
one card, confirm `torch._grouped_mm` runs on CUDA for the gate shapes (the CPU op is a
non-faithful shim — `moe_grouped_mm_dispatch.md` §5 — so this both proves the GPU kernel
exists and that we are not accidentally measuring the shim).

```bash
python -c "import torch; print(torch.cuda.device_count())"        # expect 8
# prereg: a tiny CUDA a@W[e] grouped mm vs a per-group reference, on the gate dim/inter
```

Expected: 8; the op produces a finite tensor. Failure action: if count != 8, stop — this is
the node/container (infra block §7.4), not a model bug. If the op is absent or only the shim
responds on CUDA, stop the GPU path and keep the CPU loop; do not train on an unverified
dispatch.

**1b. GPU fast path allcloses the CPU loop oracle.** This is the gate that lets the grouped
dispatch replace `v41f/moe.py:79-89`'s per-expert loop. The CPU form of it already passes at
**0.0** — both dtypes, because the reference uses the same `F.linear` operand layout as the
loop (`tests/v41f/test_p1_grouped_dispatch.py`, `moe_grouped_mm_dispatch.md` §4). On GPU the
same test runs with `torch._grouped_mm` where the skeleton calls `_grouped_linear`.

- CPU expectation for this gate: **0.0** — it compares dispatch math against the loop, and
  grouping is exact once the operand layout is held fixed.
- The GPU kernel is forced to the `[E,K,N]` weight layout, which the CPU reference does NOT
  use. That layout difference is a separate, **shape-dependent** CPU proxy: bf16
  `dim=1024/inter=1728` measured 0.031–0.25 across seeds, `dim=8/16` and `dim=64/128` 0.0
  (`moe_grouped_mm_dispatch.md` §5). This is the operand-layout rounding the GPU repack can
  introduce, and its real on-GPU magnitude is **prereg #3 — do not gate on the CPU proxy and
  do not copy it as a threshold**. Gating the GPU run on the proxy range would either fail a
  correct kernel or invite a silent atol bump.
- `prereg`: the measured GPU dispatch-vs-loop max_abs at the gate shape, and whether the
  kernel needs the 16-byte-aligned repack at all (moe doc §7 prereg #1). Set the working atol
  from that measurement, against the loop, not from any CPU proxy.
- The CPU loop stays the oracle: a GPU run that disagrees with it fails, full stop.

Failure action: do not enable the fast path; the loop remains the production path and the
throughput item stays open.

## 2. Whole-net forward+backward on GPU — no NaN, bit-exact to the CPU fp32 reference

The P0/P1 suites on `main` already prove, **on CPU**, that every leaf module and every
assembled Block/Model matches the vendored upstream reference
(`tests/v41f/p0_selftest.py`, `tests/v41f/p1_selftest.py`; oracle
`third_party/deepseek_v41_ref`). The GPU gate re-runs the same oracle comparison with the
model on the device, and adds the two things a CPU run cannot show: GPU kernels emit no NaN,
and GPU backward agrees.

**This gate is re-run at each assembly step, not once.** The current `v41f_small()`
(`v41f/config.py:110`) and the whole-net comparison run with all four new subsystems OFF
(`engram_layer_ids=()`, `n_mtp_layers=0`, `dspark_block_size=0`,
`dspark_target_layer_ids=()`). A green run there says nothing about the model once Engram,
DSpark, or the indexer STE are wired in. Per the assembly plan (`docs/standards/v41f_assembly_plan.md`,
open PR #468 at the time of writing — this section tracks that plan's A–D steps), re-run this
gate after each step with that step's mechanism ON:

| step | mechanism ON for this gate | prerequisite |
|---|---|---|
| base | all four OFF — today's `v41f_small` | none (this is the §2 run above) |
| A | Engram (`self.engrams`, `engram_hash`) | `engram_num_embeddings` on `V41FConfig` (else `EngramLayout.from_args` AttributeErrors) + the bf16 `Engram.q_weight`/`k_weight` fix |
| B | DSpark draft block (`self.mtp` + training entry) | #454 merged |
| C | indexer STE (non-default path) | #456 merged |
| D | ckpt covers the three new subsystems | after A/B/C (parameters settled) |

Steps A/B/C are **fully serial**, not merely ordered (A and B both edit `v41f/model.py`; C is
staggered to isolate regressions) — the prerequisite column is a gate, and the serial ruling
is the reason no two run at once. The `engram_num_embeddings` field and the explicit-bf16
`q_weight`/`k_weight` are hard prerequisites inside step A: without them the model either
fails to construct or constructs at an implicit dtype that the #468 regression gate exists to
catch.

Each step's own pair of gates (off = regression against the step's baseline; on = correctness
vs the vendored reference) is #468's; §2 here is the **GPU execution** of that pair.

```bash
python tests/v41f/p0_selftest.py --selftest    # CPU, must stay green (baseline)
python tests/v41f/p1_selftest.py --selftest
# prereg: a GPU variant that builds v41f_small + the enabled step, fwd+bwd vs the CPU fp32 ref
```

- Forward: the assembled model on GPU, compared to the CPU fp32 reference at the existing
  P0/P1 atol — `prereg` the GPU dtype/atol pairing and the measured max_abs; do not invent it.
- Backward: every parameter receives a finite gradient; no NaN/Inf anywhere in the graph.
- Run this at the base config AND at each of A/B/C (D re-runs it via ckpt load). A step that
  is correct on CPU but NaNs or diverges on GPU is exactly what this catches.

Failure action: a NaN, or a gradient gap that survives the step's own on/off gates, blocks
that assembly step's merge and the training launch — cheaper to find here than at step 40 of
a real run.

## 3. Memory peak and first-step throughput — recorded the r3 way

Run a short smoke (the V4.1 stack at the gate shape) and record it in the convention the
r3-era smokes used: loss at named steps, grad-norm, **peak GiB/rank**, **tok/s/gpu**, with
the config that produced them — never a bare number.

**Measure this after assembly steps A and B**, not at the base config: Engram and DSpark both
add parameters (and DSpark adds a draft block), so a peak taken with them off under-reports
the training shape. Step C (indexer STE) is attention-path and does not change parameter
count, but re-record if it moves the attention activation peak.

- The record format and the r3 reference points are in the same fact entry:
  `facts/v41.json#v41.smoke_compiled_flash_h_i_0911` records batch 8 OOMs pre-step at **94.6
  GiB/rank** and batch 4 accum 4 driving **381 steps at 72.64 GiB/rank, steady ~18K
  tok/s/gpu**; `#v41.smoke_first_steps_f_0910` is the eager dense-FFN counterpart (25.2
  GiB/rank, 7K/12K tok/s/gpu at steps 20/30).
- Those are **r3-era facts on the r3 model family**. v41f is a new model family: its
  architecture flags differ and it does **not inherit r3 weights or r3 numbers**. Read the
  r3 entry for the *format*; the numbers above are r3's, not a target for v41f.
- `prereg`: v41f's own peak and steady tok/s at the batch/accum chosen, measured with A+B on.
  B4×accum is fixed where the gate recipe fixes it; the smoke reports what the new node does.

Failure action: an unexpected OOM is a config/headroom finding for the prereg (mirroring the
`smoke_oom_ladder` batch-8 record), not a reason to change batch silently.

## 4. What each gate proves, and the sequence

| # | gate | proves | CPU status | GPU status |
|---|---|---|---|---|
| 1a | 8 H20 + CUDA `_grouped_mm` runs | node is the promised node; the kernel exists (not the shim) | n/a | prereg |
| 1b | GPU fast path allcloses the CPU loop | the dispatch swap is safe | **0.0** (dispatch vs loop) | value prereg; layout proxy not a gate |
| 2 | whole-net fwd/bwd, no NaN, matches ref | the model runs on device, gradients are finite | P0/P1 green on CPU | on-GPU atol prereg; re-run per assembly step |
| 3 | smoke: peak GiB/rank + tok/s, r3 format | the node's real capacity and speed for v41f | n/a | prereg; measure after A+B |

Order: 1a/1b are one-time (node + dispatch); 2 is re-run at base and at each assembly step
A/B/C; 3 is the budget, measured once A+B are in. A failure in 1 stops the GPU dispatch path;
in 2 it blocks that assembly step's merge; in 3 it is a config finding.

## 5. prereg — everything this doc does not yet know

Explicitly unmeasured, to be filled from the first node's run, never guessed:

1. GPU `_grouped_mm` vs the loop at the gate shape: max_abs and the working atol (set from
   this measurement, against the loop — not from the CPU `[E,K,N]` layout proxy).
2. Whether the kernel needs the 16-byte-aligned `[E,K,N]` repack at dim=1024/inter=1728.
3. GPU bf16 backward correctness and its atol vs the CPU fp32 reference, at base and at each
   assembly step.
4. v41f's own smoke peak GiB/rank and steady tok/s/gpu on the new node, with A+B on.
5. Whether any of the above is sensitive to the H20 driver/torch build on the new image.
