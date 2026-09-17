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
**0.0** (`tests/v41f/test_p1_grouped_dispatch.py`, in `moe_grouped_mm_dispatch.md` §4). On
GPU the same test runs with `torch._grouped_mm` where the skeleton calls `_grouped_linear`.

- bf16, at the gate magnitudes (`dim=1024, inter=1728`): the CPU pre-transposed-layout
  figure there is **0.031–0.25 across seeds** and is shape-dependent, so the GPU gate gets a
  **threshold, not a copied value**. Proposed bound `atol=2e-2` (the project MoE convention)
  on the *dispatch-vs-loop* comparison — the GPU kernel does NOT need to match the CPU
  layout artifact; it must match the loop it replaces.
- `prereg`: the measured GPU max_abs at the gate shape, and whether the kernel needs the
  16-byte-aligned `[E,K,N]` repack (moe doc §7 prereg #1). If the real kernel's bf16
  accumulation differs enough to exceed 2e-2, that is a finding for the prereg, not a silent
  threshold bump.
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

```bash
python tests/v41f/p0_selftest.py --selftest    # CPU, must stay green (baseline)
python tests/v41f/p1_selftest.py --selftest
# prereg: a GPU variant that builds v41f_small on cuda, forward+backward vs the CPU fp32 ref
```

- Forward: whole `v41f_small` (`v41f/config.py:110`, the upstream-small reference shape) on
  GPU, compared to the CPU fp32 reference at the existing P0/P1 atol — `prereg` that the GPU
  bf16 path reproduces the CPU comparison inside that atol (the suites are bf16 on CPU today;
  the GPU dtype/atol pairing is the thing to pin).
- Backward: every parameter receives a finite gradient; no NaN/Inf anywhere in the graph.
- `prereg`: the exact GPU atol and the measured max_abs; do not invent it.

Failure action: a NaN or a shape-dependent gradient gap blocks the training launch; it is a
model/kernel defect, and it is cheaper to find here than at step 40 of a real run.

## 3. Memory peak and first-step throughput — recorded the r3 way

Once §2 passes, run a short smoke (the V4.1 stack at the gate shape) and record it in the
convention the r3-era smokes already used (`facts/v41.json#v41.smoke_compiled_flash_h_i_0911`,
`#v41.smoke_first_steps_f_0910`): loss at named steps, grad-norm, **peak GiB/rank**, and
**tok/s/gpu** at named steps — with the config that produced them, never a bare number.

- Record: cfg (dim/layers/heads/seq/batch/accum/world/dtype/compile), loss + gnorm + tok/s at
  steps 10/20/30..., peak GiB/rank, NaN yes/no, checkpoint written (path) or not.
- The r3 reference points (batch 8 OOMs pre-step at 94.6 GiB; batch 4 accum 4 drove 381 steps
  at 72.64 GiB/rank, steady ~18K tok/s/gpu) are **r3-era facts on the r3 model family**.
  v41f is a new model family: its architecture flags differ and it does **not inherit r3
  weights or r3 numbers**. Use the r3 record as the *format*, not as the expected values.
- `prereg`: v41f's own peak and steady tok/s at the batch/accum chosen. B4×accum is fixed
  where the gate recipe fixes it; the smoke reports what the new node actually does.

Failure action: an unexpected OOM is a config/headroom finding for the prereg (mirroring the
`smoke_oom_ladder` batch-8 record), not a reason to change batch silently.

## 4. What each gate proves, and the sequence

| # | gate | proves | CPU status | GPU status |
|---|---|---|---|---|
| 1a | 8 H20 + CUDA `_grouped_mm` runs | node is the promised node; the kernel exists (not the shim) | n/a | prereg |
| 1b | GPU fast path allcloses the CPU loop | the dispatch swap is safe | 0.0 on CPU | `atol=2e-2` proposed, value prereg |
| 2 | whole-net fwd/bwd, no NaN, matches ref | the model runs on device, gradients are finite | P0/P1 green on CPU | on-GPU atol prereg |
| 3 | smoke: peak GiB/rank + tok/s, r3 format | the node's real capacity and speed for v41f | n/a | prereg |

Order matters: 1a is the node, 1b is the dispatch, 2 is the model, 3 is the budget. A failure
in 1 stops the GPU dispatch path; a failure in 2 stops the training launch; a failure in 3 is
a config finding.

## 5. prereg — everything this doc does not yet know

Explicitly unmeasured, to be filled from the first node's run, never guessed:

1. GPU `_grouped_mm` vs the loop at the gate shape: max_abs and the working atol.
2. Whether the kernel needs the 16-byte-aligned `[E,K,N]` repack at dim=1024/inter=1728.
3. GPU bf16 backward correctness and its atol vs the CPU fp32 reference for `v41f_small`.
4. v41f's own smoke peak GiB/rank and steady tok/s/gpu on the new node at the launch shape.
5. Whether any of the above is sensitive to the H20 driver/torch build on the new image.
