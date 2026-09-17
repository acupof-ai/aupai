---
question: Replace the v41f MoE per-expert Python `torch.where` loop with a GPU `torch._grouped_mm` dispatch without changing numerics — what is the exact math, the kernel contract, and the parity gate that lets the swap be kernel-only?
status: draft (P4 throughput pre-research, 2026-09-17, 66). Design + CPU parity skeleton only; v41f/moe.py is unchanged. Awaits fb/de sign-off, then a GPU prereg before the fast path lands.
source: v41f/moe.py:72-89 (the loop); third_party/deepseek_v41_ref/model_ref.py.ref:889-904 (upstream MoE); torch._grouped_mm probed on local torch 2.12.0 CPU 2026-09-17; CPU parity tests/v41f/test_p1_grouped_dispatch.py
---

# Grouped-GEMM MoE dispatch — design

## 1. Problem

`v41f/moe.py:79-89` computes the routed experts with a Python loop over all 48 experts:
`torch.where(indices == i)` per expert, a gather, one SwiGLU FFN, and an fp32 scatter-add.
The docstring says training dispatch uses `torch._grouped_mm`; the kernel is **not
implemented** — the loop is the only path. At 48 experts that is 48 Python round trips, 48
small launches for each of w1/w3/w2, and 48 gathers/scatters per layer per step: the leading
MoE throughput bottleneck on the gate.

This PR is pre-research only. It does not touch `v41f/moe.py`. It pins the math on CPU so the
GPU change is a kernel substitution behind one flag, with the numerics already gated.

## 2. Scope and the one invariant

- Add a **GPU-only** grouped dispatch path in a later PR. The CPU per-expert loop stays, and
  stays the **oracle** the GPU path is tested against.
- The two paths must be `allclose` on identical gate output `(weights, indices)` and expert
  parameters. The gate, the routing weights, the shared expert, and the residual are shared;
  only the routed-expert execution changes.
- Single process (world_size=1). Multi-rank all2all dispatch is out of scope; the upstream
  inference ref itself keeps the per-expert loop and only `dist.all_reduce`s the combined `y`
  (`model_ref.py.ref:889-904`).

## 3. The grouped math

The loop does, for each expert `i`, the rows assigned to it:

```
idx, top = torch.where(indices == i)              # rows [token] whose top-k chose expert i
y[idx] += Expert_i(x[idx], weights[idx, top])     # weight multiplied in fp32 BEFORE w2
```

The grouped form computes every expert at once. Flatten the `(token, k)` assignments and
**sort by expert**, so rows for one expert are contiguous:

1. `flat = indices.flatten()` (length `Mtot = n_tokens * top_k`); `order = argsort(flat,
   stable=True)`; `exp = flat[order]`; `tok = order // top_k`; `cho = order % top_k`. The
   sort **must be stable**: `torch.where(indices == i)` lists an expert's rows in flattened
   row-major order, and an unstable sort permutes rows inside a group, which silently
   scrambles bf16 outputs (this cost a debugging pass).
2. `batch_sizes = bincount(exp, minlength=E).to(int32)` — rows per group; `sum = Mtot`.
3. `x_g = x[tok]` `[Mtot,dim]`, grouped-contiguous by expert; `w_g = weights[tok, cho, None]`.
4. Stack expert weights and run three grouped GEMMs:
   - `g1 = grouped(x_g, W1)` and `g3 = grouped(x_g, W3)` `[Mtot,inter]`,
   - activation in **fp32**: `h = w_g * (silu(g1.float()) * g3.float())` — the routing weight
     is multiplied **before** the down projection, matching `_expert_weighted`
     (`v41f/moe.py:19-31`),
   - `gd = grouped(h.to(x.dtype), W2)` `[Mtot,dim]`.
5. Scatter-add: `y = zeros(n_tokens, dim); y.index_add_(0, tok, gd.float())`. A token with
   `top_k` DISTINCT experts contributes `top_k` rows; `index_add_` sums them in fp32, matching
   the loop's fp32 `y[idx] +=`. Top-k routing never repeats an expert within a token, so there
   is no interaction with the loop's advanced-index assignment (a repeated-expert fixture made
   the two differ — it is an impossible input, asserted out by the skeleton).
6. Add the shared expert exactly as today: `y += shared_expert(x).float()`; `y.type_as(x)`.

The CPU reference for "grouped" in the skeleton is a plain per-group matmul over the
contiguous slices; on GPU each one is a single `torch._grouped_mm`. The arithmetic order is
unchanged: groups are still visited in ascending expert id and `index_add_` accumulates a
token's `top_k` contributions in that expert-sorted order, which matches the loop's ascending
expert order — this is why the fp32 diff is at round-off (§6), not reorder-noise.

## 4. Numerical parity (the gate)

| path | gate | measured max_abs (CPU, torch 2.12.0, 2026-09-17) |
|---|---|---|
| fp32 params/activations | `allclose(atol=1e-5, rtol=1e-3)` | **0.0** |
| bf16 experts, weight-before-w2 in fp32 | `allclose(atol=2e-2, rtol=1e-3)` + finite | **0.0** |
| weight-before-w2 vs weight-after-w2 (bf16, n=64) | must be a different function, `> 1e-3` | 3.4e-3 |

Measured on small CPU shapes (n=11/13, E=4/5, top_k=2), every expert used plus a
deterministically empty `[8,9,7,0]` group, real `v41f.Expert` parameters
(`tests/v41f/test_p1_grouped_dispatch.py`). The exact 0.0 on both dtypes is because the
grouped reference uses the **same** `F.linear` operand layout as the loop and a stable sort,
so the two paths are the same GEMM calls in the same order. The fp32 scalar-weight
before/after-w2 commute to round-off, so that guard deliberately runs bf16 (where the
rounding point moves, per the P0 lesson).

Rules the GPU kernel must not violate, or parity is not the same function:

- Routing weight multiplied in **fp32 before w2**. Multiplying after the down projection is
  algebraically equal but rounds elsewhere and already broke the bf16 gate by up to 0.5 in
  P0 (`v41f/moe.py:19-31`).
- g1/g3 upcast to fp32 around the activation; the h→w2 input casts back to the stream dtype
  exactly as `_expert_weighted` does.
- Empty experts (`batch_sizes[i]==0`) contribute nothing — the loop skips them; the grouped
  op carries a zero-length group, which is the same.
- Gate output `(weights, indices)` is held fixed between the two paths; this gate tests
  dispatch only, never the router.

## 5. `torch._grouped_mm` contract (measured, not assumed)

Probed on local torch 2.12.0. The op exists and runs on CPU, but see the warning below.

- Inputs: `a[Mtot, K]` rows laid out **contiguously per group in expert order**; group
  weights. The real op wants `[E, K, N]` and computes `a_g @ b[g]` (nn.Linear stores
  `[N,K]`, so the GPU build transposes/contiguously repacks). Third argument per-group
  **batch sizes, int32** (positional). Confirmed semantics against a manual per-group matmul.
- **The CPU test does NOT feed a transposed operand.** A pre-transposed contiguous `[E,K,N]`
  matrix makes the CPU bf16 GEMM round at a different point than the loop's `F.linear` over
  `weight.T` (measured max_abs 0.18): that is operand-LAYOUT rounding inside the kernel, not
  dispatch math, and it would make the CPU gate measure the wrong thing. The skeleton's
  `_grouped_linear` stacks weights in the native `[E,N,K]` layout and calls `F.linear`, so
  it isolates grouping (exact 0.0). The GPU kernel's mandatory `[E,K,N]` repack gets its own
  bf16 layout-parity check on the GPU under prereg #3; it is not asserted on CPU.
- Offsets (alternative third arg) must be **int32**; a `[0, …, Mtot]` prefix form int64 is
  rejected.
- Alignment: strides must be multiples of 16 bytes (an SM-grouped kernel constraint). Small
  CPU test dims (K=N=5) rejected; K/N multiples of 16 ran. The GPU path must pad dims/inter
  or dispatch on shapes that satisfy this; gate dim=1024/inter=1728 do not both divide 16
  evenly (1728 is 16-aligned; verify the exact kernel requirement on the H20 build) — **prereg**.
- Autograd is supported (a and b grads returned) on the aligned CPU shape; the training path
  needs it, confirmed on GPU before relying on it — **prereg**.

**WARNING — do not call the op in the CPU test.** On CPU the built-in `_grouped_mm` output
did not match a per-group reference for a small case (max_abs 8.2, mean 0.93 — the CPU path
is not the faithful kernel; it is a shape/registration shim). The skeleton therefore defines
its own pure-torch `_grouped_linear` (an explicit per-group slice matmul over the contiguous
layout) as the CPU grouped semantics and compares **that** to the where-loop. The GPU
implementation swaps only this one function for `torch._grouped_mm`; the math around it
(sort, batch_sizes, fp32 weight point, index_add) is identical and is what the CPU gate
pins. Never point a CI atol at the CPU `_grouped_mm` op.

## 6. What changes when the GPU path lands (later PR, gated)

- `v41f/moe.py` gains a `grouped: bool` (or device-dispatch) branch: CPU/`--grouped=0` keeps
  the current loop verbatim (oracle); CUDA/`--grouped=1` runs the §3 sequence with
  `torch._grouped_mm` in place of `_grouped_linear`.
- The parity test grows a GPU section that runs the real MoE both ways on the gate shape
  (8/top2 and 48/top6) under the §4 atols; the CPU section here already holds regardless.
- Stack the `nn.ModuleList` expert weights into `[E,K,N]` once (parameter repack or a
  grouped layout at build). Whether to repack per call or hold grouped parameters is a
  memory/autograd trade-off — **prereg** (measure; per-call `.t().contiguous()` over 48
  experts must not eat the launch savings).
- Registration/selftest: the skeleton is standalone in this PR and intentionally not added to
  the P0/P1 allclose discovery (it exercises no upstream-equivalent kernel yet). It joins
  `tests/v41f/p1_selftest.py` discovery when the GPU branch exists.

## 7. Open / prereg (self-decided items, need a GPU measurement to close)

1. 16-byte alignment on H20 for dim=1024/inter=1728/top-k gather widths — pad or assert.
2. Keep parameters grouped permanently vs repack per forward (memory + autograd).
3. Real-GPU `_grouped_mm` backward correctness/performance; bf16 accumulation vs the fp32
   weight point (some grouped kernels only accumulate in the input dtype — measure atol).
4. Throughput delta vs the loop at 48 experts on H20 (the payoff this exists for); prereg
   target before enabling by default.
