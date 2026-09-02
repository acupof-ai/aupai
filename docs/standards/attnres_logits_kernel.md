---
question: What does a fused AttnRes logits+mixing kernel compute, what may it assume, and what proves it correct?
status: design
source: trace_p200m_3step.json (measured); model.py:244-253; scripts/attnres_fused_reference.py (b0, commit ccbc089)
---

# AttnRes logits kernel — design

## The target, measured

The AttnRes block is **138.9 ms/step, 8.6% of wall** on p200m
(`[16,4096,1024]`, L=12, n=25 sources). Every kernel whose launching op carries
the AttnRes idiom (`select`+`unsqueeze`, or `softmax` over the stacked logits),
grouped by role with no size threshold so the group closes exactly:

| group | ms/step | launches/step |
|---|---|---|
| **logits forward** — `model.py:248`, `triton_red_..._squeeze_sum_unsqueeze_*` | **68.131** | 632 |
| softmax forward + mixing — `triton_poi_fused__softmax_...` | 50.307 | 142 |
| softmax backward | 12.200 | 100 |
| other `select`+`unsqueeze` | 8.243 | 34 |
| **AttnRes total** | **138.881** | 908 |

Three accounting notes, each of which corrects an error in an earlier version
of this table:

- **35.960 ms/step** of `..._silu_silu_backward_slice_...` kernels match the
  same name pattern but belong to `DeltaRecurrence`'s shortconv
  (`model.py:107-113`), not AttnRes. Excluded.
- **The mixing's 30.2 ms ceiling is not a trace line and must never be added to
  one.** It is the ablation-confirmed upper bound on what a mixing-only fusion
  could remove (b0, `attnres_fused_reference.py`); the numbers above are
  measured kernel times. Summing a bound with a measurement yields a quantity
  that is neither, which is what produced an earlier unexplained 31.2 ms gap.
- **`logits forward` is not one kernel.** Inductor specializes it per source
  count, so there are ~18 compiled variants (`..._unsqueeze_3` through `_16`
  and beyond), one per distinct `n`. The 67.861 ms reported earlier was the
  single largest variant; the group is 68.131. This matters for the design: a
  fused kernel taking `n` at runtime replaces all 18.

The logits group is **6.25× off the bandwidth roofline** — the largest single
deviation anywhere in the trace, worse than any KDA kernel (worst 3.55×, class
1.84–2.13×). It reads a whole `[B,T,D]` tensor per source read and emits one
scalar per row: 1024 elements in, 1 out.

## What the kernel computes

Per AttnRes call, over sources `i = 0..n-1`, each `v_i` of shape `[B,T,D]`:

```
logit[i,b,t] = (Σ_d v_i[b,t,d] · gq[d]) · scale_i[b,t]      # model.py:248
a[:,b,t]     = softmax over i of logit[:,b,t]                # model.py:249
out[b,t,d]   = Σ_i a[i,b,t] · v_i[b,t,d]                     # model.py:250-252
```

`gq = g * q` is `[D]`, computed once per call outside the kernel. `scale_i` is
`[B,T,1]`, the RMS factor `Source` stores instead of a normalized copy
(`model.py:75-80`). **`scale` belongs to the logits, never to the mixing** — a
kernel that folds it into `dV` passes a weaker test than the model deserves
(b0 records this at `attnres_fused_reference.py:124-126`).

## The one non-obvious design decision: row blocking

`a[i,b,t]` depends on **every** `v_j` at the same `(b,t)`, so mixing cannot
start before all logits at that row are done. That looks like it forces two
full passes over `v`, and at tensor granularity it does: one source is
**134.2 MB**, the H20 L2 is 60 MB, so nothing survives from pass 1 to pass 2.

**But the dependency is per-row, not global.** Rows `(b,t)` are independent, so
the traversal blocks by rows: for a block of `R` rows, compute all `n` logits,
softmax across `i` within the block, then mix.

**The blocking size is set by SMEM, not L2, and an earlier version of this
section got that wrong.** Holding `n` tiles per CTA (`n×R×D`) does not survive
78 SMs running concurrently — at R=128 that is 6.55 MB per CTA and **511 MB
across the machine, against a 60 MB L2**. The fix is that a CTA never needs
`n` tiles resident: it streams one source at a time, keeping only the **fp32
accumulator `out[R,D]`** plus the current tile.

| R rows | accumulator (fp32) | + one tile (bf16) | per CTA | fits 228 KB SMEM |
|---|---|---|---|---|
| 16 | 64 KB | 32 KB | 96 KB | yes |
| 32 | 128 KB | 64 KB | 192 KB | tight |
| 64 | 256 KB | — | — | no |

So **R ≤ 16**, and the accumulator stays in SMEM instead of round-tripping to
HBM.

**The single pass is only valid with an online softmax.** `a_i` needs every
logit, so a kernel that streams sources cannot know `a_i` when it reads `v_i`.
The kernel keeps a running max `m`, a running denominator `l`, and the
accumulator, rescaling both by `exp(m_old − m_new)` when a new source raises
the max — flash-attention's trick, applied over the source axis instead of the
key axis. Final `out = acc / l`. Verified in fp64 against `softmax(0)` then
mix: **max error 2.2e-16** over n=25, i.e. exact to rounding. Without this the
form silently needs two passes, which is what an earlier version assumed.

`v` is then read **once per source read** rather than twice, which is the
whole saving — exactly the factor 2 b0 measured in the `add_` count
(`add_ = 2 × source_reads`, one edge from the logits read and one from the
mixing read; confirmed by ablation, ratio 2.00 at L=2/3/4/12).

The byte budget below is unchanged by this correction: it counts HBM traffic
per source read, which does not depend on `R`. If the implementation cannot
hold the accumulator in SMEM, two passes each with their own read is the
retreat, not the plan.

## Bytes, per step at L=12 (325 source reads forward)

One `[B,T,D]` bf16 tensor = 0.1342 GB.

| | GB/step | ideal ms at 4.0 TB/s |
|---|---|---|
| today: logits pass | 43.6 | 10.9 |
| today: mixing pass | 43.6 | 10.9 |
| today: `out` writes (25 calls) | 3.4 | 0.8 |
| **today total (forward)** | **90.6** | **22.6** |
| **fused, row-blocked** | **47.0** | **11.7** |

Measured forward today is **118.4 ms** (68.131 logits + 50.307 softmax and
mixing). At the 1.5× roofline target that is **17.5 ms**, i.e. about
**101 ms/step, 6.3% of wall** — but see the note below before quoting it.

**What this estimate is worth.** The 22.6 ms "today ideal" is what the current
two-pass structure would cost at the roofline; the measured 118.4 ms is 5.24×
that. Attributing the whole gap to fusion assumes the new kernel reaches 1.5×
where the existing one sits at 6.25×, and nothing here proves that is
reachable. **The defensible claim is the byte reduction: 43.6 GB/step removed
= 10.9 ms at 4.0 TB/s, floor.** Everything above that is the unexplained 6.25×,
which fusion may or may not touch. Quote 10.9 ms as the guaranteed part.

## Backward

Gradients (b0's derivation, `grads_analytic`, verified against autograd to
1e-12 in fp64):

```
dV_i[b,t,d] = a[i,b,t] · dout[b,t,d]              elementwise
dA[i,b,t]   = Σ_d dout[b,t,d] · v_i[b,t,d]        reduction over D
```

Backward is also **one pass**, but only because forward saves `a`. The chain is

```
dA[i,b,t]     = Σ_d dout·v_i                    one pass over v
s[b,t]        = Σ_i a[i]·dA[i]                  a per-row scalar
dlogit[i,b,t] = a[i]·(dA[i] − s)                softmax backward
dV_i[b,t,d]   = a[i,b,t]·dout[b,t,d]            elementwise
```

`dlogit` needs **all** `dA` before any of it is final, and `dV` needs `a`.
Both dependencies are per-row scalars, so if forward stores `a[n,B,T]` the
backward streams `v` once. Verified in fp64 against autograd: `dlogit` max
error 2.2e-16, `dV` exactly 0.

**Storing `a` rather than recomputing it is not a close call**: `a[n,B,T]` fp32
is 0.0066 GB per call against 3.36 GB to re-read `v` and rebuild the logits —
**512× cheaper**. (Contrast KDA, where the analogous `disable_recompute=True`
trade stores ~1.6 GB/call; that one is genuinely arguable, this one is not.)
The `sum` backward is a broadcast-multiply; its reads are already inside b0's
261.7 GB/step accounting.

**The gradient wrt `v_i` has a second term the mixing does not own**: each `v`
is read by its own logit too, so autograd's `v.grad` is the mixing term *plus*
the logits term. A fused kernel that computes both paths owns both terms —
which is a change from the mixing-only design, where the logits residue stays
in autograd. **This is the parity trap**: b0's `known_answer()` deliberately
reports that residue as expected. A logits+mixing kernel must match the
**total**, so the check inverts, and copying b0's tolerance without noticing
would let a wrong kernel pass.

## Correctness gates

1. **Known-answer against the real module.** `known_answer()` from
   `scripts/attnres_fused_reference.py` — builds a real `model.AttnRes` with
   randomized `q` and `g` (both are identity elements at their init values, so
   a wrong weighting is invisible without this), fp32, `atol=1e-6`. Forward
   must match; `dV` must match the **total** (see the trap above).
2. **Gradient check** on `dA` through the softmax, at the logits, where both
   sides have the same variable — `a` is not a leaf.
3. **`add_count(L)` regression.** The closed form `n(n+1)` at L=2/3/4/12 is a
   cheap structural check that the AttnRes wiring did not change. A fused
   forward changes this count; the new expected values go in the same
   assertion rather than the assertion being deleted.
4. **20-step loss delta ≤ 1e-3** against the unfused arm, same seed.
5. **Negative control.** Run every gate above against a deliberately wrong
   kernel (e.g. `scale` folded into the mixing) and confirm each fails. A gate
   never seen red is a hypothesis about a gate.

## Constraints

- **Flag-gated, default off.** The unfused path stays until the A/B lands.
- **No `[n,B,T,D]` stack in forward.** `model.py:247` records that copy
  dominating at L=24, which is why the loop exists.
- **`n` varies per call** (source `p` is read by calls `p..2L`), so the kernel
  takes `n` as a runtime argument, not a constexpr — or it recompiles 25 times
  per step.

## What this design does not address

The 6.25× itself. The fused kernel is designed to halve the bytes; whether it
also closes the gap to roofline depends on why the current kernel is 6.25× off,
and that has not been diagnosed. **Candidate: 632 kernel launches/step for 18
calls means ~35 launches per call, so each launch does a small slice of a
reduction whose only parallel axis is D=1024.** If that is the cause, row
blocking fixes it as a side effect (rows become the parallel axis). If not, the
byte saving stands alone at 10.9 ms and the rest needs its own investigation.

**Criterion: 18 KERNEL launches per step, one per call — not 18 CTAs.** At
R=16 each call launches 65536/16 = 4096 CTAs, so the step runs 73728 CTAs in
18 launches. CTAs are cheap; kernel launches are what the 632 counts. Reading
this criterion as a CTA count would reject a correct kernel.
