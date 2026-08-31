---
question: "Can the LM head run in fp8, what does it actually save, and what has to be true for it to be correct?"
status: recorded
source: "t57 (fb ruling 2026-09-01); facts/efficiency.json#eff.lm_head_is_compute_bound, #eff.steady_state_composition; liger_kernel 0.8.2 ops/fused_linear_cross_entropy.py; train.py FP8LinearFunction"
---

# fp8 LM head: design

The head is the largest single lever with a mechanism and a computed ceiling. This is the
design, not an implementation: it names the three GEMMs, what fp8 covers, what it saves, and
the two ways it can be wrong.

## Why it is a lever at all

The claim it replaces was mine and it was wrong: I told fb and the user that FP8 had no
headroom because the head is memory-bound. It is not. Liger's chunked head GEMM is
M=2048, K=1024, N=32784, so **AI is 668.7 FLOP/B against H20's balance point of 37**
(`eff.lm_head_is_compute_bound`). It is compute-bound by a factor of 18, and it is excluded
from fp8 today by `_fp8_ok`'s **name** check on `"head"` — not by alignment (32832×1024 and the
32784 slice are both 16-aligned) and not by tied-parameter detection.

## The three GEMMs, and they are equal

`liger_kernel/ops/fused_linear_cross_entropy.py` does the head as three matmuls per chunk:

| line | GEMM | shape | GFLOP/chunk |
|---|---|---|---|
| 114 | `logits_chunk = _input_chunk @ weight.t()` | M×K·K×N | 137.5 |
| 232 | `grad_input = grad_logits_chunk @ weight` | M×N·N×K | 137.5 |
| 236 | `grad_weight += grad_logits.t() @ _input_chunk` | N×M·M×K | 137.5 |

All three are the same size, so **the measured 224.0 ms/step splits roughly 75 / 75 / 75**.
That matters for scoping: converting only the forward buys ~1/3 of the ceiling.

| scope | ideal 2× saving | % of a 1702 ms step |
|---|---|---|
| forward only | 37 ms | 2.2% — **under the ship gate** |
| forward + grad_input | 75 ms | 4.4% |
| all three | **112 ms** | **6.6%** |

Forward-only cannot pass the ≥3% gate on its own. The design therefore has to cover at least
two GEMMs to be worth shipping, which makes it a backward change and not a forward tweak.

## What we already have, and it is most of the work

`train.py`'s `FP8LinearFunction` (:352) already implements exactly this pattern for the body
linears, and it answers the objection I expected to be hardest. It does **all three** GEMMs
through `torch._scaled_mm` with per-tensor e4m3 scaling, and it produces `grad_w` with
`out_dtype=torch.bfloat16` — so an fp8 GEMM does not need an fp32 output and the
`grad_weight` accumulation is not a blocker. It also caches the fp8 operands and their scales
for backward, cutting five quantisations to three.

So the design is **not** "write an fp8 head kernel". It is "make Liger's three matmuls use the
scaled-mm path we already trust", by one of:

- **A. Patch Liger's chunk loop.** Replace the three `@` with `_scaled_mm` calls carrying
  per-chunk scales. Smallest diff, but it edits a site-packages file, so it needs vendoring or
  a runtime monkeypatch and it re-breaks on every Liger bump.
- **B. Replace FLCE for the head with our own chunked loss.** Reuses `FP8LinearFunction`
  directly and drops the Liger dependency for this path. Larger diff, and it gives up Liger's
  in-place logits-gradient trick, which is what keeps the V=32784 logits from materialising
  twice — a memory regression we have not budgeted.
- **C. Keep FLCE, precompute the projection in fp8 outside it.** Not viable: the fusion's whole
  point is that logits never leave the kernel.

**A is the default** on the AGENTS.md rule of the laziest correct implementation, with vendoring
rather than monkeypatching so the version is pinned in-tree.

## The two ways it is wrong, and the gates

The head feeds the loss directly, so an fp8 error lands on **every** gradient, not on one layer's
activations. That is a stronger parity requirement than the body linears carry.

1. **Range.** Measured on the live run's step-7000 checkpoint, and it comes out **benign for the
   weight**: head `absmax 26.875`, `std 4.32`, so `w_scale = 26.875/448 = 0.060`. Per-vocab-row
   absmax is p50 15.69, p99 19.84, max 26.88 — **max/p50 is only 1.71×**, so no single row
   dominates and per-tensor scaling costs little. What is still unmeasured is the **activation**
   side: the pre-softcap projection is unbounded and one outlier logit sets the scale for a
   32784-wide chunk. Gate before any A/B: max |logit| and its per-chunk spread on real batches,
   the same way the weight was just checked.
2. **Loss curvature.** Cross-entropy is sensitive where the softmax is confident; a quantisation
   error that is invisible in the projection can move the loss. Gate: `|Δval| ≤ 0.04 nat` on a
   50-step A/B, which is the standing rule, plus a parity check of loss and grad-norm against
   the bf16 path on the same batch.

Ship gate unchanged: 50-step 7-card A/B, tok/s ≥ +3%, `|Δval| ≤ 0.04 nat`, parity test.

## What this design does not establish

The 112 ms is an **ideal** 2× on measured GEMM time. It will not be realised in full: scaling
work is not free, FLCE's chunking overhead is untouched by a faster matmul, and the
`_scaled_mm` path has its own alignment constraints at N=32784. No implementation exists to
measure yet, and the 224.0 ms itself comes from a single-card trace
(`eff.steady_state_composition`), so the 7-card share may differ. Treat 6.6% as the ceiling to
measure against, never as the expected gain.
