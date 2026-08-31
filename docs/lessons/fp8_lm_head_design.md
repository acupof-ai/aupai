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

**MEASURED, not inferred (b0, reproduced by me).** Each of the three is separately identifiable
in the trace by its transpose signature, at exactly 1280 calls over 20 steps = 64/step, matching
FLCE's chunk count:

| kernel | ms/step | share | µs/call |
|---|---|---|---|
| `nvjet_tst_272x128_..._coopA_TNN` | 62.5 | 32.9% | 977.0 |
| `nvjet_tst_128x160_..._splitK_NNT` | 63.0 | 33.1% | 984.0 |
| `nvjet_tst_256x160_..._coopA_NTT` | 64.5 | 34.0% | 1008.5 |
| **head total** | **190.0** | | |

The FLOP-equality inference held — 32.9 / 33.1 / 34.0, equal within 3% — but b0's prior that it
would NOT was the better-reasoned one: grad_W is a thin-K GEMM (M=32784, K=2048, N=1024) writing
33.6M output elements, classically the least efficient of the three on tensor cores. It came in
largest by 1.1 points. cuBLAS chose splitK for one and coop variants for the others, which is
presumably what levels them.

**The head is 190.0 ms/step, not 224.0.** The 224 figure was the whole `nvjet_tst` NAME family:
6328 calls and 221.3 ms, of which the head is 3840 calls and 190.0 ms. The remaining 2488 calls
and 31.2 ms are `aten::bmm` and `aten::baddbmm` — attention's batched matmuls, confirmed by
correlation id to cpu_op. Same naming-family trap as the `_scaled_mm` misattribution: a regex on
a kernel-name prefix is not a semantic group.

| scope | ideal 2× saving | % of a 1702 ms step |
|---|---|---|
| forward only | 31 ms | 1.8% — **under the ship gate** |
| forward + grad_input | 63 ms | 3.7% |
| all three | **95 ms** | **5.6%** |

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
  directly and drops the Liger dependency for this path. **The memory cost is now priced and it
  is small (b0's number, my verification): 2048 × 32784 × 2 bytes = 134.3 MB for one chunk, and
  `logits_chunk` is created inside a SEQUENTIAL `for chunk_id in range(num_chunks)` loop
  (fused_linear_cross_entropy.py:108-114), so one chunk is live at a time, not all 64.** The 8.6 GB
  worst case b0 asked about does not occur. 134.3 MB against 52.7 GB used of 55 is 0.25% of card
  memory, so my original rejection of B rested on a cost that is negligible.
- **C. Keep FLCE, precompute the projection in fp8 outside it.** Not viable: the fusion's whole
  point is that logits never leave the kernel.

**A remains the default, but on a different argument than the one I first gave.** My original
reason for rejecting B — an unbudgeted memory regression — is void: the cost is 134.3 MB, one
chunk at a time. What survives is narrower: A keeps Liger's in-place logits-gradient write and
its fused CE kernel, so it changes only the matmuls and leaves the numerically delicate part
untouched, whereas B reimplements the loss itself on the path where an error reaches every
gradient. A is the smaller correctness surface, not the smaller memory footprint. Vendored
rather than monkeypatched so the version is pinned in-tree.

## The two ways it is wrong, and the gates

The head feeds the loss directly, so an fp8 error lands on **every** gradient, not on one layer's
activations. That is a stronger parity requirement than the body linears carry.

1. **Range.** Measured on the live run's step-7000 checkpoint, and it comes out **benign for the
   weight**: head `absmax 26.875`, `std 4.32`, so `w_scale = 26.875/448 = 0.060`. Per-vocab-row
   absmax is p50 15.69, p99 19.84, max 26.88 — **max/p50 is only 1.71×**, so no single row
   dominates and per-tensor scaling costs little. What is still unmeasured is the **activation**
   side: the pre-softcap projection is unbounded and one outlier logit sets the scale for a
   32784-wide chunk. **MEASURED and the gate passes** (`eff.fp8_head_activation_range`): over 128
   chunks from step-11000 on real cot rows, per-chunk pre-softcap absmax is p50 48.05, p99 62.24,
   max 82.43 against e4m3's 448. Zero chunks clip, the largest is 5.4× under the ceiling, and the
   spread that decides the question is **p99/p50 = 1.30×**. So **per-tensor scaling is sufficient**
   and the per-chunk / amax-history branch is not needed — the design's largest open risk is closed
   and its most complicated option is removed. Measured on the forward projection only; the
   backward's `grad_logits` distribution is the other fp8 operand and is not yet measured.
2. **Loss curvature.** Cross-entropy is sensitive where the softmax is confident; a quantisation
   error that is invisible in the projection can move the loss. Gate: `|Δval| ≤ 0.04 nat` on a
   50-step A/B, which is the standing rule, plus a parity check of loss and grad-norm against
   the bf16 path on the same batch.

### Ship gate, stricter here than for the body linears (fb, 2026-09-01)

This is the loss path, so the standing gate is not sufficient on its own:

| check | bar |
|---|---|
| throughput | tok/s ≥ +3%, 50 steps, 7 cards |
| validation | `\|Δval\| ≤ 0.04 nat` |
| per-domain loss | every domain's delta within seed noise (σ̂ 0.0516) |
| digit/FoNE probe | **assertion, not a note** (b0): the fp8 head path must REFUSE when `Cfg.fone` is set until the digit probe has run against it. `--fone` changes the token stream and the head's input distribution, and an n/a note is a check on the reader where an assertion is a check on the code. b0 writes it when the head path lands. |
| parity | loss and grad-norm against the bf16 path on the same batch |

The per-domain bar is the one the aggregate can hide: a single domain moving while the mean
holds is exactly what an fp8 range failure on rare tokens would look like.

### GEMM 3 is an accumulator, not a gradient — and we are already losing precision there

b0 raised this against option A and it is right about the mechanism, but the conclusion inverts
once you check which branch our config takes.

FLCE accumulates `grad_weight` across all 64 chunks. `fused_linear_cross_entropy.py:233-241`
has an fp32 path — `torch.addmm(grad_weight, grad_logits_t, input_chunk, out_dtype=torch.float32,
out=grad_weight)` — but it is **guarded on `grad_weight.dtype == torch.float32`**, and
`grad_weight` is `zeros_like(weight)` when `accum_dtype is None` (:76). We never pass
`accum_dtype`, and our weights are **bf16** (verified in the step-7000 checkpoint: `head.weight`
and `tok.weight` are both `torch.bfloat16`, per t01's bf16-params-no-fp32-master decision).

**So the fp32 branch never fires for us.** We take the `else` at :257,
`grad_weight += torch.mm(...).float()`, which computes the product in fp32 and then rounds the
running sum back to bf16 on every one of the 64 `+=`.

Measured on the pod, 64 chunk contributions accumulated bf16 vs fp32: **median relative error
0.803%**, p99 33%, cosine similarity 0.999954. b0's independent 0.75% figure reproduces.

Three consequences, and the third is the one that matters for scoping:

1. b0's hazard is real but **pre-existing**, not introduced by fp8. The bf16 accumulator is
   today's behaviour on the live 15B run.
2. It cannot be a reason to prefer B over A, since B would have to reimplement the same
   accumulation and would face the identical choice.
3. **It is a separate, possibly larger lever than fp8 on this GEMM.** Passing
   `accum_dtype=torch.float32` is one keyword argument, costs 33.6M × 4 bytes = 134 MB of fp32
   accumulator, and buys back a 0.8% systematic bias on every head weight gradient. That is a
   correctness change with no throughput claim, so it is not gated on the 3% rule — and it should
   be tested independently of the fp8 work rather than bundled into it.

For the fp8 design itself, b0's operational points stand and are adopted: GEMM 3's replacement
must not silently substitute a bf16 output for whatever accumulator is in force, the vendored
loop should **assert the accumulator dtype** rather than assume it, and `grad_weight` gets an
elementwise comparison against the current path on one step before any training A/B — a
systematic 0.8% bias is exactly what a 50-step `|Δval|` check cannot see.

### The forward-only refusal is the gate working

Forward-only measures 37 ms = 2.2%, and it is refused for being under 3% even though it is the
smallest and safest diff available. That is the ship gate doing its job rather than an obstacle
to route around: a 2.2% change to the loss path carries the same parity risk as a 6.6% one and
a third of the return.

## What this design does not establish

The 112 ms is an **ideal** 2× on measured GEMM time. It will not be realised in full: scaling
work is not free, FLCE's chunking overhead is untouched by a faster matmul, and the
`_scaled_mm` path has its own alignment constraints at N=32784. No implementation exists to
measure yet, and the 224.0 ms itself comes from a single-card trace
(`eff.steady_state_composition`), so the 7-card share may differ. Treat 5.6% as the ceiling to
measure against, never as the expected gain. The 190.0 ms is single-card, so the 7-card share
may differ, and every percentage above uses a 1702 ms step derived from 77K tok/s rather than a
timed step. The 190.0 ms is single-card **at batch 16 accum 2** (b0): the head's chunk count
scales with BT, and it is the 64 chunks that make the accumulator question sharp — at a different
batch the accumulation error and the per-chunk absmax distribution both change.
