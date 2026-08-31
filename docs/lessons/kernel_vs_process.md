---
question: "is our throughput headroom in kernels or in process, and does the HF kernels hub help"
status: recorded
source: "e1-3 2026-09-01; eff.step_remainder_attribution (t56/tilerl-4), eff.lm_head_is_compute_bound, eff.kda_occupancy_bound, eff.dynamo_recompile_not_a_lever, eff.max_autotune_dynamic_shape_noship"
---

# Kernels or process

Research, 2026-09-01 (e1-3). The question is where stage-1 throughput headroom
lives and whether the Hugging Face kernels hub reaches any of it.

## The answer is decided before any literature arrives

GPU kernel time is **1607 ms/step**. The idle beside it depends on which step
figure you divide by, and the two available figures come from different time
bases:

| base | step | idle | busy |
|---|---|---|---|
| same trace window (1607 + measured 186.6 gap) | 1793.6 ms | 186.6 ms | **89.6%** |
| steady state derived from 77K tok/s | 1702.0 ms | 95.0 ms | 94.4% |

**Quote 89.6%.** `eff.step_remainder_attribution` records both: its opening text
says 94% and its own correction later in the same entry withdraws that as mixing
two time bases, since 1607 ms came from the trace window and 1702 ms from
throughput. I read the stale half first and fb caught it. The same-window pair
is the one measured against a single clock.

So the process-side budget is **95–187 ms, 5.6–10.4% of a step**, and only the
part not already overlapped is recoverable. Against that, three kernel-side
levers at their ideals:

| lever | ideal ms | % of step |
|---|---|---|
| fp8 path through the LM head | 112.0 | 6.2–6.6 |
| fusion audit cutting 20% of fused-region time | 103.5 | 5.8–6.1 |
| KDA retune doubling occupancy-limited throughput | 87.0 | 4.9–5.1 |
| **sum, all three, all ideal** | 302.5 | **16.9–17.8** |

**The headroom is in kernels.** At either bound the kernel side is three times
the process side, and the single largest kernel lever alone matches or exceeds
the entire process-side ceiling. The conclusion does not depend on which base is
right, which is why it was worth stating before the survey returned.

A steady-state trace replacing both numbers with one measurement is running on
the lane and lands ~00:45. It will move these figures inside the range above; it
will not move the ranking.

Two prior measurements say the same thing from the other direction, and both are
retractions of earlier process-side recommendations:

| fact | what it killed |
|---|---|
| `eff.dynamo_recompile_not_a_lever` | cu bucketing as t57's first lever: the recompiling frame carries 2.2 ms/step, 0.1% of compiled-region time. Ceiling ~0%. |
| `eff.max_autotune_dynamic_shape_noship` | max-autotune: 106 GEMM choices re-benchmarked per dynamic cu shape, never reaches steady state, 1K tok/s against a 72K baseline. |

So the two most obvious process levers have already been measured and are gone.
This document should not re-propose them, and it does not.

## The attribution, with per-launch cost

From `eff.step_remainder_attribution`, one 20-step trace, lane card, stage-1
shapes, 100% named coverage. Percentages of step use the same-window base
(1793.6 ms). The µs/launch column is mine and it is the one that settles the
launch-overhead question.

| group | ms/step | % kernel | % step | launches | µs/launch |
|---|---|---|---|---|---|
| inductor fusion | 517.6 | 32.2 | 28.9 | 3762 | 137.6 |
| fp8 GEMM | 493.1 | 30.7 | 27.5 | 378 | 1304.5 |
| bf16 GEMM LM head | 224.0 | 13.9 | 12.5 | 328 | 682.9 |
| KDA gated-delta | 174.0 | 10.8 | 9.7 | 324 | 537.0 |
| elementwise-copy | 107.0 | 6.7 | 6.0 | 1816 | 58.9 |
| flash attention | 44.4 | 2.8 | 2.5 | 12 | 3700.0 |
| liger FLCE | 38.5 | 2.4 | 2.1 | 64 | 601.6 |

**Launch overhead is not our problem.** At 6684 launches/step, pure overhead
costs 13 ms at 2 µs/launch and 67 ms at 10 µs. Overhead only bites when it is not
overlapped, and total idle is 95–187 ms. The mean kernel is 138 µs, two orders of
magnitude above per-launch cost. A step whose kernels average 138 µs is not
launch-bound; it is kernel-bound on a nearly-saturated card.

The one group where launch cost is plausibly material is elementwise-copy at
58.9 µs mean over 1816 launches — still 30× a 2 µs launch, so even there the
copies dominate their own dispatch.

## The idle is one seam, not many launches

The 186.6 ms of idle is not spread evenly, and that changes what a process-side
lever would even be. From `eff.dynamo_recompile_not_a_lever`, over 138,460 gaps
in the profiled window:

| gaps | total |
|---|---|
| 95.3% of gaps, each under 5 µs | 10.2 ms |
| the largest 20 gaps | 164.9 ms |

The largest single gap is 645 ms, and all twenty sit at **the same
`rms_norm → flash` seam**, with `compile_attempt_1` 81 ms, `bytecode_tracing`
36 ms and `build_guards` 34 ms inside them.

So roughly 165 of 187 ms of idle is twenty events at one boundary, and 10 ms is
spread across 130,000 small ones. Reducing launch count attacks the 10 ms.
Fixing one seam attacks the 165 ms. These are different projects with a 16×
difference in ceiling, and only the second is worth a row in the ranking.

The caveat that keeps this honest: those twenty gaps are compile events, and the
window is a warmup window. `eff.dynamo_recompile_not_a_lever` already measured
what survives into steady state and found the recompiling frame carries 2.2
ms/step, 0.1% of compiled-region time. So the seam may largely be a warmup
artifact — which is exactly what tilerl's steady-state trace will settle. If the
seam persists at steady state it is the single best process lever; if it does
not, the process side collapses toward the 10 ms of small gaps and the answer to
fb's question becomes emphatic rather than merely clear.

## Half A: does the kernels hub reach any of it

**Short answer: no, and the one lead worth chasing is not on the hub.**

`kernels` (`pip install kernels`, `get_kernel` from the Hub, torch≥2.5) fetches
**prebuilt** artifacts at runtime, keyed by a build variant like
`torch213-cxx11-cu130-x86_64-linux`. Better than feared on the compile question:
native kernels must register as real custom ops in `torch.ops.<id>`, so inductor
sees an opaque-but-schematized op rather than a graph break, and `kernelize(model,
mode=Mode.TORCH_COMPILE)` resolves dispatch ahead of time instead of branching at
runtime. Two traps: inductor can schedule around a custom op but **cannot fuse
into it**, and `kernelize` **silently falls back** to the original layer when a
kernel does not declare `can_torch_compile` (default `False`) — pass
`use_fallback=False` or you will believe a kernel is active when it is not.

Against our table:

| our group | ms/step | hub candidate | verdict |
|---|---|---|---|
| inductor fusion | 517.6 | none | a hub kernel adds an unfusable op; it cannot reduce fused-region count |
| fp8 GEMM | 493.1 | finegrained-fp8, deep-gemm, fp8-fbgemm | inference/quantized-GEMM oriented, not an fp8 *training* path; already at 30.7% of kernel time and not the weak spot |
| bf16 LM head | 224.0 | none for a tied fp8 head | the lever is our own `_fp8_ok` exclusion, not a missing kernel |
| KDA gated-delta | 174.0 | `kernels-community/fla` | **the same Triton we already run** |
| elementwise-copy | 107.0 | activation, rmsnorm, rotary, layer-norm | all sm90-valid, no published benchmarks, and inductor already fuses these |

**On question C, the expected answer was almost right but for the wrong reason.**
`kernels-community/fla` does exist and exports `chunk_kda`,
`fused_recurrent_kda`, `chunk_gated_delta_rule`. But its `metadata.json` has no
`archs` key and its single variant is `build/torch-cuda` — the leaves are
`@triton.jit` source, built from upstream `addf474`. It is the flash-linear-
attention library we already pip-install, redistributed. Same code, same speed, a
new runtime Hub dependency, 0 downloads. **Do not adopt it.**

Nothing in the hub is tuned for H20. That matters more here than usual: H20 is
sm90 with ~148 bf16 TFLOPS against ~4 TB/s, a ~6.7× FLOP derate from H100 at
near-identical bandwidth, so tile and occupancy choices tuned on H100's ratio are
mistuned for ours. Prebuilt sm90 binaries are *valid* on H20 (same compute
capability); nobody publishes H20 validation for them.

### FlashKDA: already known here, and inference-only

The survey surfaced MoonshotAI/FlashKDA with H20-specific benchmarks — 1.85–2.31×
over `fla chunk_kda` at T=8192, D=128. We already have this recorded:
`eff.kda_kernel_path` says FlashKDA is inference-only, and eight files in this
repo set `FLA_FLASH_KDA=0` because the backend needs fp32 `A_log`/`dt_bias` and
errors during validation otherwise. The Hub build's own verifier rejects it with
"FlashKDA only supports inference mode" when `torch.is_grad_enabled()`. So the
2.3× is real silicon on our exact card and **cannot touch the training step**.

### The one genuinely new lead: FlashQLA

[QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA), TileLang, claims **2–3×
forward and 2× backward** over the FLA Triton kernel, supports sm90, and plugs in
through the standard FLA API. The backward claim is what distinguishes it from
everything else here — it is the only candidate that could touch our 174 ms/step.

Not on the kernels hub, so the mirror-chain cost does not apply; it needs CUDA
≥12.8 and torch ≥2.8, which is a version question we must check against the pod
before anything else. Its published benchmarks are H200 and GB200, **no H20**,
and given the FLOP derate those numbers do not transfer. Treat the 2× backward as
a hypothesis to measure on our card, not a number to plan with.

## Half B: the process-side levers, and one that is not process-side at all

Three of the four items in the brief are already dead or bounded, and the one
useful finding turned out to be a memory-traffic lever wearing a config flag.

| lever | verdict |
|---|---|
| cu bucketing | dead. `eff.dynamo_recompile_not_a_lever`: recompiling frame carries 2.2 ms/step, 0.1% of compiled-region time. |
| max-autotune | dead. `eff.max_autotune_dynamic_shape_noship`: 1K tok/s against a 72K baseline. |
| CUDA graphs under dynamic shapes | bounded and risky, below. |
| launch-count reduction | bounded at ~33 ms, 1.9%, and unreachable. |

**CUDA graphs deserve a specific warning rather than a bounded dismissal.** Our
shapes are fixed at 16×4096, so capture is feasible, but vLLM downgrades FA2 from
full cudagraphs because the launch configuration freezes at capture shape — their
[PR #20059](https://github.com/vllm-project/vllm/pull/20059) reports a kernel
launching 494 blocks/SM when eager needed 4.6. Our KDA kernels already run at
6–12% occupancy from register pressure. Freezing a launch config on top of that is
a live risk, not a hypothetical, and it points the wrong way.

The scale comparison also settles why vLLM's cudagraph gains do not transfer.
vLLM measures ~5 ms of GPU exec time per step for Llama-8B on H100, which is why
per-launch CPU overhead binds there. Ours is 1607 ms, roughly 320× larger. Their
+8.6% to +24% is a number from a launch-bound regime we are not in.

### The one lever worth having: `pad_dynamic_shapes`

`torch._inductor.config.pad_dynamic_shapes` defaults to `False`, and the gate is
in `ir.py`, verified verbatim in the pod's torch 2.11:

```python
is_dynamic = not all(
    isinstance(s, (int, sympy.Integer))
    for s in itertools.chain(in_strides, size)
)
if not config.pad_dynamic_shapes and is_dynamic:
    return in_strides
```

`is_dynamic` is true if **any** size or stride is a symbol. Our doc-mask `cu` is
marked dynamic, so inductor skips 128-byte stride padding on pointwise and
reduction buffers across the graph — 64 elements for bf16, 128 for fp8. Measured
on the pod: `pad_dynamic_shapes False`, `comprehensive_padding True`,
`padding_stride_threshold 1024`. So padding is enabled in general and disabled
*specifically because our shapes are dynamic*.

This is the only lever found tonight that aims at memory coalescing rather than
launch count, and coalescing is what governs the 517.6 ms of fusion plus the
107 ms of copies — **39% of kernel time**.

**The sign is not guaranteed and this must be A/B'd, not assumed.** PyTorch's own
in-source note records padding swinging AllenaiLongformerBase amp training from a
1.09× loss to a 1.05× win (71.09 → 77.38 → 67.77 ms) depending on
`padding_stride_threshold`. Padding creates non-contiguous strides, and inductor
then uses a smaller persistent-reduction threshold and can emit worse code.

## Ranked levers

Scored on the three axes fb asked for. "% of step" uses the same-window base.
Expected gain is a fraction of an ideal that is explicitly unreachable.

| # | lever | % of step touched | expected gain | ship cost |
|---|---|---|---|---|
| 1 | `pad_dynamic_shapes=True` | 34.9% (fusion + copies) | unknown sign, ±1–5% | one line, one A/B on the lane |
| 2 | fp8 path through the LM head | 12.5% | ≤6.2%, less in practice | train.py change, frozen until between-stages |
| 3 | FlashQLA for KDA fwd+bwd | 9.7% | claims 2× bwd, unverified on H20 | source install, microbenchmark first |
| — | fusion audit (273 distinct regions) | 28.9% | unknown | a day of work, no measurement yet |
| — | anything from the kernels hub | 0% | none | blocked: pod torch 2.11 vs matrix 2.12+ |

**Rank 1 is first because it is one line and touches the largest surface.** It is
also the only one shippable tonight: it is an inductor config, not a train.py
edit, so the freeze does not apply — though it does change compiled output, so it
belongs in a lane A/B before it goes near the block.

**Rank 2 is the largest certain gain and the slowest to ship.** `_fp8_ok` excludes
the head by name; the arithmetic intensity is 668.7 FLOP/B against H20's 37 FLOP/B
balance point, so it is eighteen times inside the compute-bound region and an fp8
path is real. It waits for the between-stages window.

**Rank 3 is the only external dependency worth the risk**, and only after a
microbenchmark: FLA against FlashQLA on our shapes, forward and backward, on the
lane card. H200 numbers do not transfer at H20's FLOP derate.

The fusion audit is listed without a rank because nobody has measured what a 273→N
reduction would return. It is the largest surface in the table and the least
understood, which makes it the right thing to measure next rather than the right
thing to ship next.

## Ship cost that applies to every hub row

Two costs, and the first one is fatal on its own.

**The pod's torch is 2.11.0+cu129. The `kernels` prebuilt matrix covers torch
2.12, 2.13 and 2.14.** No build variant matches, so `get_kernel` has nothing to
resolve to. The hub path is closed on version grounds regardless of anything
else, and the fix — moving torch under a running stage-1 job — is not a fix
anyone should want. Measured on the pod, not inferred:

| | pod | needed |
|---|---|---|
| torch | 2.11.0+cu129 | 2.12 / 2.13 / 2.14 for a hub variant |
| CUDA | 12.9 | ≥12.8 for FlashQLA |
| compute capability | 9.0 (H20) | sm90 for both |
| fla | 0.5.2 | — |

**Pod egress to huggingface.co is broken**, so even with a matching torch, every
`get_kernel` fetch needs the mirror chain. `kernels lock` plus `kernels download`
would move that cost from runtime to install time, which is the right shape, but
it is a second blocker behind the first.

FlashQLA is unaffected by both: it is a source install from GitHub, and the pod's
torch 2.11 / CUDA 12.9 / sm90 all clear its stated requirements. That is the
practical reason it outranks everything on the hub, separate from it being the
only candidate with a backward-pass claim.

## Ceilings

- Everything here rests on one 20-step trace on one card. The fact notes the
  profiled job matched the 7-card run's 77K tok/s and MFU 32%, so the shapes are
  representative, but a single trace is a single trace, and its window is a
  warmup window rather than steady state.
- The idle figure is a range, 95–187 ms, because the two available step figures
  come from different clocks. tilerl's steady-state trace (~00:45) replaces both
  with one measurement. It moves the numbers inside the range; it does not move
  the ranking, since the kernel side is 3× the process side at either bound.
- The busy figure to quote is **89.6%**. The 94.4% in the same fact's opening
  text was withdrawn by its own later correction, and I quoted the stale half
  before fb caught it — a reminder that a long fact entry can contain both a
  claim and its retraction.
- Ideal ceilings (fp8 2×, doubled occupancy, 20% of fusion time) bound the
  search. None is achievable in full and none should be quoted as a forecast.
