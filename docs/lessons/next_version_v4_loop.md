---
question: "what is the adoption loop from SMELT and DeepSeek-V4 for our next version, and what does the NoPE/RoPE break cost"
status: open
source: "facts/deepseek_v4.json (arXiv 2606.19348, read 2026-09-08) + facts/smelt_deeploop.json (arXiv 2609.01343); 4c's 2026-09-08 brief"
---

# Next-version loop: SMELT + sparse MoE + CSA-with-SWA + HCA

The two papers point at the same target from different scales. SMELT
(arXiv 2609.01343, our recorded facts) measures the compute-matched CE gain
from looped middle layers + sparse MoE at 100M-1.6B active params. DeepSeek-V4
(arXiv 2606.19348) ships the production-scale version of the same instincts:
sparse MoE everywhere, compressed+sparse attention for 1M context, and a
positional-encoding change that is the one part we cannot adopt for free.

## The loop, in dependency order

**Loop 1 — SMELT looped layers + sparse MoE. No attention change, no PE change.**
The middle 50% of physical layers execute twice per forward, every layer is a
sparse MoE FFN (SMELT: top-8; V4: top-6 with 256/384 routed experts + 1 shared),
and every sublayer update inside the looped span is scaled by 1/r to stop
weight-tied updates inflating the residual stream. SMELT measured 6.8-18.0%
compute-matched CE gain at 1e20-1e21 FLOPs, growing ~8pp per 10x compute. This
loop is checkpoint-compatible: it touches the FFN/MoE path only, and our
checkpoints are NoPE so there is no position encoding to preserve. V4's MoE
details that sharpen SMELT's recipe: Sqrt(Softplus) affinity instead of Sigmoid,
aux-loss-free load balancing + a slight sequence-wise balance loss, Hash routing
in the first 3 layers, and the routing-target-node constraint removed.

**Loop 2 — CSA-with-SWA for long context. This is where the PE break lives.**
CSA compresses every m=4 tokens into one KV entry (overlapped grouping, 2m
neighbors), a Lightning Indexer selects top-k=512 compressed entries per query,
and an SWA branch restores the n_win=128 most recent uncompressed KV entries so
a query keeps the local detail its own compressed block lost. This is the
1M-context mechanism: Pro reaches 27% of V3.2's single-token FLOPs and 10% of
its KV cache at 1M. CSA is the layer type that needs the positional-encoding
decision, because V4's CSA ships with partial RoPE (see below).

**Loop 3 — HCA for the layers that do not need sparsity.**
HCA is CSA without the Lightning Indexer: dense attention over KV compressed at
m'=128. V4 interleaves CSA and HCA, and Pro's first 2 layers are pure HCA. HCA
is the cheaper layer for the positions where sparse selection buys nothing
(early layers, or layers whose attention pattern is already diffuse). It carries
the same partial-RoPE dependency as CSA.

## The NoPE/RoPE break, stated as a cost

Our checkpoints are NoPE. `GatedMLA` (model.py:227) carries no positional
encoding; position is handled by KDA/DeltaRecurrence. DeepSeek-V4 uses partial
RoPE on the last 64 dimensions of queries, compressed KV entries, and
attention outputs, with a -i position correction on the outputs so the result
encodes relative rather than absolute position. V4 ships this PE change inside
the same layers as CSA/HCA — the attention upgrade and the PE change are not
separable in their design.

The cost of Loop 2 is therefore not just the attention implementation. It is
one of:

- **PE conversion**: map the NoPE checkpoint into a partial-RoPE world. No
  measured path exists; the 64-dim partial RoPE with the -i output correction
  is not a standard NoPE->RoPE upgrade, and a conversion that preserves the
  KDA recurrence is unstudied.
- **Attention-path re-pretrain**: keep the NoPE checkpoint for the FFN/MoE path
  and re-pretrain only the attention path under partial RoPE. Cheaper than a
  full re-pretrain, but it forfeits the SMELT-style "continue the checkpoint"
  property for the attention weights, and the interaction between a
  re-pretrained attention path and a continued NoPE KDA path is unmeasured.

Loop 1 does not pay this cost. Loop 2 does. The break is the gate between them:
we can take the SMELT + sparse-MoE gain on our current checkpoints, and we can
take CSA/HCA only by also taking the partial-RoPE decision and its conversion
or re-pretrain cost.

## What is not yet answered

- Whether CSA's Lightning Indexer works as described under NoPE — V4 gives no
  NoPE ablation, so the PE-attention coupling is asserted by construction, not
  measured.
- Whether SMELT's 1/r loop scaling holds under V4's mHC residual (n_hc=4,
  doubly-stochastic B) instead of a standard residual. SMELT measured with a
  plain residual; mHC changes the residual stream the looped updates land in.
- The partial-RoPE dimension (64) is small relative to V4's head dims
  (d_c=512/1536). Whether a 64-dim RoPE carries enough position signal at 1M
  context, or whether the SWA branch (n_win=128) is doing the real positional
  work, is not separated in the paper.
