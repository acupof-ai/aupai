# Pivot: V2 abandoned, target is DeepSeek-V4.1-Flash (user order 2026-09-10)

User order 2026-09-10: stop all V2 work; build DeepSeek-V4.1-Flash's architecture.
Source: DeepSeek_V41_Tech_Report.pdf (51 pp), read in full 2026-09-10. The acceptance gate is
unchanged: HumanEval pass@1 >= 30% at ~350M (docs/standards/p1_data_recipe.md:256).

## What V4.1 is, in the terms we will build

- **CED, Causal Encoder-Decoder.** 40 causal layers at 552B = 20 encoder (bottom) + 20 decoder
  (top). Decoder layers do NOT project their own global KV: every decoder layer's global KV entries
  C^l and compression weights Z^l are projected directly from the final encoder hidden state
  H_{L/2} (Eq.1, §2.2), per-layer unshared weights. Decoder layers keep their own Q and their own
  local SWA KV. Attention is causal everywhere — not bidirectional.
- **CSA2, Compressed Sparse Attention 2.** Learned non-overlapping m-token KV entries; a DEDICATED
  lightweight indexer (indexer-Q from H, indexer-K over the entries) selects top-k entries; then
  ONE concatenated softmax over [selected global entries ; local SWA window keys]. No branch gate.
  Layers run in modes Full (build the package: entries, indexer-K, top-k indices), Reuse (consume
  the package, own Q + SWA only), or Reindex (consume KV, re-run own indexer). This is cross-layer
  KV reuse and it replaces HCA.
- **SWA local branch in every layer**, pure SWA for the first two layers, n_win=128 at 552B.
- Partial RoPE retained (we already have it, model.py:541). KDA / NoPE is dropped — V4.1 has no
  recurrent state.
- MoE in every block, DeepSeekMoE 1-shared + routed, aux-free bias selection (we already have this).
- **Deferred past the gate (single-node 350M / 4096 context):** FP4 KV cache + QAT, SSD/host KV
  tiers, SWA bounded replay's prefill-skip scheduling, Hierarchical Sparse Indexer (16k candidate
  pool), DSpark drafter, Engram n-gram memory, mHC kernels, head-wise Muon, the vision stack, the
  64K/1M curriculum and all 552B-scale hyperparameters.

## Build order

- **Step 0, config on paper first** — L, CSA2 mode map, m, indexer geometry, top-k, n_win, all as
  Cfg fields so checkpoints carry them. No 552B value is copied; every number is re-chosen for
  d=1024/h=8/seq=4096.
- **Step 1, CSA2 core in a flat stack** (no CED): rewrite CompressedSparseAttention — learned
  entries replace mean-pool; dedicated indexer; single concatenated softmax over entries + SWA;
  delete branch_gate and the full-resolution select branch. Behind a csa2 flag.
- **Step 2, Full/Reuse cross-layer package**, threaded through the body. Skip Reindex (decoder-only).
- **Step 3, SWA placement** — pure-SWA first-two-blocks option; n_win 128 vs 256 A/B.
  Steps 1-3 = a complete trainable flat CSA2 model and the first HumanEval-gate candidate.
- **Step 4, positional encoding**: rope_dims>0, attn_every=1, KDA blocks replaced by CSA2/SWA.
- **Step 5, CED topology WITHOUT prefill skip** (Eq.1), full teacher-forced; A/B vs flat CSA2.
- **Steps 6-7, prefill skip + bounded replay in the inference harness** — only if Step 5 wins.

## Decisions taken now (controller, with the gap-map reasoning)

- SUPERSEDED 2026-09-10 by the user's option A (faithful V4.1): CED is IN SCOPE as Step 5 (6
  encoder + 6 decoder at L=12, Eq.1 decoder package from H_{L/2}, teacher-forced, no prefill
  skip), owned by de as de-105. The flat CSA2 stack still ships first: it is the CED encoder and
  the control arm. The first gate run uses whichever of the two is trainable when the UltraData
  shards land; the other is the A/B.
- **KDA deleted on this line.** V4.1 carries position by RoPE; a KDA+CSA2 hybrid is unaddressed by
  the paper and CED makes it near-impossible (recurrent state cannot flow into H_{L/2}-projected
  decoder KV). HCA not wired into new layers; its shared helpers stay.
- **AttnRes**: do not carry across the CED boundary; evaluate within-half only if CED is built.
- MoE: reuse MoEFFN as-is, all blocks, 48/top-3/1-shared — the 384-expert / top-6 / expert-2304
  numbers are 552B-only. Keep SiTU-GLU; do not add SwiGLU clamp without an A/B.
- **The data recipe CHANGED 2026-09-10 (user order).** Teacher synthesis (textbooks, exercises;
  de-101 / 0e-1) is stopped and dropped. The p1 code+exercise corpus is openbmb/UltraData-Code L2
  (natural code) and L3 (exercises with tests), python subsets, fetched from hf-mirror,
  decontaminated against HumanEval/MBPP, mixed with the existing math/CoT/en domains (task 0e-3,
  PR #221). Only the highest-quality tier is kept; the 2.8116B keep set stays as a domain. The
  tokenizer is re-measured on an UltraData sample before the gate run (ae).
