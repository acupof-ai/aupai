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

## Config cross-check against the released config.json (fb, 2026-09-10)

Source: `docs/audits/deepseek_v41_flash_config.json`, fetched from
hf-mirror `deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/config.json` (text_config). Our column
is train.py `Cfg` as wired by PR #222. "open" rows are for de (model) and ae (Step 0 sizing) to
close before the gate run; each needs a one-line ruling in this table, not a new doc.

| field | V4.1-Flash | ours | status |
|---|---|---|---|
| SWA-only layers | compress_ratios 0 at layers 0,1 (then 3 MTP layers at the end) | n_swa_only_layers 2 | match |
| sliding_window | 128 | csa2_n_win 128 | match |
| qk_rope_head_dim | 64 (partial RoPE, theta 10000) | rope_dims 64 | match |
| candidate_block_size | 8 | csa2_m 8 | match |
| index_topk | 512 entries (4096 tokens at 8/entry; seq up to 1M) | csa2_top_k 64 (512 tokens at seq 4096) | open: a top-k that is 1/8 of the sequence is a re-chosen number; ae states the ratio rule in scripts/v41_size.py |
| candidate_topk_blocks | 2048 (hierarchical indexer pool) | none | out of scope (HSI struck) |
| kv_source_layer_ids | [2, 8, 14, 20]: 3 Full layers in the 18-layer encoder + the first decoder layer | csa2_modes F,R,R,R,X,R,R,R,R,R: one Full source for 10 layers | open: paper refreshes KV every 6 layers; at L=12 that is F at 2 and 8. de rules F,R,R,R,R,R,F,R,R,R or keeps one source with a reason |
| index_source_layer_ids | [2,8,14,20,24,28,32,36] | X at position 4 (runs as Reuse; Reindex deferred) | deferred (Step 2 Reindex) |
| compress_ratios | 2 in encoder layers 2-19, 1 in decoder layers 20-39 | not modelled | open: de reads what the ratio compresses (KV width vs entry stride) against the tech report and states whether CSA2Package carries it |
| num_experts_per_tok / n_shared / n_routed | 6 / 1 / 384 | 3 / 1 / 48 | re-chosen for 350M active (Decisions above) |
| moe_intermediate_size | 2304 | moe_expert_ffn 1728 | re-chosen (v41_size.py) |
| scoring_func / topk_method / norm_topk_prob / routed_scaling_factor | sqrtsoftplus / noaux_tc / true / 1.5 | sigmoid-style gate, aux-loss-free bias (moe_bias_gamma), no routed scaling | open: de states our router's exact function and whether sqrtsoftplus + 1.5 scaling is adopted; A/B if changed |
| hidden_act / swiglu_limit | silu / 10.0 (clamped SwiGLU) | SiTU-GLU (tanh-bounded) | open: Decisions said no clamp without an A/B; the released config HAS the clamp. de files the A/B or adopts |
| rms_norm_eps | 1e-20 | 1e-6 | open: trivially adoptable; de rules |
| head_dim / num_key_value_heads / q_lora_rank / o_lora_rank / o_groups | 512 / 1 / 1280 / 1024 / 8 (MLA latent geometry) | GatedMLA latent at d 1024, h 8, head_dim 128 | re-chosen; ae records our latent dims beside these |
| hc_mult / hc_sinkhorn_iters (mHC) | 4 / 20 | none | struck (Decisions) |
| engram_layer_ids | [1, 14] | none | struck |
| num_nextn_predict_layers (MTP) | 3 | none | struck for the gate run |
| rope_scaling | yarn x16 from 64K | none | irrelevant at seq 4096 |
| tie_word_embeddings | false | untie_head flag exists, default tied | open: ae counts the parameter cost at vocab 32784 and rules |
| vocab_size | 129280 | 32784 frozen; re-measured on UltraData by ae | ae ruling pending |

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
- **The data recipe CHANGED 2026-09-10 (user order); state as of 2026-09-11.** Teacher
  synthesis (textbooks/exercises, de-101) is stopped; the gate corpus is
  openbmb/UltraData-Code L2 (natural code) and L3 (task/analysis/solution/test) python
  subsets, fetched from hf-mirror and kept under 0e's L2/L3 keep rules (#237), 13-gram
  decontaminated against HumanEval/MBPP (`filters/decontam_ngram.py`; results in
  facts/contamination.json), and mixed with the existing math/CoT/en domains and the
  2.8116B classifier keep set (now `code_keep_p1`, assembled flat by
  scripts/assemble_keep_p1.py). The mix is `data/mix_v41_gate.json`: total_tokens 30.0B,
  one epoch, eight domains (PR #246; weights re-normalise to 0e's measured totals).
  **Tokeniser: rebuilt 2026-09-10 to 32,768 slots under unfreeze condition 2** — the
  measured freeze tax on UltraData was ~8.5% (PR #233,
  facts/tokenizer.json#tok.ultra_freeze_tax_0910); the 32,773-slot vocab is preserved on
  the pod as `data/tokenizer_frozen_0829.json`, and every gate cache is stamped at the new
  vocab f1f860970d15d623. Gate-run recipe (fb 2026-09-11, prereg v41_gate_0911 amendment 1):
  world 6, B4/accum8, 786,432 tokens/step, 38.1K steps; warmup 500, warmdown 0.65,
  anneal_frac 0.10.
