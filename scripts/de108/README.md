# de-108: CSA2 joint entries+window softmax on flash (spike / WIP)

Replaces the materialized `B,H,T,(T+T/m)` score tensor in `_forward_csa2` with two
flash calls joined by an fp32 LSE combine. B8 T4096 H8: sc_cat is 4.50 GiB fp32/layer
x10 CSA2 layers = ~45 GiB forward residency (~90 with backward; the 94.62 GiB OOM on
v41_smoke_0911h). Post-flash: window in-kernel, entries branch B·H·T·64 = 0.0625
GiB/layer (0.6 GiB x10).

These scripts are the proven de-risking pieces (2026-09-11). The model wiring and the
ragged-gather backward are not yet landed; report at gather-backward parity.

## Proven

- `check_split_softmax_combine.py` (CPU float64): two softmax branches combined by
  `c_j=softmax(lse)` equal the concatenated softmax; forward max 4.4e-16 and all four
  gradients (se/sw/Ve/Vw) max 4.4e-16. Backward rule: feed each flash branch
  `dout_j=c_j*dout` and `dLSE_j=sum_d(dout_j.o_j)`; flash computes
  `p(dV.dout_j - dLSE_j)`, the exact within-branch score grad.
- `check_ragged_varlen_flash.py` (pod H20 card 7, flash_attn.cute): a varlen stream
  with one 1-query segment per query and ragged per-segment KV lengths matches the
  reference softmax (bf16 maxdiff 0.0078). This is how OOB selected entries are
  excluded: flash dense `mask_mod` is position-only and `gather_kv_indices` is gated to
  the MLA-absorption kernel, so neither expresses a per-query top-k validity mask; the
  ragged segment length does.
- `check_ragged_ordering.py` (CPU): argsort(stable,descending) packs each query's
  valid selected entries first so the first `n_sel` gathered slots are exactly the
  selected set.
- `check_ragged_gather_backward.py` (CPU float64): `torch.gather` followed by a boolean
  compress is differentiable end to end -- autograd scatter-adds the ragged gradient
  back to kc/vc, accumulating when an entry is selected by more than one query. So the
  custom flash Function only returns grads to the ragged inputs; no manual scatter is
  written. This was the open risk for the backward.

## Flash hooks verified

- `flash_attn.cute.interface._flash_attn_fwd(..., return_lse=True)` and
  `_flash_attn_bwd(q,k,v,out,dout,lse,...,cu_seqlens_q/k,max_seqlen_q/k,causal,
  window_size_left/right, dlse=)` (interface.py:292 / :1208). The autograd wrapper
  `FlashAttnVarlenFunc.backward(ctx, dout, dlse)` passes dlse through (interface.py:2094).
- Window branch: `_flash_attn_fwd(causal=True, window_size_left=n_win-1, right=0)`
  over the document cu (the shipped #236 PureSWA call).
- Entries branch: non-causal, non-windowed ragged varlen over gathered K/V.

## Design notes

- Hard top-k is realized by the ragged gather (only selected entries are in the
  stream), so no per-entry mask is needed inside the branch softmax. The STE gradient
  to the indexer lives on `isc`/`soft_sel` upstream and is outside this combine.
- Indexer heads (4) share one selection across each attention-head group (8 heads), so
  `n_sel` is per (batch, query) and uniform over heads: one varlen call.

## PR acceptance (fb, 2026-09-11)

Single card (CUDA_VISIBLE_DEVICES=7) on the smoke shape (d=1024, L=12, csa2 x10):
1. forward max-abs diff and backward grad diff vs the materialized path on a fixed
   batch, bf16 tolerance stated in the PR;
2. peak memory and tok/s/gpu at batch 4 and batch 8.
Measured numbers decide the PR, not the derivation.
