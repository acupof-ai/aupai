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
  `n_sel` is per (batch, query) and uniform over heads: one varlen call per group.
- Flash token layout is `(B*T,H,D)`; feed the untransposed `k,v` (`B,T,H,D`) reshaped
  directly. The transposed `kh,vh` (`B,H,T,D`) reshaped to `(B*T,H,D)` scrambles K/V.

## PR acceptance (fb, 2026-09-11)

## CRITICAL: the indexer STE gradient must be rebuilt (check_ste_indexer_*.py)

A naive flash port that selects entries via detached top-k SILENTLY REMOVES the
indexer's gradient. In the materialized path the entry weight is `w_entry * ste` where
`ste = hard + soft_sel - soft_sel.detach()`; the `soft_sel - soft_sel.detach()` term is
the ONLY route by which the indexer scores `isc` learn (proved: hard gather alone gives
`indexer isc grad nonzero: False`, materialized gives True).

The attention flash calls cannot produce it (their K/V are the detached-selection
gather). It is computed separately from the SMALL indexer softmax (B,ih,T,NB, no T*T):

    g_e[b,h,t,n] = dL/d(entry_weight_n)           # per selected entry, from flash bwd
    dL/disc     = soft_sel * (g_e - sum_n soft_sel*g_e)   # softmax Jacobian, exact
                                                   # (maxdiff 0.0 vs autograd, analytic)

g_e is recovered at B,H,T,topk (0.0625 GiB/layer) from the entry branch attention
probabilities P_e = exp(qk_e - lse_e) and the upstream dO: g_e = P_e * (c_e*dO @ ve),
with the branch mixing c_e from the LSE combine. Masked/dead entries contribute 0.
The attention-value gradient (to the compressors/kc/vc) still flows normally through the
ragged gather backward; ONLY the indexer-score gradient is the separate small term.

model.py contains a draft `_CSA2JointFlash`/`csa2_joint_flash` (compiles, not yet wired
into `_forward_csa2`, not yet parity-tested); it does the two flash calls and LSE
combine but still needs: (1) the indexer STE gradient term above added to the
indexer_q/ik_weight grads; (2) a pure-PyTorch combine reference for CPU CI; (3) wire
into `_forward_csa2` under HAS_FA with the materialized path as fallback; (4) CPU parity
+ card-7 B4/B8 numbers.
