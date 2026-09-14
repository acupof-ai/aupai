"""de-109 step-cost breakdown: why 3.2x on the attention module is ~1.1x on the full step.

Lane-free analytic accounting (no GPU): per-token FLOPs of each full-step component from
the gate architecture dims, plus the Amdahl inversion of the MEASURED step speedup. The
point is the gap between the window branch's FLOP share and its pre-flag TIME share:
flashing it removes a memory-bound T*T score materialization, not FLOPs.

Numbers below are active (per-token-routed) FLOPs, forward only; backward roughly doubles
the attention-score term and does not change the conclusion. A real CUDA timeline is the
post-gate follow-up; this is the architecture-derived bound.
"""

d = 1024
L = 12
n_csa2 = 10
n_win = 128
NB = 512
moe_experts_active = 4
expert_ffn = 1728


def fmt(x):
    return f"{x/1e6:6.2f}M"


rows = []

# MoE: 3 SwiGLU matmuls (gate,up,down) each 2*d*expert_ffn, times active experts, all layers
moe = 3 * 2 * d * expert_ffn * moe_experts_active * L
rows.append(("MoE FFN (4 active experts/layer, SwiGLU gate+up+down)", moe))

# MLA Q/K/V/gate/O projections per GatedMLA (model.py GatedMLA.__init__): latent KV
# compression, not four d->d matmuls. kv_down d->d/4; kv_up d/4->2d (fused k|v);
# qg d->2d (fused q|gate); o d->d. Sum weights = 0.25 + 0.5 + 2 + 1 = 3.75 d^2/layer.
latent = d // 4
proj = 2 * (d * latent + latent * 2 * d + d * 2 * d + d * d) * L
rows.append(("MLA projections kv_down+kv_up+qg+o (3.75 d^2/layer, latent KV)", proj))

# CSA2 dense selected-entries scores: 2*NB*d per CSA2 layer (QK + AV), unchanged dense
entries = 2 * NB * d * n_csa2
rows.append(("CSA2 selected-entry scores (2*NB*d, dense)", entries))

# window scores: 2*n_win*d per layer -- the ONLY part csa2_win_flash moves to flash
window = 2 * n_win * d * L
rows.append(("SWA window scores (2*n_win*d) -- flashed", window))

# indexer + norms + router + embedding (small; indexer ~2*d*256 bound from size fact)
other = 40 * d * d
rows.append(("indexer + RMSNorm + router + embedding (approx)", other))

total = sum(v for _, v in rows)
print(f"per-token forward FLOPs (L={L}, d={d}, NB={NB}, n_win={n_win})")
for name, v in rows:
    print(f"  {fmt(v)}  {100*v/total:5.2f}%  {name}")
print(f"  {fmt(total)}  total")
print()
print(f"window FLOP share: {100*window/total:.3f}%  (the flag speeds only this compute)")
print()

# Amdahl inversion of the measured step speedup S: S = 1 / (1 - f + f/s_attn)
# window module was s_attn=3.2x faster in the attention-only bench; measured full-step S.
for s_attn, S in ((3.2, 1.11), (3.2, 1.40)):
    f = (1 - 1 / S) / (1 - 1 / s_attn)
    print(f"if full-step speedup S={S}x with attention-module gain {s_attn}x: "
          f"window pre-flag TIME share f={f:.3f}")
print()
print("Interpretation: window is <1% of FLOPs but ~14% of pre-flag step time. The gap is")
print("the materialized B,H,T,T fp32 score tensor + causal/document mask build (write- and")
print("launch-bound, not compute). Flash removes that memory work; it cannot cut the 86%")
print("(MoE grouped-mm/dispatch, projections, dense entries matmul, fp8 casts, optim, IO).")
