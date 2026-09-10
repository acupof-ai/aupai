#!/usr/bin/env python3
"""V4.1 Step 0 (task ae-2): derive the small-scale CSA2 config for the p1 model.

The paper's numbers (32 indexer heads x 128, 64 core heads x 512, latent 1280,
top-k 512, m=2/1, 40 layers, n_win 128) are all 552B / seq 64K-1M and do not
transfer. This script re-derives every value for d=1024, L=12, seq 4096 and
prints the arithmetic. The proposed Cfg defaults at the bottom are what de-103
implements; nothing here trains anything.

    python3 scripts/v41_size.py

Sources: DeepSeek_V41_Tech_Report.pdf section 2 (pp. 7-15); PR #210
docs/standards/v41_pivot.md. The pivot doc's v41_arch_spec.md and the
2026-09-10 gap map exist in no tree any session can read, so L=12 is taken as
the gap map's reported suggestion and every other value is derived here.
"""

# --- proposed Cfg defaults (d=1024 width is fixed: the ladder, the OOM facts and
# the indexer geometry below all assume it) -----------------------------------
D = 1024  # residual width
HEADS = 8  # core attention heads, hd=128 (KDA is gone, so hd=128 is no
# longer kernel-pinned; 8x128 still divides d and keeps every width fact valid)
LAYERS = 12  # gap map suggestion (fb, 2026-09-10)
VOCAB = 32784  # frozen tokenizer, tied lm_head (model.py:1988)
SEQ = 4096  # gate-run sequence length

MOE_EXPERTS = 48  # pivot: reuse MoEFFN as-is, 48/top-3/1-shared
MOE_TOP_K = 3
MOE_SHARED = 1
MOE_EXPERT_FFN = 1728  # the knob that hits ~350M active at L=12, d=1024.
# fb ruling 2026-09-10: do NOT relax MoEFFN's equal-active parity check -- set
# ffn_hidden = (top_k+shared)*expert_ffn = 6912 so the config stays within it.
# ffn_hidden is the parity reference only (all 12 layers are MoE), so it adds
# no active params. 3b's PR #213 review corrected the per-layer attention
# counts (Reuse reuses Main KV, so it saves 2d^2): at 1536 the count was 324.0M,
# 26M under target; 1728 restores 352.3M. The 340-360M assertion band catches
# that class of overcount (the old 300-400M band passed it).

M = 8  # tokens per main-KV entry (fb's start range 4..8; see below)
TOP_K = 64  # entries selected per query (fb's range 64..128; see below)
N_WIN = 128  # SWA width in tokens (paper value; Step 3 A/Bs 128 vs 256)
IDX_HEADS = 4  # indexer heads
IDX_DIM = 64  # indexer per-head dim; 4x64 = 256 = D//4 latent
BYTES = 2  # bf16; FP4 KV is deferred past the gate (pivot)

# flat-stack mode map, one entry per layer:
#   S = pure SWA (paper: first two layers are SWA-only)
#   F = Full     (own main KV + indexer K, fresh top-K; stores global KV)
#   X = Reindex, DEFERRED: Step 2 ships Full/Reuse only (pivot build order), so
#       until Reindex lands the module runs X as Reuse. The slot is kept in the
#       map so the mid-stack selection refresh has a named place, not a live mode
#   R = Reuse    (reuses the nearest preceding F/X's main KV AND top-K)
MODE_MAP = "S,S,F,R,R,R,X,R,R,R,R,R"


def active_params():
    """Active params per token. Embedding counted once (tied head). Per the
    paper's Fig. 4 (3b's PR #213 review): only Full computes Main K/V -- Reuse
    reuses the preceding Full's Main KV and Top-K, Reindex reuses Main KV and
    runs its own indexer. So:
      S = Q + SWA(K,V,O)        = 4d^2   (no global branch)
      F = Q + Main(K,V,O) + SWA(K,V,O) + idx = 7d^2 + idx
      X = Q + Main O + SWA(K,V,O) + idx   = 5d^2 + idx
      R = Q + Main O + SWA(K,V,O)         = 5d^2
    (paper text 2.3.1 also reuses indexer K in Reindex, worth 0.26M; 3b's
    prescription counts the full indexer, kept as the conservative reading.)
    Active FFN = (top_k + shared) experts, each 3*d*w."""
    d2 = D * D
    ffn = (MOE_TOP_K + MOE_SHARED) * 3 * D * MOE_EXPERT_FFN
    idx = 2 * D * (IDX_HEADS * IDX_DIM)  # indexer Q proj + K proj, F/X only
    emb = D * VOCAB

    per = {
        "S": 4 * d2 + ffn,
        "F": 7 * d2 + idx + ffn,
        "X": 5 * d2 + idx + ffn,
        "R": 5 * d2 + ffn,
    }
    modes = MODE_MAP.split(",")
    layers = sum(per[m] for m in modes)
    active = layers + emb
    # total = non-FFN params + ALL experts (active already counts the 4/layer
    # routed+shared, so subtract the per-layer active FFN before adding the set)
    all_experts = LAYERS * (MOE_EXPERTS + MOE_SHARED) * 3 * D * MOE_EXPERT_FFN
    total = active - LAYERS * ffn + all_experts
    return active, total, emb, layers, per, modes


def kv_bytes_per_token():
    """Global KV per token, the number the architecture exists for.

    Dense at the same width: every token stores K+V in every layer.
    CSA2: only F layers store main KV, T/M entries each holding K+V, amortized
    over the M tokens an entry covers. X/R store nothing (they reuse F's KV)."""
    dense_per_layer = 2 * D * BYTES
    dense = LAYERS * dense_per_layer
    n_full = MODE_MAP.split(",").count("F")
    csa2 = n_full * dense_per_layer / M
    return dense, csa2, n_full


def density():
    """top-k is not effectively dense iff the main branch attends a small
    fraction of what dense would. Both sides of the proof:
      keys/query: TOP_K entries x M tokens + the N_WIN window keys, vs T
      entry selectivity: TOP_K of the T/M entries scored survive the indexer"""
    entries = SEQ // M
    keys = TOP_K * M + N_WIN
    return {
        "entries_per_layer": entries,
        "attended_keys_per_query": keys,
        "dense_keys_per_query": SEQ,
        "attended_fraction": keys / SEQ,
        "entry_selectivity": TOP_K / entries,
        "indexer_macs_per_query": IDX_HEADS * entries * IDX_DIM,
        "dense_qk_macs_per_query": HEADS * SEQ * (D // HEADS),
    }


def check_mode_map(modes):
    assert len(modes) == LAYERS, f"mode map has {len(modes)} entries, need {LAYERS}"
    assert modes[:2] == ["S", "S"], "paper: first two layers are SWA-only"
    first_csa2 = next(i for i, m in enumerate(modes) if m != "S")
    assert modes[first_csa2] == "F", "the first CSA2 layer must be Full (nothing to reuse)"
    for i, m in enumerate(modes):
        if m in "XR":
            prior = [j for j in range(i) if modes[j] in "FX"]
            assert prior, f"layer {i} ({m}) has no preceding Full/Reindex to reuse"
    # spacing: how many CSA2 layers one Full's KV serves (paper encoder bulk:
    # one Full at the top serves ~17 Reuse layers)
    fulls = [i for i, m in enumerate(modes) if m == "F"]
    bounds = fulls[1:] + [LAYERS]
    served = max(b - f for f, b in zip(fulls, bounds, strict=True)) - 1
    return first_csa2, served


def main():
    active, total, emb, layers, per, modes = active_params()
    dense_kv, csa2_kv, n_full = kv_bytes_per_token()
    dens = density()
    first_csa2, max_gap = check_mode_map(modes)

    # the checks: fail loud if a proposed value stops satisfying its constraint
    assert 340e6 < active < 360e6, f"active {active / 1e6:.1f}M outside ~350M (band 340-360)"
    assert M in (4, 8) or 4 <= M <= 8, "m must start in 4..8 (fb)"
    assert 64 <= TOP_K <= 128, "top-k must start in 64..128 (fb)"
    assert IDX_HEADS * IDX_DIM == D // 4, "indexer latent must be d//4"
    assert D % (IDX_HEADS * IDX_DIM) == 0 and (IDX_HEADS * IDX_DIM) % IDX_HEADS == 0
    assert dens["attended_fraction"] < 0.5, "top-k is effectively dense"
    assert (MOE_TOP_K + MOE_SHARED) * MOE_EXPERT_FFN == 6912, "ffn_hidden must be 6912"

    print(f"active params/token: {active / 1e6:.1f}M (emb {emb / 1e6:.1f}M tied, layers {layers / 1e6:.1f}M)")
    print(
        f"total params:        {total / 1e9:.2f}B (name: {total / 1e9:.1f}b-a{active / 1e6:.0f}m-e{MOE_EXPERTS})"
    )
    print(f"per-layer active:    S {per['S'] / 1e6:.2f}M  F/X {per['F'] / 1e6:.2f}M  R {per['R'] / 1e6:.2f}M")
    print()
    print(f"global KV/token:     {csa2_kv:.0f} B (CSA2, {n_full} Full layer(s), m={M}, bf16)")
    print(f"dense same width:    {dense_kv:.0f} B ({LAYERS} layers)  ->  {dense_kv / csa2_kv:.0f}x smaller")
    print(f"  (SWA KV at the gate: {dense_kv:.0f} B/token, stored not yet replayed;")
    print("   bounded replay is Step 7, so the persistent total is not this good yet)")
    print()
    print(f"entries/layer:       {dens['entries_per_layer']} (T/m = {SEQ}/{M})")
    print(
        f"attended keys/query: {dens['attended_keys_per_query']} "
        f"(top_k*m + n_win = {TOP_K}*{M} + {N_WIN}) = {dens['attended_fraction']:.1%} of dense {SEQ}"
    )
    print(
        f"entry selectivity:   {dens['entry_selectivity']:.1%} of scored entries survive "
        f"({1 - dens['entry_selectivity']:.1%} discarded before attention)"
    )
    print(
        f"indexer cost:        {dens['indexer_macs_per_query'] / 1e3:.0f}K MACs/query "
        f"= {dens['indexer_macs_per_query'] / dens['dense_qk_macs_per_query']:.1%} of dense QK"
    )
    print()
    print(
        f"mode map:            {MODE_MAP}  (first CSA2 = Full at layer {first_csa2}, "
        f"one Full serves {max_gap} reuse layers; paper encoder bulk runs ~17)"
    )
    print()
    print("proposed Cfg defaults for de-103 (train.py Cfg + Step 1/2 modules):")
    print("  layers = 12  d = 1024  heads = 8  (unchanged)")
    print(
        f"  moe_experts = {MOE_EXPERTS}  moe_top_k = {MOE_TOP_K}  moe_shared = {MOE_SHARED}"
        f"  moe_expert_ffn = {MOE_EXPERT_FFN}  ffn_hidden = {(MOE_TOP_K + MOE_SHARED) * MOE_EXPERT_FFN}"
    )
    print(f"  csa2_m = {M}  csa2_top_k = {TOP_K}  csa2_n_win = {N_WIN}")
    print(f'  csa2_indexer_heads = {IDX_HEADS}  csa2_indexer_dim = {IDX_DIM}  csa2_modes = "{MODE_MAP}"')
    print("  csa2_modes: X is Reindex-DEFERRED (fb ruling 2026-09-10) -- the module runs it as")
    print("  Reuse until Step 2 ships Reindex; the slot names the mid-stack refresh point")
    print("  (csa_compress/csa_topk/csa_window belong to the old CSA class Step 1 rewrites")
    print("   and should be removed with it, not reused -- same prefix, different semantics)")


if __name__ == "__main__":
    main()
