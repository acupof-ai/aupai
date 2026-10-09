"""Streaming forward pass for MLX v42 model.
Loads layer weights on demand, releases after use. Keeps RSS < 6GB.
"""
import os
import mlx.core as mx
from .model import (apply_rope_real, _rms_norm, hc_mixes, hc_pre, hc_post,
                    sparse_attention, engram_forward_with_embed,
                    compressor_forward, indexer_forward, IndexerWeights, _deq)
from .sparse_moe import sparse_moe_forward


def forward_streaming(model, sw, token_ids):
    """Streaming forward pass. sw is StreamingWeights.
    token_ids: [b,s] int32. Returns logits [b,s,vocab].
    """
    cfg = model.cfg
    b, s = token_ids.shape

    # Embed
    embed_w = sw.embed_weight
    h = mx.take(embed_w.astype(mx.bfloat16), token_ids.astype(mx.int32), axis=0)
    h = mx.broadcast_to(h[:, :, None, :], (b, s, cfg.hc_mult, cfg.dim))

    pre_mix = mx.concatenate([mx.ones((b, s, 1), dtype=mx.float32),
                               mx.zeros((b, s, cfg.hc_mult - 1), dtype=mx.float32)], axis=-1)

    attn_cos = model.attn_cos[:s][None]
    attn_sin = model.attn_sin[:s][None]

    # Layer state for compressor/indexer
    compress_kv = None
    index_k = None

    for L in range(cfg.n_layers):
        WL = sw.get_layer(L)

        # Engram injection
        if L in cfg.engram_layer_ids:
            hash_ids = model._engram_hash(L, token_ids)
            eW = sw.get_engram(L)
            if L in model.ssd_lookups:
                ssd_emb = model.ssd_lookups[L].lookup(hash_ids)
                h = engram_forward_with_embed(h, hash_ids, ssd_emb, eW, cfg)

        # Attention sublayer
        residual = h
        attn_pre, attn_post, attn_comb = hc_mixes(
            h, WL["hc"]["attn_fn"], WL["hc"]["attn_scale"], WL["hc"]["attn_base"],
            cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
        a = hc_pre(h, pre_mix)
        a = _rms_norm(a, WL["attn_norm"]["weight"], cfg.norm_eps)

        q, qr = model._qproj(a, WL)
        q = apply_rope_real(q, attn_cos[0], attn_sin[0])
        kv = model._kvproj(a, WL)
        kv = apply_rope_real(kv, attn_cos[0], attn_sin[0])

        ratio = cfg.compress_ratios[L]
        comp_kv_out = None
        comp_idx_out = None

        if ratio > 0:
            if L in cfg.kv_source_layers and "compressor" in WL:
                latent = compressor_forward(a, WL["compressor"], ratio, cfg)
                g_cos = model.comp_cos[::ratio][:latent.shape[1]][None]
                g_sin = model.comp_sin[::ratio][:latent.shape[1]][None]
                rot_lat = apply_rope_real(latent, g_cos[0], g_sin[0])
                compress_kv = rot_lat
                if L in cfg.index_source_layers:
                    ik_w = WL["index_key"]
                    ik_wk = _deq(ik_w["wk"]["weight"], mx.bfloat16)
                    ik = _rms_norm(latent @ ik_wk.T, ik_w["k_norm"]["weight"], cfg.norm_eps)
                    ik = apply_rope_real(ik, g_cos[0], g_sin[0])
                    index_k = ik

            if L in cfg.index_source_layers and index_k is not None:
                idx_w = IndexerWeights(
                    wq_b=WL["indexer"]["wq_b"]["weight"],
                    weights_proj=WL["indexer"]["weights_proj"]["weight"],
                )
                comp_idx_out = indexer_forward(
                    a, qr, index_k, attn_cos[0], attn_sin[0], idx_w, cfg)

            if compress_kv is not None:
                comp_kv_out = compress_kv

        win_idx = model._window_idx(s)

        if comp_kv_out is not None:
            full_kv = mx.concatenate([kv, comp_kv_out], axis=1)
            if comp_idx_out is not None:
                comp_gather = mx.where(comp_idx_out >= 0, comp_idx_out + s, -1)
            else:
                comp_gather = None
        else:
            full_kv = kv
            comp_gather = None

        o = sparse_attention(q, full_kv, WL["attn_sink"], win_idx, comp_gather,
                             cfg.head_dim ** -0.5, cfg.attn_logit_softcap)
        o = apply_rope_real(o, attn_cos[0], attn_sin[0], inverse=True)
        a_out = model._oproj(o, WL)

        h = hc_post(a_out, residual, attn_post, attn_comb)

        # FFN sublayer
        residual = h
        ffn_pre, ffn_post, ffn_comb = hc_mixes(
            h, WL["hc"]["ffn_fn"], WL["hc"]["ffn_scale"], WL["hc"]["ffn_base"],
            cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
        f = hc_pre(h, attn_pre)
        f = _rms_norm(f, WL["ffn_norm"]["weight"], cfg.norm_eps)
        # Routed MoE: use fused Q8 gather kernels for interactive inference.
        # Set V42_MOE_BACKEND=bf16 to run the exact but slower reference path.
        moe_backend = os.environ.get("V42_MOE_BACKEND", "q8")
        if moe_backend == "bf16":
            f = sparse_moe_forward(f,
                                   WL["ffn"]["gate"]["weight"],
                                   WL["ffn"]["gate"]["bias"],
                                   WL["ffn"]["shared"]["w1"]["weight"],
                                   WL["ffn"]["shared"]["w3"]["weight"],
                                   WL["ffn"]["shared"]["w2"]["weight"],
                                   'ckpt_local/sft_mlx_bf16/expert',
                                   L, cfg)
        else:
            from .q8_moe import q8_moe_forward
            f = q8_moe_forward(f,
                               WL["ffn"]["gate"]["weight"],
                               WL["ffn"]["gate"]["bias"],
                               WL["ffn"]["shared"]["w1"]["weight"],
                               WL["ffn"]["shared"]["w3"]["weight"],
                               WL["ffn"]["shared"]["w2"]["weight"],
                               L, cfg)
        h = hc_post(f, residual, ffn_post, ffn_comb)

        pre_mix = ffn_pre

        # Release expert weights for this layer
        sw.release_layer(L)

    # Final norm + head
    h = hc_pre(h, pre_mix)
    h = _rms_norm(h, sw.norm_weight, cfg.norm_eps)
    logits = h.astype(mx.bfloat16) @ sw.head_weight.astype(mx.bfloat16).T
    return logits
