"""Incremental v42 forward compatible with mlx-lm generate_step.

The implementation reuses mlx-lm KVCache semantics. It preserves v42-specific
HyperConnection, SSD Engram, compressed attention and routed MoE.
"""
from __future__ import annotations

import os
import mlx.core as mx

from .model import (
    IndexerWeights,
    _deq,
    _rms_norm,
    apply_rope_real,
    compressor_forward,
    engram_forward_with_embed,
    hc_mixes,
    hc_post,
    hc_pre,
    indexer_forward,
    sparse_attention,
)
from .q8_moe import q8_moe_forward
from .sparse_moe import sparse_moe_forward


def _append(old, new):
    return new if old is None else mx.concatenate([old, new], axis=1)


def _source_for_layer(layer_id):
    if layer_id < 8:
        return 2
    if layer_id < 12:
        return 8
    return 12


def _update_compressed(model, stage, a, wl, ratio, start_pos, cfg):
    """Append complete compressed groups produced by this input chunk."""
    if ratio == 1:
        latent = compressor_forward(a, wl["compressor"], 1, cfg)
        group_start = start_pos
    else:
        pending = stage.pending_x
        merged = a if pending is None else mx.concatenate([pending, a], axis=1)
        n_complete = merged.shape[1] // ratio
        if n_complete == 0:
            stage.pending_x = merged
            return
        consumed = n_complete * ratio
        source_x = merged[:, :consumed]
        stage.pending_x = merged[:, consumed:] if consumed < merged.shape[1] else None
        latent = compressor_forward(source_x, wl["compressor"], ratio, cfg)
        group_start = (start_pos - (0 if pending is None else pending.shape[1])) // ratio

    n = latent.shape[1]
    positions = group_start + mx.arange(n)
    cos = model.comp_cos[positions]
    sin = model.comp_sin[positions]
    rot = apply_rope_real(latent, cos, sin)
    stage.compress_kv = _append(stage.compress_kv, rot)

    if "index_key" in wl:
        ikw = wl["index_key"]
        proj = _deq(ikw["wk"]["weight"], mx.bfloat16)
        ik = _rms_norm(latent @ proj.T, ikw["k_norm"]["weight"], cfg.norm_eps)
        ik = apply_rope_real(ik, cos, sin)
        stage.index_k = _append(stage.index_k, ik)


def forward_cached(model, sw, token_ids, cache):
    cfg = model.cfg
    global_cache = cache[0]
    layer_caches = cache[1:]
    start_pos = global_cache.offset
    all_tokens = global_cache.append_tokens(token_ids)
    b, s = token_ids.shape

    h = mx.take(sw.embed_weight.astype(mx.bfloat16), token_ids.astype(mx.int32), axis=0)
    h = mx.broadcast_to(h[:, :, None, :], (b, s, cfg.hc_mult, cfg.dim))
    pre_mix = mx.concatenate([
        mx.ones((b, s, 1), dtype=mx.float32),
        mx.zeros((b, s, cfg.hc_mult - 1), dtype=mx.float32),
    ], axis=-1)

    positions = start_pos + mx.arange(s)
    attn_cos = model.attn_cos[positions]
    attn_sin = model.attn_sin[positions]

    for layer_id in range(cfg.n_layers):
        wl = sw.get_layer(layer_id)

        if layer_id in cfg.engram_layer_ids:
            # Hash on the full token history, then retain only the current chunk.
            hashes = model._engram_hash(layer_id, all_tokens)[:, -s:]
            ew = sw.get_engram(layer_id)
            emb = model.ssd_lookups[layer_id].lookup(hashes)
            h = engram_forward_with_embed(h, hashes, emb, ew, cfg)

        residual = h
        attn_pre, attn_post, attn_comb = hc_mixes(
            h, wl["hc"]["attn_fn"], wl["hc"]["attn_scale"], wl["hc"]["attn_base"],
            cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
        a = _rms_norm(hc_pre(h, pre_mix), wl["attn_norm"]["weight"], cfg.norm_eps)

        q, qr = model._qproj(a, wl)
        q = apply_rope_real(q, attn_cos, attn_sin)
        kv = apply_rope_real(model._kvproj(a, wl), attn_cos, attn_sin)

        # mlx-lm KVCache stores [B, heads, time, dim]. v42 MQA has one KV head.
        lc = layer_caches[layer_id]
        full_kv, _ = lc.window.update_and_fetch(
            kv[:, None, :, :], kv[:, None, :, :])
        full_kv = full_kv[:, 0]

        ratio = cfg.compress_ratios[layer_id]
        comp_idx = None
        comp_kv = None
        if ratio > 0:
            source = _source_for_layer(layer_id)
            stage = global_cache.stages[source]
            if layer_id in cfg.kv_source_layers and "compressor" in wl:
                _update_compressed(model, stage, a, wl, ratio, start_pos, cfg)

            if layer_id in cfg.index_source_layers and stage.index_k is not None:
                idxw = IndexerWeights(
                    wq_b=wl["indexer"]["wq_b"]["weight"],
                    weights_proj=wl["indexer"]["weights_proj"]["weight"],
                )
                comp_idx = indexer_forward(
                    a, qr, stage.index_k, attn_cos, attn_sin, idxw, cfg,
                    start_pos=start_pos, key_ratio=cfg.compress_ratios[source])
                stage.topk = comp_idx
            # Match the validated full-prefix forward: compressed selections are
            # consumed at index-source layers only. Non-index layers do not reuse
            # an earlier query's top-k indices.
            comp_kv = stage.compress_kv

        total_window = full_kv.shape[1]
        lo = mx.maximum(positions - cfg.window_size + 1, 0)
        win_idx = lo[:, None] + mx.arange(cfg.window_size)[None, :]
        win_idx = mx.where(win_idx > positions[:, None], -1, win_idx)[None].astype(mx.int32)

        if comp_kv is not None and comp_idx is not None:
            kv_cat = mx.concatenate([full_kv, comp_kv], axis=1)
            comp_gather = mx.where(comp_idx >= 0, comp_idx + total_window, -1)
        else:
            kv_cat = full_kv
            comp_gather = None

        o = sparse_attention(
            q, kv_cat, wl["attn_sink"], win_idx, comp_gather,
            cfg.head_dim ** -0.5, cfg.attn_logit_softcap)
        o = apply_rope_real(o, attn_cos, attn_sin, inverse=True)
        h = hc_post(model._oproj(o, wl), residual, attn_post, attn_comb)

        residual = h
        ffn_pre, ffn_post, ffn_comb = hc_mixes(
            h, wl["hc"]["ffn_fn"], wl["hc"]["ffn_scale"], wl["hc"]["ffn_base"],
            cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
        f = _rms_norm(hc_pre(h, attn_pre), wl["ffn_norm"]["weight"], cfg.norm_eps)
        args = (
            wl["ffn"]["gate"]["weight"], wl["ffn"]["gate"]["bias"],
            wl["ffn"]["shared"]["w1"]["weight"],
            wl["ffn"]["shared"]["w3"]["weight"],
            wl["ffn"]["shared"]["w2"]["weight"],
        )
        if os.environ.get("V42_MOE_BACKEND", "q8") == "bf16":
            f = sparse_moe_forward(
                f, *args, "ckpt_local/sft_mlx_bf16/expert", layer_id, cfg)
        else:
            f = q8_moe_forward(f, *args, layer_id, cfg)
        h = hc_post(f, residual, ffn_post, ffn_comb)
        pre_mix = ffn_pre

    h = _rms_norm(hc_pre(h, pre_mix), sw.norm_weight, cfg.norm_eps)
    return h.astype(mx.bfloat16) @ sw.head_weight.astype(mx.bfloat16).T
