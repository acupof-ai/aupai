"""V42RuntimeModel: the frozen mlx-lm runtime forward.

Conforms to mlx-lm's ``model(inputs, cache)`` contract so it can be driven by
``generate_step`` / ``BatchGenerator``.  Reuses the immutable model semantics
from ``v41f.mlx.model`` (RMSNorm, RoPE, HyperConnection, sparse attention,
compressor, indexer, Engram math, sqrtsoftplus routing).

Cache layout (frozen): cache[0] = GlobalState (tokens + 3 compressed stages +
HyperConnection pre_mix), cache[1..24] = per-layer MQA window KV caches.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from ..config import MLXV42Config
from ..model import (
    IndexerWeights,
    _deq,
    _rms_norm,
    apply_rope_real,
    compressor_forward,
    engram_forward_with_embed,
    hc_post,
    hc_pre,
    indexer_forward,
    sparse_attention,
)
from .cache import GlobalState, LayerKVCache, make_cache
from .hc_kernel import hc_mixes_fast
from .hc_kernel import warmup as warmup_sinkhorn
from .moe import SwitchGLU

_NO_COMPRESS = bool(os.environ.get("V42_NO_COMPRESS"))


def _source_for_layer(layer_id: int) -> int:
    if layer_id < 8:
        return 2
    if layer_id < 12:
        return 8
    return 12


class V42RuntimeModel(nn.Module):
    """mlx-lm compatible runtime model."""

    def __init__(self, cfg: MLXV42Config, wstore, engram_bank):
        super().__init__()
        self.cfg = cfg
        self.ws = wstore
        self.engram = engram_bank
        self.moe = SwitchGLU(wstore.qstore, wstore.group_size, wstore.bits)
        # RoPE tables (attn rope_theta vs compress_rope_theta MUST differ).
        t = mx.arange(8192, dtype=mx.float32)
        hd = cfg.rope_head_dim
        attn_inv = 1.0 / (cfg.rope_theta ** (mx.arange(0, hd, 2, dtype=mx.float32) / hd))
        self.attn_cos = mx.cos(mx.outer(t, attn_inv))
        self.attn_sin = mx.sin(mx.outer(t, attn_inv))
        comp_inv = 1.0 / (cfg.compress_rope_theta ** (mx.arange(0, hd, 2, dtype=mx.float32) / hd))
        self.comp_cos = mx.cos(mx.outer(t, comp_inv))
        self.comp_sin = mx.sin(mx.outer(t, comp_inv))
        # mlx-lm protocol attributes.
        self.args = type("V42Args", (), {"max_position_embeddings": 4096,
                                         "vocab_size": cfg.vocab_size})()
        self.layers = [None] * cfg.n_layers
        self._layer_w = None
        self._engram_w = None

    def prepare(self):
        """Cache layer dicts and compile the Sinkhorn kernel before the first request."""
        self._layer_w = [self.ws.get_layer(i) for i in range(self.cfg.n_layers)]
        self._engram_w = {i: self.ws.get_engram(i) for i in self.cfg.engram_layer_ids}
        warmup_sinkhorn()

    def _layer(self, layer_id: int) -> dict:
        if self._layer_w is None:
            return self.ws.get_layer(layer_id)
        return self._layer_w[layer_id]

    def make_cache(self):
        return make_cache(self.cfg.n_layers)

    # -- sub-projections ----------------------------------------------------
    def _qproj(self, a, WL):
        wq_a = _deq(WL["qproj"]["wq_a"]["weight"], mx.bfloat16)
        qr = _rms_norm(a @ wq_a.T, WL["qproj"]["q_norm"]["weight"], self.cfg.norm_eps)
        wq_b = _deq(WL["qproj"]["wq_b"]["weight"], mx.bfloat16)
        q = (qr @ wq_b.T).reshape(*a.shape[:2], self.cfg.n_heads, self.cfg.head_dim)
        return q, qr

    def _kvproj(self, a, WL):
        wkv = _deq(WL["kvproj"]["wkv"]["weight"], mx.bfloat16)
        return _rms_norm(a @ wkv.T, WL["kvproj"]["kv_norm"]["weight"], self.cfg.norm_eps)

    def _oproj(self, o, WL):
        b, s, h, d = o.shape
        g = self.cfg.o_groups
        pg = h // g
        wo_a = _deq(WL["oproj"]["wo_a"], mx.bfloat16)
        og = o.reshape(b, s, g, pg * d)
        lat = mx.einsum("bsgd,grd->bsgr", og, wo_a)
        wo_b = _deq(WL["oproj"]["wo_b"]["weight"], mx.bfloat16)
        return lat.reshape(b, s, g * self.cfg.o_lora_rank) @ wo_b.T

    def _update_compressed(self, stage, a, wl, ratio, start_pos):
        """Append completed ratio-groups; half groups are kept in pending_x."""
        cfg = self.cfg
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
        # RoPE position of compressed group j = its group's first absolute token
        # position j*ratio (PyTorch gold: freqs_cis[:kept:ratio]; MLX gold:
        # comp_cos[::ratio]). Buggy group-index indexing caused long-prompt drift.
        positions = (group_start + mx.arange(n)) * ratio
        cos = self.comp_cos[positions]
        sin = self.comp_sin[positions]
        rot = apply_rope_real(latent, cos, sin)
        stage.compress_kv = rot if stage.compress_kv is None else \
            mx.concatenate([stage.compress_kv, rot], axis=1)

        if "index_key" in wl:
            ikw = wl["index_key"]
            proj = _deq(ikw["wk"]["weight"], mx.bfloat16)
            ik = _rms_norm(latent @ proj.T, ikw["k_norm"]["weight"], cfg.norm_eps)
            ik = apply_rope_real(ik, cos, sin)
            stage.index_k = ik if stage.index_k is None else \
                mx.concatenate([stage.index_k, ik], axis=1)

    def __call__(self, inputs: mx.array, cache=None):
        """inputs: [T] or [1,T] int32. cache: [GlobalState] + 24 LayerKVCache."""
        cfg = self.cfg
        if cache is None:
            cache = self.make_cache()
        gstate: GlobalState = cache[0]
        layer_caches = cache[1:]
        start_pos = gstate.offset

        if inputs.ndim == 1:
            inputs = inputs[None]
        b, s = inputs.shape
        gstate.append_tokens(inputs)
        # Engram rows are ready before the layer loop, so the GPU graph stays one eval.
        en_emb, en_hash = None, None
        if cfg.engram_layer_ids:
            en_emb, en_hash = self.engram.embed_chunk(gstate.cpu_hist, start_pos, s)

        h = mx.take(self.ws.embed_weight.astype(mx.bfloat16),
                    inputs.astype(mx.int32), axis=0)
        h = mx.broadcast_to(h[:, :, None, :], (b, s, cfg.hc_mult, cfg.dim))
        # Layer 0 always starts from one-hot identity HyperConnection mixing.
        pre_mix = mx.concatenate([mx.ones((b, s, 1), mx.float32),
                                  mx.zeros((b, s, cfg.hc_mult - 1), mx.float32)], axis=-1)

        positions = start_pos + mx.arange(s)
        attn_cos = self.attn_cos[positions]
        attn_sin = self.attn_sin[positions]

        for layer_id in range(cfg.n_layers):
            wl = self._layer(layer_id)

            if layer_id in cfg.engram_layer_ids:
                ew = (
                    self.ws.get_engram(layer_id)
                    if self._engram_w is None
                    else self._engram_w[layer_id]
                )
                h = engram_forward_with_embed(
                    h, en_hash[layer_id], en_emb[layer_id], ew, cfg)

            residual = h
            attn_pre, attn_post, attn_comb = hc_mixes_fast(
                h, wl["hc"]["attn_fn"], wl["hc"]["attn_scale"], wl["hc"]["attn_base"],
                cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
            a = _rms_norm(hc_pre(h, pre_mix), wl["attn_norm"]["weight"], cfg.norm_eps)

            q, qr = self._qproj(a, wl)
            q = apply_rope_real(q, attn_cos, attn_sin)
            kv = apply_rope_real(self._kvproj(a, wl), attn_cos, attn_sin)

            lc: LayerKVCache = layer_caches[layer_id]
            full_kv, _ = lc.window.update_and_fetch(kv[:, None, :, :], kv[:, None, :, :])
            # Slice off the KVCache's step-aligned padding so total_window equals
            # the real token count (matches the validated full-prefix forward).
            full_kv = full_kv[:, 0, :gstate.offset, :]

            ratio = cfg.compress_ratios[layer_id]
            comp_idx = None
            comp_kv = None
            if ratio > 0 and not _NO_COMPRESS:
                source = _source_for_layer(layer_id)
                stage = gstate.stages[source]
                if layer_id in cfg.kv_source_layers and "compressor" in wl:
                    self._update_compressed(stage, a, wl, ratio, start_pos)
                # compressed selection consumed only at index-source layers.
                if layer_id in cfg.index_source_layers and stage.index_k is not None:
                    idxw = IndexerWeights(
                        wq_b=wl["indexer"]["wq_b"]["weight"],
                        weights_proj=wl["indexer"]["weights_proj"]["weight"])
                    comp_idx = indexer_forward(
                        a, qr, stage.index_k, attn_cos, attn_sin, idxw, cfg,
                        start_pos=start_pos, key_ratio=cfg.compress_ratios[source])
                    stage.topk = comp_idx
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

            o = sparse_attention(q, kv_cat, wl["attn_sink"], win_idx, comp_gather,
                                 cfg.head_dim ** -0.5, cfg.attn_logit_softcap)
            o = apply_rope_real(o, attn_cos, attn_sin, inverse=True)
            h = hc_post(self._oproj(o, wl), residual, attn_post, attn_comb)

            residual = h
            ffn_pre, ffn_post, ffn_comb = hc_mixes_fast(
                h, wl["hc"]["ffn_fn"], wl["hc"]["ffn_scale"], wl["hc"]["ffn_base"],
                cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
            f = _rms_norm(hc_pre(h, attn_pre), wl["ffn_norm"]["weight"], cfg.norm_eps)
            f = self.moe(f, wl["ffn"]["gate"]["weight"], wl["ffn"]["gate"]["bias"],
                         wl["ffn"]["shared"]["w1"]["weight"],
                         wl["ffn"]["shared"]["w3"]["weight"],
                         wl["ffn"]["shared"]["w2"]["weight"], layer_id, cfg)
            h = hc_post(f, residual, ffn_post, ffn_comb)
            pre_mix = ffn_pre

        h = _rms_norm(hc_pre(h, pre_mix), self.ws.norm_weight, cfg.norm_eps)
        return h.astype(mx.bfloat16) @ self.ws.head_weight.astype(mx.bfloat16).T
