"""Correct KV-cached streaming decode for the MLX v42 bf16-active-sparse path.

This mirrors `forward_streaming.forward_streaming` line-for-line (same weights,
same math, same order) but:
  * prefill runs the prompt once and caches every per-layer RoPE'd raw KV slot,
  * every compressor/indexer source layer also caches its compressed-group state
    incrementally (softmax-pooled groups + RoPE),
  * decode is a single-token forward that only touches the new token plus the
    cached window / compressed slots, instead of recomputing the whole prefix.

Correctness invariants (must match forward_streaming / PyTorch gold):
  * window indices use ABSOLUTE positions.
  * compressor pools consecutive `ratio` tokens; the in-progress partial group
    keeps per-token kv/score so it can be finalized exactly when it fills.
  * indexer causal visibility uses absolute position: visible groups
    col < (P+1)//key_ratio, key_ratio = (P+1)//n_groups.
  * Engram n-gram hashes are computed over the FULL token history (last slot).
  * HyperConn pre/post/comb mixes are recomputed per token (position-free); the
    previous layer's ffn_pre is threaded as pre_mix, exactly as the accurate
    path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx

from .model import (
    apply_rope_real, _rms_norm, hc_mixes, hc_pre, hc_post,
    sparse_attention, indexer_forward, IndexerWeights, _deq,
    engram_forward_with_embed,
)
from .sparse_moe import sparse_moe_forward_batched


@dataclass
class KVBench:
    prefill_ms: float = 0.0
    decode_ms_sum: float = 0.0
    decode_steps: int = 0

    @property
    def decode_ms_per_tok(self) -> float:
        return self.decode_ms_sum / max(1, self.decode_steps)

    @property
    def decode_tps(self) -> float:
        s = self.decode_ms_sum / 1000.0
        return self.decode_steps / s if s > 0 else 0.0


class _SourceState:
    """Incremental compressed-KV state for one kv_source layer."""
    __slots__ = ("ratio", "partial_kv", "partial_score", "comp_groups", "index_groups",
                 "has_index")

    def __init__(self, ratio: int, has_index: bool):
        self.ratio = ratio
        self.has_index = has_index
        self.partial_kv = []      # list of [head_dim] fp32 (unpooled)
        self.partial_score = []   # list of [head_dim] fp32
        self.comp_groups = []     # list of [head_dim] bf16 RoPE'd
        self.index_groups = []    # list of [index_head_dim] bf16 RoPE'd


class KVStreamingEngine:
    """Stateful prefill + single-token decode. One session = one prefix."""

    def __init__(self, model, sw, expert_dir="ckpt_local/sft_mlx_bf16/expert",
                 max_seq_len=8192):
        self.model = model
        self.sw = sw
        self.cfg = model.cfg
        self.expert_dir = expert_dir
        self.max_seq_len = max_seq_len
        self.bench = KVBench()

        cfg = self.cfg
        # per-layer raw RoPE'd KV slots (MQA, single head of head_dim), appended per token.
        self._raw: list[list] = [[] for _ in range(cfg.n_layers)]
        # source-layer incremental compressor state
        self._src = {L: _SourceState(cfg.compress_ratios[L],
                                    L in cfg.index_source_layers)
                     for L in cfg.kv_source_layers}
        self._len = 0            # number of committed tokens
        self._history = []       # python list of token ids (for engram n-gram)

    # ------------------------------------------------------------------ utils
    def _engram_ids(self, L, full_ids_mx):
        """n-gram hash ids for layer L over the FULL history; return last-slot [n_hash_cols]."""
        all_ids = self.model.engram_hash(full_ids_mx, start_pos=0)  # [1,s,n_layers,cols]
        lidx = self.cfg.engram_layer_ids.index(L)
        return all_ids[:, -1:, lidx, :]  # [1,1,cols]

    def _attn_cossin(self, pos):
        return self.model.attn_cos[pos:pos + 1], self.model.attn_sin[pos:pos + 1]

    # ------------------------------------------------------ source layer update
    def _update_source(self, L, a_t, qr_t, WL):
        """Feed the new token's attention-normed activation `a_t` [1,1,d] into source
        layer L's compressor. Finalise a group if it fills. Returns nothing; mutates
        self._src[L]. Caller then reads .comp_groups / .index_groups."""
        st = self._src[L]
        cfg = self.cfg
        ratio = st.ratio
        eW = WL["compressor"]
        wkv = _deq(eW["wkv"]["weight"], mx.float32)  # [head_dim, d]
        kv_t = (a_t.astype(mx.float32) @ wkv.T)      # [1,1,head_dim]

        if ratio == 1:
            # per-token group immediately
            lat = _rms_norm(kv_t.astype(mx.bfloat16), eW["norm"]["weight"], cfg.norm_eps)  # [1,1,hd]
            pos = self._len  # absolute position of this token = number committed so far
            g_cos = self.model.comp_cos[pos:pos + 1]
            g_sin = self.model.comp_sin[pos:pos + 1]
            rot = apply_rope_real(lat, g_cos, g_sin)  # [1,1,hd]
            st.comp_groups.append(rot[0, 0])           # [hd]
            if st.has_index and "index_key" in WL:
                ikw = WL["index_key"]
                ik_wk = _deq(ikw["wk"]["weight"], mx.bfloat16)  # [index_head_dim, d]? maps hd->ihd
                ik = _rms_norm(lat.astype(mx.bfloat16) @ ik_wk.T, ikw["k_norm"]["weight"], cfg.norm_eps)
                ik = apply_rope_real(ik, g_cos, g_sin)  # [1,1,ihd]
                st.index_groups.append(ik[0, 0])
            return

        # ratio > 1: accumulate kv/score, pool when group fills
        wgate = _deq(eW["wgate"]["weight"], mx.float32)
        sc_t = a_t.astype(mx.float32) @ wgate.T        # [1,1,hd]
        st.partial_kv.append(kv_t[0, 0])               # [hd]
        st.partial_score.append(sc_t[0, 0])
        if len(st.partial_kv) == ratio:
            kv = mx.stack(st.partial_kv, axis=0)        # [ratio, hd]
            sc = mx.stack(st.partial_score, axis=0)     # [ratio, hd]
            weight = mx.softmax(sc, axis=0)             # [ratio, hd]
            pooled = mx.sum(kv * weight, axis=0)        # [hd]
            lat = _rms_norm(pooled[None, None].astype(mx.bfloat16),
                            eW["norm"]["weight"], cfg.norm_eps)  # [1,1,hd]
            g = len(st.comp_groups)                      # group index
            gpos = g * ratio
            g_cos = self.model.comp_cos[gpos:gpos + 1]
            g_sin = self.model.comp_sin[gpos:gpos + 1]
            rot = apply_rope_real(lat, g_cos, g_sin)     # [1,1,hd]
            st.comp_groups.append(rot[0, 0])
            if st.has_index and "index_key" in WL:
                ikw = WL["index_key"]
                ik_wk = _deq(ikw["wk"]["weight"], mx.bfloat16)
                ik = _rms_norm(lat.astype(mx.bfloat16) @ ik_wk.T, ikw["k_norm"]["weight"], cfg.norm_eps)
                ik = apply_rope_real(ik, g_cos, g_sin)
                st.index_groups.append(ik[0, 0])
            st.partial_kv.clear()
            st.partial_score.clear()

    # ---------------------------------------------------------- the layer body
    def _run_token(self, token_id, pos, full_ids_mx):
        """Run one token at absolute position `pos`. Returns logits [vocab]."""
        cfg = self.cfg
        sw = self.sw
        L_ = cfg.n_layers

        h = mx.take(sw.embed_weight.astype(mx.bfloat16),
                    mx.array([[token_id]], dtype=mx.int32), axis=0)  # [1,1,d]
        h = mx.broadcast_to(h[:, :, None, :], (1, 1, cfg.hc_mult, cfg.dim))

        pre_mix = mx.concatenate([mx.ones((1, 1, 1), dtype=mx.float32),
                                  mx.zeros((1, 1, cfg.hc_mult - 1), dtype=mx.float32)], axis=-1)

        acos, asin = self._attn_cossin(pos)

        # threaded cross-layer state (mirror forward_streaming locals)
        cur_comp_kv = None   # [1, n, hd] latest published
        cur_index_k = None   # [1, n, ihd] latest published
        cur_comp_len = 0

        for L in range(L_):
            WL = sw.get_layer(L)

            # ---- Engram injection (full-history n-gram) ----
            if L in cfg.engram_layer_ids:
                eW = sw.get_engram(L)
                if L in self.model.ssd_lookups:
                    hash_ids = self._engram_ids(L, full_ids_mx)  # [1,1,cols]
                    ssd_emb = self.model.ssd_lookups[L].lookup(hash_ids)  # [1,1,cols,ed]
                    h = engram_forward_with_embed(h, hash_ids, ssd_emb, eW, cfg)

            # ---- attention sublayer ----
            residual = h
            attn_pre, attn_post, attn_comb = hc_mixes(
                h, WL["hc"]["attn_fn"], WL["hc"]["attn_scale"], WL["hc"]["attn_base"],
                cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
            a = hc_pre(h, pre_mix)
            a = _rms_norm(a, WL["attn_norm"]["weight"], cfg.norm_eps)  # [1,1,d]

            q, qr = self.model._qproj(a, WL)                  # q[1,1,nh,hd], qr[1,1,qlr]
            q = apply_rope_real(q, acos, asin)
            kv = self.model._kvproj(a, WL)                   # [1,1,head_dim]
            kv = apply_rope_real(kv, acos, asin)

            # store raw RoPE'd KV slot
            self._raw[L].append(kv[0, 0])  # [head_dim]

            ratio = cfg.compress_ratios[L]
            comp_kv_out = None
            comp_idx_out = None

            if ratio > 0:
                # publish compressed KV if source layer
                if L in cfg.kv_source_layers and "compressor" in WL:
                    self._update_source(L, a, qr, WL)
                    st = self._src[L]
                    if st.comp_groups:
                        cur_comp_kv = mx.stack(st.comp_groups, axis=0)[None]  # [1,n,hd]
                        cur_comp_len = len(st.comp_groups)
                    if st.has_index and st.index_groups:
                        cur_index_k = mx.stack(st.index_groups, axis=0)[None]  # [1,n,ihd]

                # run indexer if this layer is an index source
                if L in cfg.index_source_layers and cur_index_k is not None and cur_comp_len > 0:
                    comp_idx_out = self._decode_indexer(a, qr, cur_index_k, pos, WL)

                if cur_comp_kv is not None:
                    comp_kv_out = cur_comp_kv
                    cur_comp_len = cur_comp_kv.shape[1]

            # ---- assemble attention inputs ----
            # raw KV = all committed slots [0..pos]; window indices use absolute
            # positions with -1 padding (take wraps -1 -> last committed token),
            # exactly matching forward_streaming._window_idx.
            raw_all = mx.stack(self._raw[L], axis=0)[None]        # [1,pos+1,hd]
            win_lo = max(0, pos - cfg.window_size + 1)
            win_idx = mx.arange(win_lo, win_lo + cfg.window_size, dtype=mx.int32)
            win_idx = mx.where(win_idx > pos, -1, win_idx)[None, None]  # [1,1,w]

            if comp_kv_out is not None:
                full_kv = mx.concatenate([raw_all, comp_kv_out], axis=1)  # [1, pos+1+n, hd]
                if comp_idx_out is not None:
                    comp_gather = mx.where(comp_idx_out >= 0, comp_idx_out + (pos + 1), -1)
                else:
                    comp_gather = None
            else:
                full_kv = raw_all
                comp_gather = None

            o = sparse_attention(q, full_kv, WL["attn_sink"], win_idx, comp_gather,
                                 cfg.head_dim ** -0.5, cfg.attn_logit_softcap)
            o = apply_rope_real(o, acos, asin, inverse=True)
            a_out = self.model._oproj(o, WL)
            h = hc_post(a_out, residual, attn_post, attn_comb)

            # ---- FFN sublayer ----
            residual = h
            ffn_pre, ffn_post, ffn_comb = hc_mixes(
                h, WL["hc"]["ffn_fn"], WL["hc"]["ffn_scale"], WL["hc"]["ffn_base"],
                cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
            f = hc_pre(h, attn_pre)
            f = _rms_norm(f, WL["ffn_norm"]["weight"], cfg.norm_eps)
            f = sparse_moe_forward_batched(
                f, WL["ffn"]["gate"]["weight"], WL["ffn"]["gate"]["bias"],
                WL["ffn"]["shared"]["w1"]["weight"], WL["ffn"]["shared"]["w3"]["weight"],
                WL["ffn"]["shared"]["w2"]["weight"], self.expert_dir, L, cfg)
            h = hc_post(f, residual, ffn_post, ffn_comb)
            pre_mix = ffn_pre

        h = hc_pre(h, pre_mix)
        h = _rms_norm(h, sw.norm_weight, cfg.norm_eps)
        logits = h.astype(mx.bfloat16) @ sw.head_weight.astype(mx.bfloat16).T  # [1,1,vocab]
        return logits[0, 0, :]

    def _decode_indexer(self, a, qr, index_k, pos, WL):
        """Indexer for the single decode query at absolute `pos`.

        Matches model.indexer_forward but uses absolute-position causal visibility:
          key_ratio = max(1,(pos+1)//n); visible groups col < (pos+1)//key_ratio.
        """
        cfg = self.cfg
        n = index_k.shape[1]
        idx_w = IndexerWeights(
            wq_b=WL["indexer"]["wq_b"]["weight"],
            weights_proj=WL["indexer"]["weights_proj"]["weight"],
        )
        wq_b = _deq(idx_w.wq_b, mx.bfloat16)  # [n_heads*ihd, q_lora]
        q = (qr.astype(mx.bfloat16) @ wq_b.T).reshape(1, 1, cfg.index_n_heads, cfg.index_head_dim)
        qrot = apply_rope_real(q, self.model.attn_cos[pos:pos + 1], self.model.attn_sin[pos:pos + 1])

        weights = _deq(idx_w.weights_proj, mx.float32)  # [n_heads, d]
        w_scale = (cfg.index_head_dim ** -0.5) * (cfg.index_n_heads ** -0.5)
        head_weights = (a.astype(mx.float32) @ weights.T) * w_scale  # [1,1,n_heads]

        scores = mx.einsum("bshd,bnd->bshn", qrot.astype(mx.float32), index_k.astype(mx.float32))
        scores = mx.maximum(scores, 0.0)
        scores = mx.sum(scores * head_weights[..., None], axis=2)  # [1,1,n]

        # absolute causal visibility
        key_ratio = max(1, (pos + 1) // n) if n else 1
        vis = (pos + 1) // key_ratio                      # visible columns < vis
        col = mx.arange(n)
        mask = col[None, :] >= vis                        # [1,n]
        scores = mx.where(mask[None], -float("inf"), scores)

        k = min(cfg.index_topk, n)
        _, idx = self._topk_local(scores, k=k, axis=-1)
        idx = mx.sort(idx, axis=-1)
        return idx.astype(mx.int32)  # [1,1,k]

    @staticmethod
    def _topk_local(x, k, axis=-1):
        n = x.shape[axis]
        idx = mx.argpartition(x, kth=n - k, axis=axis)
        top_idx = idx[..., n - k:]
        top_val = mx.take_along_axis(x, top_idx, axis=axis)
        return top_val, top_idx

    # ------------------------------------------------------------ public API
    def prefill(self, token_ids):
        """One-shot batch prefill over the whole prompt (mirrors forward_streaming),
        then cache every per-layer RoPE'd raw KV slot and the source-layer
        compressor/indexer state. Returns last-token logits [vocab].

        We run the prompt as a batch (s=P) so the window `-1` padding wraps to the
        last prompt token exactly like the accurate path -- a token-by-token replay
        would wrap to the wrong (current) token and diverge.
        """
        t0 = time.time()
        ids = list(token_ids)
        self._history = list(ids)
        P = len(ids)
        sw = self.sw
        cfg = self.cfg

        h = mx.take(sw.embed_weight.astype(mx.bfloat16),
                    mx.array([ids], dtype=mx.int32), axis=0)  # [1,P,d]
        h = mx.broadcast_to(h[:, :, None, :], (1, P, cfg.hc_mult, cfg.dim))
        pre_mix = mx.concatenate([mx.ones((1, P, 1), dtype=mx.float32),
                                  mx.zeros((1, P, cfg.hc_mult - 1), dtype=mx.float32)], axis=-1)

        acos = self.model.attn_cos[:P][None]
        asin = self.model.attn_sin[:P][None]
        full_ids = mx.array([ids], dtype=mx.int32)

        cur_comp_kv = None
        cur_index_k = None
        cur_comp_len = 0

        for L in range(cfg.n_layers):
            WL = sw.get_layer(L)

            if L in cfg.engram_layer_ids:
                eW = sw.get_engram(L)
                if L in self.model.ssd_lookups:
                    hash_ids = self.model._engram_hash(L, full_ids)  # [1,P,cols]
                    ssd_emb = self.model.ssd_lookups[L].lookup(hash_ids)
                    h = engram_forward_with_embed(h, hash_ids, ssd_emb, eW, cfg)

            residual = h
            attn_pre, attn_post, attn_comb = hc_mixes(
                h, WL["hc"]["attn_fn"], WL["hc"]["attn_scale"], WL["hc"]["attn_base"],
                cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
            a = hc_pre(h, pre_mix)
            a = _rms_norm(a, WL["attn_norm"]["weight"], cfg.norm_eps)  # [1,P,d]

            q, qr = self.model._qproj(a, WL)
            q = apply_rope_real(q, acos[0], asin[0])
            kv = self.model._kvproj(a, WL)                     # [1,P,head_dim]
            kv = apply_rope_real(kv, acos[0], asin[0])
            self._raw[L] = [kv[0, i] for i in range(P)]         # list of [head_dim]

            ratio = cfg.compress_ratios[L]
            comp_kv_out = None
            comp_idx_out = None

            if ratio > 0:
                if L in cfg.kv_source_layers and "compressor" in WL:
                    self._prefill_source(L, a, WL)
                    st = self._src[L]
                    if st.comp_groups:
                        cur_comp_kv = mx.stack(st.comp_groups, axis=0)[None]
                        cur_comp_len = len(st.comp_groups)
                    if st.has_index and st.index_groups:
                        cur_index_k = mx.stack(st.index_groups, axis=0)[None]

                if L in cfg.index_source_layers and cur_index_k is not None and cur_comp_len > 0:
                    idx_w = IndexerWeights(
                        wq_b=WL["indexer"]["wq_b"]["weight"],
                        weights_proj=WL["indexer"]["weights_proj"]["weight"])
                    comp_idx_out = indexer_forward(a, qr, cur_index_k, acos[0], asin[0],
                                                   idx_w, cfg, start_pos=0)
                if cur_comp_kv is not None:
                    comp_kv_out = cur_comp_kv
                    cur_comp_len = cur_comp_kv.shape[1]

            win_idx = self.model._window_idx(P)  # [1,P,w]
            if comp_kv_out is not None:
                full_kv = mx.concatenate([kv, comp_kv_out], axis=1)
                if comp_idx_out is not None:
                    comp_gather = mx.where(comp_idx_out >= 0, comp_idx_out + P, -1)
                else:
                    comp_gather = None
            else:
                full_kv = kv
                comp_gather = None

            o = sparse_attention(q, full_kv, WL["attn_sink"], win_idx, comp_gather,
                                 cfg.head_dim ** -0.5, cfg.attn_logit_softcap)
            o = apply_rope_real(o, acos[0], asin[0], inverse=True)
            a_out = self.model._oproj(o, WL)
            h = hc_post(a_out, residual, attn_post, attn_comb)

            residual = h
            ffn_pre, ffn_post, ffn_comb = hc_mixes(
                h, WL["hc"]["ffn_fn"], WL["hc"]["ffn_scale"], WL["hc"]["ffn_base"],
                cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
            f = hc_pre(h, attn_pre)
            f = _rms_norm(f, WL["ffn_norm"]["weight"], cfg.norm_eps)
            f = sparse_moe_forward_batched(
                f, WL["ffn"]["gate"]["weight"], WL["ffn"]["gate"]["bias"],
                WL["ffn"]["shared"]["w1"]["weight"], WL["ffn"]["shared"]["w3"]["weight"],
                WL["ffn"]["shared"]["w2"]["weight"], self.expert_dir, L, cfg)
            h = hc_post(f, residual, ffn_post, ffn_comb)
            pre_mix = ffn_pre

        h = hc_pre(h, pre_mix)
        h = _rms_norm(h, sw.norm_weight, cfg.norm_eps)
        logits = h.astype(mx.bfloat16) @ sw.head_weight.astype(mx.bfloat16).T  # [1,P,vocab]
        last = logits[0, -1, :]
        mx.eval(last)
        self._len = P
        self.bench.prefill_ms = (time.time() - t0) * 1000
        return last

    def _prefill_source(self, L, a, WL):
        """Build source-layer compressor state over the whole prompt activation a [1,P,d]."""
        st = self._src[L]
        cfg = self.cfg
        ratio = st.ratio
        eW = WL["compressor"]
        wkv = _deq(eW["wkv"]["weight"], mx.float32)
        kv_all = a.astype(mx.float32) @ wkv.T               # [1,P,hd]
        if ratio == 1:
            lat = _rms_norm(kv_all.astype(mx.bfloat16), eW["norm"]["weight"], cfg.norm_eps)
            rot = apply_rope_real(lat, self.model.comp_cos[:kv_all.shape[1]],
                                  self.model.comp_sin[:kv_all.shape[1]])  # [1,P,hd]
            st.comp_groups = [rot[0, i] for i in range(kv_all.shape[1])]
            if st.has_index and "index_key" in WL:
                ikw = WL["index_key"]
                ik_wk = _deq(ikw["wk"]["weight"], mx.bfloat16)
                ik = _rms_norm(lat.astype(mx.bfloat16) @ ik_wk.T, ikw["k_norm"]["weight"], cfg.norm_eps)
                ik = apply_rope_real(ik, self.model.comp_cos[:kv_all.shape[1]],
                                     self.model.comp_sin[:kv_all.shape[1]])
                st.index_groups = [ik[0, i] for i in range(kv_all.shape[1])]
            st.partial_kv.clear(); st.partial_score.clear()
            return

        wgate = _deq(eW["wgate"]["weight"], mx.float32)
        sc_all = a.astype(mx.float32) @ wgate.T              # [1,P,hd]
        P = kv_all.shape[1]
        n_groups = P // ratio
        kv_g = kv_all[0, :n_groups * ratio].reshape(n_groups, ratio, -1)
        sc_g = sc_all[0, :n_groups * ratio].reshape(n_groups, ratio, -1)
        w = mx.softmax(sc_g, axis=1)
        pooled = mx.sum(kv_g * w, axis=1)                    # [n_groups, hd]
        lat = _rms_norm(pooled[None].astype(mx.bfloat16), eW["norm"]["weight"], cfg.norm_eps)
        g_cos = self.model.comp_cos[::ratio][:n_groups]
        g_sin = self.model.comp_sin[::ratio][:n_groups]
        rot = apply_rope_real(lat, g_cos, g_sin)             # [1,n_groups,hd]
        st.comp_groups = [rot[0, i] for i in range(n_groups)]
        if st.has_index and "index_key" in WL:
            ikw = WL["index_key"]
            ik_wk = _deq(ikw["wk"]["weight"], mx.bfloat16)
            ik = _rms_norm(lat.astype(mx.bfloat16) @ ik_wk.T, ikw["k_norm"]["weight"], cfg.norm_eps)
            ik = apply_rope_real(ik, g_cos, g_sin)
            st.index_groups = [ik[0, i] for i in range(n_groups)]
        # tail partial group
        st.partial_kv = [kv_all[0, i] for i in range(n_groups * ratio, P)]
        st.partial_score = [sc_all[0, i] for i in range(n_groups * ratio, P)]

    def decode_step(self, token_id):
        """Commit the new token and return logits [vocab] for it."""
        pos = self._len
        self._history.append(int(token_id))
        full_ids = mx.array([self._history], dtype=mx.int32)
        t0 = time.time()
        lg = self._run_token(int(token_id), pos, full_ids)
        mx.eval(lg)
        self._len = pos + 1
        self.bench.decode_ms_sum += (time.time() - t0) * 1000
        self.bench.decode_steps += 1
        return lg
