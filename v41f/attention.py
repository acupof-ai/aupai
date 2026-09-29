"""Full V4.1-Flash Attention block assembled from the P0 leaf modules.

Faithful to upstream Attention (model_ref.py.ref class Attention, forward :765-789,
_window_kv :700-720, _compress_kv :739-763, _compress_topk_idxs :722-737):

  q  = unflatten(wq_b(q_norm(wq_a(x)))) ; RoPE on the rope tail
  kv = kv_norm(wkv(x))                 ; RoPE tail (MQA, one KV shared by all heads)
  window KV + per-query window idxs
  if compress_ratio > 0:
      owner layers compress their own KV (Compressor) and build the index keys;
      index sources run the Indexer; other layers REUSE the published compressed KV,
      index keys and topk from an explicit runtime container (NOT the upstream process
      global `shared_attn`, which is unsafe across checkpointing / multiple backwards);
      concat [window_kv ; compressed_kv] and [window_idxs ; compressed_idxs+offset]
  o  = sparse_attn(q, kv, attn_sink, idxs, scale)   ; single softmax, shared sink
  inverse RoPE on the output rope tail
  grouped block-diagonal wo_a einsum -> wo_b back to dim

Scope for v41f P0/training: prefill only (start_pos=0). The decode ring buffer and the
fp8/fp4 act_quant calls are omitted — on the CPU bf16 oracle they are identities and the
production training path never decodes. The two-level candidate indexer is config-gated
(v41f-S sets candidate_source_layer=-1), but the plumbing is passed through.
"""

import torch
from torch import nn

from v41f import docpack
from v41f.compressor import Compressor
from v41f.indexer import Indexer
from v41f.indexer_ste import ste_slot_weight
from v41f.norm_gate import RMSNorm
from v41f.projections import GroupedOProj, KVProj, QProj
from v41f.rope import apply_rotary_emb, apply_rotary_real, precompute_freqs_cis, rope_cos_sin
from v41f.sparse_attn import sparse_attn
from v41f.window import get_window_topk_idxs


def _rot_tail(t, fn, rd):
    """RoPE on the last rd dims, out of place: a q/kv from a Float8Linear is a custom Function
    output and autograd refuses an in-place write through a view of it. rope_impl real
    allocates nothing but the result; complex clones the 64-dim tail, not the tensor."""
    return torch.cat((t[..., :-rd], fn(t[..., -rd:])), dim=-1)


class SharedAttnState:
    """Explicit per-forward replacement for the upstream process-global `shared_attn`.

    One instance is created per forward pass (or per model when state genuinely persists,
    e.g. decode caches) and threaded through layers, so layer L reads exactly what the source
    layer published and two concurrent/backward passes cannot alias each other's tensors.
    For prefill training the only cross-layer fields are the compressed KV, the index keys
    and the topk idxs; window caches are per-layer and decode-only.
    """

    def __init__(self):
        self.compress_kv = None  # [b, n_compressed, head_dim], published by kv source
        self.index_k = None  # [b, n_compressed, index_head_dim], published by owner
        self.topk_idxs = None  # [b, s, index_topk], published by an index source
        self.candidates = None  # level-one block mask, from the candidate source
        self.sel_scores = None  # continuous scores at the published slots (training only)
        self.cu = None  # int32 cu_seqlens over the flattened b*s stream; None = one doc per row
        # packed-row path (v41f/docpack.py): per-token layout, entry metadata, indexer loss
        self.doc = self.pos = self.doclen = None  # [b,s]
        self.cu_docs = None  # int32 merged cu_seqlens (fused window branch)
        self.entry_meta = None  # (e_valid, e_doc, e_last, ratio) from the kv source
        self.indexer_kl = []  # (sum, count) per index-source layer under indexer_train_mode "kl"
        # qk-scale probe (train health): when record_qk, every layer appends
        # (layer_id, per-head q RMS [h], kv RMS) after RoPE; read by V41FModel into model.qk_stats
        self.record_qk = False
        self.qk_stats = []


class IndexKeyProj(nn.Module):
    """Owner-side builder of the shared index keys from the RoPE-free compressed latent.

    Lives only on a layer that BOTH is an index source AND owns the compression (the latent
    it projects is produced in that same layer). Upstream puts wk/k_norm inside Indexer when
    `owns_k`; kept here as a named module so the leaf Indexer stays a pure scorer and the
    owner/publisher boundary is explicit. k: head_dim -> index_head_dim, bf16 upstream; the
    RoPE tail is applied by the caller at the compressed group positions.
    """

    def __init__(self, head_dim: int, index_head_dim: int, eps: float):
        super().__init__()
        self.wk = nn.Linear(head_dim, index_head_dim, bias=False)
        self.k_norm = RMSNorm(index_head_dim, eps)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.k_norm(self.wk(latent))


class Attention(nn.Module):
    def __init__(self, cfg, layer_id: int, max_batch_size: int = 4):
        super().__init__()
        self.layer_id = layer_id
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rd = cfg.rope_head_dim
        self.n_groups = cfg.o_groups
        self.window_size = cfg.window_size
        ratio = cfg.compress_ratios[layer_id]
        self.compress_ratio = ratio
        self.softmax_scale = cfg.head_dim**-0.5

        # finite default; the checkpoint loader overwrites. empty fp32 can be NaN, which
        # would make the allclose oracle non-deterministic (the gate-test flaky lesson).
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))

        self.qproj = QProj(cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.head_dim, cfg.norm_eps)
        self.kvproj = KVProj(cfg.dim, cfg.head_dim, cfg.norm_eps)
        self.oproj = GroupedOProj(cfg.n_heads, cfg.head_dim, cfg.o_groups, cfg.o_lora_rank, cfg.dim)

        backbone = layer_id < cfg.n_layers
        self.is_kv_source = backbone and layer_id in cfg.kv_source_layers
        self.is_index_source = backbone and layer_id in cfg.index_source_layers
        self.owns_index_k = self.is_index_source and self.is_kv_source

        self.compressor = (
            Compressor(cfg.dim, cfg.head_dim, ratio, norm_eps=cfg.norm_eps, max_batch_size=max_batch_size)
            if self.is_kv_source
            else None
        )
        self.indexer = (
            Indexer(
                cfg.dim,
                cfg.q_lora_rank,
                cfg.index_n_heads,
                cfg.index_head_dim,
                cfg.rope_head_dim,
                cfg.index_topk,
                ratio,
            )
            if self.is_index_source
            else None
        )
        self.index_key = (
            IndexKeyProj(cfg.head_dim, cfg.index_head_dim, cfg.norm_eps) if self.owns_index_k else None
        )

        orig, theta = (cfg.original_seq_len, cfg.compress_rope_theta) if ratio else (0, cfg.rope_theta)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.rd,
                4096,
                original_seq_len=orig,
                base=theta,
                factor=cfg.rope_factor,
                beta_fast=cfg.beta_fast,
                beta_slow=cfg.beta_slow,
            ),
            persistent=False,
        )

    # ------------------------------------------------------------------ prefill helpers
    def _window_kv(self, x, freqs, bsz):
        """Sliding-window raw KV (MQA) and the per-query window position idxs (prefill)."""
        kv = self.kvproj(x).clone()  # clone: rotated in place below; a Linear under fp8/compile returns a custom-Function output whose view cannot take an in-place write
        apply_rotary_emb(kv[..., -self.rd :], freqs)
        seqlen = x.size(1)
        # prefill attends over the chunk directly; the ring-buffer write is decode-only and
        # omitted, so only the returned kv and idxs matter for training.
        idxs = get_window_topk_idxs(self.window_size, bsz, seqlen, 0, device=x.device)
        return kv, idxs

    def _group_freqs(self, seqlen, ratio):
        """Compressed latent at group j takes the position of its group's first token, j*r."""
        kept = seqlen - seqlen % ratio
        return self.freqs_cis[:kept:ratio]

    def _compress(self, x, qr, freqs, window_len, bsz, state):
        """Publish (source) or reuse the shared compressed KV + index keys, run/reuse topk.

        Returns compressed KV [b, n, head_dim] and compressed idxs [b, s, k] already offset
        to sit AFTER the window positions, or (None, None) when no compressed position is
        reachable yet.
        """
        seqlen = x.size(1)
        latent = self.compressor(x, 0) if self.is_kv_source else None

        if self.owns_index_k and latent is not None:
            k = self.index_key(latent).clone()  # index keys, RoPE-free; rotated in place below
            apply_rotary_emb(k[..., -self.rd :], self._group_freqs(seqlen, self.compress_ratio))
            state.index_k = k
        if self.is_kv_source and latent is not None:
            rot = latent.clone()
            apply_rotary_emb(rot[..., -self.rd :], self._group_freqs(seqlen, self.compress_ratio))
            state.compress_kv = rot  # published KV is RoPE'd

        if self.is_index_source:
            # the leaf indexer scores against the published keys and returns local positions
            assert state.index_k is not None, "index source reached with no published keys"
            if self.cfg.indexer_train_mode == "ste":
                idxs, sc = self.indexer.select(x, qr, state.index_k, freqs, 0, window_len)
                state.sel_scores = sc
            else:
                idxs = self.indexer(x, qr, state.index_k, freqs, 0, window_len)
            state.topk_idxs = idxs
            return state.compress_kv, idxs

        # non-index layers reuse the source's published KV and topk for this same q set
        return state.compress_kv, state.topk_idxs

    # ------------------------------------------------------------------ packed-row path
    def _compress_docs(self, x, qr, freqs, state):
        """_compress for packed rows: per-document groups and visibility (docpack). Returns
        (compressed KV [b,n,d] RoPE'd, selected entry indices [b,s,k] with -1 empty)."""
        if self.is_kv_source:
            latent, e_start, e_valid = self.compressor.forward_docs(x, state.pos, state.doclen)
            efreqs = self.freqs_cis[state.pos.gather(1, e_start)]
            if self.cfg.rope_impl == "real":
                ecos, esin = rope_cos_sin(efreqs)
                erot = lambda t: apply_rotary_real(t, ecos, esin)  # noqa: E731
            else:
                erot = lambda t: apply_rotary_emb(t.clone(), efreqs)  # noqa: E731
            if self.owns_index_k:
                # detached latent: the index keys learn only from the indexer loss
                state.index_k = _rot_tail(self.index_key(latent.detach()), erot, self.rd)
            state.compress_kv = _rot_tail(latent, erot, self.rd)
            e_last = e_start + self.compress_ratio - 1
            state.entry_meta = (e_valid, state.doc.gather(1, e_start), e_last, self.compress_ratio)
        if self.is_index_source:
            assert state.index_k is not None, "index source reached with no published keys"
            e_valid, e_doc, e_last, m = state.entry_meta
            visible = docpack.entry_visibility(e_valid, e_doc, e_last, state.doc)
            state.topk_idxs = docpack.select_visible(self.indexer, x, qr, state.index_k, freqs, visible, m)
        return state.compress_kv, state.topk_idxs

    def _forward_docs(self, x, state):
        if self.cfg.indexer_train_mode == "ste":
            raise NotImplementedError("indexer_train_mode 'ste' has no packed-row path; use 'kl'")
        bsz, seqlen, _ = x.size()
        freqs = self.freqs_cis[state.pos]  # [b,s,rd/2]: positions restart per document
        real = self.cfg.rope_impl == "real"
        if real:
            cos, sin = rope_cos_sin(freqs)
            rot = lambda t, inv=False: apply_rotary_real(t, cos, sin, inverse=inv)  # noqa: E731
        else:
            rot = lambda t, inv=False: apply_rotary_emb(t.clone(), freqs, inverse=inv)  # noqa: E731
        q, qr = self.qproj(x)
        q = _rot_tail(q, rot, self.rd)
        kv = _rot_tail(self.kvproj(x), rot, self.rd)
        if state.record_qk:
            # per-token norms only, no sequence matmul; RoPE is norm-preserving so this is the q the
            # softmax sees. Side effect on the state object, replayed by dynamo like indexer_kl.
            state.qk_stats.append((self.layer_id, q.detach().float().square().mean(dim=(0, 1, 3)).sqrt(),
                                   kv.detach().float().square().mean().sqrt()))
        comp_kv = comp_idx = None
        if self.compress_ratio:
            comp_kv, comp_idx = self._compress_docs(x, qr, freqs, state)
        if self.cfg.attn_impl in ("chunked", "fused"):
            m = state.entry_meta[3] if comp_kv is not None else 1
            if self.cfg.attn_impl == "fused":
                o, lse = docpack.fused_sparse_attn(
                    q, kv, comp_kv, comp_idx, self.attn_sink, self.softmax_scale, state.doc, state.cu_docs,
                    self.window_size, m)
            else:
                o, lse = docpack.windowed_sparse_attn(
                    q, kv, comp_kv, comp_idx, self.attn_sink, self.softmax_scale, state.doc, self.window_size, m)
            if self.is_index_source and self.cfg.indexer_train_mode == "kl":
                state.indexer_kl.append(docpack.indexer_kl(
                    self.indexer, x, qr, state.index_k, freqs, q, comp_kv, comp_idx, lse,
                    self.softmax_scale, m))
        else:
            win = get_window_topk_idxs(self.window_size, bsz, seqlen, 0, device=x.device)
            start = torch.arange(seqlen, device=x.device) - state.pos  # [b,s] document start
            idxs = torch.where(win >= start[..., None], win, -1)
            if comp_kv is not None:
                kv = torch.cat([kv, comp_kv], dim=1)
                idxs = torch.cat([idxs, torch.where(comp_idx >= 0, comp_idx + seqlen, -1)], dim=-1)
            o = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale)
        o = _rot_tail(o.to(q.dtype), lambda t: rot(t, True), self.rd)
        return self.oproj(o), state

    def forward(self, x, state=None):
        """Prefill forward. `state` is the shared per-pass SharedAttnState; a fresh one is
        made when the caller omits it (single-layer tests), but multi-layer models pass one."""
        bsz, seqlen, _ = x.size()
        freqs = self.freqs_cis[:seqlen]
        if state is None:
            state = SharedAttnState()
        if state.doc is not None:
            return self._forward_docs(x, state)

        q, qr = self.qproj(x)
        q = q.clone()  # rotated in place below
        apply_rotary_emb(q[..., -self.rd :], freqs)

        kv, idxs = self._window_kv(x, freqs, bsz)
        sel_scores = None
        if self.compress_ratio:
            comp_kv, comp_idxs = self._compress(x, qr, freqs, kv.size(1), bsz, state)
            if comp_kv is not None and comp_idxs is not None:
                kv = torch.cat([kv, comp_kv], dim=1)
                idxs = torch.cat([idxs, comp_idxs], dim=-1)
                # The window slots LEAD the concatenated selection and are not indexer-chosen,
                # so they carry no straight-through weight. Pad the front with ones (the
                # identity) to line the indexer's own scores up with their slots: the whole
                # tensor is then 1.0 in forward and only the tail carries a softmax gradient.
                if state.sel_scores is not None:
                    pad = idxs.size(-1) - state.sel_scores.size(-1)
                    ones = torch.ones(
                        *state.sel_scores.shape[:-1], pad,
                        device=x.device, dtype=state.sel_scores.dtype)
                    sel_scores = torch.cat([ones, state.sel_scores], dim=-1)

        if sel_scores is None:
            # THE OFF PATH MAKES THE IDENTICAL CALL. Not `slot_weight=None`: passing the
            # keyword at all changes the call signature every inference stub and monkeypatch
            # sees, so the faithful path is kept literally byte-for-byte the old call.
            o = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale)
        else:
            o = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale,
                            slot_weight=ste_slot_weight(sel_scores))
        # the kernel accumulates in fp32 against the fp32 attn_sink but stores empty_like(q),
        # so its output is the activation dtype; match that boundary before wo_a (bf16).
        o = o.to(q.dtype).clone()  # inverse RoPE writes in place below
        apply_rotary_emb(o[..., -self.rd :], freqs, inverse=True)
        return self.oproj(o), state
