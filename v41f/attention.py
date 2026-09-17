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

from v41f.compressor import Compressor
from v41f.indexer import Indexer
from v41f.indexer_ste import ste_slot_weight
from v41f.norm_gate import RMSNorm
from v41f.projections import GroupedOProj, KVProj, QProj
from v41f.rope import apply_rotary_emb, precompute_freqs_cis
from v41f.sparse_attn import sparse_attn
from v41f.window import get_window_topk_idxs


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
        kv = self.kvproj(x)
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
            k = self.index_key(latent)  # index keys, RoPE-free
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

    def forward(self, x, state=None):
        """Prefill forward. `state` is the shared per-pass SharedAttnState; a fresh one is
        made when the caller omits it (single-layer tests), but multi-layer models pass one."""
        bsz, seqlen, _ = x.size()
        freqs = self.freqs_cis[:seqlen]
        if state is None:
            state = SharedAttnState()

        q, qr = self.qproj(x)
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
        o = o.to(q.dtype)
        apply_rotary_emb(o[..., -self.rd :], freqs, inverse=True)
        return self.oproj(o), state
