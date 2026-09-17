"""DSpark MTP stage-1 training loss (P3 first cut, n_mtp_layers=1, block_size=5).

Scope, deliberately smaller than upstream DSparkBlock
(third_party/deepseek_v41_ref/model_ref.py.ref:1100-1156): the reference implements
the INFERENCE path only — a sparse-attn draft block plus Markov/confidence heads and
autoregressive speculative sampling (forward_head/forward_spec). P3 omits all three:
no Markov-rank head (config dspark_markov_rank=0), no confidence head, no
forward_spec. This module is the TRAINING-only scaffolding the reference does not
have, namely:

1. main projection: the anchor layer's hidden is the concat of the attn INPUTS of
   `dspark_target_layer_ids` target layers, [B, T, dim] flattened to [B, dim*T],
   projected back to dim and RMSNormed — the same order as forward_embed:
       main_x = main_norm(main_proj(main_hidden))         (model_ref:1130)
2. draft input construction: a block_size query whose position 0 is the real anchor
   token and positions 1..block_size-1 are learned NOISE tokens, exactly
   forward_embed:1131-1135. One draft forward predicts all block_size next tokens in
   parallel; noise lives on the INPUT side only.
3. teacher-forced multi-token CE: draft position i is supervised against the real
   token at main-sequence position anchor+1+i. Every target is a REAL future token;
   a noise query position is never a supervision target.

Upstream provides NO training loss to match, so this is not an allclose module: the
weighting is v41f-defined (uniform mean CE over the real future tokens, optional
per-position weights) and is registered in the v41f prereg. The draft transformer
block and the LM head are injected callables — the real DSparkBlock (sparse-attn,
hyper-connections) and the tied head plug in at P3; here a deterministic mock lets
the loss/position/noise contract be checked in a known-answer world.
"""

import torch
import torch.nn.functional as F
from torch import nn

from v41f.norm_gate import RMSNorm


class DSparkMTP(nn.Module):
    """Stage-1 DSpark training pieces: main projection, draft ids, multi-token CE.

    dim: model width. n_target_layers: number of layers whose attn-input hidden is
    concatenated (len(dspark_target_layer_ids)). The embedding is standalone here
    (plain nn.Embedding); training ties it to the shared token embedding.
    """

    def __init__(
        self,
        dim: int,
        n_target_layers: int,
        vocab_size: int,
        block_size: int,
        noise_token_id: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if n_target_layers < 1:
            raise ValueError("DSpark needs at least one target layer")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.dim = dim
        self.block_size = block_size
        self.noise_token_id = noise_token_id
        # model_ref:1113 main_proj is a Linear with the default no-bias contract.
        self.main_proj = nn.Linear(dim * n_target_layers, dim, bias=False)
        self.main_norm = RMSNorm(dim, eps)
        self.embed = nn.Embedding(vocab_size, dim)

    def project_main(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """[B, T, dim] target-layer attn-input hidden -> projected anchor [B, dim].

        Flatten order is target-layer order then dim: position t*dim:(t+1)*dim carries
        target layer t, matching the concat the caller feeds main_proj upstream.
        """
        if main_hidden.dim() != 3 or main_hidden.shape[-1] != self.dim:
            raise ValueError(f"expected [B,T,dim], got {tuple(main_hidden.shape)}")
        return self.main_norm(self.main_proj(main_hidden.flatten(-2)))

    def draft_input_ids(self, anchor_ids: torch.Tensor) -> torch.Tensor:
        """[B] anchor token -> [B, block_size] with the real token at 0, noise after."""
        if anchor_ids.dim() != 1:
            raise ValueError(f"anchor_ids must be [B], got {tuple(anchor_ids.shape)}")
        draft = anchor_ids.new_full((anchor_ids.size(0), self.block_size), self.noise_token_id)
        draft[:, 0] = anchor_ids
        return draft

    def draft_embed(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """[B, block_size] -> [B, block_size, dim]. The hc_mult expansion happens inside
        the real draft block's hyper-connections; this stays on plain dim."""
        return self.embed(draft_ids)

    def forward(
        self,
        main_hidden: torch.Tensor,
        anchor_ids: torch.Tensor,
        future_ids: torch.Tensor,
        draft_block,
        head,
        ignore_index: int = -100,
        pos_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-token teacher-forced loss.

        main_hidden: [B, T, dim] attn inputs of the target layers at the anchor.
        anchor_ids: [B] the real token at the anchor position p.
        future_ids: [B, block_size] REAL tokens at p+1 .. p+block_size (use ignore_index
            to mask a position); noise query positions are inputs, never these labels.
        draft_block: (x:[B,S,dim], main_x:[B,dim]) -> [B,S,dim], the future DSparkBlock.
        head: (hidden:[B,S,dim]) -> logits [B,S,vocab].
        pos_weights: optional [block_size] draft-position weights (v41f-defined).

        Returns (mean CE over non-ignored targets, logits [B,S,vocab]).
        """
        if future_ids.shape != (anchor_ids.size(0), self.block_size):
            raise ValueError(f"future_ids must be [B,{self.block_size}], got {tuple(future_ids.shape)}")
        main_x = self.project_main(main_hidden)
        draft_ids = self.draft_input_ids(anchor_ids)
        x = self.draft_embed(draft_ids)
        hidden = draft_block(x, main_x)
        logits = head(hidden)
        if logits.shape[:2] != future_ids.shape:
            raise ValueError(f"head logits {tuple(logits.shape)} must lead with [B,{self.block_size}]")

        log_probs = F.log_softmax(logits.float(), dim=-1)
        valid = future_ids.ne(ignore_index)
        # clamp the (masked) label to a legal column for gather; its contribution is zeroed
        # by the validity weight, so the value gathered there is never read.
        safe_labels = future_ids.clamp_min(0)
        picked = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        if pos_weights is None:
            pos_weights = torch.ones(self.block_size, device=future_ids.device, dtype=logits.dtype)
        w = pos_weights.to(logits.dtype) * valid.to(logits.dtype)
        denom = w.sum().clamp_min(1.0)
        loss = -(picked * w).sum() / denom
        return loss, logits


# --------------------------------------------------------------------------------------
# Real draft block (P1): DSparkAttention + DSparkBlock, faithful to vendored
# model_ref DSparkAttention:1032-1074 / DSparkBlock:1100-1144.
#
# TRAINING uses one PARALLEL teacher-forced causal forward over all block_size draft
# positions (doc §2.6 prereg): KV = concat(main window prefix, this draft block's own
# KV), RoPE starts at main_len, and each draft query i causally sees draft positions
# 0..i only. This is numerically equivalent to feeding gold shifted ids through the ref
# start_pos>0 sequential decode column-by-column; the inference-only autoregressive
# sample() loop / Markov / confidence heads are out of scope (dspark_markov_rank=0).


from v41f.attention import Attention  # noqa: E402
from v41f.block import Block  # noqa: E402
from v41f.hyperconn import HyperConn  # noqa: E402
from v41f.moe import MoE  # noqa: E402
from v41f.rope import apply_rotary_emb  # noqa: E402
from v41f.sparse_attn import sparse_attn  # noqa: E402


def dspark_causal_topk(main_len: int, block_size: int, batch: int, device):
    """Per-query draft topk for the parallel teacher-forced path.

    KV layout after concat is [0:main_len]=main window prefix,
    [main_len:main_len+block_size]=this draft block's KV. Query (draft position) i sees
    the ENTIRE main prefix plus draft positions 0..i (causal). Returns
    [batch, block_size, main_len + block_size] int indices (no empty slots here); the
    trailing draft slots past i are simply not emitted for that row.

    Rows have varying length (i+1 draft slots), which sparse_attn handles as a normal
    (unpadded) index set; we pad the ragged draft tail to block_size with -1 so the tensor
    is rectangular, matching sparse_attn's empty-slot convention.
    """
    device = torch.device(device)
    main_idx = torch.arange(main_len, device=device)
    draft_base = main_len
    rows = []
    for i in range(block_size):
        draft_idx = draft_base + torch.arange(i + 1, device=device)
        rows.append(torch.cat([main_idx, draft_idx]))
    padded = torch.full((block_size, main_len + block_size), -1, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        padded[i, : row.numel()] = row
    return padded.unsqueeze(0).expand(batch, -1, -1).contiguous()


class DSparkAttention(Attention):
    """Window-only draft attention over a main prefix plus this block's draft KV.

    Always compress_ratio=0 (draft blocks have no CSA2 compressor/indexer). Prefill
    start_pos=0 only seeds the main window cache in the reference; the TRAINING path here
    is the parallel teacher-forced forward given the precomputed main prefix, so it never
    touches a decode ring buffer.
    """

    def __init__(self, cfg, virtual_layer_id: int, max_batch_size: int = 4):
        # A draft attention is window-only with no compressor/indexer (ref asserts
        # compress_ratio == 0). The base Attention indexes cfg.compress_ratios[layer_id],
        # which does not exist for a virtual draft layer (id >= n_layers), so hand it a
        # cfg whose ratio table is extended with a trailing 0 for this virtual layer. The
        # backbone gate (layer_id < n_layers) then makes is_kv/index_source False on its
        # own, so no draft compressor/indexer is built.
        from dataclasses import replace

        n_extra = virtual_layer_id - len(cfg.compress_ratios) + 1
        draft_cfg = replace(
            cfg,
            compress_ratios=tuple(cfg.compress_ratios) + (0,) * max(n_extra, 1),
        )
        super().__init__(draft_cfg, virtual_layer_id, max_batch_size=max_batch_size)
        if self.compress_ratio != 0 or self.indexer is not None or self.compressor is not None:
            raise ValueError("DSpark draft attention must be window-only with no compressor/indexer")

    def forward(self, x, main_kv, main_len):
        """Parallel teacher-forced draft attention.

        x:       [b, block_size, dim] draft token hidden (pre RoPE inputs).
        main_kv: [b, main_len, head_dim] ALREADY-RoPE'd main-window KV prefix (MQA shared).
        main_len: number of prefix positions; RoPE for draft q/kv starts there.
        Returns [b, block_size, dim].
        """
        b, block_size, _ = x.size()
        freqs = self.freqs_cis[main_len : main_len + block_size]

        q, _ = self.qproj(x)
        apply_rotary_emb(q[..., -self.rd :], freqs)
        draft_kv = self.kvproj(x)
        apply_rotary_emb(draft_kv[..., -self.rd :], freqs)

        kv = torch.cat([main_kv, draft_kv], dim=1)
        idxs = dspark_causal_topk(main_len, block_size, b, x.device)
        o = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale)
        o = o.to(q.dtype)
        apply_rotary_emb(o[..., -self.rd :], freqs, inverse=True)
        return self.oproj(o)

    def seed_main_prefix(self, main_x, main_len=None):
        """RoPE the anchor-layer hidden into the main-window KV prefix the draft attends
        to. Ref DSparkAttention:1040-1042 (kv_norm/wkv/RoPE on main_x); CPU path drops the
        fp8 act_quant (identity on the bf16 oracle)."""
        b, seqlen, _ = main_x.size()
        n = main_len if main_len is not None else seqlen
        freqs = self.freqs_cis[:n]
        main_kv = self.kvproj(main_x)
        apply_rotary_emb(main_kv[..., -self.rd :], freqs)
        return main_kv


class DSparkBlock(Block):
    """One draft block (mtp.* checkpoint namespace), structurally a Block but:
    - attention_cls is the window-only DSparkAttention;
    - stage 0 owns main_proj/main_norm over the concat of target-layer attn inputs;
    - the FFN MoE is a NEW independent parameter set sized by get_moe_config (dspark count
      0 -> backbone counts), never shared with a backbone block;
    - Markov/confidence/head live only on the last draft layer when markov_rank>0 (omitted).

    stage_id = (n_layers + stage_index) - n_layers = stage_index, passed explicitly so the
    draft block is built WITHOUT being in a backbone layer table.
    """

    def __init__(self, cfg, stage_index: int, n_target_layers: int, max_batch_size: int = 4):
        # build as a virtual layer id n_layers+stage_index so Attention's ratio table /
        # backbone gating resolve to window-only, then replace attention + MoE below.
        nn.Module.__init__(self)
        self.layer_id = cfg.n_layers + stage_index
        self.stage_id = stage_index
        self.hc_mult = cfg.hc_mult
        self.attn = DSparkAttention(cfg, self.layer_id, max_batch_size=max_batch_size)

        n_routed = cfg.dspark_n_routed_experts or cfg.n_routed_experts
        n_activated = cfg.dspark_n_activated_experts or cfg.n_activated_experts
        self.ffn = MoE(
            cfg.dim,
            n_routed,
            n_activated,
            cfg.moe_inter_dim,
            route_scale=cfg.route_scale,
            swiglu_limit=cfg.swiglu_limit,
            gate_temp=cfg.gate_temp,
            norm_topk_prob=cfg.norm_topk_prob,
            score_func=cfg.score_func,
        )
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.hc = HyperConn(
            cfg.dim,
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=cfg.hc_sinkhorn_iters,
            eps=cfg.hc_eps,
            norm_eps=cfg.norm_eps,
        )
        self.block_size = cfg.dspark_block_size
        self.noise_token_id = cfg.dspark_noise_token_id
        if stage_index == 0:
            assert n_target_layers > 0, "DSpark needs target layers"
            self.main_proj = nn.Linear(cfg.dim * n_target_layers, cfg.dim, bias=False)
            self.main_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        # rank>0: last draft stage gets norm + LM/markov/confidence heads (later PR)
        if cfg.dspark_markov_rank > 0:
            raise NotImplementedError("Markov/confidence inference heads are a later stage")

    def project_main(self, main_hidden):
        """[b, t, dim*n_target] concat of target-layer attn inputs -> [b, t, dim].

        The concat is over the LAST dim (ref Transformer cats main_hiddens on dim=-1),
        per position; t is the anchor context length and is not flattened."""
        return self.main_norm(self.main_proj(main_hidden))

    def forward(self, x, pre_mix, main_kv, main_len):
        """One parallel teacher-forced draft Block over block_size positions.

        x:        [b, block_size, hc, dim] expanded draft stream.
        main_kv:  [b, main_len, head_dim] RoPE'd main-window KV prefix (seeded once from
                  the projected anchor by seed_main_prefix).
        main_len: prefix length; draft RoPE starts there.
        Returns (x, ffn_pre). Mirrors Block.forward with the parallel DSpark attention.
        """
        # attention sublayer collapses on the incoming pre_mix
        residual = x
        attn_pre, attn_post, attn_comb = self.hc.hc_mixes(
            x, self.hc.hc_attn_fn, self.hc.hc_attn_scale, self.hc.hc_attn_base
        )
        a = self.hc.hc_pre(x, pre_mix)
        a = self.attn_norm(a)
        a = self.attn(a, main_kv, main_len)
        x = self.hc.hc_post(a, residual, attn_post, attn_comb)
        # FFN collapses on this block's attention pre
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc.hc_mixes(
            x, self.hc.hc_ffn_fn, self.hc.hc_ffn_scale, self.hc.hc_ffn_base
        )
        f = self.hc.hc_pre(x, attn_pre)
        f = self.ffn_norm(f)
        f = self.ffn(f, None)
        x = self.hc.hc_post(f, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def forward_embed(self, main_hidden, anchor_ids, embed, identity_pre_mix):
        """Ref DSparkBlock.forward_embed:1129-1138. Builds the expanded draft stream, the
        projected anchor, and the seeded main-window KV prefix the draft attends to.

        main_hidden: [b, t, dim*n_target] concat of the target layers' attn INPUTS
            (read BEFORE each target block runs).
        anchor_ids:  [b] real token at the anchor position.
        embed: callable ids->[b,S,dim], tied to the shared token embedding in training.
        Returns (x [b,S,hc,dim], pre_mix, main_kv [b,t,head_dim], main_len t).
        """
        draft_ids = anchor_ids.new_full((anchor_ids.size(0), self.block_size), self.noise_token_id)
        draft_ids[:, 0] = anchor_ids
        x = embed(draft_ids)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = identity_pre_mix(x, self.hc_mult)
        main_anchor = self.project_main(main_hidden)  # [b,t,dim]
        main_kv = self.attn.seed_main_prefix(main_anchor)
        return x, pre_mix, main_kv, main_anchor.size(1)

    def forward_train_embed(self, main_hidden, draft_input_ids, embed, identity_pre_mix):
        """Parallel teacher-forced TRAINING entry (v41f-defined, ref has no training path).

        Unlike forward_embed's inference noise fill, training feeds GOLD shifted ids:
        draft_input_ids is [b, block_size] with position 0 = anchor token, position i =
        the real main token at anchor-1+i (the label one step left). The causal mask in
        DSparkAttention keeps each query at or before its own column, so this is the
        gold-fed sequential decode made parallel. Returns the same tuple as
        forward_embed."""
        x = embed(draft_input_ids)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = identity_pre_mix(x, self.hc_mult)
        main_anchor = self.project_main(main_hidden)
        main_kv = self.attn.seed_main_prefix(main_anchor)
        return x, pre_mix, main_kv, main_anchor.size(1)
