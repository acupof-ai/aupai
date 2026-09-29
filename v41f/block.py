"""One assembled v41f transformer Block: Hyper-Connections around Attention and MoE.

P1 assembly (docs/standards/v41_faithful_repro.md §2.1): the leaf modules from P0 are
wired exactly as vendored model_ref.Block (third_party/deepseek_v41_ref/model_ref.py.ref
:907-996). The residual stream is `hc_mult` parallel copies `x` [b,s,hc,d]. Each
sublayer sits between an hc_pre collapse and an hc_post expansion; a sublayer collapses
on the coefficient set the PREVIOUS sublayer produced -- attention on the pre_mix handed
in from the prior layer, the FFN on this block's own attention pre.

Prefill-only: start_pos is 0 and the cross-layer KV/index tensors travel in an explicit
SharedAttnState container (P0 convention), never a process global. v41f has no vision
stack, so image_mask is accepted for signature parity and ignored. Engram (n-gram) and
DSpark are later assembly stages and are absent here (engram_layout=None).
"""

import torch
from torch import nn

from .attention import Attention
from .hyperconn import HyperConn
from .moe import MoE
from .norm_gate import RMSNorm


class Block(nn.Module):
    """Attention + MoE between Hyper-Connection pre/post, matching ref Block.

    forward(x, start_pos, pre_mix, state) -> (x, ffn_pre, state):
      x        [b,s,hc_mult,dim] residual stream
      pre_mix  [b,s,hc_mult] fp32 collapse weights from the previous layer
      state    SharedAttnState for the cross-layer window/compressed/index KV
    returns the expanded stream and the FFN's pre for the NEXT block's attention.
    """

    def __init__(self, cfg, layer_id: int, max_batch_size: int = 4):
        super().__init__()
        self.layer_id = layer_id
        self.hc_mult = cfg.hc_mult
        self.attn = Attention(cfg, layer_id, max_batch_size=max_batch_size)
        self.ffn = MoE(
            cfg.dim,
            cfg.n_routed_experts,
            cfg.n_activated_experts,
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

    def hc_pre(self, x, pre_mix):
        return self.hc.hc_pre(x, pre_mix)

    def forward(self, x, start_pos, pre_mix, state):
        # ---- attention sublayer ----
        residual = x
        attn_pre, attn_post, attn_comb = self.hc.hc_mixes(
            x, self.hc.hc_attn_fn, self.hc.hc_attn_scale, self.hc.hc_attn_base)
        a = self.hc.hc_pre(x, pre_mix)
        a = self.attn_norm(a)
        a, state = self.attn(a, state)
        x = self.hc.hc_post(a, residual, attn_post, attn_comb)

        # ---- FFN sublayer collapses on the attention's own pre ----
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc.hc_mixes(
            x, self.hc.hc_ffn_fn, self.hc.hc_ffn_scale, self.hc.hc_ffn_base)
        f = self.hc.hc_pre(x, attn_pre)
        f = self.ffn_norm(f)
        f = self.ffn(f, None)
        x = self.hc.hc_post(f, residual, ffn_post, ffn_comb)
        # hand the FFN's pre to the NEXT block's attention (ref Block returns ffn_pre)
        return x, ffn_pre, state


def make_identity_pre_mix(x, hc_mult):
    """Initial one-hot mix (ref make_identity_pre_mix): every token collapses onto
    residual copy 0. fp32 regardless of the bf16 stream."""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix
