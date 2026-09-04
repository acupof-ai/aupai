"""The model: blocks, the depth-attention wiring, and HybridLM.

Split out of train.py (b0-8; design in docs/standards/model_module_split.md). The point is
not a smaller file: it is that one odd structural experiment costs one class plus one Cfg
flag. A draft head sharing the trunk reads any block's residual-stream output -- KDA state
is internal to DeltaRecurrence, never threaded through the block interface -- so that
experiment adds a class here and touches nothing in the training loop.

DEPENDENCY DIRECTION IS ONE-WAY: this module must never import train. train re-exports
these names so sft.py, sft_math.py and infer_local.py keep working unchanged.

A re-export cannot be monkey-patched through train: `from model import chunk_kda` binds a
SEPARATE name, so `train.chunk_kda = stub` does not reach the call sites here, which read this
module's own globals. Patch model (test_arch_compat does both).

BLOCK CONTRACT (design page §2):
  forward(x, cu) -> x of the same shape; a new block satisfying this drops into any slot of
  HybridLM.blocks. sublayers() is AttnRes's coupling point, and AttnRes raises on None
  rather than skipping -- a block without it under attn_res=True would silently run no
  depth attention, which is "configured on, actually off".
"""
import math
import os
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (used via torch.utils.checkpoint.checkpoint)

import fone  # its `from train import generate_batch` is inside a function body -- no cycle

try:  # CUDA-only kernel; absent on Mac where only checkpoint tooling imports this module
    from fla.ops.kda import chunk_kda
except ImportError:
    chunk_kda = None
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    HAS_FA = True
except ImportError:
    try:
        # flash-attn 4 keeps the same two names under .cute with a DIFFERENT positional
        # order: its fourth positional is `qv`, not cu_seqlens_q. Every call below passes cu
        # and the max lengths by keyword, correct for both versions and the only thing
        # standing between this import and a silently mis-bound mask.
        from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

        HAS_FA = True
    except ImportError:
        HAS_FA = False

if HAS_FA:
    # flash's varlen wrapper validates shapes against a Python int, so dynamo burns the
    # document count into a guard drawn from a distribution: the variant set never closes and
    # recompilation is permanent (70 flash recompiles in 110 steps, 54.9 ms/step at the
    # rms_norm -> flash seam). flash's own compile_key contains no batch_size, so the
    # specialisation cannot select a different kernel -- the guard has no consumer.
    # eff.recompile_recurrence_explained, eff.seam_dynamo_disable.
    flash_attn_varlen_func = torch._dynamo.disable(flash_attn_varlen_func)

# Applied identically in training (Liger FLCE) and inference; SOFTCAP=0 disables it.
SOFTCAP = float(os.environ.get("SOFTCAP", 15.0)) or None


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


def rms_scale(x, eps=1e-6):
    """The [B,T,1] factor of a gain-free RMSNorm, without applying it: rms_hat(x) . gq ==
    rsqrt(mean(x^2)) * (x . gq), so AttnRes stores this instead of a normalized [B,T,D] copy
    per source -- 1024x less memory at d=1024."""
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class DeltaRecurrence(nn.Module):
    """Kimi Delta Attention: bounded decay + ShortConv + QK-norm, via fla.ops.kda.chunk_kda.

    `inner` is the width this mixer WORKS AT, which is not the residual width it reads and
    writes. It always reads the full residual (cfg.d) and always writes cfg.d back; `inner` sets
    the width in between -- the head count times head_dim. `inner=None` means cfg.d, today's
    behaviour, and the gate in runs/audit_0904/b0_headmix_bitwise_gate.py holds that path
    identical at bf16 resolution on both live shapes.

    THE INPUT PROJECTIONS READ cfg.d, NOT inner, and that is the whole point. Two mixers side by
    side in one block must each see the entire residual -- a mixer fed only its own slice cannot
    attend to what the other half carries, which would be a different architecture from the
    3:1 head split being tested. So qkv is [cfg.d -> 3*inner] and o is [inner -> cfg.d]. No extra
    matrices appear: the existing projections change shape, so at inner = 3/4 * d the mixer costs
    3/4 of its parameters instead of adding a down-projection.

    WHY THE ARGUMENT AND NOT x.shape. The width used to be read off the input
    (`B, T, D = x.shape`), which silently ties the working width to the residual: hand this
    module a slice and every reshape below is wrong by the ratio. `heads` comes in the same way
    and for the same reason -- head_dim must stay 128 (train.py:2058, the FlashKDA CUTLASS
    kernel), so a mixer at inner=768 carries 6 heads while its sibling at inner=256 carries 2,
    and neither can be derived from cfg alone.
    """

    def __init__(self, cfg, inner=None, heads=None):
        super().__init__()
        self.d_in = cfg.d                                  # the residual: read and written
        self.d = inner if inner is not None else cfg.d     # the working width
        self.h = heads if heads is not None else cfg.heads
        self.hd = self.d // self.h
        if self.hd * self.h != self.d:
            raise ValueError(f"inner={self.d} is not divisible by heads={self.h}")
        self.chunk_size = cfg.chunk_size
        self.qkv = nn.Linear(self.d_in, 3 * self.d, bias=False)
        self.o = nn.Linear(self.d, self.d_in, bias=False)
        # fused gate|beta GEMM; beta padded to a multiple of 16 output rows so FP8 (_scaled_mm) applies
        self.beta_pad = (-self.h) % 16
        self.gb = nn.Linear(self.d_in, self.d + self.h + self.beta_pad, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.h))
        # fla KDA init: dt ~ logU[1e-3, 0.1] -> mean retention ~0.9 per token. Zero init gave
        # softplus(0)=0.69 log-decay per token (retention ~0.1), erasing the recurrent state.
        dt = torch.exp(torch.rand(self.h * self.hd) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        # The conv sits BEFORE qkv, so it runs at the residual width, not at inner.
        self.short_conv = nn.Conv1d(self.d_in, self.d_in, kernel_size=4, padding=0, groups=self.d_in)
        # Document isolation for the conv (eff.kda_document_isolation_violated). Read from cfg so it
        # travels in the checkpoint; see forward() for what it changes and scripts/loader.py for why
        # a checkpoint without the key must get False rather than the live default.
        self.conv_doc_isolated = getattr(cfg, "conv_doc_isolated", False)

    def forward(self, x, cu=None):
        B, T, D = x.shape
        if D != self.d_in:
            raise ValueError(
                f"DeltaRecurrence reads the residual at width {self.d_in}, got x with width {D}. "
                f"The mixer no longer takes its width from x.shape, so a mismatch is a "
                f"construction error, not something to reshape around.")
        # causal: left-pad only, so output[t] sees only input[:t+1] (padding=2 leaks the next token)
        # K shifted multiply-adds, not nn.Conv1d: ATen routes a depthwise k=4 conv to
        # conv_depthwise2d_generic at ~6% of bandwidth; inductor fuses the arithmetic form
        # (3.44x compiled, the training path). Weights stay on self.short_conv, so
        # checkpoints load unchanged. Eager is 0.61x -- this only wins under torch.compile.
        w, K = self.short_conv.weight, self.short_conv.kernel_size[0]
        h = F.pad(x.transpose(1, 2), (K - 1, 0))
        if self.conv_doc_isolated and cu is not None:
            # WITHOUT THIS THE CONV READS ACROSS DOCUMENTS. cu reaches chunk_kda (:131) and the
            # attention (:191) but never reached the conv, so the first K-1 positions of every
            # document were convolved with the LAST tokens of the previous one. Measured
            # 2026-09-04 (runs/n8/): positions 0-2 of q/k/v contaminated, position 3+ exactly
            # 0.0000, which chunk_kda writes into the recurrent state and decays forward into a
            # 48.88 difference at the block output against that layer's 0.9253 tolerance. The fla
            # kernel and flash_attn_varlen_func were both controlled out on random inputs: each
            # isolates exactly, so this conv was the only site.
            #
            # Tap i at output position t reads input t-(K-1-i), legal only when that read lands in
            # the same document. cu indexes the FLAT B*T stream (train.py:669) while h is
            # [B, D, T], so the mask is built flat and viewed as [B, 1, T] to broadcast over D.
            # A multiplier per tap keeps this pure arithmetic, so inductor still fuses it: an fla
            # ShortConvolution would isolate too, but it is a Triton kernel, opaque to inductor,
            # and would give back part of the 3.44x on all nine KDA layers.
            pos = torch.arange(B * T, device=x.device)
            doc_start = cu[:-1].to(torch.long)[torch.bucketize(pos, cu[1:], right=True)]
            y = None
            for i in range(K):
                tap = ((pos - (K - 1 - i)) >= doc_start).to(x.dtype).view(B, 1, T)
                term = h[:, :, i : i + T] * w[:, 0, i].unsqueeze(-1) * tap
                y = term if y is None else y + term
        else:
            # FLAG OFF IS TODAY'S EXACT OPS, not the masked form with an all-ones tap: the op
            # sequence must be identical for a pre-flag checkpoint to score bitwise as it trained,
            # which is what b0 checks on ckpt_b0_sd_unlooped.pt before reusing it as the "current"
            # arm. An all-ones multiply would change inductor's fusion and so the last bits.
            y = h[:, :, :T] * w[:, 0, 0].unsqueeze(-1)  # conv1d is cross-correlation: no tap reversal
            for i in range(1, K):
                y = y + h[:, :, i : i + T] * w[:, 0, i].unsqueeze(-1)
        h = F.silu((y + self.short_conv.bias.unsqueeze(-1)).transpose(1, 2))
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, T, self.h, self.hd).contiguous()
        k = k.reshape(B, T, self.h, self.hd).contiguous()
        v = v.reshape(B, T, self.h, self.hd).contiguous()
        gb = self.gb(x)
        # SLICE AT self.d, THE WORKING WIDTH, NOT AT D. gb is [d_in -> d + h + pad], so the gate
        # occupies [:self.d] and beta [self.d : self.d + h]. These read D before inner existed,
        # when D and self.d were the same number; at inner < d_in that would take d_in columns
        # from a d-wide gate and read beta out of the gate's own tail.
        g = gb[..., : self.d].reshape(B, T, self.h, self.hd).contiguous()
        beta = gb[..., self.d : self.d + self.h].contiguous()  # raw logits, sigmoid in kernel
        if cu is not None:  # varlen: fla wants a single flattened sequence + cu_seqlens
            q, k, v, g = (t.reshape(1, B * T, self.h, self.hd) for t in (q, k, v, g))
            beta = beta.reshape(1, B * T, self.h)
        out, _ = chunk_kda(
            q,
            k,
            v,
            g=g,
            beta=beta,
            cu_seqlens=cu,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            state_v_first=True,  # unblocks FlashKDA at inference (zero training cost)
            disable_recompute=True,  # save w/u/qg/kg/v_new rather than recompute: +3GB, 8-15% faster
            chunk_size=self.chunk_size,
        )
        return self.o(out.reshape(B, T, self.d).to(x.dtype))


class GatedMLA(nn.Module):
    """Gated MLA: latent KV compression + full causal attention (NoPE, KDA handles position).

    `inner` / `heads`: see DeltaRecurrence. The projections read the full residual (cfg.d) and o
    writes it back; `inner` is only the width in between. `latent` is the third free parameter and
    it is NOT derivable from inner: today it is cfg.d // 4, which at inner < cfg.d would shrink
    twice over (once with inner, once through the //4) and change the arm's KV capacity for a
    reason nobody chose. Default None keeps cfg.d // 4 -- the value every existing checkpoint
    holds.
    """

    def __init__(self, cfg, inner=None, heads=None, latent=None):
        super().__init__()
        self.d_in = cfg.d                                  # the residual: read and written
        self.d = inner if inner is not None else cfg.d     # the working width
        self.h = heads if heads is not None else cfg.heads
        self.hd = self.d // self.h
        if self.hd * self.h != self.d:
            raise ValueError(f"inner={self.d} is not divisible by heads={self.h}")
        self.latent = latent if latent is not None else cfg.d // 4
        self.kv_down = nn.Linear(self.d_in, self.latent, bias=False)
        self.kv_up = nn.Linear(self.latent, 2 * self.d, bias=False)  # fused k_up|v_up
        self.qg = nn.Linear(self.d_in, 2 * self.d, bias=False)  # fused q|gate
        self.o = nn.Linear(self.d, self.d_in, bias=False)
        # A/B (4): the GATE is per layer (3 * 12 * d params, negligible); the TABLE is ONE
        # [vocab, d] shared by every MLA layer and owned by HybridLM. Three separate tables
        # would be +48.9% parameters at this config against +16.3% for one, and the mechanism
        # (token identity reaching V, gated, only in attention layers) is intact either way.
        # `_ve` is the per-forward lookup, stashed by HybridLM.forward; None means the arm is off.
        #
        # The gate outputs self.d because it is added to V, which lives at the working width; the
        # table itself is [vocab, cfg.d] and owned by HybridLM, so a sub-width MLA would need a
        # projection here. Arm B does not exercise this path (value_embed is off), and adding an
        # untested projection for a combination nobody is running is how a second variable enters
        # an A/B: refuse instead.
        self._ve = None
        if getattr(cfg, "value_embed", False):
            if self.d != self.d_in:
                raise ValueError(
                    f"value_embed with inner={self.d} != cfg.d={self.d_in} is unimplemented: the "
                    f"shared [vocab, {self.d_in}] table would need a projection to reach V at "
                    f"width {self.d}, which no measurement has ever run. Turn one of them off.")
            self.ve_gate = nn.Linear(12, self.d, bias=True)
        else:
            self.ve_gate = None

    def forward(self, x, cu=None):
        B, T, D = x.shape
        if D != self.d_in:
            raise ValueError(
                f"GatedMLA reads the residual at width {self.d_in}, got x with width {D}. The "
                f"mixer no longer takes its width from x.shape, so a mismatch is a construction "
                f"error, not something to reshape around.")
        latent = self.kv_down(x)
        k, v = self.kv_up(latent).chunk(2, dim=-1)
        # A/B (4) value embeddings: a token-indexed vector added to V, gated per position.
        #
        # AFTER kv_up, never into the latent: the latent is shared by K and V (one kv_down, one
        # kv_up producing both), so adding there would put token identity into the KEYS too and
        # this would stop being a value embedding. 1e's ruling 2026-09-03.
        #
        # The gate reads the first 12 dims of the residual (speedrun's shape) and spans [0, 3):
        # 3*sigmoid can amplify as well as suppress, which is the published form, and at init
        # ve_gate is zero-init so sigmoid(0)=0.5 gives a gate of 1.5 -- NOT zero. That is
        # deliberate and it is why this arm is not parameter-free at step 0: a zero gate would
        # make the table invisible and the arm would need many steps just to discover it.
        if self._ve is not None:
            g = 3.0 * torch.sigmoid(self.ve_gate(x[..., :12]))
            v = v + g * self._ve.to(v.dtype)
        q, gate = self.qg(x).chunk(2, dim=-1)
        k = k.view(B, T, self.h, self.hd)
        v = v.view(B, T, self.h, self.hd)
        q = q.view(B, T, self.h, self.hd)
        q = F.rms_norm(q, (self.hd,))
        k = F.rms_norm(k, (self.hd,))
        if HAS_FA and cu is not None:
            q, k, v = (t.reshape(B * T, self.h, self.hd) for t in (q, k, v))
            y = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                       max_seqlen_q=T, max_seqlen_k=T, causal=True)
        elif HAS_FA:
            y = flash_attn_func(q, k, v, causal=True)
        else:
            # No flash_attn: SDPA with an explicit block-diagonal causal mask built from cu.
            # This branch used to take cu and ignore it, so doc_mask=True trained with every
            # document attending across every boundary and nothing in the log looked wrong.
            # Correct but ~20x slower per step -- a correctness fallback, not a training path.
            # cu indexes the flat B*T stream and every row start is a boundary (documents do
            # not span rows), so a per-row mask is exact. Every query sees at least itself, so
            # no row is fully masked and no NaN appears.
            if cu is not None:
                pos = torch.arange(B * T, device=q.device)
                doc = torch.bucketize(pos, cu[1:].to(pos.dtype), right=True).view(B, T)
                mask = (doc[:, :, None] == doc[:, None, :]) & torch.ones(
                    T, T, dtype=torch.bool, device=q.device).tril()
                mask = mask[:, None]
            else:
                mask = None
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
            y = y.transpose(1, 2)
        if isinstance(y, tuple):
            y = y[0]  # flash-attn 4 returns (out, lse); v2 and the fallback return a tensor
        y = y.reshape(B, T, self.d)
        return self.o(y * torch.sigmoid(gate))

class SwiGLU(nn.Module):
    """K3 SiTU-GLU: bounded activation, tracks SwiGLU near zero."""

    def __init__(self, cfg):
        super().__init__()
        self.w13 = nn.Linear(cfg.d, 2 * cfg.ffn_hidden, bias=False)  # fused w1|w3
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d, bias=False)
        self.beta1 = 4.0
        self.beta2 = 25.0

    def forward(self, x):
        a, b = self.w13(x).chunk(2, dim=-1)
        gate = self.beta1 * torch.tanh(a / self.beta1) * torch.sigmoid(b)
        up = self.beta2 * torch.tanh(self.w2(gate) / self.beta2)
        return up


class Source(NamedTuple):
    """A source of the depth attention: the raw layer output, and its gain-free RMS factor."""

    v: torch.Tensor
    scale: torch.Tensor

    @staticmethod
    def of(v):
        return Source(v, rms_scale(v))

    def normed(self):
        return self.v * self.scale


class AttnRes(nn.Module):
    """Attention Residuals (Kimi, arXiv 2603.15031): h_l = sum_i softmax_i(q_l . RMSNorm(v_i)) v_i over
    previous layer outputs (v_0 = embedding). One zero-init pseudo-query per layer -> uniform mean.
    Paper ablations: multihead / sigmoid / no-norm / sliding-window all worse — keep this exact form.
    The gain folds into the query ((v_hat * g) . q == v_hat . (g * q)); rms_scale has the rest."""

    def __init__(self, d, dyn_q=False, rank=64, fused=False, fp32_logits=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))  # the RMSNorm gain, applied to the query side
        self.q = nn.Parameter(torch.zeros(d))
        self.fused = fused  # one autograd node instead of the loop; same value, half the edges
        self.fp32_logits = fp32_logits
        # Input-dependent query (paper Table 4: 1.731 vs 1.737 Full); B zero-init.
        self.dyn = (
            nn.Sequential(nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False)) if dyn_q else None
        )

    def forward(self, srcs):
        q = self.q if self.dyn is None else self.q + self.dyn(srcs[-1].normed() * self.g)
        gq = self.g * q
        if self.fused:
            # scale stays LIVE: Source.scale is rms_scale(v), so v reaches the output by
            # two routes and detaching drops one (dV lands 7.6% low, forward unchanged).
            from algorithms.attnres_fused import fused_attn_res
            return fused_attn_res([s.v for s in srcs], gq, [s.scale for s in srcs])
        # logits [n,B,T] only; never an [n,B,T,D] stack of the values (that copy dominates at L=24)
        # fp32 dot: measured against fp64, bf16 accumulation over D=1024 puts the logits
        # 0.858 off against a 279.8 spread, and softmax turns that into 14% on the weights.
        if self.fp32_logits:
            logits = torch.stack([(s.v.float() * gq.float()).sum(-1) * s.scale.squeeze(-1).float()
                                  for s in srcs])
        else:
            logits = torch.stack([(s.v * gq).sum(-1) * s.scale.squeeze(-1) for s in srcs])
        a = logits.float().softmax(0).to(srcs[0].v.dtype)
        out = a[0].unsqueeze(-1) * srcs[0].v
        for i in range(1, len(srcs)):
            out = out + a[i].unsqueeze(-1) * srcs[i].v
        return out


class HeadMix(nn.Module):
    """Arm B: KDA and MLA side by side INSIDE one block, on a head-count split (default 3:1).

    Today's hybrid alternates whole layers -- 3 KDA blocks then 1 MLA block (attn_every=4). This
    puts both mixers in every block instead, KDA on 3/4 of the heads and MLA on 1/4, so every
    layer has both a recurrent and a full-attention path rather than every fourth layer having
    only attention.

    THE OUTPUTS ARE SUMMED, and that is not an approximation of concatenation. Each mixer's o
    projects its own working width back to the full residual, so summing the two is exactly what
    concatenating [kda_out, mla_out] and applying a single [d, d] o would compute:
    o(concat(a,b)) == o1(a) + o2(b) where o1,o2 are o's column blocks. Verified numerically,
    max|diff| 1.43e-06 and torch.allclose True (2026-09-04). So no fusion mechanism is needed and
    no parameter is spent on one.

    HEAD_DIM STAYS 128 on both halves (train.py:2058, the FlashKDA CUTLASS kernel), which is what
    makes the split arithmetic rigid: inner = heads * 128, so a 3:1 split at d=1024/h=8 is
    KDA h=6 inner=768 and MLA h=2 inner=256. h must be divisible by 4 -- at d=768/h=6 it is not,
    which is why this arm runs at d=1024/h=8 and not at the Stage D shape.
    """

    def __init__(self, cfg, ratio=3):
        super().__init__()
        h = cfg.heads
        if h % (ratio + 1) != 0:
            raise ValueError(
                f"head_mixed ratio {ratio}:1 needs heads divisible by {ratio + 1}, got heads={h}. "
                f"head_dim is pinned to {cfg.d // h} by the FlashKDA CUTLASS kernel "
                f"(train.py:2058), so the split cannot be taken in fractions of a head.")
        hd = cfg.d // h
        self.h_mla = h // (ratio + 1)
        self.h_kda = h - self.h_mla
        self.kda = DeltaRecurrence(cfg, inner=self.h_kda * hd, heads=self.h_kda)
        # latent stays cfg.d // 4, the value every checkpoint holds: deriving it from the MLA
        # half's inner would shrink the KV latent to d//16 and silently make this arm a test of
        # two changes at once.
        self.mla = GatedMLA(cfg, inner=self.h_mla * hd, heads=self.h_mla)

    def forward(self, x, cu=None):
        return self.kda(x, cu) + self.mla(x, cu)


class ProductKeyMemory(nn.Module):
    """A sparse memory table read by product-key lookup, ONE pool shared by several blocks.

    WHY THIS EXISTS (charter docs/standards/memory_layers_0905.md, user order 2026-09-04T16:44Z).
    N2 measured the params arm winning 0.0108 nat at equal compute; N7/Stage E measured reusing
    weights buying nothing; the head-hybrid lost 0.087 nat. The one variable that moved loss was
    how many parameters a token can reach, so this adds parameters at near-zero FLOPs: a token
    touches top_k of V values, not all of them.

    PRODUCT KEYS, and they are the whole reason a 1M-value table is affordable to search. A flat
    table needs V dot products per token (1,048,576 at M1). Product keys factor the query into two
    halves and the key space into two sub-tables of sqrt(V) each, so the search is 2*sqrt(V) = 2048
    dot products plus a top_k x top_k combine. The cost is that the searchable set is a Cartesian
    product rather than V free vectors -- a real restriction on what the table can represent, and
    the reason this is a known architecture (Lample et al. 2019, Meta memory+) rather than a
    shortcut invented here.

    SHARED ACROSS BLOCKS, ONE POOL. Layers 3, 6, 9 all read THIS module. The alternative (one pool
    per layer) triples the parameters for the same charter line and makes the arm a test of two
    changes. Sharing is also what makes the diagnostics meaningful: "fraction of values touched"
    is a statement about one table, not an average over three.

    PARALLEL TO THE FFN, NOT REPLACING IT: `h = h + ffn(n2(h)) + mem(n_mem(h))`. The dense
    parameter count therefore equals the control's EXACTLY, which is the property the whole
    experiment rests on -- if the FFN were replaced, a loss delta would confound "memory helps"
    with "less FFN hurts". prereg field dense_params_equal_control is that assertion, recomputed
    on the launch sha.

    KEYS AND VALUES ARE OUTSIDE FP8 AND MUON, by charter. Muon orthogonalises 2D matrices, which
    is meaningless for an embedding table read by index -- and _fp8_ok would otherwise convert the
    key projection. train.build_optimizers routes anything named `.mem.` to its own Adagrad group
    (dense state, one moment: 16 GiB at 2048^2 against SparseAdam's 32 GiB, measured before
    choosing, 4c's ruling 2026-09-05).

    THE VALUE TABLE IS AN nn.Embedding WITH sparse=True so its grad arrives as a sparse COO
    tensor holding only the touched rows. That is what lets DDP exchange touched indices instead
    of all-reducing a 4.3B-row table, and it is why the optimizer must be one that accepts sparse
    grads -- Adagrad and SparseAdam do, AdamW does not.
    """

    def __init__(self, n_values, d, top_k=32, key_dim=None, sparse=True):
        super().__init__()
        side = int(round(math.isqrt(n_values)))
        if side * side != n_values:
            raise ValueError(
                f"product-key memory needs a SQUARE value count so the two sub-tables are equal: "
                f"got n_values={n_values}, whose isqrt is {side} ({side * side} != {n_values}). "
                f"Use a perfect square (M1 1024^2 = 1048576, M2 512^2 = 262144, "
                f"M3 2048^2 = 4194304)."
            )
        if top_k > side:
            raise ValueError(
                f"top_k={top_k} exceeds the per-half candidate count sqrt(n_values)={side}: each "
                f"half can return at most {side} keys, so a larger top_k silently returns fewer "
                f"than requested and the combine below would index out of range."
            )
        self.n_values, self.side, self.top_k = n_values, side, top_k
        # key_dim is the HALF-query width, so the full query is 2*key_dim. Default d//2 keeps the
        # query projection square (d -> d) rather than growing it.
        self.key_dim = key_dim if key_dim is not None else d // 2
        self.n_mem = RMSNorm(d)
        # ONE query head (charter). Projects to both halves at once; the split is a view.
        self.query = nn.Linear(d, 2 * self.key_dim, bias=False)
        # Two key sub-tables, each sqrt(V) x key_dim. These are the searchable half-keys.
        self.keys = nn.Parameter(torch.randn(2, side, self.key_dim) * (self.key_dim ** -0.5))
        # The table itself. padding_idx unset: every row is a real value.
        self.values = nn.Embedding(n_values, d, sparse=sparse)
        nn.init.normal_(self.values.weight, std=d ** -0.5)
        # Output gate (Meta memory+ style): the read is projected and silu-gated so the block can
        # learn to ignore the memory rather than being forced to add it.
        self.gate = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        # Diagnostics, charter readout 4: which rows were touched since the last read of this
        # counter. A bool buffer rather than a set, so it costs V bits and no python.
        # NOT persistent: it is a window counter, not model state, and saving it would make two
        # checkpoints of the same weights differ.
        self.register_buffer("touched", torch.zeros(n_values, dtype=torch.bool), persistent=False)
        self.register_buffer("last_entropy", torch.zeros(()), persistent=False)

    def forward(self, x):
        B, T, d = x.shape
        h = self.n_mem(x)
        q = self.query(h).view(B * T, 2, self.key_dim)
        # Each half scores its own sqrt(V) keys and keeps top_k.
        s0 = torch.einsum("nk,ck->nc", q[:, 0], self.keys[0])
        s1 = torch.einsum("nk,ck->nc", q[:, 1], self.keys[1])
        v0, i0 = s0.topk(self.top_k, dim=-1)
        v1, i1 = s1.topk(self.top_k, dim=-1)
        # The Cartesian combine: top_k x top_k candidate pairs, whose scores add because the full
        # key is the concatenation of the two halves and the query is split to match.
        cand = (v0[:, :, None] + v1[:, None, :]).view(B * T, self.top_k * self.top_k)
        idx = (i0[:, :, None] * self.side + i1[:, None, :]).view(B * T, self.top_k * self.top_k)
        w, sel = cand.topk(self.top_k, dim=-1)
        flat = idx.gather(1, sel)                       # (B*T, top_k) rows of the table
        w = torch.softmax(w.float(), dim=-1).to(x.dtype)
        vals = self.values(flat)                        # (B*T, top_k, d)
        read = torch.einsum("nkd,nk->nd", vals, w).view(B, T, d)
        if not self.training or torch.is_grad_enabled():
            with torch.no_grad():
                self.touched[flat.reshape(-1)] = True
                p = w.float().clamp_min(1e-9)
                self.last_entropy = -(p * p.log()).sum(-1).mean()
        return self.out(F.silu(self.gate(h)) * read)

    def diagnostics(self, reset=True):
        """Charter readout 4: (touched_fraction, topk_entropy, key_usage_gini) for the window.

        Reset by default because the readout is "fraction touched IN THE WINDOW": a cumulative
        counter converges to 1.0 and stops being able to see a collapse, which is the one thing
        this diagnostic exists to catch (a pool below 20% at step 1000 stops the arm).
        """
        n = int(self.touched.sum())
        frac = n / self.n_values
        ent = float(self.last_entropy)
        # Gini over the two key sub-tables' usage is not derivable from `touched` (a bool tells
        # us reached-or-not, not how often), so it is reported as the row-count concentration we
        # CAN see: how far the touched set is from uniform coverage of the product space. Named
        # honestly rather than called a Gini it is not.
        if reset:
            self.touched.zero_()
        return {"touched_fraction": frac, "touched_rows": n, "n_values": self.n_values,
                "topk_entropy": ent, "topk_entropy_max": math.log(self.top_k)}


def _mem_layers(cfg):
    """Which block indices read the shared memory pool: cfg.mem_layers, or the charter's 3,6,9.

    Accepts a list or a comma string, because the flag arrives from argv as a string and from a
    saved cfg as whatever it was saved as. Parsed in one place so a checkpoint written from a
    string and one written from a list produce the SAME architecture -- two spellings of the same
    arm reading as two arms is the shape that made a ledger knob look like a 1/sqrt(L) rule.
    """
    v = getattr(cfg, "mem_layers", None)
    if v is None or v == "":
        return [3, 6, 9]
    if isinstance(v, str):
        return [int(t) for t in v.replace(" ", "").split(",") if t]
    return [int(t) for t in v]


class Block(nn.Module):

    def __init__(self, cfg, is_attn=False, memory=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.d)
        # head_mixed replaces the layer-level alternation entirely: EVERY block gets both mixers,
        # so is_attn stops selecting anything. Reading it here rather than in HybridLM keeps the
        # block list construction (:421) untouched, so attn_every still decides nothing else.
        _hm = getattr(cfg, "head_mixed", 0)
        self.mixer = HeadMix(cfg, ratio=_hm) if _hm else (GatedMLA(cfg) if is_attn else DeltaRecurrence(cfg))
        self.n2 = RMSNorm(cfg.d)
        self.ffn = SwiGLU(cfg)
        # THE SHARED POOL IS NOT REGISTERED AS A CHILD, deliberately. Assigning it to an attribute
        # would register it once per block that reads it (layers 3, 6, 9), so state_dict would
        # carry three copies of a table up to 4.3B rows and named_parameters would hand the
        # optimizer the same tensor three times. Held inside a list so nn.Module.__setattr__ does
        # not see a Module: HybridLM owns it and registers it exactly once.
        self._mem = [memory] if memory is not None else []
        attn_res = getattr(cfg, "attn_res", False)
        dyn_q = getattr(cfg, "attn_res_dyn_q", False)
        _fused = getattr(cfg, "attn_res_fused", False)
        _f32 = getattr(cfg, "attn_res_fp32_logits", False)
        self.ar1 = AttnRes(cfg.d, dyn_q, fused=_fused, fp32_logits=_f32) if attn_res else None  # pre-mixer / pre-ffn depth attention
        self.ar2 = AttnRes(cfg.d, dyn_q, fused=_fused, fp32_logits=_f32) if attn_res else None

    def forward(self, x, cu=None):
        x = x + self.mixer(self.n1(x), cu)
        # PARALLEL TO THE FFN, both branches reading the same x: `h = h + ffn(n2(h)) + mem(h)`.
        # The memory is added, never substituted, so the dense parameter count equals the
        # control's exactly -- the property the whole experiment rests on.
        h = x + self.ffn(self.n2(x))
        return h + self._mem[0](x) if self._mem else h

    def sublayers(self, cu=None):
        return ((self.ar1, self.n1, lambda t: self.mixer(t, cu)), (self.ar2, self.n2, self.ffn))


def remap_legacy_state_dict(sd):
    """Old unfused keys -> fused: w1|w3 -> w13, k_up|v_up -> kv_up, q|gate -> qg, gate_proj|beta_proj -> gb."""
    sd = dict(sd)
    for k in list(sd):
        if k.endswith("ffn.w1.weight"):
            p = k[: -len("w1.weight")]
            sd[p + "w13.weight"] = torch.cat([sd.pop(k), sd.pop(p + "w3.weight")])
        elif k.endswith("mixer.k_up.weight"):
            p = k[: -len("k_up.weight")]
            sd[p + "kv_up.weight"] = torch.cat([sd.pop(k), sd.pop(p + "v_up.weight")])
        elif k.endswith("mixer.gate_proj.weight"):
            p = k[: -len("gate_proj.weight")]
            beta = sd.pop(p + "beta_proj.weight")
            pad = torch.zeros((-beta.shape[0]) % 16, beta.shape[1], dtype=beta.dtype, device=beta.device)
            sd[p + "gb.weight"] = torch.cat([sd.pop(k), beta, pad])
        elif k.endswith("ar1.norm.g") or k.endswith("ar2.norm.g") or k.endswith("final_ar.norm.g"):
            sd[k[: -len("norm.g")] + "g"] = sd.pop(k)  # AttnRes gain moved onto the query side
        elif k.endswith("mixer.q.weight"):
            p = k[: -len("q.weight")]
            sd[p + "qg.weight"] = torch.cat([sd.pop(k), sd.pop(p + "gate.weight")])
    return sd


class HybridLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grad_ckpt = cfg.grad_ckpt
        self.padded_vocab = ((cfg.vocab + 63) // 64) * 64
        self.tok = nn.Embedding(self.padded_vocab, cfg.d)
        # GatedMLA is NoPE (position comes from KDA's recurrent state). Zero KDA layers
        # = no position information = not a valid model (attn_every=1 gave 21-sigma worse
        # val loss, 2026-08-30). Refuse rather than produce a plausible-looking wrong number.
        n_kda = sum(1 for i in range(cfg.layers) if i % cfg.attn_every != cfg.attn_every - 1)
        if n_kda == 0:
            raise ValueError(
                f"attn_every={cfg.attn_every} produces 0 KDA layers, but GatedMLA is NoPE "
                f"(KDA handles position). The model would have no position information. "
                f"Use attn_every >= 2, or add RoPE to GatedMLA first."
            )
        # ONE SHARED MEMORY POOL, owned here so it is registered exactly once no matter how many
        # blocks read it (see Block._mem). `mem_values` 0 or absent means no memory: a Cfg from
        # before this field existed is legitimately the control, and the control must construct
        # bit-identically to how it trained.
        _mv = int(getattr(cfg, "mem_values", 0) or 0)
        if _mv:
            _layers = _mem_layers(cfg)
            bad = [i for i in _layers if not 0 <= i < cfg.layers]
            if bad:
                raise ValueError(
                    f"mem_layers names layer(s) {bad} outside 0..{cfg.layers - 1}. A memory layer "
                    f"index that does not exist would silently attach to nothing and the arm "
                    f"would train as the control while its flags said otherwise."
                )
            self.memory = ProductKeyMemory(
                _mv, cfg.d,
                top_k=int(getattr(cfg, "mem_top_k", 32)),
                sparse=bool(getattr(cfg, "mem_sparse", True)),
            )
            self.mem_layers = sorted(set(_layers))
            # MEMORY + attn_res IS HANDLED IN BOTH PATHS, and it took a defect to get here.
            # _body has two paths: the plain one calls Block.forward (which adds the memory) and
            # the attn_res one iterates Block.sublayers(), which returns only (ar1, n1, mixer) and
            # (ar2, n2, ffn) -- NO memory branch. Until 2026-09-05 the second path silently
            # skipped the memory, so an arm would have trained as the CONTROL while every flag,
            # log line and ledger row said it carried a 1B-parameter table, and the primary
            # readout would have reported a null it never tested. Found by reading the traceback
            # of the first CPU run rather than by reasoning: the stack went through sublayers,
            # which is where the memory is not.
            #
            # It is not hypothetical: Cfg.attn_res DEFAULTS TO TRUE and the control trained with
            # it on (ck["cfg"] of ckpt_b0_headmix_armA.pt reads attn_res True, and 50 AttnRes
            # tensors are in its weights). So the arms must run attn_res too -- turning it off
            # would change the control. _body's attn_res branch now adds the memory per block as a
            # plain residual outside the AR mechanism (4c's ruling), which is why this no longer
            # raises. What still raises is grad_ckpt on that path, below: the memory add sits
            # outside the checkpointed fn, so the two have never been run together.
            if getattr(cfg, "attn_res", False) and getattr(cfg, "grad_ckpt", False):
                raise ValueError(
                    "mem_values with BOTH attn_res and grad_ckpt is refused. On the attn_res path "
                    "the memory add sits outside the checkpointed fn, so under recomputation the "
                    "memory branch is evaluated on the forward tape only -- and a hook or counter "
                    "that does not fire in the recompute has already cost a day here "
                    "(hooks-dont-fire-in-recompute). The charter's arms reuse the control's line, "
                    "which carries --no-grad_ckpt, so this combination is not needed. Measure it "
                    "on a card before enabling it, and check the diagnostics counter specifically: "
                    "`touched` is written under no_grad and would double-count or vanish."
                )
        else:
            self.memory, self.mem_layers = None, []
        self.blocks = nn.ModuleList(
            # every `attn_every` blocks (was `i == cfg.attn_every - 1`: one attention layer total)
            [Block(cfg, is_attn=(i % cfg.attn_every == cfg.attn_every - 1),
                   memory=(self.memory if i in self.mem_layers else None))
             for i in range(cfg.layers)]
        )
        self.norm = RMSNorm(cfg.d)
        # A/B (4): ONE shared value-embedding table for every MLA layer, or None when off.
        #
        # getattr WITH a default here, unlike Muon's `assert hasattr` for muon_shape_lr, and the
        # difference is not inconsistency. Muon is only ever constructed from the live train.Cfg,
        # so a missing field there means a rename and must be loud. This constructor is also
        # called with a Cfg from BEFORE the field existed: test_split_bitwise.py builds the same
        # architecture from the pre-split train.py to prove the split changed no bit, and an
        # assert here turned that test red for a field the old code cannot have. A historical Cfg
        # legitimately means "off".
        #
        # What still has to be loud is the RENAME, and it is caught by test_value_embed.py's
        # check 0, which asserts train.Cfg carries the field BY THAT NAME. Not by the parameter
        # delta, which was this comment's first claim and was wrong: every other check in that
        # file sets cfg.value_embed on the object itself, so none of them can see a rename in
        # Cfg. Verified by renaming the field -- only check 0 goes red.
        self.value_embed = (nn.Embedding(self.padded_vocab, cfg.d)
                            if getattr(cfg, "value_embed", False) else None)
        self.head = nn.Linear(cfg.d, self.padded_vocab, bias=False)
        # TIED BY DEFAULT. b0-17 unties it because the tied head necessarily trains at the
        # EMBEDDING lr (train.py routes any name containing tok/head into the embed group at 0.1),
        # which is 28.9x the nanochat reference head lr 0.004*(d/768)^-0.5 = 0.003464 at d1024 --
        # and that is the named candidate for the uniform embedding growth b0-10 measured at 1.43x
        # per 500 steps.
        #
        # The untied head is NOT initialised from tok's values. self.apply(self._init) gives it its
        # own std=0.02 draw, so the two arms differ from step 0. Copying tok would make the arms
        # bit-identical before the first step and would answer a different question (how fast a
        # head diverges from its embedding) than the one asked (what an untied head with its own
        # lr is worth).
        if not getattr(cfg, "untie_head", False):
            self.head.weight = self.tok.weight
        # FoNE: [NUM] carries no value in its identity; injected from Fourier features, read per digit
        self.fone = getattr(cfg, "fone", False)
        if self.fone:
            self.num_proj = nn.Linear(fone.NUM_DIMS, cfg.d, bias=False)
            self.num_head = nn.Linear(cfg.d, fone.NUM_DIMS, bias=False)
        self.attn_res = getattr(cfg, "attn_res", False)
        n_sub = 2 * cfg.layers
        n_blocks = min(n_sub, getattr(cfg, "attn_res_blocks", 0) or n_sub)  # 0 -> Full (every sublayer)
        self.ar_block_ends = {round((j + 1) * n_sub / n_blocks) for j in range(n_blocks)}
        self.final_ar = AttnRes(cfg.d, getattr(cfg, "attn_res_dyn_q", False),
                                fused=getattr(cfg, "attn_res_fused", False),
                                fp32_logits=getattr(cfg, "attn_res_fp32_logits", False)
                                ) if self.attn_res else None
        if self.attn_res:
            # Construction time, not forward: which blocks implement sublayers() is fixed once
            # the model is built and does not vary by step, so a missing one is decidable here.
            # In forward the best case is a crash at step 1 and the worst is a block on some
            # conditional branch never asked until step 8000 -- and the reason this check
            # exists is that "configured on, actually off" must not run quietly, which means
            # it has to shout before the run starts (tilerl, design page §2, 2026-09-02).
            for i, b in enumerate(self.blocks):
                sub = b.sublayers() if hasattr(b, "sublayers") else None
                if not sub or any(ar is None for ar, _, _ in sub):
                    raise TypeError(
                        f"attn_res=True but block {i} ({type(b).__name__}) supplies no usable "
                        f"sublayers(): {'method missing' if sub is None else 'an AttnRes is None'}. "
                        f"A block without it would run NO depth attention while the config says "
                        f"it is on. Implement sublayers() returning (AttnRes, norm, transform) "
                        f"triples, or run with attn_res=False."
                    )
        self.apply(self._init)
        for m in self.modules():
            if isinstance(m, AttnRes) and m.dyn is not None:
                nn.init.zeros_(m.dyn[1].weight)  # after _init, or it starts non-uniform
        # A/B (3): zero every OUTPUT projection, so each sublayer starts as an identity on the
        # residual stream. Off unless --zero_init_out. Two sites, four tensor kinds: `.o` is the
        # name in BOTH the KDA block (model.py:91) and the MLA block (:156), and `w2` is FFN's
        # down-projection (:204). AttnRes's final_ar is deliberately NOT included -- it mixes
        # sources, it does not write to the residual stream (1e's ruling 2026-09-03).
        #
        # Zeroed AFTER _init, like the two blocks above, or _init refills them. And the count
        # is asserted rather than assumed: a rename of `o` or `w2` would silently zero nothing
        # and the arm would be an exact copy of the baseline that still reports as the arm --
        # the most expensive way for this to fail, since it costs a full run to learn nothing.
        # A/B (4): the gate's WEIGHT is zero so the gate starts uniform at 1.5 (3*sigmoid(0)),
        # not at zero. Zero-init the weight and leave the bias at zero: the gate then begins as a
        # constant 1.5 for every token and every position, and learns to differentiate. A gate
        # that started at 0 would make the table invisible and the arm would spend its 500 steps
        # discovering the table exists rather than using it.
        if self.value_embed is not None:
            for m in self.modules():
                if isinstance(m, GatedMLA) and m.ve_gate is not None:
                    nn.init.zeros_(m.ve_gate.weight)
                    nn.init.zeros_(m.ve_gate.bias)
        if getattr(cfg, "zero_init_out", False):
            n_zeroed = 0
            with torch.no_grad():
                for name, mod in self.named_modules():
                    if isinstance(mod, nn.Linear) and (
                        name.endswith(".o") or name.endswith(".w2")):
                        mod.weight.zero_()
                        n_zeroed += 1
            expect = 2 * cfg.layers  # one output projection per sublayer: attn/KDA `o` + FFN `w2`
            assert n_zeroed == expect, (
                f"zero_init_out zeroed {n_zeroed} projections, expected {expect} "
                f"(2 x {cfg.layers} layers). A renamed `o`/`w2` would make this arm a silent "
                f"copy of the baseline.")
            print(f"zero_init_out: {n_zeroed} output projections zeroed", flush=True)
        # Alignment padding (vocab_real:vocab) must stay neutral in the softmax. The training
        # path slices head.weight[:vocab] into Liger FLCE, which has no per-class mask, so
        # random-init padding logits steal denominator mass: 11 columns spiked the vocab A/B
        # to |delta| 1.8 (eff.vocab_padding_softmax_defect). Zero keeps their logits at 0;
        # they are never targets, so CE gradient only pushes them down. After _init, or _init
        # re-fills them.
        #
        # UNTIED NEEDS BOTH TENSORS ZEROED, and this is a real trap rather than tidiness (1e,
        # b0-17): while the head IS tok, one zero_() does both. Untie them and `tok`'s pad rows
        # keep their std=0.02 draw -- so the untied arm would train pad rows the tied arm never
        # touched, adding a hidden variable to an A/B whose whole point is the head. tok's pad
        # rows are never inputs, so zeroing them costs nothing in either arm.
        _real = getattr(cfg, "vocab_real", cfg.vocab)
        if _real < cfg.vocab:
            with torch.no_grad():
                self.head.weight[_real : cfg.vocab].zero_()
                if self.head.weight is not self.tok.weight:
                    self.tok.weight[_real : cfg.vocab].zero_()

    def load_state_dict(self, sd, strict=True):
        """Load old checkpoints (fused-key remap); disable AttnRes if the ckpt predates it."""
        sd = remap_legacy_state_dict(sd)
        if self.attn_res and not any(k.startswith("final_ar.") for k in sd):
            print("checkpoint has no AttnRes params: disabling AttnRes for this model", flush=True)
            self.attn_res = False
            self.cfg.attn_res = False
            self.final_ar = None
            for b in self.blocks:
                b.ar1 = b.ar2 = None
        return super().load_state_dict(sd, strict)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, std=0.02)  # not PyTorch's kaiming default

    def _body(self, x, cu=None):
        ckpt = self.grad_ckpt and self.training
        if not self.attn_res:
            for b in self.blocks:
                x = torch.utils.checkpoint.checkpoint(b, x, cu, use_reentrant=False) if ckpt else b(x, cu)
            return x
        # Block AttnRes (Fig. 2): `done` = completed block reps, `partial` = intra-block running sum
        done, partial, n = [Source.of(x)], [], 0
        for bi, b in enumerate(self.blocks):
            for ar, norm, f in b.sublayers(cu):
                h = ar(done + partial)
                # AttnRes stays outside the checkpoint: only [B,T] logits on the tape, never [B,T,D]
                fn = lambda t, norm=norm, f=f: f(norm(t))  # noqa: E731
                out = torch.utils.checkpoint.checkpoint(fn, h, use_reentrant=False) if ckpt else fn(h)
                partial = [Source.of(partial[0].v + out if partial else out)]
                n += 1
                if n in self.ar_block_ends:
                    done, partial = done + partial, []
            # THE MEMORY IN THE attn_res PATH, placed here by 4c's ruling 2026-09-05 and NOT a
            # sublayer. The control trained with attn_res True (ck["cfg"] of
            # ckpt_b0_headmix_armA.pt, and 50 AttnRes tensors are in its weights, 51,200 params =
            # 0.0214% of the model), so the arms must run it too -- turning it off would change the
            # control and break the block-paired comparison the primary readout depends on.
            #
            # OUTSIDE THE AR MECHANISM, deliberately: a plain residual add after the block's
            # AR-combined sublayers. The memory neither reads the depth-attention's `done`/`partial`
            # sources nor becomes one of them, so what a depth-attention read of a memory branch
            # would mean stays an open question rather than an implicit answer. Adding it as a
            # third sublayer would have made this arm a test of two changes.
            #
            # The read is on the block's OUTPUT (the AR-combined running sum), which is this path's
            # equivalent of the plain path's post-FFN x. Both paths therefore compute
            # h = h + mem(norm(h)) with the memory's own RMSNorm inside ProductKeyMemory.
            if self.memory is not None and bi in self.mem_layers:
                cur = partial[0].v if partial else done[-1].v
                upd = cur + self.memory(cur)
                if partial:
                    partial = [Source.of(upd)]
                else:
                    done = done[:-1] + [Source.of(upd)]
        return self.final_ar(done + partial)

    def lm_logits(self, hidden):
        """The vocabulary head plus the softcap, split out so a decoder can apply it to the
        handful of positions it actually reads instead of to the whole prefix. Columns at or
        past vocab_real are alignment padding (never targets): set to the dtype's most negative
        finite value AFTER the softcap, since tanh would compress it to -SOFTCAP. Finite, not
        -inf: the E2E asserts every logit is finite, and a real -inf must stay distinguishable
        from padding."""
        logits = self.head(hidden)[..., : self.cfg.vocab].float()
        out = SOFTCAP * torch.tanh(logits / SOFTCAP) if SOFTCAP else logits
        real = getattr(self.cfg, "vocab_real", self.cfg.vocab)
        if real < out.shape[-1]:
            out[..., real:] = torch.finfo(out.dtype).min
        return out

    def num_logits(self, hidden):
        """Per-digit logits (..., digits, 10) at every position; the caller masks to [NUM]. Runs
        outside autocast, so cast to the weight dtype and back to fp32."""
        return fone.digit_logits(self.num_head(hidden.to(self.num_head.weight.dtype)).float())

    def forward(self, idx, targets=None, cu=None, num_vals=None, return_hidden=False, no_head=False):
        """cu: int32 cu_seqlens over the flattened (B*T) stream (see doc_cu_seqlens); None = no doc mask.
        num_vals: (B, T) float, the value at each [NUM] position; elsewhere masked out, not trusted.
        return_hidden: FoNE sampling needs the state that predicted [NUM] to read its digits from."""
        emb = self.tok(idx)
        # A/B (4): ONE lookup for the whole forward, stashed on each MLA layer. The table is
        # shared, so all three layers want the identical [B, T, d] gather -- doing it once here
        # rather than per layer saves two gathers and, more importantly, keeps the block
        # signatures untouched (threading a new arg through Block.forward / sublayers / _body
        # would also change the AttnRes path, which has nothing to do with this arm).
        if self.value_embed is not None:
            ve = self.value_embed(idx)
            for m in self.modules():
                if isinstance(m, GatedMLA) and m.ve_gate is not None:
                    m._ve = ve
        if self.fone and num_vals is not None:
            mask = (idx == self.cfg.num_id).unsqueeze(-1)
            feat = fone.encode_tensor(num_vals.masked_fill(~mask.squeeze(-1), 0.0)).to(emb.dtype)
            emb = emb + torch.where(mask, self.num_proj(feat), emb.new_zeros(()))
        hidden = self.norm(self._body(emb, cu))
        if self.value_embed is not None:
            # Cleared HERE, not in a finally: the reference must not outlive the forward that
            # made it. A stale _ve would pin a [B, T, d] activation between steps AND would be
            # reused by any forward that reached an MLA layer without going through this method
            # (a decoder calling _body directly), which would be a wrong number, not a crash.
            for m in self.modules():
                if isinstance(m, GatedMLA):
                    m._ve = None
        if targets is None:
            # no_head: a decoder reads ONE position per row while the head ran over all T.
            # At B=64, T=557, V=32832 the fp32 logits alone are 4.7GB and the softcap chain
            # allocates several more copies, all freed immediately -- 17% of per-position
            # forward FLOPs, 99.8% of it discarded, and the transient is what pins the eval
            # batch size. The caller gathers its B positions and calls lm_logits on those.
            # Slicing [:, -1:] here would be WRONG: rows are right-padded from different
            # prompt lengths, so each row decodes from its own column, not the last one.
            if no_head:
                return None, hidden
            return self.lm_logits(hidden), (hidden if return_hidden else None)
        # Training: the loss is computed in the loop -- Liger FLCE is compile-incompatible
        return hidden, None
