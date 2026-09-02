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
    """Kimi Delta Attention: bounded decay + ShortConv + QK-norm, via fla.ops.kda.chunk_kda."""

    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.heads, cfg.d // cfg.heads
        self.chunk_size = cfg.chunk_size
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)
        # fused gate|beta GEMM; beta padded to a multiple of 16 output rows so FP8 (_scaled_mm) applies
        self.beta_pad = (-cfg.heads) % 16
        self.gb = nn.Linear(cfg.d, cfg.d + cfg.heads + self.beta_pad, bias=False)
        self.A_log = nn.Parameter(torch.zeros(cfg.heads))
        # fla KDA init: dt ~ logU[1e-3, 0.1] -> mean retention ~0.9 per token. Zero init gave
        # softplus(0)=0.69 log-decay per token (retention ~0.1), erasing the recurrent state.
        dt = torch.exp(torch.rand(cfg.heads * self.hd) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.short_conv = nn.Conv1d(cfg.d, cfg.d, kernel_size=4, padding=0, groups=cfg.d)

    def forward(self, x, cu=None):
        B, T, D = x.shape
        # causal: left-pad only, so output[t] sees only input[:t+1] (padding=2 leaks the next token)
        # K shifted multiply-adds, not nn.Conv1d: ATen routes a depthwise k=4 conv to
        # conv_depthwise2d_generic at ~6% of bandwidth; inductor fuses the arithmetic form
        # (3.44x compiled, the training path). Weights stay on self.short_conv, so
        # checkpoints load unchanged. Eager is 0.61x -- this only wins under torch.compile.
        w, K = self.short_conv.weight, self.short_conv.kernel_size[0]
        h = F.pad(x.transpose(1, 2), (K - 1, 0))
        y = h[:, :, :T] * w[:, 0, 0].unsqueeze(-1)  # conv1d is cross-correlation: no tap reversal
        for i in range(1, K):
            y = y + h[:, :, i : i + T] * w[:, 0, i].unsqueeze(-1)
        h = F.silu((y + self.short_conv.bias.unsqueeze(-1)).transpose(1, 2))
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, T, self.h, self.hd).contiguous()
        k = k.reshape(B, T, self.h, self.hd).contiguous()
        v = v.reshape(B, T, self.h, self.hd).contiguous()
        gb = self.gb(x)
        g = gb[..., :D].reshape(B, T, self.h, self.hd).contiguous()
        beta = gb[..., D : D + self.h].contiguous()  # raw logits, sigmoid in kernel
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
        return self.o(out.reshape(B, T, D).to(x.dtype))


class GatedMLA(nn.Module):
    """Gated MLA: latent KV compression + full causal attention (NoPE, KDA handles position)."""

    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.heads, cfg.d // cfg.heads
        self.latent = cfg.d // 4
        self.kv_down = nn.Linear(cfg.d, self.latent, bias=False)
        self.kv_up = nn.Linear(self.latent, 2 * cfg.d, bias=False)  # fused k_up|v_up
        self.qg = nn.Linear(cfg.d, 2 * cfg.d, bias=False)  # fused q|gate
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)
        # A/B (4): the GATE is per layer (3 * 12 * d params, negligible); the TABLE is ONE
        # [vocab, d] shared by every MLA layer and owned by HybridLM. Three separate tables
        # would be +48.9% parameters at this config against +16.3% for one, and the mechanism
        # (token identity reaching V, gated, only in attention layers) is intact either way.
        # `_ve` is the per-forward lookup, stashed by HybridLM.forward; None means the arm is off.
        self._ve = None
        self.ve_gate = nn.Linear(12, cfg.d, bias=True) if getattr(cfg, "value_embed", False) else None

    def forward(self, x, cu=None):
        B, T, D = x.shape
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
        y = y.reshape(B, T, D)
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

    def __init__(self, d, dyn_q=False, rank=64):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))  # the RMSNorm gain, applied to the query side
        self.q = nn.Parameter(torch.zeros(d))
        # Input-dependent query (paper Table 4: 1.731 vs 1.737 Full); B zero-init.
        self.dyn = (
            nn.Sequential(nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False)) if dyn_q else None
        )

    def forward(self, srcs):
        q = self.q if self.dyn is None else self.q + self.dyn(srcs[-1].normed() * self.g)
        gq = self.g * q
        # logits [n,B,T] only; never an [n,B,T,D] stack of the values (that copy dominates at L=24)
        logits = torch.stack([(s.v * gq).sum(-1) * s.scale.squeeze(-1) for s in srcs])
        a = logits.float().softmax(0).to(srcs[0].v.dtype)
        out = a[0].unsqueeze(-1) * srcs[0].v
        for i in range(1, len(srcs)):
            out = out + a[i].unsqueeze(-1) * srcs[i].v
        return out


class Block(nn.Module):
    def __init__(self, cfg, is_attn=False):
        super().__init__()
        self.n1 = RMSNorm(cfg.d)
        self.mixer = GatedMLA(cfg) if is_attn else DeltaRecurrence(cfg)
        self.n2 = RMSNorm(cfg.d)
        self.ffn = SwiGLU(cfg)
        attn_res = getattr(cfg, "attn_res", False)
        dyn_q = getattr(cfg, "attn_res_dyn_q", False)
        self.ar1 = AttnRes(cfg.d, dyn_q) if attn_res else None  # pre-mixer / pre-ffn depth attention
        self.ar2 = AttnRes(cfg.d, dyn_q) if attn_res else None

    def forward(self, x, cu=None):
        x = x + self.mixer(self.n1(x), cu)
        return x + self.ffn(self.n2(x))

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
        self.blocks = nn.ModuleList(
            # every `attn_every` blocks (was `i == cfg.attn_every - 1`: one attention layer total)
            [Block(cfg, is_attn=(i % cfg.attn_every == cfg.attn_every - 1)) for i in range(cfg.layers)]
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
        self.final_ar = AttnRes(cfg.d, getattr(cfg, "attn_res_dyn_q", False)) if self.attn_res else None
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
        # they are never targets, so CE gradient only pushes them down. The tied embedding
        # rows are zeroed too -- their ids never appear as inputs. After _init, or _init
        # re-fills them.
        _real = getattr(cfg, "vocab_real", cfg.vocab)
        if _real < cfg.vocab:
            with torch.no_grad():
                self.head.weight[_real : cfg.vocab].zero_()

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
        for b in self.blocks:
            for ar, norm, f in b.sublayers(cu):
                h = ar(done + partial)
                # AttnRes stays outside the checkpoint: only [B,T] logits on the tape, never [B,T,D]
                fn = lambda t, norm=norm, f=f: f(norm(t))  # noqa: E731
                out = torch.utils.checkpoint.checkpoint(fn, h, use_reentrant=False) if ckpt else fn(h)
                partial = [Source.of(partial[0].v + out if partial else out)]
                n += 1
                if n in self.ar_block_ends:
                    done, partial = done + partial, []
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
