#!/usr/bin/env python3
"""Train a ~200M Chinese LLM with hybrid recurrent architecture.

Acceleration techniques (from karpathy/nanochat):
- bf16 autocast + TF32 matmuls
- FP8 weight storage (--fp8, Hopper): fp8 weights, fp32 master weights, bf16 compute
- Muon optimizer (Newton-Schulz orthogonalized updates) for 2D params
- AdamW for embeddings and 1D params, with separate LRs
- Flash Attention for sliding window (SDPA fallback)
- torch.compile(dynamic=False)
- DDP multi-GPU via torchrun --nproc_per_node=N
- Gradient accumulation
- Linear warmup + linear warmdown LR schedule
- Logit softcap + QK-norm (fused into fla kernel)
- PYTORCH_ALLOC_CONF=expandable_segments
- nanochat recipe: momentum ramp, WD decay, no gradient checkpointing (97GB H20)
- FP8 via torchao Float8Linear (compile-friendly); FP8_RECIPE=legacy for the hand-rolled path
- Fused GEMMs: SwiGLU w1|w3, MLA k_up|v_up and q|gate (old checkpoints remapped on load)
- Attention Residuals (--attn_res, arXiv 2603.15031): softmax over depth instead of residual sum
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import copy
import datetime
import glob
import json
import math
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # used by --grad_ckpt
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = os.path.dirname(os.path.abspath(__file__))


class RunLog:
    """Tee training prints to runs/<name>.log and plot curves at the end (scripts/plot_curves.py)."""

    def __init__(self, name):
        os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
        self.path = os.path.join(ROOT, "runs", f"{name}.log")
        self.f = open(self.path, "a", encoding="utf-8")

    def __call__(self, msg):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def plot(self):
        import subprocess
        import sys

        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "plot_curves.py"), self.path])


DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "ckpt.pt")
TOK_PATH = os.path.join(DATA, "tokenizer.json")
TOKEN_CACHE = "/data00/pretrain_1b_tokens.pt"  # NVMe for fast loading

try:  # CUDA-only kernels; absent on Mac where only checkpoint tooling imports this module
    from fla.ops.kda import chunk_kda
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:
    chunk_kda = LigerFusedLinearCrossEntropyLoss = None
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    HAS_FA = True
except ImportError:
    HAS_FA = False


# Logit softcap, applied identically in training (Liger FLCE) and inference. SOFTCAP=0 in the
# environment disables it, which is only there to bisect throughput regressions.
SOFTCAP = float(os.environ.get("SOFTCAP", 15.0)) or None


class Cfg:
    d = 1024
    heads = 8  # hd=128, required for FlashKDA CUTLASS kernel
    layers = 12
    attn_every = 4
    attn_window = 1024
    ffn_hidden = 3072
    vocab = 32772
    seq = 4096  # start 4K, curriculum to 10K; recurrent arch handles arbitrary length at inference
    batch = 72  # was 60; H20 97GB, FP8 compute allows larger batch
    accum = 1  # was 3; no_sync makes accumulation cheap
    epochs = 3  # 3 epochs on 2.26B tokens, FP8 for speed
    warmup = 20
    warmdown = 0.65
    final_lr_frac = 0.05
    clip = 1.0
    val_frac = 0.05
    seed = 42
    compile = True  # model body only; FLCE loss kept outside (Liger compile-incompatible)
    grad_ckpt = False  # was True; costs 25% wall-clock for ~15GB savings, not needed on 97GB H20
    # Attention Residuals (arXiv 2603.15031): softmax over depth replaces the residual sum. Off by default
    # so SFT/eval stay comparable. blocks=0 -> Full (every sublayer a source; ~+10% HBM traffic at L=24;
    # layer outputs are already kept for backward, the per-source norms are recomputed under compile);
    # N>0 -> Block AttnRes with exactly N blocks. dyn_q: low-rank input-dependent query.
    attn_res = False
    attn_res_blocks = 0
    attn_res_dyn_q = False
    attn_res_lr = 0.01  # AdamW lr for the zero-init pseudo-queries (wd=0)
    # Document boundaries: <eos> positions become cu_seqlens so KDA state and SWA attention reset per
    # document instead of leaking across the ~10 docs packed into each 4K row (train/infer mismatch).
    doc_mask = True
    mix = "data/mix.json"  # domain mix (weights / epoch caps / anneal); absent -> legacy flat corpus
    anneal_frac = 0.10  # last fraction of tokens uses each domain's "anneal" weight (MiniCPM-style)
    val_every = 500  # steps between fixed-subset validations (0 = epoch end only)
    val_batches = 20  # batches of the val split used for the periodic check
    val_batches_full = 100  # batches used for the epoch-end validation (fixed prefix, comparable)
    val_rows_max = 5000  # cap on val rows kept per mix domain (validation only ever reads a prefix)
    # Muon (matrix params) — nanochat recipe
    muon_lr = 0.01
    muon_momentum = 0.95
    muon_ns_steps = 5
    muon_wd = 0.10  # was 0.28; nanochat 1/width² law @ d=1024
    # AdamW (embedding + 1D params)
    embed_lr = 0.1  # was 0.05; nanochat batch-scaled
    embed_betas = (0.8, 0.995)
    embed_wd = 0.001
    scalar_lr = 0.15  # was 0.05; nanochat batch-scaled
    scalar_betas = (0.8, 0.95)
    scalar_wd = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


def rms_hat(x, eps=1e-6):
    """RMS normalization with no learned gain -- the gain lives on the AttnRes query instead, so this
    is one tensor per source rather than one per (consumer, source) pair."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class DeltaRecurrence(nn.Module):
    """Kimi Delta Attention (KDA): bounded decay + ShortConv + QK-norm.

    Uses fla.ops.kda.chunk_kda with fused gate (A_log + dt_bias + data-dependent
    gate input), auto-dispatches to FlashKDA CUTLASS kernel when installed.
    """

    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.heads, cfg.d // cfg.heads
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)
        # K3 KDA gating: data-dependent gate (per-head-per-dim) + beta
        # fused gate|beta GEMM; beta padded to a multiple of 16 output rows so FP8 (_scaled_mm) applies
        self.beta_pad = (-cfg.heads) % 16
        self.gb = nn.Linear(cfg.d, cfg.d + cfg.heads + self.beta_pad, bias=False)
        # Learned per-head decay and per-head-per-dim gate bias
        self.A_log = nn.Parameter(torch.zeros(cfg.heads))
        # fla KDA init: dt ~ logU[1e-3, 0.1], dt_bias = inv_softplus(dt) -> mean retention ~0.9 per token.
        # Zero init gave softplus(0)=0.69 log-decay per token (retention ~0.1), erasing the recurrent state.
        dt = torch.exp(torch.rand(cfg.heads * self.hd) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        # K3: causal short convolution for local pattern before qkv
        self.short_conv = nn.Conv1d(cfg.d, cfg.d, kernel_size=4, padding=0, groups=cfg.d)

    def forward(self, x, cu=None):
        B, T, D = x.shape
        # ShortConv + Swish (K3), causal: left-pad only so output[t] sees only input[:t+1]
        h = F.pad(x.transpose(1, 2), (self.short_conv.kernel_size[0] - 1, 0))
        h = F.silu(self.short_conv(h).transpose(1, 2))
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, T, self.h, self.hd).contiguous()
        k = k.reshape(B, T, self.h, self.hd).contiguous()
        v = v.reshape(B, T, self.h, self.hd).contiguous()
        # K3 KDA: gate computed in-kernel = lower_bound * sigmoid(exp(A_log)*(g+dt_bias))
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
            disable_recompute=True,  # save w/u/qg/kg/v_new instead of recomputing in backward (+3GB, 8-15% faster)
        )
        return self.o(out.reshape(B, T, D).to(x.dtype))


class SlidingWindowAttention(nn.Module):
    """K3 Gated MLA: latent KV compression + sliding window attention (NoPE, KDA handles position)."""

    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.heads, cfg.d // cfg.heads
        self.latent = cfg.d // 4  # KV compression ratio 4:1
        self.attn_window = cfg.attn_window
        # Down-project KV to latent, up-project to per-head K,V
        self.kv_down = nn.Linear(cfg.d, self.latent, bias=False)
        self.kv_up = nn.Linear(self.latent, 2 * cfg.d, bias=False)  # fused k_up|v_up
        self.qg = nn.Linear(cfg.d, 2 * cfg.d, bias=False)  # fused q|gate (full-rank output gate, K3)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)

    def forward(self, x, cu=None):
        B, T, D = x.shape
        latent = self.kv_down(x)  # (B, T, latent)
        k, v = self.kv_up(latent).chunk(2, dim=-1)
        q, gate = self.qg(x).chunk(2, dim=-1)
        k = k.view(B, T, self.h, self.hd)
        v = v.view(B, T, self.h, self.hd)
        q = q.view(B, T, self.h, self.hd)
        q = F.rms_norm(q, (self.hd,))
        k = F.rms_norm(k, (self.hd,))
        if HAS_FA and cu is not None:
            q, k, v = (t.reshape(B * T, self.h, self.hd) for t in (q, k, v))
            y = flash_attn_varlen_func(
                q, k, v, cu, cu, T, T, causal=True, window_size=(self.attn_window - 1, 0)
            )
        elif HAS_FA:
            # flash_attn_func wants (B, T, h, hd); sliding window = (left, right)
            y = flash_attn_func(q, k, v, causal=True, window_size=(self.attn_window - 1, 0))
        else:  # CPU fallback: causal only, no document mask (ponytail: block-diag mask if ever needed)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2)
        y = y.reshape(B, T, D)
        # Gated output
        return self.o(y * torch.sigmoid(gate))


# --- FP8 compute: e4m3 for both forward and backward (e5m2 backward was unstable without grad_ckpt) ---
_FP8_MAX_E4M3 = 448.0  # forward: weights + activations; backward: grad_output (4-bit mantissa, ~6% error)


class FP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        x_scale = (x2d.detach().abs().max().clamp(min=1e-12) / _FP8_MAX_E4M3).float()
        w_scale = (weight.detach().abs().max().clamp(min=1e-12) / _FP8_MAX_E4M3).float()
        x_fp8 = (x2d / x_scale).to(torch.float8_e4m3fn)
        w_fp8 = (weight.contiguous() / w_scale).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_fp8,
            w_fp8.t(),
            scale_a=x_scale,
            scale_b=w_scale,
            out_dtype=torch.bfloat16,
        )
        if bias is not None:
            out = out + bias
        # Cache fp8 tensors + scales: backward reuses them (5 quants -> 3; saved
        # activations drop from bf16 to fp8, ~halving per-linear saved memory)
        ctx.save_for_backward(x_fp8, w_fp8, bias, x_scale, w_scale)
        ctx.orig_shape = x.shape
        return out.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x_fp8, w_fp8, bias, x_scale, w_scale = ctx.saved_tensors
        go2d = grad_out.reshape(-1, grad_out.shape[-1])
        go_scale = (go2d.detach().abs().max().clamp(min=1e-12) / _FP8_MAX_E4M3).float()
        go_fp8 = (go2d / go_scale).to(torch.float8_e4m3fn)
        # w_fp8 is (out, in) contiguous; need (out, in) column-major = .t() of (in, out) contiguous
        w_t = w_fp8.t().contiguous()  # (in, out) contiguous
        grad_x = torch._scaled_mm(
            go_fp8,
            w_t.t(),
            scale_a=go_scale,
            scale_b=w_scale,
            out_dtype=torch.bfloat16,
        ).reshape(ctx.orig_shape)
        x_t = x_fp8.t().contiguous()  # [D, M] fp8; .t() below is the column-major mat2
        grad_w = torch._scaled_mm(
            go_fp8.t().contiguous(),
            x_t.t(),
            scale_a=go_scale,
            scale_b=x_scale,
            out_dtype=torch.bfloat16,
        )
        grad_b = go2d.sum(0) if bias is not None else None
        return grad_x, grad_w, grad_b


class FP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear with FP8 forward + bf16 backward."""

    def __init__(self, linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data)
        self.bias = nn.Parameter(linear.bias.data) if linear.bias is not None else None
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    # Inductor's min-cut partitioner recomputes FP8LinearFunction's saved fp8 tensors in
    # the backward graph, re-dividing already-scaled values -> NaN grads at step 1 when
    # grad_ckpt is off (scripts/nan_probe.py, 2026-08-26). Keep it out of the compiled graph.
    @torch._dynamo.disable
    def forward(self, x):
        return FP8LinearFunction.apply(x, self.weight, self.bias)


def _fp8_ok(mod, name):
    return name != "head" and mod.weight.shape[0] % 16 == 0 and mod.weight.shape[1] % 16 == 0


def convert_to_fp8_compute(model):
    """FP8 linears. Prefers torchao Float8Linear (compile-friendly: no graph break per linear, fused casts);
    falls back to the hand-rolled FP8Linear (dynamo-disabled -> ~200 graph breaks) if torchao is missing.
    FP8_RECIPE=tensorwise|rowwise selects the torchao scaling recipe; FP8_RECIPE=legacy forces the old path."""
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError:
        print("torchao not found: using legacy FP8Linear (slow path, graph breaks)", flush=True)
        return _convert_to_fp8_legacy(model)
    recipe = os.environ.get("FP8_RECIPE", "tensorwise")
    if recipe == "legacy":
        return _convert_to_fp8_legacy(model)
    cfg = Float8LinearConfig.from_recipe_name(recipe)
    convert_to_float8_training(
        model, config=cfg, module_filter_fn=lambda m, fqn: _fp8_ok(m, fqn.rsplit(".", 1)[-1])
    )
    return model


def _convert_to_fp8_legacy(model):
    """Replace nn.Linear with FP8Linear (skip tied LM head and tiny layers not divisible by 16)."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and _fp8_ok(module, name):
            setattr(model, name, FP8Linear(module))  # torch._scaled_mm needs dims % 16 == 0
        else:
            _convert_to_fp8_legacy(module)
    return model


class SwiGLU(nn.Module):
    """K3 SiTU-GLU: bounded activation, tracks SwiGLU near zero."""

    def __init__(self, cfg):
        super().__init__()
        self.w13 = nn.Linear(cfg.d, 2 * cfg.ffn_hidden, bias=False)  # fused w1|w3: one GEMM instead of two
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d, bias=False)
        self.beta1 = 4.0
        self.beta2 = 25.0

    def forward(self, x):
        # SiTU-GLU: β1*tanh(W_g x/β1) * σ(W_gate x) * β2*tanh(W_u x/β2)
        a, b = self.w13(x).chunk(2, dim=-1)
        gate = self.beta1 * torch.tanh(a / self.beta1) * torch.sigmoid(b)
        up = self.beta2 * torch.tanh(self.w2(gate) / self.beta2)
        return up


class AttnRes(nn.Module):
    """Attention Residuals (Kimi, arXiv 2603.15031): h_l = sum_i softmax_i(q_l . RMSNorm(v_i)) v_i over
    previous layer outputs (v_0 = embedding). One zero-init pseudo-query per layer -> starts as uniform mean.
    Paper ablations: multihead / sigmoid / no-norm / sliding-window all worse — keep this exact form.

    The RMSNorm gain is folded into the query, which is an exact rewrite, not an approximation:
    (v_hat * g) . q == v_hat . (g * q). That leaves v_hat parameter-free, so it is the SAME tensor for
    every consumer of that source and is computed once per source instead of once per (consumer,
    source) pair. At L=12 that is 25 normalizations instead of 325 for Full, and the backward tape
    holds 25 shared [B,T,D] tensors instead of 325 per-pair products -- 6.7GB rather than 87GB at
    batch 32, which is what makes Full affordable at all."""

    def __init__(self, d, dyn_q=False, rank=64):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))  # the RMSNorm gain, applied to the query side
        self.q = nn.Parameter(torch.zeros(d))
        # Input-dependent query (paper Table 4: 1.731 vs 1.737 Full): q = w + B(A norm(latest source)),
        # low-rank d->rank->d, B zero-init so training still starts from the uniform mean.
        self.dyn = (
            nn.Sequential(nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False)) if dyn_q else None
        )

    def forward(self, srcs, hats):
        """srcs: raw sources (mixed by the softmax weights). hats: the same sources under a
        parameter-free RMS normalization, shared across every AttnRes in the model."""
        q = self.q if self.dyn is None else self.q + self.dyn(hats[-1] * self.g)
        gq = self.g * q
        # logits [n,B,T] only; no torch.stack of the [n,B,T,D] values (that copy dominates at L=24)
        logits = torch.stack([(vh * gq).sum(-1) for vh in hats])
        a = logits.float().softmax(0).to(srcs[0].dtype)
        out = a[0].unsqueeze(-1) * srcs[0]
        for i in range(1, len(srcs)):
            out = out + a[i].unsqueeze(-1) * srcs[i]
        return out


class Block(nn.Module):
    def __init__(self, cfg, is_attn=False):
        super().__init__()
        self.n1 = RMSNorm(cfg.d)
        self.mixer = SlidingWindowAttention(cfg) if is_attn else DeltaRecurrence(cfg)
        self.n2 = RMSNorm(cfg.d)
        self.ffn = SwiGLU(cfg)
        attn_res = getattr(cfg, "attn_res", False)
        dyn_q = getattr(cfg, "attn_res_dyn_q", False)
        self.ar1 = AttnRes(cfg.d, dyn_q) if attn_res else None  # pre-mixer depth attention
        self.ar2 = AttnRes(cfg.d, dyn_q) if attn_res else None  # pre-ffn depth attention

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
            pad = torch.zeros((-beta.shape[0]) % 16, beta.shape[1], dtype=beta.dtype)
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
        self.grad_ckpt = cfg.grad_ckpt  # set via --grad_ckpt
        self.padded_vocab = ((cfg.vocab + 63) // 64) * 64
        self.tok = nn.Embedding(self.padded_vocab, cfg.d)
        self.blocks = nn.ModuleList(
            # Periodic placement: an attention layer every `attn_every` blocks
            # (was `i == cfg.attn_every - 1`, which created only a single attention layer)
            [Block(cfg, is_attn=(i % cfg.attn_every == cfg.attn_every - 1)) for i in range(cfg.layers)]
        )
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, self.padded_vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.attn_res = getattr(cfg, "attn_res", False)
        n_sub = 2 * cfg.layers
        n_blocks = min(n_sub, getattr(cfg, "attn_res_blocks", 0) or n_sub)  # 0 -> Full (every sublayer)
        self.ar_block_ends = {round((j + 1) * n_sub / n_blocks) for j in range(n_blocks)}  # exactly N blocks
        self.final_ar = AttnRes(cfg.d, getattr(cfg, "attn_res_dyn_q", False)) if self.attn_res else None
        self.apply(self._init)
        for m in self.modules():
            if isinstance(m, AttnRes) and m.dyn is not None:
                nn.init.zeros_(m.dyn[1].weight)  # after _init: start as plain zero pseudo-query

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

    def _body(self, x, cu=None):
        ckpt = self.grad_ckpt and self.training
        if not self.attn_res:
            for b in self.blocks:
                # Gradient checkpointing: recompute block activations in backward, trading compute for memory
                x = torch.utils.checkpoint.checkpoint(b, x, cu, use_reentrant=False) if ckpt else b(x, cu)
            return x
        # Block AttnRes (Fig. 2 of the paper): `blocks` = completed block reps (embedding first),
        # `partial` = intra-block running sum. Block size 1 sublayer == Full AttnRes.
        blocks, hats, partial, partial_hat, n = [x], [rms_hat(x)], None, None, 0
        for b in self.blocks:
            for ar, norm, f in b.sublayers(cu):
                srcs = blocks if partial is None else blocks + [partial]
                hs = hats if partial_hat is None else hats + [partial_hat]
                h = ar(srcs, hs)
                # The AttnRes call sits outside the checkpoint, so its per-source products stay on
                # the tape: one [B,T,D] per (consumer, source) pair, which at L=12 is 325 pairs for
                # Full (196GB/GPU at batch 72), 85 for blocks=4, 61 for blocks=2. Folding ar() into
                # the checkpointed function would leave only the block reps alive and make Full
                # affordable; not done here because it changes what gets recomputed in backward and
                # this run is not the place to validate that.
                fn = lambda t, norm=norm, f=f: f(norm(t))  # noqa: E731
                out = torch.utils.checkpoint.checkpoint(fn, h, use_reentrant=False) if ckpt else fn(h)
                partial = out if partial is None else partial + out
                partial_hat = rms_hat(partial)
                n += 1
                if n in self.ar_block_ends:
                    blocks.append(partial)
                    hats.append(partial_hat)
                    partial = partial_hat = None
        return self.final_ar(
            blocks + ([partial] if partial is not None else []),
            hats + ([partial_hat] if partial_hat is not None else []),
        )

    def forward(self, idx, targets=None, cu=None):
        """cu: int32 cu_seqlens over the flattened (B*T) stream (see doc_cu_seqlens); None = no doc mask."""
        hidden = self.norm(self._body(self.tok(idx), cu))
        if targets is None:
            # Inference: compute logits with softcap
            logits = self.head(hidden)[..., : self.cfg.vocab].float()
            if SOFTCAP:
                logits = SOFTCAP * torch.tanh(logits / SOFTCAP)  # same cap as training
            return logits, None
        # Training: return hidden states; loss computed eagerly in the loop
        # (Liger FLCE is compile-incompatible, must stay outside torch.compile)
        return hidden, None


# --- Muon optimizer (from karpathy/nanochat, simplified per-param) ---

POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class Muon(torch.optim.Optimizer):
    """Muon: Nesterov momentum + Polar Express orthogonalization + cautious weight decay.
    Stacked by shape for fewer kernel launches + torch.compile."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.28):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._compiled = {}  # (shape, ns_steps, tall, device) -> compiled function
        self._scalar_tensors = {}  # device -> (lr_t, mom_t, wd_t) 0-D CUDA tensors

    def _get_scalar_tensors(self, device):
        if device not in self._scalar_tensors:
            self._scalar_tensors[device] = (
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )
        return self._scalar_tensors[device]

    def _get_compiled(self, shape, ns_steps, tall, device):
        """Get or create compiled update function for a given shape."""
        key = (shape, ns_steps, tall, device)
        if key not in self._compiled:
            coeffs = POLAR_EXPRESS[:ns_steps]
            a_coeffs = [c[0] for c in coeffs]
            b_coeffs = [c[1] for c in coeffs]
            c_coeffs = [c[2] for c in coeffs]

            def muon_update(
                grads,
                weights,
                momentums,
                lr,
                momentum,
                wd,
                a_c=a_coeffs,
                b_c=b_coeffs,
                c_c=c_coeffs,
                is_tall=tall,
            ):
                momentums.lerp_(grads, 1 - momentum)
                g = grads.lerp(momentums, momentum)
                X = g.bfloat16()
                X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
                for i in range(len(a_c)):
                    if is_tall:
                        A = X.mT @ X
                        B = torch.baddbmm(b_c[i] * A, A, A, beta=1.0, alpha=c_c[i])
                        X = torch.baddbmm(a_c[i] * X, X, B, beta=1.0, alpha=1.0)
                    else:
                        A = X @ X.mT
                        B = torch.baddbmm(b_c[i] * A, A, A, beta=1.0, alpha=c_c[i])
                        X = torch.baddbmm(a_c[i] * X, B, X, beta=1.0, alpha=1.0)
                mask = (grads * weights) >= 0
                weights.sub_(lr * X.to(weights.dtype) + lr * wd * weights * mask)
                return weights, momentums

            # 0-D tensor hparams are NOT value-specialized -> momentum ramp / WD decay
            # no longer trigger recompiles (the old float args recompiled every step)
            self._compiled[key] = torch.compile(muon_update, dynamic=False)
        return self._compiled[key]

    @torch.no_grad()
    def step(self):
        # Group params by shape for stacked computation
        shape_groups = {}
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                shape = tuple(p.shape)
                if shape not in shape_groups:
                    shape_groups[shape] = {
                        "params": [],
                        "grads": [],
                        "mbs": [],
                        "lr": lr,
                        "momentum": momentum,
                        "ns_steps": ns_steps,
                        "wd": wd,
                    }
                sg = shape_groups[shape]
                sg["params"].append(p)
                sg["grads"].append(p.grad)
                st = self.state[p]
                if "mb" not in st:
                    st["mb"] = torch.zeros_like(p.grad)
                sg["mbs"].append(st["mb"])

        for shape, sg in shape_groups.items():
            n = len(sg["params"])
            tall = shape[-2] > shape[-1] if len(shape) >= 2 else True
            # 0-D CUDA tensor hparams: not value-specialized by Dynamo, so momentum
            # ramp / WD decay don't trigger recompiles every step
            device = sg["params"][0].device
            lr_t, mom_t, wd_t = self._get_scalar_tensors(device)
            lr_t.fill_(sg["lr"])
            mom_t.fill_(sg["momentum"])
            wd_t.fill_(sg["wd"])
            if n == 1:
                # Single param: avoid stacking overhead
                p, g, mb = sg["params"][0], sg["grads"][0], sg["mbs"][0]
                fn = self._get_compiled(shape, sg["ns_steps"], tall, device)
                W = p.unsqueeze(0)
                G = g.unsqueeze(0)
                M = mb.unsqueeze(0)
                W, M = fn(G, W, M, lr_t, mom_t, wd_t)
                p.data.copy_(W[0])
                mb.copy_(M[0])
            else:
                # Stack same-shape params for batched update
                W = torch.stack(sg["params"])
                G = torch.stack(sg["grads"])
                M = torch.stack(sg["mbs"])
                fn = self._get_compiled(shape, sg["ns_steps"], tall, device)
                W, M = fn(G, W, M, lr_t, mom_t, wd_t)
                for i, p in enumerate(sg["params"]):
                    p.data.copy_(W[i])
                    sg["mbs"][i].copy_(M[i])


def doc_cu_seqlens(idx, eos_id):
    """cu_seqlens for a (B, T) batch: every row start and every position after an <eos> opens a new
    document; returned over the flattened B*T stream as int32 [n_docs + 1]. Length varies per batch,
    so it is marked dynamic for torch.compile."""
    B, T = idx.shape
    flat = idx.reshape(-1)
    starts = torch.nonzero(flat == eos_id).squeeze(1) + 1
    starts = torch.cat([torch.arange(0, B * T, T, device=idx.device), starts])
    starts = starts[starts < B * T]
    cu = torch.cat([starts.unique(), torch.tensor([B * T], device=idx.device)]).to(torch.int32)
    torch._dynamo.mark_dynamic(cu, 0)
    return cu


def validate(model, raw_model, Xva, Yva, batch, device, amp_dtype, eos_id=None, max_batches=None):
    """Mean FLCE loss over the (fixed, disjoint) validation split. All DDP ranks call it in lockstep."""
    model.eval()
    flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=SOFTCAP)
    weight = raw_model.head.weight[: raw_model.cfg.vocab]
    vl = []
    with torch.no_grad():
        for j in range(0, len(Xva), batch):
            if max_batches is not None and len(vl) >= max_batches:
                break
            xva = Xva[j : j + batch].to(device)
            yva = Yva[j : j + batch].to(device)
            cu = doc_cu_seqlens(xva, eos_id) if eos_id is not None else None
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda")):
                hidden, _ = model(xva, yva, cu)
            D = hidden.shape[-1]
            vl.append(flce(weight, hidden.to(weight.dtype).reshape(-1, D), yva.reshape(-1)).item())
    model.train()
    return sum(vl) / len(vl)


def build_optimizers(model, cfg, lr_scale=1.0):
    """Muon for 2D matrices; AdamW for embeddings, for 1D norm gains/biases, and (low lr, wd=0) for the
    short-conv kernels (3D, were mis-routed to the 15x scalar lr) and AttnRes pseudo-queries."""
    muon, embed, scalar, arq = [], [], [], []
    for n, p in model.named_parameters():
        if "tok" in n or "head" in n:
            embed.append(p)
        elif p.ndim == 2:
            muon.append(p)
        elif p.ndim == 3 or n.endswith("ar1.q") or n.endswith("ar2.q") or n.endswith("final_ar.q"):
            arq.append(p)
        else:
            scalar.append(p)
    opts = [
        Muon(
            muon,
            lr=cfg.muon_lr * lr_scale,
            momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps,
            weight_decay=cfg.muon_wd,
        ),
        torch.optim.AdamW(
            embed, lr=cfg.embed_lr * lr_scale, betas=cfg.embed_betas, weight_decay=cfg.embed_wd, fused=True
        ),
        torch.optim.AdamW(
            scalar,
            lr=cfg.scalar_lr * lr_scale,
            betas=cfg.scalar_betas,
            weight_decay=cfg.scalar_wd,
            fused=True,
        ),
    ]
    if arq:
        opts.append(
            torch.optim.AdamW(
                arq,
                lr=getattr(cfg, "attn_res_lr", 0.01) * lr_scale,
                betas=cfg.scalar_betas,
                weight_decay=0.0,
                fused=True,
            )
        )
    for opt in opts:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]
            g["initial_wd"] = g["weight_decay"]
    return opts


def set_schedule(optimizers, step, total, cfg):
    """LR warmup/warmdown for every group; nanochat momentum ramp and WD decay-to-zero for Muon only
    (the old loop also overwrote the embedding group's wd 0.001 with muon_wd)."""
    m = lr_mult(step, total, cfg)
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * m
            if isinstance(opt, Muon):
                g["momentum"] = 0.85 + 0.10 * min(1.0, step / 150)
                g["weight_decay"] = g["initial_wd"] * max(0.0, 1.0 - step / total)


def opt_snapshot(optimizers):
    """Real CPU copies of optimizer state (state_dict() values are dicts, so cloning at the top level
    used to alias the live CUDA moments)."""
    out = []
    for opt in optimizers:
        sd = opt.state_dict()
        out.append(
            {
                "state": {
                    i: {k: (v.cpu().clone() if torch.is_tensor(v) else v) for k, v in st.items()}
                    for i, st in sd["state"].items()
                },
                "param_groups": copy.deepcopy(sd["param_groups"]),
            }
        )
    return out


def ddp_even_len(n, batch, ddp):
    """Rows every rank can iterate: min over ranks of n//batch*batch, so all ranks take the same number
    of steps (a strided shard leaves ranks off by one row -> different lr per rank + NCCL hang)."""
    if not ddp:
        return n // batch * batch
    t = torch.tensor([n // batch], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item()) * batch


# --- Data ---


def load_texts():
    texts = []
    for name in ("core.txt", "framework.md", "method.txt"):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            texts.append(open(p, encoding="utf-8").read())
    for p in sorted(glob.glob(os.path.join(DATA, "corpus", "*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                texts.append(json.loads(line)["content"])
    for p in sorted(glob.glob(os.path.join(DATA, "corpus", "primary", "*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                texts.append(json.loads(line)["content"])
    mix = os.path.join(DATA, "mix", "mixed.jsonl")
    if os.path.exists(mix):
        for line in open(mix, encoding="utf-8"):
            line = line.strip()
            if line:
                texts.append(json.loads(line)["content"])
    for name in ("pretrain_full.jsonl",):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line:
                    texts.append(json.loads(line)["content"])
    return texts


def build_tokenizer(texts):
    if os.path.exists(TOK_PATH):
        tok = Tokenizer.from_file(TOK_PATH)
        assert tok.get_vocab_size() == Cfg.vocab, (
            f"tokenizer vocab {tok.get_vocab_size()} != Cfg.vocab {Cfg.vocab}"
        )
        return tok
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=Cfg.vocab, special_tokens=["<unk>", "<eos>"])
    tok.train_from_iterator(texts, trainer)
    tok.save(TOK_PATH)
    return tok


def encode(texts, tok, chunk=50_000, log=None):
    """Documents -> one <eos>-separated int32 stream.

    Chunked, and one C-level copy per document rather than a per-token python loop. Measured on
    20K web documents (2026-08-26): the tokenizer itself runs at 6.7M tok/s, but appending through
    array("i").extend(e.ids) dropped the pipeline to 1.4M tok/s -- 107 minutes for the 9.1B-token
    web domain. np.asarray per document plus encode_batch_fast (which skips the offsets, word_ids
    and masks that nothing here reads) gives 3.3M tok/s, 46 minutes. Chunking keeps the Encoding
    objects and the intermediate arrays bounded."""
    eos = tok.token_to_id("<eos>")
    batch_fn = getattr(tok, "encode_batch_fast", tok.encode_batch)
    parts = []
    for i in range(0, len(texts), chunk):
        enc = batch_fn(texts[i : i + chunk])
        parts.append(np.concatenate([np.asarray(e.ids + [eos], dtype=np.int32) for e in enc]))
        if log and (i // chunk) % 20 == 0:
            log(
                f"  encode {min(i + chunk, len(texts))}/{len(texts)} docs, "
                f"{sum(len(p) for p in parts) / 1e6:.0f}M tokens"
            )
    return torch.from_numpy(np.concatenate(parts))  # int32: vocab 32772 fits, halves bandwidth


def _domain_seqs(domain, tok, is_main, ddp):
    """Tokenize data/corpus/<domain>/*.jsonl once (rank 0), cache next to TOKEN_CACHE, return [N, seq+1]."""
    cache = os.path.join(os.path.dirname(TOKEN_CACHE), f"tokens_{domain}.pt")
    if is_main and not os.path.exists(cache):
        texts = []
        for p in sorted(glob.glob(os.path.join(DATA, "corpus", domain, "*.jsonl"))):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    texts.append(json.loads(line)["content"])
        assert texts, f"mix domain {domain}: no data/corpus/{domain}/*.jsonl"
        random.Random(Cfg.seed).shuffle(texts)
        print(f"mix: tokenizing {domain} ({len(texts)} docs) -> {cache}", flush=True)
        data = encode(texts, tok, log=lambda m: print(m, flush=True))
        del texts
        torch.save(data, cache)
        print(f"mix: {domain} cached {len(data) / 1e6:.0f}M tokens", flush=True)
        del data
    if ddp:
        dist.barrier()
    data = torch.load(cache, map_location="cpu", weights_only=True)  # int32; .long() once, per rank
    n = len(data) // (Cfg.seq + 1)
    return data[: n * (Cfg.seq + 1)].view(-1, Cfg.seq + 1)


def build_mix(cfg_path, tok, is_main, ddp, rank=0, world=1):
    """Domain mix -> (this rank's train rows in schedule order, val rows). mix.json:
    {"total_tokens": 11.5e9, "domains": {"web": {"weight": .83, "epochs": 2, "anneal": .42}, ...}}
    weight = share of the main phase; anneal = share of the last Cfg.anneal_frac tokens (default = weight);
    epochs = max repeats of that domain's data (the schedule is capped, never the filter thresholds).

    The schedule is built as an index plan -- (domain, row) pairs, ~22MB -- and only this rank's
    1/world slice of it is turned back into token rows. Materializing the whole schedule on every
    rank the way the obvious version does costs ~2.3TB of host RAM at 11.5B tokens x 8 ranks, which
    is more than the box has; it would have died after the 40 minutes spent tokenizing. Rows are
    pre-shuffled per phase and consumed sequentially, so main -> anneal order is exact."""
    mix = json.load(open(cfg_path, encoding="utf-8"))
    rows = mix["total_tokens"] / Cfg.seq
    phases = [(1 - Cfg.anneal_frac, "weight"), (Cfg.anneal_frac, "anneal")]
    g = torch.Generator().manual_seed(Cfg.seed)
    names = list(mix["domains"])
    pools, val, used = {}, [], {}
    for name in names:
        seqs = _domain_seqs(name, tok, is_main, ddp)
        # Capped: validation reads at most Cfg.val_batches_full batches, so a 5% split of a 1.9M-row
        # domain would keep 95K rows alive to look at 4.8K of them.
        n_val = min(max(1, int(len(seqs) * Cfg.val_frac)), Cfg.val_rows_max)
        val.append(seqs[:n_val])
        pools[name] = seqs[n_val:]
        used[name] = 0
    plan = []
    for frac, key in phases:
        parts = []
        for di, name in enumerate(names):
            d = mix["domains"][name]
            want = int(rows * frac * d.get(key, d["weight"]))
            pool = pools[name]
            cap = int(len(pool) * d.get("epochs", 1)) - used[name]
            if want > cap:
                if is_main:
                    print(
                        f"mix: {name} {key} wants {want} rows, epoch cap leaves {cap} -> capped", flush=True
                    )
                want = max(0, cap)
            if want:
                idx = torch.arange(used[name], used[name] + want) % len(pool)
                parts.append(torch.stack([torch.full_like(idx, di), idx]))
            used[name] += want
        if parts:
            ph = torch.cat(parts, dim=1)
            plan.append(ph[:, torch.randperm(ph.shape[1], generator=g)])
    plan = torch.cat(plan, dim=1)
    if is_main:
        for name in names:
            print(
                f"mix: {name} {used[name]} rows = {used[name] / max(len(pools[name]), 1):.2f} epochs",
                flush=True,
            )
        print(
            f"mix: {plan.shape[1]} rows = {plan.shape[1] * Cfg.seq / 1e9:.2f}B tokens scheduled", flush=True
        )
    # Truncate to a multiple of world so every rank takes the same number of steps: a strided shard
    # that leaves one rank a row short gives that rank a different lr and then hangs the all-reduce.
    n = (plan.shape[1] // world) * world
    mine = plan[:, :n][:, rank::world]
    out = torch.empty((mine.shape[1], Cfg.seq + 1), dtype=torch.int32)
    for di, name in enumerate(names):
        m = mine[0] == di
        if m.any():
            out[m] = pools[name][mine[1][m]]
    del pools
    vcat = torch.cat(val)
    return out, vcat[torch.randperm(len(vcat), generator=torch.Generator().manual_seed(Cfg.seed))]


# --- LR schedule: linear warmup, constant, cosine warmdown ---


def lr_mult(step, total, cfg):
    if step < cfg.warmup:
        return (step + 1) / cfg.warmup
    wd_steps = max(1, int(cfg.warmdown * total))
    wd_start = total - wd_steps
    if step < wd_start:
        return 1.0
    progress = min(1.0, (step - wd_start) / wd_steps)  # 0.0 → 1.0, clamped past total (resume)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))  # 1.0 → 0.0
    return cfg.final_lr_frac + (1 - cfg.final_lr_frac) * cosine


# --- DDP ---


def setup_ddp():
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    # 2h, not NCCL's default 10min: rank 0 does the per-domain tokenization alone while the other
    # ranks sit on the barrier in _domain_seqs, and the web domain takes ~45 minutes on its own.
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return True, rank, world, local


# --- Train ---


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain HybridLM; any --flag below overrides Cfg.<flag>")
    for name, help_ in {
        "seq": "sequence length",
        "batch": "batch size per GPU",
        "accum": "gradient accumulation steps",
        "vocab": "vocab size (e.g. 32000 for a 32K tokenizer)",
        "attn_res_blocks": "Block AttnRes with N blocks (0 = Full)",
        "val_every": "steps between fixed-subset validations (0 = epoch end only)",
        "val_batches": "val batches per periodic check",
        "warmup": "linear warmup steps (scale with total steps; 20 is a rounding error at 20K steps)",
    }.items():
        parser.add_argument(f"--{name}", type=int, default=None, help=f"{help_} (default: Cfg.{name})")
    for name, help_ in {
        "grad_ckpt": "gradient checkpointing (recompute sublayers in backward)",
        "attn_res": "Attention Residuals (arXiv 2603.15031)",
        "attn_res_dyn_q": "AttnRes input-dependent pseudo-query",
    }.items():
        parser.add_argument(f"--{name}", action="store_true", help=help_)
    parser.add_argument(
        "--fp8", action="store_true", help="FP8 linears (torchao; FP8_RECIPE=legacy for old path)"
    )
    parser.add_argument("--no_doc_mask", action="store_true", help="let KDA state / attention cross <eos>")
    parser.add_argument(
        "--mix", type=str, default=None, help='domain mix json (default Cfg.mix; "" = flat corpus)'
    )
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="stop after N optimizer steps (ablations)"
    )
    parser.add_argument("--name", type=str, default="pretrain", help="runs/<name>.log, ckpt_<name>.pt")
    args = parser.parse_args()
    for k, v in vars(args).items():
        if hasattr(Cfg, k) and v:  # int override or store_true flag; unset/False keeps the Cfg default
            setattr(Cfg, k, v)
    if args.no_doc_mask:
        Cfg.doc_mask = False

    torch.manual_seed(Cfg.seed)
    torch.set_float32_matmul_precision("high")
    ddp, rank, world, local = setup_ddp()
    device = f"cuda:{local}" if ddp else ("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = not ddp or rank == 0
    runlog = RunLog(args.name) if is_main else print
    if args.mix is not None:
        Cfg.mix = args.mix
    ckpt_path = CKPT if args.name == "pretrain" else os.path.join(ROOT, f"ckpt_{args.name}.pt")
    amp = device.startswith("cuda")

    mix_path = os.path.join(ROOT, Cfg.mix) if Cfg.mix else None
    use_mix = bool(mix_path and os.path.exists(mix_path))
    if use_mix:
        assert os.path.exists(TOK_PATH), "mix mode needs a trained data/tokenizer.json"
        tok = build_tokenizer([])
        eos_id = tok.token_to_id("<eos>")
        seqs, vseqs = build_mix(mix_path, tok, is_main, ddp, rank, world)
        seqs, vseqs = seqs.long(), vseqs.long()  # int32 on disk and in the pools; long for embedding
        Xtr, Ytr, Xva, Yva = seqs[:, :-1], seqs[:, 1:], vseqs[:, :-1], vseqs[:, 1:]
        data, X = seqs, seqs  # for the params print below
        Cfg.epochs = 1  # repeats are encoded in the schedule
    else:
        texts = load_texts()
        random.seed(Cfg.seed)
        random.shuffle(texts)  # document-level shuffle so sources are well mixed
        if ddp and rank == 0 and not os.path.exists(TOK_PATH):
            build_tokenizer(texts)  # rank 0 trains and saves
        if ddp:
            dist.barrier()  # all ranks wait for tokenizer to exist
        tok = build_tokenizer(texts)
        eos_id = tok.token_to_id("<eos>")
        # Rank-0-only tokenization + NVMe cache (avoids 8x redundant work)
        if is_main and not os.path.exists(TOKEN_CACHE):
            data = encode(texts, tok)
            torch.save(data, TOKEN_CACHE)
            print(f"Saved token cache ({len(data) / 1e6:.0f}M tokens) to {TOKEN_CACHE}", flush=True)
            data = data.long()  # int32 -> long for embedding lookup
        if ddp:
            dist.barrier()  # all ranks wait for rank 0
        if not is_main or (is_main and os.path.exists(TOKEN_CACHE) and "data" not in dir()):
            data = torch.load(TOKEN_CACHE, map_location="cpu", weights_only=True)
            data = data.long()  # int32 -> long for embedding lookup
            if is_main:
                print(f"Loaded cached tokens from {TOKEN_CACHE} ({len(data) / 1e6:.0f}M)", flush=True)
        n_seq = len(data) // (Cfg.seq + 1)
        data = data[: n_seq * (Cfg.seq + 1)]
        seqs = data.view(-1, Cfg.seq + 1)
        X, Y = seqs[:, :-1], seqs[:, 1:]
        n_val = max(1, int(len(X) * Cfg.val_frac))
        Xtr, Ytr, Xva, Yva = X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]

    if ddp and not use_mix:  # the mix path already handed this rank its own slice, in schedule order
        n_even = ddp_even_len(len(Xtr[rank::world]), Cfg.batch, ddp)
        Xtr, Ytr = Xtr[rank::world][:n_even], Ytr[rank::world][:n_even]  # strided: same phase per rank
    Xtr, Ytr = Xtr.contiguous().pin_memory(), Ytr.contiguous().pin_memory()  # nanochat: async H2D
    # Xtr[idx] allocates an unpinned temp, which makes .to(non_blocking=True) synchronous. Stage through two
    # pinned buffers; each is reused only after the event recorded behind its previous H2D copy completes.
    pin = [
        (
            torch.empty((Cfg.batch, Cfg.seq), dtype=Xtr.dtype).pin_memory(),
            torch.empty((Cfg.batch, Cfg.seq), dtype=Ytr.dtype).pin_memory(),
            torch.cuda.Event() if amp else None,
        )
        for _ in range(2)
    ]
    raw_model = HybridLM(Cfg).to(device)
    resume_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ck["model"])
        # parse step from filename like ckpt.pt.step2000
        import re

        m = re.search(r"step(\d+)", args.resume)
        if m:
            resume_step = int(m.group(1))
        resume_step = ck.get("step", resume_step)
        if is_main:
            print(f"Resumed from {args.resume} (step {resume_step})", flush=True)
    fp8 = args.fp8 and amp
    amp_dtype = torch.bfloat16
    if fp8:
        raw_model = raw_model.to(torch.bfloat16)
        convert_to_fp8_compute(raw_model)
        if is_main:
            print("FP8 compute enabled", flush=True)
    if is_main:
        n_params = sum(p.numel() for p in raw_model.parameters())
        # dense peak per GPU for MFU; override with PEAK_TFLOPS (H20: 296 FP8 / 148 bf16)
        peak_tflops = float(os.environ.get("PEAK_TFLOPS", 296 if fp8 else 148))
        print(
            f"params {n_params / 1e6:.1f}M | tokens {len(data)} | seqs {len(X)} | "
            f"device {device} | world {world} | fa {HAS_FA} | fp8 {fp8}"
        )
        # param-count assert removed: architecture now scales well beyond the original ~23M target (e.g. 200M)

    optimizers = build_optimizers(raw_model, Cfg)
    if args.resume and "opt" in ck:
        for opt, sd in zip(optimizers, ck["opt"], strict=True):
            opt.load_state_dict(sd)  # Muon momentum + Adam moments continue instead of restarting from 0

    model = raw_model
    if ddp:
        model = DDP(
            model, device_ids=[local], bucket_cap_mb=100, gradient_as_bucket_view=True, static_graph=True
        )
    if Cfg.compile and amp:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 256
        if os.environ.get("COMPILE_SUPPRESS_ERRORS", "0") == "1":
            torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, dynamic=False)

    # Initial good state (before any training, so NaN at step 1 can recover)
    good_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
    good_opt = [None] * len(optimizers)  # CPU copies of optimizer state_dicts
    total_steps = Cfg.epochs * (len(Xtr) // (Cfg.batch * Cfg.accum))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)  # LR schedule completes within the short run
    step = resume_step
    n_skip = 0  # consecutive skipped optimizer steps (non-finite gradients)
    GOOD_SAVE_INTERVAL = 200
    for ep in range(Cfg.epochs):
        model.train()
        perm = torch.arange(len(Xtr)) if use_mix else torch.randperm(len(Xtr))  # mix: schedule order
        i0 = step * Cfg.batch * Cfg.accum if use_mix else 0  # resume continues where the schedule stopped
        t0 = time.time()
        last = 0.0
        t_log = time.time()
        for i in range(i0, len(Xtr) - Cfg.batch + 1, Cfg.batch):
            idx = perm[i : i + Cfg.batch]
            xb_pin, yb_pin, ev = pin[(i // Cfg.batch) % 2]
            if ev is not None:
                ev.synchronize()
            torch.index_select(Xtr, 0, idx, out=xb_pin)
            torch.index_select(Ytr, 0, idx, out=yb_pin)
            xb = xb_pin.to(device, non_blocking=True)
            yb = yb_pin.to(device, non_blocking=True)
            if ev is not None:
                ev.record()
            cu = doc_cu_seqlens(xb, eos_id) if Cfg.doc_mask else None
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
                hidden, _ = model(xb, yb, cu)  # pass targets so compile traces hidden branch
            B, T, D = hidden.shape
            weight = raw_model.head.weight[: raw_model.cfg.vocab]
            loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=SOFTCAP)(
                weight, hidden.to(weight.dtype).reshape(-1, D), yb.reshape(-1)
            )
            loss = loss / Cfg.accum
            # nanochat: skip DDP all-reduce on non-final microbatches
            if ddp and Cfg.accum > 1 and (i // Cfg.batch + 1) % Cfg.accum != 0:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            if (i // Cfg.batch + 1) % Cfg.accum == 0:
                grad_norm = nn.utils.clip_grad_norm_(raw_model.parameters(), Cfg.clip)
                # One CPU sync per step: finite(loss) & finite(grad_norm), MIN-reduced across ranks
                flag = (torch.isfinite(loss.detach()) & torch.isfinite(grad_norm)).float()
                if ddp:
                    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                healthy = flag.item() > 0.5
                if step % 10 == 9:
                    last = loss.item() * Cfg.accum  # only sync the loss value on log steps
                if not healthy:
                    # This check runs BEFORE opt.step(), so the parameters are still the last
                    # healthy ones and only the gradients are non-finite: dropping the gradients
                    # is the entire fix. Restoring the snapshot here was strictly worse -- for the
                    # first GOOD_SAVE_INTERVAL steps it is the pre-training random init, good_opt
                    # is still empty so the optimizer moments are NOT rolled back with it, and
                    # `step` is not rewound, so a single non-finite grad at step 300 silently put
                    # the run back at initialization while the log read "restored last good state".
                    n_skip += 1
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)
                    if fp8:
                        raw_model.zero_grad(set_to_none=True)
                    if is_main:
                        runlog(f"step {step}/{total_steps} non-finite grad — step skipped ({n_skip})")
                    if n_skip >= 20 and good_state is not None:
                        # 20 in a row is not a transient spike; the parameters themselves are
                        # suspect, so fall back to the last snapshot and its optimizer state.
                        raw_model.load_state_dict(good_state)
                        for j, opt in enumerate(optimizers):
                            if good_opt[j] is not None:
                                opt.load_state_dict(good_opt[j])
                        n_skip = 0
                        if is_main:
                            runlog(f"step {step}/{total_steps} 20 skips in a row — rolled back to snapshot")
                    step += 1
                    if step >= total_steps:
                        break
                    continue
                n_skip = 0
                set_schedule(optimizers, step, total_steps, Cfg)
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
                # Periodically save a CPU copy of the healthy state
                if step % GOOD_SAVE_INTERVAL == 0:
                    good_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                    good_opt = opt_snapshot(optimizers)
                    if is_main and step % 1000 == 0:
                        torch.save(
                            {
                                "model": good_state,
                                "opt": good_opt,
                                "step": step,
                                "cfg": {k: v for k, v in vars(Cfg).items() if not k.startswith("_")},
                            },
                            ckpt_path + f".step{step}",
                        )
                if Cfg.val_every and step % Cfg.val_every == 0:
                    v = validate(
                        model,
                        raw_model,
                        Xva,
                        Yva,
                        Cfg.batch,
                        device,
                        amp_dtype,
                        eos_id if Cfg.doc_mask else None,
                        Cfg.val_batches,
                    )
                    if is_main:
                        runlog(f"step {step}/{total_steps} val {v:.3f}")
                if is_main and step % 10 == 0:
                    now = time.time()
                    tps = 10 * Cfg.batch * Cfg.accum * Cfg.seq / (now - t_log)  # tokens/s per GPU
                    mfu = 6 * n_params * tps / (peak_tflops * 1e12)
                    t_log = now
                    runlog(
                        f"step {step}/{total_steps} loss {last:.3f} | {tps / 1e3:.0f}K tok/s/gpu "
                        f"| MFU {mfu * 100:.0f}%"
                    )
                if step >= total_steps:
                    break

        # Flush leftover gradients from incomplete accumulation at epoch end
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if fp8:
            raw_model.zero_grad(set_to_none=True)  # clear bf16 model grads too

        # All ranks run validation to keep DDP in lockstep; only rank 0 prints. Capped at a fixed
        # prefix of the split: uncapped it is 27.6K sequences (113M tokens) recomputed identically on
        # every rank, which on a --max_steps ablation costs more wall-clock than the training it is
        # measuring. A fixed prefix keeps the number comparable across runs.
        v = validate(
            model,
            raw_model,
            Xva,
            Yva,
            Cfg.batch,
            device,
            amp_dtype,
            eos_id if Cfg.doc_mask else None,
            max_batches=Cfg.val_batches_full,
        )
        if is_main:
            runlog(f"ep {ep + 1}/{Cfg.epochs} train {last:.3f} val {v:.3f} {time.time() - t0:.0f}s")
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "cfg": {k: v for k, v in vars(Cfg).items() if not k.startswith("_")},
                },
                ckpt_path + f".ep{ep + 1}",
            )
        if step >= total_steps:
            break  # --max_steps reached (validation + epoch ckpt already done above)

    if is_main:
        torch.save(
            {
                "model": raw_model.state_dict(),
                "cfg": {k: v for k, v in vars(Cfg).items() if not k.startswith("_")},
            },
            ckpt_path,
        )
        print(f"saved {ckpt_path}")
        runlog.plot()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
