#!/usr/bin/env python3
"""Train a ~200M Chinese LLM with hybrid recurrent architecture (KDA + gated MLA).

Muon (2D params) + AdamW (embeddings/1D), bf16 autocast, optional FP8 via torchao, Flash
Full causal attention, torch.compile, DDP, Liger FLCE with logit softcap, and Attention
Residuals (arXiv 2603.15031) in place of the residual sum (--attn_res, on by default).

    torchrun --nproc_per_node=8 train.py --fp8 [--attn_res] [--fone] [--name X]
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import copy
import datetime
import glob
import hashlib
import json
import math
import multiprocessing as mp
import random
import re
import shutil
import tempfile
import time
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (used via torch.utils.checkpoint.checkpoint)
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

import fone

ROOT = os.path.dirname(os.path.abspath(__file__))


class RunLog:
    """Tee prints to runs/<name>.log; track=True also mirrors parsed metrics to trackio."""

    _STEP_RE = re.compile(
        r"step (\d+)/\d+.*?loss ([\d.]+).*?lr ([\d.eE+-]+).*?gnorm ([\d.]+).*?"
        r"([\d.]+)K tok/s/gpu \| MFU (\d+)%"
    )
    _VAL_RE = re.compile(r"step (\d+)/\d+ val ([\d.]+)")

    def __init__(self, name, track=False):
        os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
        self.path = os.path.join(ROOT, "runs", f"{name}.log")
        self.f = open(self.path, "a", encoding="utf-8")
        self.track = None
        if track:
            import trackio

            trackio.init(project=os.environ.get("TRACKIO_PROJECT", "aupai"), name=name)
            self.track = trackio

    def __call__(self, msg):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()
        if self.track:
            m = self._STEP_RE.search(msg)
            if m:
                s, loss, lr, gnorm, tps, mfu = m.groups()
                self.track.log(
                    {
                        "loss": float(loss),
                        "lr": float(lr),
                        "gnorm": float(gnorm),
                        "tok_s_per_gpu_k": float(tps),
                        "mfu": int(mfu),
                    },
                    step=int(s),
                )
            elif v := self._VAL_RE.search(msg):
                self.track.log({"val_loss": float(v.group(2))}, step=int(v.group(1)))

    def plot(self):
        import subprocess
        import sys

        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "plot_curves.py"), self.path])


DATA = os.path.join(ROOT, "data")
TOK_PATH = os.path.join(DATA, "tokenizer.json")
TOKEN_CACHE = "/data00/pretrain_1b_tokens.pt"

# Committed audit evidence in data/corpus/sample/ (d76a8a1; read by datagen/fasttext_junk.py)
# is jsonl but not shards. Enumerated here, not by schema-sniffing rows: a sniffer silently
# skips a shard whose first line broke; this list fails loud on the next such file.
NON_SHARD_JSONL = {
    "cci3_audit_400.jsonl",
    "cci3_audit_400_labels.jsonl",
    "cci3_handread_150.jsonl",
    "cci3_iaa_50.jsonl",
    "web_labels.jsonl",
}

try:  # CUDA-only kernels; absent on Mac where only checkpoint tooling imports this module
    from fla.ops.kda import chunk_kda
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:
    chunk_kda = LigerFusedLinearCrossEntropyLoss = None
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    HAS_FA = True
except ImportError:
    try:
        # flash-attn 4 keeps the same two names under .cute with a DIFFERENT positional
        # order: its fourth positional is `qv`, not cu_seqlens_q. Every call below passes
        # cu and the max lengths by keyword, which is correct for both versions and is the
        # only thing standing between this import and a silently mis-bound mask.
        from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

        HAS_FA = True
    except ImportError:
        HAS_FA = False


# Applied identically in training (Liger FLCE) and inference; SOFTCAP=0 disables it.
SOFTCAP = float(os.environ.get("SOFTCAP", 15.0)) or None


class Cfg:
    d = 1024
    heads = 8  # hd=128, required for FlashKDA CUTLASS kernel
    chunk_size = 32  # fla chunk_kda chunk size: +19.1% KDA kernel (~2% step, single-layer
    # isolation, tilerl 2026-08-30), numerically neutral vs 64 (eff.chunk_size_parity). The
    # throughput gain is isolated-layer, not seven-card real-model: the merge run must show it.
    bucket_cap_mb = 50  # DDP gradient bucket: 50 ties 25 at 75K tok/s/gpu, wins on fewer
    # allreduces; 100 leaves comm unhidden (eff.bucket_cap_mb_ab). A Cfg field (not just an
    # argparse arg) so the checkpoint records what it trained under.
    layers = 12
    attn_every = 4
    ffn_hidden = 3072
    vocab = 32784  # multiple of 16: 8 for the cuBLAS aligned kernel (32773 fell back to the
    # SM75 align-1 GEMM on Hopper, 41% vs 92% of bf16 peak, +13.9% end-to-end, measured
    # 2026-08-30), 16 so _fp8_ok passes and the fp8-head option stays open (same cost: the
    # extra columns are never targets). 32776 was 8-aligned only; the A/B at 32784 is
    # eff.vocab_align_parity.
    # The 11 columns above vocab_real are alignment padding: never targets, set to
    # finfo(dtype).min in lm_logits (finite, so the all-finite E2E assert holds).
    # padded_vocab (32832) is unchanged, so head/embedding shapes and old checkpoints
    # are unaffected -- this is not a tokenizer change and does not touch vocab_id.
    vocab_real = 32773  # the frozen tokenizer's size (2026-08-29): 32768 BPE merges + 4 chat
    # specials + [NUM], with <unk>/<eos> inside the merges; vocab - vocab_real is padding
    fone = False
    num_id = 32772  # [NUM] is the last id, always in the vocab, so --fone resizes nothing
    fone_loss_w = 1.0
    seq = 4096  # the recurrent arch handles arbitrary length at inference
    batch = (
        32  # throughput_bisect 2026-08-27: 90K tok/s at batch 32 no-ckpt; 72 needs grad_ckpt (2.4x slower)
    )
    accum = 1
    warmup = 20  # absolute steps, not a fraction: momentum/second-moment reliability needs a
    # roughly constant count (eff.warmup_absolute_not_fractional: at the 0.2b point, 2 steps
    # lost 0.52 val vs 20). The fraction varies 9.2% (0.2b) to 0.57% (3.24b) -- a known
    # confound that overestimates beta; the proportional alternative biased the same way harder.
    warmdown = 0.65
    final_lr_frac = 0.05
    clip = 1.0
    val_frac = 0.05
    seed = 42
    compile = True  # model body only; FLCE loss kept outside (Liger compile-incompatible)
    grad_ckpt = False  # costs 25% wall-clock for ~15GB savings; batch 32 fits without it on H20
    attn_res = True  # blocks=0 -> Full (every sublayer a source); N>0 -> exactly N blocks
    attn_res_blocks = 0
    attn_res_dyn_q = False
    attn_res_lr = 0.01  # AdamW lr for the zero-init pseudo-queries (wd=0)
    # <eos> -> cu_seqlens: KDA state and SWA reset per document instead of leaking across the
    # ~10 docs packed into each 4K row.
    doc_mask = True
    # Must name a live mix: a retired one here trains the retired recipe in silence.
    mix = "data/mix_scale_3.24b.json"  # domain mix (weights / epoch caps / anneal)
    # Symlinked shards, or a domain whose live bytes mismatch its build_corpus_stats.json
    # fingerprint, refuse to start: a swapped-in corpus trains under another domain's name
    # (the voided 0.2b run: CCI3 shards under web_hq's name). The flag pardons known,
    # intended byte drift; it never pardons a symlink.
    allow_corpus_drift = False
    # Pod code must match the committed manifest (scripts/pod_drift.py). The flag pardons a
    # known, intended hotfix; the default refuses, because a pod behind HEAD trains under
    # rules the repo no longer has (142 files drifted before this guard existed).
    allow_pod_drift = False
    # Environment fingerprint mismatch on resume: the checkpoint's env differs from
    # the current one (container restart, package update). Default refuses -- a resume
    # under a changed environment is not the same run.
    allow_env_drift = False
    anneal_frac = 0.10  # last fraction of tokens uses each domain's "anneal" weight (MiniCPM-style)
    val_every = 500  # 0 = epoch end only
    val_batches = 20
    val_batches_full = 100  # fixed prefix, so the epoch-end number is comparable across runs
    val_rows_max = 5000  # per mix domain; validation only ever reads a prefix
    # Muon (matrix params) — nanochat recipe
    muon_lr = 0.01
    muon_momentum = 0.95
    muon_ns_steps = 5
    muon_wd = 0.10  # was 0.28; nanochat 1/width² law @ d=1024
    # AdamW (embedding + 1D params) — batch-scaled per nanochat
    embed_lr = 0.1
    embed_betas = (0.8, 0.995)
    embed_wd = 0.001
    scalar_lr = 0.15
    scalar_betas = (0.8, 0.95)
    scalar_wd = 0.0


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

    def forward(self, x, cu=None):
        B, T, D = x.shape
        latent = self.kv_down(x)
        k, v = self.kv_up(latent).chunk(2, dim=-1)
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


# --- FP8 compute: e4m3 for both forward and backward (e5m2 backward was unstable without grad_ckpt) ---
_FP8_MAX_E4M3 = 448.0


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
        # Cache fp8 + scales for backward: 5 quants -> 3, saved activations bf16 -> fp8
        ctx.save_for_backward(x_fp8, w_fp8, bias, x_scale, w_scale)
        ctx.orig_shape = x.shape
        return out.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x_fp8, w_fp8, bias, x_scale, w_scale = ctx.saved_tensors
        go2d = grad_out.reshape(-1, grad_out.shape[-1])
        go_scale = (go2d.detach().abs().max().clamp(min=1e-12) / _FP8_MAX_E4M3).float()
        go_fp8 = (go2d / go_scale).to(torch.float8_e4m3fn)
        # _scaled_mm wants mat2 column-major: .t() of an (in, out) contiguous tensor
        w_t = w_fp8.t().contiguous()
        grad_x = torch._scaled_mm(
            go_fp8,
            w_t.t(),
            scale_a=go_scale,
            scale_b=w_scale,
            out_dtype=torch.bfloat16,
        ).reshape(ctx.orig_shape)
        x_t = x_fp8.t().contiguous()
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
    def __init__(self, linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data)
        self.bias = nn.Parameter(linear.bias.data) if linear.bias is not None else None
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    # Kept out of the compiled graph: Inductor's min-cut partitioner recomputes the saved fp8
    # tensors, re-dividing already-scaled values -> NaN grads at step 1 without grad_ckpt
    # (eval/nan_probe.py, 2026-08-26).
    @torch._dynamo.disable
    def forward(self, x):
        return FP8LinearFunction.apply(x, self.weight, self.bias)


def _fp8_ok(mod, name):
    # FoNE heads excluded by NAME: num_head runs on hidden[nmask], whose row count is the batch's
    # [NUM] count -- rarely a multiple of 16, and not visible from the weight shape.
    return name not in ("head", "num_proj", "num_head") and all(d % 16 == 0 for d in mod.weight.shape)


def convert_to_fp8_compute(model):
    """FP8 linears via torchao Float8Linear; falls back to FP8Linear (dynamo-disabled -> ~200
    graph breaks) when torchao is missing. FP8_RECIPE=rowwise|tensorwise|legacy overrides.

    Default e4m3_tensorwise = tensorwise scaling with grad_output in e4m3: stock `tensorwise`
    uses e5m2 there, the unstable backward the legacy path abandoned, and stock `rowwise` keeps
    e4m3 but raises `aten.clone.default with axiswise scaling is not supported yet` under
    compile on the AttnRes .chunk() path."""
    try:
        from torchao.float8 import (
            CastConfig,
            Float8LinearConfig,
            ScalingGranularity,
            ScalingType,
            convert_to_float8_training,
        )
    except ImportError:
        print("torchao not found: using legacy FP8Linear (slow path, graph breaks)", flush=True)
        return _convert_to_fp8_legacy(model)
    recipe = os.environ.get("FP8_RECIPE", "e4m3_tensorwise")
    if recipe == "legacy":
        return _convert_to_fp8_legacy(model)
    if recipe in ("rowwise", "tensorwise"):
        cfg = Float8LinearConfig.from_recipe_name(recipe)
    else:

        def cc():
            return CastConfig(
                scaling_type=ScalingType.DYNAMIC,
                scaling_granularity=ScalingGranularity.TENSORWISE,
                target_dtype=torch.float8_e4m3fn,
            )

        cfg = Float8LinearConfig(
            cast_config_input=cc(), cast_config_weight=cc(), cast_config_grad_output=cc()
        )
    convert_to_float8_training(
        model, config=cfg, module_filter_fn=lambda m, fqn: _fp8_ok(m, fqn.rsplit(".", 1)[-1])
    )
    return model


def _convert_to_fp8_legacy(model):
    """Replace nn.Linear with FP8Linear (skip tied LM head and tiny layers not divisible by 16)."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and _fp8_ok(module, name):
            setattr(model, name, FP8Linear(module))
        else:
            _convert_to_fp8_legacy(module)
    return model


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
        self.apply(self._init)
        for m in self.modules():
            if isinstance(m, AttnRes) and m.dyn is not None:
                nn.init.zeros_(m.dyn[1].weight)  # after _init, or it starts non-uniform
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
        if self.fone and num_vals is not None:
            mask = (idx == self.cfg.num_id).unsqueeze(-1)
            feat = fone.encode_tensor(num_vals.masked_fill(~mask.squeeze(-1), 0.0)).to(emb.dtype)
            emb = emb + torch.where(mask, self.num_proj(feat), emb.new_zeros(()))
        hidden = self.norm(self._body(emb, cu))
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


# --- Muon optimizer (from karpathy/nanochat, simplified per-param) ---

POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class Muon(torch.optim.Optimizer):
    """Nesterov momentum + Polar Express orthogonalization + cautious weight decay."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.28):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._compiled = {}  # (shape, ns_steps, tall, device) -> compiled function
        self._scalar_tensors = {}

    def _get_scalar_tensors(self, device):
        if device not in self._scalar_tensors:
            self._scalar_tensors[device] = (
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )
        return self._scalar_tensors[device]

    def _get_compiled(self, shape, ns_steps, tall, device):
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

            self._compiled[key] = torch.compile(muon_update, dynamic=False)
        return self._compiled[key]

    @torch.no_grad()
    def step(self):
        shape_groups = {}  # by shape, so same-shape params update in one batched call
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
            # 0-D tensors, not floats: Dynamo value-specializes floats and recompiled every step
            device = sg["params"][0].device
            lr_t, mom_t, wd_t = self._get_scalar_tensors(device)
            lr_t.fill_(sg["lr"])
            mom_t.fill_(sg["momentum"])
            wd_t.fill_(sg["wd"])
            if n == 1:
                p, g, mb = sg["params"][0], sg["grads"][0], sg["mbs"][0]
                fn = self._get_compiled(shape, sg["ns_steps"], tall, device)
                W = p.unsqueeze(0)
                G = g.unsqueeze(0)
                M = mb.unsqueeze(0)
                W, M = fn(G, W, M, lr_t, mom_t, wd_t)
                p.data.copy_(W[0])
                mb.copy_(M[0])
            else:
                W = torch.stack(sg["params"])
                G = torch.stack(sg["grads"])
                M = torch.stack(sg["mbs"])
                fn = self._get_compiled(shape, sg["ns_steps"], tall, device)
                W, M = fn(G, W, M, lr_t, mom_t, wd_t)
                for i, p in enumerate(sg["params"]):
                    p.data.copy_(W[i])
                    sg["mbs"][i].copy_(M[i])


def doc_cu_seqlens(idx, eos_id):
    """cu_seqlens over the flattened B*T stream: every row start and every position after an <eos>
    opens a document. Length varies per batch, hence mark_dynamic for torch.compile.

    A *run* of <eos> opens one document, not one per token. SFT rows are padded to seq with
    <eos> (mean 489 per 4097-token row, max 3721), and one boundary per pad token made every
    pad its own length-1 document: fla's varlen grid is per-document, so batch 16 produced
    grid=(2, 78936, 1) against CUDA's gridDim.Y limit of 65535 and cuLaunchKernel returned a
    bare `invalid argument`. Zero-length documents are meaningless everywhere, so this is a
    correctness fix, not an SFT workaround."""
    B, T = idx.shape
    flat = idx.reshape(-1)
    starts = torch.nonzero(flat == eos_id).squeeze(1) + 1
    starts = starts[starts < B * T]
    starts = starts[flat[starts] != eos_id]
    rows = torch.arange(0, B * T, T, device=idx.device)  # every row start, always: rows must not merge
    end = torch.tensor([B * T], dtype=starts.dtype, device=idx.device)
    cu = torch.cat([rows, starts, end]).unique().to(torch.int32)
    torch._dynamo.mark_dynamic(cu, 0)
    return cu


def validate(
    model, raw_model, Xva, Yva, batch, device, amp_dtype, eos_id=None, max_batches=None, Vva=None, Wva=None
):
    """Mean FLCE loss over the fixed validation split; all DDP ranks call it in lockstep. Under
    --fone the values must ride along, or every [NUM] is fed zero and the loss is not comparable."""
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
            vva = Vva[j : j + batch].to(device) if Vva is not None else None
            cu = doc_cu_seqlens(xva, eos_id) if eos_id is not None else None
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda")):
                hidden, _ = model(xva, yva, cu, vva)
            D = hidden.shape[-1]
            v = flce(weight, hidden.to(weight.dtype).reshape(-1, D), yva.reshape(-1))
            if Wva is not None:
                nm = yva == raw_model.cfg.num_id
                if nm.any():
                    wva = Wva[j : j + batch].to(device)
                    v = v + raw_model.cfg.fone_loss_w * F.cross_entropy(
                        raw_model.num_logits(hidden[nm].float()).reshape(-1, 10),
                        fone.digit_targets(wva[nm]).reshape(-1),
                    )
            vl.append(v.item())
    model.train()
    return sum(vl) / len(vl)


class MasterWeights:
    """fp32 copies of the bf16 parameters for the optimizer to own and step.

    Measured 2026-08-30: at the schedule's own LR floor (final_lr_frac 0.05) 91% of parameter
    elements came back from step() bit-identical -- bf16 has 8 mantissa bits, so an update
    ~2^-9 below the parameter cannot change it. The work was computed, applied and discarded
    with nothing raising. Keeping the optimizer's copy in fp32 lets those updates accumulate
    until they are large enough to move the bf16 weight.

    Only the optimizer changes. The model, DDP's gradient traffic, torchao's fp8 cast and the
    fla kernels all keep bf16 -- autocast would not have: chunk_kda is a custom op, so autocast
    leaves its inputs in whatever dtype the parameters are.
    """

    def __init__(self, model):
        self.pairs = [(p, p.detach().float().clone().requires_grad_(True)) for p in model.parameters()]
        self.map = {p: m for p, m in self.pairs}

    def pull_grads(self):
        """Copy p.grad into m.grad and clear p.grad -- both halves, in one place.

        The optimizer holds `m`, so its zero_grad() clears m.grad and NOTHING clears
        p.grad: the next backward() accumulated into the old one and this arm trained on
        a running sum (grad 2.0, 4.0, 6.0 over three steps). Clearing here rather than
        next to opt.step() keeps the copy and the clear in the same function, so they
        cannot drift apart again."""
        for p, m in self.pairs:
            m.grad = None if p.grad is None else p.grad.float()
            p.grad = None

    def push(self):
        with torch.no_grad():
            for p, m in self.pairs:
                p.copy_(m)

    def resync(self):
        """After a rollback the model is the truth again; the fp32 copies must follow or the
        next step would push the pre-rollback weights straight back in."""
        with torch.no_grad():
            for p, m in self.pairs:
                m.copy_(p)


def build_optimizers(model, cfg, master=None):
    """Muon for 2D matrices; AdamW for embeddings, 1D norm gains, and (low lr, wd=0) for the 3D
    short-conv kernels (were mis-routed to the 15x scalar lr) and AttnRes pseudo-queries. Base LRs
    only -- lr_scale is applied in set_schedule, so a resume cannot keep a stale scale."""
    muon, embed, scalar, arq = [], [], [], []
    for n, p in model.named_parameters():
        # Grouping is by the MODEL's name and shape; the tensor handed to the optimizer is the
        # fp32 master when there is one, so every group keeps its own lr and weight decay.
        q = p if master is None else master[p]
        if "tok" in n or "head" in n:
            embed.append(q)
        elif p.ndim == 2:
            muon.append(q)
        elif (
            p.ndim == 3
            or ".dyn." in n
            or n.endswith("ar1.q")
            or n.endswith("ar2.q")
            or n.endswith("final_ar.q")
        ):
            arq.append(q)
        else:
            scalar.append(q)
    opts = [
        Muon(
            muon,
            lr=cfg.muon_lr,
            momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps,
            weight_decay=cfg.muon_wd,
        ),
        torch.optim.AdamW(
            embed, lr=cfg.embed_lr, betas=cfg.embed_betas, weight_decay=cfg.embed_wd, fused=True
        ),
        torch.optim.AdamW(
            scalar,
            lr=cfg.scalar_lr,
            betas=cfg.scalar_betas,
            weight_decay=cfg.scalar_wd,
            fused=True,
        ),
    ]
    if arq:
        opts.append(
            torch.optim.AdamW(
                arq,
                lr=getattr(cfg, "attn_res_lr", 0.01),
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


def set_schedule(optimizers, step, total, cfg, lr_scale=1.0):
    """LR warmup/warmdown for every group; momentum ramp and WD decay-to-zero for Muon ONLY -- the
    old loop overwrote the embedding group's wd 0.001 with muon_wd."""
    m = lr_mult(step, total, cfg)
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * lr_scale * m
            if isinstance(opt, Muon):
                g["momentum"] = 0.85 + 0.10 * min(1.0, step / 150)
                g["weight_decay"] = g["initial_wd"] * max(0.0, 1.0 - step / total)


def opt_snapshot(optimizers):
    """Real CPU copies of optimizer state: state_dict() values are dicts, so a top-level clone
    aliases the live CUDA moments."""
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
    """Rows every rank can iterate: min over ranks of n//batch*batch, so all take the same number of
    steps (one row off -> different lr per rank, then an NCCL hang)."""
    if not ddp:
        return n // batch * batch
    t = torch.tensor([n // batch], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item()) * batch


def _jsonl_content(path):
    return [json.loads(ln)["content"] for ln in open(path, encoding="utf-8") if ln.strip()]


VOCAB_ID = None  # fingerprint of the id->token map the run trained against

# t50: fork()ed encode workers inherit these globals (copy-on-write, no pickling
# the doc list). Set by _encode_domain right before it spawns workers.
_PARALLEL_TEXTS = None
_PARALLEL_TOK = None


# Greedy/sampled decoding, the ONE implementation. There were seven near-identical copies
# (three probes, infer.py, test_e2e.py, rlvr_generate.py, and this one); only this one
# batches, truncates the prompt to the trained context, and handles the FoNE value channel,
# so every other copy was slower and silently unbounded. It lives here because `model` is a
# HybridLM and scripts/loader.py must stay importable without torch.
_EOS = 1
_MAX_CTX = 4096  # the trained seq len; smaller truncates the model's own long reasoning away


@torch.no_grad()
def generate_batch(model, prompts, max_new, device, temperature=0.0, prompt_values=None,
                   tokenizer=None, rep_stop=True):
    """Greedy (temperature=0) or sampled decoding for a list of token-id lists. Returns generated ids.

    prompt_values switches on the FoNE path: a per-position value list for each prompt.
    Then the return is (ids, values) per row, because a [NUM] token carries no number
    of its own -- the digit head reads it off the same hidden state that predicted the
    token, and fone.decode_text writes it back into the text.

    rep_stop: repetition stop (whitespace 8-gram repeated 3x, checked every 32 tokens).
    Requires tokenizer for decoding. A correct answer never trips it; degenerate ones
    stop at ~100 tokens instead of running to max_new.
    """
    B = len(prompts)
    keep = max(0, _MAX_CTX - max_new)  # prompt budget; 0 means "keep all" (p[-0:] == p[0:])
    fone_on = prompt_values is not None
    if fone_on:
        prompt_values = [v[-keep:] for v in prompt_values]
    prompts = [p[-keep:] for p in prompts]
    lengths = [len(p) for p in prompts]
    x = torch.full((B, max(lengths)), _EOS, dtype=torch.long, device=device)
    v = torch.zeros((B, max(lengths)), device=device) if fone_on else None
    for i, p in enumerate(prompts):
        x[i, : lengths[i]] = torch.tensor(p, device=device)
        if fone_on:
            v[i, : lengths[i]] = torch.tensor(prompt_values[i], device=device)
    ends = torch.tensor(lengths, device=device)  # next write position per row
    done = torch.zeros(B, dtype=torch.bool, device=device)
    ar = torch.arange(B, device=device)
    num_id = model.cfg.num_id if fone_on else None
    rep_stop = rep_stop and tokenizer is not None

    for step in range(max_new):
        # DEFECT (tilerl): this recomputes the full prefix per token -- no KDA/MLA
        # state is carried across steps. Each step is O(T) not O(1), making
        # generation O(T^2) per sequence. A state-carrying decode path would
        # cut wall time ~max_new-fold for long contexts.
        _, hidden = model(x[:, -_MAX_CTX:], num_vals=v[:, -_MAX_CTX:] if fone_on else None, no_head=True)
        # hidden covers only the last _MAX_CTX positions; index relative to that slice, then
        # run the head on those B rows alone rather than on B x T.
        off = max(0, x.size(1) - _MAX_CTX)
        step_hidden = hidden[ar, ends - off - 1]
        step_logits = model.lm_logits(step_hidden)
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(step_logits.float() / temperature, dim=-1), 1).squeeze(1)
        else:
            nxt = step_logits.argmax(dim=-1)
        if x.size(1) <= int(ends.max()):
            x = torch.cat([x, torch.full((B, 1), _EOS, dtype=torch.long, device=device)], dim=1)
            if fone_on:
                v = torch.cat([v, torch.zeros((B, 1), device=device)], dim=1)
        x[ar, ends] = torch.where(done, torch.full_like(nxt, _EOS), nxt)
        if fone_on:
            val = fone.decode(model.num_logits(step_hidden.float())).to(v.dtype)
            v[ar, ends] = torch.where(nxt == num_id, val, torch.zeros_like(val))
        ends += (~done).long()
        done |= nxt == _EOS
        # Repetition stop: check every 32 tokens. Whitespace 8-gram for
        # English/code; character 12-gram for CJK-majority text (no spaces).
        if rep_stop and step > 0 and step % 32 == 31:
            from collections import Counter
            for i in range(B):
                if done[i]:
                    continue
                gen_ids = x[i, lengths[i] : ends[i]].tolist()
                if len(gen_ids) < 64:
                    continue
                text = tokenizer.decode(gen_ids)
                hit = False
                words = text.split()
                if len(words) >= 24:  # need at least 3x 8-grams
                    grams = [tuple(words[j : j + 8]) for j in range(len(words) - 7)]
                    hit = any(c >= 3 for c in Counter(grams).values())
                if not hit:
                    cjk = sum(1 for c in text if '一' <= c <= '鿿')
                    if cjk > len(text) * 0.3 and len(text) >= 36:
                        chars = list(text)
                        cg = [tuple(chars[j : j + 12]) for j in range(len(chars) - 11)]
                        hit = any(c >= 3 for c in Counter(cg).values())
                if hit:
                    done[i] = True
        if bool(done.all()):
            break
    ids = [x[i, lengths[i] : ends[i]].tolist() for i in range(B)]
    if not fone_on:
        return ids
    vals = [v[i, lengths[i] : ends[i]][x[i, lengths[i] : ends[i]] == num_id].tolist() for i in range(B)]
    return ids, vals


def _env_fp():
    """The effective training environment fingerprint (scripts/env_fp.py).

    Stored in every checkpoint and compared on resume. A container restart can
    change the environment -- dropping hand-installed packages -- without anyone
    noticing. Three sessions spent an hour chasing three wrong hypotheses because
    nothing recorded what the environment WAS. This is the record."""
    import sys as _sys

    _sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from env_fp import env_fingerprint

    return env_fingerprint()


def save_checkpoint(path, model_state, cfg, vocab_id, opt=None, step=None):
    """The ONLY way a checkpoint is written, so no writer can forget the vocabulary.

    There were nine torch.save sites and four schemas. Stamping vocab_id on two of them left
    sft.py and algorithms/rlvr_trainer.py still writing a checkpoint that load_tokenizer can
    only WARN about before loading whatever data/tokenizer.json currently is -- the file that
    is rebuilt in place. `cfg` may be a Cfg class, a namespace or a dict; the private keys are
    stripped here rather than at each call site."""
    assert vocab_id, f"refusing to write {path} with no vocab_id: it could never be scored safely"
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    # Corpus fingerprint alongside vocab_id: which corpus this checkpoint trained on.
    # _corpus_fp is the same hash the startup guard compares to the build-time stamp.
    corpus_fp = {}
    mix_path = cfg.get("mix")
    if mix_path:
        for dom in json.load(open(mix_path, encoding="utf-8")).get("domains", {}):
            ddir = os.path.join("data", "corpus", dom)
            if os.path.isdir(ddir):
                corpus_fp[dom] = _corpus_fp(ddir)
    ck = {
        "model": model_state,
        "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "vocab_id": vocab_id,
        "corpus_fp": corpus_fp,
        "env_fp": _env_fp(),
    }
    if opt is not None:
        ck["opt"] = opt
    if step is not None:
        ck["step"] = step
    torch.save(ck, path)


def vocab_fingerprint(tok):
    """Hash of the id->token map, so a checkpoint records WHICH vocabulary it saw. Size does not
    identify one: two 32,773-token files can disagree on every id, and a pack built against the
    wrong one trained at 4.77 instead of 1.28 without raising (2026-08-28)."""
    import hashlib

    h = hashlib.sha256()
    for t, i in sorted(tok.get_vocab().items(), key=lambda kv: kv[1]):
        h.update(t.encode())
    return h.hexdigest()[:16]


def build_tokenizer(texts):
    """Load data/tokenizer.json; never build one here. The old inline BPE fallback registered only
    <unk>/<eos>, dropping the chat specials and [NUM] while still matching Cfg.vocab, so every id
    shifted and checkpoints read scrambled embeddings with nothing raising."""
    assert os.path.exists(TOK_PATH), (
        f"{TOK_PATH} is missing. Build it with `python scripts/build_tokenizer.py --force`, "
        "which is the only supported path -- it registers the chat specials and [NUM] that "
        "an inline BPE would silently drop."
    )
    global VOCAB_ID
    tok = Tokenizer.from_file(TOK_PATH)
    assert tok.get_vocab_size() == Cfg.vocab_real, (
        f"tokenizer vocab {tok.get_vocab_size()} != Cfg.vocab_real {Cfg.vocab_real}"
    )
    assert Cfg.vocab % 8 == 0 and Cfg.vocab >= Cfg.vocab_real, (
        f"Cfg.vocab={Cfg.vocab} must be a multiple of 8 >= vocab_real={Cfg.vocab_real}: "
        "an unaligned head width falls back to the SM75 align-1 cuBLAS kernel on Hopper (-55% step time)"
    )
    VOCAB_ID = vocab_fingerprint(tok)
    return tok


def encode(texts, tok, chunk=50_000, log=None):
    """Documents -> one <eos>-separated int32 stream.

    np.asarray per document + encode_batch_fast, not array("i").extend(e.ids): the latter dropped
    the pipeline from 3.3M to 1.4M tok/s, 107 min instead of 46 on web (2026-08-26). Chunking
    keeps the Encoding objects bounded."""
    eos = tok.token_to_id("<eos>")
    batch_fn = getattr(tok, "encode_batch_fast", tok.encode_batch)
    parts, vparts = [], []
    t0 = time.time()
    for i in range(0, len(texts), chunk):
        if Cfg.fone:
            # values ride alongside in stream order: the k-th [NUM] takes the k-th value
            pieces, vals = fone.encode_text(texts[i : i + chunk], tok, Cfg.num_id)
            # np.int32(eos), not eos: a python int promotes the pieces to int64, which
            # build_mix's int32 destination then refuses.
            parts.append(np.concatenate([np.append(p, np.int32(eos)) for p in pieces]))
            vparts.append(vals)
            if log and (i // chunk) % 4 == 0:
                ntok = sum(len(p) for p in parts)
                dt = time.time() - t0
                log(
                    f"  encode {min(i + chunk, len(texts))}/{len(texts)} docs, "
                    f"{ntok / 1e6:.0f}M tokens ({ntok / dt / 1e6:.1f}M tok/s), "
                    f"{sum(len(v) for v in vparts) / 1e6:.1f}M numbers"
                )
            continue
        enc = batch_fn(texts[i : i + chunk])
        parts.append(np.concatenate([np.asarray(e.ids + [eos], dtype=np.int32) for e in enc]))
        if log and (i // chunk) % 4 == 0:
            ntok = sum(len(p) for p in parts)
            dt = time.time() - t0
            log(
                f"  encode {min(i + chunk, len(texts))}/{len(texts)} docs, "
                f"{ntok / 1e6:.0f}M tokens ({ntok / dt / 1e6:.1f}M tok/s)"
            )
    ids = torch.from_numpy(np.concatenate(parts))  # int32: vocab 32773 fits, halves bandwidth
    assert ids.dtype == torch.int32, f"token stream is {ids.dtype}, not int32"
    if Cfg.fone:
        return ids, torch.from_numpy(np.concatenate(vparts)) if vparts else torch.zeros(0)
    return ids


def scatter_values(ids, vals, num_id):
    """Compact per-number values -> a dense tensor shaped like ids. The [NUM] slots in row-major
    order line up one-for-one with `vals`, so any row-level slicing must happen AFTER this. vals
    may be longer (trailing tokens dropped by the reshape); shorter is a corrupt cache."""
    out = torch.zeros(ids.shape, dtype=torch.float32)
    mask = ids == num_id
    k = int(mask.sum())
    assert k <= len(vals), f"cache has {len(vals)} values for {k} [NUM] tokens"
    out[mask] = vals[:k].float()
    return out


def _encode_worker(i, lo, hi, path):
    # Part-files, not a Queue: a worker that dies leaves the parent's q.get()
    # blocking forever with no error (observed 2026-08-31); exitcode!=0 surfaces.
    out = encode(_PARALLEL_TEXTS[lo:hi], _PARALLEL_TOK)
    if Cfg.fone:
        ids, vals = out
        np.savez(path, ids=ids.numpy(), vals=vals.numpy())
    else:
        np.savez(path, ids=out.numpy(), vals=np.empty(0))


def _encode_domain(texts, tok, workers, log=None):
    """t50: encode a domain's (already shuffled) docs into the <eos>-separated stream.

    workers>1 splits the doc list into CONTIGUOUS blocks across fork()ed processes
    and concatenates streams in worker order, so the result is element-identical to
    the single-process encode() (t49: stream_sum matched on math_seed and wiki_chat).
    Process parallelism is the lever: the tokenizers rayon pool inside one process
    does not scale (t49: 1 proc x 180 threads 2.45M tok/s vs 8 proc x 22 threads
    18.2M on math_seed). Callers MUST set RAYON_NUM_THREADS=nproc/workers before
    importing tokenizers so the inherited pools sum to nproc; pretokenize.py does.
    The calling process MUST NOT have encoded before the first parallel call: a
    live rayon pool does not survive fork() (the child deadlocks in inherited
    locks -- observed 2026-08-31). pretokenize.py --workers N satisfies this by
    construction: with N>1 the parent never encodes.
    Workers write part-files (see _encode_worker), so per-domain stream size is
    bounded by disk, not by a pipe buffer.
    """
    if workers <= 1:
        return encode(texts, tok, log=log)
    global _PARALLEL_TEXTS, _PARALLEL_TOK
    _PARALLEL_TEXTS, _PARALLEL_TOK = texts, tok
    n = len(texts)
    bounds = [int(k * n / workers) for k in range(workers + 1)]
    tmpdir = tempfile.mkdtemp(prefix="encode_parts_")
    ctx = mp.get_context("fork")
    procs = [
        ctx.Process(target=_encode_worker, args=(i, bounds[i], bounds[i + 1], os.path.join(tmpdir, f"p{i}.npz")))
        for i in range(workers)
    ]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"encode worker {p.pid} exited {p.exitcode}")
        parts = [np.load(os.path.join(tmpdir, f"p{i}.npz")) for i in range(workers)]
        ids = torch.from_numpy(np.concatenate([p["ids"] for p in parts]))
        if Cfg.fone:
            return ids, torch.from_numpy(np.concatenate([p["vals"] for p in parts]))
        return ids
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _domain_cache_path(domain):
    """Token cache path. --fone is part of the NAME, not just the freshness check: it changes the
    token stream while leaving the vocabulary fingerprint identical. Reuse across the flag is
    silent both ways -- a plain cache read as FoNE dies 40 minutes in unpacking `ids, vals`, and a
    FoNE cache read as plain gives len(data)==2, i.e. zero rows, and trains on nothing."""
    return os.path.join(os.path.dirname(TOKEN_CACHE), f"tokens_{domain}{'_fone' if Cfg.fone else ''}.pt")


def _domain_seqs(domain, tok, is_main, ddp, workers=1):
    """Tokenize data/corpus/<domain>/*.jsonl once (rank 0), cache next to TOKEN_CACHE, [N, seq+1].

    Reused only while newer than every shard, carrying the same vocabulary fingerprint, AND
    carrying the source directory's corpus fingerprint (.srcfp): the 2026-08-30 swap rebuilt
    the cache from a different corpus and reused it with nothing raising, because mtime and
    vocab both matched. A stale source fingerprint retokenizes, same as a stale vocabulary."""
    cache = _domain_cache_path(domain)
    stamp = cache + ".vocab"
    srcfp = cache + ".srcfp"
    shards = sorted(
        p
        for p in glob.glob(os.path.join(DATA, "corpus", domain, "*.jsonl"))
        if os.path.basename(p) not in NON_SHARD_JSONL
    )
    same_vocab = os.path.exists(stamp) and open(stamp).read().strip() == (VOCAB_ID or "")
    live_fp = _corpus_fp(os.path.join(DATA, "corpus", domain))
    same_source = os.path.exists(srcfp) and open(srcfp).read().strip() == live_fp
    fresh = (
        os.path.exists(cache)
        and shards
        and same_vocab
        and same_source
        and os.path.getmtime(cache) >= max(os.path.getmtime(p) for p in shards)
    )
    if is_main and not fresh and os.path.exists(cache) and not same_vocab:
        print(f"mix: {domain} cache was built by another vocabulary, retokenizing", flush=True)
    if is_main and not fresh and os.path.exists(cache) and not same_source:
        print(f"mix: {domain} cache's source changed since caching, retokenizing", flush=True)
    if is_main and not fresh:
        texts = []
        for p in shards:
            texts += _jsonl_content(p)
        assert texts, f"mix domain {domain}: no data/corpus/{domain}/*.jsonl"
        random.Random(Cfg.seed).shuffle(texts)
        print(f"mix: tokenizing {domain} ({len(texts)} docs, workers={workers}) -> {cache}", flush=True)
        data = _encode_domain(texts, tok, workers, log=lambda m: print(m, flush=True))
        del texts
        torch.save(data, cache)
        with open(stamp, "w") as f:
            f.write(VOCAB_ID or "")
        with open(srcfp, "w") as f:
            f.write(live_fp)
        n_tok = len(data[0] if Cfg.fone else data)
        print(f"mix: {domain} cached {n_tok / 1e6:.0f}M tokens", flush=True)
        del data
    if ddp:
        dist.barrier()
    data = torch.load(cache, map_location="cpu", weights_only=True)
    if not Cfg.fone:
        n = len(data) // (Cfg.seq + 1)
        return data[: n * (Cfg.seq + 1)].view(-1, Cfg.seq + 1)
    ids, vals = data
    n = len(ids) // (Cfg.seq + 1)
    ids = ids[: n * (Cfg.seq + 1)].view(-1, Cfg.seq + 1)
    return ids, scatter_values(ids, vals, Cfg.num_id)


def _corpus_fp(ddir):
    """Hash of sorted (shard name, size, sha256 of first/last 64KB) for one domain dir.
    Canonical implementation: datagen/corpus_fingerprint.py (fp_dir); inline so train.py
    imports nothing from scripts/ -- this is the same value save_checkpoint writes into
    the checkpoint under corpus_fp. The two must agree bit-for-bit; corpus_fingerprint.py
    --self-check asserts parity on every run. Content-based, not mtime-based: a transfer
    (podput/rsync) changes mtime without touching a byte."""
    fh = hashlib.sha1()
    for nm in sorted(os.listdir(ddir)):
        if nm == "build_corpus_stats.json" or nm.startswith("."):
            continue
        p = os.path.join(ddir, nm)
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            head = f.read(65536)
            if size > 65536:
                f.seek(-65536, os.SEEK_END)
                tail = f.read(65536)
            else:
                tail = b""
        fh.update(f"{nm}:{size}:{hashlib.sha256(head).hexdigest()}:{hashlib.sha256(tail).hexdigest()}\n".encode())
    return fh.hexdigest()[:16]


def _assert_mix_domains(names, corpus_dir, allow_drift=False):
    """Reject a mix that names the unfiltered corpus, a domain with no shards on disk, a
    domain reached through symlinks, or a domain whose live bytes no longer match its
    build-time fingerprint. Called from main() where the mix json is read -- the one point
    run_ddp.sh and a bare `python train.py` share. Not from build_mix,
    which test_arch_compat.py drives with a synthetic mix that has no corpus.

    Returns {domain: live_fp} so main() can log data identity next to the mix line.
    allow_drift pardons fingerprint drift (known, intended byte changes); it never pardons a
    symlink -- a symlinked domain is a different corpus wearing another domain's name, not
    drift of the same corpus."""
    fps = {}
    for name in names:
        assert name != "web", (
            "mix domain 'web' is the UNFILTERED corpus: train.py globs data/corpus/<domain>/*.jsonl, "
            "so this trains on 2,991,648 unfiltered documents and silently discards every quality "
            "filter the corpus-v3 rebuild applied. Use web_hq."
        )
        ddir = os.path.join(corpus_dir, name)
        # Redundant with _domain_seqs' `assert texts`, but fails before the 40-minute tokenize.
        assert glob.glob(os.path.join(ddir, "*.jsonl")), (
            f"mix domain '{name}' has no shards: {os.path.join(ddir)}/*.jsonl matches "
            "nothing. _domain_seqs would raise on this too, but only after tokenizing the domains "
            "that do exist."
        )
        if os.path.islink(ddir):
            links = [f"{name}/ (the domain dir itself)"]
        else:
            links = [f"{name}/{nm}" for nm in sorted(os.listdir(ddir)) if os.path.islink(os.path.join(ddir, nm))]
        assert not links, (
            f"mix domain '{name}' is reached through symlink(s): {links[:5]}. A symlinked domain is a "
            "different corpus wearing another domain's name -- the voided 0.2b run trained on CCI3 "
            "shards under web_hq's name this way. Name the real corpus in the mix instead."
        )
        live = _corpus_fp(ddir)
        fps[name] = live
        if allow_drift:
            continue
        stats = os.path.join(ddir, "build_corpus_stats.json")
        try:
            with open(stats, encoding="utf-8") as f:
                stamped = json.load(f).get("fingerprint")
        except Exception:
            stamped = None
        assert stamped, (
            f"mix domain '{name}' carries no build-time fingerprint ({stats} missing or unstamped). "
            "An unstamped domain cannot be distinguished from a swapped-in one -- rebuild it with "
            "build_corpus.py, or pass --allow_corpus_drift to train on the live bytes knowingly."
        )
        assert live == stamped, (
            f"mix domain '{name}' drifted since build: stamped {stamped} != live {live}. The bytes on "
            "disk are not the ones build_corpus.py recorded. Rebuild the domain, or pass "
            "--allow_corpus_drift to train on the live bytes knowingly."
        )
    return fps


def _selftest_mix_guard():
    """Runs at import, so it fails in CI rather than 40 minutes into a tokenize on the pod.

    LOGIC only. That the guard is ON main()'s path -- the actual bug, which every logic test
    here passes without -- is scripts/harness.py check_guard_on_path, which asserts it by AST
    in CI and has a broken world. It used to be duplicated here, costing an inspect.getsource
    + ast.parse of 1,500 lines on every `import train`, i.e. on every DDP rank."""
    import tempfile

    def expect_raise(names, **kw):
        try:
            _assert_mix_domains(names, d, **kw)
        except AssertionError:
            return
        raise AssertionError(f"mix guard accepted {names} {kw}")

    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "web_hq")
        os.makedirs(good)
        with open(os.path.join(good, "a.jsonl"), "w"):
            pass
        with open(os.path.join(good, "build_corpus_stats.json"), "w") as f:
            json.dump({"fingerprint": _corpus_fp(good)}, f)
        os.makedirs(os.path.join(d, "web"))
        with open(os.path.join(d, "web", "a.jsonl"), "w"):
            pass
        os.makedirs(os.path.join(d, "empty"))
        assert _assert_mix_domains(["web_hq"], d) == {"web_hq": _corpus_fp(good)}
        for bad in (["web"], ["web_hq", "web"], ["empty"], ["web_hq", "missing"]):
            expect_raise(bad)
        # drift: stamped, then mutated -- refused, and pardoned only by allow_drift
        with open(os.path.join(good, "a.jsonl"), "a", encoding="utf-8") as f:
            f.write('{"question": "drifted"}\n')
        expect_raise(["web_hq"])
        assert _assert_mix_domains(["web_hq"], d, allow_drift=True)
        # a domain with no stamp at all -- refused, pardoned by allow_drift
        nostamp = os.path.join(d, "nostamp")
        os.makedirs(nostamp)
        with open(os.path.join(nostamp, "c.jsonl"), "w"):
            pass
        expect_raise(["nostamp"])
        assert _assert_mix_domains(["nostamp"], d, allow_drift=True)
        # a symlinked domain dir, and a real dir holding a symlinked shard -- both refused,
        # and allow_drift does not pardon either
        real = os.path.join(d, "real_corpus")
        os.makedirs(real)
        with open(os.path.join(real, "b.jsonl"), "w"):
            pass
        os.symlink(real, os.path.join(d, "linked"))
        expect_raise(["linked"])
        expect_raise(["linked"], allow_drift=True)
        inside = os.path.join(d, "inside")
        os.makedirs(inside)
        os.symlink(os.path.join(real, "b.jsonl"), os.path.join(inside, "s.jsonl"))
        with open(os.path.join(inside, "r.jsonl"), "w"):
            pass
        expect_raise(["inside"])
        expect_raise(["inside"], allow_drift=True)


_selftest_mix_guard()


def build_mix(cfg_path, tok, is_main, ddp, rank=0, world=1):
    """Domain mix -> (this rank's train rows in schedule order, val rows). mix.json:
    {"total_tokens": 11.5e9, "domains": {"web": {"weight": .83, "epochs": 2, "anneal": .42}, ...}};
    weight = share of the main phase, anneal = share of the last Cfg.anneal_frac tokens.

    The schedule is an index plan -- (domain, row) pairs, ~22MB -- and only this rank's 1/world
    slice becomes token rows: materializing it per rank costs ~2.3TB of host RAM at 11.5B x 8.
    Rows are pre-shuffled per phase and consumed in order, so main -> anneal is exact."""
    mix = json.load(open(cfg_path, encoding="utf-8"))
    rows = mix["total_tokens"] / Cfg.seq
    phases = [(1 - Cfg.anneal_frac, "weight"), (Cfg.anneal_frac, "anneal")]
    g = torch.Generator().manual_seed(Cfg.seed)
    names = list(mix["domains"])
    pools, val, used = {}, [], {}
    vpools, vval = {}, []  # --fone: per-position number values, shadowing pools/val exactly
    for name in names:
        seqs = _domain_seqs(name, tok, is_main, ddp)
        seqs, vseq = seqs if Cfg.fone else (seqs, None)
        # Capped: an uncapped 5% split of a 1.9M-row domain keeps 95K rows alive to read 4.8K.
        n_val = min(max(1, int(len(seqs) * Cfg.val_frac)), Cfg.val_rows_max)
        val.append(seqs[:n_val])
        pools[name] = seqs[n_val:]
        if Cfg.fone:
            vval.append(vseq[:n_val])
            vpools[name] = vseq[n_val:]
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
    # Multiple of world, or a rank left a row short gets a different lr and hangs the all-reduce.
    n = (plan.shape[1] // world) * world
    mine = plan[:, :n][:, rank::world]
    out = torch.empty((mine.shape[1], Cfg.seq + 1), dtype=torch.int32)
    vout = torch.empty_like(out, dtype=torch.float32) if Cfg.fone else None
    for di, name in enumerate(names):
        m = mine[0] == di
        if m.any():
            out[m] = pools[name][mine[1][m]]
            if Cfg.fone:
                vout[m] = vpools[name][mine[1][m]]
    del pools, vpools
    # One permutation for both, so a validation row keeps its own numbers.
    vperm = torch.randperm(sum(len(v) for v in val), generator=torch.Generator().manual_seed(Cfg.seed))
    vcat = torch.cat(val)[vperm]
    if Cfg.fone:
        return (out, vout), (vcat, torch.cat(vval)[vperm])
    return out, vcat


def lr_mult(step, total, cfg):
    if step < cfg.warmup:
        return (step + 1) / cfg.warmup
    wd_steps = max(1, int(cfg.warmdown * total))
    wd_start = total - wd_steps
    if step < wd_start:
        return 1.0
    progress = min(1.0, (step - wd_start) / wd_steps)  # clamped past total (resume)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.final_lr_frac + (1 - cfg.final_lr_frac) * cosine


def setup_ddp():
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    # 2h, not NCCL's 10min: rank 0 tokenizes alone (~45 min for web) while the others barrier
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return True, rank, world, local


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
        "warmup": "warmup steps in absolute terms (default 20; a fraction lost 0.52 val at the 0.2b point -- eff.warmup_absolute_not_fractional)",
        "seed": "RNG seed for init, data order and dropout",
        "attn_every": "one attention layer every N blocks",
    }.items():
        parser.add_argument(f"--{name}", type=int, default=None, help=f"{help_} (default: Cfg.{name})")
    for name, help_ in {
        "warmdown": "fraction of total steps for the cosine warmdown tail (WSD; 0 keeps lr at stable for a stage-1 join)",
        "anneal_frac": "fraction of tokens using each domain's anneal weight (0 = no anneal, for a WSD stage-1)",
    }.items():
        parser.add_argument(f"--{name}", type=float, default=None, help=f"{help_} (default: Cfg.{name})")
    for name, help_ in {
        "grad_ckpt": "gradient checkpointing (recompute sublayers in backward)",
        "attn_res": "Attention Residuals (arXiv 2603.15031)",
        "attn_res_dyn_q": "AttnRes input-dependent pseudo-query",
        "fone": "Fourier number embedding: one [NUM] per number, value in, digits out",
    }.items():
        parser.add_argument(f"--{name}", action="store_true", help=help_)
    parser.add_argument(
        "--fp8", action="store_true", help="FP8 linears (torchao; FP8_RECIPE=legacy for old path)"
    )
    parser.add_argument(
        "--fp32_master",
        action="store_true",
        help="optimizer owns fp32 copies of the bf16 weights (see MasterWeights)",
    )
    parser.add_argument(
        "--frozen_probe",
        action="store_true",
        help="report what fraction of elements step() left bit-identical, on the last step",
    )
    parser.add_argument("--allow_slow_attn", action="store_true",
                        help="run without flash_attn on a GPU (~20x slower, correct)")
    parser.add_argument(
        "--mix", type=str, default=None, help='domain mix json (default Cfg.mix; "" = flat corpus)'
    )
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="stop after N optimizer steps (ablations)"
    )
    parser.add_argument(
        "--save_every", type=int, default=1000,
        help="write a resumable checkpoint (opt+step) every N steps; the t38 resume test and the 16h interval both need this tunable",
    )
    parser.add_argument("--name", type=str, default="pretrain", help="runs/<name>.log, ckpt_<name>.pt")
    parser.add_argument(
        "--track", action="store_true", help="mirror step metrics to trackio (local, TRACKIO_PROJECT)"
    )
    parser.add_argument("--profile", action="store_true", help="export a chrome trace of N steps (measurement only, no behavior change)")
    parser.add_argument("--profile_warmup", type=int, default=15)
    parser.add_argument("--profile_steps", type=int, default=20)
    parser.add_argument(
        "--allow_corpus_drift", action="store_true",
        help="train even if a domain's live bytes mismatch its build-time fingerprint; never pardons symlinks",
    )
    parser.add_argument(
        "--allow_pod_drift", action="store_true",
        help="train on a pod whose code is behind the committed manifest (known hotfix only)",
    )
    parser.add_argument(
        "--allow_env_drift", action="store_true",
        help="resume even if the checkpoint's environment fingerprint differs (container restart, package change)",
    )
    parser.add_argument("--no_attn_res", action="store_true", help="disable AttnRes (A/B measurement)")
    parser.add_argument("--bucket_cap_mb", type=int, default=50, help="DDP gradient bucket size in MB (50: +14.1%% vs 100, eff.bucket_cap_mb_ab)")
    parser.add_argument("--no_static_graph", action="store_true", help="disable DDP static_graph (A/B: 5K overhead hunt)")
    parser.add_argument("--no_bucket_view", action="store_true", help="disable DDP gradient_as_bucket_view (A/B: 5K overhead hunt)")
    # nanochat's rates assume 1.77M tokens/step; at batch 24 x 8 (786K) unscaled they made the
    # loss bottom out at step 610 and climb, 3.45 -> 4.36 by step 1060 (val 3.03 -> 3.56).
    parser.add_argument("--lr_scale", type=float, default=1.0, help="multiplier on every optimizer lr")
    args = parser.parse_args()
    for k, v in vars(args).items():
        if hasattr(Cfg, k) and v:
            setattr(Cfg, k, v)
    # warmdown/anneal_frac are floats whose valid value 0.0 is falsy: the loop above skips
    # them (the --warmdown 0 stage-1 join would silently keep Cfg.warmdown 0.65). Apply
    # them explicitly by is-not-None so 0.0 lands.
    for k in ("warmdown", "anneal_frac"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(Cfg, k, v)
    if args.no_attn_res:
        Cfg.attn_res = False

    torch.manual_seed(Cfg.seed)
    torch.set_float32_matmul_precision("high")
    ddp, rank, world, local = setup_ddp()
    device = f"cuda:{local}" if ddp else ("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = not ddp or rank == 0
    runlog = RunLog(args.name, track=args.track) if is_main else print
    if args.mix is not None:
        Cfg.mix = args.mix
    ckpt_path = os.path.join(
        ROOT, f"ckpt_{args.name}.pt"
    )  # always name-derived; pipeline evals ckpt_<name>.pt
    amp = device.startswith("cuda")

    # The mix is the only data path. The old flat-corpus fallback was where a named-but-missing
    # mix landed, silently training on whatever data/corpus/*.jsonl held (244KB).
    assert Cfg.mix, "no mix configured. The flat-corpus fallback is gone; use --mix data/mix_sample.json"
    mix_path = os.path.join(ROOT, Cfg.mix)
    assert os.path.exists(mix_path), (
        f"--mix names {Cfg.mix}, which does not exist. Push the file. There is no fallback "
        f"corpus to train on instead."
    )
    assert os.path.exists(TOK_PATH), "mix mode needs a trained data/tokenizer.json"
    fps = _assert_mix_domains(
        list(json.load(open(mix_path, encoding="utf-8"))["domains"]),
        os.path.join(DATA, "corpus"),
        allow_drift=Cfg.allow_corpus_drift,
    )
    if is_main:
        print(f"corpus_fp: {fps}", flush=True)
    if not os.path.isdir(os.path.join(ROOT, ".git")) and not Cfg.allow_pod_drift:
        # The pod is not a git repo: its code must match the committed manifest, or it
        # trains under rules the repo no longer has (142 files drifted before this guard).
        import sys

        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import pod_drift

        _ok, _evidence = pod_drift.check_pod(ROOT)
        if not _ok:
            raise RuntimeError(
                f"pod code drift vs the committed manifest: {_evidence}. Sync the pod to "
                "HEAD (scripts/pod_sync_check.sh) or pass --allow_pod_drift for a known hotfix."
            )
    if is_main:
        # GEMM-hostile dimensions (vocab 32773 ran the LM head at 41% of bf16 peak) must
        # fail before tokenizing 40 minutes of corpus. Static, CPU-only, seconds. Canonical
        # implementation: scripts/shape_audit.py -- a subprocess so train.py imports nothing
        # from scripts/, same pattern as the corpus_fingerprint.py parity assertion.
        import subprocess
        import sys

        _audit = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "shape_audit.py")],
            capture_output=True,
            text=True,
        )
        if _audit.returncode != 0:
            print(_audit.stdout, _audit.stderr)
            raise RuntimeError("shape_audit FAIL: GEMM-hostile Cfg dimensions, see above")
    tok = build_tokenizer([])
    eos_id = tok.token_to_id("<eos>")
    tr, va = build_mix(mix_path, tok, is_main, ddp, rank, world)
    (seqs, num_tr), (vseqs, num_va) = (tr, va) if Cfg.fone else ((tr, None), (va, None))
    seqs, vseqs = seqs.long(), vseqs.long()
    Xtr, Ytr, Xva, Yva = seqs[:, :-1], seqs[:, 1:], vseqs[:, :-1], vseqs[:, 1:]
    # V* feeds the embedding (aligned with X); W* is the digit target (aligned with Y).
    Vtr = num_tr[:, :-1].contiguous() if Cfg.fone else None
    Wtr = num_tr[:, 1:].contiguous() if Cfg.fone else None
    Vva = num_va[:, :-1].contiguous() if Cfg.fone else None
    Wva = num_va[:, 1:].contiguous() if Cfg.fone else None
    data, X = seqs, seqs  # for the params print below
    Cfg.epochs = 1  # repeats are encoded in the schedule
    Xtr, Ytr = Xtr.contiguous().pin_memory(), Ytr.contiguous().pin_memory()
    if Cfg.fone:
        Vtr, Wtr = Vtr.contiguous().pin_memory(), Wtr.contiguous().pin_memory()
    # Xtr[idx] allocates an unpinned temp, which makes .to(non_blocking=True) synchronous. Stage
    # through two pinned buffers, each reused only after its previous H2D copy's event completes.
    pin = [
        (
            torch.empty((Cfg.batch, Cfg.seq), dtype=Xtr.dtype).pin_memory(),
            torch.empty((Cfg.batch, Cfg.seq), dtype=Ytr.dtype).pin_memory(),
            torch.cuda.Event() if amp else None,
            torch.empty((Cfg.batch, Cfg.seq), dtype=torch.float32).pin_memory() if Cfg.fone else None,
            torch.empty((Cfg.batch, Cfg.seq), dtype=torch.float32).pin_memory() if Cfg.fone else None,
        )
        for _ in range(2)
    ]
    raw_model = HybridLM(Cfg).to(device)
    resume_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Environment fingerprint: a container restart can change the effective
        # environment (dropping hand-installed packages) without anyone noticing.
        # A mismatch means this is not the same run -- refuse unless overridden.
        ck_fp = ck.get("env_fp")
        if ck_fp:
            live_fp = _env_fp()
            if ck_fp != live_fp and not args.allow_env_drift:
                raise RuntimeError(
                    f"env fingerprint mismatch: {args.resume} trained in env {ck_fp}, "
                    f"current env is {live_fp}. The environment changed (container "
                    f"restart? package update?). Refusing to resume -- this is not the "
                    f"same run. Pass --allow_env_drift to override."
                )
            if is_main:
                print(f"env fingerprint {'OK' if ck_fp == live_fp else 'DRIFT (overridden)'} ({live_fp})", flush=True)
        raw_model.load_state_dict(ck["model"])
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
        # runlog, not print: an unrecorded batch size once cost 90 minutes of regression-chasing
        runlog(
            f"params {n_params / 1e6:.1f}M | tokens {len(data)} | seqs {len(X)} | "
            f"device {device} | world {world} | fa {HAS_FA} | fp8 {fp8}"
        )
        # The fallback is correct now, so this no longer guards correctness -- it guards
        # against a run nobody meant to start. Measured on an H20: bool-mask SDPA is ~20x
        # slower per step, which turns a 16-hour run into 13 days while every log line looks
        # normal. A 20x-slower run is a wrong run unless someone says otherwise, out loud.
        assert not (torch.cuda.is_available() and not HAS_FA and not args.allow_slow_attn), (
            "flash_attn is unavailable, so attention falls back to masked SDPA: correct, and "
            "~20x slower -- a 16h run becomes 13 days. Install flash-attn, or pass "
            "--allow_slow_attn to say you meant it."
        )
        runlog(
            f"cfg batch {Cfg.batch} accum {Cfg.accum} seq {Cfg.seq} grad_ckpt {Cfg.grad_ckpt} "
            f"doc_mask {Cfg.doc_mask} attn_res {Cfg.attn_res}/{Cfg.attn_res_blocks} "
            f"softcap {SOFTCAP} warmup {Cfg.warmup} epochs {Cfg.epochs} "
            f"lr_scale {args.lr_scale} mix {Cfg.mix or 'flat'} fone {Cfg.fone}"
        )

    master = MasterWeights(raw_model) if args.fp32_master else None
    optimizers = build_optimizers(raw_model, Cfg, master.map if master else None)
    if master is not None and is_main:
        print(f"fp32 master weights: {sum(m.numel() for _, m in master.pairs) * 4 / 1e9:.2f} GB", flush=True)
    if args.resume and "step" in ck and "opt" not in ck:
        raise RuntimeError(
            f"{args.resume} records step {ck['step']} but carries no optimizer state. "
            f"Resuming would zero Muon momentum and AdamW moments -- the loss would dip "
            f"and recover, looking like noise. This checkpoint cannot be safely resumed."
        )
    if args.resume and "opt" in ck:
        for opt, sd in zip(optimizers, ck["opt"], strict=True):
            opt.load_state_dict(sd)  # momentum/moments continue instead of restarting from 0

    model = raw_model
    if ddp:
        model = DDP(
            model, device_ids=[local], bucket_cap_mb=args.bucket_cap_mb,
            gradient_as_bucket_view=not args.no_bucket_view, static_graph=not args.no_static_graph
        )
    if Cfg.compile and amp:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 256
        if Cfg.attn_res:
            # The AttnRes loop builds one compiled graph per distinct source count: 1 + 2*layers
            # in Full mode (25), 1 + n_blocks in Block mode. Below that, torch.compile silently
            # falls back to eager from source limit+1 on -- 980.8 -> 1463.9 ms/step (-33%,
            # measured 2026-08-30). Read the EFFECTIVE limit back; an unreadable one raises,
            # because a check that cannot run is not a check.
            try:
                import torch._dynamo as _dynamo

                _dynamo_limit = _dynamo.config.cache_size_limit
            except Exception as e:
                raise RuntimeError(f"cannot read torch._dynamo.config.cache_size_limit: {e}") from e
            _dynamo_need = 1 + min(Cfg.attn_res_blocks or 2 * Cfg.layers, 2 * Cfg.layers)
            assert _dynamo_limit >= _dynamo_need, (
                f"cache_size_limit={_dynamo_limit} < {_dynamo_need} AttnRes sources: eager fallback "
                f"from source {_dynamo_limit + 1} on, -33% step time, silently. Keep it at 64 next "
                "to the model = torch.compile line."
            )
            if is_main:
                print(f"dynamo cache_size_limit={_dynamo_limit} (need {_dynamo_need})", flush=True)
        if os.environ.get("COMPILE_SUPPRESS_ERRORS", "0") == "1":
            torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, dynamic=False, mode=os.environ.get("COMPILE_MODE") or None)

    good_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
    good_opt = [None] * len(optimizers)
    total_steps = Cfg.epochs * (len(Xtr) // (Cfg.batch * Cfg.accum))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)  # LR schedule completes within the short run
    step = resume_step
    if args.resume and is_main:
        # WSD stage join: the resumed lr must continue from where stage 1 stopped. Print it
        # loudly so a two-stage launch can be verified at a glance rather than inferred.
        _jm = lr_mult(step, total_steps, Cfg)
        runlog(
            f"WSD JOIN: resumed at step {step}/{total_steps} under mix {Cfg.mix or 'flat'} | "
            f"lr_mult {_jm:.4f} | warmdown {Cfg.warmdown} anneal_frac {Cfg.anneal_frac} | "
            f"warmdown starts at step {total_steps - max(1, int(Cfg.warmdown * total_steps))}"
        )
    n_skip = 0  # consecutive optimizer steps skipped for non-finite gradients
    _prof = None
    if getattr(args, "profile", False):
        import torch.profiler as _tp
        os.makedirs("/work/aupai/bench_eff", exist_ok=True)
        _prof = _tp.profile(
            activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA],
            schedule=_tp.schedule(wait=1, warmup=args.profile_warmup, active=args.profile_steps, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(f"/work/aupai/bench_eff/ddp_trace_rank{rank}.json"),
        )
        _prof.start()
    GOOD_SAVE_INTERVAL = 200
    for ep in range(Cfg.epochs):
        model.train()
        perm = torch.arange(len(Xtr))  # the schedule is already in order; never reshuffle it
        i0 = step * Cfg.batch * Cfg.accum
        t0 = time.time()
        last = 0.0
        t_log = time.time()
        for i in range(i0, len(Xtr) - Cfg.batch + 1, Cfg.batch):
            idx = perm[i : i + Cfg.batch]
            xb_pin, yb_pin, ev, vb_pin, wb_pin = pin[(i // Cfg.batch) % 2]
            if ev is not None:
                ev.synchronize()
            torch.index_select(Xtr, 0, idx, out=xb_pin)
            torch.index_select(Ytr, 0, idx, out=yb_pin)
            xb = xb_pin.to(device, non_blocking=True)
            yb = yb_pin.to(device, non_blocking=True)
            vb = wb = None
            if Cfg.fone:
                torch.index_select(Vtr, 0, idx, out=vb_pin)
                torch.index_select(Wtr, 0, idx, out=wb_pin)
                vb = vb_pin.to(device, non_blocking=True)
                wb = wb_pin.to(device, non_blocking=True)
            if ev is not None:
                ev.record()
            cu = doc_cu_seqlens(xb, eos_id) if Cfg.doc_mask else None
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
                hidden, _ = model(xb, yb, cu, vb)  # targets given so compile traces the hidden branch
            B, T, D = hidden.shape
            weight = raw_model.head.weight[: raw_model.cfg.vocab]
            loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=SOFTCAP)(
                weight, hidden.to(weight.dtype).reshape(-1, D), yb.reshape(-1)
            )
            if Cfg.fone:
                # [NUM] says a number comes next but not which one, so digits are supervised
                # separately against the SHIFTED value slice (wb), not the embedding's (vb).
                nmask = yb == Cfg.num_id
                if nmask.any():
                    nlog = raw_model.num_logits(hidden[nmask].float())
                    ntgt = fone.digit_targets(wb[nmask])
                    loss = loss + Cfg.fone_loss_w * F.cross_entropy(nlog.reshape(-1, 10), ntgt.reshape(-1))
            loss = loss / Cfg.accum
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
                    # Runs BEFORE opt.step(), so the parameters are still healthy and dropping the
                    # gradients is the whole fix. Restoring the snapshot here instead put a run back
                    # at random init on one bad grad at step 300, logging "restored last good state".
                    n_skip += 1
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)
                    # Unconditional: dropping the gradients IS this path. An optimizer may
                    # not hold the parameters (--fp32_master) or all of them (fp8), and in
                    # a plain run this is a no-op because opt.zero_grad already cleared them.
                    raw_model.zero_grad(set_to_none=True)
                    if is_main:
                        runlog(f"step {step}/{total_steps} non-finite grad — step skipped ({n_skip})")
                    if n_skip >= 20 and good_state is not None:  # not a transient spike
                        raw_model.load_state_dict(good_state)
                        for j, opt in enumerate(optimizers):
                            if good_opt[j] is not None:
                                opt.load_state_dict(good_opt[j])
                        if master is not None:
                            master.resync()
                        n_skip = 0
                        if is_main:
                            runlog(f"step {step}/{total_steps} 20 skips in a row — rolled back to snapshot")
                    step += 1
                    if step >= total_steps:
                        break
                    continue
                n_skip = 0
                set_schedule(optimizers, step, total_steps, Cfg, args.lr_scale)
                # Two readings. Measured 2026-08-30 at 0.2b: model bf16 84.5% frozen without
                # master, 77.9-79.1% with -- a single step still rounds away in bf16, but the
                # fp32 copy keeps it and over 217 steps part of the accumulation clears the
                # bf16 ULP and moves the weight. So the first reading DOES fall, just far less
                # than the second (0.1%), which is where the update is actually kept.
                probe = None
                if args.frozen_probe and step == total_steps - 1:
                    probe = ([(p, p.detach().clone()) for p in raw_model.parameters()],
                             [(m, m.detach().clone()) for _, m in master.pairs] if master else None)
                if master is not None:
                    master.pull_grads()
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if master is not None:
                    master.push()
                if probe is not None and is_main:
                    if probe[1] is None:
                        # Say it, or the next reader takes the missing line for a dropped
                        # measurement: without --fp32_master the optimizer steps the model's
                        # own parameters, so the two readings would be one number.
                        runlog("frozen[optimizer] n/a — no --fp32_master, the optimizer holds "
                               "the model's own parameters")
                    for tag, pairs, bits in (("model bf16", probe[0], torch.int16),
                                             ("optimizer", probe[1], torch.int32)):
                        if pairs is None:
                            continue
                        same = sum(int((a.view(bits) == b.view(bits)).sum()) for a, b in pairs)
                        tot = sum(a.numel() for a, _ in pairs)
                        runlog(f"frozen[{tag}] {100 * same / tot:.1f}% of {tot / 1e6:.1f}M elements")
                step += 1
                if _prof is not None:
                    _prof.step()
                # Refresh the rollback buffer on the finer of the two cadences so a save
                # never writes a stale snapshot (save_every can be < GOOD_SAVE_INTERVAL).
                if step % min(GOOD_SAVE_INTERVAL, args.save_every) == 0:
                    good_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                    good_opt = opt_snapshot(optimizers)
                if step > 0 and step % args.save_every == 0 and is_main:
                    save_checkpoint(ckpt_path + f".step{step}", good_state, Cfg, VOCAB_ID, good_opt, step)
                    # keep the newest 3; resume only needs the latest
                    stale = sorted(
                        glob.glob(ckpt_path + ".step*"),
                        key=lambda p: int(p.rsplit(".step", 1)[1]),
                    )[:-3]
                    for p in stale:
                        os.remove(p)
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
                        Vva,
                        Wva,
                    )
                    if is_main:
                        runlog(f"step {step}/{total_steps} val {v:.3f}")
                if is_main and step % 10 == 0:
                    now = time.time()
                    dt = now - t_log
                    tps = 10 * Cfg.batch * Cfg.accum * Cfg.seq / dt
                    mfu = 6 * n_params * tps / (peak_tflops * 1e12)
                    t_log = now
                    phase = " [anneal]" if step > (1 - Cfg.anneal_frac) * total_steps else " [main]"
                    eta = (total_steps - step) * dt / 10
                    runlog(
                        f"step {step}/{total_steps} {step / total_steps:.0%}{phase} | loss {last:.3f} "
                        f"| lr {optimizers[0].param_groups[0]['lr']:.2e} | gnorm {grad_norm.item():.2f} "
                        f"| {step * Cfg.batch * Cfg.accum * Cfg.seq * world / 1e9:.2f}B tok "
                        f"| {tps / 1e3:.0f}K tok/s/gpu | MFU {mfu * 100:.0f}% | ETA {eta / 3600:.1f}h"
                    )
                if step >= total_steps:
                    break

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if fp8:
            raw_model.zero_grad(set_to_none=True)  # clear bf16 model grads too

        # All ranks validate to keep DDP in lockstep; only rank 0 prints. Fixed prefix: the full
        # split is 27.6K sequences per rank, more wall-clock than a --max_steps ablation itself.
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
            Vva=Vva,
            Wva=Wva,
        )
        if is_main:
            runlog(f"ep {ep + 1}/{Cfg.epochs} train {last:.3f} val {v:.3f} {time.time() - t0:.0f}s")
            save_checkpoint(
                ckpt_path + f".ep{ep + 1}",
                raw_model.state_dict(),
                Cfg,
                VOCAB_ID,
                opt_snapshot(optimizers),
                step,
            )
        if step >= total_steps:
            break  # --max_steps reached (validation + epoch ckpt already done above)

    if is_main:
        # A checkpoint that took no optimizer steps is a 206M random init that every
        # downstream check passes. test_e2e stage 4 measures the postcondition; this makes
        # the simplest form of it impossible.
        assert step > 0, "refusing to save: the training loop ran zero optimizer steps"
        save_checkpoint(ckpt_path, raw_model.state_dict(), Cfg, VOCAB_ID)
        print(f"saved {ckpt_path}")
        runlog.plot()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
