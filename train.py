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
import signal
import shutil
import tempfile
import time

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
# A shard is NAMED like one: ShardWriter emits <prefix>_<NNN>.jsonl and nothing else does.
# This is a whitelist because the blacklist above only knows the non-shards that existed
# when someone wrote it down -- on 2026-09-01 datagen started writing
# holdout_slice_<domain>.jsonl INTO the corpus directory it describes, _domain_seqs globbed
# it as a shard, and its header row {phase, rule_fp, n} died on ["content"]. Four domains
# carried one, code_py_starcoder among them, so the 20B run would have died before step 0
# with eight idle cards and nobody watching.
#
# Neither side was wrong -- putting the slice beside its corpus is right, and globbing
# *.jsonl was right for two years. The direction of the default is the whole fix: a
# blacklist reads an unknown new file as DATA, a whitelist ignores it. Verified against
# every corpus on the pod: 3453 of 3459 files match, and the 6 that do not are exactly
# the 4 slices plus 2 sample/ label files.
SHARD_RE = re.compile(r"_\d{3,}\.jsonl$")

# Known non-shard artifacts written INTO a corpus dir, by pattern rather than by name:
# holdout_slice_<domain>.jsonl is emitted per domain, so enumerating them would need a
# new entry for every domain ever added. These are skipped silently; anything matching
# neither this nor SHARD_RE stops the run (see _domain_seqs).
NON_SHARD_RE = re.compile(r"^holdout_slice_")

try:  # CUDA-only kernel; the FLCE loss path that uses it stays here
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None

# The model lives in model.py (b0-8, docs/standards/model_module_split.md). Re-exported
# rather than left to each caller: sft.py, sft_math.py and infer_local.py import these from
# train, and the split is meant to be invisible to them. Direction is one-way -- model.py
# never imports train. F401: re-exports are consumed by importers, not by this file.
from model import (  # noqa: E402,F401
    HAS_FA,
    SOFTCAP,
    AttnRes,
    Block,
    DeltaRecurrence,
    GatedMLA,
    HybridLM,
    RMSNorm,
    Source,
    SwiGLU,
    chunk_kda,
    remap_legacy_state_dict,
    rms_scale,
)


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
    # DERIVED, not asserted. [NUM] is the last id in today's tokenizer, so this default is
    # correct -- and it stays correct only while the vocabulary is frozen. resolve_num_id()
    # below reads it from the tokenizer and raises if absent; nothing should read this
    # constant directly. WHY IT IS A HARD FAILURE RATHER THAN A WARNING: num_id is read at
    # three sites (fone masking :808, digit CE :999, value write-back :1229) and every one
    # of them fails SILENTLY into plausible training -- a stale id masks an ordinary BPE
    # token as numeric, computes digit cross-entropy on the wrong positions, and writes
    # values into the wrong slots, with no shape error and no crash. The danger is that
    # property, not the probability of a rebuild.
    num_id = 32772
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
    # SCOPE OF THAT EVIDENCE (b0 2026-09-01): it is ONE-SIDED. 2-vs-20 establishes a floor above
    # 2; nothing above 20 has ever been measured, so 20 is the bottom of a tested range being
    # used as an optimum, not a fitted value. It is also being carried far past where it was
    # measured: 10.486% of the 191-step run it was fitted on, 0.061% of a 32,697-step 500M run.
    # Untested at that length, not wrong. Before lengthening this to protect an early window,
    # read docs/lessons/warmup_momentum_interaction.md -- warmup ends at step 20 while Muon's
    # momentum is 13% into the hardcoded 150-step ramp at line ~1121, and that ramp, not this
    # constant, is what the mechanism points at.
    warmdown = 0.65
    final_lr_frac = 0.05
    clip = 1.0
    val_frac = 0.05
    seed = 42
    #: The seed for the corpus shuffle, separate from `seed` so a seed sweep does not
    #: rebuild every cache. Cfg.seed also drives weight init, the val permutation and
    #: the batch generator, and the p02_s* arms exist to measure INIT variance at a
    #: fixed corpus -- binding the cache to it would change their training data and fold
    #: data variance into ds.seed_variance_0p2b, which the readout's thresholds derive
    #: from. None means "follow Cfg.seed" (de-7 design).
    sample_seed = None
    compile = True  # model body only; FLCE loss kept outside (Liger compile-incompatible)
    grad_ckpt = False  # costs 25% wall-clock for ~15GB savings; batch 32 fits without it on H20
    attn_res = True  # blocks=0 -> Full (every sublayer a source); N>0 -> exactly N blocks
    attn_res_blocks = 0
    attn_res_dyn_q = False
    # One autograd node for the whole AttnRes call instead of the eager loop. Same value,
    # half the graph edges: the eager form reads every source twice (logits, then mixing)
    # and add_ per step is exactly n(n+1); fused it is n(n+1)/2, measured 2.00x at
    # L=2/3/4/12. Default OFF until the A/B; algorithms/attnres_fused.py holds the gates.
    attn_res_fused = False
    # The AttnRes logit is a D=1024 dot product. In bf16 it accumulates 0.858 off against
    # a 279.8 spread (measured vs fp64, scripts/attnres_triton_bf16_gate.py), and softmax
    # turns that into 14% on the mixing weights. Whether 14% on the weights moves the loss
    # is an A/B, not an inference: default OFF until it wins.
    attn_res_fp32_logits = False
    attn_res_lr = 0.01  # AdamW lr for the zero-init pseudo-queries (wd=0)
    # A/B (3), speedrun record: zero-init every OUTPUT projection, so each sublayer starts as
    # an identity on the residual stream and learns its way out. Off by default -- this is an
    # A/B arm, not a production change, until it wins. Covers `o` (both KDA's and MLA's, both
    # named .o) and FFN's `w2`; NOT AttnRes's final_ar, which mixes sources rather than
    # writing to the residual stream (1e's ruling 2026-09-03).
    zero_init_out = False
    # A/B (2a): Muon lr scaled by max(1, out/in)^0.5 per parameter shape (modded-nanogpt).
    muon_shape_lr = False
    # A/B (4): ONE shared [vocab, d] value-embedding table added to V in every MLA layer, gated
    # per position by 3*sigmoid over the residual's first 12 dims. One table, not three: three
    # would be +48.9% parameters at 200M against +16.3% (1e's ruling 2026-09-03).
    value_embed = False
    # b0-17: untie the LM head from the token embedding, and give it its own AdamW lr.
    # untie_head acts only at __init__ (model.py:359), so it is in harness's _FROZEN_KEYS beside
    # value_embed -- a resume silently ignores it and the arm's weights, not the flag, carry the
    # architecture. head_lr is the A/B knob: 0.0 means FOLLOW embed_lr, which is what the tied
    # head has always done, so the default reproduces today's behaviour byte for byte.
    untie_head = False
    head_lr = 0.0
    # <eos> -> cu_seqlens: KDA state and SWA reset per document instead of leaking across the
    # ~10 docs packed into each 4K row.
    doc_mask = True
    # Must name a live mix: a retired one here trains the retired recipe in silence.
    mix = "data/mix_500m.json"  # domain mix (weights / epoch caps / anneal)
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
    # 44-12: a resume that would discard a domain cursor (sample_seed or corpus-fingerprint
    # mismatch) refuses instead of printing and restarting that domain at row 0. The flag
    # pardons a known, intended discard; the default refuses.
    allow_partial_cursor = False
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


# The class-body defaults, snapshotted BEFORE any flag is applied. This is what the cfg-diff
# line at :2187 compares against, and it must be taken here rather than recomputed later:
# argparse writes onto Cfg in place, so after that point the "default" is gone.
_CFG_DEFAULTS = {k: v for k, v in vars(Cfg).items() if not k.startswith("_")}




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


_FP8_WSCALE = {}  # id(weight) -> scale, cleared per step; the head weight is constant within a
                  # step and reused across all 64 FLCE chunks, so recomputing its absmax 64 times
                  # is 64 reductions over 32784x1024 for one value (b0). FP8LinearFunction caches
                  # its scales for the same reason.


def _fp8_mm(a, b, out_dtype, cache_b=False):
    """One fp8 GEMM with per-tensor e4m3 scaling. a @ b, both 2D, b already oriented.

    Same recipe as FP8LinearFunction: scale each operand by its own absmax so the largest
    element lands at e4m3's 448, multiply in fp8, and let _scaled_mm apply the scales to the
    accumulator. Measured range on the head's activations is p50 48 / p99 62 / max 82 against
    448, spread p99/p50 = 1.30x, so per-tensor is sufficient and per-chunk scaling is not
    needed (eff.fp8_head_activation_range).

    Per-tensor scaling is robust to an outlier, tested rather than assumed (probes/t57_outlier.py,
    b0's point that the range fact should not simply wave the question off). Planting one element
    at 82, 448 and 4480 while the rest sit near p50 moves the ORDINARY rows' median relative
    error only 3.766% -> 3.764% -> 3.791%, cosine 0.99929 throughout. The reason is that the
    scale is a DIVISOR, not a clamp: a 55x outlier shrinks every element by 55x together, and
    e4m3 keeps 3 mantissa bits regardless of exponent, so relative precision is scale-invariant.
    Absolute range would only matter if the outlier pushed ordinary elements below e4m3's
    subnormal floor, which needs ~10^5 of spread, not 55x.
    """
    sa = (a.detach().abs().amax().clamp(min=1e-12) / _FP8_MAX_E4M3).float()
    # Every path binds sb explicitly. The earlier walrus form was correct but left the hit case
    # binding sb only as a side effect of the condition, so any added elif or reorder made it
    # unbound -- a NameError rather than a wrong number, but a trap for the next reader (b0).
    if not cache_b:
        sb = (b.detach().abs().amax().clamp(min=1e-12) / _FP8_MAX_E4M3).float()
    elif (cached := _FP8_WSCALE.get(id(b))) is not None:
        sb = cached
    else:
        sb = _FP8_WSCALE[id(b)] = (b.detach().abs().amax().clamp(min=1e-12) / _FP8_MAX_E4M3).float()
    qa = (a / sa).to(torch.float8_e4m3fn)
    qb = (b / sb).to(torch.float8_e4m3fn)
    return torch._scaled_mm(qa.contiguous(), qb.t().contiguous().t(),
                            scale_a=sa, scale_b=sb, out_dtype=out_dtype)


def patch_liger_flce_fp8():
    """Route Liger FLCE's three head GEMMs through _scaled_mm. Returns True if patched.

    The head is the largest single lever with a mechanism: arithmetic intensity 668.7 FLOP/B
    against H20's balance point of 37, so it is compute-bound by 18x, and it is excluded from
    fp8 today only by _fp8_ok's NAME check (eff.lm_head_is_compute_bound). All three matmuls are
    the same M=2048 K=1024 N=32784 shape and measure 62.5 / 63.0 / 64.5 ms per step, so
    converting only the forward buys a third of the ceiling and lands under the ship gate --
    this patches all three, ideal 2x = 95 ms = 5.6% of a 1702 ms step.

    Patched by source rewrite of the three call sites rather than by vendoring all 487 lines,
    so a Liger bump surfaces as a failed patch instead of as a silently stale copy. The
    grad_weight site keeps whatever accumulator dtype is in force: substituting an fp8 GEMM's
    own output dtype there would replace an accumulator with a single product, which is the one
    substitution that breaks correctness (b0). Our grad_weight is bf16 today because weights are
    bf16 and accum_dtype is unset (eff.head_grad_accum_bf16), and that stays true after this.
    """
    try:
        import inspect

        from liger_kernel.ops import fused_linear_cross_entropy as _flce
    except ImportError:
        return False
    src = inspect.getsource(_flce.fused_linear_cross_entropy_forward)
    # Clear the weight-scale cache at the top of every FLCE call. id() is only unique among LIVE
    # objects, so a freed tensor's id can be reused by a different one -- caching across calls
    # would then hand a stale scale to an unrelated weight. Within one call the weight is pinned
    # by the caller's reference, which is exactly the lifetime the 64 chunks need.
    src = src.replace("    for chunk_id in range(num_chunks):",
                      "    _FP8_WSCALE.clear()\n    for chunk_id in range(num_chunks):", 1)
    subs = [
        ("logits_chunk = _input_chunk @ weight.t()",
         "logits_chunk = _fp8_mm(_input_chunk, weight.t(), _input_chunk.dtype, cache_b=True)"),
        ("grad_input[start_idx:end_idx] = grad_logits_chunk @ weight",
         "grad_input[start_idx:end_idx] = _fp8_mm(grad_logits_chunk, weight, grad_input.dtype)"),
        ("grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()",
         "grad_weight += _fp8_mm(grad_logits_chunk.t(), _input_chunk, torch.float32).to(grad_weight.dtype)"),
    ]
    for old, _ in subs:
        if old not in src:
            print(f"fp8 head: PATCH FAILED, Liger source changed -- missing {old[:44]!r}", flush=True)
            return False
    for old, new in subs:
        src = src.replace(old, new)
    ns = dict(_flce.__dict__)
    ns["_fp8_mm"] = _fp8_mm
    ns["_FP8_WSCALE"] = _FP8_WSCALE
    exec(compile(src, "<fp8_flce>", "exec"), ns)  # noqa: S102
    _flce.fused_linear_cross_entropy_forward = ns["fused_linear_cross_entropy_forward"]
    print("fp8 head: Liger FLCE's three GEMMs routed through _scaled_mm", flush=True)
    return True


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
        # Read ONCE, here, not per step: hasattr rather than getattr-with-default, because
        # getattr returns the same False whether the field is missing or present-and-False, so
        # a renamed field would make the arm silently run the baseline while still reporting as
        # the arm (de's shape 7, docs/lessons/fact_and_inference.md).
        assert hasattr(Cfg, "muon_shape_lr"), (
            "Cfg.muon_shape_lr is gone; A/B (2a) cannot tell its arm from the baseline")
        self._shape_lr = Cfg.muon_shape_lr
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
            # A/B (2a), modded-nanogpt's shape-based lr: max(1, out/in)^0.5, where the weight
            # is PyTorch's [out, in]. Muon's update is orthogonalised, so every column of X has
            # roughly unit RMS regardless of shape -- for a tall (fan-out) matrix that means the
            # per-element step is smaller relative to the weight, and the factor puts it back.
            # Wide matrices get exactly 1.0 from the clamp, so they are untouched.
            #
            # It goes HERE, not in the group lr, because params are already batched by shape:
            # every tensor in this call has the same [out, in], so the factor is one scalar per
            # group and costs no recompile. Sharing a group lr across shapes would be wrong.
            shape_mult = 1.0
            if self._shape_lr and len(shape) >= 2:
                shape_mult = max(1.0, shape[-2] / shape[-1]) ** 0.5
            lr_t.fill_(sg["lr"] * shape_mult)
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
    muon, embed, scalar, arq, head = [], [], [], [], []
    for n, p in model.named_parameters():
        # Grouping is by the MODEL's name and shape; the tensor handed to the optimizer is the
        # fp32 master when there is one, so every group keeps its own lr and weight decay.
        q = p if master is None else master[p]
        # THE THIRD ARM LIVES OR DIES HERE. An untied head that still lands in `embed` trains at
        # embed_lr 0.1 -- which is exactly arm 2 (untied, embed lr) and is a legitimate arm, but it
        # is NOT arm 3. The two differ only in this branch, and their logs are otherwise identical,
        # so a mistake here makes two different experiments print the same number.
        # scripts/test_untie_head.py asserts all three arms' head lr separately.
        if "head" in n and getattr(cfg, "untie_head", False) and getattr(cfg, "head_lr", 0.0) > 0:
            head.append(q)
        elif "tok" in n or "head" in n:
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
    # Appended AFTER the three fixed groups, like arq: eval and analysis code indexes opts[0]
    # (Muon) and opts[1] (embed), and inserting a group earlier would silently repoint every one
    # of those readings -- including scripts/embed_norm_sdr.py, which reads ck["opt"][1].
    if head:
        opts.append(
            torch.optim.AdamW(
                head, lr=cfg.head_lr, betas=cfg.embed_betas, weight_decay=cfg.embed_wd, fused=True
            )
        )
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
    # The group's NAME, carried on the optimizer itself so a reader of the step line does not
    # have to know this function's construction order (b0-14). Attached as an attribute rather
    # than returned alongside: ten call sites unpack this list (sft.py:101, sft_math.py:181,
    # bench_eff/*, probes/, scripts/test_arch_*), and returning a tuple would break every one.
    # `arq` is present only when the model has attn_res pseudo-queries, so the zip below must
    # not assume four.
    _names = ["muon", "embed", "scalar"] + (["head"] if head else []) + (["arq"] if arq else [])
    for opt, nm in zip(opts, _names, strict=False):
        opt.aupai_group = nm
    return opts


def set_schedule(optimizers, step, total, cfg, lr_scale=1.0):
    """LR warmup/warmdown for every group; momentum ramp and WD decay-to-zero for Muon ONLY -- the
    old loop overwrote the embedding group's wd 0.001 with muon_wd."""
    m = lr_mult(step, total, cfg)
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * lr_scale * m
            if isinstance(opt, Muon):
                # The 150 is a constant fitted at L=12 and never revisited. It interacts with
                # cfg.warmup: LR reaches full value at step `warmup` (20) while momentum is only
                # 13% through this ramp -- an averaging window of ~7.3 steps against ~20 at the
                # final 0.95. Two constants, one early window, neither examined since L=12.
                # docs/lessons/warmup_momentum_interaction.md has the table and the reasoning.
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
    # Read the cursor attributes BEFORE cfg is rebound. The rebind below turns cfg into a
    # dict on every path, so a `not isinstance(cfg, dict)` guard after it is always false
    # and getattr never finds a data key on a dict anyway -- the whole cursor block below
    # was unreachable, and no checkpoint any run ever wrote carried a cursor (de-8 D1).
    _cur = None if isinstance(cfg, dict) else getattr(cfg, "_row_cursor", None)
    _fps = None if isinstance(cfg, dict) else getattr(cfg, "_row_cursor_srcfp", None)
    _dom_idx = None if isinstance(cfg, dict) else getattr(cfg, "_plan_domains", None)
    _names = None if isinstance(cfg, dict) else getattr(cfg, "_plan_names", None)
    _batch = None if isinstance(cfg, dict) else getattr(cfg, "batch", None)
    _accum = None if isinstance(cfg, dict) else getattr(cfg, "accum", None)
    _origin = 0 if isinstance(cfg, dict) else (getattr(cfg, "_plan_step_origin", 0) or 0)
    _base = {} if isinstance(cfg, dict) else (getattr(cfg, "_row_cursor_base", None) or {})
    _discarded = [] if isinstance(cfg, dict) else (getattr(cfg, "_cursor_discarded", None) or [])
    _total_steps = None if isinstance(cfg, dict) else getattr(cfg, "_total_steps", None)
    _seed = None if isinstance(cfg, dict) else getattr(cfg, "sample_seed", None)
    _seed = (None if isinstance(cfg, dict) else getattr(cfg, "seed", None)) if _seed is None else _seed
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
    # The schedule this checkpoint was written under. A resume recomputes total_steps from
    # the plan it builds, and until 2026-09-02 nothing could compare the two: the field did
    # not exist, so an inflated recomputation was silent by construction -- there was
    # nothing to be inconsistent with. Writing it makes the next resume able to say the
    # schedule moved, and makes a reader of the file able to say what the run was aiming at.
    if _total_steps:
        ck["total_steps"] = int(_total_steps)
    # The row cursor: how many rows of each domain's pool the plan has consumed, and the
    # corpus fingerprint each count is measured against. Stage 2 builds a NEW plan from
    # its own mix, so without this every domain restarts at row 0 and the tail is never
    # read. Top-level, not inside cfg -- the cfg dict strips _-prefixed keys, and a
    # cursor is state, not configuration (de-7).
    cur, fps = _cur, _fps
    if cur:
        # AS OF THIS STEP, not the plan's end. The plan-complete counts describe a run
        # that finished; a checkpoint at step k describes one that has read k*batch*accum
        # rows per rank. Seeding stage 2 from the plan-complete figure would skip
        # everything between k and the end -- and --auto-resume makes a mid-plan
        # checkpoint the expected case, not the rare one (fb, 2026-09-01).
        dom_idx, names = _dom_idx, _names
        if step is not None and dom_idx is not None and names:
            # RELATIVE to this plan, not the absolute step. _plan_domains holds only the
            # current plan's rows, so after a resume an absolute step indexes past its end
            # -- and a Python slice CLAMPS rather than raising, so the cursor written is
            # the plan-complete count wearing an as-of-step label. At step 24000 that is
            # 768,000 rows into a 523,158-row plan, silently (tilerl, 2026-09-01). Correct
            # below the plan length, wrong above it, which is the shape that survives
            # testing: stage 1 starts at 0, where absolute and relative are equal.
            rows_done = (step - _origin) * _batch * _accum  # this rank's share, plan order
            if rows_done < 0 or rows_done > len(dom_idx):
                # Refuse rather than clamp, and record WHY in the checkpoint. A wrong
                # cursor is not recoverable by a later reader -- it looks exactly like a
                # right one -- while a missing cursor costs a resume that repeats rows,
                # which is. No early return: one save at the end of the function, so a
                # future field added below cannot be skipped by this path.
                ck["row_cursor_refused"] = (
                    f"step {step} - origin {_origin} = {step - _origin} steps x "
                    f"{_batch}x{_accum} = {rows_done} rows against a {len(dom_idx)}-row plan; "
                    f"the origin is wrong or the plan is not this run's. No cursor written."
                )
            else:
                head = dom_idx[:rows_done]
                counts = torch.bincount(head.to(torch.int64), minlength=len(names))
                world = int(os.environ.get("WORLD_SIZE", 1))
                # x world: every rank walks its own stripe of the same plan at the same
                # rate, so the whole-run consumption of a domain is this rank's count x world.
                # PLUS the cursor this plan started from: dom_idx holds only the current
                # plan's rows, so counting it alone describes THIS segment and every earlier
                # one is invisible. MEASURED on p500m_20b_0902: .interrupt.step83 summed to
                # 13,056 = (83-32)x256 while step 83 had consumed 21,248 = 83x256, and the
                # 8,192 difference is exactly segment one -- which every later resume then
                # re-read (de-13).
                ck["row_cursor"] = {n: int(counts[i]) * world + int(_base.get(n, 0))
                                    for i, n in enumerate(names)}
                ck["row_cursor_as_of_step"] = step
                # The identity that would have caught the segment-only bug above the day it
                # landed, and which ds.second_resume_rereads_one_segment's uncertainty named
                # as the missing check: an ABSOLUTE cursor sums to exactly the rows the run
                # has consumed, which is as_of_step x batch x accum x world.
                #
                # KNOWN ANSWERS, read off three real checkpoints of p200m_4b_0902 (rows/step
                # = 16x2x8 = 256): step500 128,000; interrupt.step832 212,992;
                # interrupt.step1192 305,152. Only the last has discriminating power -- it
                # was written after a resume (origin 832), so absolute and segment-only
                # disagree there (305,152 against 92,160) and the difference from step832 is
                # 92,160 = 360x256, that segment added in whole. The other two sit at origin
                # 0 where both implementations agree, so a known-answer set without a
                # post-resume checkpoint is blind to exactly this defect.
                #
                # Skipped, not relaxed, when a domain's cursor was DISCARDED: that domain
                # restarted at row 0 and contributes 0 to _base by design, so the sum is
                # legitimately short and firing here would train an operator to ignore it.
                _sum = sum(ck["row_cursor"].values())
                _want = step * _batch * _accum * world
                if _discarded:
                    ck["row_cursor_sum_unchecked"] = (
                        f"{len(_discarded)} domain(s) restarted at row 0, so the cursor sum "
                        f"{_sum} is legitimately below {step}x{_batch}x{_accum}x{world} = "
                        f"{_want}: {'; '.join(_discarded[:4])}"
                    )
                elif _sum != _want:
                    raise AssertionError(
                        f"refusing to write {path}: row_cursor sums to {_sum}, but step {step} "
                        f"x batch {_batch} x accum {_accum} x world {world} = {_want} rows have "
                        f"been consumed. A cursor that is not absolute makes the next resume "
                        f"re-read {_want - _sum} rows silently (the defect measured as "
                        f"ds.second_resume_rereads_one_segment and fixed in 52aec31). No "
                        f"domain reported a discarded cursor, so this is not the discard path."
                    )
        else:
            ck["row_cursor"] = dict(cur)  # no step (run-end save): the plan is complete
        ck["row_cursor_srcfp"] = dict(fps or {})
        ck["row_cursor_seed"] = _seed if _seed is not None else _sample_seed()
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


NUM_TOKEN = "[NUM]"


def resolve_num_id(tok):
    """[NUM]'s id, read from the tokenizer rather than trusted from a constant.

    Cfg.num_id was a hardcoded 32772 with a comment saying [NUM] is "always in the
    vocab". True of the frozen tokenizer and false the moment one is rebuilt: if a
    rebuild yields a smaller vocabulary, [NUM] moves down and 32772 addresses an
    ordinary BPE token instead. Nothing raises -- FoNE then masks that token as
    numeric, computes digit cross-entropy at the wrong positions, and writes values
    into the wrong slots, all with correct shapes. Three read sites, three silent
    failures into training that looks fine.

    So this raises rather than warns, and it raises on the ONE input a rebuild
    actually produces: a tokenizer with no [NUM] at all, or with it somewhere new.
    """
    nid = tok.token_to_id(NUM_TOKEN)
    if nid is None:
        raise SystemExit(
            f"REFUSE: {NUM_TOKEN} is absent from the tokenizer (vocab {tok.get_vocab_size()}). "
            "build_tokenizer.py registers it as a chat special; a vocabulary built without it "
            "cannot run --fone, and a stale Cfg.num_id would silently address a BPE token."
        )
    return nid


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
    Cfg.num_id = resolve_num_id(tok)
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


def _sample_seed():
    """The corpus-shuffle seed: Cfg.sample_seed when set, else Cfg.seed.

    Separate from Cfg.seed on purpose. Cfg.seed also drives weight init, the val
    permutation and the batch generator, and the p02_s* arms measure INIT variance at a
    fixed corpus -- binding the cache to Cfg.seed would change their training data and
    fold data variance into ds.seed_variance_0p2b, the number the readout's thresholds
    derive from. Pinning sample_seed lets a seed sweep share one cache (de-7)."""
    return Cfg.seed if Cfg.sample_seed is None else Cfg.sample_seed


def _token_cache_dir():
    """Directory holding the token caches; AUPAI_TOKEN_CACHE_DIR relocates it, unset is unchanged.
    scripts/test_domain_loss_val.py set HARNESS_TOKEN_CACHE_DIR to redirect its cache and train.py
    read no such variable, so the test wrote a real cache into the pod's shared /data00 beside the
    live run's (2026-09-02); it only stopped mattering because probe_domain is not in mix_500m."""
    return os.environ.get("AUPAI_TOKEN_CACHE_DIR") or os.path.dirname(TOKEN_CACHE)


def _domain_cache_path(domain):
    """Token cache path. --fone is part of the NAME, not just the freshness check: it changes the
    token stream while leaving the vocabulary fingerprint identical. Reuse across the flag is
    silent both ways -- a plain cache read as FoNE dies 40 minutes in unpacking `ids, vals`, and a
    FoNE cache read as plain gives len(data)==2, i.e. zero rows, and trains on nothing."""
    return os.path.join(_token_cache_dir(), f"tokens_{domain}{'_fone' if Cfg.fone else ''}.pt")


def _domain_seqs(domain, tok, is_main, ddp, workers=1):
    """Tokenize data/corpus/<domain>/*.jsonl once (rank 0), cache next to TOKEN_CACHE, [N, seq+1].

    Reused only while newer than every shard, carrying the same vocabulary fingerprint, AND
    carrying the source directory's corpus fingerprint (.srcfp): the 2026-08-30 swap rebuilt
    the cache from a different corpus and reused it with nothing raising, because mtime and
    vocab both matched. A stale source fingerprint retokenizes, same as a stale vocabulary."""
    cache = _domain_cache_path(domain)
    stamp = cache + ".vocab"
    srcfp = cache + ".srcfp"
    # The shuffle below is seeded (Random(Cfg.seed).shuffle), so the seed decides WHICH
    # rows land where in the pool -- two caches built at different seeds hold the same
    # documents in different order. A resume cursor is a row count into that order, so a
    # seed change silently reinterprets every cursor rather than invalidating it. A
    # sidecar, not a name: unlike --fone the stream is identical, so two seeds must not
    # coexist as two caches, they must force one rebuild (de-7 prerequisite).
    seedfp = cache + ".seed"
    # Three branches, not two. A file is a shard, a KNOWN non-shard, or unknown -- and
    # the third refuses. Silently ignoring the unknown would keep new artifacts out of
    # the training data (the point of the whitelist) while throwing away the property
    # the original blacklist was chosen for: "a sniffer silently skips a shard whose
    # first line broke; this list fails loud on the next such file" (train.py:96). The
    # blacklist picked the wrong DEFAULT, not the wrong volume, so the fix keeps the
    # volume. A real shard that someone names oddly must stop the run, not vanish from
    # it -- silently training on 8 of 9 domains is the expensive failure; this one costs
    # two minutes at step 0.
    seen = sorted(glob.glob(os.path.join(DATA, "corpus", domain, "*.jsonl")))
    shards, unknown = [], []
    for p in seen:
        b = os.path.basename(p)
        if b in NON_SHARD_JSONL or NON_SHARD_RE.search(b):
            continue
        (shards if SHARD_RE.search(b) else unknown).append(p)
    if unknown:
        raise SystemExit(
            f"REFUSING: data/corpus/{domain} holds {len(unknown)} .jsonl file(s) that are "
            f"neither shards (<prefix>_NNN.jsonl) nor known non-shards: "
            f"{', '.join(os.path.basename(u) for u in unknown[:4])}. "
            f"A shard misnamed here would be dropped from training in silence. Rename it "
            f"to the shard pattern, or add it to NON_SHARD_JSONL / NON_SHARD_RE in train.py."
        )
    # bool(VOCAB_ID) first, and compared against VOCAB_ID rather than (VOCAB_ID or ""):
    # the old form made an empty stamp equal an unset VOCAB_ID, so the check passed by
    # construction and the seven 0-byte .vocab files on the pod read as fresh (fb,
    # 2026-09-02: eval/ppl.py was two minutes from retokenizing the 20B run's nine caches).
    same_vocab = bool(VOCAB_ID) and os.path.exists(stamp) and open(stamp).read().strip() == VOCAB_ID
    live_fp = _corpus_fp(os.path.join(DATA, "corpus", domain))
    same_source = os.path.exists(srcfp) and open(srcfp).read().strip() == live_fp
    # An unstamped cache REBUILDS. The 17 caches that predate this stamp were written
    # explicitly by scripts/stamp_cache_seeds.py after auditing what each was actually
    # shuffled at (all 42: the ladder caches predate the seed-1/2/3 arms, and every
    # stage-1 cache was built by a run passing --seed 0, which the truthiness bug
    # dropped). Lazy adoption was the first version and 44 was right to reject it: it
    # cannot distinguish "predates the stamp" from "written by a seed nobody recorded",
    # so it would launder exactly the wrong-order cache this exists to catch.
    same_seed = os.path.exists(seedfp) and open(seedfp).read().strip() == str(_sample_seed())
    fresh = (
        os.path.exists(cache)
        and shards
        and same_vocab
        and same_source
        and same_seed
        and os.path.getmtime(cache) >= max(os.path.getmtime(p) for p in shards)
    )
    if is_main and not fresh and os.path.exists(cache) and not same_vocab:
        print(f"mix: {domain} cache was built by another vocabulary, retokenizing", flush=True)
    if is_main and not fresh and os.path.exists(cache) and not same_source:
        print(f"mix: {domain} cache's source changed since caching, retokenizing", flush=True)
    if is_main and not fresh and os.path.exists(cache) and same_source and same_vocab and not same_seed:
        was = open(seedfp).read().strip() if os.path.exists(seedfp) else "unstamped"
        print(f"mix: {domain} cache was shuffled at sample_seed {was}, now "
              f"{_sample_seed()}: retokenizing (the row ORDER differs, so a resume "
              f"cursor into the old order is meaningless)", flush=True)
    if is_main and not fresh:
        texts = []
        for p in shards:
            texts += _jsonl_content(p)
        assert texts, f"mix domain {domain}: no data/corpus/{domain}/*.jsonl"
        random.Random(_sample_seed()).shuffle(texts)
        print(f"mix: tokenizing {domain} ({len(texts)} docs, workers={workers}) -> {cache}", flush=True)
        data = _encode_domain(texts, tok, workers, log=lambda m: print(m, flush=True))
        del texts
        torch.save(data, cache)
        # Before open(), not inside the with: raising after it leaves behind exactly the
        # 0-byte stamp this guards against. `VOCAB_ID or ""` wrote one, and the read side
        # above then accepted it as a match -- seven such files landed on the pod 2026-09-01.
        if not VOCAB_ID:
            raise RuntimeError(
                f"refusing to stamp {stamp} with no vocabulary: VOCAB_ID is unset, so this "
                f"cache could never be told apart from one built by another vocabulary. "
                f"Only train.build_tokenizer sets it -- an eval reaching _domain_seqs "
                f"through scripts.loader must set train.VOCAB_ID from the checkpoint's "
                f"vocab_id first (eval/cache_guard.set_vocab_id)."
            )
        with open(stamp, "w") as f:
            f.write(VOCAB_ID)
        with open(srcfp, "w") as f:
            f.write(live_fp)
        with open(seedfp, "w") as f:
            f.write(str(_sample_seed()))
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


def build_mix(cfg_path, tok, is_main, ddp, rank=0, world=1, row_cursor=None,
              cursor_srcfp=None, cursor_seed=None):
    """Domain mix -> (this rank's train rows in schedule order, val rows). mix.json:
    {"total_tokens": 11.5e9, "domains": {"web": {"weight": .83, "epochs": 2, "anneal": .42}, ...}};
    weight = share of the main phase, anneal = share of the last Cfg.anneal_frac tokens.

    The schedule is an index plan -- (domain, row) pairs, ~22MB -- and only this rank's 1/world
    slice becomes token rows: materializing it per rank costs ~2.3TB of host RAM at 11.5B x 8.
    Rows are pre-shuffled per phase and consumed in order, so main -> anneal is exact."""
    # Refused here, not defaulted: _domain_seqs compares every cache's .vocab stamp against
    # VOCAB_ID, so an unset one makes all nine read as another vocabulary's and retokenize.
    # eval/ppl.py reached this state through scripts.loader (which never sets VOCAB_ID) and
    # was two minutes from rebuilding the live 20B run's caches (fb, 2026-09-02).
    #
    # Gated on `tok`, because the risk is the cache path and not this function. With tok=None
    # nothing is encoded and no stamp is read or written -- test_arch_compat:208 builds four
    # rank shards that way to check the plan partition, and an ungated guard failed it, i.e.
    # broke a CI gate and one of the three pre-relaunch gates to protect a path that call
    # cannot reach. A guard whose condition is wider than its hazard turns working callers
    # into refusals, which is the same cost as the bug it prevents.
    if tok is not None and not VOCAB_ID:
        raise RuntimeError(
            "refusing to build the mix with train.VOCAB_ID unset: every token cache's "
            "vocabulary stamp would read as a mismatch and the domains would be "
            "retokenized. Only train.build_tokenizer sets it; a caller coming through "
            "scripts.loader must set train.VOCAB_ID from the checkpoint's vocab_id "
            "(eval/cache_guard.set_vocab_id) before calling build_mix."
        )
    mix = json.load(open(cfg_path, encoding="utf-8"))
    rows = mix["total_tokens"] / Cfg.seq
    phases = [(1 - Cfg.anneal_frac, "weight"), (Cfg.anneal_frac, "anneal")]
    g = torch.Generator().manual_seed(Cfg.seed)
    names = list(mix["domains"])
    pools, val, used = {}, [], {}
    # Which domains had their cursor thrown away, and why. Published on Cfg because the
    # cursor-sum equality asserted at save time is only exact when nothing was discarded:
    # a discarded domain restarts at row 0 and contributes 0 to _row_cursor_base by design,
    # so the sum falls legitimately short of as_of_step x rows_per_step. Without this the
    # assert would have to be dropped or made inexact, and an inexact identity is not one.
    discarded = []
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
        # Seed the cursor from the checkpoint so a resume under a DIFFERENT mix continues
        # where stage 1 stopped. Within one run the loop already seeks correctly
        # (i0 = step * batch * accum into an ordered plan); the break is across mixes,
        # where a fresh plan restarts every domain at row 0 and leaves the tail unread --
        # 26% of code_rp1t, 34% of en_c4, 92% of zh_web (b0). A count, not an index:
        # pools are read modulo their length, so a cursor past one epoch is meaningful.
        used[name] = 0
        if row_cursor and name in row_cursor:
            # The shuffle seed too, beside srcfp: identical bytes shuffled at another
            # seed give the same fingerprint and a different row ORDER, so the cursor
            # would index different documents while every other check passes (fb).
            if cursor_seed is not None and cursor_seed != _sample_seed():
                if is_main:
                    print(f"mix: {name} cursor discarded -- written at sample_seed "
                          f"{cursor_seed}, this run uses {_sample_seed()}; the pool is "
                          f"shuffled differently so the row count indexes other rows",
                          flush=True)
                discarded.append(f"{name} (sample_seed {cursor_seed} != {_sample_seed()})")
                continue
            want_fp = (cursor_srcfp or {}).get(name)
            ddir = os.path.join(DATA, "corpus", name)
            live_fp = _corpus_fp(ddir) if os.path.isdir(ddir) else None
            if want_fp and live_fp and want_fp != live_fp:
                # Per domain, not all-or-nothing: a rebuilt corpus invalidates ITS cursor.
                # A row count means nothing against different bytes, and silently reusing
                # it is the failure this guard exists for.
                if is_main:
                    print(f"mix: {name} cursor {row_cursor[name]} discarded -- corpus "
                          f"changed ({want_fp[:8]} -> {live_fp[:8]}), restarting at row 0",
                          flush=True)
                discarded.append(f"{name} (corpus {want_fp[:8]} -> {live_fp[:8]})")
            else:
                used[name] = int(row_cursor[name])
                # The base the next save adds to. Set HERE, in the branch that actually
                # applied the cursor: a domain whose cursor was discarded above restarted at
                # row 0, so adding its old value back would claim rows this run never read.
                # Cfg, not a return value, because build_mix runs BEFORE the resume block
                # (train.py:2278 vs :2307) and already publishes _plan_* the same way.
                base = getattr(Cfg, "_row_cursor_base", None)
                if base is None:
                    base = Cfg._row_cursor_base = {}
                base[name] = used[name]
                if is_main:
                    print(f"mix: {name} resuming at row {used[name]} "
                          f"({used[name] / max(len(pools[name]), 1):.2f} epochs consumed)",
                          flush=True)
    # True only when a cursor SEEDED used[] before the plan was built. Named for what it
    # knows: the old name _plan_trimmed asserted the plan HAD been trimmed, which this code
    # did not do -- :1597 allocated the full budget regardless -- and both consumers read
    # the name (tilerl). Computed here, before the phase loop mutates used[] into post-plan
    # counts, which are non-zero on every run and would make the flag meaningless.
    Cfg._cursor_seeded = any(v > 0 for v in used.values())
    Cfg._cursor_discarded = list(discarded)
    if discarded and not Cfg.allow_partial_cursor:
        # 44-12: the print above is the only signal a 66h log gives. Refuse, named, so the
        # operator chooses --allow_partial_cursor knowingly instead of discovering the
        # restarted domains in a readout.
        raise RuntimeError(
            f"refusing to start: {len(discarded)} domain cursor(s) would be silently "
            f"discarded and those domains would restart at row 0 -- "
            f"{'; '.join(discarded[:4])}{'; ...' if len(discarded) > 4 else ''}. "
            f"Pass --allow_partial_cursor to proceed knowingly (44-12)."
        )
    # NOW it is trimmed, and the flag above is the reason it can be: the budget already
    # spent by earlier segments is subtracted, so a resume allocates the REMAINDER instead
    # of the whole recipe a second time. Without this, `rows` is the full budget at :1528
    # and :2140 turns the over-allocated plan into total_steps -- p200m_4b_0902 planned
    # 976,556 rows = 4.00B in the fresh run AND in the resume from step 832, so the run
    # would have ended 832 steps past its own recipe. Bounded below at 0: a spent budget
    # must reach the refusal at :1615, not go negative and plan a garbage count.
    spent = sum(used.values())
    if spent:
        rows = max(0.0, rows - spent)
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
    if not plan:
        # Every domain allocated zero rows, so torch.cat below would raise
        # `expected a non-empty list of Tensors` -- a message about a tensor library, for a
        # condition about the training budget, which sends the reader to the wrong file.
        # Reachable two ways, and the resume way is the one the remainder subtraction above
        # creates: a cursor at or past the budget leaves nothing to allocate. The other is a
        # mix whose weights sum to ~0 or whose pools are all empty.
        raise RuntimeError(
            f"refusing to train on an empty plan: {len(names)} domain(s) allocated 0 rows. "
            f"budget {mix['total_tokens'] / Cfg.seq:.0f} rows, cursor already consumed "
            f"{sum(int(v) for v in (row_cursor or {}).values())}, so "
            f"{rows:.0f} remain. If the cursor equals the budget this checkpoint has "
            f"finished its mix -- point --mix at the next stage instead of resuming into "
            f"the same one."
        )
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
    # The cursor AFTER this plan is consumed, for the checkpoint: what stage 2 seeds from.
    # The per-row DOMAIN INDEX for this rank, in plan order. save_checkpoint counts a
    # prefix of it to get the cursor AS OF a step, instead of the plan-complete counts
    # below: with --auto-resume a mid-plan crash is the expected case, and seeding from
    # the plan's end would skip every row the run never reached. int8, ~0.5MB per rank
    # at stage-1 size (523,158 rows), so keeping it costs nothing next to a 959MB
    # checkpoint (fb ruling: gating for stage 2).
    Cfg._plan_domains = None  # set below, once `mine` exists
    Cfg._row_cursor = dict(used)
    # Only domains whose corpus dir is present: a mix can name a domain served from a
    # cache whose source is not on this box, and an unguarded fingerprint raises there.
    Cfg._row_cursor_srcfp = {
        nm: _corpus_fp(os.path.join(DATA, "corpus", nm))
        for nm in names
        if os.path.isdir(os.path.join(DATA, "corpus", nm))
    }
    # Multiple of world, or a rank left a row short gets a different lr and hangs the all-reduce.
    n = (plan.shape[1] // world) * world
    mine = plan[:, :n][:, rank::world]
    Cfg._plan_domains = mine[0].to(torch.int8).clone()
    Cfg._plan_names = list(names)
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

    # The twelve knobs runs/recipe_provenance.json argues for. Omitting one of these is
    # what four launches on 2026-09-02 did while all nine gates stayed green, because an
    # omitted flag lands on a default -- and five of those defaults EQUAL the recipe
    # value, so no check that reads the effective config could see them missing at all.
    # required=True removes the failure class instead of detecting it.
    #
    # Spelled here rather than imported from scripts/launch_gate.py: train.py is the
    # training entry point and must not grow a dependency on the gate that audits it
    # (the gate imports nothing from train.py either). scripts/test_recipe_required.py
    # reads RECIPE_FLAGS from launch_gate and asserts every one of them is refused when
    # omitted, so the two lists cannot drift without that test going red -- e1-9's whole
    # finding was two lists disagreeing and the gate going blind to four keys.
    RECIPE_REQUIRED = {
        "dim", "layers", "heads", "ffn_hidden", "batch", "accum",
        "lr_scale", "warmdown", "anneal_frac", "warmup", "save_every", "grad_ckpt",
    }

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
        # --dim, not --d: run_ddp.sh's args pass through torchrun's own parser, where
        # argparse prefix matching makes "--d" ambiguous against --duplicate-*-filters
        # and torchrun exits before train.py is ever reached.
        "dim": "model width (must be 128*heads: FlashKDA CUTLASS pins head_dim at 128)",
        "heads": "attention/KDA heads (head_dim = d/heads must be 128)",
        "layers": "number of blocks",
        "ffn_hidden": "FFN inner width",
    }.items():
        parser.add_argument(f"--{name}", type=int, default=None, required=name in RECIPE_REQUIRED,
                            help=f"{help_} (default: Cfg.{name})")
    for name, help_ in {
        "warmdown": "fraction of total steps for the cosine warmdown tail (WSD; 0 keeps lr at stable for a stage-1 join)",
        "anneal_frac": "fraction of tokens using each domain's anneal weight (0 = no anneal, for a WSD stage-1)",
    }.items():
        parser.add_argument(f"--{name}", type=float, default=None, required=name in RECIPE_REQUIRED,
                            help=f"{help_} (default: Cfg.{name})")
    for name, help_ in {
        "grad_ckpt": "gradient checkpointing (recompute sublayers in backward)",
        "attn_res": "Attention Residuals (arXiv 2603.15031)",
        "attn_res_dyn_q": "AttnRes input-dependent pseudo-query",
        "fone": "Fourier number embedding: one [NUM] per number, value in, digits out",
    }.items():
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction,
                            default=None, required=name in RECIPE_REQUIRED, help=help_)
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
        "--save_every", type=int, required=True,
        help="write a resumable checkpoint (opt+step) every N steps; the t38 resume test and the 16h interval both need this tunable",
    )
    parser.add_argument("--name", type=str, default="pretrain", help="runs/<name>.log, ckpt_<name>.pt")
    parser.add_argument(
        "--untie_head", action="store_true",
        help="b0-17: give the LM head its own weights instead of tying to tok",
    )
    parser.add_argument(
        "--head_lr", type=float, default=None,
        help="b0-17: AdamW lr for the untied head; 0 or unset follows embed_lr "
             "(nanochat reference at d1024 is 0.003464)",
    )
    parser.add_argument(
        "--value_embed", action="store_true",
        help="A/B (4): one shared token-indexed table added to V in every MLA layer, gated by "
             "3*sigmoid over 12 residual dims (+33.6M params at 200M)")
    parser.add_argument(
        "--muon_shape_lr", action="store_true",
        help="A/B (2a): Muon lr x max(1, out/in)^0.5 per parameter shape (modded-nanogpt)")
    parser.add_argument(
        "--zero_init_out", action="store_true",
        help="A/B (3): zero-init output projections (every .o and FFN .w2), so each sublayer "
             "starts as an identity on the residual stream")
    parser.add_argument(
        "--attn_res_fp32_logits", action="store_true",
        help="accumulate the AttnRes logit dot product in fp32 (bf16 puts the mixing "
             "weights 14% off against fp64 truth)")
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
        "--allow_partial_cursor", action="store_true",
        help="resume even if a domain cursor is discarded (sample_seed/corpus mismatch); "
             "the discarded domain restarts at row 0. Default refuses (44-12).",
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
    parser.add_argument("--lr_scale", type=float, required=True, help="multiplier on every optimizer lr")
    args = parser.parse_args()
    # Apply by IS-NOT-NONE against the parser's own defaults, not by truthiness.
    # `and v` dropped every zero: --seed 0 kept Cfg.seed 42, --val_every 0 kept 500
    # despite its help text saying "0 = epoch end only", and --attn_res_blocks 0 landed
    # only because the Cfg default was already 0. Ten int flags were affected; the
    # warmdown/anneal_frac rescue loop this replaces was the same fix for two of them.
    #
    # BooleanOptionalAction switches (grad_ckpt, attn_res, attn_res_dyn_q, fone) now
    # default to None, so absence IS expressible and `is not None` is finally the right
    # test for them too -- False from --no-grad_ckpt writes False, absent writes nothing.
    # That is why the store_true exclusion below no longer needs to name them: the class
    # it protected against is gone for the four that moved. Any store_true still in the
    # parser keeps the old handling, because for those absent and False remain one value.
    #
    # store_true flags are excluded and handled below: argparse gives them False when
    # absent, not None, so is-not-None would overwrite the Cfg default on every run --
    # and Cfg.attn_res defaults to TRUE, so a blanket sweep would silently disable
    # Attention Residuals everywhere. Absence of a switch is not a request to turn it off.
    _switches = {a.dest for a in parser._actions if isinstance(a, argparse._StoreTrueAction)}
    if args.dim is not None:
        Cfg.d = args.dim  # --dim -> Cfg.d; the loop below matches on name and "d" has no flag
    for k, v in vars(args).items():
        if not hasattr(Cfg, k):
            continue
        if k in _switches:
            # A store_true is False both when absent and when the Cfg default is False,
            # so `is not None` would write False over a True default -- Cfg.attn_res
            # defaults to True, and a blanket sweep would silently disable Attention
            # Residuals on every run. Only a switch actually passed sets its field;
            # turning one OFF is what the --no_* flags below are for.
            if v:
                setattr(Cfg, k, True)
        elif v is not None:
            setattr(Cfg, k, v)
    if args.no_attn_res:
        Cfg.attn_res = False
    # --d/--heads off the default pair is a shape experiment; head_dim is not free.
    # The FlashKDA CUTLASS kernel is compiled for head_dim 128 and a mismatch is an
    # illegal memory access deep in the kernel, not a shape error at the boundary.
    if Cfg.d // Cfg.heads != 128 or Cfg.d % Cfg.heads:
        raise SystemExit(f"head_dim must be 128: d={Cfg.d} heads={Cfg.heads} -> {Cfg.d / Cfg.heads}")

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
    _mix_obj = json.load(open(mix_path, encoding="utf-8"))
    assert not _mix_obj.get("_retired"), (
        f"{Cfg.mix} is retired: {_mix_obj['_retired']}. A retired mix is frozen "
        f"(ladder_config_frozen) and its supply can no longer be made to match its demand, "
        f"so it cannot be corrected either -- pick a live mix."
    )
    fps = _assert_mix_domains(
        list(_mix_obj["domains"]),
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
    # The cursor must reach build_mix, which runs BEFORE the checkpoint is loaded at
    # :1929 (the model load needs the mix's vocab). Read just the two fields here rather
    # than reordering the load: torch.load of a 959MB checkpoint twice would cost a
    # minute per rank, and mmap keeps this to the header.
    _cursor = _cursor_fp = _cursor_seed = _pre_written = None
    _prev_total_steps = None
    if args.resume and os.path.exists(args.resume):
        try:
            _pre = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
            _cursor, _cursor_fp = _pre.get("row_cursor"), _pre.get("row_cursor_srcfp")
            _cursor_seed = _pre.get("row_cursor_seed")
            # The schedule the earlier segment ran under, for the comparison at :2231.
            # Read here rather than off the full load below so the check does not depend on
            # `ck` still being bound 180 lines later.
            _prev_total_steps = _pre.get("total_steps")
            del _pre
            # The file's own mtime, for the message below: whether a missing cursor is
            # an old format or a live write bug turns on when the file was written, and
            # nothing inside the checkpoint records that.
            _pre_written = time.strftime("%Y-%m-%d", time.gmtime(os.path.getmtime(args.resume)))
        except (OSError, RuntimeError, TypeError) as e:
            if is_main:
                print(f"resume: could not pre-read the row cursor ({e}); every domain "
                      f"starts at row 0 -- the tail of each pool stays unread", flush=True)
    if args.resume and is_main and not _cursor:
        # Two reasons a checkpoint carries no cursor, and until 2026-09-01 this message
        # asserted the innocent one. save_checkpoint dropped the cursor on EVERY save
        # (the :1289 rebind made the write block unreachable), so "predates the field"
        # was printed for checkpoints written minutes earlier by the current code --
        # a message that explains away its own symptom, addressed to precisely the
        # person who would otherwise investigate. Only the write date can tell the two
        # apart, so say which one this is instead of assuming.
        _fixed = "2026-09-01"
        _written = _pre_written or "unknown"
        _why = ("predates the field" if _written < _fixed else
                f"written {_written}, AFTER the {_fixed} cursor-save fix -- this is a "
                f"live defect, not an old format; do not reconstruct over it, report it")
        print(f"resume: checkpoint carries no row_cursor ({_why}). Every "
              "domain restarts at row 0, leaving each pool's tail unread (92% of "
              "zh_web at stage-1 weights). Reconstruct it with "
              "`python3 scripts/replay_cursor.py --ckpt <ckpt> --write` before "
              "launching stage 2 if that tail matters.", flush=True)
    tr, va = build_mix(mix_path, tok, is_main, ddp, rank, world,
                       row_cursor=_cursor, cursor_srcfp=_cursor_fp,
                       cursor_seed=_cursor_seed)
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
        # fb, 2026-09-02 / b0 / tilerl: the end-of-run save (save_checkpoint without
        # opt/step, end of main) has neither field -- resuming from it silently restarts
        # at step 0 with a cold optimizer, both invisible in the logs. Refuse at the
        # point of use. Fields are the fact, the filename is convention: milestone
        # hardlinks (ckpt_<name>.milestone_<tok>.pt) are valid resume targets without
        # stepN in the name, and a save_every checkpoint carries both fields. SFT is the
        # legitimate step-0 case and has its own loader in sft_math.py.
        missing = [k for k in ("step", "opt") if k not in ck]
        if missing:
            raise RuntimeError(
                f"refusing to resume from {args.resume}: checkpoint is missing {missing} "
                f"-- a train.py resume needs step and opt (a save_every checkpoint, "
                f"ckpt_<name>.pt.step<N>.pt, or a milestone hardlink of one). Resuming "
                f"from anything else silently restarts at step 0 with a cold optimizer."
            )
        # The cursor's step origin. save_checkpoint indexes _plan_domains, which holds
        # only THIS plan's rows, with an absolute step -- so after a resume the index
        # runs past the array and Python's slice clamps to its end, writing the
        # plan-complete count under a `row_cursor_as_of_step` label. Verified at step
        # 24000: 768,000 rows indexed into a 523,158-row plan (tilerl, 2026-09-01).
        # replay_cursor.py:44 states the same rule for its caller; this is the writer's
        # half of it.
        Cfg._plan_step_origin = resume_step
        if is_main:
            print(f"Resumed from {args.resume} (step {resume_step})", flush=True)
    fp8 = args.fp8 and amp
    amp_dtype = torch.bfloat16
    if fp8:
        raw_model = raw_model.to(torch.bfloat16)
        convert_to_fp8_compute(raw_model)
        # Off by default: this is the loss path, so an fp8 error reaches EVERY gradient rather
        # than one layer's activations. Behind a flag until the four-arm A/B judges it against
        # an in-config noise floor (docs/lessons/shipment_ab_plan.md).
        if os.environ.get("FP8_HEAD") == "1":
            if Cfg.fone:
                raise SystemExit(
                    "FP8_HEAD=1 with --fone: FoNE changes the token stream and the head's input "
                    "distribution, and the digit probe has not been run against an fp8 head. "
                    "Run the probe first or drop one of the two flags."
                )
            # Fail CLOSED. The function detects a Liger bump and returns False; discarding that
            # return would let FP8_HEAD=1 train in bf16, turning the A/B's fp8 arm into a second
            # baseline and reading "no difference" for a candidate that never ran. Same class as
            # --seed 0: a flag declaring a property the consumer relies on, false while set (b0).
            if not patch_liger_flce_fp8():
                raise SystemExit(
                    "FP8_HEAD=1 but the Liger FLCE patch did not apply -- see the line above for "
                    "which source string is missing. An explicit opt-in must take effect or refuse."
                )
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
            f"lr_scale {args.lr_scale} mix {Cfg.mix or 'flat'} fone {Cfg.fone} "
            # The EFFECTIVE values, from Cfg after the flags are applied -- not the
            # argv the caller typed. --seed 0 was silently dropped for weeks and
            # nothing in the log said so, because the log did not print the seed at
            # all: the invisibility is what let the truthiness bug live (44).
            f"seed {Cfg.seed} sample_seed {_sample_seed()}"
            + (" (pinned)" if Cfg.sample_seed is not None else " (follows seed)")
            + f" warmdown {Cfg.warmdown} anneal_frac {Cfg.anneal_frac} "
            f"val_every {Cfg.val_every} attn_every {Cfg.attn_every}"
        )
        # EVERY field that differs from the class-body default, named. The line above prints a
        # hand-picked list, and the picking is the defect: A/B (2a)'s arm flag (--muon_shape_lr)
        # appears in NO log line, so its arm identity could only be recovered from the
        # checkpoint's cfg after the fact, and a renamed or silently-dropped flag would have run
        # the BASELINE while every log line said "arm". Same shape as the --seed truthiness bug
        # noted just above, which stayed invisible for weeks because the log did not print the
        # seed -- fixed there by adding one field, which leaves the next field uncovered. This
        # prints the DIFF, so a future arm's flag needs nobody to remember it.
        _diff = {k: getattr(Cfg, k) for k, v in _CFG_DEFAULTS.items()
                 if getattr(Cfg, k, None) != v}
        runlog("cfg non-default: "
               + (", ".join(f"{k}={v!r}" for k, v in sorted(_diff.items()))
                  or "NONE (every field is at its class-body default)"))

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
        # Derived from depth, not a literal. AttnRes Full builds one compiled graph per
        # distinct source count -- 1 + 2*layers -- so the limit that fit at layers=12
        # (need 25) is one SHORT at layers=32 (need 65), and the assert below refuses
        # the launch. It refuses rather than degrading, which is right, but a constant
        # that does not move with the shape it bounds turns a shape flag into a
        # tripwire. +8 is headroom so a launch does not sit on the boundary; max(64,..)
        # so nothing shrinks below today's value at the old depth (de, 2026-09-01).
        #
        # grad_ckpt does NOT change the count, measured not assumed
        # (scripts/probe_gradckpt_sources.py): 65 distinct source counts at L=32, the
        # same sequence with checkpointing off and on. AttnRes sits outside the
        # checkpoint by construction (see _body: only [B,T] logits on the tape), so
        # recompute never re-enters it. Recorded here because the alternative is
        # sizing this constant against the wrong quantity, which is the same defect
        # one level up from the one it fixes (fb's challenge, 2026-09-01).
        _cache_need = max(64, 2 * Cfg.layers + 8)
        torch._dynamo.config.cache_size_limit = _cache_need
        torch._dynamo.config.accumulated_cache_size_limit = 4 * _cache_need
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
    # A cursor-seeded plan holds only the rows THIS segment will read -- true by
    # construction since the remainder subtraction at :1600, and true before it only for
    # domains the epoch cap happened to trim. Either way the count above is this segment's
    # steps alone, while `step` stays absolute at the resume point because it drives the LR
    # schedule, which the cursor does not relieve it of. Leaving them on different scales
    # makes total_steps smaller than the resume step, and the loop exits without running:
    # the rehearsal saw 16000/7998, zero steps (fb's ruling on tilerl's challenge).
    # Rebasing `step` instead would restart the schedule and break the t47 join, so the
    # totals are what move. With the subtraction in place this addition RESTORES the
    # original total rather than inflating past it: segment steps + resume_step == the
    # steps a fresh run would have had (scripts/test_plan_length.py, assertion 2).
    if getattr(Cfg, "_cursor_seeded", False) and resume_step:
        total_steps += resume_step
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)  # LR schedule completes within the short run
    # For save_checkpoint, and for the comparison below. Published on Cfg the same way
    # build_mix publishes _row_cursor: save_checkpoint reads state off cfg rather than
    # taking it through five call sites.
    Cfg._total_steps = total_steps
    # The schedule must not move across a resume. Loud and non-fatal on purpose: by the
    # time this runs the cards are allocated and the plan is in memory, and an inflated
    # total still trains -- it just trains a different recipe than the one the run started
    # under, which is what p200m_4b_0902 did in silence (4.87B against a 4.00B budget).
    # --max_steps is exempt: naming it IS asking for a different total.
    _prev_total = _prev_total_steps
    if _prev_total and not args.max_steps and int(_prev_total) != total_steps:
        _d = total_steps - int(_prev_total)
        runlog(
            f"SCHEDULE MOVED: this resume computes total_steps {total_steps}, the checkpoint "
            f"was written under {_prev_total} ({_d:+d}). The warmdown start and every lr_mult "
            f"move with it, so the LR shape is not the one the earlier segment trained under. "
            f"A positive delta is the plan being allocated for tokens already spent; pass "
            f"--max_steps {_prev_total} to hold the original schedule."
        )
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
    # An interrupt writes a checkpoint before it dies. The periodic save is the floor,
    # not the only line: SIGTERM (container stop, torchrun teardown, an operator's kill)
    # and SIGINT reach the main thread between bytecodes, so the handler can snapshot
    # GPU to CPU at the moment it arrives. SIGKILL and a dead node cannot -- for those
    # the periodic save is all there is, which is why it stays.
    _step_now = [resume_step]

    def _save_on_interrupt(signum, _frame):
        if is_main:
            p = ckpt_path + f".interrupt.step{_step_now[0]}"
            try:
                save_checkpoint(p, {k: v.cpu() for k, v in raw_model.state_dict().items()},
                                Cfg, VOCAB_ID, opt_snapshot(optimizers), _step_now[0])
                print(f"interrupt (signal {signum}): saved {p}", flush=True)
            except Exception as e:
                print(f"interrupt (signal {signum}): SAVE FAILED {type(e).__name__}: {e}", flush=True)
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _save_on_interrupt)
    signal.signal(signal.SIGINT, _save_on_interrupt)

    for ep in range(Cfg.epochs):
        model.train()
        perm = torch.arange(len(Xtr))  # the schedule is already in order; never reshuffle it
        # 0 whenever a row cursor SEEDED the plan, because every row in it is already
        # unread: build_mix indexes each domain from arange(used, used+want), so the plan
        # starts past what earlier segments consumed and seeking into it would skip rows a
        # second time. The rehearsal saw exactly that -- 16000 * 16 * 2 = 512,000 into a
        # 1,791,741-row plan, past the end, zero steps executed.
        #
        # Re-judged with the remainder subtraction at :1600 (de, 2026-09-02): unchanged, and
        # for the same reason. The subtraction changed HOW MANY rows the plan holds, never
        # WHERE it starts -- that was always the cursor at :1607 -- so 0 was right before it
        # and is right after. What the subtraction does fix is the count this 0 is paired
        # with: previously the plan held the full budget from the cursor onward, so `i0 = 0`
        # correctly read unread rows and there were simply too many of them.
        #
        # Without a cursor the seek is still required: the plan is then the WHOLE run's
        # rows and a resume must skip what it already read (the within-run case that has
        # always worked). `step` stays absolute either way -- it drives the LR schedule,
        # which the cursor does not relieve it of (tilerl's challenge, fb's ruling).
        i0 = 0 if getattr(Cfg, "_cursor_seeded", False) else step * Cfg.batch * Cfg.accum
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
                # EVERY rank, not just rank 0: the NaN rollback below loads good_state
                # unguarded, so a rank whose buffer never refreshed would restore itself to
                # initialisation while rank 0 restored a real step, and DDP synchronises
                # gradients, not parameters -- the divergence would not heal and would not
                # raise. Folding this into the is_main save reopened exactly that (tilerl,
                # 2026-09-02, caught before it ran).
                _step_now[0] = step
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
                    # ...but never a step something has pinned. A milestone checkpoint is
                    # one we have promised to keep, and the roller does not know that: on
                    # 2026-08-31 the 3.24B own-mix baseline was lost because step3500
                    # rotated out while its rescore sat in the lane queue, and those
                    # weights are unrepeatable. harness milestone hardlinks the file to
                    # ckpt_<run>.milestone_<token>.pt, so the pin is discoverable from
                    # disk: same inode, no list to keep in sync and go stale.
                    pinned_inodes = set()
                    for m in glob.glob(os.path.join(os.path.dirname(ckpt_path) or ".",
                                                    "*.milestone_*.pt")):
                        try:
                            pinned_inodes.add(os.stat(m).st_ino)
                        except OSError:
                            pass
                    for p in stale:
                        try:
                            if os.stat(p).st_ino in pinned_inodes:
                                if is_main:
                                    runlog(f"roller: keeping {os.path.basename(p)} -- pinned")
                                continue
                        except OSError:
                            pass
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
                    # Peak memory decides whether a batch fits at a given world, and it
                    # was not readable from a training run at all -- the 500M shape work
                    # had to read nvidia-smi, which reports the caching allocator's
                    # reservation rather than the high-water mark of live tensors. This
                    # is rank 0 only (the whole log line is is_main); ranks differ by
                    # their DDP bucket, so rank 0 is a lower bound on the worst rank, not
                    # the max. Reset each window so a late spike is not hidden by an
                    # earlier, larger one (2026-09-01).
                    peak_gib = torch.cuda.max_memory_allocated() / 2**30
                    torch.cuda.reset_peak_memory_stats()
                    # EVERY optimizer group's lr, not just optimizers[0]'s. Two sessions read
                    # this line's `lr 1.00e-02` as the embedding lr on 2026-09-02 while the
                    # embedding group was at 0.1 (Cfg.embed_lr, train.py:263) -- a log that
                    # shows one of four optimizers' lr guarantees that misread (b0-14).
                    #
                    # THE BARE `lr <value>` STAYS FIRST, and that is a constraint rather than a
                    # style choice: RunLog._STEP_RE (train.py:46) parses this line to feed
                    # trackio, and it has no test. Measured against the real regex -- a fully
                    # labeled field (`lr muon 7.00e-03 embed 1.00e-01 ...`) does NOT match, and
                    # neither does a slash-joined one, so either would silently stop every
                    # trackio metric on this line while the log still looked richer. The parser
                    # now has a test (test_step_line_parses) that fails on exactly that.
                    lrs = " ".join(
                        f"{getattr(o, 'aupai_group', f'opt{i}')} {o.param_groups[0]['lr']:.2e}"
                        for i, o in enumerate(optimizers)
                    )
                    runlog(
                        f"step {step}/{total_steps} {step / total_steps:.0%}{phase} | loss {last:.3f} "
                        f"| lr {optimizers[0].param_groups[0]['lr']:.2e} ({lrs}) "
                        f"| gnorm {grad_norm.item():.2f} "
                        f"| {step * Cfg.batch * Cfg.accum * Cfg.seq * world / 1e9:.2f}B tok "
                        f"| {tps / 1e3:.0f}K tok/s/gpu | MFU {mfu * 100:.0f}% "
                        f"| peak {peak_gib:.2f}GiB | ETA {eta / 3600:.1f}h"
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
        save_checkpoint(ckpt_path, raw_model.state_dict(), Cfg, VOCAB_ID,
                        opt_snapshot(optimizers), step)
        print(f"saved {ckpt_path}")
        runlog.plot()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
