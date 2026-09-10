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

#: Seed for readout 6's row-checksum probe vector (ProductKeyMemory.row_probe). A CONSTANT, not
#: Cfg.seed: the projection must be identical on every rank and across a resume so the checksums
#: are comparable, and Cfg.seed varies by arm, which would make two arms' counts incomparable and
#: a resumed run's first count meaningless. Not persistent in the checkpoint either -- the vector
#: is recomputed from this number, so it is the same after a reload.
_ROW_PROBE_SEED = 20260905

#: Rows per block in ProductKeyMemory.row_checksums. 65,536 rows at d=1024 is a 256 MiB fp32
#: temporary, fixed regardless of table size -- against 5.45 GiB at side 1195 and 8.00 at 1448 for
#: the unchunked `weight.float()`, which would put the diagnostic's own allocation into the peak
#: that decides whether the arm fits (4c's instruction, 2026-09-05).
_ROW_CHECKSUM_BLOCK = 65536


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
    and for the same reason -- head_dim must stay 128 (train.py's `head_dim must be 128` raise in
    main(), the FlashKDA CUTLASS kernel), so a mixer at inner=768 carries 6 heads while its sibling
    at inner=256 carries 2, and neither can be derived from cfg alone.
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
            # the same document. cu indexes the FLAT B*T stream (train.py's doc_cu_seqlens) while h is
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


class _CfgView:
    """A read-only cfg with a few fields overridden, for per-layer architecture choices.

    Attribute reads fall through to the real cfg, so a field added later needs no change here and
    a typo in an override name cannot silently shadow anything -- the override dict is explicit
    and small. Not a copy: cfg carries tensors and a copy would duplicate them per layer.
    """

    def __init__(self, cfg, **over):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_over", over)

    def __getattr__(self, k):
        o = object.__getattribute__(self, "_over")
        return o[k] if k in o else getattr(object.__getattribute__(self, "_cfg"), k)

    def __setattr__(self, k, v):
        raise AttributeError(
            f"_CfgView is read-only; setting {k!r} here would change one layer's view of the "
            f"config and leave the checkpoint's recorded cfg describing a different model.")


def masked_attend(sc, mask, v):
    """softmax(sc masked by `mask`) @ v, finite in the BACKWARD as well as the forward.

    THE BUG THIS REPLACES, found 2026-09-09 by a CPU forward/backward of the p1 config before it
    reached a card: `torch.nan_to_num(torch.softmax(sc.masked_fill(~mask, -inf), -1) @ v)` gives
    the right forward value and a NaN GRADIENT. nan_to_num rewrites the output, not the graph, so
    a row that is entirely -inf still differentiates through softmax as 0/0 and BmmBackward0
    propagates NaN into every parameter upstream. Measured: 89 of 102 parameter tensors
    non-finite after one backward; forward finite throughout, which is why no forward-only test
    or parity check could see it.

    Fully-masked rows are not an edge case in either caller. A compressed block is visible only
    once its LAST member is at or before the query, so every query before its document's first
    complete block has one -- in HCA at m'=128 that is the first 128 positions of every document,
    and in CSA's compress branch the first m. CSA has carried this since b0-35 and it never fired
    because CSA has never been trained; its first backward would have NaN'd.

    The fix gives a dead row a uniform FINITE score and zeroes its output afterwards, so the
    softmax never sees an all--inf row. Same forward value as before -- zero for a query with no
    visible context, which is the correct attention result -- with a defined gradient of zero
    instead of NaN.
    """
    alive = mask.any(-1, keepdim=True)
    sc = sc.masked_fill(~mask, float("-inf"))
    sc = torch.where(alive, sc, torch.zeros_like(sc))
    return torch.softmax(sc, dim=-1) @ v * alive.to(v.dtype)


def _doc_blocks(q, cu, m):
    """Per-document m-token block assignment and the visibility mask.

    The shared arithmetic of pool_per_doc (mean pooling, HCA) and entries_per_doc
    (learned compression, CSA2). Two of this file's recorded incidents are a
    compressed block that straddled a document boundary and a block gated on its
    FIRST position instead of its last, so the subtle part has ONE implementation
    and the reducers differ.

    Returns (block_id, doc, vis, NB): per-position block id (B,T), per-position
    document id (B,T), the (B,T,NB) visibility mask (block complete and past the
    query, same document, real slot), and the block count.
    """
    B, T, _, _ = q.shape
    pos = torch.arange(B * T, device=q.device)
    cu_l = cu.to(pos.dtype)
    doc = torch.bucketize(pos, cu_l[1:], right=True).view(B, T)   # (B,T) doc id per pos
    doc_start = cu_l[doc]                                         # (B,T)
    nb_doc = (cu_l[1:] - cu_l[:-1] + m - 1) // m                  # blocks per doc
    offset = torch.cumsum(nb_doc, 0) - nb_doc                     # first block id per doc
    first_doc = doc[:, 0]
    # per-batch block id: doc's global offset + within-doc block, minus the batch's first
    # doc's offset, so block ids run 0..NB-1 inside each batch
    block_id = (offset[doc] + (pos.view(B, T) - doc_start) // m
                - offset[first_doc][:, None])                    # (B,T)
    NB = int((offset[doc[:, -1]] + nb_doc[doc[:, -1]]
              - offset[first_doc]).max().item())

    # block metadata: last position (the blk_last <= t visibility rule, unchanged),
    # owning document (cross-document refusal), and the pad-slot marker (NB is a max over
    # batches, so shorter batches carry empty slots that must stay invisible)
    blk_last = torch.full((B, NB), -1, device=q.device, dtype=torch.long)
    blk_last.scatter_reduce_(1, block_id, pos.view(B, T), reduce="amax",
                             include_self=True)
    blk_doc = torch.zeros(B, NB, dtype=torch.long, device=q.device)
    blk_doc.scatter_(1, block_id, doc)
    ar = torch.arange(T, device=q.device)
    vis = ((blk_last >= 0)[:, None, :]
           & (blk_last[:, None, :] <= ar[None, :, None])
           & (blk_doc[:, None, :] == doc[:, :, None]))       # (B,T,NB)
    return block_id, doc, vis, NB


def pool_per_doc(q, kh, vh, cu, m):
    """Mean-pool K/V into blocks of `m` PER DOCUMENT, with the visibility mask.

    Lifted verbatim out of CompressedSparseAttention._forward_packed so CSA and HCA share one
    implementation. They differ only in `m` (16 vs 128) and in what they do with the result, and
    the part that is subtle is identical: two of this file's recorded incidents are a compressed
    block that straddled a document boundary and a block gated on its FIRST position instead of
    its last, and neither raises -- the second makes the model better at training time and only
    surfaces as a generation collapse nobody can attribute. A second copy of these lines is a
    second chance to reintroduce them.

    Returns (kc, vc, vis, block_id, doc): pooled K/V as (B,H,NB,D), the (B,T,NB) visibility mask,
    the per-position block id and the per-position document id.
    """
    B, T, H, D = q.shape
    block_id, doc, vis, NB = _doc_blocks(q, cu, m)

    # --- pool K/V per document block ----------------------------------------------
    idx = block_id[:, None, :, None].expand(B, H, T, D)
    kc = kh.new_zeros(B, H, NB, D)
    vc = vh.new_zeros(B, H, NB, D)
    kc.scatter_add_(2, idx, kh)
    vc.scatter_add_(2, idx, vh)
    # count REAL positions per block in float32: bf16 cannot hold every count up to T
    # exactly, and a partial block must be a mean over what it holds
    cnt = q.new_zeros(B, NB, dtype=torch.float32)
    cnt.scatter_add_(1, block_id, torch.ones(B, T, device=q.device))
    cnt = cnt.clamp(min=1)
    kc = kc / cnt[:, None, :, None].to(kc.dtype)
    vc = vc / cnt[:, None, :, None].to(vc.dtype)
    return kc, vc, vis, block_id, doc


def entries_per_doc(q, kh, vh, cu, m, compress_k, compress_v):
    """Learned non-overlapping m-token KV entries (DeepSeek-V4.1 CSA2, tech report §2.3).

    Same blocks and visibility as pool_per_doc; the reducer is a learned linear over
    the zero-padded m-token block instead of a mean. The paper removes V4-CSA's
    overlap and absolute positional embedding but does not publish the m>1 combine
    weights, so a head-shared Linear over the concatenated m positions is the
    chosen form (de-103) -- a gated weighted mean is the alternative. A document's
    tail block keeps its real positions and zero-pads the rest. m=1 is the paper's
    uncompressed-main-KV special case: compress_k/v are None and the gathered block
    IS the KV, unchanged.

    Returns (kc, vc, vis, doc): compressed K/V as (B,H,NB,D), the (B,T,NB)
    visibility mask and the per-position document id.
    """
    B, T, H, D = q.shape
    block_id, doc, vis, NB = _doc_blocks(q, cu, m)
    cu_l = cu.to(torch.long)
    off = (torch.arange(B * T, device=q.device).view(B, T) - cu_l[doc]) % m
    flat = (block_id * m + off)[:, None, :, None].expand(B, H, T, D)
    bk = kh.new_zeros(B, H, NB * m, D)
    bv = vh.new_zeros(B, H, NB * m, D)
    bk.scatter_add_(2, flat, kh)
    bv.scatter_add_(2, flat, vh)
    if compress_k is None:  # m=1: the gathered block is the KV itself
        return bk.view(B, H, NB, D), bv.view(B, H, NB, D), vis, doc
    kc = compress_k(bk.reshape(B * H, NB, m * D)).view(B, H, NB, D)
    vc = compress_v(bv.reshape(B * H, NB, m * D)).view(B, H, NB, D)
    return kc, vc, vis, doc


class HeavilyCompressedAttention(nn.Module):
    """HCA: dense attention over aggressively compressed KV, no sparse selection.

    facts/deepseek_v4.json#dsv4.hybrid_attention -- V4 interleaves HCA with CSA after the first
    two layers. HCA is the cheap half: compression m'=128 against CSA's 16, and every compressed
    entry is attended densely, so there is no top-k, no full-resolution branch and no gate. One
    branch means one number, which is also why it needs no learned mixing.

    Two consequences of m'=128 that are easy to miss:

    A query in the first 128 positions of its document sees NO complete block of that document,
    because a block is visible only once its last member is at or before the query -- the same
    `blk_last <= t` rule the compress branch of CSA uses, and for the same causal reason. In CSA
    the sliding window covers that prefix exactly; HCA has no window, so those queries attend to
    nothing and the softmax over an all-masked row is undefined. They are given a zero output
    here rather than a NaN, which is correct as an attention result (no visible context) and is
    the reason for the nan_to_num rather than an oversight.

    That prefix is why HCA is INTERLEAVED and never used alone: at m'=128 the first 128 tokens of
    every document are handled entirely by the CSA layers around it. A model of pure HCA layers
    would be blind over exactly the span most code completions live in.
    """

    def __init__(self, cfg, h, hd):
        super().__init__()
        self.h, self.hd = h, hd
        self.m = int(getattr(cfg, "hca_compress", 128))
        if self.m < 1:
            raise ValueError(f"hca needs a positive hca_compress, got {self.m}")
        self.scale = hd ** -0.5

    def forward(self, q, k, v, cu=None):
        B, T, H, D = q.shape
        qh = q.transpose(1, 2)
        kh, vh = k.transpose(1, 2), v.transpose(1, 2)
        if cu is None:
            # Unpacked: one document, so the per-doc arithmetic degenerates to a plain
            # contiguous blocking. Built through the same helper by handing it the trivial cu
            # rather than a second implementation of the visibility rule.
            cu = torch.arange(0, B * T + 1, T, device=q.device, dtype=torch.int32)
        kc, vc, vis, _, _ = pool_per_doc(q, kh, vh, cu, self.m)
        sc = (qh @ kc.transpose(-1, -2)) * self.scale            # B,H,T,NB
        y = masked_attend(sc, vis[:, None], vc)
        return y.transpose(1, 2)                                # B,T,H,D


class CompressedSparseAttention(nn.Module):
    """CSA: coarse attention over compressed KV blocks, top-k selection, plus a sliding window.

    b0-35. Three branches over the SAME q/k/v the dense path builds, summed by a learned gate:

      compress  every `m` positions of K/V are pooled into one entry, so a query attends to
                T/m entries instead of T and sees the whole prefix at reduced resolution
      select    the compressed scores pick the top-k BLOCKS, and the query attends to those
                blocks at FULL resolution -- coarse scores choosing where to look closely
      window    the last `n_win` tokens at full resolution, unconditionally

    Shape follows the NSA/DSA family (arXiv 2502.11089 §3). The window branch is what makes the
    other two safe to sparsify: local structure is where a missed token hurts most, and it is
    the branch that costs nothing to keep exact.

    WHY A REFERENCE IMPLEMENTATION AND NOT A KERNEL. This is O(T^2) in the select branch's
    gather and materialises the coarse score matrix; it is for deciding whether the ARCHITECTURE
    is worth a kernel, not for training at speed. b0-35 asks whether CSA-with-SWA replaces gated
    MLA at 200M active, and that question is answered by loss at matched tokens, not by tok/s.
    The per-step cost goes in facts/efficiency.json so the trade is on the record before anyone
    proposes a kernel. A fast wrong answer to the architecture question would cost more than the
    slow right one.

    DOCUMENT ISOLATION WITH `cu`. `cu` marks document boundaries in the packed stream. Blocks
    are built PER DOCUMENT -- a compressed entry never pools two documents -- and every branch
    is masked to the query's own document: compress and select by the block's owning document,
    window by the same-document mask. The residual rule is one rule across all three branches:
    a document's tail is its own partial block, pooled over the positions it holds and visible
    only once its last member is past the query (the same blk_last <= t rule as the unpacked
    path); a query before that sees no block of its document and is covered by the window
    branch. A block that straddled a boundary would be a second rule, and a second rule is
    where the leak comes back. eff.kda_document_isolation_violated is the same failure in the
    KDA short_conv, and it went unnoticed for days.
    """

    def __init__(self, cfg, h, hd):
        super().__init__()
        self.h, self.hd = h, hd
        self.m = int(getattr(cfg, "csa_compress", 16))       # positions per compressed entry
        self.topk = int(getattr(cfg, "csa_topk", 8))         # compressed blocks attended at full res
        self.n_win = int(getattr(cfg, "csa_window", 256))    # sliding-window width
        if self.m < 1 or self.topk < 1 or self.n_win < 1:
            raise ValueError(f"csa needs positive csa_compress/csa_topk/csa_window, got "
                             f"{self.m}/{self.topk}/{self.n_win}")
        self.scale = hd ** -0.5
        # CSA2 (DeepSeek-V4.1, tech report §2.3): the V4.1 replacement for this class,
        # behind cfg.csa2. Learned non-overlapping m-token entries replace the mean pool,
        # a dedicated indexer (indexer-Q from the layer input H, indexer-K projected from
        # the entries) selects top-k, and ONE softmax runs over [selected entries ; SWA
        # window keys]. No branch gate, no full-resolution select branch. Full Mode only:
        # cross-layer Reuse/Reindex is a later step. The paper publishes no m>1 combine
        # weights, so the compressor is a head-shared Linear over the m positions -- a
        # choice to A/B, not a paper value. The indexer's hard top-k is non-differentiable,
        # so selection carries a straight-through softmax for its gradient -- also an
        # engineering choice, flagged in the PR, since the paper does not say how the
        # indexer learns.
        self.csa2 = bool(getattr(cfg, "csa2", False))
        if self.csa2:
            if not getattr(cfg, "csa", False):
                raise ValueError(
                    "csa2=True needs csa=True: csa2 is the V4.1 variant of the csa arm, "
                    "not a second arm -- GatedMLA constructs the module only when csa is on")
            self.m = int(getattr(cfg, "csa2_m", 8))
            self.topk = int(getattr(cfg, "csa2_top_k", 64))
            self.n_win = int(getattr(cfg, "csa2_n_win", 128))
            if self.m < 1 or self.topk < 1 or self.n_win < 1:
                raise ValueError(f"csa2 needs positive m/top_k/n_win, got "
                                 f"{self.m}/{self.topk}/{self.n_win}")
            self.di = int(getattr(cfg, "csa2_indexer_dim", 64))  # indexer low dim
            self.ih = int(getattr(cfg, "csa2_indexer_heads", 4))  # indexer heads (<= h)
            if self.di < 1 or self.h % self.ih:
                raise ValueError(f"csa2_indexer_dim must be positive and csa2_indexer_heads="
                                 f"{self.ih} must divide heads={self.h}")
            # m=1 is the paper's uncompressed-main-KV special case: no compressor.
            self.compress_k = nn.Linear(self.hd * self.m, self.hd) if self.m > 1 else None
            self.compress_v = nn.Linear(self.hd * self.m, self.hd) if self.m > 1 else None
            self.indexer_q = nn.Linear(cfg.d, self.ih * self.di, bias=False)
            self.ik_weight = nn.Parameter(torch.empty(self.ih, self.hd, self.di))
            nn.init.normal_(self.ik_weight, std=0.02)
        else:
            # The branch gate. Zero-init so all three branches start at sigmoid(0)=0.5 and none is
            # born dominant; the model discovers the mixture rather than inheriting my guess of it.
            self.branch_gate = nn.Linear(hd, 3, bias=True)
            nn.init.zeros_(self.branch_gate.weight)
            nn.init.zeros_(self.branch_gate.bias)

    def forward(self, q, k, v, cu=None, x=None):
        if self.csa2:
            return self._forward_csa2(q, k, v, cu, x)
        if cu is not None:
            return self._forward_packed(q, k, v, cu)
        B, T, H, D = q.shape
        qh = q.transpose(1, 2)                                # B,H,T,D
        kh, vh = k.transpose(1, 2), v.transpose(1, 2)
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()

        # --- compress: mean-pool K/V over blocks of m -------------------------------------
        # A BLOCK IS VISIBLE ONLY WHEN IT IS COMPLETE AND PAST, and this is where my first
        # version leaked. I gated on the block's FIRST position (`blk_first <= t`), reasoning
        # that a query should see a block it is inside. But the pooled entry is a mean over all
        # m members, so a query at t=5 with m=4 read block 1 = mean(k[4..7]) and thereby saw
        # tokens 6 and 7 -- its own future. Measured by perturbation: k[5] += 7 moved outputs at
        # positions 0..4 by 1.44 (max|delta|), which is a causal leak, not a rounding artifact.
        # The window branch was clean at 0.0 in the same test, which is what localised it.
        # A leak here is invisible in the loss: it makes the model BETTER at training time and
        # only shows up as a generation-time collapse nobody can attribute.
        # So the mask is `blk_last <= t`: every member of a visible block is at or before the
        # query. The prefix a query cannot see coarsely is covered exactly by the window branch.
        nb = (T + self.m - 1) // self.m
        pad = nb * self.m - T
        kp = F.pad(kh, (0, 0, 0, pad))
        vp = F.pad(vh, (0, 0, 0, pad))
        # count the REAL positions per block so the last, partly-padded block is a mean over
        # what it holds rather than a mean diluted by zeros
        cnt = F.pad(torch.ones(T, device=q.device), (0, pad)).view(nb, self.m).sum(-1)
        kc = kp.view(B, H, nb, self.m, D).sum(3) / cnt[None, None, :, None]
        vc = vp.view(B, H, nb, self.m, D).sum(3) / cnt[None, None, :, None]
        sc = (qh @ kc.transpose(-1, -2)) * self.scale          # B,H,T,nb
        blk_last = (torch.arange(nb, device=q.device) * self.m + self.m - 1)[None, :]
        vis = blk_last <= torch.arange(T, device=q.device)[:, None]            # T,nb
        # MASK sc IN PLACE, BEFORE the topk below reads it. The select branch picks its blocks
        # from THIS tensor, so an unmasked score here lets topk rank a block the query cannot
        # see; `sel &= vis` then drops it and the query silently attends to fewer blocks than
        # topk promised. Moving this line into masked_attend (2026-09-09) did exactly that and
        # test_arch_compat's CSA causality case caught it -- the masking is load-bearing twice
        # and only one of the two uses is the attention below.
        sc = sc.masked_fill(~vis, float("-inf"))
        # the first m-1 queries see NO complete block. They are covered by the window branch,
        # so this branch contributes exactly zero -- see masked_attend for why the obvious
        # nan_to_num spelling of that is correct forward and NaN backward.
        y_cmp = masked_attend(sc, vis, vc)

        # --- select: top-k compressed blocks, attended at full resolution ------------------
        kk = min(self.topk, nb)
        idx = sc.topk(kk, dim=-1).indices                       # B,H,T,kk
        sel = torch.zeros(B, H, T, nb, dtype=torch.bool, device=q.device)
        sel.scatter_(-1, idx, True)
        sel &= vis[None, None]                    # a -inf block can still be picked when kk>visible
        # expand block selection to positions, then intersect with the causal mask
        pos_sel = sel.repeat_interleave(self.m, dim=-1)[..., :T]
        full = (qh @ kh.transpose(-1, -2)) * self.scale
        m_sel = pos_sel & causal[None, None]
        # a query whose selected set is empty contributes nothing (masked_attend)
        y_sel = masked_attend(full, m_sel, vh)

        # --- window: the last n_win tokens, exact -----------------------------------------
        ar = torch.arange(T, device=q.device)
        m_win = causal & ((ar[:, None] - ar[None, :]) < self.n_win)
        y_win = torch.softmax(full.masked_fill(~m_win[None, None], float("-inf")), dim=-1) @ vh

        g = torch.sigmoid(self.branch_gate(q)).transpose(1, 2).unsqueeze(-1)   # B,H,T,3,1
        y = g[..., 0, :] * y_cmp + g[..., 1, :] * y_sel + g[..., 2, :] * y_win
        return y.transpose(1, 2)                                # B,T,H,D

    def _forward_csa2(self, q, k, v, cu, x):
        """CSA2: learned entries, indexer top-k, ONE softmax over entries + SWA window.

        Full Mode only (tech report §2.3.1): the layer builds its own entries and runs
        its own indexer; cross-layer Reuse/Reindex is a later step. One code path for
        packed and unpacked inputs -- the unpacked case is one document per batch row,
        handed to entries_per_doc through the trivial cu, as HCA does.

        `x` is the layer input H; the indexer-Q is projected from it and stays
        DISTINCT from the attention Q (fb ruling 2026-09-10: the distinction is
        load-bearing). GatedMLA passes it; a caller that does not gets ValueError
        rather than a silent fallback to q.
        """
        if x is None:
            raise ValueError("csa2 needs the layer input x for the indexer-Q; pass it")
        B, T, H, D = q.shape
        qh = q.transpose(1, 2)                                # B,H,T,D
        kh, vh = k.transpose(1, 2), v.transpose(1, 2)
        if cu is None:
            cu = torch.arange(0, B * T + 1, T, device=q.device, dtype=torch.int32)
        kc, vc, vis, doc = entries_per_doc(q, kh, vh, cu, self.m,
                                           self.compress_k, self.compress_v)
        NB = kc.shape[2]

        # --- indexer: score entries, mask BEFORE top-k, select -----------------------
        # The mask is load-bearing twice, as in CSA: topk ranks from THIS tensor, so an
        # unmasked score lets it pick a block the query cannot see (a future block or
        # another document's), and the attention below must never see it either.
        # Indexer heads are FEWER than attention heads (ae-2: 4 vs 8): each indexer head
        # scores the mean of its group's entries, and its selection serves the group.
        iq = self.indexer_q(x).view(B, T, self.ih, self.di)
        kc_g = kc.view(B, self.ih, self.h // self.ih, NB, D).mean(dim=2)
        ik = torch.einsum("bhnd,hde->bhne", kc_g, self.ik_weight)   # B,ih,NB,di
        isc = torch.einsum("bthd,bhnd->bhtn", iq, ik) * (self.di ** -0.5)
        isc = isc.masked_fill(~vis[:, None], float("-inf"))
        kk = min(self.topk, NB)
        sel = torch.zeros(B, self.ih, T, NB, dtype=torch.bool, device=q.device)
        sel.scatter_(-1, isc.topk(kk, dim=-1).indices, True)
        sel &= vis[:, None]                    # a -inf block can still be picked when kk>visible
        sel = sel.repeat_interleave(self.h // self.ih, dim=1)    # B,H,T,NB
        # straight-through: hard selection in the forward, softmax over the indexer
        # scores in the backward so the indexer gets a gradient (the hard top-k has none)
        alive = vis[:, None].any(-1, keepdim=True)               # B,1,T,1
        s = torch.where(alive, isc, torch.zeros_like(isc))
        soft_sel = (torch.softmax(s, dim=-1) * alive.to(isc.dtype)
                    ).repeat_interleave(self.h // self.ih, dim=1)  # B,H,T,NB
        ste = sel.to(soft_sel.dtype) + soft_sel - soft_sel.detach()

        # --- one softmax over [selected entries ; SWA window keys] ------------------
        sc = (qh @ kc.transpose(-1, -2)) * self.scale            # B,H,T,NB
        ar = torch.arange(T, device=q.device)
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        same = doc[:, :, None] == doc[:, None, :]
        m_win = same & causal & ((ar[:, None] - ar[None, :]) < self.n_win)
        full = (qh @ kh.transpose(-1, -2)) * self.scale
        sc_cat = torch.cat([sc, full], dim=-1)                   # B,H,T,NB+T
        mask_cat = torch.cat([vis[:, None] & sel,
                              m_win[:, None].expand(B, H, T, T)], dim=-1)
        # masked_attend's body, split so the straight-through selector gates the ENTRY
        # WEIGHTS: y = (w_entry * ste) @ vc + w_win @ vh. Gating values instead would
        # materialise a (T,NB,D) tensor the matmul never holds.
        alive = mask_cat.any(-1, keepdim=True)
        s = sc_cat.masked_fill(~mask_cat, float("-inf"))
        s = torch.where(alive, s, torch.zeros_like(s))
        w = torch.softmax(s, dim=-1) * alive.to(sc_cat.dtype)
        y = (w[..., :NB] * ste) @ vc + w[..., NB:] @ vh
        return y.transpose(1, 2)                                # B,T,H,D

    def _forward_packed(self, q, k, v, cu):
        """The cu path: per-document blocks, every branch masked to the query's document.
        See the class docstring for the residual rule. cu indexes the flat B*T stream
        (train.py's doc_cu_seqlens: every row start and every position after an <eos>);
        documents never span rows, so per-batch block numbering is exact."""
        B, T, H, D = q.shape
        qh = q.transpose(1, 2)                                # B,H,T,D
        kh, vh = k.transpose(1, 2), v.transpose(1, 2)
        kc, vc, vis, block_id, doc = pool_per_doc(q, kh, vh, cu, self.m)
        NB = kc.shape[2]      # the select branch's top-k is bounded by the block count
        ar = torch.arange(T, device=q.device)

        # --- compress: same three lines as the unpacked path, over per-document blocks ----
        sc = (qh @ kc.transpose(-1, -2)) * self.scale            # B,H,T,NB
        # Masked in place BEFORE the topk below reads it, for the same reason as the unpacked
        # path: the select branch ranks blocks from THIS tensor, and an unmasked score lets it
        # pick a block belonging to another document. Caught by test_arch_compat's CSA
        # cross-document case when this line briefly moved inside masked_attend.
        sc = sc.masked_fill(~vis[:, None], float("-inf"))
        y_cmp = masked_attend(sc, vis[:, None], vc)

        # --- select: top-k blocks, attended at full resolution, same document only --------
        kk = min(self.topk, NB)
        sel = torch.zeros(B, H, T, NB, dtype=torch.bool, device=q.device)
        sel.scatter_(-1, sc.topk(kk, dim=-1).indices, True)
        sel &= vis[:, None]                    # a -inf block can still be picked when kk>visible
        pos_sel = sel.gather(3, block_id[:, None, :, None].expand(B, H, T, T))
        same = doc[:, :, None] == doc[:, None, :]
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        full = (qh @ kh.transpose(-1, -2)) * self.scale
        m_sel = pos_sel & same[:, None] & causal[None, None]
        y_sel = masked_attend(full, m_sel, vh)

        # --- window: the last n_win tokens inside the document ---------------------------
        m_win = same & causal & ((ar[:, None] - ar[None, :]) < self.n_win)
        y_win = torch.softmax(full.masked_fill(~m_win[:, None], float("-inf")),
                              dim=-1) @ vh

        g = torch.sigmoid(self.branch_gate(q)).transpose(1, 2).unsqueeze(-1)   # B,H,T,3,1
        y = g[..., 0, :] * y_cmp + g[..., 1, :] * y_sel + g[..., 2, :] * y_win
        return y.transpose(1, 2)                                # B,T,H,D


class PartialRoPE(nn.Module):
    """RoPE on the LAST `dims` of each head only (facts/deepseek_v4.json#dsv4.partial_rope).

    Partial and not full, because the leading hd-dims are what CSA and HCA pool. A compressed
    entry is a mean over m positions; mean-pooling a rotated vector mixes m different rotation
    angles and the result encodes no usable position at all. Leaving the leading dims
    position-free is what lets the same K carry content into the pooled branches and position
    into the exact ones.

    POSITIONS ARE PER DOCUMENT. `cu` marks boundaries in the packed stream and every other
    branch in this file is masked to the query's own document; a position counted from the row
    start would make RoPE encode the distance between two tokens in DIFFERENT documents. That
    error raises nothing and is invisible in the loss -- the same shape as the CSA compress-branch
    leak recorded above, and as eff.kda_document_isolation_violated. The arithmetic is CSA's own:
    `flat_pos - cu[doc(flat_pos)]`.

    NOT APPLIED to the attention OUTPUT. dsv4.partial_rope also specifies RoPE at position -i on
    each output because V4's compressed KV entries double as values, so a pooled value carries the
    absolute position baked into its keys. Our CSA pools K and V through separate projections
    (kv_up produces both, then they are pooled independently) and only K is rotated here, so V
    never carries an angle and there is nothing to undo. If V is ever rotated, that correction
    becomes required -- this comment is the record of why it is absent, not an oversight.
    """

    def __init__(self, dims, hd, base=10000.0):
        super().__init__()
        if dims <= 0 or dims % 2 or dims > hd:
            raise ValueError(
                f"rope_dims={dims} must be positive, even, and at most head_dim={hd}: RoPE rotates "
                f"pairs, so an odd width leaves one dimension with no partner and a width above "
                f"the head has no dimensions left to rotate.")
        self.dims = dims
        inv = base ** (-torch.arange(0, dims, 2, dtype=torch.float32) / dims)
        # persistent=False: derived from (dims, base), so it is not state. A persistent buffer
        # would add a key to every checkpoint and make rope_dims un-changeable after the fact.
        self.register_buffer("inv_freq", inv, persistent=False)

    def positions(self, B, T, cu, device):
        """Per-document positions for the packed stream; plain arange when cu is absent."""
        if cu is None:
            return torch.arange(T, device=device).expand(B, T)
        pos = torch.arange(B * T, device=device)
        cu_l = cu.to(pos.dtype)
        doc = torch.bucketize(pos, cu_l[1:], right=True)
        return (pos - cu_l[doc]).view(B, T)

    def forward(self, x, pos):
        """x: (B, T, H, hd). Rotates x[..., -dims:] in place-free fashion, returns the whole."""
        head, tail = x[..., :-self.dims], x[..., -self.dims:]
        ang = pos.to(torch.float32)[:, :, None, None] * self.inv_freq  # B,T,1,dims/2
        cos, sin = torch.cos(ang), torch.sin(ang)
        a, b = tail.float().chunk(2, dim=-1)
        rot = torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)
        return torch.cat([head, rot.to(x.dtype)], dim=-1)


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
        # CSA (b0-35), OFF by default and None when off. A submodule that is None registers no
        # parameters, so a flag-off model's state_dict and parameter count are unchanged and every
        # existing checkpoint loads without a shim. Constructing it unconditionally and skipping it
        # in forward would add tensors to the checkpoint of every run that does not use it.
        if getattr(cfg, "csa2", False) and not getattr(cfg, "csa", False):
            raise ValueError(
                "csa2=True needs csa=True: csa2 is the V4.1 variant of the csa arm, not a "
                "second arm -- GatedMLA constructs the module only when csa is on")
        self.csa = CompressedSparseAttention(cfg, self.h, self.hd) if getattr(
            cfg, "csa", False) else None
        # HCA (dsv4.hybrid_attention): the other half of V4's hybrid attention. CSA and HCA are
        # both constructed only when their flag is on, for the same state_dict reason as above.
        self.hca = HeavilyCompressedAttention(cfg, self.h, self.hd) if getattr(
            cfg, "hca", False) else None
        if self.csa is not None and self.hca is not None:
            raise ValueError(
                "csa and hca are both on for this layer. V4 INTERLEAVES them across layers -- it "
                "does not stack them inside one -- so a layer is one or the other. HybridLM picks "
                "per layer from cfg.attn_kind; setting both on cfg means every layer would run "
                "both branches and the arm would be neither architecture.")
        # PARTIAL RoPE (dsv4.partial_rope). Zero = off, which is what every existing checkpoint
        # was trained under (dsv4.nope_rope_break: this stack is NoPE and position comes from
        # KDA). It carries no parameters and its one buffer is non-persistent, so turning it on
        # does not change any state_dict key -- but it DOES change what the weights mean, which
        # is why a checkpoint records the flag rather than the layer inferring it.
        rd = int(getattr(cfg, "rope_dims", 0) or 0)
        self.rope = PartialRoPE(rd, self.hd) if rd else None

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
        # PARTIAL RoPE, applied AFTER the norm and to q/k only. After, because rms_norm rescales
        # the whole head and would otherwise rescale the rotated tail by a factor that depends on
        # the angle, making the same relative offset mean different things at different absolute
        # positions. q/k only, because V carries no angle -- see PartialRoPE's docstring for why
        # V4's -i output correction is not needed here.
        if self.rope is not None:
            p = self.rope.positions(B, T, cu, x.device)
            q, k = self.rope(q, p), self.rope(k, p)
        if self.hca is not None:
            y = self.hca(q, k, v, cu)
            return self.o(y.reshape(B, T, self.d) * torch.sigmoid(gate))
        # CSA (b0-35). ONE `if` ON THE UNCHANGED PATH, placed after q/k/v are built and normed
        # so that with the flag off every tensor below is bit-for-bit what it was before this
        # branch existed -- no reordering, no re-association, nothing recomputed. The parity
        # case in scripts/test_arch_compat.py asserts that against a saved reference, because
        # "I did not mean to change the default" is not a measurement.
        if self.csa is not None:
            y = self.csa(q, k, v, cu, x)
            y = y.reshape(B, T, self.d)
            return self.o(y * torch.sigmoid(gate))
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

    HEAD_DIM STAYS 128 on both halves (train.py's `head_dim must be 128` raise, the FlashKDA CUTLASS
    kernel), which is what
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
                f"(train.py refuses any other head_dim in main()), so the split cannot be taken in "
                f"fractions of a head.")
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

    def __init__(self, n_values, d, top_k=32, key_dim=None, sparse=True, query_norm="none"):
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
        # QUERY NORMALISATION, and "none" reproduces the M1/M2/M3 arms bit-for-bit -- that is what
        # makes this flag safe to land before it is chosen. Added because those three arms were
        # stopped 2026-09-05 under readout 4 with key-usage collapse (M1 touched 0.0945 at step
        # 1000, gini 0.9192), and the query goes straight from this Linear into the two half-key
        # top-k with nothing bounding its scale.
        #
        # "bn": Lample et al. 2019 section 3.3 BatchNorm the query network's output and report it
        # as what keeps key usage spread. Over the flat (B*T, 2*key_dim) query it is a per-feature
        # statistic across tokens, so it carries no position information and cannot leak across the
        # causal boundary. Its cost is that the statistic is batch-dependent: under DDP each rank
        # normalises by its own tokens, and eval uses running stats, so train and eval are not the
        # same function.
        # "l2": per-half L2-normalise query AND keys, then scale by a learned temperature. Bounds
        # every score to [-temp, temp] so no single key's norm can dominate the top-k, with no
        # batch statistic -- identical under DDP, at batch 1, and in eval. The temperature is
        # stored as a log so the optimizer cannot drive it through zero; init 1/sqrt(key_dim) is
        # the scale a dot product of two unit vectors in key_dim dimensions would otherwise have.
        if query_norm not in ("none", "l2", "bn"):
            raise ValueError(
                f"query_norm must be none/l2/bn, got {query_norm!r}. An unrecognised value must "
                f"raise rather than fall back to 'none': a typo would silently run the control "
                f"arm under a name that says otherwise, which is how an arm reports a clean null "
                f"for a change it never applied.")
        self.query_norm = query_norm
        if query_norm == "bn":
            self.q_bn = nn.BatchNorm1d(2 * self.key_dim)
        elif query_norm == "l2":
            self.q_log_temp = nn.Parameter(
                torch.tensor(math.log(math.sqrt(float(self.key_dim)))))
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
        # KEY USAGE COUNTS, which `touched` cannot supply. The ledger's key_gini is required and
        # is a real Gini over how OFTEN each half-key is selected; a bool "was it reached" tells
        # us reached-or-not and cannot produce one. The two diagnostics separate the two collapse
        # shapes: a pool can be 90% touched (pool_touched_frac healthy) while one key wins every
        # lookup (key_gini near 1), and readout 4 exists to catch exactly that. 2 x side counts,
        # so 4096 int32 at M1 against the 1.07B-parameter table -- free.
        self.register_buffer("key_hits", torch.zeros(2, side, dtype=torch.long), persistent=False)
        self.register_buffer("windows", torch.zeros((), dtype=torch.long), persistent=False)
        # READOUT 6, WRITE EFFECTIVENESS (4c's ruling 2026-09-05). The four diagnostics above are
        # all functions of the SELECTION distribution, which the keys produce -- and the keys are
        # a normal Linear with an fp32 master. So a table whose every update rounds away in bf16
        # reads 100% healthy on all four: touched counts reads, not effective writes. This is the
        # instrument that sees the difference.
        #
        # A PER-ROW CHECKSUM, NOT A COPY OF THE ROWS. The direct form -- keep the previous diag
        # step's table and diff it -- is 2 bytes per PARAMETER: 4 GiB at M1. The dot of each row
        # with one fixed random vector is 4 bytes per ROW: 4 MiB at M1, 8 MiB at 1448^2, three
        # orders of magnitude less for the same question. It misses only an update whose elements
        # cancel in that projection exactly, which for a real gradient is measure zero.
        #
        # fp32 CHECKSUM OVER A bf16 TABLE ON PURPOSE: the sum of 1024 bf16 products accumulated in
        # bf16 would round away the very small change this exists to detect, so the reduction is
        # the one place the precision must not follow the table's.
        #
        # SEEDED, so the projection is the same on every rank and across a resume. A per-rank
        # vector would make the checksums incomparable, and the DDP reduction below compares them.
        _g = torch.Generator().manual_seed(_ROW_PROBE_SEED)
        self.register_buffer("row_probe", torch.randn(d, generator=_g, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("row_sum_prev", torch.zeros(n_values, dtype=torch.float32),
                             persistent=False)
        # -1 rather than 0: "no previous checksum yet" and "no row changed" are different
        # findings, and a 0 here would report the first diag row as 0.0 changed -- the same
        # number a frozen table gives.
        self.register_buffer("rows_changed", torch.full((), -1, dtype=torch.long),
                             persistent=False)
        # An EXPLICIT armed flag, not "is row_sum_prev still all zeros". That test would be a
        # second, weaker definition of the same state, and it reads wrong in the one case that
        # matters: if a table ever produced an all-zero projection, the baseline would be taken
        # twice and the first real comparison silently skipped.
        self.register_buffer("row_probe_armed", torch.zeros((), dtype=torch.bool),
                             persistent=False)

    @torch.no_grad()
    def row_checksums(self):
        """Per-row fp32 projection of the value table onto the fixed probe vector.

        CHUNKED OVER ROW BLOCKS, and that is a peak-memory requirement rather than a speed one.
        `weight.float()` on the whole table allocates a full-size fp32 temporary -- verified, not
        assumed: .float() on a bf16 tensor never shares storage, so it is 4 B/param, which is
        5.45 GiB at side 1195 and 8.00 at 1448. That allocation would land at the diag step, on
        top of a peak already measured at 88.15 GiB reserved of 95.22 usable, so the instrument
        would decide whether the arm fits. A block of 65,536 rows costs 256 MiB at d=1024 instead,
        independent of table size.

        BOTH OPERANDS FORCED TO fp32, not assumed from the buffer's dtype. `row_probe` is a
        buffer, so `model.to(torch.bfloat16)` -- which train.py applies to the whole model
        under --fp8 (`raw_model = raw_model.to(torch.bfloat16)`) -- casts it along with everything
        else. The projection would then accumulate in
        bf16, where each partial sum over d terms is ~d times the size of a one-ULP change, and the
        smallest real updates would round out of the checksum: readout 6 would report a barely
        moving table as frozen, which is the failure it exists to detect. Caught by the one-ULP
        case in test_arch_compat, on a module cast to bf16 exactly as the arms cast it.
        """
        w = self.values.weight.detach()
        probe = self.row_probe.float()
        out = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
        for i in range(0, w.shape[0], _ROW_CHECKSUM_BLOCK):
            j = min(i + _ROW_CHECKSUM_BLOCK, w.shape[0])
            out[i:j] = w[i:j].float() @ probe
        return out

    @torch.no_grad()
    def note_row_changes(self):
        """Count rows whose checksum moved since the last call; arm the next comparison.

        Called at diag steps only, on every rank (the caller reduces the count across ranks).
        The FIRST call establishes the baseline and returns -1: with no previous checksum there
        is nothing to compare, and returning 0 would be indistinguishable from a table that
        changed in no row -- which is exactly the failure this measures.

        THE STORE IS RE-FLOATED for the same reason row_checksums forces its operands: `to(bfloat16)`
        casts row_sum_prev as well, and copy_ into a bf16 buffer would round every stored checksum
        back to bf16 precision -- discarding the resolution the fp32 projection was computed for and
        leaving the comparison exactly as blind as a bf16 accumulator.
        """
        cur = self.row_checksums()
        if self.row_sum_prev.dtype is not torch.float32:
            self.row_sum_prev = self.row_sum_prev.float()
        if not bool(self.row_probe_armed):
            self.row_sum_prev.copy_(cur)
            self.row_probe_armed.fill_(True)
            return -1
        n = int((cur != self.row_sum_prev).sum())
        self.row_sum_prev.copy_(cur)
        self.rows_changed.fill_(n)
        return n

    def forward(self, x):
        B, T, d = x.shape
        h = self.n_mem(x)
        q = self.query(h).view(B * T, 2, self.key_dim)
        k0, k1 = self.keys[0], self.keys[1]
        # THE ONE BRANCH THE QUERY-NORM FLAG ADDS. "none" leaves q and the keys exactly as the
        # M1/M2/M3 arms had them, which is what test_arch_compat asserts bitwise -- a flag whose
        # default changed the arm would make every earlier measurement incomparable.
        if self.query_norm == "bn":
            # BatchNorm1d over the flat (B*T, 2*key_dim) query: the two halves are normalised
            # together because they are one projection's output, and splitting them would give the
            # halves different statistics for no reason. In eval it uses running stats.
            q = self.q_bn(q.reshape(B * T, 2 * self.key_dim)).view(B * T, 2, self.key_dim)
        elif self.query_norm == "l2":
            # BOTH SIDES normalised, then one learned temperature. Normalising only the query
            # would leave a key free to win every top-k by growing its norm, which is the same
            # concentration by another route.
            q = F.normalize(q, dim=-1) * self.q_log_temp.exp()
            k0, k1 = F.normalize(k0, dim=-1), F.normalize(k1, dim=-1)
        # Each half scores its own sqrt(V) keys and keeps top_k.
        s0 = torch.einsum("nk,ck->nc", q[:, 0], k0)
        s1 = torch.einsum("nk,ck->nc", q[:, 1], k1)
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
                # MEAN OVER THE WINDOW, accumulated, not the last batch's value. The ledger field
                # says "mean over the window"; writing the last micro-batch's entropy under that
                # name would report one batch as if it were 100 steps.
                self.last_entropy += -(p * p.log()).sum(-1).mean()
                self.windows += 1
                # Counted on the SELECTED pairs (i0/i1 gathered through `sel`), not on the
                # per-half topk. The per-half topk keeps top_k of each side unconditionally, so
                # counting there would report every half-key as used top_k/side of the time by
                # construction and the Gini would measure the constant, not the model.
                _s0 = sel // self.top_k
                _s1 = sel % self.top_k
                self.key_hits[0] += torch.bincount(
                    i0.gather(1, _s0).reshape(-1), minlength=self.side)
                self.key_hits[1] += torch.bincount(
                    i1.gather(1, _s1).reshape(-1), minlength=self.side)
        return self.out(F.silu(self.gate(h)) * read)

    def diagnostics(self, reset=True):
        """Charter readout 4: touched fraction, top-k entropy and key-usage Gini for the window.

        Reset by default because the readout is "fraction touched IN THE WINDOW": a cumulative
        counter converges to 1.0 and stops being able to see a collapse, which is the one thing
        this diagnostic exists to catch (a pool below 20% at step 1000 stops the arm).
        """
        n = int(self.touched.sum())
        frac = n / self.n_values
        nwin = max(1, int(self.windows))
        ent = float(self.last_entropy) / nwin
        # THE GINI IS OVER KEY-USAGE COUNTS, and it is the reason key_hits exists. `touched` is a
        # bool -- reached or not -- and no Gini follows from it; the ledger field is a real
        # concentration measure and the two diagnostics separate the two collapse shapes: a pool
        # can be 90% touched while one key wins every lookup. Computed on the pooled 2*side
        # counts, sorted, by the standard formula G = (2*sum(i*x_i)/(n*sum(x))) - (n+1)/n. Zero
        # counts are kept: a key never used is the observation, not a missing datum.
        c = self.key_hits.reshape(-1).float().sort().values
        tot = float(c.sum())
        if tot <= 0:
            gini = 0.0   # nothing was read this window; touched_fraction 0 is the finding
        else:
            k = c.numel()
            i = torch.arange(1, k + 1, dtype=torch.float32, device=c.device)
            gini = float((2.0 * (i * c).sum() / (k * tot)) - (k + 1) / k)
            gini = min(1.0, max(0.0, gini))
        if reset:
            self.touched.zero_()
            self.key_hits.zero_()
            self.last_entropy.zero_()
            self.windows.zero_()
        return {"touched_fraction": frac, "touched_rows": n, "n_values": self.n_values,
                "topk_entropy": ent, "topk_entropy_max": math.log(self.top_k),
                "key_gini": gini, "windows": nwin}


class MoEFFN(nn.Module):
    """Fine-grained MoE replacing the dense SwiGLU: N routed experts + 1 always-on shared expert.

    WHY THIS EXISTS (charter docs/standards/moe_0905.md, prereg runs/prereg.jsonl#moe_0905).
    The memory program measured that what moves loss is how many parameters a token can REACH.
    A memory table adds reachable parameters in PARALLEL to the FFN; this replaces the FFN with a
    sparse one, so total capacity grows at constant active compute rather than at constant
    parameters. E1 is 24 routed + 1 shared per layer, all 12 layers, 0.801B total against the
    control's 206M.

    EQUAL ACTIVE COMPUTE IS THE DESIGN CONSTRAINT, not an outcome. SwiGLU costs 3*d*ffn_hidden of
    matmul per token (w13 is 2*d*ffn, w2 is d*ffn). Each expert at width w costs 3*d*w, and a
    token traverses top_k routed experts plus the shared one, so parity requires
    (top_k + 1) * w == ffn_hidden. At ffn_hidden 3072 that means (k+1) must divide 3072 = 2^10*3,
    i.e. k in {3,5,7,11,15}; the charter's cell is k=3, w=768. __init__ REFUSES a config that
    misses parity rather than training an arm whose loss delta confounds sparsity with FLOPs --
    the bench's import-time version of this assertion certified a configuration its timed code
    never ran (b0's review of moe_dispatch_bench, 2026-09-05), so here it is checked against the
    shapes actually constructed.

    THE ACTIVATION IS K3 SiTU-GLU, NOT TEXTBOOK SwiGLU, and this is the subtlest correctness
    requirement in the module. model.SwiGLU is beta1*tanh(a/beta1)*sigmoid(b) then
    beta2*tanh(w2(gate)/beta2) with beta1=4.0, beta2=25.0 (SwiGLU.forward). An expert computing
    the textbook a*sigmoid(b) would make the arm differ from the control in ACTIVATION as well as
    in sparsity, and readout 1's delta would not be attributable to the sparsity it is registered
    to measure. The betas are read off a real SwiGLU instance rather than re-typed, so a change to
    the dense module propagates here instead of silently diverging. scripts/test_moe_module.py's
    tied-weights witness is what proves it: one expert holding the dense module's own weights must
    reproduce its output.

    AUX-LOSS-FREE LOAD BALANCING, with the bias affecting SELECTION ONLY. A per-expert bias is
    added to the affinity scores for the top-k choice, and the gating value multiplied into the
    expert output comes from the UN-BIASED score (DeepSeek-V3, arXiv:2412.19437 section 2.1.2:
    "Note that the bias term is only used for routing"). Biasing the gate as well would change
    the function the arm computes, not merely which experts it picks. gamma = 0.001 and the
    sequence-wise balance alpha = 1e-4 are OUR pre-registered values BORROWED from
    facts/moe.json -- paper-reported at 1,966,080 tokens per routed expert per step against our
    32,768, a factor of 60.0, so they are a starting point and not an inherited result. Neither is
    tuned after seeing a curve; if readout 4's stop rule fires, that is the finding.

    THE BIAS IS PERSISTENT STATE (4c's ruling 2026-09-05). It lives in the state dict, and a
    checkpoint without the key loads zeros. The cards are the RL team's and can be recalled
    mid-run, so a resume is the expected case rather than a contingency, and a balancer that
    restarted cold after a resume would make readout 4 unattributable. It is N floats per layer.

    DISPATCH: torch._grouped_mm over experts sorted by assignment, with a sort-and-loop fallback.
    Three API constraints, each measured by tilerl on card 7 (2026-09-05) after every group size
    reported a shape refusal:
      - mat2 must be TRANSPOSED; a contiguous (E, K, N) operand is refused on CUDA with
        "Expected mat2 to be transposed". Weights are stored (E, N, K) and transposed at use, so
        no copy happens. NOTE, measured here 2026-09-05: CPU ACCEPTS a contiguous mat2, so a
        CPU-only test cannot catch a regression on this -- the layout is asserted directly in
        scripts/test_moe_module.py instead of being left to the op to reject.
      - `offs` is CUMULATIVE ENDS, not starts.
      - the grouped backward refuses a grad with stride [0, 0], which is what expanding a scalar
        (`.sum().backward()`) produces. Not this module's problem in training -- the real loss
        produces a real grad -- but tests must pass an explicit contiguous grad.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d
        self.n_routed = int(getattr(cfg, "moe_experts", 0))
        self.top_k = int(getattr(cfg, "moe_top_k", 3))
        self.n_shared = int(getattr(cfg, "moe_shared", 1))
        w = int(getattr(cfg, "moe_expert_ffn", 0) or 0)
        # THE LATENT VARIANT, off unless moe_latent is set (prereg moe_0905 amendment 13). The
        # routed experts run in a PROJECTED width: down-project d -> d_latent once per token,
        # route and run the experts entirely at d_latent, up-project back to d. The point is the
        # dispatch payload and the expert matmul dimensions, both of which shrink with d_latent;
        # at the registered cell (d_latent 384) the payload is 0.38x the 24-expert arm's.
        #
        # THE SHARED EXPERT KEEPS ITS OWN WIDTH, which is why moe_shared_ffn exists. Under the
        # latent variant the equal-active budget is spent on three things instead of two, so the
        # shared expert cannot also be at expert_ffn without breaking parity -- and letting it
        # move is what makes an EXACT parity-and-parameter match possible at all (measured: with
        # the shared width pinned at 768 there is no integer solution, and with it free there are
        # three at 1.000x the 24-expert arm's routed parameters). It defaults to expert_ffn, so
        # every existing config keeps its shape.
        self.d_latent = int(getattr(cfg, "moe_latent", 0) or 0)
        w_sh = int(getattr(cfg, "moe_shared_ffn", 0) or 0) or w
        if w <= 0:
            raise ValueError("moe_expert_ffn must be > 0 when moe_experts > 0")
        if self.top_k < 1 or self.top_k > self.n_routed:
            raise ValueError(
                f"moe_top_k {self.top_k} must be in [1, moe_experts {self.n_routed}]")
        if self.d_latent < 0:
            raise ValueError(f"moe_latent {self.d_latent} must be >= 0 (0 = off)")
        if self.d_latent and self.n_shared != 1:
            raise ValueError(
                f"moe_latent with moe_shared {self.n_shared}: the latent parity below counts ONE "
                f"shared expert at moe_shared_ffn. Use moe_shared 1, or extend the parity "
                f"expression before using more")
        # EQUAL-ACTIVE PARITY, refused rather than warned. This is the property the whole arm
        # rests on: an arm at 1.25x active compute measures compute, not sparsity.
        if self.d_latent:
            # COUNTED IN MULTIPLIES, not in widths, because the latent arm spends its budget on
            # three different shapes: the two projections (d x d_latent each, every token), the
            # routed experts at the LATENT width (3 * d_latent * expert_ffn per expert, top_k of
            # them), and the shared expert at the FULL width (3 * d * moe_shared_ffn). The dense
            # SwiGLU costs 3 * d * ffn_hidden. Equality of those two totals is the constraint;
            # the width-product form above cannot express it.
            active_mul = (2 * d * self.d_latent
                          + 3 * self.top_k * self.d_latent * w
                          + 3 * d * w_sh)
            dense_mul = 3 * d * cfg.ffn_hidden
            if active_mul != dense_mul:
                raise ValueError(
                    f"latent MoE active multiplies {active_mul} != dense {dense_mul}: "
                    f"2*d*d_latent ({2 * d * self.d_latent}) + 3*top_k*d_latent*expert_ffn "
                    f"({3 * self.top_k * self.d_latent * w}) + 3*d*shared_ffn "
                    f"({3 * d * w_sh}) must equal 3*d*ffn_hidden exactly, or the arm differs "
                    f"from the control in FLOPs as well as in sparsity and readout 1's delta is "
                    f"unattributable. The registered cell is d_latent 384, expert_ffn 2048, "
                    f"moe_shared_ffn 512 at d 1024 / ffn_hidden 3072")
        else:
            active = (self.top_k + self.n_shared) * w
            if active != cfg.ffn_hidden:
                raise ValueError(
                    f"active FFN width {active} != dense ffn_hidden {cfg.ffn_hidden}: "
                    f"(moe_top_k {self.top_k} + moe_shared {self.n_shared}) * moe_expert_ffn {w} "
                    f"must equal ffn_hidden exactly, or the arm differs from the control in FLOPs as "
                    f"well as in sparsity and readout 1's delta is unattributable. "
                    f"ffn_hidden {cfg.ffn_hidden} admits moe_top_k in "
                    f"{sorted(k - self.n_shared for k in range(2, cfg.ffn_hidden + 1) if cfg.ffn_hidden % k == 0 and k - self.n_shared >= 1)[:8]}"
                    f" at the matching widths")
        self.shared_ffn = w_sh
        self.expert_ffn = w
        # STORED (E, N, K) AND TRANSPOSED AT USE. _grouped_mm needs a transposed mat2 on CUDA;
        # keeping the parameter in (E, N, K) means the transpose is a view and no copy happens in
        # the hot path. Held as one stacked Parameter per matrix rather than a ModuleList: the
        # grouped op consumes per-expert matrices with a row offset, so a list would be stacked
        # on every call and the arm would pay for the stacking.
        # THE EXPERTS' INPUT/OUTPUT WIDTH IS d_latent UNDER THE LATENT VARIANT, d otherwise. This
        # one name is the whole shape difference: w13 is (E, 2w, de) and w2 is (E, de, w), so the
        # grouped matmuls shrink in their contracted dimension and the rows dispatched to each
        # expert are de wide instead of d.
        de = self.d_latent or d
        self.w13 = nn.Parameter(torch.empty(self.n_routed, 2 * w, de))
        self.w2 = nn.Parameter(torch.empty(self.n_routed, de, w))
        # THE PROJECTIONS, latent variant only. No bias and no activation between them and the
        # experts: they exist to change the width the routed path runs at, not to add a
        # nonlinearity, and an activation here would make the arm a different architecture than
        # the one registered. Initialised at their own fan_in like everything else below.
        if self.d_latent:
            self.down = nn.Linear(d, self.d_latent, bias=False)
            self.up = nn.Linear(self.d_latent, d, bias=False)
        # THE SHARED EXPERT is a plain dense FFN, always at the FULL width d and its own
        # moe_shared_ffn -- it is not routed, so there is nothing for a latent projection to save
        # on it, and putting it behind the projection would change what every token traverses.
        self.sh13 = nn.Linear(d, 2 * w_sh, bias=False)
        self.sh2 = nn.Linear(w_sh, d, bias=False)
        self.router = nn.Linear(d, self.n_routed, bias=False)
        # THE SAME INITIALISATION THE DENSE FFN GETS, per expert. nn.Linear's default is
        # kaiming_uniform on fan_in, so an expert of width w initialised as if it were width
        # ffn_hidden would start at the wrong scale and the arm's first steps would measure the
        # initialisation. Each expert matrix is initialised at ITS own fan_in.
        #
        # `de`, NOT `d`, FOR w13 UNDER THE LATENT VARIANT, and this line is why the variant needed
        # a test rather than a read: w13 contracts over the expert INPUT width, which the latent
        # projection changes from d to d_latent. Leaving `d` here would init 24 experts at
        # 1/sqrt(1024) when their fan_in is 384 -- a 1.63x scale error, silent, and exactly the
        # "first steps measure the initialisation" failure this comment already warns about.
        for t, fan_in in ((self.w13, de), (self.w2, w)):
            bound = 1.0 / math.sqrt(fan_in)
            with torch.no_grad():
                t.uniform_(-bound, bound)
        # THE BETAS COME FROM THE DENSE MODULE, not re-typed here. If SwiGLU's bounds change, the
        # experts follow; a literal 4.0/25.0 in this file would diverge silently and the tied-
        # weights witness would then be the only thing standing between that and a launched arm.
        _ref = SwiGLU(cfg)
        self.beta1, self.beta2 = _ref.beta1, _ref.beta2
        self.gamma = float(getattr(cfg, "moe_bias_gamma", 0.001))
        self.balance_alpha = float(getattr(cfg, "moe_balance_alpha", 1e-4))
        # PERSISTENT, and zeros for a checkpoint that predates it (4c's ruling). Not a Parameter:
        # it is not updated by a gradient, so handing it to an optimizer would let weight decay
        # and momentum act on a control-loop variable.
        #
        # fp32 EXPLICITLY, and it is load-bearing rather than tidy. This buffer inherits the
        # module's dtype otherwise, which is bf16 under --fp8, and bf16's spacing at magnitude
        # 0.5 is 0.00391 -- FOUR TIMES the gamma 0.001 step. MEASURED on
        # ckpt_b0_moe48_8b.pt.step1000: 10 of 12 layers had 2-3 DISTINCT bias values left across
        # 48 experts, i.e. the differential signal that actually routes had been rounded away.
        # It is 48 floats x 12 layers = 2,304 values, so fp32 costs 9 KiB and buys the
        # resolution the control loop is built on. A bf16 saved tensor casts into this buffer on
        # load (verified: dtype stays fp32, values preserved), so old checkpoints still resume.
        #
        # THE dtype= ARGUMENT ALONE DOES NOTHING, and saying so here is the whole point: this
        # buffer was ALREADY fp32 at construction (torch.zeros defaults to fp32), and train.py
        # :3134 and :3162 both call `raw_model.to(torch.bfloat16)`, which casts every floating
        # buffer -- so it became bf16 before step 0 either way. That is why the failed run's
        # checkpoint holds bf16. _apply below is what actually keeps it fp32; the argument is
        # kept only so the declaration states the intent (4c, reviewing 87ef5985).
        self.register_buffer("expert_bias", torch.zeros(self.n_routed, dtype=torch.float32),
                             persistent=True)
        # READOUT 4's counters. Non-persistent: they describe a window, and a resume that restored
        # a half-finished window would report a fraction over a denominator from another run.
        self.register_buffer("tokens_per_expert", torch.zeros(self.n_routed, dtype=torch.long),
                             persistent=False)
        self.register_buffer("windows", torch.zeros((), dtype=torch.long), persistent=False)
        # THE BALANCER'S OWN COUNTER, separate from readout 4's. tokens_per_expert accumulates over
        # a 200-step window and is cleared by diagnostics(reset=True), so it cannot feed a per-step
        # control loop: the bias would move on the window's total (and only when a diagnostics step
        # happened to fire) rather than on this step's load. Non-persistent for the same reason
        # tokens_per_expert is -- a resume restoring a half-finished step's counts would update the
        # bias from another run's load.
        self.register_buffer("step_tokens_per_expert", torch.zeros(self.n_routed, dtype=torch.long),
                             persistent=False)
        # THE MICRO-BATCH SCRATCH SLOT AND ITS FORWARD COUNTER (b0-27). forward accumulates the
        # cumulative counters as it always did, and additionally records this micro-batch's counts
        # plus how many times it ran for them; commit_token_counts() subtracts the surplus once
        # per micro-batch. Gradient checkpointing runs forward twice and nothing inside it can
        # tell the passes apart, so the correction happens after backward instead. Non-persistent
        # for the same reason the others are, and additionally because they hold at most one
        # micro-batch of state -- a resume that restored them would subtract a surplus belonging
        # to another run's micro-batch.
        self.register_buffer("micro_tokens_per_expert",
                             torch.zeros(self.n_routed, dtype=torch.long), persistent=False)
        self.register_buffer("micro_forwards", torch.zeros((), dtype=torch.long),
                             persistent=False)
        # The sequence-wise balance loss for the current forward, read by train.py and added to
        # the loss there. Kept as an attribute rather than returned so Block.forward's signature
        # and the AttnRes sublayer protocol stay unchanged.
        self.aux_loss = None

    def _situ(self, a, b, w2_apply):
        """model.SwiGLU's bounded activation, applied to whatever the second matmul is."""
        gate = self.beta1 * torch.tanh(a / self.beta1) * torch.sigmoid(b)
        return self.beta2 * torch.tanh(w2_apply(gate) / self.beta2)

    @torch.no_grad()
    def _apply(self, fn, recurse=True):
        """Cast/move the module while keeping expert_bias's fp32 VALUES, not just its dtype.

        `raw_model.to(torch.bfloat16)` (train.py, under --fp8 and again under --bf16) walks every
        floating buffer, so register_buffer's `dtype=torch.float32` is overwritten before step 0 --
        the buffer was already fp32 at construction and became bf16 anyway, which is why
        ckpt_b0_moe48_8b.pt.step1000 holds bf16. Same trap the store's checksum path records under
        "THE STORE IS RE-FLOATED"; this is that pattern for the control loop.

        WHY fp32 IS REQUIRED HERE, not preferred: the bias is an accumulator stepped by
        gamma 0.001, and bf16's spacing at magnitude 0.5 is 0.00391 -- four times the step. On the
        run that died, 10 of 12 layers had 2-3 distinct values left across 48 experts: the
        DIFFERENTIAL, which is the only part topk can see, had been rounded away while the
        common mode kept climbing. 2,304 values, 9 KiB.

        THE ORIGINAL TENSOR IS KEPT, NOT RE-FLOATED AFTER THE FACT, and that distinction is the
        whole point (4c, reviewing 2f95a797). `fn(eb).float()` restores the DTYPE and has already
        destroyed the VALUES: fn rounds to bf16's grid first, so .float() returns rounded numbers
        in an fp32 box and every dtype assertion passes. MEASURED on [0.5, 0.501, 0.502] ->
        [0.5, 0.5, 0.50390625]: three distinct biases became two, which is exactly the
        differential collapse this override exists to prevent. train.py loads the checkpoint at
        :3043 and casts at :3134/:3162 -- LOAD BEFORE CAST -- so that rounding would be applied
        once per resume to every value the previous run accumulated.
        So: keep the pre-cast tensor and follow only the DEVICE.

        Hooked at _apply rather than at .to() because .to(), .cuda(), .float() and DDP's own
        movement all route through here, so a single override covers every path instead of the
        one call site I happened to look at. This REPLACES the buffer entry rather than mutating
        in place (an earlier version of this docstring claimed in-place, wrongly); that is safe
        because every reader reaches the buffer through _buffers/getattr, never through a
        long-lived reference to the tensor object.
        """
        orig = self._buffers.get("expert_bias")
        out = super()._apply(fn, recurse)
        if orig is not None:
            moved = out._buffers["expert_bias"]
            out._buffers["expert_bias"] = orig.to(device=moved.device, dtype=torch.float32)
        return out

    def commit_token_counts(self):
        """Remove the recompute surplus from this micro-batch's counts. Once per micro-batch.

        Called by train.py after loss.backward() -- a site that is once-per-micro-batch BY
        CONSTRUCTION and, critically, AFTER the recompute, which happens during backward. forward
        has by then run either once (no checkpointing) or twice (checkpointed), and recorded which
        in micro_forwards. This subtracts the extra copies:

            surplus = micro_forwards - 1        # 0 or 1 in practice
            tokens_per_expert      -= surplus * micro_tokens_per_expert
            step_tokens_per_expert -= surplus * micro_tokens_per_expert
            windows                -= surplus

        THE ARITHMETIC STAYS ON THE DEVICE. `int(self.micro_forwards)` would read the surplus to
        the host, and this runs once per MoE layer per micro-batch -- 12 layers x accum 4 = 48
        device-to-host syncs per step on the 1.5b-a0.2b-e48_8b shape, in the training loop's hot
        path. clamp_min(0) on the tensor gives the same result with no sync, verified equal for
        both the 1-pass and 2-pass cases. The `if surplus > 0` shortcut was dropped for the same
        reason: branching on a tensor's value requires reading it, and multiplying by a zero
        surplus is already a no-op.

        WHY NOT DETECT RECOMPUTE INSIDE forward: `self.training` and `torch.is_grad_enabled()`
        are both true in the recompute pass, and torch exposes no public recompute marker. Counting
        the passes and correcting afterwards needs no such marker.

        WHY NOT SIMPLY ASSIGN IN forward: tried and rejected. It is correct for train.py and it
        silently zeroes every OTHER caller -- scripts/profile_moe_step.py runs a bare m(x) and
        reads step_tokens_per_expert to pick its perturbation victim, and
        scripts/test_moe_bias_wiring.py case D asserts 1 forward gives `want` and 2 give `2*want`.
        Neither uses checkpointing and neither calls this method, so for them accumulate-in-forward
        is already right. A fix that breaks the callers it does not know about is the shape where
        a new guard lands after its callers.

        Measured before this existed (b0-27, card 7): grad_ckpt on read exactly 2.0000x the
        grad_ckpt-off counts, uniform across every expert on all four MoE layers, `windows` 2 -> 4.

        WHAT THE DOUBLE COUNT DID AND DID NOT CORRUPT, since the guard this unblocks claimed
        otherwise. NOT the balancer: update_bias steps on torch.sign(counts - mean), doubling every
        count doubles the mean too, so err doubles and sign() is invariant to a positive scale --
        the bias step was bit-identical, exact ties included. NOT any measured diagnostic:
        usage_frac, used_experts, entropy_norm and load_gini are functions of the count
        DISTRIBUTION, and load_gini is normalised by tot, so a uniform factor cancels. It corrupted
        `tokens` and `window_steps`, the two absolute counts -- and since both doubled together,
        even their ratio was right. A NON-uniform doubling would have moved the statistics; the
        measurement says it was uniform. So this fix makes the bookkeeping honest and unblocks the
        configuration; it corrects no published number, because no arm has ever run checkpointed.
        """
        with torch.no_grad():
            surplus = (self.micro_forwards - 1).clamp_min(0)
            self.tokens_per_expert -= surplus * self.micro_tokens_per_expert
            self.step_tokens_per_expert -= surplus * self.micro_tokens_per_expert
            self.windows -= surplus
            self.micro_forwards.zero_()

    def update_bias(self, counts):
        """The aux-loss-free bias step: -gamma where overloaded, +gamma where underloaded.

        NOT A GRADIENT (arXiv:2412.19437 section 2.1.2). The step is a fixed size on the SIGN of
        the load error, so an expert that is slightly over and one that is wildly over move by
        the same gamma -- that is the paper's rule, and a magnitude-proportional version would be
        a different balancer than the one whose value we borrowed.

        Called by train.py once per optimizer step with the step's summed counts, NOT per
        micro-batch: with accum 2 a per-micro-batch update would move the bias twice per step and
        the effective gamma would silently be 2x the registered one.

        WIRED 2026-09-06. Before that this method existed and NOTHING in train.py called it: all
        three E1' arms (dense has no router; moe24 and moe48) finished 3815 steps with every
        expert_bias buffer identically zero (absmax 0.0, 12/12 on ckpt_b0_e1p_moe48.pt), so their
        load-balance numbers are the sequence-wise loss acting alone and say nothing about this
        balancer at any gamma.
        """
        mean = counts.float().mean()
        err = counts.float() - mean
        self.expert_bias -= self.gamma * torch.sign(err).to(self.expert_bias.dtype)
        # ZERO-MEAN BY CONSTRUCTION, and this is a FIX for a measured failure, not a nicety.
        # sign() returns 0 only on an exact tie, and with 48 experts on integer counts ties are
        # rare, so the 48 signs almost never cancel: the mean bias moves gamma*(48-2k)/48 per
        # step for k experts above mean, and any persistent asymmetry INTEGRATES WITHOUT BOUND.
        # There is no clamp anywhere in this update.
        #
        # MEASURED on b0_moe48_8b, steps 500 -> 1000: common mode +0.1067 -> +0.4997 (x4.68,
        # a drift of 0.786 gamma/step, i.e. only ~5 of 48 experts reading below mean each step)
        # while the per-layer SPREAD collapsed 0.0992 -> 0.0101 (x0.10). Load gini went 0.0935 ->
        # 0.5959 and the run was stopped on its pre-registered band at step 1000.
        #
        # THE COMMON MODE CANNOT AFFECT ROUTING AT ALL: selection is
        # `(affinity + expert_bias).topk(top_k)` below, and topk is INVARIANT to a constant added
        # to every entry. So the runaway term was pure windup -- it did nothing for balance while
        # consuming the bf16 resolution that the spread needed. Subtracting the mean therefore
        # changes NO routing decision at the moment it is applied, and stops the integrator dead.
        self.expert_bias -= self.expert_bias.mean()

    def forward(self, x):
        B, T, d = x.shape
        n = B * T
        flat = x.reshape(n, d)
        # ROUTER LOGITS AT fp32. The affinity scores decide WHICH parameters a token reaches, so a
        # tie broken differently by bf16 rounding is not a tolerance question -- it changes the
        # function. The charter's kernel question 3 asks the same thing of fp8.
        logits = self.router(flat).float()
        affinity = torch.softmax(logits, dim=-1)
        # THE BIAS ENTERS SELECTION ONLY. `affinity` (un-biased) supplies the gate below; the
        # biased score chooses. Reversing this is the single easiest way to get a plausible-looking
        # module that computes the wrong function.
        sel = (affinity + self.expert_bias.float()).topk(self.top_k, dim=-1).indices
        gate = affinity.gather(1, sel)                                    # (n, top_k) UN-biased
        gate = gate / gate.sum(-1, keepdim=True).clamp_min(1e-9)
        if self.training:
            # THE SEQUENCE-WISE BALANCE LOSS (eq. 17), alpha = 1e-4. Complementary to the bias,
            # not an alternative: the bias balances across the step's batch, this term catches
            # extremes inside ONE sequence. Computed per sequence and averaged over the batch,
            # which is what "sequence-wise" means -- computing it over the flattened batch would
            # be a batch-wise loss under a sequence-wise name.
            f = torch.zeros(B, self.n_routed, device=x.device, dtype=torch.float32)
            sel_b = sel.view(B, T * self.top_k)
            f.scatter_add_(1, sel_b, torch.ones_like(sel_b, dtype=torch.float32))
            f = f * (self.n_routed / (T * self.top_k))
            P = affinity.view(B, T, self.n_routed).mean(1)
            self.aux_loss = self.balance_alpha * (f * P).sum(-1).mean()
        else:
            self.aux_loss = None
        # Sorted dispatch: one contiguous row block per expert, which is what _grouped_mm reads.
        e_of_row = sel.reshape(-1)
        order = torch.argsort(e_of_row)
        counts = torch.bincount(e_of_row, minlength=self.n_routed)
        if self.training or torch.is_grad_enabled():
            with torch.no_grad():
                # ACCUMULATE AS BEFORE, and additionally record what this micro-batch's counts
                # were and how many times this forward ran for it (b0-27, 2026-09-06).
                #
                # Gradient checkpointing runs this forward TWICE per micro-batch -- measured on
                # card 7: tokens_per_expert read exactly 2x, uniformly per expert on every MoE
                # layer (84 -> 168, 98 -> 196, ...), with `windows` 2 -> 4. No condition here can
                # tell the passes apart: `self.training` and `torch.is_grad_enabled()` are both
                # true in the recompute, and torch publishes no recompute marker.
                #
                # So the surplus is CORRECTED downstream instead of prevented here.
                # commit_token_counts() subtracts (micro_forwards - 1) copies, which is right for
                # one pass (subtract nothing) and for two (subtract one) without either site
                # needing to know which happened. Accumulating here rather than only assigning
                # keeps every caller that runs a bare forward and then reads the counters --
                # scripts/profile_moe_step.py and scripts/test_moe_bias_wiring.py both do, and
                # neither uses checkpointing, so for them the pre-b0-27 behaviour is unchanged.
                self.tokens_per_expert += counts
                self.step_tokens_per_expert += counts
                self.windows += 1
                self.micro_tokens_per_expert.copy_(counts)
                self.micro_forwards += 1
        # OFFSETS ARE CUMULATIVE ENDS, and int32 -- the op's convention, measured by tilerl.
        offs = torch.cumsum(counts, 0).to(torch.int32)
        # ONE CAST, ONE PLACE, for every expert matmul in this module (tilerl's review,
        # 2026-09-05). autocast covers nn.Linear and does NOT cover _grouped_mm: under the same
        # bf16 autocast the dense SwiGLU takes fp32 in and returns bf16, while _grouped_mm raises
        # "Expected inputs of BF16 type but got mat_a.dtype=torch.float32". So the dense FFN this
        # arm replaces never showed the defect, and E1 died before step 1 at this line
        # (runs/b0_moe_e1.log:76, 2026-09-05T06:52Z, dynamo's meta for aten._grouped_mm).
        #
        # `xw` FEEDS BOTH PATHS, routed and shared, so the dtype is decided here and not in four
        # places. sh13/sh2 work today only because they are nn.Linear and autocast covers them --
        # a dtype the routed path could not use. Casting once removes that asymmetry rather than
        # relying on two mechanisms to agree.
        xw = flat.to(self.w13.dtype)
        # THE ROUTED PATH'S INPUT IS THE PROJECTION, latent variant only. Down-project ONCE per
        # token, before the gather: projecting after would run d->d_latent top_k times per token
        # instead of once and break the parity the constructor asserts. The router still reads the
        # FULL-width token (logits are computed above from `flat`), because routing is a decision
        # about the token and not about its projection -- and because a d_latent-width router
        # would be a second architecture change bundled into this one.
        xr = self.down(xw) if self.d_latent else xw
        # THE EXPERT DTYPE MUST BE bf16, AND THIS IS A REFUSAL RATHER THAN A CAST (4c's ruling (c),
        # 2026-09-05). torch._grouped_mm's eager kernel accepts fp32 and its META registration does
        # not, so under torch.compile an fp32 model raises "Expected inputs of BF16 type but got
        # mat_a.dtype=torch.float32 and mat_b.dtype=torch.float32" -- measured on card 1: eager
        # fp32/fp32 returns fp32, compiled fp32/fp32 raises, both bf16 paths fine.
        #
        # The model reaches bf16 under --fp8 (`fp8 = args.fp8 and amp` in train.py's main) and, since
        # readout 5', under --bf16 as well -- but under neither by default.
        # run_ddp.sh passes --fp8; scripts/test_e2e.py invoked train.py without it, so the walk ran
        # a configuration the launch never runs and found this.
        #
        # NOT CHECKED AT CONSTRUCTION, and this is why: nn.Parameter is fp32 by default, so EVERY
        # MoEFFN is fp32 when __init__ returns -- verified 2026-09-05 -- and the cast to bf16
        # happens later, on the whole model. A constructor assert would refuse every arm including
        # the one that works. The first forward is the earliest point where the dtype is the one
        # the run will use.
        #
        # NOT A SILENT CAST TO bf16 EITHER: that would change the arithmetic of an fp32 arm without
        # anyone asking, which is the same objection amendments 6/7 raise about the fp8/bf16
        # confound. A crash naming its cause costs one run; a silent precision change costs the
        # interpretation of every number the arm produces.
        # THE bf16 REQUIREMENT IS ENFORCED AT LAUNCH, NOT HERE, and the attempt to put it here is
        # why the comment says so. torch._grouped_mm's eager kernel accepts fp32 while its META
        # registration does not, so only the COMPILED path is bf16-only -- an unconditional raise in
        # forward refused all 10 CPU checks, which run this module in fp32 where eager is correct.
        # Scoping it with `torch.compiler.is_compiling()` does not work either: dynamo traces the
        # raise, turns it into a graph break, and re-runs the segment in eager where the condition
        # is False -- measured 2026-09-05, the user still received the op's own
        # "Expected inputs of BF16 type but got mat_a.dtype=torch.float32 and mat_b.dtype=
        # torch.float32" instead of a message naming the cause. A guard that the compiler can
        # elide is not a guard. It lives in train.py, before the first step, where the model's
        # dtype is a plain fact.
        # ROW-OF-SLOT WITHOUT MATERIALISING IT, and it replaces two allocations plus a gather.
        # The old code built `tok = arange(n).repeat_interleave(top_k)` and then indexed it as
        # `tok[order]` -- twice, once for the dispatch gather and again inside index_add_ below.
        # But tok[i] == i // top_k BY CONSTRUCTION, so integer division gives tok[order]
        # directly and the arange, the repeat_interleave and both gathers of it are removable.
        # PROVEN, not assumed: torch.equal(tok[order], order // top_k) is True on the arm's
        # shape, and a negative control with top_k+1 as the divisor makes it False (so the
        # equality test can fail). Measured on card 3 at 4096 tok x top_k 3 x 48 experts:
        # arange+repeat_interleave 0.0174 ms against 0.0075 ms for the div.
        #
        # HOW SMALL THIS IS, stated so nobody budgets a speed win on it: the whole change saves
        # 0.0195 ms of a 0.854 ms dispatch+combine, which scales to 0.0037 s of the 0.517 s
        # step gap against dense -- 0.72%, i.e. 1.265x dense becomes 1.263x. The cost is not
        # the index arithmetic. It is index_add_ at 0.2073 ms (70% of these two regions), split
        # about evenly between materialising the fp32 y*gate temp (0.0953 ms, 37.7 MB) and the
        # scatter itself (0.1003 ms). Scaling in bf16 first would save 0.0137 ms of that and was
        # REJECTED: it rounds the gate to bf16 and moves the FFN output by 3.3e-03 relative,
        # which is a precision change bought for 0.03% of a step.
        row_of_slot = torch.div(order, self.top_k, rounding_mode="floor")
        rows = xr[row_of_slot]
        h = torch._grouped_mm(rows, self.w13.transpose(-2, -1), offs=offs)
        a, b = h.chunk(2, dim=-1)
        y = self._situ(a, b, lambda g: torch._grouped_mm(
            g.contiguous(), self.w2.transpose(-2, -1), offs=offs))
        # Scatter back, weighted by the un-biased gate. index_add_ over the un-sorted token index
        # so a token's top_k contributions land on its own row.
        #
        # THE GATE MULTIPLY AND THE ACCUMULATION STAY fp32 (tilerl's review). `gate` is a softmax
        # probability and up to top_k of them are summed into the same row, so this is the one
        # place with no dense analogue to match: the control's FFN has no gate and no scatter.
        # bf16 has 8 mantissa bits, and the same argument that put the optimizer's master copy in
        # fp32 (train.py's fp32_master, 91% of bf16 updates were bit-identical no-ops) applies to
        # a sum of three weighted contributions. Cast once at the return instead.
        #
        # THE ACCUMULATOR IS d_latent WIDE UNDER THE LATENT VARIANT, and the gated sum happens
        # THERE, before the up-projection. Two reasons it is this way round and not the other: the
        # up-projection is linear, so up(sum_k g_k * y_k) == sum_k g_k * up(y_k) and the cheaper
        # form is exact rather than an approximation; and projecting each of the top_k
        # contributions separately would cost top_k up-projections per token, which breaks the
        # parity the constructor asserts.
        de = self.d_latent or d
        gflat = gate.reshape(-1)[order]
        out = torch.zeros(n, de, device=x.device, dtype=torch.float32)
        out.index_add_(0, row_of_slot, y.float() * gflat[:, None])
        if self.d_latent:
            # UP-PROJECT ONCE, in the accumulator's dtype rather than casting back first: `up` is
            # an nn.Linear so autocast decides its compute dtype, and the fp32 accumulation above
            # is preserved through the projection's input.
            out = self.up(out.to(xw.dtype)).float()
        # THE SHARED EXPERT, which every token also traverses. Fed from the SAME cast input at the
        # FULL width -- it is not routed, so the latent projection does not apply to it.
        hs = self.sh13(xw)
        sa, sb = hs.chunk(2, dim=-1)
        out = out + self._situ(sa, sb, self.sh2).float()
        # BACK TO THE INPUT DTYPE. Measured on this model: under bf16 autocast the dense SwiGLU
        # returns bf16 for a bf16 input, and the real call site is `x + self.ffn(self.n2(x))` at
        # model.py:1056 where x is bf16 -- so at the launch shape this cast returns bf16 and the
        # arm's residual has the control's dtype. It is written as x.dtype rather than hardcoded
        # bf16 so the module stays correct under an fp32 or an autocast-off caller.
        return out.to(x.dtype).view(B, T, d)

    @torch.no_grad()
    def diagnostics(self, reset=True):
        """Readout 4: usage fraction, load entropy normalised by ln(N), and load Gini.

        Reset by default because the readout is per WINDOW: a cumulative counter converges on
        every expert having been used at least once and stops being able to see a collapse, which
        is the one thing this exists to catch. Same reasoning as ProductKeyMemory.diagnostics.
        """
        c = self.tokens_per_expert.float()
        tot = float(c.sum())
        used = int((self.tokens_per_expert > 0).sum())
        frac = used / self.n_routed
        if tot <= 0:
            # NOT 0.0 silently: an all-zero window means no forward ran, which is a different
            # finding from a perfectly collapsed router and must not print as one.
            ent_norm, gini = 0.0, 0.0
        else:
            p = (c / tot).clamp_min(1e-12)
            ent = float(-(p * p.log()).sum())
            ent_norm = ent / math.log(self.n_routed)
            s = c.sort().values
            k = s.numel()
            i = torch.arange(1, k + 1, dtype=torch.float32, device=s.device)
            gini = float((2.0 * (i * s).sum() / (k * tot)) - (k + 1) / k)
            gini = min(1.0, max(0.0, gini))
        out = {"usage_frac": frac, "used_experts": used, "n_routed": self.n_routed,
               "entropy_norm": ent_norm, "load_gini": gini,
               "window_steps": int(self.windows), "tokens": int(tot)}
        if reset:
            self.tokens_per_expert.zero_()
            self.windows.zero_()
        return out


def _moe_layers(cfg):
    """Which block indices are MoE, from cfg.moe_layers. Accepts "0-11", "0,3,6" or a list.

    Parsed in ONE place for the reason _mem_layers documents: a checkpoint written from a string
    and one written from a list must produce the SAME architecture, or two spellings of one arm
    read as two arms. The range form is accepted because the charter's cell is "all 12 layers"
    and writing that as a 12-element list in a launch line invites a typo the parity check
    cannot catch.
    """
    v = getattr(cfg, "moe_layers", None)
    if v is None or v == "":
        return list(range(cfg.layers))
    if isinstance(v, str):
        v = v.replace(" ", "")
        out = []
        for tok in v.split(","):
            if not tok:
                continue
            if "-" in tok[1:]:
                lo, hi = tok.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(tok))
        return out
    return [int(t) for t in v]


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

    def __init__(self, cfg, is_attn=False, memory=None, moe=False, attn_kind=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.d)
        # head_mixed replaces the layer-level alternation entirely: EVERY block gets both mixers,
        # so is_attn stops selecting anything. Reading it here rather than in HybridLM keeps the
        # block list construction (:421) untouched, so attn_every still decides nothing else.
        _hm = getattr(cfg, "head_mixed", 0)
        # attn_kind is the V4 hybrid's per-layer choice ("csa" or "hca"), decided by HybridLM
        # because interleaving is a property of the STACK and no layer can see its own index.
        # None = read the flags off cfg, which is every pre-hybrid config and every A/B that
        # turns one branch on globally. Implemented as a shallow cfg copy rather than another
        # GatedMLA argument: csa/hca are already read through getattr(cfg, ...) in three places
        # (the two constructors and the both-on refusal), and a second source for the same two
        # booleans is how they come to disagree.
        if attn_kind is not None and is_attn and not _hm:
            cfg = _CfgView(cfg, csa=(attn_kind == "csa"), hca=(attn_kind == "hca"))
        self.mixer = HeadMix(cfg, ratio=_hm) if _hm else (GatedMLA(cfg) if is_attn else DeltaRecurrence(cfg))
        self.n2 = RMSNorm(cfg.d)
        # THE FFN IS REPLACED, NOT SUPPLEMENTED, and that is the structural difference from the
        # memory arm. The memory pool ADDED a branch and left the dense parameter count identical
        # to the control's; MoE substitutes a sparse FFN for the dense one at equal ACTIVE width,
        # so the arm's assertion is the four integers of prereg#moe_0905, not equality with the
        # control. `self.ffn` keeps its name either way: sublayers() below hands it to the AttnRes
        # path by attribute, and every checkpoint remap, fp8 filter and optimizer rule that names
        # `.ffn.` keeps working for the arm without a second spelling to maintain.
        self.ffn = MoEFFN(cfg) if moe else SwiGLU(cfg)
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
        if n_kda == 0 and not int(getattr(cfg, "rope_dims", 0) or 0):
            raise ValueError(
                f"attn_every={cfg.attn_every} produces 0 KDA layers, but GatedMLA is NoPE "
                f"(KDA handles position). The model would have no position information. "
                f"Use attn_every >= 2, or set rope_dims > 0."
            )
        # The refusal above is about POSITION, not about KDA, so rope_dims lifts it -- that was
        # always the stated fix ("add RoPE to GatedMLA first"). What it must not do is lift it by
        # accident: rope_dims is checked here rather than assumed from the attention flags,
        # because a p1-shaped config that sets csa/hca and forgets rope_dims is exactly the
        # 21-sigma-worse model of 2026-08-30 wearing the new architecture's name.
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
                query_norm=str(getattr(cfg, "mem_query_norm", "none")),
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
        # WHICH BLOCKS ARE MoE. Resolved here rather than inside Block so the list is one object
        # the diagnostics and the optimizer can both read, and so an out-of-range index is caught
        # at construction instead of silently selecting nothing.
        self.moe_layers = []
        if int(getattr(cfg, "moe_experts", 0)) > 0:
            self.moe_layers = _moe_layers(cfg)
            _bad = [i for i in self.moe_layers if not 0 <= i < cfg.layers]
            if _bad:
                raise ValueError(
                    f"moe_layers names block(s) {_bad} outside [0, {cfg.layers - 1}]: "
                    f"moe_layers={getattr(cfg, 'moe_layers', None)!r}. An out-of-range index would "
                    f"silently convert fewer layers than the launch line says, and the params "
                    f"assertion would then be the only thing that noticed")
            # GRAD_CKPT IS ALLOWED (b0-27, 2026-09-07). It was refused here on the reasoning that
            # recomputation runs forward twice, so tokens_per_expert -- written under no_grad
            # inside forward -- would double, and readout 4's usage fraction and load Gini are
            # computed from that counter. The doubling is real and was measured on a card: ratio
            # exactly 2.0000, uniform across all four MoE layers, with the surplus correction
            # disabled. The CONSEQUENCE was not real. A uniform 2x moves neither consumer:
            # update_bias steps on sign(counts - mean), and sign is invariant under a positive
            # scale (ties included), so the bias step is bit-identical; usage_frac, used_experts,
            # entropy_norm and load_gini are all functions of the count distribution or are
            # normalised by the total, and were verified bit-identical at 1x and 2x on the measured
            # counts. Only `tokens` and `window_steps` moved -- two bookkeeping fields, not the
            # stop rule. So the refusal's specific claim, that a doubled denominator would report
            # a healthier spread, was false for the doubling that actually occurs.
            # MoEFFN.commit_token_counts now subtracts the surplus after backward, so the counter
            # is exact rather than merely harmless: post-fix ratio 1.0000 on the same card, with a
            # --no-commit control reading 2.0000 in the same claim to prove the test can still see
            # the defect.
        # V4 HYBRID ATTENTION (dsv4.hybrid_attention): CSA and HCA INTERLEAVED across layers, not
        # stacked inside one. Off by default, and off means every layer reads cfg.csa/cfg.hca as
        # before, so no existing arm changes shape.
        #
        # The first two attention layers are CSA and not HCA, which is the one place this departs
        # from copying V4-Flash verbatim (its first two are pure SWA). Reason, not preference:
        # HCA at m'=128 has no window branch, so a query in the first 128 positions of its
        # document sees no complete block and attends to nothing. Whatever occupies the bottom of
        # the stack has to cover that prefix exactly, and CSA's window branch does; SWA would too,
        # but that is a third attention class to implement and CSA already contains it.
        _hyb = bool(getattr(cfg, "attn_hybrid", False))
        attn_idx = [i for i in range(cfg.layers) if i % cfg.attn_every == cfg.attn_every - 1]
        kind = {}
        if _hyb:
            for n, i in enumerate(attn_idx):
                kind[i] = "csa" if n < 2 or n % 2 == 0 else "hca"
        self.blocks = nn.ModuleList(
            # every `attn_every` blocks (was `i == cfg.attn_every - 1`: one attention layer total)
            [Block(cfg, is_attn=(i % cfg.attn_every == cfg.attn_every - 1),
                   memory=(self.memory if i in self.mem_layers else None),
                   moe=(i in self.moe_layers),
                   attn_kind=kind.get(i))
             for i in range(cfg.layers)]
        )
        self.attn_kinds = kind
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

    def commit_moe_token_counts(self):
        """Commit every MoE layer's token counts for this micro-batch. Returns the layers swept.

        One call per micro-batch, after loss.backward(). Wraps MoEFFN.commit_token_counts, whose
        docstring carries the arithmetic and why it runs after backward rather than inside forward.

        RETURNS THE COUNT SO A CALLER CAN ASSERT IT IS NOT ZERO (tilerl). A sweep that resolves
        zero layers leaves the surplus in place and the ratio reads 2.0 -- indistinguishable from
        the correction not working at all. The count separates those two failures. It also splits
        what `getattr(model, "moe_layers", None) or []` folds together at an external call site:
        "this model has no MoE layers" and "the attribute is gone" both yield an empty list there,
        and only one of them is a bug.

        `self.moe_layers` holds block INDICES, not modules, so the layers are reached as
        self.blocks[i].ffn -- the same shape train.py uses for its balancer sweep.

        WHAT THE COUNT CANNOT CATCH, because both sides read this same attribute (tilerl): the
        caller's check is `swept != len(_moe_balance_layers)`, and train.py builds that list as
        `[raw_model.blocks[i].ffn for i in (getattr(raw_model, "moe_layers", None) or [])]` -- from
        this same attribute. Anything that corrupts the LIST moves both sides equally, so the check
        sees only a broken SWEEP. Measured against these two functions:

            moe_layers      balancer  sweep  fires?  per-block commits
            [0,1,2,3]              4      4      no  [1, 1, 1, 1]
            [0,1]                  2      2      no  [1, 1, 0, 0]
            []                     0      0      no  [0, 0, 0, 0]
            [0,0,0,0]              4      4      no  [4, 0, 0, 0]

        Three of those four are caught elsewhere by construction, and this was verified by BUILDING
        each one rather than by reading (layers=4, d=256, moe_experts=4, parity defaults):

            moe_layers   len  total params   MoE-weight params
            [0,1,2,3]      4    21,370,640           9,441,280
            [0,1]          2    20,188,944           4,720,640
            [0]            1    19,598,096           2,360,320
            [0,0,0,0]      4    19,598,096           2,360,320

        A truncated list builds fewer experts and an out-of-range index raises ValueError in
        __init__ before the blocks are built. The DUPLICATED index builds a model
        parameter-identical to [0], because `i in self.moe_layers` is a membership test and converts
        block 0 once however many times it is named -- while len(moe_layers) still reads 4.

        NOTHING EXECUTABLE CATCHES THE DUPLICATE, and an earlier version of this docstring said a
        parameter count did. It does not: train.py computes `n_params = sum(p.numel() ...)` and only
        formats it into the runlog's `params {n_params/1e6:.1f}M` line -- no comparison, no assert.
        Three places NAME a params gate and none runs one: that print, a comment in scripts/harness.py
        citing "prereg#moe_0905: 9,437,184 active / 800,965,704 total", and prereg#moe_0905's own
        `active_params_equal_control` field, which asserts that a gate checks four integers. The
        integers are RECORDED AT LAUNCH BY A PERSON, not enforced by code. So any check comparing
        total params against the INTENDED shape would catch a duplicate, and no such check runs
        today. Found by tilerl, who went looking for the gate this docstring claimed and could not
        find an executable one; the overstatement was the worse error, because a reader who believes
        a case is covered stops looking.

        No guard added: no code path generates a duplicated index, and a check on a state with no
        producer reads like coverage without being coverage. What is unguarded is the double commit
        -- layer 0's counts committed four times and layers 1-3 never, with both sides of the sweep
        check reading 4.
        """
        n = 0
        for i in self.moe_layers:
            ffn = getattr(self.blocks[i], "ffn", None)
            commit = getattr(ffn, "commit_token_counts", None)
            if commit is None:
                continue
            commit()
            n += 1
        return n

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
