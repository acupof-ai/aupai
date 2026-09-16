"""CPU-only loader for the vendored upstream reference (third_party/deepseek_v41_ref).

The upstream modules assume tilelang SM100 fp8/fp4 kernels, a vision stack, and a
distributed world. P0 compares pure-tensor math on CPU, so we inject lightweight
stubs, force bf16 weights (linear() then takes the plain F.linear branch and never
calls the quant kernels), and provide pure-torch reference implementations for the
two math kernels P0 actually validates: hc_split_sinkhorn and sparse_attn.

Nothing here is allowed to silently replace a module under test: act/fp8/fp4 GEMM
stubs raise, because a v41f-S test that accidentally hit a quant path would not be
the comparison it claims to be.
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch

REF_DIR = Path(__file__).resolve().parents[2] / "third_party" / "deepseek_v41_ref"


class _QuantNotAvailable(RuntimeError):
    pass


def _install_stubs() -> None:
    if getattr(sys, "_v41f_ref_stubbed", False):
        return

    # tilelang + tilelang.language: the kernel module touches them at import time.
    # No v41f-S math path calls them, so a permissive stub is enough; the kernels
    # themselves are stubbed below and refuse if actually invoked.
    if "tilelang" not in sys.modules:
        tl = types.ModuleType("tilelang")
        tl_language = types.ModuleType("tilelang.language")

        def _jit(*_a, **_k):
            def deco(fn):
                return fn
            return deco

        tl.jit = _jit
        tl.set_log_level = lambda *_a, **_k: None
        tl.PassConfigKey = types.SimpleNamespace(
            TL_DISABLE_WARP_SPECIALIZED=0, TL_DISABLE_TMA_LOWER=1)
        for nm in ("float8_e4m3", "float4_e2m1fn", "bfloat16", "float32", "int32"):
            setattr(tl_language, nm, nm)
        for nm in ("symbolic", "prim_func", "Kernel", "Parallel", "Pipelined",
                   "alloc_shared", "alloc_fragment", "copy", "gemm", "reduce_absmax",
                   "reduce_max", "reduce_sum", "Cast", "clamp", "max", "if_then_else",
                   "ceildiv", "use_swizzle", "fill", "clear", "alloc"):
            def _missing(*_a, _nm=nm, **_k):
                raise _QuantNotAvailable(f"tilelang.{_nm} called in a CPU P0 path")
            setattr(tl_language, nm, _missing)
        tl.language = tl_language
        sys.modules["tilelang"] = tl
        sys.modules["tilelang.language"] = tl_language

    # kernel: real names the model imports, with pure-torch math where P0 needs it.
    kernel = types.ModuleType("kernel")

    def _no_quant(*_a, **_k):
        raise _QuantNotAvailable(
            "quant GEMM path reached in P0: force bf16 weights in the reference model")

    kernel.act_quant = _no_quant
    kernel.fp4_act_quant = _no_quant
    kernel.fp4_gemm = _no_quant
    kernel.fp8_gemm = _no_quant

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
        """Pure-torch port of kernel.hc_split_sinkhorn_kernel, one token at a time
        but vectorised over [N]. Returns pre[N,hc], post[N,hc], comb[N,hc,hc]."""
        n = mixes.shape[0]
        # upstream layout per token: pre[hc] | post[hc] | comb[hc*hc]
        pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult].view(1, -1)) + eps
        post = 2 * torch.sigmoid(
            mixes[:, hc_mult:2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult:2 * hc_mult].view(1, -1))
        comb_raw = (mixes[:, 2 * hc_mult:].reshape(n, hc_mult, hc_mult) * hc_scale[2]
                    + hc_base[2 * hc_mult:].view(1, hc_mult, hc_mult))
        comb = torch.softmax(comb_raw, dim=-1) + eps
        # kernel does one column normalization outside the serial loop, then iters-1
        # full row/column rounds.
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        """Pure-torch port of kernel.sparse_attn for [b,m,h,d] q, gathered kv [b,n,d].
        MQA-style shared kv (no head dim). -1 index = empty slot."""
        b, m, h, d = q.shape
        valid = topk_idxs >= 0
        safe = topk_idxs.clamp_min(0)
        gathered = kv[
            torch.arange(b, device=q.device)[:, None, None],
            safe,
        ]                                   # [b,m,topk,d]
        scores = torch.einsum("bmhd,bmtd->bmht", q, gathered) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        row_max = scores.amax(dim=-1, keepdim=True)
        exp = torch.exp(scores - row_max)
        exp = torch.nan_to_num(exp, nan=0.0, posinf=0.0, neginf=0.0)
        sink_term = torch.exp(attn_sink.view(1, 1, h) - row_max.squeeze(-1))
        denom = exp.sum(dim=-1) + sink_term
        out = torch.einsum("bmht,bmtd->bmhd", exp, gathered) / denom.unsqueeze(-1)
        return out

    kernel.hc_split_sinkhorn = hc_split_sinkhorn
    kernel.sparse_attn = sparse_attn
    sys.modules["kernel"] = kernel

    # image_processor: only constants are imported; vision disabled (n_layers=0).
    img = types.ModuleType("image_processor")
    for nm in ("IMAGE", "IMAGE_END", "IMAGE_NEW_LINE", "IMAGE_START"):
        setattr(img, nm, -1)
    sys.modules["image_processor"] = img

    vision = types.ModuleType("vision")
    class _VisionUnused:
        def __init__(self, *_a, **_k):
            raise _QuantNotAvailable("vision is out of scope for v41f")
    vision.ViT = _VisionUnused
    vision.Aligner = _VisionUnused
    sys.modules["vision"] = vision

    sys._v41f_ref_stubbed = True


def _load(mod_name: str, file_name: str):
    _install_stubs()
    from importlib.machinery import SourceFileLoader
    path = REF_DIR / file_name
    spec = importlib.util.spec_from_loader(
        mod_name, SourceFileLoader(mod_name, str(path)))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_reference():
    """Import upstream engram + model with stubs installed; return the model module."""
    engram = _load("engram", "engram_ref.py.ref")
    model = _load("model", "model_ref.py.ref")
    return model, engram


def bf16_args(model, **over):
    """Reference ModelArgs forced onto the CPU pure-torch path: bf16 weights,
    single process, no vision, no engram hash (layers=()), no DSpark."""
    import dataclasses
    base = dict(dtype="bf16", expert_dtype=None, vision_n_layers=0,
                engram_layer_ids=(), dspark_block_size=0,
                dspark_target_layer_ids=(), max_batch_size=2, max_seq_len=4096)
    base.update(over)
    return dataclasses.replace(model.ModelArgs(), **base)
