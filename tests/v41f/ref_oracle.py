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

REF_DIR = Path(__file__).absolute().parents[2] / "third_party" / "deepseek_v41_ref"


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
    single process, no vision, no DSpark. Engram defaults OFF (engram_layer_ids=()) so the
    ordinary whole-model comparison builds with no tokenizer; the ON path overrides it --
    use engram_on_args / build_engram_transformer, which fill the derived engram fields and
    validate the tokenizer/table config BEFORE construction can fail deep in the ref with a
    bare AttributeError/IndexError (the de A4 diagnostic-order trap)."""
    import dataclasses
    base = dict(dtype="bf16", expert_dtype=None, vision_n_layers=0,
                engram_layer_ids=(), dspark_block_size=0,
                dspark_target_layer_ids=(), max_batch_size=2, max_seq_len=4096)
    base.update(over)
    return dataclasses.replace(model.ModelArgs(), **base)


# A synthetic tokenizer for the ENGRAM ON path: enough surface for
# build_compressed_token_map (backend_tokenizer.decode / id_to_token / len) and nothing
# else, so an ON ref Transformer builds on CPU with no data/tokenizer.json on disk. The
# pieces are the P0 engram known-answer set (case/accent/whitespace folding + the raw U+FFFD
# piece); a caller needing another compressed-vocab size passes its own pieces.
_DEFAULT_ENGRAM_PIECES = [
    "a", "A", "y", " The", "the", "THE", "\tThe\n", "cafe", "café", " ", "\n", "�",
]


class _SyntheticBackend:
    def __init__(self, pieces):
        self.pieces = list(pieces)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[i] for i in ids)

    def id_to_token(self, i):
        return f"<raw{i}>"


class SyntheticTokenizer:
    """Stand-in for the HF tokenizer the ref Engram path reads. Disk-free CPU double."""

    def __init__(self, pieces=None):
        self.backend_tokenizer = _SyntheticBackend(
            _DEFAULT_ENGRAM_PIECES if pieces is None else pieces)

    def __len__(self):
        return len(self.backend_tokenizer.pieces)


def synthetic_tokenizer(pieces=None):
    return SyntheticTokenizer(pieces)


class _NamespaceLike:
    """Just enough attribute access for EngramLayout.from_args over a plain dict."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def _engram_num_embeddings(engram_mod, over):
    """Table rows per engram layer = sum of that layer's bucket primes, derived from the
    SAME prime layout the ref builds (never a hand-written literal). from_args only passes
    num_embeddings through; primes are all we read, so a placeholder () is supplied."""
    probe = dict(over, engram_num_embeddings=())
    layout = engram_mod.EngramLayout.from_args(_NamespaceLike(probe))
    return tuple(sum(p for ngram in layer for p in ngram) for layer in layout.primes)


def engram_on_args(model, engram_mod, *, engram_layer_ids=(1,), tokenizer=None,
                   engram_max_ngram_size=4, engram_n_heads=2, engram_head_dim=8,
                   engram_vocab_size=20, engram_pad_id=2, **shape):
    """Ref ModelArgs for an engram-ON CPU comparison. Derives engram_num_embeddings from the
    bucket primes and engram_compressed_vocab_size by measuring the tokenizer (vendored map;
    P0 proves it bit-equal to v41f/engram). Validates in a clear order BEFORE Transformer
    construction: (1) a non-empty engram layer REQUIRES a tokenizer -- the ref otherwise dies
    in NgramHashState with a NoneType AttributeError; (2) num_embeddings is filled from the
    layout -- an empty tuple otherwise raises IndexError at ParallelEngramEmbedding."""
    engram_layer_ids = tuple(engram_layer_ids)
    if engram_layer_ids and tokenizer is None:
        raise ValueError(
            "engram ON path requires a tokenizer: pass synthetic_tokenizer() (or an HF "
            "tokenizer). Transformer(args, None) with engram_layer_ids="
            f"{engram_layer_ids} fails deep in NgramHashState with a NoneType AttributeError; "
            "the tokenizer feeds build_compressed_token_map.")
    _, compressed_size = engram_mod.build_compressed_token_map(tokenizer)
    over = dict(
        engram_layer_ids=engram_layer_ids,
        engram_max_ngram_size=engram_max_ngram_size,
        engram_n_heads=engram_n_heads,
        engram_head_dim=engram_head_dim,
        engram_vocab_size=engram_vocab_size,
        engram_pad_id=engram_pad_id,
        engram_compressed_vocab_size=compressed_size,
    )
    over.update(shape)
    over["engram_num_embeddings"] = _engram_num_embeddings(engram_mod, over)
    return bf16_args(model, **over), over["engram_num_embeddings"], compressed_size


def build_engram_transformer(model, args, tokenizer=None, eval_mode=True):
    """Construct a ref Transformer, enforcing the engram/tokenizer contract BEFORE the ref
    can fail opaquely. None is valid only on the OFF path (engram_layer_ids=()). Disk-free:
    this harness never reads data/tokenizer.json."""
    if tuple(args.engram_layer_ids) and tokenizer is None:
        raise ValueError(
            "build_engram_transformer: engram_layer_ids is non-empty but no tokenizer was "
            "given. Pass ref_oracle.synthetic_tokenizer(); None only matches the OFF path.")
    import torch
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        t = model.Transformer(args, tokenizer)
    finally:
        torch.set_default_dtype(prev)
    return t.eval() if eval_mode else t

