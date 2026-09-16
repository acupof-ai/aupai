"""P0: Compressor (softmax-gated ratio pooling) matches the vendored upstream module.

Covered:
- ratio=2 prefill over an even length, fp32 pooling path (atol 1e-5),
- ratio=2 prefill with a TRAILING partial group: the latent set is only the complete
  groups and the tail is parked (upstream returns the pooled complete groups; states
  hold the remainder),
- ratio=1: gate-less bf16 linear + norm (atol 2e-2),
- within-group causality: the softmax weight is normalized over ONLY the ratio
  positions of that token's own group, so pooling can never read another group;
- bf16 output at ratio=2 uses the same 2e-2 tolerance.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import cmp
from ref_oracle import bf16_args, load_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.compressor import Compressor


def _pair(model, args, layer_id, dim, head_dim, ratio, norm_eps, max_bs, seed=0):
    """Build reference and v41f compressors for one layer and copy weights by name.

    The upstream module allocates projections with torch.empty (the checkpoint loader fills
    them); uninitialized bf16 memory can be NaN, which would make an allclose "match" on
    garbage meaningless. So we first write FINITE random values into the REFERENCE weights
    (in the reference's own dtype), then copy them into ours by name.
    """
    g = torch.Generator().manual_seed(seed)
    ref = model.Compressor(args, layer_id).eval()
    ours = Compressor(dim, head_dim, ratio, norm_eps=norm_eps,
                      max_batch_size=max_bs).eval()

    def _fill_and_copy(rw, ow):
        with torch.no_grad():
            rw.copy_(torch.randn(rw.shape, generator=g, dtype=rw.dtype) * 0.1)
            ow.data.copy_(rw.data)

    _fill_and_copy(ref.wkv.weight, ours.wkv.weight)
    if ratio > 1:
        _fill_and_copy(ref.wgate.weight, ours.wgate.weight)
    with torch.no_grad():
        ref.norm.weight.copy_(
            1.0 + 0.01 * torch.randn(ref.norm.weight.shape, generator=g,
                                     dtype=ref.norm.weight.dtype))
        ours.norm.weight.data.copy_(ref.norm.weight.data)
    return ref, ours


def test_compressor_ratio2_fp32():
    model, _ = load_reference()
    torch.manual_seed(0)
    dim, head_dim, ratio, eps, bsz = 48, 32, 2, 1e-6, 2
    args = bf16_args(model, dim=dim, head_dim=head_dim, norm_eps=eps,
                     compress_ratios=(ratio,), max_batch_size=bsz)
    ref, ours = _pair(model, args, 0, dim, head_dim, ratio, eps, bsz)
    x = torch.randn(bsz, 8, dim)
    with torch.no_grad():
        want = ref(x, 0)
        got = ours(x, 0)
    cmp("Compressor ratio2 fp32", got, want, atol=1e-5)
    # pre-RoPE latent count = seqlen // ratio
    assert want.shape == (bsz, 8 // ratio, head_dim), want.shape


def test_compressor_ratio2_partial_tail():
    model, _ = load_reference()
    torch.manual_seed(1)
    dim, head_dim, ratio, eps, bsz = 48, 32, 2, 1e-6, 2
    args = bf16_args(model, dim=dim, head_dim=head_dim, norm_eps=eps,
                     compress_ratios=(ratio,), max_batch_size=bsz)
    ref, ours = _pair(model, args, 0, dim, head_dim, ratio, eps, bsz)
    x = torch.randn(bsz, 7, dim)   # 3 complete groups + 1 trailing token
    with torch.no_grad():
        want = ref(x, 0)
        got = ours(x, 0)
    cmp("Compressor partial-tail pooled", got, want, atol=1e-5)
    assert want.shape == (bsz, 3, head_dim), want.shape
    # the one leftover token is parked, not pooled
    cmp("Compressor tail kv_state", ours.kv_state[:bsz, 0],
        ref.kv_state[:bsz, 0], atol=1e-5)
    cmp("Compressor tail score_state", ours.score_state[:bsz, 0],
        ref.score_state[:bsz, 0], atol=1e-5)
    assert torch.isinf(ours.score_state[:bsz, 1]).all(), "empty slot should stay -inf"


def test_compressor_ratio1_bf16():
    model, _ = load_reference()
    torch.manual_seed(2)
    dim, head_dim, ratio, eps, bsz = 48, 128, 1, 1e-6, 2
    args = bf16_args(model, dim=dim, head_dim=head_dim, norm_eps=eps,
                     compress_ratios=(ratio,), max_batch_size=bsz)
    ref, ours = _pair(model, args, 0, dim, head_dim, ratio, eps, bsz)
    x = torch.randn(bsz, 5, dim)
    with torch.no_grad():
        # ratio 1 keeps bf16 weights; compare the bf16 path
        want = ref(x.bfloat16(), 0)
        got = ours(x.bfloat16(), 0)
    cmp("Compressor ratio1 bf16", got, want, atol=2e-2)
    assert not hasattr(ours, "wgate"), "ratio 1 must not build a gate"


def test_compressor_ratio2_bf16_output():
    model, _ = load_reference()
    torch.manual_seed(3)
    dim, head_dim, ratio, eps, bsz = 48, 32, 2, 1e-6, 2
    args = bf16_args(model, dim=dim, head_dim=head_dim, norm_eps=eps,
                     compress_ratios=(ratio,), max_batch_size=bsz)
    ref, ours = _pair(model, args, 0, dim, head_dim, ratio, eps, bsz)
    x = torch.randn(bsz, 6, dim)
    with torch.no_grad():
        want = ref(x.bfloat16(), 0)
        got = ours(x.bfloat16(), 0)
    cmp("Compressor ratio2 bf16 out", got, want, atol=2e-2)


def test_compressor_pooling_is_causal_within_group():
    """Pooling weight must be normalized over exactly the ratio positions of one group;
    a token's gate must not move any other group's pooled latent."""
    torch.manual_seed(4)
    dim, head_dim, ratio, eps, bsz = 32, 16, 2, 1e-6, 1
    ours = Compressor(dim, head_dim, ratio, norm_eps=eps, max_batch_size=bsz).eval()
    x = torch.randn(bsz, 4, dim)
    with torch.no_grad():
        # reference: explicit within-group softmax over the ratio=2 axis only
        kv = ours.wkv(x.float()).unflatten(1, (-1, ratio))
        sc = ours.wgate(x.float()).unflatten(1, (-1, ratio))
        manual = (kv * sc.softmax(dim=2)).sum(dim=2)
        manual = ours.norm(manual.to(x.dtype))
        got = ours(x, 0)
    cmp("Compressor causal grouping", got, manual, atol=1e-5)
