"""Partial rotary embedding, faithful to upstream model.py.

- precompute_freqs_cis: model.py:368 (YaRN when original_seq_len>0; v41f-S passes 0).
- apply_rotary_emb: model.py:392 — rotates adjacent complex pairs; inverse=True
  conjugates, used to un-rotate the attention output.

Upstream rotates the LAST rope_head_dim of each head. The reference works in complex64;
we mirror it numerically in float32.
"""
import math

import torch


def precompute_freqs_cis(dim: int, seqlen: int, *, original_seq_len: int = 0,
                         base: float = 10000.0, factor: float = 40.0,
                         beta_fast: int = 32, beta_slow: int = 1):
    freqs = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (
                2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low)
                / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """In-place rotation (mirrors upstream): writes the rotated values back into x and
    returns the same storage. The five attention.py call sites invoke it for the side
    effect (a sliced q/kv tail rotated in place) and discard the return, so a functional
    no-copy return would silently leave q/kv unrotated on the real forward path."""
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    lead = freqs_cis.size(0) if freqs_cis.dim() == 3 else 1  # [b,s,d/2] per-token positions
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(lead, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(lead, xc.size(1), 1, xc.size(-1))
    out = torch.view_as_real(xc * freqs_cis).flatten(-2)
    y.copy_(out.to(y.dtype))  # in-place, matching model_ref apply_rotary_emb (return discarded by callers)
    return y


def rope_cos_sin(freqs_cis: torch.Tensor):
    """complex [..., d/2] -> (cos, sin) fp32 of the same shape, for apply_rotary_real."""
    return freqs_cis.real.contiguous(), freqs_cis.imag.contiguous()


def apply_rotary_real(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False):
    """apply_rotary_emb without the complex64 round trip, OUT of place: the rotation of adjacent
    pairs written as real multiplies, so torch.compile fuses it into the neighbouring casts
    (inductor has no complex kernels and falls back to eager on view_as_complex). fp32 math on
    the pairs; agrees with apply_rotary_emb to bf16 rounding (tests/v41f/test_p1_fused.py)."""
    xf = x.float().unflatten(-1, (-1, 2))
    a, b = xf[..., 0], xf[..., 1]
    if inverse:
        sin = -sin
    lead = cos.size(0) if cos.dim() == 3 else 1
    if xf.ndim == 4:  # [b,s,d/2,2]
        cos = cos.view(lead, xf.size(1), xf.size(2))
        sin = sin.view(lead, xf.size(1), xf.size(2))
    else:  # [b,s,h,d/2,2]
        cos = cos.view(lead, xf.size(1), 1, xf.size(3))
        sin = sin.view(lead, xf.size(1), 1, xf.size(3))
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2).to(x.dtype)
