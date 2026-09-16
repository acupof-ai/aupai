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
    """In-place-free rotation. Accepts [b,s,d] or [b,s,h,d]; mirrors upstream exactly."""
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    out = torch.view_as_real(xc * freqs_cis).flatten(-2)
    return out.to(y.dtype)
