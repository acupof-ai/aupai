"""P0: RoPE frequencies and rotary application match upstream, including inverse."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import cmp
from ref_oracle import load_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.rope import apply_rotary_emb, precompute_freqs_cis


def test_freqs_cis_no_yarn():
    model, _ = load_reference()
    for dim in (32, 64):
        got = torch.view_as_real(precompute_freqs_cis(dim, 128))
        want = torch.view_as_real(model.precompute_freqs_cis(dim, 128, 0, 10000.0, 40, 32, 1))
        cmp(f"freqs_cis dim{dim}", got, want, atol=1e-6)


def test_freqs_cis_yarn():
    model, _ = load_reference()
    dim, osl = 64, 4096
    got = torch.view_as_real(precompute_freqs_cis(dim, 128, original_seq_len=osl, factor=16))
    want = torch.view_as_real(
        model.precompute_freqs_cis(dim, 128, osl, 10000.0, 16, 32, 1))
    cmp("freqs_cis yarn", got, want, atol=1e-6)


def test_apply_forward_inverse():
    model, _ = load_reference()
    torch.manual_seed(0)
    b, s, h, d = 2, 20, 4, 64
    rd = 32
    freqs = model.precompute_freqs_cis(rd, s, 0, 10000.0, 40, 32, 1)
    x = torch.randn(b, s, h, d)
    # rotate only the tail in both, to match usage
    xg = x.clone()
    xw = x.clone()
    xg_tail = apply_rotary_emb(xg[..., -rd:], freqs)
    model.apply_rotary_emb(xw[..., -rd:], freqs)
    cmp("apply forward", xg_tail, xw[..., -rd:], atol=1e-5)
    # inverse on the rotated tensor recovers it
    back = apply_rotary_emb(xg_tail, freqs, inverse=True)
    cmp("apply inverse recovers", back, x[..., -rd:], atol=1e-4)
