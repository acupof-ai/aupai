"""P0: Expert SwiGLU clamp semantics match upstream Expert."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).absolute().parent))
from ref_oracle import load_reference

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
from v41f.expert import Expert


def _matched_expert(dim, inter, limit):
    model, _ = load_reference()
    ref = model.Expert(dim, inter, dtype=torch.bfloat16, swiglu_limit=limit)
    ours = Expert(dim, inter, swiglu_limit=limit).bfloat16()
    # nn.Linear weights are torch.empty: fill the reference with finite values first
    # (wide enough to bind the clamps) then mirror them into ours.
    with torch.no_grad():
        for lyr in (ref.w1, ref.w3, ref.w2):
            lyr.weight.copy_(torch.randn_like(lyr.weight) * 0.2)
        ours.w1.weight.data.copy_(ref.w1.weight.data)
        ours.w3.weight.data.copy_(ref.w3.weight.data)
        ours.w2.weight.data.copy_(ref.w2.weight.data)
    return ref, ours


def test_expert_matches():
    ref, ours = _matched_expert(64, 96, 10.0)
    x = (torch.randn(32, 64) * 8.0).bfloat16()
    with torch.no_grad():
        d = (ours(x) - ref(x)).abs().max().item()
    assert torch.isfinite(ours(x)).all()
    assert d < 2e-2, d


def test_limit_zero_is_plain_swiglu():
    ref, ours = _matched_expert(32, 48, 0.0)
    x = (torch.randn(8, 32) * 20.0).bfloat16()
    with torch.no_grad():
        d = (ours(x) - ref(x)).abs().max().item()
    assert d < 2e-2, d
