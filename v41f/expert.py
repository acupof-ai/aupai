"""One SwiGLU expert and the MoE combine, faithful to upstream model.py:830-904.

Expert: up clamped both sides, gate clamped from above only, all in fp32 around the
activation, then back to input dtype. MoE dispatch in training uses torch._grouped_mm
(P1/GPU); P0 validates the per-expert math and the shared-expert sum, not the
dispatch kernel.
"""
import torch
from torch import nn


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = torch.nn.functional.silu(gate) * up
        return self.w2(h.to(dtype))
