"""RMSNorm and the sqrt(softplus) MoE gate, faithful to upstream model.py.

RMSNorm: model.py RMSNorm (eps configurable; v41f-S uses 1e-6, upstream 1e-20).
Gate: model.py Gate.forward — sqrt(softplus(logits/T)), selection-only bias,
weights gathered from the UN-biased score, normalized over topk, * route_scale.
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class Gate(nn.Module):
    """Top-k router matching upstream Gate (no VL bias in v41f)."""

    def __init__(self, dim: int, n_routed: int, top_k: int,
                 score_func: str = "sqrtsoftplus", gate_temp: float = 1.0,
                 norm_topk_prob: bool = True, route_scale: float = 1.5):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.score_func = score_func
        self.gate_temp = gate_temp
        self.norm_topk_prob = norm_topk_prob
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(n_routed, dim))
        # persistent selection-only bias; zeros like a checkpoint without the key
        self.register_buffer("bias", torch.zeros(n_routed, dtype=torch.float32),
                             persistent=True)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor):
        """x: [n, dim] -> weights [n, top_k], indices [n, top_k]."""
        scores = torch.nn.functional.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = torch.nn.functional.softplus(scores).sqrt()
        # bias picks experts; weights come from raw scores
        indices = (scores + self.bias).topk(self.top_k, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices
