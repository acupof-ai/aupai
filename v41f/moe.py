"""MoE top-layer combine, faithful to upstream model_ref.MoE (:854-904), CPU P0.

Gate selects top-k experts; each routed token is computed by those experts, weighted by
the UN-biased gate score, and summed; exactly one shared expert runs on every token
unconditionally and its output is ADDED. Training dispatch uses torch._grouped_mm
(P1/GPU); here a single-process per-expert gather loop matches the reference shape.
No distributed all_reduce (world_size == 1).

The routed weight is multiplied inside the expert BEFORE the down projection
(upstream Expert.forward(x, weights)); `_expert_weighted` reproduces that path on
the shared Expert's linears so bf16 rounding matches to 2e-2.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .expert import Expert
from .norm_gate import Gate


def _expert_weighted(expert, x, weights):
    """Upstream Expert.forward(x, weights): the routing weight is multiplied in fp32
    BEFORE the down projection, not after it. We can't pass weights through v41f.Expert
    (its signature is weight-free), so replicate its exact path on its own linears:
    silu(clamped w1) * clamped w3, times weight in fp32, then the (bf16) w2. Applying
    the weight after w2 instead is algebraically equal but rounds at a different point
    and misses the 2e-2 bf16 allclose by up to 0.5."""
    dtype = x.dtype
    gate = expert.w1(x).float()
    up = expert.w3(x).float()
    lim = expert.swiglu_limit
    if lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    h = weights * (F.silu(gate) * up)
    return expert.w2(h.to(dtype))


class MoE(nn.Module):
    def __init__(
        self,
        dim,
        n_routed_experts,
        n_activated_experts,
        moe_inter_dim,
        route_scale=1.5,
        swiglu_limit=0.0,
        gate_temp=1.0,
        norm_topk_prob=True,
        score_func="sqrtsoftplus",
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.gate = Gate(
            dim,
            n_routed_experts,
            n_activated_experts,
            score_func=score_func,
            gate_temp=gate_temp,
            norm_topk_prob=norm_topk_prob,
            route_scale=route_scale,
        )
        self.experts = nn.ModuleList(
            [Expert(dim, moe_inter_dim, swiglu_limit=swiglu_limit) for _ in range(n_routed_experts)]
        )
        # exactly one shared expert, unconditional
        self.shared_experts = Expert(dim, moe_inter_dim, swiglu_limit=swiglu_limit)

    def forward(self, x, image_mask=None):
        # v41f has no vision stack: image_mask is accepted for signature parity but
        # unused; the P0 Gate (norm_gate.Gate) has no VL-bias path.
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += _expert_weighted(self.experts[i], x[idx], weights[idx, top, None])
        y += self.shared_experts(x).float()
        return y.type_as(x).view(shape)
