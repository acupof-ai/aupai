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

import math

import torch
import torch.nn.functional as F
from torch import nn

from .deepgemm_moe import grouped_linear_fp8
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


def swiglu_clamp(gate, up, weight, lim: float):
    """silu(clamp(gate)) * clamp(up) * routing weight, fp32, as ONE elementwise expression: under
    torch.compile inductor fuses the clamps, silu, two multiplies and the surrounding casts into
    one kernel over the [rows, inter] activation (the 'fused SwiGLU'); eager runs them separately.
    Bit-identical to the reference's clamp-then-silu order."""
    if lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    return weight.float() * (F.silu(gate) * up)


class MoE(nn.Module):
    grouped_on_cpu = False  # tests flip it: torch._grouped_mm runs on CPU when strides are 16B-aligned

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
        stacked=False,
        moe_gemm="grouped_mm",
    ):
        super().__init__()
        self.moe_gemm = moe_gemm
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.stacked = bool(stacked)
        self.swiglu_limit = float(swiglu_limit)
        self.gate = Gate(
            dim,
            n_routed_experts,
            n_activated_experts,
            score_func=score_func,
            gate_temp=gate_temp,
            norm_topk_prob=norm_topk_prob,
            route_scale=route_scale,
        )
        if self.stacked:
            # three stacked parameters, [E,inter,dim] / [E,inter,dim] / [E,dim,inter]; grouped_mm
            # reads them without a per-forward torch.stack (which saved 3*E*inter*dim bf16 per
            # layer for backward). Init matches nn.Linear's kaiming_uniform (bound 1/sqrt(fan_in)).
            self.experts = None
            for name, out_f, in_f in (("w1", moe_inter_dim, dim), ("w3", moe_inter_dim, dim), ("w2", dim, moe_inter_dim)):
                w = torch.empty(n_routed_experts, out_f, in_f)
                nn.init.uniform_(w, -in_f ** -0.5, in_f ** -0.5)
                setattr(self, name, nn.Parameter(w))
        else:
            self.experts = nn.ModuleList(
                [Expert(dim, moe_inter_dim, swiglu_limit=swiglu_limit) for _ in range(n_routed_experts)]
            )
        # exactly one shared expert, unconditional
        self.shared_experts = Expert(dim, moe_inter_dim, swiglu_limit=swiglu_limit)
        # Trainer-side balancing (V4.1 report §4.2.2). Both default off, so the reference
        # forward is unchanged: balance_alpha is the sequence-level balance loss weight (1e-4 in
        # V4.1) and gamma the aux-loss-free bias step (0.001), set by v41f.lm.V42LM.
        self.top_k = n_activated_experts
        self.balance_alpha = 0.0
        self.gamma = 0.0
        self.aux_loss = None
        for name in ("tokens_per_expert", "step_tokens_per_expert"):
            self.register_buffer(name, torch.zeros(n_routed_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("windows", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, x, image_mask=None):
        # v41f has no vision stack: image_mask is accepted for signature parity but
        # unused; the P0 Gate (norm_gate.Gate) has no VL-bias path.
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices, scores = self.gate(x, return_scores=True)
        # scatter_add, not bincount: same numbers, static [E] shape (bincount's output shape is
        # data-dependent to dynamo and broke the compiled graph at every MoE layer)
        flat = indices.flatten()
        counts = torch.zeros(self.n_routed_experts, dtype=torch.long, device=x.device).scatter_add_(
            0, flat, torch.ones_like(flat))
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                c = counts.float()
                self.tokens_per_expert += c
                self.step_tokens_per_expert += c
                self.windows += 1
        if x.is_cuda or self.grouped_on_cpu:
            y = self._routed_grouped(x, weights, indices, counts)
        else:
            y = torch.zeros_like(x, dtype=torch.float32)
            for i, n in enumerate(counts.tolist()):
                if n == 0:
                    continue
                idx, top = torch.where(indices == i)
                if self.stacked:
                    xi = x[idx]
                    h = swiglu_clamp(F.linear(xi, self.w1[i]).float(), F.linear(xi, self.w3[i]).float(),
                                     weights[idx, top, None], self.swiglu_limit)
                    y[idx] += F.linear(h.to(x.dtype), self.w2[i])
                else:
                    y[idx] += _expert_weighted(self.experts[i], x[idx], weights[idx, top, None])
        y += self.shared_experts(x).float()
        self.aux_loss = None
        if self.balance_alpha > 0 and self.training and len(shape) == 3:
            # Sequence-wise balance loss (DeepSeek-V3 eq. 17-20): per sequence, f_i is expert
            # i's selected share scaled by E/(T*k), P_i the mean NORMALIZED affinity; averaged
            # over the batch rows, so it is sequence-wise and not batch-wise.
            b, t = shape[0], shape[1]
            e = self.n_routed_experts
            sel = indices.view(b, t * self.top_k)
            f = torch.zeros(b, e, device=x.device, dtype=torch.float32)
            f.scatter_add_(1, sel, torch.ones_like(sel, dtype=torch.float32))
            f = f * (e / (t * self.top_k))
            p = (scores / scores.sum(-1, keepdim=True).clamp_min(1e-20)).view(b, t, e).mean(1)
            self.aux_loss = self.balance_alpha * (f * p).sum(-1).mean()
        return y.type_as(x).view(shape)

    def _routed_grouped(self, x, weights, indices, counts):
        """GPU dispatch: sort the (token, slot) pairs by expert and run the three expert GEMMs as
        torch._grouped_mm over contiguous per-expert row blocks. Same math as the loop above
        (fp32 gate/up, clamp, routing weight before w2, fp32 accumulate). The expert weights are
        stacked per call; torch.stack is differentiable, so every expert Linear keeps its own
        parameter and state_dict key.
        ponytail: stack per forward costs 3*E*inter*dim bf16 of saved activations per layer; store
        the experts stacked if that memory is ever the limit."""
        k = indices.size(1)
        order = torch.argsort(indices.reshape(-1), stable=True)
        offs = counts.cumsum(0).to(torch.int32)
        tok = torch.div(order, k, rounding_mode="floor")
        rows = x[tok].to(torch.bfloat16)

        def gmm(a, name):
            if self.stacked:
                w = getattr(self, name).to(torch.bfloat16)
            else:
                w = torch.stack([getattr(e, name).weight for e in self.experts]).to(torch.bfloat16)
            if self.moe_gemm == "deepgemm":
                return grouped_linear_fp8(a, w, counts, offs)
            return torch._grouped_mm(a, w.transpose(-2, -1), offs=offs)

        gate = gmm(rows, "w1").float()
        up = gmm(rows, "w3").float()
        h = swiglu_clamp(gate, up, weights.reshape(-1)[order].unsqueeze(-1), self.swiglu_limit)
        out = gmm(h.to(torch.bfloat16).contiguous(), "w2")
        y = torch.zeros_like(x, dtype=torch.float32)
        return y.index_add(0, tok, out.float())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Loader remap between the two expert layouts: per-expert `experts.{i}.w{1,3,2}.weight`
        keys become the stacked `w{1,3,2}` (and back), so a checkpoint from either layout loads."""
        pe = f"{prefix}experts."
        if self.stacked and any(k.startswith(pe) for k in state_dict):
            for name in ("w1", "w3", "w2"):
                keys = [f"{pe}{i}.{name}.weight" for i in range(self.n_routed_experts)]
                if all(k in state_dict for k in keys):
                    state_dict[f"{prefix}{name}"] = torch.stack([state_dict.pop(k) for k in keys])
        elif not self.stacked and f"{prefix}w1" in state_dict:
            for name in ("w1", "w3", "w2"):
                w = state_dict.pop(f"{prefix}{name}")
                for i in range(self.n_routed_experts):
                    state_dict[f"{pe}{i}.{name}.weight"] = w[i]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def update_bias(self, counts):
        """Aux-loss-free bias step (arXiv 2412.19437 §2.1.2): -gamma where the step's load is above
        the mean, +gamma below; then re-centred, since topk ignores a common offset and an
        uncentred sign sum integrates without bound (model.py MoEFFN.update_bias, measured)."""
        err = counts.float() - counts.float().mean()
        self.gate.bias -= self.gamma * torch.sign(err)
        self.gate.bias -= self.gate.bias.mean()

    def commit_token_counts(self):
        """No recompute pass exists (v42 refuses grad_ckpt), so there is no surplus to remove."""

    def diagnostics(self, reset=True):
        """Same fields as model.MoEFFN.diagnostics, for runs/moe_diag.jsonl."""
        c = self.tokens_per_expert.float()
        tot = float(c.sum())
        e = self.n_routed_experts
        used = int((c > 0).sum())
        ent_norm, gini = 0.0, 0.0
        if tot > 0:
            p = (c / tot).clamp_min(1e-12)
            ent_norm = float(-(p * p.log()).sum()) / math.log(e)
            srt = c.sort().values
            i = torch.arange(1, e + 1, dtype=torch.float32, device=c.device)
            gini = min(1.0, max(0.0, float(2.0 * (i * srt).sum() / (e * tot) - (e + 1) / e)))
        out = {"usage_frac": used / e, "used_experts": used, "n_routed": e,
               "entropy_norm": ent_norm, "load_gini": gini,
               "window_steps": int(self.windows), "tokens": int(tot)}
        if reset:
            self.tokens_per_expert.zero_()
            self.windows.zero_()
        return out
