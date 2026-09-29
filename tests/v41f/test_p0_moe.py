"""P0: MoE combine (Gate + routed experts + one shared expert) matches upstream MoE.

Single process (world_size=1): every expert is local, no dist all_reduce. We fill the
reference's torch.empty gate/expert/shared tensors with finite values first, then copy
them weight-for-weight into our MoE, and compare forward on identical bf16 input.
Shapes: v41f_small (8 routed / top2) and full (48 / top6).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).absolute().parent))
from allclose import cmp
from ref_oracle import load_reference

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
from v41f.moe import MoE, _expert_weighted

_MODEL = None


class _Args:
    """Minimal ModelArgs for upstream MoE(layer_id, args); vision off, bf16, no fp4."""

    def __init__(self, dim, inter, n_routed, topk, swiglu_limit=0.0):
        self.dim = dim
        self.moe_inter_dim = inter
        self.n_routed_experts = n_routed
        self.n_activated_experts = topk
        self.n_shared_experts = 1
        self.score_func = "sqrtsoftplus"
        self.gate_temp = 1.0
        self.norm_topk_prob = True
        self.route_scale = 1.5
        self.swiglu_limit = swiglu_limit
        self.expert_dtype = None  # bf16 experts, no fp4/fp8 storage
        self.vision_enabled = False
        self.n_layers = 1

    def get_moe_config(self, layer_id):
        return self.n_routed_experts, self.n_activated_experts


def _fill_ref_and_copy(ref, ours, seed):
    """Give the reference finite gate/expert weights (they are torch.empty) and mirror
    every parameter into our MoE by structural position+name."""
    g = torch.Generator().manual_seed(seed)

    def finite_like(t, scale=0.2):
        return torch.randn(t.shape, generator=g, dtype=torch.float32).to(t.dtype) * scale

    with torch.no_grad():
        # gate weights are fp32 upstream (Gate builds them under set_dtype(float32));
        # our Gate is fp32 too, so copy the finite values directly.
        ref.gate.weight.copy_(finite_like(ref.gate.weight))
        ref.gate.bias.copy_(finite_like(ref.gate.bias, scale=0.5))
        ours.gate.weight.data.copy_(ref.gate.weight.float())
        ours.gate.bias.data.copy_(ref.gate.bias.float())
        # routed experts are bf16 on both sides (our expert Linears are .bfloat16())
        for routed_e, oe in zip(ref.experts, ours.experts, strict=True):
            for nm in ("w1", "w2", "w3"):
                rl = getattr(routed_e, nm)
                ol = getattr(oe, nm)
                rl.weight.copy_(finite_like(rl.weight))
                ol.weight.data.copy_(rl.weight)
        # shared expert
        for nm in ("w1", "w2", "w3"):
            rl = getattr(ref.shared_experts, nm)
            ol = getattr(ours.shared_experts, nm)
            rl.weight.copy_(finite_like(rl.weight))
            ol.weight.data.copy_(rl.weight)


def _matched(dim, inter, n_routed, topk, limit, seed):
    global _MODEL
    if _MODEL is None:
        _MODEL, _ = load_reference()
    args = _Args(dim, inter, n_routed, topk, swiglu_limit=limit)
    # Expert is built with dtype=None -> Linear falls back to the module global
    # default_dtype (import-time fp8). Force bf16 so the reference runs the plain
    # non-quant Linear branch on CPU.
    saved = _MODEL.default_dtype
    _MODEL.default_dtype = torch.bfloat16
    try:
        ref = _MODEL.MoE(0, args)
    finally:
        _MODEL.default_dtype = saved
    ours = MoE(dim, n_routed, topk, inter, route_scale=1.5, swiglu_limit=limit)
    # experts run bf16 (matching upstream non-quant Linear); gate stays fp32.
    for e in ours.experts:
        e.bfloat16()
    ours.shared_experts.bfloat16()
    _fill_ref_and_copy(ref, ours, seed)
    return ref, ours, args


def _run(dim, inter, n_routed, topk, limit, seed):
    ref, ours, _args = _matched(dim, inter, n_routed, topk, limit, seed)
    torch.manual_seed(seed + 1)
    x = (torch.randn(4, 9, dim) * 3.0).bfloat16()
    with torch.no_grad():
        ro = ref(x, None)
        oo = ours(x, None)
    cmp(f"MoE {n_routed}/top{topk} bf16", oo, ro, atol=2e-2)
    assert oo.shape == x.shape
    return ref, ours, x


def test_moe_small_8_top2():
    _run(32, 48, 8, 2, 0.0, seed=11)


def test_moe_full_48_top6():
    _run(64, 96, 48, 6, 10.0, seed=23)


def test_shared_adds_to_routed():
    """Output is routed sum PLUS the unconditional shared expert, not one or the other."""
    _, ours, x = _run(32, 48, 8, 2, 0.0, seed=37)
    with torch.no_grad():
        weights, indices = ours.gate(x.reshape(-1, 32))
        routed = torch.zeros(x.shape[0] * x.shape[1], 32, dtype=torch.float32)
        xr = x.reshape(-1, 32)
        counts = torch.bincount(indices.flatten(), minlength=8).tolist()
        for i in range(8):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            routed[idx] += _expert_weighted(ours.experts[i], xr[idx], weights[idx, top, None]).float()
        shared = ours.shared_experts(xr).float()
        expect = (routed + shared).type_as(x).view_as(x)
    cmp("routed + shared reconstructs MoE", ours(x, None), expect, atol=1e-4)
    # shared expert genuinely moves the output (it is unconditional, not gated by topk)
    assert shared.abs().sum() > 0


def test_weights_are_unbiased_scores():
    """Routing weights come from the UN-biased score (bias selects experts only)."""
    _, ours, x = _run(32, 48, 8, 2, 0.0, seed=51)
    xr = x.reshape(-1, 32)
    with torch.no_grad():
        w, idx = ours.gate(xr)
        # recompute the sqrt(softplus) score by hand from the gate weight, NO bias
        raw = torch.nn.functional.linear(xr.float(), ours.gate.weight.float())
        score = torch.nn.functional.softplus(raw).sqrt()
        gathered = score.gather(1, idx)
        if ours.gate.norm_topk_prob:
            gathered = gathered / (gathered.sum(-1, keepdim=True) + 1e-20)
        gathered = gathered * ours.gate.route_scale
    cmp("gate weights ignore bias", w, gathered, atol=1e-5)
    # but the bias changes SELECTION for at least one token when perturbed
    with torch.no_grad():
        big = ours.gate.bias.clone()
        big.fill_(0.0)
        _, idx0 = ours.gate(xr)
        ours.gate.bias.add_(torch.randn_like(ours.gate.bias) * 5.0)
        _, idxB = ours.gate(xr)
        ours.gate.bias.copy_(big)
    assert not torch.equal(idx0, idxB), "a large bias must steer expert selection"
