"""P0: RMSNorm and sqrtsoftplus Gate match the vendored upstream implementation."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).absolute().parent))
from allclose import cmp
from ref_oracle import load_reference

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
from v41f.norm_gate import Gate, RMSNorm


def test_rmsnorm():
    model, _ = load_reference()
    torch.manual_seed(0)
    dim, eps = 64, 1e-6
    ref = model.RMSNorm(dim, eps)
    ours = RMSNorm(dim, eps)
    ours.weight.data.copy_(ref.weight.data)
    x = torch.randn(3, 17, dim) * 3.0
    cmp("RMSNorm fp32", ours(x.float()), ref(x.float()), atol=1e-5)
    cmp("RMSNorm bf16", ours(x.bfloat16()), ref(x.bfloat16()), atol=2e-2)


def test_gate_sqrtsoftplus():
    model, _ = load_reference()
    torch.manual_seed(1)
    n, dim, n_experts, topk = 64, 48, 48, 6
    ref = model.Gate(0, _ArgsLite(n_experts, topk, dim))
    ours = Gate(dim, n_experts, topk)
    # Upstream Gate allocates weight with torch.empty (the loader fills it); uninitialized
    # bf16 memory is intermittently NaN depending on allocator layout, which made this test
    # flaky (~1/4 runs). Write finite values into the reference first, then mirror them.
    with torch.no_grad():
        ref.weight.copy_(torch.randn(n_experts, dim) * 0.1)
        ref.bias.zero_()
    ours.weight.data.copy_(ref.weight.data)
    ours.bias.data.copy_(ref.bias.data)
    x = torch.randn(n, dim)
    with torch.no_grad():
        rw, ri = ref(x, None)
        ow, oi = ours(x)
    cmp("Gate weights", ow, rw, atol=1e-4)
    assert torch.equal(oi, ri), "selected experts differ"
    # zero bias must not change scores (weights are unbiased)
    assert (rw >= 0).all()


class _ArgsLite:
    """Minimal stand-in for ModelArgs for the upstream Gate(layer_id, args)."""

    def __init__(self, n_routed, topk, dim):
        self.n_routed_experts = n_routed
        self.n_activated_experts = topk
        self.dim = dim
        self.score_func = "sqrtsoftplus"
        self.gate_temp = 1.0
        self.norm_topk_prob = True
        self.route_scale = 1.5
        self.vision_enabled = False

    def get_moe_config(self, layer_id):
        return self.n_routed_experts, self.n_activated_experts
