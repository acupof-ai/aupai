"""P1 training smoke: one CPU fp32-master training step works end to end.

This is the smallest closed loop -- forward, shift-CE (#439), backward, one AdamW update
-- and it proves the assembled model learns on CPU with no tilelang/fp8 kernel. The P0
quant stubs and the pure bf16 Linear path mean nothing here needs GPU.

Grad coverage is asserted by PARAMETER CATEGORY, not by "the total number of nonzero
grads", which a routing change would move around and which cannot name an unconnected
subsystem. The hard-topk Indexer is pinned the other way as a DESIGN premise: its scores
leave the graph as discrete indices, so the CE path gives those four params no gradient
(any training of them needs an STE or an auxiliary loss, decided separately).

CPU uses an fp32 master copy because bf16 backward underflows these small-scale grads to
zero; production keeps bf16 forward + fp32 master weights (AMP), which is the same math
with the optimizer holding the fp32 copy.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.config import v41f_small
from v41f.loss import shifted_cross_entropy
from v41f.model import V41FModel
from v41f.train import train_step

# one leaf per trainable subsystem; each must carry a nonzero finite CE gradient.
_NONZERO_PROBES = (
    "embed.weight",
    "norm.weight",
    "head.weight",
    "layers.0.attn.qproj.wq_b.weight",
    "layers.0.attn.kvproj.wkv.weight",
    "layers.0.attn.oproj.wo_b.weight",
    "layers.0.attn.oproj.wo_a",
    "layers.0.attn.attn_sink",
    "layers.0.ffn.gate.weight",
    "layers.0.ffn.experts.0.w1.weight",
    "layers.0.ffn.shared_experts.w1.weight",
    "layers.0.hc.hc_attn_fn",
    "layers.1.attn.compressor.wkv.weight",
    "layers.1.attn.compressor.wgate.weight",
    "layers.1.attn.compressor.norm.weight",
)

# hard-topk Indexer leaves: scores leave the graph as discrete selected indices, so the
# next-token CE loss has no path to them. Training them requires STE/aux (by design).
_ZERO_GRAD_INDEXER = (
    "layers.1.attn.indexer.wq_b.weight",
    "layers.1.attn.indexer.weights_proj.weight",
    "layers.1.attn.index_key.wk.weight",
    "layers.1.attn.index_key.k_norm.weight",
)


def _fp32_model():
    cfg = v41f_small()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = V41FModel(cfg, max_batch_size=2).train()
    torch.set_default_dtype(prev)
    return model.float(), cfg


def test_single_step_grad_coverage_by_category():
    torch.manual_seed(0)
    model, cfg = _fp32_model()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss = train_step(model, ids, opt)
    assert torch.isfinite(loss)
    params = dict(model.named_parameters())
    for name in _NONZERO_PROBES:
        g = params[name].grad
        assert g is not None, f"{name}: no grad (unconnected subsystem)"
        assert torch.isfinite(g).all() and g.abs().sum() > 0, f"{name}: zero/non-finite grad"
    # design premise: hard-topk indexer params get NO CE gradient
    for name in _ZERO_GRAD_INDEXER:
        g = params[name].grad
        assert g is None or g.abs().sum() == 0, f"{name}: hard-topk indexer unexpectedly received CE grad"


def test_overfit_one_batch_large_drop():
    torch.manual_seed(1)
    model, cfg = _fp32_model()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = train_step(model, ids, opt).item()
    for _ in range(39):
        train_step(model, ids, opt)
    last = train_step(model, ids, opt).item()
    assert last < first * 0.05, f"model did not overfit one batch: {first:.3f} -> {last:.3f}"


def test_lr_zero_mutation_keeps_loss_flat():
    torch.manual_seed(1)
    model, cfg = _fp32_model()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=0.0)  # weight_decay-free frozen update
    opt.param_groups[0]["weight_decay"] = 0.0
    first = train_step(model, ids, opt).item()
    for _ in range(20):
        train_step(model, ids, opt)
    last = train_step(model, ids, opt).item()
    # lr=0 changes nothing, so the repeated loss must be constant to fp32 noise
    assert abs(last - first) < 1e-4, f"lr=0 moved the loss: {first:.6f} -> {last:.6f}"


def test_detached_loss_mutation_has_no_grad():
    """train_step returns a detached scalar; a caller cannot backprop twice through it and
    the graph is freed after the step. Pins the return contract mutation-sensitively."""
    model, cfg = _fp32_model()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss = train_step(model, ids, opt)
    assert not loss.requires_grad, "train_step must return a detached loss"


def test_loss_feeds_gradient_not_random():
    """Sanity that grads derive from the data/loss: a second batch changes which routed
    experts fire and the embed grad is data-dependent."""
    model, cfg = _fp32_model()
    torch.manual_seed(7)
    a = torch.randint(0, cfg.vocab_size, (2, 8))
    shifted_cross_entropy(model(a)[0], a).backward()
    ga = dict(model.named_parameters())["embed.weight"].grad.detach().clone()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(8)
    b = torch.randint(0, cfg.vocab_size, (2, 8))
    shifted_cross_entropy(model(b)[0], b).backward()
    gb = dict(model.named_parameters())["embed.weight"].grad.detach().clone()
    assert (ga - gb).abs().sum() > 0, "embed grad is identical for different data -- disconnected"
