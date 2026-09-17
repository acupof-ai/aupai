"""P1 training gate: the default-constructed model is trainable before any checkpoint load.

Upstream allocates every weight as torch.empty and fills it from a checkpoint, so the
vendored model specifies no init. The v41f training side must initialise its own weights;
an uninitialised leaf that happens to read zero (the fp32 LM head) makes the logits
constant, and the head then backprops a zero gradient into the entire backbone while its
own weight still trains -- a silent "everything looks fine, nothing learns" failure that
forward allclose cannot see because those tests copy finite weights in.

These gates pin the training-critical init properties: every parameter is finite (no
torch.empty garbage), random-ids logits are non-constant, and a backward from shift-CE
reaches the backbone with nonzero finite gradients. The CPU step runs an fp32 master copy
(bf16 backward underflows at this scale; production keeps bf16 forward + fp32 master/AMP).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.config import v41f_small
from v41f.model import V41FModel

# backbone leaves one per trainable sub-system; the LM head alone must not be the only
# parameter with a gradient.
_BACKBONE_PROBES = (
    "embed.weight",
    "layers.0.attn_norm.weight",
    "layers.0.attn.attn_sink",
    "layers.0.attn.qproj.wq_b.weight",
    "layers.0.ffn.gate.weight",
    "layers.0.ffn.shared_experts.w2.weight",
    "layers.0.hc.hc_attn_fn",
)


def _build(cfg, fp32_master):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = V41FModel(cfg, max_batch_size=2).train()
    torch.set_default_dtype(prev)
    return model.float() if fp32_master else model


def _shift_ce(logits, ids, vocab):
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1))


def test_no_uninitialized_or_nonfinite_leaves():
    cfg = v41f_small()
    model = _build(cfg, fp32_master=False)
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert not bad, f"non-finite (uninitialized empty) parameters: {bad}"


def test_default_model_logits_non_constant_and_backbone_gets_grad():
    cfg = v41f_small()
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))

    # production bf16 forward: random-ids logits must carry position/vocab variance, not
    # collapse to one constant vector (the zero-head signature).
    m16 = _build(cfg, fp32_master=False)
    with torch.no_grad():
        logits16, _ = m16(ids)
    assert torch.isfinite(logits16).all()
    assert logits16.std().item() > 0.1, f"constant logits, std={logits16.std().item()}"

    # fp32 master backward: the shift-CE gradient must reach the backbone, not only the head
    m = _build(cfg, fp32_master=True)
    logits, _ = m(ids)
    loss = _shift_ce(logits, ids, cfg.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
    params = dict(m.named_parameters())
    head_grad = params["head.weight"].grad
    assert head_grad is not None and torch.isfinite(head_grad).all() and head_grad.abs().sum() > 0
    for name in _BACKBONE_PROBES:
        g = params[name].grad
        assert g is not None, f"{name} got no gradient (head severed the backbone)"
        assert torch.isfinite(g).all() and g.abs().sum() > 0, f"{name} zero/non-finite grad"


def test_zero_head_mutation_severs_backbone_and_is_red():
    """The failure the init must prevent: a zeroed (uninitialised) head makes logits
    constant and the backbone grad vanish. This is the mutation that goes green silently
    without the gates above."""
    cfg = v41f_small()
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    m = _build(cfg, fp32_master=True)
    with torch.no_grad():
        m.head.weight.zero_()
    logits, _ = m(ids)
    assert logits.std().item() <= 0.1, "zero head must collapse logits"
    _shift_ce(logits, ids, cfg.vocab_size).backward()
    embed_grad = dict(m.named_parameters())["embed.weight"].grad
    assert embed_grad.abs().sum().item() == 0.0, "zero head must zero the backbone grad"
