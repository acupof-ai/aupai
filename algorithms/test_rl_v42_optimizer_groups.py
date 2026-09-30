#!/usr/bin/env python3
"""Under --arch v42 the RL trainer must group parameters the way the V4.1 optimizer does.

algorithms/rl_code_trainer.build_optimizer assigned Muon by `p.ndim == 2` and AdamW to
everything else. On the v41f V4.1 stack that mis-groups in three directions at once, silently:

  embed.weight / head.weight  2-D -> Muon        must be SinkhornMomentum
  *norm.weight                1-D -> AdamW wd 0  must be AdamW wd 0.1
  stacked expert tensors      3-D -> AdamW wd 0  must be Muon

Nothing raises in any of the three; the run trains and the weights are updated by the wrong
rule. So this test asserts MEMBERSHIP -- every trainable parameter, by name, in exactly one
group with the right weight decay -- not that build succeeds. The negative control is the old
ndim==2 rule on the same model, which must place embed.weight in Muon.

Second subject: activation checkpointing. v42 has no `grad_ckpt`; V41FModel reads
`self.block_ckpt`, assigned at construction (v41f/model.py:82) and consumed at :239. The old
`cfg.grad_ckpt = True; model.grad_ckpt = True` therefore set a dead attribute on a V42LM and
left every block un-checkpointed with no error.

CPU only, tiny shapes, no checkpoint on disk.

    python3 algorithms/test_rl_v42_optimizer_groups.py            # same as --selftest
"""

import os
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

VOCAB = 97


def _cfg():
    """A live train.Cfg namespace naming the v42 arch at a tiny shape (same recipe as
    tests/test_loader_arch.py, which is the committed precedent for a CPU-sized V41FConfig)."""
    from train import Cfg
    from v41f.config import v42_s24

    vc = replace(
        v42_s24(vocab_size=VOCAB),
        dim=64, n_layers=5, n_heads=4, head_dim=32, rope_head_dim=16, q_lora_rank=32,
        o_groups=2, o_lora_rank=16, window_size=4, compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layers=(1, 3), index_source_layers=(1, 3), index_n_heads=4, index_head_dim=32,
        index_topk=3, n_routed_experts=4, n_activated_experts=2, moe_inter_dim=32, hc_mult=2,
    )
    vc.validate()
    d = {k: v for k, v in vars(Cfg).items() if not k.startswith("_") and not callable(v)}
    d.update(arch="v42", vocab=VOCAB, v42_cfg=asdict(vc), grad_ckpt=False)
    return SimpleNamespace(**d)


def _ids(params):
    return {id(p) for p in params}


def test_groups():
    from train import build_model
    from v41f.optim import SinkhornMomentum, V42Muon, v42_param_groups

    from rl_code_trainer import build_optimizer, build_rl_optimizers, is_v42

    cfg = _cfg()
    model = build_model(cfg)
    assert is_v42(cfg) and type(model).__name__ == "V42LM", type(model).__name__

    opts = build_rl_optimizers(model, cfg, lr=1e-3)
    assert [type(o) for o in opts] == [V42Muon, SinkhornMomentum, torch.optim.AdamW], \
        [type(o).__name__ for o in opts]
    muon, sink, adam = opts
    muon_ids = _ids(p for g in muon.param_groups for p in g["params"])
    sink_ids = _ids(p for g in sink.param_groups for p in g["params"])
    decay_ids = _ids(adam.param_groups[0]["params"])
    nodecay_ids = _ids(adam.param_groups[1]["params"])
    assert adam.param_groups[0]["weight_decay"] == 0.1, adam.param_groups[0]["weight_decay"]
    assert adam.param_groups[1]["weight_decay"] == 0.0, adam.param_groups[1]["weight_decay"]

    # EXACT PARTITION. Every trainable parameter in exactly one group, and the four groups
    # together covering all of them -- a mis-grouping that moved a parameter and a
    # mis-grouping that dropped one look the same from a per-group count.
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    all_ids = muon_ids | sink_ids | decay_ids | nodecay_ids
    total = len(muon_ids) + len(sink_ids) + len(decay_ids) + len(nodecay_ids)
    assert total == len(all_ids), f"a parameter is in two optimizer groups ({total} vs {len(all_ids)})"
    assert all_ids == _ids(p for _, p in trainable), (
        f"{len(all_ids)} grouped against {len(trainable)} trainable parameters")

    # MEMBERSHIP BY NAME, the four rules of v41f/optim.py:130 stated independently here so the
    # test fails if v42_param_groups itself changes rule without this being revisited.
    wrong = []
    for n, p in trainable:
        if n in ("embed.weight", "head.weight"):
            want, got = "sinkhorn", sink_ids
        elif n.endswith("norm.weight") or n.endswith("ffn.gate.weight"):
            want, got = "adamw_decay", decay_ids
        elif p.ndim >= 2:
            want, got = "muon", muon_ids
        else:
            want, got = "adamw_nodecay", nodecay_ids
        if id(p) not in got:
            where = next(k for k, s in (("muon", muon_ids), ("sinkhorn", sink_ids),
                                        ("adamw_decay", decay_ids), ("adamw_nodecay", nodecay_ids))
                         if id(p) in s)
            wrong.append(f"{n} ({tuple(p.shape)}): wanted {want}, got {where}")
    assert not wrong, "mis-grouped:\n  " + "\n  ".join(wrong)

    counts = {k: len(v) for k, v in v42_param_groups(model).items()}
    assert counts["sinkhorn"] == 2, counts          # embed.weight + head.weight, both present
    assert counts["muon"] > 0 and counts["adamw_decay"] > 0 and counts["adamw_nodecay"] > 0, counts
    print(f"  v42 partition exact over {len(trainable)} trainable params: {counts}; "
          f"adamw wd 0.1 / 0.0")

    # NEGATIVE CONTROL: the rule this replaced, on the same model. It must put the embedding
    # under Muon -- otherwise the assertions above would pass on the defect too.
    from rl_code_trainer import build_rounder

    old = build_optimizer(list(model.parameters()), lr=1e-3, rounder=build_rounder())
    old_muon = _ids(p for o in old if isinstance(o, V42Muon) or type(o).__name__ == "Muon"
                    for g in o.param_groups for p in g["params"])
    emb = dict(model.named_parameters())["embed.weight"]
    assert id(emb) in old_muon, "the ndim==2 rule did not put embed.weight in Muon -- the " \
                                "negative control is not exercising the defect"
    n_3d = sum(1 for _, p in trainable if p.ndim >= 3)
    assert n_3d > 0, "this shape has no 3-D expert tensor, so the third mis-grouping is untested"
    old_rest = _ids(p for o in old if not (isinstance(o, V42Muon) or type(o).__name__ == "Muon")
                    for g in o.param_groups for p in g["params"])
    assert all(id(p) in old_rest for _, p in trainable if p.ndim >= 3), \
        "the ndim==2 rule did not drop the 3-D expert tensors out of Muon"
    print(f"  negative control: the ndim==2 rule sends embed.weight to Muon and drops "
          f"{n_3d} 3-D expert tensors out of it")
    return model, cfg


def test_checkpointing(model, cfg):
    from rl_code_trainer import set_activation_checkpointing

    assert model.block_ckpt is False, model.block_ckpt
    # THE DEFECT: the two lines this replaced, on a V42LM.
    cfg.grad_ckpt = True
    model.grad_ckpt = True
    assert model.block_ckpt is False, (
        "setting grad_ckpt changed block_ckpt -- the defect is not reproducible here, so the "
        "fix below proves nothing")
    flag = set_activation_checkpointing(model, cfg, True)
    assert flag == "block_ckpt", flag
    assert model.block_ckpt is True
    assert cfg.v42_cfg["block_ckpt"] is True, "the saved cfg must record the flag it trained under"
    print("  grad_ckpt=True leaves block_ckpt False on a V42LM; "
          "set_activation_checkpointing sets block_ckpt and records it in cfg.v42_cfg")


def test_hybrid_unchanged():
    """The hybrid arm must keep the ndim==2 split: this is a v42 branch, not a rewrite."""
    import torch.nn as nn

    from rl_code_trainer import build_rl_optimizers, build_rounder, set_activation_checkpointing

    cfg = SimpleNamespace(arch="hybrid", grad_ckpt=False)
    m = nn.Sequential(nn.Linear(4, 4))
    m.grad_ckpt = False
    opts = build_rl_optimizers(m, cfg, lr=1e-3, rounder=build_rounder())
    kinds = sorted(type(o).__name__ for o in opts)
    assert kinds == ["AdamW", "Muon"], kinds
    assert set_activation_checkpointing(m, cfg, True) == "grad_ckpt"
    assert cfg.grad_ckpt is True and m.grad_ckpt is True
    print(f"  hybrid arm unchanged: {kinds}, flag grad_ckpt")


def selftest():
    model, cfg = test_groups()
    test_checkpointing(model, cfg)
    test_hybrid_unchanged()
    print("rl v42 optimizer-group selftest OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--selftest":
        sys.exit(f"usage: {sys.argv[0]} [--selftest]")
    selftest()
