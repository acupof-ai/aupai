"""--moe_router_wd: the router AdamW group gets the configured decay, it survives an optimizer-state
resume from a checkpoint saved at 0.0, and one step on a zero gradient shrinks the rows by lr*wd.
CPU only: python3 scripts/test_router_wd.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from train import Cfg, HybridLM, build_optimizers, reapply_router_wd  # noqa: E402

Cfg.d, Cfg.heads, Cfg.layers, Cfg.ffn_hidden, Cfg.vocab, Cfg.seq = 64, 2, 2, 128, 100, 16
Cfg.moe_experts, Cfg.moe_top_k, Cfg.moe_shared, Cfg.moe_expert_ffn, Cfg.moe_layers = 8, 3, 1, 32, "0-1"
Cfg.moe_router_lr = 1e-3


def router_opt(model, wd):
    Cfg.moe_router_wd = wd
    opts = build_optimizers(model, Cfg)
    return opts, {o.aupai_group: o for o in opts}["moe_router"]


torch.manual_seed(0)
m = HybridLM(Cfg)
old_opts, old_r = router_opt(m, 0.0)
assert all(g["weight_decay"] == 0.0 for g in old_r.param_groups)
saved = [o.state_dict() for o in old_opts]

new_opts, new_r = router_opt(m, 0.1)
assert all(g["weight_decay"] == 0.1 for g in new_r.param_groups), "build ignores moe_router_wd"
for o, sd in zip(new_opts, saved, strict=True):
    o.load_state_dict(sd)
assert all(g["weight_decay"] == 0.0 for g in new_r.param_groups), (
    "precondition: load_state_dict is expected to restore the saved wd 0.0; if it no longer does, "
    "this test no longer exercises the re-apply")
assert reapply_router_wd(new_opts, Cfg) == 0.1
assert all(g["weight_decay"] == 0.1 and g["initial_wd"] == 0.1 for g in new_r.param_groups)

rows = [p for g in new_r.param_groups for p in g["params"]]
before = [p.detach().clone() for p in rows]
for p in rows:
    p.grad = torch.zeros_like(p)
new_r.step()
lr = new_r.param_groups[0]["lr"]
for b, p in zip(before, rows, strict=True):
    torch.testing.assert_close(p.detach(), b * (1 - lr * 0.1), rtol=0, atol=1e-6)

Cfg.moe_router_wd = 0.0
assert reapply_router_wd(new_opts, Cfg) == 0.0
print("router wd: build applies it, resume re-applies over the checkpoint's 0.0, zero-grad step shrinks rows by lr*wd OK")
