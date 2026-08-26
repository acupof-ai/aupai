"""CPU self-check for train.py architecture changes (no GPU deps needed).

Checks: AttnRes fwd/bwd (Full, Block, grad_ckpt), zero-init == uniform mean, and legacy checkpoint
round-trip: old-key state_dict -> load (remap) -> save -> load, identical key set and outputs.
Run: python scripts/test_arch_compat.py
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402

if train.chunk_kda is None:  # no fla on this machine: shape-preserving stand-in
    train.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)
from train import Cfg, HybridLM, build_optimizers  # noqa: E402

Cfg.d, Cfg.heads, Cfg.layers, Cfg.ffn_hidden, Cfg.vocab, Cfg.seq = 64, 2, 4, 128, 100, 16
x = torch.randint(0, 100, (2, 16))
y = torch.randint(0, 100, (2, 16))

for blocks, ckpt, dyn in [
    (0, False, False),
    (3, False, False),
    (5, False, False),
    (0, True, False),
    (0, False, True),
]:
    Cfg.attn_res, Cfg.attn_res_blocks, Cfg.grad_ckpt, Cfg.attn_res_dyn_q = True, blocks, ckpt, dyn
    m = HybridLM(Cfg)
    h, _ = m(x, y)
    h.sum().backward()
    assert m.final_ar.q.grad is not None and torch.isfinite(h).all()
    assert blocks == 0 or len(m.ar_block_ends) == blocks, "Block AttnRes must produce exactly N blocks"
    assert len(build_optimizers(m, Cfg)) == 4

srcs = [torch.randn(1, 3, 8) for _ in range(5)]
assert torch.allclose(train.AttnRes(8)(srcs), sum(srcs) / 5, atol=1e-6), "zero-init must equal uniform mean"
dq = train.AttnRes(8, dyn_q=True)
nn.init.zeros_(dq.dyn[1].weight)
assert torch.allclose(dq(srcs), sum(srcs) / 5, atol=1e-6), "dyn_q zero-init must equal uniform mean"
Cfg.attn_res_dyn_q = False

# legacy checkpoint: split fused weights back into old keys
Cfg.attn_res, Cfg.grad_ckpt = False, False
old = HybridLM(Cfg)
legacy = {}
for k, v in old.state_dict().items():
    if k.endswith(".gb.weight"):  # gate|beta|pad -> gate_proj, beta_proj
        legacy[k.replace("gb", "gate_proj")] = v[: Cfg.d]
        legacy[k.replace("gb", "beta_proj")] = v[Cfg.d : Cfg.d + Cfg.heads]
        continue
    for fused, (a, b) in {"w13": ("w1", "w3"), "kv_up": ("k_up", "v_up"), "qg": ("q", "gate")}.items():
        if k.endswith(f".{fused}.weight"):
            va, vb = v.chunk(2)
            legacy[k.replace(fused, a)] = va
            legacy[k.replace(fused, b)] = vb
            break
    else:
        legacy[k] = v
Cfg.attn_res = True
new = HybridLM(Cfg)
new.load_state_dict(legacy)
assert new.attn_res is False and Cfg.attn_res is False, "old ckpt must disable AttnRes"
assert set(new.state_dict()) == set(old.state_dict()), "round-trip key set changed"
again = HybridLM(Cfg)
again.load_state_dict(new.state_dict())
with torch.no_grad():
    assert torch.allclose(old(x)[0], new(x)[0]) and torch.allclose(old(x)[0], again(x)[0])
# optimizer plumbing: schedule gates wd decay to Muon, snapshot is a real copy, conv kernels off the scalar group
Cfg.attn_res = False
m = HybridLM(Cfg)
opts = build_optimizers(m, Cfg)
assert all(p.ndim != 3 for p in opts[2].param_groups[0]["params"]), "conv kernels must not be in scalar group"
train.set_schedule(opts, 0, 100, Cfg)
assert opts[1].param_groups[0]["weight_decay"] == Cfg.embed_wd, "embedding wd must not be overwritten"
assert opts[0].param_groups[0]["weight_decay"] == Cfg.muon_wd
train.set_schedule(opts, 100, 100, Cfg)
assert opts[0].param_groups[0]["weight_decay"] == 0.0
assert train.lr_mult(10**6, 100, Cfg) == Cfg.final_lr_frac, "lr must stay at the floor past total (resume)"
m(x, y)[0].sum().backward()
for o in opts:
    o.step()
snap = train.opt_snapshot(opts)
before = next(v for st in snap[1]["state"].values() for v in st.values() if torch.is_tensor(v)).clone()
m(x, y)[0].sum().backward()
for o in opts:
    o.step()
after = next(v for st in snap[1]["state"].values() for v in st.values() if torch.is_tensor(v))
assert torch.equal(before, after), "snapshot must not alias live optimizer state"
# KDA decay init: mean retention exp(-softplus(dt_bias)) ~ 0.9 (was 0.5 with zero init at g=0)
dt_bias = m.blocks[0].mixer.dt_bias
ret = torch.exp(-torch.nn.functional.softplus(dt_bias)).mean().item()
assert 0.85 < ret < 0.99, ret
# doc boundaries: row starts + positions after <eos>, over the flattened stream
idx = torch.tensor([[5, 1, 7, 7], [1, 1, 3, 3]])
cu = train.doc_cu_seqlens(idx, eos_id=1)
assert cu.tolist() == [0, 2, 4, 5, 6, 8] and cu.dtype == torch.int32, cu
m = HybridLM(Cfg)
assert m(x, y, train.doc_cu_seqlens(x, 1))[0].shape == (2, 16, Cfg.d)
print("test_arch_compat OK")
