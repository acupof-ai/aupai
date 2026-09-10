"""Per-rank fixed memory of the V4.1 flat stack under DDP (no sharding).

Element groups are counted by instantiating HybridLM at the exact smoke configs, so the
optimizer grouping matches train.py build_optimizers rather than a hand-maintained table.
Float8Linear persistence is checked by converting a build and re-reading dtypes.

Output is the source table for facts/v41.json#v41.per_rank_memory_budget_0910. Run anywhere
the repo deps import (numbers are CPU, no GPU); the assertion at the bottom fails if the
model definition drifts away from the fact.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import HybridLM  # noqa: E402
from train import Cfg, convert_to_fp8_compute  # noqa: E402

GIB = 2**30

V41_MOE = dict(
    d=1024, dim=1024, layers=12, heads=8, ffn_hidden=6912,
    attn_res=False, attn_every=1, csa=True, csa2=True, rope_dims=64,
    n_swa_only_layers=2, moe_experts=48, moe_top_k=3, moe_shared=1,
    moe_expert_ffn=1728,
)
V41_DENSE = dict(
    d=1024, dim=1024, layers=12, heads=8, ffn_hidden=3072,
    attn_res=False, attn_every=1, csa=True, csa2=True, rope_dims=64,
    n_swa_only_layers=2, moe_experts=0,
)


def group_bytes(model):
    groups = {k: 0 for k in ("muon_expert3d", "muon_dense2d", "embed_head",
                             "arq3d", "router", "scalar")}
    total_el = 0
    for n, p in model.named_parameters():
        total_el += p.numel()
        if n.endswith("ffn.router.weight"):
            groups["router"] += p.numel()
        elif n.startswith("blocks.") and (n.endswith("ffn.w13") or n.endswith("ffn.w2")):
            groups["muon_expert3d"] += p.numel()
        elif ("tok" in n) or ("head" in n):
            groups["embed_head"] += p.numel()
        elif p.ndim == 3:
            groups["arq3d"] += p.numel()
        elif p.ndim == 2:
            groups["muon_dense2d"] += p.numel()
        else:
            groups["scalar"] += p.numel()
    return groups, total_el


def budget(tag, over):
    for k, v in over.items():
        setattr(Cfg, k, v)
    Cfg.vocab = 32768  # V4.1 gate tokenizer (rebuilt 2026-09-10), zero padding
    model = HybridLM(Cfg)
    groups, total_el = group_bytes(model)

    # Float8Linear keeps a bf16 master weight and casts per call: verify no persistent fp8.
    model_bf = model.to(torch.bfloat16)
    convert_to_fp8_compute(model_bf)
    persistent = {str(p.dtype): 0 for p in model_bf.parameters()}
    for p in model_bf.parameters():
        persistent[str(p.dtype)] = persistent.get(str(p.dtype), 0) + p.numel() * p.element_size()
    assert "torch.float8_e4m3fn" not in persistent, "unexpected persistent fp8 weight storage"

    weights_bf16 = total_el * 2
    grads_bf16 = total_el * 2
    muon_el = groups["muon_expert3d"] + groups["muon_dense2d"]
    muon_mom_bf16 = muon_el * 2  # zeros_like(grad), Muon Nesterov buffer
    # AdamW m,v on embeddings and routers (fused, fp32 state).
    adamw_el = groups["embed_head"] + groups["router"]
    adamw_state_fp32 = adamw_el * 8
    fixed = weights_bf16 + grads_bf16 + muon_mom_bf16 + adamw_state_fp32

    print(f"== {tag}: {total_el/1e6:.1f}M params")
    for k, v in groups.items():
        print(f"   {k:14s} {v/1e6:9.2f}M el")
    print(f"   weights bf16            {weights_bf16/GIB:6.2f} GiB/rank")
    print(f"   grads bf16              {grads_bf16/GIB:6.2f} GiB/rank")
    print(f"   Muon momentum bf16      {muon_mom_bf16/GIB:6.2f} GiB/rank")
    print(f"   AdamW m+v fp32          {adamw_state_fp32/GIB:6.3f} GiB/rank")
    print(f"   fixed total             {fixed/GIB:6.2f} GiB/rank "
          f"(+ 50 MiB DDP buckets, context; fp32 masters off would add {total_el*4/GIB:.2f})")
    return total_el, fixed


if __name__ == "__main__":
    moe_el, moe_fixed = budget("MoE-48 flat v41smoke", V41_MOE)
    dense_el, dense_fixed = budget("dense ffn3072 (runs e/f)", V41_DENSE)
    assert abs(moe_el - 3_209_500_000) < 5e6, moe_el
    assert abs(dense_el - 200_800_000) < 5e5, dense_el
    assert abs(moe_fixed / GIB - 18.2) < 0.3, moe_fixed / GIB
