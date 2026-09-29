"""Latent Q, head-shared KV, and grouped low-rank output projections.

Faithful to upstream model.py Attention (single-process shape; the reference shards
these across a tensor-parallel world, v41f-S runs DDP so every projection is local):

- wq_a: dim -> q_lora_rank; RMSNorm; wq_b -> n_heads*head_dim.
- wkv: dim -> head_dim, ONE KV shared by all heads (MQA); kv_norm.
- wo_a: block-diagonal over o_groups, weight [o_groups, o_lora_rank, heads_per_group
  * head_dim], applied by einsum "bsgd,grd->bsgr" (NOT a dense Linear); wo_b maps the
  concatenated group latents back to dim.
"""
import torch
from torch import nn

from v41f.norm_gate import RMSNorm


class QProj(nn.Module):
    def __init__(self, dim: int, q_lora_rank: int, n_heads: int, head_dim: int, eps: float, qk_norm: bool = False):
        super().__init__()
        self.wq_a = nn.Linear(dim, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank, eps)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        # cfg.qk_norm (deviation from the reference): RMSNorm per head on q, before RoPE
        self.head_norm = RMSNorm(head_dim, eps) if qk_norm else None
        self.n_heads = n_heads
        self.head_dim = head_dim

    def latent(self, x):
        return self.q_norm(self.wq_a(x))

    def forward(self, x):
        qr = self.latent(x)
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        if self.head_norm is not None:
            q = self.head_norm(q)
        return q, qr

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # a checkpoint saved without qk_norm loads into a qk_norm model at the identity weight
        k = f"{prefix}head_norm.weight"
        if self.head_norm is not None and k not in state_dict:
            state_dict[k] = torch.ones_like(self.head_norm.weight)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class KVProj(nn.Module):
    """One KV line for every head (MQA): dim -> head_dim."""

    def __init__(self, dim: int, head_dim: int, eps: float):
        super().__init__()
        self.wkv = nn.Linear(dim, head_dim, bias=False)
        self.kv_norm = RMSNorm(head_dim, eps)

    def forward(self, x):
        return self.kv_norm(self.wkv(x))


class GroupedOProj(nn.Module):
    def __init__(self, n_heads: int, head_dim: int, o_groups: int, o_lora_rank: int, dim: int):
        super().__init__()
        if n_heads % o_groups:
            raise ValueError("n_heads must divide o_groups")
        self.n_groups = o_groups
        self.per_group = n_heads // o_groups
        self.o_lora_rank = o_lora_rank
        in_per_group = self.per_group * head_dim
        # block-diagonal: each group's heads project only to that group's latents
        self.wo_a = nn.Parameter(torch.empty(o_groups, o_lora_rank, in_per_group))
        self.wo_b = nn.Linear(o_groups * o_lora_rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.wo_a, a=5 ** 0.5)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        # o: [b, s, n_heads, head_dim] -> group -> latents -> dim
        b, s = o.shape[:2]
        og = o.view(b, s, self.n_groups, self.per_group * o.shape[-1])
        lat = torch.einsum("bsgd,grd->bsgr", og, self.wo_a)
        return self.wo_b(lat.flatten(2))
