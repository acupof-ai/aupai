"""Sparse attention over gathered KV with a learned attention sink, one softmax.

Upstream kernel.sparse_attn (tilelang, SM100) is not importable on CPU, so P0
validates two independent pure-torch formulations against each other:
- sparse_attn here: online-softmax gather, the shape the training kernel replaces,
- a brute-force explicit softmax in the test (no shared code).

Semantics (model.py:613 and kernel.py:310): q [b,m,h,d]; kv [b,n,d] is head-shared
(MQA); topk_idxs [b,m,topk] selects kv slots per query, -1 = empty; one learned
attn_sink[h] is always added to the softmax denominator but carries no value.
"""
import torch


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    b, m, h, d = q.shape
    valid = topk_idxs >= 0
    safe = topk_idxs.clamp_min(0)                       # [b,m,topk]
    bidx = torch.arange(b, device=q.device)[:, None, None]
    gathered = kv[bidx, safe]                            # [b,m,topk,d]
    scores = torch.einsum("bmhd,bmtd->bmht", q, gathered) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    row_max = scores.amax(dim=-1, keepdim=True)
    row_max = torch.nan_to_num(row_max, neginf=0.0)     # all-empty row -> sink only
    exp = torch.exp(scores - row_max)
    exp = torch.nan_to_num(exp, nan=0.0, posinf=0.0, neginf=0.0)
    sink_term = torch.exp(
        attn_sink.view(1, 1, h) - row_max.squeeze(-1))  # [b,m,h]
    denom = exp.sum(dim=-1) + sink_term
    out = torch.einsum("bmht,bmtd->bmhd", exp, gathered) / denom.unsqueeze(-1)
    return out
