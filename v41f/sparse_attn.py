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
                topk_idxs: torch.Tensor, softmax_scale: float,
                slot_weight: "torch.Tensor | None" = None) -> torch.Tensor:
    """`slot_weight` [b,m,1,k] optionally scales each gathered slot's softmax numerator.

    It exists for the training-only straight-through indexer signal (`v41f/indexer_ste.py`),
    which supplies a tensor that is exactly 1.0 in forward and carries a softmax derivative
    in backward. The default None is the faithful path: no multiplication happens at all, so
    the inferential numerics are the ones the reference defines.

    Callers that DO pass it must be aware that `exp * w` with `w == 1.0` is bit-equal to
    `exp` (the multiply is exact for 1.0, and `exp` is finite and non-negative by the
    nan_to_num above), which is what keeps the hard forward bit-identical rather than merely
    close.
    """
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
    if slot_weight is not None:
        exp = exp * slot_weight
    sink_term = torch.exp(
        attn_sink.view(1, 1, h) - row_max.squeeze(-1))  # [b,m,h]
    denom = exp.sum(dim=-1) + sink_term
    out = torch.einsum("bmht,bmtd->bmhd", exp, gathered) / denom.unsqueeze(-1)
    return out
