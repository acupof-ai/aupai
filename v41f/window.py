"""Sliding-window cache index math, faithful to upstream model.py:409.

get_window_topk_idxs returns which ring-cache slots each query attends; -1 marks an
empty slot. Training (start_pos==0, prefill) is what v41f needs first: one row per
query over its causal window. The decode ring case is implemented for the later
generator but not on the training hot path.
"""
import torch


def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int,
                         device=None):
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), device=device)
        idxs = torch.where(idxs > end, torch.full_like(idxs, -1), idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size, device=device),
                          torch.arange(oldest, device=device)])
        idxs = torch.where(idxs > start_pos, torch.full_like(idxs, -1), idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
