"""Shared decoding helpers for chat.py / serve.py."""

import torch


def top_p_sample(logits, top_p):
    """Nucleus (top-p) sampling.

    logits: [1, vocab], already temperature-scaled. Returns the next-token id
    tensor of shape [1, 1] (so callers can torch.cat it and call .item()).
    """
    probs = torch.softmax(logits, dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    keep = torch.cumsum(sp, dim=-1) - sp <= top_p
    sp[~keep] = 0
    sp /= sp.sum(dim=-1, keepdim=True)
    return si.gather(-1, torch.multinomial(sp, 1))
