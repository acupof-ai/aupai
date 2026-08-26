#!/usr/bin/env python3
"""Batched autoregressive sampling (top-p nucleus).

torch is imported lazily inside generate() so this module is importable
without torch/GPU. Model-agnostic: any model returning (logits, _).
"""


def generate(model, prompt_ids, n, max_new, temperature, top_p, device):
    """n responses for one prompt (same length, no padding needed).
    Returns list of n generated token-id lists."""
    import torch
    import torch.nn.functional as F

    eos = 1  # <eos>
    x = torch.tensor(prompt_ids, device=device).repeat(n, 1)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(x[:, -1024:])
            logits = logits[:, -1, :].float() / temperature
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs > top_p
            remove[..., 1:] = remove[..., :-1].clone()  # keep first token above threshold
            remove[..., 0] = False
            probs = F.softmax(sorted_logits.masked_fill(remove, float("-inf")), dim=-1)
            nxt = torch.multinomial(probs, 1)
            nxt = sorted_idx.gather(-1, nxt).squeeze(-1)
            nxt = torch.where(finished, torch.full_like(nxt, eos), nxt)
            x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
            finished |= nxt == eos
            if finished.all():
                break
    return x[:, len(prompt_ids) :].tolist()
