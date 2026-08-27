"""Shared helpers for the eval harness."""

import torch
import torch.nn.functional as F


def log_likelihood_joint(model, tok, prompt, choice, device="cuda"):
    """Log-likelihood of choice tokens given prompt (joint tokenization)."""
    ids_p = tok.encode(prompt).ids
    ids_f = tok.encode(prompt + choice).ids
    if len(ids_f) <= len(ids_p):
        return -1e9
    x = torch.tensor([ids_f], device=device)
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        log_probs = F.log_softmax(logits[0], dim=-1)
    return sum(
        log_probs[len(ids_p) + i - 1, tid].item()
        for i, tid in enumerate(ids_f[len(ids_p):])
        if len(ids_p) + i - 1 >= 0
    )
