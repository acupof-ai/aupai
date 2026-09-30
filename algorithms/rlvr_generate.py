#!/usr/bin/env python3
"""Batched autoregressive sampling (top-p nucleus).

torch is imported lazily inside generate() so this module is importable
without torch/GPU. Model-agnostic: any model returning (logits, _).
"""


def generate(model, prompt_ids, n, max_new, temperature, top_p, device, ctx_window=None):
    """n responses for one prompt (same length, no padding needed).
    Returns list of n generated token-id lists.

    ctx_window=None (the default and the only value any trainer passes) feeds the WHOLE
    prompt+prefix to the model at every step, because that is the context the loss is
    computed over: rlvr_trainer.seq_logprob runs one forward on cat([prompt, gen]) and reads
    log_probs at positions plen-1..-1. Until 2026-09-30 this line read `model(x[:, -1024:])`,
    a fixed 1024-token window. The policy that ACTED was then conditioned on a different
    context from the policy being SCORED, so exp(seq_lp - old_lp) was not an importance ratio
    between two policies on one context -- it compared two different conditionals, and the
    GSPO clip bounded the wrong quantity. It rarely bit at the math shape (MAX_PROMPT 512 +
    max_new 512 = 1024, so the window clipped only the final token) and bites from the first
    generated token on the stdin code shape (MAX_PROMPT_STDIN 1024 + MAX_NEW_STDIN 562 =
    1586). ctx_window survives only as the negative control in
    algorithms/test_rollout_context_parity.py, which asserts the default reproduces
    seq_logprob's own conditionals and that ctx_window=1024 does not.
    """
    import torch
    import torch.nn.functional as F

    eos = 1  # <eos>
    x = torch.tensor(prompt_ids, device=device).repeat(n, 1)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(x if ctx_window is None else x[:, -ctx_window:])
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
    # Truncate each row at its own first <eos>: the loop forces <eos> into rows
    # that stopped early, and a rectangular slice would put that padding inside
    # the RL loss mask for tokens the policy never emitted.
    out = []
    for row in x[:, len(prompt_ids) :].tolist():
        if eos in row:
            row = row[: row.index(eos) + 1]  # keep the <eos> the policy chose
        out.append(row)
    return out
