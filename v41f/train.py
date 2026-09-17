"""One CPU training step for v41f.

Deliberately minimal: the whole forward/backward/update the smoke test needs, not a
trainer loop (no data loader, scheduler, checkpointing, AMP wrapper, DDP). Production runs
bf16 forward with fp32 master weights (the optimizer owns the fp32 copy); on CPU the smoke
test simply passes a model cast to fp32, so this helper is dtype-agnostic.
"""


from .loss import shifted_cross_entropy


def train_step(model, input_ids, optimizer, ignore_index: int = -100):
    """forward -> shift-CE -> backward -> AdamW(step); returns the detached scalar loss."""
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(input_ids)
    loss = shifted_cross_entropy(logits, input_ids, ignore_index=ignore_index)
    loss.backward()
    optimizer.step()
    return loss.detach()
