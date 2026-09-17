"""Next-token causal LM loss for v41f.

The head emits fp32 logits regardless of the activation dtype; this upcasts defensively so
a bf16 caller still gets an fp32 reduction. No upstream reference exists (the vendored
model is inference-only), so this is the standard shift-and-cross-entropy and is checked
against an independent hand-written reduction in tests, not against another call into
F.cross_entropy.
"""

import torch
import torch.nn.functional as F


def shifted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Mean next-token cross entropy.

    logits [b, t, vocab], labels [b, t]. Logits at position t predict label t+1, so both
    are shifted (drop the last logit / first label), flattened, and averaged over targets
    not equal to ignore_index. Returns a scalar; an all-masked batch yields NaN (loud),
    matching F.cross_entropy -- callers must mask out such batches rather than get a 0.
    """
    if logits.dim() != 3 or labels.dim() != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            f"expected logits [b,t,v] and labels [b,t], got {tuple(logits.shape)} / {tuple(labels.shape)}"
        )
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
    )
