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


def multi_token_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Direct (UNSHIFTED) cross entropy: logits [b, n, vocab] predict labels [b, n] at the
    SAME position.

    This is the draft loss, not the backbone loss, and the difference is the shift. The
    backbone's logits at t predict t+1, so `shifted_cross_entropy` drops a column from each
    side. A DSpark draft already carries its own offset: query i is fed the token at p+i and
    is scored against the token at p+i+1, so the alignment is built into how the draft inputs
    and labels are constructed, and shifting again here would score every position one step
    off.

    Returns a scalar; an all-masked batch yields NaN (loud), same convention as
    `shifted_cross_entropy`.
    """
    if logits.dim() != 3 or labels.dim() != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            f"expected logits [b,n,v] and labels [b,n], got {tuple(logits.shape)} / {tuple(labels.shape)}"
        )
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def dspark_targets(input_ids: torch.Tensor, prefix_len: int, block_size: int):
    """Split a teacher-forced sequence into (draft inputs, draft labels) for the DSpark draft.

    input_ids [b, t]. The prefix is [0, prefix_len); the draft predicts the next
    `block_size` tokens after it. Teacher forcing means query i is fed the token BEFORE the
    one it predicts:

        draft_input[j] = ids[:, prefix_len - 1 + j]     j = 0..block_size-1
        labels[j]      = ids[:, prefix_len + j]

    so query 0 is fed the last prefix token (the anchor) and is scored against the first
    future token, and query i sees draft positions 0..i under the causal mask -- tokens
    prefix_len-1 .. prefix_len-1+i -- while predicting prefix_len+i. Nothing is visible
    before it is predicted, which is what makes the parallel forward equal to a gold-fed
    sequential decode (prereg v41f_dspark_train_equiv_0917).
    """
    if input_ids.dim() != 2:
        raise ValueError(f"expected input_ids [b,t], got {tuple(input_ids.shape)}")
    t = input_ids.size(1)
    if prefix_len < 1 or block_size < 1 or prefix_len + block_size > t:
        raise ValueError(
            f"need 1 <= prefix_len and prefix_len + block_size <= t, got prefix_len "
            f"{prefix_len}, block_size {block_size}, t {t}")
    draft_inputs = input_ids[:, prefix_len - 1 : prefix_len - 1 + block_size]
    labels = input_ids[:, prefix_len : prefix_len + block_size]
    return draft_inputs, labels
