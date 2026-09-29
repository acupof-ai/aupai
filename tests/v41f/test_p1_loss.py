"""P1: shifted next-token cross entropy. The vendored tree has no training loss, so there
is no upstream number to match; correctness is pinned against an independent hand-written
log-sum-exp reduction and exact hand-computed cases, plus shift/label and ignore-index
mutations."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.loss import shifted_cross_entropy


def _manual_ce(logits, labels, ignore_index=-100):
    """Independent reference reduction (no F.cross_entropy): shift, log-softmax via max
    stability, mean over non-ignored targets."""
    sl = logits[:, :-1, :].float()
    ta = labels[:, 1:]
    b, t, v = sl.shape
    flat = sl.reshape(b * t, v)
    targets = ta.reshape(-1)
    picked = flat[torch.arange(flat.size(0)), targets.clamp_min(0)]
    lse = flat.amax(-1) + (flat - flat.amax(-1, keepdim=True)).exp().sum(-1).log()
    nll = lse - picked
    valid = targets.ne(ignore_index)
    return (nll * valid).sum() / valid.sum()


def test_known_uniform_distribution():
    # all logits equal -> each token predicts a uniform distribution -> ln(vocab), exactly
    b, t, v = 2, 5, 17
    logits = torch.zeros(b, t, v)
    labels = torch.randint(0, v, (b, t))
    loss = shifted_cross_entropy(logits, labels)
    assert torch.allclose(loss, torch.tensor(math.log(v)), atol=1e-6), loss.item()


def test_perfect_one_hot_predictions_zero_loss():
    # logits that put huge mass on the NEXT label give loss ~0
    b, t, v = 1, 4, 8
    labels = torch.tensor([[3, 7, 1, 5]])
    logits = torch.full((b, t, v), -1e4)
    for pos in range(t - 1):
        logits[0, pos, labels[0, pos + 1]] = 1e4
    loss = shifted_cross_entropy(logits, labels)
    assert loss.item() < 1e-6, loss.item()


def test_matches_manual_reduction_random():
    torch.manual_seed(0)
    logits = torch.randn(3, 11, 23)
    labels = torch.randint(0, 23, (3, 11))
    assert torch.allclose(shifted_cross_entropy(logits, labels), _manual_ce(logits, labels), atol=1e-5)


def test_fp32_output_and_bf16_input():
    logits = torch.randn(2, 6, 10).bfloat16()
    labels = torch.randint(0, 10, (2, 6))
    loss = shifted_cross_entropy(logits, labels)
    assert loss.dtype == torch.float32
    ref = _manual_ce(logits.float(), labels)
    assert torch.allclose(loss, ref, atol=2e-2)


def test_ignore_index_excluded():
    torch.manual_seed(1)
    b, t, v = 1, 4, 12
    logits = torch.randn(b, t, v)
    labels = torch.tensor([[5, -100, 9, -100]])  # shifted targets: -100, 9, -100 -> only label 9 counts
    got = shifted_cross_entropy(logits, labels)
    # exactly one valid shifted target: predict label 9 at position 1
    sl = logits[:, :-1, :].float()
    one = torch.nn.functional.cross_entropy(sl[0, 1].reshape(1, -1), torch.tensor([9]))
    assert torch.allclose(got, one, atol=1e-6), (got.item(), one.item())


def test_all_masked_is_loud_nan_not_zero():
    logits = torch.randn(1, 4, 8)
    labels = torch.full((1, 4), -100)
    loss = shifted_cross_entropy(logits, labels)
    assert torch.isnan(loss), "an all-masked batch must be NaN (loud), not a silent 0"


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        shifted_cross_entropy(torch.randn(2, 5, 8), torch.randint(0, 8, (2, 6)))


def test_shift_is_by_one_mutation():
    """Mutation guard: aligning the wrong position (predicting the CURRENT token) changes
    the answer; the loss must use logits t -> label t+1."""
    torch.manual_seed(2)
    b, t, v = 2, 7, 16
    logits = torch.randn(b, t, v)
    labels = torch.randint(0, v, (b, t))
    shifted = shifted_cross_entropy(logits, labels)
    # wrong: no shift, logits t -> label t
    wrong = torch.nn.functional.cross_entropy(logits.float().reshape(-1, v), labels.reshape(-1))
    assert not torch.allclose(shifted, wrong, atol=1e-4), (shifted.item(), wrong.item())

    # and an explicit label-left/right swap must move it
    swapped = shifted_cross_entropy(logits, torch.roll(labels, shifts=1, dims=-1))
    assert not torch.allclose(shifted, swapped, atol=1e-4), (shifted.item(), swapped.item())
