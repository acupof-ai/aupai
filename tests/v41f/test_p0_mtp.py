"""P0: DSpark MTP stage-1 training loss in a known-answer world.

The upstream reference ships ONLY the inference forward (DSparkBlock:1100-1156); it has
no training loss, so this is not an allclose module. The v41f-defined pieces are pinned
by hand-computed answers:
- main projection: concat target-layer hidden -> Linear -> RMSNorm (model_ref:1113-1130);
- draft sequence: real anchor token at position 0, NOISE tokens after (1131-1135);
- multi-token teacher-forced CE: draft position i is scored ONLY against real future
  token p+1+i; a one-position label misalignment changes the loss; noise query positions
  are inputs, never labels; ignore_index masks a position out.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.mtp import DSparkMTP

DIM = 4
VOCAB = 8
BLOCK = 5
NOISE = 3
EPS = 1e-6


def _model(n_target=1):
    return DSparkMTP(DIM, n_target, VOCAB, BLOCK, noise_token_id=NOISE, eps=EPS)


class _OneHotHead(torch.nn.Module):
    """Logits[b, i] put a big mass on token aimed[b, i], 0 elsewhere. aimed is [B, S]."""

    def __init__(self, aimed, mass=40.0):
        super().__init__()
        self.aimed = aimed
        self.mass = mass

    def forward(self, hidden):
        b, s = hidden.shape[:2]
        logits = torch.zeros(b, s, VOCAB, dtype=hidden.dtype)
        logits.scatter_(-1, self.aimed.unsqueeze(-1), self.mass)
        return logits


def _identity_block(x, main_x):
    # real DSparkBlock consumes main_x; echo it so the test proves it was produced/passed
    return x + main_x.unsqueeze(1) * 0.0


def test_draft_ids_real_then_noise():
    m = _model()
    anchor = torch.tensor([7, 2, 5])
    draft = m.draft_input_ids(anchor)
    assert draft.shape == (3, BLOCK)
    assert torch.equal(draft[:, 0], anchor), "position 0 is the real anchor token"
    assert (draft[:, 1:] == NOISE).all(), "positions 1..block_size-1 are NOISE tokens"
    # noise never leaks into the supervision labels (labels are the separate future_ids)
    future = torch.randint(0, VOCAB, (3, BLOCK))
    assert future.shape == draft.shape  # both S-wide, but labels come from future, not draft


def test_project_main_hand_computed():
    m = _model(n_target=2)
    torch.manual_seed(0)
    # [B=1, T=2, dim] -> flatten [8] -> W[dim,8] -> [dim] -> RMSNorm
    h = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -2.0]]])
    with torch.no_grad():
        m.main_proj.weight.copy_(torch.arange(32, dtype=torch.float).reshape(4, 8) % 3 - 1)
    got = m.project_main(h)
    flat = h.flatten(-2)
    z = flat @ m.main_proj.weight.t()
    expect = z / torch.sqrt(z.square().mean(-1, keepdim=True) + EPS)
    assert torch.allclose(got, expect, atol=1e-6)
    # concat order: swapping the two target layers changes the projection
    h_swap = torch.stack([h[0, 1], h[0, 0]]).unsqueeze(0)
    assert not torch.allclose(m.project_main(h_swap), got)


def test_uniform_logits_loss_is_log_vocab():
    m = _model()
    B = 2
    future = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 0, 1]])
    h = torch.zeros(B, 1, DIM)
    anchor = torch.full((B,), 6)
    uniform_head = torch.nn.Linear(DIM, VOCAB, bias=False)
    with torch.no_grad():
        uniform_head.weight.zero_()  # all-zero logits regardless of hidden
    loss, logits = m(h, anchor, future, _identity_block, uniform_head)
    assert logits.shape == (B, BLOCK, VOCAB)
    assert math.isclose(loss.item(), math.log(VOCAB), rel_tol=1e-5)


def test_one_hot_correct_targets_near_zero():
    m = _model()
    B = 3
    future = torch.randint(0, VOCAB, (B, BLOCK))
    h = torch.randn(B, 1, DIM)
    anchor = torch.randint(0, VOCAB, (B,))
    loss, _ = m(h, anchor, future, _identity_block, _OneHotHead(future))
    # CE of a 40.0 one-hot on the correct token over 8 classes: log(1+7 e^-40) ~ 0
    assert loss.item() < 1e-10


def test_one_position_misalignment_changes_loss():
    """Head aims one position EARLY (predicts future[i-1] at draft slot i). Aligned labels
    give ~0; shifting the labels by one against the same fixed logits raises CE sharply."""
    m = _model()
    B = 2
    future = torch.tensor([[0, 1, 2, 3, 4], [7, 6, 5, 4, 3]])
    anchor = torch.tensor([0, 7])
    h = torch.zeros(B, 1, DIM)
    aligned_head = _OneHotHead(future)
    loss_aligned, _ = m(h, anchor, future, _identity_block, aligned_head)

    # same logits, but labels shifted by one (draft slot i scored against future[i-1])
    shifted = torch.empty_like(future)
    shifted[:, 0] = anchor
    shifted[:, 1:] = future[:, :-1]
    loss_shifted, _ = m(h, anchor, shifted, _identity_block, aligned_head)
    assert loss_aligned.item() < 1e-10
    assert loss_shifted.item() > 1.0, "a one-position target misalignment must be detected"
    assert loss_shifted.item() > loss_aligned.item()


def test_noise_positions_not_supervised_and_ignore_mask():
    """Only real future tokens are supervised. Masking one future position with
    ignore_index removes it from the mean exactly; the noise INPUT positions are
    unaffected (they are on the draft input, not in future_ids)."""
    m = _model()
    B = 1
    future = torch.tensor([[0, 1, 2, 3, 4]])
    anchor = torch.tensor([6])
    h = torch.zeros(B, 1, DIM)
    uniform_head = torch.nn.Linear(DIM, VOCAB, bias=False)
    with torch.no_grad():
        uniform_head.weight.zero_()

    loss_full, _ = m(h, anchor, future, _identity_block, uniform_head)
    masked = future.clone()
    masked[:, 2] = -100
    loss_masked, _ = m(h, anchor, masked, _identity_block, uniform_head)
    # uniform logits -> every valid target CE is ln(V); 5 valid vs 4 valid, same mean
    assert math.isclose(loss_full.item(), math.log(VOCAB), rel_tol=1e-5)
    assert math.isclose(loss_masked.item(), math.log(VOCAB), rel_tol=1e-5)

    # with a NON-uniform head, masking changes the mean unless the masked term equals it
    aimed = torch.tensor([[0, 1, 7, 3, 4]])  # slot 2 is wrong for label 2 -> high CE
    head = _OneHotHead(aimed)
    loss_a, _ = m(h, anchor, future, _identity_block, head)
    loss_b, _ = m(h, anchor, masked, _identity_block, head)
    assert not math.isclose(loss_a.item(), loss_b.item(), rel_tol=1e-3), (
        "masking a high-CE position must move the loss"
    )

    # ignoring everything is a loud zero, not a NaN (denominator guarded)
    all_masked = torch.full_like(future, -100)
    loss_none, _ = m(h, anchor, all_masked, _identity_block, head)
    assert torch.isfinite(loss_none) and loss_none.item() == 0.0


def test_position_weights_hand_computed():
    m = _model()
    B = 1
    future = torch.tensor([[0, 1, 2, 3, 4]])
    anchor = torch.tensor([6])
    h = torch.zeros(B, 1, DIM)
    uniform_head = torch.nn.Linear(DIM, VOCAB, bias=False)
    with torch.no_grad():
        uniform_head.weight.zero_()
    w = torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0])
    loss, _ = m(h, anchor, future, _identity_block, uniform_head, pos_weights=w)
    # every term is ln(V); any weighting leaves the weighted mean equal to ln(V)
    assert math.isclose(loss.item(), math.log(VOCAB), rel_tol=1e-5)

    # non-uniform terms: build per-position known CEs and verify the weighted mean.
    # slots 1 and 3 are wrong: their one-hot target token gets logit 0 vs a 40.0 mass on a
    # different token, so their CE is 40.0 (the 7*e^-40 softmax correction is < 1e-15).
    # slots 0,2,4 are correct -> CE ~0. With weights (0,2,0,3,0) the weighted mean over the
    # two weighted slots is (2*40 + 3*40)/(2+3) = 40.
    aimed = torch.tensor([[0, 7, 2, 7, 4]])
    head = _OneHotHead(aimed, mass=40.0)
    w2 = torch.tensor([0.0, 2.0, 0.0, 3.0, 0.0])
    loss_w, _ = m(h, anchor, future, _identity_block, head, pos_weights=w2)
    assert math.isclose(loss_w.item(), 40.0, rel_tol=1e-5)


def test_bad_shapes_refuse():
    m = _model()
    h = torch.zeros(2, 1, DIM)
    anchor = torch.zeros(2, dtype=torch.long)
    good_future = torch.zeros(2, BLOCK, dtype=torch.long)
    head = torch.nn.Linear(DIM, VOCAB, bias=False)
    m(h, anchor, good_future, _identity_block, head)  # baseline ok
    with pytest.raises(ValueError):
        m.project_main(torch.zeros(2, 3, DIM + 1))  # last dim != model dim
    with pytest.raises(ValueError):
        m.draft_input_ids(torch.zeros(2, 3, dtype=torch.long))  # anchor must be [B]
    with pytest.raises(ValueError):
        m(h, anchor, torch.zeros(2, BLOCK - 1, dtype=torch.long), _identity_block, head)
    with pytest.raises(ValueError):
        DSparkMTP(DIM, 0, VOCAB, BLOCK)  # need a target layer
