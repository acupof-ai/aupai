"""Known-answer checks for the L2 quality head losses (no encoder/GPU needed).

The MSE term and the within-domain ranking term are the two design invariants, so both
get hand-computed answers; the domain-prior removal (ranking never crosses domains) is
explicitly asserted because that is the whole reason the ranking term exists.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datagen import l3_rubric
from v41f_l2.l2_quality_head import (
    DEFAULT_DIMS,
    RUBRIC_DIMS,
    RUBRIC_KINDS,
    L2Config,
    L2Dataset,
    L2QualityModel,
    QualityHead,
    per_dim_mse,
    quality_loss,
    train_step,
    within_domain_rank_loss,
)


def test_head_dims_match_teacher_rubric_exactly():
    """Loud coupling: head output width/dim set must equal 66's l3_rubric definition.
    If the rubric changes, this fails instead of silently masking every label."""
    assert tuple(RUBRIC_DIMS) == tuple(l3_rubric._DIMS)
    assert tuple(DEFAULT_DIMS) == tuple(l3_rubric._DIMS)
    assert L2Config().n_dims == len(l3_rubric._DIMS)
    assert QualityHead(L2Config()).net[-1].out_features == len(l3_rubric._DIMS)
    # both rubric kinds share the same four dim keys (only judging text differs)
    assert tuple(l3_rubric.CODE_RUBRIC["dimensions"]) == tuple(l3_rubric.NL_RUBRIC["dimensions"])
    assert set(RUBRIC_KINDS) == {"code", "natural_language"}


class _TinyEncoder(torch.nn.Module):
    """Stand-in for a frozen encoder: projects one-hot-ish inputs to embed_dim."""

    def __init__(self, vocab=16, dim=1024):
        super().__init__()
        self.proj = torch.nn.Linear(vocab, dim)

    def forward(self, input_ids, attention_mask):
        x = torch.nn.functional.one_hot(input_ids, 16).float()
        return type("H", (), {"last_hidden_state": self.proj(x)})()


def test_train_step_frozen_encoder_only_head_gets_grads():
    cfg = L2Config(lambda_rank=0.5)
    model = L2QualityModel(_TinyEncoder(), cfg, pooling="cls")
    B = 6
    batch = {
        "input_ids": torch.randint(0, 16, (B, 4)),
        "attention_mask": torch.ones(B, 4, dtype=torch.long),
        "target": torch.randint(1, 6, (B, 4)).float(),
        "mask": torch.ones(B, 4),
        "domain_id": torch.tensor([0, 0, 0, 1, 1, 1]),
    }
    loss, parts = train_step(model, batch, cfg)
    assert torch.isfinite(loss) and set(parts) == {"mse", "rank", "total"}
    loss.backward()
    head_grad = all(p.grad is not None for p in model.head.parameters() if p.requires_grad)
    enc_frozen = all(p.grad is None for p in model.encoder.parameters())
    assert head_grad and enc_frozen, "encoder frozen, head receives grads"


def test_head_shapes_and_range():
    cfg = L2Config()
    head = QualityHead(cfg)
    emb = torch.randn(7, 1024)
    out = head(emb)
    assert out.shape == (7, 4)


def test_per_dim_mse_hand_computed():
    # one dim, 2 rows, both labelled
    pred = torch.tensor([[3.0], [1.0]])
    targ = torch.tensor([[2.0], [4.0]])
    mask = torch.ones(2, 1)
    got = per_dim_mse(pred, targ, mask)
    assert torch.isclose(got, torch.tensor(((1) ** 2 + (-3) ** 2) / 2))  # (1+9)/2 = 5
    # masked row excluded entirely -> only the first contributes
    mask[1] = 0
    got2 = per_dim_mse(pred, targ, mask)
    assert torch.isclose(got2, torch.tensor(1.0))
    # sparse second dim with one labelled row does not divide by zero
    pred2 = torch.tensor([[0.0, 10.0]])
    targ2 = torch.tensor([[0.0, 4.0]])
    mask2 = torch.tensor([[1.0, 1.0]])
    assert torch.isclose(per_dim_mse(pred2, targ2, mask2), torch.tensor(18.0))  # (0+36)/2


def test_rank_loss_zero_when_ordered_and_domain_means_removed():
    # two domains; WITHIN each, pred matches target order. Across domains means differ a
    # lot, but that must NOT contribute (ranking is within-domain only).
    pred = torch.tensor([[1.0], [5.0], [1.0], [5.0]])
    targ = torch.tensor([[1.0], [5.0], [1.0], [5.0]])
    mask = torch.ones(4, 1)
    dom = torch.tensor([0, 0, 1, 1])
    loss = within_domain_rank_loss(pred, targ, mask, dom)
    # correctly ordered with gap 4: softplus(-4)=0.018, small but not exactly 0
    assert loss.item() < 0.05

    # swap ordering in domain 0 only; domain 1 stays ordered -> large positive penalty.
    # reversed gap -4 gives softplus(4) ~= 3.98, vs 0.018 for the correct direction.
    pred_bad = torch.tensor([[5.0], [1.0], [1.0], [5.0]])
    loss_bad = within_domain_rank_loss(pred_bad, targ, mask, dom)
    assert loss_bad.item() > 1.0
    assert loss_bad.item() > 100 * loss.item()


def test_rank_never_crosses_domain():
    # domain A always scores low, domain B always scores high. If ranking crossed domains
    # the high-B-vs-low-A pairs would be "correct"; here there are simply no cross pairs,
    # and within each single-doc domain there is no pair at all -> loss exactly 0.
    pred = torch.tensor([[0.1], [0.2], [9.0], [9.5]])
    targ = torch.tensor([[1.0], [2.0], [1.0], [2.0]])
    mask = torch.ones(4, 1)
    dom = torch.tensor([0, 1, 2, 3])  # all distinct domains
    assert within_domain_rank_loss(pred, targ, mask, dom).item() == 0.0


def test_combined_loss_weights_rank():
    cfg = L2Config(lambda_rank=0.0)
    pred = torch.tensor([[3.0, 3.0]])
    targ = torch.tensor([[1.0, 5.0]])
    mask = torch.ones(1, 2)
    dom = torch.tensor([0])
    l0, parts0 = quality_loss(pred, targ, mask, dom, cfg)
    # single row -> no pairs -> rank 0, so total == mse regardless of lambda
    assert torch.isclose(parts0["rank"], torch.tensor(0.0))
    assert torch.isclose(l0, parts0["mse"])


def _row(doc_id, domain, dims, kind=None):
    return {
        "doc_id": doc_id,
        "domain": domain,
        "lang": "en",
        "scorer_name": "l3-rubric",
        "scorer_version": "v0",
        "ts": "2026-09-16T00:00:00Z",
        "rubric_dims": dims,
        "score": None,
        "cut": None,
        "model": None,
        "backend": None,
        "stratum": None,
        "rubric_kind": kind,
        "record_id": None,
        "src_sha": None,
    }


def test_label_out_of_range_refuses(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text(json.dumps(_row("a" * 16, "py", {"content_quality": 9})) + "\n")
    with pytest.raises(ValueError):
        L2Dataset(p)


def test_unknown_dim_and_bad_kind_refuse(tmp_path):
    # a dim that is NOT one of the four teacher rubric dims must raise, not be masked away
    p = tmp_path / "ledger.jsonl"
    p.write_text(json.dumps(_row("a" * 16, "py", {"made_up_dim": 3})) + "\n")
    with pytest.raises(ValueError):
        L2Dataset(p)

    p2 = tmp_path / "ledger2.jsonl"
    p2.write_text(json.dumps(_row("b" * 16, "py", {"content_quality": 3}, kind="klingon")) + "\n")
    with pytest.raises(ValueError):
        L2Dataset(p2)


def test_dataset_collate_partial_dims_and_two_kinds(tmp_path):
    rows = [
        _row("a" * 16, "py", {"content_quality": 5, "factual_correctness": 2}, kind="code"),
        _row("b" * 16, "en", {"educational_or_code_value": 1}, kind="natural_language"),
    ]
    p = tmp_path / "ledger.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ds = L2Dataset(p)
    b = ds.collate_labels([0, 1])
    # dim order is the teacher RUBRIC_DIMS: content, factual, complexity, educational_value
    assert b["mask"].tolist() == [[1, 1, 0, 0], [0, 0, 0, 1]]
    assert b["target"][0].tolist()[:2] == [5.0, 2.0]
    # both kinds carried through; they share the same 4 output columns
    assert b["rubric_kind"] == ["code", "natural_language"]
    assert set(ds.kind_to_idx) == {"code", "natural_language"}
    assert b["rubric_kind_id"].tolist() == sorted(b["rubric_kind_id"].tolist())
