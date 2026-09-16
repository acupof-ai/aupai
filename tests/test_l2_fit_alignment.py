"""Small-scale end-to-end fit: de's #397 ledger loader -> our head, 4-dim column alignment.

Runs only when datagen/l2_dataset.py (PR #397) is importable; skips otherwise so this file
is safe on main before that merge. It overfits a tiny synthetic ledger and proves:
1. the loader's label column order equals the head's DEFAULT_DIMS (no silent column swap);
2. per-dim MSE actually descends when encoder+head train on aligned labels.
A fake embedding-bag encoder stands in for bge-m3 (no weights/GPU needed).
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

l2_dataset = pytest.importorskip("datagen.l2_dataset")

from datagen.l3_rubric import CODE_RUBRIC  # noqa: E402
from datagen.score_ledger import ScoreRow  # noqa: E402
from v41f_l2.l2_quality_head import (  # noqa: E402
    DEFAULT_DIMS,
    L2Config,
    L2QualityModel,
    per_dim_mse,
    quality_loss,
)

VOCAB = 256


class BagEncoder(torch.nn.Module):
    """Mean embedding bag: differentiable, no real weights, enough to overfit 40 rows."""

    def __init__(self, dim=1024):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(VOCAB, dim, mode="mean")

    def forward(self, input_ids, attention_mask):
        b, length = input_ids.shape
        flat = input_ids.reshape(-1)
        off = torch.arange(b) * length
        return type("H", (), {"last_hidden_state": self.bag(flat, off).unsqueeze(1)})()


def make_encode():
    def encode(text):
        return [(abs(hash(w)) % (VOCAB - 1)) + 1 for w in text.split()[:64]] or [1]

    return encode


def write_fixture(tmp, n=40):
    dims = tuple(CODE_RUBRIC["dimensions"])
    assert dims == DEFAULT_DIMS
    led, pool = tmp / "ledger.jsonl", tmp / "pool.jsonl"
    with open(led, "w") as fl, open(pool, "w") as fp:
        for i in range(n):
            content = f"doc number {i} " + " ".join(f"tok{i}_{k}" for k in range(20))
            labels = {d: (i % 5) + 1 for d in dims}
            # force spread independent of i so within-domain rank has pairs
            labels[dims[0]] = (i % 5) + 1
            labels[dims[1]] = ((i + 2) % 5) + 1
            labels[dims[2]] = ((40 - i) % 5) + 1
            labels[dims[3]] = ((i * 3) % 5) + 1
            row = ScoreRow(
                doc_id=f"doc{i:03d}", domain="py", lang="en",
                scorer_name="l3-rubric", scorer_version="r1",
                ts="2026-09-16T00:00:00Z", rubric_dims=labels, rubric_kind="code",
                model="t", backend="stub", stratum=None,
            )
            fl.write(json.dumps(row.to_dict()) + "\n")
            fp.write(json.dumps({"doc_id": f"doc{i:03d}", "content": content}) + "\n")
    return led, pool


def test_loader_column_order_matches_head_dims(tmp_path):
    led, pool = write_fixture(tmp_path)
    pairs = l2_dataset.load_pairs(led, pool, scorer_name="l3-rubric")
    assert len(pairs) == 40
    ex = pairs[0]
    assert tuple(ex.labels) and len(ex.labels) == len(DEFAULT_DIMS) == 4
    # the loader orders columns by CODE_RUBRIC dimensions; head assumes exactly this tuple
    assert tuple(CODE_RUBRIC["dimensions"]) == DEFAULT_DIMS


def test_small_fit_reduces_mse(tmp_path):
    led, pool = write_fixture(tmp_path)
    pairs = l2_dataset.load_pairs(led, pool, scorer_name="l3-rubric")
    train, val = l2_dataset.split_pairs(pairs, val_frac=0.1)
    vocab = l2_dataset.build_vocab(e.domain for e in train + val)
    dl = l2_dataset.make_torch_loader(
        train, make_encode(), batch_size=8, max_len=64, shuffle=True, domain_vocab=vocab
    )
    cfg = L2Config(freeze_encoder=False, lambda_rank=0.1)
    model = L2QualityModel(BagEncoder(), cfg, pooling="cls")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def avg_mse():
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for b in dl:
                pred = model(b["input_ids"], b["attention_mask"])
                tot += per_dim_mse(pred, b["labels"], b["label_mask"]).item()
                cnt += 1
        return tot / cnt

    mse0 = avg_mse()
    model.train()
    for _ in range(60):
        for b in dl:
            pred = model(b["input_ids"], b["attention_mask"])
            loss, _ = quality_loss(pred, b["labels"], b["label_mask"], b["domain_id"], cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
    mse1 = avg_mse()
    assert mse1 < 0.5 * mse0, f"no fit: {mse0:.3f} -> {mse1:.3f}"
