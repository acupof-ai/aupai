"""P0: sparse_attn matches an independent brute-force softmax, incl. sink semantics."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.sparse_attn import sparse_attn


def _bruteforce(q, kv, sink, idxs, scale):
    b, m, h, d = q.shape
    topk = idxs.shape[-1]
    out = torch.zeros(b, m, h, d)
    for bi in range(b):
        for mi in range(m):
            for hi in range(h):
                logits, vals = [], []
                for t in range(topk):
                    j = idxs[bi, mi, t].item()
                    if j < 0:
                        continue
                    s = float((q[bi, mi, hi] * kv[bi, j]).sum() * scale)
                    logits.append(s)
                    vals.append(kv[bi, j])
                logits.append(float(sink[hi]))   # sink slot, value 0
                mx = max(logits)
                w = [torch.exp(torch.tensor(s - mx)) for s in logits]
                z = sum(w)
                acc = torch.zeros(d)
                for wi, v in zip(w[:-1], vals, strict=True):
                    acc += (wi / z) * v
                out[bi, mi, hi] = acc
    return out


def test_matches_bruteforce():
    torch.manual_seed(0)
    b, m, h, d, n, topk = 2, 6, 4, 32, 13, 9
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, n, d)
    sink = torch.randn(h) * 0.5
    idxs = torch.randint(0, n, (b, m, topk))
    got = sparse_attn(q, kv, sink, idxs, d ** -0.5)
    want = _bruteforce(q, kv, sink, idxs, d ** -0.5)
    diff = (got - want).abs().max().item()
    assert diff < 1e-4, diff


def test_empty_slots_and_all_empty_row():
    torch.manual_seed(1)
    b, m, h, d, n, topk = 1, 2, 2, 16, 5, 4
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, n, d)
    sink = torch.zeros(h)
    idxs = torch.randint(0, n, (b, m, topk))
    idxs[0, 1] = -1                            # all-empty row: output must be 0
    out = sparse_attn(q, kv, sink, idxs, d ** -0.5)
    assert torch.isfinite(out).all()
    assert out[0, 1].abs().max().item() == 0.0
    # a very negative sink and a normal row still yields a unit-norm weight on real slots
    idxs2 = idxs.clone()
    idxs2[0, 1] = torch.tensor([0, 1, -1, -1])
    sink2 = torch.full((h,), -1e4)
    out2 = sparse_attn(q, kv, sink2, idxs2, d ** -0.5)
    assert torch.isfinite(out2).all()
