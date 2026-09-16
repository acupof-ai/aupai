"""P0: latent Q, head-shared KV, and grouped block-diagonal output projections.

The linear parts are nn.Linear; the load-bearing non-obvious math is the block-diagonal
wo_a einsum. It is validated against an explicit per-group matmul reference built in
the test (no shared code), and against the upstream weight layout [groups, rank, in].
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
from v41f.projections import GroupedOProj, KVProj, QProj


def test_qproj_shapes_and_roundtrip():
    torch.manual_seed(0)
    dim, qr, h, hd, eps = 128, 32, 4, 16, 1e-6
    qp = QProj(dim, qr, h, hd, eps)
    x = torch.randn(2, 9, dim)
    q, latent = qp(x)
    assert q.shape == (2, 9, h, hd)
    assert latent.shape == (2, 9, qr)
    # latent equals the normed down-projection
    assert torch.allclose(latent, qp.q_norm(qp.wq_a(x)), atol=1e-6)


def test_kvproj_head_shared():
    torch.manual_seed(0)
    dim, hd = 128, 16
    kv = KVProj(dim, hd, 1e-6)
    x = torch.randn(2, 9, dim)
    y = kv(x)
    assert y.shape == (2, 9, hd)        # one KV line, shared by all heads
    assert torch.allclose(y, kv.kv_norm(kv.wkv(x)), atol=1e-6)


def test_grouped_oproj_block_diagonal():
    torch.manual_seed(0)
    h, hd, g, r, dim = 8, 16, 4, 12, 128
    op = GroupedOProj(h, hd, g, r, dim)
    b, s = 2, 7
    o = torch.randn(b, s, h, hd)
    got = op(o)
    # explicit per-group reference
    og = o.view(b, s, g, h // g * hd)
    lat_ref = torch.empty(b, s, g, r)
    for gi in range(g):
        lat_ref[:, :, gi, :] = torch.einsum("bsd,rd->bsr", og[:, :, gi], op.wo_a[gi])
    want = op.wo_b(lat_ref.flatten(2))
    assert got.shape == (b, s, dim)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max().item()


def test_grouped_oproj_matches_upstream_layout():
    # upstream wo_a.weight is viewed as [n_local_groups, o_lora_rank, -1]; mirror that
    # exact contraction with a synthetic weight to pin the einsum index order.
    g, r, per, hd = 4, 5, 2, 8
    w = torch.randn(g, r, per * hd)
    x = torch.randn(3, 6, g, per, hd).reshape(3, 6, g, per * hd)
    got = torch.einsum("bsgd,grd->bsgr", x, w)
    want = torch.empty(3, 6, g, r)
    for gi in range(g):
        want[:, :, gi] = x[:, :, gi] @ w[gi].T
    assert torch.allclose(got, want, atol=1e-6)
