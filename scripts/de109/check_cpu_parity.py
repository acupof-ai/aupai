"""de-109 CPU float64 parity: materialized CSA2 vs dense-entries + window split-softmax.

The win-flash branch with use_flash=False is the same split math in eager torch
(entries logsumexp + window materialized softmax, ce/cw combine), so it must match the
single concatenated softmax to float64 precision on forward AND every gradient,
including the STE indexer grad. The GPU swaps only the window half for flash.
"""
from types import SimpleNamespace

import torch

import model as M


def build(win, seed_w):
    cfg = SimpleNamespace(
        csa=True, csa2=True, csa2_m=4, csa2_top_k=3, csa2_n_win=6,
        csa2_indexer_dim=8, csa2_indexer_heads=2, d=16)
    if win:
        cfg.csa2_win_flash = True
    torch.manual_seed(seed_w)
    return M.CompressedSparseAttention(cfg, h=4, hd=8).to(torch.float64)


def main():
    torch.manual_seed(0)
    B, T, H, hd, d = 2, 12, 4, 8, 16
    q0 = torch.randn(B, T, H, hd, dtype=torch.float64)
    k0 = torch.randn(B, T, H, hd, dtype=torch.float64)
    v0 = torch.randn(B, T, H, hd, dtype=torch.float64)
    x0 = torch.randn(B, T, d, dtype=torch.float64)
    cu = torch.tensor([0, 7, 12, 24], dtype=torch.int32)
    gout = torch.randn(B, T, H, hd, dtype=torch.float64)
    seed_w = 123

    def one(win):
        mdl = build(win, seed_w)
        q, k, v, x = (t.clone().requires_grad_() for t in (q0, k0, v0, x0))
        y = mdl(q, k, v, cu=cu, x=x)
        y.backward(gout)
        pg = {n: p.grad.detach().clone() for n, p in mdl.named_parameters()}
        return y.detach(), (q.grad, k.grad, v.grad, x.grad), pg

    ym, igm, gm = one(False)
    yw, igw, gw = one(True)

    fwd = (ym - yw).abs().max()
    print(f"forward abs maxdiff {fwd:.3e}  (|y|max {ym.abs().max():.3e})")
    assert fwd < 1e-8, fwd
    for nm, (a, b) in zip(("q", "k", "v", "x"), zip(igm, igw, strict=True), strict=True):
        dd = (a - b).abs().max()
        print(f"input grad {nm:2s} abs maxdiff {dd:.3e}")
        assert dd < 1e-7, (nm, dd)
    for n in sorted(set(gm) | set(gw)):
        a, b = gm[n], gw[n]
        assert a is not None and b is not None, n
        dd = (a - b).abs().max()
        nz = b.abs().max().item()
        print(f"param {n:14s} abs maxdiff {dd:.3e}  win|grad|max {nz:.3e}")
        assert dd < 1e-7, (n, dd)
        assert nz > 0, n
    assert gw["indexer_q.weight"].abs().max() > 0
    assert gw["ik_weight"].abs().max() > 0
    print("CPU PARITY OK")


if __name__ == "__main__":
    main()
