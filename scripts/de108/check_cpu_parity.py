"""CPU parity for de-108: materialized CSA2 joint softmax vs the pure-torch LSE-combine
reference (csa2_joint_ref), identical weights and identical fixed upstream grad. Checks
forward max-abs and gradients to every parameter that must learn: compress_k/v,
indexer_q, ik_weight, plus the q/k/v inputs. Float64. The flash path runs the same combine
and gets its A/B on the GPU (card 7).
"""
from types import SimpleNamespace

import torch

import model as M


def build(joint, seed_w):
    cfg = SimpleNamespace(
        csa=True, csa2=True, csa2_m=4, csa2_top_k=3, csa2_n_win=6,
        csa2_indexer_dim=8, csa2_indexer_heads=2, d=16)
    if joint:
        cfg.csa2_joint = True
    torch.manual_seed(seed_w)
    return M.CompressedSparseAttention(cfg, h=4, hd=8).to(torch.float64)


def main():
    torch.manual_seed(0)
    B, T, H, hd, d = 2, 12, 4, 8, 16
    q0 = torch.randn(B, T, H, hd, dtype=torch.float64)
    k0 = torch.randn(B, T, H, hd, dtype=torch.float64)
    v0 = torch.randn(B, T, H, hd, dtype=torch.float64)
    x0 = torch.randn(B, T, d, dtype=torch.float64)
    # flat B*T stream: row0 split into two docs at 7, row1 one document of 12
    cu = torch.tensor([0, 7, 12, 24], dtype=torch.int32)
    gout = torch.randn(B, T, H, hd, dtype=torch.float64)
    seed_w = 123

    def one(joint):
        mdl = build(joint, seed_w)
        q, k, v, x = (t.clone().requires_grad_() for t in (q0, k0, v0, x0))
        y = mdl(q, k, v, cu=cu, x=x)
        y.backward(gout)
        pg = {n: p.grad.detach().clone() for n, p in mdl.named_parameters()}
        return y.detach(), (q.grad, k.grad, v.grad, x.grad), pg

    ym, igm, gm = one(False)
    yj, igj, gj = one(True)

    fwd = (ym - yj).abs().max()
    print(f"forward abs maxdiff {fwd:.3e}  (|y|max {ym.abs().max():.3e})")
    assert fwd < 1e-8, fwd
    for nm, (a, b) in zip(("q", "k", "v", "x"), zip(igm, igj, strict=True), strict=True):
        dd = (a - b).abs().max()
        print(f"input grad {nm:2s} abs maxdiff {dd:.3e}")
        assert dd < 1e-7, (nm, dd)
    keys = sorted(set(gm) | set(gj))
    for n in keys:
        if gm[n] is None or gj[n] is None:
            print(f"param {n:14s} grad missing mat={gm[n] is not None} jnt={gj[n] is not None}")
            assert gm[n] is not None and gj[n] is not None, n
        dd = (gm[n] - gj[n]).abs().max()
        nz = gj[n].abs().max().item()
        print(f"param {n:14s} abs maxdiff {dd:.3e}  joint|grad|max {nz:.3e}")
        assert dd < 1e-7, (n, dd)
        assert nz > 0, n
    # explicit: the indexer MUST receive a gradient on the joint path
    assert gj["indexer_q.weight"].abs().max() > 0
    assert gj["ik_weight"].abs().max() > 0
    print("CPU PARITY OK")


if __name__ == "__main__":
    main()
