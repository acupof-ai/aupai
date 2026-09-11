"""de-108 GPU acceptance (card 7, single process). Stacks 10 CSA2 attention layers at
the v41 smoke shape (d=1024, h=8, hd=128, T=4096) and compares the materialized joint
softmax (cfg.csa2_joint=False) against the flash LSE-combine path (True).

Part A -- parity on a fixed batch (T smaller, bf16): forward max-abs and every parameter
gradient max-abs (and indexer_q/ik_weight nonzero on both).
Part B -- peak memory and tok/s of forward+backward at B=4 and B=8, T=4096. The
materialized path OOMs at B8 by construction (94.62 GiB was the stack OOM); that is the
comparison de-108 exists to win. Run:
    CUDA_VISIBLE_DEVICES=7 python3 scripts/de108/check_gpu_parity.py
"""
import argparse
import time
from types import SimpleNamespace

import torch

import model as M


def cfg(joint):
    c = SimpleNamespace(
        csa=True, csa2=True, csa2_m=8, csa2_top_k=64, csa2_n_win=128,
        csa2_indexer_dim=64, csa2_indexer_heads=4, d=1024)
    if joint:
        c.csa2_joint = True
    return c


class Stack(torch.nn.Module):
    def __init__(self, joint, L, h, hd):
        super().__init__()
        torch.manual_seed(123)
        self.layers = torch.nn.ModuleList(
            [M.CompressedSparseAttention(cfg(joint), h=h, hd=hd) for _ in range(L)])

    def forward(self, q, k, v, x, cu):
        for layer in self.layers:
            y = layer(q, k, v, cu=cu, x=x)
            q = y
        return q


def build(joint, L=10, h=8, hd=128, dtype=torch.bfloat16, device="cuda"):
    m = Stack(joint, L, h, hd).to(dtype).to(device)
    return m


def inputs(B, T, h, hd, d, dtype, device):
    q = torch.randn(B, T, h, hd, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, h, hd, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, h, hd, dtype=dtype, device=device) * 0.1
    x = torch.randn(B, T, d, dtype=dtype, device=device) * 0.1
    return q, k, v, x


def parity(T, h, hd, d, device):
    print(f"=== PARITY bf16 T={T} B=2 ===", flush=True)
    B = 2
    mm = build(False, device=device)
    mj = build(True, device=device)
    # identical params
    sd = {k: v.clone() for k, v in mm.state_dict().items()}
    mj.load_state_dict(sd)
    torch.manual_seed(7)
    q, k, v, x = inputs(B, T, h, hd, d, torch.bfloat16, device)
    cu = torch.tensor([0, T // 2, T, 2 * T], dtype=torch.int32, device=device)
    gout = torch.randn(B, T, h, hd, dtype=torch.bfloat16, device=device)

    def run(mdl, q, k, v, x):
        q, k, v, x = (t.clone() for t in (q, k, v, x))
        y = mdl(q, k, v, x, cu)
        y.backward(gout)
        grads = {n: p.grad.detach().float() for n, p in mdl.named_parameters()}
        return y.detach().float(), grads

    ym, gm = run(mm, q, k, v, x)
    yj, gj = run(mj, q, k, v, x)
    fwd = (ym - yj).abs().max().item()
    scale = ym.abs().max().clamp_min(1e-6).item()
    print(f"fwd abs max {fwd:.4e}  rel {fwd/scale:.4e}  |y|max {scale:.3f}", flush=True)
    worst = 0.0
    for n in sorted(gm):
        if gm[n] is None or gj[n] is None:
            print(f"  {n}: grad missing mat={gm[n] is not None} jnt={gj[n] is not None}")
            continue
        dd = (gm[n] - gj[n]).abs().max().item()
        worst = max(worst, dd)
    print(f"worst param-grad abs max {worst:.4e}", flush=True)
    iq = [n for n in gj if n.endswith("indexer_q.weight")]
    ik = [n for n in gj if n.endswith("ik_weight")]
    print(f"indexer_q grad nonzero mat={any(gm[n].abs().max()>0 for n in iq)} "
          f"jnt={any(gj[n].abs().max()>0 for n in iq)}", flush=True)
    print(f"ik_weight grad nonzero mat={any(gm[n].abs().max()>0 for n in ik)} "
          f"jnt={any(gj[n].abs().max()>0 for n in ik)}", flush=True)
    return fwd, worst


def measure(joint, B, T, h, hd, d, device, iters=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = build(joint, device=device)
    q, k, v, x = inputs(B, T, h, hd, d, torch.bfloat16, device)
    # two equal docs per row over the B*T stream: 0,T/2,T, 3T/2,... -> B*T
    cu = (torch.arange(0, 2 * B + 1, dtype=torch.int32, device=device) * (T // 2))

    def step():
        for z in (q, k, v, x):
            z.grad = None
        for p in m.parameters():
            p.grad = None
        y = m(q, k, v, x, cu)
        y.sum().backward()

    try:
        step()
        step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        toks = B * T / dt
        print(f"joint={int(joint)} B={B}: peak {peak:.2f} GiB  {dt*1000:.1f} ms/fwdbwd  "
              f"{toks:.0f} tok/s", flush=True)
        return peak, toks
    except torch.cuda.OutOfMemoryError:
        print(f"joint={int(joint)} B={B}: OOM", flush=True)
        torch.cuda.empty_cache()
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--parity_T", type=int, default=512)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    device = "cuda"
    h, hd, d = 8, 128, 1024
    fwd, grad = parity(args.parity_T, h, hd, d, device)
    print("=== MEMORY / SPEED (10 layers, bf16, fwd+bwd) ===", flush=True)
    measure(False, 4, args.T, h, hd, d, device, args.iters)
    measure(True, 4, args.T, h, hd, d, device, args.iters)
    # B8: joint first -- its numbers are the deliverable; the materialized B8 OOM is the
    # already-recorded 94.62 GiB stack OOM, run last so it cannot take the joint result.
    measure(True, 8, args.T, h, hd, d, device, args.iters)
    measure(False, 8, args.T, h, hd, d, device, args.iters)
    print(f"SUMMARY parity_fwd={fwd:.3e} parity_grad={grad:.3e}", flush=True)


if __name__ == "__main__":
    main()
