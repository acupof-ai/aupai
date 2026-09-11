"""de-109 GPU acceptance (single card). Stacks 10 CSA2 layers at the v41 smoke shape
(d=1024 h=8 hd=128 T=4096) and compares materialized (cfg.csa2_win_flash=False) against
dense-entries + flash-window split softmax (True).

Part A: bf16 fixed-batch forward max-abs + every parameter grad max-abs, indexer grad
nonzero on both. Part B: peak GiB and tok/s of fwd+bwd at B=4 and B=8. Target:
tok/s >= materialized at B4 and B8 fitting (joint de-108 halved B8 memory but was
2.3-3.2x slower; de-109 keeps entries dense and flashes only the window).
Run: CUDA_VISIBLE_DEVICES=<lane> python3 scripts/de109/check_gpu_parity.py
"""
import argparse
import time
from types import SimpleNamespace

import torch

import model as M


def cfg(win):
    c = SimpleNamespace(
        csa=True, csa2=True, csa2_m=8, csa2_top_k=64, csa2_n_win=128,
        csa2_indexer_dim=64, csa2_indexer_heads=4, d=1024)
    if win:
        c.csa2_win_flash = True
    return c


class Stack(torch.nn.Module):
    def __init__(self, win, L, h, hd):
        super().__init__()
        torch.manual_seed(123)
        self.layers = torch.nn.ModuleList(
            [M.CompressedSparseAttention(cfg(win), h=h, hd=hd) for _ in range(L)])

    def forward(self, q, k, v, x, cu):
        for layer in self.layers:
            q = layer(q, k, v, cu=cu, x=x)
        return q


def build(win, L=10, h=8, hd=128, dtype=torch.bfloat16, device="cuda"):
    return Stack(win, L, h, hd).to(dtype).to(device)


def inputs(B, T, h, hd, d, dtype, device):
    return (torch.randn(B, T, h, hd, dtype=dtype, device=device) * 0.1 for _ in range(4))


def parity(T, h, hd, d, device):
    print(f"=== PARITY bf16 T={T} B=2 ===", flush=True)
    B = 2
    mm, mw = build(False, device=device), build(True, device=device)
    mw.load_state_dict({k: v.clone() for k, v in mm.state_dict().items()})
    torch.manual_seed(7)
    q, k, v, x = [t for t in inputs(B, T, h, hd, d, torch.bfloat16, device)]
    cu = torch.tensor([0, T // 2, T, 2 * T], dtype=torch.int32, device=device)
    gout = torch.randn(B, T, h, hd, dtype=torch.bfloat16, device=device)

    def run(mdl):
        qq, kk, vv, xx = (t.clone() for t in (q, k, v, x))
        y = mdl(qq, kk, vv, xx, cu)
        y.backward(gout)
        return y.detach().float(), {n: p.grad.detach().float() for n, p in mdl.named_parameters()}

    ym, gm = run(mm)
    yw, gw = run(mw)
    fwd = (ym - yw).abs().max().item()
    print(f"fwd abs max {fwd:.4e}  rel {fwd/ym.abs().max().clamp_min(1e-6).item():.4e}", flush=True)
    worst = 0.0
    for n in sorted(gm):
        dd = (gm[n] - gw[n]).abs().max().item()
        worst = max(worst, dd)
    print(f"worst param-grad abs max {worst:.4e}", flush=True)
    for tag, g in (("indexer_q", gw), ("ik_weight", gw)):
        names = [n for n in g if n.endswith(tag + ".weight")]
        print(f"{tag} grad nonzero mat={any(gm[n].abs().max()>0 for n in names)} "
              f"win={any(gw[n].abs().max()>0 for n in names)}", flush=True)
    return fwd, worst


def measure(win, B, T, h, hd, d, device, iters=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = build(win, device=device)
    q, k, v, x = [t for t in inputs(B, T, h, hd, d, torch.bfloat16, device)]
    cu = (torch.arange(0, 2 * B + 1, dtype=torch.int32, device=device) * (T // 2))

    def step():
        for z in (q, k, v, x):
            z.grad = None
        for p in m.parameters():
            p.grad = None
        m(q, k, v, x, cu).sum().backward()

    try:
        step(); step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        print(f"win={int(win)} B={B}: peak {peak:.2f} GiB  {dt*1000:.1f} ms/fwdbwd  "
              f"{B*T/dt:.0f} tok/s", flush=True)
        return peak, B * T / dt
    except torch.cuda.OutOfMemoryError:
        print(f"win={int(win)} B={B}: OOM", flush=True)
        torch.cuda.empty_cache()
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--parity_T", type=int, default=512)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    h, hd, d = 8, 128, 1024
    fwd, grad = parity(a.parity_T, h, hd, d, "cuda")
    print("=== MEMORY / SPEED (10 layers, bf16, fwd+bwd) ===", flush=True)
    measure(False, 4, a.T, h, hd, d, "cuda", a.iters)
    measure(True, 4, a.T, h, hd, d, "cuda", a.iters)
    measure(True, 8, a.T, h, hd, d, "cuda", a.iters)
    measure(False, 8, a.T, h, hd, d, "cuda", a.iters)
    print(f"SUMMARY parity_fwd={fwd:.3e} parity_grad={grad:.3e}", flush=True)


if __name__ == "__main__":
    main()
