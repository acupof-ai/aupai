"""Gated MLA: fp64 reference + G1/G2/G5 harness. No kernel yet — skeleton only.

Layer shape (aupai target config): x(B,T,d) -> kv_down(d->latent) -> kv_up(latent->2d)
-> split k|v (B,T,H,hd) -> rms_norm q,k -> full-causal attn -> * sigmoid(gate) -> o_proj.
The kv_up 2d materialization + the gate multiply are what a fused kernel removes.

Run: python gated_mla_bench.py            # G1+G2 on a tiny shape, CPU
     python gated_mla_bench.py --full     # target config, needs a GPU
"""

import argparse
import json
import platform
import time

import torch

CFG = dict(d=1024, latent=256, heads=8, hd=128, B=32, T=4096, dtype="bfloat16")
TINY = dict(CFG, B=2, T=64)


class Weights:
    """One Gated MLA layer's parameters, held in fp64 as the single source of truth."""

    def __init__(self, cfg, device, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        d, lat, H, hd = cfg["d"], cfg["latent"], cfg["heads"], cfg["hd"]

        def p(*shape, scale=None):
            t = torch.randn(*shape, generator=g, dtype=torch.float64)
            return (t * (scale or shape[-1] ** -0.5)).to(device).requires_grad_(True)

        self.cfg = cfg
        self.kv_down = p(d, lat)
        self.kv_up = p(lat, 2 * d)  # the 2*d materialization
        self.qg = p(d, 2 * d)  # q | gate
        self.o = p(H * hd, d)
        self.q_norm = torch.ones(hd, dtype=torch.float64, device=device).requires_grad_(True)
        self.k_norm = torch.ones(hd, dtype=torch.float64, device=device).requires_grad_(True)
        self.all = [self.kv_down, self.kv_up, self.qg, self.o, self.q_norm, self.k_norm]

    def cast(self, dtype):
        w = object.__new__(Weights)
        w.cfg = self.cfg
        for n in ("kv_down", "kv_up", "qg", "o", "q_norm", "k_norm"):
            setattr(w, n, getattr(self, n).detach().to(dtype).requires_grad_(True))
        w.all = [w.kv_down, w.kv_up, w.qg, w.o, w.q_norm, w.k_norm]
        return w


def rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def gated_mla(x, w, *, accum=None):
    """The math. `accum` promotes compute dtype without changing the reference math."""
    cfg = w.cfg
    H, hd, d = cfg["heads"], cfg["hd"], cfg["d"]
    B, T, _ = x.shape
    c = x if accum is None else x.to(accum)
    p = (lambda t: t if accum is None else t.to(accum))

    kv = (c @ p(w.kv_down)) @ p(w.kv_up)  # (B,T,2d) — the materialization
    k, v = kv.split(d, dim=-1)
    qg = c @ p(w.qg)
    q, gate = qg.split(d, dim=-1)

    q = rms_norm(q.view(B, T, H, hd), p(w.q_norm)).transpose(1, 2)
    k = rms_norm(k.view(B, T, H, hd), p(w.k_norm)).transpose(1, 2)
    v = v.view(B, T, H, hd).transpose(1, 2)

    scores = (q @ k.transpose(-1, -2)) * (hd**-0.5)
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    o = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, T, H * hd)
    return (o * torch.sigmoid(gate)) @ p(w.o)


def reference(x, w):
    """G1 ground truth: same math, fp64 throughout, no speed claims."""
    return gated_mla(x.to(torch.float64), w)


def err(a, b):
    a, b = a.to(torch.float64), b.to(torch.float64)
    return dict(max_abs=(a - b).abs().max().item(),
                rel=((a - b).norm() / b.norm().clamp_min(1e-30)).item())


def g1(cfg, device, candidate=None, trials=20):
    """New kernel's fwd+bwd error must be <= the incumbent's, against the same fp64 ref."""
    w64 = Weights(cfg, device)
    dtype = getattr(torch, cfg["dtype"])
    wlo = w64.cast(dtype)
    rows = []
    for i in range(trials):
        g = torch.Generator(device="cpu").manual_seed(1000 + i)
        x = torch.randn(cfg["B"], cfg["T"], cfg["d"], generator=g, dtype=torch.float64).to(device)

        def run(fn, weights, xin):
            xin = xin.detach().requires_grad_(True)
            for p in weights.all:
                p.grad = None
            out = fn(xin, weights)
            out.sum().backward()
            return out.detach(), xin.grad, [p.grad for p in weights.all]

        ref_o, ref_dx, ref_dw = run(reference, w64, x)
        base_o, base_dx, base_dw = run(gated_mla, wlo, x.to(dtype))
        row = {"trial": i, "baseline_fwd": err(base_o, ref_o),
               "baseline_dx": err(base_dx, ref_dx),
               "baseline_dw": max(err(a, b)["max_abs"] for a, b in zip(base_dw, ref_dw))}
        if candidate is not None:
            cand_o, cand_dx, cand_dw = run(candidate, wlo, x.to(dtype))
            row |= {"cand_fwd": err(cand_o, ref_o), "cand_dx": err(cand_dx, ref_dx),
                    "cand_dw": max(err(a, b)["max_abs"] for a, b in zip(cand_dw, ref_dw))}
            row["pass"] = (row["cand_fwd"]["max_abs"] <= row["baseline_fwd"]["max_abs"]
                           and row["cand_dx"]["max_abs"] <= row["baseline_dx"]["max_abs"]
                           and row["cand_dw"] <= row["baseline_dw"])
        rows.append(row)
    return rows


def g2(cfg, device, fn):
    """Determinism: same input twice must be bit-exact."""
    w = Weights(cfg, device).cast(getattr(torch, cfg["dtype"]))
    x = torch.randn(cfg["B"], cfg["T"], cfg["d"], device=device, dtype=getattr(torch, cfg["dtype"]))
    a, b = fn(x, w), fn(x, w)
    return torch.equal(a, b)


def config_stamp(cfg, device):
    """G5: no number without its config."""
    s = dict(cfg, device=str(device), torch=torch.__version__, platform=platform.platform())
    if device.type == "cuda":
        s["gpu"] = torch.cuda.get_device_name(device)
        s["tf32"] = torch.backends.cuda.matmul.allow_tf32
    return s


def bench(cfg, device, fn, iters=20, warmup=5):
    w = Weights(cfg, device).cast(getattr(torch, cfg["dtype"]))
    x = torch.randn(cfg["B"], cfg["T"], cfg["d"], device=device, dtype=getattr(torch, cfg["dtype"]))
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn(x, w)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, w)
    sync()
    return {"ms_per_call": (time.perf_counter() - t0) * 1e3 / iters,
            "config": config_stamp(cfg, device)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="target config (needs a GPU)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CFG if a.full else TINY
    if not a.full:
        cfg = dict(cfg, dtype="bfloat16" if dev.type == "cuda" else "float32")

    rows = g1(cfg, dev, trials=3)
    assert all(r["baseline_fwd"]["rel"] < 1e-2 for r in rows), rows
    assert g2(cfg, dev, gated_mla), "baseline is non-deterministic — G2 floor is broken"
    print(json.dumps({"g1_baseline_vs_fp64": rows[0], "g2_bit_exact": True,
                      "bench": bench(cfg, dev, gated_mla, iters=3, warmup=1)}, indent=2))
