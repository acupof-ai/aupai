"""Gated MLA: fp64 reference + G1/G2/G5 harness. No kernel yet — reference and gate only.

Layer shape (target config): x(B,T,d) -> kv_down(d->latent) -> kv_up(latent->2d) -> split k|v
(B,T,H,hd) -> rms_norm q,k -> full-causal attn -> * sigmoid(gate) -> o_proj. A fused kernel
removes the 2*d materialization and the separate gate multiply.

Two attention spellings of the SAME math:
  explicit  materializes [B,H,T,T] scores. The fp64 reference needs it; it is 34 TB at the
            target shape, so it is never the thing benchmarked.
  sdpa      scaled_dot_product_attention(is_causal=True) — what the model actually runs.
            This is the incumbent a fused kernel has to beat.

G1 (agreed criterion): over a FROZEN seed list, fwd / dx / dw each need
    max(candidate) / max(baseline) <= 1.10   AND   mean(candidate/baseline) <= 1.0
against the same fp64 reference. A per-trial "must be <=" tests the sign of rounding noise,
not kernel quality (measured: 7/20 for a mathematically equivalent reimplementation).

Run: python scripts/bench_gated_mla.py            # CPU, tiny shape
     python scripts/bench_gated_mla.py --full     # target config, needs a GPU
"""

import argparse
import json
import platform
import statistics
import time

import torch
import torch.nn.functional as F

CFG = dict(d=1024, latent=256, heads=8, hd=128, B=32, T=4096, dtype="bfloat16")
TINY = dict(CFG, B=2, T=64)
# G1 keeps d/latent/heads/hd at target and shrinks B*T: the fp64 reference materializes
# [B,H,T,T] scores, which is 34 TB at the target shape.
G1_CFG = dict(CFG, B=2, T=256)
SEEDS = tuple(range(1000, 1020))  # frozen: "20 trials" must mean the same 20 every run
MAX_RATIO, MEAN_RATIO = 1.10, 1.0


class Weights:
    """One Gated MLA layer's parameters, held in fp64 as the single source of truth."""

    def __init__(self, cfg, device, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        d, lat, H, hd = cfg["d"], cfg["latent"], cfg["heads"], cfg["hd"]

        def p(*shape):
            t = torch.randn(*shape, generator=g, dtype=torch.float64) * shape[-1] ** -0.5
            return t.to(device).requires_grad_(True)

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


def _project(x, w):
    """Everything before attention. Returns q, k, v as (B,H,T,hd) plus the gate."""
    cfg = w.cfg
    H, hd, d = cfg["heads"], cfg["hd"], cfg["d"]
    B, T, _ = x.shape
    k, v = ((x @ w.kv_down) @ w.kv_up).split(d, dim=-1)  # the materialization
    q, gate = (x @ w.qg).split(d, dim=-1)
    q = rms_norm(q.view(B, T, H, hd), w.q_norm).transpose(1, 2)
    k = rms_norm(k.view(B, T, H, hd), w.k_norm).transpose(1, 2)
    return q, k, v.view(B, T, H, hd).transpose(1, 2), gate


def _finish(o, gate, w):
    B, H, T, hd = o.shape
    return (o.transpose(1, 2).reshape(B, T, H * hd) * torch.sigmoid(gate)) @ w.o


def gated_mla_explicit(x, w):
    """Materializes [B,H,T,T]. Only for the fp64 reference and small-shape G1."""
    q, k, v, gate = _project(x, w)
    T, hd = q.shape[-2], q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) * (hd**-0.5)
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    return _finish(torch.softmax(scores.masked_fill(~causal, float("-inf")), -1) @ v, gate, w)


def gated_mla(x, w):
    """What the model runs: flash/SDPA attention. The incumbent a kernel must beat."""
    q, k, v, gate = _project(x, w)
    return _finish(F.scaled_dot_product_attention(q, k, v, is_causal=True), gate, w)


def reference(x, w):
    """G1 ground truth: the same math in fp64. Makes no speed claim."""
    return gated_mla_explicit(x.to(torch.float64), w)


def err(a, b):
    a, b = a.to(torch.float64), b.to(torch.float64)
    return dict(
        max_abs=(a - b).abs().max().item(),
        rel=((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
    )


def _fwd_bwd(fn, w, x):
    x = x.detach().requires_grad_(True)
    for p in w.all:
        p.grad = None
    out = fn(x, w)
    out.sum().backward()
    return out.detach(), x.grad, [p.grad for p in w.all]


def _score(got, ref):
    return dict(
        fwd=err(got[0], ref[0])["max_abs"],
        dx=err(got[1], ref[1])["max_abs"],
        dw=max(err(a, b)["max_abs"] for a, b in zip(got[2], ref[2], strict=True)),
    )


def g1(cfg, device, candidate=None, baseline=gated_mla, seeds=SEEDS):
    """Candidate's fwd+bwd error vs the incumbent's, both against the same fp64 reference."""
    w64 = Weights(cfg, device)
    dtype = getattr(torch, cfg["dtype"])
    wlo = w64.cast(dtype)
    per_trial = []
    for seed in seeds:
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(cfg["B"], cfg["T"], cfg["d"], generator=g, dtype=torch.float64).to(device)
        ref = _fwd_bwd(reference, w64, x)
        row = {"seed": seed, "baseline": _score(_fwd_bwd(baseline, wlo, x.to(dtype)), ref)}
        if candidate is not None:
            row["candidate"] = _score(_fwd_bwd(candidate, wlo, x.to(dtype)), ref)
        per_trial.append(row)

    verdict = {"seeds": list(seeds), "max_ratio_limit": MAX_RATIO, "mean_ratio_limit": MEAN_RATIO}
    for k in ("fwd", "dx", "dw"):
        b = [t["baseline"][k] for t in per_trial]
        verdict[k] = {"baseline_max": max(b)}
        if candidate is None:
            continue
        c = [t["candidate"][k] for t in per_trial]
        verdict[k] |= {
            "candidate_max": max(c),
            "max_ratio": max(c) / max(b),
            "mean_ratio": statistics.mean(x / y for x, y in zip(c, b, strict=True)),
        }
        verdict[k]["pass"] = verdict[k]["max_ratio"] <= MAX_RATIO and verdict[k]["mean_ratio"] <= MEAN_RATIO
    if candidate is not None:
        verdict["pass"] = all(verdict[k]["pass"] for k in ("fwd", "dx", "dw"))
    return verdict, per_trial


def g2(cfg, device, fn):
    """Determinism: same input twice, bit-exact."""
    dtype = getattr(torch, cfg["dtype"])
    w = Weights(cfg, device).cast(dtype)
    x = torch.randn(cfg["B"], cfg["T"], cfg["d"], device=device, dtype=dtype)
    return torch.equal(fn(x, w), fn(x, w))


def config_stamp(cfg, device):
    """G5: no number without its config."""
    s = dict(cfg, device=str(device), torch=torch.__version__, platform=platform.platform())
    if device.type == "cuda":
        s["gpu"] = torch.cuda.get_device_name(device)
        s["tf32"] = torch.backends.cuda.matmul.allow_tf32
    return s


def bench(cfg, device, fn, iters=10, warmup=3):
    dtype = getattr(torch, cfg["dtype"])
    w = Weights(cfg, device).cast(dtype)
    x = torch.randn(cfg["B"], cfg["T"], cfg["d"], device=device, dtype=dtype)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn(x, w)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, w)
    sync()
    return {"ms_per_call": (time.perf_counter() - t0) * 1e3 / iters, "config": config_stamp(cfg, device)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="target config; needs a GPU")
    ap.add_argument("--json", default=None, help="write the result here")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CFG if args.full else TINY
    gcfg = G1_CFG if args.full else TINY
    if dev.type != "cuda":
        cfg = gcfg = dict(gcfg, dtype="float32")  # no bf16 arithmetic worth measuring on CPU

    verdict, trials = g1(gcfg, dev)
    assert trials[0]["baseline"]["fwd"] < 1e-2, trials[0]
    assert g2(gcfg, dev, gated_mla), "the incumbent is non-deterministic — G2 floor is broken"
    out = {"g1_baseline_vs_fp64": verdict, "g1_config": config_stamp(gcfg, dev), "g2_bit_exact": True}
    # The model runs under torch.compile, so the eager number is not the bar to beat.
    out["bench_sdpa"] = bench(cfg, dev, gated_mla)
    out["bench_sdpa_compiled"] = bench(cfg, dev, torch.compile(gated_mla), warmup=5)
    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out | {"per_trial": trials}, f, indent=2)
