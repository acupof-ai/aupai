"""AttnRes depth-softmax: fp64 reference + the one-pass online-softmax algorithm + G1/G2/G5.

Mirrors train.py's AttnRes exactly (arXiv 2603.15031). Two implementations of the SAME math:
  two_pass   what ships today: stack all logits, softmax, then a second read to accumulate.
             Reads every source twice — 2n * [B,T,D] per sublayer.
  one_pass   flash-style running max/sum on the DEPTH axis. Reads every source once.

`one_pass` is a pure-torch stand-in for the kernel, written in the kernel's reduction order, so
G1 prices the reordering's numerical cost before any kernel exists.

G1 (agreed criterion): over a FROZEN seed list, fwd / dv / dq each need
    max(candidate) / max(baseline) <= 1.10   AND   mean(candidate/baseline) <= 1.0
against the same fp64 reference. A per-trial "must be <=" was tried and rejected: it passed
7/20 for a mathematically equivalent reimplementation, i.e. it tests the sign of rounding noise.

Run: python scripts/attn_res_bench.py            # CPU, tiny shape
     python scripts/attn_res_bench.py --full     # B=32 T=4096 D=1024 n=25 bf16, needs a GPU
"""

import argparse
import json
import platform
import statistics
import time

import torch

CFG = dict(B=32, T=4096, D=1024, n=25, dtype="bfloat16")
TINY = dict(CFG, B=2, T=16, D=64, n=5)
# G1 holds D and n at target but shrinks B*T: the fp64 reference materializes 3 copies of
# n*[B,T,D], which is ~80 GB at CFG and OOMs before it measures anything.
G1_CFG = dict(CFG, B=2, T=512)
SEEDS = tuple(range(1000, 1020))  # frozen: "20 trials" must mean the same 20 every run
MAX_RATIO, MEAN_RATIO = 1.10, 1.0


def rms_scale(v, eps=1e-6):
    return torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)


def two_pass(vs, gq):
    """Today's implementation: all logits first, then a second read to accumulate."""
    logits = torch.stack([(v * gq).sum(-1) * rms_scale(v).squeeze(-1) for v in vs])
    a = logits.float().softmax(0).to(vs[0].dtype)
    out = a[0].unsqueeze(-1) * vs[0]
    for i in range(1, len(vs)):
        out = out + a[i].unsqueeze(-1) * vs[i]
    return out


def one_pass(vs, gq):
    """Online softmax down the depth axis — the kernel's reduction order, in torch.

    One read per source gives both the logit (a full-D reduction, so the D row must stay
    resident: 1024 bf16 = 2 KB/row, SRAM-sized) and the weighted accumulate. Running stats and
    accumulator are fp32, matching the incumbent's fp32 softmax.
    """
    m = torch.full(vs[0].shape[:-1], float("-inf"), device=vs[0].device, dtype=torch.float32)
    s = torch.zeros_like(m)
    acc = torch.zeros(vs[0].shape, device=vs[0].device, dtype=torch.float32)
    for v in vs:
        logit = ((v * gq).sum(-1) * rms_scale(v).squeeze(-1)).float()
        m_new = torch.maximum(m, logit)
        # exp(-inf - -inf) is nan on the first source, so the first rescale is forced to 0.
        rescale = torch.where(torch.isinf(m), torch.zeros_like(m), torch.exp(m - m_new))
        p = torch.exp(logit - m_new)
        s = s * rescale + p
        acc = acc * rescale.unsqueeze(-1) + p.unsqueeze(-1) * v.float()
        m = m_new
    return (acc / s.unsqueeze(-1)).to(vs[0].dtype)


def reference(vs, gq):
    """G1 ground truth: the same math in fp64. Makes no speed claim."""
    vs = [v.to(torch.float64) for v in vs]
    gq = gq.to(torch.float64)
    logits = torch.stack([(v * gq).sum(-1) * rms_scale(v).squeeze(-1) for v in vs])
    return (logits.softmax(0).unsqueeze(-1) * torch.stack(vs)).sum(0)


def err(a, b):
    a, b = a.to(torch.float64), b.to(torch.float64)
    return dict(max_abs=(a - b).abs().max().item(), rel=((a - b).norm() / b.norm().clamp_min(1e-30)).item())


def make(cfg, device, seed, need_fp64=True):
    """`need_fp64=False` for timing and memory arms: the fp64 reference tensors are 25 x 1.07 GB
    of HOST ram at the target shape, which is what made every full-config run crawl or die."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dt = getattr(torch, cfg["dtype"])
    if not need_fp64:
        vs = [torch.randn(cfg["B"], cfg["T"], cfg["D"], dtype=dt, device=device) for _ in range(cfg["n"])]
        gq = (torch.randn(cfg["D"], generator=g, dtype=torch.float64) * cfg["D"] ** -0.5).to(device, dt)
        return None, None, vs, gq
    vs64 = [
        torch.randn(cfg["B"], cfg["T"], cfg["D"], generator=g, dtype=torch.float64).to(device)
        for _ in range(cfg["n"])
    ]
    # 1/sqrt(D) so the logits land at O(1). Unit-variance gq gives logits with std sqrt(D) = 32,
    # which saturates the softmax to a hard argmax and prices bf16 against a regime the trained
    # model never occupies (q is zero-init, so real logits start at 0).
    gq64 = torch.randn(cfg["D"], generator=g, dtype=torch.float64).to(device) * cfg["D"] ** -0.5
    return vs64, gq64, [v.to(dt) for v in vs64], gq64.to(dt)


def _fwd_bwd(fn, V, Q):
    V = [v.detach().requires_grad_(True) for v in V]
    Q = Q.detach().requires_grad_(True)
    out = fn(V, Q)
    out.sum().backward()
    return out.detach(), [v.grad for v in V], Q.grad


def _score(x, ref):
    return dict(
        fwd=err(x[0], ref[0])["max_abs"],
        fwd_rel=err(x[0], ref[0])["rel"],
        dv=max(err(a, b)["max_abs"] for a, b in zip(x[1], ref[1], strict=True)),
        dq=err(x[2], ref[2])["max_abs"],
    )


def g1(cfg, device, candidate=one_pass, baseline=two_pass, seeds=SEEDS):
    """Candidate's fwd+bwd error vs baseline's, both against the same fp64 reference."""
    per_trial = []
    for seed in seeds:
        vs64, gq64, vs, gq = make(cfg, device, seed)
        ref = _fwd_bwd(reference, vs64, gq64)

        per_trial.append(
            {
                "seed": seed,
                "baseline": _score(_fwd_bwd(baseline, vs, gq), ref),
                "candidate": _score(_fwd_bwd(candidate, vs, gq), ref),
            }
        )

    verdict = {"seeds": list(seeds), "max_ratio_limit": MAX_RATIO, "mean_ratio_limit": MEAN_RATIO}
    for k in ("fwd", "dv", "dq"):
        c = [t["candidate"][k] for t in per_trial]
        b = [t["baseline"][k] for t in per_trial]
        verdict[k] = {
            "candidate_max": max(c),
            "baseline_max": max(b),
            "max_ratio": max(c) / max(b),
            "mean_ratio": statistics.mean(x / y for x, y in zip(c, b, strict=True)),
        }
        verdict[k]["pass"] = verdict[k]["max_ratio"] <= MAX_RATIO and verdict[k]["mean_ratio"] <= MEAN_RATIO
    verdict["pass"] = all(verdict[k]["pass"] for k in ("fwd", "dv", "dq"))
    return verdict, per_trial


def g2(cfg, device, fn):
    """Determinism: same input twice, bit-exact."""
    _, _, vs, gq = make(cfg, device, SEEDS[0], need_fp64=False)
    return torch.equal(fn(vs, gq), fn(vs, gq))


def config_stamp(cfg, device):
    s = dict(cfg, device=str(device), torch=torch.__version__, platform=platform.platform())
    if device.type == "cuda":
        s["gpu"] = torch.cuda.get_device_name(device)
        s["tf32"] = torch.backends.cuda.matmul.allow_tf32
    return s


def peak_mem_fwd_bwd(cfg, device, fn):
    """Peak allocated bytes for one fwd+bwd. The memory claim has to be measured, not derived:
    eager autograd does not save the accumulate loop's temporaries (add's backward saves
    nothing), while one_pass's fp32 running accumulator is saved by every rescale multiply."""
    if device.type != "cuda":
        return None
    _, _, vs, gq = make(cfg, device, SEEDS[0], need_fp64=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _fwd_bwd(fn, vs, gq)
    return {"peak_MB": (torch.cuda.max_memory_allocated() - base) / 1e6}


def bench(cfg, device, fn, iters=10, warmup=3):
    _, _, vs, gq = make(cfg, device, SEEDS[0], need_fp64=False)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn(vs, gq)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(vs, gq)
    sync()
    ms = (time.perf_counter() - t0) * 1e3 / iters
    src_bytes = cfg["B"] * cfg["T"] * cfg["D"] * torch.finfo(getattr(torch, cfg["dtype"])).bits // 8
    return {
        "ms_per_call": ms,
        "source_MB": src_bytes / 1e6,
        "GBps_if_read_once": cfg["n"] * src_bytes / (ms * 1e6),
        "config": config_stamp(cfg, device),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="target config; needs a GPU")
    ap.add_argument("--json", default=None, help="write the result here")
    ap.add_argument("--bench-only", action="store_true", help="skip G1/G2 (the fp64 reference is slow)")
    ap.add_argument("--eager-only", action="store_true", help="skip the compiled arms (slow to build)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CFG if args.full else TINY
    gcfg = G1_CFG if args.full else TINY
    if dev.type != "cuda":
        cfg = gcfg = dict(cfg, dtype="float32")  # no bf16 arithmetic worth measuring on CPU

    out = {}
    trials = None
    if not args.bench_only:
        verdict, trials = g1(gcfg, dev)
        # Relative, not absolute: at D=1024 in bf16 a max|d| of ~4e-2 is just bf16's mantissa.
        assert trials[0]["baseline"]["fwd_rel"] < 1e-2, trials[0]
        assert g2(gcfg, dev, one_pass), "one_pass is non-deterministic — G2 floor is broken"
        out = {"g1": verdict, "g1_config": config_stamp(gcfg, dev), "g2_bit_exact": True}
    for name, fn in (("two_pass", two_pass), ("one_pass", one_pass)):
        out[f"bench_{name}"] = bench(cfg, dev, fn)
        # The shipped model runs under torch.compile, so the eager pair is not the
        # baseline a kernel has to beat — compile fuses the two-pass reads too.
        if not args.eager_only:
            out[f"bench_{name}_compiled"] = bench(cfg, dev, torch.compile(fn), warmup=5)
        out[f"mem_{name}"] = peak_mem_fwd_bwd(cfg, dev, fn)
    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out if trials is None else out | {"per_trial": trials}, f, indent=2)
