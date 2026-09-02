#!/usr/bin/env python3
"""Step-time breakdown + full-causal-vs-1024-window A/B for the 200M hybrid on H20.

Single GPU (CUDA_VISIBLE_DEVICES=0). Production-matched config:
fp8 (e4m3_tensorwise), attn_res=True (Full), compile=True, grad_ckpt=False,
batch=32, seq=4096, doc_mask via cu_seqlens.

All timings via CUDA events or torch.profiler (no wall-clock subtraction).
Outputs JSON on stdout.
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

import train as T

VOCAB = T.Cfg.vocab
B, S = T.Cfg.batch, T.Cfg.seq
DEV = "cuda"
EOS = 1


def make_batch():
    x = torch.randint(10, VOCAB, (B, S), device=DEV)
    for _ in range(8):  # sprinkle eos so cu_seqlens has real doc structure
        x[torch.randint(0, B, (1,)), torch.randint(0, S, (1,))] = EOS
    y = torch.cat([x[:, 1:], torch.full((B, 1), EOS, device=DEV)], dim=1).contiguous()
    return x.contiguous(), y, T.doc_cu_seqlens(x, EOS)


def build(window=False):
    if window:
        orig = T.flash_attn_varlen_func

        def win(*a, **kw):
            kw["window_size"] = (1023, 0)
            return orig(*a, **kw)

        T.flash_attn_varlen_func = win
    raw = T.HybridLM(T.Cfg).to(DEV).to(torch.bfloat16)
    T.convert_to_fp8_compute(raw)
    opts = T.build_optimizers(raw, T.Cfg)
    if T.Cfg.compile:
        torch._dynamo.config.cache_size_limit = 64
        m = torch.compile(raw, dynamic=False)
    else:
        m = raw
    return raw, m, opts


def step(raw, m, opts, batch):
    x, y, cu = batch
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden, _ = m(x, y, cu, None)
    D = hidden.shape[-1]
    w = raw.head.weight[:VOCAB]
    flce = T.LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=T.SOFTCAP)
    loss = flce(w, hidden.to(w.dtype).reshape(-1, D), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(raw.parameters(), T.Cfg.clip)
    for o in opts:
        o.step()
    for o in opts:
        o.zero_grad(set_to_none=True)
    return loss


def measure_steps(raw, m, opts, n, warmup):
    batch = make_batch()
    for _ in range(warmup):
        step(raw, m, opts, batch)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        step(raw, m, opts, batch)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / n
    return {"ms_per_step": ms, "tok_per_s": B * S / (ms / 1000)}


def profile_steps(raw, m, opts, n, path):
    batch = make_batch()
    for _ in range(5):
        step(raw, m, opts, batch)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(n):
            step(raw, m, opts, batch)
            prof.step()
    prof.export_chrome_trace(path)


def categorize(name):
    n = name.lower()
    if "kda" in n:
        return "kda_kernel"
    if "flash" in n or "fmha" in n:
        return "attention"
    if "liger" in n or "flce" in n:
        return "flce"
    if "nccl" in n or "allreduce" in n:
        return "ddp_comm"
    if "memcpy" in n or "memset" in n:
        return "memcpy_memset"
    if any(k in n for k in ("gemm", "cutlass", "ampere", "sgemm", "dgrad", "wgrad", "dot_")):
        return "gemm_linear"
    if "triton" in n:
        return "triton_compiled"
    if "elementwise" in n or "reduce" in n or "norm" in n:
        return "elementwise_norm"
    return "other"


def parse_trace(path, step_ms):
    cats = {}
    total_us = 0.0
    with open(path) as f:
        for ev in json.load(f)["traceEvents"]:
            if ev.get("cat") == "kernel" and "dur" in ev:
                c = categorize(ev["name"])
                cats[c] = cats.get(c, 0.0) + ev["dur"]
                total_us += ev["dur"]
    rows = []
    for c, us in sorted(cats.items(), key=lambda kv: -kv[1]):
        rows.append(
            {
                "category": c,
                "gpu_ms_per_step": us / 1000 / 20,
                "pct_of_step": 100 * us / 1000 / 20 / step_ms,
            }
        )
    accounted = sum(r["pct_of_step"] for r in rows)
    return rows, accounted, 100 - accounted


def measure_data(n=50):
    """Production data path: pinned CPU tensor -> index_select into pinned buffer -> H2D + cu build."""
    X = torch.randint(10, VOCAB, (10000, S + 1)).pin_memory()
    buf = torch.empty((B, S), dtype=torch.long).pin_memory()
    idx = torch.randint(0, 10000, (B,))
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        torch.index_select(X, 0, idx, out=buf)
        xb = buf.to(DEV, non_blocking=True)
        T.doc_cu_seqlens(xb, EOS)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--trace", default="/work/aupai/bench_eff/trace.json")
    args = ap.parse_args()

    out = {"config": {"batch": B, "seq": S, "gpu": 1, "fp8": True, "compile": T.Cfg.compile,
                      "grad_ckpt": False, "attn_res": T.Cfg.attn_res, "model": "HybridLM 12L (3 KDA+1 MLA)x3"}}

    # Arm 1: full causal (current, b3cad87)
    raw, m, opts = build(window=False)
    r1 = measure_steps(raw, m, opts, args.steps, args.warmup)
    out["full_causal"] = r1
    profile_steps(raw, m, opts, 20, args.trace)
    rows, accounted, gap = parse_trace(args.trace, r1["ms_per_step"])
    out["breakdown"] = {"rows": rows, "accounted_pct": accounted, "residual_pct": gap}
    out["data_load_ms"] = measure_data()
    del raw, m, opts
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    # Arm 2: 1024 sliding window (pre-b3cad87)
    raw, m, opts = build(window=True)
    out["window_1024"] = measure_steps(raw, m, opts, args.steps, args.warmup)

    out["delta"] = {
        "ms_per_step": out["window_1024"]["ms_per_step"] - out["full_causal"]["ms_per_step"],
        "tok_per_s_pct": 100 * (out["full_causal"]["tok_per_s"] / out["window_1024"]["tok_per_s"] - 1),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
