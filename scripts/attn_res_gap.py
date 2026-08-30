#!/usr/bin/env python3
"""Split the AttnRes end-to-end cost into forward / backward / optimizer, one GPU, one config
per process (train.Cfg is global and torch.compile caches per graph).

The ablation ladder, each config differing from the previous in ONE thing:
    off       plain residual                       -- the --no_attn_res baseline
    n2        AttnRes with attn_res_blocks=1        -- all 25 modules and all 25 rms_scale calls
              still run, but every call sees <=2 sources, so the 325 source-reads collapse to 49
    noscale   AttnRes Full, rms_scale -> ones       -- Full source-reads, no [B,T,D] reduction
    full      AttnRes Full                          -- what ships

    full - noscale = the rms_scale reductions
    noscale - n2   = the source-read math
    n2 - off       = everything AttnRes costs that is not source-reads or rms_scale
                     (extra kernels, worse fusion elsewhere, memory pressure)

Run (one per pod call, each under 4 minutes):
    CUDA_VISIBLE_DEVICES=7 python -u scripts/attn_res_gap.py --config full --batch 16
"""

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402


def patch_no_scale():
    """Source.of without the [B,T,D] rsqrt-of-mean-square reduction."""
    one = {}

    def of(v):
        key = (v.shape[0], v.shape[1], v.dtype, v.device)
        if key not in one:
            one[key] = torch.ones(v.shape[0], v.shape[1], 1, dtype=v.dtype, device=v.device)
        return train.Source(v, one[key])

    train.Source.of = staticmethod(of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["off", "n2", "noscale", "full"], required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--no_fp8", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--opt", action="store_true", help="also time the optimizer step")
    ap.add_argument("--limit", type=int, default=None, help="torch._dynamo recompile limit")
    ap.add_argument("--vocab", type=int, default=None, help="override Cfg.vocab (LM-head GEMM alignment)")
    a = ap.parse_args()

    if a.limit:
        for k in ("recompile_limit", "cache_size_limit"):
            if hasattr(torch._dynamo.config, k):
                setattr(torch._dynamo.config, k, a.limit)

    cfg = train.Cfg
    cfg.batch = a.batch
    if a.vocab:
        cfg.vocab = a.vocab
    if a.seq:
        cfg.seq = a.seq
    cfg.attn_res = a.config != "off"
    cfg.attn_res_blocks = 1 if a.config == "n2" else 0
    if a.no_compile:
        cfg.compile = False
    if a.config == "noscale":
        patch_no_scale()

    dev = "cuda"
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    model = train.HybridLM(cfg).to(dev).to(torch.bfloat16)
    if not a.no_fp8:
        train.convert_to_fp8_compute(model)
    if cfg.compile:
        model = torch.compile(model, dynamic=False)
    opts = train.build_optimizers(model, cfg) if a.opt else []
    flce = train.LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=train.SOFTCAP)
    raw = getattr(model, "_orig_mod", model)

    x = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq), device=dev)
    y = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq), device=dev)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    acc = [0.0, 0.0, 0.0]

    def step(record):
        ev[0].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, _ = model(x, y)
        w = raw.head.weight[: cfg.vocab]
        loss = flce(w, hidden.to(w.dtype).reshape(-1, hidden.shape[-1]), y.reshape(-1))
        ev[1].record()
        loss.backward()
        ev[2].record()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        ev[3].record()
        torch.cuda.synchronize()
        if record:
            for i in range(3):
                acc[i] += ev[i].elapsed_time(ev[i + 1])

    t0 = time.time()
    for _ in range(a.warmup):
        step(False)
    print(f"[{a.config}] warmup+compile {time.time() - t0:.0f}s", flush=True)
    for i in range(a.steps):
        step(True)
        n = i + 1
        print(
            f"[{a.config}] b{cfg.batch} n={n} fwd {acc[0] / n:8.2f}  bwd {acc[1] / n:8.2f}"
            f"  opt {acc[2] / n:7.2f}  fwd+bwd {(acc[0] + acc[1]) / n:8.2f} ms"
            f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
            flush=True,
        )
    fb = (acc[0] + acc[1]) / a.steps
    print(
        f"RESULT config={a.config} batch={cfg.batch} seq={cfg.seq} vocab={cfg.vocab} fwd={acc[0] / a.steps:.2f} "
        f"bwd={acc[1] / a.steps:.2f} fwd_bwd={fb:.2f} "
        f"tok_s={cfg.batch * cfg.seq / (fb / 1000) / 1e3:.1f}K",
        flush=True,
    )


if __name__ == "__main__":
    main()
