#!/usr/bin/env python3
"""Phase timing + full-vs-window A/B for the 200M hybrid on H20 (single GPU).

Production-matched: fp8 (e4m3_tensorwise), attn_res=True (Full), compile=True,
grad_ckpt=False, batch=32, seq=4096, doc_mask via cu_seqlens.

Timings via CUDA events. JSON written to --out (flushed) so a kill leaves partial results.
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
    for _ in range(8):
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


def fwd_bwd(raw, m, batch):
    x, y, cu = batch
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden, _ = m(x, y, cu, None)
    D = hidden.shape[-1]
    w = raw.head.weight[:VOCAB]
    flce = T.LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=T.SOFTCAP)
    loss = flce(w, hidden.to(w.dtype).reshape(-1, D), y.reshape(-1))
    loss.backward()
    return loss


def opt_step(raw, opts):
    torch.nn.utils.clip_grad_norm_(raw.parameters(), T.Cfg.clip)
    for o in opts:
        o.step()
    for o in opts:
        o.zero_grad(set_to_none=True)


def ev_time(fn, n, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def measure_data(n=50):
    X = torch.randint(10, VOCAB, (10000, S + 1)).pin_memory()
    buf = torch.empty((B, S), dtype=torch.long).pin_memory()
    idx = torch.randint(0, 10000, (B,))
    return ev_time(lambda: (torch.index_select(X, 0, idx, out=buf),
                            buf.to(DEV, non_blocking=True),
                            T.doc_cu_seqlens(buf.to(DEV), EOS)), n, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["full", "window"], required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--out", default="/work/aupai/bench_eff/result.json")
    args = ap.parse_args()

    out = {"arm": args.arm,
           "config": {"batch": B, "seq": S, "gpu": 1, "fp8": True, "compile": T.Cfg.compile,
                      "grad_ckpt": False, "attn_res": T.Cfg.attn_res}}
    with open(args.out, "w") as f:
        json.dump(out, f); f.flush()

    raw, m, opts = build(window=(args.arm == "window"))
    batch = make_batch()

    out["fwd_bwd_ms"] = ev_time(lambda: fwd_bwd(raw, m, batch), args.steps, args.warmup)
    with open(args.out, "w") as f:
        json.dump(out, f); f.flush()

    # optimizer phase: direct CUDA-event measurement around opt_step (grads from a fresh fwd_bwd)
    for _ in range(5):
        fwd_bwd(raw, m, batch)
        opt_step(raw, opts)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.steps):
        fwd_bwd(raw, m, batch)
        opt_step(raw, opts)
    e.record()
    torch.cuda.synchronize()
    out["full_ms"] = s.elapsed_time(e) / args.steps
    # opt phase alone: per-iter events around opt_step only
    for _ in range(3):
        fwd_bwd(raw, m, batch)
        opt_step(raw, opts)
    torch.cuda.synchronize()
    opts_ms = []
    for _ in range(args.steps):
        fwd_bwd(raw, m, batch)
        s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s2.record()
        opt_step(raw, opts)
        e2.record()
        torch.cuda.synchronize()
        opts_ms.append(s2.elapsed_time(e2))
    out["opt_ms"] = sum(opts_ms) / len(opts_ms)

    out["tok_per_s"] = B * S / (out["full_ms"] / 1000)
    out["data_load_ms"] = measure_data()
    with open(args.out, "w") as f:
        json.dump(out, f); f.flush()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
