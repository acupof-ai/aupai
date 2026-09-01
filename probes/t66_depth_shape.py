"""Three ~500M depth candidates, measured on a real card. fb/user order 2026-09-01.

Depth is the variable: A is wide-shallow, C is narrow-deep, B is between, all at ~500M.
Measured rather than derived -- three derived figures missed measured ones badly today, and
the one this replaces ("deep costs more because KDA is recurrent") is an expectation, not a
number.

Real HybridLM from train.py, synthetic token batches: no corpus, no tokenizer, no checkpoint.
The question is throughput and memory at a shape, and neither depends on which tokens flow.

Every candidate runs AttnRes ON (decided, not retested) and head_dim 128 (FlashKDA), so
heads = d/128 and d is a multiple of 128.

    python3 t66.py <A|B|C> [--seq N] [--batch N] [--steps N] [--find-batch]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("AUPAI_ROOT", "/work/aupai"))
import train as T  # noqa: E402

CANDIDATES = {  # d, layers; heads := d//128, ffn fixed 3072 for the ruled shape
    "A": (1280, 24),
    "B": (1152, 30),
    "C": (1024, 36),
    "S": (1024, 32),   # the ruled shape: d1024 L32 heads8 ffn3072
    # The depth control: identical to S but L=12. The 200M run's 32% MFU is NOT a baseline --
    # different model, different world/batch/ckpt. This is the only comparison on the board
    # where depth is the sole variable (fb, 2026-09-01).
    "S12": (1024, 12),
}
WARMUP = 50  # fb: steady-state window past step 50
_AR_BLOCKS = [0]  # set by --ar-blocks; 0 is Full, the pinned default


def configure(name, seq, batch):
    d, layers = CANDIDATES[name]
    T.Cfg.d, T.Cfg.layers = d, layers
    T.Cfg.heads = d // 128            # head_dim pinned at 128 by the FlashKDA kernel
    T.Cfg.ffn_hidden = 3072 if name.startswith("S") else 3 * d
    T.Cfg.seq, T.Cfg.batch = seq, batch
    T.Cfg.attn_res = True             # ON in every measurement, per the order
    T.Cfg.attn_res_blocks = _AR_BLOCKS[0]  # 0 = Full (every sublayer a source); N = N blocks
    T.Cfg.fone = False
    return d, layers


def params_of(model):
    # tok and head share storage (train.py:717), so counting both double-counts the embedding.
    seen, n = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n += p.numel()
    return n


def _apply_compile():
    """train.py:2317-2330's dynamo settings. AttnRes Full builds 1 + 2*layers graphs, so the
    cache limit has to move with depth or compile silently falls back to eager from graph 65
    on -- which is the same eager-vs-compiled confusion this function exists to remove."""
    need = max(64, 2 * T.Cfg.layers + 8)
    torch._dynamo.config.cache_size_limit = need
    torch._dynamo.config.accumulated_cache_size_limit = 4 * need


def run(name, seq, batch, steps, grad_ckpt=False, compile_=True):
    d, layers = configure(name, seq, batch)
    T.Cfg.grad_ckpt = grad_ckpt
    torch.manual_seed(0)
    dev = "cuda"
    model = T.HybridLM(T.Cfg).to(dev).to(torch.bfloat16)
    if compile_:
        _apply_compile()
        model = torch.compile(model, dynamic=False)
    opts = T.build_optimizers(model, T.Cfg)
    nparams = params_of(model)
    torch.cuda.reset_peak_memory_stats()
    V = T.Cfg.vocab
    x = torch.randint(0, V, (batch, seq + 1), device=dev)
    idx, tgt = x[:, :-1].contiguous(), x[:, 1:].contiguous()

    # The real step, not an approximation of it: forward returns hidden, and the loss is
    # Liger FLCE over the head weight (train.py:2374-2379). Substituting a plain
    # cross_entropy here would drop the fused-head kernel, which is the largest single
    # consumer in the step -- the throughput number would then describe a model we do not run.
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=T.SOFTCAP)
    W = model.head.weight[: T.Cfg.vocab]

    def one():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden, _ = model(idx, tgt)
        B, Tt, D = hidden.shape
        loss = flce(W, hidden.to(W.dtype).reshape(-1, D), tgt.reshape(-1))
        loss.backward()
        for o in (opts if isinstance(opts, (list, tuple)) else [opts]):
            o.step()
            o.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tok = batch * seq * steps
    peak = torch.cuda.max_memory_allocated() / 2**30
    # MFU: 6ND is the standard train-FLOPs approximation (fwd+bwd), H20 bf16 peak 148 TFLOPS.
    flops = 6 * nparams * tok
    return {"candidate": name, "d": d, "layers": layers, "heads": d // 128,
            "grad_ckpt": grad_ckpt, "compiled": compile_, "attn_res_blocks": _AR_BLOCKS[0],
            "world": 1, "accum": 1,
            "params_M": round(nparams / 1e6, 1), "seq": seq, "batch": batch,
            "steps_timed": steps, "warmup": WARMUP,
            "ms_per_step": round(dt / steps * 1000, 1),
            "tok_per_s": round(tok / dt),
            "peak_mem_GiB": round(peak, 2),
            "mfu_pct_derived": round(100 * flops / dt / 148e12, 1)}


def find_batch(name, seq, lo=1, hi=64, grad_ckpt=False, compile_=True):
    """Largest batch that fits: double until OOM. Measured, not estimated.

    Doubling only -- it reports the largest POWER OF TWO that fits, which is what a batch
    is chosen from in practice. A bisect would give a tighter bound and a batch nobody uses.
    """
    ok, peak = 0, 0.0
    b = lo
    while b <= hi:
        try:
            configure(name, seq, b)
            T.Cfg.grad_ckpt = grad_ckpt
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(0)
            m = T.HybridLM(T.Cfg).to("cuda").to(torch.bfloat16)
            if compile_:
                _apply_compile()
                m = torch.compile(m, dynamic=False)
            o = T.build_optimizers(m, T.Cfg)
            x = torch.randint(0, T.Cfg.vocab, (b, seq + 1), device="cuda")
            ii, tt = x[:, :-1].contiguous(), x[:, 1:].contiguous()
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            fl = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=T.SOFTCAP)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hid, _ = m(ii, tt)
            Wb = m.head.weight[: T.Cfg.vocab]
            fl(Wb, hid.to(Wb.dtype).reshape(-1, hid.shape[-1]), tt.reshape(-1)).backward()
            for oo in (o if isinstance(o, (list, tuple)) else [o]):
                oo.step()
            ok, peak = b, torch.cuda.max_memory_allocated() / 2**30
        except torch.cuda.OutOfMemoryError:
            break
        finally:
            for v in ("m", "o", "x", "loss"):
                if v in dir():
                    pass
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        b *= 2
    return ok, round(peak, 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", choices=list(CANDIDATES))
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--find-batch", action="store_true")
    ap.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true")
    # Default ON: Cfg.compile is True, so an eager measurement describes a model we do not run.
    ap.add_argument("--eager", action="store_true", help="measure uncompiled (NOT the shipped path)")
    ap.add_argument("--ar-blocks", type=int, default=0,
                    help="attn_res_blocks: 0=Full (pinned default), N=N blocks. Full source "
                         "reads are O(L^2) -- 2145 at L=32 vs 353 at blocks=8 (b0 t71).")
    a = ap.parse_args()
    _AR_BLOCKS[0] = a.ar_blocks
    if a.find_batch:
        got, pk = find_batch(a.candidate, a.seq, grad_ckpt=a.grad_ckpt, compile_=not a.eager)
        print(json.dumps({"candidate": a.candidate, "seq": a.seq, "grad_ckpt": a.grad_ckpt,
                          "compiled": not a.eager, "attn_res_blocks": a.ar_blocks,
                          "largest_batch_that_fits": got, "peak_mem_GiB_at_that_batch": pk}))
    else:
        print(json.dumps(run(a.candidate, a.seq, a.batch, a.steps, a.grad_ckpt,
                             compile_=not a.eager)))
