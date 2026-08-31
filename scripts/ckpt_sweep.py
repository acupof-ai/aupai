"""Selective activation recompute: (peak memory, step time) for a few module combinations.

The full-model grad_ckpt is 25% slower; not checkpointing at all OOMs at batch 32. This sweeps
the middle: wrap only the modules that actually hold the activations, per probes/mem_account.py
(DeltaRecurrence 30.2%, SwiGLU 26.7%, RMSNorm 11.1% of saved bytes).

Each combination is printed as it finishes, so a run killed partway still leaves usable rows.

Run: CUDA_VISIBLE_DEVICES=0 python -u scripts/ckpt_sweep.py --batch 16
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.utils.checkpoint as tuc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402

MB = 1e6
COMBOS = [
    [],
    ["RMSNorm"],
    ["SwiGLU"],
    ["DeltaRecurrence"],
    ["DeltaRecurrence", "SwiGLU"],
    ["DeltaRecurrence", "SwiGLU", "RMSNorm"],
]


def wrap(model, names):
    """Route the named module types through checkpoint. use_reentrant=False so the modules keep
    their kwargs (cu=...) and so backward works under a graph that was not built reentrantly."""
    for mod in model.modules():
        if type(mod).__name__ in names:
            inner = mod.forward

            def fwd(*a, _f=inner, **kw):
                return tuc.checkpoint(_f, *a, use_reentrant=False, **kw)

            mod.forward = fwd


def measure(cfg, names, compiled, steps=5, warmup=2):
    model = train.HybridLM(cfg).cuda().to(torch.bfloat16)
    wrap(model, names)
    opts = train.build_optimizers(model, cfg)
    idx = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq + 1), device="cuda")
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    run = torch.compile(model) if compiled else model

    def step():
        run(x, y)[0].sum().backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / steps
    peak = torch.cuda.max_memory_allocated() / MB
    torch.cuda.empty_cache()
    return {"wrapped": names or ["none"], "peak_MB": peak, "ms_per_step": ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = train.Cfg
    cfg.batch, cfg.compile = args.batch, False
    rows = []
    for names in COMBOS:
        try:
            r = measure(cfg, names, args.compile, steps=args.steps)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            r = {"wrapped": names or ["none"], "peak_MB": None, "ms_per_step": None, "oom": True}
        rows.append(r)
        base = rows[0]
        if base["peak_MB"] and r["peak_MB"]:
            r["mem_vs_none"] = r["peak_MB"] / base["peak_MB"]
            r["time_vs_none"] = r["ms_per_step"] / base["ms_per_step"]
        print(json.dumps(r), flush=True)

    out = {
        "config": {
            "batch": cfg.batch,
            "seq": cfg.seq,
            "d": cfg.d,
            "layers": cfg.layers,
            "vocab": cfg.vocab,
            "ffn_hidden": cfg.ffn_hidden,
            "attn_res": cfg.attn_res,
            "attn_res_blocks": cfg.attn_res_blocks,
            "compiled": args.compile,
            "fp8": False,
            "dtype": "bfloat16",
            "ddp": False,
            "steps": args.steps,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "rows": rows,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out["config"]), flush=True)


if __name__ == "__main__":
    main()
