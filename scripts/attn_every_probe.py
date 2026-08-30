"""Peak memory and step time as a function of attn_every, before an A/B commits to it.

The KDA-vs-full-attention A/B needs both arms at the same batch, so the question that has to be
answered first is whether attn_every=1 (every block a GatedMLA) even fits. Finding that out from
a probe costs minutes; finding it out from eight budget runs costs hours.

Single process, random tokens: activation memory does not care what the tokens say, and the
corpus is being rebuilt. That makes this a LOWER BOUND -- DDP adds gradient buckets and NCCL
buffers this does not measure, so a result close to the card is a fail, not a pass.

Run: CUDA_VISIBLE_DEVICES=0 python -u scripts/attn_every_probe.py --batch 16
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402

MB = 1e6


def measure(cfg, attn_every, steps=3, warmup=2):
    cfg.attn_every = attn_every
    model = train.HybridLM(cfg).cuda().to(torch.bfloat16)
    n_attn = sum(1 for i in range(cfg.layers) if i % attn_every == attn_every - 1)
    opts = train.build_optimizers(model, cfg)
    idx = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq + 1), device="cuda")
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()

    def step():
        model(x, y)[0].sum().backward()
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
    return {
        "attn_every": attn_every,
        "attn_layers": n_attn,
        "peak_MB": peak,
        "ms_per_step": ms,
        "tok_s": cfg.batch * cfg.seq / (ms / 1e3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--values", default="4,1")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = train.Cfg
    cfg.batch, cfg.compile = args.batch, False
    rows = []
    for v in [int(x) for x in args.values.split(",")]:
        try:
            r = measure(cfg, v, steps=args.steps)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            r = {"attn_every": v, "peak_MB": None, "ms_per_step": None, "oom": True}
        rows.append(r)
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
            "grad_ckpt": cfg.grad_ckpt,
            "compiled": False,
            "fp8": False,
            "ddp": False,
            "dtype": "bfloat16",
            "steps": args.steps,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "note": "single process, random tokens -- a LOWER bound; DDP buckets and "
            "NCCL buffers are not included",
        },
        "rows": rows,
    }
    print(json.dumps(out["config"]), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
