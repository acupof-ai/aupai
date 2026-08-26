#!/usr/bin/env python3
"""SFT: fine-tune a pretrained HybridLM checkpoint on packed instruction data.

Same acceleration stack as train.py: FP8 compute, torch.compile, DDP, Muon,
Liger FLCE loss. Prompt tokens are masked (labels=-100); loss is computed on
answer + <eos> tokens only.

Usage: torchrun --nproc_per_node=8 sft.py --resume /work/aupai/ckpt.pt.step6000
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Do NOT set FLA_FLASH_KDA=1 — A_log float32 errors during validation.

import argparse
import math
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP

from train import (
    SOFTCAP,
    Cfg,
    HybridLM,
    RunLog,
    build_optimizers,
    convert_to_fp8_compute,
    ddp_even_len,
    opt_snapshot,
    set_schedule,
    setup_ddp,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SFT_DATA = os.path.join(ROOT, "data", "sft", "sft_all.pt")
CKPT_SFT = os.path.join(ROOT, "ckpt_sft.pt")
SAVE_INTERVAL = 500
LOG_INTERVAL = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="pretrained checkpoint path")
    parser.add_argument("--sft_path", default=SFT_DATA)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr_scale", type=float, default=0.1, help="SFT LR = pretrain LR x scale")
    parser.add_argument("--no_fp8", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    for k, v in ck.get("cfg", {}).items():
        setattr(Cfg, k, v)  # model architecture (d, layers, attn_res, ...) comes from the checkpoint
    Cfg.batch = args.batch
    Cfg.epochs = args.epochs
    Cfg.grad_ckpt = True  # required for stability (removing it causes NaN)

    torch.manual_seed(Cfg.seed)
    torch.set_float32_matmul_precision("high")
    ddp, rank, world, local = setup_ddp()
    device = f"cuda:{local}" if ddp else ("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = not ddp or rank == 0
    runlog = RunLog("sft") if is_main else print
    amp = device.startswith("cuda")

    # data: packed (N, seq+1), labels=-100 on prompt tokens
    d = torch.load(args.sft_path, map_location="cpu", weights_only=True)
    X = d["input_ids"][:, :-1].long().contiguous()
    Y = d["labels"][:, 1:].long().contiguous()
    del d
    if ddp:
        X = X[rank::world].contiguous()
        Y = Y[rank::world].contiguous()
    n_even = ddp_even_len(len(X), Cfg.batch, ddp)  # identical step count on every rank
    X, Y = X[:n_even], Y[:n_even]
    X = X.pin_memory()
    Y = Y.pin_memory()
    if is_main:
        print(f"sft rows {len(X)} per rank (world {world})", flush=True)

    # model: pretrained weights -> bf16 -> FP8 compute
    raw_model = HybridLM(Cfg).to(device)
    raw_model.load_state_dict(ck["model"])
    fp8 = not args.no_fp8 and amp
    if fp8:
        raw_model = raw_model.to(torch.bfloat16)
        convert_to_fp8_compute(raw_model)
    if is_main:
        print(
            f"resumed {args.resume} | params {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M | fp8 {fp8}",
            flush=True,
        )

    # optimizers: Muon for 2D params, AdamW for embeddings + 1D (same as train.py)
    optimizers = build_optimizers(raw_model, Cfg, lr_scale=args.lr_scale)

    model = raw_model
    if ddp:
        model = DDP(
            model, device_ids=[local], bucket_cap_mb=100, gradient_as_bucket_view=True, static_graph=True
        )
    if Cfg.compile and amp:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 256
        model = torch.compile(model, dynamic=False)

    good_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
    good_opt = [None] * len(optimizers)
    total_steps = Cfg.epochs * (len(X) // Cfg.batch)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    step = 0
    flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=SOFTCAP)
    weight = raw_model.head.weight[: raw_model.cfg.vocab]

    for ep in range(Cfg.epochs):
        model.train()
        perm = torch.randperm(len(X))
        t0 = time.time()
        for i in range(0, len(X) - Cfg.batch + 1, Cfg.batch):
            idx = perm[i : i + Cfg.batch]
            xb = X[idx].to(device, non_blocking=True)
            yb = Y[idx].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                hidden, _ = model(xb, yb)  # targets passed so compile traces hidden branch
            B, T, D = hidden.shape
            loss = flce(weight, hidden.to(weight.dtype).reshape(-1, D), yb.reshape(-1))
            loss.backward()
            last = loss.item()
            grad_norm = nn.utils.clip_grad_norm_(raw_model.parameters(), Cfg.clip)

            # sync NaN check across DDP ranks: if any rank is bad, all skip
            if ddp:
                flag = torch.tensor([float(math.isfinite(last) and math.isfinite(grad_norm))], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                healthy = flag.item() > 0.5
            else:
                healthy = math.isfinite(last) and math.isfinite(grad_norm)
            if not healthy:
                raw_model.load_state_dict(good_state)
                for j, opt in enumerate(optimizers):
                    if good_opt[j] is not None:
                        opt.load_state_dict(good_opt[j])
                if is_main:
                    runlog(f"step {step}/{total_steps} NaN — restored last good state")
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                step += 1
                continue

            set_schedule(optimizers, step, total_steps, Cfg)
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1

            if step % SAVE_INTERVAL == 0:
                good_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                good_opt = opt_snapshot(optimizers)
                if is_main:
                    torch.save(
                        {
                            "model": good_state,
                            "cfg": {k: v for k, v in vars(Cfg).items() if not k.startswith("_")},
                        },
                        CKPT_SFT + f".step{step}",
                    )
            if is_main and step % LOG_INTERVAL == 0:
                runlog(f"step {step}/{total_steps} loss {last:.3f} {time.time() - t0:.0f}s")
                t0 = time.time()
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    if is_main:
        torch.save(
            {
                "model": raw_model.state_dict(),
                "cfg": {k: v for k, v in vars(Cfg).items() if not k.startswith("_")},
            },
            CKPT_SFT,
        )
        print(f"saved {CKPT_SFT}", flush=True)
        runlog.plot()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
