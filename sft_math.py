#!/usr/bin/env python3
"""Stage-2 math SFT: same as sft.py but with --out (never overwrites ckpt_sft.pt)
and SAVE_INTERVAL=200 (stage-2 runs are ~300-360 steps).

Usage: torchrun --nproc_per_node=6 sft_math.py --resume ckpt_sft.pt \
  --sft_path data/sft/sft_math.pt --out ckpt_sft_math.pt --epochs 2 --lr_scale 0.05
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import math
import re
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP

import fone
from train import (
    SOFTCAP,
    Cfg,
    HybridLM,
    RunLog,
    build_optimizers,
    convert_to_fp8_compute,
    ddp_even_len,
    doc_cu_seqlens,
    opt_snapshot,
    set_schedule,
    setup_ddp,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SFT_DATA = os.path.join(ROOT, "data", "sft", "sft_all.pt")
EOS_ID = 1  # <eos> id in data/tokenizer.json; packed SFT rows carry ~10 samples each
SAVE_INTERVAL = 200
LOG_INTERVAL = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="pretrained checkpoint path")
    parser.add_argument("--sft_path", default=SFT_DATA)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr_scale", type=float, default=0.1, help="SFT LR = pretrain LR x scale")
    parser.add_argument("--no_fp8", action="store_true")
    parser.add_argument(
        "--no_grad_ckpt",
        action="store_true",
        help="disable activation checkpointing (FP8 backward goes NaN without it)",
    )
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "ckpt_sft_math.pt"),
        help="output checkpoint path (default: ckpt_sft_math.pt)",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="override the base's vocabulary fingerprint, for a checkpoint saved before "
        "train.py started recording it (print it with scripts/ckpt_info.py)",
    )
    args = parser.parse_args()

    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    for k, v in ck.get("cfg", {}).items():
        setattr(Cfg, k, v)  # model architecture (d, layers, attn_res, ...) comes from the checkpoint
    Cfg.batch = args.batch
    Cfg.epochs = args.epochs
    # grad_ckpt must stay ON: FP8 e4m3 backward produced NaN at step ~100 without it (perf_sweep 2026-08-26)
    Cfg.grad_ckpt = not args.no_grad_ckpt

    torch.manual_seed(Cfg.seed)
    torch.set_float32_matmul_precision("high")
    ddp, rank, world, local = setup_ddp()
    device = f"cuda:{local}" if ddp else ("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = not ddp or rank == 0
    runlog = (
        RunLog(re.sub(r"^ckpt_", "", os.path.splitext(os.path.basename(args.out))[0])) if is_main else print
    )
    amp = device.startswith("cuda")

    d = torch.load(args.sft_path, map_location="cpu", weights_only=True)
    X = d["input_ids"][:, :-1].long().contiguous()
    Y = d["labels"][:, 1:].long().contiguous()
    # V feeds the embedding, W is the digit target one position later -- the same
    # split train.py uses. A FoNE base on a pack without values would read every
    # number as zero, so the mismatch is an error rather than a silent fallback.
    # A pack built against a different vocabulary trains happily at four times the
    # loss: every id is wrong and every id is still in range, so nothing raises.
    # Measured 2026-08-28 -- k5 on a k6-vocab pack sat at loss 4.77 where the matched
    # run was at 1.28. Vocab SIZE does not catch it either (the two differ by one id
    # that a non-FoNE pack never emits), so the pack carries a fingerprint of the
    # id->token map and --vocab is checked against it.
    ck_vocab = args.vocab or ck.get("vocab_id")
    if ck_vocab and "vocab" in d:
        assert d["vocab"] == ck_vocab, (
            f"{args.sft_path} was packed against vocabulary {d['vocab']} but "
            f"{args.resume} was trained on {ck_vocab}; repack with "
            "`prepare_sft_math.py --tokenizer <the base's tokenizer.json>`"
        )
    elif is_main:
        missing = "the checkpoint" if not ck_vocab else "the pack"
        print(f"WARNING {missing} predates vocabulary fingerprinting; verify by hand", flush=True)
    assert Cfg.fone == ("values" in d), (
        f"checkpoint fone={Cfg.fone} but {args.sft_path} "
        f"{'has' if 'values' in d else 'has no'} values; repack with prepare_sft_math.py --fone"
    )
    V = d["values"][:, :-1].contiguous() if Cfg.fone else None
    W = d["values"][:, 1:].contiguous() if Cfg.fone else None
    del d
    if ddp:
        X = X[rank::world].contiguous()
        Y = Y[rank::world].contiguous()
        if Cfg.fone:
            V, W = V[rank::world].contiguous(), W[rank::world].contiguous()
    n_even = ddp_even_len(len(X), Cfg.batch, ddp)  # identical step count on every rank
    X, Y = X[:n_even].pin_memory(), Y[:n_even].pin_memory()
    if Cfg.fone:
        V, W = V[:n_even].pin_memory(), W[:n_even].pin_memory()
    if is_main:
        print(f"sft rows {len(X)} per rank (world {world})", flush=True)

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

    optimizers = build_optimizers(raw_model, Cfg)

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
            vb = V[idx].to(device, non_blocking=True) if Cfg.fone else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                hidden, _ = model(xb, yb, doc_cu_seqlens(xb, EOS_ID) if Cfg.doc_mask else None, vb)
            B, T, D = hidden.shape
            loss = flce(weight, hidden.to(weight.dtype).reshape(-1, D), yb.reshape(-1))
            if Cfg.fone:
                # Only supervised [NUM] positions: a prompt-masked one carries no loss here
                # either, or the model would be scored on digits it was told to ignore.
                nmask = yb == Cfg.num_id
                if nmask.any():
                    wb = W[idx].to(device, non_blocking=True)
                    loss = loss + Cfg.fone_loss_w * F.cross_entropy(
                        raw_model.num_logits(hidden[nmask].float()).reshape(-1, 10),
                        fone.digit_targets(wb[nmask]).reshape(-1),
                    )
            loss.backward()
            last = loss.item()
            grad_norm = nn.utils.clip_grad_norm_(raw_model.parameters(), Cfg.clip)

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
                if args.max_steps and step >= args.max_steps:
                    break
                continue

            set_schedule(optimizers, step, total_steps, Cfg, args.lr_scale)
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
                        args.out + f".step{step}",
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
            args.out,
        )
        print(f"saved {args.out}", flush=True)
        runlog.plot()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
