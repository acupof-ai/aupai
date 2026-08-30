"""How many parameter updates does bf16 swallow?

`run_ddp.sh` always passes `--fp8`, and `train.py:1626` then casts the whole model to
bfloat16 before training. There are no fp32 master weights: the optimizers hold the bf16
parameters directly, so their state is bf16 too, and every `step()` rounds the result back
to bf16. bfloat16 carries 8 mantissa bits, so an update smaller than ~2^-9 of a
parameter's magnitude cannot change its bit pattern -- it is computed, applied, and
silently discarded.

This measures the discard rate directly: run real steps and count the parameter elements
whose *bit pattern* is unchanged across `optimizer.step()`. A stalled parameter looks
exactly like a converged one from the loss curve, which is why this needs its own number.

    E2E_GPU=3 python scripts/bf16_update_loss.py --ckpt ckpt_p324.pt --steps 8

The reading that matters is how the rate moves with lr_scale: the 30B run spends its tail
at a small LR, and that is where updates fall under the representable step.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import Cfg, HybridLM, build_optimizers, convert_to_fp8_compute, doc_cu_seqlens  # noqa: E402


def frozen_fraction(before, after):
    """Fraction of elements whose bf16 bit pattern did not move. Bit-level, not allclose:
    two bf16 values that compare equal ARE the same value, and that is the point."""
    frozen = total = 0
    for name, b in before.items():
        a = after[name]
        same = (a.view(torch.int16) == b.view(torch.int16)).sum().item()
        frozen += same
        total += a.numel()
    return frozen, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_p324.pt")
    ap.add_argument("--sft_path", default="data/sft/sft_all.pt", help="real tokens for real gradients")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument(
        "--lr_scales",
        default="1.0,0.1,0.01",
        help="the schedule's tail is where an update stops being representable",
    )
    args = ap.parse_args()

    gpu = os.environ.get("E2E_GPU")
    assert gpu is not None, "set E2E_GPU=<idx>; this probe needs a card and will not pick one"
    dev = "cuda:0"  # CUDA_VISIBLE_DEVICES is set by the caller; fla kernels use the current device

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    for k, v in ck.get("cfg", {}).items():
        if not k.startswith("_") and hasattr(Cfg, k):
            setattr(Cfg, k, v)
    Cfg.batch = args.batch
    Cfg.compile = False  # the probe measures the optimizer, not the compiler

    d = torch.load(args.sft_path, map_location="cpu", weights_only=True)
    X = d["input_ids"][:, :-1].long()
    Y = d["labels"][:, 1:].long()

    print(f"ckpt {args.ckpt} | batch {args.batch} | steps {args.steps} | rows {len(X)}")
    print(f"param dtype as trained: {next(iter(ck['model'].values())).dtype}")

    for scale in [float(s) for s in args.lr_scales.split(",")]:
        m = HybridLM(Cfg).to(dev)
        m.load_state_dict(ck["model"])
        m = m.to(torch.bfloat16)  # exactly what train.py:1626 does under --fp8
        convert_to_fp8_compute(m)
        m.train()
        opts = build_optimizers(m, Cfg)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["lr"] * scale

        rates = []
        for step in range(args.steps):
            i = (step * args.batch) % (len(X) - args.batch)
            xb = X[i : i + args.batch].to(dev)
            yb = Y[i : i + args.batch].to(dev)
            before = {n: p.detach().clone() for n, p in m.named_parameters()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = m(xb, yb, doc_cu_seqlens(xb, 1) if Cfg.doc_mask else None)
            loss.backward()
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            after = {n: p.detach() for n, p in m.named_parameters()}
            frozen, total = frozen_fraction(before, after)
            rates.append(100.0 * frozen / total)
            print(f"  lr_scale {scale:<6} step {step}  loss {loss.item():.4f}  frozen {rates[-1]:.2f}%")
        print(f"lr_scale {scale}: mean frozen {sum(rates) / len(rates):.2f}% of parameter elements")
        del m, opts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
