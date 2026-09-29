"""Does the v42 trainer path train under DDP + torch.compile? (2 cards, torchrun; SKIP off CUDA)

The 8-card trial (v42_arch_b_0929, 2026-09-29) died at the first backward with "element 0 of tensors
does not require grad and does not have a grad_fn" under compile + DDP, with or without the liger
RMSNorm, while the single-process compile of the same model trained. This reproduces the exact
train.py wrapping order on 2 ranks -- build -> .to(bf16) -> convert_to_fp8_compute -> DDP(same kwargs
as train.py's `model = DDP(` call) -> torch.compile(dynamic=False) -> fwd + Liger FLCE + aux + backward -- once with
torch._dynamo.config.optimize_ddp True (DDPOptimizer graph splitting, the default) and once False,
and asserts per arm that the loss has a grad_fn and every trainable parameter received a grad. The
arm that fails is named in the output.

  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tests/v41f/test_gpu_ddp_compile.py [--layers 4] [--B 1]
  python3 tests/v41f/test_gpu_ddp_compile.py --help          # CPU: help only

--layers 4 (default) truncates v42_s24 to layers 0-3 (window-only 0-1, m=2 kv/index source 2, reuse 3)
so a run is a minute, not ten; --layers 24 is the trial's shape. Switches default to the trial's
(fused/real/stacked/liger/liger); --impl overrides, e.g. --impl attn_impl=chunked,norm_impl=torch.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_cfg(a):
    from v41f.config import v42_s24

    L = a.layers
    over = {} if L == 24 else dict(
        n_layers=L, compress_ratios=((0, 0) + (2,) * 10 + (1,) * 12)[:L],
        kv_source_layers=tuple(s for s in (2, 8, 12) if s < L),
        index_source_layers=tuple(s for s in (2, 8, 12, 16, 20) if s < L))
    impl = dict(attn_impl="fused", rope_impl="real", moe_stacked=True, hc_impl="liger", norm_impl="liger")
    for kv in filter(None, a.impl.split(",")):
        k, v = kv.split("=", 1)
        impl[k] = v in ("1", "true", "True") if k == "moe_stacked" else v
    cfg = v42_s24(vocab_size=a.vocab, **over, **impl)
    cfg.validate()
    return cfg


def one_arm(a, optimize_ddp, rank, local, world):
    from torch.nn.parallel import DistributedDataParallel as DDP

    from train import convert_to_fp8_compute
    from v41f.lm import V42LM

    torch._dynamo.reset()
    torch._dynamo.config.optimize_ddp = optimize_ddp
    torch.manual_seed(0)
    cfg = build_cfg(a)
    raw = V42LM(cfg, max_batch_size=a.B).to(local)
    raw = raw.to(torch.bfloat16)  # train.py `raw_model = raw_model.to(torch.bfloat16)`, then the fp8 conversion
    if a.fp8:
        convert_to_fp8_compute(raw)
    model = DDP(raw, device_ids=[local], bucket_cap_mb=50, gradient_as_bucket_view=True, static_graph=True)
    model = torch.compile(model, dynamic=False)
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
    except ImportError:
        flce = None
    g = torch.Generator().manual_seed(rank)
    ids = torch.randint(0, a.vocab, (a.B, a.T), generator=g).to(local)
    tgt = torch.roll(ids, -1, 1)
    hidden, _ = model(ids, cu=None)
    h = hidden.reshape(-1, hidden.size(-1))
    loss = flce(raw.head.weight, h, tgt.reshape(-1)) if flce is not None else \
        torch.nn.functional.cross_entropy((h @ raw.head.weight.t()).float(), tgt.reshape(-1))
    aux = raw.aux_loss()
    if aux is not None:
        loss = loss + aux
    problems = []
    if not loss.requires_grad or loss.grad_fn is None:
        problems.append(f"loss has no grad_fn (requires_grad={loss.requires_grad}); hidden.grad_fn={hidden.grad_fn}")
    else:
        loss.backward()
        missing = [n for n, p in raw.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            problems.append(f"{len(missing)} trainable params without grad, e.g. {missing[:5]}")
        nonfinite = [n for n, p in raw.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if nonfinite:
            problems.append(f"{len(nonfinite)} params with non-finite grad, e.g. {nonfinite[:5]}")
    ok = torch.tensor([0 if problems else 1], device=local)
    torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
    if rank == 0:
        tag = f"optimize_ddp={optimize_ddp} layers={a.layers} fp8={a.fp8} impl={a.impl or 'trial'}"
        print(f"{'ok  ' if ok.item() else 'FAIL'} {tag} loss={float(loss):.4f}" + ("" if not problems else f" :: {problems}"),
              flush=True)
    del model, raw
    torch.cuda.empty_cache()
    return bool(ok.item())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--fp8", type=int, default=1)
    ap.add_argument("--impl", default="", help="k=v,... overrides of the trial's switches")
    ap.add_argument("--arms", default="1,0", help="optimize_ddp values to run, in order")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("test_gpu_ddp_compile: CUDA only; launch with torchrun --nproc_per_node=2")
    torch.distributed.init_process_group("nccl")
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    results = {}
    for arm in a.arms.split(","):
        opt = arm in ("1", "True", "true")
        try:
            results[opt] = one_arm(a, opt, rank, local, world)
        except Exception as e:  # the arm's own failure is the finding; the other arm still runs
            results[opt] = False
            if rank == 0:
                print(f"FAIL optimize_ddp={opt}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank == 0:
        print("ddp+compile: " + ", ".join(f"optimize_ddp={k}: {'ok' if v else 'FAIL'}" for k, v in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
