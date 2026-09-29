"""Step-time bench of the v42 (v41f, preset v42_s24) trainer paths. CUDA only; --help works on CPU.

One micro-batch fwd+bwd (Liger FLCE + aux loss) at --B x --T, every combination of
  attn   : chunked | fused        (V41FConfig.attn_impl; fused = flash_attn.cute window + merged entries)
  moe    : restack | stacked      (V41FConfig.moe_stacked; stacked = one [E,inter,dim] parameter per w)
  compile: 0 | 1                  (torch.compile over the trunk forward; rope_impl real, see --rope)
  fp8    : 0 | 1                  (train.convert_to_fp8_compute, torchao Float8Linear, e4m3 tensorwise)
  --hc / --norm torch | liger     (liger_kernel mHC kernels / LigerRMSNorm; one value per run, not a grid axis)
  --grad_ckpt                     (per-Block recompute; with --B this answers which micro-batch fits)
and prints ms/step (median of --iters), peak GiB, and a CUDA-time split attention / MoE / other from
torch.profiler. The split is by kernel-name substrings (ATTN_KEYS, MOE_KEYS): the ten heaviest kernels
are printed under it so a misfiled kernel is visible, and an unmatched kernel lands in `other`.

  CUDA_VISIBLE_DEVICES=0 python3 scripts/v42_step_bench.py                        # the 16-cell grid
  CUDA_VISIBLE_DEVICES=0 python3 scripts/v42_step_bench.py --attn fused --moe stacked --compile 1 --fp8 0
  CUDA_VISIBLE_DEVICES=0 python3 scripts/v42_step_bench.py --profile --attn fused --moe stacked
       --profile: the full train step (fwd, bwd, V4.1 optimizers step, zero_grad) and a chrome trace
       at --trace (default runs/v42_step_bench_trace.json).
"""

# restartable: prints one row per cell/layer as it goes and holds no state; an interrupt costs one rerun
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GIB = 2**30
ATTN_KEYS = ("flash", "fmha", "attn", "softmax", "logsumexp", "_scaled_dot_product", "rotary")
MOE_KEYS = ("grouped_mm", "gmm", "group_gemm", "index_add", "scatter_add", "sort", "silu", "sigmoid", "topk")


def packed_cu(b, t, g, mean_doc=1500):
    cuts = [0]
    for r in range(b):
        p = r * t
        while True:
            p += int(torch.randint(max(1, mean_doc // 8), 2 * mean_doc, (1,), generator=g))
            if p >= (r + 1) * t:
                break
            cuts.append(p)
        cuts.append((r + 1) * t)
    return torch.tensor(cuts, dtype=torch.int32, device="cuda")


def build(cfg, fp8, compile_):
    from v41f.lm import V42LM

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            m = V42LM(cfg, max_batch_size=1).train()
    finally:
        torch.set_default_dtype(prev)
    if fp8:
        from train import convert_to_fp8_compute

        convert_to_fp8_compute(m)
    fwd = torch.compile(m, dynamic=False) if compile_ else m
    return m, fwd


def make_loss():
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        return lambda w, h, y: flce(w, h, y), "liger_flce"
    except ImportError:
        return (lambda w, h, y: torch.nn.functional.cross_entropy((h @ w.t()).float(), y)), "torch_ce"


def step(m, fwd, loss_fn, ids, tgt, cu, opts=None):
    hidden, _ = fwd(ids, cu=cu)
    loss = loss_fn(m.head.weight, hidden.reshape(-1, hidden.size(-1)), tgt.reshape(-1))
    aux = m.aux_loss()
    if aux is not None:
        loss = loss + aux
    loss.backward()
    if opts is not None:
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        m.commit_moe_token_counts()
    else:
        m.zero_grad(set_to_none=True)
    return float(loss)


def split_cuda_time(prof):
    tot = {"attention": 0.0, "moe": 0.0, "other": 0.0}
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = getattr(e, "self_cuda_time_total", 0.0)
        if t <= 0:
            continue
        n = e.key.lower()
        k = "attention" if any(s in n for s in ATTN_KEYS) else "moe" if any(s in n for s in MOE_KEYS) else "other"
        tot[k] += t
        rows.append((t, k, e.key[:80]))
    rows.sort(reverse=True)
    return {k: v / 1e3 for k, v in tot.items()}, rows[:10]


def bench_cell(a, attn, moe, compile_, fp8, ids, tgt, cu, loss_fn):
    from v41f.config import v42_s24

    cfg = v42_s24(vocab_size=a.vocab, attn_impl=attn, rope_impl=a.rope, moe_stacked=(moe == "stacked"),
                  hc_impl=a.hc, norm_impl=a.norm, block_ckpt=a.grad_ckpt)
    cfg.validate()
    torch._dynamo.reset()
    m, fwd = build(cfg, fp8, compile_)
    opts = None
    if a.profile:
        from v41f.optim import build_v42_optimizers

        opts = build_v42_optimizers(m, cfg, 1e-3)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(a.warmup):
        step(m, fwd, loss_fn, ids, tgt, cu, opts)
    torch.cuda.synchronize()
    times = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        step(m, fwd, loss_fn, ids, tgt, cu, opts)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated() / GIB
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        step(m, fwd, loss_fn, ids, tgt, cu, opts)
        torch.cuda.synchronize()
    split, top = split_cuda_time(prof)
    if a.profile:
        prof.export_chrome_trace(a.trace)
    row = {"attn": attn, "moe": moe, "compile": compile_, "fp8": fp8, "hc": a.hc, "norm": a.norm, "grad_ckpt": a.grad_ckpt, "B": a.B, "T": a.T,
           "mode": "train_step" if a.profile else "fwd_bwd",
           "ms_median": statistics.median(times), "ms_min": min(times), "peak_gib": peak, "cuda_ms": split}
    print(json.dumps(row), flush=True)
    for t, k, name in top:
        print(f"    {t / 1e3:8.2f} ms  {k:9s} {name}", flush=True)
    del m, fwd, opts
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--attn", default="chunked,fused")
    ap.add_argument("--moe", default="restack,stacked")
    ap.add_argument("--compile", default="0,1")
    ap.add_argument("--fp8", default="0,1")
    ap.add_argument("--grad_ckpt", action="store_true", help="V41FConfig.block_ckpt: recompute each Block in backward")
    ap.add_argument("--hc", default="torch", choices=["torch", "liger"])
    ap.add_argument("--norm", default="torch", choices=["torch", "liger"])
    ap.add_argument("--rope", default="real", choices=["real", "complex"],
                    help="rope_impl for every cell; complex has no inductor kernel, so compile=1 needs real")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="full train step with the V4.1 optimizers + chrome trace")
    ap.add_argument("--trace", default=str(ROOT / "runs" / "v42_step_bench_trace.json"))
    ap.add_argument("--json", default=None, help="append one row per cell to this jsonl")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("v42_step_bench: CUDA only (no card visible)")
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(0, a.vocab, (a.B, a.T), generator=g).cuda()
    tgt = torch.roll(ids, -1, 1)
    cu = packed_cu(a.B, a.T, g)
    loss_fn, loss_name = make_loss()
    print(f"v42_step_bench: B{a.B} T{a.T} docs={cu.numel() - 1} loss={loss_name} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name()}", flush=True)
    rows = []
    for attn in a.attn.split(","):
        for moe in a.moe.split(","):
            for c in a.compile.split(","):
                for f in a.fp8.split(","):
                    try:
                        rows.append(bench_cell(a, attn, moe, int(c), int(f), ids, tgt, cu, loss_fn))
                    except Exception as e:  # one failing cell must not hide the others
                        print(json.dumps({"attn": attn, "moe": moe, "compile": int(c), "fp8": int(f),
                                          "error": f"{type(e).__name__}: {str(e)[:200]}"}), flush=True)
                        torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
