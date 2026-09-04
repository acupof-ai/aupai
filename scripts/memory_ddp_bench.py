#!/usr/bin/env python3
# restartable: two ranks on one card, <5 min, writes its json only at the end. An
# interrupt loses the run, never a partial result.
"""Sparse gradient exchange for a 1.07B memory table: what does "never dense" buy?

Charter: docs/standards/memory_layers_0905.md. The rule was "gather touched indices,
never all-reduce the dense 1B table"; 4c struck it on 2026-09-05 after the arithmetic
below, so the exchange is now chosen PER ARM by measured bytes per step. This produces
that number. Sparse gradients stay for the OPTIMIZER either way -- Adagrad state is
dense but the update is row-wise, which is a separate question from the exchange.

THE ARITHMETIC FIRST, so the measurement has something to disagree with. M1's table is
1,048,576 values x 1024 dims in bf16 = 2.0 GiB. A dense all-reduce moves ~2x that per
step (ring all-reduce is 2(N-1)/N x size, ~2.0 GiB at world 2). The sparse side moves
only the rows some rank touched: at batch 16 x seq 4096 = 65,536 tokens x top-k 32 =
2,097,152 index slots per rank per pooled layer, against 1,048,576 distinct values --
so the interesting quantity is how many DISTINCT rows are touched, which is a coupon-
collector question and NOT 2.1M. Expected distinct = M(1 - (1-1/M)^n); at n=2.1M,
M=1.05M that is ~86% of the table, and three pooled layers share one pool, so the union
over layers is higher still.

That is the finding this script exists to expose: at M1's shape the "sparse" path may
touch most of the table anyway, in which case gathering indices costs MORE than a dense
all-reduce (indices + rows + the dedup) and the charter's rule is wrong for M1 while
still being right for M2. Measured, not assumed -- the uniform model above is an upper
bound on distinctness, and real queries are not uniform.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist


def touched_rows(n_tok, topk, n_vals, device, concentration):
    """Indices one rank touches in a step. `concentration` bends the draw away from
    uniform: real product-key queries are peaked, and uniform is the WORST case for
    sparsity (maximum distinct rows). Reported both ways rather than picked."""
    if concentration <= 0:
        return torch.randint(0, n_vals, (n_tok * topk,), device=device)
    # Zipf-ish: draw from a power law over the table, so a few rows dominate.
    u = torch.rand(n_tok * topk, device=device)
    idx = (u.pow(concentration) * n_vals).long().clamp_(0, n_vals - 1)
    return idx


def sparse_exchange(values_grad_rows, idx, world, device):
    """What the sparse path actually costs: all_gather the touched index sets, then the
    rows for them. Two collectives with different shapes per rank, so the sizes must be
    exchanged first -- that handshake is part of the cost and is measured with it."""
    uniq = torch.unique(idx)
    n = torch.tensor([uniq.numel()], device=device)
    sizes = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(sizes, n)
    m = int(max(int(s.item()) for s in sizes))
    pad_i = torch.zeros(m, dtype=uniq.dtype, device=device)
    pad_i[: uniq.numel()] = uniq
    gi = [torch.zeros_like(pad_i) for _ in range(world)]
    dist.all_gather(gi, pad_i)
    rows = values_grad_rows[: uniq.numel()]
    pad_r = torch.zeros(m, rows.shape[1], dtype=rows.dtype, device=device)
    pad_r[: rows.shape[0]] = rows
    gr = [torch.zeros_like(pad_r) for _ in range(world)]
    dist.all_gather(gr, pad_r)
    bytes_moved = (pad_i.numel() * pad_i.element_size()
                   + pad_r.numel() * pad_r.element_size()) * (world - 1) * 2
    return uniq.numel(), bytes_moved


def dense_allreduce(table_grad, world):
    dist.all_reduce(table_grad)
    return table_grad.numel() * table_grad.element_size() * 2 * (world - 1) / world


def run(rank, a):
    dist.init_process_group("nccl", rank=rank, world_size=a.world)
    torch.cuda.set_device(a.device)
    dev = torch.device("cuda", a.device)
    n_vals = a.n_side ** 2
    n_tok = a.batch * a.seq
    torch.manual_seed(42 + rank)

    idx = touched_rows(n_tok, a.topk, n_vals, dev, a.concentration)
    uniq_local = torch.unique(idx).numel()
    grad_rows = torch.randn(uniq_local, a.dv, device=dev, dtype=torch.bfloat16)

    # Sparse
    for _ in range(3):
        sparse_exchange(grad_rows, idx, a.world, dev)
    dist.barrier()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(a.iters):
        n_uniq, sparse_bytes = sparse_exchange(grad_rows, idx, a.world, dev)
    torch.cuda.synchronize()
    sparse_s = (time.perf_counter() - t) / a.iters

    # Dense, the thing the charter forbids -- measured so "forbidden" has a price.
    table_grad = torch.zeros(n_vals, a.dv, device=dev, dtype=torch.bfloat16)
    for _ in range(3):
        dense_allreduce(table_grad, a.world)
    dist.barrier()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(a.iters):
        dense_bytes = dense_allreduce(table_grad, a.world)
    torch.cuda.synchronize()
    dense_s = (time.perf_counter() - t) / a.iters

    if rank == 0:
        res = {
            "n_values": n_vals, "dv": a.dv, "topk": a.topk, "world": a.world,
            "tokens_per_step": n_tok, "concentration": a.concentration,
            "index_slots_per_rank": n_tok * a.topk,
            "distinct_rows_touched": n_uniq,
            "frac_table_touched": n_uniq / n_vals,
            "table_gib": n_vals * a.dv * 2 / 2**30,
            "sparse_bytes_per_step": sparse_bytes,
            "dense_bytes_per_step": dense_bytes,
            "sparse_s_per_step": sparse_s,
            "dense_s_per_step": dense_s,
            "sparse_over_dense_bytes": sparse_bytes / dense_bytes,
            "sparse_over_dense_time": sparse_s / dense_s,
        }
        res["verdict"] = (
            "sparse wins" if res["sparse_over_dense_time"] < 1
            else "SPARSE LOSES at this shape -- the table is mostly touched"
        )
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        for k, v in res.items():
            print(f"  {k}: {v}")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-side", type=int, default=1024)
    ap.add_argument("--dv", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--concentration", type=float, default=0.0,
                    help="0 = uniform (worst case for sparsity); >0 = power-law peaked")
    ap.add_argument("--out", default="runs/memory_ddp_bench.json")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA", file=sys.stderr)
        return 1
    rank = int(os.environ.get("RANK", "0"))
    run(rank, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
