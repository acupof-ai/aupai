#!/usr/bin/env python3
# restartable: one GPU, <20 min of card time, writes its json only at the end -- an
# interrupt loses the run, never a partial result. Rerun costs less than checkpointing.
"""Product-key memory: is the lookup+gather cheap enough to be "near-zero FLOPs"?

Charter: docs/standards/memory_layers_0905.md, arm M1. My half is the kernel and the
sparse DDP gradient path; b0 owns the module. This measures the two pieces in isolation
so the answer does not wait on the module, and so a slow arm can be attributed to the
lookup rather than to everything at once.

THE BAR, stated as the charter states it: the control b0_headmix_armA runs at 82K
tok/s/gpu (readout 5; below 70K the arm is stopped). Its step is batch 16 x seq 4096 =
65,536 tokens per gpu, so 82K tok/s is 1.25 optimizer steps/s, i.e. 799 ms/step. The
memory adds three pooled layers (3, 6, 9), so the lookup budget at the charter's <=3%
FLOP bound is 3% x 799 ms = 24.0 ms/step for all three, 8.0 ms each.

That budget is the honest way to read ">=85% of 82K": a lookup consuming X ms/step costs
82K x (799 / (799 + X)) tok/s, and 85% of 82K = 69.7K -- which is 141 ms/step, right at
the charter's own 70K stop line. So the two constraints agree, and the number to report
is ms/step, from which both follow.

WHAT THIS IS NOT: an end-to-end training number. No backbone, no optimizer, no DDP
overlap. It bounds the lookup's cost from below -- the integrated step-30 figure can only
be worse. Reported as a bound, never summed with a trace row.
"""
import argparse
import json
import os
import sys
import time

import torch


def product_keys(n_side, dq, device, dtype):
    """Two half-key tables. The product of two 1024-entry sides is a 1,048,576-entry
    table addressed by 2 x 1024 comparisons instead of 1M -- that is the whole reason
    product keys exist, and why the lookup is not itself a 1M-row matmul."""
    g = torch.Generator(device="cpu").manual_seed(42)
    k1 = torch.randn(n_side, dq // 2, generator=g).to(device=device, dtype=dtype)
    k2 = torch.randn(n_side, dq // 2, generator=g).to(device=device, dtype=dtype)
    return k1, k2


def topk_indices(q, k1, k2, topk, n_side):
    """Product-key top-k: top-k on each half, then the best `topk` of the k x k
    candidate cross-products. The cross-product step is the part people get wrong --
    the top-k of the product is NOT the product of the top-ks in general, but it is
    contained in the k x k grid of the per-half top-ks, which is why this is exact
    for the grid and why the grid must be searched rather than just its diagonal."""
    q1, q2 = q.chunk(2, dim=-1)
    s1, i1 = (q1 @ k1.T).topk(topk, dim=-1)     # [N, topk]
    s2, i2 = (q2 @ k2.T).topk(topk, dim=-1)
    cand = s1.unsqueeze(-1) + s2.unsqueeze(-2)  # [N, topk, topk]
    flat_s, flat_i = cand.flatten(1).topk(topk, dim=-1)
    row = flat_i // topk
    col = flat_i % topk
    idx = i1.gather(1, row) * n_side + i2.gather(1, col)
    return idx, flat_s.softmax(-1)


def gather_embedding_bag(values, idx, w):
    """torch.nn.functional.embedding_bag with per_sample_weights: one fused kernel for
    gather + weighted sum, and the path with a real sparse-gradient implementation."""
    return torch.nn.functional.embedding_bag(
        idx, values, per_sample_weights=w, mode="sum", sparse=True
    )


def gather_index_select(values, idx, w):
    """The naive path: materialize [N, topk, dv] then weight-and-sum. Same math, but it
    writes N*topk*dv elements to HBM before reducing them -- at N=65536, topk=32,
    dv=1024 that is 8.6 GB of traffic per call in bf16. This is the arithmetic the
    Triton kernel would be written to avoid, so measuring it says what a kernel could buy."""
    return (values[idx] * w.unsqueeze(-1)).sum(1)


def bench(fn, *a, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*a)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-side", type=int, default=1024)   # 1024^2 = 1,048,576 values (M1)
    ap.add_argument("--dv", type=int, default=1024)
    ap.add_argument("--dq", type=int, default=512)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--pools", type=int, default=3)       # layers 3, 6, 9 share one pool
    ap.add_argument("--out", default="runs/memory_lookup_bench.json")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA: this measures a GPU kernel and refuses to report a CPU number "
              "as if it were one", file=sys.stderr)
        return 1

    dev = torch.device("cuda")
    dt = torch.bfloat16
    n_vals = a.n_side ** 2
    n_tok = a.batch * a.seq

    torch.manual_seed(42)
    values = torch.randn(n_vals, a.dv, device=dev, dtype=dt)
    k1, k2 = product_keys(a.n_side, a.dq, dev, dt)
    q = torch.randn(n_tok, a.dq, device=dev, dtype=dt)

    res = {
        "n_values": n_vals, "dv": a.dv, "topk": a.topk, "tokens_per_step": n_tok,
        "pools": a.pools, "gpu": torch.cuda.get_device_name(0),
        "values_gib": values.numel() * values.element_size() / 2**30,
    }

    idx, w = topk_indices(q, k1, k2, a.topk, a.n_side)
    res["ms_topk"] = bench(topk_indices, q, k1, k2, a.topk, a.n_side)

    # embedding_bag reads the [N, topk] form directly as N bags of topk each.
    res["ms_gather_embedding_bag"] = bench(gather_embedding_bag, values, idx, w.float())
    res["ms_gather_index_select"] = bench(gather_index_select, values, idx, w)

    best = min(res["ms_gather_embedding_bag"], res["ms_gather_index_select"])
    res["ms_lookup_total_one_pool"] = res["ms_topk"] + best
    res["ms_lookup_total_all_pools"] = res["ms_lookup_total_one_pool"] * a.pools

    # The bar, derived rather than asserted: the control's 82K tok/s/gpu at 65,536
    # tokens/step is 799 ms/step. A lookup costing X ms/step leaves 82K*(799/(799+X)).
    ctrl_toks = 82_000.0
    ctrl_ms = n_tok / ctrl_toks * 1000
    x = res["ms_lookup_total_all_pools"]
    res["control_ms_per_step"] = ctrl_ms
    res["projected_tok_s"] = ctrl_toks * ctrl_ms / (ctrl_ms + x)
    res["pct_of_control"] = res["projected_tok_s"] / ctrl_toks * 100
    res["budget_ms_at_3pct_flops"] = ctrl_ms * 0.03
    res["verdict"] = (
        "PASS >=85%" if res["pct_of_control"] >= 85 else
        "STOP <70K" if res["projected_tok_s"] < 70_000 else "between: measured, not adopted"
    )

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
