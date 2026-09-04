#!/usr/bin/env python3
"""The table gradient: what does the scatter-add cost, and what is cheaper?

The decomposition (runs/mem_decomp_0905.jsonl, 2026-09-05) put 97.3 ms per pooled layer in
backward, scaling with top_k -- 105 ms per layer for 131,072 token-lookups x 32 rows, 25 ns
per row-gradient, which is atomics-bound rather than bandwidth-bound. This measures the
candidates 4c named against that baseline, at the arm's shape, with no model around them.

WHAT IS BEING TIMED. Only the gradient of `values` with respect to the read. That is the
whole of the per-layer backward cost the fit isolated; the query GEMMs and the gate/out
projections are the same in every variant and are not timed.

  dense_embedding   nn.Embedding's own backward, what the arm runs today.
  index_add         index_add_ on the unsorted flat indices, the one-liner.
  sort_segment      sort the flat indices first, then index_add_ -- 4c's option (i).

4c's option (ii), deferring the table gradient so one scatter serves all three layers, is
NOT here: it saves at most the two extra zero-fills and the kernel launches, because the
row-gradients themselves still have to be scattered once each. If the cost is atomic
contention per row-gradient, and k16 says it scales with top_k, then merging three scatters
into one moves no atomics. Measured first, proposed second.

BITWISE OR SUMMATION ORDER, reported per variant rather than assumed. Every variant computes
the same mathematical gradient, so a difference is float addition being non-associative --
but "non-associative" is the mechanism, not a measurement, and a variant that is bitwise
identical is a strictly safer swap than one that is not. Both are reported, and the max
absolute difference is printed next to the table's own scale so a reader can judge it.

    python3 scripts/mem_grad_bench.py                    # M1 shape, one card
    python3 scripts/mem_grad_bench.py --selftest         # correctness, no card
"""
import argparse
import json
import os
import sys

import torch


def make_case(n_vals, dv, n_tok, top_k, device, seed=0):
    """Indices and upstream gradient at the arm's shape. Uniform draw: the real query
    distribution is peaked, and peaked is EASIER for a scatter (fewer distinct rows, more
    coalescing), so uniform is the honest worst case for the thing being replaced."""
    g = torch.Generator(device=device).manual_seed(seed)
    flat = torch.randint(0, n_vals, (n_tok, top_k), device=device, generator=g)
    # d(read)/d(values[row]) is w * grad_out, so the per-row gradient rows are what a real
    # backward hands the scatter. Built once and shared by every variant.
    rows = torch.randn(n_tok, top_k, dv, device=device, dtype=torch.bfloat16, generator=g)
    return flat, rows


def grad_dense_embedding(values, flat, rows):
    """What the arm runs: autograd's own embedding backward."""
    v = values.detach().requires_grad_(True)
    out = torch.nn.functional.embedding(flat, v)
    out.backward(rows)
    return v.grad


def grad_index_add(values, flat, rows):
    """The one-liner: index_add_ into a zeroed dense grad, unsorted."""
    g = torch.zeros_like(values)
    g.index_add_(0, flat.reshape(-1), rows.reshape(-1, rows.shape[-1]))
    return g


def grad_sort_segment(values, flat, rows):
    """Sort the flat indices first, then index_add_. The sort costs O(n log n) but every
    write to one row becomes contiguous, which is the whole point: the dense path's cost is
    atomic contention on rows many tokens share, not the bytes."""
    f = flat.reshape(-1)
    order = f.argsort()
    g = torch.zeros_like(values)
    g.index_add_(0, f[order], rows.reshape(-1, rows.shape[-1])[order])
    return g


def bench(fn, *a, iters=20, warmup=5):
    for _ in range(warmup):
        fn(*a)
    torch.cuda.synchronize()
    import time
    t = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


VARIANTS = {
    "dense_embedding": grad_dense_embedding,
    "index_add": grad_index_add,
    "sort_segment": grad_sort_segment,
}


def _selftest():
    """Every variant computes the same gradient, on a shape small enough to check exactly.
    No card needed -- the variants are device-agnostic."""
    bad = 0
    torch.manual_seed(0)
    n_vals, dv, n_tok, top_k = 32, 4, 16, 3
    values = torch.randn(n_vals, dv)
    flat = torch.randint(0, n_vals, (n_tok, top_k))
    rows = torch.randn(n_tok, top_k, dv)

    ref = grad_dense_embedding(values, flat, rows)
    for name, fn in VARIANTS.items():
        got = fn(values, flat, rows)
        ok = torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} {name} matches the dense gradient "
              f"(max|d| {(got - ref).abs().max():.2e})")

    # A DUPLICATED INDEX IS THE WHOLE POINT. If every row were touched once, a scatter and a
    # plain assignment agree and the test proves nothing about accumulation -- which is
    # exactly the property that separates a correct scatter from a broken one.
    flat_dup = torch.zeros(n_tok, top_k, dtype=torch.long)
    ref_dup = grad_dense_embedding(values, flat_dup, rows)
    ok = torch.allclose(ref_dup[0], rows.reshape(-1, dv).sum(0), rtol=1e-5, atol=1e-5)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} all-same-index accumulates rather than overwrites "
          f"(row 0 == sum of every upstream row)")
    for name, fn in VARIANTS.items():
        got = fn(values, flat_dup, rows)
        ok = torch.allclose(got, ref_dup, rtol=1e-5, atol=1e-5)
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} {name} accumulates duplicates correctly")

    n = 1 + 2 * len(VARIANTS)
    print(f"mem_grad_bench selftest: {n - bad}/{n} pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-values", type=int, default=1048576)
    ap.add_argument("--dv", type=int, default=1024)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--out", default="runs/mem_grad_bench.json")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not torch.cuda.is_available():
        print("no CUDA: this measures a GPU kernel and will not report a CPU number as one",
              file=sys.stderr)
        return 1

    dev = torch.device("cuda")
    # The step's token count, accum included: one pooled layer scatters this many lookups per
    # optimizer step, and the decomposition's 97.3 ms/layer is per step, not per micro-batch.
    n_tok = a.batch * a.seq * a.accum
    values = torch.randn(a.n_values, a.dv, device=dev, dtype=torch.bfloat16)
    flat, rows = make_case(a.n_values, a.dv, n_tok, a.top_k, dev)

    res = {"n_values": a.n_values, "dv": a.dv, "top_k": a.top_k, "tokens_per_step": n_tok,
           "row_grads": n_tok * a.top_k, "gpu": torch.cuda.get_device_name(0),
           "measured_per_layer_ms": 97.3, "variants": {}}
    ref = None
    for name, fn in VARIANTS.items():
        ms = bench(fn, values, flat, rows)
        g = fn(values, flat, rows)
        if ref is None:
            ref = g
            same, maxd = True, 0.0
        else:
            same = bool(torch.equal(g, ref))
            maxd = float((g.float() - ref.float()).abs().max())
        res["variants"][name] = {
            "ms": round(ms, 2),
            "ns_per_row_grad": round(ms * 1e6 / (n_tok * a.top_k), 1),
            "bitwise_identical_to_dense": same,
            "max_abs_diff": maxd,
            "vs_dense": round(ms / res["variants"]["dense_embedding"]["ms"], 3)
            if "dense_embedding" in res["variants"] else 1.0,
        }
        print(f"  {name:16s} {ms:8.2f} ms  "
              f"{ms * 1e6 / (n_tok * a.top_k):5.1f} ns/row  "
              f"{'bitwise identical' if same else f'max|d| {maxd:.2e}'}", flush=True)
        del g
        torch.cuda.empty_cache()

    # THE PROJECTION, so the number answers the question that was asked. Three pooled layers
    # at the measured 97.3 ms each sit inside a 2016.9 ms step; a variant saving X ms per
    # layer moves the step to 2016.9 - 3X and the ratio to 1592.6/(that).
    step_m1, step_off = 2016.9, 1592.6
    base = res["variants"]["dense_embedding"]["ms"]
    for v in res["variants"].values():
        saved = 3 * (base - v["ms"])
        step = step_m1 - saved
        v["projected_step_ms"] = round(step, 1)
        v["projected_ratio_vs_control"] = round(step_off / step, 3)
    print("\nprojection to the arm (3 pooled layers, M1 step 2016.9 ms, control 1592.6):")
    for name, v in res["variants"].items():
        print(f"  {name:16s} step {v['projected_step_ms']:7.1f} ms  "
              f"ratio {v['projected_ratio_vs_control']:.3f}"
              f"{'  <- clears 0.85' if v['projected_ratio_vs_control'] >= 0.85 else ''}")
    print("  A PROJECTION, not a measurement: it assumes the scatter is the only thing that "
          "changes and that the rest of backward is untouched. The arm-shape cell is what "
          "would settle it.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
