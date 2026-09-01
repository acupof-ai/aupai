"""Cache the head weight's fp8 BYTES, not just its scale: how much is that worth?

_FP8_WSCALE (train.py:448) caches the head weight's SCALE across the 64 FLCE chunks, because
recomputing an absmax over 32784x1024 sixty-four times for one scalar is waste. But the
quantised BYTES are not cached, so every chunk still runs the divide and the cast over 67.1 MB
-- and at two call sites, in two orientations. Site 2 (`grad_input`) does not even pass
cache_b, so it recomputes the scale as well.

The weight is `self.head.weight`, tied to `self.tok.weight` (train.py:717), and it does not
change within a step: the optimizer runs after FLCE. So both orientations could be quantised
once per step instead of 64 times.

This is not an epilogue. No CUTLASS, no Liger kernel change -- a dict holding tensors instead
of scalars. That makes it the cheapest thing on the ladder if the number holds, so it is worth
measuring rather than deriving: my first arithmetic on it had a 1000x unit error, and the
traffic model that produced 135.1 ms was fitted, not measured.

Three arms, interleaved, same statistic as t58:
    live     what production does now -- quantise W at both sites, every chunk
    cached   quantise both orientations once, reuse for all 64
    scale    cache the scale only (today's behaviour at site 1) as the middle case

    CUDA_VISIBLE_DEVICES=7 python -u probes/t60_weight_cache.py
"""
import json
import statistics
import sys

import torch

sys.argv = ["t60"]
sys.path.insert(0, "/work/aupai/scripts")
sys.path.insert(0, "/work/aupai")
import train  # noqa: E402

M, K, N = 2048, 1024, 32784
CHUNKS, ITERS, WARMUP = 64, 60, 10
E4M3 = 448.0


def _time(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def _q(t):
    sc = (t.detach().abs().amax().clamp(min=1e-12) / E4M3).float()
    return (t / sc).to(torch.float8_e4m3fn), sc


def main():
    dev = "cuda"
    torch.manual_seed(0)
    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16).normal_(0, 0.5)
    G = torch.empty(M, N, device=dev, dtype=torch.bfloat16).normal_(0, 0.02)
    Wn = torch.empty(N, K, device=dev, dtype=torch.bfloat16).normal_(0, 0.02)  # [vocab, hidden]
    Wt = Wn.t().contiguous()

    # Quantised once, outside every timed window: what a per-step byte cache would hold.
    qWt, sWt = _q(Wt)
    qWn, sWn = _q(Wn)

    def sm(a, sa, b, sb, out):
        return torch._scaled_mm(a, b.t().contiguous().t(), scale_a=sa, scale_b=sb, out_dtype=out)

    def live():
        # exactly production: site 1 caches the scale, site 2 does not
        train._fp8_mm(A, Wt, A.dtype, cache_b=True)
        train._fp8_mm(G, Wn, torch.bfloat16)

    def cached():
        # the weight's fp8 bytes are already in hand; only the activation is quantised
        qA, sA = _q(A)
        sm(qA, sA, qWt, sWt, torch.bfloat16)
        qG, sG = _q(G)
        sm(qG, sG, qWn, sWn, torch.bfloat16)

    def scale_only():
        # today's site-1 behaviour applied to BOTH sites: scale cached, bytes re-cast
        qA, sA = _q(A)
        qWt2 = (Wt / sWt).to(torch.float8_e4m3fn)
        sm(qA, sA, qWt2, sWt, torch.bfloat16)
        qG, sG = _q(G)
        qWn2 = (Wn / sWn).to(torch.float8_e4m3fn)
        sm(qG, sG, qWn2, sWn, torch.bfloat16)

    ARMS = {"live": live, "cached": cached, "scale": scale_only}
    samples = {k: [] for k in ARMS}
    for _ in range(3):
        for k, fn in ARMS.items():
            samples[k].append(_time(fn, iters=ITERS // 3, warmup=WARMUP // 3))
    per_chunk = {k: statistics.median(v) for k, v in samples.items()}
    per_step = {k: v * CHUNKS for k, v in per_chunk.items()}

    print(f"\nhead weight quantisation, 2 of the 3 call sites, {CHUNKS} chunks/step")
    print(f"{'arm':8s} {'ms/chunk':>10s} {'ms/step':>10s}   spread")
    for k in ARMS:
        print(f"{k:8s} {per_chunk[k]:10.3f} {per_step[k]:10.1f}   {max(samples[k]) - min(samples[k]):.3f}")

    byte_cache = per_step["live"] - per_step["cached"]
    scale_half = per_step["live"] - per_step["scale"]
    print(f"\nbyte cache saves        {byte_cache:6.1f} ms/step  (live - cached)")
    print(f"  of which scale-only   {scale_half:6.1f} ms/step  (already done at site 1 today)")
    print(f"  the new part          {per_step['scale'] - per_step['cached']:6.1f} ms/step  (caching the BYTES)")
    print("\nfor scale: t58 measured the whole 3-site quant tax at 135.1 ms/step.")
    print(f"  this lever is {byte_cache / 135.1 * 100:.0f}% of it, and needs no kernel work.")

    out = {"probe": "t60_weight_cache", "shape": {"M": M, "K": K, "N": N, "chunks": CHUNKS},
           "statistic": "median of 3 interleaved blocks, cuda events",
           "ms_per_step": per_step, "byte_cache_saving_ms": byte_cache,
           "scale_only_saving_ms": scale_half,
           "bytes_beyond_scale_ms": per_step["scale"] - per_step["cached"],
           "t58_total_quant_tax_ms": 135.1}
    with open("/work/aupai/runs/t60_weight_cache.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote runs/t60_weight_cache.json")


if __name__ == "__main__":
    main()
