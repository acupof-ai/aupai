"""GPU parity, memory and speed of the v42 packed-row attention (v41f/docpack.py).

One card, run after the stage-2 window frees it:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/v42_attn_gpu_probe.py            # layers 2 and 12
  CUDA_VISIBLE_DEVICES=0 python3 scripts/v42_attn_gpu_probe.py --full     # + whole 24-layer trunk

Prints, per probed layer at the v42_s24 shape: bf16 ref and bf16 chunked error against the
chunked path in fp32 (B1 T1024), the
bytes a layer keeps for backward, peak GiB of one fwd+bwd, and median ms. --full adds the
trunk's fwd+bwd peak at --B x --T (params bf16, no optimizer state).
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v41f.attention import Attention, SharedAttnState  # noqa: E402
from v41f.config import v42_s24  # noqa: E402
from v41f.docpack import doc_layout  # noqa: E402
from v41f.model import V41FModel  # noqa: E402

GIB = 2**30
DEV = "cuda"
PT = 1024  # parity length


def build(make):
    """Construct under bf16 default dtype on DEV, as the trainer does: the m>1 compressor
    keeps its explicit fp32 projections, which a blanket .to(bf16) would break."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(DEV):
            return make()
    finally:
        torch.set_default_dtype(prev)


def packed_cu(b, t, g, mean_doc=1500):
    """cu_seqlens over b rows of t with random document lengths around mean_doc."""
    cuts = [0]
    for r in range(b):
        p = r * t
        while True:
            p += int(torch.randint(max(1, mean_doc // 8), 2 * mean_doc, (1,), generator=g))
            if p >= (r + 1) * t:
                break
            cuts.append(p)
        cuts.append((r + 1) * t)
    return torch.tensor(cuts, dtype=torch.int32, device=DEV)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def mem(kind):
    """cuda allocator reading; 0 on the cpu smoke path."""
    if DEV != "cuda":
        return 0
    if kind == "reset":
        return torch.cuda.reset_peak_memory_stats()
    return torch.cuda.max_memory_allocated() if kind == "peak" else torch.cuda.memory_allocated()


def layer_state(x, cu):
    """Fresh packed-row state; the probed layers (2, 12) are kv+index sources, so they need
    no owner layer run before them."""
    st = SharedAttnState()
    st.doc, st.pos, st.doclen = doc_layout(cu, x.size(0), x.size(1), x.device)
    return st


def run_layer(attn, x, cu, cfg, layer_id):
    st = layer_state(x, cu)
    y, st = attn(x, st)
    loss = y.float().square().mean()
    if st.indexer_kl:
        t, c = st.indexer_kl[0]
        loss = loss + t / c.clamp_min(1)
    return y, loss


def parity(layer_id):
    """Relative error of the bf16 ref and bf16 chunked paths against the chunked path run in
    fp32 (same weights, same input), for the output and for dL/dx."""
    out = {}
    for impl, dt in (("ref", torch.bfloat16), ("chunked", torch.bfloat16), ("chunked", torch.float32)):
        cfg = v42_s24(attn_impl=impl, indexer_train_mode="kl" if impl == "chunked" else "off")
        torch.manual_seed(0)
        attn = build(lambda: Attention(cfg, layer_id))
        if dt == torch.float32:
            attn = attn.float()
        torch.manual_seed(1)
        x = torch.randn(1, PT, cfg.dim, device=DEV).to(dt).requires_grad_()
        cu = packed_cu(1, PT, torch.Generator().manual_seed(2), mean_doc=PT // 3)
        y, _ = run_layer(attn, x, cu, cfg, layer_id)
        y.float().square().mean().backward()
        out[(impl, dt)] = (y.detach().float(), x.grad.float())
    yt, gt = out[("chunked", torch.float32)]

    def rel(key):
        y, g = out[key]
        return ((y - yt).norm() / yt.norm()).item(), ((g - gt).norm() / gt.norm()).item()
    return rel(("ref", torch.bfloat16)), rel(("chunked", torch.bfloat16))


def bench(layer_id, B, T, g, iters=5):
    cfg = v42_s24()
    attn = build(lambda: Attention(cfg, layer_id))
    x = torch.randn(B, T, cfg.dim, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    cu = packed_cu(B, T, g)
    times = []
    for i in range(iters + 2):
        sync()
        mem("reset")
        base = mem("now")
        t0 = time.perf_counter()
        y, loss = run_layer(attn, x, cu, cfg, layer_id)
        kept = mem("now") - base
        loss.backward()
        sync()
        if i >= 2:
            times.append((time.perf_counter() - t0) * 1e3)
        peak = mem("peak") - base
        x.grad = None
        del y, loss
    return kept / GIB, peak / GIB, statistics.median(times)


def full(B, T, g):
    cfg = v42_s24()
    m = build(lambda: V41FModel(cfg, max_batch_size=B))
    ids = torch.randint(2, cfg.vocab_size, (B, T), device=DEV)
    cu = packed_cu(B, T, g)
    sync()
    mem("reset")
    t0 = time.perf_counter()
    h, kl = m(ids, cu=cu, return_hidden=True)
    (h.float().square().mean() + kl).backward()
    sync()
    return mem("peak") / GIB, (time.perf_counter() - t0) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--layers", default="2,12")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--device", default="cuda", help="cpu: tiny smoke of the script itself")
    a = ap.parse_args()
    global DEV, PT
    DEV = a.device
    PT = min(PT, a.T)
    g = torch.Generator().manual_seed(0)
    for lid in (int(v) for v in a.layers.split(",")):
        (ry, rg), (cy, cg) = parity(lid)
        kept, peak, ms = bench(lid, a.B, a.T, g)
        print(f"layer {lid}: vs fp32, rel y/dx: ref-bf16 {ry:.1e}/{rg:.1e} chunked-bf16 {cy:.1e}/{cg:.1e}"
              f" | B{a.B} T{a.T}: kept-for-bwd "
              f"{kept:.2f} GiB, fwd+bwd peak {peak:.2f} GiB, {ms:.1f} ms")
    if a.full:
        peak, ms = full(a.B, a.T, g)
        print(f"trunk 24L B{a.B} T{a.T}: fwd+bwd peak {peak:.2f} GiB (params bf16, no optimizer), {ms:.0f} ms")


if __name__ == "__main__":
    main()
