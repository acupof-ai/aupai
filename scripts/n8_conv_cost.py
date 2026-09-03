#!/usr/bin/env python3
"""What does document-isolating the short_conv cost in throughput?

THE DECISION THIS FEEDS. eff.kda_document_isolation_violated located the isolation break at
model.py:109-113: the depthwise k=4 causal conv is applied to the whole row and never sees cu, so
positions 0-2 of q/k/v in every document after the first are convolved with the previous document's
last tokens. 6e's fix is to call fla's ShortConvolution instead, which takes cu_seqlens and honours
it (verified on the pod: fla 0.5.2, cu_seqlens reaches the kernel at two sites inside forward).

WHY THE COST IS NOT OBVIOUS, and why the ruling should not be executed without this number.
model.py:104-108 chose the current form deliberately:

    "K shifted multiply-adds, not nn.Conv1d: ATen routes a depthwise k=4 conv to
     conv_depthwise2d_generic at ~6% of bandwidth; inductor fuses the arithmetic form
     (3.44x compiled, the training path). Eager is 0.61x -- this only wins under torch.compile."

The 3.44x comes from inductor FUSING the multiply-adds into the surrounding graph. ShortConvolution's
default backend is a Triton kernel, which inductor treats as opaque -- it cannot fuse across it. This
model has 9 KDA layers of 12, so whatever the per-layer cost is, it is paid nine times per forward on
every future training run, to remove a 3-position-per-document error whose loss cost is unmeasured.

THREE FORMS MEASURED, all under torch.compile since that is the training path:
  current      the multiply-add form at model.py:109-113, row-padded, LEAKS across documents
  shortconv    fla ShortConvolution with cu_seqlens, isolates, opaque to inductor
  masked       the multiply-add form with the left pad ZEROED at document starts -- isolates, and
               stays fusible because it is still pure arithmetic. This is the third option the
               throughput number may make the right one, so it is measured before being proposed
               rather than after.

The masked form is the same algebra as the current one with a [1, 1, T] multiplier on h per tap: tap i
contributes only where the source position is inside the same document. Cheap to state, and if it is
within noise of `current` while isolating, the fix costs nothing.

NOT A CORRECTNESS TEST. Isolation of the masked form is asserted here on a small case so a wrong
formula cannot be reported as a fast one, but the real gate is scripts/n7c_pack_isolation.py on a
fresh init, per 6e's ruling. This script answers "what does it cost", nothing else.

USAGE
    CUDA_VISIBLE_DEVICES=4 python3 scripts/n8_conv_cost.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

D, K = 1024, 4
B, T = 16, 4096      # the real training shape (train.py: batch 16, seq 4096)
DOC = 512            # documents of 512 tokens: 8 per row, close to the real pack's document count
LAYERS = 9           # KDA layers in the 12-block model, so the per-forward multiplier
ITERS = 50


def main():
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    dev = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(0)

    conv = torch.nn.Conv1d(D, D, kernel_size=K, padding=0, groups=D).to(dev).to(dt)
    w, bias = conv.weight, conv.bias
    x = torch.randn(1, B * T, D, device=dev, dtype=dt)          # flattened varlen layout
    cu = torch.arange(0, B * T + 1, DOC, dtype=torch.int32, device=dev)

    def current(x):
        """model.py:109-113 verbatim: row-padded, cu never consulted."""
        Bx, Tx, _ = x.shape
        h = F.pad(x.transpose(1, 2), (K - 1, 0))
        y = h[:, :, :Tx] * w[:, 0, 0].unsqueeze(-1)
        for i in range(1, K):
            y = y + h[:, :, i : i + Tx] * w[:, 0, i].unsqueeze(-1)
        return F.silu((y + bias.unsqueeze(-1)).transpose(1, 2))

    # THE TAP MASKS, built once from cu. Tap i at output position t reads input position t-(K-1-i);
    # that read is legal only when it lands in the same document, i.e. when t - (K-1-i) >= the
    # document's start. Precomputed as [1, 1, T] multipliers so the hot path stays pure arithmetic
    # with no gather and no branch -- which is the whole point of keeping inductor able to fuse it.
    pos = torch.arange(B * T, device=dev)
    seg_start = torch.bucketize(pos, cu[1:], right=True)        # which document each position is in
    doc_start = cu[:-1].to(torch.long)[seg_start]               # that document's first position
    taps = [((pos - (K - 1 - i)) >= doc_start).to(dt).view(1, 1, -1) for i in range(K)]

    def masked(x):
        """The same algebra with each tap zeroed where it would cross a document boundary."""
        Bx, Tx, _ = x.shape
        h = F.pad(x.transpose(1, 2), (K - 1, 0))
        y = h[:, :, :Tx] * w[:, 0, 0].unsqueeze(-1) * taps[0]
        for i in range(1, K):
            y = y + h[:, :, i : i + Tx] * w[:, 0, i].unsqueeze(-1) * taps[i]
        return F.silu((y + bias.unsqueeze(-1)).transpose(1, 2))

    from fla.modules.convolution import ShortConvolution  # noqa: PLC0415
    sc = ShortConvolution(D, K, bias=True, activation="silu").to(dev).to(dt)
    with torch.no_grad():  # same weights, so this is a speed comparison and not a shape comparison
        sc.weight.copy_(w)
        if sc.bias is not None:
            sc.bias.copy_(bias)

    def shortconv(x):
        out, _ = sc(x, cu_seqlens=cu.to(torch.long))
        return out

    # CORRECTNESS FIRST, on a small case: a fast wrong formula is worse than a slow right one, and
    # reporting a throughput number for a form that does not isolate would be exactly that.
    print("== isolation check on 2 documents of 8 tokens (correctness before speed)")
    small_cu = torch.tensor([0, 8, 16], dtype=torch.int32, device=dev)
    xs = torch.randn(1, 16, D, device=dev, dtype=dt)
    sp = torch.arange(16, device=dev)
    ss = torch.bucketize(sp, small_cu[1:], right=True)
    ds = small_cu[:-1].to(torch.long)[ss]
    small_taps = [((sp - (K - 1 - i)) >= ds).to(dt).view(1, 1, -1) for i in range(K)]
    saved, taps[:] = list(taps), small_taps
    with torch.no_grad():
        packed_m, packed_c = masked(xs), current(xs)
        solo_m = masked(xs[:, 8:])
        sc_packed, _ = sc(xs, cu_seqlens=small_cu.to(torch.long))
        sc_solo, _ = sc(xs[:, 8:])
    taps[:] = saved
    for name, a, b in (("current  ", packed_c[0, 8:], masked(xs[:, 8:])[0] if False else None),
                       ("masked   ", packed_m[0, 8:], solo_m[0]),
                       ("shortconv", sc_packed[0, 8:], sc_solo[0])):
        if b is None:  # `current` is compared against the correct form, not against itself
            b = solo_m[0]
        d = (a.float() - b.float()).abs().max().item()
        print(f"  {name}: document 1 packed vs alone, max|diff| {d:.6f}   "
              f"{'ISOLATES' if d < 1e-3 else 'LEAKS'}")
    print("  `current` is expected to LEAK -- that is the defect being priced.")

    # TIMING. torch.compile because that is the training path, and the whole question is whether
    # inductor can still fuse the form. Each variant compiled separately, warmed up, then CUDA-event
    # timed -- wall clock around an async launch measures the launch, not the kernel.
    print(f"\n== throughput under torch.compile, B*T={B * T} d={D} K={K}, {ITERS} iters")
    print(f"  {'form':12s} {'ms/call':>9s} {'vs current':>11s} {'x9 layers ms':>13s}   isolation")
    base = None
    for name, fn, isolates in (("current", current, "LEAKS"),
                               ("masked", masked, "isolates"),
                               ("shortconv", shortconv, "isolates")):
        c = torch.compile(fn, dynamic=False)
        with torch.no_grad():
            for _ in range(5):
                c(x)
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(ITERS):
                c(x)
            ev1.record()
            torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / ITERS
        base = base or ms
        print(f"  {name:12s} {ms:9.3f} {ms / base:10.2f}x {ms * LAYERS:13.2f}   {isolates}")
    print("  The x9 column is what a forward pays, since 9 of this model's 12 blocks are KDA. It is\n"
          "  not the whole step: attention, FFN and the head are unchanged, so the fraction of a\n"
          "  training step is smaller -- take this as the conv-side delta, not as a tokens/s claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
