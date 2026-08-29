#!/usr/bin/env python3
"""Deeper comparison of two RL runs' weight deltas.

Run A: ckpt_sft_k4.pt -> ckpt_rl_k4_bf16.pt
Run B: ckpt_k4_11b_lr05.pt -> ckpt_rl_direct_bf16.pt

Per-component cosine similarity of delta directions, per-layer delta norms,
and geometry of where run B ended up.
"""

import torch
from collections import defaultdict


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model"]


def component(name):
    if name.startswith("blocks."):
        return "blocks." + name.split(".")[2]
    return name.split(".")[0]


def layer_of(name):
    if name.startswith("blocks."):
        return int(name.split(".")[1])
    return None


def main():
    base = load("ckpt_k4_11b_lr05.pt")
    sft = load("ckpt_sft_k4.pt")
    rl_a = load("ckpt_rl_k4_bf16.pt")
    rl_b = load("ckpt_rl_direct_bf16.pt")

    keys = [k for k in sft if k in rl_a and k in base and k in rl_b]
    dA = {k: (rl_a[k].float() - sft[k].float()) for k in keys}
    dB = {k: (rl_b[k].float() - base[k].float()) for k in keys}

    comps = sorted({component(k) for k in keys})

    print("=== per-component cosine of delta directions (A: sft->rl, B: base->rl_direct) ===")
    print(
        f"{'component':<16} {'cos(concat)':>12} {'mean per-tensor cos':>20} {'n_tensors':>10} {'||dA||/||wA||':>14} {'||dB||/||wB||':>14}"
    )
    for c in comps:
        ks = [k for k in keys if component(k) == c]
        va = torch.cat([dA[k].flatten() for k in ks])
        vb = torch.cat([dB[k].flatten() for k in ks])
        cos = torch.nn.functional.cosine_similarity(va, vb, dim=0).item()
        per = []
        for k in ks:
            a, b = dA[k].flatten(), dB[k].flatten()
            na, nb = a.norm(), b.norm()
            if na > 0 and nb > 0:
                per.append(torch.dot(a, b) / (na * nb))
        mcos = torch.stack(per).mean().item() if per else float("nan")
        wa = torch.cat([sft[k].float().flatten() for k in ks]).norm()
        wb = torch.cat([base[k].float().flatten() for k in ks]).norm()
        print(
            f"{c:<16} {cos:>12.4f} {mcos:>20.4f} {len(ks):>10} {va.norm() / wa:>14.4f} {vb.norm() / wb:>14.4f}"
        )

    # whole-model cosine
    va = torch.cat([dA[k].flatten() for k in keys])
    vb = torch.cat([dB[k].flatten() for k in keys])
    print(f"\nwhole-model cosine: {torch.nn.functional.cosine_similarity(va, vb, dim=0).item():.4f}")

    print("\n=== per-layer delta norm (all tensors in the layer, relative rms) ===")
    print(f"{'layer':>6} {'A ||d||/||w||':>16} {'B ||d||/||w||':>16} {'ratio B/A':>12}")
    layA = defaultdict(lambda: [0.0, 0.0])
    layB = defaultdict(lambda: [0.0, 0.0])
    for k in keys:
        L = layer_of(k)
        if L is None:
            continue
        layA[L][0] += dA[k].pow(2).sum().item()
        layA[L][1] += sft[k].float().pow(2).sum().item()
        layB[L][0] += dB[k].pow(2).sum().item()
        layB[L][1] += base[k].float().pow(2).sum().item()
    for L in sorted(layA):
        ra = layA[L][0] ** 0.5 / (layA[L][1] ** 0.5 + 1e-12)
        rb = layB[L][0] ** 0.5 / (layB[L][1] ** 0.5 + 1e-12)
        print(f"{L:>6} {ra:>16.5f} {rb:>16.5f} {rb / (ra + 1e-12):>12.2f}")

    # per-layer per-component breakdown (where in each layer did B move more)
    print("\n=== per-layer x component rms-delta (A / B) ===")
    print(
        f"{'layer':>6} {'mixer A/B':>18} {'ffn A/B':>18} {'ar1 A/B':>18} {'ar2 A/B':>18} {'n1 A/B':>14} {'n2 A/B':>14}"
    )
    cell = defaultdict(lambda: [0.0, 0.0])
    for k in keys:
        L = layer_of(k)
        if L is None:
            continue
        c = component(k)
        cell[(L, c)][0] += dA[k].pow(2).sum().item()
        cell[(L, c)][1] += dB[k].pow(2).sum().item()
    for L in range(12):
        row = [f"{L:>6}"]
        for c in ["blocks.mixer", "blocks.ffn", "blocks.ar1", "blocks.ar2", "blocks.n1", "blocks.n2"]:
            da, db = cell[(L, c)]
            row.append(f"{da**0.5:>8.2e}/{db**0.5:<8.2e}")
        print(" ".join(row))

    # geometry: where did B end up?
    print("\n=== endpoint geometry (relative rms distances) ===")

    def dist(x, y):
        s = sum((x[k].float() - y[k].float()).pow(2).sum().item() for k in keys)
        w = sum(y[k].float().pow(2).sum().item() for k in keys)
        return s**0.5 / (w**0.5 + 1e-12)

    print(f"||base - sft|| / ||sft||          = {dist(base, sft):.4f}  (SFT's own move)")
    print(f"||rl_direct - sft|| / ||sft||     = {dist(rl_b, sft):.4f}  (B endpoint vs SFT)")
    print(f"||rl_direct - base|| / ||base||   = {dist(rl_b, base):.4f}  (B's own move)")
    print(f"||rl_a - sft|| / ||sft||          = {dist(rl_a, sft):.4f}  (A's own move)")
    print(f"||rl_direct - rl_a|| / ||rl_a||   = {dist(rl_b, rl_a):.4f}  (B endpoint vs A endpoint)")

    # collapse signature: norm layers and ar params — absolute movement in B
    print("\n=== norm/ar absolute movement in B (collapse signature check) ===")
    for pat in ["n1", "n2", "norm", "ar1", "ar2", "final_ar"]:
        ks = [k for k in keys if pat in k]
        da = sum(dA[k].abs().max().item() for k in ks) / len(ks)
        db = sum(dB[k].abs().max().item() for k in ks) / len(ks)
        print(f"  {pat:<10} mean|max delta|  A={da:.2e}  B={db:.2e}")

    # per-tensor cosine distribution for the big movers (ar q vectors)
    print("\n=== per-tensor cosine for top-moving tensors ===")
    movers = sorted(keys, key=lambda k: -(dA[k].pow(2).sum() + dB[k].pow(2).sum()))[:15]
    for k in movers:
        a, b = dA[k].flatten(), dB[k].flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        print(f"  {k:<24} cos={cos:>7.4f}  ||dA||={a.norm():.2e}  ||dB||={b.norm():.2e}")


if __name__ == "__main__":
    main()
