#!/usr/bin/env python3
"""Compare checkpoints: what each training stage actually changed.

    python scripts/ckpt_diff.py ckpt_k4_11b_lr05.pt ckpt_sft_k4.pt ckpt_rl_k4_bf16.pt

Per checkpoint: config flags and param count. Per consecutive pair: per-component
relative RMS delta (||b-a|| / ||a||) and the most-shifted individual tensors.
"""

import sys
from collections import defaultdict

import torch


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model"], ck.get("cfg", {})


def component(name):
    # blocks.3.mixer.q.weight -> blocks.mixer; blocks.0.ar1.g -> blocks.ar1; tok.weight -> tok
    if name.startswith("blocks."):
        return "blocks." + name.split(".")[2]
    return name.split(".")[0]


def main(paths):
    cks = [load(p) for p in paths]
    for p, (sd, cfg) in zip(paths, cks, strict=True):
        n = sum(t.numel() for t in sd.values())
        print(
            f"{p}: {len(sd)} tensors, {n / 1e6:.1f}M params, "
            f"attn_res={cfg.get('attn_res')} blocks={cfg.get('attn_res_blocks')} "
            f"batch={cfg.get('batch')} seq={cfg.get('seq')}"
        )
    print()

    for i in range(len(cks) - 1):
        p1, p2 = paths[i], paths[i + 1]
        (sd1, cfg1), (sd2, cfg2) = cks[i], cks[i + 1]
        print(f"== {p1} -> {p2}")
        diff_cfg = {
            k: (cfg1.get(k), cfg2.get(k)) for k in set(cfg1) | set(cfg2) if cfg1.get(k) != cfg2.get(k)
        }
        if diff_cfg:
            print("  cfg changes: " + ", ".join(f"{k}: {a}->{b}" for k, (a, b) in sorted(diff_cfg.items())))
        agg = defaultdict(lambda: [0.0, 0.0])
        per = []
        for k in sd1:
            if k not in sd2:
                continue
            a, b = sd1[k].float(), sd2[k].float()
            agg[component(k)][0] += (b - a).pow(2).sum().item()
            agg[component(k)][1] += a.pow(2).sum().item()
            rel = ((b - a).pow(2).mean().sqrt() / (a.pow(2).mean().sqrt() + 1e-12)).item()
            per.append((rel, k, tuple(a.shape)))
        for c, (ds, ws) in sorted(agg.items()):
            print(f"  {c:<16} rms-delta {ds**0.5 / (ws**0.5 + 1e-12):.4f}")
        print("  top shifted tensors:")
        for rel, k, shape in sorted(per, reverse=True)[:8]:
            print(f"    {rel:.4f}  {k} {shape}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
