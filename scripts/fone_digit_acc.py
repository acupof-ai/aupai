#!/usr/bin/env python3
"""Is the FoNE digit head actually learning? Per-digit accuracy at [NUM] positions.

The training log adds the digit cross-entropy into the same scalar as the token loss,
so a falling loss does not say whether the number path works at all. This reads a
checkpoint and scores the digit head directly against held-out rows from a cached
domain, next to the two baselines that matter: 10% is chance, and the majority digit
(0, because place-value padding is mostly zeros) is what a head that learned nothing
useful would converge to.

    python scripts/fone_digit_acc.py --ckpt ckpt_k6_fone.pt.step1000 [--domain math] [--rows 256]
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fone  # noqa: E402
import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain", default="math")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = type("C", (), ck["cfg"])
    assert getattr(cfg, "fone", False), "checkpoint was not trained with --fone"
    model = train.HybridLM(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(a.device).eval()
    if next(model.parameters()).dtype == torch.float32 and a.device.startswith("cuda"):
        model = model.to(torch.bfloat16)

    # _domain_cache_path, not a hand-built name: the FoNE cache is tokens_<domain>_fone.pt
    # and this read tokens_<domain>.pt, a bare 1-D id tensor, so `ids, vals = ...` unpacked a
    # million-element tensor ("too many values to unpack"). train.py:1041 documents this exact
    # failure and the fix is to go through the helper that owns the name.
    train.Cfg.fone = True
    cache = train._domain_cache_path(a.domain)
    ids, vals = torch.load(cache, map_location="cpu", weights_only=True)
    # The LAST rows of the cache: training consumes it from the front, so the tail is
    # the least-seen part of a domain that is only ~5 epochs deep.
    n = len(ids) // (cfg.seq + 1)
    rows = ids[: n * (cfg.seq + 1)].view(-1, cfg.seq + 1)[-a.rows :]
    dense = train.scatter_values(ids[: n * (cfg.seq + 1)].view(-1, cfg.seq + 1), vals, cfg.num_id)[-a.rows :]

    hit = tot = 0
    exact_hit = exact_tot = 0
    zeros = all_zero = 0
    copy_hit = copy_tot = 0
    with torch.no_grad():
        for j in range(0, len(rows), a.batch):
            x = rows[j : j + a.batch].long().to(a.device)
            v = dense[j : j + a.batch].to(a.device)
            xb, yb, vb, wb = x[:, :-1], x[:, 1:], v[:, :-1], v[:, 1:]
            with torch.autocast("cuda", torch.bfloat16, enabled=a.device.startswith("cuda")):
                h, _ = model(xb, yb, None, vb)
            nm = yb == cfg.num_id
            if not nm.any():
                continue
            pred = model.num_logits(h[nm].float()).argmax(-1)  # (N, digits)
            tgt = fone.digit_targets(wb[nm])
            hit += int((pred == tgt).sum())
            tot += tgt.numel()
            exact_hit += int((pred == tgt).all(-1).sum())
            exact_tot += len(tgt)
            zeros += int((tgt == 0).sum())
            all_zero += int((tgt == 0).all(-1).sum())
            # Copying baseline: the previous number in the same row. Math text restates
            # numbers constantly, so a head that only learned "repeat the last one" would
            # already score here -- the exact rate is only evidence above THIS line.
            for r in range(len(x)):
                vals_r = wb[r][nm[r]]
                if len(vals_r) > 1:
                    copy_hit += int((vals_r[1:] == vals_r[:-1]).sum())
                    copy_tot += len(vals_r) - 1

    assert tot, f"no [NUM] targets in the last {a.rows} rows of {a.domain}"
    print(f"{a.ckpt} on {a.domain} ({exact_tot} numbers, {tot} digits)")
    print(f"  per-digit accuracy  {100 * hit / tot:5.1f}%")
    print(f"  whole-number exact  {100 * exact_hit / exact_tot:5.1f}%")
    print(f"  baselines: per-digit chance 10.0%, always-0 {100 * zeros / tot:5.1f}%")
    print(f"             whole-number always-0 {100 * all_zero / exact_tot:5.1f}%", end="")
    print(f", copy-previous {100 * copy_hit / max(copy_tot, 1):5.1f}%")


if __name__ == "__main__":
    main()
