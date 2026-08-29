#!/usr/bin/env python3
"""Held-out perplexity, reported PER DOMAIN.

Rebuilds exactly the rows train.py holds out (same caches, same val_frac, same
val_rows_max) and scores each domain on its own; the summary is an unweighted
mean across domains, because a row-weighted blend is dominated by the largest
domain and cannot show a small one moving.

    python eval/ppl.py --ckpt ckpt_k5_clean_0827.pt --tokenizer data/tokenizer_k5.json
"""

import argparse
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--mix", default=os.path.join(ROOT, "data", "mix_v3.json"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rows", type=int, default=512, help="val rows scored per domain")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import json

    import train

    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    tok = load_tokenizer(a.tokenizer, cfg)
    for p in model.parameters():
        p.data = p.data.contiguous()
    # _domain_seqs reads Cfg for seq/fone/num_id, so the checkpoint's own config drives it.
    for k, v in vars(cfg).items():
        if not k.startswith("_"):
            setattr(train.Cfg, k, v)

    mix = json.load(open(a.mix, encoding="utf-8"))
    out = {}
    for name in mix["domains"]:
        seqs = train._domain_seqs(name, tok, True, False)
        seqs = seqs[0] if train.Cfg.fone else seqs
        n_val = min(max(1, int(len(seqs) * train.Cfg.val_frac)), train.Cfg.val_rows_max)
        rows = seqs[:n_val][: a.rows].long()
        if not len(rows):
            continue
        X, Y = rows[:, :-1], rows[:, 1:]
        tot = ntok = 0.0
        with torch.no_grad():
            for i in range(0, len(X), a.batch):
                xb, yb = X[i : i + a.batch].to(a.device), Y[i : i + a.batch].to(a.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, _ = model(xb)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]).float(), yb.reshape(-1)
                    )
                n = yb.numel()
                tot += loss.item() * n
                ntok += n
        out[name] = tot / ntok
        print(
            f"  {name:<6} loss {out[name]:.4f}  ppl {math.exp(out[name]):7.2f}  ({len(X)} rows)", flush=True
        )

    if out:
        # Unweighted mean across domains, not the row-weighted blend train.py prints:
        # a blend hides a small domain moving.
        m = sum(out.values()) / len(out)
        print(f"  {'MEAN':<6} loss {m:.4f}  ppl {math.exp(m):7.2f}  (unweighted across domains)")


if __name__ == "__main__":
    main()
