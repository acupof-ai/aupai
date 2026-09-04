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
sys.path.insert(0, os.path.join(ROOT, "eval"))

from scripts.loader import EOS_ID, load_checkpoint, load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--mix", default=os.path.join(ROOT, "data", "mix_scale_3.24b.json"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rows", type=int, default=512, help="val rows scored per domain")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cu_path", choices=["cu_none", "doc_cu"], default="cu_none",
                    help="doc_cu passes the document mask; cu_none is what every published "
                         "ppl figure was taken with (audit_0904 E10). train._domain_seqs "
                         "returns PACKED rows, so the two differ.")
    a = ap.parse_args()

    import json

    import train
    from train import doc_cu_seqlens
    from cache_guard import guard

    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    tok = load_tokenizer(a.tokenizer, cfg)
    for p in model.parameters():
        p.data = p.data.contiguous()
    # _domain_seqs reads Cfg for seq/fone/num_id, so the checkpoint's own config drives it.
    for k, v in vars(cfg).items():
        if not k.startswith("_"):
            setattr(train.Cfg, k, v)

    mix = json.load(open(a.mix, encoding="utf-8"))
    # Before the first _domain_seqs call, not inside the loop: this ran on card 7 against
    # the live 20B run's nine caches and printed "cache was built by another vocabulary,
    # retokenizing" two minutes in (fb killed it by exact PID, 2026-09-02). train.VOCAB_ID
    # is set only by train.build_tokenizer; load_checkpoint never touches it, so it was
    # None, every stamp read as a mismatch, and the rebuild would have re-stamped nine
    # training caches with an empty vocabulary. Cfg is set above first because the guard's
    # cache path depends on Cfg.fone.
    guard(cfg, list(mix["domains"]))
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
                # cu REACHES THE FORWARD. train._domain_seqs packs multiple documents per row,
                # so without the mask attention reads across the boundaries inside a row while
                # training used doc_cu_seqlens (E10: this file's docstring claimed it "rebuilds
                # exactly the rows train.py held out" while scoring them on a different path).
                cu = doc_cu_seqlens(xb, EOS_ID) if a.cu_path == "doc_cu" else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, _ = model(xb, cu=cu) if cu is not None else model(xb)
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
