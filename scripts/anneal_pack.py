#!/usr/bin/env python3
# restartable: writes one pack file; rerun overwrites it.
"""Continued-pretraining pack from the gate caches, rows the 30B run never read.

    python3 scripts/anneal_pack.py --cursor_ckpt ckpt_v41_ced_0926.pt --tokens 1.6e9 \
        --out data/sft/anneal2/anneal2.pt

train.build_mix is called with the finished run's row_cursor, so every row comes after the
cursor in the run's own shuffle: unseen documents of the same decontaminated caches. Rows are
fully supervised (labels == input_ids), in the sft_math pack format, so sft_math.py can train
them from avg3 weights with a fresh optimizer and a linear-to-zero LR.
"""
import argparse
import json
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import train  # noqa: E402

WEIGHTS = {"code_ultra_l3_noexec_dc": 0.70, "code_py_starcoder_dc": 0.20, "math_owm_stage2_dc": 0.10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cursor_ckpt", required=True)
    ap.add_argument("--tokens", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open("data/mix_v41_gate.json") as g:
        gate = json.load(g)
    ck = torch.load(args.cursor_ckpt, map_location="cpu", weights_only=False, mmap=True)
    cursor = {n: int(v) for n, v in ck["row_cursor"].items() if n in WEIGHTS}
    # build_mix reads total_tokens as a total over cursor + this plan and allocates the remainder.
    total = args.tokens + sum(cursor.values()) * train.Cfg.seq
    mix = {"total_tokens": total, "anneal_frac": 0.0,
           "domains": {n: {**gate["domains"][n], "weight": w, "anneal": w} for n, w in WEIGHTS.items()}}
    print(f"cursor from {args.cursor_ckpt}: {cursor} seed {ck.get('row_cursor_seed')}", flush=True)
    train.Cfg.anneal_frac = 0.0  # one phase; build_mix refuses a mix/Cfg disagreement
    tok = train.build_tokenizer(None)
    assert train.VOCAB_ID, "build_tokenizer did not set VOCAB_ID"
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir="runs", delete=False) as f:
        json.dump(mix, f)
    try:
        out, _ = train.build_mix(f.name, tok, True, False, row_cursor=cursor,
                                 cursor_srcfp=ck.get("row_cursor_srcfp"), cursor_seed=ck.get("row_cursor_seed"))
    finally:
        os.unlink(f.name)
    got = {n: train.Cfg._row_cursor[n] - cursor.get(n, 0) for n in WEIGHTS}
    assert all(v > 0 for v in got.values()), f"a domain contributed no unseen rows: {got}"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"input_ids": out, "labels": out, "vocab_id": train.VOCAB_ID,
                "build_stats": {"rows_per_domain": got, "cursor_from": args.cursor_ckpt, "weights": WEIGHTS}},
               args.out)
    print(f"wrote {args.out}: {out.shape[0]} rows = {out.numel() / 1e9:.2f}B tokens, {got}", flush=True)


if __name__ == "__main__":
    main()
