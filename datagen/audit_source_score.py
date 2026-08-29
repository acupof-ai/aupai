#!/usr/bin/env python3
"""Check whether a published corpus's own quality score is usable as a filter.

Scores the same documents with our own distilled head and reports Spearman
correlation plus our mean score per band of theirs. A published score passes
only if our score rises monotonically across its bands -- published score
columns have been non-monotonic with quality, so never cut on one unmeasured.

    python datagen/audit_source_score.py --parquet '/work/newdata/cosmo/*.parquet' \\
        --head data/quality_head.pt --ckpt ckpt_k5_clean_0827.pt \\
        --tokenizer data/tokenizer_k5.json
"""

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))


def spearman(a, b):
    import numpy as np

    def rank(x):
        o = np.argsort(x)
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x))
        return r

    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = (ra**2).sum() ** 0.5 * (rb**2).sum() ** 0.5
    return float((ra * rb).sum() / d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--score_col", default="score")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--head", default="data/quality_head.pt")
    ap.add_argument("--ckpt", default="ckpt_k5_clean_0827.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer_k5.json")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import random

    import numpy as np
    import pyarrow.parquet as pq
    import torch

    from loader import load_checkpoint, load_tokenizer
    from train_quality_head import _encode, _features

    files = sorted(glob.glob(a.parquet))
    per = max(1, a.n // len(files) + 1)
    rng = random.Random(0)
    texts, theirs = [], []
    for f in files:
        t = pq.ParquetFile(f).read(columns=[a.text_col, a.score_col]).to_pydict()
        idx = rng.sample(range(len(t[a.text_col])), min(per, len(t[a.text_col])))
        for i in idx:
            texts.append(t[a.text_col][i] or "")
            theirs.append(float(t[a.score_col][i]))
    texts, theirs = texts[: a.n], np.array(theirs[: a.n])

    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    for p in model.parameters():
        p.data = p.data.contiguous()
    tok = load_tokenizer(a.tokenizer, cfg)
    w = torch.load(a.head, map_location="cpu", weights_only=True)
    ours = (_features(model, _encode(texts, tok), a.device) @ w["w"] + w["b"]).numpy()

    print(f"{len(texts)} documents from {len(files)} files")
    print(f"  their score : min {theirs.min():.3f} median {np.median(theirs):.3f} max {theirs.max():.3f}")
    print(f"  our score   : min {ours.min():.2f} median {np.median(ours):.2f} max {ours.max():.2f}")
    print(f"  Spearman rank correlation {spearman(theirs, ours):+.3f}")
    print("\n  their band            n   our mean score")
    edges = np.quantile(theirs, [0, 0.25, 0.5, 0.75, 1.0])
    means = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (theirs >= lo) & (theirs <= hi)
        if m.sum():
            means.append(ours[m].mean())
            print(f"  {lo:.3f}-{hi:.3f}  {m.sum():>6}   {ours[m].mean():+.3f}")
    mono = all(x < y for x, y in zip(means[:-1], means[1:], strict=True))
    print(
        f"\n  monotonic across their bands: {mono}\n"
        f"  {'their score is usable as a filter threshold' if mono else 'their score is NOT usable as a threshold -- rank with ours instead'}"
    )


if __name__ == "__main__":
    main()
