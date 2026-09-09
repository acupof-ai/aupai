#!/usr/bin/env python3
# 3b hit dump (2026-09-09): dump (doc, holdout, containment) triples from a few
# shards to adjudicate whether the 11,744 containment hits are real contamination.
import json, os, sys
sys.path.insert(0, "/work/aupai/datagen")
import numpy as np
from scan_code_contamination import load_holdouts, HoldoutIndex, bigrams

SHARDS = [
    "/work/aupai/data/corpus/code_rp1t_dd09/code_rp1t_000.jsonl",
    "/work/aupai/data/corpus/code_rp1t_b2v2_dd/code_rp1t_b2v2_dd_000.jsonl",
    "/work/aupai/data/corpus/code_dedup08/code_py_starcoder_120.jsonl",  # the exact-hit shard
]
TH = 0.5

holdouts = load_holdouts()
idx = HoldoutIndex(holdouts)
print(f"holdouts: {len(holdouts)} ({len(idx.long_cols)} long, {len(idx.short)} short)")

for sp in SHARDS:
    docs = []
    for line in open(sp, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            docs.append(json.loads(line).get("content", ""))
        except Exception:
            docs.append("")
    # IDF from this shard alone (few shards, ~100K docs -- stable enough for the
    # 16K bigrams present; the full-corpus IDF was used for the verdict)
    df = np.zeros(len(idx.g2i), dtype=np.float64)
    for d in docs:
        for g in set(bigrams(d)):
            j = idx.g2i.get(g)
            if j is not None:
                df[j] += 1
    idx.set_idf(df, len(docs))

    BATCH = 2048
    n_hits = 0
    print(f"\n===== {os.path.basename(sp)} ({len(docs)} docs) =====")
    for i0 in range(0, len(docs), BATCH):
        texts = docs[i0:i0 + BATCH]
        exact, mc, nf, hr, hidx = idx.scan_chunk(texts, TH)
        for row, t in exact[:3]:
            print(f"  EXACT row {i0+row}: {t[:150]}")
        for r in hidx:
            # find the top holdout for this row
            R = idx._matrix([texts[r]])
            if R is None:
                continue
            cont = np.asarray((R @ idx.Hw_n).todense()).ravel()
            # long_cols only: the dump adjudicates the verdict scan's 11,744,
            # which counted long-holdout hits (scan hit_rows maxes long_cols)
            long = cont[idx.long_cols]
            top_local = int(long.argmax())
            if long[top_local] < TH:
                continue
            n_hits += 1
            if n_hits <= 12:
                print(f"  hit row {i0+r}: cont={long[top_local]:.3f} holdout={idx.long_cols[top_local]} "
                      f"doc={texts[r][:120]!r}")
                print(f"    holdout_text: {holdouts[idx.long_cols[top_local]][:120]!r}")
    print(f"  total hit rows: {n_hits}")
