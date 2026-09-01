#!/usr/bin/env python3
"""P0 corpus-duplication measurement (fb 2026-09-01). For each domain dir the run read
(the stamped shards, fingerprints carried by data/mix_30b_stage2.json), report:
  1. exact duplicate rate, document-level and paragraph-level
  2. near-duplicate rate via the near-dedup post-pass's MinHash path at the FUNCTION'S
     OWN DEFAULT (bands=64 rows=2 perms=128) -- config reported beside the number
  3. effective-unique tokens per domain vs tokens consumed (the multi-epoch multiplier)

Numbers only, config beside every rate. No verdict. Imports build_corpus primitives so
the exact/near measures are the SAME functions the dedup path uses, not a reimplementation.

Near-dup uses Fork C's two-pass to stay memory-bounded: Pass A stores only each exact-
unique doc's packed signature (1KB) + its source location; Pass B builds the LSH band
tables; Pass C re-reads ONLY the candidate docs' source lines to compute shingles and
run exact-J on candidate pairs (streaming, never materializing the pair set - the N7
shape). Memory is bounded by signatures + the candidate doc set, not the pair count."""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import build_corpus as B  # noqa: E402

PERMS, BANDS, ROWS = 128, 64, 2  # the near-dedup post-pass DEFAULT; reported with the number
JACCARD = 0.5
PARA_SPLIT = re.compile(r"\n\s*\n|\r\n\s*\r\n")


def _read_rows(shard):
    for line in open(shard, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            t = B.SPECIAL_TOKEN.sub("", d.get("content") or "").strip()
            if t:
                yield t


def measure(domain_dir, tk):
    shards = sorted(
        p for p in glob_shards(domain_dir)
        if os.path.basename(p).endswith(".jsonl") and not os.path.basename(p).startswith("holdout_slice_")
    )
    exact = set()
    para_exact = set()
    para_total = 0
    n = 0
    unique_tokens = 0
    sigs = {}       # exact-unique-doc ordinal -> packed MinHash signature (bytes)
    src = {}        # ordinal -> (shard_path, line_row) for candidate re-read
    ab, mask = B._near_coeffs(PERMS, 17)
    ordinal = 0
    for shard in shards:
        row = 0
        for t in _read_rows(shard):
            row_pos = row  # index among this shard's yielded rows (for candidate re-read)
            row += 1
            n += 1
            for p in (p.strip() for p in PARA_SPLIT.split(t) if p.strip()):
                para_exact.add(B.exact_key(p))
                para_total += 1
            if B.exact_key(t) in exact:
                continue
            exact.add(B.exact_key(t))
            unique_tokens += len(tk.encode(t).ids) if tk else len(t)
            sh = B._word_shingle_hashes(B._norm_skeleton(t))
            if sh:
                sigs[ordinal] = B._minhash(sh, ab, mask)
                src[ordinal] = (shard, row_pos)
            ordinal += 1

    # Fork C Pass B+C: band tables -> candidate doc set -> re-read candidates -> exact-J.
    candidate_docs = set()
    for b in range(BANDS):
        table = {}
        for o, sig in sigs.items():
            table.setdefault(sig[b * ROWS * 8 : (b + 1) * ROWS * 8], []).append(o)
        for members in table.values():
            if len(members) > 1:
                candidate_docs.update(members)
    # re-read candidate docs' shingles, grouped by shard
    shingle = {}
    by_shard = {}
    for o in candidate_docs:
        by_shard.setdefault(src[o][0], set()).add(o)
    for shard, ords in by_shard.items():
        rows = list(_read_rows(shard))
        for o in ords:
            t = rows[src[o][1]]
            sh = B._word_shingle_hashes(B._norm_skeleton(t))
            if sh:
                shingle[o] = sh
    find, union = B._union_find(len(exact))  # ordinals are 0..len(exact)-1 (short docs have an ordinal but no sig)
    for b in range(BANDS):
        table = {}
        for o, sig in sigs.items():
            table.setdefault(sig[b * ROWS * 8 : (b + 1) * ROWS * 8], []).append(o)
        for members in table.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    oi, oj = members[i], members[j]
                    si, sj = shingle.get(oi), shingle.get(oj)
                    if si and sj:
                        a, bset = set(si), set(sj)
                        jj = len(a & bset) / len(a | bset) if (a | bset) else 0.0
                        if jj >= JACCARD:
                            union(oi, oj)
    roots = {}
    for o in sigs:
        roots.setdefault(find(o), []).append(o)
    near_dup_docs = sum(len(m) - 1 for m in roots.values() if len(m) > 1)
    return {
        "domain_dir": domain_dir, "shards": len(shards), "docs": n,
        "exact_doc_dup": n - len(exact), "exact_doc_rate": (n - len(exact)) / n if n else 0.0,
        "para_total": para_total, "para_dup": para_total - len(para_exact),
        "para_rate": (para_total - len(para_exact)) / para_total if para_total else 0.0,
        "exact_unique_docs": len(exact),
        "near_dup_docs": near_dup_docs, "near_dup_rate": near_dup_docs / len(exact) if exact else 0.0,
        "near_config": {"perms": PERMS, "bands": BANDS, "rows": ROWS, "jaccard": JACCARD, "seed": 17},
        "unique_tokens": unique_tokens,
    }


def glob_shards(domain_dir):
    import glob

    return glob.glob(os.path.join(domain_dir, "*_*.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain_dirs", nargs="+", help="domain dir path; epochs appended as dir:epochs")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    a = ap.parse_args()
    try:
        from tokenizers import Tokenizer

        tk = Tokenizer.from_file(a.tokenizer)
        print(f"tokenizer loaded: {a.tokenizer}", file=sys.stderr)
    except Exception as e:
        tk = None
        print(f"no tokenizer ({e}); unique_tokens will be char counts", file=sys.stderr)
    for spec in a.domain_dirs:
        parts = spec.split(":")
        d = parts[0]
        epochs = float(parts[1]) if len(parts) > 1 else 1.0
        r = measure(d, tk)
        r["epochs"] = epochs
        r["consumed_tokens"] = int(round(r["unique_tokens"] * epochs))
        r["epoch_multiplier"] = (epochs * r["unique_tokens"]) / r["unique_tokens"] if r["unique_tokens"] else 0.0
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()