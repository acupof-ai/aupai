"""Exact-duplicate rate per corpus domain, on the bytes _domain_seqs actually feeds the tokenizer.

Written to test the P0 hypothesis that the 22B model's verbatim-repetition generations come from
a duplicated corpus. It does NOT test near-duplication; a paraphrase or a boilerplate-heavy pair
hashes differently and is counted unique. Read the result as "exact duplication is/is not
present", never as "the corpus is clean".

The read path is deliberately the same one train.py uses: data/corpus/<domain>/*.jsonl, the
`text` field, every shard. A scan of the first N rows would sample the first shards only, and
duplication concentrated in a later shard would read as zero -- so the default is the whole
domain and the row cap exists only for a quick look.

    python3 t62_corpus_dup_rate.py <domain> [max_rows]
    python3 t62_corpus_dup_rate.py --selftest

A zero from this probe is worth nothing until --selftest passes: a hash-based counter that
silently reads no text (wrong field name, unparseable lines) reports zero duplicates and zero
is the answer people want to hear. The selftest plants known duplicates and asserts they are
found -- the known-answer verification a null result requires (docs/lessons/instrument_not_system.md).
"""
import glob
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter

ROOT = "/work/aupai/data/corpus"


def scan(root, domain, max_rows=10**9):
    shards = sorted(p for p in glob.glob(os.path.join(root, domain, "*.jsonl"))
                    if "build_corpus_stats" not in os.path.basename(p))
    seen = Counter()
    n = nbytes = skipped = 0
    for p in shards:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    skipped += 1
                    continue
                t = o.get("text") or o.get("content") or ""
                if not t:
                    skipped += 1
                    continue
                seen[hashlib.sha1(t.encode()).digest()] += 1
                n += 1
                nbytes += len(t)
                if n >= max_rows:
                    break
        if n >= max_rows:
            break
    top = seen.most_common(5)
    return {"domain": domain, "shards": len(shards), "rows_scanned": n,
            "unique": len(seen), "dup_rows": n - len(seen),
            "dup_pct": round(100 * (n - len(seen)) / max(n, 1), 2),
            "max_copies": top[0][1] if top else 0,
            "top_counts": [c for _, c in top],
            "rows_with_no_text": skipped,
            "mean_doc_bytes": round(nbytes / max(n, 1))}


def selftest():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "kat"))
    rows = [{"text": f"unique document number {i} with padding"} for i in range(100)]
    rows += [{"text": "DUPLICATED BODY"}] * 7      # one body, 7 copies -> 6 dup rows
    rows += [rows[3], rows[3]]                     # another at 3 copies -> 2 dup rows
    with open(os.path.join(d, "kat", "kat_000.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    got = scan(d, "kat")
    assert got["rows_scanned"] == 109, got
    assert got["unique"] == 101, got
    assert got["dup_rows"] == 8, got
    assert got["max_copies"] == 7, got
    # the failure this guards: a counter that reads nothing reports a clean zero
    assert got["rows_with_no_text"] == 0, got
    print("selftest OK: 109 scanned, 101 unique, 8 dup rows, max 7 copies")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        cap = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
        print(json.dumps(scan(ROOT, sys.argv[1], cap)))
