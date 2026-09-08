#!/usr/bin/env python3
"""Exact token + packed-row count over a corpus directory. Full population, no sample."""
import glob, json, os, sys, time
import multiprocessing as mp

ROOT = "/work/aupai"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))
TOK = os.path.join(ROOT, "data", "tokenizer.json")
SEQ = 4097
_tok = None


def _init():
    global _tok
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(TOK)


def _one(p):
    from count_tokens import count_docs
    texts, docs = [], 0
    raw = open(p, "rb").read()
    # split("\n"), never splitlines(): splitlines breaks on U+2028/U+2029, which occur
    # INSIDE json string values in this corpus, so a valid row becomes 2+ unparseable
    # fragments and its document is silently dropped from the count. Measured on
    # code_rp1t_dd09 2026-09-07: 1,312 fragments from 288 documents.
    for line in raw.decode("utf-8", "replace").split("\n"):
        if not line.strip():
            continue
        try:
            texts.append(json.loads(line)["content"])
        except (json.JSONDecodeError, KeyError):
            continue
    docs = len(texts)
    toks = 0
    for i in range(0, docs, 2000):
        toks += count_docs(texts[i:i + 2000], _tok)
    return p, toks, docs, len(raw)


def main():
    d = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    dom = os.path.basename(d)
    # Shard prefix is the SOURCE domain, not the directory: code_rp1t_dd09/ holds
    # code_rp1t_*.jsonl. Glob every non-stats .jsonl, the same population fp_dir hashes.
    # holdout_slice_<phase>.jsonl is a frozen slice record, not corpus: it carries no
    # `content` key, so it adds 0 tokens and 0 docs but WOULD add a shard to n_shards and
    # its bytes to the total. fp_dir hashes it (it is part of the stamped domain), the
    # supply count must not.
    shards = sorted(p for p in glob.glob(os.path.join(d, "*.jsonl"))
                    if os.path.basename(p) != "build_corpus_stats.json"
                    and not os.path.basename(p).startswith("holdout_slice_")
                    and not os.path.basename(p).startswith("."))
    assert shards, f"no shards in {d}"
    t0 = time.time()
    tot = docs = nbytes = 0
    with mp.Pool(nw, initializer=_init) as pool:
        for i, (p, t, n, b) in enumerate(pool.imap_unordered(_one, shards), 1):
            tot += t; docs += n; nbytes += b
            if i % 20 == 0 or i == len(shards):
                el = time.time() - t0
                print(f"  {i}/{len(shards)} | {tot:,} tok | {docs:,} docs | "
                      f"{nbytes/1e6/el:.1f} MB/s | {el:.0f}s", flush=True)
    el = time.time() - t0
    from corpus_fingerprint import fp_dir
    fp = fp_dir(d)
    out = {
        "dir": d, "n_shards": len(shards), "docs": docs, "bytes": nbytes,
        "tokens": tot, "packed_rows": tot // SEQ, "fingerprint": fp,
        "wall_s": round(el, 1),
    }
    print(json.dumps(out, indent=1), flush=True)
    with open(f"/work/aupai/runs/count_{dom}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
