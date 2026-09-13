#!/usr/bin/env python3
"""Build token caches for named mix domains with train.py's own path, CPU side.

Thin driver over train._domain_seqs: it shard-globs data/corpus/<d>, shuffles with the
sample seed, encodes via the fork worker pool, and writes tokens_<d>.pt plus the
.vocab/.srcfp/.seed stamps. A fresh cache is skipped (mmap load), so re-running reports
without re-encoding. CPU only -- set CUDA_VISIBLE_DEVICES empty in the environment.

workers x RAYON_NUM_THREADS must stay <= nproc (train._encode_domain contract); this
script does not set RAYON_NUM_THREADS for you because nproc is machine-specific.

Report line per domain (stdout):
  REPORT <d> rows=<pool rows> pool_tokens=<rows*(seq+1)> seconds=<s> sha256=<16> bytes=<n>
For an already-cached domain seconds is the mmap load, not the encode.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "eval"))
import cache_guard  # noqa: E402

# Same normalisation build_l3_stub.py / datagen/gen_exercises.py use for the global dedup
# key, so the manifest's content_sha1_norm keys the stub docs exactly.
_NORM_WS = re.compile(r"\s+")


def _norm(s):
    return _NORM_WS.sub(" ", s).strip()


def sha256_prefix(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 26), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def _load_exclude_keys(manifest_path):
    """(urls, content_sha1_norm) sets from a phi holdout manifest jsonl."""
    urls, shas = set(), set()
    with open(manifest_path, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("url"):
                urls.add(r["url"])
            if r.get("content_sha1_norm"):
                shas.add(r["content_sha1_norm"])
    return urls, shas


def build_excluded_cache(domain, manifest_path, tok, workers):
    """Encode a domain MINUS the holdout docs named in the manifest.

    Mirrors train._domain_seqs (same shard glob/whitelist, same sample-seed shuffle, same
    _encode_domain and int32 save) but drops a doc when its url OR sha1(_norm(content)) is
    in the manifest. Writes a DISTINCT cache so it cannot be read as the full domain:
      tokens_<domain>.excl<manifest_sha256_16>.pt
    with .srcfp = "<domain corpus fp>|exclude=<manifest_sha256_16>:<ndropped>" (the cache's
    bytes are a function of both) and the normal .vocab/.seed stamps. train.py is untouched.
    """
    corpus_dir = os.path.join(train.DATA, "corpus", domain)
    seen = sorted(os.listdir(corpus_dir))
    shards, unknown = [], []
    for b in seen:
        if b == "build_corpus_stats.json" or b.startswith("."):
            continue
        p = os.path.join(corpus_dir, b)
        if b in train.NON_SHARD_JSONL or train.NON_SHARD_RE.search(b):
            continue
        if train.SHARD_RE.search(b):
            shards.append(p)
        else:
            unknown.append(b)
    if unknown:
        raise SystemExit(f"REFUSE unknown files in {corpus_dir}: {unknown[:4]}")
    assert shards, f"{domain}: no shards"

    urls, shas = _load_exclude_keys(manifest_path)
    msha = sha256_prefix(manifest_path)
    texts, ndocs, ndropped = [], 0, 0
    for p in shards:
        with open(p, encoding="utf-8") as sfh:
            for ln in sfh:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                content = r.get("content")
                if content is None:
                    continue
                ndocs += 1
                if r.get("url") in urls or (
                    content and hashlib.sha1(_norm(content).encode()).hexdigest() in shas
                ):
                    ndropped += 1
                    continue
                texts.append(content)
    assert texts, f"{domain}: all docs excluded"
    random.Random(train._sample_seed()).shuffle(texts)
    print(f"mix(excl {msha}): tokenizing {domain} ({ndocs} docs, dropped {ndropped} holdout, "
          f"kept {len(texts)}, workers={workers})", flush=True)
    data = train._encode_domain(texts, tok, workers, log=lambda m: print(m, flush=True))
    del texts

    cache = os.path.join(train._token_cache_dir(), f"tokens_{domain}.excl{msha}.pt")
    torch.save(data, cache)
    with open(cache + ".vocab", "w") as fh:
        fh.write(train.VOCAB_ID)
    with open(cache + ".srcfp", "w") as fh:
        fh.write(f"{train._corpus_fp(corpus_dir)}|exclude={msha}:{ndropped}")
    with open(cache + ".seed", "w") as fh:
        fh.write(str(train._sample_seed()))
    return cache, ndocs, ndropped, len(data), data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domains", nargs="+")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--exclude-manifest", default="",
                    help="phi holdout manifest jsonl; build each domain's cache MINUS its docs")
    a = ap.parse_args()
    tok = train.build_tokenizer([])
    print("cache dir", train._token_cache_dir(), "vocab", train.VOCAB_ID[:12], flush=True)
    for d in a.domains:
        # Building writes the same tens of GB to /data00 that a read does, so the co-residency
        # rule applies on this side too: refuse while a live claim holds the cards. No claim ->
        # silent pass. An unrecorded domain size WARNs and proceeds on a lower bound.
        cache_guard.assert_not_co_resident([d])
        t = time.time()
        if a.exclude_manifest:
            cache, ndocs, ndropped, n_ids, data = build_excluded_cache(
                d, a.exclude_manifest, tok, a.workers)
            rows = n_ids // (train.Cfg.seq + 1)
            dt = time.time() - t
            print(
                f"REPORT {d}.excl rows={rows} pool_tokens={rows * (train.Cfg.seq + 1)} "
                f"seconds={dt:.0f} sha256={sha256_prefix(cache)} bytes={os.path.getsize(cache)} "
                f"docs={ndocs} dropped_holdout={ndropped}",
                flush=True,
            )
            del data
            continue
        seqs = train._domain_seqs(d, tok, True, False, workers=a.workers)
        dt = time.time() - t
        rows = len(seqs)
        path = train._domain_cache_path(d)
        print(
            f"REPORT {d} rows={rows} pool_tokens={rows * (train.Cfg.seq + 1)} "
            f"seconds={dt:.0f} sha256={sha256_prefix(path)} bytes={os.path.getsize(path)}",
            flush=True,
        )
        del seqs
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
