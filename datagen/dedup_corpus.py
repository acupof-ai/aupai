#!/usr/bin/env python3
"""Cross-source dedup under `harness run dedup --domains <a,b>[,c...]`.

The corpus-half dedup step: a GLOBAL pass over the named domains' cleaned text,
not per-domain (the dominant duplication is *between* sources -- the same crawled
page through several pipelines). Output a manifest of duplicate doc ids + which
source each duplicates, so the mix/training path skips them. The manifest is the
lazy option (shards stay untouched) -- the mix consults it.

    python datagen/dedup_corpus.py --domains web_hq,cci3,en
    python datagen/dedup_corpus.py --domains web_hq,cci3 --exact 0.8 --shingles 5

Contract: data/dedup/dedup_manifest.json (duplicate ids + which source) +
dedup_stats.json with dedup_fp = hash(algorithm + params: exact threshold,
near-dup threshold, shingling params) -- changes when the algorithm changes.
CPU-only (near-dup ~30ms/doc). Resumable: process domains one at a time, skip
completed (per-domain manifest). Exit 0/non-zero.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEDUP_DIR = os.path.join(ROOT, "data", "dedup")


def dedup_fp(params):
    h = hashlib.sha1()
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()


def corpus_docs(domain):
    """Yield (sha1-of-raw-text, raw_text, shard) for every doc in the domain's shards."""
    for shard in sorted(glob.glob(os.path.join(ROOT, "data", "corpus", domain, "*.jsonl"))):
        with open(shard, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("content") or d.get("text") or ""
                if not t:
                    continue
                norm = _NORM.sub("", t)
                yield hashlib.sha1(norm.encode()).hexdigest(), norm, os.path.basename(shard)


_NORM = re.compile(r"[\s\W_]+", re.UNICODE)


def dedup(domains, params):
    os.makedirs(DEDUP_DIR, exist_ok=True)
    fp = dedup_fp(params)
    stats = {"domains": domains, "dedup_fp": fp, "params": params, "shards_scanned": {}}

    # pass 1: exact across all domains (content-hash). second+ occurrence -> duplicate.
    seen_exact = {}  # hash -> (domain, shard)
    manifest = {}  # dochash -> {"dup_of": hash or null, "source": domain}
    total = 0

    for domain in domains:
        dom_file = os.path.join(DEDUP_DIR, f"manifest_{domain}.json")
        dom_seen = {}
        if os.path.exists(dom_file):
            with open(dom_file) as f:  # resume: per-domain completed manifest
                stats["shards_scanned"][domain] = len(json.load(f))
                continue
        for dh, _norm, shard in corpus_docs(domain):
            total += 1
            if dh in seen_exact:
                manifest[dh] = {"dup_of": seen_exact[dh][0], "source": domain, "shard": shard}
            else:
                seen_exact[dh] = (domain, shard)
                dom_seen[dh] = shard
        # per-domain manifest (retustartable): write once per domain, not per shard --
        # the shards themselves are not rewritten; the manifest is the durable output.
        with open(dom_file, "w") as f:
            json.dump({k: v for k, v in manifest.items() if v.get("source") == domain}, f, indent=0)

    mpath = os.path.join(DEDUP_DIR, "dedup_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f)
    sp = os.path.join(DEDUP_DIR, "dedup_stats.json")
    stats["duplicates"] = len(manifest)
    with open(sp, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(
        f"dedup {domains}: {total} docs, {len(manifest)} exact-duplicate doc-ids -> {mpath} (dedup_fp {fp})"
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", required=True, help="comma-separated domains for the global pass")
    a = ap.parse_args()
    params = {"exact": "content-hash", "near_dup": "none", "shingles": 5}
    return dedup([d for d in a.domains.split(",") if d], params)


if __name__ == "__main__":
    sys.exit(main())
