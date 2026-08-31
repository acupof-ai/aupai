#!/usr/bin/env python3
"""Clean a fetched raw source into a corpus domain under `harness run clean --domain`.

The corpus-half clean step. de owns the harness wrapper; this script owns the
substance. It reuses `datagen/build_corpus.py`, which already does the format
chain (reject_reason incl reject_holdout + garbage_topic) per document, stamps
`filters_fp` into build_corpus_stats.json, writes shard-by-shard (restartable),
and holds the "web" name as a guarded staging path.

    python scripts/clean_corpus.py --domain web_hq      # uses this domain's raw source
    python scripts/clean_corpus.py --domain web_hq --source fineweb2  # explicit raw

Contract: output data/corpus/<domain>/ + build_corpus_stats.json with filters_fp
content-hash of filters/*.py (a changed filter -> different fp). Shard-level
resumability is build_corpus's own (writes per shard, dedups incrementally).
Exit 0 on success, non-zero otherwise.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

# domain -> the named fetch source it is cleaned from. The fetched raw lives at
# data/raw/<source>/. This is the composition side: which raw becomes which domain.
DOMAIN_SOURCE = {
    "web_hq": "fineweb2",
    "cci3": "cci3_hq",
    "en": "fineweb2",  # a real-English slice replaces the cosmopedia-mislabeled en; source to be set
    "code": "rp1t_github",  # 30B code-raw cell: RedPajama-1T github (fetch t21)
}


def _stamp_filters_fp(domain):
    """Confirm build_corpus_stats.json has a filters_fp; fail loudly if build_corpus
    did not stamp it (silent filter drift is the incident class we guard here)."""
    sp = os.path.join(ROOT, "data", "corpus", domain, "build_corpus_stats.json")
    if not os.path.exists(sp):
        print(f"missing {sp}: build_corpus did not stamp corpus stats", file=sys.stderr)
        return False
    try:
        with open(sp) as f:
            stats = json.load(f)
    except Exception:
        print(f"{sp} unreadable", file=sys.stderr)
        return False
    if not stats.get("filters_fp"):
        print(
            f"{sp} has no filters_fp -- the domain's build did not record its filter chain", file=sys.stderr
        )
        return False
    print(f"{domain}: filters_fp {stats['filters_fp']}")
    return True


def clean(domain, source):
    src_dir = os.path.join(RAW, source)
    # the fetched raw is parquet (the sources we fetch are parquet corpora)
    parquet = sorted(glob.glob(os.path.join(src_dir, "*.parquet")))
    if parquet:
        source_arg = f"parquet:{os.path.join(src_dir, '*.parquet')}"
    else:
        jsonl = sorted(glob.glob(os.path.join(src_dir, "*.jsonl")))
        if not jsonl:
            print(f"no fetched raw for source {source!r} under {src_dir}", file=sys.stderr)
            return 2
        source_arg = f"jsonl:{os.path.join(src_dir, '*.jsonl')}"

    # The fetched raw is already holdout-and-format-checked by fetch? No: fetch only
    # downloads bytes; cleaning applies the filter chain. build_corpus does that
    # (+ reject_holdout + garbage_topic) and stamps filters_fp.
    cmd = [
        sys.executable,
        os.path.join(ROOT, "datagen", "build_corpus.py"),
        "--domain",
        domain,
        "--source",
        source_arg,
        "--host_cap",
        "0",  # single-source corpora: not a per-host web crawl
    ]
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"build_corpus exited {r.returncode}", file=sys.stderr)
        return r.returncode
    ok = _stamp_filters_fp(domain)
    return 0 if ok else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="corpus domain to produce (web_hq, cci3, en, ...)")
    ap.add_argument("--source", default=None, help="raw source to clean from (default: this domain's source)")
    a = ap.parse_args()
    source = a.source or DOMAIN_SOURCE.get(a.domain)
    if not source:
        print(f"no raw source known for domain {a.domain!r}; pass --source", file=sys.stderr)
        return 2
    return clean(a.domain, source)


if __name__ == "__main__":
    sys.exit(main())
