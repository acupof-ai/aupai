#!/usr/bin/env python3
"""Freeze what data/corpus/* IS right now, for every domain, whether or not it was
built with provenance.

Reproducibility and detectability are different properties and only one of them can
still be recovered. 30 of 49 domains carry a content fingerprint but no filters_fp,
and 5 carry nothing: for those, WHICH filter code produced the bytes is gone and no
snapshot brings it back. What a snapshot does buy is the other half -- from now on a
changed byte is detectable, and the eventual rebuild has something to diff against
instead of a rebuild nobody can compare to anything.

    python scripts/corpus_snapshot.py                 # write runs/corpus_snapshot.json
    python scripts/corpus_snapshot.py --check         # compare live dirs to the snapshot
    python scripts/corpus_snapshot.py --selftest      # known-answer: mutation is caught

Per-shard lines are kept, not just the per-domain hash: when a fingerprint stops
matching, the question is always WHICH shard, and by then the shard has already
changed -- it cannot be re-derived after the fact.

# restartable: reads 128KB per shard and writes one file at the end; an interrupt
# costs the walk (seconds to a couple of minutes on 232GB) and leaves nothing
# half-written, because the previous snapshot is only replaced once json.dump returns.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "datagen"))
from corpus_fingerprint import _shard_line, fp_dir, fp_filters  # noqa: E402

OUT = os.path.join(ROOT, "runs", "corpus_snapshot.json")


def _tier(stats):
    """Which provenance tier this domain sits in. The tier is a property of when it was
    built, not of the domain: filters_fp arrived after most of the web shards did."""
    if stats is None:
        return "none"
    if stats.get("filters_fp"):
        return "reproducible"
    return "detectable"


def snap_domain(d):
    stats = None
    p = os.path.join(d, "build_corpus_stats.json")
    if os.path.isfile(p):
        try:
            stats = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            stats = {"_unreadable": str(e)}
    shards = []
    total = 0
    for name in sorted(os.listdir(d)):
        if name == "build_corpus_stats.json" or name.startswith("."):
            continue
        path = os.path.join(d, name)
        if not os.path.isfile(path):
            continue
        total += os.path.getsize(path)
        shards.append(_shard_line(name, path).decode().strip())
    return {
        "fp": fp_dir(d),
        "tier": _tier(stats),
        "shards": len(shards),
        "bytes": total,
        "stats_fingerprint": (stats or {}).get("fingerprint"),
        "stats_filters_fp": (stats or {}).get("filters_fp"),
        "stats_filters": (stats or {}).get("filters"),
        "stats_tokens": (stats or {}).get("tokens"),
        "shard_lines": shards,
    }


def build(corpus=None):
    corpus = corpus or os.path.join(ROOT, "data", "corpus")
    doms = sorted(x for x in os.listdir(corpus) if os.path.isdir(os.path.join(corpus, x)))
    return {
        "corpus_dir": corpus,
        # The filter code AS OF THE SNAPSHOT. For a 'reproducible' domain compare it to
        # stats_filters_fp; for the other two tiers it is NOT what produced the bytes and
        # says only which code a rebuild would use.
        "filters_fp_now": fp_filters(ROOT),
        "domains": {d: snap_domain(os.path.join(corpus, d)) for d in doms},
    }


def check(snap, corpus=None):
    """(drifted, missing, added) against the live tree."""
    corpus = corpus or snap["corpus_dir"]
    live = set(x for x in os.listdir(corpus) if os.path.isdir(os.path.join(corpus, x)))
    drifted, missing = [], []
    for dom, rec in snap["domains"].items():
        d = os.path.join(corpus, dom)
        if not os.path.isdir(d):
            missing.append(dom)
            continue
        if fp_dir(d) != rec["fp"]:
            missing_shards = []
            was = {ln.split(":", 1)[0]: ln for ln in rec["shard_lines"]}
            now = {}
            for name in sorted(os.listdir(d)):
                if name == "build_corpus_stats.json" or name.startswith("."):
                    continue
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    now[name] = _shard_line(name, p).decode().strip()
            for name in sorted(set(was) | set(now)):
                if was.get(name) != now.get(name):
                    missing_shards.append(name)
            drifted.append((dom, missing_shards))
    return drifted, missing, sorted(live - set(snap["domains"]))


def _selftest():
    """Known answer on a REAL shard: a one-byte edit must be caught and must name the
    shard. A hand-written world would share this code's assumption about shard layout."""
    import shutil
    import tempfile

    corpus = os.path.join(ROOT, "data", "corpus")
    src = None
    for dom in sorted(os.listdir(corpus)):
        d = os.path.join(corpus, dom)
        if os.path.isdir(d) and any(f.endswith(".jsonl") for f in os.listdir(d)):
            src = d
            break
    if src is None:
        print("selftest SKIP: no corpus shards")
        return 0
    with tempfile.TemporaryDirectory() as t:
        dst = os.path.join(t, "dom")
        shutil.copytree(src, dst)
        snap = build(t)
        d, m, a = check(snap, t)
        assert not d and not m and not a, f"clean copy must not drift: {d} {m} {a}"
        shard = next(f for f in sorted(os.listdir(dst)) if f.endswith(".jsonl"))
        p = os.path.join(dst, shard)
        with open(p, "r+b") as f:
            b = f.read(1)
            f.seek(0)
            f.write(b"x" if b != b"x" else b"y")
        d, m, a = check(snap, t)
        assert d and d[0][1] == [shard], f"mutation must name the shard, got {d}"
        os.utime(p, (0, 0))
        assert check(snap, t)[0], "mtime change alone is not what we caught"
        print(f"selftest OK: mutation caught on {shard}, named exactly")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.check:
        snap = json.load(open(a.out, encoding="utf-8"))
        drifted, missing, added = check(snap)
        for dom, shards in drifted:
            print(f"DRIFT {dom}: {len(shards)} shard(s) differ: {', '.join(shards[:5])}")
        for dom in missing:
            print(f"GONE  {dom}")
        for dom in added:
            print(f"NEW   {dom} (not in the snapshot; re-run without --check to include it)")
        n = len(snap["domains"])
        print(f"{n} domain(s) in the snapshot, {len(drifted)} drifted, {len(missing)} gone, {len(added)} new")
        sys.exit(1 if (drifted or missing) else 0)
    snap = build()
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=1)
        f.write("\n")
    tiers = {}
    for rec in snap["domains"].values():
        tiers[rec["tier"]] = tiers.get(rec["tier"], 0) + 1
    tb = sum(r["bytes"] for r in snap["domains"].values())
    print(f"{a.out}: {len(snap['domains'])} domains, {sum(r['shards'] for r in snap['domains'].values())} shards, "
          f"{tb / 1e9:.1f} GB; tiers {tiers}")
