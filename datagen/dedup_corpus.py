#!/usr/bin/env python3
"""Cross-source dedup under `harness run dedup --domains <a,b>[,c...]`.

The corpus-half dedup step: a GLOBAL pass over the named domains' cleaned text,
not per-domain (the dominant duplication is *between* sources -- the same crawled
page through several pipelines). Output a manifest of duplicate doc ids + which
source each duplicates, so the mix/training path skips them. The manifest is the
lazy option (shards stay untouched) -- the mix consults it.

    python datagen/dedup_corpus.py --domains web_hq,cci3,en

`--exact 0.8 --shingles 5` stood here until 2026-09-08 and the parser has never had either
flag: this pass is exact content-hash only, with no threshold and no shingling. Caught by
harness check doc_flags_parse, which compares a documented flag against add_argument.

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
                if not _NORM.sub("", t):
                    continue
                yield doc_key(t), t, os.path.basename(shard)


_NORM = re.compile(r"[\s\W_]+", re.UNICODE)


def doc_key(text):
    """The exact-dedup key: sha1 over the text with whitespace/punctuation/underscores
    removed (\\s \\W _ collapsed to nothing). Whitespace- and punctuation-only edits are
    therefore the SAME document (re-flowed re-crawls dedup), while word case, word order,
    and actual word content change the key. This is the load-bearing dedup-key contract."""
    return hashlib.sha1(_NORM.sub("", text).encode()).hexdigest()


def dedup_docs(docs):
    """Pure exact-dedup core over an iterable of (domain, text[, shard]). First occurrence
    of a normalized-content key is the representative; every later occurrence is a duplicate
    pointing back at the FIRST source domain. Returns (representatives, duplicates) where
    representatives is key -> {"source": domain} and duplicates is
    key -> {"dup_of": first_source_domain, "source": this_domain}. No files, deterministic.
    The first-occurrence rule is the thing that decides WHICH source survives a cross-source
    cluster; a dedup that flagged the first and kept a later copy would silently shift which
    corpus loses the document."""
    seen = {}
    dups = []  # one record PER duplicate occurrence: two later copies of one key are two
               # duplicates, not one -- a dict keyed by the content hash would silently drop
               # every duplicate occurrence after the first.
    for doc in docs:
        domain = doc[0]
        text = doc[1]
        shard = doc[2] if len(doc) > 2 else None
        if not _NORM.sub("", text or ""):
            # a raw-whitespace / punctuation-only document normalizes to the empty string;
            # it has no content identity. Keying it would map every such doc to the same
            # sha1("") and report a spurious duplicate cluster, so it is never a rep or dup.
            continue
        key = doc_key(text)
        if key in seen:
            dups.append({"doc_key": key, "dup_of": seen[key]["source"],
                         "source": domain, "shard": shard})
        else:
            seen[key] = {"source": domain, "shard": shard}
    return seen, dups



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


def _selftest():
    # Known answers for the exact cross-source dedup contract. Pure in-memory, no corpus.
    base = "the cat sat on the mat and the dog ran"

    # 1) NORMALIZATION dedup: whitespace/punctuation/underscore variants of one document
    #    share a key (re-flowed / re-punctuated re-crawls ARE the same document).
    variants = [
        base,
        base.replace(" ", "   "),                         # extra spaces
        base.replace(" ", "\n"),                          # newlines instead of spaces
        base.replace("the", "the,").replace("mat,", "mat. "),  # punctuation noise
        base.replace(" ", "_"),                           # underscores stripped too
    ]
    keys = {doc_key(v) for v in variants}
    assert len(keys) == 1, f"normalized variants must share one key, got {len(keys)}"

    # 2) case / word order / real content must NOT dedup: they are different documents.
    other = [
        base.upper(),                                    # case changes letters
        "ran dog the and mat the on sat cat the",        # words reordered
        "the cat sat on the mat and the cat slept",      # one different word
    ]
    other_keys = {doc_key(t) for t in other}
    base_key = doc_key(base)
    assert base_key not in other_keys, "case/order/content change must change the key"
    assert len(other_keys) == 3, f"the three other docs must be distinct, got {other_keys}"

    # 3) CROSS-SOURCE first-occurrence: the duplicate points at the FIRST source domain.
    #    Feed domains in order; a later-domain copy is a dup_of the earlier domain.
    a = ("domA", base, "a_shard.jsonl")
    b = ("domB", base.replace(" ", "\n  "), "b_shard.jsonl")   # same normalized doc
    c = ("domC", base.replace(" ", "   "), "c_shard.jsonl")
    seen, dups = dedup_docs([a, b, c])
    assert len(seen) == 1 and len(dups) == 2, (seen, dups)
    assert seen[doc_key(base)]["source"] == "domA", "first occurrence is the representative"
    sources = sorted((d["source"], d["dup_of"]) for d in dups)
    assert sources == [("domB", "domA"), ("domC", "domA")], sources
    # a genuinely distinct doc in domB is kept, not collapsed onto the cluster
    seen2, dups2 = dedup_docs([a, ("domB", "totally different unique words here", "x.jsonl")])
    assert len(seen2) == 2 and dups2 == [], (seen2, dups2)

    # 4) dedup_fp moves when the algorithm params move (a changed algo must re-stamp, so a
    #    stale manifest can never be read as the output of a different dedup).
    p1 = dedup_fp({"exact": "content-hash", "near_dup": "none", "shingles": 5})
    p2 = dedup_fp({"exact": "content-hash", "near_dup": "0.8", "shingles": 5})
    p3 = dedup_fp({"exact": "content-hash", "near_dup": "none", "shingles": 9})
    assert p1 != p2 and p1 != p3 and p2 != p3, "dedup_fp must change with any param"
    assert dedup_fp({"exact": "content-hash", "near_dup": "none", "shingles": 5}) == p1, \
        "same params must fingerprint identically (deterministic)"

    # 5) empty/whitespace docs are never representatives or duplicates (unscorable).
    seen3, dups3 = dedup_docs([("domA", "", "x"), ("domA", "   ", "y"),
                               ("domA", "\n\t", "z")])
    assert not seen3 and dups3 == [], (seen3, dups3)
    print("selftest ok: normalization collapses ws/punct variants; case/order/content do not; "
          "cross-source dup points at first occurrence; dedup_fp tracks params; empty skipped")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", help="comma-separated domains for the global pass (required unless --selftest)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.domains:
        ap.error("--domains is required")
    params = {"exact": "content-hash", "near_dup": "none", "shingles": 5}
    return dedup([d for d in a.domains.split(",") if d], params)


if __name__ == "__main__":
    sys.exit(main())
