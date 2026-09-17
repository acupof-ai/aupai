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

Outputs (data/dedup/): dedup_manifest.json is a JSON ARRAY with one record per
duplicate OCCURRENCE {doc_key, dup_of, source, shard} (a later copy points at the
first source domain); dedup_kept.json maps representative key -> {source, shard};
dedup_stats.json carries dedup_fp = hash(algorithm + params: exact threshold,
near-dup threshold, shingling params) -- it changes when the algorithm changes.
CPU-only (near-dup ~30ms/doc). Resumable: a per-domain manifest_<domain>.json marker
skips a completed domain; if all are present the global manifest is left as-is.
As of 2026-09-17 NO production code consumes dedup_manifest.json yet (verified by
repo grep; only docs/facts mention it) -- it is the lazy manifest a future mix/
training skip will consult, so its shape is set and gated here before a reader lands.
Exit 0/non-zero.
"""

import argparse
import contextlib
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atomic_io import durable_publish  # noqa: E402

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


def _write_json_atomic(path, obj, *, indent=None):
    """Durable, atomic JSON publish via datagen/atomic_io.durable_publish (#450): same-dir
    .tmp -> flush -> fsync(file) -> os.replace -> fsync(parent dir). A reader never sees a
    half-written path. Every dedup output MUST go through this; a bare open(final,'w') is
    the regression the spy selftest turns red."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
        durable_publish(fh, tmp, path)


def _load_valid_json(path):
    """Parsed JSON at path, or None if absent/corrupt/torn. Resume treats a torn manifest
    as 'not complete' and rebuilds, never a permanent JSONDecodeError."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def dedup(domains, params, *, doc_source=corpus_docs, out_dir=DEDUP_DIR):
    """Production cross-source exact dedup. It MUST go through dedup_docs (the tested core):
    gather every domain's documents in domain order, then keep the first occurrence and emit
    one duplicate record PER later occurrence (a list -- never a hash-keyed dict that collapses
    two later copies into one and mis-attributes them to the last domain).

    doc_source/out_dir are injected only for the selftest (stub corpus + temp dir)."""
    os.makedirs(out_dir, exist_ok=True)
    fp = dedup_fp(params)
    stats = {"domains": list(domains), "dedup_fp": fp, "params": params, "shards_scanned": {}}

    # Build the single ordered document stream across domains. A per-domain marker file makes a
    # completed domain resumable: if present, its docs are skipped and the recorded count used.
    docs = []
    total = 0
    for domain in domains:
        dom_file = os.path.join(out_dir, f"manifest_{domain}.json")
        marker = _load_valid_json(dom_file)
        if marker is not None:
            stats["shards_scanned"][domain] = marker.get("docs_scanned", 0)
            continue
        n = 0
        for _key, text, shard in doc_source(domain):
            docs.append((domain, text, shard))
            n += 1
        stats["shards_scanned"][domain] = n
        total += n
        # resumable per-domain marker: counts the docs scanned from this domain so a re-run
        # skips it (the shards and the global manifest are rebuilt from the gathered stream).
        _write_json_atomic(dom_file, {"domain": domain, "docs_scanned": n})

    # Resume: if every requested domain was already scanned (markers present, nothing gathered),
    # the global manifest on disk is complete -- leave it rather than rebuild from an empty
    # stream (which would wipe it). A partial prior run (some domains unmarked) re-scans those
    # domains and rebuilds from the full gathered stream. A torn/corrupt global manifest under
    # complete markers is NOT left to JSONDecodeError forever: invalidate the markers and
    # re-scan every domain this run.
    all_marked = all(_load_valid_json(os.path.join(out_dir, f"manifest_{d}.json")) is not None
                     for d in domains)
    mpath0 = os.path.join(out_dir, "dedup_manifest.json")
    if not docs and all_marked:
        existing = _load_valid_json(mpath0)
        if isinstance(existing, list):
            print(f"dedup {list(domains)}: all domains already scanned; {mpath0} left as-is")
            return 0
        for d in domains:
            with contextlib.suppress(OSError):
                os.remove(os.path.join(out_dir, f"manifest_{d}.json"))
        docs = []
        total = 0
        for domain in domains:
            dom_file = os.path.join(out_dir, f"manifest_{domain}.json")
            n = 0
            for _key, text, shard in doc_source(domain):
                docs.append((domain, text, shard))
                n += 1
            stats["shards_scanned"][domain] = n
            total += n
            _write_json_atomic(dom_file, {"domain": domain, "docs_scanned": n})

    seen, dups = dedup_docs(docs)  # THE tested decision; dups is one record per occurrence

    # global manifest: a JSON array of duplicate records (one per later occurrence).
    _write_json_atomic(mpath0, dups)
    # representatives manifest (the surviving first occurrences), key -> {source, shard}.
    _write_json_atomic(os.path.join(out_dir, "dedup_kept.json"), seen)
    sp = os.path.join(out_dir, "dedup_stats.json")
    stats["duplicates"] = len(dups)
    stats["representatives"] = len(seen)
    _write_json_atomic(sp, stats, indent=1)
    print(
        f"dedup {list(domains)}: {total} docs scanned, {len(seen)} representatives, "
        f"{len(dups)} exact-duplicate occurrences -> {mpath0} (dedup_fp {fp})"
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

    # 6) PRODUCTION WIRING: dedup() itself must drive dedup_docs and persist one record per
    #    duplicate occurrence. The pure tests above cannot catch a dedup() that inlines an old
    #    hash-keyed dict (which collapses 3 copies to 1 dup and attributes it to the LAST
    #    domain). Stub the corpus reader and output dir; three copies of one normalized doc
    #    across A/B/C must yield exactly two dup records, both pointing at the first source.
    import tempfile
    base = "the cat sat on the mat and the dog ran"
    corpus = {
        "domA": [(doc_key(base), base, "a0.jsonl"),
                 (doc_key("only in alpha"), "only in alpha unique words", "a1.jsonl")],
        "domB": [(doc_key(base), base.replace(" ", "  "), "b0.jsonl")],
        "domC": [(doc_key(base), base.replace(" ", "\n"), "c0.jsonl")],
    }

    def stub_docs(domain):
        return iter(corpus[domain])

    with tempfile.TemporaryDirectory() as td:
        rc = dedup(["domA", "domB", "domC"],
                   {"exact": "content-hash", "near_dup": "none", "shingles": 5},
                   doc_source=stub_docs, out_dir=td)
        assert rc == 0
        with open(os.path.join(td, "dedup_manifest.json")) as f:
            on_disk = json.load(f)
        with open(os.path.join(td, "dedup_stats.json")) as f:
            on_stats = json.load(f)
        # a LIST with one entry per duplicate OCCURRENCE (B and C), not one hash-keyed entry
        assert isinstance(on_disk, list), "global manifest must be a per-occurrence list"
        assert len(on_disk) == 2, f"3 copies must give 2 dup records, got {len(on_disk)}"
        pairs = sorted((r["source"], r["dup_of"]) for r in on_disk)
        assert pairs == [("domB", "domA"), ("domC", "domA")], pairs
        # both later copies are attributed to the FIRST source, never the last
        assert all(r["dup_of"] == "domA" for r in on_disk)
        # two distinct representatives survive (the shared doc + alpha-only doc)
        assert on_stats["representatives"] == 2 and on_stats["duplicates"] == 2, on_stats
        # rerun with markers present rescans nothing but still rewrites a consistent manifest
        rc2 = dedup(["domA", "domB", "domC"],
                    {"exact": "content-hash", "near_dup": "none", "shingles": 5},
                    doc_source=stub_docs, out_dir=td)
        assert rc2 == 0
        with open(os.path.join(td, "dedup_manifest.json")) as f:
            again = json.load(f)
        assert again == on_disk, "resume must produce an identical manifest"

    # 7) DURABLE/ATOMIC PUBLICATION: every output lands via tmp -> flush/fsync ->
    #    os.replace(tmp != dst) -> dir fsync, never a bare open(final,'w') a crash can tear.
    #    Spy on durable_publish in THIS module's globals (under __main__, `import
    #    datagen.dedup_corpus` is a second module object; patching that attribute never
    #    reaches _write_json_atomic). A direct write bypasses durable_publish and goes red.
    calls = []
    _real_pub = durable_publish

    def _spy_publish(fh, tmp, dst):
        calls.append((tmp, dst))
        return _real_pub(fh, tmp, dst)

    with tempfile.TemporaryDirectory() as td:
        globals()["durable_publish"] = _spy_publish
        try:
            dedup(["domA"], {"exact": "content-hash", "near_dup": "none", "shingles": 5},
                  doc_source=lambda dom: iter([(doc_key("one two three four"),
                                                "one two three four", "a.jsonl")]),
                  out_dir=td)
        finally:
            globals()["durable_publish"] = _real_pub
        got = {os.path.basename(dst) for _tmp, dst in calls}
        expected = {"dedup_manifest.json", "dedup_kept.json", "dedup_stats.json",
                    "manifest_domA.json"}
        assert expected <= got, f"not every output published durably: missing {expected - got}"
        for tmp, dst in calls:
            assert tmp != dst and os.path.basename(tmp).endswith(".tmp"), (tmp, dst)
            assert os.path.dirname(os.path.abspath(tmp)) == os.path.dirname(
                os.path.abspath(dst)), (tmp, dst)
        assert not [f for f in os.listdir(td) if f.endswith(".tmp")], "stale .tmp left behind"

    # 8) TORN-MANIFEST RECOVERY: complete markers + a half-written global manifest must be
    #    REBUILT (rc 0, valid correct manifest), not left failing every resume with
    #    JSONDecodeError.
    stub2 = {
        "domA": [(doc_key(base), base, "a0.jsonl")],
        "domB": [(doc_key(base), base.replace(" ", "  "), "b0.jsonl")],
    }
    with tempfile.TemporaryDirectory() as td:
        dedup(["domA", "domB"],
              {"exact": "content-hash", "near_dup": "none", "shingles": 5},
              doc_source=stub2.__getitem__, out_dir=td)
        with open(os.path.join(td, "dedup_manifest.json"), "w") as f:
            f.write('[{"doc_key": "deadbeef", "dup_of": "do')  # torn mid-JSON
        rc3 = dedup(["domA", "domB"],
                    {"exact": "content-hash", "near_dup": "none", "shingles": 5},
                    doc_source=stub2.__getitem__, out_dir=td)
        assert rc3 == 0, "torn manifest must rebuild, not raise"
        rebuilt = _load_valid_json(os.path.join(td, "dedup_manifest.json"))
        assert isinstance(rebuilt, list) and len(rebuilt) == 1, rebuilt
        assert rebuilt[0]["source"] == "domB" and rebuilt[0]["dup_of"] == "domA", rebuilt

    print("selftest ok: normalization; cross-source first-occurrence; dedup_fp; empty skip; "
          "production drives core (3 copies -> 2 dup); all 4 outputs atomic tmp->rename "
          "(spy-verified); torn global manifest with complete markers rebuilds, never raises")
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
