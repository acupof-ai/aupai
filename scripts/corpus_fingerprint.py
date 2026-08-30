#!/usr/bin/env python3
"""Per-domain corpus fingerprint: hash of (shard name, size, mtime).

The incident: two corpora both called data/corpus/math, one 0.0% contaminated and
one 30.0%, and nothing but an audit distinguished them. A checkpoint needs to say
which corpus it trained on the way it already says which tokenizer (vocab_id).

Stat-only on purpose: name+size+mtime is O(shards), so the fingerprint of a 108 GB
domain costs a directory scan, not a line count. It catches the drift that matters
(shard deleted, shard rewritten) and misses same-size same-mtime edits -- the same
ceiling as the scan ledger, documented rather than hidden. Row counts would close
it but cost minutes per domain on web-scale data.

    python scripts/corpus_fingerprint.py [mix.json]   # print {domain: fp} for a mix
    python scripts/corpus_fingerprint.py --self-check # broken world: mutate a real shard

build_corpus.py stamps the fingerprint into build_corpus_stats.json at build time;
harness check corpus_fp_matches compares that stamp to the live directory.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fp_domain(domain, corpus_dir=None):
    """Hash of sorted (shard name, size, mtime) for one corpus domain."""
    d = os.path.join(corpus_dir or os.path.join(ROOT, "data", "corpus"), domain)
    if not os.path.isdir(d):
        return None
    h = hashlib.sha1()
    for name in sorted(os.listdir(d)):
        if name == "build_corpus_stats.json" or name.startswith("."):
            continue
        st = os.stat(os.path.join(d, name))
        h.update(f"{name}:{st.st_size}:{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:16]


def fp_mix(mix_path):
    """{domain: fp} for every domain named in a mix file."""
    mix = json.load(open(mix_path, encoding="utf-8"))
    return {dom: fp_domain(dom) for dom in mix["domains"]}


def self_check():
    """Broken world: copy a REAL shard into a temp corpus, mutate it, fp must change."""
    real = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", "math", "*.jsonl")))
    if not real:
        print("self-check SKIP: no math shards to copy")
        return 0
    with tempfile.TemporaryDirectory() as d:
        dom = os.path.join(d, "math")
        os.makedirs(dom)
        shard = os.path.join(dom, "real_shard.jsonl")
        with open(real[0], "rb") as f, open(shard, "wb") as g:
            g.write(f.read())
        fp1 = fp_domain("math", corpus_dir=d)
        with open(shard, "a", encoding="utf-8") as f:
            f.write(json.dumps({"question": "指纹自检：改一行必须变", "output": "1"}, ensure_ascii=False) + "\n")
        fp2 = fp_domain("math", corpus_dir=d)
        assert fp1 and fp2 and fp1 != fp2, f"mutation did not change fingerprint: {fp1} -> {fp2}"
        os.unlink(shard)
        fp3 = fp_domain("math", corpus_dir=d)
        assert fp3 is None or fp3 != fp1, "deleting the only shard must change the fingerprint too"
    print(f"self-check OK (mutate {fp1} -> {fp2}, delete -> {fp3})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mix", nargs="?", help="mix json; default: the live Cfg.mix")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        sys.exit(self_check())
    mix = args.mix
    if not mix:
        import ast
        src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ClassDef) and node.name == "Cfg":
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and stmt.targets[0].id == "mix":
                        mix = ast.literal_eval(stmt.value)
        assert mix, "no mix arg and no Cfg.mix"
    print(json.dumps(fp_mix(os.path.join(ROOT, mix) if not os.path.isabs(mix) else mix), indent=1))


if __name__ == "__main__":
    main()
