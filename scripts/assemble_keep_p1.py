#!/usr/bin/env python3
"""Assemble data/corpus/code_keep_p1 from the p1 KEEP set as hard links, and stamp it.

The mix loader globs a flat data/corpus/<domain>/*.jsonl, but the audited keep set lives
nested at data/p1/keep_set/{code_rp1t_dd09,code_rp1t_b2v2_dd,code_dedup08}. This builds the
flat domain dir with hard links (same filesystem, zero extra bytes), each link prefixed
"<source_dir>__" so provenance survives and names cannot collide. It refuses to create a
symlink (the mix guard rejects a symlinked domain) and writes build_corpus_stats.json with
the train._corpus_fp fingerprint so a launch's _assert_mix_domains accepts the domain.

Idempotent: existing links are left in place and re-verified, never replaced. The three
source dirs are read only. Pod-only -- the linked corpus bytes are not in git.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import train  # noqa: E402

ROOT = train.ROOT
SRC = os.path.join(ROOT, "data", "p1", "keep_set")
SUBS = ["code_rp1t_dd09", "code_rp1t_b2v2_dd", "code_dedup08"]
DST = os.path.join(ROOT, "data", "corpus", "code_keep_p1")
EXPECTED = 685


def main():
    os.makedirs(DST, exist_ok=True)
    linked = 0
    for sub in SUBS:
        for src in sorted(glob.glob(os.path.join(SRC, sub, "*.jsonl"))):
            dst = os.path.join(DST, f"{sub}__{os.path.basename(src)}")
            if os.path.lexists(dst):
                if os.path.islink(dst):
                    raise SystemExit(f"refuse: {dst} is a symlink; the mix guard rejects those")
            else:
                os.link(src, dst)
            linked += 1
    present = [x for x in os.listdir(DST) if x.endswith(".jsonl")]
    assert linked == EXPECTED and len(present) == EXPECTED, f"{linked} linked, {len(present)} on disk"
    assert not any(os.path.islink(os.path.join(DST, x)) for x in os.listdir(DST)), "symlink in domain dir"

    fp = train._corpus_fp(DST)
    stats = os.path.join(DST, "build_corpus_stats.json")
    if os.path.exists(stats):
        with open(stats) as fh:
            if json.load(fh).get("fingerprint") != fp:
                raise SystemExit(f"refuse: existing {stats} stamps a different fingerprint; investigate before overwriting")
    with open(stats, "w") as fh:
        json.dump({
            "fingerprint": fp,
            "domain": "code_keep_p1",
            "assembled": "hard links into data/p1/keep_set/{code_rp1t_dd09,code_rp1t_b2v2_dd,code_dedup08}",
            "shards": EXPECTED,
            "name_rule": "<source_dir>__<shard filename>",
            "note": "assembled, not built by build_corpus.py; fingerprint is train._corpus_fp over the links",
        }, fh, indent=1)

    got = train._assert_mix_domains(["code_keep_p1"], os.path.join(ROOT, "data", "corpus"))
    assert got["code_keep_p1"] == fp
    print(f"ASSEMBLED links={len(present)} fingerprint={fp} guard=ACCEPT")


if __name__ == "__main__":
    main()
