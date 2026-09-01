#!/usr/bin/env python3
"""Byte-identity check for the near-dedup post-pass (Fork C, `_near_dedup_postpass`).
Builds two {domain}_*.jsonl shards with a known near-dup pair spanning the shard
boundary, runs the post-pass SERIAL and PARALLEL in separate out dirs, and asserts
the written {domain}_*.jsonl are byte-identical -- AND that the near-dup pair
collapsed to one survivor (so the byte-identity is tested over a real removal, not
an all-keep pass). Acceptance condition 3 (44, 2026-09-01): serial reference = the
same Fork C pipeline single-worker, MinHash fixed seed, survivor = lowest ordinal
per cluster."""
import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import build_corpus as B  # noqa: E402

# A skeleton and a near-variant of it: they share >0.5 normalized word-3-gram
# Jaccard, so they are ONE near-cluster and only the lowest-ordinal survivor keeps.
_SKEL = ("the quick brown fox jumps over a lazy dog near the river bank while the "
         "sun was setting behind the hills on a warm evening in the middle of june")
# One near-cluster: NEAR is SKEL plus a short appended tail, so the base trigrams
# dominate and their exact word-3-gram Jaccard is ~0.9 (well above the 0.5 gate).
SKEL = ("the quick brown fox jumps over a lazy dog near the river bank while the sun "
        "was setting behind the hills on a warm evening in the middle of june and the "
        "birds returned to their nests one by one as the light began to fade across "
        "the meadow and settled over the quiet village for the night")
NEAR = SKEL + " then the morning came and the people woke to a new day that began again"
UNIQ_A = "def main(x): compute the total of a list and return it rounded to two places"
UNIQ_B = "class Vector: add two vectors componentwise and return a new instance"


def main():
    # shards: NEAR and SKEL are two docs of one near-cluster, split across the shard
    # boundary (proves removal is GLOBAL, not per-shard); UNIQ_A/B are genuinely
    # distinct and must both survive (proves we do not over-merge).
    shards = [
        [{"content": SKEL, "url": "u"}, {"content": UNIQ_A, "url": "u"}],
        [{"content": NEAR, "url": "u"}, {"content": UNIQ_B, "url": "u"}],
    ]
    outs = []
    for par in (False, True):
        d = tempfile.mkdtemp()
        for i, ss in enumerate(shards):
            with open(os.path.join(d, f"t_{i:03d}.jsonl"), "w", encoding="utf-8") as f:
                for doc in ss:
                    f.write(json.dumps(doc) + "\n")
        a = argparse.Namespace(out=d, domain="t", workers=3 if par else 1)
        B._near_dedup_postpass(a, perms=128, bands=128, rows=1, jaccard=0.5, seed=17)
        merged = b"".join(
            open(os.path.join(d, p), "rb").read()  # noqa: SIM115 -- shards are small, reuse the line
            for p in sorted(os.listdir(d))
            if p.startswith("t_") and p.endswith(".jsonl")
        )
        outs.append((d, merged))
    (d1, serial), (d2, parallel) = outs
    try:
        assert serial == parallel, (
            f"BYTE MISMATCH ({len(serial)} vs {len(parallel)} bytes)\n"
            f"serial:   {serial[:200]}\nparallel: {parallel[:200]}"
        )
        # the near-dup cluster must have collapsed: 'the quick brown fox' appears once
        # across the output (SKEL and NEAR are one cluster, one survivor), not twice.
        fox = serial.count(b"the quick brown fox jumps over a lazy dog")
        assert fox == 1, f"near-dup pair did not collapse to one survivor (fox hits: {fox})"
        # UNIQ_A and UNIQ_B are genuinely distinct docs that LOOK nothing alike; both
        # must survive (proves the pass does not over-merge distinct docs).
        assert serial.count(b"def main(x): compute the total of a list") == 1, "UNIQ_A dropped or duplicated"
        assert serial.count(b"class Vector: add two vectors") == 1, "UNIQ_B dropped or duplicated"
        lines = serial.count(b"\n")
        print(f"ok: near-dedup post-pass serial==parallel byte-identical ({len(serial)}B, {lines} lines); near-cluster -> 1 survivor")
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
