#!/usr/bin/env python3
"""Byte-identity check for the parallel exact-dup+holdout global pass.
Builds small w* shards with known exact dups + a holdout-pattern doc, runs the
SERIAL and PARALLEL exact-only global pass in separate out dirs, and asserts the
written {domain}_*.jsonl are byte-identical. Fails loudly on a real mismatch."""
import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import build_corpus as B  # noqa: E402

DUP = "public static void paginate records count cache size large enough text" * 2
UNIQ = "def g(x): return x * 2 + 1 and the rest of this function body is long enough"
HOLD = "REVEAL_ME_SENTINEL_HOLDOUT_0003 is a holdout marker that must be dropped"
# A genuinely-held-out doc (real eval problem in data/eval/holdout_hashes.txt), to
# exercise the holdout-slice gate end-to-end: captured in the pass, through _write_stats,
# into holdout_slice_{phase}.jsonl whose row count must match the stamp's holdout reason.
REAL_HOLD = "小明有10个苹果，他送给小红3个，还剩几个？"


def main():
    # three shards; doc 0 x2 (exact dup across shard 0), UNIQ x2 (across shard 0/1),
    # UNIQ x1 (shard 2), one real holdout doc (HOLD sentinel is NOT a real holdout by
    # itself, so REAL_HOLD is what the slice captures under phase="t").
    shards = [
        [{"content": DUP, "url": "u"}, {"content": DUP, "url": "u"},
         {"content": UNIQ, "url": "u"}, {"content": HOLD, "url": "u"}, {"content": REAL_HOLD, "url": "u"}],
        [{"content": UNIQ, "url": "u"}, {"content": "another unique body is here and long", "url": "u"}],
        [{"content": "another unique body is here and long", "url": "u"}],
    ]
    outs = []
    for par in (False, True):
        d = tempfile.mkdtemp()
        w = os.path.join(d, "w")
        os.makedirs(w)
        for i, ss in enumerate(shards):
            with open(os.path.join(w, f"w{i}__000.jsonl"), "w") as f:
                for doc in ss:
                    f.write(json.dumps(doc) + "\n")
        a = argparse.Namespace(
            domain="t", out=w, source=[], filters="light", no_near_dedup=True,
            workers=3 if par else 1, global_only=True, dry=False, exclude=[],
            limit=None, rg_mod=None, rg_idx=None, cache_dir=None, phase="t",
        )
        if par:
            B._parallel_exact_pass(a)
        else:
            B._global_pass(a)
        # H3 (e1): end-to-end --phase coverage -- a real held-out doc captured in the
        # pass, through _write_stats, into the frozen slice, row count == stamp's reason.
        sp = os.path.join(w, "holdout_slice_t.jsonl")
        assert os.path.exists(sp), f"phase='t' did not emit a holdout slice at {sp}"
        with open(sp, encoding="utf-8") as _sf:
            slice_rows = sum(1 for _ in _sf) - 1  # minus the header line
        with open(os.path.join(w, "build_corpus_stats.json"), encoding="utf-8") as _st:
            stats = json.load(_st)
        assert slice_rows == stats["reasons"]["holdout"], (
            f"slice rows {slice_rows} != stamp holdout reason {stats['reasons']['holdout']}"
        )
        merged = os.path.join(w, "t_000.jsonl")
        with open(merged, "rb") as f:
            got = f.read()  # raw bytes: byte-identity is an exact-file comparison
        outs.append((d, w, got))
    (d1, w1, serial) = outs[0]
    (d2, w2, parallel) = outs[1]
    assert serial == parallel, (
        f"BYTE MISMATCH ({len(serial)} vs {len(parallel)} bytes)\n"
        f"serial:   {serial[:200]}\nparallel: {parallel[:200]}"
    )
    # sanity: the cross-shard duplicate (UNIQ) appears exactly once across all
    # output lines, proving global exact-dedup ran (not per-shard only).
    uniq_hits = serial.count(b"def g(x): return x * 2")
    assert uniq_hits == 1, f"expected UNIQ deduped to 1 occurrence, got {uniq_hits}"
    n_lines = serial.count(b"\n")
    print(f"ok: serial==parallel byte-identical ({len(serial)}B, {n_lines} lines); UNIQ deduped to 1")
    shutil.rmtree(d1, ignore_errors=True)
    shutil.rmtree(d2, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
