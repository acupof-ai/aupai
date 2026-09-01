#!/usr/bin/env python3
"""Build data/corpus/code_py_rp1t (fb ruling 2026-09-01): the code_rp1t ast.parse
survivors (0.42B / 209,668 rows) as a SECOND code domain referenced by mix_500m. The
0.42B parse-verified from code_rp1t is not thrown away; it joins starcoder. Same filter
(ast.parse = language ID + syntax), same --phase holdout slice (fresh-source empty,
allow_empty programmatic). New dir, nothing in a mix_scale_* domain.
RESTARTABILITY, and it is real now rather than asserted. `.built_shards` is appended
after EVERY source shard, and the ShardWriter is re-opened past the shards already on
disk, so a kill costs at most the source shard in flight.

It did not work that way before (e1, 2026-09-01). `.built_shards` was written once,
after all 235 shards, so a kill at shard 212 left the file absent -- the next run saw
an empty `built` set, redid all 235, and ShardWriter restarted numbering at 0 and
OVERWROTE the eleven jsonl already there. The docstring said "a kill loses at most the
current raw source shard". It lost everything and corrupted the output.

The comment was fixed by changing the CODE to match it, not by weakening the sentence:
the next person reads the comment, not the loop."""
import ast
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import build_corpus as B  # noqa: E402

SRC = "/work/aupai/data/corpus/code_rp1t"
DST = "/work/aupai/data/corpus/code_py_rp1t"
PHASE = "code_py_rp1t"
DONE = os.path.join(DST, ".built_shards")


def main():
    built = set()
    if os.path.exists(DONE):
        built = set(open(DONE).read().split())
    shards = [p for p in sorted(glob.glob(os.path.join(SRC, "code_rp1t_*.jsonl")))
              if os.path.basename(p) != "build_corpus_stats.json"]
    todo = [p for p in shards if os.path.basename(p) not in built]
    if not todo:
        print(json.dumps({"ndone": len(built), "note": "nothing new"}))
        return
    B._LOCK_FD = B._build_lock(DST)
    os.makedirs(DST, exist_ok=True)
    # Resume past what is already on disk. ShardWriter numbers from 0, so on a restart
    # it would otherwise reopen code_py_rp1t_000.jsonl and overwrite completed output.
    w = B.ShardWriter(DST, "code_py_rp1t")
    existing = sorted(glob.glob(os.path.join(DST, "code_py_rp1t_*.jsonl")))
    if existing and built:
        nxt = max(int(os.path.basename(p).split("_")[-1].split(".")[0])
                  for p in existing) + 1
        w.n = nxt  # ShardWriter's counter is `n` (build_corpus.py:275), not `idx`
        print(f"resuming: {len(built)} source shard(s) done, writing from shard {nxt}",
              flush=True)
    rows_keep = 0
    held_out = []
    done = list(built)
    for p in todo:
        shard = os.path.basename(p)
        sh_rows = 0
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("content") or ""
            if not t:
                continue
            try:
                ast.parse(t)
            except (SyntaxError, MemoryError, RecursionError, ValueError):
                continue
            if B.is_holdout(t):
                held_out.append(B.exact_key(t))
                continue
            rows_keep += 1
            sh_rows += 1
            w.write(d)
        print(f"{shard}: rows_kept={sh_rows} cumulative={rows_keep}", flush=True)
        done.append(shard)
        # Appended per source shard, not once at the end: a kill between here and the
        # final write used to discard every shard already processed.
        with open(DONE, "a") as f:
            f.write(shard + "\n")
    w.close()
    from collections import Counter

    B._emit_holdout_slice(DST, PHASE, held_out, allow_empty=True)
    B._write_stats(DST, "code_py_rp1t",
                   B.argparse.Namespace(domain="code_py_rp1t", workers=1, phase=PHASE, allow_empty_slice=True,
                                        filters="rp1t-python-ast", no_near_dedup=True),
                   Counter({"kept": rows_keep}), rows_keep, 0,
                   len(glob.glob(os.path.join(DST, "code_py_rp1t_*.jsonl"))), held_out)
    print(json.dumps({"phase": PHASE, "src": SRC, "dst": DST, "rows": rows_keep}))


if __name__ == "__main__":
    main()