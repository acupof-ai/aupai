#!/usr/bin/env python3
"""Convert UltraData-Code python shards to repo corpus shards.

    python3 datagen/ultradata_shards.py --level L3 --first 1 --last 3

Document field: L3=full_content (the dataset's own task/analysis/solution/
test assembly), L2=content (the code text). Decontamination against
HumanEval+MBPP reuses the 0e-1 harness (datagen/gen_exercises.py): a missing
benchmark file is a loud SystemExit and the planted HumanEval/0 control must
be caught on the main path. MultiPL-E's python split IS HumanEval, so the
HumanEval file covers the python slice of MultiPL-E. Exact dedup is by
normalized-content hash. Tokens are counted EXACTLY with the frozen tokenizer
(per-doc ids + one <eos>, the code_rp1t convention).

Output: data/corpus/code_ultra_<level>/<prefix>_NNN.jsonl (100MB shards,
{"content","source","url"}) plus build_corpus_stats.json.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datagen.gen_exercises import _norm, decontam, load_benchmarks, planted_control

DOC_FIELD = {"L2": "content", "L3": "full_content"}
N_SHARDS = {"L2": 119, "L3": 147}
SHARD_BYTES = 100 * 1024 * 1024


def shard_name(level, i):
    return f"UltraData-Code-{level}-py-part-{i:05d}-of-{N_SHARDS[level]:05d}.parquet"


class ShardWriter:
    def __init__(self, out_dir, prefix, limit=SHARD_BYTES):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix, self.limit = out_dir, prefix, limit
        self.n = 0
        self.fh = None
        self.bytes = 0

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        size = len(line.encode())
        if self.fh is None or self.bytes + size > self.limit:
            if self.fh:
                self.fh.close()
            self.fh = open(os.path.join(self.out_dir, f"{self.prefix}_{self.n:03d}.jsonl"),
                           "w", encoding="utf-8")
            self.n += 1
            self.bytes = 0
        self.fh.write(line)
        self.bytes += size

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


def fp_of(*paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["L2", "L3"])
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--raw", default="data/raw/ultradata")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--limit-rows", type=int, default=0, help="stop after N rows (dry run)")
    args = ap.parse_args()
    out = args.out or f"data/corpus/code_ultra_{args.level.lower()}"
    prefix = os.path.basename(out.rstrip("/"))
    for stale in glob.glob(os.path.join(out, f"{prefix}_*.jsonl")):
        os.remove(stale)

    bench = load_benchmarks()
    control = planted_control(bench)
    print(f"decontam OK: {len(bench)} benchmark rows, planted control {control} caught", flush=True)
    tok = Tokenizer.from_file(args.tokenizer)

    field = DOC_FIELD[args.level]
    seen = set()
    stats = {"kept": 0, "decontam": 0, "dup": 0, "empty": 0}
    kept_chars = 0
    kept_tokens = 0
    total = 0
    writer = ShardWriter(out, prefix)

    for i in range(args.first, args.last + 1):
        path = os.path.join(args.raw, shard_name(args.level, i))
        if not os.path.exists(path):
            print(f"MISSING {path} -- run fetch_ultradata.py first", flush=True)
            continue
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=8192, columns=None):
            cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
            for r in range(batch.num_rows):
                total += 1
                doc = cols[field][r]
                if not doc or not doc.strip():
                    stats["empty"] += 1
                    continue
                if decontam({"prompt": doc, "output": ""}, bench):
                    stats["decontam"] += 1
                    continue
                sig = hashlib.sha1(_norm(doc).encode()).hexdigest()
                if sig in seen:
                    stats["dup"] += 1
                    continue
                seen.add(sig)
                if args.level == "L3":
                    source, url = "openbmb/UltraData-Code-L3/py", cols["uuid"][r]
                else:
                    source, url = f"github:{cols['repo_name'][r]}", cols["relative_path"][r]
                writer.write({"content": doc, "source": source, "url": url})
                stats["kept"] += 1
                kept_chars += len(doc)
                kept_tokens += len(tok.encode(doc).ids) + 1
                if total % 100000 == 0:
                    print(f"[{i}] rows={total} kept={stats['kept']} decontam={stats['decontam']} "
                          f"dup={stats['dup']} empty={stats['empty']}", flush=True)
                if args.limit_rows and total >= args.limit_rows:
                    break
            if args.limit_rows and total >= args.limit_rows:
                break
        print(f"done shard {i}: rows={total} kept={stats['kept']}", flush=True)
    writer.close()

    fingerprint = fp_of(*sorted(os.path.join(out, f) for f in os.listdir(out)
                                if f.endswith(".jsonl")))
    record = {
        "domain": os.path.basename(out),
        "source": f"openbmb/UltraData-Code/{args.level}/py shards {args.first}-{args.last} "
                  f"of {N_SHARDS[args.level]}",
        "kept": stats["kept"],
        "kept_chars": kept_chars,
        "kept_tokens": kept_tokens,
        "tokens": kept_tokens,
        "tokens_status": "measured",
        "tokens_config": f"{args.tokenizer}, exact per-doc ids + one <eos> per doc "
                         "(code_rp1t convention)",
        "filters": "decontam(humaneval,mbpp)+exact-dedup",
        "workers": 1,
        "n_shards": writer.n,
        "filters_fp": fp_of(__file__, sys.modules["datagen.gen_exercises"].__file__),
        "fingerprint": fingerprint,
        "near_dedup": False,
        "near_dedup_note": "exact dedup only; near-dedup not run at small scale",
        "total_rows": total,
        "reasons": dict(stats),
    }
    with open(os.path.join(out, "build_corpus_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    print(f"DONE {json.dumps(record, indent=1)}", flush=True)


if __name__ == "__main__":
    main()
