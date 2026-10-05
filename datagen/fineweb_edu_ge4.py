#!/usr/bin/env python3
"""fineweb-edu sample/350BT -> w*_*.jsonl kept shards (int_score >= 4).

Stage 1 of the standard build: this applies the score cut plus build_corpus's
own `reject_light` (short/long/bad_bytes/holdout) and per-worker exact dedup,
writing worker shards byte-compatible with `build_corpus.py --global-only`,
which then runs the global MinHash near-dedup + holdout + fingerprint stamp.
Why a separate stage: the edu corpus is a SUPERSET already filtered at score 3;
the int_score>=4 cut has to read a column iter_parquet never projects.

Parquet columns: text,url,int_score (verified on sample/350BT, 2026-10-05).
--delete_raw removes each parquet as its kept rows land, so peak disk is
shards-plus-24-parquet instead of the full 560 GB download.
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datagen"))

import build_corpus as B  # noqa: E402  (reject_light, exact_key, ShardWriter, SPECIAL_TOKEN)


def process_slice(job):
    files, out, prefix, delete_raw = job
    import pyarrow.parquet as pq

    reject = B.reject_light
    exact = set()
    tot = {"seen": 0, "below4": 0, "reject": 0, "kept": 0, "kept_chars": 0}
    reject_reasons = {}
    w = B.ShardWriter(out, prefix)  # ONE writer for the whole slice -> unique shard numbers
    for fi, path in enumerate(files):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000, columns=["text", "url", "int_score"]):
            d = batch.to_pydict()
            for text, url, score in zip(d["text"], d["url"], d["int_score"]):
                tot["seen"] += 1
                if score is None or score < 4:
                    tot["below4"] += 1
                    continue
                text = B.SPECIAL_TOKEN.sub("", text or "").strip()
                why = reject(text)
                if why is None:
                    k = B.exact_key(text)
                    if k in exact:
                        why = "exact_dup"
                    else:
                        exact.add(k)
                if why:
                    tot["reject"] += 1
                    reject_reasons[why] = reject_reasons.get(why, 0) + 1
                    continue
                tot["kept"] += 1
                tot["kept_chars"] += len(text)
                w.write({"content": text, "source": os.path.basename(path).split(".")[0], "url": url})
        print(
            f"[{prefix.rstrip('_')}] {fi + 1}/{len(files)} {os.path.basename(path)} "
            f"| kept {tot['kept']} | ~{tot['kept_chars'] / B.CHARS_PER_TOKEN / 1e9:.2f}B tok",
            flush=True,
        )
        if delete_raw:
            os.remove(path)
    w.close()
    return {**tot, "reasons": reject_reasons, "prefix": prefix, "n_files": len(files)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_glob", required=True, help="parquet glob, e.g. /data00/fwe350/*.parquet")
    ap.add_argument("--out", required=True, help="worker-shard dir (mkdir is the operator's step)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--delete_raw", action="store_true")
    ap.add_argument("--limit_files", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(args.raw_glob))
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        sys.exit(f"no parquet under {args.raw_glob}")
    os.makedirs(args.out, exist_ok=True)
    if glob.glob(os.path.join(args.out, "w*_*.jsonl*")):
        sys.exit(f"REFUSE worker shards already in {args.out} -- wipe first")

    # disjoint FILE slices, one process and one writer each (same contract as _clean_piece)
    slices = [[] for _ in range(args.workers)]
    for i, p in enumerate(files):
        slices[i % args.workers].append(p)
    jobs = [(grp, args.out, f"w{wi}_", args.delete_raw) for wi, grp in enumerate(slices) if grp]
    print(f"{len(files)} parquet across {len(jobs)} workers", flush=True)

    tot = {"seen": 0, "below4": 0, "reject": 0, "kept": 0, "kept_chars": 0}
    with mp.Pool(len(jobs)) as pool:
        for r in pool.imap_unordered(process_slice, jobs):
            for k in ("seen", "below4", "reject", "kept", "kept_chars"):
                tot[k] += r[k]
            print(
                f"slice {r['prefix'].rstrip('_')} done ({r['n_files']} files): {r['reasons']}",
                flush=True,
            )
    print(
        f"stage1 done: seen {tot['seen']} below4 {tot['below4']} reject {tot['reject']} "
        f"kept {tot['kept']} (~{tot['kept_chars'] / B.CHARS_PER_TOKEN / 1e9:.2f}B tok)"
    )


if __name__ == "__main__":
    main()
