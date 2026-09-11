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
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datagen.gen_exercises import _norm, decontam, load_benchmarks, planted_control
from datagen.ud_solution_exec import FAIL, PASS, TIMEOUT, execute, nontrivial

DOC_FIELD = {"L2": "content", "L3": "full_content"}
N_SHARDS = {"L2": 119, "L3": 147}
DROP_CATEGORIES = {"CONFIG", "TEST"}  # fb ruling 2026-09-10, 3b's category audit
SHARD_BYTES = 100 * 1024 * 1024


def shard_name(level, i):
    return f"UltraData-Code-{level}-py-part-{i:05d}-of-{N_SHARDS[level]:05d}.parquet"


class ShardWriter:
    def __init__(self, out_dir, prefix, limit=SHARD_BYTES, tag=""):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix, self.limit, self.tag = out_dir, prefix, limit, tag
        self.n = 0
        self.fh = None
        self.bytes = 0

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        size = len(line.encode())
        if self.fh is None or self.bytes + size > self.limit:
            if self.fh:
                self.fh.close()
            fn = f"{self.prefix}{self.tag}_{self.n:03d}.jsonl"
            self.fh = open(os.path.join(self.out_dir, fn), "w", encoding="utf-8")
            self.n += 1
            self.bytes = 0
        self.fh.write(line)
        self.bytes += size

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


def _exec_pair(pair):
    solution, test = pair
    return execute(solution, test)[0], nontrivial(solution)


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
    ap.add_argument("--exec-workers", type=int, default=32,
                    help="L3 solution-test sandbox concurrency (0e-3 fb ruling filter)")
    ap.add_argument("--tag", default="",
                    help="shard filename tag (parallel shard groups); '' = single writer")
    ap.add_argument("--stats-name", default="build_corpus_stats.json")
    ap.add_argument("--aggregate", default="",
                    help="aggregate the group stats_<tag>.json files in out and exit")
    args = ap.parse_args()
    out = args.out or f"data/corpus/code_ultra_{args.level.lower()}"
    prefix = os.path.basename(out.rstrip("/"))

    if args.aggregate:
        aggregate(out, args.aggregate, prefix, args.tokenizer, args.level)
        return

    tag = f"_{args.tag}" if args.tag else ""
    stale_pat = os.path.join(out, f"{prefix}{tag}_*.jsonl") if args.tag else \
        os.path.join(out, f"{prefix}_*.jsonl")
    for stale in glob.glob(stale_pat):
        os.remove(stale)

    bench = load_benchmarks()
    control = planted_control(bench)
    print(f"decontam OK: {len(bench)} benchmark rows, planted control {control} caught", flush=True)
    tok = Tokenizer.from_file(args.tokenizer)

    field = DOC_FIELD[args.level]
    seen = set()
    stats = {"kept": 0, "decontam": 0, "dup": 0, "empty": 0, "category_drop": 0,
             "exec_fail": 0, "exec_timeout": 0, "non_trivial": 0}
    kept_chars = 0
    kept_tokens = 0
    total = 0
    writer = ShardWriter(out, prefix, tag=tag)
    exec_pool = ThreadPoolExecutor(max_workers=args.exec_workers) if args.level == "L3" else None

    for i in range(args.first, args.last + 1):
        path = os.path.join(args.raw, shard_name(args.level, i))
        if not os.path.exists(path):
            raise SystemExit(f"MISSING {path} -- run fetch_ultradata.py first; "
                             "refusing to emit a zero-row stats for an unread shard")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048 if args.level == "L3" else 8192,
                                     columns=None):
            cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
            # Cheap filters first; L3 survivors are batch-executed against their own
            # bundled test with the one predicate shared with 3b's audit (ud_solution_exec).
            survivors = []
            for r in range(batch.num_rows):
                total += 1
                doc = cols[field][r]
                if not doc or not doc.strip():
                    stats["empty"] += 1
                elif args.level == "L2" and cols["category"][r] in DROP_CATEGORIES:
                    stats["category_drop"] += 1
                elif decontam({"prompt": doc, "output": ""}, bench):
                    stats["decontam"] += 1
                else:
                    sig = hashlib.sha1(_norm(doc).encode()).hexdigest()
                    if sig in seen:
                        stats["dup"] += 1
                    else:
                        seen.add(sig)
                        survivors.append(r)
                if args.limit_rows and total >= args.limit_rows:
                    break
            if exec_pool is not None and survivors:
                pairs = [(cols["solution"][r], cols["test"][r]) for r in survivors]
                verdicts = list(exec_pool.map(_exec_pair, pairs))
            else:
                verdicts = [(PASS, True)] * len(survivors)
            for r, (verdict, nt) in zip(survivors, verdicts, strict=True):
                if verdict == TIMEOUT:
                    stats["exec_timeout"] += 1
                    continue
                if verdict == FAIL:
                    stats["exec_fail"] += 1
                    continue
                if not nt:
                    stats["non_trivial"] += 1
                    continue
                doc = cols[field][r]
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
                      f"dup={stats['dup']} empty={stats['empty']} cat={stats['category_drop']} "
                      f"exec_fail={stats['exec_fail']} exec_timeout={stats['exec_timeout']} "
                      f"non_trivial={stats['non_trivial']}", flush=True)
            if args.limit_rows and total >= args.limit_rows:
                break
        print(f"done shard {i}: rows={total} kept={stats['kept']}", flush=True)
    if exec_pool is not None:
        exec_pool.shutdown()
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
        "filters": ("decontam(humaneval,mbpp)+exact-dedup+exec-pass+non-triviality"
                    if args.level == "L3"
                    else "decontam(humaneval,mbpp)+exact-dedup+drop-CONFIG,TEST"),
        "workers": args.exec_workers if args.level == "L3" else 1,
        "n_shards": writer.n,
        "filters_fp": fp_of(__file__, sys.modules["datagen.gen_exercises"].__file__,
                            sys.modules["datagen.ud_solution_exec"].__file__),
        "fingerprint": fingerprint,
        "near_dedup": False,
        "near_dedup_note": "exact dedup only; near-dedup not run at small scale",
        "total_rows": total,
        "reasons": dict(stats),
    }
    with open(os.path.join(out, args.stats_name), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    print(f"DONE {json.dumps(record, indent=1)}", flush=True)


def aggregate(out, pattern, prefix, tokenizer_path, level):
    """Sum group stats, then: (1) ae's 13-gram solution-body decontamination
    against HE+MBPP prompts+solutions+tests, (2) the GLOBAL exact-dedup pass
    groups cannot do. Survivors are re-emitted as final prefix_NNN shards with
    exact token recount; tagged intermediates removed only after. Bytes are not
    mix-legal until this runs (fb ruling 2026-09-11).
    """
    paths = sorted(glob.glob(os.path.join(out, pattern)))
    if not paths:
        raise SystemExit(f"aggregate: no stats match {pattern} in {out}")
    records = [json.load(open(p, encoding="utf-8")) for p in paths]
    reasons = {}
    for r in records:
        for k, v in r.get("reasons", {}).items():
            reasons[k] = reasons.get(k, 0) + v
    group_kept = sum(r.get("kept", 0) for r in records)

    # this file lives at <root>/datagen/ultradata_shards.py -> root is one up
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        from filters.decontam_ngram import Decontaminator
    except ImportError as e:
        raise SystemExit(
            "aggregate requires filters/decontam_ngram.py (ae 13-gram decontam, "
            f"fb ruling 2026-09-11) on the import path: {e}")
    decon = Decontaminator.load_default(root)

    tagged = sorted(glob.glob(os.path.join(out, f"{prefix}_g??_*.jsonl")))
    if not tagged:
        raise SystemExit(f"aggregate: no tagged group shards {prefix}_g??_*.jsonl in {out}")
    tok = Tokenizer.from_file(tokenizer_path)
    for stale in glob.glob(os.path.join(out, f"{prefix}_[0-9][0-9][0-9].jsonl")):
        os.remove(stale)
    seen = set()
    kept = 0
    kept_chars = 0
    kept_tokens = 0
    scanned = 0
    ngram_drop = 0
    ngram_problems = set()
    ngram_parts = {}
    cross_dup = 0
    writer = ShardWriter(out, prefix, tag="")
    for shard in tagged:
        for line in open(shard, encoding="utf-8"):
            scanned += 1
            rec = json.loads(line)
            # L3 full_content already joins task+analysis+solution+test; L2 is
            # the code doc itself. One assembled string, per the gate key.
            hit = decon.hit(rec["content"])
            if hit is not None:
                ngram_drop += 1
                ngram_problems.add(str(hit.get("problem", "?")))
                part = str(hit.get("part", "unknown"))
                ngram_parts[part] = ngram_parts.get(part, 0) + 1
                continue
            sig = hashlib.sha1(_norm(rec["content"]).encode()).hexdigest()
            if sig in seen:
                cross_dup += 1
                continue
            seen.add(sig)
            writer.write(rec)
            kept += 1
            kept_chars += len(rec["content"])
            kept_tokens += len(tok.encode(rec["content"]).ids) + 1
    writer.close()
    assert scanned == group_kept, (
        f"tagged shards hold {scanned} rows, group stats say kept {group_kept}")
    for shard in tagged:
        os.remove(shard)

    # Partition invariant: final kept + rejects + cross-group dup + ngram drop
    # == total input rows.
    reasons["kept"] = kept
    reasons["cross_group_dup"] = cross_dup
    reasons["decontam_ngram"] = ngram_drop
    try:
        ngram_fp = decon.decontam_fp()
    except AttributeError:
        ngram_fp = None
    final_shards = sorted(glob.glob(os.path.join(out, f"{prefix}_[0-9]*.jsonl")))
    canonical = {
        "domain": os.path.basename(out.rstrip("/")),
        "level": level,
        "source": records[0].get("source", "").split(" shards ")[0]
                  + f" shards, {len(records)} parallel groups",
        "kept": kept,
        "kept_chars": kept_chars,
        "kept_tokens": kept_tokens,
        "tokens": kept_tokens,
        "tokens_status": "measured",
        "tokens_config": f"{tokenizer_path}, exact per-doc ids + one <eos> per doc, "
                         "recounted after n-gram decontam + global dedup",
        "filters": records[0].get("filters", "") + "+ngram13-decontam+global-exact-dedup",
        "workers": records[0].get("workers", 1),
        "n_shards": len(final_shards),
        "filters_fp": records[0].get("filters_fp", ""),
        "decontam_ngram_fp": ngram_fp,
        "decontam_ngram": {
            "rows_in": scanned,
            "rows_dropped": ngram_drop,
            "distinct_problems_hit": len(ngram_problems),
            "by_part": ngram_parts,
            "problems": sorted(ngram_problems),
        },
        "fingerprint": fp_of(*final_shards),
        "near_dedup": False,
        "near_dedup_note": "exact dedup global across all groups; near-dedup not run",
        "total_rows": sum(r.get("total_rows", 0) for r in records),
        "reasons": reasons,
        "groups": [os.path.basename(p) for p in paths],
    }
    with open(os.path.join(out, "build_corpus_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(canonical, fh, indent=1)
    print(f"AGGREGATED {len(records)} groups: {json.dumps(canonical, indent=1)}", flush=True)


if __name__ == "__main__":
    main()
