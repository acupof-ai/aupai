#!/usr/bin/env python3
"""0e-8: HumanEval-stub domain from the static L3 survivors.

One document = the solution's last top-level function: its def line and
docstring (the task text is folded into a new docstring when the function
has none) followed directly by the body. No repeated signature, no
problem:/solution: scaffolding. Rows whose last top-level def carries a
docstring but no body are dropped.

The survivor set is the url(=uuid) set of data/corpus/code_ultra_l3_noexec_dc,
which is left untouched. Raw solution/task columns are read from the 147
UltraData-Code-L3 parquet shards and joined to that set.
"""
import argparse
import ast
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.decontam_ngram import Decontaminator, decontam_fp  # noqa: E402
from datagen.gen_exercises import _norm  # noqa: E402
from datagen.corpus_fingerprint import fp_dir  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corpus")
RAW = "/data00/aupai_raw/ultradata"
SOURCE = "openbmb/UltraData-Code-L3/py"
SHARD = 100 * 1024 * 1024


class Writer:
    def __init__(self, d, prefix):
        self.d, self.prefix, self.n, self.fh, self.b = d, prefix, 0, None, 0

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        size = len(line.encode())
        if self.fh is None or self.b + size > SHARD:
            if self.fh:
                self.fh.close()
            self.fh = open(os.path.join(self.d, f"{self.prefix}_{self.n:03d}.jsonl"), "w")
            self.n += 1
            self.b = 0
        self.fh.write(line)
        self.b += size

    def close(self):
        if self.fh:
            self.fh.close()


def last_top_func(tree):
    fn = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = node
    return fn


def docstring_no_body(fn):
    if ast.get_docstring(fn, clean=False) is None:
        return False
    real = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                       and isinstance(s.value.value, str))]
    return len(real) == 0


def make_stub(solution, task):
    """Return (stub_src, reason). reason in {ok, no_func, parse_fail, docstring_empty, body_empty}."""
    try:
        tree = ast.parse(solution)
    except (SyntaxError, ValueError):
        return None, "parse_fail"
    fn = last_top_func(tree)
    if fn is None:
        return None, "no_func"
    folded = False
    if ast.get_docstring(fn, clean=False) is None and task.strip():
        fn.body.insert(0, ast.Expr(ast.Constant(value=task.strip())))
        folded = True
    if docstring_no_body(fn):
        return None, "docstring_empty"
    # non-docstring real statements; pass/...-only bodies are dropped too
    real = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                       and isinstance(s.value.value, str))]
    if not real or all(isinstance(s, ast.Pass) or
                       (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                        and s.value.value is Ellipsis) for s in real):
        return None, "body_empty"
    try:
        src = ast.unparse(fn)
    except Exception:
        return None, "unparse_fail"
    return (src, folded), "ok"


def load_survivors(dc_dir):
    s = set()
    for f in sorted(glob.glob(os.path.join(dc_dir, "code_ultra_l3_noexec_[0-9]*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                s.add(json.loads(line)["url"])
    return s


def _shard_worker(args):
    part, survivors, out_dir, do_fold_decon = args
    import pyarrow.parquet as pq
    p = os.path.join(RAW, f"UltraData-Code-L3-py-part-{part:05d}-of-00147.parquet")
    decon = Decontaminator.load_default(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) \
        if do_fold_decon else None
    pf = pq.ParquetFile(p)
    stats = {"rows_in_shard": 0, "survivor": 0, "candidate": 0,
             "no_func": 0, "parse_fail": 0, "docstring_empty": 0, "body_empty": 0,
             "unparse_fail": 0, "folded_docstring": 0, "decontam": 0}
    out = os.path.join(out_dir, f"part_{part:03d}.jsonl")
    with open(out, "w") as w:
        for batch in pf.iter_batches(batch_size=2048, columns=["uuid", "task", "solution"]):
            cols = {n: batch.column(n).to_pylist() for n in batch.schema.names}
            for r in range(len(cols["uuid"])):
                stats["rows_in_shard"] += 1
                uid = cols["uuid"][r]
                if uid not in survivors:
                    continue
                stats["survivor"] += 1
                solution, task = cols["solution"][r] or "", cols["task"][r] or ""
                made, reason = make_stub(solution, task)
                if reason != "ok":
                    stats[reason] += 1
                    continue
                src, folded = made
                if decon is not None and decon.hit(src) is not None:
                    stats["decontam"] += 1
                    continue
                if folded:
                    stats["folded_docstring"] += 1
                w.write(json.dumps({"content": src, "source": SOURCE, "url": uid},
                                   ensure_ascii=False) + "\n")
                stats["candidate"] += 1
    return stats


def _stat_l3_shard(part):
    import pyarrow.parquet as pq
    p = os.path.join(RAW, f"UltraData-Code-L3-py-part-{part:05d}-of-00147.parquet")
    c = {"rows": 0, "parse_fail": 0, "no_func": 0, "docstring_no_body": 0}
    pf = pq.ParquetFile(p)
    for batch in pf.iter_batches(batch_size=4096, columns=["solution"]):
        for solution in batch.column("solution").to_pylist():
            c["rows"] += 1
            try:
                tree = ast.parse(solution or "")
            except (SyntaxError, ValueError):
                c["parse_fail"] += 1
                continue
            fn = last_top_func(tree)
            if fn is None:
                c["no_func"] += 1
                continue
            if docstring_no_body(fn):
                c["docstring_no_body"] += 1
    return c


def l3_solution_stats(workers):
    with mp.Pool(workers) as pool:
        parts = pool.map(_stat_l3_shard, range(1, 148))
    agg = {"shards": 147, "rows": 0, "parse_fail": 0, "no_func": 0, "docstring_no_body": 0}
    for c in parts:
        for k in ("rows", "parse_fail", "no_func", "docstring_no_body"):
            agg[k] += c[k]
    return agg


def _stat_file(f):
    c = {"rows": 0, "parse_fail": 0, "no_func": 0, "docstring_no_body": 0}
    with open(f) as fh:
        for line in fh:
            c["rows"] += 1
            try:
                tree = ast.parse(json.loads(line)["content"])
            except (SyntaxError, ValueError):
                c["parse_fail"] += 1
                continue
            fn = last_top_func(tree)
            if fn is None:
                c["no_func"] += 1
                continue
            if docstring_no_body(fn):
                c["docstring_no_body"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CORPUS, "code_ultra_l3_stub_dc"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    l3_dir = os.path.join(CORPUS, "code_ultra_l3_noexec_dc")
    l2_dir = os.path.join(CORPUS, "code_ultra_l2_dc")
    print("[stat] L3 raw-solution docstring-no-body (all 147 shards) ...", flush=True)
    l3stat = l3_solution_stats(args.workers)
    print("[stat] L2 natural-code docstring-no-body ...", flush=True)
    files = sorted(glob.glob(os.path.join(l2_dir, "code_ultra_l2_[0-9]*.jsonl")))
    with mp.Pool(args.workers) as pool:
        l2parts = pool.map(_stat_file, files)
    l2stat = {"files": len(files), "rows": 0, "parse_fail": 0, "no_func": 0,
              "docstring_no_body": 0}
    for c in l2parts:
        for k in ("rows", "parse_fail", "no_func", "docstring_no_body"):
            l2stat[k] += c[k]
    print("L3_STAT", json.dumps(l3stat), flush=True)
    print("L2_STAT", json.dumps(l2stat), flush=True)
    if args.stats_only:
        return

    os.makedirs(args.out, exist_ok=True)
    parts_dir = os.path.join(args.out, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    survivors = load_survivors(l3_dir)
    print(f"[build] survivors={len(survivors)}", flush=True)
    tasks = [(i, survivors, parts_dir, True) for i in range(1, 148)]
    total = {"rows_in_shard": 0, "survivor": 0, "candidate": 0,
             "no_func": 0, "parse_fail": 0, "docstring_empty": 0, "body_empty": 0,
             "unparse_fail": 0, "folded_docstring": 0, "decontam": 0}
    with mp.Pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(_shard_worker, tasks), 1):
            for k in total:
                total[k] += st[k]
            if i % 25 == 0:
                print(f"[build] {i}/147 candidates={total['candidate']}", flush=True)

    # Global exact-dedup pass over the per-shard parts, writing final 100MB shards.
    seen = set()
    kept = 0
    kept_tokens = 0
    w = Writer(args.out, "code_ultra_l3_stub_dc")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "tokenizer.json"))
    for f in sorted(glob.glob(os.path.join(parts_dir, "part_[0-9]*.jsonl"))):
        for line in open(f):
            rec = json.loads(line)
            sig = hashlib.sha1(_norm(rec["content"]).encode()).hexdigest()
            if sig in seen:
                continue
            seen.add(sig)
            kept += 1
            kept_tokens += len(tok.encode(rec["content"]).ids) + 1
            w.write(rec)
    w.close()
    n_shards = w.n
    for f in glob.glob(os.path.join(parts_dir, "part_[0-9]*.jsonl")):
        os.remove(f)
    os.rmdir(parts_dir)
    ngram_fp = decontam_fp(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "eval", "humaneval", "humaneval_164.jsonl"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "eval", "mbpp_holdouts.jsonl"))
    record = {
        "domain": "code_ultra_l3_stub_dc",
        "source": SOURCE,
        "survivor_source_dir": l3_dir,
        "stats_l3_docstring_no_body": l3stat,
        "stats_l2_docstring_no_body": l2stat,
        "kept": kept,
        "kept_tokens": kept_tokens,
        "tokens": kept_tokens,
        "tokens_status": "measured",
        "tokens_config": "data/tokenizer.json, exact per-doc ids+1",
        "filters": "last-top-func stub form + task-folded docstring when absent + drop docstring-no-body + 13gram decontam + global exact dedup",
        "n_shards": n_shards,
        "decontam_fp": ngram_fp,
        "fingerprint": fp_dir(args.out),
        "reasons": {**{k: total[k] for k in
                       ("survivor", "candidate", "no_func", "parse_fail", "docstring_empty",
                        "body_empty", "unparse_fail", "folded_docstring", "decontam")},
                    "written_after_global_dedup": kept},
    }
    json.dump(record, open(os.path.join(args.out, "build_corpus_stats.json"), "w"), indent=1)
    print(f"STUB_KEPT docs={kept} tokens={kept_tokens} shards={n_shards} "
          f"folded={total['folded_docstring']} dropped_docstring_empty={total['docstring_empty']} "
          f"fp={record['fingerprint']}", flush=True)


if __name__ == "__main__":
    main()
