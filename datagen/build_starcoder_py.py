#!/usr/bin/env python3
"""Build data/corpus/code_py_starcoder from the fetched starcoderdata python parquet
(fb P0 2026-09-01). ast.parse(t) = language ID AND syntax filter; a labelled source
loses a few % to syntax, not 94% to language. Per-shard cumulative rows/tokens, written
to a NEW dir (never a ladder-mix dir).
# restartable: idempotent via .built_shards, appended per-shard (e1); an interrupt loses
# at most the current raw-parquet shard. STAGE is cleared at start (e1) so a killed run's
# orphan jsonl can never double-enter DST.

Parallelism (fb 2026-09-01): mp.Pool(16) parallelizes ONLY _process_shard (ast+tokenize);
the ShardWriter stays a single SERIAL writer in the master, so jsonl numbering never
races. Fresh source -> empty allow_empty holdout slice + _write_stats stamp (fingerprint).

Reconciliation (3b 2026-09-02): the delivered corpus is stamped parallel (workers=16,
fingerprint e1a14839 -- the mp.Pool version, which produced 6,180,174 rows). The serial
a264716 variant added the is_holdout filter + renumber-publish but never produced a corpus.
This keeps the parallel build that is provenance-of-record AND carries the holdout filter
+ renumber-publish a264716 added, so a clean rebuild is both fast and holdout-clean."""
import ast
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import build_corpus as B  # noqa: E402

RAW = "/work/aupai/data/raw/ms_starcoder_py"
DST = "/work/aupai/data/corpus/code_py_starcoder"
PHASE = "code_py_starcoder"
DONE = os.path.join(DST, ".built_shards")
STAGE = os.path.join(DST, ".stage")  # cleared each run; publish renumbers-append, no delete
NJOBS = 16


def _process_shard(args):
    """Collect (content, ntok) for the rows that ast.parse cleanly; NO writer here --
    the master ShardWriter stays serial, so numbering cannot race."""
    path, = args
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file("/work/aupai/data/tokenizer.json")
    out = []
    col = None
    try:
        pf = pq.ParquetFile(path)
        for c in pf.schema_arrow.names:
            if c in ("content", "text", "code"):
                col = c
                break
        if col is None:
            return [], 0, "no-content-col"
        for batch in pf.iter_batches(batch_size=1024, columns=[col]):
            for v in batch.to_pydict()[col]:
                if v is None:
                    continue
                t = str(v)
                if not t.strip():
                    continue
                try:
                    ast.parse(t)
                except (SyntaxError, MemoryError, RecursionError, ValueError):
                    continue
                ntok = len(tk.encode(t).ids)
                out.append((t, ntok))
    except Exception as e:
        return [], 0, f"err:{type(e).__name__}:{str(e)[:60]}"
    print(f"  parswave {os.path.basename(path)[:30]} parsed {len(out)} rows", flush=True)
    return out, 0, "ok"


def _built_set():
    if not os.path.exists(DONE):
        return set()
    return set(open(DONE).read().split())


def main():
    shards = sorted(glob.glob(os.path.join(RAW, "train-*.parquet")))
    todo = [p for p in shards if os.path.basename(p) not in _built_set()]
    if not todo:
        print(json.dumps({"ndone": len(_built_set()), "note": "nothing new"}))
        return
    shutil.rmtree(STAGE, ignore_errors=True)  # e1: clear orphan jsonl from a killed run
    os.makedirs(STAGE, exist_ok=True)
    B._LOCK_FD = B._build_lock(DST)
    w = B.ShardWriter(STAGE, "code_py_starcoder")  # single serial writer, stage dir
    rows_keep = tok_keep = 0
    held_out = []
    done = list(_built_set())
    with mp.Pool(NJOBS) as pool:
        # imap: streaming, memory-bounded; results arrive in todo order
        for (p,), res in zip([(p,) for p in todo], pool.imap(_process_shard, [(p,) for p in todo])):
            shard = os.path.basename(p)
            out, _, err = res
            sh_rows = sh_tok = 0
            for t, ntok in out:
                if B.is_holdout(t):
                    held_out.append(B.exact_key(t))
                    continue
                w.write({"content": t, "source": "starcoderdata:python", "url": shard})
                rows_keep += 1
                tok_keep += ntok
                sh_rows += 1
                sh_tok += ntok
            if err:
                print(f"WARN {shard}: {err}", flush=True)
            done.append(shard)
            with open(DONE, "a") as f:  # per-shard append (e1): a kill keeps prior progress
                f.write(shard + "\n")
            print(f"{shard}: rows_kept={sh_rows} tokens_kept={sh_tok} "
                  f"cumulative_tokens={tok_keep}", flush=True)
    w.close()
    # publish: renumber staged jsonl to continue after DST's existing max, NEVER delete
    existing = sorted(glob.glob(os.path.join(DST, "code_py_starcoder_*.jsonl")))
    nxt = int(os.path.basename(existing[-1]).split("_")[-1].split(".")[0]) + 1 if existing else 0
    for sp in sorted(glob.glob(os.path.join(STAGE, "code_py_starcoder_*.jsonl"))):
        os.replace(sp, os.path.join(DST, f"code_py_starcoder_{nxt:03d}.jsonl"))
        nxt += 1
    from collections import Counter

    B._emit_holdout_slice(DST, PHASE, held_out, allow_empty=True)
    B._write_stats(DST, "code_py_starcoder",
                   B.argparse.Namespace(domain="code_py_starcoder", workers=NJOBS, phase=PHASE, allow_empty_slice=True,
                                        filters="starcoder-python-ast", no_near_dedup=True),
                   Counter({"kept": rows_keep}), rows_keep, 0,
                   len(glob.glob(os.path.join(DST, "code_py_starcoder_*.jsonl"))), held_out)
    print(json.dumps({"rows": rows_keep, "tokens_lenids": tok_keep, "shards": len(shards),
                      "stamp": os.path.join(DST, "build_corpus_stats.json")}))


if __name__ == "__main__":
    main()