#!/usr/bin/env python3
"""Build data/corpus/code_py_starcoder from the fetched starcoderdata python parquet
(fb P0 2026-09-01): the labelled-Python source. ast.parse(t) = language-ID AND syntax
filter. Per-shard cumulative rows+tokens, written to a NEW dir (never a ladder-mix dir).
# restartable: idempotent via .built_shards (written at the end of each run); a kill
# loses at most the current raw parquet shard, and re-running re-processes it. Fresh
# source, empty holdout slice (allow_empty)."""
import ast
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import build_corpus as B  # noqa: E402

RAW = "/work/aupai/data/raw/ms_starcoder_py"
DST = "/work/aupai/data/corpus/code_py_starcoder"
PHASE = "code_py_starcoder"
DONE = os.path.join(DST, ".built_shards")
STAGE = os.path.join(DST, ".stage")  # per-run staging; jsonl move into DST only on completion,
# so an interrupt never re-clobbers prior output (ShardWriter numbers jsonl from 0).


def built_set():
    if not os.path.exists(DONE):
        return set()
    return set(open(DONE).read().split())


def _process_shard(args):
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
        return [], 0, f"err:{type(e).__name__}"
    return out, 0, "ok"


def main():
    built = built_set()
    shards = sorted(glob.glob(os.path.join(RAW, "train-*.parquet")))
    todo = [p for p in shards if os.path.basename(p) not in built]
    if not todo:
        print(json.dumps({"ndone": len(built), "note": "nothing new to build"}))
        return
    B._LOCK_FD = B._build_lock(DST)
    os.makedirs(DST, exist_ok=True)
    os.makedirs(STAGE, exist_ok=True)
    w = B.ShardWriter(STAGE, "code_py_starcoder")  # stage; move to DST only on completion
    rows_keep = 0
    tok_keep = 0
    done = list(built)
    held_out = []
    per_shard = []
    for args in [(p,) for p in todo]:
        res, _, _ = _process_shard(args)
        # single-process per shard is enough (ast.parse is fast); report as land
        shard = os.path.basename(args[0])
        sh_rows = 0
        sh_tok = 0
        for t, ntok in res:
            tok_keep += ntok
            rows_keep += 1
            sh_rows += 1
            sh_tok += ntok
            if B.is_holdout(t):
                held_out.append(B.exact_key(t))
                continue
            w.write({"content": t, "source": "starcoderdata:python", "url": shard})
        per_shard.append({"shard": shard, "rows": sh_rows, "tokens": sh_tok})
        print(f"{shard}: rows_kept={sh_rows} tokens_kept={sh_tok} "
              f"cumulative_tokens={tok_keep}", flush=True)
        done.append(shard)
    w.close()
    # publish: renumber staged jsonl to continue AFTER DST's existing max, NEVER delete.
    # A resume's todo (via .built_shards) only has the remaining shards, so STAGE holds
    # just the new rows; deleting DST's prior output would clear done work and then
    # .built_shards claims it complete (fb's catch, 2026-09-01). Appending at the next
    # index across resumes keeps prior shards' rows and only adds the new ones.
    existing = sorted(glob.glob(os.path.join(DST, "code_py_starcoder_*.jsonl")))
    nxt = int(os.path.basename(existing[-1]).split("_")[-1].split(".")[0]) + 1 if existing else 0
    for sp in sorted(glob.glob(os.path.join(STAGE, "code_py_starcoder_*.jsonl"))):
        os.replace(sp, os.path.join(DST, f"code_py_starcoder_{nxt:03d}.jsonl"))
        nxt += 1
    with open(DONE, "w") as f:
        f.write("\n".join(done) + "\n")
    B._emit_holdout_slice(DST, PHASE, held_out, allow_empty=True)
    from collections import Counter

    reasons = Counter({"kept": rows_keep})
    B._write_stats(DST, "code_py_starcoder",
                   B.argparse.Namespace(domain="code_py_starcoder", workers=1, phase=PHASE, allow_empty_slice=True,
                                        filters="starcoder-python-ast", no_near_dedup=True),
                   reasons, rows_keep, 0, len(glob.glob(os.path.join(DST, "code_py_starcoder_*.jsonl"))), held_out)
    print(json.dumps({"new_shards_built": len(todo), "cum_rows": rows_keep, "cum_tokens": tok_keep,
                      "total_built": len(done)}))


if __name__ == "__main__":
    main()
