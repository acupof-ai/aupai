#!/usr/bin/env python3
"""Starcoder python source-shape check (fb P0 2026-09-01). Two cheap measurements on
data already on disk, while the build runs:
  1. FAILING-reason histogram of rows that fail ast.parse (~200 sample): is `unexpected
     EOF` prominent (= rows are chunked/fragments) vs py2-ism / invalid-syntax / not-
     Python? ast.parse cannot detect truncation in the SURVIVING half, so a chunked
     source would be passing fragments -- the one way starcoder could be worse than it
     looks at a projected 34% of the mix.
  2. LENGTH percentiles of SURVIVING rows: real Python files are long-tailed (median
     ~hundreds of lines); a tight uniform cluster means rows were cut to a length.
If unexpected-EOF is prominent OR lengths cluster, fb stops before the mix; otherwise the
52% is py2/non-code and the source is what it says."""
import ast
import json
import random
from collections import Counter

random.seed(17)
SRC = "/work/aupai/data/raw/ms_starcoder_py/train-00000-of-00059.parquet"


def main():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file("/work/aupai/data/tokenizer.json")
    pqf = pq.ParquetFile(SRC)
    col = next(c for c in pqf.schema_arrow.names if c in ("content", "text", "code"))
    fails = []
    surv_lens = []
    fail_msgs = Counter()
    n = 0
    for batch in pqf.iter_batches(batch_size=2000, columns=[col]):
        for v in batch.to_pydict()[col]:
            if v is None:
                continue
            t = str(v)
            if not t.strip():
                continue
            n += 1
            try:
                ast.parse(t)
            except SyntaxError as e:
                m = str(e.msg)
                if "EOF" in m or "incomplete" in m:
                    key = "unexpected_EOF"
                elif "invalid character" in m:
                    key = "invalid_character"
                elif "invalid syntax" in m:
                    key = "invalid_syntax"
                elif "unterminated" in m or "line continuation" in m:
                    key = "unterminated_string"
                elif "decimal" in m or "leading zeros" in m or "print" in m and "parentheses" in m:
                    key = "py2_ism"
                else:
                    key = f"other:{m[:30]}"
                fail_msgs[key] += 1
                fails.append(m)
            except (MemoryError, RecursionError, ValueError):
                fail_msgs["parser_exception"] += 1
            else:
                surv_lens.append(len(tk.encode(t).ids))
            if len(fails) >= 500 and len(surv_lens) >= 500:
                break
    surv_lens.sort()
    def pct(q):
        return surv_lens[int(q * (len(surv_lens) - 1))] if surv_lens else 0
    print(json.dumps({
        "scanned": n,
        "failing_histogram": {k: fail_msgs[k] for k in sorted(fail_msgs)},
        "fail_total": len(fails),
        "surviving_rows": len(surv_lens),
        "surviving_token_percentiles": {"p05": pct(.05), "p25": pct(.25), "p50": pct(.5),
                                        "p75": pct(.75), "p95": pct(.95), "p99": pct(.99)},
        "survivor_median_roughly_lines": pct(.5) // 4,  # ~4 tokens/line heuristic, config noted
        "config": {"source": "starcoderdata python shard0", "tokenizer": "data/tokenizer.json",
                   "lines_heuristic": "tokens/4, rough"},
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
