#!/usr/bin/env python3
"""Steps 1-9 of the cot_fable cleaning spec over a Fable trace parquet, counted.

Produces facts/corpus_supply.json#cs.cot_fable_supply. Run on the pod, where the raw
sits; the ROOT default below is the pod path because that is the only place the 981 MB
parquet exists.

    python3 datagen/fable5/measure_fable5_supply.py

Steps 4 and 5 of the spec are NOT implemented here and the fact says so: there is no
MinHash path against cot/cot_open_thoughts, and MBPP is absent from datagen/holdout.py's
registry, so a contamination hit against it cannot be detected. Reporting them as zero
would be the defect this repo keeps paying for; they are unmeasured.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"),
)
import pyarrow.parquet as pq  # noqa: E402
from count_tokens import count_docs  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from filters.secrets import redact_text  # noqa: E402

POD = "/work/aupai"
BOILER = re.compile(r"^(let me think|okay,? let'?s|hmm,?)\W*$", re.I)


def measure(parquet, tokenizer):
    tok = Tokenizer.from_file(tokenizer)
    f = pq.ParquetFile(parquet)
    n = kept = red_rows = 0
    rej = collections.Counter()
    by_src = collections.Counter()
    tok_by_src = collections.Counter()
    red_by_src = collections.Counter()
    seen = set()

    for b in f.iter_batches(batch_size=4000, columns=["row_json"]):
        for r in b.to_pylist():
            n += 1
            try:
                d = json.loads(r["row_json"])
            except Exception:
                rej["parse_fail"] += 1
                continue
            if "cot" not in d:
                rej["no_cot_schema"] += 1
                continue
            cot = re.sub(r"\s+", " ", str(d.get("cot") or "")).strip()
            if not cot:
                rej["empty_cot"] += 1
                continue
            if len(cot) < 200:
                rej["cot_under_200"] += 1
                continue
            if BOILER.match(cot):
                rej["boilerplate"] += 1
                continue
            if d.get("output_type") != "text":
                rej["tool_call_only"] += 1
                continue
            ans = str(d.get("output") or "").strip()
            if not ans:
                rej["no_answer"] += 1
                continue
            head = cot[:600]
            if sum(1 for c in head if ord(c) > 127) > len(head) * 0.10:
                rej["not_english"] += 1
                continue
            # sha1, NOT the builtin hash(): PYTHONHASHSEED is random per process, so a
            # builtin-hash dedup makes the kept count and therefore the recorded token
            # total non-reproducible across runs of this same file.
            h = hashlib.sha1(cot[:400].encode("utf-8")).hexdigest()
            if h in seen:
                rej["dup_within"] += 1
                continue
            seen.add(h)
            body, nred = redact_text(f"{str(d.get('context') or '')}\n\n{cot}\n\n{ans}")
            src = str(d.get("source_file") or d.get("origin") or "unknown")
            src = src.split("/")[-1][:40]
            if nred:
                red_rows += 1
                red_by_src[src] += nred
            kept += 1
            by_src[src] += 1
            tok_by_src[src] += count_docs([body], tok)

    return {
        "rows_in": n,
        "kept": kept,
        "rejects": dict(rej.most_common()),
        "tokens_total": sum(tok_by_src.values()),
        "redaction_rows": red_rows,
        "redactions_total": sum(red_by_src.values()),
        "top_sources": dict(by_src.most_common(12)),
        "tokens_top": dict(tok_by_src.most_common(12)),
        "redactions_by_source": dict(red_by_src.most_common(12)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=f"{POD}/data/raw/fable5_2m/data/train.parquet")
    ap.add_argument("--tokenizer", default=f"{POD}/data/tokenizer.json")
    a = ap.parse_args()
    print(json.dumps(measure(a.parquet, a.tokenizer), indent=1))
