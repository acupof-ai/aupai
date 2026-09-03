#!/usr/bin/env python3
"""Build a cot corpus dir from fetched parquet, applying cot_criterion_0903.

Reads data/raw/<src>/*.parquet, extracts the chain via the schema-verified
binding (shared with cot_pilot), applies the 5 checks, writes kept docs as
jsonl to data/corpus/<out_dir>/<src>_<slice>.jsonl, and stamps
build_corpus_stats.json (docs in/kept, per-check rejects, tokens). Check #4 is
recorded per source: N/A when the source carries no ground-answer column
(OpenThoughts), or read from correctness_math_verify when it does (OpenR1).

    python datagen/build_cot.py --source cot_open_thoughts \
        --schema openthoughts --raw data/raw/cot_open_thoughts \
        --out data/corpus/cot_open_thoughts
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cot_pilot import THINK_TAG, chain_of  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ok_flags(chain, ans, ans_verified):
    """The 5 keep checks. chain/ans already bound. ans_verified: bool|None
    (None => no verifiable ground answer so check #4 is N/A)."""
    f = {}
    f["no_chain"] = not (chain and chain.strip())
    norm = re.sub(r"\s+", " ", chain).strip() if chain else ""
    f["too_short"] = (not f["no_chain"]) and len(norm) < 200
    f["truncated"] = (not f["no_chain"]) and bool(
        re.search(r"(\.\.\.\s*$|truncat|\[\s*\.\.\.\s*\])", chain, re.I))
    if ans_verified is None:
        f["math_unchecked"] = False  # N/A: no ground answer in this source
    else:
        f["math_wrong"] = not ans_verified
    f["dirty_answer"] = bool(ans and THINK_TAG.search(ans))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--schema", choices=["openthoughts", "skywork", "openr1"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    TOK = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(ROOT, a.out)
    parqs = sorted(f for f in os.listdir(os.path.join(ROOT, a.raw)) if f.endswith(".parquet"))
    assert parqs, f"no parquet in {a.raw}"
    docs_in = docs_kept = tok_tot = 0
    ck = Counter()
    wrote = 0
    for i, pq_name in enumerate(parqs):
        rows = pq.read_table(os.path.join(ROOT, a.raw, pq_name)).to_pylist()
        out = os.path.join(base, f"{os.path.basename(a.out)}_{i}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                chain, ans = chain_of(a.schema, r)
                body = THINK_TAG.sub(" ", chain).strip()
                verified = None
                if a.schema == "openr1":
                    gens = r.get("generations") or []
                    cmv = r.get("correctness_math_verify") or []
                    verified = bool(next((ok for g, ok in zip(gens, cmv) if g), False))
                fl = ok_flags(body, ans, verified)
                ck.update({k for k, v in fl.items() if v} or ("keep",))
                docs_in += 1
                if any(fl.values()):
                    continue
                docs_kept += 1
                t = len(TOK.encode(body).ids)
                tok_tot += t
                f.write(json.dumps({"src": a.source, "problem": r.get("problem") or
                        r.get("input") or "", "chain": body, "answer": ans or "",
                        "tokens": t}, ensure_ascii=False) + "\n")
        wrote += 1
        print(f"  {pq_name}: {len(rows)} rows", flush=True)
    stats = {"domain": os.path.basename(a.out), "source": a.source,
             "filters": "cot_criterion_0903", "srcfp": None,
             "criterion": "docs/standards/cot_criterion_0903.md",
             "schema": a.schema, "docs_in": docs_in, "docs_kept": docs_kept,
             "docs_deleted": docs_in - docs_kept,
             "reject_checks": dict(ck), "tokens_kept": tok_tot,
             "check4": "verified-from-source" if a.schema == "openr1" else "N/A-no-ground-answer",
             "n_shards": wrote}
    with open(os.path.join(base, "build_corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps({"docs_in": docs_in, "docs_kept": docs_kept,
                      "docs_deleted": docs_in - docs_kept,
                      "tokens_kept": tok_tot, "shards": wrote}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()