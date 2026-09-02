#!/usr/bin/env python3
"""Cascade on code_rp1t ast.parse survivors (fb P0 2026-09-01): what the 0.42B verified
Python becomes after boilerplate-drop then length floor -- the number the mix is built on.
  ast.parse(py3) -> boilerplate-drop (>50% content lines high-DF) -> length floor.
Reports rows+tokens at each stage and the token-length distribution of survivors so the
floor is a choice, not a hidden constant."""
import glob
import json
import multiprocessing as mp
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

DF_MIN_DOCS = 8
BOILERPLATE_FRAC = 0.50
FLOOR_TOKENS = 50  # a doc below this many tokens is a fragment; report the distribution too


def _emit_survivors(path):
    import ast as _ast

    from tokenizers import Tokenizer

    tk = Tokenizer.from_file("/work/aupai/data/tokenizer.json")
    import json as _j

    surv = []  # (content, tokens)
    rows = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows += 1
        t = None
        try:
            t = _j.loads(line).get("content") or ""
        except _j.JSONDecodeError:
            continue
        if not t:
            continue
        try:
            _ast.parse(t)
        except (SyntaxError, MemoryError, RecursionError, ValueError):
            continue
        try:
            surv.append((t, len(tk.encode(t).ids)))
        except (MemoryError, RecursionError, ValueError):
            pass
    return rows, surv


def main():
    shards = sorted(glob.glob("/work/aupai/data/corpus/code_rp1t/*.jsonl"))
    with mp.Pool(min(16, len(shards))) as pool:
        parts = pool.map(_emit_survivors, shards)
    total_rows = sum(p[0] for p in parts)
    surv = [s for p in parts for s in p[1]]
    tokens_parse = sum(t for _, t in surv)

    # boilerplate: content-line DF across survivors
    line_df = Counter()
    for t, _ in surv:
        for ln in set(l.strip() for l in t.split("\n") if l.strip()):
            line_df[ln] += 1
    high_df = {ln for ln, c in line_df.items() if c >= DF_MIN_DOCS}
    aft_boil = [s for s in surv if not (lambda lines: lines and
                 sum(1 for l in lines if l in high_df) / len(lines) > BOILERPLATE_FRAC)([l.strip() for l in s[0].split("\n") if l.strip()])]
    tokens_boil = sum(t for _, t in aft_boil)

    # length floor + distribution
    lens = sorted(t for _, t in aft_boil)
    aft_floor = [s for s in aft_boil if s[1] >= FLOOR_TOKENS]
    tokens_floor = sum(t for _, t in aft_floor)
    def pct(q):
        return lens[int(q * (len(lens) - 1))] if lens else 0
    print(json.dumps({
        "total_rows": total_rows,
        "stage_parse": {"rows": len(surv), "tokens": tokens_parse},
        "stage_boilerplate": {"rows": len(aft_boil), "tokens": tokens_boil,
                              "db_drop": len(surv) - len(aft_boil),
                              "boil_config": {"df_min_docs": DF_MIN_DOCS, "drop_frac": BOILERPLATE_FRAC,
                                              "df_basis": "survivor-docs, content-lines"}},
        "stage_length_floor": {"rows": len(aft_floor), "tokens": tokens_floor,
                               "floor_tokens": FLOOR_TOKENS, "floor_drop": len(aft_boil) - len(aft_floor)},
        "survivor_token_percentiles": {"p05": pct(.05), "p25": pct(.25), "p50": pct(.5),
                                       "p75": pct(.75), "p95": pct(.95)},
        "config": "ast.parse(py3) over all code_rp1t; cascade then boilerplate then length-floor",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
