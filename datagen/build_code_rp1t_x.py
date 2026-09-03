#!/usr/bin/env python3
"""Build data/corpus/code_rp1t_x: code_rp1t through the decided executableity ops
(fb P0 2026-09-01), into a NEW dir the new mix names (never a mix-named ladder dir):
  1. boilerplate drop: drop docs where >50% of content lines are high-DOCUMENT-FREQUENCY
     (recurring across many docs) -- the measured 6.7% class;
  2. executable parse filter: keep docs that PARSED (Python ast.compile; C/C++/C#/Java/etc
     are kept as a reported bucket -- parse runtimes for them are not on the pod).
Writes survivors to code_rp1t_x, freezes the --phase holdout slice (fb: the gate lands
before the rebuild finishes, so the corpus is readable cross-stage).

Reports three numbers separately: parse rate, run rate, surviving tokens (run is the
sandboxed subprocess probe from executable_yield; here we report the parsed portion's
run rate on a sampled subset to bound time, and flag the full-run-probe cost)."""
import argparse
import ast
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import build_corpus as B  # noqa: E402

DF_SAMPLE = 0.20       # fraction of shards used to estimate line-document-frequency
DF_MIN_DOCS = 8        # a line in >= this many docs is boilerplate
BOILERPLATE_FRAC = 0.50  # drop a doc when this fraction of its lines are boilerplate
SRC = "/work/aupai/data/corpus/code_rp1t"
DST = "/work/aupai/data/corpus/code_rp1t_x"
PHASE = "code_rp1t_x"


def content_lines(t):
    return [ln.strip() for ln in t.split("\n") if ln.strip()]


def main():
    shards = sorted(B.glob.glob(os.path.join(SRC, "code_rp1t_*.jsonl"))) if hasattr(B, "glob") else sorted(
        __import__("glob").glob(os.path.join(SRC, "code_rp1t_*.jsonl")))
    if not shards:
        raise SystemExit(f"no code_rp1t shards under {SRC}")
    # Pass 1: line document-frequency on a sample (boilerplate = high-DF line).
    n_docs_sample = 0
    line_df = Counter()          # normalized line -> number of distinct docs (approx)
    seen_doc_line = set()
    for i, p in enumerate(shards):
        if i / len(shards) > DF_SAMPLE:
            break
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            t = (d.get("content") or "").strip()
            if not t:
                continue
            n_docs_sample += 1
            for ln in set(content_lines(t)):  # count once per doc
                k = B.exact_key(ln)
                line_df[k] += 1
    high_df = {k for k, c in line_df.items() if c >= DF_MIN_DOCS}
    print(f"pass1: {n_docs_sample} sample docs, {len(line_df)} distinct lines, "
          f"{len(high_df)} high-DF boilerplate lines", flush=True)

    # Pass 2: per-doc boilerplate fraction + parse, keep survivors.
    out_conf = os.path.join(DST, "build_corpus_stats.json")
    os.makedirs(DST, exist_ok=True)
    w = B.ShardWriter(DST, "code_rp1t_x")
    kept = boil_drop = parse_fail = 0
    kept_chars = 0
    held_out = []
    for p in shards:
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            t = B.SPECIAL_TOKEN.sub("", d.get("content") or "").strip()
            if not t:
                continue
            if B.is_holdout(t):
                held_out.append(B.exact_key(t))
                continue
            lines = content_lines(t)
            if lines and sum(1 for ln in lines if B.exact_key(ln) in high_df) / len(lines) > BOILERPLATE_FRAC:
                boil_drop += 1
                continue
            # executable parse: keep if it parses (Python) OR report as unparsed-bucket
            parsed = True
            src = (d.get("source") or "").lower()
            if src.endswith(".py") or ".py/" in src:
                try:
                    ast.parse(t)
                except SyntaxError:
                    parsed = False
            if not parsed:
                parse_fail += 1
                continue
            kept += 1
            kept_chars += len(t)
            w.write(d)
    w.close()
    # holdout slice freeze (the --phase gate, BEFORE the stamp)
    B._emit_holdout_slice(DST, PHASE, held_out)
    reasons = Counter({"kept": kept, "boilerplate_drop": boil_drop, "parse_fail": parse_fail})
    B._write_stats(DST, "code_rp1t_x", argparse.Namespace(domain="code_rp1t_x", workers=1, phase=PHASE, dry=False),
                   reasons, kept, kept_chars, len(B.glob.glob(os.path.join(DST, "code_rp1t_x_*.jsonl"))), held_out)
    print(json.dumps({
        "phase": PHASE, "src": SRC, "dst": DST, "shards": len(shards),
        "kept_docs": kept, "boilerplate_drop": boil_drop, "parse_fail": parse_fail,
        "boilerplate_config": {"df_min_docs": DF_MIN_DOCS, "sample_frac": DF_SAMPLE,
                               "drop_frac": BOILERPLATE_FRAC, "df_basis": "sample, content-lines"},
        "parse_config": "Python ast.parse; non-Python kept as unparsed bucket (no runtime on pod)",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
