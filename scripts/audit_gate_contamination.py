#!/usr/bin/env python3
"""Gate-mix contamination audit vs HumanEval + MBPP (ae-6, fb order 2026-09-11).

The HumanEval pass@1 gate is worthless if the training mix contains the test. For
every gate-mix domain this scans each corpus document for a 13-token n-gram that
also appears in either the benchmark's CANONICAL SOLUTION or its PROMPT/DOCSTRING.

Normalisation (shared with the decontam pipeline, datagen/gen_exercises._norm):
collapse all whitespace runs to one space, strip. Tokens for the n-gram are
whitespace tokens (not BPE ids): the gate is about a verbatim TEXT span, which
must survive across vocabularies. A 13-whitespace-token match cannot arise from
shared boilerplate (imports, def signatures) the way a bigram can.

Relation to the UltraData pipeline's own decontam: that check
(datagen/gen_exercises.decontam, called at datagen/ultradata_shards.py:147) is
WHOLE-prompt substring containment of the normalized benchmark prompt (>=100 HE /
40 MBPP chars) inside a corpus doc, plus exact prompt+solution equality. It is not
13-gram and its containment key is the prompt/docstring, not the canonical
solution -- so this audit is the stronger, separate check the gate requires, and it
covers the five non-UltraData domains the UltraData pipeline never saw.

CPU only; default 16 workers. Writes per-domain hit rows (offending corpus row id
+ which problem + solution/prompt), and a summary JSON.

    python3 scripts/audit_gate_contamination.py \
        --domains code_py_starcoder,code_py_rp1t,cot,math_owm_stage2,en_c4_stage2 \
        --out runs/contam_gate_audit.json
"""
# restartable: read-only scan that appends one summary at the end; an interrupt
# loses only the in-memory tally and the command re-runs (no shard is mutated).
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = os.environ.get("AUPAI_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUMANEVAL = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP = os.path.join(ROOT, "data", "eval", "mbpp_holdouts.jsonl")
N = 13


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _grams(text, n=N):
    toks = _norm(text).split(" ")
    return set(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))


def load_bench():
    """Return (problem grams -> set, per-problem meta). Two disjoint key sets:
    solution grams and prompt/docstring grams."""
    sol, prompt, meta = defaultdict(set), defaultdict(set), {}
    files = (("humaneval", HUMANEVAL), ("mbpp", MBPP))
    for source, path in files:
        if not os.path.exists(path):
            sys.exit(f"benchmark missing: {path}")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            tid = f"{source}:{r['task_id']}"
            if source == "humaneval":
                p, sol_text = r["prompt"], r.get("canonical_solution", "")
            else:
                p, sol_text = r.get("text", ""), r.get("code", "")
            p, sol_text = _norm(p), _norm(sol_text)
            prompt[tid] = _grams(p)
            sol[tid] = _grams(sol_text)
            meta[tid] = {"source": source, "n_prompt_grams": len(prompt[tid]),
                         "n_sol_grams": len(sol[tid])}
    return dict(sol), dict(prompt), meta


def _scan_doc(args):
    """Worker: does one doc's 13-grams intersect any benchmark problem? Returns
    [(row_index, problem_id, kind), ...]."""
    idx, text, sol_grams, prompt_grams = args
    g = _grams(text)
    if not g:  # fewer than N whitespace tokens -> no 13-gram can exist
        return []
    hits = []
    for tid, bg in sol_grams.items():
        if bg and (g & bg):
            hits.append((idx, tid, "solution"))
    for tid, bg in prompt_grams.items():
        if bg and (g & bg):
            hits.append((idx, tid, "prompt"))
    return hits


def scan_file(path, sol_grams, prompt_grams, workers):
    """Stream a shard; batch docs to the pool. Returns list of hit tuples."""
    docs = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line).get("content", "")
            except json.JSONDecodeError:
                c = ""
            docs.append((i, c))
    hits = []
    # chunk so a huge shard does not copy gram sets per-doc at submit granularity
    chunk = 2000
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s in range(0, len(docs), chunk):
            batch = [(i, c, sol_grams, prompt_grams) for i, c in docs[s:s + chunk]]
            for r in ex.map(_scan_doc, batch, chunksize=64):
                hits.extend(r)
    return hits, len(docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--domains",
                    default="code_py_starcoder,code_py_rp1t,cot,math_owm_stage2,en_c4_stage2")
    ap.add_argument("--max_shards", type=int, default=0, help="0 = all shards")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_hits_per_domain", type=int, default=0,
                    help="0 = record every contaminated row id")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "contam_gate_audit.json"))
    a = ap.parse_args()

    corpus = os.path.join(a.root, "data", "corpus")
    sol_grams, prompt_grams, meta = load_bench()
    print(f"benchmark problems: {len(meta)} (humaneval+mbpp), {N}-whitespace-token grams",
          flush=True)

    domains = [d for d in a.domains.split(",") if d]
    report = {"n": N, "normalisation": "whitespace collapsed to single spaces, stripped; "
              "n-grams over whitespace tokens", "benchmark_n_problems": len(meta),
              "domains": {}}
    for d in domains:
        fs = sorted(glob.glob(os.path.join(corpus, d, "*.jsonl")))
        if a.max_shards:
            fs = fs[:a.max_shards]
        if not fs:
            print(f"{d}: NO SHARDS", flush=True)
            report["domains"][d] = {"shards": 0, "error": "no shards"}
            continue
        per_problem = defaultdict(lambda: {"solution": 0, "prompt": 0})
        rows = {}
        n_docs = 0
        for f in fs:
            hits, nd = scan_file(f, sol_grams, prompt_grams, a.workers)
            for idx, tid, kind in hits:
                per_problem[tid][kind] += 1
                if a.max_hits_per_domain and len(rows) >= a.max_hits_per_domain:
                    continue
                # one contaminated row, even if it matches several problems/grams
                key = f"{os.path.basename(f)}:{idx}"
                rec = rows.setdefault(key, {"problem": tid, "match": kind, "all": []})
                rec["all"].append(f"{tid}:{kind}")
            n_docs += nd
        n_hit_docs = len(rows)
        report["domains"][d] = {
            "shards": len(fs), "docs_scanned": n_docs,
            "hit_rows_recorded": n_hit_docs, "record_cap": a.max_hits_per_domain,
            "hit_rows_capped": bool(a.max_hits_per_domain and n_docs and len(rows) >= a.max_hits_per_domain),
            "problems": {tid: dict(v) for tid, v in sorted(per_problem.items())},
            "rows": rows,
        }
        n_prob = len(per_problem)
        print(f"{d}: {n_docs} docs, {len(fs)} shards, {n_prob} problem(s) matched, "
              f"{n_hit_docs} distinct contaminated rows", flush=True)

    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {a.out}")


def _selftest():
    """Detector cannot false-negative: a doc that only BARELY spans a 13-gram
    (15 tokens -> 3 unique windows) must still match. The first implementation guarded
    on `len(gram_set) < N` (3 < 13) and returned early, missing exactly the real case."""
    g13 = " ".join(f"w{i}" for i in range(13))
    key = {"p": _grams(g13)}
    # 15 tokens: 3 windows, one of which is the exact benchmark gram
    hit = _scan_doc((0, "aa " + g13 + " bb", key, {}))
    assert hit == [(0, "p", "solution")], hit
    # 12 tokens -> no 13-gram -> empty, not an error
    short = " ".join(f"w{i}" for i in range(12))
    assert _scan_doc((1, short, key, {})) == []
    # unrelated text does not fire
    assert _scan_doc((2, "x " * 30, key, {})) == []
    print("audit_gate_contamination selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
