#!/usr/bin/env python3
# restartable: an interrupt costs only the current run -- it writes one small JSON at
# the end and no partial file a later run could mistake for complete; re-run from scratch.
"""Generic 13-gram eval-gold contamination scanner (N4 stamp prereq, controller 2026-09-03).

Inputs: one or more corpus directories and the eval gold files. Method (reused from
e1_28_leak_scan, never reimplemented): whitespace 13-gram containment + e1-28's char-13
window with the MIN_CHARSET=4 low-entropy filter (a char gram from <4 distinct chars —
markdown rules, repeated dashes — is formatting, not evidence). Output per (domain,
eval): gold docs hit, corpus docs hit, rate, and a hit-list file naming the matched
gold + corpus doc + gram.

A hit requires an EXACT 13-token or 13-char gram of the gold to appear in a corpus doc,
so the gold extractor is deliberately loose (all string leaves) — a short numeric answer
cannot form a 13-gram, so loose extraction cannot false-positive.

    python3 datagen/scan_eval_golds.py --domains en_c4 en_c4_stage2 \
        --golds data/eval/humaneval/humaneval_164.jsonl ... --out runs/scan_eval_golds.json
    python3 datagen/scan_eval_golds.py --selftest   # fixture: verbatim gold MUST hit, shuffled MUST NOT
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import e1_28_leak_scan as E  # noqa: E402  (ws_grams, char_grams, scan_text, low_entropy, NGRAM, MIN_CHARSET)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_GOLDS = [
    "data/eval/humaneval/humaneval_164.jsonl",
    "data/eval/math_test_500.jsonl",
    "data/synthetic/math_hard_eval_v2_1k.jsonl",
    "data/eval/lambada_en/lambada_test_en.jsonl",
    "data/eval/code_holdout_v2_500.jsonl",
]


def string_leaves(o):
    """All string values in a record (recursive), as a list. Loose by design: a hit
    needs an exact 13-gram, and a short numeric answer forms none."""
    out = []
    if isinstance(o, str):
        if o.strip():
            out.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            out.extend(string_leaves(v))
    elif isinstance(o, (list, tuple)):
        for v in o:
            out.extend(string_leaves(v))
    return out


def load_golds(path):
    """name -> needle sets {whitespace gram ids, char gram ids} for the reference text."""
    ext = os.path.splitext(path)[1].lower()
    recs = []
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
    elif ext == ".json":
        o = json.load(open(path, encoding="utf-8"))
        recs = o if isinstance(o, list) else [o]
    else:
        raise SystemExit(f"unsupported gold type {path} ({ext}); use .jsonl or .json")
    name = os.path.basename(path)
    return name, recs


def needles(texts, use_char=True):
    ws, ch = {}, {}
    for t in texts:
        for g in E.ws_grams(t):
            ws[g] = t
        if use_char:
            for g in E.char_grams(t):
                ch[g] = t
    return ws, ch


def scan(paths, golds, use_char=True):
    """[(name, ws_need, ch_need)] x corpus shard rows -> {name: [hit records]}."""
    loaded = []
    for gp in golds:
        name, recs = load_golds(gp)
        texts = [t for r in recs for t in string_leaves(r)]
        w, c = needles(texts, use_char)
        loaded.append((name, gp, w, c, len(recs)))
    hits = {n: [] for n, *_ in loaded}
    n_shard = 0
    for path in paths:
        n_shard += 1
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("content") or d.get("text")
                if not t:
                    continue
                for name, gp, w, c, nrec in loaded:
                    ws, ch, _ = E.scan_text(t, w, c, use_char)
                    if ws or ch:
                        hits[name].append({"gold_file": gp, "gold_records": nrec,
                                           "corpus_doc": f"{path}:{ln}",
                                           "hit": t[:300]})
    return hits, n_shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=[])
    ap.add_argument("--golds", nargs="+", default=DEFAULT_GOLDS)
    ap.add_argument("--out", default="runs/scan_eval_golds.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        import tempfile
        import shutil
        # one gold verbatim -> MUST hit; a shuffled gold -> MUST NOT
        tmp = tempfile.mkdtemp()
        try:
            gold_text = "def is_prime(n):\n    if n < 2: return False\n    return all(n % i for i in range(2, int(n**0.5) + 1))"
            fixture_dir = os.path.join(tmp, "corpus", "fix")
            os.makedirs(fixture_dir)
            # verbatim shard
            with open(os.path.join(fixture_dir, "fix_000.jsonl"), "w") as f:
                f.write(json.dumps({"content": gold_text}) + "\n")
                f.write(json.dumps({"content": "unrelated english prose repeated enough tokens to matter."}) + "\n")
            # shuffled gold shard (word-order randomized -> no 13-gram survives)
            shuf_text = " ".join(sorted(gold_text.split()))
            with open(os.path.join(tmp, "corpus", "fix_shuf.jsonl"), "w") as f:
                f.write(json.dumps({"content": shuf_text}) + "\n")
            golds = [os.path.join(tmp, "golds.jsonl")]
            with open(golds[0], "w") as f:
                f.write(json.dumps({"solution": gold_text}) + "\n")
            hits, n_shard = scan(sorted(glob.glob(os.path.join(fixture_dir, "*.jsonl"))) + [os.path.join(tmp, "corpus", "fix_shuf.jsonl")], golds)
            name = os.path.basename(golds[0])
            verbatim_found = len(hits[name]) > 0
            shuf_found = any("fix_shuf" in h["corpus_doc"] for h in hits[name])
            assert verbatim_found, "verbatim gold NOT found -- scanner broken"
            assert not shuf_found, f"shuffled gold matched {len([h for h in hits[name] if 'fix_shuf' in h['corpus_doc']])} docs -- 13-gram survives shuffle, test wrong"
            print(f"selftest OK: verbatim gold found, shuffled gold {len([h for h in hits[name] if 'fix_shuf' in h['corpus_doc']])} hits (must be 0)")
        finally:
            shutil.rmtree(tmp)
        return

    for gp in a.golds:
        if not os.path.isfile(gp):
            raise SystemExit(f"REFUSING: gold file {gp} missing -- a named gold that cannot be read is not a scan")
    if not a.domains:
        raise SystemExit("REFUSING: need --domains; a scan with no corpus measures nothing")

    result = {}
    for dom in a.domains:
        base = os.path.join(ROOT, "data", "corpus", dom)
        if not os.path.isdir(base):
            raise SystemExit(f"REFUSING: corpus dir {dom} missing")
        shards = sorted(glob.glob(os.path.join(base, "*.jsonl")))
        print(f"{dom}: scanning {len(shards)} shards", flush=True)
        hits, n_shard = scan(shards, a.golds)
        per_eval = {}
        for name, ghits in hits.items():
            per_eval[name] = {"corpus_docs_hit": len(ghits), "hit_rate": round(len(ghits) / max(1, n_shard), 6)}
        result[dom] = {"shards": n_shard, "evals": per_eval, "hit_list": hits}
        print(f"  {json.dumps(per_eval)}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()