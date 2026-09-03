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


QUESTION_KEYS = {"instruction", "problem", "prompt", "question", "query", "source", "input", "task", "statement"}
ANSWER_KEYS = {"output", "answer", "solution", "canonical_solution", "target", "completion", "gold", "result", "text"}


def tagged_leaves(o, key=""):
    """String values with their source key, so a needle can be typed question/answer.
    A hit on a problem statement is the strong contamination (the eval's question
    appears in the corpus); a hit on an answer/working string is weaker. Reporting
    per needle type scopes the verdict, rather than lumping both under one 'hit'."""
    out = []
    if isinstance(o, str):
        if o.strip():
            out.append((key.lower(), o))
    elif isinstance(o, dict):
        for k, v in o.items():
            out.extend(tagged_leaves(v, str(k)))
    elif isinstance(o, (list, tuple)):
        for v in o:
            out.extend(tagged_leaves(v, key))
    return out


def needle_type(key):
    if key in QUESTION_KEYS:
        return "question"
    if key in ANSWER_KEYS:
        return "answer"
    return "other"


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


def needles(texts_by_type, use_char=False):
    """texts_by_type: {needle_type: [text...]} -> {needle_type: (ws_need, ch_need)}.
    use_char defaults FALSE: e1-28's design applies the char-13 window only to the
    dense-chardomains (zh_web/chatml/chat_qa); on code/math/English it 'shreds
    indentation and identifiers into matches that mean nothing' -- a measured
    18,000%/doc false hit on the math corpus when left on. Whitespace 13-gram is
    the containment unit for everything else."""
    out = {}
    for ty, texts in texts_by_type.items():
        ws, ch = {}, {}
        for t in texts:
            for g in E.ws_grams(t):
                ws[g] = t
            if use_char:
                for g in E.char_grams(t):
                    ch[g] = t
        out[ty] = (ws, ch)
    return out


def _count_docs(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


_G_LOADED = []  # (name, gold_file, needles_by_type, n_records) set once for pool workers


def _scan_shard(path):
    """One shard's hits: {name: {needle_type: [hit records]}}. Runs in a pool worker."""
    shard_hits = {n: {ty: [] for ty in ("question", "answer", "other")} for n, *_ in _G_LOADED}
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
            for name, gp, by_type, nrec in _G_LOADED:
                for ty, (w, c) in by_type.items():
                    ws, ch, _ = E.scan_text(t, w, c, True)
                    if ws or ch:
                        shard_hits[name][ty].append({"gold_file": gp, "gold_records": nrec,
                                                     "corpus_doc": f"{path}:{ln}", "needle_type": ty,
                                                     "hit": t[:300]})
    return shard_hits


def _init_loaded(loaded):
    global _G_LOADED
    _G_LOADED = loaded


def scan(paths, golds, use_char=False, pool_n=8):
    """(name, needles_by_type) x corpus shards -> {name: {type: [hits]}}, shards in
    parallel since 2026-09-03. use_char defaults FALSE here too (the needles()
    default was changed, but this signature's own True override silently kept
    char-13 on for every scan -- the 18,000%/doc shredding). e1-28 reserves the
    char window for dense-chardomains only."""
    import multiprocessing as mp
    loaded = []
    for gp in golds:
        name, recs = load_golds(gp)
        by_type = {}
        for r in recs:
            for key, t in tagged_leaves(r):
                by_type.setdefault(needle_type(key), []).append(t)
        loaded.append((name, gp, needles(by_type, use_char), len(recs)))
    if pool_n <= 1:
        return merge(_scan_shard(p) for p in paths), len(paths)
    with mp.Pool(pool_n, initializer=_init_loaded, initargs=(loaded,)) as pool:
        return merge(pool.imap_unordered(_scan_shard, paths, chunksize=1)), len(paths)


def merge(parts):
    """Merge per-shard hit dicts {name: {type: [..]}}. Row identity preserved; a hit
    is appended to its (eval, type)."""
    merged = {}
    for part in parts:
        for name, by_type in (part or {}).items():
            merged.setdefault(name, {ty: [] for ty in ("question", "answer", "other")})
            for ty, hh in by_type.items():
                merged[name].setdefault(ty, []).extend(hh)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=[])
    ap.add_argument("--golds", nargs="+", default=DEFAULT_GOLDS)
    ap.add_argument("--out", default="runs/scan_eval_golds.json")
    ap.add_argument("--pool", type=int, default=8, help="pool size for the shard-parallel scan; keep modest to leave cores for training")
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
            verbatim_found = any(hits[name].values())
            shuf_found = any("fix_shuf" in h["corpus_doc"] for hh in hits[name].values() for h in hh)
            n_shuf = sum(1 for hh in hits[name].values() for h in hh if "fix_shuf" in h["corpus_doc"])
            assert verbatim_found, "verbatim gold NOT found -- scanner broken"
            assert not shuf_found, f"shuffled gold matched {n_shuf} docs -- 13-gram survives shuffle, test wrong"
            print(f"selftest OK: verbatim gold found, shuffled gold {n_shuf} hits (must be 0)")
        finally:
            shutil.rmtree(tmp)
        return

    for gp in a.golds:
        if not os.path.isfile(gp):
            raise SystemExit(f"REFUSING: gold file {gp} missing -- a named gold that cannot be read is not a scan")
    if not a.domains:
        raise SystemExit("REFUSING: need --domains; a scan with no corpus measures nothing")

    import time
    result = {}
    for dom in a.domains:
        base = os.path.join(ROOT, "data", "corpus", dom)
        if not os.path.isdir(base):
            raise SystemExit(f"REFUSING: corpus dir {dom} missing")
        shards = sorted(glob.glob(os.path.join(base, "*.jsonl")))
        print(f"{dom}: scanning {len(shards)} shards (pool {a.pool})", flush=True)
        t0 = time.perf_counter()
        hits, n_shard = scan(shards, a.golds, pool_n=a.pool)
        wall = round(time.perf_counter() - t0)
        docs = sum(_count_docs(os.path.join(base, s)) for s in shards)
        print(f"  {dom}: {len(shards)} shards ({docs} docs) in {wall}s", flush=True)
        per_eval = {}
        for name, ghits in hits.items():
            per_eval[name] = {}
            for ty, hh in ghits.items():
                if not hh:
                    continue
                # hits are per-doc (a doc hits an eval once), so hits > docs is
                # impossible: it means the unit over-matches (char-13 shredding),
                # e.g. the measured 18,000%/doc. Refuse rather than print it.
                if len(hh) > docs:
                    raise SystemExit(f"REFUSING: {dom} {name}[{ty}] hit {len(hh)} docs > {docs} scanned -- wrong unit (char-13 shredding); re-run whitespace-13")
                per_eval[name][ty] = {"corpus_docs_hit": len(hh), "hit_rate": round(len(hh) / max(1, docs), 6)}
        result[dom] = {"shards": n_shard, "docs": docs, "wall_s": wall, "evals": per_eval, "hit_list": hits}
        print(f"  {json.dumps(per_eval)}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()