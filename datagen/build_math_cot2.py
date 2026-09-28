#!/usr/bin/env python3
"""Build math_cot2_dc: word-problem + step-by-step-solution corpora NOT already in
cot_dc (which is the full NuminaMath-CoT set).

Sources (all continuous Question:/Answer: pretraining text, never ChatML):
  - microsoft/orca-math-word-problems-200k (real word problems, human answers)
  - open-math/OpenMathInstruct-2 train shards 0..6 (GSM8K/MATH originals + augmentations)
  - meta-math/MetaMathQA-395K (GSM8K/MATH AnsAug/Rephrased/SV/FOBAR)

Filters, in order:
  1. 13-word-token containment vs FOUR gates, reusing filters/decontam_ngram.py's
     normaliser/ngrams on every side: GSM8K test (Q+A), English MATH-500 (P+S),
     HumanEval (prompt/solution/test), MBPP (text/code). The tracked math_test_500.jsonl
     is a Chinese translation and is also keyed (instruction/output).
     Gate files not in git (pod-only eval bytes):
       data/eval/gsm8k_test.jsonl      = openai/gsm8k main/test (1319 rows), same file eval/gsm8k.py reads
       data/eval/math_500_en_test.jsonl = HuggingFaceH4/MATH-500 test.jsonl (500 rows),
         curl -4 .../hf-mirror.com/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl
     A missing gate file raises (no silent unfiltered build).
  2. Question-set dedup against NuminaMath-CoT: cot_dc already trained on every NuminaMath
     row, and orca/gsm/math subsets recur verbatim in these sources.
  3. Within-domain exact (question,answer) dedup: whitespace-collapsed, lowercased.
     OM2 repeats one original problem with several independently generated solutions;
     distinct solutions survive, only byte-identical pairs drop.

    python3 datagen/build_math_cot2.py --root /work/aupai \
        --om2-shards 9 --out data/corpus/math_cot2_dc
"""

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "filters"))

from decontam_ngram import ngrams  # noqa: E402  (shared 13-word-token shingler)

_WS = re.compile(r"\s+")
SHARD_ROWS = 50_000

_WORKER = {}


def qkey(q):
    """Question identity key: whitespace-collapsed, lowercased. Conservative vs a
    paraphrase; meant to catch verbatim recurrence across/within these sources."""
    return _WS.sub(" ", (q or "").strip().lower())


def cakey(q, a):
    """(question, answer) identity. OM2 emits several independently generated
    solutions for one original problem: distinct solutions survive, byte-identical
    pairs drop. Keying the question alone discarded the multi-solution diversity."""
    return qkey(q) + "\n" + _WS.sub(" ", (a or "").strip().lower())


def gate_parts(root):
    """{pid: {part: gram_set}} for the four math/code gates, shingled with the ONE
    normaliser decontam_ngram defines so the gate stays vocab-independent."""
    parts = {}

    def add(pid, part, text):
        g = ngrams(text)
        if g:
            parts.setdefault(pid, {})[part] = g

    p = os.path.join(root, "data", "eval", "gsm8k_test.jsonl")
    with open(p, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            add(f"gsm8k:{i}", "question", r.get("question", ""))
            add(f"gsm8k:{i}", "answer", r.get("answer", ""))

    p = os.path.join(root, "data", "eval", "math_500_en_test.jsonl")
    with open(p, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            add(f"math500en:{i}", "problem", r.get("problem", ""))
            add(f"math500en:{i}", "solution", r.get("solution", ""))

    p = os.path.join(root, "data", "eval", "math_test_500.jsonl")
    with open(p, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            add(f"math500zh:{i}", "instruction", r.get("instruction", ""))
            add(f"math500zh:{i}", "output", r.get("output", ""))

    he = os.path.join(root, "data", "eval", "humaneval", "humaneval_164.jsonl")
    with open(he, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            tid = f"humaneval:{r['task_id']}"
            add(tid, "prompt", r.get("prompt", ""))
            add(tid, "solution", r.get("canonical_solution", ""))
            add(tid, "test", r.get("test", ""))

    mb = os.path.join(root, "data", "eval", "mbpp_holdouts.jsonl")
    with open(mb, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            tid = f"mbpp:{r['task_id']}"
            add(tid, "prompt", r.get("text", ""))
            add(tid, "solution", r.get("code", ""))
    return parts


def iter_numma_questions(raw_dir):
    import pyarrow.parquet as pq

    for f in sorted(glob.glob(os.path.join(raw_dir, "*.parquet"))):
        t = pq.read_table(f, columns=["problem"])
        for q in t.column(0).to_pylist():
            k = qkey(q)
            if k:
                yield k


def _init_worker(parts):
    _WORKER["parts"] = parts


def gate_hit(content, gram2pid):
    """O(len(row grams)): one flat dict over every gate gram; return the first
    pid:part owning a row 13-gram. 196k gate grams, so iterating the 3.3k problem
    parts per row would cost that many intersections for every one of ~3.5M rows."""
    for g in ngrams(content):
        pid = gram2pid.get(g)
        if pid is not None:
            return pid
    return None


def flat_gate_map(parts):
    m = {}
    for pid, pmap in parts.items():
        for part, grams in pmap.items():
            for g in grams:
                m.setdefault(g, f"{pid}:{part}")
    return m


def scan_rows(rows, source, gram2pid, seen, numma_qs):
    """Apply the three filters to one chunk of (question, answer) rows.
    Returns kept (q,a) list and a per-reason counter. Mutates `seen` with new keys."""
    kept = []
    st = {"scanned": 0, "drop_empty": 0, "drop_gate": 0, "drop_numma": 0, "drop_dup": 0}
    hits = {}
    for q, a in rows:
        st["scanned"] += 1
        q, a = (q or "").strip(), (a or "").strip()
        if not q or not a:
            st["drop_empty"] += 1
            continue
        content = f"Question: {q}\nAnswer: {a}"
        h = gate_hit(content, gram2pid)
        if h:
            st["drop_gate"] += 1
            hits[h] = hits.get(h, 0) + 1
            continue
        kq = qkey(q)
        if kq in numma_qs:
            st["drop_numma"] += 1
            continue
        k = cakey(q, a)
        if k in seen:
            st["drop_dup"] += 1
            continue
        seen.add(k)
        kept.append((content, source))
    return kept, st, hits


def read_orca(p):
    import pyarrow.parquet as pq

    d = pq.read_table(p).to_pydict()
    return list(zip(d["question"], d["answer"], strict=True)), "orca_math"


def read_metamath(p):
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    return [(r["query"], r["response"]) for r in d], "metamath"


def read_om2(p):
    import pyarrow.parquet as pq

    d = pq.read_table(p, columns=["problem", "generated_solution"]).to_pydict()
    return list(zip(d["problem"], d["generated_solution"], strict=True)), "openmathinstruct2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="data/corpus/math_cot2_dc")
    ap.add_argument("--om2-shards", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="rows per source, 0 = all (smoke)")
    a = ap.parse_args()

    raw = os.path.join(a.root, "data", "raw")
    out_dir = os.path.join(a.root, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(out_dir, exist_ok=True)

    print("building 4-benchmark gate ...", flush=True)
    gram2pid = flat_gate_map(gate_parts(a.root))
    print(f"gate: {len(gram2pid):,} distinct grams", flush=True)

    print("loading numina question set (cot_dc dedup) ...", flush=True)
    numma_qs = set(iter_numma_questions(os.path.join(raw, "hf_numma")))
    print(f"numma questions: {len(numma_qs):,}", flush=True)

    inputs = []
    inputs.append(read_orca(os.path.join(raw, "orca_math", "train-00000-of-00001.parquet")))
    for i in range(a.om2_shards):
        inputs.append(read_om2(os.path.join(raw, "om2", f"train-{i:05d}-of-00032.parquet")))
    inputs.append(read_metamath(os.path.join(raw, "metamath", "MetaMathQA-395K.json")))
    if a.limit:
        inputs = [(rows[: a.limit], src) for rows, src in inputs]

    seen = set()
    stats = {}
    all_hits = {}
    shard_idx = 0
    buf = []
    om2_i = 0

    def flush(buf):
        nonlocal shard_idx
        if not buf:
            return
        p = os.path.join(out_dir, f"math_cot2_{shard_idx:03d}.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for content, source in buf:
                fh.write(
                    json.dumps({"content": content, "source": source, "url": ""}, ensure_ascii=False) + "\n"
                )
        shard_idx += 1

    for rows, source in inputs:
        # OM2 arrives as several shards that all share the source name; the stamp
        # key must be unique per input or shard stats overwrite each other.
        label = source
        if source == "openmathinstruct2":
            label = f"openmathinstruct2_shard{om2_i:02d}"
            om2_i += 1
        st = {"scanned": 0, "drop_empty": 0, "drop_gate": 0, "drop_numma": 0, "drop_dup": 0}
        src_hits = {}
        CH = 20_000
        for i in range(0, len(rows), CH):
            kept, cst, hits = scan_rows(rows[i : i + CH], source, gram2pid, seen, numma_qs)
            for k in st:
                st[k] += cst[k]
            for h, c in hits.items():
                src_hits[h] = src_hits.get(h, 0) + c
                all_hits[h] = all_hits.get(h, 0) + c
            buf.extend(kept)
            while len(buf) >= SHARD_ROWS:
                flush(buf[:SHARD_ROWS])
                del buf[:SHARD_ROWS]
        st["kept"] = st["scanned"] - sum(st[k] for k in ("drop_empty", "drop_gate", "drop_numma", "drop_dup"))
        stats[label] = {"rows": st, "gate_hits": src_hits}
        print(f"{label}: " + json.dumps(st), flush=True)
    flush(buf)

    from datagen.corpus_fingerprint import fp_dir

    stamp = {
        "domain": "math_cot2_dc",
        "fingerprint": fp_dir(out_dir),
        "filter": "13-word-token containment vs GSM8K test + MATH-500(en+zh) + HumanEval + MBPP; "
        "question dedup vs NuminaMath-CoT; within-source exact (question,answer) dedup",
        "n": 13,
        "sources_in_order": [
            "orca_math",
            f"openmathinstruct2 (train shards 0..{a.om2_shards - 1})",
            "metamath",
        ],
        "numma_question_set": len(numma_qs),
        "per_source": stats,
        "totals": {
            k: sum(v["rows"][k] for v in stats.values())
            for k in ("scanned", "drop_empty", "drop_gate", "drop_numma", "drop_dup", "kept")
        },
        "gate_hits": all_hits,
        "shards_total": shard_idx,
    }
    with open(os.path.join(out_dir, "build_corpus_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, indent=1, ensure_ascii=False)
    print("fingerprint", stamp["fingerprint"], "shards", shard_idx, flush=True)


if __name__ == "__main__":
    main()
