#!/usr/bin/env python3
"""3b 2026-09-24: SFT mix A prose slice -- pretrain-shaped continuation pairs.

Output: data/sft/sfta/prose.jsonl + prose_stats.json

Each sample is a cut of an en_c4_stage2_dc document: a masked prefix
(prompt, 32..256 tokens) followed by a supervised tail (output, 64..256
tokens). No ChatML, no special framing: this is a pretraining-shaped
continuation, the model's native distribution, so the 10% slice regularises
against a pure-code SFT collapse rather than teaching a format.

Gates:
  - source domain is 13-gram decontaminated at build; the rendered cuts are
    rescanned anyway (plan section 4: scan the final rendered text).
  - tail length 64..256 gate tokens; deterministic windows from each doc tail,
    seed 42 shuffles and selects.
# restartable: pure CPU read of 40 immutable jsonl shards, full run is tens of
# seconds on 16 workers; an interrupt loses at most that run and reruns to the
# same ordered bytes.
CPU only; run pinned to cores away from training, e.g.
  taskset -c 96-119 nice -n 19 python3 scripts/sfta_prose_extract.py
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "filters"))

from tokenizers import Tokenizer
from decontam_ngram import Decontaminator, decontam_fp, ngrams

DEFAULT_CORPUS = os.path.join(ROOT, "data", "corpus", "en_c4_stage2_dc")
DEFAULT_OUT_DIR = os.path.join(ROOT, "data", "sft", "sfta")
DEFAULT_TOK = os.path.join(ROOT, "data", "tokenizer.json")
HE = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP = os.path.join(ROOT, "data", "eval", "mbpp_holdouts.jsonl")

MAX_OUT, MIN_OUT, MAX_PROMPT = 256, 64, 256
MAX_DOC_CHARS = 8000
SEED = 42

_TOK = _UNION = None


def _init(tok_path):
    global _TOK, _UNION
    _TOK = Tokenizer.from_file(tok_path)
    dcon = Decontaminator.load_default(ROOT)
    _UNION = set()
    for pmap in dcon.parts.values():
        for grams in pmap.values():
            _UNION |= grams


def _doc_pairs(doc):
    """Up to 3 non-overlapping (prompt,output) cuts walking back from the tail."""
    ids = _TOK.encode(doc[:MAX_DOC_CHARS]).ids
    n = len(ids)
    if n < MIN_OUT * 2 + 32:
        return []
    out, pos = [], n
    while len(out) < 3 and pos - MIN_OUT >= 32:
        lo = max(MIN_OUT, min(MAX_OUT, pos - 32))
        start = pos - lo
        p_start = max(0, start - MAX_PROMPT)
        ptext = _TOK.decode(ids[p_start:start])
        otext = _TOK.decode(ids[start:pos])
        pos = start
        if not (otext.strip() and ptext.strip()):
            continue
        if ngrams(ptext) & _UNION or ngrams(otext) & _UNION:
            continue
        out.append((ptext, otext))
    return out


def _shard(path):
    pairs = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = r.get("content") or r.get("text") or ""
            if len(text) < 200:
                continue
            pairs.extend(_doc_pairs(text))
            if len(pairs) >= 12000:
                break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--quota", type=int, default=1_530_000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-shards", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.corpus, "*.jsonl")))[:args.max_shards]
    assert shards, f"no jsonl under {args.corpus}"

    all_pairs = []
    # ordered imap: shards and the final list follow glob order, so the eligible
    # pool is reproducible (the seed-42 shuffle below then makes selection stable).
    with mp.Pool(args.workers, _init, (args.tokenizer,)) as pool:
        for i, pairs in enumerate(pool.imap(_shard, shards)):
            all_pairs.extend(pairs)
            print(f"shard {i+1}/{len(shards)} pairs={len(all_pairs)}", flush=True)

    random.Random(SEED).shuffle(all_pairs)
    tok = Tokenizer.from_file(args.tokenizer)
    kept, total = [], 0
    for p, o in all_pairs:
        lo = len(tok.encode(o).ids)
        if total + lo > args.quota:
            continue
        total += lo
        kept.append((p, o, lo))
        if total >= args.quota:
            break

    out_path = os.path.join(args.out_dir, "prose.jsonl")
    with open(out_path, "w") as f:
        for p, o, lo in kept:
            f.write(json.dumps({"prompt": p, "output": o,
                                "source": os.path.basename(args.corpus),
                                "body_tokens": lo}, ensure_ascii=False) + "\n")
    stats = {"shards_scanned": len(shards), "eligible_cuts": len(all_pairs),
             "kept": len(kept), "loss_tokens": total, "quota": args.quota,
             "gates": {"tail_tokens": [MIN_OUT, MAX_OUT], "prompt_max": MAX_PROMPT,
                       "decontam_ngram_n": 13,
                       "decontam_fp_inputs": decontam_fp(HE, MBPP)}}
    with open(os.path.join(args.out_dir, "prose_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
