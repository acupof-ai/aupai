#!/usr/bin/env python3
"""3b 2026-09-24: SFT mix A stage 1 -- 13-gram decontam + length gate on code_if pairs.

Input : function-level (signature+docstring -> body) pairs extracted from
        starcoder jsonl by datagen/extract_code_if_pairs.py.
Output: <out> jsonl of pairs that pass both gates, plus a stats json.

Gates:
  - 13-gram DROP against HumanEval+MBPP via filters/decontam_ngram on prompt and
    output independently. The extractor's own scan is bidirectional containment
    with a length floor, NOT this engine, so this rerun is the real 13-gram gate.
  - output (body) <= 256 gate tokens, prompt <= 512 (split-encoding counts:
    answer encoded alone, matching the packer).
The output order is the INPUT order (ordered pool), so the file is byte
reproducible; the extractor shuffles with seed 42 before writing.
# restartable: pure CPU over a fixed 100k-row jsonl, full run is seconds on 16
# workers; an interrupt loses at most that run, rerun reproduces the same bytes.
CPU only; pin it away from any GPU-training ranks, e.g.
  taskset -c 96-119 nice -n 19 python3 scripts/sfta_codeif_filter.py \\
      --root /work/aupai --src data/sft/code_if_pairs_dc_train.jsonl
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DEFAULT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT_DEFAULT)
sys.path.insert(0, os.path.join(ROOT_DEFAULT, "filters"))

from tokenizers import Tokenizer
from decontam_ngram import Decontaminator, decontam_fp, ngrams

MAX_BODY_DEFAULT = 256
MAX_PROMPT_DEFAULT = 512
WORKERS_DEFAULT = 16

_tok = _union = None


def _init(tok_path, root):
    global _tok, _union
    sys.path.insert(0, os.path.join(root, "filters"))
    _tok = Tokenizer.from_file(tok_path)
    dcon = Decontaminator.load_default(root)
    _union = set()
    for pmap in dcon.parts.values():
        for grams in pmap.values():
            _union |= grams


def _work(line):
    r = json.loads(line)
    p, o = r["prompt"], r["output"]
    if ngrams(p) & _union:
        return ("ngram_prompt", 0, None)
    if ngrams(o) & _union:
        return ("ngram_o", 0, None)
    lp = len(_tok.encode(p).ids)
    lo = len(_tok.encode(o).ids)
    return ("len", lo, {"prompt": p, "output": o,
                        "body_tokens": lo, "prompt_tokens": lp})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--src", default="data/sft/code_if_pairs_dc_train.jsonl")
    ap.add_argument("--out", default="data/sft/sfta/code_if_clean.jsonl")
    ap.add_argument("--stats", default="data/sft/sfta/code_if_stats.json")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--max-body", type=int, default=MAX_BODY_DEFAULT)
    ap.add_argument("--max-prompt", type=int, default=MAX_PROMPT_DEFAULT)
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    args = ap.parse_args()

    def p(x):
        return x if os.path.isabs(x) else os.path.join(args.root, x)

    src, out, stats, tok_path = p(args.src), p(args.out), p(args.stats), p(args.tokenizer)
    he = os.path.join(args.root, "data", "eval", "humaneval", "humaneval_164.jsonl")
    mbpp = os.path.join(args.root, "data", "eval", "mbpp_holdouts.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    lines = open(src).read().splitlines()
    n = kept = 0
    drops = {"ngram_prompt": 0, "ngram_o": 0, "body_gt_max": 0, "prompt_gt_max": 0}
    body_lens = []
    # ordered imap: output follows input order, so the jsonl is reproducible.
    out_records = []
    with mp.Pool(args.workers, _init, (tok_path, args.root)) as pool:
        for kind, lo, rec in pool.imap(_work, lines, chunksize=200):
            n += 1
            if kind == "ngram_prompt":
                drops["ngram_prompt"] += 1
            elif kind == "ngram_o":
                drops["ngram_o"] += 1
            elif lo > args.max_body:
                drops["body_gt_max"] += 1
            elif rec["prompt_tokens"] > args.max_prompt:
                drops["prompt_gt_max"] += 1
            else:
                rec["source"] = "code_if_pairs_dc"
                body_lens.append(lo)
                kept += 1
                out_records.append(rec)
    with open(out, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    body_lens.sort()
    def pct(q):
        return body_lens[min(len(body_lens) - 1, int(q * len(body_lens)))] if body_lens else 0
    buckets = {"1-50": sum(x <= 50 for x in body_lens),
               "51-150": sum(50 < x <= 150 for x in body_lens),
               f"151-{MAX_BODY_DEFAULT}": sum(x > 150 for x in body_lens)}
    result = {"input_rows": n, "kept": kept, "drops": drops,
             "body_token_total": sum(body_lens),
             "body_len_p25": pct(.25), "body_len_median": pct(.5),
             "body_len_p75": pct(.75), "buckets": buckets,
             "gates": {"max_body_tokens": args.max_body,
                       "max_prompt_tokens": args.max_prompt,
                       "decontam_ngram_n": 13,
                       "decontam_fp_inputs": decontam_fp(he, mbpp)}}
    with open(stats, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
