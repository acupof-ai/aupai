#!/usr/bin/env python3
"""t21 report: exact token count of CLEANED rp1t_github, the 5 numbers.

Tokens = the frozen tokenizer over the CLEANED text (not bytes x a ratio -- that
is the 85.7B->73.6B correction class). Reads the clean's output shards
(data/corpus/code/*.jsonl), which carry filters_fp. Reports:

  fetched docs   (from the raw rp1t_github fetch_stats/docs)
  kept docs      (rows in the cleaned shards)
  retention      kept / fetched
  landed tokens  frozen-tokenizer tokens over cleaned text
  landed tok/byte

Also checks whether the 751.3M-tokens/file projection holds within 10% (13 files
were the fetch target); if not, the 73.6B code-supply figure changes by the gap.

Usage (on pod): python3 scripts/count_cleaned_code.py
"""
import glob
import json
import os

from tokenizers import Tokenizer  # type: ignore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
SHARDS = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", "code", "*.jsonl")))


def main():
    kept = 0
    tokens = 0
    tb = 0  # text UTF-8 bytes of kept docs
    for shard in SHARDS:
        with open(shard, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = d.get("content") or d.get("text") or ""
                if not t:
                    continue
                kept += 1
                tb += len(t.encode("utf-8"))
                tokens += len(TOK.encode(t).ids)
    # fetched docs: the raw fetch wrote N jsonl docs; counted as kept+rejected is
    # not available, so fetched is derived from the raw files (the clean's reject
    # reasons live in build_corpus_stats / the raw). We count raw docs here.
    raw_docs = 0
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "rp1t_github", "*.sampled.jsonl"))):
        with open(p, encoding="utf-8") as f:
            raw_docs += sum(1 for l in f if l.strip())
    # filters_fp present?
    stats = {}
    sp = os.path.join(ROOT, "data", "corpus", "code", "build_corpus_stats.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            stats = json.load(fh)
    fp = stats.get("filters_fp")

    per_file = tokens / max(1, len(SHARDS))
    project = 751303817  # 751.3M tokens/file projection measured 2026-08-30
    gap = (per_file - project) / project

    print(f"shards: {len(SHARDS)}  filters_fp: {fp or 'MISSING -> FAILS the derived-artifact rule'}")
    print(f"fetched docs (sample): {raw_docs}   kept docs: {kept}")
    print(f"retention (kept/fetched, one-file denominator): {kept / max(1, raw_docs):.2%}")
    print(f"landed tokens (frozen tok over cleaned text): {tokens} ({tokens / 1e9:.2f}B)")
    print(f"landed tok/byte: {tokens / max(1, tb):.4f}")
    print(f"per-file tokens: {per_file / 1e6:.2f}M  vs projection 751.3M: gap {gap:+.1%} "
          f"({'WITHIN 10%' if abs(gap) <= 0.10 else 'BEYOND 10% -> 73.6B changes by ' + f'{gap:+.1%}'})")
    print(f"disk /work free: {os.statvfs(os.path.join(ROOT,'data')).f_bavail * os.statvfs(os.path.join(ROOT,'data')).f_frsize / 1e9:.0f}G")


if __name__ == "__main__":
    main()

