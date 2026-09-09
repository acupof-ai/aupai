#!/usr/bin/env python3
# Actual-count frozen-tokenizer tokens over one swapped-in clean domain (4c 2026-09-10).
# Usage: COUNT_DOMAIN=code_dedup08 python3 datagen/count_domain_tokens.py
# COUNT_BASE overrides the shard root (default data/corpus; use
# data/p1/keep_set for the keep set's own token count).
import glob, json, multiprocessing as mp, os, sys
ROOT = "/work/aupai"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DOM = os.environ.get("COUNT_DOMAIN", "code_dedup08")
SHARDS = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", DOM, "*.jsonl")))
_TOK = None

def tok():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    return _TOK

def count_shard(shard):
    from count_tokens import count_docs
    kept = tokens = tb = 0
    texts = []
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
            texts.append(t)
            if len(texts) >= 2000:
                tokens += count_docs(texts, tok())
                texts = []
    tokens += count_docs(texts, tok())
    return kept, tokens, tb

def main():
    assert SHARDS, f"no shards under data/corpus/{DOM}"
    with mp.Pool(int(os.environ.get("COUNT_WORKERS", "16"))) as pool:
        counts = pool.map(count_shard, SHARDS)
    kept = sum(c[0] for c in counts)
    tokens = sum(c[1] for c in counts)
    tb = sum(c[2] for c in counts)
    print(f"domain={DOM} shards={len(SHARDS)} kept_docs={kept} tokens={tokens} ({tokens/1e9:.4f}B) text_bytes={tb} tok/byte={tokens/max(1,tb):.4f}", flush=True)

if __name__ == "__main__":
    main()
