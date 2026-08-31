#!/usr/bin/env python3
"""Token counting, one definition.

Training consumes what `train.py`'s `encode` produces: each document's ids plus
one `<eos>` terminator. A counter that omits the terminator reads ~0.35% low on
math_owm (5.6 tokens/doc, 2026-08-31: the stamp said 4,020,618,525 and an
independent count said 4,034,824,812). The convention is not a detail of either
counter -- it is what the model is trained on -- so both call this.

Selftest: `python3 scripts/count_tokens.py --selftest`.
"""
import json
import os
import sys

CONVENTION = "ids + one <eos> per document (train.py encode)"


def count_docs(texts, tok):
    """Tokens in these documents as training will see them."""
    batch_fn = getattr(tok, "encode_batch_fast", tok.encode_batch)
    return sum(len(e.ids) + 1 for e in batch_fn(list(texts)))


def count_shards(paths, tok, field="content", sample=None):
    """(tokens, bytes) over jsonl shards. With `sample`, reads only the first
    `sample` shards and scales by total bytes -- the estimate every corpus stamp
    uses, since tokenizing 20GB to write one number is not worth the hour."""
    all_bytes = sum(os.path.getsize(p) for p in paths)
    read = paths[:sample] if sample else paths
    toks = nbytes = 0
    for p in read:
        raw = open(p, "rb").read()
        nbytes += len(raw)
        texts = []
        for line in raw.decode("utf-8", "replace").splitlines():
            if line.strip():
                try:
                    texts.append(json.loads(line)[field])
                except (json.JSONDecodeError, KeyError):
                    continue  # truncated final line (a killed pass) or a row without the field
        toks += count_docs(texts, tok)
    if not nbytes:
        return 0, all_bytes
    return int(all_bytes * toks / nbytes), all_bytes


def _selftest():
    """Known answer: N documents must exceed the no-terminator count by exactly N."""
    from tokenizers import Tokenizer

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tok = Tokenizer.from_file(os.path.join(root, "data", "tokenizer.json"))
    texts = ["hello world", "def f(x):\n    return x + 1", "中文测试"]
    bare = sum(len(e.ids) for e in tok.encode_batch(texts))
    got = count_docs(texts, tok)
    assert got == bare + len(texts), f"{got} != {bare} + {len(texts)}"
    assert count_docs([], tok) == 0
    print(f"count_tokens selftest OK: {len(texts)} docs, {bare} ids + {len(texts)} <eos> = {got}")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else 0)
