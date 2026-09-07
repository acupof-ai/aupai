#!/usr/bin/env python3
# restartable: a pure counter -- reads shards, writes nothing, returns a number. An
# interrupt costs only the tokenizing done so far, and callers that want to spend
# minutes rather than the hour pass `sample`. Nothing to resume.
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
        # split("\n"), never splitlines(): splitlines also breaks on U+2028/U+2029,
        # which ShardWriter writes through literally (json.dumps ensure_ascii=False,
        # build_corpus.py:292), so a row carrying one becomes two unparseable fragments
        # and its document is dropped -- silently, because the JSONDecodeError below is
        # the handler for a truncated final line. The bias is one-directional: every
        # count through this path reads LOW by the tokens of such documents.
        for line in raw.decode("utf-8", "replace").split("\n"):
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
    """Known answer: N documents must exceed the no-terminator count by exactly N.
    Plus the U+2028 case: a shard written the way ShardWriter writes it must count
    every document, and the splitlines reading of the same bytes must count fewer --
    the negative control, so the case fails if the reader silently reverts."""
    import tempfile

    from tokenizers import Tokenizer

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tok = Tokenizer.from_file(os.path.join(root, "data", "tokenizer.json"))
    texts = ["hello world", "def f(x):\n    return x + 1", "中文测试"]
    bare = sum(len(e.ids) for e in tok.encode_batch(texts))
    got = count_docs(texts, tok)
    assert got == bare + len(texts), f"{got} != {bare} + {len(texts)}"
    assert count_docs([], tok) == 0

    # Written exactly as ShardWriter does: json.dumps(ensure_ascii=False) passes
    # U+2028 through as the literal character, inside the string value.
    rows = ["plain one", "a b", "c d", "plain two"]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s_000.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for t in rows:
                f.write(json.dumps({"content": t}, ensure_ascii=False) + "\n")
        want = count_docs(rows, tok)
        got_shard, _ = count_shards([p], tok)
        assert got_shard == want, f"shard count {got_shard} != {want} over {len(rows)} docs"

        with open(p, "rb") as f:
            raw = f.read().decode("utf-8")
        assert len(raw.splitlines()) == len(rows) + 2, "fixture must carry 2 extra breaks"
        old = []
        for line in raw.splitlines():
            if line.strip():
                try:
                    old.append(json.loads(line)["content"])
                except (json.JSONDecodeError, KeyError):
                    continue
        assert len(old) == 2, f"negative control: splitlines must drop 2 docs, kept {len(old)}"
        assert count_docs(old, tok) < want, "negative control must read low"

    print(
        f"count_tokens selftest OK: {len(texts)} docs, {bare} ids + {len(texts)} <eos> = {got}; "
        f"U+2028/29 shard {got_shard} tok over {len(rows)} docs (splitlines would see {len(old)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else 0)
