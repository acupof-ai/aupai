#!/usr/bin/env python3
"""Traditional -> Simplified over data/corpus/, then drop the duplicates the conversion creates.

Measured on the built corpus (2026-08-26): 59.4% of fineweb-2's cmn_Hani subset is Traditional,
against 1.2% of data/pretrain_full.jsonl. That is a Simplified-Chinese model spending most of its
new pretraining data on a script it is never evaluated on, and it also wrecks the tokenizer, which
was trained on the Simplified corpus: fineweb-2 compresses at 1.07 chars/token versus 1.52 for
pretrain_full, so the same text costs 42% more tokens.

Conversion is a character table lifted from opencc (convert every CJK codepoint once, keep the ones
that change) applied with str.translate: a C loop at ~7M chars/s against opencc.convert's 0.5M/s,
and on the sampled text the two agree exactly -- this opencc build does not do phrase-level
substitution either, so nothing is lost.

    python scripts/t2s_corpus.py [--dir data/corpus] [--domains web,chat] [--workers 32]
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_PATH = os.path.join(ROOT, "data", "t2s_table.json")
_NORM = re.compile(r"[\s\W_]+", re.UNICODE)
_TABLE = None


def table():
    """{codepoint: simplified char}, cached to data/t2s_table.json so opencc is only needed once."""
    global _TABLE
    if _TABLE is None:
        if os.path.exists(TABLE_PATH):
            _TABLE = {int(k): v for k, v in json.load(open(TABLE_PATH, encoding="utf-8")).items()}
        else:
            import opencc

            c = opencc.OpenCC("t2s")
            _TABLE = {}
            for cp in range(0x4E00, 0xA000):
                s = c.convert(chr(cp))
                if s != chr(cp) and len(s) == 1:
                    _TABLE[cp] = s
            os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
            json.dump({str(k): v for k, v in _TABLE.items()}, open(TABLE_PATH, "w", encoding="utf-8"))
    return _TABLE


def convert_file(path):
    """Rewrite one shard to <path>.t2s. Returns (docs, chars, chars_changed)."""
    t = table()
    docs = chars = changed = 0
    with open(path, encoding="utf-8") as f, open(path + ".t2s", "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = r["content"]
            c = s.translate(t)
            docs += 1
            chars += len(s)
            if c != s:
                changed += sum(1 for a, b in zip(s, c) if a != b)
                r["content"] = c
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    return docs, chars, changed


def dedup_domain(d):
    """Converting Traditional to Simplified makes two copies of the same article identical.
    One sequential pass per domain, replacing each .t2s shard with its deduplicated form."""
    seen = set()
    kept = dropped = 0
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl.t2s"))):
        rows = []
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            k = hashlib.sha1(_NORM.sub("", r["content"]).encode("utf-8")).digest()[:12]
            if k in seen:
                dropped += 1
                continue
            seen.add(k)
            rows.append(line)
            kept += 1
        with open(p[: -len(".t2s")], "w", encoding="utf-8") as out:
            out.writelines(rows)
        os.remove(p)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "corpus"))
    ap.add_argument("--domains", help="comma-separated (default: every subdirectory)")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    a = ap.parse_args()

    table()  # build/load once in the parent so the workers inherit the cache file
    domains = (
        a.domains.split(",")
        if a.domains
        else sorted(n for n in os.listdir(a.dir) if os.path.isdir(os.path.join(a.dir, n)))
    )
    for name in domains:
        d = os.path.join(a.dir, name)
        files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
        if not files:
            print(f"{name}: no shards, skipped", flush=True)
            continue
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(convert_file, files))
        docs = sum(r[0] for r in res)
        chars = sum(r[1] for r in res)
        changed = sum(r[2] for r in res)
        kept, dropped = dedup_domain(d)
        print(
            f"{name:<6} {docs:>9} docs  {chars / 1e9:>6.2f}B chars  {changed / max(chars, 1):>6.2%} converted"
            f"  |  dedup after conversion: dropped {dropped} ({dropped / max(docs, 1):.2%}), kept {kept}",
            flush=True,
        )
    print("T2S_DONE")


if __name__ == "__main__":
    sys.exit(main())
