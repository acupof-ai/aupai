#!/usr/bin/env python3
"""Build data/corpus/chat_qa/ (fb ruling 2026-09-01): the same chat rows (问：/答：,
from data/corpus/chat/) in their ORIGINAL format, as a SEPARATE domain alongside
chatml/ (the ChatML render). TWO domains, same source rows; chat_qa = original form
(fingerprint + form differ from chatml, so if one format works and the other does
not, the corpus can read it). New dir + --phase holdout slice; nothing written into a
mix_scale_*-named domain (data/corpus/chat/ itself stays untouched)."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import build_corpus as B  # noqa: E402

SRC = "/work/aupai/data/corpus/chat"
DST = "/work/aupai/data/corpus/chat_qa"
PHASE = "chat_qa"


def main():
    shards = [p for p in sorted(glob.glob(os.path.join(SRC, "*.jsonl")))
              if os.path.basename(p) != "build_corpus_stats.json"]
    if not shards:
        raise SystemExit(f"no chat shards under {SRC}")
    B._LOCK_FD = B._build_lock(DST)
    os.makedirs(DST, exist_ok=True)
    w = B.ShardWriter(DST, "chat_qa")
    kept = 0
    kept_chars = 0
    held_out = []
    for p in shards:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            c = d.get("content") or ""
            if not c:
                continue
            if B.is_holdout(c):
                held_out.append(B.exact_key(c))
                continue
            kept += 1
            kept_chars += len(c)
            w.write(d)  # original form, unchanged
    w.close()
    from collections import Counter

    B._emit_holdout_slice(DST, PHASE, held_out, allow_empty=True)  # fresh source, 0 holdout overlap
    B._write_stats(DST, "chat_qa",
                   B.argparse.Namespace(domain="chat_qa", workers=1, phase=PHASE, allow_empty_slice=True,
                                        dry=False, filters="chat-original", no_near_dedup=True),
                   Counter({"kept": kept}), kept, kept_chars,
                   len(glob.glob(os.path.join(DST, "chat_qa_*.jsonl"))), held_out)
    print(json.dumps({"phase": PHASE, "src": SRC, "dst": DST, "rows": kept, "chars": kept_chars}))


if __name__ == "__main__":
    main()
