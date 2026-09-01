#!/usr/bin/env python3
"""Build data/corpus/chatml/ (fb ruling 2026-09-01): the SAME chat rows (问：/答：,
from data/corpus/chat/) rendered in ChatML markup. Two SEPARATE domains in the mix --
chat/ keeps the original form, chatml/ carries the ChatML render (so if one format
works and the other does not, the corpus can read it). New dir + --phase holdout slice
+ nothing written into a mix_scale_*-named domain. chat supply = 0.038B one-epoch;
chatml weight 0.76% at the 4-epoch cap (0.152B want) per fb. The purpose is to make the
ChatML prefix in-distribution (a format), not to memorise 38M tokens of QA."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import build_corpus as B  # noqa: E402
from loader import format_example  # noqa: E402

SRC = "/work/aupai/data/corpus/chat"
DST = "/work/aupai/data/corpus/chatml"
PHASE = "chatml"
QM = "问："
AN = "\n答："


def render_chatml(content):
    """问：Q\n答：A -> ChatML 'user/Q im_end assistant/A im_end'. A row with no 答： is a
    question-only turn (format_example's empty answer -> just the user prompt)."""
    q, a = content, ""
    if AN in content:
        q, a = content.split(AN, 1)
        a = a.strip()
    q = q[len(QM):].strip() if q.startswith(QM) else q.strip()
    prompt, completion = format_example(q, a)
    return prompt + completion


def main():
    shards = [p for p in sorted(glob.glob(os.path.join(SRC, "*.jsonl")))
              if os.path.basename(p) != "build_corpus_stats.json"]
    if not shards:
        raise SystemExit(f"no chat shards under {SRC}")
    B._LOCK_FD = B._build_lock(DST)
    os.makedirs(DST, exist_ok=True)
    w = B.ShardWriter(DST, "chatml")
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
            chatml = render_chatml(c)
            if not chatml:
                continue
            kept += 1
            kept_chars += len(chatml)
            w.write({"content": chatml, "source": d.get("source"), "url": d.get("url")})
    w.close()
    from collections import Counter

    B._emit_holdout_slice(DST, PHASE, held_out, allow_empty=True)  # fresh source, 0 holdout overlap
    B._write_stats(DST, "chatml",
                   B.argparse.Namespace(domain="chatml", workers=1, phase=PHASE, allow_empty_slice=True,
                                        dry=False, filters="chatml-render", no_near_dedup=True),
                   Counter({"kept": kept}), kept, kept_chars,
                   len(glob.glob(os.path.join(DST, "chatml_*.jsonl"))), held_out)
    print(json.dumps({"phase": PHASE, "src": SRC, "dst": DST, "rows": kept, "chars": kept_chars,
                      "one_epoch_b": kept_chars / 1.5 / 1e9,  # chars/1.5 chars-per-token heuristic
                      "config": "chat rows 问：/答： rendered ChatML via loader.format_example"}))


if __name__ == "__main__":
    main()
