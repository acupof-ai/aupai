#!/usr/bin/env python3
"""Render Fable CoT traces as plain-continuation jsonl for build_corpus.py.

Step 7 of the cot_fable cleaning spec, and the last piece before the fold. It writes rows
carrying ONE field that build_corpus reads -- `text` -- so the corpus build itself needs no
change: `--source jsonl:<glob>` already accepts this shape (datagen/build_corpus.py:243).

    python3 datagen/fable5/render_fable5_cot.py --out data/raw/fable5_cot_rendered
    python datagen/build_corpus.py --domain cot --source jsonl:data/raw/fable5_cot_rendered/*.jsonl \
        --phase fable5_0906 --target_tokens 3e6

`--phase` is REQUIRED and not a stylistic choice: cot is mix-named, so build_corpus.py:1288
refuses without one and freezes this phase's held-out slice before stamping. cot is NOT a
ladder domain -- no data/mix_scale_*.json names it -- so the frozen-corpus rule does not
forbid writing into it, which is why this folds into cot rather than needing a new dir.

WHY `text` AND NOT `instruction`. iter_jsonl (build_corpus.py:206-209) falls back to
format_example() when a row has `instruction` and no text, which wraps the row in ChatML.
The pretraining corpus contains effectively no ChatML -- 0 occurrences of <|im_start|> in
168,000 rows across all 42 domains -- and base evals prompt in continuation format, so a
ChatML-wrapped row would be the only one of its kind in the domain. Writing `text` takes the
first branch and never reaches the fallback.

WHY NOT build_cot.py, which looks like the right tool. It reads parquet and binds a chain per
schema, but the `cot` domain on the pod was NOT built by it: that stamp carries
filters/kept/kept_chars/near_dedup and nulls for schema/source, which is build_corpus.py's
shape. build_cot.py also emits no filters_fp at all (its stats dict has "srcfp": None and no
filters_fp key), and filters_fp is the whole requirement of the fold. Routing through
build_corpus keeps one stamper for the domain.

The cleaning steps are NOT reimplemented here. This reads what
datagen/fable5/measure_fable5_supply.py already decided -- same predicates, same order -- and
its own selftest asserts the two agree on row count, because a renderer that keeps a different
set than the measurement makes the recorded token count describe a corpus nobody built.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datagen.fable5.measure_fable5_supply import BOILER, POD  # noqa: E402
from filters.secrets import redact_text  # noqa: E402


def keep(d):
    """(text, None) for a row that survives steps 1-9, or (None, reject_reason).

    Mirrors measure_fable5_supply.measure()'s predicate order exactly. Kept as a separate
    function rather than shared with the measurement because the measurement counts and this
    renders; the selftest below is what holds them equal.
    """
    if "cot" not in d:
        return None, "no_cot_schema"
    import re

    cot = re.sub(r"\s+", " ", str(d.get("cot") or "")).strip()
    if not cot:
        return None, "empty_cot"
    if len(cot) < 200:
        return None, "cot_under_200"
    if BOILER.match(cot):
        return None, "boilerplate"
    if d.get("output_type") != "text":
        return None, "tool_call_only"
    ans = str(d.get("output") or "").strip()
    if not ans:
        return None, "no_answer"
    head = cot[:600]
    if sum(1 for c in head if ord(c) > 127) > len(head) * 0.10:
        return None, "not_english"
    return f"{str(d.get('context') or '')}\n\n{cot}\n\n{ans}", None


def render(parquet, out_dir, shard_rows=50000):
    import hashlib

    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    f = pq.ParquetFile(parquet)
    seen, rows, shard, kept, red = set(), [], 0, 0, 0
    for b in f.iter_batches(batch_size=4000, columns=["row_json"]):
        for r in b.to_pylist():
            try:
                d = json.loads(r["row_json"])
            except Exception:
                continue
            text, why = keep(d)
            if why:
                continue
            h = hashlib.sha1(text[:400].encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            text, n = redact_text(text)
            red += n
            kept += 1
            rows.append(json.dumps({"text": text, "src": "fable5_2m"}, ensure_ascii=False))
            if len(rows) >= shard_rows:
                _flush(out_dir, shard, rows)
                shard, rows = shard + 1, []
    if rows:
        _flush(out_dir, shard, rows)
        shard += 1
    return {"kept": kept, "redactions": red, "shards": shard}


def _flush(out_dir, i, rows):
    with open(os.path.join(out_dir, f"fable5_cot_{i}.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def _selftest():
    """The renderer's keep() must accept and reject exactly what the measurement's does.

    A drift here is silent and expensive: the fold would land a different row set than
    facts/corpus_supply.json#cs.cot_fable_supply describes, so the recorded 2.76M would
    describe a corpus that was never built.
    """
    fails = []
    cot = "x" * 250
    cases = [
        ({"cot": cot, "output_type": "text", "output": "ans", "context": "q"}, None),
        ({"output_type": "text", "output": "a"}, "no_cot_schema"),
        ({"cot": "", "output_type": "text", "output": "a"}, "empty_cot"),
        ({"cot": "short", "output_type": "text", "output": "a"}, "cot_under_200"),
        ({"cot": cot, "output_type": "tool_use", "output": "a"}, "tool_call_only"),
        ({"cot": cot, "output_type": "text", "output": ""}, "no_answer"),
        ({"cot": "Ж" * 250, "output_type": "text", "output": "a"}, "not_english"),
    ]
    for d, want in cases:
        _t, got = keep(d)
        if got != want:
            fails.append(f"keep({sorted(d)}) -> {got!r}, expected {want!r}")

    # The rendered text must carry NO ChatML: build_corpus's iter_jsonl only falls back to
    # format_example when `text` is absent, and this is the assertion that the field name
    # chosen above actually avoids that branch.
    text, _ = keep(cases[0][0])
    if "<|im_start|>" in text or "<|im_end|>" in text:
        fails.append("rendered text carries ChatML markers")
    if not text.startswith("q"):
        fails.append("rendered text does not open with the context")

    # A credential in a kept row must not survive rendering.
    _u = "_"
    gsk = "gsk" + _u + "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOP"
    t2, _ = keep({"cot": cot, "output_type": "text", "output": f"key={gsk}", "context": "q"})
    red, n = redact_text(t2)
    if n != 1 or gsk in red:
        fails.append(f"credential survived rendering: {n} replacement(s)")

    for f in fails:
        print(f"  FAIL {f}", file=sys.stderr)
    if fails:
        print(f"render_fable5_cot selftest: {len(fails)} failure(s)", file=sys.stderr)
        return 1
    print(
        f"render_fable5_cot selftest OK: {len(cases)} keep/reject cases match "
        f"measure_fable5_supply's predicates, rendered text is ChatML-free and "
        f"context-first, and a credential in a kept row is redacted"
    )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=f"{POD}/data/raw/fable5_2m/data/train.parquet")
    ap.add_argument("--out", default=f"{POD}/data/raw/fable5_cot_rendered")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    print(json.dumps(render(a.parquet, a.out), indent=1))
