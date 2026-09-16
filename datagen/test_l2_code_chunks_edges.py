#!/usr/bin/env python3
"""Edge-case validator for datagen/l2_code_chunk_pool (ae's L2 code chunker), gen-B 2026-09-16.

Independent of the main file: imports its functions, asserts the contract on synthetic
known-answer worlds AND (optionally) audits real code_py_starcoder. Never edits a corpus.

    python3 datagen/test_l2_code_chunks_edges.py --selftest          # synthetic asserts (CPU)
    python3 datagen/test_l2_code_chunks_edges.py --tokenizer T.json  # real-BPE mechanism check
    python3 datagen/test_l2_code_chunks_edges.py --root /work/aupai  # corpus audit (needs a pod)

The chunker's #399-B API is the length-oracle contract: chunk_document(text, tok) and
_cut_oversized_line/_pack_lines take a tokenizer-like object (.encode(s).ids, .decode(ids),
optional .cut_drift). char/4 counters CANNOT prove the hard cap because BPE is non-additive
across joins; the default selftest therefore injects ae's _FakeTokenizer (exact oracle +
cut_drift=4) and asserts every emitted chunk RE-ENCODES <= MAX_TOK under that drift.

Contract, one master invariant + four edges:
  M. ''.join(chunk_document(text, tok)) == text   -- exact partition, 0 bytes lost/reordered.
  1. ast.parse failure -> fallback partitions the whole doc, 0 drop.
  2. a top-level def/class that fits (<=MAX ids) lands whole in exactly one chunk; a fitting
     class keeps all its methods in one chunk.
  3. a unit >MAX is line-packed and any over-long physical line is BPE-cut so every piece
     RE-ENCODES <= MAX (this is the drift case char/4 cannot see).
  4. BOM / mixed indentation / semicolon multi-statement lines do not break partition
     exactness (semicolon lines were the real partition_fail root cause, 218/150k docs).
"""
import argparse
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datagen.l2_code_chunk_pool import (  # noqa: E402
    MAX_TOK,
    _cut_oversized_line,
    _FakeTokenizer,
    _pack_lines,
    build_rows,
    chunk_document,
    top_level_line_units,
)


def _n(tok):
    def f(s):
        return len(tok.encode(s).ids)
    return f


def check_partition(text, tok):
    chunks = chunk_document(text, tok)
    assert "".join(chunks) == text, "partition not exact"
    return chunks


def check_fitting_units_whole(text, tok):
    """Every top-level unit that fits (<=MAX ids) is a substring of exactly one chunk."""
    n = _n(tok)
    chunks = chunk_document(text, tok)
    bad = []
    for u in top_level_line_units(text):
        if n(u) > MAX_TOK:
            continue  # oversized: covered by the re-encode-cap check
        owners = sum(1 for c in chunks if u in c)
        if owners != 1:
            bad.append((u.splitlines()[0][:50] if u.strip() else "<blank>", owners))
    return bad


def check_all_chunks_within_cap(text, tok):
    """Every emitted chunk RE-ENCODES <= MAX under the injected tokenizer (drift-aware)."""
    n = _n(tok)
    over = [(i, n(c)) for i, c in enumerate(chunk_document(text, tok)) if n(c) > MAX_TOK]
    return over


# ------------------------------------------------------------------ synthetic known answers
def _selftest():
    tok = _FakeTokenizer()          # exact oracle + cut_drift=4 (BPE non-additivity, 3b-measured)
    n = _n(tok)
    fails = []

    def funcs(names, body="    x = 1\n    return x\n"):
        return "".join(f"def {nm}(a):\n{body}" for nm in names)

    # M + 2: ordinary multi-def doc partitions exactly and no fitting def is bisected.
    doc = "import os\n\n" + funcs([f"fn{i}" for i in range(80)])
    try:
        check_partition(doc, tok)
    except AssertionError as e:
        fails.append(str(e))
    if check_fitting_units_whole(doc, tok):
        fails.append(f"fitting units bisected: {check_fitting_units_whole(doc, tok)[:3]}")
    if check_all_chunks_within_cap(doc, tok):
        fails.append(f"chunk over cap: {check_all_chunks_within_cap(doc, tok)[:3]}")

    # 1: unparseable input -> fallback is the whole text, still exact, nothing dropped.
    bad_src = "def broken(:\n    x=1\n" * 300
    if "".join(chunk_document(bad_src, tok)) != bad_src:
        fails.append("unparseable partition not exact")
    if top_level_line_units(bad_src) != [bad_src]:
        fails.append("unparseable did not fall back to whole-doc unit")

    # 2 (methods): a class that fits keeps all its methods in one chunk.
    cls = "class C:\n" + "".join(f"    def m{i}(self):\n        return {i}\n" for i in range(5))
    doc2 = "import x\n\n" + cls + funcs(["after"])
    ch2 = chunk_document(doc2, tok)
    owners = [c for c in ch2 if "class C:" in c]
    if len(owners) != 1:
        fails.append(f"class split across {len(owners)} chunks")
    elif sum(f"def m{i}(self):" in owners[0] for i in range(5)) != 5:
        fails.append("class methods not all in the class chunk")

    # 3: one function longer than the window is line-packed, rejoins, every chunk re-encodes <=MAX.
    big = "def big():\n" + "".join(f"    y{i} = {i}\n" for i in range(4000))
    assert n(big) > MAX_TOK
    ch = chunk_document(big, tok)
    if "".join(ch) != big or len(ch) < 2:
        fails.append("oversized function not partitioned/split")
    over = [(i, n(c)) for i, c in enumerate(ch) if n(c) > MAX_TOK]
    if over:
        fails.append(f"oversized-function chunk over cap under drift: {over[:3]}")

    # 3 (pathological single line > window): _cut_oversized_line cuts so pieces re-encode <=MAX.
    longline = "    z = [" + ",".join(str(i) for i in range(4000)) + "]\n"
    pieces = _cut_oversized_line(longline, tok)
    if "".join(pieces) != longline:
        fails.append("single-long-line rejoin not exact")
    over = [n(p) for p in pieces if n(p) > MAX_TOK]
    if over:
        fails.append(f"_cut_oversized_line piece over cap under drift +{tok.cut_drift}: {over[:3]}")

    # 3 (drift boundary): _pack_lines assembled candidate must re-encode <=MAX at every join.
    lines = [f"row_{i} = compute({i})\n" for i in range(2000)]
    packed = _pack_lines(lines, tok)
    if "".join(packed) != "".join(lines):
        fails.append("_pack_lines partition not exact")
    over = [n(p) for p in packed if n(p) > MAX_TOK]
    if over:
        fails.append(f"_pack_lines piece over cap under drift: {over[:3]}")

    # 4a: BOM + mixed tabs/spaces -> partition exact (ast may fail on mixed indent, fallback holds).
    for pre in ("\ufeff", "\t \t"):
        doc4 = pre + "def g():\n\treturn 1\n" + funcs(["h"])
        if "".join(chunk_document(doc4, tok)) != doc4:
            fails.append(f"BOM/mixed-indent partition not exact ({pre!r})")

    # 4b: SEMICOLON multi-statement lines -- the real partition_fail root cause (218/150k docs).
    semi_cases = [
        "from gevent import monkey; monkey.patch_all()\n\ndef f():\n    return 1\n",
        "NR = 2; NC = 1\nimport os\n" + funcs(["g"]),
        "import logging; module_logger = logging.getLogger(__name__)\n\nclass C:\n    pass\n",
        "months = []; total_m = 1; net_total = 0; changes = []\n" + funcs(["h"]),
        "def a():\n    x = 1; y = 2; return x + y\n\nb = 3; c = 4\n" + funcs(["d"]),
    ]
    for i, doc in enumerate(semi_cases):
        j = "".join(chunk_document(doc, tok))
        if j != doc:
            fails.append(f"semicolon-line partition not exact [case {i}]: len {len(doc)}->{len(j)}")
        if check_fitting_units_whole(doc, tok):
            fails.append(f"semicolon-line fitting unit bisected [case {i}]")

    if fails:
        print("FAIL\n" + "\n".join(fails))
        raise SystemExit(1)
    print("edge validator selftest ok: partition, fitting-units-whole, methods, re-encode-cap "
          "under cut_drift, unparseable fallback, BOM/mixed-indent, semicolon-lines")


# ------------------------------------------------------------------ real-BPE mechanism check
def _mech_check(tok_path):
    """Parameterized on a real tokenizer (fb 2026-09-16): the SAME cap assertion as the selftest
    but on the real BPE, which has its own cut drift. Threshold self-adapts to the vocab; the
    local surviving vocab is aupai-de/data/tokenizer.json (32768, <eos>=1, [NUM]=32767)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    n = _n(tok)
    fails = []

    big = "def big():\n" + "".join(f"    val_{i} = compute({i}) + offset\n" for i in range(6000))
    for c in chunk_document(big, tok):
        if n(c) > MAX_TOK:
            fails.append(f"real-BPE chunk over window: {n(c)}")
    doc = "def f():\n    x = [" + ",".join(f'"{i}"' for i in range(8000)) + "]\n"
    ch = chunk_document(doc, tok)
    if "".join(ch) != doc:
        fails.append("real-BPE long-line partition not exact")
    for c in ch:
        if n(c) > MAX_TOK:
            fails.append(f"real-BPE long-line piece over window: {n(c)}")
    mixed = ("import os; import sys\n\n"
             + "".join(f"def fn{i}(a):\n    return a + {i}\n" for i in range(200)))
    if "".join(chunk_document(mixed, tok)) != mixed:
        fails.append("real-BPE mixed-doc partition not exact")
    if fails:
        print("FAIL (mech)\n" + "\n".join(fails))
        raise SystemExit(1)
    print(f"mech check ok on {os.path.basename(tok_path)} (vocab {tok.get_vocab_size()}): "
          f"every chunk real ntok<=%d, partition exact." % MAX_TOK)


# ------------------------------------------------------------------ real-corpus audit (pod)
def _audit(root, domain, n_shards, limit):
    import glob
    import json

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(root, "data", "tokenizer.json"))
    n = _n(tok)

    files = sorted(f for f in glob.glob(os.path.join(root, "data", "corpus", domain, "*.jsonl"))
                   if "manifest" not in f and "gate_exclude" not in f)
    step = max(1, len(files) // n_shards)
    pick = files[::step][:n_shards]

    st = {"docs": 0, "parse_fail": 0, "partition_fail": 0, "fitting_unit_bisected": 0,
          "chunks": 0, "chunks_over_window": 0, "band_not_parent_level": 0, "empty_after_chunk": 0}
    overages = []
    fail_samples = []
    for f in pick:
        with open(f, encoding="utf-8") as fh:
            lines_iter = list(fh)
        for line in lines_iter:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            content = d.get("content") or ""
            if len(content) < 50:
                continue
            st["docs"] += 1
            if st["docs"] > limit:
                break
            parsed = True
            try:
                ast.parse(content)
            except SyntaxError:
                parsed = False
                st["parse_fail"] += 1
            chunks = chunk_document(content, tok)
            st["chunks"] += len(chunks)
            if "".join(chunks) != content:
                st["partition_fail"] += 1
                if len(fail_samples) < 5:
                    fail_samples.append({"url": d.get("url"), "reason": "partition"})
                continue
            if any(c == "" for c in chunks):
                st["empty_after_chunk"] += 1
            for c in chunks:
                nt = n(c)
                if nt > MAX_TOK:
                    st["chunks_over_window"] += 1
                    overages.append(nt)
            if parsed:
                rows = build_rows(content, d.get("source"), d.get("url"), tok)
                if len({r["length_band"] for r in rows}) != 1:
                    st["band_not_parent_level"] += 1
                for u in top_level_line_units(content):
                    if n(u) <= MAX_TOK and sum(1 for c in chunks if u in c) != 1:
                        st["fitting_unit_bisected"] += 1
                        break
        if st["docs"] > limit:
            break
    st["parse_fail_rate"] = round(st["parse_fail"] / max(1, st["docs"]), 5)
    st["shards_sampled"] = len(pick)
    st["fail_samples"] = fail_samples
    if overages:
        overages.sort()
        m = len(overages)
        st["overage_p50"] = overages[m // 2]
        st["overage_p95"] = overages[min(m - 1, int(m * 0.95))]
        st["overage_max"] = overages[-1]
    st["chunks_over_window_share"] = round(st["chunks_over_window"] / max(1, st["chunks"]), 6)
    print("AUDIT " + json.dumps(st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tokenizer", default="", help="tokenizer.json for the real-BPE mechanism "
                    "check; threshold self-adapts to this vocab (fb 2026-09-16)")
    ap.add_argument("--root", default="")
    ap.add_argument("--domain", default="code_py_starcoder_dc")
    ap.add_argument("--n_shards", type=int, default=24)
    ap.add_argument("--limit", type=int, default=200000)
    a = ap.parse_args()
    if a.selftest or (not a.root and not a.tokenizer):
        _selftest()
    if a.tokenizer:
        _mech_check(a.tokenizer)
    if a.root:
        _audit(a.root, a.domain, a.n_shards, a.limit)


if __name__ == "__main__":
    main()
