#!/usr/bin/env python3
# restartable: read-only sampling pass over source shards; an interrupt before the final
# write leaves no output at all and re-running re-reads the same shards (fixed seed), costing
# only CPU. The single pool file is written once at the end; there is no incremental state to
# corrupt.
"""Build the code side of the L2 label pool: 512-1024-token chunks of code_py_starcoder cut
on AST top-level def/class boundaries. Schema-aligned with 3b's datagen/l2_label_pool_build
(nl side); the two files cat into runs/l2label/pool_50k.jsonl for 66 to label.

    python3 datagen/l2_code_chunk_pool.py --selftest
    python3 datagen/l2_code_chunk_pool.py --root /work/aupai --target_chunks 15000 \\
        --out runs/l2label/pool_code_py_starcoder.jsonl

LENGTH IS MEASURED ONLY BY A REAL ENCODE (de/98/fb, PR #399 review). There is no additive or
junction estimate anywhere: byte-level BPE is non-additive across a join (worst +1 measured
on 1.09M junctions) and a 1024-id slice decodes to text that RE-ENCODES up to +4. The AST
only supplies candidate cut points; whether text fits is decided by tok.encode(candidate).ids
on the assembled text. An over-long physical line is cut by slicing ids, decoding, and
walking the boundary back until the decoded text re-encodes <= 1024 (same measured cap as 3b
#400 _cap_segments).

  - boundary = a top-level FunctionDef/AsyncFunctionDef/ClassDef, never cut through one;
  - semicolon multi-statement lines (several AST nodes, one physical line) are emitted once;
  - chunks never span source documents; a whole short document stays one <512 chunk;
  - byte-exact partition: ''.join(chunks) == text;
  - sample_id=sha256(final chunk text)[:16], parent_doc_id=sha256(full doc)[:16];
  - kind="code"; ppl/ppl_band null (no code KenLM; the nl model is not applied to code).
"""
import argparse
import ast
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datagen.l3_stratified_sample import LENGTH_BINS, length_band  # noqa: E402
from datagen.score_ledger import content_doc_id  # noqa: E402

MIN_TOK, MAX_TOK = 512, 1024
KIND = "code"
PPL_NULL = None


def language_of(source):
    """Last path/tag segment of `source`: 'starcoderdata:python' and '.../py' both -> 'python'."""
    seg = str(source or "").rstrip("/").replace(":", "/").split("/")[-1].strip()
    return seg or "_"


def top_level_line_units(text):
    """Partition source lines into top-level units. Each top-level def/class is one unit;
    loose statements/comments/blanks before a def/class are attached to it, a trailing loose
    run is attached to the last unit. A single forward line cursor assigns every physical line
    exactly once, so ''.join(units) == text.

    Semicolon multi-statement lines (`a=1; b=2`, `from x import y; y.z()`) are several AST nodes
    sharing one (lineno,end_lineno): a node wholly inside already-emitted lines is skipped, so
    the shared physical line is not emitted once per node (the 218/150k duplication bug)."""
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [text]
    nodes = list(tree.body)
    if not nodes:
        return [text]
    DEF = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    units, loose, cursor = [], [], 0

    def add_def(seg):
        units.append("".join(loose) + seg)
        loose.clear()

    for nd in nodes:
        lo = max(nd.lineno - 1, cursor)
        hi = max(nd.lineno, getattr(nd, "end_lineno", nd.lineno))
        if hi <= cursor:
            continue                         # node on a semicolon line already emitted
        loose.extend(lines[cursor:lo])
        seg = "".join(lines[lo:hi])
        if isinstance(nd, DEF):
            add_def(seg)
        else:
            loose.append(seg)
        cursor = hi
    loose.extend(lines[cursor:])
    if loose:
        tail = "".join(loose)
        if units:
            units[-1] += tail
        else:
            units.append(tail)
    return units


def _ids(tok, text):
    return tok.encode(text).ids


def _cut_oversized_line(line, tok):
    """Cut ONE physical line that already encodes over MAX into pieces that RE-ENCODE <= MAX.

    Slice at most MAX ids, decode, RE-ENCODE, and walk the end boundary back id-by-id while the
    measured length exceeds MAX -- the boundary is chosen by the real metric, not the id count
    (a 1024-id cut re-encodes up to +4). A tokenizer reports that cut-point drift through
    tok.cut_drift (0 for an exact tokenizer; the selftest fake sets 4). One id of forward
    progress is forced for a pathological token, so this terminates. The decoded pieces are
    exact substrings in id space and partition the line when concatenated."""
    ids = _ids(tok, line)
    drift = int(getattr(tok, "cut_drift", 0))
    pieces, start = [], 0
    while start < len(ids):
        end = min(start + MAX_TOK, len(ids))
        while end > start and len(_ids(tok, tok.decode(ids[start:end]))) + drift > MAX_TOK:
            end -= 1
        if end == start:
            end = start + 1
        pieces.append(tok.decode(ids[start:end]))
        start = end
    return pieces


def _pack_lines(lines, tok):
    """Pack source LINES (keepends=True) into pieces that re-encode <=MAX, on line boundaries.
    A line that alone exceeds MAX is BPE-cut by _cut_oversized_line. The join decision is a
    real encode of the assembled candidate. Byte-exact because every input line is kept whole
    in exactly one output piece."""
    pieces, cur = [], []

    def flush():
        if cur:
            pieces.append("".join(cur))
            cur.clear()

    for ln in lines:
        if len(_ids(tok, ln)) > MAX_TOK:
            flush()
            pieces.extend(_cut_oversized_line(ln, tok))
            continue
        if cur and len(_ids(tok, "".join(cur) + ln)) > MAX_TOK:
            flush()
        cur.append(ln)
    flush()
    return pieces


def chunk_document(text, tok):
    """Cut one document into chunks that re-encode in [512,1024] on AST def/class boundaries.
    `tok` is the only length oracle. Returns list[str] with ''.join(chunks) == text; a whole
    short document stays one <512 chunk; a trailing <512 tail merges into the previous chunk
    only while the merged text re-encodes <= MAX."""
    units = top_level_line_units(text)
    chunks, cur = [], []

    def flush():
        if cur:
            chunks.append("".join(cur))
            cur.clear()

    for u in units:
        if len(_ids(tok, u)) > MAX_TOK:
            flush()
            chunks.extend(_pack_lines(u.splitlines(keepends=True), tok))
            continue
        if cur and len(_ids(tok, "".join(cur) + u)) > MAX_TOK:
            flush()
        cur.append(u)
    flush()
    if len(chunks) >= 2 and len(_ids(tok, chunks[-1])) < MIN_TOK \
            and len(_ids(tok, chunks[-2] + chunks[-1])) <= MAX_TOK:
        tail = chunks.pop()
        chunks[-1] += tail
    return chunks


def build_rows(content, source, url, tok):
    """All chunk rows for one parent doc. sample_id over the FINAL chunk text. length_band is
    the PARENT-DOC char band (aligned with 3b's nl pool, PR #396): every chunk of one doc
    shares it, because a chunk-level band collapses every ~512-1024-token chunk to m/l."""
    parent = content_doc_id(content)
    band = length_band(len(content))
    chunks = chunk_document(content, tok)
    rows = []
    for idx, ch in enumerate(chunks):
        rows.append({
            "sample_id": content_doc_id(ch),
            "parent_doc_id": parent,
            "chunk_idx": idx,
            "n_chunks": len(chunks),
            "language": language_of(source),
            "kind": KIND,
            "length_band": band,
            "ppl_band": PPL_NULL,
            "ppl": PPL_NULL,
            "source": source,
            "url": url,
            "content": ch,
            "teacher_labels": PPL_NULL,
            "hand_read": PPL_NULL,
        })
    return rows


# ---------------------------------------------------------------------------------------
class _FakeTokenizer:
    """Deterministic tokenizer for selftests reproducing the real BPE's non-idealities without
    the gate vocabulary:
      - whole-text length is 1 id per char PLUS 1 for every INTERNAL (non-trailing) newline -- a
        join across a newline can cost +1; a final trailing newline costs 0. This is 98's exact
        oracle: len(ids) == sum(char in non-last lines) + (#internal newlines). encode/decode
        round-trip the text exactly, so byte partition is checkable on plain strings.
      - cut_drift=4: a piece produced by an id-slice mid-line re-encodes +4 (the byte-BPE
        boundary effect 3b measured); _cut_oversized_line walks the boundary back for it."""

    cut_drift = 4

    def encode(self, text):
        # id = codepoint for each char, except a NEWLINE shared between two non-empty lines is
        # encoded as the pair (0x0A, INTERNAL_NL=10), i.e. +1 over the single newline char.
        ids = []
        lines = text.split("\n")
        for k, line in enumerate(lines):
            ids.extend(ord(ch) for ch in line)
            if k < len(lines) - 1:
                ids.append(10)             # newline char id
                ids.append(10 + 0x100)     # +1 internal-newline marker id
        return _FakeEncoding(ids)

    def decode(self, ids, skip_special_tokens=False):
        out = []
        i = 0
        while i < len(ids):
            t = ids[i]
            if t == 10 and i + 1 < len(ids) and ids[i + 1] == 10 + 0x100:
                out.append("\n")
                i += 2
                continue
            if t == 10:
                out.append("\n")
            elif 0 <= t <= 0x10FFFF:
                out.append(chr(t))
            i += 1
        return "".join(out)


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


def _selftest():
    # Real-length oracle, two instances: exact (no cut drift) and the packer's measured re-encode
    tok = _FakeTokenizer()

    def n(s):
        return len(tok.encode(s).ids)

    def funcs(names, body="    x = 1\n    return x\n"):
        return "".join(f"def {nm}(a):\n{body}" for nm in names)

    fails = []

    def assert_cap(doc, label):
        ch = chunk_document(doc, tok)
        if "".join(ch) != doc:
            fails.append(f"{label}: partition not exact")
        for c in ch:
            if n(c) > MAX_TOK:
                fails.append(f"{label}: chunk {n(c)} > {MAX_TOK}")
        return ch

    # 1) exact partition on short/multifunction/long docs, every chunk re-encodes <= 1024.
    assert_cap("import os\n\n" + funcs([f"f{i}" for i in range(40)]), "multifunc")
    assert_cap("def g():\n    return 2\n", "tiny")
    assert_cap("\n".join(f"x = {i}" for i in range(5000)), "loose-long")

    # 2) each fitting top-level def lands intact in exactly one chunk.
    doc = "\n".join("import os" for _ in range(6)) + "\n\n" + funcs([f"fn{i}" for i in range(60)])
    ch = chunk_document(doc, tok)
    for nm in [f"fn{i}" for i in range(60)]:
        if sum(1 for c in ch if f"def {nm}(a):\n" in c) != 1:
            fails.append(f"{nm} split/duplicated")

    # 3) tiny doc -> one kept chunk with correct defaults.
    tiny = "def g():\n    return 2\n"
    rows = build_rows(tiny, "starcoderdata:python", "u", tok)
    if len(rows) != 1 or rows[0]["n_chunks"] != 1 or rows[0]["kind"] != "code" \
            or rows[0]["ppl"] is not None or rows[0]["language"] != "python":
        fails.append("tiny row schema wrong")

    # 4) oversized function hard-splits on lines, exact, in cap, line-aligned.
    big = "def big():\n" + "".join(f"    y{i} = {i}\n" for i in range(2000))
    ch = assert_cap(big, "oversized-fn")
    if len(ch) < 2 or any(not c.endswith("\n") for c in ch):
        fails.append("oversized function split shape wrong")

    # 4b) semicolon multi-statement physical lines emitted exactly once.
    semi = ("from gevent import monkey; monkey.patch_all()\n"
            "NR = 2; NC = 1\nmonths = []; total_m = 1; net_total = 0\n"
            + funcs(["a", "b", "c"]))
    if "".join(top_level_line_units(semi)) != semi:
        fails.append("semicolon duplicated in top_level_line_units")
    assert_cap(semi, "semicolon")

    # 4c) mixed oversized unit + short tail: the tail-merge setitem regression.
    mixed = "q\n" * 2000 + "z\n" * 40
    assert_cap(mixed, "mixed-tail")

    # 4d) +4 CUT DRIFT. A physical line that is exactly at/above the id cap: an id-slice would
    # re-encode to 1028. _cut_oversized_line must walk the boundary back so EVERY returned piece
    # re-encodes <=1024 even after decode->re-encode adds 4. Build a no-newline line whose raw
    # id length forces a cut (decode stamps +4).
    line = "a" * (MAX_TOK + 100)          # single physical line, 1124 ids, must be cut
    pieces = _cut_oversized_line(line, tok)
    if "".join(pieces) != line:
        fails.append("cut pieces do not partition the line")
    if len(pieces) < 2:
        fails.append("over-long line was not cut")
    for p in pieces:
        if n(p) > MAX_TOK:
            fails.append(f"cut piece re-encodes {n(p)} > {MAX_TOK} (drift not walked back)")
    # and through the full document path
    assert_cap(line + "\ny = 1\n" * 50, "doc-with-overlong-line")

    # 4e) exhaustive random regression under the exact +1/internal-newline oracle.
    import random as _r
    rr = _r.Random(98)
    for t in range(1600):
        nlines = rr.randint(1, 400)
        doc = "\n".join("x" * rr.randint(0, 1300) for _ in range(nlines))
        ch = chunk_document(doc, tok)
        if "".join(ch) != doc:
            fails.append(f"random[{t}] partition broken")
            break
        bad = [n(c) for c in ch if n(c) > MAX_TOK]
        if bad:
            fails.append(f"random[{t}] over cap: {bad[:3]}")
            break

    # 5) ids: sample_id per final chunk, parent stable.
    rows = build_rows(funcs(["a", "b", "c", "d", "e", "f", "g", "h"]) * 3,
                      "starcoderdata:python", None, tok)
    if len({r["sample_id"] for r in rows}) != len(rows):
        fails.append("chunk sample_ids not unique")
    if len({r["parent_doc_id"] for r in rows}) != 1:
        fails.append("parent_doc_id not constant")
    if rows[0]["sample_id"] != content_doc_id(rows[0]["content"]):
        fails.append("sample_id not sha256(chunk)")

    # 6) unparseable input partitions exactly (defensive; the corpus is ast-gated).
    bad = "def broken(:\n" * 300
    assert_cap(bad, "unparseable")

    if fails:
        print("FAIL\n" + "\n".join(fails))
        raise SystemExit(1)
    print("l2_code_chunk_pool selftest ok: real-encode hard cap, cut-drift walk-back, "
          "exact partition, 1600-doc random, AST boundaries")


def _load_tokenizer(root):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(root, "data", "tokenizer.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--domain", default="code_py_starcoder_dc")
    ap.add_argument("--out", default="")
    ap.add_argument("--target_chunks", type=int, default=15000)
    ap.add_argument("--n_shards", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return

    tok = _load_tokenizer(a.root)
    files = sorted(f for f in glob.glob(os.path.join(a.root, "data", "corpus", a.domain, "*.jsonl"))
                   if "manifest" not in f and "gate_exclude" not in f)
    step = max(1, len(files) // a.n_shards)
    pick = files[::step][:a.n_shards]
    docs_by_band = {nm: [] for _, _, nm in LENGTH_BINS}
    n_parent = 0
    for f in pick:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                content = d.get("content") or ""
                if len(content) < 50:
                    continue
                rows = build_rows(content, d.get("source"), d.get("url"), tok)
                docs_by_band[rows[0]["length_band"]].append(rows)
                n_parent += 1

    bands = [nm for _, _, nm in LENGTH_BINS]
    per = a.target_chunks // len(bands)
    rng = random.Random(a.seed)
    chosen, manifest_bands = [], {}
    for band in bands:
        families = docs_by_band.get(band, [])
        rng.shuffle(families)
        got = drawn_docs = 0
        for rows in families:
            if got >= per:
                break
            chosen.extend(rows)
            got += len(rows)
            drawn_docs += 1
        manifest_bands[band] = {"parent_docs_population": len(families),
                                "parent_docs_drawn": drawn_docs, "chunks": got}
    chosen.sort(key=lambda r: (r["kind"], r["language"], r["length_band"],
                               str(r["ppl_band"]), r["parent_doc_id"], r["chunk_idx"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in chosen:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "out": a.out, "domain": a.domain, "kind": KIND,
        "shards_sampled": len(pick), "shards_total": len(files),
        "parent_docs_read": n_parent, "chunks_total": len(chosen),
        "target_chunks": a.target_chunks, "per_band_cap": per, "seed": a.seed,
        "token_band": [MIN_TOK, MAX_TOK], "tokenizer": "data/tokenizer.json",
        "length_measure": "real tok.encode(candidate).ids; cut points decode->re-encode walked back",
        "ppl": "null: no code KenLM; nl KenLM not applied to code (agreed 2026-09-16)",
        "bands": manifest_bands,
        "doc_id": "datagen/score_ledger.content_doc_id sha256(text)[:16]",
    }
    with open(a.out + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps({"chunks": len(chosen), "parents": n_parent, "bands": manifest_bands}))


if __name__ == "__main__":
    main()
