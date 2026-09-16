#!/usr/bin/env python3
# restartable: read-only sampling pass over source shards; an interrupt before the final
# write leaves no output at all and re-running re-reads the same shards (fixed seed), costing
# only CPU. The single pool file is written once at the end; there is no incremental state to
# corrupt.
"""Build the code side of the L2 label pool: 512-1024 BPE-token chunks of code_py_starcoder
cut on AST top-level def/class boundaries. Schema-aligned with 3b's datagen/l2_label_pool_build
(nl side); the two files cat into runs/l2label/pool_50k.jsonl for 66 to label.

    python3 datagen/l2_code_chunk_pool.py --selftest
    python3 datagen/l2_code_chunk_pool.py --root /work/aupai --target_chunks 15000 \
        --out runs/l2label/pool_code_py_starcoder.jsonl

Cutting rules (agreed with 3b/de/genB 2026-09-16, PR #399 review):
  - boundary = a top-level FunctionDef/AsyncFunctionDef/ClassDef, never cut through one;
  - runs of other module statements (imports/assigns) are packed with adjacent defs;
  - greedy pack to [512, 1024] BPE ids; a short trailing chunk merges into the previous;
  - HARD CAP by additive id sum + junction headroom (each join measured at worst +1 id),
    then each emitted chunk is true-encoded once and split if it still exceeds 1024. An
    additive-only bound let 8.97% of chunks exceed 1024 (max 1973) because BPE is non-additive;
    a join that would exceed 1024 flushes first;
  - a single top-level unit longer than 1024 ids is hard-split on complete LINES, and one
    over-long physical line is cut at its BPE id boundary (never dropped, or long programs
    vanish systematically and bias the length bands);
  - semicolon multi-statement lines (`a=1; b=2`) are several AST nodes on one physical line:
    a line cursor emits each physical line once, so ''.join(units) is byte-exact;
  - sample_id = sha256(chunk text)[:16], parent_doc_id = sha256(full source doc)[:16]
    (datagen/score_ledger.content_doc_id); ids are always over the FINAL chunk text;
  - kind="code"; ppl/ppl_band null (no code KenLM exists; the nl model is not applied to code);
  - chunks never span source documents, so a whole short document stays one <512 chunk.
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
# BPE is non-additive across a join, but measured on 1.09M real starcoder line junctions the
# per-junction change is in {-1,0,+1} (worst +1). The packers therefore budget
# additive-id-sum + number-of-new-junctions (linear, one encode per unit), then true-encode
# each emitted chunk once and hard-split any that still exceeds the window as a backstop.
KIND = "code"
PPL_NULL = None


def language_of(source):
    """Last path/tag segment of `source`: 'starcoderdata:python' and '.../py' both -> 'python'."""
    seg = str(source or "").rstrip("/").replace(":", "/").split("/")[-1].strip()
    return seg or "_"


def top_level_line_units(text):
    """Partition source lines into top-level units. Each top-level def/class is one unit;
    loose statements/comments/blanks before a def/class are attached to it, a trailing loose
    run is attached to the last unit. A single forward cursor assigns every line exactly once,
    so ''.join(units) == text."""
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [text]
    nodes = [nd for nd in tree.body]
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
        loose.extend(lines[cursor:lo])       # uncovered lines before this node
        seg = "".join(lines[lo:hi])
        if isinstance(nd, DEF):
            add_def(seg)
        else:
            loose.append(seg)
        cursor = hi
    loose.extend(lines[cursor:])             # trailing comments/blanks
    if loose:
        tail = "".join(loose)
        if units:
            units[-1] += tail
        else:
            units.append(tail)
    return units


def _fit_line_piece(line, ntok):
    """One physical line longer than MAX_TOK: cut at a real BPE id boundary. ntok carries the
    tokenizer in `_enc`; a non-tokenizer stand-in (selftest) returns the line whole."""
    if ntok(line) <= MAX_TOK:
        return [line]
    enc = getattr(ntok, "_enc", None)
    if enc is None:
        return [line]
    ids = enc.encode(line).ids
    pieces, start = [], 0
    while start < len(ids):
        pieces.append(enc.decode(ids[start:start + MAX_TOK], skip_special_tokens=False))
        start += MAX_TOK
    return pieces


def _hard_split_long(unit, ntok):
    """Split one >MAX unit into pieces whose REAL encoded length is <=MAX, on line boundaries.
    Packing is linear: budget = additive token sum + (#line junctions), since each junction was
    measured at worst +1 id over 1.09M real starcoder junctions. An over-long physical line is
    cut at its BPE boundary by _fit_line_piece."""
    lines = unit.splitlines(keepends=True)
    pieces, cur, cur_n, cur_j = [], [], 0, 0

    def flush():
        nonlocal cur, cur_n, cur_j
        if cur:
            pieces.append("".join(cur))
            cur, cur_n, cur_j = [], 0, 0

    for ln in lines:
        ln_n = ntok(ln)
        if ln_n > MAX_TOK:
            flush()
            pieces.extend(_fit_line_piece(ln, ntok))
            continue
        # adding this line introduces one new junction (+<=1 id worst case).
        add_j = 1 if cur else 0
        if cur and cur_n + add_j + ln_n > MAX_TOK:
            flush()
            add_j = 0
        cur.append(ln)
        cur_n += ln_n
        cur_j += add_j
    flush()
    return pieces


def chunk_document(text, ntok):
    """Greedily pack top-level units into [512,1024]-id chunks on AST boundaries.
    Returns list[str]. Exact partition: ''.join(chunks) == text. A lone <512 document is one
    chunk; a trailing <512 tail merges into the previous chunk when it stays in window."""
    units = top_level_line_units(text)
    chunks, cur, cur_n, cur_j = [], [], 0, 0

    def flush():
        nonlocal cur, cur_n, cur_j
        if cur:
            chunks.append("".join(cur))
            cur, cur_n, cur_j = [], 0, 0

    for u in units:
        un = ntok(u)
        if un > MAX_TOK:
            flush()
            chunks.extend(_hard_split_long(u, ntok))
            continue
        uj = u.count("\n")
        add_j = min(uj, 1) if cur else 0     # one junction joins this unit to the current chunk
        # HARD CAP by additive sum + junction headroom; flush even if cur is below MIN_TOK
        # (de review, PR #399 -- never satisfy the lower target by overflowing the window).
        if cur and cur_n + add_j + un > MAX_TOK:
            flush()
            add_j = 0
        cur.append(u)
        cur_n += un
        cur_j += add_j
    flush()
    # merge a short trailing tail into the previous chunk while the additive headroom says it
    # stays in window. Do NOT write `chunks[-2] += chunks.pop()`: with exactly two chunks the
    # pop shrinks the list first, so the augmented-assignment setitem at -2 is out of range.
    if len(chunks) >= 2 and ntok(chunks[-1]) < MIN_TOK \
            and ntok(chunks[-2]) + ntok(chunks[-1]) + 1 <= MAX_TOK:
        tail = chunks.pop()
        chunks[-1] += tail
    # FINAL hard-cap backstop: true-encode each emitted chunk once; any that still exceeds
    # MAX_TOK is split on lines. One encode per output chunk, never per packing decision.
    final = []
    for c in chunks:
        final.extend(_hard_split_long(c, ntok) if ntok(c) > MAX_TOK else [c])
    return final


def build_rows(content, source, url, ntok):
    """All chunk rows for one parent doc. sample_id over the FINAL chunk text. length_band is
    the PARENT-DOC char band (aligned with 3b's nl pool, PR #396): every chunk of one doc
    shares it, because a chunk-level band collapses every ~512-1024-token chunk to m/l."""
    parent = content_doc_id(content)
    band = length_band(len(content))
    chunks = chunk_document(content, ntok)
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
def _selftest():
    # EXACTLY ADDITIVE token stand-in: 1 char = 1 id. The packer budgets additive ids plus the
    # number of new junctions (each measured <= +1 on the real tokenizer); the selftest uses the
    # zero-growth boundary so flush decisions are exact here. Non-additivity is tested against
    # the real tokenizer in the pod read-only audit, not here.
    def ntok(s):
        return len(s)

    def funcs(names, body="    x = 1\n    return x\n"):
        return "".join(f"def {nm}(a):\n{body}" for nm in names)

    fails = []

    # 1) exact partition: chunking never loses or reorders a byte (short and long docs).
    for doc in ["import os\n\n" + funcs([f"f{i}" for i in range(40)]),
                "def g():\n    return 2\n",
                "\n".join(f"x = {i}" for i in range(5000))]:
        ch = chunk_document(doc, ntok)
        if "".join(ch) != doc:
            fails.append(f"partition not exact for doc len={len(doc)}")

    # 2) ordinary multi-function doc: every middle chunk is in band, boundaries don't bisect
    #    a function that itself fits.
    doc = "\n".join("import os" for _ in range(6)) + "\n\n" + funcs([f"fn{i}" for i in range(60)])
    ch = chunk_document(doc, ntok)
    for c in ch[1:-1]:
        if not (MIN_TOK - 1 <= ntok(c) <= MAX_TOK):
            fails.append(f"middle chunk out of band: {ntok(c)}")
    # each whole small function appears intact in exactly one chunk
    for nm in [f"fn{i}" for i in range(60)]:
        header = f"def {nm}(a):\n"
        owners = [c for c in ch if header in c]
        if len(owners) != 1:
            fails.append(f"{nm} split/duplicated across {len(owners)} chunks")

    # 3) single tiny document -> one chunk, idx 0, kept (never dropped).
    tiny = "def g():\n    return 2\n"
    rows = build_rows(tiny, "starcoderdata:python", "u", ntok)
    if len(rows) != 1 or rows[0]["chunk_idx"] != 0 or rows[0]["n_chunks"] != 1:
        fails.append("tiny doc did not yield exactly one chunk")
    if rows[0]["kind"] != "code" or rows[0]["ppl"] is not None:
        fails.append("kind/ppl defaults wrong")
    if rows[0]["language"] != "python":
        fails.append(f"language segment wrong: {rows[0]['language']!r}")

    # 4) one function longer than the window is hard-split on complete lines, still exact.
    big = "def big():\n" + "".join(f"    y{i} = {i}\n" for i in range(120))
    assert ntok(big) > MAX_TOK
    ch = chunk_document(big, ntok)
    if "".join(ch) != big or len(ch) < 2:
        fails.append("oversized function not hard-split exactly")
    for c in ch:
        if not c.endswith("\n"):
            fails.append("hard-split piece not line-aligned")

    # 4b) semicolon multi-statement lines are several AST nodes sharing one physical line;
    # the line cursor must emit it once (genB: 218/150k real docs duplicated bytes here).
    semi = ("from gevent import monkey; monkey.patch_all()\n"
            "NR = 2; NC = 1\nmonths = []; total_m = 1; net_total = 0\n"
            + funcs(["a", "b", "c"]))
    u = top_level_line_units(semi)
    if "".join(u) != semi:
        fails.append("semicolon line duplicated/lost in top_level_line_units")
    if "".join(chunk_document(semi, ntok)) != semi:
        fails.append("semicolon doc partition not exact")

    # 4c) de hard-cap regression: a <MIN prefix that would overflow MAX when it absorbs a fitting
    # unit must FLUSH, never emit a >MAX chunk. prefix 400 chars (<512); a ~800-char fitting
    # block; joined 1200+ (>1024) while the block alone fits.
    prefix = "z\n" * 200                        # 400 chars, < MIN_TOK
    bigunit = "q\n" * 400                        # 800 chars <= MAX_TOK
    if not (ntok(prefix) < MIN_TOK):
        fails.append(f"hard-cap prefix not <MIN ({ntok(prefix)})")
    if not (ntok(bigunit) <= MAX_TOK and ntok(prefix) + ntok(bigunit) > MAX_TOK):
        fails.append("hard-cap fixture mis-sized")

    # 4d) EXHAUSTIVE hard cap: no chunk may exceed MAX on every case (char stand-in is exact).
    # Mixed oversized unit (hard-split pieces) followed by a short trailing unit: exercises the
    # two-chunk tail-merge setitem that raised IndexError on a real doc (`chunks[-2] += pop()`).
    mixed = "q\n" * 1200 + "z\n" * 40
    mch = chunk_document(mixed, ntok)
    if "".join(mch) != mixed or len(mch) < 2:
        fails.append("mixed oversized+tail partition/count wrong")
    for d2 in [prefix + "\n" + bigunit, semi, big, mixed,
               "import os\n\n" + funcs([f"f{i}" for i in range(40)])]:
        for c in chunk_document(d2, ntok):
            if ntok(c) > MAX_TOK:
                fails.append(f"chunk over MAX ({ntok(c)}) in doc head {d2[:20]!r}")

    # 5) ids: sample_id is content hash of THAT chunk, parent stable, content_doc_id reused.
    rows = build_rows(funcs(["a", "b", "c", "d", "e", "f", "g", "h"]) * 3,
                      "starcoderdata:python", None, ntok)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        fails.append("chunk sample_ids not unique")
    if len({r["parent_doc_id"] for r in rows}) != 1:
        fails.append("parent_doc_id not constant within doc")
    r0 = rows[0]
    if r0["sample_id"] != content_doc_id(r0["content"]):
        fails.append("sample_id not sha256(chunk)")

    # 6) unparseable input still chunks and partitions (defensive; corpus is ast-gated).
    bad = "def broken(:\n" * 200
    ch = chunk_document(bad, ntok)
    if "".join(ch) != bad:
        fails.append("unparseable doc partition not exact")

    if fails:
        print("FAIL\n" + "\n".join(fails))
        raise SystemExit(1)
    print("l2_code_chunk_pool selftest ok: exact partition, AST boundaries, banding, "
          "hard-split, ids, defensive parse")


def _bpe_counter(root):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(root, "data", "tokenizer.json"))

    def ntok(s):
        return len(tok.encode(s).ids)
    ntok._enc = tok          # used by _fit_line_piece to cut an over-long line at id boundary
    return ntok


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

    ntok = _bpe_counter(a.root)
    files = sorted(f for f in glob.glob(os.path.join(a.root, "data", "corpus", a.domain, "*.jsonl"))
                   if "manifest" not in f and "gate_exclude" not in f)
    step = max(1, len(files) // a.n_shards)
    pick = files[::step][:a.n_shards]
    # bucket by PARENT-doc length band, keeping each doc's chunk family as one unit so a
    # multi-chunk document is never split across the cap.
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
                rows = build_rows(content, d.get("source"), d.get("url"), ntok)
                docs_by_band[rows[0]["length_band"]].append(rows)
                n_parent += 1

    # cap CHUNKS per band equally; draw whole docs, accepting the doc that crosses the cap so
    # long programs are not truncated at the selection boundary either.
    bands = [nm for _, _, nm in LENGTH_BINS]
    per = a.target_chunks // len(bands)
    rng = random.Random(a.seed)
    chosen, manifest_bands = [], {}
    for band in bands:
        families = docs_by_band.get(band, [])
        rng.shuffle(families)
        got = 0
        drawn_docs = 0
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
        "ppl": "null: no code KenLM; nl KenLM not applied to code (agreed 2026-09-16)",
        "bands": manifest_bands,
        "doc_id": "datagen/score_ledger.content_doc_id sha256(text)[:16]",
    }
    with open(a.out + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps({"chunks": len(chosen), "parents": n_parent, "bands": manifest_bands}))


if __name__ == "__main__":
    main()
