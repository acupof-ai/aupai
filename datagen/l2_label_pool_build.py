#!/usr/bin/env python3
"""Build the L2 encoder teacher-label TRAINING pool: 50k stratified, chunked docs.

The L2 labeler (datagen/l3_label_pilot.py) needs an input already chunked,
stratified and content-id-keyed waiting for it. This produces that input, reusing
the L3 pilot row schema so the labeler consumes it unchanged; it only ADDS chunk
fields and a KenLM PPL stratum.

Two streaming passes (CPU, bounded memory, deterministic from --seed):
  pass 1, per source: score every doc with the matching saved KenLM model
    (l1_ppl_kenlm.py --save-model) and keep ONLY a light ref (path, line-no, source,
    url, char-len, ppl) -- never the text -- plus the PPL list for terciles. Memory
    is O(docs) small dicts, not O(corpus bytes).
  derive: high/mid/low PPL bands at the population 33/67 terciles (real boundaries,
    not guessed constants) and per-stratum chunk-population counts.
  pass 2: re-read the same lines, chunk 512-1024 tokens with the gate tokenizer and
    reservoir-sample within each (kind, domain, char-length-band, ppl-band) stratum
    to 35k NL + 15k code chunks, allocated proportionally with a per-stratum floor so
    high- and low-PPL tails are both present. Only reservoir-held chunk text lives in
    memory (bounded by the target).

Identity: sample_id = content_doc_id(chunk_text) = sha256(chunk utf8)[:16], exactly
the #389 ledger content id; parent_doc_id = content_doc_id(full source doc), with
chunk_idx/n_chunks. No threshold is chosen -- terciles describe the pool, not a cut.

# restartable: reservoir scan emits only at the end; an interrupt costs the two
read passes and leaves no partial output.
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l1_ppl_kenlm import InterpolatedKneserNey, tokenize  # noqa: E402
from l3_stratified_sample import language_of, length_band  # noqa: E402

NL_TARGET_CHUNKS = 35_000
CODE_TARGET_CHUNKS = 15_000
PPL_BANDS = ("low", "mid", "high")
MIN_STRATUM_CHUNKS = 40  # every populated stratum is represented


def content_doc_id(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def terciles(values):
    v = sorted(values)
    if not v:
        return None

    def q(p):
        return v[min(len(v) - 1, int(p * len(v)))]

    return q(1 / 3), q(2 / 3)


def ppl_band(ppl, lo, hi):
    return "low" if ppl <= lo else "mid" if ppl <= hi else "high"


MIN_FRAGMENT_TOKENS = 128  # whole docs shorter than this are dropped, not labeled


def _cap_segments(segments, lo, hi, measure, split):
    """Impose the HARD max on MEASURED segments. Sliding windows leave a 1..lo-1 tail;
    folding it into a full window would overflow (encoder truncation). Repartition the
    last two segments so both halves measure in [lo, hi], searching around the measured
    midpoint (an exact token-id midpoint is only even after decode->re-encode drift)."""
    if len(segments) >= 2 and 0 < measure(segments[-1]) < lo:
        tail = segments.pop()
        prev = segments.pop()
        combo = prev + tail
        mid = len(combo) // 2
        best, best_gap = mid, None
        for cut in range(max(lo, mid - 32), min(len(combo) - lo, mid + 32) + 1):
            a, b = split(combo, cut)
            ma, mb = measure(a), measure(b)
            if ma > hi or mb > hi:
                continue
            gap = abs(ma - mb)
            if best_gap is None or gap < best_gap:
                best, best_gap = cut, gap
        a, b = split(combo, best)
        segments.extend([a, b])
    assert all(measure(s) <= hi for s in segments), "chunk over hard measured max"
    return segments


def _chunk_fn(tokenizer_path, allow_word_count=False):
    """text -> list[chunk_text] each measuring 512-1024 tokens under the GATE
    TOKENIZER. The cap is checked on decode->re-ENCODED text, not on sliced token ids:
    byte-level BPE is not invertible at cut points and a 1024-id slice decodes to text
    that re-encodes up to ~1028 (measured), so an id-based cap emits >1024 chunks.
    A whole document shorter than MIN_FRAGMENT_TOKENS is dropped; single-doc chunks of
    128-511 tokens are kept (the xs/s length strata).

    A failure to load the HF tokenizer is FATAL: silently falling back to word count
    would make the measured <=1024 token gate vacuous (the thing three rounds just
    fixed). Word count is allowed only when the caller explicitly opts in
    (allow_word_count / --word-count), and then the manifest marks the chunks as
    NON-BPE so they can never be mistaken for gate-tokenizer output."""
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)

        def ntok(text):
            return len(tok.encode(text).ids)

        def measured(ids):
            return len(tok.encode(tok.decode(ids)).ids)

        def chunks(text, lo=512, hi=1024):
            ids = tok.encode(text).ids
            if not ids or len(ids) < MIN_FRAGMENT_TOKENS:
                return []
            segs, start = [], 0
            while start < len(ids):
                end = min(start + hi, len(ids))
                # shrink to a boundary whose DECODED text re-encodes within the max
                while end > start and measured(ids[start:end]) > hi:
                    end -= 1
                if end == start:
                    end = start + 1  # forward progress on a pathological token
                segs.append(ids[start:end])
                start = end
            if len(segs) == 1 and measured(segs[0]) < lo:
                return [tok.decode(segs[0])]  # lone 128-511 doc: keep as one short chunk
            segs = _cap_segments(
                segs, lo, hi, measured, lambda s, k: (s[:k], s[k:]))
            return [tok.decode(s) for s in segs]

        return chunks, "hf-tokenizer", ntok
    except Exception as e:
        if not allow_word_count:
            raise SystemExit(
                f"REFUSE: could not load gate HF tokenizer at {tokenizer_path!r}: "
                f"{type(e).__name__}: {e}. The measured 512-1024 token hard cap is only "
                "meaningful under the real tokenizer; refusing rather than silently "
                "falling back to word count. Pass --word-count explicitly to accept the "
                "non-BPE proxy (the manifest will mark the pool accordingly).") from e

        def ntok(text):
            return len(text.split())  # ~1 token/word proxy for the fallback path

        def chunks(text, lo=120, hi=260):  # ~4 tokens/word proxy
            words = text.split()
            if not words:
                return []
            win = [words[i:i + hi] for i in range(0, len(words), hi)]
            win = _cap_segments(win, lo, hi, len, lambda s, k: (s[:k], s[k:]))
            return [" ".join(s) for s in win]
        return chunks, "word-fallback", ntok


def _atomic_write_jsonl(path, lines):
    """Write JSONL atomically: temp file in the same dir, fsync, then os.replace so a
    crash can never leave a half-written pool at `path` (the restartability promise)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for ln in lines:
            fh.write(ln)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _iter_paths(kind2glob):
    for kind, pattern in sorted(kind2glob.items()):
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"REFUSE: no files match {pattern}")
        for p in paths:
            yield kind, p


def _check_quota_feasible(quotas, doc_pop, chunk_pop):
    """Refuse when a stratum is assigned a positive quota but produced zero chunks.
    doc_pop counts KenLM-scorable docs; the selectable unit is a BPE chunk, so a stratum
    whose docs all fall below the MIN_FRAGMENT BPE threshold can have doc_pop>0 with
    chunk_pop==0. Names every unsatisfiable stratum with its quota and populations."""
    unsatisfiable = {
        "/".join(k): {"quota": quotas[k], "doc_population": doc_pop.get(k, 0),
                      "chunk_population": chunk_pop.get(k, 0)}
        for k in quotas if quotas[k] > 0 and chunk_pop.get(k, 0) == 0}
    if unsatisfiable:
        raise SystemExit(
            "REFUSE: quota assigned to strata that produced 0 chunks (doc_pop counts "
            "KenLM-token docs, not BPE-chunkable length): "
            + json.dumps(unsatisfiable, ensure_ascii=False, indent=2)
            + "\nRelax the strata / fragment threshold or source more long docs; an empty "
              "stratum must not be silently under-filled.")


def build(nl_glob, nl_model_path, *, seed, tokenizer_path, out,
          code_glob=None, code_model_path=None, allow_word_count=False):
    chunk_fn, tok_kind, ntok = _chunk_fn(tokenizer_path, allow_word_count)
    # Code is optional: ae owns AST-boundary code chunks (l2_code_chunk_pool.py)
    # with ppl=null until a code KenLM exists. NL-only runs emit pool_en_c4.jsonl.
    models = {"nl": InterpolatedKneserNey.load(nl_model_path)}
    if code_model_path:
        models["code"] = InterpolatedKneserNey.load(code_model_path)
    for m in models.values():
        m._build_continuation()
    globs = {"nl": nl_glob}
    if code_glob:
        if "code" not in models:
            raise SystemExit("REFUSE: --code-glob needs --code-model")
        globs["code"] = code_glob
    targets = {"nl": NL_TARGET_CHUNKS}
    if "code" in globs:
        targets["code"] = CODE_TARGET_CHUNKS

    # ---- pass 1: light refs + PPL + source file shas ----
    refs = {k: [] for k in globs}
    ppls = {k: [] for k in globs}
    fps = {}
    for kind, p in _iter_paths(globs):
        with open(p, "rb") as srcf:
            fps[os.path.basename(p)] = hashlib.sha256(srcf.read()).hexdigest()
        model = models[kind]
        with open(p, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("content") or ""
                toks = tokenize(text)
                if not toks:
                    continue
                ppl = model.perplexity(text)
                if not (isinstance(ppl, (int, float)) and math.isfinite(ppl)):
                    continue
                refs[kind].append((p, lineno, row.get("source"), row.get("url"),
                                   len(text), ppl))
                ppls[kind].append(ppl)

    bounds, key_of = {}, {}
    for kind in refs:
        t = terciles(ppls[kind])
        if t is None:
            raise SystemExit(f"REFUSE: no scorable {kind} docs")
        bounds[kind] = t
        lo, hi = t
        # per-stratum chunk population: must chunk to count, but use char length as a
        # cheap proxy is wrong; instead chunk only here is costly in pass1. Count a
        # doc's chunks by token-window estimate from char len is also avoidable: do an
        # exact count lazily while building a doc->key map, streaming text not stored.
        key_of[kind] = {}
        for r in refs[kind]:
            _, _, source, _, charlen, ppl = r
            key_of[kind][(r[0], r[1])] = (
                kind, language_of({"source": source}),
                length_band(charlen), ppl_band(ppl, lo, hi))

    # exact chunk populations require reading text once more and chunking; fold that
    # into pass 2 by first sizing quotas from DOC counts (one doc >= 1 chunk) then
    # sampling chunk-by-chunk. Quotas proportional to docs are stable across strata.
    doc_pop = collections.Counter()
    for kind in refs:
        for r in refs[kind]:
            doc_pop[key_of[kind][(r[0], r[1])]] += 1
    quotas = _allocate_quotas(doc_pop, targets)

    # ---- pass 2: re-read refs in order, chunk, reservoir within stratum ----
    # group refs by path so each file is opened once; refs already in scan order.
    by_path = collections.defaultdict(list)
    for kind in refs:
        for r in refs[kind]:
            by_path[r[0]].append((kind, r))

    rng = random.Random(seed)
    reservoirs = collections.defaultdict(list)
    chunk_pop = collections.Counter()
    # whole scorable docs the chunker returned nothing for: these are the
    # <MIN_FRAGMENT_TOKENS extreme fragments, dropped and COUNTED (never silent).
    dropped_docs = collections.Counter()
    dropped_tokens = collections.Counter()
    for p, items in by_path.items():
        by_line = {r[1]: (kind, r) for kind, r in items}
        with open(p, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh):
                if lineno not in by_line:
                    continue
                kind, r = by_line[lineno]
                _, _, source, url, _charlen, _ppl = r
                text = json.loads(line).get("content") or ""
                chs = chunk_fn(text)
                key = key_of[kind][(p, lineno)]
                if not chs:
                    # pass1 already scored this doc, so empty here = whole doc <
                    # MIN_FRAGMENT_TOKENS: dropped, counted per kind, not silent.
                    dropped_docs[kind] += 1
                    dropped_tokens[kind] += ntok(text)
                    continue
                q = quotas.get(key, 0)
                parent = content_doc_id(text)
                for ci, ch in enumerate(chs):
                    chunk_pop[key] += 1
                    if q == 0:
                        continue
                    t = chunk_pop[key]
                    entry = {
                        "sample_id": content_doc_id(ch),
                        "parent_doc_id": parent,
                        "chunk_idx": ci,
                        "n_chunks": len(chs),
                        "language": key[1],
                        "kind": key[0],
                        "length_band": key[2],
                        "ppl_band": key[3],
                        "ppl": round(r[5], 4),
                        "source": source,
                        "url": url,
                        "content": ch,
                        "teacher_labels": None,
                        "hand_read": None,
                    }
                    pool = reservoirs[key]
                    if len(pool) < q:
                        pool.append(entry)
                    else:
                        j = rng.randrange(t)
                        if j < q:
                            pool[j] = entry

    # feasibility gate after pass 2 has counted real chunks, before any pool byte writes
    _check_quota_feasible(quotas, doc_pop, chunk_pop)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    out_lines = []
    over_max = collections.Counter()
    short_kept = collections.Counter()      # 128-511 token single-doc chunks kept
    short_kept_tokens = collections.Counter()
    token_hist = collections.Counter()
    for key in sorted(reservoirs):
        for entry in reservoirs[key]:
            tlen = ntok(entry["content"])
            token_hist[min(tlen // 128 * 128, 2048)] += 1
            if tlen > 1024:
                over_max[key[0]] += 1
            if tlen < 512:
                short_kept[key[0]] += 1
                short_kept_tokens[key[0]] += tlen
            out_lines.append(json.dumps(entry, ensure_ascii=False) + "\n")
    written = len(out_lines)

    manifest = {
        "nl_glob": nl_glob, "code_glob": code_glob,
        "source_sha256": fps, "seed": seed,
        "tokenizer_chunk": tok_kind, "chunk_tokens": [512, 1024],
        "targets": targets,
        "ppl_terciles": {k: [round(bounds[k][0], 4), round(bounds[k][1], 4)]
                         for k in bounds},
        "ppl_bands_note": "low<=q33<=mid<=q67<high on the scorable pool; descriptive, not a cut",
        "n_docs_scored": {k: len(refs[k]) for k in refs},
        "n_chunks_written": written,
        "min_fragment_tokens": MIN_FRAGMENT_TOKENS,
        "dropped_fragments": {
            k: {"docs": dropped_docs.get(k, 0),
                "tokens": dropped_tokens.get(k, 0)}
            for k in targets},
        "short_chunks_kept_128_511": {
            k: {"chunks": short_kept.get(k, 0),
                "tokens": short_kept_tokens.get(k, 0)}
            for k in targets},
        "chunks_over_1024_must_be_zero": {k: over_max.get(k, 0) for k in targets},
        "chunk_token_hist_128bins": {str(k): token_hist[k]
                                     for k in sorted(token_hist)},
        "strata": {
            "/".join(k): {"quota": quotas.get(k, 0),
                          "doc_population": doc_pop.get(k, 0),
                          "chunk_population": chunk_pop.get(k, 0),
                          "drawn": len(reservoirs[k])}
            for k in sorted(set(doc_pop) | set(chunk_pop))},
    }
    # manifest first, pool second: a reader that sees the pool always sees a manifest
    # describing exactly these rows; both writes are tmp+fsync+rename (no half files).
    _atomic_write_json(out + ".manifest.json", manifest)
    _atomic_write_jsonl(out, out_lines)
    return written, manifest


def _allocate_quotas(doc_pop, target):
    """Proportional allocation of each kind's chunk target across its strata using
    doc population as the weight, with a per-populated-stratum floor capped at its
    population; the remainder fills the largest strata."""
    quotas = {}
    for kind, want in target.items():
        keys = [k for k in doc_pop if k[0] == kind]
        populated = [k for k in keys if doc_pop[k] > 0]
        for k in populated:
            quotas[k] = min(MIN_STRATUM_CHUNKS, doc_pop[k])
        remaining = want - sum(quotas[k] for k in populated)
        while remaining > 0:
            eligible = [k for k in populated if quotas[k] < doc_pop[k]]
            if not eligible:
                break
            space_total = sum(doc_pop[k] - quotas[k] for k in eligible)
            gave = 0
            for k in eligible:
                share = int(round(remaining * (doc_pop[k] - quotas[k]) / space_total))
                add = min(share, doc_pop[k] - quotas[k])
                quotas[k] += add
                gave += add
            if gave == 0:
                k = max(eligible, key=lambda x: doc_pop[x] - quotas[x])
                quotas[k] += 1
                gave = 1
            remaining -= gave
    return quotas


def _selftest() -> int:
    import tempfile

    # quota feasibility: a positive quota on a stratum with 0 chunks must refuse (the
    # doc_pop vs BPE-chunk-population gap); normal populated strata still get their quota.
    k_full = ("nl", "en", "m", "mid")
    k_empty = ("nl", "en", "xs", "low")
    doc_pop = {k_full: 100, k_empty: 50}
    chunk_pop = {k_full: 120}  # the xs/low stratum's docs all chunk to nothing
    quotas = _allocate_quotas(doc_pop, {"nl": 40})
    assert quotas[k_full] > 0 and quotas[k_empty] > 0, quotas
    try:
        _check_quota_feasible(quotas, doc_pop, chunk_pop)
    except SystemExit:
        pass
    else:
        raise AssertionError("quota>0 with chunk_pop==0 must refuse")
    # once every quota'd stratum has at least one chunk, the gate passes (it does NOT
    # demand full satisfaction -- reservoir under-sampling is normal, only an empty stratum)
    chunk_pop[k_empty] = 3
    _check_quota_feasible(quotas, doc_pop, chunk_pop)
    # a stratum with quota 0 and 0 chunks is not an error (nothing was promised)
    _check_quota_feasible({k_full: 10}, doc_pop, chunk_pop)

    # hard-max contract on MEASURED segments: a short trailing window must NEVER
    # overflow the previous full window to >hi (encoder truncation, the de-399
    # shape). Exhaustive over every possible tail length 1..lo-1 and 0-3 full windows.
    for hi, lo in ((1024, 512), (260, 120)):
        def split(s, k):
            return s[:k], s[k:]
        for n_full in range(0, 4):
            for tail in range(0, lo):
                segs = [[0] * hi for _ in range(n_full)]
                if tail:
                    segs.append([0] * tail)
                out = _cap_segments([list(s) for s in segs], lo, hi, len, split)
                assert all(len(s) <= hi for s in out), (hi, n_full, tail, out)
                if segs and 0 < len(segs[-1]) < lo and n_full >= 1:
                    assert len(out[-1]) >= lo, (hi, n_full, tail, out)
    # a MEASURED-drift measure (decode->re-encode inflates near the boundary) still
    # caps: measure = id count + 4 on long pieces, emulating the observed +4 BPE drift.
    def drift(s):  # emulate decode->re-encode +4 inflation on long pieces
        return len(s) + (4 if len(s) >= 512 else 0)

    def drift_split(s, k):
        return s[:k], s[k:]

    drift_in = [[0] * 1024, [0] * 40]
    drift_out = _cap_segments([list(s) for s in drift_in], 512, 1024, drift,
                              drift_split)
    assert all(drift(s) <= 1024 for s in drift_out), [drift(s) for s in drift_out]
    # the word-fallback chunker itself honors the cap on a tail-triggering doc
    wchunks, wkind, _ = _chunk_fn("", allow_word_count=True)
    assert wkind == "word-fallback"
    for n_words in (261, 300, 521, 1000, 260 * 3 + 7):
        c = wchunks(" ".join(f"w{i}" for i in range(n_words)))
        assert all(len(x.split()) <= 260 for x in c), (n_words, [len(x.split()) for x in c])
    # broken world: a bad/missing tokenizer WITHOUT explicit word-count must REFUSE,
    # never silently degrade (that would void the measured token hard cap).
    for bad_path in ("", "/nonexistent/tokenizer/path.json"):
        try:
            _chunk_fn(bad_path, allow_word_count=False)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"tokenizer load failure at {bad_path!r} must refuse")

    sents = ["the", "model", "reads", "each", "document", "and", "scores",
             "the", "words", "in", "order", "natural", "prose", "repeats",
             "the", "same", "common", "function", "words", "often"]
    gib = ["zqxwk", "kjvbn", "mxqhz", "qwpzx", "vbnmk", "xzjlq"]

    def doc(kind, n_words):
        if kind == "low":
            words = [sents[i % len(sents)] for i in range(n_words)]
        elif kind == "mid":
            words = [(sents[i % len(sents)] if i % 2 else gib[i % len(gib)])
                     for i in range(n_words)]
        else:
            words = [gib[(i * 7 + 3) % len(gib)] for i in range(n_words)]
        return " ".join(words)

    kinds = ["low"] * 3 + ["mid"] * 3 + ["high"] * 3
    lens = [110, 110, 580, 110, 580, 110, 580, 110, 110]
    rows = [{"content": doc(k, n), "source": "selftest/en"}
            for k, n in zip(kinds, lens, strict=True)]

    tmp = tempfile.mkdtemp()
    corpus = os.path.join(tmp, "nl.jsonl")
    with open(corpus, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    model = InterpolatedKneserNey(order=2)
    model.train_lines([" ".join(sents)] * 40)
    model._finalize_discount()
    mpath = os.path.join(tmp, "nl.model.json")
    model.save(mpath)
    out = os.path.join(tmp, "pool.jsonl")
    written, man = build(corpus, mpath, seed=7, tokenizer_path="", out=out, allow_word_count=True)
    assert written > 0, "no chunks written"
    assert man["n_docs_scored"]["nl"] == 9, man["n_docs_scored"]
    assert man["tokenizer_chunk"] == "word-fallback", man["tokenizer_chunk"]
    with open(out, encoding="utf-8") as fh:
        got = [json.loads(l) for l in fh]
    bands = {e["ppl_band"] for e in got}
    assert {"low", "mid", "high"} <= bands, bands
    lbands = {e["length_band"] for e in got}
    assert {"s", "l"} <= lbands, lbands
    for e in got:
        assert e["sample_id"] == content_doc_id(e["content"]), "sample_id not content hash"
        assert e["kind"] == "nl" and e["language"] == "en"
    assert any(e["n_chunks"] >= 2 for e in got), "no multi-chunk doc drawn"
    for e in got:
        assert 0 <= e["chunk_idx"] < e["n_chunks"], (e["chunk_idx"], e["n_chunks"])
    out2 = os.path.join(tmp, "pool2.jsonl")
    w2, _ = build(corpus, mpath, seed=7, tokenizer_path="", out=out2, allow_word_count=True)

    def ids_at(path):
        with open(path, encoding="utf-8") as fh:
            return sorted(json.loads(l)["sample_id"] for l in fh)

    ids1, ids2 = ids_at(out), ids_at(out2)
    assert w2 == written and ids1 == ids2, "not deterministic at fixed seed"

    # atomicity known-answer: both outputs present and complete, no .tmp left, and the
    # manifest's n_chunks_written equals the actual pool row count (crash can never leave
    # a half pool at the target path, nor a pool whose manifest is missing/short).
    for p in (out, out + ".manifest.json"):
        assert os.path.exists(p), p
        assert not os.path.exists(p + ".tmp"), f"stale temp at {p}.tmp"
    with open(out, encoding="utf-8") as nf:
        assert sum(1 for _ in nf) == man["n_chunks_written"]
    # the atomic helpers themselves: complete target, no tmp, content exact
    ap_path = os.path.join(tmp, "atomic.jsonl")
    _atomic_write_jsonl(ap_path, ['{"a":1}\n', '{"b":2}\n'])
    with open(ap_path, encoding="utf-8") as af:
        assert af.read() == '{"a":1}\n{"b":2}\n'
    assert not os.path.exists(ap_path + ".tmp")
    aj_path = os.path.join(tmp, "atomic.json")
    _atomic_write_json(aj_path, {"k": 3})
    with open(aj_path, encoding="utf-8") as jf:
        assert json.load(jf) == {"k": 3}
    assert not os.path.exists(aj_path + ".tmp")

    # end-to-end WIRING test for the quota-feasibility gate (a unit test of
    # _check_quota_feasible alone cannot prove build() calls it). Enough docs land in one
    # char band/PPL band to earn a positive per-stratum floor quota, but the chunker yields
    # NOTHING for them (simulates the HF MIN_FRAGMENT drop where a stratum is populated by
    # KenLM-token docs yet every doc is below the BPE fragment threshold). build() must
    # SystemExit and leave NO pool and NO manifest behind. An all-dropping chunker is
    # injected because the offline word proxy deliberately has no fragment drop.
    empty_dir = tempfile.mkdtemp()
    empty_corpus = os.path.join(empty_dir, "nl.jsonl")
    short_text = " ".join(sents) * 3
    with open(empty_corpus, "w", encoding="utf-8") as fh:
        for _ in range(80):
            fh.write(json.dumps({"content": short_text, "source": "selftest/en"}) + "\n")
    empty_out = os.path.join(empty_dir, "pool_empty.jsonl")

    def all_dropping_chunker(_path, _allow=False):
        return (lambda _t: []), "hf-tokenizer", lambda _t: 0

    globals()["_chunk_fn"], saved_chunk_fn = all_dropping_chunker, globals()["_chunk_fn"]
    raised = False
    try:
        build(empty_corpus, mpath, seed=7, tokenizer_path="", out=empty_out,
              allow_word_count=True)
    except SystemExit:
        raised = True
    finally:
        globals()["_chunk_fn"] = saved_chunk_fn
    assert raised, "build() must refuse a positive-quota stratum that chunks to 0"
    assert not os.path.exists(empty_out), "no pool written on infeasible quota"
    assert not os.path.exists(empty_out + ".manifest.json"), "no manifest on refusal"

    print(f"selftest ok: {written} chunks, bands={sorted(bands)}, "
          f"length_bands={sorted(lbands)}, multi-chunk rows present, tokenizer-failure "
          f"refused, pool+manifest atomic (no tmp, counts match), build() end-to-end "
          f"refuses an infeasible positive-quota stratum and writes nothing")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--nl-glob", default=None)
    ap.add_argument("--code-glob", default=None)
    ap.add_argument("--nl-model", default=None)
    ap.add_argument("--code-model", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--word-count", action="store_true",
                    help="explicitly accept the non-BPE word-count chunker; the manifest "
                         "marks the pool word-fallback instead of hf-tokenizer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260916)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    missing = [n for n, v in (("--nl-glob", a.nl_glob), ("--nl-model", a.nl_model),
                              ("--tokenizer", a.tokenizer), ("--out", a.out))
               if not v]
    if missing:
        ap.error(f"missing required: {' '.join(missing)}")
    written, manifest = build(
        a.nl_glob, a.nl_model,
        code_glob=a.code_glob, code_model_path=a.code_model,
        seed=a.seed, tokenizer_path=a.tokenizer, out=a.out,
        allow_word_count=a.word_count)
    print(f"wrote {written} chunks -> {a.out}")
    print(json.dumps({k: manifest[k] for k in
                      ("n_docs_scored", "n_chunks_written", "ppl_terciles")}, indent=2))


if __name__ == "__main__":
    main()
