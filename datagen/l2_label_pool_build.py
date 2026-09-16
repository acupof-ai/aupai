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


def _cap_chunks(segments, lo, hi):
    """Impose the HARD max: every emitted segment is <= hi items. Sliding windows
    of width hi leave a 1..lo-1 tail; folding it into the previous full window would
    overflow to hi+lo-1 (encoder truncation). Instead repartition the last
    hi+tail items into two pieces both in [lo, hi]. A lone document shorter than lo
    stays one short chunk (represents the xs length band)."""
    if len(segments) >= 2 and 0 < len(segments[-1]) < lo:
        tail = segments.pop()
        prev = segments.pop()
        combo = prev + tail
        cut = max(lo, min(len(combo) // 2, len(combo) - lo))
        segments.append(combo[:cut])
        segments.append(combo[cut:])
    assert all(len(s) <= hi for s in segments), "chunk over hard token max"
    return segments


def _chunk_fn(tokenizer_path):
    """text -> list[chunk_text] of 512-1024 tokens via the gate tokenizer; every chunk
    is hard-capped at 1024 tokens (a short tail repartitions the last boundary, never
    overflows). Word-count fallback (flagged in the manifest) only if the HF tokenizer
    is unavailable."""
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)

        def chunks(text, lo=512, hi=1024):
            ids = tok.encode(text).ids
            if not ids:
                return []
            win = [ids[i:i + hi] for i in range(0, len(ids), hi)]
            win = [list(s) for s in _cap_chunks(win, lo, hi)]
            return [tok.decode(w) for w in win]

        return chunks, "hf-tokenizer"
    except Exception:
        def chunks(text, lo=120, hi=260):  # ~4 tokens/word proxy
            words = text.split()
            if not words:
                return []
            win = [words[i:i + hi] for i in range(0, len(words), hi)]
            win = _cap_chunks(win, lo, hi)
            return [" ".join(s) for s in win]
        return chunks, "word-fallback"


def _iter_paths(kind2glob):
    for kind, pattern in sorted(kind2glob.items()):
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"REFUSE: no files match {pattern}")
        for p in paths:
            yield kind, p


def build(nl_glob, nl_model_path, *, seed, tokenizer_path, out,
          code_glob=None, code_model_path=None):
    chunk_fn, tok_kind = _chunk_fn(tokenizer_path)
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
                if not chs:
                    continue
                key = key_of[kind][(p, lineno)]
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

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    written = 0
    with open(out, "w", encoding="utf-8") as fo:
        for key in sorted(reservoirs):
            for entry in reservoirs[key]:
                fo.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

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
        "strata": {
            "/".join(k): {"quota": quotas.get(k, 0),
                          "doc_population": doc_pop.get(k, 0),
                          "chunk_population": chunk_pop.get(k, 0),
                          "drawn": len(reservoirs[k])}
            for k in sorted(set(doc_pop) | set(chunk_pop))},
    }
    with open(out + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
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

    # hard-max contract: a short trailing window must NEVER overflow the previous
    # full window to >1024 (encoder truncation, the de-399 shape). Exhaustive over
    # every possible tail length 1..511 and multiples of the window.
    for hi, lo in ((1024, 512), (260, 120)):
        for n_full in range(0, 4):
            for tail in range(0, lo):
                segs = [[0] * hi for _ in range(n_full)]
                if tail:
                    segs.append([0] * tail)
                out = _cap_chunks([list(s) for s in segs], lo, hi)
                assert all(len(s) <= hi for s in out), (hi, n_full, tail, out)
                if segs and 0 < len(segs[-1]) < lo and n_full >= 1:
                    assert len(out[-1]) >= lo, (hi, n_full, tail, out)
    # the word-fallback chunker itself honors the cap on a tail-triggering doc
    wchunks, wkind = _chunk_fn("")  # no tokenizer file -> word fallback
    assert wkind == "word-fallback"
    for n_words in (261, 300, 521, 1000, 260 * 3 + 7):
        c = wchunks(" ".join(f"w{i}" for i in range(n_words)))
        assert all(len(x.split()) <= 260 for x in c), (n_words, [len(x.split()) for x in c])

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
    written, man = build(corpus, mpath, seed=7, tokenizer_path="", out=out)
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
    w2, _ = build(corpus, mpath, seed=7, tokenizer_path="", out=out2)

    def ids_at(path):
        with open(path, encoding="utf-8") as fh:
            return sorted(json.loads(l)["sample_id"] for l in fh)

    ids1, ids2 = ids_at(out), ids_at(out2)
    assert w2 == written and ids1 == ids2, "not deterministic at fixed seed"
    print(f"selftest ok: {written} chunks, bands={sorted(bands)}, "
          f"length_bands={sorted(lbands)}, multi-chunk rows present")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--nl-glob", default=None)
    ap.add_argument("--code-glob", default=None)
    ap.add_argument("--nl-model", default=None)
    ap.add_argument("--code-model", default=None)
    ap.add_argument("--tokenizer", default=None)
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
        seed=a.seed, tokenizer_path=a.tokenizer, out=a.out)
    print(f"wrote {written} chunks -> {a.out}")
    print(json.dumps({k: manifest[k] for k in
                      ("n_docs_scored", "n_chunks_written", "ppl_terciles")}, indent=2))


if __name__ == "__main__":
    main()
