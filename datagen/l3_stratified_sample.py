"""Stratified pilot sampler for the L3 teacher-labeling funnel (data-quality P0).

# restartable: CPU-only single-pass reservoir over the corpus; an interrupt costs one
# re-read and leaves no partial output (nothing is written until the whole pass ends),
# matching scripts/audit_sample.py's deliberate pattern.

The old quality head died from one scalar score plus register drift
(docs/lessons/data_quality_methods.md); the Nemotron-CC fix is multi-stratum
sampling so the teacher does not spend 90% of its reads on homogeneous junk.

Strata here are derivable from a corpus row WITHOUT a model:
- language/source: the last path segment of `source` (L3 is all `.../py` today;
  other corpora carry en/zh/...); unknown -> "_".
- length band: binned by a cheap char count, not tokens (no tokenizer on the
  sampling CPU path). Bins are logged in the manifest so a re-draw is comparable.

A single pass over the shards; per-stratum RESERVOIR sampling (Algorithm R) gives
an exactly-uniform-within-stratum fixed-size draw without holding the corpus in
memory. Deterministic from --seed; refuses short strata unless --allow-short.

Output rows carry the stratum metadata the labels need for versioning:
{sample_id, language, length_band, source, url, content}.
A sibling .manifest.json pins source shas, seed, n, bins, and counts per stratum.

Run:
  python datagen/l3_stratified_sample.py \\
      --glob 'data/corpus/code_ultra_l3_noexec_dc/*.jsonl' \\
      --out runs/l3label/pilot.jsonl --per-stratum 100 --seed 20260916
"""

import argparse
import glob
import hashlib
import json
import os
import random

# length bands in characters of `content`; the label set's distribution over these
# is what stops a uniform-random teacher from seeing only short/easy or only long rows.
LENGTH_BINS = [(0, 400, "xs"), (400, 1200, "s"), (1200, 3000, "m"), (3000, 8000, "l"), (8000, 10**12, "xl")]


def length_band(n_chars: int) -> str:
    for lo, hi, name in LENGTH_BINS:
        if lo <= n_chars < hi:
            return name
    return "xl"


def language_of(row: dict) -> str:
    src = str(row.get("source") or "")
    seg = src.rstrip("/").split("/")[-1].strip()
    return seg or "_"


def strata_key(row: dict) -> tuple[str, str]:
    return (language_of(row), length_band(len(str(row.get("content") or ""))))


def content_doc_id(text: str) -> str:
    """Stable per-document CONTENT id, identical to datagen/score_ledger.content_doc_id
    (sha256 of utf-8 content, first 16 hex). Used for the locked English hand-read set so
    fineweb-edu AUC / PPL / the L3 teacher rubric all join on one content-derived id;
    never url/row-number (those move when the corpus is rebuilt)."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def sample_stream(
    paths, per_stratum, seed, allow_short, text_warn_empty=True, doc_id_mode="sequence", only_bands=None
):
    """One reservoir per (lang,band) over every row of every path.

    Returns (reservoirs dict, total_seen dict, stats). Reservoir entries are
    (row_with_meta,). Algorithm R with a per-stratum deterministic RNG so adding or
    removing an unrelated stratum never perturbs another stratum's draw.

    stats = {"empty": rows with no usable content, "bad": corrupt rows,
             "scanned": non-blank lines examined, "bad_by_file": basename -> bad count}.
    Streaming is tolerant: a corrupt row is counted, never silently sampled and never
    aborts the pass; the caller enforces a bad-fraction threshold before writing. A row
    is bad when it is not a JSON object or its `content` is present but not a string;
    missing/null/blank content is a tolerated empty skip, not corruption.

    doc_id_mode: "sequence" -> sample_id assigned at write time (L3 labelling pilot);
    "content" -> each row's doc_id is sha256(content)[:16] at draw time (locked sets
    that must join to score_ledger on content identity).
    """
    pools = {}
    seen = {}
    rngs = {}
    empty = 0
    bad = 0
    scanned = 0
    bad_by_file = {}
    for path in paths:
        bname = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            for _lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                scanned += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    bad_by_file[bname] = bad_by_file.get(bname, 0) + 1
                    continue
                if not isinstance(row, dict):
                    bad += 1
                    bad_by_file[bname] = bad_by_file.get(bname, 0) + 1
                    continue
                content = row.get("content")
                if content is None or (isinstance(content, str) and not content.strip()):
                    empty += 1
                    continue
                if not isinstance(content, str):
                    bad += 1
                    bad_by_file[bname] = bad_by_file.get(bname, 0) + 1
                    continue
                key = strata_key(row)
                if only_bands is not None and key[1] not in only_bands:
                    continue
                seen[key] = seen.get(key, 0) + 1
                rng = rngs.setdefault(key, _seeded(seed, key))
                t = seen[key]
                pool = pools.setdefault(key, [])
                meta = {
                    "sample_id": content_doc_id(content) if doc_id_mode == "content" else "",
                    "language": key[0],
                    "length_band": key[1],
                    "source": row.get("source"),
                    "url": row.get("url"),
                    "content": content,
                }
                if len(pool) < per_stratum:
                    pool.append(meta)
                else:
                    j = rng.randrange(t)
                    if j < per_stratum:
                        pool[j] = meta
    if text_warn_empty and empty:
        print(f"WARN skipped {empty} empty-content rows")
    stats = {"empty": empty, "bad": bad, "scanned": scanned, "bad_by_file": bad_by_file}
    return pools, seen, stats


def _seeded(seed, key):
    h = hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode()).hexdigest()
    r = random.Random()
    r.seed(int(h[:16], 16))
    return r


def _write_shard(path):
    rows = [
        json.dumps({"source": "src/en", "url": "u0", "content": "alpha " * 20}),
        "{not valid json",                                   # corrupt JSON -> bad
        json.dumps(["a", "list", "not", "an", "object"]),    # non-object -> bad
        json.dumps({"source": "src/en", "url": "u2", "content": 12345}),  # non-string -> bad
        json.dumps({"source": "src/en", "url": "u3", "content": "   "}),  # blank -> empty
        "",                                                  # blank line -> ignored
    ]
    # two more clearly different-length valid rows so two strata exist and neither bad
    # row's text can be sampled.
    rows += [
        json.dumps({"source": "src/en", "url": f"v{i}", "content": ("word " * (30 + 400 * i))})
        for i in range(1, 5)
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")


def _selftest() -> int:
    import subprocess
    import sys
    import tempfile

    d = tempfile.mkdtemp()
    shard = os.path.join(d, "corpus.jsonl")
    _write_shard(shard)

    # unit: sample_stream classifies every bad row, never samples one, counts them.
    pools, seen, stats = sample_stream([shard], per_stratum=4, seed=1, allow_short=True)
    assert stats["bad"] == 3, stats
    assert stats["empty"] == 1, stats
    assert stats["scanned"] == 9, stats            # 10 lines minus the one blank line
    assert stats["bad_by_file"][os.path.basename(shard)] == 3, stats["bad_by_file"]
    sampled = [m["content"] for k in pools for m in pools[k]]
    assert all(isinstance(c, str) for c in sampled), "non-string content was sampled"
    assert "12345" not in sampled and "['a'" not in sampled, "a bad row reached a reservoir"
    assert sum(seen.values()) == 5, seen           # only the 5 valid objects are population

    script = os.path.abspath(__file__)
    out = os.path.join(d, "pilot.jsonl")

    def run(*extra):
        return subprocess.run(
            [sys.executable, script, "--glob", shard, "--out", out,
             "--per-stratum", "2", "--seed", "1", "--allow-short", *extra],
            capture_output=True, text=True, timeout=60)

    # 3/9 = 33% bad; default 1% threshold refuses nonzero and writes nothing.
    p = run()
    assert p.returncode != 0 and "REFUSE" in p.stderr, (p.returncode, p.stderr)
    assert not os.path.exists(out), "pool must not be written when bad fraction exceeds"
    assert not os.path.exists(out + ".manifest.json"), "no manifest on refusal"

    # an explicit permissive threshold completes; the manifest records the bad rows.
    p = run("--max-bad-frac", "0.5")
    assert p.returncode == 0, p.stderr
    with open(out, encoding="utf-8") as fh:
        n_out = sum(1 for _ in fh)
    with open(out + ".manifest.json", encoding="utf-8") as fh:
        man = json.load(fh)
    assert n_out == man["n_written"] > 0
    assert man["bad_lines"] == 3 and man["empty_content_rows"] == 1, man
    assert man["input_rows_scanned"] == 9 and man["bad_line_fraction"] == round(3 / 9, 8), man

    # --max-bad-frac 0 forbids any bad row even one, so the same shard still refuses.
    p0 = run("--max-bad-frac", "0")
    assert p0.returncode != 0 and "REFUSE" in p0.stderr, p0.stderr

    # a fully clean shard passes the strictest threshold and reports zero bad.
    clean = os.path.join(d, "clean.jsonl")
    with open(clean, "w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps({"source": "src/en", "url": f"c{i}",
                                 "content": "clean content words " * 30}) + "\n")
    cout = os.path.join(d, "clean_pilot.jsonl")
    pc = subprocess.run(
        [sys.executable, script, "--glob", clean, "--out", cout, "--per-stratum", "2",
         "--seed", "1", "--allow-short", "--max-bad-frac", "0"],
        capture_output=True, text=True, timeout=60)
    assert pc.returncode == 0, pc.stderr
    with open(cout + ".manifest.json", encoding="utf-8") as fh:
        cman = json.load(fh)
    assert cman["bad_lines"] == 0 and cman["input_rows_scanned"] == 3, cman
    print("selftest ok: per-row validation counts bad JSON/non-object/non-string content "
          "(bad=3, empty=1, scanned=9) and never samples one; bad-fraction gate refuses "
          "33% under default 1% and under 0, permits under 50% with manifest accounting, "
          "clean shard passes strict 0")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--glob", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--per-stratum", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument(
        "--max-strata",
        type=int,
        default=None,
        help="optional cap on number of strata (keeps the pilot bounded)",
    )
    ap.add_argument(
        "--doc-id",
        choices=["sequence", "content"],
        default="sequence",
        help="content = sha256(content)[:16] (locked hand-read sets); sequence = stratum-order id (L3 pilot)",
    )
    ap.add_argument(
        "--only-bands", default=None, help="comma list of length bands to keep, e.g. 's,m,l'; others skipped"
    )
    ap.add_argument(
        "--max-bad-frac",
        type=float,
        default=0.01,
        help="refuse nonzero if corrupt input rows (bad JSON / non-object / non-string "
        "content) exceed this fraction of non-blank lines; 0 forbids any bad row",
    )
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    if not a.glob or not a.out:
        ap.error("--glob and --out are required")
    if not 0.0 <= a.max_bad_frac <= 1.0:
        raise SystemExit("REFUSE: --max-bad-frac must be in [0,1]")
    only_bands = set(a.only_bands.split(",")) if a.only_bands else None

    paths = sorted(glob.glob(a.glob))
    if not paths:
        raise SystemExit(f"REFUSE: no files match {a.glob}")
    fps = {
        os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: SIM115
        for p in paths
    }

    pools, seen, stats = sample_stream(
        paths, a.per_stratum, a.seed, a.allow_short, doc_id_mode=a.doc_id, only_bands=only_bands
    )
    bad, scanned = stats["bad"], stats["scanned"]
    bad_frac = (bad / scanned) if scanned else 0.0
    if bad:
        top = sorted(stats["bad_by_file"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        where = ", ".join(f"{f}x{n}" for f, n in top)
        print(f"WARN {bad} bad input row(s) of {scanned} scanned "
              f"({bad_frac:.4%}); never sampled; by file: {where}")
    # corrupt input is loud: sparse corruption is tolerated and COUNTED (a torn tail must
    # not waste a whole pass), but above the fraction the input is judged broken and the
    # run refuses BEFORE any pool byte is written, rather than drawing strata on bad data.
    if bad_frac > a.max_bad_frac:
        raise SystemExit(
            f"REFUSE: bad-line fraction {bad_frac:.4%} ({bad}/{scanned}) exceeds "
            f"--max-bad-frac {a.max_bad_frac:.4%}; fix or rebuild the source shards"
        )
    keys = sorted(pools)
    if a.max_strata:
        # keep the most-populated strata so a cap does not prefer a rare tail.
        keys = sorted(keys, key=lambda k: seen[k], reverse=True)[: a.max_strata]
    short = [f"{k[0]}/{k[1]}={len(pools[k])}<{a.per_stratum}" for k in keys if len(pools[k]) < a.per_stratum]
    if short and not a.allow_short:
        raise SystemExit(
            "REFUSE: short strata " + ", ".join(short) + "; pass --allow-short to accept a smaller pilot"
        )

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    written = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for key in keys:  # stratum order so a reviewer reads each band together
            for meta in pools[key]:
                if a.doc_id == "sequence":
                    meta["sample_id"] = f"{key[0]}-{key[1]}-{written:07d}"
                # locked-set placeholder columns, filled later (never by the sampler):
                meta.setdefault("teacher_labels", None)
                meta.setdefault("hand_read", None)
                out.write(json.dumps(meta, ensure_ascii=False) + "\n")
                written += 1

    manifest = {
        "source_glob": a.glob,
        "source_sha256": fps,
        "n_sources": len(paths),
        "seed": a.seed,
        "per_stratum": a.per_stratum,
        "doc_id": a.doc_id,
        "only_bands": sorted(only_bands) if only_bands else None,
        "length_bins_chars": [[lo, hi, nm] for lo, hi, nm in LENGTH_BINS],
        "n_written": written,
        "input_rows_scanned": scanned,
        "empty_content_rows": stats["empty"],
        "bad_lines": bad,
        "bad_line_fraction": round(bad_frac, 8),
        "bad_lines_by_file": stats["bad_by_file"],
        "max_bad_frac": a.max_bad_frac,
        "strata": {f"{k[0]}/{k[1]}": {"pool": len(pools[k]), "population": seen[k]} for k in keys},
    }
    with open(a.out + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"wrote {written} rows across {len(keys)} strata -> {a.out}")


if __name__ == "__main__":
    main()
