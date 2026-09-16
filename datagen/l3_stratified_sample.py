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


def sample_stream(paths, per_stratum, seed, allow_short, text_warn_empty=True):
    """One reservoir per (lang,band) over every row of every path.

    Returns (reservoirs dict, total_seen dict, empty count). Reservoir entries are
    (row_with_meta,). Algorithm R with a per-stratum deterministic RNG so adding or
    removing an unrelated stratum never perturbs another stratum's draw.
    """
    pools = {}
    seen = {}
    rngs = {}
    empty = 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                content = row.get("content")
                if not content:
                    empty += 1
                    continue
                key = strata_key(row)
                seen[key] = seen.get(key, 0) + 1
                rng = rngs.setdefault(key, _seeded(seed, key))
                t = seen[key]
                pool = pools.setdefault(key, [])
                meta = {
                    "sample_id": "",
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
    return pools, seen, empty


def _seeded(seed, key):
    h = hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode()).hexdigest()
    r = random.Random()
    r.seed(int(h[:16], 16))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-stratum", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument(
        "--max-strata",
        type=int,
        default=None,
        help="optional cap on number of strata (keeps the pilot bounded)",
    )
    a = ap.parse_args()

    paths = sorted(glob.glob(a.glob))
    if not paths:
        raise SystemExit(f"REFUSE: no files match {a.glob}")
    fps = {
        os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: SIM115
        for p in paths
    }

    pools, seen, _empty = sample_stream(paths, a.per_stratum, a.seed, a.allow_short)
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
                meta["sample_id"] = f"{key[0]}-{key[1]}-{written:07d}"
                out.write(json.dumps(meta, ensure_ascii=False) + "\n")
                written += 1

    manifest = {
        "source_glob": a.glob,
        "source_sha256": fps,
        "n_sources": len(paths),
        "seed": a.seed,
        "per_stratum": a.per_stratum,
        "length_bins_chars": [[lo, hi, nm] for lo, hi, nm in LENGTH_BINS],
        "n_written": written,
        "strata": {f"{k[0]}/{k[1]}": {"pool": len(pools[k]), "population": seen[k]} for k in keys},
    }
    with open(a.out + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"wrote {written} rows across {len(keys)} strata -> {a.out}")


if __name__ == "__main__":
    main()
