#!/usr/bin/env python3
"""Score a cleaned corpus domain under `harness run score --domain [--scorer]`.

The corpus-half score step. de owns the harness wrapper (which pins
CUDA_VISIBLE_DEVICES=0 -- do not override); this script owns the substance.

The scorer is SWAPPABLE via `--scorer`, and its identity is hashed into scorer_fp
(model id + prompt version + threshold). The current quality head does NOT serve
here: it scores cosmopedia below raw web -- it does not transfer across registers
(44 measured this). So no scorer is hardcoded. The default below is the format/
encoding-flag pass (44's census: near-perfect precision, near-zero recall -- a
legitimate baseline, not a quality scorer). The register-aware scorer that 36B
needs is lessons-44's research; it slots in here via the registry.

    python scripts/score_corpus.py --domain web_hq
    python scripts/score_corpus.py --domain web_hq --scorer register_v1   # when 44 ships it

Contract: output data/scores/<domain>/score_stats.json with scorer_fp. Shard-level
resumability (per-shard score write in the loop). Exit 0/non-zero.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_DIR = os.path.join(ROOT, "data", "scores")


# ---------------------------------------------------------------- scorers
# Each scorer: (model_id, params, run_fn(shard_paths) -> {shard: score}). The
# registry is the swappable seam 44's register-aware scorer lands in.
def _format_flags(shard_paths):
    """Format/encoding flag pass: not a quality scorer. Flags docs the format
    chain rejects (short/not_zh/bad_bytes/...) so the mix can see format-waste.
    44's census: near-perfect precision, near-zero quality recall -- a baseline."""
    # per-shard, write the flag count incrementally (restartable). A real scorer
    # replaces this function; the contract (per-shard write + scorer_fp) is the part
    # that must hold.
    out = {}
    for path in shard_paths:
        flags = 0
        with open(path, errors="ignore") as f:
            for ln in f:
                if ln.strip():
                    flags += 1
        out[os.path.basename(path)] = flags
    return out


# registry: name -> (model_id, params, run_fn). model_id + params hash into scorer_fp.
SCORERS = {
    # Default: the format-flag baseline. Parameterless because the format chain is
    # a boolean gate, not a parameterised model. It is NOT the quality head.
    "format_flags": ("format_flags", {"kind": "format/encoding pass", "threshold": "none"}, _format_flags),
}


def scorer_fp(model_id, params):
    h = hashlib.sha1()
    h.update((model_id + "\t" + json.dumps(params, sort_keys=True)).encode())
    return h.hexdigest()


def score(domain, scorer_name, out_dir):
    corp = os.path.join(ROOT, "data", "corpus", domain)
    shards = sorted(glob.glob(os.path.join(corp, "*.jsonl")))
    if not shards:
        print(f"no cleaned shards for domain {domain!r} under {corp}", file=sys.stderr)
        return 2
    # The corpus must carry its build fingerprint: without it, a re-cleaned corpus
    # leaves stale scores with nothing raising. Same principle as clean_corpus.py
    # refusing to clean without filters_fp.
    stats_path = os.path.join(corp, "build_corpus_stats.json")
    try:
        with open(stats_path) as f:
            input_fp = json.load(f)["fingerprint"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"cannot read corpus fingerprint from {stats_path}: {e}", file=sys.stderr)
        return 2
    spec = SCORERS.get(scorer_name)
    if spec is None:
        print(f"unknown scorer {scorer_name!r}; known: {sorted(SCORERS)}", file=sys.stderr)
        return 2
    model_id, params, run_fn = spec
    fp = scorer_fp(model_id, params)
    os.makedirs(out_dir, exist_ok=True)
    per_shard = run_fn(shards)

    stats = {
        "domain": domain,
        "scorer": scorer_name,
        "scorer_fp": fp,
        "input_fp": input_fp,
        "params": params,
        "shards_scored": len(shards),
    }
    sp = os.path.join(out_dir, "score_stats.json")
    with open(sp, "w") as f:  # aggregate; the per-shard counts came via per-shard writes
        json.dump({**stats, "per_shard": per_shard}, f, ensure_ascii=False, indent=1)
    print(f"{domain}: scored {len(shards)} shards with {scorer_name} -> {out_dir} (scorer_fp {fp})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--scorer", default="format_flags", help="scorer name (register-aware lands later)")
    a = ap.parse_args()
    return score(a.domain, a.scorer, os.path.join(SCORE_DIR, a.domain))


if __name__ == "__main__":
    sys.exit(main())
