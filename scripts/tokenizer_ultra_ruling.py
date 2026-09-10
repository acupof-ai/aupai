#!/usr/bin/env python3
"""Tokenizer ruling on UltraData-Code L2+L3 python (user order 2026-09-10, via fb):
# restartable: single deterministic BPE fit plus reads; an interrupt just re-runs
# (no accumulated state to lose), so there is no per-shard write to make.
frozen 32K vs a fresh 20K fitted on the same distribution.

Unfreeze condition 2 is corpus distribution change; condition 3 is the extrinsic
two-vocab test this script runs. The candidate is fitted here on UltraData code --
the distribution V4.1 actually trains on -- not on the old keep-set+textbooks proxy
(data/vocab_sweep/p1_v20000.json, measured as a third column when present).

Fit/held-out are disjoint document sets (a vocab scored on its fit text flatters its
own tail). One fit (seed 7, deterministic BPE); held-out metrics drawn at three
seeds -- the measurement side is where never-used sampling noise lives
(facts/tokenizer.json#tok.never_used_not_decidable).

Gates per scripts/tokenizer_eval.py: round-trip lossless and all-256-bytes are vetoes;
hanzi whole-char is undefined on English code (byte-fragment tokens reported instead);
ref fertility is corpus-independent (REF_EN fixed string), its recorded value printed.

    python3 scripts/tokenizer_ultra_ruling.py \
        --corpus_dirs data/corpus/code_ultra_l2,data/corpus/code_ultra_l3 \
        --shards 3 --fit_chars 120000000 --eval_chars 80000000 \
        --json runs/tokenizer_ultra_ruling.json
"""
import argparse
import glob
import json
import os
import random
import sys

ROOT = os.environ.get("AUPAI_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_p1_tokenizer import fit_vocab  # noqa: E402
from tokenizer_eval import collect  # noqa: E402

SEEDS = (7, 13, 21)
TARGET_V = 20000


def read_level(d, shards, cap_chars, rng):
    fs = sorted(glob.glob(os.path.join(d, "*.jsonl")))
    if not fs:
        sys.exit(f"no converted shards under {d}")
    fs = rng.sample(fs, min(shards, len(fs)))
    rows = []
    got = 0
    for f in fs:
        with open(f, encoding="utf-8") as fh:
            lines = fh.readlines()
        rng.shuffle(lines)
        for line in lines:
            c = json.loads(line).get("content", "")
            if c:
                rows.append(c)
                got += len(c.encode("utf-8"))
                if got >= cap_chars:
                    return rows, got
    return rows, got


def draw(corpus_root, dirs, shards, fit_chars, eval_chars, n_seeds, seed):
    """Per level: fit rows and n_seeds DISJOINT held-out segments. Read fit +
    n_seeds*eval bytes, cut the fit, then shuffle the tail with one fixed RNG and
    slice it into contiguous byte-budgeted segments -- the metrics are
    order-invariant token counts, so three reshuffles of one pool give three
    identical columns (the 3b review finding); the spread has to come from
    disjoint documents."""
    rng = random.Random(seed)
    fit, ev, counts = {}, {}, {}
    for d in dirs:
        rows, got = read_level(os.path.join(corpus_root, d), shards,
                               fit_chars + n_seeds * eval_chars, rng)
        rng.shuffle(rows)
        cut = next(i for i in range(len(rows))
                   if sum(len(r.encode("utf-8")) for r in rows[:i]) >= fit_chars)
        fit[d] = rows[:cut]
        tail = rows[cut:]
        parts, i = [], 0
        for _ in range(n_seeds):
            picked, used = [], 0
            while i < len(tail) and used < eval_chars:
                picked.append(tail[i])
                used += len(tail[i].encode("utf-8"))
                i += 1
            parts.append(picked)
        ev[d] = parts
        counts[d] = {
            "n_docs": len(rows), "n_fit": cut,
            "n_eval": [len(p) for p in parts],
            "fit_bytes": sum(len(r.encode("utf-8")) for r in rows[:cut]),
            "eval_bytes": [sum(len(r.encode("utf-8")) for r in p) for p in parts],
            "shards": [os.path.basename(f) for f in sorted(glob.glob(os.path.join(corpus_root, d, "*.jsonl")))[:shards]],
        }
    return fit, ev, counts


def measure(tok_path, corpus):
    _, m, g = collect(tok_path, corpus, [], [], False)
    keys = ("chars/token", "never used frac", "utilised", "undertrained frac",
            "byte-fragment tokens", "en fertility", "ref fertility", "renyi")
    return {k: round(m[k], 6) for k in keys if k in m}, g


ROOT_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT, help="repo root (defaults to this script's repo)")
    ap.add_argument("--corpus_dirs", default="data/corpus/code_ultra_l2,data/corpus/code_ultra_l3")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--fit_chars", type=int, default=80_000_000, help="per level")
    ap.add_argument("--eval_chars", type=int, default=40_000_000,
                    help="per held-out segment, per level (disjoint across seeds)")
    ap.add_argument("--target_v", type=int, default=TARGET_V)
    ap.add_argument("--candidate_out", default="/tmp/ultra_v20000.json")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    dirs = [d.strip() for d in a.corpus_dirs.split(",") if d.strip()]
    frozen = os.path.join(a.root, "data", "tokenizer.json")
    proxy = os.path.join(a.root, "data", "vocab_sweep", "p1_v20000.json")
    fit, ev_parts, counts = draw(a.root, dirs, a.shards, a.fit_chars, a.eval_chars,
                                 len(SEEDS), 7)
    fit_rows = [r for d in dirs for r in fit[d]]
    random.Random(7).shuffle(fit_rows)
    print(f"fitting V={a.target_v} on {len(fit_rows)} docs "
          f"({sum(c['fit_bytes'] for c in counts.values()) / 1e6:.0f}M bytes)...", flush=True)
    tok = fit_vocab(a.target_v, fit_rows)
    tok.save(a.candidate_out)
    print(f"wrote {a.candidate_out}", flush=True)

    toks = {"frozen_32k": frozen, "ultra_v20000": a.candidate_out}
    if os.path.exists(proxy):
        toks["proxy_p1_v20000"] = proxy

    result = {"config": {"dirs": dirs, "shards": a.shards, "seeds": list(SEEDS),
                         "target_v": a.target_v, "counts": counts}, "draws": {}}
    for si, seed in enumerate(SEEDS):
        corpus = {d: ev_parts[d][si] for d in dirs}
        result["draws"][str(seed)] = {}
        # per level and pooled: collect() aggregates every key in the corpus dict, so
        # per-level numbers need their own call
        subsets = {**{d: {d: corpus[d]} for d in dirs}, "pooled": corpus}
        for label, sub in subsets.items():
            result["draws"][str(seed)][label] = {}
            for name, path in toks.items():
                m, g = measure(path, sub)
                result["draws"][str(seed)][label][name] = {"metrics": m, "gates": g}

    # freeze tax convention shared with build_p1_tokenizer: frozen/candidate - 1,
    # positive = frozen spends more tokens per byte
    print(f"\n{'seed':>4} {'subset':<16}{'vocab':<16}{'chars/tok':>10}{'never_used':>12}"
          f"{'utilised':>10}{'tax':>8}")
    for seed in SEEDS:
        for label in [*dirs, "pooled"]:
            fr = result["draws"][str(seed)][label]["frozen_32k"]["metrics"]["chars/token"]
            for name in toks:
                m = result["draws"][str(seed)][label][name]["metrics"]
                tax = f"{(fr / m['chars/token'] - 1) * 100:+.1f}%"
                print(f"{seed:>4} {label:<16}{name:<16}{m['chars/token']:>10.4f}"
                      f"{m['never used frac']:>12.4f}{m['utilised']:>10.4f}{tax:>8}")
    g0 = result["draws"]["7"]["pooled"]["ultra_v20000"]["gates"]
    print("\ncandidate gates (seed-7 draw, corpus-independent):",
          json.dumps(g0), "ref fertility recorded: frozen 1.4286, proxy 1.3117")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
