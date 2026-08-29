#!/usr/bin/env python3
"""Data distribution at a glance: per-domain token counts vs the mix target.

Counts come from the pretokenized caches (/data00/tokens_<domain>.pt, exact and
instant) when present; otherwise data/corpus/<domain>/*.jsonl is tokenized with
the project tokenizer and cached in <corpus>/.counts.json, so the first run
counts and every later run is free.

    python scripts/data_overview.py [--mix data/mix_v3.json] [--corpus data/corpus]
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness  # single source of truth for the configured mix

import train  # noqa: E402

TOKEN_CACHE = train.TOKEN_CACHE
TOK_PATH = train.TOK_PATH


def cache_tokens(domain):
    """Token count from the pretokenized cache: int32, 4 bytes each, plus a small header."""
    p = os.path.join(os.path.dirname(TOKEN_CACHE), f"tokens_{domain}.pt")
    return os.path.getsize(p) // 4 if os.path.exists(p) else None


def corpus_tokens(domain, corpus_dir, tok, sidecar):
    """Count a corpus/<domain> directory via the tokenizer, caching per file (size, mtime)."""
    d = os.path.join(corpus_dir, domain)
    if not os.path.isdir(d):
        return None
    total = 0
    dirty = False
    for f in sorted(os.listdir(d)):
        if not f.endswith(".jsonl"):
            continue
        p = os.path.join(d, f)
        st = os.stat(p)
        key = f"{domain}/{f}"
        rec = sidecar.get(key)
        if rec and rec[0] == st.st_size and rec[1] == st.st_mtime:
            total += rec[2]
            continue
        n = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (r.get("title") or "") + "\n" + (r.get("content") or r.get("text") or "")
                n += len(tok.encode(text).ids)
        sidecar[key] = [st.st_size, st.st_mtime, n]
        total += n
        dirty = True
    if dirty:
        print(f"  counted {domain}: {total / 1e6:.1f}M tokens", file=sys.stderr)
    return total


def fmt_tokens(n):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.2f}{suf}"
    return str(n)


def main():
    ap = argparse.ArgumentParser()
    # The configured mix, not a second copy of its name: four scripts each hardcoded
    # "data/mix.json" and all four went stale the day it was deleted.
    ap.add_argument("--mix", default=os.path.join(ROOT, harness.cfg_default("mix")))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus"))
    a = ap.parse_args()

    with open(a.mix, encoding="utf-8") as f:
        mix = json.load(f)
    domains = list(mix["domains"])
    extra = sorted(
        d
        for d in os.listdir(a.corpus)
        if os.path.isdir(os.path.join(a.corpus, d)) and d not in mix["domains"]
    )
    if extra:
        print(f"corpus domains not in mix: {extra}", file=sys.stderr)

    sidecar_path = os.path.join(a.corpus, ".counts.json")
    if os.path.exists(sidecar_path):
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)
    else:
        sidecar = {}
    tok = None
    counts = {}
    for d in domains + extra:
        n = cache_tokens(d)
        if n is None:
            if tok is None:
                from tokenizers import Tokenizer

                tok = Tokenizer.from_file(TOK_PATH)
            n = corpus_tokens(d, a.corpus, tok, sidecar)
        counts[d] = n
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f)

    missing = [d for d in domains if counts[d] is None]
    if missing:
        print(f"MISSING data for mix domains: {missing}", file=sys.stderr)
    total = sum(v for v in counts.values() if v is not None)
    if total == 0:
        print("no data found anywhere — build the corpus or mount the token caches", file=sys.stderr)
        return 1

    bar_w = 20
    max_share = max(counts[d] for d in counts if counts[d] is not None) / total
    print(
        f"\naupai data — {fmt_tokens(total)} tokens across {len(counts)} domains "
        f"(mix target {mix['total_tokens'] / 1e9:.1f}B)\n"
    )
    print(
        f"{'domain':<8}{'tokens':>9}{'share':>8}  {'bar':<{bar_w}}  {'mix_w':>6}{'Δ':>8}{'epochs':>7}{'anneal':>8}"
    )
    for d in counts:
        if counts[d] is None:
            print(f"{d:<8}{'MISSING':>9}")
            continue
        share = counts[d] / total
        m = mix["domains"].get(d)
        w = m["weight"] if m else 0.0
        filled = round(share / max_share * bar_w) if share > 0 else 0
        bar = "█" * filled if filled else ("▏" if share > 0 else "")
        delta = share - w
        if m:
            print(
                f"{d:<8}{fmt_tokens(counts[d]):>9}{share:>7.1%}  {bar:<{bar_w}}  {w:>5.0%}{delta:>+7.1%}"
                f"{m['epochs']:>7}{m['anneal']:>7.0%}"
            )
        else:
            print(
                f"{d:<8}{fmt_tokens(counts[d]):>9}{share:>7.1%}  {bar:<{bar_w}}  {'—':>6}{'—':>8}{'—':>7}{'—':>8}"
            )

    ann = {d: c["anneal"] for d, c in mix["domains"].items()}
    print("\nmain phase: " + " ".join(f"{d} {c['weight']:.0%}" for d, c in mix["domains"].items()))
    print("anneal:     " + " ".join(f"{d} {w:.0%}" for d, w in ann.items()))
    print("schedule dry-run (rows/phase, epoch caps, step count): python scripts/check_mix.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
