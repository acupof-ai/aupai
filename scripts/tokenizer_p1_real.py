#!/usr/bin/env python3
"""Tokenizer gates + fertility on the REAL p1 composition: frozen vocab vs the 20K candidate.

4c 2026-09-09: the condition-2 ruling must rest on the four GATES (AGENTS.md:233),
not on never_used (reported, not gated -- its 0.01 threshold sits inside its own
seed noise). This script reports all four on the real composition and the
decision number: the freeze tax = frozen-32K vs fresh-20K chars/token on the same
samples. The fresh candidate was fitted on the English proxy (build_p1_vocab_sweep.py);
the frozen vocab was fitted on the old bilingual mix and spends 64.9% of its slots
on hanzi-bearing tokens (facts/tokenizer.json#tok.minicpm5_slot_budget_vs_ours).

Gates:
  round-trip lossless  - tokenizer property, same on every corpus
  all 256 bytes        - tokenizer property
  hanzi whole-char     - UNDEFINED on this composition (English textbooks + English
                         code contain no hanzi; the guard protects Chinese, which p1
                         does not train). Frozen's recorded value on the old
                         composition: 0.9913. p1_v20000 has no hanzi tokens at all.
  ref fertility <=1.55 - corpus-independent by construction (REF_EN is a fixed 616-word
                         string, tokenizer_report.py:285). Frozen 1.4286 (recorded),
                         p1_v20000 1.3117 (sweep). Both pass.

The tax number is chars/token per subset, frozen vs candidate: fewer chars/token =
more tokens per byte of content = more compute per training example, forever.

    python3 /work/aupai/runs/tokenizer_p1_real.py
"""
import json
import os
import random
import sys

ROOT = "/work/aupai"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import tokenizer_report as R  # noqa: E402
from tokenizer_eval import collect  # noqa: E402

TOKS = {
    "frozen_32k": os.path.join(ROOT, "data", "tokenizer.json"),
    "p1_v20000": os.path.join(ROOT, "data", "vocab_sweep", "p1_v20000.json"),
}
TEXTBOOKS = "/work/aupai/data/p1/textbooks_pilot.jsonl"
SEEDS = (7, 13, 21)

# token-share ratio of the three code domains (recipe doc, 4c 2026-09-09)
CODE_DOMAINS = ("code_rp1t_dd09", "code_rp1t_b2v2_dd", "code_dedup08")
CODE_SHARES = (0.332, 0.191, 0.477)
CODE_CHARS = 4_000_000  # the code side is cheap to sample; chars/token is a ratio

# recipe gate corpus: ~6B filtered code + 0.8B textbooks -> code:textbooks ~= 88:12
CODE_FRAC = 0.88


def sample_domain(domain, want_chars, rng):
    fs = R.shard_paths(domain)
    if not fs:
        sys.exit("no shards for " + domain)
    rows, got = [], 0
    while got < want_chars:
        for f in rng.sample(fs, min(8, len(fs))):
            with open(f, encoding="utf-8") as fh:
                lines = fh.readlines()
            for x in rng.sample(lines, min(len(lines), 4000)):
                try:
                    c = json.loads(x).get("content", "")
                except Exception:
                    continue
                if c:
                    rows.append(c)
                    got += len(c)
                    if got >= want_chars:
                        return rows
    return rows


def load_textbooks(rng):
    """ALL non-empty lines of the textbook file -- no cap, no sampling -- so the
    chapter count is identical across seeds; the seed varies only the shuffle
    (and the code side). The recorded fact config states this explicitly."""
    rows = []
    for line in open(TEXTBOOKS, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line)["text"])
    rng.shuffle(rows)
    return rows


def measure(tok_path, label, rows, seed):
    tok, m, g = collect(tok_path, {label: rows}, None, None, False)
    out = {"tokenizer": os.path.basename(tok_path), "label": label, "seed": seed,
           "n_rows": len(rows), "chars_M": round(sum(len(r) for r in rows) / 1e6, 2)}
    for k in ("chars/token", "ref fertility", "en fertility", "never used frac",
              "byte-fragment tokens", "utilised"):
        if k in m:
            out[k] = round(m[k], 4)
    out["roundtrip"] = g["round-trip lossless"]
    out["bytes"] = f'{g["_bytes"]}/256'
    return out


def main():
    results = []
    for seed in SEEDS:
        rng = random.Random(seed)
        textbooks = load_textbooks(rng)
        tb_chars = sum(len(t) for t in textbooks)
        print(f"seed {seed}: {len(textbooks)} textbook chapters, {tb_chars/1e6:.2f}M chars", flush=True)

        code = []
        for dom, share in zip(CODE_DOMAINS, CODE_SHARES, strict=True):
            code += sample_domain(dom, CODE_CHARS * share, rng)
        mix = textbooks + sample_domain_mix(code, tb_chars, rng)
        rng.shuffle(mix)

        for path in TOKS.values():
            results.append(measure(path, "textbooks_only", textbooks, seed))
            results.append(measure(path, "code_3domains_4M", code, seed))
            results.append(measure(path, "full_mix_88_12", mix, seed))

        for r in results[-6:]:
            print(json.dumps(r), flush=True)

    print("\n=== gates (tokenizer properties, corpus-independent) ===", flush=True)
    for name, path in TOKS.items():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(path)
        rf = R.ref_fertility(tok)["ref fertility"]
        print(f"  {name:12s} ref_fertility={rf:.4f} (<=1.55: {'PASS' if rf <= 1.55 else 'FAIL'})"
              f"  hanzi=undefined (no hanzi in p1 composition)", flush=True)

    print("\n=== chars/token summary (the freeze tax: frozen vs candidate) ===", flush=True)
    for label in ("textbooks_only", "code_3domains_4M", "full_mix_88_12"):
        for tok_name in TOKS:
            vals = [r["chars/token"] for r in results
                    if r["label"] == label and r["tokenizer"] == os.path.basename(TOKS[tok_name])]
            print(f"  {label:20s} {tok_name:12s} min {min(vals):.4f}  max {max(vals):.4f}", flush=True)
        f = [r["chars/token"] for r in results if r["label"] == label and r["tokenizer"] == "tokenizer.json"]
        c = [r["chars/token"] for r in results if r["label"] == label and r["tokenizer"] == "p1_v20000.json"]
        # tokens/byte = 1/(chars/token), so the frozen-vs-candidate token-per-byte
        # ratio is candidate/frozen on chars/token. Positive = freeze costs more.
        tax = (sum(c) / len(c)) / (sum(f) / len(f)) - 1
        print(f"  {'':20s} {'TAX':12s} frozen uses {100*tax:+.1f}% more tokens per byte", flush=True)

    print("\n=== never_used (reported, NOT gated) ===", flush=True)
    for label in ("textbooks_only", "code_3domains_4M", "full_mix_88_12"):
        for tok_name in TOKS:
            vals = [r["never used frac"] for r in results
                    if r["label"] == label and r["tokenizer"] == os.path.basename(TOKS[tok_name])]
            print(f"  {label:20s} {tok_name:12s} min {min(vals):.4f}  max {max(vals):.4f}", flush=True)


def sample_domain_mix(code, tb_chars, rng):
    """Code at the recipe ratio, resampled to the mix size (code is 88%, textbooks 12%)."""
    want = tb_chars * CODE_FRAC / (1 - CODE_FRAC)
    rng.shuffle(code)
    out, got = [], 0
    for c in code:
        out.append(c)
        got += len(c)
        if got >= want:
            break
    if got < want:
        print(
            f"WARNING: code pool {got / 1e6:.2f}M chars < wanted {want / 1e6:.2f}M; "
            f"the mix is not {CODE_FRAC:.0%}:{1 - CODE_FRAC:.0%}",
            flush=True,
        )
    return out


if __name__ == "__main__":
    main()
