#!/usr/bin/env python3
"""p1 vocab-size sweep: bits/char vs V on the English proxy composition.

Freeze condition 1 for the V=20,000 working target (4c, 2026-09-09): the
20K->32K marginal must be measured on the p1 composition, not inferred from the
old-mix 32K->64K sweep (+2.8% compression for +33.6M params,
facts/tokenizer.json#tok.vocab_sweep_32k_64k) -- code identifier long tails
could make 20K much more expensive than 32K, and that is exactly the segment
nobody measured.

Candidates are fitted on the English proxy (en_c4_stage2 + code_py_starcoder/
code_py_rp1t at a 3:0.95:0.05 byte budget -- the p1 composition's prose:code
ratio with the code half split 95/5 like tok.gates_p1_proxy_composition) at
16K/20K/24K/32K, then scored on ONE held-out proxy set, identical for every
candidate: only the segmentation differs. The frozen vocabulary (fitted on the
old bilingual mix) is scored on the same held-out set as a reference row, to
split the fit effect from the size effect.

Build rules follow AGENTS.md and build_tokenizer.py: initial_alphabet seeds all
256 bytes, same specials as the production vocabulary. Candidates are written
under data/vocab_sweep/ -- never to data/tokenizer.json, whose ids every live
checkpoint inherits.

    python scripts/build_p1_vocab_sweep.py
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_tokenizer import CHAT_SPECIALS, domain_texts  # noqa: E402
from tokenizer_eval import collect  # noqa: E402
from tokenizer_report import shard_paths  # noqa: E402
from tokenizer_sweep import bits_per_char  # noqa: E402

CORPUS = os.path.join(ROOT, "data", "corpus")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
OUTDIR = os.path.join(ROOT, "data", "vocab_sweep")

# prose:code = 3:1 by bytes; the code half splits 95/5 like the gate measurement.
WEIGHTS = {"en_c4_stage2": 3.0, "code_py_starcoder": 0.95, "code_py_rp1t": 0.05}
SIZES = (16384, 20000, 24576, 32768)
N_TRAIN_ROWS, N_EVAL_ROWS = 3000, 800
SEED = 11


def build_candidate(target, texts, outdir):
    """Same trainer shape as build_tokenizer.py, parameterized by target size."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=target - len(CHAT_SPECIALS),
        special_tokens=["<unk>", "<eos>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    tok.add_special_tokens(CHAT_SPECIALS)
    got = tok.get_vocab_size()
    if got != target:
        sys.exit(f"REFUSE: built vocab {got} != target {target} (corpus too small?)")
    path = os.path.join(outdir, f"p1_v{target}.json")
    tok.save(path)
    return path


def held_out_rows():
    """Proxy-composition rows held out from training: last-quartile shards
    (training reads the front of the sorted shard list), 3:1 prose:code by count."""
    rng = random.Random(SEED)

    def draw(domain, n):
        fs = shard_paths(domain)
        tail = fs[-(len(fs) // 4):]
        rows = []
        for f in rng.sample(tail, min(2, len(tail))):
            with open(f, encoding="utf-8") as fh:
                lines = fh.readlines()
            for x in rng.sample(lines, min(len(lines), n)):
                try:
                    c = json.loads(x).get("content", "")
                except Exception:
                    continue
                if c:
                    rows.append(c)
                    if len(rows) >= n:
                        return rows
        return rows

    def split(n):
        return int(n * 0.75), int(n * 0.95 * 0.25), int(n * 0.05 * 0.25)

    out = {}
    for phase, n in (("train", N_TRAIN_ROWS), ("eval", N_EVAL_ROWS)):
        np_, ns, nr = split(n)
        rows = draw("en_c4_stage2", np_) + draw("code_py_starcoder", ns) + draw("code_py_rp1t", nr)
        rng.shuffle(rows)
        out[phase] = rows
    return out["train"], out["eval"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-tokens", type=int, default=62_500_000,
                    help="training sample cap in tokens (default: build_tokenizer's)")
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    budget = a.sample_tokens * 4  # BYTES_PER_TOKEN_EST in build_tokenizer
    tot_w = sum(WEIGHTS.values())
    texts = []
    for d, w in WEIGHTS.items():
        docs = domain_texts(CORPUS, d, budget * w / tot_w)
        print(f"  train {d}: {len(docs)} docs", flush=True)
        texts += docs

    train_rows, eval_rows = held_out_rows()
    print(f"held-out: {len(train_rows)} train rows, {len(eval_rows)} eval rows", flush=True)

    results = []
    for target in SIZES:
        path = build_candidate(target, texts, a.outdir)
        tok, m, g = collect(path, {"p1_proxy": eval_rows}, None, None, False)
        bc = bits_per_char(tok, train_rows, eval_rows)
        row = {"vocab": target, "bits/char": round(bc["bits/char"], 4),
               "bits/token": round(bc["bits/token"], 4),
               "chars/token": round(bc["chars/token"], 4),
               "never_used_frac": m["never used frac"], "ref_fertility": m["ref fertility"],
               "roundtrip": g["round-trip lossless"], "bytes": g["_bytes"]}
        results.append(row)
        print(json.dumps(row), flush=True)

    # reference: the frozen vocabulary, fitted on the OLD mix, same held-out set
    from tokenizers import Tokenizer
    ftok = Tokenizer.from_file(TOK_PATH)
    fbc = bits_per_char(ftok, train_rows, eval_rows)
    print(json.dumps({"vocab": f"frozen_oldmix_{ftok.get_vocab_size()}",
                      "bits/char": round(fbc["bits/char"], 4),
                      "bits/token": round(fbc["bits/token"], 4),
                      "chars/token": round(fbc["chars/token"], 4)}), flush=True)

    with open(os.path.join(a.outdir, "p1_sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump({"weights": WEIGHTS, "sample_tokens": a.sample_tokens,
                   "held_out": {"train_rows": len(train_rows), "eval_rows": len(eval_rows), "seed": SEED},
                   "results": results}, f, indent=1)

    # the pipeline check: more merges must segment into strictly longer tokens
    ct = [r["chars/token"] for r in results]
    assert all(a < b for a, b in zip(ct, ct[1:])), f"chars/token not monotone in V: {ct}"
    assert all(r["roundtrip"] and r["bytes"] == 256 for r in results), "a candidate failed a gate"
    print("sweep OK: chars/token monotone in V, all candidates pass round-trip + 256 bytes", flush=True)


if __name__ == "__main__":
    main()
