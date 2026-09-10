#!/usr/bin/env python3
"""Fit the p1 vocabulary on the FILTERED corpus (4c ruling 2026-09-09: rebuild, V=20,000).

# restartable: single deterministic BPE training run -- the output is one vocab file,
# so an interrupt just re-runs the fit (no accumulated partial state to lose; the fit
# sample is deterministic given --sample_tokens and the sorted shard order).

Unfreeze condition 2 is satisfied: the frozen vocab spends 64.9% of its slots on
hanzi tokens (facts/tokenizer.json#tok.minicpm5_slot_budget_vs_ours) and reads
never_used 0.69-0.71 on the p1 composition -- ~11.5K slots serving English+code
against a 20K candidate's 20K. The measured freeze tax is +3.4% tokens/byte on the
88:12 mix (lower bound; the candidate was proxy-fitted, this script fits the real
composition).

Fit corpus = the classifier keep set (three code domains after e1's threshold) +
synthetic textbooks, at the recipe ratio. NOT the raw pool: the classifier keeps
~17% (phi-1 rate), so fitting on the raw pool would feed the vocab stats from
70-80% of documents we discard.

Build rules match build_tokenizer.py / build_p1_vocab_sweep.py: initial_alphabet
seeds all 256 bytes, same specials. Writes to data/tokenizer_p1.json -- NEVER
data/tokenizer.json, whose ids every live checkpoint inherits. The swap is a
separate decision.

    python scripts/build_p1_tokenizer.py \
        --code_dirs code_rp1t_dd09_kept,code_rp1t_b2v2_dd_kept,code_dedup08_kept \
        --textbooks data/p1/textbooks_pilot.jsonl
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_tokenizer import BYTES_PER_TOKEN_EST, CHAT_SPECIALS, domain_texts  # noqa: E402
from tokenizer_eval import GATES, collect  # noqa: E402

CORPUS = os.path.join(ROOT, "data", "corpus")
TOK_FROZEN = os.path.join(ROOT, "data", "tokenizer.json")
OUT = os.path.join(ROOT, "data", "tokenizer_p1.json")

DEFAULT_CODE_SHARES = "0.332,0.191,0.477"  # token share of the three code domains (recipe)
DEFAULT_CODE_FRAC = 0.88                   # code:textbooks = 88:12 by tokens (recipe gate corpus)


def textbook_texts(path, max_bytes, skip_bytes=0):
    """content strings from the textbook jsonl, capped at max_bytes (front-loaded).

    skip_bytes: skip chapters until at least this many bytes are skipped. A held-out
    read with skip_bytes = the fit read's budget starts at the chapter AFTER the
    fit's front-loaded prefix (same file, same byte counts, same stopping rule), so
    fit and held-out are disjoint by construction."""
    out, nbytes, skipped = [], 0, 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line).get("text", "")
        if not c:
            continue
        if skipped < skip_bytes:
            skipped += len(c.encode("utf-8"))
            continue
        out.append(c)
        nbytes += len(c.encode("utf-8"))
        if nbytes >= max_bytes:
            break
    return out


def fit_vocab(target, texts):
    """Same trainer shape as build_tokenizer.py / build_p1_vocab_sweep.py."""
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
    return tok


def _sample_random(domain, want_bytes, rng):
    """Random sample across shards, disjoint from domain_texts' front-loaded fit read
    (the corpus is ~100x the fit budget, so a random draw lands outside it)."""
    import glob
    fs = sorted(glob.glob(os.path.join(CORPUS, domain, "*.jsonl")))
    if not fs:
        sys.exit(f"no shards for {domain}")
    rows, got = [], 0
    while got < want_bytes:
        for f in rng.sample(fs, min(8, len(fs))):
            with open(f, encoding="utf-8") as fh:
                lines = fh.readlines()
            for x in rng.sample(lines, min(len(lines), 4000)):
                try:
                    c = json.loads(x).get("content", "")
                except json.JSONDecodeError:
                    continue
                if c:
                    rows.append(c)
                    got += len(c.encode("utf-8"))
                    if got >= want_bytes:
                        return rows
    return rows


def gate_failures(label, m, g):
    """The p1 gate set on one held-out subset: round-trip and 256 bytes are vetoes,
    ref fertility the regression guard (threshold from tokenizer_eval.GATES). Hanzi
    is undefined on this composition (English textbooks + code contain no hanzi)."""
    fails = []
    if not g["round-trip lossless"]:
        fails.append(f"{label}: round-trip lossless FAIL")
    if g["_bytes"] != 256:
        fails.append(f"{label}: all-256-bytes FAIL ({g['_bytes']}/256)")
    thr = GATES["ref fertility"][0]
    if m["ref fertility"] > thr:
        fails.append(f"{label}: ref fertility {m['ref fertility']:.4f} > {thr}")
    return fails


def held_out(code_dirs, code_shares, textbooks_path, fit_tb_budget, seed=17):
    """Code half: random sample across shards. Textbook half: chapters after the fit
    prefix (skip_bytes = the fit read's budget), disjoint by construction."""
    rng = random.Random(seed)
    rows = []
    for d, share in zip(code_dirs, code_shares, strict=True):
        rows += _sample_random(d, int(2_000_000 * share), rng)  # ~2M chars code held-out
    tb = textbook_texts(textbooks_path, 300_000, skip_bytes=fit_tb_budget)
    rng.shuffle(rows), rng.shuffle(tb)
    return rows, tb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dirs", required=True,
                    help="comma-separated filtered code domains (classifier keep set), under data/corpus/")
    ap.add_argument("--code_shares", default=DEFAULT_CODE_SHARES)
    ap.add_argument("--textbooks", required=True, help="synthetic textbook jsonl")
    ap.add_argument("--code_frac", type=float, default=DEFAULT_CODE_FRAC)
    ap.add_argument("--target_v", type=int, default=20000)
    ap.add_argument("--sample_tokens", type=int, default=62_500_000,
                    help="fit-sample cap in tokens (default: build_p1_vocab_sweep's)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; remove it by hand to refit (never overwrite a vocab in use)")

    code_dirs = [d.strip() for d in a.code_dirs.split(",") if d.strip()]
    code_shares = [float(s) for s in a.code_shares.split(",")]
    assert len(code_dirs) == len(code_shares), "code_dirs and code_shares must match"

    budget = a.sample_tokens * BYTES_PER_TOKEN_EST
    code_budget = budget * a.code_frac
    tb_budget = budget * (1 - a.code_frac)

    texts = []
    for d, share in zip(code_dirs, code_shares, strict=True):
        docs = domain_texts(CORPUS, d, code_budget * share)
        print(f"  fit {d}: {len(docs)} docs, {sum(len(x.encode()) for x in docs)/1e6:.1f}M bytes", flush=True)
        texts += docs
    tb = textbook_texts(a.textbooks, tb_budget)
    print(f"  fit textbooks: {len(tb)} chapters, {sum(len(x.encode()) for x in tb)/1e6:.1f}M bytes", flush=True)
    texts += tb

    print(f"fitting V={a.target_v} on {len(texts)} docs...", flush=True)
    tok = fit_vocab(a.target_v, texts)

    # gates run on the SAVED file, so write a tmp name first; the final path appears
    # only if every gate passes -- a vocab that drops a byte must not land at
    # data/tokenizer_p1.json and then block every refit as "exists"
    tmp = a.out + ".tmp"
    tok.save(tmp)

    # gates + the tax, now a READING on the real composition (was an inference at proxy-fit)
    code_ho, tb_ho = held_out(code_dirs, code_shares, a.textbooks, tb_budget)
    subsets = {"code_held_out": code_ho}
    if tb_ho:
        subsets["textbooks_held_out"] = tb_ho
        subsets["mix_held_out"] = code_ho + tb_ho
    else:
        print(
            "NOTE: textbook held-out is empty -- the fit prefix exhausts the "
            "textbook file; the textbook tax is not measurable on this input",
            flush=True,
        )

    fails = []
    try:
        for label, rows in subsets.items():
            _, m_new, g_new = collect(tmp, {label: rows}, None, None, False)
            _, m_frz, _ = collect(TOK_FROZEN, {label: rows}, None, None, False)
            fails += gate_failures(label, m_new, g_new)
            # tax convention, shared with tokenizer_p1_real.py: candidate/frozen - 1,
            # positive = frozen uses more tokens per byte
            tax = m_new["chars/token"] / m_frz["chars/token"] - 1
            print(
                json.dumps(
                    {
                        "subset": label,
                        "new_chars/token": round(m_new["chars/token"], 4),
                        "frozen_chars/token": round(m_frz["chars/token"], 4),
                        "freeze_tax": f"{100 * tax:+.1f}%",
                        "new_roundtrip": g_new["round-trip lossless"],
                        "new_bytes": f"{g_new['_bytes']}/256",
                        "new_ref_fertility": round(m_new["ref fertility"], 4),
                        "hanzi": "undefined (no hanzi in p1 composition)",
                    }
                ),
                flush=True,
            )
    finally:
        if fails:
            os.remove(tmp)
            sys.exit("REFUSE: gate failures, vocab not written:\n  " + "\n  ".join(fails))

    os.replace(tmp, a.out)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
