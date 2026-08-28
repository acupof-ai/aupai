#!/usr/bin/env python3
"""Build data/tokenizer.json: the fixed tokenizer-training pipeline.

train.py's build_tokenizer rebuild path registers only <unk>/<eos>, so a freshly retrained
tokenizer drops the 4 chat/think specials the model relies on (a known bug). This script is the
single source of truth for the rebuild: ByteLevel BPE, vocab 32768, THEN the specials at their fixed
ids -> vocab 32772, matching the existing data/tokenizer.json exactly.

    python scripts/build_tokenizer.py [--force] [--sample-tokens N] [--mix data/mix.json]

Trains on the content field of every data/corpus/<domain>/*.jsonl (stratified: an equal per-domain
byte budget so math/code symbols earn merges instead of drowning in web). Prints a coverage report.
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402  (TOK_PATH, Cfg.vocab, DATA)

# The 4 specials build_tokenizer forgets, plus [NUM]. ids follow the 32768 base vocab.
# [NUM] is last (32772) and always present: --fone gives it a value embedding, and
# without --fone it simply never appears in the data, so the vocab is one id wider
# either way and no checkpoint needs resizing to switch.
CHAT_SPECIALS = ["<|im_start|>", "<|im_end|>", "<|think|>", "<|/think|>", "[NUM]"]
BYTES_PER_TOKEN_EST = 4  # only to turn --sample-tokens into a per-domain byte budget
# Default sample. Feeding the whole corpus took 45+ minutes and bought nothing: the
# BPE merge loop is inherently sequential (~5 of the box's 180 cores), and a sweep on
# this corpus showed sample size barely moves the result. 25K docs vs 395K (16x the
# data, 6x the time): chars/token 2.6093 -> 2.6296, hanzi occurrence-coverage
# 99.51% -> 99.62%, and the 5K-10K frequency tier stays at 0% either way. The binding
# constraint is the 32K vocab budget, not the sample -- the vocab spends itself on
# frequent word pieces and rare hanzi never fit, however much corpus it sees.
# Override with --sample-tokens when retuning vocab size, where the tradeoff may differ.
DEFAULT_SAMPLE_TOKENS = 250_000 * 250  # ~250K docs at the measured ~250 tokens/doc


def domain_texts(corpus, domain, max_bytes):
    """content strings from corpus/<domain>/*.jsonl, capped at max_bytes."""
    out, nbytes = [], 0
    for p in sorted(glob.glob(os.path.join(corpus, domain, "*.jsonl"))):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = r.get("content") or r.get("text")
                if not c:
                    continue
                out.append(c)
                nbytes += len(c.encode("utf-8"))
                if nbytes >= max_bytes:
                    return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing data/tokenizer.json")
    ap.add_argument("--sample-tokens", type=int, default=DEFAULT_SAMPLE_TOKENS,
                    help=f"cap the training sample (tokens; default {DEFAULT_SAMPLE_TOKENS:,}, 0 = whole corpus)")
    ap.add_argument("--mix", default=os.path.join(train.DATA, "mix.json"))
    a = ap.parse_args()

    if os.path.exists(train.TOK_PATH) and not a.force:
        print(f"{train.TOK_PATH} exists; pass --force to retrain", file=sys.stderr)
        return 1

    corpus = os.path.join(train.DATA, "corpus")
    mix_names = []
    if os.path.exists(a.mix):
        with open(a.mix, encoding="utf-8") as f:
            mix_names = list(json.load(f)["domains"])
    domains = [d for d in mix_names if os.path.isdir(os.path.join(corpus, d))]
    for d in sorted(os.listdir(corpus)) if os.path.isdir(corpus) else []:
        if os.path.isdir(os.path.join(corpus, d)) and d not in domains:
            domains.append(d)
    if not domains:
        print("no data/corpus/<domain>/ to train on", file=sys.stderr)
        return 1

    budget = a.sample_tokens * BYTES_PER_TOKEN_EST if a.sample_tokens else float("inf")  # 0 -> whole corpus
    per_domain = budget / len(domains)
    print(f"sampling {len(domains)} domains ({', '.join(domains)})", file=sys.stderr)
    samples = {d: domain_texts(corpus, d, per_domain) for d in domains}
    texts = [t for docs in samples.values() for t in docs]
    print(f"{len(texts)} docs", file=sys.stderr)

    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    # BPE base holds <unk>/<eos> and merges; the specials sit on top. Cfg.vocab (32773) is the
    # FINAL total, so the trainer targets Cfg.vocab - 5 = 32768 to land at 32773 after adding them.
    base_vocab = train.Cfg.vocab - len(CHAT_SPECIALS)
    # Seed all 256 ByteLevel characters. Without this the trainer only keeps the ones
    # the corpus happened to contain -- measured 2026-08-28: 193/256, with 63 missing
    # (0xC0, 0xC1, 0xD2-0xDF, ...). Chinese text rarely hits them, so the gap is quiet
    # rather than loud: it does not raise, and it does not even emit <unk> (BPE falls
    # back to finer pieces), but a NUL byte is silently dropped -- "café\x00binary"
    # decodes back as "cafébinary". It also violates the totality invariant that
    # ByteLevel exists to provide, which is why gigatoken refuses this vocab outright
    # ("no single-byte vocab entry for byte 0x00") and fastokens raises on corpus text.
    trainer = BpeTrainer(
        vocab_size=base_vocab,
        special_tokens=["<unk>", "<eos>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    # THE FIX: register the chat/think specials build_tokenizer drops.
    tok.add_special_tokens(CHAT_SPECIALS)

    vsize = tok.get_vocab_size()
    expected = train.Cfg.vocab
    if vsize != expected:
        print(
            f"WARNING vocab {vsize} != {expected} (corpus too small to fill {base_vocab} merges?)",
            file=sys.stderr,
        )
    tok.save(train.TOK_PATH)
    print(f"saved {train.TOK_PATH} (vocab {vsize})")

    print(f"\n{'domain':<8}{'tokens':>10}{'chars/tok':>12}{'unk_rate':>10}")
    unk_id = tok.token_to_id("<unk>")
    for d, docs in samples.items():
        if not docs:
            continue
        chars = sum(len(t) for t in docs)
        ids = [i for t in docs for i in tok.encode(t).ids]
        unks = sum(1 for i in ids if i == unk_id)
        ntok = len(ids)
        cpt = chars / ntok if ntok else 0.0
        unk = unks / ntok if ntok else 0.0
        print(f"{d:<8}{ntok:>10}{cpt:>12.3f}{unk * 100:>9.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
