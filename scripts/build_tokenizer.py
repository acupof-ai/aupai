#!/usr/bin/env python3
"""Build data/tokenizer.json: the fixed tokenizer-training pipeline.

train.py's build_tokenizer rebuild path registers only <unk>/<eos> and silently drops the 4
chat/think specials the model relies on; this script is the single source of truth instead.
Stratified equal per-domain byte budget, so math/code symbols earn merges instead of drowning in web.

    python scripts/build_tokenizer.py [--force] [--sample-tokens N] [--mix data/mix_v3.json]
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402  (TOK_PATH, Cfg.vocab, DATA)

# [NUM] is last (32772) and always present: without --fone it never appears in the data, so the
# vocab is one id wider either way and no checkpoint needs resizing to switch.
CHAT_SPECIALS = ["<|im_start|>", "<|im_end|>", "<|think|>", "<|/think|>", "[NUM]"]
BYTES_PER_TOKEN_EST = 4  # only to turn --sample-tokens into a per-domain byte budget
# Sample size barely moves the result and the whole corpus costs 45+ minutes: 25K docs vs 395K
# (16x data, 6x time) moved chars/token 2.6093 -> 2.6296 and hanzi coverage 99.51% -> 99.62%.
# The binding constraint is the 32K vocab budget, not the sample. Raise when retuning vocab size.
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
    ap.add_argument(
        "--sample-tokens",
        type=int,
        default=DEFAULT_SAMPLE_TOKENS,
        help=f"cap the training sample (tokens; default {DEFAULT_SAMPLE_TOKENS:,}, 0 = whole corpus)",
    )
    ap.add_argument("--mix", default=os.path.join(train.DATA, os.path.basename(train.Cfg.mix)))
    a = ap.parse_args()

    if os.path.exists(train.TOK_PATH) and not a.force:
        print(f"{train.TOK_PATH} exists; pass --force to retrain", file=sys.stderr)
        return 1

    corpus = os.path.join(train.DATA, "corpus")
    mix_names = []
    if os.path.exists(a.mix):
        with open(a.mix, encoding="utf-8") as f:
            mix_names = list(json.load(f)["domains"])
    # ONLY the domains the mix names. Enumerating data/corpus/* instead silently pulls in the
    # UNFILTERED `web` (kept on disk to re-threshold later) and wastes the whole sample on it.
    domains = [d for d in mix_names if os.path.isdir(os.path.join(corpus, d))]
    if not domains:
        print("no data/corpus/<domain>/ named by the mix recipe (update the mix domains)", file=sys.stderr)
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
    # Cfg.vocab (32773) is the FINAL total, so the trainer targets 32768 to land there after
    # the specials are added on top.
    base_vocab = train.Cfg.vocab - len(CHAT_SPECIALS)
    # initial_alphabet seeds all 256 ByteLevel chars. Without it only the bytes the corpus
    # contains survive (measured 193/256) and a NUL is silently dropped: "café\x00binary"
    # decodes back as "cafébinary", with no raise and not even an <unk>.
    trainer = BpeTrainer(
        vocab_size=base_vocab,
        special_tokens=["<unk>", "<eos>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    # the specials train.py's rebuild path drops
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
