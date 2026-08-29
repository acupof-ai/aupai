#!/usr/bin/env python3
"""Train candidate vocabularies that differ in one decision each, for ranking.

The decisions worth sweeping, and why each is in doubt:

  size          CLAUDE.md fits a scaling law putting the optimum near 12-20K for
                this 166M non-embedding model, and 32K was kept because a 64K
                sweep bought +2.8% compression for +14% compute per character.
                That reasoning was entirely about compression; bits/char can now
                check it.
  digits        `scripts/tokenizer_report.py` measures all six test numbers
                tokenising differently by context and only 3 of 6 place-value
                aligned. `Digits(individual_digits=True)` fixes both by rule.
                Cost unknown until measured.
  domain mix    the current vocabulary was trained on an equal-byte sample per
                domain. Whether that is the right split is untested.

Every variant trains on the SAME stratified sample so only the named decision
differs, and every one gets the four ChatML specials plus [NUM] at fixed ids so
the loader's fingerprint check keeps working.

    python scripts/train_vocab_variants.py --out data/vocab_sweep
"""

import argparse
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|think|>", "<|/think|>", "[NUM]"]
DOMAINS = ["web_hq", "textbook", "wiki", "math", "chat", "code", "en"]


def sample_texts(per_domain_bytes, seed=13):
    """Equal byte budget per domain -- the stratification CLAUDE.md records as the
    fix for web drowning everything else (1.04 chars/token before, 1.484 after)."""
    rng = random.Random(seed)
    out = []
    for d in DOMAINS:
        fs = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", d, "*.jsonl")))
        if not fs:
            continue
        got = 0
        rng.shuffle(fs)
        for f in fs:
            with open(f, encoding="utf-8") as fh:
                lines = fh.readlines()
            rng.shuffle(lines)
            for line in lines:
                if got >= per_domain_bytes:
                    break
                t = json.loads(line).get("content", "")
                if t:
                    out.append(t)
                    got += len(t.encode())
            if got >= per_domain_bytes:
                break
        print(f"  {d:<10}{got / 1e6:>7.1f} MB", flush=True)
    rng.shuffle(out)
    return out


def train_one(texts, size, split_digits, out_path):
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token="<unk>"))
    pre = ByteLevel(add_prefix_space=False)
    # Digits BEFORE ByteLevel: the split has to happen on the text, not on the
    # byte-level re-encoding of it.
    tok.pre_tokenizer = Sequence([Digits(individual_digits=True), pre]) if split_digits else pre
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=size - len(SPECIALS),
        special_tokens=["<unk>", "<eos>"],
        initial_alphabet=ByteLevel.alphabet(),  # without this only the bytes the
        show_progress=False,  # corpus happens to contain survive (193/256 measured)
    )
    tok.train_from_iterator(texts, trainer)
    tok.add_tokens(SPECIALS)
    tok.save(out_path)
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "vocab_sweep"))
    ap.add_argument("--mb", type=int, default=40, help="MB sampled per domain")
    ap.add_argument(
        "--variants",
        default="16384:0,32768:0,32768:1,49152:0,49152:1,65536:1",
        help="comma-separated size:split_digits",
    )
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    print(f"sampling {a.mb} MB per domain")
    texts = sample_texts(a.mb * 1_000_000)
    print(f"{len(texts)} documents, {sum(len(t.encode()) for t in texts) / 1e6:.0f} MB total\n")

    for spec in a.variants.split(","):
        size, split = spec.split(":")
        size, split = int(size), bool(int(split))
        name = f"v{size}{'_digits' if split else ''}.json"
        path = os.path.join(a.out, name)
        if os.path.exists(path):
            print(f"{name}: exists, skipping")
            continue
        print(f"training {name} ...", flush=True)
        tok = train_one(texts, size, split, path)
        probe = tok.encode(" 63 + 122 = 185", add_special_tokens=False).tokens
        print(f"  saved {path}  vocab {tok.get_vocab_size()}  ' 63 + 122 = 185' -> {probe}", flush=True)


if __name__ == "__main__":
    main()
