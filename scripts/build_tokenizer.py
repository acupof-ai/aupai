#!/usr/bin/env python3
"""Build data/tokenizer.json: the fixed tokenizer-training pipeline.

Use this, not train.py's rebuild path: it registers only <unk>/<eos> and silently
drops the chat/think specials the model relies on.
Stratified equal per-domain byte budget, so math/code symbols earn merges instead
of drowning in web.

    python scripts/build_tokenizer.py [--force] [--sample-tokens N] [--mix data/mix_scale_3.24b.json]
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402

# [NUM] is always present (last id): unused without --fone, so no checkpoint needs
# resizing to switch.
CHAT_SPECIALS = ["<|im_start|>", "<|im_end|>", "<|think|>", "<|/think|>", "[NUM]"]
BYTES_PER_TOKEN_EST = 4
# Sample size barely moves the result (25K vs 395K docs moved chars/token 2.6093 -> 2.6296);
# the binding constraint is the 32K vocab budget, not the sample. Raise when retuning vocab size.
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
    ap.add_argument("--weights", default="", help="per-domain sample weight, e.g. en=3,code=2")
    ap.add_argument("--out", default="", help="write here instead of data/tokenizer.json")
    ap.add_argument("--mix", default=os.path.join(train.DATA, os.path.basename(train.Cfg.mix)))
    a = ap.parse_args()

    # --out keeps a candidate out of data/tokenizer.json: ids do not survive a rebuild,
    # so writing there would invalidate every live checkpoint.
    out_path = a.out or train.TOK_PATH
    if os.path.exists(out_path) and not a.force:
        print(f"{out_path} exists; pass --force to retrain", file=sys.stderr)
        return 1

    corpus = os.path.join(train.DATA, "corpus")
    mix_names = []
    if os.path.exists(a.mix):
        with open(a.mix, encoding="utf-8") as f:
            mix_names = list(json.load(f)["domains"])
    # Only the domains the mix names: enumerating data/corpus/* would silently pull in
    # the unfiltered `web` kept on disk.
    domains = [d for d in mix_names if os.path.isdir(os.path.join(corpus, d))]
    if not domains:
        print("no data/corpus/<domain>/ named by the mix recipe (update the mix domains)", file=sys.stderr)
        return 1

    budget = a.sample_tokens * BYTES_PER_TOKEN_EST if a.sample_tokens else float("inf")
    # Sample balance is NOT the corpus mix: it decides what earns merges, so weight it by
    # what the vocabulary must be good at (English runs hot at fertility 1.87 under equal-byte).
    w = dict(kv.split("=") for kv in a.weights.split(",")) if a.weights else {}
    wt = {d: float(w.get(d, 1.0)) for d in domains}
    tot_w = sum(wt.values())
    print(
        f"sampling {len(domains)} domains, byte shares "
        + ", ".join(f"{d} {wt[d] / tot_w:.0%}" for d in domains),
        file=sys.stderr,
    )
    samples = {d: domain_texts(corpus, d, budget * wt[d] / tot_w) for d in domains}
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
    # vocab_real is the TOKENIZER's size; Cfg.vocab (32784) is that padded up to a
    # multiple of 16 for the aligned cuBLAS head kernel. Padding is embedding width, not
    # tokens -- targeting Cfg.vocab would train 11 extra merges and produce a 32784-token
    # vocabulary that load_tokenizer's `vocab == vocab_real` assert then rejects on every
    # existing checkpoint. The trainer targets vocab_real - len(specials).
    base_vocab = train.Cfg.vocab_real - len(CHAT_SPECIALS)
    # initial_alphabet seeds all 256 bytes. Without it, bytes absent from the corpus vanish
    # silently (NUL drops with no raise, not even an <unk>).
    trainer = BpeTrainer(
        vocab_size=base_vocab,
        special_tokens=["<unk>", "<eos>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    tok.add_special_tokens(CHAT_SPECIALS)

    vsize = tok.get_vocab_size()
    expected = train.Cfg.vocab_real
    if vsize != expected:
        # HARD FAILURE, not a warning. A short vocabulary moves [NUM] down from its
        # expected id, and train.py reads num_id at three sites -- fone masking, digit
        # cross-entropy, value write-back -- each of which then addresses an ordinary BPE
        # token with correct shapes and no error. Training proceeds and looks fine. A
        # warning printed to stderr during a build nobody watches is not a control for
        # that; the danger is the silence, not the likelihood.
        raise SystemExit(
            f"REFUSE: built vocab {vsize} != Cfg.vocab_real {expected} "
            f"(corpus too small to fill {base_vocab} merges?). Saving it would move [NUM] "
            f"off id {train.Cfg.num_id} and silently mis-address every --fone read site."
        )
    if tok.token_to_id("[NUM]") is None:
        raise SystemExit("REFUSE: [NUM] missing from the built vocabulary -- --fone cannot run")
    tok.save(out_path)
    print(f"saved {out_path} (vocab {vsize})")

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
