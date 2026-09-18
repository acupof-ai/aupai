"""Vocabulary fingerprint for v41f checkpoints (torch-free).

A checkpoint records a `vocab_id`; a loader refuses to build it with a different tokenizer.
The hash is the SAME convention as `scripts/loader.py:vocab_fingerprint` so a v41f training
blob and the older checkpoint/packs agree on what a vocabulary is:

    sha256 over the tokens in ascending ID order, token bytes concatenated with no
    separator, first 16 hex chars.

Ordering by ID (not by token text) is the identity: two maps that contain the same pieces in
a different order are different tokenizers. v41f must not import scripts/, so the hash lives
here and the cross-check that it equals scripts.loader.vocab_fingerprint lives in the test.

Two adapters are provided because v41f takes the tokenizer by injection and never assumes an
HF surface: `id_to_token_map` normalises either an HF-style tokenizer (`.get_vocab()` returns
a {piece: id} dict) or a backend exposing `.id_to_token(i)`/`.get_vocab()`/`.pieces`.
"""

import hashlib


def id_to_token_map(tok):
    """Return a normalised {id(int): token(str)} from an injected tokenizer.

    Accepted surfaces, in order:
      - tok.get_vocab() -> {piece: id} (HF tokenizers, and the scripts/loader convention);
      - tok.backend_tokenizer with .get_vocab() (HF backend);
      - a backend with .id_to_token(i) and len(tok) (the disk-free SyntheticTokenizer).
    Raises on a tokenizer it cannot read rather than guessing.
    """
    get_vocab = getattr(tok, "get_vocab", None)
    if callable(get_vocab):
        piece_to_id = get_vocab()
        if piece_to_id:
            return {int(i): t for t, i in piece_to_id.items()}
    backend = getattr(tok, "backend_tokenizer", None)
    if backend is not None:
        bgv = getattr(backend, "get_vocab", None)
        if callable(bgv) and bgv():
            return {int(i): t for t, i in bgv().items()}
        # An ordered pieces list (id -> token) is the most direct identity; prefer it over an
        # id_to_token accessor that may return a rendered form rather than the stored piece
        # (the disk-free SyntheticTokenizer renders "<raw i>" while pieces holds the bytes).
        pieces = getattr(backend, "pieces", None)
        if pieces is not None:
            return {i: t for i, t in enumerate(pieces)}
        id_to_token = getattr(backend, "id_to_token", None)
        if callable(id_to_token):
            n = getattr(tok, "__len__", None)
            size = n() if callable(n) else len(backend)
            return {i: id_to_token(i) for i in range(size)}
    raise TypeError("vocab fingerprint: tokenizer exposes no get_vocab()/id_to_token surface")


def fingerprint(tok):
    """sha256 over tokens in ascending id order, [:16]; byte-identical to loader's."""
    id_to_tok = id_to_token_map(tok)
    h = hashlib.sha256()
    for _id in sorted(id_to_tok):
        h.update(id_to_tok[_id].encode())
    return h.hexdigest()[:16]
