#!/usr/bin/env python3
"""domain_loss must score val, not the training pool (de, 2026-09-01).

tilerl measured the defect: the old path took the head of each domain's
alphabetically-first shard, and 0.625% of those docs landed in val against 0.587%
expected by pure chance. Indistinguishable. Every per-domain nat ever recorded was
training-set loss.

The fix routes both callers -- domain_loss.py's CLI and score_matrix's metric --
through val_seqs, which reconstructs val the way train.py does. This asserts the
property that makes the fix real:

    the rows val_seqs returns are the rows train.py holds out, and they are
    DISJOINT from the training pool it hands the loader

A fix that returned the same rows under a new function name would pass every
type-check and change nothing. That is what this exists to catch.

No GPU: builds a small fake corpus, runs the real _domain_seqs through the real cache,
and compares the real slices. Tokenizing a few hundred short documents is seconds.

    python3 scripts/test_domain_loss_val.py

# restartable: writes only into a tempdir it removes on exit. An interrupt costs the
# tokenization of a few hundred synthetic documents -- seconds -- and leaves nothing a
# rerun would have to reconcile.
"""

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))


def main():
    import torch

    import train
    from domain_loss import seqs_fp, val_seqs

    bad = []
    d = tempfile.mkdtemp(prefix="dlval_")
    dom = "probe_domain"
    corpus = os.path.join(d, "corpus", dom)
    os.makedirs(corpus)
    # Enough documents to pack into several sequences at a short seq length. Distinct
    # text per row so a row's identity is visible in the tokens.
    # "content", not "text". train.py:1197 `_jsonl_content` reads ["content"] and raises
    # KeyError on anything else -- the same KeyError('content') that killed the 20B launch
    # at step 0. This test wrote "text" and so died in its own setup, which is why it read
    # as an unrunnable candidate for deletion (de-5, 2026-09-02): a stale fixture key, not
    # a dead test. Measured on the pod after the fix: OK, val_seqs returns train.py's
    # held-out rows, disjoint from the training pool.
    with open(os.path.join(corpus, "shard_000.jsonl"), "w", encoding="utf-8") as f:
        for i in range(4000):
            f.write(json.dumps({"content": f"document number {i} " + ("alpha beta gamma " * 8)}) + "\n")

    from scripts.loader import load_tokenizer

    tok_path = os.path.join(ROOT, "data", "tokenizer.json")
    if not os.path.exists(tok_path):
        # Gitignored, present only where the corpus is (the pod). SKIP rather than fail:
        # a red on every laptop is a red nobody acts on. Run it on the pod, where the
        # thing it guards actually runs.
        print(f"SKIP: no {os.path.relpath(tok_path, ROOT)} on this machine; run on the pod")
        shutil.rmtree(d, ignore_errors=True)
        return 0
    tok = load_tokenizer(tok_path, None)

    old_data, old_seq, old_frac, old_max = train.DATA, train.Cfg.seq, train.Cfg.val_frac, train.Cfg.val_rows_max
    old_token_cache, old_vid = train.TOKEN_CACHE, train.VOCAB_ID
    try:
        train.DATA = d
        train.Cfg.seq = 128           # short: many sequences from a small corpus
        train.Cfg.val_frac = 0.05
        train.Cfg.val_rows_max = 5000
        # train.TOKEN_CACHE, not os.environ["HARNESS_TOKEN_CACHE_DIR"]. train.py never
        # reads that variable -- only scripts/harness.py's _token_cache_dir does -- so
        # setting it redirected nothing and _domain_cache_path still returned
        # /data00/tokens_probe_domain.pt. MEASURED 2026-09-02: this test wrote a real
        # cache into the pod's shared /data00 at 05:50:42Z with a 0-byte .vocab beside
        # it, because load_tokenizer leaves train.VOCAB_ID None and :1686 writes
        # `VOCAB_ID or ""`. probe_domain is not in mix_500m so the live run was unharmed,
        # and that was luck: any domain name colliding with the mix would have poisoned a
        # cache the run reads. A test that writes outside its tempdir is not a CPU-only
        # test (fb, 2026-09-02).
        os.makedirs(os.path.join(d, "cache"), exist_ok=True)
        train.TOKEN_CACHE = os.path.join(d, "cache", "tokens.pt")
        assert train._domain_cache_path(dom).startswith(d), (
            f"the cache would land at {train._domain_cache_path(dom)}, outside this "
            f"test's tempdir {d}"
        )
        # And the vocabulary fingerprint, for the same reason one line up: load_tokenizer
        # does not set it, so without this the cache this test writes carries a 0-byte
        # .vocab -- which val_seqs' freshness guard now refuses, correctly. Setting it from
        # the tokenizer rather than from a checkpoint because there is no checkpoint here.
        from scripts.loader import vocab_fingerprint
        train.VOCAB_ID = vocab_fingerprint(tok)

        seqs = train._domain_seqs(dom, tok, True, False)
        seqs = seqs[0] if train.Cfg.fone else seqs
        if seqs is None or len(seqs) < 20:
            print(f"SKIP: the fake corpus packed into {0 if seqs is None else len(seqs)} "
                  f"sequences, too few to split")
            return 0

        n_val = min(max(1, int(len(seqs) * train.Cfg.val_frac)), train.Cfg.val_rows_max)
        val_expected = seqs[:n_val]
        pool = seqs[n_val:]          # exactly what build_mix hands the loader

        got = val_seqs(dom, tok, cap=len(val_expected))
        if got is None:
            bad.append("val_seqs returned None for a domain with shards")
        else:
            if not torch.equal(got.long(), val_expected[: len(got)].long()):
                bad.append("val_seqs did not return train.py's val slice")
            # The property, not the implementation: no val row may appear in the pool.
            # Compared as row bytes, because a row is only 'the same row' if identical.
            pool_rows = {r.numpy().tobytes() for r in pool}
            leaked = sum(1 for r in got if r.numpy().tobytes() in pool_rows)
            if leaked:
                bad.append(f"{leaked}/{len(got)} val rows also appear in the training pool")

            # And the fix must not be a rename: the OLD path (head of the first shard)
            # must produce different bytes from the new one. If these agree, either the
            # corpus is degenerate or nothing changed.
            from domain_loss import head_texts
            head = head_texts(os.path.join(corpus, "shard_000.jsonl"), 4000)
            head_ids = []
            for t in head[:64]:
                head_ids.extend(tok.encode(t).ids)
            n = len(head_ids) // train.Cfg.seq
            if n:
                head_rows = torch.tensor(head_ids[: n * train.Cfg.seq]).view(n, train.Cfg.seq)
                if seqs_fp(head_rows[: len(got)]) == seqs_fp(got):
                    bad.append("the shard head and the val slice hash the same -- either the "
                               "shuffle is off or the fix changed nothing")

    finally:
        train.DATA, train.Cfg.seq = old_data, old_seq
        train.Cfg.val_frac, train.Cfg.val_rows_max = old_frac, old_max
        train.TOKEN_CACHE = old_token_cache
        train.VOCAB_ID = old_vid
        shutil.rmtree(d, ignore_errors=True)

    if bad:
        print("FAIL: domain_loss is not scoring val")
        for b in bad:
            print(f"  {b}")
        return 1
    print("OK: val_seqs returns train.py's held-out rows, disjoint from the training "
          "pool, and differs from the old shard-head slice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
