#!/usr/bin/env python3
"""The .counts sidecar says the same thing the cache does, and the two paths agree on pool_rows.

    python3 scripts/test_cache_counts.py

WHAT THIS IS FOR. scripts/write_mix_500m.py sizes every pool by torch.load(mmap) of the whole
cache. That read goes through assert_not_co_resident, which refuses any domain over 10 GB while
a run is live -- which is every large one -- so the mix could not be re-derived beside the job
it was sizing for. train.py now writes <cache>.counts at build time and _cache_pool reads it
first. The number is only worth having if it is the SAME number the expensive path produces.

THE ASSERTION IS pool_rows, NOT rows. _cache_pool returns pool_rows = rows - min(int(rows*0.05),
5000): comparing `rows` would pass while the two paths disagreed by up to 5000 rows further down,
because a mix draws against pool_rows and nothing else (62, 2026-09-07). Both arms are compared
whole, key by key, so a field added to one path and not the other fails here.

THE FIXTURE IS A REAL CACHE FROM _domain_seqs, not a hand-written tensor. The sidecar's whole
claim is "written by the code that produced the tensor", and a fixture that writes both halves
itself asserts that this file can do arithmetic. So the cache and its four sidecars come out of
one real _domain_seqs call, and the two arms differ only in whether .counts is on disk.

WHAT THIS DOES NOT TEST: the co-residency refusal itself (scripts/test_cache_absent_refusal.py
and eval/cache_guard.py own that), and the fone cache path -- _cache_pool's torch.load arm
raises AttributeError on a fone cache's tuple and the bare `except Exception` returns None, a
pre-existing defect this change does not touch, so there is no second arm to compare against.
"""
import io
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class _Tok:
    """Enough tokenizer for encode(): data/tokenizer.json is gitignored and the vocabulary is
    not the variable here."""

    def token_to_id(self, t):
        return 1

    def encode_batch(self, texts):
        return [_Enc([2, 3, 4, 5, 6, 7, 8, 9]) for _ in texts]


def _isolate(train, d):
    """Point train at a tempdir and REFUSE if the cache would land outside it.

    scripts/test_domain_loss_val.py set an environment variable train.py never reads and wrote
    a real cache into the pod's shared /data00 beside the live run's (2026-09-02). Nothing
    raised. This asserts the path, rather than trusting the knob."""
    train.DATA = d
    os.makedirs(os.path.join(d, "cache"), exist_ok=True)
    train.TOKEN_CACHE = os.path.join(d, "cache", "tokens.pt")
    train.VOCAB_ID = "test-cache-counts"
    train.Cfg.seq = 8
    train.Cfg.fone = False
    train.Cfg.seed = 0
    train.Cfg.sample_seed = None
    p = train._domain_cache_path("probe")
    if not os.path.abspath(p).startswith(os.path.abspath(d)):
        raise RuntimeError(f"cache would land at {p}, outside this test's tempdir {d}")


def _mkcorpus(root, dom, ndocs=400):
    # "content", not "text": _jsonl_content reads ["content"] and a KeyError here would be the
    # test dying in its own setup rather than measuring anything.
    cdir = os.path.join(root, "corpus", dom)
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, f"{dom}_000.jsonl"), "w", encoding="utf-8") as f:
        for i in range(ndocs):
            f.write(json.dumps({"content": f"document {i} alpha beta gamma"}) + "\n")
    return cdir


def _quiet(fn, *a, **kw):
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        return fn(*a, **kw), buf.getvalue()
    finally:
        sys.stdout = old


def main():
    import scripts.write_mix_500m as w
    import train

    d = tempfile.mkdtemp(prefix="cachecounts_")
    _isolate(train, d)
    _mkcorpus(d, "probe")
    # SEQ is a module constant in the writer and Cfg.seq is what the cache was built at. The
    # fixture is small, so it is the writer that bends: without this, rows is computed at 4096
    # against a cache built at 8 and both arms agree on a number that describes nothing.
    w.SEQ = train.Cfg.seq

    _quiet(train._domain_seqs, "probe", _Tok(), True, False)
    cache = train._domain_cache_path("probe")
    side = cache + ".counts"

    check(os.path.exists(side), f"_domain_seqs built {cache} and wrote no {side}")
    if FAILS:
        return _report()

    # ARM 1: the sidecar. ARM 2: the same call with the file moved away, which is the only
    # difference between them -- same cache, same Cfg, same function.
    from_counts, _ = _quiet(w._cache_pool, "probe")
    os.rename(side, side + ".hidden")
    try:
        from_cache, _ = _quiet(w._cache_pool, "probe")
    finally:
        os.rename(side + ".hidden", side)

    check(from_counts is not None and from_counts["source"] == "counts",
          f"the sidecar arm did not answer from .counts: {from_counts}")
    check(from_cache is not None and from_cache["source"] == "cache",
          f"the fallback arm did not answer from the cache read: {from_cache}")
    if FAILS:
        return _report()

    # THE WHOLE DICT, minus the field whose job is to differ. Comparing only pool_rows would
    # pass while `tokens` disagreed, and `tokens` is what a mix's epoch arithmetic divides by.
    a = {k: v for k, v in from_counts.items() if k != "source"}
    b = {k: v for k, v in from_cache.items() if k != "source"}
    check(a == b, f"the two paths disagree: from .counts {a}, from the cache {b}")
    check(a["pool_rows"] < a["rows"],
          f"pool_rows {a['pool_rows']} did not subtract a holdout from rows {a['rows']}, so "
          f"comparing them above compared rows to rows and the holdout is untested")

    # AND AGAINST THE TENSOR ITSELF, so this is not two readers agreeing on one wrong file.
    import torch

    flat = torch.load(cache, map_location="cpu", mmap=True)
    check(a["tokens"] == flat.numel(),
          f".counts says {a['tokens']} tokens, the tensor holds {flat.numel()}")
    check(a["rows"] == flat.numel() // (train.Cfg.seq + 1),
          f"rows {a['rows']} against len(flat)//(seq+1) "
          f"{flat.numel() // (train.Cfg.seq + 1)}")

    # A MODE MISMATCH RAISES rather than falling back to the expensive read of the same wrong
    # file. Written by hand, because producing it for real needs a second cache under --fone.
    with open(side, "w", encoding="utf-8") as f:
        json.dump({"tokens": a["tokens"], "seq": train.Cfg.seq, "fone": True}, f)
    try:
        got, _ = _quiet(w._cache_pool, "probe")
        check(False, f"a fone mismatch returned {got} instead of raising")
    except ValueError as e:
        check("fone" in str(e), f"the refusal must name the field that mismatched: {e}")
    with open(side, "w", encoding="utf-8") as f:
        json.dump({"tokens": a["tokens"], "seq": train.Cfg.seq + 1, "fone": False}, f)
    try:
        got, _ = _quiet(w._cache_pool, "probe")
        check(False, f"a seq mismatch returned {got} instead of raising")
    except ValueError as e:
        check("seq" in str(e), f"the refusal must name the field that mismatched: {e}")

    # A DAMAGED file is not a mismatch: it falls back, and says so.
    with open(side, "w", encoding="utf-8") as f:
        f.write("{not json")
    got, log = _quiet(w._cache_pool, "probe")
    check(got is not None and got["source"] == "cache",
          f"an unreadable .counts must fall back, not answer: {got}")
    check("unreadable" in log,
          f"the fallback must be loud -- otherwise `source` is the only trace: {log!r}")

    # A WELL-FORMED OBJECT MISSING A KEY is the same class: damaged, not a mismatch.
    # {"tokens": N} alone used to reach the mismatch comparison outside the read's try
    # and escape as an uncaught KeyError, taking down the whole mix write (3b's case 2).
    with open(side, "w", encoding="utf-8") as f:
        json.dump({"tokens": a["tokens"]}, f)
    got, log = _quiet(w._cache_pool, "probe")
    check(got is not None and got["source"] == "cache",
          f"a .counts missing seq/fone must fall back, not answer or raise: {got}")
    check("unreadable" in log, f"the fallback must be loud: {log!r}")

    return _report()


def _report():
    for f in FAILS:
        print(f"FAIL: {f}")
    if FAILS:
        return 1
    print("ok  .counts and the cache read produce the same pool_rows, a mode mismatch raises, "
          "and a damaged sidecar falls back loudly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
