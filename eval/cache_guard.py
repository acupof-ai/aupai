#!/usr/bin/env python3
"""An eval may READ a training token cache. It may never rebuild one.

The incident (fb, 2026-09-02, killed by exact PID two minutes in). `eval/ppl.py --mix
data/mix_500m.json` on card 7 printed

    mix: math_owm_stage2 cache was built by another vocabulary, retokenizing

against the nine caches the live 20B run is training on. train.VOCAB_ID is set only by
train.build_tokenizer (train.py:1462-1472); ppl.py goes through
scripts.loader.load_checkpoint, which never touches it, so VOCAB_ID stayed None.
train.py:1647 then reads

    same_vocab = os.path.exists(stamp) and open(stamp).read().strip() == (VOCAB_ID or "")

which is False against any real stamp, and :1686 writes `VOCAB_ID or ""` -- a 0-byte
stamp. So the eval would have retokenized all nine and left them stamped empty, and the
run's next resume would have retokenized them AGAIN. Cost: 851,965 documents and five
minutes of idle cards per domain, in the middle of a 20B run.

Two properties this asserts before any eval calls _domain_seqs, and the second is the
one a mirror of train.py:1472 alone does not buy:

1. VOCAB_ID is set, from the CHECKPOINT's vocab_id. Setting it makes the stamp compare
   correctly; it does not make rebuilding wrong. The default action is still "rebuild".
2. Every domain's cache is present and its three stamps -- .vocab, .srcfp, .seed --
   match, and the cache is newer than its shards. Any mismatch RAISES and names the
   domain and the stamp. That is the difference between an eval that scores what the run
   trained on and an eval that quietly redefines it.

Why a separate module and not a line in each tool: three tools call _domain_seqs off the
training path (eval/ppl.py, eval/domain_loss.py, eval/score_matrix.py through
val_seqs), and a guard copied three times is a guard that diverges in two of them. The
freshness arithmetic is READ from train.py's own values -- _domain_cache_path,
_corpus_fp, _sample_seed -- rather than restated, so a change there cannot leave this
agreeing with a rule that no longer exists.

The permanent fix belongs in build_mix, which must refuse when VOCAB_ID is None instead
of rebuilding, and must treat an empty stamp as stale. train.py is frozen for
p500m_20b_0902; this is what the eval side can do today, and it is strictly narrower --
it protects the caches from evals, not from a mis-launched training run.

    python3 eval/cache_guard.py --selftest
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STAMPS = (".vocab", ".srcfp", ".seed")


class CacheWouldRebuild(RuntimeError):
    """An eval was about to retokenize a training cache. Raised, never warned."""


def set_vocab_id(cfg):
    """Set train.VOCAB_ID from a checkpoint's cfg. Returns the fingerprint.

    load_checkpoint puts the checkpoint's vocab_id on cfg (scripts/loader.py:51), and
    load_tokenizer has already refused a tokenizer that disagrees with it -- so by the
    time an eval gets here the value is verified, and no re-fingerprinting is needed.

    A checkpoint with no vocab_id (pre-2026-08-29) raises rather than defaulting: a None
    here is exactly the state that wrote seven 0-byte stamps on the pod.
    """
    import train

    vid = getattr(cfg, "vocab_id", None)
    if not vid:
        raise CacheWouldRebuild(
            "this checkpoint carries no vocab_id, so train.VOCAB_ID cannot be set and "
            "every cache stamp would read as a mismatch: _domain_seqs would retokenize "
            "each domain and stamp it with an empty vocabulary. Score this checkpoint "
            "with eval/domain_loss.py's own text path, or re-save it with a vocab_id."
        )
    train.VOCAB_ID = vid
    return vid


def assert_caches_fresh(domains, root=ROOT):
    """Raise unless every domain's cache exists and _domain_seqs would reuse it as-is.

    Same five conditions train.py:1652-1665 ANDs together, read from train.py's own
    helpers. Reports every domain that fails, not the first: an eval run whose mix has
    two stale domains should learn both in one line rather than one per rerun.
    """
    import train

    bad = []
    for name in domains:
        cache = train._domain_cache_path(name)
        if not os.path.exists(cache):
            bad.append(f"{name}: no cache at {cache}")
            continue
        ddir = os.path.join(train.DATA, "corpus", name)
        if not os.path.isdir(ddir):
            bad.append(f"{name}: cache exists but data/corpus/{name} does not, so "
                       f"freshness cannot be checked")
            continue
        want = {
            ".vocab": train.VOCAB_ID or "",
            ".srcfp": train._corpus_fp(ddir),
            ".seed": str(train._sample_seed()),
        }
        for ext in STAMPS:
            p = cache + ext
            if not os.path.exists(p):
                bad.append(f"{name}: {os.path.basename(p)} missing -- an unstamped cache "
                           f"rebuilds")
                continue
            got = open(p, encoding="utf-8").read().strip()
            if not got:
                # The 0-byte case, which is the incident's own signature: train.py's
                # `== (VOCAB_ID or "")` makes an empty stamp MATCH a None VOCAB_ID, so
                # without this branch a cache poisoned by an earlier unstamped eval reads
                # as fresh to the next one. Seven such .vocab files sit on the pod dated
                # 09-01, all written before today (tilerl, 2026-09-02).
                bad.append(f"{name}: {os.path.basename(p)} is empty -- written by a run "
                           f"with no vocabulary fingerprint; it cannot certify anything")
                continue
            if got != want[ext]:
                bad.append(f"{name}: {os.path.basename(p)} is {got[:16]}, this run needs "
                           f"{want[ext][:16]}")
        shards = [
            os.path.join(ddir, b) for b in sorted(os.listdir(ddir))
            if b.endswith(".jsonl")
            and b not in train.NON_SHARD_JSONL
            and not train.NON_SHARD_RE.search(b)
            and train.SHARD_RE.search(b)
        ]
        if shards and os.path.getmtime(cache) < max(os.path.getmtime(p) for p in shards):
            bad.append(f"{name}: a shard is newer than the cache")
    if bad:
        raise CacheWouldRebuild(
            "REFUSING: an eval must never rebuild a training token cache. "
            + f"{len(bad)} problem(s):\n  " + "\n  ".join(bad)
            + "\n\nRetokenizing here would overwrite caches the live run reads and stamp "
              "them from this process's state, so the run's next resume retokenizes too. "
              "Warm the cache from the training path instead "
              "(`python datagen/pretokenize.py --domains <names>`), or score a mix whose "
              "domains are already cached."
        )
    return len(list(domains))


def guard(cfg, domains, root=ROOT):
    """Both halves, in the order an eval needs them: fingerprint, then freshness."""
    set_vocab_id(cfg)
    return assert_caches_fresh(domains, root=root)


def _selftest():
    """Four worlds, three of which must raise. A guard whose refusal is never provoked
    is an assertion about the author's intent."""
    import json
    import shutil
    import tempfile
    from types import SimpleNamespace

    import train

    d = tempfile.mkdtemp(prefix="cacheguard_")
    old_data, old_cache, old_vid = train.DATA, train.TOKEN_CACHE, train.VOCAB_ID
    try:
        dom = "guard_probe"
        corpus = os.path.join(d, "corpus", dom)
        os.makedirs(corpus)
        with open(os.path.join(corpus, "shard_000.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"content": "one document"}) + "\n")
        os.makedirs(os.path.join(d, "cache"))
        train.DATA = d
        train.TOKEN_CACHE = os.path.join(d, "cache", "tokens.pt")

        # 1. no vocab_id on the cfg -> refuse (the incident's own state).
        try:
            set_vocab_id(SimpleNamespace(vocab_id=None))
            raise AssertionError("a cfg with no vocab_id was accepted")
        except CacheWouldRebuild as e:
            assert "no vocab_id" in str(e), e

        set_vocab_id(SimpleNamespace(vocab_id="deadbeefdeadbeef"))
        assert train.VOCAB_ID == "deadbeefdeadbeef"

        # 2. no cache at all -> refuse. This is what a fresh /data00 looks like, and an
        #    eval must not be the process that fills it.
        try:
            assert_caches_fresh([dom])
            raise AssertionError("a missing cache was accepted")
        except CacheWouldRebuild as e:
            assert "no cache at" in str(e), e

        cache = train._domain_cache_path(dom)
        open(cache, "w").write("x")
        for ext, val in ((".vocab", "deadbeefdeadbeef"),
                         (".srcfp", train._corpus_fp(corpus)),
                         (".seed", str(train._sample_seed()))):
            with open(cache + ext, "w") as f:
                f.write(val)
        # The cache must be newer than the shard, as train.py requires.
        os.utime(cache, None)

        # 3. every stamp right -> pass. Without this the other three prove only that the
        #    guard refuses, which a `raise` on line 1 also does.
        assert assert_caches_fresh([dom]) == 1

        # 4. one byte changed in .vocab -> refuse, naming the domain and the stamp
        #    (fb's stated broken world).
        with open(cache + ".vocab", "w") as f:
            f.write("deadbeefdeadbeee")
        try:
            assert_caches_fresh([dom])
            raise AssertionError("a one-byte .vocab difference was accepted")
        except CacheWouldRebuild as e:
            assert dom in str(e) and "tokens_guard_probe.pt.vocab" in str(e), e

        # 5. the 0-byte stamp, which train.py's `== (VOCAB_ID or "")` would call a match
        #    whenever VOCAB_ID is None. Checked with VOCAB_ID None to reproduce exactly
        #    that pairing: train.py says fresh, this says stale.
        with open(cache + ".vocab", "w") as f:
            f.write("")
        train.VOCAB_ID = None
        assert open(cache + ".vocab").read().strip() == (train.VOCAB_ID or "")
        try:
            assert_caches_fresh([dom])
            raise AssertionError("an empty .vocab stamp was accepted")
        except CacheWouldRebuild as e:
            assert "is empty" in str(e), e
    finally:
        train.DATA, train.TOKEN_CACHE, train.VOCAB_ID = old_data, old_cache, old_vid
        shutil.rmtree(d, ignore_errors=True)
    print("selftest OK: refuses a missing vocab_id, a missing cache, a one-byte stamp "
          "difference and a 0-byte stamp; accepts a fully-stamped fresh cache")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else 0)
