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

#: Where the co-residency refusal reads its threshold and its per-domain byte table.
#: scripts/ is not a package, so this is a path load rather than an import.
_ELC = None


def _elc():
    global _ELC
    if _ELC is None:
        import importlib.util

        p = os.path.join(ROOT, "scripts", "eval_load_cost.py")
        spec = importlib.util.spec_from_file_location("_eval_load_cost", p)
        _ELC = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ELC)
    return _ELC


class CacheWouldRebuild(RuntimeError):
    """An eval was about to retokenize a training cache. Raised, never warned."""


class CoResidentCacheRead(RuntimeError):
    """An eval was about to read a large token cache while a training run holds the box."""


def _live_run_cards():
    """{card: claim_name} for cards a LIVE claim holds, or {} if that cannot be read.

    Reads runs/claims/ through scripts/card_claim.py, which is the only writer and already
    knows the two things a naive reader gets wrong: a claim whose pid is gone is stale, and
    a zombie pid is NOT stale (deleting that claim hands its cards away -- card_claim.py:294).

    {} on any failure, and the caller WARNS rather than refusing on it. A guard that
    refuses when it cannot read the claims would block every eval on a laptop with no
    runs/claims/, which is where its own selftest runs.
    """
    try:
        import importlib.util

        p = os.path.join(ROOT, "scripts", "card_claim.py")
        spec = importlib.util.spec_from_file_location("_card_claim", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        live, _stale = mod.claims()
        return mod.held_cards(live)
    except Exception:
        return {}


def assert_not_co_resident(domains, root=ROOT):
    """Refuse a >10 GB token-cache read while a live claim holds cards. AGENTS.md's
    "no lane card at all" rule, at the moment of the decision instead of in a table.

    THE RULE'S OWN QUANTITY, which is why this sits here and not in score_matrix: the rule
    says co-residency is judged by host IO, not by metric class, and host IO enters the
    process through _domain_seqs. Every eval that reads a cache passes val_seqs or
    _domain_seqs, and assert_caches_fresh is already the one function they all reach, so a
    refusal here covers ppl.py, domain_loss.py, domain_bpb.py and score_matrix's
    domain_loss metric with no per-caller line to forget.

    WHY score_matrix WAS THE WRONG PLACE (6e proposed it, and the table says otherwise):
    of its fourteen metrics exactly one reaches a cache. Refusing at its entry would refuse
    the thirteen that read only a checkpoint -- including the four likelihood metrics whose
    46 s is the measurement that says scoring beside a run is fine. The refusal has to key
    on the read, not on the tool.

    THREE OUTCOMES, and the middle one is the point:
      no live claim            -> pass, silently. Nothing to be co-resident with.
      cost unknown             -> WARN by name and proceed. 39 of 42 evals have no
                                  throughput-dip measurement, so refusing on unknown would
                                  ban scoring beside any run, which the 46 s contradicts.
      >= CO_RESIDENCY_BYTES    -> raise. This is ppl.py's 166 GB, the read that was killed
                                  as 事前止损 on 2026-09-02 before it happened.

    AUPAI_ALLOW_CORESIDENT_CACHE=1 overrides, because the controller does lend block cards
    and a rescore ON the run's own card is sometimes the order. It prints what it allowed.
    """
    elc = _elc()
    want, unknown = elc.domains_cache_bytes(domains)
    held = _live_run_cards()
    if not held:
        return 0
    who = sorted({n for names in held.values() for n in names})
    if unknown:
        # NAMED, and pointing at where the measurement goes. An unmeasured domain is a hole
        # in `want`, so `want` is a lower bound here and the print says so.
        print(f"WARNING co-residency cost partly unknown: {len(unknown)} domain(s) with no "
              f"recorded cache size ({', '.join(sorted(unknown)[:4])}) while cards "
              f"{','.join(sorted(held))} are claimed by {','.join(who)}. Proceeding on a lower "
              f"bound of {want / 1e9:.1f} GB. Measure with scripts/eval_load_cost.py and add "
              f"the domain to its CACHE_BYTES.", flush=True)
    if want < elc.CO_RESIDENCY_BYTES:
        return 0
    if os.environ.get("AUPAI_ALLOW_CORESIDENT_CACHE"):
        print(f"AUPAI_ALLOW_CORESIDENT_CACHE=1: reading {want / 1e9:.1f} GB of token cache "
              f"while cards {','.join(sorted(held))} are claimed by {','.join(who)}.",
              flush=True)
        return 0
    raise CoResidentCacheRead(
        f"REFUSING: this eval is about to read {want / 1e9:.1f} GB of token cache off /data00 "
        f"while cards {','.join(sorted(held))} are held by a live claim ({','.join(who)}).\n"
        f"  domains: {', '.join(sorted(domains))}\n"
        f"  threshold: {elc.CO_RESIDENCY_BYTES / 1e9:.0f} GB "
        f"(scripts/eval_load_cost.py CO_RESIDENCY_BYTES)\n\n"
        f"AGENTS.md, Lanes: with no lane card, co-residency is judged by host IO, not by "
        f"metric class. A 2 GB checkpoint load costs about what the run's own save costs "
        f"(78 s control, and score_matrix's four likelihood metrics 46 s); a whole-cache read "
        f"is a different quantity and has never been allowed to finish beside a run, so its "
        f"cost is unmeasured rather than small. On 2026-09-02 eval/ppl.py was killed two "
        f"minutes in for exactly this read.\n"
        f"Wait for the run, score a mix whose domains are already in host RAM, or "
        f"AUPAI_ALLOW_CORESIDENT_CACHE=1 if the controller granted the card."
    )


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

    Also the co-residency refusal, BEFORE the freshness loop: both answer "may this
    process read these caches now", and putting them in one function is what makes the
    second one reach every caller (the freshness guard's own docstring reason -- a guard
    each caller has to remember is a guard one of them eventually will not).
    """
    import train

    assert_not_co_resident(domains, root=root)
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
    n_co = _selftest_co_resident()
    print("selftest OK: refuses a missing vocab_id, a missing cache, a one-byte stamp "
          f"difference and a 0-byte stamp; accepts a fully-stamped fresh cache; "
          f"co-residency refusal correct on {n_co} worlds")
    return 0


def _selftest_co_resident():
    """The co-residency refusal, on a claim dir this test writes.

    Every world drives the decision through a REAL claim file read by card_claim.claims(),
    not by stubbing _live_run_cards: the thing under test is whether a claim on disk reaches
    the refusal, and a stub asserts my belief about the reader instead of the reader.

    The claim must name a LIVE pid, so it names this process. A dead pid is filed stale and
    the guard would pass for the right reason on the wrong world -- which is the mistake
    this docstring exists to stop the next reader repeating.
    """
    import json
    import shutil
    import tempfile

    elc = _elc()
    d = tempfile.mkdtemp(prefix="coresident_")
    old_dir = os.environ.get("AUPAI_CLAIM_DIR")
    old_allow = os.environ.pop("AUPAI_ALLOW_CORESIDENT_CACHE", None)
    # BIG must exceed the threshold and SMALL must not, read from the table rather than
    # hardcoded: a future re-measurement moves the bytes and a literal here would go stale
    # while still passing.
    big = [n for n, b in elc.CACHE_BYTES.items() if b >= elc.CO_RESIDENCY_BYTES]
    small = [n for n, b in elc.CACHE_BYTES.items() if b < elc.CO_RESIDENCY_BYTES]
    assert big and small, (
        f"the byte table no longer straddles CO_RESIDENCY_BYTES "
        f"({elc.CO_RESIDENCY_BYTES / 1e9:.0f} GB): {len(big)} above, {len(small)} below. "
        f"With one side empty this selftest cannot tell a refusal from a blanket ban")
    n = 0
    try:
        def _must_pass(doms, why):
            """Call it and turn a refusal into an AssertionError that says which world.

            Not `assert assert_not_co_resident(...) == 0`: CoResidentCacheRead is not an
            AssertionError, so a refusal in a must-pass world propagated as a bare
            exception and the message saying WHICH world was lost. Measured while mutating
            -- three mutants failed with only the refusal text, which reads identically
            whether the world expected a refusal or not.
            """
            try:
                return assert_not_co_resident(doms)
            except CoResidentCacheRead as e:
                raise AssertionError(f"{why}: {str(e).splitlines()[0]}") from None

        # 1. NO CLAIM -> pass even on the largest read. This is the half that keeps the
        #    guard from becoming a ban: an empty claim dir is a laptop, or the pod between
        #    runs, and neither is co-residency.
        os.environ["AUPAI_CLAIM_DIR"] = os.path.join(d, "empty")
        os.makedirs(os.environ["AUPAI_CLAIM_DIR"], exist_ok=True)
        assert _must_pass(big, "refused with no live claim -- that is a ban, not a guard") == 0
        n += 1

        held = os.path.join(d, "held")
        os.makedirs(held)
        with open(os.path.join(held, "p500m.json"), "w", encoding="utf-8") as fh:
            json.dump({"name": "p500m_selftest", "cards": ["0", "1"], "pid": os.getpid(),
                       "cmdline": "python train.py --name p500m_selftest",
                       "acquired": "2026-09-04 00:00:00", "note": "cache_guard selftest"}, fh)
        os.environ["AUPAI_CLAIM_DIR"] = held

        # 2. A live claim AND a big read -> refuse, naming the cards, the claimant and the
        #    threshold. Without all three an operator cannot act on the message.
        try:
            assert_not_co_resident(big)
            raise AssertionError(
                f"a {sum(elc.CACHE_BYTES[x] for x in big) / 1e9:.0f} GB cache read was "
                f"allowed while a live claim held cards 0,1")
        except CoResidentCacheRead as e:
            for want in ("p500m_selftest", "0,1", "10 GB"):
                assert want in str(e), f"the refusal does not name {want!r}: {str(e)[:200]}"
        n += 1

        # 3. A live claim and a SMALL read -> pass. The rule is about bytes, not about
        #    whether a run exists, and this is the assertion that says so.
        assert _must_pass(
            small[:1],
            f"{small[0]} is {elc.CACHE_BYTES[small[0]] / 1e9:.2f} GB and was refused beside a "
            f"live claim -- the threshold is not being applied, the presence of a run is") == 0
        n += 1

        # 4. UNKNOWN cost -> warn and proceed, not refuse. 6e's ruling, and the number
        #    behind it: 39 of 42 evals are unmeasured, so refusing here bans everything.
        #    THE WARN TEXT IS ASSERTED, not just the return value: proceeding silently on an
        #    unknown cost is indistinguishable from proceeding on a measured-small one, and
        #    6e's ruling was specifically that the warning names the file where the
        #    measurement goes. Found by mutation -- `if unknown:` -> `if False:` left every
        #    other assertion green.
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = _must_pass(
                ["a_domain_nobody_measured"],
                "an unmeasured domain was refused -- 39 of 42 evals have no measurement, so "
                "this turns the guard into a blanket ban beside any run")
        assert got == 0
        warned = buf.getvalue()
        for want in ("a_domain_nobody_measured", "eval_load_cost.py", "lower bound"):
            assert want in warned, (
                f"the unknown-cost path did not warn with {want!r} -- it proceeded silently, "
                f"which reads the same as a measured-small cost. Printed: {warned[:200]!r}")
        n += 1

        # 5. The override works and is loud. It exists because the controller does lend
        #    block cards; a silent override would be indistinguishable from no guard.
        os.environ["AUPAI_ALLOW_CORESIDENT_CACHE"] = "1"
        assert _must_pass(big, "the documented override does not override") == 0
        n += 1
        del os.environ["AUPAI_ALLOW_CORESIDENT_CACHE"]

        # 6. THE CHOKEPOINT ACTUALLY ASKS. Worlds 1-5 call assert_not_co_resident directly,
        #    so all five stay GREEN when the call is deleted from assert_caches_fresh -- a
        #    working guard nothing invokes, which is the shape that let a registered
        #    selftest guard zero commits (e1, 2026-09-04). MEASURED: with the call removed
        #    from assert_caches_fresh, `--selftest` passed until this world existed.
        #
        #    Reached through assert_caches_fresh with a domain that has NO cache on disk:
        #    the freshness loop would refuse it too, so the assertion is that the
        #    CO-RESIDENCY refusal comes FIRST -- CacheWouldRebuild here means the
        #    co-residency call is gone and the freshness check answered instead.
        try:
            assert_caches_fresh(big)
            raise AssertionError(
                "assert_caches_fresh allowed a 212 GB co-resident read: the chokepoint does "
                "not call assert_not_co_resident, so every eval bypasses the refusal")
        except CoResidentCacheRead:
            pass
        except CacheWouldRebuild as e:
            raise AssertionError(
                f"assert_caches_fresh reached the FRESHNESS check first, so the co-residency "
                f"call is missing or below it: {str(e).splitlines()[0]}") from None
        n += 1

        # 7. A claim whose pid is DEAD must not refuse: card_claim files it stale, and a
        #    crashed run must not lock the caches forever. pid 2 is init's child on Linux
        #    and absent in this container's namespace on macOS; assert it is really gone
        #    rather than assuming, so a live pid 2 cannot make this pass for the wrong
        #    reason.
        dead = os.path.join(d, "dead")
        os.makedirs(dead)
        gone = 2 ** 22 - 1
        try:
            os.kill(gone, 0)
            raise AssertionError(f"pid {gone} exists; pick another for the dead-claim world")
        except ProcessLookupError:
            pass
        except PermissionError:
            raise AssertionError(f"pid {gone} exists (EPERM); pick another") from None
        with open(os.path.join(dead, "crashed.json"), "w", encoding="utf-8") as fh:
            json.dump({"name": "crashed_run", "cards": ["0"], "pid": gone,
                       "cmdline": "python train.py", "acquired": "2026-09-04 00:00:00"}, fh)
        os.environ["AUPAI_CLAIM_DIR"] = dead
        assert _must_pass(
            big,
            "a claim whose pid is gone still refused -- a crashed run would lock every "
            "cache read until someone deleted the file by hand") == 0
        n += 1
    finally:
        if old_dir is None:
            os.environ.pop("AUPAI_CLAIM_DIR", None)
        else:
            os.environ["AUPAI_CLAIM_DIR"] = old_dir
        if old_allow is not None:
            os.environ["AUPAI_ALLOW_CORESIDENT_CACHE"] = old_allow
        else:
            os.environ.pop("AUPAI_ALLOW_CORESIDENT_CACHE", None)
        shutil.rmtree(d, ignore_errors=True)
    return n


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else 0)
