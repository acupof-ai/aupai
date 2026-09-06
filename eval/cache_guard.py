#!/usr/bin/env python3
"""An eval may READ a training token cache. It may never rebuild one.

The incident (fb, 2026-09-02, killed by exact PID two minutes in). `eval/ppl.py --mix
data/mix_500m.json` on card 7 printed

    mix: math_owm_stage2 cache was built by another vocabulary, retokenizing

against the nine caches the live 20B run is training on. train.VOCAB_ID is set only by
train.build_tokenizer; ppl.py goes through
scripts.loader.load_checkpoint, which never touches it, so VOCAB_ID stayed None.
_domain_seqs' freshness test then read

    same_vocab = os.path.exists(stamp) and open(stamp).read().strip() == (VOCAB_ID or "")

which is False against any real stamp, and the stamp write used `VOCAB_ID or ""` -- a 0-byte
stamp. So the eval would have retokenized all nine and left them stamped empty, and the
run's next resume would have retokenized them AGAIN. Cost: 851,965 documents and five
minutes of idle cards per domain, in the middle of a 20B run.

Two properties this asserts before any eval calls _domain_seqs, and the second is the
one a mirror of build_tokenizer alone does not buy:

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

THE PERMANENT FIX HAS LANDED IN build_mix, and this paragraph used to ask for it. Read
2026-09-07: _domain_seqs' freshness test now opens with `bool(VOCAB_ID) and`, and compares
against VOCAB_ID rather than `(VOCAB_ID or "")` -- so an empty stamp no longer equals an
unset VOCAB_ID -- and the stamp write raises `refusing to stamp ... with no vocabulary`
BEFORE open(), rather than writing 0 bytes. This module is still strictly narrower and
still worth having: it names every stale domain in one line instead of the first, and it
carries the co-residency refusal, which build_mix has no reason to. (The line here that
said train.py was frozen for
p500m_20b_0902 was removed 2026-09-05: that run ended 09-03 and train.py has taken many
commits since, so the sentence was an expired excuse for not doing the wider fix.)

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
    """{card: claim_name} for cards a live claim held by SOMEONE ELSE holds, or {} if that
    cannot be read.

    Reads runs/claims/ through scripts/card_claim.py, which is the only writer and already
    knows the two things a naive reader gets wrong: a claim whose pid is gone is stale, and
    a zombie pid is NOT stale (deleting that claim hands its cards away -- card_claim.py's
    claims() docstring).

    SOMEONE ELSE, and that clause is the fix for a self-deadlock (4c, 2026-09-06). Every GPU
    entry point now claims its own cards at startup via scripts/loader.claim_my_cards, so
    score_matrix wrote a claim and then read it back here and refused its own cache read. A
    claim naming this process, or an ancestor of it, is not something this process can be
    co-resident WITH -- it IS this process. `harness launch` claims for the job's ancestor, so
    the ancestor case is as ordinary as the self case.

    The cost of getting it wrong was not a stuck job: the refusal exited 0, so three rows
    landed in runs/score_matrix.jsonl with every metric an {"error": ...} dict and nothing
    said they were not scores (e1; main 5c0319dc makes score_matrix_present see them).

    Ancestry by the real ppid chain from card_claim, capped at MAX_DEPTH, so this cannot loop
    on a cyclic ppid table. A pid the table cannot see is treated as foreign -- the safe
    direction, since an unreadable table must not silently disable the guard.

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
        mine = _own_pids(mod)
        others = [c for c in live if int(c.get("pid", -1)) not in mine]
        return mod.held_cards(others)
    except Exception:
        return {}


def _own_pids(card_claim):
    """{this pid} plus every ancestor pid, for excluding this job's own claim.

    Walks the ppid chain from card_claim._ps_table(), which is the reader that works in both
    namespaces (macOS has no /proc, the pod's container has both). Depth-capped by
    card_claim.MAX_DEPTH: a cyclic ppid table would otherwise loop forever.

    Falls back to {this pid, its parent} when ps cannot run, rather than to everything or to
    nothing: the direct parent is the case `harness launch` produces, and treating an
    unreachable grandparent as foreign leaves the guard on.
    """
    me = os.getpid()
    out = {me}
    table = card_claim._ps_table()
    if not table:
        out.add(os.getppid())
        return out
    parent = {p: ppid for p, ppid, _args in table}
    cur = me
    for _ in range(card_claim.MAX_DEPTH):
        nxt = parent.get(cur)
        if nxt is None or nxt in out or nxt <= 0:
            break
        out.add(nxt)
        cur = nxt
    return out


def assert_not_co_resident(domains, root=ROOT):
    """Refuse a >10 GB token-cache read while a live claim holds cards. AGENTS.md's
    "no lane card at all" rule, at the moment of the decision instead of in a table.

    THE RULE'S OWN QUANTITY, which is why this sits here and not in score_matrix: the rule
    says co-residency is judged by host IO, not by metric class, and host IO enters the
    process through _domain_seqs. Every eval that reads a cache passes val_seqs or
    _domain_seqs, and assert_caches_fresh is already the one function they all reach, so a
    refusal here covers ppl.py, domain_loss.py, domain_bpb.py and score_matrix's
    domain_loss metric with no per-caller line to forget.

    THE TRAINING PATH IS NOT COVERED, AND THAT IS DELIBERATE (de ruling 2026-09-05, found by
    58 with /proc/<pid>/status VmRSS 101,434,380 kB and fd/8 -> tokens_zh_web.pt). The
    paragraph above is true of every EVAL and reads as though it were true of _domain_seqs
    itself; it is not. train.py calls _domain_seqs directly -- neither train.py nor
    scripts/loader.py contains assert_caches_fresh -- and _domain_seqs' cache reuse path is a
    full `torch.load(cache, map_location="cpu", weights_only=True)`, so a plan build over the
    E1 mix reads 166.2 GB unrefused. That is the same
    quantity this guard's own refusal text calls out as ppl.py's 166 GB.

    A training launch is the job the lane EXISTS for, so refusing it because another team
    holds a card inverts the priority: this guard was written to stop a 166 GB SCORING read
    from contending with a run, not to stop the run. Routing the training path through here
    would have blocked E1 on 2026-09-05. The exemption is recorded rather than left implicit
    because the sentence above had already been read as coverage once.

    What would change the ruling: a measurement that a full plan build under
    torch.load(mmap=True) is cheaper in practice. 58 measured that mmap gives the shape at
    0.0 MB rchar, but build_mix then indexes pools[name][mine[1][m]] across the whole plan
    and the pages fault in anyway -- unmeasured, so not a basis for editing the data path.

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


NVME_CACHE_DIR = "/mnt/data02/tokens"


def assert_cache_dir_not_overlay(root=ROOT):
    """Raise if the cache dir is on the overlay while the NVMe copy exists.

    MEASURED 2026-09-05: the container's /data00 is a DIRECTORY on the overlay (vda2, rotational,
    87% full) reading at 193 MB/s, while the same 8 GB off nvme2n1 reads at 1.3 GB/s -- 6.5x. The
    caches were copied to /mnt/data02/tokens for that reason.

    The failure this prevents is not slowness, it is SILENT slowness. Both directories hold a file
    called tokens_<domain>.pt with matching stamps, so a run that forgot AUPAI_TOKEN_CACHE_DIR is
    correct in every observable way and merely takes 6.5x longer to read its data -- which reads as
    "training is slow today", the least actionable symptom there is.

    IT REFUSES ONLY WHEN THE NVME COPY IS PRESENT. On a laptop, on CI, and on any host without that
    mount there is nothing to prefer, so this must not fire: a check that refuses where its remedy
    does not exist is a permanent red. That also makes it self-disarming if the mount is ever
    dropped -- the run proceeds off the overlay rather than refusing to start, which is the right
    failure direction for a data path.
    """
    import train

    if not os.path.isdir(NVME_CACHE_DIR):
        return None
    # AN EXPLICIT CHOICE IS NOT A FORGOTTEN VARIABLE. If the caller set the variable, they named
    # this directory on purpose and the refusal has nothing to add. Without this branch the message
    # below would tell them to set a variable they have already set, which is the shape of advice
    # that gets a check disabled.
    if os.environ.get("AUPAI_TOKEN_CACHE_DIR"):
        return train._token_cache_dir()
    cur = train._token_cache_dir()
    if os.path.realpath(cur) == os.path.realpath(NVME_CACHE_DIR):
        return cur
    try:
        same = os.stat(cur).st_dev == os.stat("/").st_dev
    except OSError:
        return None
    if not same:
        return cur  # somewhere else, but not the overlay: not this check's business
    raise CacheWouldRebuild(
        f"the token cache dir is {cur}, which is on the ROOT FILESYSTEM (the container overlay, "
        f"rotational, measured 193 MB/s), while the NVMe copy exists at {NVME_CACHE_DIR} "
        f"(1.3 GB/s, 6.5x). Reading the overlay copy is not wrong, it is 6.5x slower with no "
        f"symptom other than 'slow today' -- both dirs hold the same filenames with the same "
        f"stamps.\n"
        f"  export AUPAI_TOKEN_CACHE_DIR={NVME_CACHE_DIR}\n"
        f"If the overlay copy is deliberate (comparing them, or the NVMe copy is suspect), set "
        f"AUPAI_TOKEN_CACHE_DIR to it explicitly -- naming it is the difference between a choice "
        f"and a forgotten variable, and this refusal only fires on the default.")


def assert_caches_fresh(domains, root=ROOT):
    """Raise unless every domain's cache exists and _domain_seqs would reuse it as-is.

    The same conditions _domain_seqs' `fresh` ANDs together, read from train.py's own
    helpers. SIX, not five: cache exists, shards exist, same_vocab, same_source, same_seed,
    and the cache's mtime is at least the newest shard's. (This said five until 2026-09-07 --
    a count beside a line number, both of them wrong.) Reports every domain that fails, not
    the first: an eval run whose mix has
    two stale domains should learn both in one line rather than one per rerun.

    Also the co-residency refusal, BEFORE the freshness loop: both answer "may this
    process read these caches now", and putting them in one function is what makes the
    second one reach every caller (the freshness guard's own docstring reason -- a guard
    each caller has to remember is a guard one of them eventually will not).

    And the overlay refusal, for the same reason and in the same place: every cache reader
    already passes through here, so the question "am I about to read the slow copy" is asked
    once rather than in each tool.
    """
    import train

    assert_not_co_resident(domains, root=root)
    assert_cache_dir_not_overlay(root=root)
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
    n_ov = _selftest_overlay()
    print("selftest OK: refuses a missing vocab_id, a missing cache, a one-byte stamp "
          f"difference and a 0-byte stamp; accepts a fully-stamped fresh cache; "
          f"co-residency refusal correct on {n_co} worlds; overlay refusal correct on "
          f"{n_ov} worlds")
    return 0


def _selftest_overlay():
    """The overlay refusal, on a fake NVMe dir and a fake cache dir.

    NVME_CACHE_DIR is rebound to a temp path, so the worlds do not depend on the pod's real mount --
    a test that only runs where the mount exists cannot gate a commit, which is the point of having
    it. `same device as /` is the predicate, and a temp dir on a laptop IS on the root filesystem, so
    the overlay world is real rather than stubbed.
    """
    import shutil
    import tempfile

    global NVME_CACHE_DIR
    real_nvme = NVME_CACHE_DIR
    import train

    old_cache = train.TOKEN_CACHE
    old_env = os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
    d = tempfile.mkdtemp(prefix="overlay_guard_")
    n = 0
    try:
        fake_nvme = os.path.join(d, "nvme", "tokens")
        overlay = os.path.join(d, "overlay")
        os.makedirs(fake_nvme)
        os.makedirs(overlay)

        # 1. NO NVMe DIR: nothing to prefer, so it must not fire. This is CI and every laptop, and
        #    a refusal here would be a permanent red -- the check would be turned off by week's end.
        NVME_CACHE_DIR = os.path.join(d, "does_not_exist")
        train.TOKEN_CACHE = os.path.join(overlay, "pretrain_1b_tokens.pt")  # cache-path-ok: fixture world
        assert assert_cache_dir_not_overlay() is None, (
            "it fired with no NVMe copy present: a refusal whose remedy does not exist on this host")
        n += 1

        # 2. NVMe DIR EXISTS and the cache dir is on the root filesystem: refuse, naming the
        #    variable. A temp dir is genuinely on / here, so `st_dev == stat('/').st_dev` is the
        #    real predicate and not a stub.
        NVME_CACHE_DIR = fake_nvme
        try:
            assert_cache_dir_not_overlay()
            raise AssertionError(
                "the overlay cache dir was accepted while the NVMe copy existed -- a run would "
                "read the 6.5x slower copy with no symptom but 'slow today'")
        except CacheWouldRebuild as e:
            for want in ("AUPAI_TOKEN_CACHE_DIR", fake_nvme, "6.5x"):
                assert want in str(e), (
                    f"the refusal does not name {want!r}, so a reader cannot act on it: {e}")
        n += 1

        # 3. THE CACHE DIR IS THE NVMe DIR: accept. Without this the case above would pass for a
        #    predicate that refuses everything.
        train.TOKEN_CACHE = os.path.join(fake_nvme, "pretrain_1b_tokens.pt")  # cache-path-ok: fixture world
        assert assert_cache_dir_not_overlay() == fake_nvme, "it refused the NVMe dir itself"
        n += 1

        # 4. AN EXPLICIT AUPAI_TOKEN_CACHE_DIR IS A CHOICE, not a forgotten variable, even when it
        #    names the overlay. Telling someone to set a variable they have already set is how a
        #    check gets disabled.
        os.environ["AUPAI_TOKEN_CACHE_DIR"] = overlay
        assert assert_cache_dir_not_overlay() == overlay, (
            "an explicitly chosen overlay dir was refused, and the message would have told the "
            "caller to set the variable they just set")
        n += 1
        del os.environ["AUPAI_TOKEN_CACHE_DIR"]

        # 5. THE CHOKEPOINT ACTUALLY ASKS. Worlds 1-4 call the predicate directly, so all four stay
        #    green if the call is deleted from assert_caches_fresh -- a working guard nothing
        #    invokes. Reached with a domain that has no cache: the freshness loop would refuse it
        #    too, so the assertion is that THIS refusal comes first, identified by its own text.
        train.TOKEN_CACHE = os.path.join(overlay, "pretrain_1b_tokens.pt")  # cache-path-ok: fixture world
        try:
            assert_caches_fresh(["a_domain_with_no_cache"])
            raise AssertionError("assert_caches_fresh accepted an overlay cache dir")
        except CacheWouldRebuild as e:
            assert "ROOT FILESYSTEM" in str(e), (
                f"assert_caches_fresh refused for a DIFFERENT reason, so the overlay call is not "
                f"on its path: {str(e)[:160]}")
        n += 1
    finally:
        NVME_CACHE_DIR = real_nvme
        train.TOKEN_CACHE = old_cache
        if old_env is not None:
            os.environ["AUPAI_TOKEN_CACHE_DIR"] = old_env
        else:
            os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
        shutil.rmtree(d, ignore_errors=True)
    return n


def _selftest_co_resident():
    """The co-residency refusal, on a claim dir this test writes.

    Every world drives the decision through a REAL claim file read by card_claim.claims(),
    not by stubbing _live_run_cards: the thing under test is whether a claim on disk reaches
    the refusal, and a stub asserts my belief about the reader instead of the reader.

    The claim must name a LIVE pid, so a real subprocess is spawned and the claim names IT.
    A dead pid is filed stale and the guard would pass for the right reason on the wrong world
    -- which is the mistake this docstring exists to stop the next reader repeating.

    A FOREIGN live pid, not os.getpid(), since 2026-09-06. The guard now excludes a claim held
    by this process or an ancestor of it -- score_matrix claims its own cards at startup and
    then read that claim back here and refused its own cache read (4c's #46) -- so every world
    below that names this pid would test the exclusion instead of the refusal, and worlds 2-6
    went red the moment the exclusion landed. The subprocess is a `sleep`: alive, unrelated,
    and reaped in the finally.
    """
    import json
    import shutil
    import subprocess
    import sys
    import tempfile

    elc = _elc()
    d = tempfile.mkdtemp(prefix="coresident_")
    old_dir = os.environ.get("AUPAI_CLAIM_DIR")
    old_allow = os.environ.pop("AUPAI_ALLOW_CORESIDENT_CACHE", None)
    # A live pid that is NOT this process and NOT an ancestor of it. Spawned rather than
    # borrowed from the process table: a pid found by scanning could be an ancestor, and the
    # world would then silently become the exclusion case again.
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
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
            json.dump({"name": "p500m_selftest", "cards": ["0", "1"], "pid": foreign.pid,
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

        # 8 + 9. THE SELF AND ANCESTOR CASES: a claim held by THIS process, or by an ancestor
        #    of it, must not refuse this process's own read (4c's #46). Every GPU entry point
        #    claims its own cards at startup, so score_matrix wrote a claim, read it back
        #    here, and refused itself -- and the refusal exited 0, so three rows landed in
        #    runs/score_matrix.jsonl with every metric an error dict and nothing said they
        #    were not scores. The ancestor case is `harness launch`, which claims for the
        #    job's ancestor rather than for the job.
        #
        #    Note the pairing with world 2: this world and that one differ ONLY in whose pid
        #    the claim names, so together they say the guard keys on WHO holds the claim and
        #    not on whether one exists. Neither is meaningful alone -- world 2 alone passes
        #    for a guard that refuses everything, and these alone pass for one that refuses
        #    nothing.
        for label, pid in (("self", os.getpid()), ("ancestor", os.getppid())):
            own = os.path.join(d, f"own_{label}")
            os.makedirs(own)
            with open(os.path.join(own, "score_matrix.json"), "w", encoding="utf-8") as fh:
                json.dump({"name": f"score_matrix_{label}", "cards": ["0", "1"], "pid": pid,
                           "cmdline": "python eval/score_matrix.py --ckpt x.pt",
                           "acquired": "2026-09-06 00:00:00"}, fh)
            os.environ["AUPAI_CLAIM_DIR"] = own
            assert _must_pass(
                big,
                f"a claim held by this process's {label} pid ({pid}) refused this process's "
                f"own cache read -- that is the self-deadlock: the job claims its cards, then "
                f"refuses itself, and the refusal used to exit 0 and write a row that looked "
                f"like a score") == 0
            n += 1

        # 10. THE MIXED DIR, which is what production actually looks like (e1's finding 2 on
        #     91531b49): score_matrix claims its own cards WHILE the lane job holds its own.
        #     Every world above points AUPAI_CLAIM_DIR at a directory holding exactly one
        #     claim, so none of them distinguishes "filter the self claim out of the set" from
        #     "if any claim is mine, disable the guard entirely" -- and that second mutant
        #     SURVIVED the whole selftest. The shipped code was correct; the world was missing.
        mixed = os.path.join(d, "mixed")
        os.makedirs(mixed)
        with open(os.path.join(mixed, "score_matrix.json"), "w", encoding="utf-8") as fh:
            json.dump({"name": "score_matrix_mine", "cards": ["0"], "pid": os.getpid(),
                       "cmdline": "python eval/score_matrix.py --ckpt x.pt",
                       "acquired": "2026-09-06 00:00:00"}, fh)
        with open(os.path.join(mixed, "lane_job.json"), "w", encoding="utf-8") as fh:
            json.dump({"name": "b0_moe48", "cards": ["1"], "pid": foreign.pid,
                       "cmdline": "python train.py --name b0_moe48",
                       "acquired": "2026-09-06 00:00:00"}, fh)
        os.environ["AUPAI_CLAIM_DIR"] = mixed
        try:
            assert_not_co_resident(big)
            raise AssertionError(
                "a big read was allowed with MY claim on card 0 and a foreign live claim on "
                "card 1: the exclusion dropped the whole set rather than my own claim, so any "
                "job that claims its own cards disables the guard for everyone")
        except CoResidentCacheRead as e:
            assert "b0_moe48" in str(e), (
                f"the refusal does not name the foreign claimant: {str(e)[:200]}")
            assert "score_matrix_mine" not in str(e), (
                f"the refusal names MY OWN claim as a reason to refuse me: {str(e)[:200]}")
            # Card 1 is the foreign one; card 0 is mine and must not appear as held.
            _cards = str(e).split("while cards ", 1)[1].split(" ", 1)[0] if "while cards " in str(e) else ""
            assert _cards == "1", (
                f"the refusal reports cards {_cards!r}, expected just the foreign card 1 -- "
                f"reporting card 0 means my own claim survived the filter")
        n += 1

        # 11-13. _own_pids WITH ps UNAVAILABLE, WITH A DEEP CHAIN, AND WITH A CYCLIC TABLE.
        #     None of those is reachable from a claim-dir world: 8-10 run where `ps` works, so
        #     the `if not table` fallback and the MAX_DEPTH cap could be deleted or inverted
        #     unnoticed. e1 reported the fallback as mutation-RED on 2026-09-06 and then
        #     RETRACTED it (review row 073f725b): both shapes SURVIVE, because the ancestor
        #     world's claim names getppid() and the fallback supplies exactly that. A coverage
        #     gap that a measurement was briefly claimed for is a reason to write the world.
        #
        #     UNIT-LEVEL, and stated as such: _own_pids is called with a stub rather than
        #     through _live_run_cards, which loads the real card_claim by path and cannot be
        #     stubbed from here. What this cannot see is whether _live_run_cards still calls
        #     _own_pids at all -- worlds 8-10 are what say that.
        class _NoPs:
            MAX_DEPTH = 6

            @staticmethod
            def _ps_table():
                return []

        got = _own_pids(_NoPs)
        assert got == {os.getpid(), os.getppid()}, (
            f"with ps unavailable _own_pids returned {sorted(got)}, expected this pid and its "
            f"parent. Dropping the parent deadlocks every job launched through `harness "
            f"launch`, which claims for the job's ancestor; returning more would exclude a "
            f"foreign claim and disable the guard")
        n += 1

        class _Chain:
            MAX_DEPTH = 6

            @staticmethod
            def _ps_table():
                # me <- 10 <- 11 <- init. Real pid, fabricated ancestry.
                return [(os.getpid(), 10, "python"), (10, 11, "bash"), (11, 0, "init")]

        got = _own_pids(_Chain)
        assert got == {os.getpid(), 10, 11}, (
            f"_own_pids walked to {sorted(got)}, expected the whole chain. A walk that stops "
            f"at the parent treats a grandparent's claim as foreign, and the pod's tree is "
            f"wrapper bash -> torchrun -> rank, so the claiming ancestor is two levels up")
        n += 1

        class _Cycle:
            MAX_DEPTH = 6

            @staticmethod
            def _ps_table():
                # A ppid cycle: a real table should never hold one, a corrupt read can.
                return [(os.getpid(), 20, "python"), (20, 21, "bash"), (21, 20, "bash")]

        got = _own_pids(_Cycle)
        assert got == {os.getpid(), 20, 21}, (
            f"_own_pids returned {sorted(got)} on a CYCLIC ppid table. It must terminate and "
            f"return what it reached -- without the depth cap and the `nxt in out` test this "
            f"call never returns and the guard hangs instead of refusing")
        n += 1
    finally:
        # The foreign process first: a leaked `sleep 300` from a failed selftest would sit in
        # the table for five minutes, and this file's own worlds are the ones that read it.
        try:
            foreign.terminate()
            foreign.wait(timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
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
