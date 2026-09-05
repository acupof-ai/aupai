#!/usr/bin/env python3
"""A mix domain with `val_frac: 0` hands its whole pool to training.

    python3 scripts/test_mix_val_frac.py

WHY THIS EXISTS. train.py:2155 took `n_val = min(max(1, int(len(seqs) * Cfg.val_frac)), ...)`
off the FRONT of every domain, injection shards included. Experiment 1
(runs/prereg.jsonl#conversion_rate_0905) injects the same 1,000 documents n times and reads a
curve against n, so a 5% hold-back makes the realised exposure count 0.95n while the axis says
n. Measured on the built arms before launch: n64 wanted 1,625 rows and could draw 1,542; n1
wanted 25 and could draw 24, i.e. ~40 of the 1,000 documents would have had ZERO exposures.

WHAT THIS TEST HAS TO DO THAT AN EASIER ONE DOES NOT. `max(1, ...)` is why an override needs a
branch: `val_frac: 0` inside the old expression floors back up to 1 row, so a test that only
checks "n_val got smaller" passes on the defect. The assertion is therefore EXACT -- pool length
equals the cache's row count, and the count of held-back rows is 0, not 'few'. And the numbers
come from the fixture's own construction (rows written / rows_per_doc), never from build_mix,
because a baseline computed by the code under test cannot disagree with it.

World 4 is the one with teeth: it asserts a natural domain
in the SAME mix still holds back its 5%. A change that simply removed the val split entirely
would pass every other world here and silently stop validating nine domains.
"""

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import train  # noqa: E402

FAILS = []
SEQ = 32  # tiny: this test is about the split arithmetic, not the tokenizer


class FakeTok:
    """Encodes each character as one id, so a document's token count is its length."""

    def encode(self, s):
        class R:
            ids = [1] * len(s)
        return R()


def _write_cache(tmp, domain, n_rows):
    """Write a token cache of exactly n_rows rows and the stamps _domain_seqs checks.

    Bypasses tokenization on purpose: the subject is the val split, and building the cache
    directly is what lets the expected row count be a FIXTURE FACT (n_rows, chosen here)
    rather than something read back from the function under test.

    AUPAI_TOKEN_CACHE_DIR, set by the caller, is what keeps this out of the shared cache
    directory. _token_cache_dir() reads that variable first; without it the path resolves to
    /data00 (read-only here, and on the pod the LIVE RUN'S cache dir -- the exact accident
    test_domain_loss_val had on 2026-09-02, writing a real cache beside a running job's).
    """
    cache = train._domain_cache_path(domain)
    assert cache.startswith(tmp), f"cache path {cache} escaped the fixture dir {tmp}"
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    # A one-shard corpus dir so the cache reads as FRESH. _domain_seqs compares the .srcfp
    # stamp against _corpus_fp(data/corpus/<domain>) and raises FileNotFoundError when the
    # directory is absent, so the fixture cannot skip it -- and the stamp has to be the
    # fingerprint of what is really there, not a literal, or the cache reads as stale and the
    # test measures a rebuild instead of the val split. train.DATA is repointed at the
    # fixture dir by the caller so nothing is written under the repo's data/.
    cdir = os.path.join(train.DATA, "corpus", domain)
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, f"{domain}_000.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "x" * 8}) + "\n")
    # FLAT, not pre-shaped. The real cache is a 1-D token stream that _domain_seqs reshapes
    # with `n = len(data) // (seq+1)` (train.py:1957). A pre-shaped [n_rows, seq+1] tensor makes
    # len(data) the ROW count, so n came out 200//33 = 6 and the function returned 198 rows of a
    # 200-row fixture -- a fixture that silently measured a different quantity than it claimed.
    torch.save(torch.arange(n_rows * (SEQ + 1), dtype=torch.int32), cache)
    for suffix, val in ((".vocab", train.VOCAB_ID),
                        (".srcfp", train._corpus_fp(cdir)),
                        (".seed", str(train._sample_seed()))):
        with open(cache + suffix, "w", encoding="utf-8") as fh:
            fh.write(str(val))
    return cache


def _mix(tmp, domains, total_tokens):
    p = os.path.join(tmp, "mix.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"total_tokens": total_tokens, "seq": SEQ, "domains": domains}, fh)
    return p


def _nval(val):
    """Held-back row count. build_mix CONCATENATES the per-domain val lists into one tensor, so
    iterating it yields ROWS and `sum(v.shape[0] for v in val)` counts the row WIDTH (33) once per
    row -- it read 330 for a 10-row split and 1650 for a 50-row one. Measured, not assumed."""
    return 0 if val is None or len(val) == 0 else int(val.shape[0])


def _run(mix_path):
    """build_mix on one rank, returning (plan_rows, val_rows, per-domain row counts)."""
    mine, val = train.build_mix(mix_path, FakeTok(), True, False, rank=0, world=1)
    return mine, val, dict(train.Cfg._row_cursor)


def main():
    tmp = tempfile.mkdtemp(prefix="mixvf_")
    caches = []
    old_seq, old_vf, old_af = train.Cfg.seq, train.Cfg.val_frac, train.Cfg.anneal_frac
    old_data, old_vocab = train.DATA, train.VOCAB_ID
    old_cachedir = os.environ.get("AUPAI_TOKEN_CACHE_DIR")
    os.environ["AUPAI_TOKEN_CACHE_DIR"] = tmp
    train.DATA = os.path.join(tmp, "data")
    train.Cfg.seq, train.Cfg.val_frac, train.Cfg.anneal_frac = SEQ, 0.05, 0.0
    train.VOCAB_ID = train.VOCAB_ID or "fixturevocab"
    try:
        # 1. val_frac 0 HOLDS BACK NOTHING, EXACTLY. 200 rows in the cache, 200 in the pool.
        #    The old code gave 190; the old code with a 0 override gave 199, because max(1, 0)
        #    is 1 -- which is why "fewer held back" is not the assertion.
        caches.append(_write_cache(tmp, "inj", 200))
        m = _mix(tmp, {"inj": {"weight": 1.0, "epochs": 1, "val_frac": 0}}, 200 * SEQ)
        mine, val, used = _run(m)
        n_val = _nval(val)
        if n_val != 0:
            FAILS.append(f"val_frac 0 still held back {n_val} row(s); it must hold back none")
        if used.get("inj") != 200:
            FAILS.append(f"val_frac 0 planned {used.get('inj')} rows of a 200-row cache, "
                         f"want 200 -- the pool is short, so a 200-row budget cannot be met")

        # 2. THE PLAN ACTUALLY CONTAINS THEM. used[] is build_mix's own bookkeeping; the plan
        #    is what the loop reads. A defect that counted rows it did not schedule would pass 1.
        if mine.shape[0] != 200:
            FAILS.append(f"the plan holds {mine.shape[0]} rows, not the 200 used[] claims")

        # 3. EVERY DOCUMENT EXACTLY ONCE, which is the property experiment 1's axis rests on.
        #    Row i of the fixture cache starts at token i*(SEQ+1), so the first column is a
        #    unique row id -- an independent handle on identity, not the plan's own index.
        ids = mine[:, 0].tolist()
        if len(set(ids)) != 200 or sorted(ids) != sorted({i * (SEQ + 1) for i in range(200)}):
            FAILS.append(f"the 200 planned rows are not the 200 distinct cache rows: "
                         f"{len(set(ids))} distinct")

        # 4. val_frac 0 IS NOT THE NEW DEFAULT. Same mix, a second domain without the key: it
        #    must still hold back 5%. Without this world, deleting the val split outright
        #    passes 1-3 and silently stops validating every natural domain.
        caches.append(_write_cache(tmp, "nat", 200))
        m = _mix(tmp, {"inj": {"weight": 0.5, "epochs": 1, "val_frac": 0},
                       "nat": {"weight": 0.5, "epochs": 1}}, 200 * SEQ)
        mine, val, used = _run(m)
        # EXACTLY 10, which is 5% of nat's 200 and 0% of inj's: the total pins both halves at
        # once. 20 would mean the override was ignored, 0 that it leaked into the default.
        held = _nval(val)
        if held != 10:
            FAILS.append(f"two domains, one with val_frac 0 and one without, held back {held} "
                         f"row(s); want exactly 10 = 5% of nat's 200 + 0% of inj's 200. 20 means "
                         f"the override was ignored, 0 that it leaked into the default")
        # NO WORLD HERE FOR "val_frac 0 must not append an EMPTY TENSOR to the val list", and the
        # absence is deliberate rather than an omission. An `if False:` block asserting exactly
        # that survived here from an earlier draft, left behind when the per-domain val read was
        # replaced by _nval(): it guarded nothing and its message described list semantics the test
        # no longer uses (44, 2026-09-05). The property is NOT OBSERVABLE from here -- build_mix
        # concatenates the per-domain val lists before returning, and torch.cat and len are no-ops
        # on a 0-row tensor, so appending one and appending nothing produce the same return value.
        # Testing it would need build_mix to expose the per-domain list, which is a change to the
        # subject to suit the test. Deleted rather than repaired.

        # 5. AN EXPLICIT NON-ZERO OVERRIDE IS HONOURED, so the key is a value and not a flag.
        m = _mix(tmp, {"inj": {"weight": 1.0, "epochs": 1, "val_frac": 0.25}}, 200 * SEQ)
        mine, val, used = _run(m)
        if _nval(val) != 50:
            FAILS.append(f"val_frac 0.25 of 200 rows held back {_nval(val)}, want 50")

        # 6. THE DEFECT IS REPRODUCED, so a green run here means the fix is present rather than
        #    that the world is easy. Same cache, no override: 10 rows go to val and the 200-row
        #    budget CANNOT be met from a 190-row pool. That shortfall is exactly what capped
        #    every experiment-1 arm.
        m = _mix(tmp, {"inj": {"weight": 1.0, "epochs": 1}}, 200 * SEQ)
        mine, val, used = _run(m)
        if _nval(val) != 10 or used.get("inj") != 190:
            FAILS.append(f"without the override the split changed: val "
                         f"{_nval(val)} (want 10), planned "
                         f"{used.get('inj')} (want 190, the capped shortfall this fix exists "
                         f"for). If this world stops holding 10 back, the fix removed the "
                         f"default split instead of overriding it per domain.")
        # 7. A GLOBAL Cfg.val_frac OF 0.0 STILL HOLDS BACK ONE ROW for a domain with no override.
        #    The override keys on the KEY's presence, not on the value being zero, and this is the
        #    world that forces that: keying on the value makes `Cfg.val_frac = 0.0` hold back 0
        #    instead of the 1 that max(1, ...) has always produced, which shifts every pool by one
        #    row. Found by scripts/test_plan_length.py, whose selftest sets exactly this and whose
        #    fresh-plan content hash moved at an identical row count. Added here because relying on
        #    another file's test to catch a regression in this line is the coverage gap, not the
        #    fix: this test is what train.py's trigger list runs for this change.
        train.Cfg.val_frac = 0.0
        try:
            m = _mix(tmp, {"nat": {"weight": 1.0, "epochs": 1}}, 200 * SEQ)
            mine, val, used = _run(m)
            if _nval(val) != 1:
                FAILS.append(f"with Cfg.val_frac 0.0 globally and NO per-domain key, the split held "
                             f"back {_nval(val)} row(s); want 1 -- max(1, ...) has always floored a "
                             f"global zero up to one row, and changing that moves every pool by a "
                             f"row (test_plan_length's fresh-plan hash)")
            # And the override still wins over a global 0.0, so the two mechanisms are independent.
            m = _mix(tmp, {"nat": {"weight": 1.0, "epochs": 1, "val_frac": 0}}, 200 * SEQ)
            mine, val, used = _run(m)
            if _nval(val) != 0:
                FAILS.append(f"with Cfg.val_frac 0.0 globally AND val_frac: 0 on the domain, the "
                             f"split held back {_nval(val)} row(s); want 0")
        finally:
            train.Cfg.val_frac = 0.05
    finally:
        train.Cfg.seq, train.Cfg.val_frac, train.Cfg.anneal_frac = old_seq, old_vf, old_af
        train.DATA, train.VOCAB_ID = old_data, old_vocab
        # RESTORED, not just saved. ruff caught this as an unused variable and it is a real leak:
        # without it the fixture's temp dir stays in AUPAI_TOKEN_CACHE_DIR after rmtree, so
        # anything running later in the same process resolves its token caches to a path that no
        # longer exists -- and _token_cache_dir treats a configured directory as a claim the
        # caches are there.
        if old_cachedir is None:
            os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
        else:
            os.environ["AUPAI_TOKEN_CACHE_DIR"] = old_cachedir
        for c in caches:
            for s in ("", ".vocab", ".srcfp", ".seed"):
                if os.path.exists(c + s):
                    os.remove(c + s)
        shutil.rmtree(tmp, ignore_errors=True)

    for f in FAILS:
        print(f"BUG {f}", file=sys.stderr)
    print(f"mix val_frac test: {'PASS (7 worlds)' if not FAILS else f'{len(FAILS)} BUG(S)'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
