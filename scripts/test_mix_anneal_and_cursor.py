#!/usr/bin/env python3
"""build_mix reads the mix's own anneal_frac, and publishes no state across calls.

    python3 scripts/test_mix_anneal_and_cursor.py

TWO DEFECTS, ONE FIXTURE, because both are properties of build_mix's own bookkeeping and both
need a real two-call world to show.

(1) THE MIX'S anneal_frac WAS NEVER READ. build_mix built `phases` from Cfg.anneal_frac, so a
mix declaring "anneal_frac": 0.0 silently got the 0.10 default and a two-phase schedule.
MEASURED across data/mix_*.json: 13 of 23 declare the key and every one of the 13 declares a
value that differs from Cfg's 0.10. The cost is not only the phase boundary -- `want = int(rows
* frac * weight)` runs once per phase and int(0.9x) + int(0.1x) <= int(x), so the spurious
second phase floors one row off every domain. e1's injection arms measured it: n1 25->24, n8
204->203, n64 1639->1638, n256 6557->6556, which on an axis whose ROW COUNT IS THE MEASUREMENT
is ~39 lost document exposures per arm.

The assertion is EXACT -- a declared 0.0 yields exactly the one-phase row count -- because "one
row fewer" is the whole defect and any test phrased as "about right" passes on it. The two row
counts are computed HERE from the fixture's own pool sizes, never read back from build_mix: a
baseline the code under test computes cannot disagree with it.

(2) _row_cursor_base LEAKED BETWEEN CALLS. Every other field build_mix publishes on Cfg is a
wholesale assignment; _row_cursor_base alone was MERGED into whatever dict Cfg already held, so
a second build_mix in one process inherited the first call's domains. Two calls in one process
is the normal case for a tool, not a corner -- and scripts/test_plan_length.py already resets
`Cfg._row_cursor_base = None` by hand before each build for exactly this reason. A fixture
working around a defect is evidence of the defect.

WHAT MAKES WORLD 3 THE ONE WITH TEETH: it asserts the refusal, i.e. that a mix declaring a
value Cfg contradicts does NOT quietly pick a side. Cfg carries the same value whether
--anneal_frac was passed or not, so nothing inside build_mix can tell an explicit flag from the
class default; picking either would be a guess that changes what the run trains on. Without
this world, "read the mix key" and "always prefer the mix key" pass identically.
"""

import contextlib
import io
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
SEQ = 32


class FakeTok:
    def encode(self, s):
        class R:
            ids = [1] * len(s)
        return R()


def _write_cache(tmp, domain, n_rows):
    """A token cache of exactly n_rows rows, plus the three stamps _domain_seqs checks.

    Flat, not pre-shaped: _domain_seqs reshapes a 1-D stream with n = len(data) // (seq+1), so
    a pre-shaped [n, seq+1] tensor makes len(data) the ROW count and the fixture silently
    measures a different quantity (test_mix_val_frac hit exactly that)."""
    cache = train._domain_cache_path(domain)
    assert cache.startswith(tmp), f"cache path {cache} escaped the fixture dir {tmp}"
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    cdir = os.path.join(train.DATA, "corpus", domain)
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, f"{domain}_000.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"content": "x" * 8}) + "\n")
    torch.save(torch.arange(n_rows * (SEQ + 1), dtype=torch.int32), cache)
    for suffix, val in ((".vocab", train.VOCAB_ID),
                        (".srcfp", train._corpus_fp(cdir)),
                        (".seed", str(train._sample_seed()))):
        with open(cache + suffix, "w", encoding="utf-8") as fh:
            fh.write(str(val))
    return cache


def _mix(tmp, name, domains, total_tokens, anneal_frac=None):
    p = os.path.join(tmp, name)
    obj = {"total_tokens": total_tokens, "seq": SEQ, "domains": domains}
    if anneal_frac is not None:
        obj["anneal_frac"] = anneal_frac
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return p


def _build(mix_path, **kw):
    """build_mix with its chatter captured. Returns (rows tensor, published cursor base, log).

    The rows tensor is [rows, seq+1], so the count is shape[0]. My first version read shape[1]
    and got 33 = SEQ+1 for every world -- a constant, so all three row assertions compared the
    same number and world 2's strictly-fewer check reported the fixture as broken rather than
    the code. build_mix's INTERNAL plan is [2, rows]; what it RETURNS is transposed from that."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mine, _val = train.build_mix(mix_path, FakeTok(), True, False, rank=0, world=1, **kw)
    return mine, dict(getattr(train.Cfg, "_row_cursor_base", None) or {}), buf.getvalue()


def _want_rows(pool_rows, weight, budget_rows, fracs):
    """The row count build_mix's own arithmetic yields, phase by phase.

    Computed from the FIXTURE's numbers so it is an independent prediction. One phase is
    int(budget * 1.0 * weight); two phases are int(budget * .9 * w) + int(budget * .1 * w),
    and the FLOORING is why they differ.

    THE POOL MUST NOT BE THE BINDING CONSTRAINT, and the first version of this fixture let it
    be: POOL 400 with a 400-row budget made both worlds return 399, because phase 1 took 360
    and phase 2 could only draw the 39 rows left -- the pool ceiling, not int(). The two counts
    agreed for a reason that has nothing to do with anneal_frac, so world 2's strictly-fewer
    check reported the FIXTURE as broken, which is the one honest thing about that run. The
    caller now keeps budget_rows well under pool_rows and this asserts it rather than trusting
    it."""
    assert budget_rows < pool_rows, (
        f"fixture error: budget {budget_rows} >= pool {pool_rows}, so the pool ceiling binds "
        f"and the flooring this test measures is invisible")
    total = 0
    for frac in fracs:
        total += int(budget_rows * frac * weight)
    return total


def main():
    tmp = tempfile.mkdtemp(prefix="mixaf_")
    old = {k: getattr(train.Cfg, k) for k in ("seq", "val_frac", "anneal_frac", "val_rows_max")}
    old_data, old_vocab = train.DATA, train.VOCAB_ID
    old_cachedir = os.environ.get("AUPAI_TOKEN_CACHE_DIR")
    os.environ["AUPAI_TOKEN_CACHE_DIR"] = tmp
    train.DATA = os.path.join(tmp, "data")
    train.Cfg.seq, train.Cfg.val_frac, train.Cfg.val_rows_max = SEQ, 0.0, 3
    train.VOCAB_ID = train.VOCAB_ID or "fixturevocab"
    try:
        POOL = 400
        _write_cache(tmp, "inj", POOL)
        # val_frac 0.0 with no per-domain key still holds back max(1, 0) = 1 row, which is
        # train.py's documented behaviour and not this test's subject.
        pool_rows = POOL - 1
        # A BUDGET WELL UNDER THE POOL, so int() is what limits each phase rather than the rows
        # available. At budget == pool the two-phase world drew 360 + 39 = 399 and the one-phase
        # world drew 399 too, and the flooring -- the entire defect -- was invisible. 305 is not
        # a multiple of 10, so int(305*0.9) + int(305*0.1) = 274 + 30 = 304 < 305 and the lost
        # row is exactly the one e1 measured in every injection arm.
        budget_rows = 305
        budget = budget_rows * SEQ

        # 1. THE MIX DECLARES 0.0 AND Cfg AGREES: exactly the one-phase row count.
        train.Cfg.anneal_frac = 0.0
        m = _mix(tmp, "one_phase.json", {"inj": {"weight": 1.0, "epochs": 1, "anneal": 1.0}},
                 budget, anneal_frac=0.0)
        rows_1, _base, _log = _build(m)
        want_1 = _want_rows(pool_rows, 1.0, budget_rows, (1.0,))
        got_1 = int(rows_1.shape[0])
        if got_1 != want_1:
            FAILS.append(f"declared anneal_frac 0.0 gave {got_1} rows, one-phase arithmetic "
                         f"says {want_1} -- the plan was not built at the mix's value")

        # 2. THE SAME MIX WITHOUT THE KEY, AT Cfg 0.10: two phases, and STRICTLY FEWER rows.
        #    This is the pair that makes world 1 mean something: if the two counts were equal
        #    the assertion above would hold no matter which value was used.
        train.Cfg.anneal_frac = 0.10
        m2 = _mix(tmp, "two_phase.json", {"inj": {"weight": 1.0, "epochs": 1, "anneal": 1.0}},
                  budget)
        rows_2, _base2, _log2 = _build(m2)
        want_2 = _want_rows(pool_rows, 1.0, budget_rows, (0.9, 0.1))
        got_2 = int(rows_2.shape[0])
        if got_2 != want_2:
            FAILS.append(f"absent anneal_frac gave {got_2} rows, Cfg-0.10 two-phase arithmetic "
                         f"says {want_2} -- an absent key must fall back to Cfg, not to 0")
        if not got_2 < got_1:
            FAILS.append(f"two phases ({got_2}) did not lose rows against one ({got_1}), so this "
                         f"test cannot see the flooring it exists for -- check the fixture, not "
                         f"the code")

        # 3. THE MIX DECLARES 0.0 WHILE Cfg SAYS 0.10: REFUSE, naming both values.
        #    Not a preference either way. Cfg holds the same 0.10 whether the operator passed
        #    --anneal_frac 0.10 or passed nothing, so silently preferring the mix would override
        #    an explicit flag and silently preferring Cfg is the original defect.
        train.Cfg.anneal_frac = 0.10
        m3 = _mix(tmp, "disagree.json", {"inj": {"weight": 1.0, "epochs": 1, "anneal": 1.0}},
                  budget, anneal_frac=0.0)
        _msgs = {}
        try:
            _r, _b, _l = _build(m3)
            FAILS.append("a mix declaring anneal_frac 0.0 against Cfg 0.10 was built anyway -- "
                         "the two schedule different runs and nothing here can tell an explicit "
                         "--anneal_frac from the class default")
        except RuntimeError as e:
            _msgs[(0.0, 0.10)] = str(e)

        # 3b. THE SAME REFUSAL AT VALUES THAT CANNOT COLLIDE WITH THE PROSE. This second world
        #     is what makes world 3's naming assertion real, and it exists because two of the
        #     three tokens in the first version could not fail (e1, MEASURED): "anneal_frac"
        #     appears in the literal "declares anneal_frac", and "0.1" is a substring of the
        #     literal "int(0.1x)" the message explains the flooring with. Deleting ` and
        #     Cfg.anneal_frac is {Cfg.anneal_frac}` -- exactly the failure this world is for --
        #     fired none of the three. The one that worked, "0.0", worked only because float
        #     formatting emits a trailing zero, so it was testing a repr.
        #
        #     Stripping the values out of the message does not fix that: str.replace removes the
        #     prose occurrence too, so the collision disappears from both sides. What separates
        #     prose from an interpolated value is that the PROSE IS CONSTANT ACROSS TWO WORLDS
        #     and the values are not. 0.25 and 0.75 appear nowhere in the message's literal text,
        #     so requiring each message to name its own pair is an assertion neither side of
        #     which the author's prose can satisfy.
        train.Cfg.anneal_frac = 0.75
        m3b = _mix(tmp, "disagree2.json", {"inj": {"weight": 1.0, "epochs": 1, "anneal": 1.0}},
                   budget, anneal_frac=0.25)
        try:
            _r, _b, _l = _build(m3b)
            FAILS.append("a mix declaring anneal_frac 0.25 against Cfg 0.75 was built anyway")
        except RuntimeError as e:
            _msgs[(0.25, 0.75)] = str(e)

        for (_dec, _cfg), _msg in _msgs.items():
            for _label, _v in (("the mix's declared value", _dec), ("Cfg's value", _cfg)):
                if str(_v) not in _msg:
                    FAILS.append(f"the refusal does not state {_label} ({_v}): {_msg[:130]}")
            if "anneal_frac" not in _msg:
                FAILS.append(f"the refusal does not name the field: {_msg[:130]}")
        # AND THE TWO MESSAGES MUST DIFFER. If they were identical the loop above would be
        # satisfied by prose containing every digit it happens to need, which is the defect one
        # level up. Different inputs producing one message is the shape, not the values.
        if len(_msgs) == 2 and len(set(_msgs.values())) == 1:
            FAILS.append("both refusals produced the SAME message, so it cannot be reporting "
                         "the values it was given")

        # 4. THE CURSOR BASE DOES NOT SURVIVE A SECOND CALL.
        #    Call A resumes domain 'inj' at row 100 and publishes a base for it. Call B is a
        #    FRESH build of a mix that does not name 'inj' at all, so a correct base is {} --
        #    under the old merge it still held {'inj': 100}, and save_checkpoint would add rows
        #    to a cursor for a domain this plan never scheduled.
        train.Cfg.anneal_frac = 0.0
        _write_cache(tmp, "other", POOL)
        srcfp = train._corpus_fp(os.path.join(train.DATA, "corpus", "inj"))
        mA = _mix(tmp, "cursorA.json", {"inj": {"weight": 1.0, "epochs": 1, "anneal": 1.0}},
                  budget, anneal_frac=0.0)
        _rA, baseA, _lA = _build(mA, row_cursor={"inj": 100}, cursor_srcfp={"inj": srcfp})
        if baseA.get("inj") != 100:
            FAILS.append(f"call A did not publish the cursor base it applied: {baseA}")
        mB = _mix(tmp, "cursorB.json", {"other": {"weight": 1.0, "epochs": 1, "anneal": 1.0}},
                  budget, anneal_frac=0.0)
        _rB, baseB, _lB = _build(mB)
        if baseB:
            FAILS.append(f"a fresh build_mix inherited the previous call's cursor base "
                         f"({baseB}) -- every published field must be per-call, and a stale "
                         f"base reads identically to an empty one at save time")
    finally:
        for k, v in old.items():
            setattr(train.Cfg, k, v)
        train.DATA, train.VOCAB_ID = old_data, old_vocab
        if old_cachedir is None:
            os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
        else:
            os.environ["AUPAI_TOKEN_CACHE_DIR"] = old_cachedir
        shutil.rmtree(tmp, ignore_errors=True)

    if FAILS:
        for f in FAILS:
            print("FAIL:", f)
        return 1
    print("ok  build_mix reads the mix's anneal_frac, refuses a disagreement, and publishes "
          "no cursor base across calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
