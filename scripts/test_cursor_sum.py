#!/usr/bin/env python3
"""save_checkpoint's cursor-sum identity, against three real checkpoints' numbers.

THE IDENTITY. An ABSOLUTE row cursor sums to exactly the rows the run has consumed:

    sum(row_cursor) == row_cursor_as_of_step x batch x accum x world

ds.second_resume_rereads_one_segment named this as the check that would have caught the
segment-only cursor the day it landed, and it did not exist. 52aec31 fixed the cursor;
this asserts it stays fixed.

THE KNOWN ANSWERS, read off ckpt_p200m_4b_0902 on the pod (rows/step = 16x2x8 = 256):

    as_of  cursor_sum   step x 256   segment-only would write
      500     128,000      128,000                    128,000
      832     212,992      212,992                    212,992
     1192     305,152      305,152                     92,160   <-- the discriminating row

ONLY step1192 DISCRIMINATES, and this is the point of the file. It was written after a
resume (origin 832), so absolute and segment-only disagree there; 305,152 - 212,992 =
92,160 = 360 x 256 is that segment added in whole. The other two sit at origin 0, where
both implementations produce the same number -- the PRE-fix code passes on them. A
known-answer set without a post-resume checkpoint is blind to exactly the defect the
assert guards, which is the blind spot train.py:1000-1005 already names in prose.

The negative case is therefore built at origin 832, not at origin 0.

No card, no corpus, no real checkpoint file: save_checkpoint is called with a one-tensor
state dict and a fabricated plan, and the write goes to a temp dir.

    python3 scripts/test_cursor_sum.py
"""
import os
import sys
import tempfile

import torch

ROOT = os.environ.get("AUPAI_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BATCH, ACCUM, WORLD = 16, 2, 8
ROWS_PER_STEP = BATCH * ACCUM * WORLD
NAMES = ["a", "b", "c"]

# The pod's three, as (as_of_step, cursor_sum, plan_origin). The third is the only one that
# separates an absolute cursor from a segment-only one.
REAL = [(500, 128000, 0), (832, 212992, 0), (1192, 305152, 832)]


class FakeCfg:
    """What save_checkpoint reads off cfg. A class, not a dict: the dict branch of every
    getattr in save_checkpoint returns None by design (the 2026-09-01 rebind bug)."""
    seq, batch, accum, seed, sample_seed = 4096, BATCH, ACCUM, 42, None
    d, layers = 512, 12


def _plan(origin, step, per_domain=None):
    """A _plan_domains vector for the rows THIS segment drew, one rank's stripe.

    save_checkpoint counts a prefix of length (step - origin) * batch * accum and
    multiplies by world, so the vector must hold at least that many entries."""
    n = (step - origin) * BATCH * ACCUM
    if per_domain is None:
        # Round-robin: each domain gets n/len(NAMES) of the rank's rows.
        return torch.tensor([i % len(NAMES) for i in range(n)], dtype=torch.int8)
    out = []
    for di, cnt in enumerate(per_domain):
        out += [di] * cnt
    assert len(out) == n, (len(out), n)
    return torch.tensor(out, dtype=torch.int8)


def _save(tmp, as_of, origin, base, discarded=None, world=WORLD):
    """Call the real save_checkpoint and return (path, loaded_ck, error_or_None)."""
    import train

    cfg = FakeCfg()
    cfg._plan_domains = _plan(origin, as_of)
    cfg._plan_names = list(NAMES)
    cfg._plan_step_origin = origin
    cfg._row_cursor = {n: 0 for n in NAMES}
    cfg._row_cursor_srcfp = {n: "fp" for n in NAMES}
    cfg._row_cursor_base = dict(base)
    cfg._cursor_discarded = list(discarded or [])
    cfg._total_steps = 3814

    p = os.path.join(tmp, f"ck_{as_of}_{origin}_{len(discarded or [])}.pt")
    prev = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = str(world)
    try:
        train.save_checkpoint(p, {"w": torch.zeros(2)}, cfg, "vocab", step=as_of)
    except AssertionError as e:
        return p, None, str(e)
    finally:
        if prev is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = prev
    return p, torch.load(p, map_location="cpu", weights_only=False), None


def _call_sites(src):
    """(lineno, arg-count) for every save_checkpoint call in `src`, by AST.

    A grep cannot do this: `save_checkpoint(` matches its own `def` at train.py:986, so a
    text test stays green on a tree where the call site was deleted (de, 2026-09-03, de-30
    shipped exactly that defect and read 9/9).
    """
    import ast

    return sorted(
        (n.lineno, len(n.args) + len(n.keywords))
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "save_checkpoint"
    )


def _short_sites(src, want=6, expect=4):
    """The predicate: which save_checkpoint calls pass fewer than `want` args, and is one missing."""
    sites = _call_sites(src)
    if len(sites) < expect:
        return sites, f"{len(sites)} call sites, expected {expect}"
    return sites, [ln for ln, n in sites if n < want]


def _check_call_sites(bad):
    """Every save_checkpoint call passes opt and step -- 6 args, not 4.

    The identity above cannot fire on a call that omits `step`: save_checkpoint skips the
    whole row_cursor block when step is None, so the checkpoint is written with no cursor at
    all and every assertion in this file passes vacuously. The run-end save at train.py:2647
    was that call until de-31. Two silent consequences, and neither raises:

      no step  the .pt carries no row_cursor, so a resume restarts every domain at row 0 and
               re-reads the corpus. Resuming had to use .ep1, which is why it survived.
      no opt   ck["opt"] is never set, and resume's `if args.resume and "opt" in ck` then
               skips restoring it without saying so -- a fresh optimizer, Muon momentum and
               Adam second moments gone, which reads as a training anomaly.

    The negative cases run on the real source with the fix textually reverted, in memory. A
    temp copy of train.py cannot be used: it is unimportable outside the repo root, so the
    broken world dies at `import fone` and reports rc=1 for the wrong reason -- a red that
    proves nothing (measured, de-31).
    """
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as fh:
        src = fh.read()
    sites, short = _short_sites(src)
    if short:
        bad.append(f"save_checkpoint at train.py:{short} passes fewer than 6 args, so it "
                   f"omits opt and/or step. A call without `step` writes no row_cursor, and "
                   f"every assertion in this file then passes vacuously on that path")
        return
    print(f"  call sites    : {len(sites)} in train.py, all pass opt+step "
          f"({', '.join(f'{ln}:{n}' for ln, n in sites)})")

    reverted = src.replace(
        "        save_checkpoint(ckpt_path, raw_model.state_dict(), Cfg, VOCAB_ID,\n"
        "                        opt_snapshot(optimizers), step)",
        "        save_checkpoint(ckpt_path, raw_model.state_dict(), Cfg, VOCAB_ID)",
    )
    if reverted == src:
        bad.append("the run-end save no longer matches the text this check reverts to build "
                   "its negative case, so the negative case is inert -- re-derive it")
        return
    _, short_rev = _short_sites(reverted)
    if not short_rev:
        bad.append("the pre-de-31 run-end save (4 args) did NOT report short: the predicate "
                   "cannot see the defect it exists for")
        return
    deleted = reverted.replace(
        "        save_checkpoint(ckpt_path, raw_model.state_dict(), Cfg, VOCAB_ID)\n", "")
    _, short_del = _short_sites(deleted)
    if not isinstance(short_del, str):
        bad.append(f"deleting the run-end call site left {len(_call_sites(deleted))} sites "
                   f"and the check stayed green: it counts args at the sites that remain, so "
                   f"a removed save path is invisible -- the de-30 shape")
        return
    print(f"  negative cases: pre-de-31 4-arg call reported short at {short_rev}; "
          f"deleting the site reported '{short_del}'")


def main():
    bad = []
    tmp = tempfile.mkdtemp()
    _check_call_sites(bad)

    # POSITIVE: the three real numbers. base is what earlier segments consumed, split
    # evenly, so sum(base) + this segment's rows == as_of x 256 exactly.
    for as_of, want_sum, origin in REAL:
        consumed_before = origin * ROWS_PER_STEP
        per = consumed_before // len(NAMES)
        base = {n: per for n in NAMES}
        base[NAMES[0]] += consumed_before - per * len(NAMES)
        p, ck, err = _save(tmp, as_of, origin, base)
        if err:
            bad.append(f"as_of {as_of} origin {origin}: the assert FIRED on a correct cursor -- {err[:150]}")
            continue
        got = sum(ck["row_cursor"].values())
        tag = "DISCRIMINATING" if origin else "origin 0, both impls agree"
        print(f"  as_of {as_of:>5} origin {origin:>4} : sum {got:>7} want {want_sum:>7} "
              f"({tag})")
        if got != want_sum:
            bad.append(f"as_of {as_of}: cursor sums to {got}, the real checkpoint holds {want_sum}")
        if ck.get("total_steps") != 3814:
            bad.append(f"as_of {as_of}: total_steps missing from the checkpoint "
                       f"({ck.get('total_steps')!r}); nothing can compare a resume's schedule")
        if ck.get("row_cursor_sum_unchecked"):
            bad.append(f"as_of {as_of}: the identity was SKIPPED with no domain discarded -- "
                       f"{ck['row_cursor_sum_unchecked'][:110]}")

    # NEGATIVE, and built at origin 832 on purpose: base dropped, which is exactly what
    # the pre-52aec31 code did. At origin 0 this same mutation is invisible -- base is
    # empty there and absolute == segment-only -- so a negative case built at origin 0
    # would pass against the defect it is meant to catch.
    p, ck, err = _save(tmp, 1192, 832, {})
    if err:
        print(f"  segment-only  : REFUSED -- {err[:120]}")
        if "92160" not in err.replace(",", "") and "92,160" not in err:
            print(f"      (the message does not quote the short sum; full text: {err[:300]})")
    else:
        got = sum(ck["row_cursor"].values())
        bad.append(f"SEGMENT-ONLY CURSOR ACCEPTED: base dropped at origin 832 wrote {got} "
                   f"where {1192 * ROWS_PER_STEP} rows are consumed, and nothing raised. "
                   f"This is the defect the identity exists to catch.")

    # And the same mutation at origin 0, to prove the negative case above needed origin 832.
    p, ck0, err0 = _save(tmp, 500, 0, {})
    if err0:
        bad.append(f"dropping base at origin 0 raised, but absolute and segment-only are EQUAL "
                   f"there -- the assert is firing on a correct cursor: {err0[:120]}")
    else:
        print(f"  same mutation at origin 0: sum {sum(ck0['row_cursor'].values())}, no raise "
              f"-- which is why the negative case is built at origin 832")

    # The DISCARD path must skip, not fire: a domain that restarted at row 0 contributes 0
    # to base by design, so the sum is legitimately short.
    p, ckd, errd = _save(tmp, 1192, 832, {}, discarded=["a (corpus abc -> def)"])
    if errd:
        bad.append(f"a discarded cursor made the identity FIRE: {errd[:150]}. A domain that "
                   f"restarted at row 0 is legitimately short, and an assert that fires there "
                   f"teaches an operator to ignore it")
    elif not ckd.get("row_cursor_sum_unchecked"):
        bad.append("a discarded cursor neither fired nor recorded row_cursor_sum_unchecked: "
                   "the skip is silent, so a reader cannot tell a checked sum from a skipped one")
    else:
        print(f"  discard path  : skipped and recorded -- {ckd['row_cursor_sum_unchecked'][:80]}")

    print()
    if bad:
        for b in bad:
            print("  FAIL:", b)
        print("cursor sum: DEFECT PRESENT")
        return 1
    print("cursor sum: OK -- the identity holds on all three real numbers, refuses a "
          "segment-only cursor at origin 832, and skips the discard path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
