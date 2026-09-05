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


def _plan_full(origin, step, per_domain=None):
    """The FULL plan's domain row, all ranks' columns, as build_mix publishes it.

    Length (step - origin) * batch * accum * world: every rank draws
    (step - origin) * batch * accum rows from its own stripe, and there are `world` stripes.
    save_checkpoint counts a prefix of rows_done * world of THIS vector.
    """
    n = (step - origin) * BATCH * ACCUM * WORLD
    if per_domain is None:
        # Round-robin over the full plan. Note this is a case where the OLD computation was
        # right: 3 domains into n rows divides evenly across the stripes, so a fixture built
        # only this way cannot see the striping defect. The discriminating case is below.
        return torch.tensor([i % len(NAMES) for i in range(n)], dtype=torch.int8)
    out = []
    for di, cnt in enumerate(per_domain):
        out += [di] * cnt
    assert len(out) == n, (len(out), n)
    return torch.tensor(out, dtype=torch.int8)


def _stripe(full, rank=0, world=WORLD):
    """This rank's stripe, sliced off the full plan exactly as build_mix does at :2415.

    Derived, not invented a second time: the two vectors have to describe one plan, or the
    fixture can be self-consistent while disagreeing with the code it tests."""
    return full[rank::world].clone()


def _save(tmp, as_of, origin, base, discarded=None, world=WORLD, full=None, tag=""):
    """Call the real save_checkpoint and return (path, loaded_ck, error_or_None)."""
    import train

    cfg = FakeCfg()
    if full is None:
        full = _plan_full(origin, as_of)
    cfg._plan_domains_full = full
    cfg._plan_domains = _stripe(full, 0, world)
    cfg._plan_names = list(NAMES)
    cfg._plan_step_origin = origin
    cfg._row_cursor = {n: 0 for n in NAMES}
    cfg._row_cursor_srcfp = {n: "fp" for n in NAMES}
    cfg._row_cursor_base = dict(base)
    cfg._cursor_discarded = list(discarded or [])
    cfg._total_steps = 3814

    p = os.path.join(tmp, f"ck_{as_of}_{origin}_{len(discarded or [])}{tag}.pt")
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


def _check_striping(bad, tmp):
    """PER-DOMAIN counts, not just the sum: the striping defect 58 found on two checkpoints.

    The plan is striped by column, so rank r holds columns r, r+world, ... For a domain whose
    row count is not a multiple of world, the stripes do not divide its rows evenly and rank
    0's count x world is wrong -- over for some domains, under for others, and the SUM stays
    exact because every rank holds n/world columns. That is why the identity above passed on
    ckpt_e1_conv_n8 (s_inject_n8 212 written against a 204-row pool) and ckpt_e1_conv_n1
    (s_inject_n1 20 against 25) while both summed to 276,096 to the row.

    The fixture is a shuffled plan with three domains of DELIBERATELY awkward sizes, so no
    domain's count divides by world. The round-robin fixture above cannot see this: 3 domains
    into n rows lands each domain's rows evenly across every stripe, which is what let the
    defect live in a file whose whole subject is the cursor.

    The assertion is against a bincount over the full plan, computed here independently -- a
    known answer, not a re-run of the code under test.
    """
    as_of, origin = 40, 0
    n = (as_of - origin) * BATCH * ACCUM * WORLD  # 40 x 16 x 2 x 8 = 10,240 rows
    # Sizes chosen so none is a multiple of 8 and the stripes must split them unevenly.
    sizes = [25, 203, n - 228]
    assert all(s % WORLD for s in sizes[:2]), sizes
    dom = []
    for di, s in enumerate(sizes):
        dom += [di] * s
    g = torch.Generator().manual_seed(7)
    full = torch.tensor(dom, dtype=torch.int8)[torch.randperm(n, generator=g)]

    truth = torch.bincount(full.to(torch.int64), minlength=len(NAMES))
    stripe0 = torch.bincount(_stripe(full).to(torch.int64), minlength=len(NAMES)) * WORLD
    if torch.equal(truth, stripe0):
        bad.append(f"the striping fixture is INERT: rank 0's count x world equals the truth "
                   f"({truth.tolist()}), so it cannot separate the two implementations. "
                   f"Re-derive the domain sizes")
        return

    p, ck, err = _save(tmp, as_of, origin, {n: 0 for n in NAMES}, full=full, tag="_stripe")
    if err:
        bad.append(f"the striping case RAISED on a correct plan: {err[:180]}")
        return
    got = [ck["row_cursor"][nm] for nm in NAMES]
    want = truth.tolist()
    print(f"  striping      : got {got} want {want} (rank0 x world would write "
          f"{stripe0.tolist()})")
    if got != want:
        bad.append(f"per-domain cursor is {got}, the plan actually consumed {want}. This is "
                   f"58's defect: rank 0's stripe x world would write {stripe0.tolist()}")
    if sum(got) != as_of * ROWS_PER_STEP:
        bad.append(f"the striping case's sum is {sum(got)}, not {as_of * ROWS_PER_STEP} -- the "
                   f"fixture is wrong, not the code")
    # And the over-count direction is the damaging one: a cursor past the pool makes stage 2
    # skip the tail. Name it explicitly so a future reader knows which side to fear.
    over = [(NAMES[i], int(stripe0[i]), sizes[i]) for i in range(len(NAMES))
            if int(stripe0[i]) > sizes[i]]
    if not over:
        bad.append("no domain's rank0 x world count exceeds its pool, so this fixture does "
                   "not reproduce the ckpt_e1_conv_n8 direction (212 against a 204 pool)")
    else:
        print(f"  over-count    : {', '.join(f'{nm} {c} against a {s}-row pool' for nm, c, s in over)}"
              f" under the old computation")


def _check_no_full_plan_refuses(bad, tmp):
    """A checkpoint with no _plan_domains_full must REFUSE, not fall back to the stripe.

    The fallback is what produced 58's two wrong checkpoints, so it may not survive as a
    compatibility path: a wrong cursor is indistinguishable from a right one to every later
    reader, while a refusal costs a resume that re-reads rows and says why in the file."""
    import train

    cfg = FakeCfg()
    full = _plan_full(0, 40)
    cfg._plan_domains = _stripe(full)
    cfg._plan_domains_full = None  # the pre-fix world: only the stripe was published
    cfg._plan_names = list(NAMES)
    cfg._plan_step_origin = 0
    cfg._row_cursor = {n: 0 for n in NAMES}
    cfg._row_cursor_srcfp = {n: "fp" for n in NAMES}
    cfg._row_cursor_base = {}
    cfg._cursor_discarded = []
    cfg._total_steps = 3814
    p = os.path.join(tmp, "ck_nofull.pt")
    prev = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = str(WORLD)
    try:
        train.save_checkpoint(p, {"w": torch.zeros(2)}, cfg, "vocab", step=40)
    finally:
        if prev is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = prev
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if "row_cursor" in ck and ck.get("row_cursor_as_of_step") is not None:
        bad.append(f"with no _plan_domains_full the cursor was still written "
                   f"({ck['row_cursor']}): the stripe fallback survived, which is the defect")
    elif not ck.get("row_cursor_refused"):
        bad.append("with no _plan_domains_full nothing was written and nothing was recorded: "
                   "the refusal is silent, so a reader cannot tell it from a run that never "
                   "reached a save")
    else:
        print(f"  no full plan  : refused and recorded -- {ck['row_cursor_refused'][:90]}")


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

    _check_striping(bad, tmp)
    _check_no_full_plan_refuses(bad, tmp)

    print()
    if bad:
        for b in bad:
            print("  FAIL:", b)
        print("cursor sum: DEFECT PRESENT")
        return 1
    print("cursor sum: OK -- the identity holds on all three real numbers, per-domain counts "
          "match a full-plan bincount under uneven striping, a missing full plan refuses, and "
          "a segment-only cursor at origin 832 is rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
