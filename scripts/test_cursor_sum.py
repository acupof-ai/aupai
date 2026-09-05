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
    cfg._plan_world = world  # what build_mix striped at; save_checkpoint refuses without it
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

    The assertion is against a bincount over the CONSUMED PREFIX of the full plan, computed
    here independently -- a known answer, not a re-run of the code under test.

    The plan is deliberately LONGER than what the run consumed (58's finding on the first
    version of this file, 2026-09-06). Every fixture here used to be plan-complete, so
    `_full[:rows_done * world]` was the identity slice in all of them and deleting the bound
    outright -- `head = _full` -- left the file GREEN. The sum identity is not a substitute:
    it catches the unbounded read on the no-discard path, but `_discarded` non-empty skips it
    by design, and mid-plan plus a changed corpus fingerprint is ordinary under --auto-resume.
    So the discard variant is asserted too, below.
    """
    as_of, origin = 40, 0
    consumed = (as_of - origin) * BATCH * ACCUM * WORLD  # 40 x 16 x 2 x 8 = 10,240 rows
    n = consumed * 2  # the plan the run did NOT finish: 20,480 rows
    # Sizes and seed searched for a fixture that reproduces BOTH directions at once: under
    # the old computation s_inject reads 32 against a 25-row pool (the ckpt_e1_conv_n8
    # direction, the damaging one) while p_format reads 72 against a truth of 104. A fixture
    # with only the under direction would pass a `cursor > pool` guard, which is exactly the
    # guard 58 measured as catching 3 of their 9 real violations.
    sizes = [25, 203, n - 228]
    assert all(s % WORLD for s in sizes[:2]), sizes
    dom = []
    for di, s in enumerate(sizes):
        dom += [di] * s
    g = torch.Generator().manual_seed(21)
    full = torch.tensor(dom, dtype=torch.int8)[torch.randperm(n, generator=g)]

    truth = torch.bincount(full[:consumed].to(torch.int64), minlength=len(NAMES))
    whole = torch.bincount(full.to(torch.int64), minlength=len(NAMES))
    if torch.equal(truth, whole):
        bad.append(f"the striping fixture is PLAN-COMPLETE: the consumed prefix and the whole "
                   f"plan give the same counts ({truth.tolist()}), so deleting the "
                   f"rows_done*world bound would not be caught. Lengthen the plan")
        return
    stripe0 = torch.bincount(_stripe(full)[:consumed // WORLD].to(torch.int64),
                             minlength=len(NAMES)) * WORLD
    if torch.equal(truth, stripe0):
        bad.append(f"the striping fixture is INERT: rank 0's count x world equals the truth "
                   f"({truth.tolist()}), so it cannot separate the two implementations. "
                   f"Re-derive the domain sizes")
        return

    # THE DISCARD PATH FIRST, mid-plan, because it is the path with NO backstop (e1,
    # 2026-09-06, finding 1 on ba05651e). `_discarded` non-empty skips the sum identity by
    # design -- a domain that restarted at row 0 is legitimately short -- so here the
    # per-domain assertion is the only thing that can fire. It ran LAST until this commit,
    # after a `return` on the no-discard raise, and all three mutants of the bound die on the
    # identity in the no-discard case: measured, the backstop never executed in the world it
    # was written for. Order is the fix, and this order cannot rot the same way -- a later
    # `return` added below cannot reach backwards.
    #
    # With the rows_done*world bound deleted: this case is WRITTEN, [25, 203, 20252], a cursor
    # 2x past every pool, labelled row_cursor_sum_unchecked, and only this assertion objects.
    # Mid-plan plus a changed corpus fingerprint is ordinary under --auto-resume, not exotic.
    want = truth.tolist()
    p, ckd, errd = _save(tmp, as_of, origin, {n: 0 for n in NAMES}, full=full,
                         discarded=[f"{NAMES[0]} (corpus abc -> def)"], tag="_stripe_disc")
    if errd:
        bad.append(f"the mid-plan discard case RAISED on a correct plan: {errd[:180]}")
    else:
        gotd = [ckd["row_cursor"][nm] for nm in NAMES]
        if gotd != want:
            bad.append(f"mid-plan WITH a discard: cursor {gotd}, the consumed prefix holds "
                       f"{want}. The sum identity is skipped on this path, so this assertion "
                       f"is the only backstop it has")
        elif not ckd.get("row_cursor_sum_unchecked"):
            bad.append("the mid-plan discard case did not record row_cursor_sum_unchecked, so "
                       "a reader cannot tell the skipped sum from a checked one")
        else:
            print(f"  discard+midplan: got {gotd}, identity skipped and labelled -- the "
                  f"per-domain assertion is this path's only backstop")

    p, ck, err = _save(tmp, as_of, origin, {n: 0 for n in NAMES}, full=full, tag="_stripe")
    if err:
        bad.append(f"the striping case RAISED on a correct plan: {err[:180]}")
        return
    got = [ck["row_cursor"][nm] for nm in NAMES]
    print(f"  striping      : got {got} want {want} (rank0 x world would write "
          f"{stripe0.tolist()}, whole plan {whole.tolist()})")
    if got != want:
        bad.append(f"per-domain cursor is {got}, the plan's consumed prefix holds {want}. "
                   f"the striping defect writes {stripe0.tolist()}; an unbounded read writes "
                   f"{whole.tolist()}")
    if sum(got) != as_of * ROWS_PER_STEP:
        bad.append(f"the striping case's sum is {sum(got)}, not {as_of * ROWS_PER_STEP} -- the "
                   f"fixture is wrong, not the code")
    # BOTH DIRECTIONS, asserted separately. e1 measured a `cursor > pool` guard catching 3 of
    # 9 real violations and calling two wholly-wrong checkpoints clean, so a fixture that
    # reproduces only the over-pool direction would certify exactly the guard that misses the
    # majority. Over-pool is the damaging one -- stage 2 skips the tail -- and under-plan is
    # the one no bound can see.
    over = [(NAMES[i], int(stripe0[i]), sizes[i]) for i in range(len(NAMES))
            if int(stripe0[i]) > sizes[i]]
    under = [(NAMES[i], int(stripe0[i]), int(truth[i])) for i in range(len(NAMES))
             if int(stripe0[i]) != int(truth[i]) and int(stripe0[i]) <= sizes[i]]
    if not over:
        bad.append("no domain's rank0 x world count exceeds its pool, so this fixture does "
                   "not reproduce the ckpt_e1_conv_n8 direction (212 against a 204 pool)")
    if not under:
        bad.append("no domain is wrong while staying under its pool, so this fixture would be "
                   "fully caught by a `cursor > pool` bound -- the guard e1 measured as "
                   "catching only 3 of 9 real violations")
    if over and under:
        print(f"  over-count    : {', '.join(f'{nm} {c} against a {s}-row pool' for nm, c, s in over)}"
              f" under the old computation")
        print(f"  under-count   : {', '.join(f'{nm} {c} against a true {t}' for nm, c, t in under)}"
              f" -- invisible to any `cursor > pool` bound")


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


def _check_world_source(bad, tmp):
    """The world used for the prefix must be the one the PLAN was striped at.

    save_checkpoint read os.environ["WORLD_SIZE"] while build_mix striped at
    dist.get_world_size() -- two independent sources for one number (58, 2026-09-06). They
    agree under torchrun, so no fixture built at one world can see the divergence; this one
    sets them to DIFFERENT values and asserts the plan's wins. Without that, `rows_done *
    world` slices a prefix the ranks never jointly consumed, silently.
    """
    import train

    as_of, origin = 40, 0
    consumed = as_of * BATCH * ACCUM * WORLD
    n = consumed * 2
    sizes = [25, 203, n - 228]
    dom = []
    for di, s in enumerate(sizes):
        dom += [di] * s
    g = torch.Generator().manual_seed(21)
    full = torch.tensor(dom, dtype=torch.int8)[torch.randperm(n, generator=g)]
    want = torch.bincount(full[:consumed].to(torch.int64), minlength=len(NAMES)).tolist()

    cfg = FakeCfg()
    cfg._plan_domains_full = full
    cfg._plan_domains = _stripe(full, 0, WORLD)
    cfg._plan_world = WORLD  # what build_mix striped at
    cfg._plan_names = list(NAMES)
    cfg._plan_step_origin = origin
    cfg._row_cursor = {nm: 0 for nm in NAMES}
    cfg._row_cursor_srcfp = {nm: "fp" for nm in NAMES}
    cfg._row_cursor_base = {}
    cfg._cursor_discarded = []
    cfg._total_steps = 3814

    p = os.path.join(tmp, "ck_worldsrc.pt")
    prev = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = str(WORLD * 2)  # a launcher that set only the environment
    try:
        train.save_checkpoint(p, {"w": torch.zeros(2)}, cfg, "vocab", step=as_of)
        err = None
    except AssertionError as e:
        err = str(e)
    finally:
        if prev is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = prev
    if err:
        bad.append(f"with WORLD_SIZE={WORLD * 2} and the plan striped at {WORLD}, the write "
                   f"RAISED: {err[:170]}. The plan's world is the correct one and the "
                   f"environment must not override it")
        return
    ck = torch.load(p, map_location="cpu", weights_only=False)
    got = [ck["row_cursor"][nm] for nm in NAMES]
    if got != want:
        bad.append(f"the environment's WORLD_SIZE={WORLD * 2} won over the plan's {WORLD}: "
                   f"cursor {got}, the prefix the ranks consumed holds {want}")
    else:
        print(f"  world source  : plan {WORLD} beat WORLD_SIZE={WORLD * 2}, got {got}")


def _check_no_plan_world_refuses(bad, tmp):
    """A plan vector with no _plan_world beside it must REFUSE at world > 1.

    e1's finding 2 on 88be635a: the reader was `_plan_world or int(os.environ[...])`, so
    deleting build_mix's publish fell back to the environment and restored the two-independent-
    sources condition that commit removed -- silently, with every test green. The fallback is
    gone; absence is a refusal above world 1, and the message says WHICH of the two fields is
    missing. At world 1 it still writes, because there the stripe is the full plan and the
    prefix length is unambiguous.
    """
    import train

    for env_world, expect_cursor in ((WORLD, False), (1, True)):
        as_of = 40
        consumed = as_of * BATCH * ACCUM * env_world
        full = torch.tensor([i % len(NAMES) for i in range(consumed)], dtype=torch.int8)
        cfg = FakeCfg()
        cfg._plan_domains_full = full
        cfg._plan_domains = _stripe(full, 0, env_world)
        cfg._plan_world = None  # the publisher stopped publishing
        cfg._plan_names = list(NAMES)
        cfg._plan_step_origin = 0
        cfg._row_cursor = {nm: 0 for nm in NAMES}
        cfg._row_cursor_srcfp = {nm: "fp" for nm in NAMES}
        cfg._row_cursor_base = {}
        cfg._cursor_discarded = []
        cfg._total_steps = 3814
        p = os.path.join(tmp, f"ck_noworld_{env_world}.pt")
        prev = os.environ.get("WORLD_SIZE")
        os.environ["WORLD_SIZE"] = str(env_world)
        try:
            train.save_checkpoint(p, {"w": torch.zeros(2)}, cfg, "vocab", step=as_of)
        finally:
            if prev is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = prev
        ck = torch.load(p, map_location="cpu", weights_only=False)
        wrote = ck.get("row_cursor_as_of_step") is not None
        if wrote != expect_cursor:
            bad.append(f"no _plan_world at WORLD_SIZE={env_world}: cursor "
                       f"{'written' if wrote else 'refused'}, expected "
                       f"{'written' if expect_cursor else 'refused'}. Above world 1 the prefix "
                       f"length cannot be derived and falling back to the environment is the "
                       f"defect; at world 1 the stripe IS the full plan")
        elif not expect_cursor and "_plan_world" not in (ck.get("row_cursor_refused") or ""):
            bad.append(f"the refusal does not name _plan_world, so a reader cannot tell it from "
                       f"a missing plan vector: {ck.get('row_cursor_refused')!r}")
        else:
            state = "wrote (world 1, stripe is the plan)" if wrote else "refused, naming _plan_world"
            print(f"  no plan world : WORLD_SIZE={env_world} -> {state}")


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
    _check_no_plan_world_refuses(bad, tmp)
    _check_world_source(bad, tmp)

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
