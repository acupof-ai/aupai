#!/usr/bin/env python3
"""build_mix's plan length against the row cursor: does a resume allocate the REMAINDER?

THE DEFECT. build_mix's per-phase allocation is `want = int(rows * frac * d.get(key,
d["weight"]))`, and `rows` there is the mix's FULL budget. The row cursor moves where
consumption starts (`arange(used[name], used[name] + want)`); it never reduces how much is
allocated. So a resume plans the whole budget again, on top of what the earlier segment
already trained, and main() turns that plan into total_steps -- the run ends LONGER than the
recipe. No line numbers: three citations in this file rotted into unrelated code and one sent
a reader chasing them (e1, 2026-09-06). Roles identify these; numbers did not.
Measured on p200m_4b_0902: plan 976,556 rows = 4.00B in the fresh run AND in the resume,
identical, so 832 steps' worth of tokens are trained twice over on the token axis.
`--max_steps` caps the symptom by truncating consumption; the allocation stays wrong, and
26,645 of 122,069 rows per rank go unread.

WHY THIS RUNS ANYWHERE. _domain_seqs is the only part of build_mix that touches disk, so
stubbing it makes the plan arithmetic testable with no corpus, no tokenizer and no card --
and the defect IS arithmetic. seq is 64 and the pool is twice the budget on purpose: at a
realistic seq the pool caps every domain, and a cap masks the cursor's effect (a capped
`want` shrinks for the right reason, which is not the property under test).

THE FOUR ASSERTIONS, and the fourth is the one the others cannot make:

  1. remainder      -- a resume's plan holds budget - consumed rows, not budget
  2. steps join     -- resume_step + resume's steps == the fresh run's total_steps
  3. warmdown       -- the absolute warmdown start is the same on both paths
  4. fresh unchanged-- the no-cursor plan is BIT-IDENTICAL to what the old code built

4 is why the fixture below carries a hash measured on the OLD code, before the fix
existed. Without it the suite only proves the new code is self-consistent: a change that
altered the fresh path too would satisfy 1-3 and silently move every ladder point.

2 is half arithmetic and half AST, and the AST half is not belt-and-braces. The
arithmetic half recomputes `segment steps + RESUME_STEP` in THIS file, so deleting
train.py's own `total_steps += resume_step` leaves it green -- measured: that mutation
survived while eight others went red. A criterion that recomputes what it is judging
cannot see the code removed, which is the same shape as a grep whose needle sits in its
own data table.

  python3 scripts/test_plan_length.py
"""
# restartable: an interrupt costs 6s and loses nothing. Builds four plans in memory from a
# stubbed pool and writes only a temp mix json -- no checkpoint, no token cache, no shard.
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile

import torch

ROOT = os.environ.get("AUPAI_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEQ = 64
POOL = 200000
BUDGET_ROWS = 100000
WORLD = 8
BATCH, ACCUM = 16, 2
ROWS_PER_STEP = BATCH * ACCUM * WORLD
RESUME_STEP = 200

# MEASURED on the code BEFORE the plan-length fix (de, 2026-09-02), so assertion 4 has a
# baseline the fix could not have produced. The fresh path must not move: every ladder
# point and every A/B reads a plan built this way.
FRESH_BASELINE = {
    "plan_rows_per_rank": 12500,
    "plan_rows_total": 100000,
    "total_steps": 390,
    "cursor": {"a": 60000, "b": 40000},
    # RECORDED on the pre-fix code (HEAD c4a9841, before any edit to build_mix), and stable
    # across three fresh processes -- the plan comes from a seeded generator, so an unstable
    # hash would mean the baseline could not detect a change at all.
    "plan_sha": "4c6037cbc0a4cb6242d1e29e5738c8f0",
}


def _stub_domain_seqs(domain, tok, is_main, ddp, workers=1):
    """A pool whose row i is identifiable: (domain tag, row index) repeated.

    Assertion 4 hashes planned CONTENT, not a count, so the rows have to differ from each
    other and between domains -- a pool of zeros would hash the same under any plan."""
    h = abs(int(hashlib.blake2b(domain.encode(), digest_size=4).hexdigest(), 16)) % 900 + 1
    return torch.arange(POOL, dtype=torch.int32).unsqueeze(1).repeat(1, SEQ + 1) + h * 1000000


def _mix_path():
    mix = {
        "total_tokens": BUDGET_ROWS * SEQ,
        "domains": {"a": {"weight": 0.6, "epochs": 1}, "b": {"weight": 0.4, "epochs": 1}},
    }
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mix_plan_probe.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mix, f)
    return p


def _build(train, p, **kw):
    """build_mix with its chatter captured. Returns (plan_rows, log, seeded_flag, cursor)."""
    Cfg = train.Cfg
    Cfg._row_cursor_base = None
    for flag in ("_cursor_seeded", "_plan_trimmed"):
        if hasattr(Cfg, flag):
            setattr(Cfg, flag, None)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tr, _va = train.build_mix(p, None, True, False, 0, WORLD, **kw)
    # Either name, so this file runs on both sides of the rename.
    seeded = getattr(Cfg, "_cursor_seeded", None)
    if seeded is None:
        seeded = getattr(Cfg, "_plan_trimmed", None)
    return tr, buf.getvalue(), seeded, dict(Cfg._row_cursor)


def _sha(rows):
    h = hashlib.blake2b(digest_size=16)
    for r in rows:
        h.update(r.numpy().tobytes())
    return h.hexdigest()


def _warmdown_start(train, total_steps):
    """The ABSOLUTE step the warmdown begins at, as train.py's own schedule line prints it."""
    return total_steps - max(1, int(train.Cfg.warmdown * total_steps))


def _reads_total_steps_join(path):
    """Does main() actually add resume_step back, guarded by the cursor flag?

    This is an AST read of the source, not a behaviour test, and it is here because the
    behaviour version was VACUOUS: assertion 2 below recomputes `segment steps +
    RESUME_STEP` in this file, so deleting train.py's own `total_steps += resume_step`
    left it green (measured -- the mutation survived while eight others went red). A
    criterion that recomputes the thing it is judging cannot see the thing removed.

    So: assert the statement exists, inside a guard naming _cursor_seeded, and that
    total_steps is published to Cfg afterwards for the checkpoint. The arithmetic
    assertion below then says the number is RIGHT; this says the code is there to produce
    it."""
    import ast

    tree = ast.parse(open(path, encoding="utf-8").read())
    joins, publishes = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            if getattr(node.target, "id", None) == "total_steps" and \
                    getattr(node.value, "id", None) == "resume_step":
                joins.append(node.lineno)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "_total_steps":
                    publishes.append(node.lineno)
    src = open(path, encoding="utf-8").read().splitlines()
    guarded = []
    for ln in joins:
        # The guard is the enclosing `if`, one to three lines above the statement.
        window = "\n".join(src[max(0, ln - 4):ln])
        if "_cursor_seeded" in window and "resume_step" in window:
            guarded.append(ln)
    return joins, guarded, publishes


def main():
    import train
    from train import Cfg

    train._domain_seqs = _stub_domain_seqs
    train.VOCAB_ID = "stub"
    Cfg.seq, Cfg.batch, Cfg.accum, Cfg.fone = SEQ, BATCH, ACCUM, False
    Cfg.val_frac, Cfg.val_rows_max = 0.0, 1
    if not Cfg.warmdown:
        Cfg.warmdown = 0.1
    p = _mix_path()

    record = "--record" in sys.argv
    bad, warn = [], []

    tr0, _log0, seeded0, cur0 = _build(train, p)
    per_rank0 = tr0.shape[0]
    total0 = per_rank0 * WORLD
    steps0 = per_rank0 // (BATCH * ACCUM)
    sha0 = _sha(tr0)
    print(f"  fresh        : {total0} rows ({per_rank0}/rank), total_steps {steps0}, "
          f"seeded={seeded0}")
    print(f"                 plan sha {sha0}")

    if record:
        print(f"\n  --record: paste into FRESH_BASELINE\n"
              f'    "plan_rows_per_rank": {per_rank0},\n'
              f'    "plan_rows_total": {total0},\n'
              f'    "total_steps": {steps0},\n'
              f'    "cursor": {json.dumps(cur0)},\n'
              f'    "plan_sha": "{sha0}",')
        return 0

    # 4: the fresh path is untouched. Counts AND content -- a plan of the right length
    # built from different rows is a different training run.
    if per_rank0 != FRESH_BASELINE["plan_rows_per_rank"] or steps0 != FRESH_BASELINE["total_steps"]:
        bad.append(f"FRESH PATH MOVED: {per_rank0} rows/rank, {steps0} steps; the old code built "
                   f"{FRESH_BASELINE['plan_rows_per_rank']}/{FRESH_BASELINE['total_steps']}")
    if cur0 != FRESH_BASELINE["cursor"]:
        bad.append(f"fresh cursor moved: {cur0} vs {FRESH_BASELINE['cursor']}")
    if not FRESH_BASELINE["plan_sha"]:
        warn.append("FRESH_BASELINE['plan_sha'] is empty, so assertion 4 checks counts only -- "
                    "run this file with --record on the pre-fix code and paste the hash")
    elif sha0 != FRESH_BASELINE["plan_sha"]:
        bad.append(f"FRESH PLAN CONTENT MOVED: sha {sha0} vs baseline "
                   f"{FRESH_BASELINE['plan_sha']} at an identical row count -- same number of "
                   f"rows, different rows")
    if seeded0:
        bad.append("the cursor flag is true with NO cursor, so it cannot mean 'a cursor seeded used[]'")

    # 5: build_mix PUBLISHES the fields save_checkpoint's cursor needs. This is a call, not a
    # fixture, and that is the whole point (e1's finding 2 on 88be635a): every case in
    # test_cursor_sum.py sets _plan_world and _plan_domains_full on a fabricated cfg, so
    # deleting build_mix's publish left them all green -- the fixture supplied what the
    # producer should have produced. save_checkpoint now refuses without them rather than
    # falling back to WORLD_SIZE, so a silent regression here turns into every checkpoint
    # carrying row_cursor_refused, which is loud but only if something asserts the publish.
    _pw = getattr(train.Cfg, "_plan_world", None)
    _pf = getattr(train.Cfg, "_plan_domains_full", None)
    if _pw is None:
        bad.append("build_mix published no Cfg._plan_world: save_checkpoint cannot derive the "
                   "prefix length rows_done x world and refuses, so every checkpoint of this "
                   "run would carry row_cursor_refused instead of a cursor")
    elif int(_pw) != WORLD:
        bad.append(f"Cfg._plan_world is {_pw}, but the plan was striped at world {WORLD} -- the "
                   f"counting world and the striping world must be one number")
    if _pf is None:
        bad.append("build_mix published no Cfg._plan_domains_full: the per-domain consumed "
                   "count cannot be computed from a rank's stripe, which is the striping "
                   "defect e1 measured on five e1_conv checkpoints")
    elif int(_pf.shape[0]) != total0:
        bad.append(f"Cfg._plan_domains_full holds {int(_pf.shape[0])} rows but the plan is "
                   f"{total0} across all ranks -- they do not describe the same plan, so the "
                   f"prefix bincount is about different rows than the run consumed")

    # 1 + 2 + 3: a cursor at exactly RESUME_STEP steps' worth of rows, split by weight so
    # nothing caps and the only variable is the cursor.
    consumed = RESUME_STEP * ROWS_PER_STEP
    cur = {"a": int(consumed * 0.6), "b": int(consumed * 0.4)}
    assert sum(cur.values()) == consumed, cur
    tr1, log1, seeded1, _cur1 = _build(train, p, row_cursor=cur, cursor_srcfp=None, cursor_seed=None)
    per_rank1 = tr1.shape[0]
    total1 = per_rank1 * WORLD
    steps1 = per_rank1 // (BATCH * ACCUM)
    want_rows = BUDGET_ROWS - consumed
    print(f"  resume@{RESUME_STEP}   : {total1} rows ({per_rank1}/rank), {steps1} steps, "
          f"seeded={seeded1}")
    print(f"                 remainder should be {want_rows} rows / "
          f"{want_rows // ROWS_PER_STEP} steps")

    if "cursor discarded" in log1:
        bad.append("the cursor was discarded on a matching seed/fp, so nothing below tests the plan")
    if not seeded1:
        bad.append("the cursor flag stayed false with a live cursor: the LR compensation "
                   "at :2148 will not fire and total_steps stays on the wrong scale")

    # 1: the plan holds the REMAINDER. Tolerance is one step's rows: int() truncation per
    # domain per phase loses under a row each, and world-rounding at :1640 drops < world.
    slack = ROWS_PER_STEP
    if abs(total1 - want_rows) > slack:
        bad.append(f"OVER-ALLOCATION: the resume plans {total1} rows where {want_rows} remain "
                   f"({total1 / max(want_rows, 1):.3f}x). :1597 multiplies the full budget by "
                   f"the weight and never subtracts the cursor.")

    # 2: the two paths must END at the same absolute step. Two halves, because the
    # arithmetic half alone is vacuous -- it recomputes the sum here, so deleting
    # train.py's own `total_steps += resume_step` leaves it green (measured: that mutation
    # survived while eight others went red). The AST half asserts the code that produces
    # the number; the arithmetic half asserts the number is right.
    joins, guarded, publishes = _reads_total_steps_join(os.path.join(ROOT, "train.py"))
    print(f"  join stmt    : total_steps += resume_step at {joins}, "
          f"cursor-guarded {guarded}, Cfg._total_steps published at {publishes}")
    if not joins:
        bad.append("train.py has no `total_steps += resume_step`: a cursor-seeded plan counts "
                   "only this segment, so total_steps would be smaller than the resume step and "
                   "the loop exits without running (the 16000/7998 rehearsal)")
    elif not guarded:
        bad.append(f"`total_steps += resume_step` at {joins} is not guarded by _cursor_seeded and "
                   f"resume_step: an unguarded add inflates the fresh path too")
    if not publishes:
        bad.append("nothing assigns Cfg._total_steps, so save_checkpoint cannot record the "
                   "schedule and the next resume has nothing to compare against")

    end1 = steps1 + RESUME_STEP
    if end1 != steps0:
        bad.append(f"STEPS DO NOT JOIN: fresh ends at {steps0}, the resume ends at "
                   f"{end1} = {steps1} + {RESUME_STEP}")

    # 3: and the warmdown must begin at the same absolute step, or the LR shape differs
    # between a run that was interrupted and one that was not.
    wd0, wd1 = _warmdown_start(train, steps0), _warmdown_start(train, end1)
    print(f"  warmdown     : fresh {wd0}, resume {wd1}")
    if wd0 != wd1:
        bad.append(f"WARMDOWN MOVED: fresh starts warmdown at {wd0}, the resume at {wd1}")

    # The end-of-budget case the fix creates: once the cursor has consumed everything,
    # every want is 0, `plan` is an empty list, and :1613 torch.cat raises a bare
    # ValueError. Measured on the pre-fix code by seeding a cursor past the epoch cap.
    # A finished budget must say so, not raise from a tensor library.
    full = {"a": 60000, "b": 40000}
    try:
        tr2, log2, _s2, _c2 = _build(train, p, row_cursor=full, cursor_srcfp=None, cursor_seed=None)
        n2 = tr2.shape[0]
        print(f"  budget spent : {n2 * WORLD} rows planned, no raise")
        if n2 == 0:
            bad.append("a spent budget planned 0 rows and returned them: the step loop runs "
                       "zero steps and the run exits looking successful")
    except ValueError as e:
        bad.append(f"a spent budget raises ValueError from torch.cat ({e}); the message names "
                   f"a tensor list, not the budget, so the operator debugs the wrong thing")
    except RuntimeError as e:
        msg = str(e)
        if "budget" in msg.lower() or "cursor" in msg.lower() or "consumed" in msg.lower():
            print(f"  budget spent : refused, naming the cause -- {msg[:80]}")
        else:
            bad.append(f"a spent budget raises RuntimeError that does not name the budget: {msg[:120]}")

    # 44-12: a cursor that WOULD be discarded refuses startup, named; the flag pardons
    # knowingly. Case 1 (sample_seed mismatch) -- the cheaper world, no corpus dir needed.
    Cfg.allow_partial_cursor = False
    try:
        _build(train, p, row_cursor={"a": 1000}, cursor_seed=train._sample_seed() + 1)
        bad.append("cursor-discard startup did not refuse (sample_seed mismatch): a print is not a gate")
    except RuntimeError as e:
        msg = str(e)
        if "a" not in msg or "discard" not in msg:
            bad.append(f"the refusal does not name the domain/reason: {msg[:100]}")
        else:
            print(f"  discard refuse: RuntimeError names the domain -- {msg[:80]}")
    # the flag pardons: the discarded cursor subtracts nothing (full budget, not budget-1000)
    Cfg.allow_partial_cursor = True
    tr_d, _log_d, _sd_d, _cur_d = _build(
        train, p, row_cursor={"a": 1000}, cursor_seed=train._sample_seed() + 1)
    Cfg.allow_partial_cursor = False
    if not Cfg._cursor_discarded or "a" not in Cfg._cursor_discarded[0]:
        bad.append(f"flag path: discard not recorded: {Cfg._cursor_discarded}")
    if tr_d.shape[0] != per_rank0:
        bad.append(f"flag path: a discarded cursor still subtracted rows: "
                   f"{tr_d.shape[0]} vs fresh {per_rank0}")

    print()
    for w in warn:
        print("  WARN:", w)
    if bad:
        for b in bad:
            print("  FAIL:", b)
        print("plan length: DEFECT PRESENT")
        return 1
    print("plan length: OK -- a resume allocates the remainder, both paths end and warm down "
          "at the same absolute step, and the fresh plan is unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
