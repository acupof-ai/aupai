#!/usr/bin/env python3
"""Do the de-33 ledger predicates hold on real history? Seven known-answer worlds.

    python3 scripts/test_ledger_predicates.py --selftest

# restartable: reads git history only, no writes; an interrupt costs 0.3s (measured, 10 worlds)

WHAT THIS GUARDS. `runs/*.jsonl` merge by union, so a row that leaves is a record disappearing:
a59ac1f staged an experiments.jsonl one line shorter than its parent and dropped b0's
ab_zeroinit AMENDMENT row, with all five pre-commit lines green. The predicate that catches it
goes in the pre-commit hook, and a predicate in a hook is only as good as the worlds it was
tested against -- so the worlds live here, built by mutating the REAL ledger, never hand-written.

THE PREDICATE IS PER-LEDGER, AND THE REASON IS NOT THE FORMAT. All seven files are one row of
JSON per line and all merge by union, yet no single predicate fits them, because their WRITE
TOOLS made different choices on different dates:

  runs/experiments.jsonl  APPEND-ONLY since 2026-08-31 (55b1d41, "exp.py is an event log").
                          exp.py:82 appends; exp.py:298 `merge` rewrites the whole file but
                          fills only EMPTY fields (exp.py:292), which is subsumption itself.
                          -> SUBSUME
  runs/tasks.jsonl        REWRITTEN IN PLACE. harness.py:6081 `r.update(state="done", ...)`
                          then harness.py:5805 `_write_tasks` writes the file in "w" mode. A
                          close CHANGES a non-empty field, so subsumption would refuse every
                          `harness task done`: 30 of its last 161 commits, all current
                          workflow.  -> KEY_PRESENT
  review / board / retro / milestones / score_matrix
                          treated as append-only pending b0's confirmation of each writer.

FALSE-POSITIVE COST, MEASURED over the last 400 single-parent commits per ledger (317 total):
subsumption everywhere refuses 46, verbatim-last-row 49. Per ledger: tasks 30/161 (18.6%),
experiments 11/57 (19.3%), retro 2/18, score_matrix 1/8, board 1/23, review 1/49.

Those two 19% figures are NOT the same kind of number, and that is the whole finding.
experiments' 11 are historical: 9 predate 55b1d41 (2026-08-26 .. 2026-08-31) and the 2 after it
are a declared history rewrite (1f07ba9) and a pod sync (f401112) -- so after the append-only
rule was established, ZERO. tasks' 30 are the live workflow and will keep arriving. A guard
tuned on the count would have read both as "14.5% noise, add an escape hatch"; the escape would
then be used weekly and the guard would stop guarding.

KNOWN BLIND SPOT of KEY_PRESENT: under a rewritten ledger, replacing a newer row with an older
one leaves the key set and the row count unchanged, so nothing sees it. For tasks.jsonl that
shape is also what a legitimate update looks like, so it is not distinguishable here.
"""

import json
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "runs/experiments.jsonl"


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def lines(rev, path=EXP):
    r = git("show", f"{rev}:{path}")
    if r.returncode:
        return None
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def obj(ln):
    try:
        return json.loads(ln)
    except Exception:
        return None


def exp_key(o):
    return (o.get("name"), o.get("started"))


def empty(v):
    return v is None or v == "" or v == [] or v == {}


def _group(rows, keyfn):
    g = defaultdict(list)
    for ln in rows:
        o = obj(ln)
        g[keyfn(o) if o else ("__unparseable__", ln.strip()[:60])].append(ln)
    return g


def subsume(head_rows, index_rows, keyfn=exp_key):
    """Keys whose HEAD LAST row is not subsumed by any index row under that key.

    Subsumed: every field with a non-empty value in the HEAD row carries the SAME value in the
    candidate. Filling an empty field is allowed; changing or emptying a non-empty one is not.
    Verbatim equality was the first version of this and it fired on 6018c62's shape500_probe,
    which did not drop the row -- it closed it, filling ended and result on a bare running row.
    """
    last = {}
    for k, rows in _group(head_rows, keyfn).items():
        last[k] = rows[-1]
    have = _group(index_rows, keyfn)
    out = []
    for k, ln in last.items():
        h = obj(ln)
        if h is None:
            continue
        ok = False
        for c_ln in have.get(k, []):
            c = obj(c_ln)
            if c is not None and all(f in c and c[f] == v for f, v in h.items() if not empty(v)):
                ok = True
                break
        if not ok:
            out.append((k, ln))
    return out


def key_present(head_rows, index_rows, keyfn=exp_key):
    """Keys present in HEAD and absent from the index. The weaker predicate, for ledgers whose
    tool rewrites rows in place."""
    have = set(_group(index_rows, keyfn))
    return [(k, rows[-1]) for k, rows in _group(head_rows, keyfn).items() if k not in have]


PREDICATE = {
    EXP: subsume,
    "runs/review.jsonl": subsume,
    "runs/board.jsonl": subsume,
    "runs/retro.jsonl": subsume,
    "runs/milestones.jsonl": subsume,
    "runs/score_matrix.jsonl": subsume,
    "runs/tasks.jsonl": key_present,
}


def _worlds():
    """(label, head, index, want_refuse, why). Real commits first, then mutations of the real
    ledger -- a hand-written world would share this file's assumptions about the schema."""
    W = []
    a, b = lines("a59ac1f^"), lines("a59ac1f")
    W.append(
        (
            "a59ac1f: the incident",
            a,
            b,
            True,
            "dropped b0's ab_zeroinit AMENDMENT, which was that key's last row",
        )
    )
    h, i = lines("71855b8"), lines("e13f09a")
    if h and i:
        W.append(("71855b8 -> e13f09a: 1e's fixture", h, i, True, "the same shape across two commits"))
    a, b = lines("6018c62^"), lines("6018c62")
    W.append(
        ("6018c62: key-fold, -97 lines", a, b, False, "folds each key to its last row; audited correct by b0")
    )
    # The acceptance negative control 1e named for the hook (de-33): an ordinary append of one
    # done event. Measured 227 -> 228 rows with 0 keys added, so the close folds onto an
    # existing key -- the shape the guard must never refuse.
    a, b = lines("c3a5a23^"), lines("c3a5a23")
    W.append(("c3a5a23: append one done event", a, b, False, "+1 row, 0 keys added"))

    real = lines("main")
    assert real, "runs/experiments.jsonl is absent on main"
    grp = defaultdict(list)
    for n, ln in enumerate(real):
        o = obj(ln)
        if o:
            grp[exp_key(o)].append(n)

    # The rollback world needs a key whose LAST row is unique under it. multi[-1] was picked
    # blindly and stopped working when ab_shapelr grew several byte-identical `done` rows: drop
    # one and the others still subsume it, so the predicate was correctly quiet and the WORLD was
    # the bug (de, 2026-09-03). Pick deliberately, and assert the property the world depends on.
    roll_key = None
    for k in [k for k, ix in grp.items() if len(ix) >= 2]:
        ix = grp[k]
        last_body = json.dumps(obj(real[ix[-1]]), sort_keys=True, ensure_ascii=False)
        copies = sum(
            1 for i in ix if json.dumps(obj(real[i]), sort_keys=True, ensure_ascii=False) == last_body
        )
        if copies == 1:
            roll_key = k
            break
    assert roll_key, "no multi-row key whose last row is unique: the rollback world cannot be built"
    ix = grp[roll_key]
    roll = list(real)
    roll[ix[-1]] = roll[ix[0]]
    W.append(
        (
            f"content rollback under {roll_key[0]!r}",
            real,
            roll,
            True,
            "newest row replaced by the oldest; row count and key set unchanged, so a "
            "line-count or key-set predicate stays quiet",
        )
    )

    keep = {ix[-1] for ix in grp.values()}
    W.append(
        (
            f"fold every key to its last row ({len(real)} -> {len(keep)})",
            real,
            [ln for n, ln in enumerate(real) if n in keep],
            False,
            "dedupe, last-row-wins",
        )
    )

    running = [n for n, ln in enumerate(real) if (obj(ln) or {}).get("status") == "running"]
    assert running, "no open row on main: the close worlds cannot be built"
    n = running[-1]
    o = obj(real[n])
    W.append(
        (
            f"append a close for {o.get('name')!r}",
            real,
            real
            + [
                json.dumps(
                    dict(o, status="ok", ended="2026-09-03 00:00", result="closed by an appended event"),
                    ensure_ascii=False,
                )
            ],
            False,
            "what exp.py done actually does (exp.py:82)",
        )
    )

    blank = [
        n
        for n, ln in enumerate(real)
        if (obj(ln) or {}).get("status") == "running" and empty((obj(ln) or {}).get("result"))
    ]
    assert blank, "no open row with an empty result: the in-place fill world cannot be built"
    n = blank[-1]
    o = obj(real[n])
    filled = list(real)
    filled[n] = json.dumps(
        dict(o, ended="2026-09-03 00:00", result="filled, nothing overwritten"), ensure_ascii=False
    )
    W.append(
        (
            f"fill empty fields in place for {o.get('name')!r}",
            real,
            filled,
            False,
            "what 6018c62 did to shape500_probe, and what exp.py merge does (exp.py:292)",
        )
    )
    return W


def selftest():
    bad = 0
    print("subsumption predicate, experiments.jsonl worlds:")
    for label, head, index, want, why in _worlds():
        if head is None or index is None:
            print(f"  SKIP {label}: a revision is absent from this clone")
            continue
        hits = subsume(head, index)
        ok = bool(hits) == want
        bad += not ok
        print(
            f"  {'ok  ' if ok else 'BUG '} {label:48} want "
            f"{'REFUSE' if want else 'PASS  '} -> {len(hits)} key(s)"
        )
        if not ok:
            print(f"       {why}")
            for k, ln in hits[:2]:
                print(f"       {k} {str(obj(ln))[:110]}")

    # EVERY LEDGER'S KEY FUNCTION MUST RESOLVE ON REAL ROWS. A keyfn reading a field the file
    # does not have returns None for every row, and then BOTH predicates go quiet rather than
    # loud: with one key holding the whole file, subsume asks only "is the last row preserved
    # somewhere" and key_present asks only "is the file non-empty". review.jsonl could lose 57
    # records and pass. Two of ledger_audit's seven keyfns were in exactly that state when this
    # check was written (de, 2026-09-03): milestones read (name, at|ts) against rows carrying
    # ckpt/milestone/measured -- 13 of 13 rows keyed to None -- and review read `id` against rows
    # carrying ts/reviewer/task -- 58 of 67. Measured replacements: ("ckpt", "milestone") gives 8
    # keys over 13 rows with no None, ("ts", "reviewer", "task") gives 62 over 67.
    #
    # This is the check that makes the class un-reintroducible, which is why it lives here rather
    # than being a one-time fix to the table.
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ledger_audit as _la
    except Exception as e:  # noqa: BLE001 -- absent or unimportable is a SKIP, not a failure
        print(f"\n  SKIP keyfn resolution: ledger_audit not importable ({e})")
    else:
        print()
        for path in sorted(_la.KEYS):
            rows_txt = lines("main", path)
            if rows_txt is None:
                print(f"  SKIP {path}: absent on main")
                continue
            parsed = [obj(ln) for ln in rows_txt]
            parsed = [o for o in parsed if o is not None]
            nones = 0
            for o in parsed:
                try:
                    k = _la.KEYS[path](o)
                except (AttributeError, TypeError):
                    nones += 1
                    continue
                if k is None or (isinstance(k, tuple) and all(x is None for x in k)):
                    nones += 1
            ok = nones == 0 and len(parsed) > 0
            bad += not ok
            print(
                f"  {'ok  ' if ok else 'BUG '} {path:28} keyfn resolves on "
                f"{len(parsed) - nones}/{len(parsed)} rows"
                + ("" if ok else "  <-- INERT: both predicates go blind on this file")
            )

    # key_present must be STRICTLY weaker: whatever it flags, subsume flags too.
    real = lines("main")
    grp = defaultdict(list)
    for n, ln in enumerate(real):
        o = obj(ln)
        if o:
            grp[exp_key(o)].append(n)
    dropped = [ln for n, ln in enumerate(real) if n not in grp[list(grp)[0]]]
    kp, sb = key_present(real, dropped), subsume(real, dropped)
    weaker = {k for k, _ in kp} <= {k for k, _ in sb}
    bad += not weaker
    print(
        f"\n  {'ok  ' if weaker else 'BUG '} key_present is weaker than subsume "
        f"({len(kp)} <= {len(sb)} key(s) on a world dropping one key)"
    )

    tasks = lines("main", "runs/tasks.jsonl")
    if tasks:

        def tkey(o):
            return o.get("id")

        # THE ROW THE FOLD KEEPS, not the last line whose state is open (found red on the real
        # tree 2026-09-04). runs/tasks.jsonl is an EVENT LOG: `task done` APPENDS a close event
        # rather than rewriting the open row, so the last `state=open` LINE for an id can be
        # followed by that id's close. Mutating it then changes a superseded line -- the fold's
        # kept row is unchanged, both predicates report 0 hits, and the case read as a BUG in the
        # predicates when the defect was in the world. de-58 was the row it picked: open at index
        # 482, closed at 483, both on main.
        #
        # Same class as harness._broken_run_commits_resolve, which planted into rows[-1] while its
        # check read exp.fold. A fixture keyed on a raw line index goes vacuous the moment anyone
        # appends to the ledger, and here it went worse than vacuous: it asserted a real bug.
        #
        # So: group first, take ids whose KEPT row is open, and mutate that row's index.
        _kept_open = [(k, rows[-1]) for k, rows in _group(tasks, tkey).items()
                      if (obj(rows[-1]) or {}).get("state") == "open"]
        if not _kept_open:
            print("  SKIP tasks world: no id on main whose current row is open")
        else:
            _k, _ln = _kept_open[-1]
            n = max(i for i, x in enumerate(tasks) if x == _ln)
            closed = list(tasks)
            closed[n] = json.dumps(
                dict(obj(tasks[n]), state="done", evidence="x", closed="2026-09-03 00:00"), ensure_ascii=False
            )
            s_hits = subsume(tasks, closed, tkey)
            k_hits = key_present(tasks, closed, tkey)
            ok = bool(s_hits) and not k_hits
            bad += not ok
            print(
                f"  {'ok  ' if ok else 'BUG '} `harness task done` in place on {_k}: subsume "
                f"refuses ({len(s_hits)}), key_present passes ({len(k_hits)}) -- why tasks is "
                f"not subsume"
            )

    # de-39's EXEMPTION, on four worlds. ledger_audit._superseded_by_ruling excuses a lost row
    # only when it is the row a recorded ruling SUPERSEDED and the row the ruling installed is
    # still present. It shipped at 54c77d5e with no test anywhere -- grep found it in
    # ledger_audit.py and nothing else -- which is the "a helper that works and is not called"
    # shape: the logic is right and nothing asserts the WIRING, so a future edit to
    # regressions() could drop the filter and every selftest would stay green.
    #
    # MUTATION-TESTED, four mutations of _superseded_by_ruling against these worlds: the filter
    # removed (red, world A), the replacement-present half made unconditional (red, world C), the
    # no-ruling case excused (red, world D). ONE SURVIVES AND IS UNOBSERVABLE BY CONSTRUCTION:
    # disabling `if fingerprint(lost) == local_fp: return False` changes nothing, because a lost
    # row that IS the ruling's current row leaves no row at local_fp behind, so the second half
    # refuses anyway. It could only bite on a file holding TWO rows at one fingerprint, i.e. two
    # byte-identical rows, which subsume treats as one -- so that drop is not a loss to begin
    # with. The first half is a redundant guard, not an untested one; recorded here rather than
    # papered over with a world that cannot exist.
    #
    # Built from a REAL ruling and the real ledger it names, so the fingerprints are the ones
    # the ruling actually recorded rather than values this test computed for itself.
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ledger_audit as _la2
        import ledger_resolutions as _lr
    except Exception as e:  # noqa: BLE001
        print(f"\n  SKIP de-39 exemption: not importable ({e})")
    else:
        rulings = [(led, key, r) for led, key, r in _lr.load() if led in _la2.KEYS]
        if not rulings:
            print("\n  SKIP de-39 exemption: no ruling names a keyed ledger")
        else:
            led, key, ruling = rulings[0]
            real_rows = lines("main", led)
            kf = _la2.KEYS[led]
            cur = [ln for ln in (real_rows or [])
                   if (o := obj(ln)) is not None and _lr.fingerprint(o) == ruling.get("local_fp")]
            if not cur:
                print(f"\n  SKIP de-39 exemption: {led} holds no row at the ruling's local_fp "
                      f"{ruling.get('local_fp')} -- the ruling is stale, which settled() reports")
            else:
                keep = cur[0]
                # The SUPERSEDED row: any other row under the same key. Its fingerprint is not
                # the ruling's local_fp, so it is the one the exemption may excuse.
                others = [ln for ln in real_rows
                          if (o := obj(ln)) is not None and kf(o) == key and ln != keep]
                if not others:
                    print(f"\n  SKIP de-39 exemption: {led} key {key} has no second row to "
                          f"supersede in this clone")
                else:
                    # THE SUPERSEDED ROW MUST BE LAST UNDER ITS KEY, or the world is vacuous.
                    # subsume compares only the LAST head row per key (:98), and in the real
                    # file the superseded row comes FIRST -- so dropping it flags nothing with
                    # or without the exemption. My first version of this test did exactly that
                    # and passed with the filter mutated away, which is the GREEN-BUG this
                    # ordering fixes: measured, `running` fp e1bcb9ce is row 1 and the ruling's
                    # `killed` fp 47301bc1 is row 2.
                    sup = others[-1]
                    rest = [ln for ln in real_rows if ln not in (sup, keep)]
                    head_txt = "\n".join(rest + [keep, sup]) + "\n"
                    # World A: the superseded row leaves, the ruling's current row stays -> EXCUSED.
                    a = "\n".join(rest + [keep]) + "\n"
                    hits_a = _la2.regressions(led, head_txt, a)
                    ok_a = not any(k == key for k, _ in hits_a)
                    # And the world must be non-vacuous: WITHOUT the exemption this same drop
                    # must be flagged, or "not flagged" says nothing about the exemption.
                    raw_a = _la2.PREDICATE.get(led, _la2.subsume)(
                        [json.dumps(r) for r in _la2._rows(led, head_txt)],
                        [json.dumps(r) for r in _la2._rows(led, a)], kf)
                    live_a = any(k == key for k, _ in raw_a)
                    # World B: the ruling's OWN current row leaves -> REFUSED, ruling or not.
                    # Its head must put `keep` last, for the same reason world A puts `sup` last.
                    head_b = "\n".join(rest + [sup, keep]) + "\n"
                    b = "\n".join(rest + [sup]) + "\n"
                    hits_b = _la2.regressions(led, head_b, b)
                    ok_b = any(k == key for k, _ in hits_b)
                    # World C: the superseded row leaves AND the ruling's current row is not
                    # there either -- REFUSED, because the exemption's second half requires the
                    # replacement to be present. Without this world, mutating that half to
                    # `return True` (any ruling excuses any loss) leaves A and B both green:
                    # measured, A is excused either way and B is caught by the first half.
                    head_c = "\n".join(rest + [keep, sup]) + "\n"
                    c = "\n".join(rest) + "\n"
                    hits_c = _la2.regressions(led, head_c, c)
                    ok_c = any(k == key for k, _ in hits_c)
                    # World D: an UNRULED key loses EVERY row -- REFUSED, because a key no ruling
                    # names is not covered by the exemption at all. Without this world, mutating
                    # `if ruling is None: return False` to `return True` (no ruling excuses
                    # everything, i.e. the guard is off for every unruled key) leaves A-C green,
                    # since all three are about the one key that IS ruled.
                    #
                    # EVERY row of the key, not just its last: my first version dropped only the
                    # last and read 0 hits, because subsume asks whether the head's last row is
                    # subsumed by ANY index row under that key -- and for a key whose earlier
                    # rows are supersets, one of them satisfies it. Dropping the key entirely is
                    # the state the predicate is actually about.
                    ok_d = None
                    other_keys = [kf(o) for ln in real_rows
                                  if (o := obj(ln)) is not None and kf(o) != key]
                    if other_keys:
                        dkey = other_keys[-1]
                        d_rows = [ln for ln in real_rows
                                  if (o := obj(ln)) is not None and kf(o) == dkey]
                        base = [ln for ln in real_rows if ln not in d_rows and ln not in (keep, sup)]
                        head_d = "\n".join(base + [keep, sup] + d_rows) + "\n"
                        d_txt = "\n".join(base + [keep, sup]) + "\n"
                        hits_d = _la2.regressions(led, head_d, d_txt)
                        ok_d = any(k == dkey for k, _ in hits_d)
                    bad += not ok_a
                    bad += not live_a
                    bad += not ok_b
                    bad += not ok_c
                    if ok_d is not None:
                        bad += not ok_d
                    print()
                    print(f"  {'ok  ' if live_a else 'BUG '} de-39: the world is NON-VACUOUS -- the "
                          f"raw predicate does flag this drop ({len(raw_a)} hit(s)), so the "
                          f"exemption is what excuses it")
                    print(f"  {'ok  ' if ok_a else 'BUG '} de-39: the row a ruling SUPERSEDED may "
                          f"leave ({led} key {key}) -- {len(hits_a)} hit(s), none on this key")
                    print(f"  {'ok  ' if ok_b else 'BUG '} de-39: the ruling's OWN current row may "
                          f"NOT leave -- {len(hits_b)} hit(s), this key among them")
                    print(f"  {'ok  ' if ok_c else 'BUG '} de-39: with the replacement ALSO gone the "
                          f"superseded row may NOT leave -- {len(hits_c)} hit(s), this key among them")
                    if ok_d is None:
                        print("  SKIP de-39: no unruled key in this clone to test the "
                              "no-ruling-excuses-nothing half")
                    else:
                        print(f"  {'ok  ' if ok_d else 'BUG '} de-39: an UNRULED key may not lose its "
                              f"row -- {len(hits_d)} hit(s), that key among them")
                    if not (ok_a and ok_b and ok_c and live_a and ok_d is not False):
                        print("       the exemption is either too wide (a ruling authorises "
                              "losing the live record), dead (regressions() lost the filter), "
                              "or the world never reached it")

    print(f"\nde-33 ledger predicates: {'PASS' if not bad else f'{bad} BUG(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(selftest())
