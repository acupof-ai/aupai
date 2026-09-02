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

        open_ix = [n for n, ln in enumerate(tasks) if (obj(ln) or {}).get("state") == "open"]
        if not open_ix:
            print("  SKIP tasks world: no open row on main to close")
        else:
            n = open_ix[-1]
            closed = list(tasks)
            closed[n] = json.dumps(
                dict(obj(tasks[n]), state="done", evidence="x", closed="2026-09-03 00:00"), ensure_ascii=False
            )
            s_hits = subsume(tasks, closed, tkey)
            k_hits = key_present(tasks, closed, tkey)
            ok = bool(s_hits) and not k_hits
            bad += not ok
            print(
                f"  {'ok  ' if ok else 'BUG '} `harness task done` in place: subsume refuses "
                f"({len(s_hits)}), key_present passes ({len(k_hits)}) -- why tasks is not subsume"
            )

    print(f"\nde-33 ledger predicates: {'PASS' if not bad else f'{bad} BUG(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(selftest())
