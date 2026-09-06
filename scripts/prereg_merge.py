#!/usr/bin/env python3
# restartable: a merge driver. It reads %O/%A/%B and writes %A only on a clean union; on any
# refusal it writes nothing and exits 1, which is the same state as never having run -- git
# leaves the conflict for a person either way. An interrupt mid-write costs a `git merge
# --abort` and a re-run, and cannot half-apply a merge, because the write is the last step.
"""git merge driver for runs/prereg.jsonl: pair rows by id, union their keys.

WHY NOT merge=union. union is a LINE merge: it keeps both sides' lines, so two edits to one
prereg row produce two lines with the same id. prereg.jsonl is amended in place -- one row per
id, keys added over time -- so the line-level answer is always wrong for it, which is why
.gitattributes deliberately leaves it unspecified and the pre-commit exemption excludes it.

THE DEADLOCK THIS EXISTS TO BREAK (b0, 2026-09-05, measured). A worktree behind main that has
edited prereg.jsonl cannot move: `git merge` refuses ("local changes would be overwritten,
commit or stash them"), and the pre-commit hook refuses the commit that would satisfy it
("this tree is N commits behind main"), because the union exemption correctly does not cover
this file. Both rules are right; their intersection is empty. `git stash` is not a third way
out -- .git/refs/stash is shared across every worktree here -- and `git merge --autostash` is
worse: measured, it writes refs/stash anyway, and when the re-apply conflicts it leaves the
work ONLY in the stash.

WHAT THIS DOES: rows are paired by `id`. A key present on one side is taken. A key both sides
carry with EQUAL values is taken once.

WHAT IT REFUSES: a key both sides carry with DIFFERENT values. That is a real disagreement
about a registration and a person has to settle it -- resolving it by rule would silently
overwrite somebody's amendment, which is more dangerous than the deadlock. It exits 1 with
conflict markers, exactly as a normal driver does.

Measured on the real case b0 hit: moe_0905 51 keys + 53 -> 53, conversion_rate_0905 54 + 50
-> 54, zero keys held at different values on both sides. Each side led on a DIFFERENT row, so
--ours would have dropped 4 keys and --theirs 2; the key union drops none and needs no ruling.

git calls this as: driver = prereg_merge.py %O %A %B %L %P
  %O ancestor  %A ours (and the file to WRITE)  %B theirs  %L conflict-marker size  %P real path
"""

import json
import sys


def _rows(path):
    """[(id, obj)] in file order, or None if the file is not readable as one-JSON-per-line.

    None rather than an exception: a driver that crashes makes git report a merge failure with
    a traceback, and the caller cannot tell "I refuse this content" from "the driver is broken".
    Returning None routes to the same clean exit-1 as a real conflict.
    """
    try:
        out = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict) or "id" not in obj:
                    return None
                out.append((obj["id"], obj))
        return out
    except (OSError, ValueError):
        return None


def merge(ours, theirs):
    """(merged_rows, conflicts). Conflicts are (id, key, ours, theirs) held at different values.

    ORDER IS OURS-THEN-NEW-FROM-THEIRS, deliberately: the file is read by people and a merge
    that reshuffles it produces a diff nobody can review. A row only theirs has is appended.
    """
    tmap = {i: o for i, o in theirs}
    merged, conflicts, seen = [], [], set()
    for rid, o in ours:
        seen.add(rid)
        t = tmap.get(rid)
        if t is None:
            merged.append(o)
            continue
        row = dict(o)
        for k, v in t.items():
            if k not in row:
                row[k] = v
            elif row[k] != v:
                conflicts.append((rid, k, row[k], v))
        merged.append(row)
    for rid, t in theirs:
        if rid not in seen:
            merged.append(t)
    return merged, conflicts


def main(argv):
    if len(argv) < 4:
        print("usage: prereg_merge.py %O %A %B [%L %P]", file=sys.stderr)
        return 2
    _base, ours_path, theirs_path = argv[1], argv[2], argv[3]
    path = argv[5] if len(argv) > 5 else ours_path

    ours, theirs = _rows(ours_path), _rows(theirs_path)
    if ours is None or theirs is None:
        which = "ours" if ours is None else "theirs"
        print(f"prereg merge: {path}: {which} is not one JSON object per line with an 'id'; "
              f"resolve by hand", file=sys.stderr)
        return 1

    merged, conflicts = merge(ours, theirs)
    if conflicts:
        for rid, k, a, b in conflicts[:10]:
            print(f"prereg merge: {path}: row {rid!r} key {k!r} differs -- "
                  f"ours {a!r}, theirs {b!r}", file=sys.stderr)
        print(f"prereg merge: {len(conflicts)} key(s) held at different values. A registration "
              f"someone amended two ways is a decision, not a merge. Resolve by hand.",
              file=sys.stderr)
        return 1

    with open(ours_path, "w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


def selftest():
    """Three worlds side by side, because proving it CAN union is only half.

    The other half is proving it does NOT silently pick a side on a real disagreement -- a
    driver that resolved conflicts by rule would pass every union world and quietly overwrite
    somebody's amendment, which is worse than the deadlock it replaces (b0's requirement).
    """
    fails = []

    # W1: only OURS added a key.
    m, c = merge([("a", {"id": "a", "x": 1, "mine": 2})], [("a", {"id": "a", "x": 1})])
    if c or m != [{"id": "a", "x": 1, "mine": 2}]:
        fails.append(f"W1 ours-only key lost or conflicted: {m} {c}")

    # W2: only THEIRS added a key. Separate from W1: an implementation that just returns `ours`
    # passes W1 and fails here, and that is the shape --ours would have.
    m, c = merge([("a", {"id": "a", "x": 1})], [("a", {"id": "a", "x": 1, "yours": 3})])
    if c or m != [{"id": "a", "x": 1, "yours": 3}]:
        fails.append(f"W2 theirs-only key dropped: {m} {c}")

    # W3: THE CONTROL THAT MUST BE RED. Same key, different value.
    m, c = merge([("a", {"id": "a", "x": 1})], [("a", {"id": "a", "x": 2})])
    if not c:
        fails.append("W3 a key held at two different values merged silently -- this driver "
                     "would overwrite someone's amendment without saying so")
    elif c[0][:2] != ("a", "x"):
        fails.append(f"W3 conflicted, but not on the differing key: {c}")

    # W4: b0's real shape -- each side leads on a DIFFERENT row, no key in disagreement. This is
    # the case that must pass with zero rulings; --ours drops theirs' row keys and --theirs drops
    # ours'.
    m, c = merge(
        [("moe", {"id": "moe", "k": 1, "amendment_14": "x"}), ("conv", {"id": "conv", "k": 1})],
        [("moe", {"id": "moe", "k": 1}),
         ("conv", {"id": "conv", "k": 1, "amendment_11": "y", "outcome_readout_1": "z"})],
    )
    if c:
        fails.append(f"W4 b0's real case conflicted, and it must not: {c}")
    else:
        by = {r["id"]: r for r in m}
        if "amendment_14" not in by.get("moe", {}):
            fails.append("W4 lost ours' row key")
        if "outcome_readout_1" not in by.get("conv", {}):
            fails.append("W4 lost theirs' row key")

    # W5: a row only one side has survives. Amendments arrive as new rows too.
    m, c = merge([("a", {"id": "a"})], [("a", {"id": "a"}), ("b", {"id": "b"})])
    if c or len(m) != 2 or {r["id"] for r in m} != {"a", "b"}:
        fails.append(f"W5 a row present on one side only was dropped: {m} {c}")

    # W6: EQUAL values on the same key are not a conflict. Without this the driver would refuse
    # every merge, since both sides share almost every key -- and a driver that always refuses
    # satisfies W3 while being useless.
    m, c = merge([("a", {"id": "a", "x": 1, "s": "same"})],
                 [("a", {"id": "a", "x": 1, "s": "same"})])
    if c:
        fails.append(f"W6 identical keys reported as a conflict; the driver would never merge: {c}")

    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        print(f"\n{len(fails)} failure(s)")
        return 1
    print("prereg_merge selftest OK: ours-only and theirs-only keys both survive, a row on one "
          "side survives, identical values are not a conflict, b0's real two-row case needs no "
          "ruling, and a key held at two DIFFERENT values is refused rather than picked")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main(sys.argv))
