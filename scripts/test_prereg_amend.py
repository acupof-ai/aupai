#!/usr/bin/env python3
"""`harness prereg amend` is the only writer of a prereg amendment, and it dates every one.

4c's ruling 2026-09-05. Amendments were hand-edited into runs/prereg.jsonl for two days and the
file's own content shows what that cost: THREE conventions for where the timestamp lives, two
amendments (b0_head_hybrid_3to1 2 and 3) with no timestamp in any of them, and one row whose
amended_N sequence sat six behind its amendment_N sequence -- so a doc citing @amended_6 was
current under one family and stale under the other. One writer that assigns both suffixes in one
step is the fix; check_prereg_amendments_dated is the guard that a hand-edit reintroducing the gap
turns red.

WHY THE NUMBER IS READ, NOT PASSED. moe_0905 has no amendment 2 -- the sequence runs 1,3,4,5,6,7 --
because a human typed the next number. N here is prereg_amendment_n(row) + 1, the max suffix over
BOTH key families, which is also why an earlier version's refusal was wrong: it treated a row
written in another convention (memory_layers_0905 amended_6 / amendment_12) as ambiguous when its
next number is plainly 13.

WHAT IS TESTED. Six worlds, each a COPY of the real ledger in its own temp dir with harness.ROOT
rebound -- never the live file, which is the artifact whose integrity the check exists to protect.
A writer test that appended to a real pre-registration would be the defect it is testing for.

  W1 the write        -> both amended_N and amendment_N appear at N+1, the date is a real
                         timestamp, prereg_amendment_n advances, AND the check passes on the
                         result. That last clause is the one that matters: a writer whose output
                         its own guard rejects is two mechanisms disagreeing
  W2 one line touched -> every other row byte-for-byte identical. A writer that reserialises the
                         file rewrites 5 rows of ~20KB and buries its real change in diff noise
  W3 unknown id       -> refuses, writes nothing, names the ids that exist
  W4 blank --text     -> refuses. argparse's required=True is satisfied by "   ", so the
                         non-empty test has to be separate from the presence test
  W5 duplicate id     -> refuses. Two rows for one id is the union-merge shape that put 13 rows
                         in the file for 5 ids; amending under it would write to one and leave
                         the other, which is how the stale versions survived in the first place
  W6 twice in a row   -> N is re-read from disk between calls, so two amendments do not collide

"""
# restartable: every world is a fresh temp dir removed at the end, harness.ROOT is restored, and
# nothing reads or writes the repository's real runs/prereg.jsonl. An interrupt costs one second
# and leaves a temp dir under /tmp.
import importlib.util
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))


def _load_harness():
    spec = importlib.util.spec_from_file_location("h_prereg", os.path.join(ROOT, "scripts", "harness.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(fails):
    if fails:
        print("test_prereg_amend FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("test_prereg_amend ok: the writer assigns amended_N and amendment_N together at a "
          "number read from the row, its output satisfies prereg_amendments_dated, it touches one "
          "line, and it refuses an unknown id, a blank text and a duplicated id without writing")
    return 0


def main():
    h = _load_harness()
    real = os.path.join(ROOT, "runs", "prereg.jsonl")
    if not os.path.exists(real):
        print("test_prereg_amend SKIP: runs/prereg.jsonl absent")
        return 0
    saved_root = h.ROOT
    fails = []
    made = []

    def fresh():
        d = tempfile.mkdtemp(prefix="prereg_amend_")
        made.append(d)
        os.makedirs(os.path.join(d, "runs"))
        shutil.copy(real, os.path.join(d, "runs", "prereg.jsonl"))
        h.ROOT = d
        return d

    def rows(d):
        p = os.path.join(d, "runs", "prereg.jsonl")
        return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

    def row(d, rid):
        return next((r for r in rows(d) if r.get("id") == rid), None)

    try:
        # W1 -- the write, and the check's verdict on it
        d = fresh()
        target = next((r["id"] for r in rows(d) if r.get("id")), None)
        if target is None:
            print("test_prereg_amend SKIP: no row with an id")
            return 0
        before = h.prereg_amendment_n(row(d, target))
        rc = h.cmd_prereg(["amend", "--id", target, "--text", "test amendment", "--by", "selftest"])
        r, want = row(d, target), before + 1
        if rc != 0:
            fails.append(f"W1: rc={rc} on a real row")
        for k in (f"amended_{want}", f"amendment_{want}"):
            if k not in r:
                fails.append(f"W1: {k} absent after the write (row was at {before})")
        if f"amended_{want}" in r and not h.PREREG_STAMP_RE.search(r[f"amended_{want}"]):
            fails.append(f"W1: amended_{want} carries no timestamp: {r[f'amended_{want}']!r}")
        if h.prereg_amendment_n(r) != want:
            fails.append(f"W1: N is {h.prereg_amendment_n(r)}, want {want}")
        st, ev = h.check_prereg_amendments_dated(d)
        if st != h.PASS:
            fails.append(f"W1: the writer's output does not satisfy its own check: {st} {ev[:90]}")

        # W2 -- exactly one line changes
        d = fresh()
        p = os.path.join(d, "runs", "prereg.jsonl")
        orig = open(p, encoding="utf-8").read().split("\n")
        h.cmd_prereg(["amend", "--id", target, "--text", "t", "--by", "selftest"])
        after = open(p, encoding="utf-8").read().split("\n")
        changed = [i for i, (a, b) in enumerate(zip(orig, after)) if a != b]
        if len(changed) != 1 or len(orig) != len(after):
            fails.append(f"W2: {len(changed)} line(s) changed and {len(orig)}->{len(after)} lines, "
                         f"want 1 changed and the count unchanged")

        # W3 -- unknown id refuses and writes nothing
        d = fresh()
        p = os.path.join(d, "runs", "prereg.jsonl")
        sz = os.path.getsize(p)
        rc = h.cmd_prereg(["amend", "--id", "no_such_row_selftest", "--text", "t"])
        if rc == 0 or os.path.getsize(p) != sz:
            fails.append(f"W3: unknown id rc={rc}, size {os.path.getsize(p)} vs {sz}")

        # W4 -- whitespace-only text refuses
        d = fresh()
        p = os.path.join(d, "runs", "prereg.jsonl")
        sz = os.path.getsize(p)
        rc = h.cmd_prereg(["amend", "--id", target, "--text", "   "])
        if rc == 0 or os.path.getsize(p) != sz:
            fails.append(f"W4: blank text rc={rc}, size {os.path.getsize(p)} vs {sz}")

        # W5 -- a duplicated id refuses
        d = fresh()
        p = os.path.join(d, "runs", "prereg.jsonl")
        lines = open(p, encoding="utf-8").read().rstrip("\n").split("\n")
        dup = next(x for x in lines if json.loads(x).get("id") == target)
        open(p, "w", encoding="utf-8").write("\n".join(lines + [dup]) + "\n")
        sz = os.path.getsize(p)
        rc = h.cmd_prereg(["amend", "--id", target, "--text", "t"])
        if rc == 0 or os.path.getsize(p) != sz:
            fails.append(f"W5: duplicate id rc={rc}, size {os.path.getsize(p)} vs {sz} -- "
                         f"amending one of two rows leaves the other stale")

        # W6 -- two amendments in sequence do not collide
        d = fresh()
        h.cmd_prereg(["amend", "--id", target, "--text", "first"])
        n1 = h.prereg_amendment_n(row(d, target))
        h.cmd_prereg(["amend", "--id", target, "--text", "second"])
        n2 = h.prereg_amendment_n(row(d, target))
        r = row(d, target)
        if n2 != n1 + 1 or r.get(f"amendment_{n1}") != "first" or r.get(f"amendment_{n2}") != "second":
            fails.append(f"W6: sequential amendments landed at {n1} then {n2} with texts "
                         f"{r.get(f'amendment_{n1}')!r}/{r.get(f'amendment_{n2}')!r}")
    finally:
        h.ROOT = saved_root
        for d in made:
            shutil.rmtree(d, ignore_errors=True)

    return _report(fails)


if __name__ == "__main__":
    sys.exit(main())
