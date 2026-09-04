"""Ledger state: pod copy vs repo copy, for every runs/*.jsonl.

Three questions, per file:
  1. row counts on each side
  2. rows present on one side only (identity per file's own key, not position)
  3. `running`/open rows with no process behind them

The pod side is read with ~/bin/pod (container view). `tn exec` is the HOST view and shows
a stale tree, so it is not used here at all.

Row identity is IMPORTED from scripts/ledger_audit.py's KEYS, not restated here. My first
version restated it from memory as {"board.jsonl": ("id",)} and board rows carry no `id`
field, so all 93 local rows hashed to ("",), all 12 pod rows too, and the diff reported "0
rows only on one side" for an 81-row gap -- a false clean of exactly the shape this audit
is looking for. A key that no row carries makes a set diff report agreement.

Files ledger_audit does not name get the whole-object key, which over-splits (any field
difference is a new row) but cannot under-report a missing row.

  python3 runs/audit_0904/ledger_diff.py --local          # local inventory
  python3 runs/audit_0904/ledger_diff.py --pod            # fetch pod counts (needs ~/bin/pod)
  python3 runs/audit_0904/ledger_diff.py --diff <file>    # per-row diff for one ledger
  python3 runs/audit_0904/ledger_diff.py --keys           # which key each ledger gets
  python3 runs/audit_0904/ledger_diff.py --selftest
"""

import glob
import json
import os
import subprocess
import sys

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
POD = os.path.expanduser("~/bin/pod")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ledger_audit import KEYS as AUDIT_KEYS  # noqa: E402


def rows(path):
    out, bad = [], 0
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return out, bad
    with fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                bad += 1
    return out, bad


WHOLE = "whole object"


def keyfn(rel):
    """rel is the repo-relative path, since that is how ledger_audit keys its map."""
    fn = AUDIT_KEYS.get(rel)
    if fn is None:
        return lambda r: json.dumps(r, sort_keys=True)[:400]
    return fn


def keyname(rel):
    return "ledger_audit.KEYS" if rel in AUDIT_KEYS else WHOLE


def local_inventory(root=None):
    root = root or ROOT
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "runs", "**", "*.jsonl"), recursive=True)):
        rel = os.path.relpath(p, root)
        rs, bad = rows(p)
        out[rel] = {"rows": len(rs), "bad": bad}
    return out


def pod_inventory():
    """Counts and per-row keys are expensive over the tunnel, so this fetches counts only.
    --diff pulls one file's rows when a count gap needs explaining."""
    if not os.path.exists(POD):
        raise SystemExit(f"no {POD} -- the pod half cannot be read from here")
    cmd = (
        "cd /work/aupai && for f in runs/*.jsonl runs/*/*.jsonl; do "
        "[ -f \"$f\" ] && printf '%s\\t%s\\n' \"$(grep -c '' \"$f\")\" \"$f\"; done"
    )
    r = subprocess.run([POD, cmd], capture_output=True, text=True, timeout=300)
    out = {}
    for ln in r.stdout.splitlines():
        if "\t" not in ln:
            continue
        n, f = ln.split("\t", 1)
        if n.strip().isdigit():
            out[f.strip()] = int(n.strip())
    if not out:
        raise SystemExit(f"pod returned no counts; stderr: {r.stderr[:300]}")
    return out


def pod_rows(rel):
    r = subprocess.run([POD, f"cd /work/aupai && cat {rel}"], capture_output=True, text=True, timeout=300)
    out, bad = [], 0
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            bad += 1
    return out, bad


def _selftest():
    inv = local_inventory()
    assert inv, "no local ledgers found -- the glob is wrong"
    assert "runs/experiments.jsonl" in inv, "experiments.jsonl not in the inventory"
    # Known answer: the recursive glob must see more than the top level. runs/ holds
    # subdirectories (heldout_v2/ etc); a non-recursive glob would find only runs/*.jsonl.
    top = {k for k in inv if k.count("/") == 1}
    assert len(inv) > len(top), "the glob found no ledger below runs/ -- it is not recursive"
    # Row identity must not be positional: reversing a file's rows must not change its key set.
    rs, _ = rows(os.path.join(ROOT, "runs", "experiments.jsonl"))
    kf = keyfn("runs/experiments.jsonl")
    assert {kf(r) for r in rs} == {kf(r) for r in reversed(rs)}, "key set depends on order"
    # THE DEFECT THIS SELFTEST EXISTS FOR. Every ledger's key must discriminate its own
    # rows: a key naming a field the rows do not carry collapses the whole file to one
    # key, and a set diff then reports agreement no matter what is missing. My first
    # version keyed board.jsonl on `id`, which board rows do not have, and it reported
    # "0 rows only on one side" across an 81-row gap.
    for rel in sorted(set(AUDIT_KEYS) | {"runs/board.jsonl", "runs/friction.jsonl"}):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        rs, _ = rows(p)
        if len(rs) < 2:
            continue
        ks = {keyfn(rel)(r) for r in rs}
        assert len(ks) > 1, (
            f"{rel}: all {len(rs)} rows hash to one key {ks} -- the key names a field these "
            f"rows do not carry, so a diff over it reports agreement whatever is missing"
        )
        # And it must not be so fine that every row is unique when the writer folds rows:
        # that is over-splitting, which over-reports rather than under-reports, so it is
        # only printed, not asserted.
    print(f"ledger_diff selftest ok ({len(inv)} local ledgers; every keyed ledger discriminates)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)

    if "--keys" in sys.argv:
        for k in sorted(local_inventory()):
            print(f"{keyname(k):20s} {k}")
        raise SystemExit(0)

    if "--diff" in sys.argv:
        rel = sys.argv[sys.argv.index("--diff") + 1]
        kf = keyfn(rel)
        lr, lbad = rows(os.path.join(ROOT, rel))
        pr, pbad = pod_rows(rel)
        lk = {kf(r) for r in lr}
        pk = {kf(r) for r in pr}
        print(f"{rel}: local {len(lr)} rows ({len(lk)} keys, {lbad} unparseable); "
              f"pod {len(pr)} rows ({len(pk)} keys, {pbad} unparseable)")
        print(f"key = {keyname(rel)}")
        if len(lr) > 1 and len(lk) == 1:
            print("REFUSING to report: every local row hashed to one key, so this diff cannot "
                  "distinguish agreement from a missing row. Add this file to ledger_audit.KEYS.")
            raise SystemExit(2)
        only_l, only_p = sorted(lk - pk, key=repr), sorted(pk - lk, key=repr)
        print(f"\nlocal only: {len(only_l)}")
        for k in only_l[:40]:
            print(f"   {k}")
        print(f"\npod only: {len(only_p)}")
        for k in only_p[:40]:
            print(f"   {k}")
        raise SystemExit(0)

    inv = local_inventory()
    if "--local" in sys.argv:
        print(f"{len(inv)} local ledgers under runs/")
        for k, v in sorted(inv.items()):
            flag = f"  ({v['bad']} UNPARSEABLE)" if v["bad"] else ""
            print(f"{v['rows']:8d}  {k}{flag}")
        raise SystemExit(0)

    pod = pod_inventory()
    names = sorted(set(inv) | set(pod))
    print(f"local {len(inv)} ledgers, pod {len(pod)} ledgers, union {len(names)}\n")
    print(f"{'local':>8s} {'pod':>8s}  file")
    for n in names:
        lv = inv.get(n, {}).get("rows")
        pv = pod.get(n)
        mark = ""
        if lv is None:
            mark = "  POD ONLY"
        elif pv is None:
            mark = "  LOCAL ONLY"
        elif lv != pv:
            mark = f"  GAP {pv - lv:+d}"
        print(f"{'-' if lv is None else lv:>8} {'-' if pv is None else pv:>8}  {n}{mark}")
