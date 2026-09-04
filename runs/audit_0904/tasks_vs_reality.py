"""runs/tasks.jsonl open rows vs reality.

The row's own fields decide what "reality" means, so this reads them rather than assuming:
`state` (not `status`), `task`, `evidence`, `blocked_on`, `closed`. My first probe asked for
`status`/`subject` and got None for all 233 ids -- every row read as open with no text.

For each open row it reports:
  age             hours since `opened` (UTC; both clocks write into this file)
  evidence        the path named, and whether it exists here
  blocked_on      the text, so a claim about a wait can be read against the artifact
  closed          set while state != done, which is a contradiction inside one row

  python3 runs/audit_0904/tasks_vs_reality.py
  python3 runs/audit_0904/tasks_vs_reality.py --owner de
  python3 runs/audit_0904/tasks_vs_reality.py --selftest
"""

import json
import os
import re
import sys
import time

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
PATH = os.path.join(ROOT, "runs", "tasks.jsonl")
DONE = {"done", "closed", "dropped", "superseded", "retracted"}


def rows(path=PATH):
    out = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def latest(rs):
    """tasks.jsonl is rewrite-style but merges by union, so an id can appear more than
    once. Last occurrence wins, which is what harness's own _write_tasks produces."""
    last = {}
    for r in rs:
        last[r.get("id")] = r
    return last


def age_hours(stamp, now=None):
    if not stamp:
        return None
    now = now or time.time()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            t = time.strptime(stamp[: len(time.strftime(fmt, time.gmtime()))], fmt)
            return (now - time.mktime(t) + time.timezone) / 3600.0
        except ValueError:
            continue
    return None


def evidence_paths(r):
    """Evidence is free text; pull out the path-shaped tokens rather than assuming one."""
    ev = r.get("evidence")
    if not ev:
        return []
    return re.findall(r"[\w./-]+\.(?:py|sh|md|json|jsonl|txt|pt|log)\b", str(ev))


def analyse(root=None):
    root = root or ROOT
    last = latest(rows(os.path.join(root, "runs", "tasks.jsonl")))
    out = []
    for tid, r in last.items():
        state = r.get("state")
        if state in DONE:
            continue
        evs = evidence_paths(r)
        missing = [p for p in evs if not os.path.exists(os.path.join(root, p))]
        out.append(
            {
                "id": tid,
                "owner": r.get("owner"),
                "state": state,
                "opened": r.get("opened"),
                "age_h": age_hours(r.get("opened")),
                "task": (r.get("task") or "")[:110],
                "blocked_on": r.get("blocked_on"),
                "evidence": evs,
                "evidence_missing": missing,
                "closed_but_open": bool(r.get("closed")) and state not in DONE,
                "no_evidence": not evs,
            }
        )
    return out, last


def _selftest():
    rs = rows()
    assert len(rs) > 200, f"only {len(rs)} rows read"
    last = latest(rs)
    # THE DEFECT THIS EXISTS FOR: the field is `state`, not `status`, and reading the wrong
    # name silently makes every row look open with no text. Assert both halves.
    assert any("state" in r for r in rs), "no row carries `state` -- the schema moved"
    assert not any("status" in r for r in rs), "a row carries `status`; this reader ignores it"
    states = {r.get("state") for r in last.values()}
    assert None not in states, f"some row has no state: {sorted(str(s) for s in states)}"
    assert "done" in states, f"no row is done, so the open filter is not filtering: {states}"
    open_rows, _ = analyse()
    assert 0 < len(open_rows) < len(last), (
        f"{len(open_rows)} of {len(last)} open -- a filter that keeps all or none is inert"
    )
    # The age parser must produce a plausible number for a known row, not None.
    a = age_hours("2026-09-04 00:23")
    assert a is not None and 0 < a < 24 * 30, f"age parser gave {a} for a row opened today"
    print(f"tasks_vs_reality selftest ok ({len(last)} ids, {len(open_rows)} open, states {sorted(states)})")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)
    open_rows, last = analyse()
    owner = None
    if "--owner" in sys.argv:
        owner = sys.argv[sys.argv.index("--owner") + 1]
        open_rows = [r for r in open_rows if r["owner"] == owner]
    print(f"{len(last)} ids in tasks.jsonl; {len(open_rows)} not in {sorted(DONE)}"
          + (f" (owner={owner})" if owner else ""))
    by_owner = {}
    for r in open_rows:
        by_owner[r["owner"]] = by_owner.get(r["owner"], 0) + 1
    print("open by owner: " + ", ".join(f"{k}={v}" for k, v in sorted(by_owner.items())))
    print(f"\nopen rows with no evidence path      : {sum(1 for r in open_rows if r['no_evidence'])}")
    print(f"open rows whose evidence path is absent: {sum(1 for r in open_rows if r['evidence_missing'])}")
    print(f"rows with a `closed` stamp but state != done: {sum(1 for r in open_rows if r['closed_but_open'])}")
    print(f"\n{'id':11s} {'owner':8s} {'state':9s} {'age_h':>7s}  blocked_on / task")
    for r in sorted(open_rows, key=lambda x: -(x["age_h"] or 0)):
        b = f"BLOCKED[{str(r['blocked_on'])[:44]}] " if r["blocked_on"] else ""
        m = f"EVIDENCE-ABSENT{r['evidence_missing']} " if r["evidence_missing"] else ""
        c = "CLOSED-STAMP-BUT-OPEN " if r["closed_but_open"] else ""
        print(f"{str(r['id']):11s} {str(r['owner']):8s} {str(r['state']):9s} "
              f"{(r['age_h'] or 0):7.1f}  {c}{m}{b}{r['task']}")
