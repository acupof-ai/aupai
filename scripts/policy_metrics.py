#!/usr/bin/env python3
"""Policy effectiveness metrics -> runs/policy_metrics.jsonl, one row per UTC day.

# restartable: appends one <1KB line per run; a torn last line is skipped on read.

User order via 4c 2026-09-05: the controller confirms its policies are effective,
not assumes. Five counts, baseline 2026-09-05, each beside yesterday's. A count
the ledgers cannot carry is null with the missing field named, never an estimate.

  1 misroutes              friction rows whose cause/blocked_what name a
                           wrong-address send. The ledger has no misroute kind:
                           it carried 1 of 2026-09-05's 3 known (b0's two live in
                           §210/chat only). kind="misroute" would close that.
  2 gate refusals          the friction ledger's cumulative grouped state
                           (whole-ledger rows/causes, the figure the controller
                           quotes as "N rows / M causes"), top 3 causes.
  3 card-hours by class    NOT COMPUTABLE: folded by (name, started), cards and
                           class are on 2/245 rows (de's probes, 2026-09-05);
                           ended is on 243/245. The missing fields are cards and
                           class, not ended. The 243 pre-field rows stay null --
                           a backfilled class is a guess (4c). Field semantics
                           (test_ledger_field_writers.py): class/cards absent =
                           UNSTATED, "" forbidden; 'none' is a STATED cards
                           answer for a CPU or corpus job.
  4 defects author vs      author: friction rows with caught_by == who.
    second reader          second: review rows with a BLOCKING/BLOCKED/REJECT/FAIL
                           verdict. Ledger carried 1 of 2026-09-05's 5 second-
                           reader catches known to the controller.
  5 open tasks per owner   tasks.jsonl open rows over roster members (same
                           population as harness check one_deliverable_per_owner).
  6 message length         words/msg to the controller, from runs/msg_log.jsonl
                           (from, words, ts; scripts/msg_log.py, c1e146a6),
                           one row per peer message received. Counting starts
                           2026-09-05 09:58Z -- the 2026-09-05 baseline is
                           partial; n_msgs in the row says how partial.

Usage:
  python3 scripts/policy_metrics.py [--date YYYY-MM-DD]   # write the row
  python3 scripts/policy_metrics.py --print               # latest two rows
  python3 scripts/policy_metrics.py --selftest
"""
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "runs", "policy_metrics.jsonl")

MISROUTE_RE = re.compile(
    r"misroute|wrong address|outside the team|bare (ListAgents )?name|matched on the substring",
    re.I)
BLOCK_RE = re.compile(r"\bBLOCKING\b|\bBLOCKED\b|\bREJECT|\bFAIL\b")


def _rows(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _folded(rows, key):
    """Last row per id wins, like harness._read_tasks."""
    out = {}
    for r in rows:
        if r.get(key):
            out[r[key]] = r
    return list(out.values())


def _date(r):
    m = re.match(r"(\d{4}-\d{2}-\d{2})",
                 str(r.get("when") or r.get("ts") or r.get("at") or ""))
    return m.group(1) if m else None


def compute(date):
    friction = _rows("runs/friction.jsonl")
    today_f = [r for r in friction if _date(r) == date]

    misroutes = [r for r in today_f
                 if MISROUTE_RE.search(f"{r.get('cause') or ''} {r.get('blocked_what') or ''}")]

    causes = Counter(str(r.get("cause") or "?")[:120] for r in friction)

    exp = _rows("runs/experiments.jsonl")
    exp = list({(r.get("name"), r.get("started")): r for r in exp}.values())
    with_ended = sum(1 for r in exp if r.get("ended"))
    with_cards = sum(1 for r in exp if r.get("cards"))
    with_class = sum(1 for r in exp if r.get("class"))

    author = [r for r in today_f
              if r.get("caught_by") and r.get("who") and r["caught_by"] == r["who"]]
    second = []
    for r in _rows("runs/review.jsonl"):
        if _date(r) != date:
            continue
        text = " ".join(str(r.get(k) or "")
                        for k in ("verdict", "finding", "findings", "new_finding"))
        if BLOCK_RE.search(text):
            second.append(r)

    roster_p = os.path.join(ROOT, "runs", "roster.json")
    roster = ({m["name"] for m in json.load(open(roster_p, encoding="utf-8"))["members"]}
              if os.path.exists(roster_p) else set())
    open_by_owner = Counter()
    for t in _folded(_rows("runs/tasks.jsonl"), "id"):
        if t.get("state") == "open" and t.get("owner") in roster:
            open_by_owner[t["owner"]] += 1

    msg = [r for r in _rows("runs/msg_log.jsonl") if _date(r) == date]
    n_msg = len(msg)
    wpm = round(sum(int(r["words"]) for r in msg) / n_msg, 1) if n_msg else None

    return {
        "date": date,
        "misroutes": {
            "n": len(misroutes),
            "basis": "friction rows whose cause/blocked_what name a wrong-address send; "
                     "ledger carried 1 of 2026-09-05's 3 known (kind=misroute would catch the rest)",
        },
        "gate_refusals": {
            "rows": len(friction), "causes": len(causes),
            "top": [[c, n] for c, n in causes.most_common(3)],
        },
        "card_hours": {
            "incremental": None, "confirmatory": None, "infra_verification": None,
            "missing": f"cards on {with_cards}/{len(exp)} folded rows, class on {with_class} "
                       f"(ended on {with_ended}); the pre-field rows stay null, no backfill (4c 2026-09-05)",
        },
        "defects": {
            "author_caught": len(author), "second_reader_caught": len(second),
            "basis": "author: friction rows caught_by==who; second: review rows with a "
                     "BLOCKING/BLOCKED/REJECT/FAIL verdict; ledger carried 1 of 2026-09-05's "
                     "5 second-reader catches known to the controller",
        },
        "open_tasks_per_owner": dict(sorted(open_by_owner.items())),
        "message_length": {
            "words_per_msg_to_fb": wpm,
            "n_msgs": n_msg,
            "basis": "runs/msg_log.jsonl (from, words, ts), one row per peer message to the "
                     "controller; counting starts 2026-09-05 09:58Z, no earlier rows exist",
        },
    }


def write_row(date):
    row = compute(date)
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def print_latest():
    if not os.path.exists(LEDGER):
        print("no runs/policy_metrics.jsonl yet; run scripts/policy_metrics.py")
        return
    rows = {}
    for line in open(LEDGER, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn append from an interrupted run; the next write replaces the day
        rows[r["date"]] = r  # fold by date, last wins
    for date in sorted(rows)[-2:]:
        r = rows[date]
        print(f"  {date}:")
        print(f"    misroutes: {r['misroutes']['n']}")
        g = r["gate_refusals"]
        print(f"    gate refusals: {g['rows']} rows / {g['causes']} causes")
        c = r["card_hours"]
        card = ("incremental {i} / confirmatory {c} / infra {v} h".format(
            i=c["incremental"], c=c["confirmatory"], v=c["infra_verification"])
            if c["incremental"] is not None else f"not computable: {c['missing']}")
        print(f"    card-hours by class: {card}")
        d = r["defects"]
        print(f"    defects author/second-reader: {d['author_caught']}/{d['second_reader_caught']}")
        print(f"    open tasks per owner: {r['open_tasks_per_owner']}")
        m = r.get("message_length")
        if m:
            ml = (f"{m['words_per_msg_to_fb']} words/msg over {m['n_msgs']} msgs"
                  if m["words_per_msg_to_fb"] is not None else "no messages logged")
            print(f"    message length: {ml}")


def _selftest():
    # The two regexes are the whole logic; pin them against the real false positives.
    assert MISROUTE_RE.search("sent to the bare ListAgents name `lessons-e1`, matched on the substring"), \
        "the 3b misroute row must match"
    assert not MISROUTE_RE.search(
        "CUDA_VISIBLE_DEVICES=4 was written into the exp rows cmd field and NOT into the launch"), \
        "a benign friction row must not match as a misroute"
    assert BLOCK_RE.search("ONE BLOCKING DEFECT (entropy stop step 1000 in the row vs 500 in the charter)"), \
        "the moe_0905 blocking-defect verdict must match"
    assert not BLOCK_RE.search("memory.values.weight + blocks.1._mem_registered + blocks.3"), \
        "'blocks.1' must not read as a blocked verdict (2026-09-05 false positive)"
    assert not BLOCK_RE.search("_ROW_CHECKSUM_BLOCK is 65,536, which at d=1024 is exactly 2"), \
        "'CHECKSUM_BLOCK' must not read as a blocked verdict (2026-09-05 false positive)"
    print("policy_metrics selftest OK")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selftest" in args:
        _selftest()
    elif "--print" in args:
        print_latest()
    else:
        date = None
        if "--date" in args:
            date = args[args.index("--date") + 1]
        else:
            date = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d")
        row = write_row(date)
        print(f"wrote runs/policy_metrics.jsonl row for {date}")
