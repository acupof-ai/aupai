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
                           An owner outside the roster is reported by name in
                           open_tasks_unknown_owner, never dropped: the filter used
                           to make such a row vanish, so the count read as complete
                           while excluding work (4c, 2026-09-07; found by 58's
                           roster sweep, which hit `db` -- the ListAgents label for
                           roster name `de` -- in review.jsonl).
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

sys.path.insert(0, os.path.join(ROOT, "scripts"))
# THE FOLD IS IMPORTED, NOT RE-DERIVED (de-71, 2026-09-07). This file used to carry its own
# `_folded` with the comment "Last row per id wins, like harness._read_tasks" -- a copy documented
# as a copy, agreeing with the original by coincidence. It had no importable home until
# harness_core existed; now it does, and the copy is gone.
#
# ONE BEHAVIOUR DIFFERENCE, stated because it is latent rather than absent: the old local `_folded`
# had `if r.get(key)` and DROPPED a row with no id, while fold_by_id keeps it under the key None.
# Measured on runs/tasks.jsonl 2026-09-07: 0 rows carry no id, so metric 5 is unchanged today. If a
# keyless row ever lands, it now folds into one None bucket and can reach open_tasks_unknown_owner
# instead of vanishing -- which is the direction this file was already fixed in an hour earlier, so
# the divergence resolves toward reporting rather than silence.
from harness_core import fold_by_id as _folded  # noqa: E402
from harness_core import refuse_in_integration_tree  # noqa: E402

MISROUTE_RE = re.compile(
    r"misroute|wrong address|outside the team|bare (ListAgents )?name|matched on the substring", re.I
)
BLOCK_RE = re.compile(r"\bBLOCKING\b|\bBLOCKED\b|\bREJECT|\bFAIL\b")


def _rows(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _split_open_owners(tasks, roster):
    """Open rows counted per roster member, with non-roster owners kept beside them by name."""
    known, unknown = Counter(), Counter()
    for t in tasks:
        if t.get("state") != "open":
            continue
        owner = t.get("owner")
        if not owner:
            continue
        (known if owner in roster else unknown)[str(owner)] += 1
    return known, unknown


def _date(r):
    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(r.get("when") or r.get("ts") or r.get("at") or ""))
    return m.group(1) if m else None


INCIDENT_HEAD_RE = re.compile(r"^### §\d+ \((\d{4}-\d{2}-\d{2})", re.M)


def _facts_measured(date):
    import glob

    n = 0
    for f in glob.glob(os.path.join(ROOT, "facts", "*.json")):
        try:
            obj = json.load(open(f, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ents = obj if isinstance(obj, list) else obj.get("entries", obj.get("facts", []))
        if isinstance(ents, dict):
            ents = list(ents.values())
        n += sum(
            1
            for e in ents
            if isinstance(e, dict) and e.get("status") == "measured" and str(e.get("measured")) == date
        )
    return n


def _commits_on_main(date):
    import subprocess

    try:
        out = subprocess.run(
            [
                "git",
                "log",
                "main",
                "--format=%cd",
                "--date=short-local",
                f"--since={date}T00:00:00Z",
                f"--until={date}T23:59:59Z",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
            # --since/--until below are stated in Z, so the rendered %cd must be UTC too or the
            # window and the label disagree by the machine's offset (+08:00 here).
            env={**os.environ, "TZ": "UTC"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None if out.returncode else len(out.stdout.split())


def _throughput(date, friction, exp):
    """User order 2026-09-07 (via 4c): does any policy reduce problems or speed up results.
    Six counts per UTC day, two ratios, all read from ledgers; a partial day says so."""
    import datetime as dt

    today_f = [r for r in friction if _date(r) == date]
    minutes = sum(float(r.get("minutes_lost") or 0) for r in today_f)
    inc_p = os.path.join(ROOT, "docs", "lessons", "gate_failure_incidents.md")
    incidents = (
        sum(1 for m in INCIDENT_HEAD_RE.finditer(open(inc_p, encoding="utf-8").read()) if m.group(1) == date)
        if os.path.exists(inc_p)
        else None
    )
    started = sum(1 for r in exp if str(r.get("started") or "").startswith(date))
    results = sum(
        1
        for r in exp
        if r.get("status") in ("ok", "done")
        and r.get("result")
        and str(r.get("ended") or r.get("started") or "").startswith(date)
    )
    now = dt.datetime.now(dt.UTC)
    hours = 24.0
    if now.strftime("%Y-%m-%d") == date:
        hours = max(now.hour + now.minute / 60.0, 1 / 60.0)
    return {
        "friction_rows": len(today_f),
        "minutes_lost": round(minutes),
        "incidents": incidents,
        "experiments_started": started,
        "experiments_with_result": results,
        "facts_measured": _facts_measured(date),
        "commits_on_main": _commits_on_main(date),
        "hours_elapsed": round(hours, 1),
        "minutes_lost_per_hour": round(minutes / hours, 1),
        "results_per_24h": round(results * 24.0 / hours, 1),
        "basis": "friction/experiments/facts ledgers and gate_failure_incidents.md headings by UTC "
        "date; git log main by committer date; a partial day is normalised by hours_elapsed",
    }


def compute(date):
    friction = _rows("runs/friction.jsonl")
    today_f = [r for r in friction if _date(r) == date]

    misroutes = [
        r for r in today_f if MISROUTE_RE.search(f"{r.get('cause') or ''} {r.get('blocked_what') or ''}")
    ]

    causes = Counter(str(r.get("cause") or "?")[:120] for r in friction)

    exp = _rows("runs/experiments.jsonl")
    exp = list({(r.get("name"), r.get("started")): r for r in exp}.values())
    with_ended = sum(1 for r in exp if r.get("ended"))
    with_cards = sum(1 for r in exp if r.get("cards"))
    with_class = sum(1 for r in exp if r.get("class"))

    author = [r for r in today_f if r.get("caught_by") and r.get("who") and r["caught_by"] == r["who"]]
    second = []
    for r in _rows("runs/review.jsonl"):
        if _date(r) != date:
            continue
        text = " ".join(str(r.get(k) or "") for k in ("verdict", "finding", "findings", "new_finding"))
        if BLOCK_RE.search(text):
            second.append(r)

    roster_p = os.path.join(ROOT, "runs", "roster.json")
    roster = (
        {m["name"] for m in json.load(open(roster_p, encoding="utf-8"))["members"]}
        if os.path.exists(roster_p)
        else set()
    )
    open_by_owner, open_unknown = _split_open_owners(_folded(_rows("runs/tasks.jsonl"), "id"), roster)

    throughput = _throughput(date, friction, exp)

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
            "rows": len(friction),
            "causes": len(causes),
            "top": [[c, n] for c, n in causes.most_common(3)],
        },
        "card_hours": {
            "incremental": None,
            "confirmatory": None,
            "infra_verification": None,
            "missing": f"cards on {with_cards}/{len(exp)} folded rows, class on {with_class} "
            f"(ended on {with_ended}); the pre-field rows stay null, no backfill (4c 2026-09-05)",
        },
        "defects": {
            "author_caught": len(author),
            "second_reader_caught": len(second),
            "basis": "author: friction rows caught_by==who; second: review rows with a "
            "BLOCKING/BLOCKED/REJECT/FAIL verdict; ledger carried 1 of 2026-09-05's "
            "5 second-reader catches known to the controller",
        },
        "open_tasks_per_owner": dict(sorted(open_by_owner.items())),
        "open_tasks_unknown_owner": dict(sorted(open_unknown.items())),
        "throughput": throughput,
        "message_length": {
            "words_per_msg_to_fb": wpm,
            "n_msgs": n_msg,
            "basis": "runs/msg_log.jsonl (from, words, ts), one row per peer message to the "
            "controller; counting starts 2026-09-05 09:58Z, no earlier rows exist",
        },
    }


def write_row(date):
    row = compute(date)
    if refuse_in_integration_tree("appending to policy_metrics.jsonl", path=LEDGER):
        raise SystemExit(1)
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
        card = (
            "incremental {i} / confirmatory {c} / infra {v} h".format(
                i=c["incremental"], c=c["confirmatory"], v=c["infra_verification"]
            )
            if c["incremental"] is not None
            else f"not computable: {c['missing']}"
        )
        print(f"    card-hours by class: {card}")
        d = r["defects"]
        print(f"    defects author/second-reader: {d['author_caught']}/{d['second_reader_caught']}")
        print(f"    open tasks per owner: {r['open_tasks_per_owner']}")
        unk = r.get("open_tasks_unknown_owner")
        if unk:
            print(f"      non-roster owners: {unk}")
        t = r.get("throughput")
        if t:
            print(
                f"    throughput ({t['hours_elapsed']} h): friction {t['friction_rows']} rows / "
                f"{t['minutes_lost']} min ({t['minutes_lost_per_hour']}/h), incidents {t['incidents']}, "
                f"experiments {t['experiments_started']} started / {t['experiments_with_result']} with "
                f"result ({t['results_per_24h']}/24h), facts {t['facts_measured']}, commits {t['commits_on_main']}"
            )
        m = r.get("message_length")
        if m:
            ml = (
                f"{m['words_per_msg_to_fb']} words/msg over {m['n_msgs']} msgs"
                if m["words_per_msg_to_fb"] is not None
                else "no messages logged"
            )
            print(f"    message length: {ml}")


def _selftest():
    # The two regexes are the whole logic; pin them against the real false positives.
    assert MISROUTE_RE.search("sent to the bare ListAgents name `lessons-e1`, matched on the substring"), (
        "the 3b misroute row must match"
    )
    assert not MISROUTE_RE.search(
        "CUDA_VISIBLE_DEVICES=4 was written into the exp rows cmd field and NOT into the launch"
    ), "a benign friction row must not match as a misroute"
    assert BLOCK_RE.search("ONE BLOCKING DEFECT (entropy stop step 1000 in the row vs 500 in the charter)"), (
        "the moe_0905 blocking-defect verdict must match"
    )
    assert not BLOCK_RE.search("memory.values.weight + blocks.1._mem_registered + blocks.3"), (
        "'blocks.1' must not read as a blocked verdict (2026-09-05 false positive)"
    )
    assert not BLOCK_RE.search("_ROW_CHECKSUM_BLOCK is 65,536, which at d=1024 is exactly 2"), (
        "'CHECKSUM_BLOCK' must not read as a blocked verdict (2026-09-05 false positive)"
    )

    # open_tasks_unknown_owner is empty on today's real ledger, so the live run cannot tell a
    # working split from a dead branch -- the vacuous-population shape. Fixture instead: `db` is
    # the real misnaming (ListAgents label for roster name `de`) that motivated the field.
    roster = {"de", "b0"}
    known, unknown = _split_open_owners(
        [
            {"state": "open", "owner": "de"},
            {"state": "open", "owner": "de"},
            {"state": "open", "owner": "db"},
            {"state": "open"},
            {"state": "done", "owner": "zz"},
        ],
        roster,
    )
    assert dict(known) == {"de": 2}, known
    assert dict(unknown) == {"db": 1}, unknown
    assert not _split_open_owners([{"state": "open", "owner": "db"}], roster)[0], (
        "a non-roster owner must not key open_tasks_per_owner"
    )

    assert [
        m.group(1)
        for m in INCIDENT_HEAD_RE.finditer(
            "### §261 (2026-09-07, R2) a title\nbody\n### §9 (2026-09-01) x\n#### §5 (2026-09-02) not a heading\n"
        )
    ] == ["2026-09-07", "2026-09-01"], "incident heading date parse"
    t = _throughput(
        "2026-09-05",
        [{"when": "2026-09-05 10:00", "minutes_lost": 30}, {"when": "2026-09-06 10:00", "minutes_lost": 99}],
        [
            {"started": "2026-09-05 01:00", "ended": "2026-09-05 02:00", "status": "ok", "result": "x"},
            {"started": "2026-09-05 03:00", "status": "running"},
        ],
    )
    assert (
        t["friction_rows"],
        t["minutes_lost"],
        t["experiments_started"],
        t["experiments_with_result"],
    ) == (1, 30, 2, 1), t
    assert t["hours_elapsed"] == 24.0 and t["results_per_24h"] == 1.0, t

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
            date = (
                __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d")
            )
        row = write_row(date)
        print(f"wrote runs/policy_metrics.jsonl row for {date}")
