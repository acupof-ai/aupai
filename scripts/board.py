#!/usr/bin/env python3
"""Shared board: topics anyone can post to and everyone can read.

    board.py who                          the roster: name, socket, what they own
    board.py topics                       one line per topic, newest state first
    board.py show <topic>                 that topic's rows, oldest first
    board.py post <topic> <kind> <text> [--artifact P] [--who W]
    board.py feed [-n N]                  everything, newest first
    board.py brief                        what others posted since you last looked
    board.py open <topic> --owner W --question Q

kind: find | rule | block | done | ask | note
A find/rule/done without --artifact is refused: a claim nobody can check is chatter.

Every command prints what others posted since this session last ran one, so
reading the board is a side effect of writing to it. Nobody has to remember.

restartable: one append per call to a union-merged JSONL; an interrupt loses
at most the row being written.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(ROOT, "runs", "board.jsonl")
NEEDS_ARTIFACT = {"find", "rule", "done"}
KINDS = ("find", "rule", "block", "done", "ask", "note", "open")


def rows():
    if not os.path.exists(BOARD):
        return []
    out = []
    with open(BOARD, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return out


def whoami():
    for var in ("AUPAI_SESSION", "CLAUDE_SESSION_NAME"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def append(row):
    # ONE IMPLEMENTATION OF THE RULE, IMPORTED, NEVER COPIED. The integration-tree refusal
    # lives in harness.refuse_in_integration_tree because tasks.jsonl and friction.jsonl reach
    # disk through harness; board.jsonl is the third ledger and 44 wrote a row into the
    # integration tree ten minutes after b0's task row (4c, 2026-09-05). A ten-line copy here
    # would be a second writer of one rule, which is how FRICTION_KINDS ended up rejecting a
    # kind this repo's own merge_main.sh emits.
    #
    # Imported INSIDE the function: harness imports board (`from board import liveness`), so a
    # module-level import here would be a cycle, and `board.py who` should not pay for harness
    # to answer a question that does not write anything.
    try:
        from harness import refuse_in_integration_tree
    except Exception as e:
        print(f"board: integration-tree guard unavailable ({type(e).__name__}: {e}); "
              f"writing anyway -- check the branch by hand", file=sys.stderr)
    else:
        if refuse_in_integration_tree("posting to board.jsonl", path=BOARD):
            raise SystemExit(1)
    os.makedirs(os.path.dirname(BOARD), exist_ok=True)
    with open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


SEEN = os.path.join(ROOT, "runs", ".board_seen.json")


def broadcast(me):
    rs = rows()
    seen = {}
    if os.path.exists(SEEN):
        try:
            seen = json.load(open(SEEN, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    mark = seen.get(me, 0)
    fresh = [r for i, r in enumerate(rs) if i >= mark and r["who"] != me]
    if fresh:
        print(f"--- {len(fresh)} new since you last looked ---")
        for r in fresh[-12:]:
            art = f"  [{r['artifact']}]" if r.get("artifact") else ""
            print(f"{r['ts']}  {r['topic']:<16}{r['who']:<10}{r['kind']:<6}{r['text'][:96]}{art}")
        blocked = [r for r in fresh if r["kind"] == "block"]
        if blocked:
            print(f"--- {len(blocked)} of them are BLOCK ---")
        print("---")
    seen[me] = len(rs)
    try:
        json.dump(seen, open(SEEN, "w", encoding="utf-8"))
    except OSError:
        pass


def cmd_post(a):
    if a.kind not in KINDS:
        sys.exit(f"kind must be one of {', '.join(KINDS)}")
    if a.kind in NEEDS_ARTIFACT and not a.artifact:
        sys.exit(f"a '{a.kind}' needs --artifact: a claim nobody can check is chatter")
    known = {r["topic"] for r in rows()}
    if a.topic not in known and a.kind != "open":
        sys.exit(f"no topic '{a.topic}'. open it first, or: {', '.join(sorted(known)) or '(none)'}")
    append({"ts": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "who": a.who or whoami(),
            "topic": a.topic, "kind": a.kind, "text": a.text, "artifact": a.artifact or ""})
    print(f"{a.topic} <- {a.kind} by {a.who or whoami()}")


def cmd_open(a):
    append({"ts": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "who": a.who or whoami(),
            "topic": a.topic, "kind": "open", "text": a.question,
            "artifact": "", "owner": a.owner})
    print(f"opened {a.topic} (owner {a.owner})")


def liveness(root=ROOT, now=None):
    """{name: {commit_min, ledger_min, open_tasks, socket, branch}} for every roster member.

    Minutes since a member last produced anything the repo can see: their newest branch tip
    and their newest row in tasks/review/board. No new file -- git and the ledgers already
    hold it.

    EVERY BRANCH THEY OWN, not just the one named after them. b0's `b0` tip is 2026-09-02
    22:10 while `b0-ve-rownorms` is 2026-09-03 21:02: a session that pushed three minutes
    ago would read as 23 hours stale if only the eponymous branch counted, and it is the
    exact member the acceptance criterion for this feature named as the expected warning.
    So `b0` plus `b0-*`, which is the convention every member here follows.

    THE TWO CLOCKS ARE DIFFERENT AND BOTH ARE HANDLED. Ledger rows are UTC (exp.py and
    harness both write time.gmtime) and git's committerdate here is +08 local, so comparing
    either against a naive `now` is off by eight hours in opposite directions -- an hour
    count is the whole output of this function, so both are parsed to epoch seconds with
    their own offset. review.jsonl carries BOTH `2026-09-03T03:10:00Z` and
    `2026-09-03 11:37` in the same file, measured, so the parser takes both rather than
    assuming the newer one."""
    import calendar

    now = now if now is not None else time.time()
    p = os.path.join(root, "runs", "roster.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        members = [m["name"] for m in json.load(fh)["members"]]

    def _epoch_utc(s):
        s = (s or "").strip()
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return calendar.timegm(time.strptime(s, fmt))
            except ValueError:
                continue
        return None

    tips = {}
    try:
        r = subprocess.run(
            ["git", "-C", root, "for-each-ref", "--format=%(refname:short)\t%(committerdate:unix)",
             "refs/heads/"], capture_output=True, text=True, timeout=15)
        for line in r.stdout.split("\n"):
            if "\t" in line:
                ref, unix = line.split("\t", 1)
                if unix.strip().lstrip("-").isdigit():
                    tips[ref] = int(unix)
    except (OSError, subprocess.SubprocessError):
        pass

    newest = dict.fromkeys(members)
    open_tasks = dict.fromkeys(members, 0)
    state = {}
    for rel, idfields, tsfields in (
        ("runs/tasks.jsonl", ("owner",), ("closed", "opened")),
        ("runs/review.jsonl", ("reviewer",), ("ts",)),
        ("runs/board.jsonl", ("who",), ("ts",)),
    ):
        fp = os.path.join(root, rel)
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                who = next((row.get(f) for f in idfields if row.get(f)), None)
                if who not in newest:
                    continue
                for tf in tsfields:
                    e = _epoch_utc(row.get(tf))
                    if e is not None:
                        newest[who] = e if newest[who] is None else max(newest[who], e)
                        break
                if rel.endswith("tasks.jsonl") and row.get("id"):
                    # Last state wins: a row is appended per event, so an id's newest row is
                    # its current state (the same fold exp.py and _read_tasks use).
                    state[(who, row["id"])] = row.get("state")
    for (who, _tid), st in state.items():
        if st == "open" and who in open_tasks:
            open_tasks[who] += 1

    with open(p, encoding="utf-8") as fh:
        sockets = {m["name"]: m.get("socket", "") for m in json.load(fh)["members"]}
    out = {}
    for name in members:
        mine = [t for ref, t in tips.items() if ref == name or ref.startswith(name + "-")]
        commit = max(mine) if mine else None
        out[name] = {
            "branch": max(
                ((ref, t) for ref, t in tips.items() if ref == name or ref.startswith(name + "-")),
                key=lambda kv: kv[1], default=("-", 0))[0],
            "commit_min": None if commit is None else int((now - commit) / 60),
            "ledger_min": None if newest[name] is None else int((now - newest[name]) / 60),
            "open_tasks": open_tasks[name],
            "socket": sockets.get(name, ""),
        }
    return out


def cmd_who(a):
    p = os.path.join(ROOT, "runs", "roster.json")
    if not os.path.exists(p):
        sys.exit("no runs/roster.json")
    r = json.load(open(p, encoding="utf-8"))
    owner = {}
    for m in r["members"]:
        for t in m["topics"]:
            owner.setdefault(t, m["name"])
    live = liveness(ROOT)
    print(f"{'name':<8}{'role':<13}{'commit':>8}{'ledger':>8}{'open':>6}  {'branch':<18}"
          f"{'socket':<34}topics")
    for m in r["members"]:
        d = live.get(m["name"], {})
        cm = "-" if d.get("commit_min") is None else f"{d['commit_min']}m"
        lm = "-" if d.get("ledger_min") is None else f"{d['ledger_min']}m"
        print(f"{m['name']:<8}{m['role']:<13}{cm:>8}{lm:>8}{d.get('open_tasks', 0):>6}  "
              f"{d.get('branch', '-'):<18}{m['socket']:<34}{','.join(m['topics']) or '-'}")
    print("\ncommit = minutes since their newest branch tip (name or name-*); "
          "ledger = minutes since their newest tasks/review/board row")
    print("\nnot on this team:")
    for m in r["not_on_this_team"]:
        print(f"  {m['name']:<22}{m['why']}")
    print("\naddress by socket, never by name -- names collide.")


def cmd_topics(a):
    by = {}
    for r in rows():
        by.setdefault(r["topic"], []).append(r)
    if not by:
        print("no topics")
        return
    order = sorted(by.items(), key=lambda kv: kv[1][-1]["ts"], reverse=True)
    for topic, rs in order:
        last = rs[-1]
        owner = next((r.get("owner") for r in rs if r.get("owner")), "?")
        blocked = sum(1 for r in rs if r["kind"] == "block")
        cleared = sum(1 for r in rs if r["kind"] == "done")
        flag = "BLOCKED " if blocked > cleared else ""
        print(f"{topic:<22} {owner:<12} {len(rs):>3} rows  {flag}{last['ts']}  "
              f"{last['kind']}: {last['text'][:70]}")


def cmd_show(a):
    rs = [r for r in rows() if r["topic"] == a.topic]
    if not rs:
        sys.exit(f"no topic '{a.topic}'")
    for r in rs:
        art = f"  [{r['artifact']}]" if r.get("artifact") else ""
        print(f"{r['ts']}  {r['who']:<12} {r['kind']:<6} {r['text']}{art}")


def cmd_feed(a):
    for r in rows()[::-1][:a.n]:
        art = f"  [{r['artifact']}]" if r.get("artifact") else ""
        print(f"{r['ts']}  {r['topic']:<20} {r['who']:<12} {r['kind']:<6} {r['text']}{art}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("post")
    q.add_argument("topic")
    q.add_argument("kind")
    q.add_argument("text")
    q.add_argument("--artifact", default="")
    q.add_argument("--who", default="")
    q.set_defaults(fn=cmd_post)

    q = sub.add_parser("open")
    q.add_argument("topic")
    q.add_argument("--owner", required=True)
    q.add_argument("--question", required=True)
    q.add_argument("--who", default="")
    q.set_defaults(fn=cmd_open)

    q = sub.add_parser("who")
    q.set_defaults(fn=cmd_who)

    q = sub.add_parser("topics")
    q.set_defaults(fn=cmd_topics)

    q = sub.add_parser("show")
    q.add_argument("topic")
    q.set_defaults(fn=cmd_show)

    q = sub.add_parser("feed")
    q.add_argument("-n", type=int, default=40)
    q.set_defaults(fn=cmd_feed)

    q = sub.add_parser("brief")
    q.set_defaults(fn=lambda a: None)

    a = p.parse_args()
    broadcast(getattr(a, "who", "") or whoami())
    a.fn(a)


if __name__ == "__main__":
    main()
