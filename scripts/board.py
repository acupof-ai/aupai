#!/usr/bin/env python3
"""Shared board: topics anyone can post to and everyone can read.

    board.py topics                       one line per topic, newest state first
    board.py show <topic>                 that topic's rows, oldest first
    board.py post <topic> <kind> <text> [--artifact P] [--who W]
    board.py feed [-n N]                  everything, newest first
    board.py open <topic> --owner W --question Q

kind: find | rule | block | done | ask | note
A find/rule/done without --artifact is refused: a claim nobody can check is chatter.

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
    os.makedirs(os.path.dirname(BOARD), exist_ok=True)
    with open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_post(a):
    if a.kind not in KINDS:
        sys.exit(f"kind must be one of {', '.join(KINDS)}")
    if a.kind in NEEDS_ARTIFACT and not a.artifact:
        sys.exit(f"a '{a.kind}' needs --artifact: a claim nobody can check is chatter")
    known = {r["topic"] for r in rows()}
    if a.topic not in known and a.kind != "open":
        sys.exit(f"no topic '{a.topic}'. open it first, or: {', '.join(sorted(known)) or '(none)'}")
    append({"ts": time.strftime("%Y-%m-%d %H:%M"), "who": a.who or whoami(),
            "topic": a.topic, "kind": a.kind, "text": a.text, "artifact": a.artifact or ""})
    print(f"{a.topic} <- {a.kind} by {a.who or whoami()}")


def cmd_open(a):
    append({"ts": time.strftime("%Y-%m-%d %H:%M"), "who": a.who or whoami(),
            "topic": a.topic, "kind": "open", "text": a.question,
            "artifact": "", "owner": a.owner})
    print(f"opened {a.topic} (owner {a.owner})")


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

    q = sub.add_parser("topics")
    q.set_defaults(fn=cmd_topics)

    q = sub.add_parser("show")
    q.add_argument("topic")
    q.set_defaults(fn=cmd_show)

    q = sub.add_parser("feed")
    q.add_argument("-n", type=int, default=40)
    q.set_defaults(fn=cmd_feed)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
