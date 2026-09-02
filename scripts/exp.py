#!/usr/bin/env python3
"""Experiment log: one JSONL row per GPU run, rendered to EXPERIMENTS.md.

Every GPU run gets a record BEFORE it starts and a result appended when it ends:

  python scripts/exp.py start --name sft_v3 --cmd "torchrun ..." --notes "350K short mix"
  python scripts/exp.py done  --name sft_v3 --result "math-500 34.2%" --status ok

Rows live in runs/experiments.jsonl; `python scripts/exp.py render` rewrites
EXPERIMENTS.md (newest first) so the table is reviewable in the repo.
"""

import argparse
import json
import os
import subprocess
import sys
import time

# --root is a FLAG, not an env var: an ambient AUPAI_ROOT would silently redirect the
# experiment log of a production run. The log is the ledger; it gets no ambient override.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "runs", "experiments.jsonl")
MD = os.path.join(ROOT, "EXPERIMENTS.md")


def set_root(root):
    global ROOT, LOG, MD
    ROOT = root
    LOG = os.path.join(root, "runs", "experiments.jsonl")
    MD = os.path.join(root, "EXPERIMENTS.md")


def fold(evs):
    """The events for one ledger reduced to one row per run. THE fold, stdlib only.

    Keyed by (name, started) -- the pair, never the name alone. Folding by name
    collapses two runs that shared a name at different times, and harness.py:4458
    did exactly that, so a re-run of a name silently replaced the earlier run's row.

    A CLOSE IS TERMINAL, REGARDLESS OF POSITION, and that is the whole reason this
    is one function rather than five. Last-write-wins on file order reopens a
    finished run whenever a duplicate start event lands after its close -- which
    this ledger CONTAINS: (sft_p324_v3, 2026-08-31 03:44) has an `ok` event at line
    44 and a `running` event at line 132, and order-only folding reported a run that
    finished in 32 minutes as 26 hours stale. A union merge concatenates two
    branches' rows in whatever order it likes, so position is not evidence of
    sequence. A run does not reopen; only `task reopen` does that, and it is a
    different ledger. Two terminal events for one run: the later one wins.

    Until e1-18 this file's rows() folded on position while harness.py:2386 folded
    terminal-wins, and rows()'s docstring asserted the divergent shape was
    impossible -- "union-merging two branches cannot produce a running row and a
    done row for the same run" -- in a repo whose harness.py:2396 records it
    happening. The two readers agreed on today's ledger (0 of 175 keys differ) and
    would have diverged on the next merge that ordered those events the other way.
    Reasoning from de and e1 independently, 2026-09-01; kept where the fold lives.
    """
    out = {}
    for r in evs:
        key = (r.get("name"), r.get("started"))
        prev = out.get(key)
        if prev is not None and prev.get("status") != "running" and r.get("status") == "running":
            continue
        out[key] = r
    return list(out.values())


def rows(raw=False):
    """The log, folded by (name, started) with a close beating a later start.

    The file is an event log, not a table -- `done` appends rather than rewriting.
    raw=True yields every event, unfolded. The fold itself is fold(), which
    scripts/harness.py reads through so the ledger has one reduction and not five.
    """
    if not os.path.exists(LOG):
        return []
    evs = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]
    return evs if raw else fold(evs)


def append(row):
    """One event. Append, never rewrite: see rows()."""
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write(rs):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def now():
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime())


def git_commit():
    """WHICH CODE this run executed. Never "" -- see below.

    NOT who ran it. The pod's sha comes from data/pod_synced_head, whose author is
    whoever PUSHED the code, not whoever started the run. Usually the same person and
    not guaranteed to be. If you need the operator, this field does not carry it.

    The old version returned "" on the pod, which has no .git -- so it failed in the
    one environment where runs actually happen, and 273 of 290 rows carried a blank
    that is indistinguishable from "nobody filled this in". The comment said it was
    "filled in when the log is synced back"; that never happened once. A value that
    reads like an omission hides the fact that a function is broken.

    Three sources, in order, and the third is explicit rather than empty:
      HEAD              a git checkout (Mac, CI)
      pod_synced_head   pod_push.sh stamps <sha> <dirty> <utc> after a full push
      "unknown"         no git and no stamp
    A PARTIAL push deletes the stamp (pod_push.sh:40), so a run started after one
    records "unknown" -- which is correct: the pod is then one tree's sha plus another
    tree's file, and the honest answer is that no single sha describes it.
    """
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        with open(os.path.join(ROOT, "data", "pod_synced_head"), encoding="utf-8") as f:
            parts = f.read().split()
        if parts:
            # dirty>0 means the push carried uncommitted files: the sha is where the
            # tree was, not what it held. Say so rather than implying a clean match.
            # Short-form to match the git branch: rev-parse --short gives 7 chars and
            # pod_synced_head stores 40, so the same ledger column carried two widths
            # and neither reader could sort or compare them.
            sha = parts[0][:7]
            return sha if len(parts) < 2 or parts[1] == "0" else f"{sha}+dirty{parts[1]}"
    except (OSError, ValueError):
        pass
    return "unknown"


def render():
    rs = sorted(rows(), key=lambda r: r.get("started", ""), reverse=True)
    n_ok = sum(1 for r in rs if r.get("status") == "ok")
    lines = [
        "# Experiments",
        "",
        "Auto-generated by `scripts/exp.py` — every GPU run is recorded here.",
        "",
        f"{len(rs)} runs, {n_ok} completed. Newest first.",
        "",
        "| started | name | status | result | notes | commit |",
        "|---|---|---|---|---|---|",
    ]
    for r in rs:
        cells = [r.get(k, "") or "" for k in ("started", "name", "status", "result", "notes", "commit")]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    lines += ["", "## What each run taught us", ""]
    for r in rs:
        if not (r.get("hypothesis") or r.get("finding") or r.get("decision")):
            continue
        lines.append(f"### {r.get('name')} — {r.get('result') or r.get('status')}")
        if r.get("hypothesis"):
            lines.append(f"- **Asked:** {r['hypothesis']}")
        if r.get("finding"):
            lines.append(f"- **Learned:** {r['finding']}")
        if r.get("decision"):
            lines.append(f"- **So:** {r['decision']}")
        lines.append("")
    lines += ["<details><summary>Commands</summary>", ""]
    for r in rs:
        lines.append(f"- **{r.get('name')}** (`{r.get('started')}`): `{r.get('cmd', '')}`")
    lines += ["", "</details>", ""]
    open(MD, "w", encoding="utf-8").write("\n".join(lines))
    return MD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="repo root to log into (tests only; default this checkout)")
    sub = ap.add_subparsers(dest="action", required=True)
    s = sub.add_parser("start")
    s.add_argument("--name", required=True)
    s.add_argument("--cmd", default="")
    s.add_argument("--notes", default="")
    s.add_argument(
        "--hypothesis", default="", help="what this run is meant to test, written BEFORE it starts"
    )
    d = sub.add_parser("done")
    d.add_argument("--name", required=True)
    d.add_argument("--result", default="")
    d.add_argument("--status", default="ok")
    d.add_argument("--finding", default="", help="what the number means — the interpretation, not the number")
    d.add_argument("--decision", default="", help="what changes next because of this result")
    d.add_argument("--started", default=None,
                   help="close THIS row (its 'started' value), not the newest running one. "
                        "Required when a name has more than one open row")
    m = sub.add_parser("merge", help="merge another experiments.jsonl into this one (pod sync)")
    m.add_argument("--from", dest="src", required=True)
    sub.add_parser("render")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.root:
        set_root(a.root)

    if a.action == "start":
        append(
            {
                "started": now(),
                "name": a.name,
                "status": "running",
                "cmd": a.cmd,
                "notes": a.notes,
                "hypothesis": a.hypothesis,
                "result": "",
                "finding": "",
                "decision": "",
                "ended": "",
                "commit": git_commit(),
            }
        )
        print(f"logged start: {a.name}")
    elif a.action == "done":
        # APPEND the closing event; never rewrite the start row. runs/*.jsonl merges
        # by union, so a rewrite means two branches closing two different runs keep
        # BOTH the running row and the done row for each (the register hit exactly
        # this: duplicate t39/t40, 2026-08-31). Identity is (name, started), and
        # readers fold on it -- `merge` already does, and rows() now does too, so an
        # appended close carries the start row's `started` to fold onto it.
        # ONE name, LAST running row: reversed() takes the newest and stops. A name with
        # two open rows needs two calls -- and the second reaches the same row the first
        # just closed, so it cannot close the older one at all. Closing eight stale rows
        # across four duplicated names on 2026-09-02 went 6 -> 4 and looked stuck; the
        # fix is to append a close carrying that row's own `started`, which is identity.
        #
        # --started IS that fix, and the refusal below is the half that matters. p200m_4b_0902
        # had three open rows in eight minutes (two OOMed launches and the live run), so a
        # bare `done` would have closed the LIVE run and written the OOM as its result. The
        # newest-row default is safe only when there is exactly one candidate; with more, the
        # default is a silent wrong answer, and picking one is not a decision this tool can
        # make (de, 2026-09-02).
        rs = rows(raw=True)
        open_rows = [r for r in rs if r["name"] == a.name and r["status"] == "running"]
        if a.started:
            base = next((r for r in open_rows if r.get("started") == a.started), None)
            if base is None:
                seen = [r.get("started") for r in open_rows]
                sys.exit(f"no open row for {a.name} started {a.started!r}. Open rows: {seen or 'none'}")
        elif len(open_rows) > 1:
            seen = [r.get("started") for r in open_rows]
            sys.exit(
                f"{a.name} has {len(open_rows)} open rows ({seen}); closing the newest by "
                f"default would write this result onto a run that may still be alive. "
                f"Pass --started <value> to say which one."
            )
        else:
            base = open_rows[-1] if open_rows else None
        ev = dict(
            base
            or {
                "started": now(), "name": a.name, "cmd": "", "notes": "",
                "hypothesis": "", "commit": git_commit(),
            },
            status=a.status, result=a.result, finding=a.finding, decision=a.decision, ended=now(),
        )
        append(ev)
        print(f"logged done: {a.name} -> {a.result}")
    elif a.action == "merge":
        incoming = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
        out, idx = [], {}
        for r in sorted(rows() + incoming, key=lambda r: r.get("started", "")):
            k = (r["name"], r["started"])
            if k in idx:
                for f in ("result", "finding", "decision", "hypothesis", "commit", "ended", "cmd", "notes"):
                    if r.get(f) and not out[idx[k]].get(f):
                        out[idx[k]][f] = r[f]
            else:
                idx[k] = len(out)
                out.append(r)
        write(out)
        print(f"merged {len(incoming)} incoming rows -> {len(out)} total")
    elif a.action == "list":
        for r in rows():
            print(f"{r['started']}  {r['name']:<20} {r['status']:<8} {r.get('result', '')}")
    print(render())


if __name__ == "__main__":
    main()
