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

    FULL 40-CHAR SHAS, both paths (de-38). This function used `rev-parse --short` on one
    side and a hardcoded `parts[0][:7]` on the other, to "match the git branch". Those
    stopped agreeing: --short is git's AUTO-SCALING abbreviation, and once the object
    count crossed a threshold it began returning 8 characters while the stamp path still
    truncated to 7. The same commit then wrote two different strings depending on which
    branch ran -- 8cd68340 against 8cd6834, measured 2026-09-03 -- so p500m_20b_0902's
    00:03 row disagreed with the pod's copy in the `commit` field alone and read as a
    provenance conflict. Identical defect to de-35's `%h` in harness.merge_reverted_content:
    an identity whose text depends on repository size is not an identity. Readers
    abbreviate for display; the ledger stores what resolves.
    """
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        with open(os.path.join(ROOT, "data", "pod_synced_head"), encoding="utf-8") as f:
            parts = f.read().split()
        if parts:
            # dirty>0 means the push carried uncommitted files: the sha is where the
            # tree was, not what it held. Say so rather than implying a clean match.
            sha = parts[0]
            return sha if len(parts) < 2 or parts[1] == "0" else f"{sha}+dirty{parts[1]}"
    except (OSError, ValueError):
        pass
    return "unknown"


PLACEHOLDER = ("unknown", "pending")


def commit_resolves(sha, root=None):
    """(ok, why) -- does this ledger's `commit` value name an object in this repository?

    A sha that resolves nowhere is worse than "unknown": it reads as provenance while
    answering nothing, and no check looked. runs/experiments.jsonl carries `cec145b` on
    p500m_20b_0902, which matches no object here and no prefix of one -- a sha from a tree
    this repo does not contain (pod-side history, or a branch since rewritten). "unknown"
    and a "+dirty" suffix are ACCEPTED: both are honest statements about what the sha can
    say, and refusing them would push writers back to the blank this function was built to
    eliminate."""
    root = root or ROOT
    if not sha:
        return False, "empty -- git_commit never returns '', so this row was written by hand"
    if sha in PLACEHOLDER:
        # Honest non-answers, and refusing them would push writers back to the blank this
        # function exists to eliminate. `pending` is what a row carries before its job takes
        # a card (harness launch writes it), and it is replaced when the job starts.
        return True, f"{sha}: an explicit non-answer, not a sha"
    base = sha.split("+dirty", 1)[0]
    r = subprocess.run(["git", "-C", root, "cat-file", "-e", f"{base}^{{commit}}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, (f"{base} names no commit in this repository -- provenance that cannot "
                       f"be resolved is not provenance")
    return True, f"{base[:12]} resolves"


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


def pick_open_row(name, started, verb):
    """The open row a command acts on, or None when the name has none.

    Shared by `done` and `note` rather than copied. The ambiguity refusal below is the
    p200m_4b_0902 incident -- three open rows in eight minutes, two OOMed launches and the
    live run -- and a second copy of that reasoning drifts from this one at the next fix.
    OPEN means "the last event for this (name, started) is running", which is what rows()
    returns; filtering raw events reports every launch a name ever had as open.
    """
    open_rows = [r for r in rows() if r["name"] == name and r["status"] == "running"]
    if started:
        base = next((r for r in open_rows if r.get("started") == started), None)
        if base is None:
            seen = [r.get("started") for r in open_rows]
            sys.exit(f"no open row for {name} started {started!r}. Open rows: {seen or 'none'}")
        return base
    if len(open_rows) > 1:
        seen = [r.get("started") for r in open_rows]
        sys.exit(
            f"{name} has {len(open_rows)} open rows ({seen}); {verb} the newest by "
            f"default would write this onto a run that may still be alive. "
            f"Pass --started <value> to say which one."
        )
    return open_rows[-1] if open_rows else None


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
    n = sub.add_parser("note", help="append a line to a RUNNING row's notes; does not close it")
    n.add_argument("--name", required=True)
    n.add_argument("--text", required=True)
    n.add_argument("--started", default=None,
                   help="annotate THIS row (its 'started' value). Required when a name has "
                        "more than one open row")
    n.add_argument("--quiet-if-absent", action="store_true",
                   help="exit 0 without writing when the name has no open row. For automation "
                        "that annotates a row it did not create")
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
        # OPEN MEANS "the LAST event for this (name, started) is running", not "a running
        # event exists". A start event stays in the raw log forever, so filtering raw events
        # on status=="running" reports every launch a name ever had as still open -- and the
        # refusal below then demands --started on every close. MEASURED on this ledger
        # (1e, 2026-09-03): p200m_4b_0902 read 5 open rows, while fold() gives
        # fail/fail/stopped/stopped and exactly one running (14:32), the live run. So the
        # count was 5 and the answer was 1, and the tool asked a human to disambiguate
        # something it already had the data to settle.
        #
        # fold() is that data and already existed -- keyed on (name, started) with a close
        # terminal regardless of file position, which is why raw order cannot substitute for
        # it. rows() without raw=True IS the fold; the bug was reaching past it.
        #
        # The refusal is NOT relaxed: with two genuinely-live rows it still fires, which is
        # the case it was written for. p200m_4b_0902 had three open rows in eight minutes
        # (two OOMed launches and the live run), and a bare `done` would have closed the LIVE
        # run and written the OOM as its result. Fixing the count does not make picking one
        # of two live runs a decision this tool can make.
        base = pick_open_row(a.name, a.started, "closing")
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
    elif a.action == "note":
        # STILL RUNNING. `note` appends an event that carries status="running" forward, so
        # fold() keeps it as the row's latest state and a later `done` folds onto the same
        # (name, started) -- the row is never rewritten, same discipline as `done`.
        #
        # It exists because the chained end-of-run score_matrix in run_ddp.sh:83-96 runs
        # after torchrun exits, inside the training shell, and wrote NOTHING: b0 double-scored
        # the params leg because no artifact said a score was already in flight. Two events --
        # one when scoring starts, one when it ends -- are what distinguish "someone is
        # scoring this now" from "this was scored"; a single line at the end cannot.
        base = pick_open_row(a.name, a.started, "annotating")
        if base is None:
            # A run started outside harness launch has no row to annotate. That is a fact
            # about the launch, not a failure of the thing being annotated, so automation
            # passes --quiet-if-absent: a scoring run that SUCCEEDED must not exit nonzero
            # because its bookkeeping had no row to write to.
            msg = f"no open row for {a.name}; nothing annotated"
            if a.quiet_if_absent:
                print(msg)
                return
            sys.exit(msg)
        stamped = f"[{now()}] {a.text}"
        notes = base.get("notes") or ""
        append(dict(base, notes=f"{notes} | {stamped}" if notes else stamped))
        print(f"logged note: {a.name} ({base.get('started')}) -> {a.text}")
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
