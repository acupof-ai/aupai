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
        # A RETRACTION IS TERMINAL BY KIND, NOT BY POSITION, for the same reason a close beats
        # a later start: a union merge concatenates two branches in whatever order it likes, so
        # file position is not evidence of sequence. Without this, an `ok` event ordered after a
        # `retracted` one UN-RETRACTS the run -- measured on a three-event fixture the moment
        # `retract` was written, and it is the worse direction of the two, because the row then
        # reads as a standing result with the retraction invisible rather than as a lost note.
        #
        # The docstring above says "two terminal events for one run: the later one wins", which
        # was written when `ok` and `fail` were the only terminals and disagreed only about
        # outcome. Retraction disagrees about VALIDITY: it is a statement about the other event,
        # so it cannot be outvoted by it. Un-retracting takes an explicit `done`, which appends
        # a new event a human chose to write.
        if prev is not None and prev.get("status") == "retracted" and r.get("status") != "retracted":
            continue
        # A MONITOR'S CLOSE IS NOT A RESULT, so it loses to a human's regardless of position.
        # Reported by 4c 2026-09-07 and reproduced on a three-event fixture: a run closed by hand
        # with `val 2.884 at step 2000, control 2.877 / stop rule 4 tripped` and closed by its
        # monitor with `exit 0 / monitor: process exited cleanly` folds to whichever row a union
        # merge happened to put last. In one of the two orders the reading a human took is
        # invisible, with nothing red -- the row still says status=ok, which is why nobody looks.
        #
        # THE ASYMMETRY IS THE ARGUMENT, and it is the same one the retraction rule above makes.
        # The two events do not disagree about the same question: the monitor reports PROCESS
        # STATE (the pid returned 0) and the human reports the RESULT (what the run measured).
        # A process that exits cleanly having produced a stopped arm is both at once, so the
        # monitor's row is never wrong -- it is just not an answer to the question the ledger is
        # read for. An event that cannot answer the question cannot outvote one that does.
        #
        # WHY HERE AND NOT IN THE MONITOR: the monitor already skips its close when it sees a
        # terminal row (`settled()`), and exp.py refuses a second close in one tree. Both are
        # races -- settled() polls at 60-second resolution, and the refusal only fires when both
        # writers share a working tree, which two sessions on two branches do not. The fold is
        # the only place that sees both rows, so it is the only place the rule holds without a
        # race. The monitor keeps its guard: skipping the write is cheaper than folding it away.
        #
        # IDENTIFIED BY `writer`, NOT BY THE TEXT. Matching on "monitor:" would be a predicate on
        # prose that any reworded finding silently escapes, and it would misfire on a human whose
        # finding quotes the monitor. exp.py's `done --writer monitor` sets the field; a row with
        # no `writer` is a human's, which is the safe default for every row already in the ledger.
        if (prev is not None and r.get("writer") == "monitor"
                and prev.get("status") in ("ok", "fail") and prev.get("writer") != "monitor"):
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


def _closed_only_by_monitor(name, started):
    """True when this run has terminal events and EVERY one of them was written by the monitor.

    One predicate, two call sites (de-70): pick_open_row's --started path and `done`'s bare-call
    path both need it, and they are frames apart. A second copy would drift -- exactly the reason
    pick_open_row is shared by `done` and `note` rather than duplicated.

    rows(raw=True), NOT rows(): the fold collapses a key to ONE row, so the folded view cannot
    answer "was every terminal event the monitor's" -- it shows only the winner, and after this
    verb runs once the winner is the human's. The question is about the population, not the winner.

    ALL, not any: one human close already present means a human has read this run, and overriding
    that would need a fold rule saying which human wins. There is none, which is why ruling (a)
    is narrow.
    """
    terminal = [r for r in rows(raw=True)
                if r.get("name") == name and r.get("started") == started
                and r.get("status") not in (None, "", "running")]
    return bool(terminal) and all(r.get("writer") == "monitor" for r in terminal)


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
            # NAME THE STATE FOUND, not just the absence. "Open rows: none" is true and useless:
            # it reads identically for a name nobody started and for the row the caller is
            # holding in their hand, already closed -- and the second is the common case, since
            # --started is copied off a row someone just looked at. Returning None instead would
            # be worse: `done`'s caller fabricates a base when it gets None, which is how the
            # orphan row of de-46 was written.
            closed = [r for r in rows() if r["name"] == name and r.get("started") == started
                      and r["status"] != "running"]
            if closed:
                # THE MONITOR EXCEPTION REACHES THIS PATH TOO (de-70, 4c's ruling (a)). `done`'s
                # own closed-row branch handles the BARE call; passing --started exits here first,
                # several frames earlier, so implementing the exception only there left it
                # unreachable for exactly the invocation the case needs -- b0 has the `started`
                # value in hand, copied off the row. Measured: the world got this refusal, not the
                # reclassify one, and the branch below had never run.
                #
                # RETURNING None IS WRONG HERE, which is why this returns the row instead: `done`
                # fabricates a base from None (the de-46 orphan), and a fabricated row loses the
                # cmd and hypothesis this close must inherit. The caller re-derives whether the
                # exception applies -- it owns the --reason check and the refusal text.
                if (verb == "closing" and closed[-1]["status"] != "retracted"
                        and _closed_only_by_monitor(name, started)):
                    return closed[-1]
                sys.exit(f"{name} ({started}) is not open -- it is already closed as "
                         f"{closed[-1]['status']!r}, result "
                         f"{str(closed[-1].get('result', ''))[:60]!r}. {verb.capitalize()} it "
                         f"again would overwrite that. `start` a new run if this is a new attempt.")
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


def pick_closed_row(name, started, verb):
    """The CLOSED row a command acts on. The mirror of pick_open_row, and separate on purpose.

    `retract` acts on a finished run, so pick_open_row cannot serve it: that function's whole
    subject is rows whose last event is `running`, and a retraction of a run that never closed
    is a `done --status fail`, not a retraction. Keeping them apart means neither can silently
    answer the other's question.

    CLOSED means "the last event for this (name, started) is terminal", which is what rows()
    returns after folding. The ambiguity refusal is the same shape as pick_open_row's and for
    the same reason: a name with two finished runs has two candidates, and choosing the newest
    silently retracts a result the caller may not have meant. Unlike the open case, there is no
    live-run hazard here -- the cost is a wrong row edited, not a running job misreported --
    but the tool still cannot make the choice.
    """
    closed = [r for r in rows() if r["name"] == name and r["status"] not in ("running",)]
    if started:
        base = next((r for r in closed if r.get("started") == started), None)
        if base is None:
            seen = [r.get("started") for r in closed]
            sys.exit(f"no closed row for {name} started {started!r}. Closed rows: {seen or 'none'}")
        return base
    if not closed:
        open_now = [r for r in rows() if r["name"] == name and r["status"] == "running"]
        if open_now:
            sys.exit(
                f"{name} has no closed row; it is still running (started "
                f"{open_now[-1].get('started')!r}). A run that never finished is closed with "
                f"`done --status fail`, not retracted -- retraction withdraws a RESULT."
            )
        sys.exit(f"no row for {name} at all. Nothing to retract.")
    if len(closed) > 1:
        seen = [(r.get("started"), r.get("status")) for r in closed]
        sys.exit(
            f"{name} has {len(closed)} closed rows ({seen}); {verb} the newest by default "
            f"would withdraw a result the caller may not mean. Pass --started <value>."
        )
    return closed[-1]


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
    # WHAT THE RUN IS FOR, as a field rather than prose in --notes. harness launch requires it and
    # passes it here; a row written by hand may omit it, and the absence is then honest -- no
    # backfill and no default, because a class the tooling chose says nothing about intent.
    # 4c's ruling 2026-09-05: the baseline for any metric over this field starts the day the
    # writer lands, and rows before it read null rather than a guessed value.
    s.add_argument("--class", dest="run_class", default="",
                   help="incremental (a number nobody has) | confirmatory (reproduces a known "
                        "number) | infra-verification (proves a tool works, not a model "
                        "measurement). Empty on a hand-written row means unstated, not zero.")
    # WHICH CARDS, as a field. harness launch has appended "; cards N" to --notes since the
    # allocation moved to the grant file, so the information existed and only 1 of 243 folded rows
    # could be READ for it -- "which card was this on" is the first question asked when two jobs
    # collide, and answering it meant grepping free text with 14 different phrasings. A field the
    # metric can group by, and prose stays prose.
    s.add_argument("--cards", default="",
                   help="the cards this run holds, CSV (e.g. 5 or 0,1,2,3,4,5,6). Empty means "
                        "unstated -- for a CPU/corpus job pass 'none' rather than leaving it blank")
    d = sub.add_parser("done")
    d.add_argument("--name", required=True)
    d.add_argument("--result", default="")
    d.add_argument("--status", default="ok")
    d.add_argument("--finding", default="", help="what the number means — the interpretation, not the number")
    d.add_argument("--decision", default="", help="what changes next because of this result")
    d.add_argument("--started", default=None,
                   help="close THIS row (its 'started' value), not the newest running one. "
                        "Required when a name has more than one open row")
    d.add_argument("--reading_artifact", default="",
                   help="repo-relative path to the file the result was READ FROM, when the row's "
                        "cmd produces no checkpoint harness.py can score. harness.py's "
                        "score_matrix_present FAILs on a path that does not exist, so this "
                        "names a real file or it names nothing")
    d.add_argument("--writer", default="",
                   help="who wrote this close. `monitor` marks a close that reports PROCESS STATE "
                        "rather than a result, and fold() lets a human's close outvote it "
                        "regardless of union-merge order. Leave empty for a human -- every row "
                        "already in the ledger has no writer, and that is the safe default")
    # ONLY MEANINGFUL WHEN RE-CLOSING A MONITOR-CLOSED ROW, and mandatory there (de-70, 4c's
    # ruling (a), 2026-09-08). Not required for an ordinary close: a first close needs no
    # justification for existing. It is required when the ledger will end up holding two terminal
    # events for one run, because that pair is otherwise indistinguishable from a double-close bug.
    d.add_argument("--reason", default="",
                   help="why the monitor's close is not the result. REQUIRED when re-closing a row "
                        "the monitor already closed (a deliberate stop lands there as exit 137); "
                        "ignored on a first close. The monitor's event is never rewritten -- this "
                        "appends a human one, which fold() prefers in either merge order")
    am = sub.add_parser("amend", help="correct a CLOSED row's reading_artifact, finding or "
                                      "decision; does not touch status or result")
    am.add_argument("--name", required=True)
    am.add_argument("--reading_artifact", default="",
                    help="repo-relative path to the file the row's result was READ FROM. Checked "
                         "here as `done` checks it: harness.py's score_matrix_present FAILs on a "
                         "path that does not exist, so this names a real file or it names nothing")
    # THE PROSE FIELDS, de-62's second half. A closed row's INTERPRETATION goes stale in a way its
    # result does not: the number stands and what it means changed, which is not a retraction (the
    # result is not withdrawn) and not a note (`note` carries status=running forward, reopening a
    # finished run). Before this no verb could do it, so the only reachable moves were to retract a
    # standing result to fix prose about it, or to leave a wrong finding in the ledger.
    #
    # NOT --result and NOT --status. Those are the measurement and its validity: changing either is
    # a different claim about the run, and the tool has `retract` for that -- which preserves the
    # withdrawn value in `retracted_result` precisely so a reader can check the retraction. An
    # amend that could rewrite `result` would be a silent retraction with no record of the old
    # number, i.e. the defect facts/*.json's retracted_value exists to prevent.
    am.add_argument("--finding", default="",
                    help="corrected interpretation of a result that still stands. The number is "
                         "not editable here -- a result that does not stand takes `retract`")
    am.add_argument("--decision", default="",
                    help="corrected 'what changes because of this'")
    am.add_argument("--started", default=None,
                   help="amend THIS row (its 'started' value). Required when a name has more "
                        "than one closed row")
    n = sub.add_parser("note", help="append a line to a RUNNING row's notes; does not close it")
    n.add_argument("--name", required=True)
    n.add_argument("--text", required=True)
    n.add_argument("--started", default=None,
                   help="annotate THIS row (its 'started' value). Required when a name has "
                        "more than one open row")
    n.add_argument("--quiet-if-absent", action="store_true",
                   help="exit 0 without writing when the name has no open row. For automation "
                        "that annotates a row it did not create")
    r = sub.add_parser("retract", help="withdraw a CLOSED row's result; appends, never rewrites")
    r.add_argument("--name", required=True)
    r.add_argument("--reason", required=True,
                   help="why the result does not stand. Required: a retraction whose reason is "
                        "absent is indistinguishable from a lost row")
    r.add_argument("--superseded_by", default="",
                   help="the 'started' value of the run that replaces this one, when one does. "
                        "Checked: it must name a row that exists, or the pointer is a dead end")
    r.add_argument("--started", default=None,
                   help="retract THIS row (its 'started' value). Required when a name has more "
                        "than one closed row")
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
                # Only when given: absent means unstated (every row before 2026-09-05), "" would
                # be indistinguishable from it, and a metric must be able to tell "nobody said"
                # from "said nothing". harness launch always supplies one.
                **({"class": a.run_class} if a.run_class else {}),
                # Same absent-vs-empty rule: no key means unstated (every row before 2026-09-05),
                # and a CPU job passes "none" so that "no cards" is a stated answer rather than a
                # gap. harness launch always supplies one.
                **({"cards": a.cards} if a.cards else {}),
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
        # None on every ordinary close. Set only by the monitor-reclassify path, and read after the
        # event is built -- so it must exist before that path can be skipped, which is every
        # normal `done`.
        _reclassifies = None
        # TWO PATHS REACH A MONITOR-CLOSED ROW and both must demand --reason (de-70). With
        # --started, pick_open_row returns the closed row itself, so `base` is non-None and the
        # closed-row branch below never runs; without it, `base` is None and that branch does the
        # work. Checking here covers the first: `base` is a row whose status is terminal, which an
        # open row's never is, so the condition cannot fire on an ordinary close.
        if base is not None and base.get("status") not in (None, "", "running"):
            _reclassifies = dict(base)
            if not a.reason:
                sys.exit(
                    f"{a.name} ({base.get('started')}) is closed as {base['status']!r} by the "
                    f"MONITOR, result {str(base.get('result', ''))[:60]!r}. A monitor reports "
                    f"process state, so a deliberate stop lands here as a failure and this is "
                    f"re-closable by hand -- but it needs --reason, because the ledger will then "
                    f"hold two terminal events for one run and a reader cannot tell a correction "
                    f"from a double-close bug without one. Re-run with --reason '<why the "
                    f"monitor's reading is not the result>'."
                )
        if base is None:
            # A CLOSED ROW IS NOT AN ABSENT ROW, and this branch could not tell them apart.
            # pick_open_row's subject is rows whose last event is `running`, so it returns None
            # for a name nobody started AND for a name already closed. The `or {...}` below then
            # fabricated a base with started=now() and this exited 0 printing "logged done".
            #
            # MEASURED over the 8 combinations of (retracted, --started, later minute) on
            # 2026-09-06 (de-46): a bare `done` on any closed row fabricates -- retraction is not
            # the variable, being closed is -- and what it produces depends on the CLOCK, because
            # now() is minute resolution:
            #
            #   same minute as the close -> the fabricated `started` collides with the original's
            #     and the event folds onto that row. On a retracted row fold() then DISCARDS it
            #     (`retracted` is terminal by kind): success printed, nothing changed.
            #   a later minute -> TWO folded rows for one run: the original keeping its result,
            #     and a new row carrying this one with cmd='', hypothesis='' and no commit -- a
            #     result orphaned from the run that produced it. EXPERIMENTS.md then shows a row
            #     whose command is blank, which reads as a ledger defect rather than a misuse.
            #
            # The fabrication is legitimate for a name with NO row: closing a run whose start
            # event was lost is the case it was written for, and run_sft.sh/run_pretrain.sh call
            # `done` unconditionally at exit. So the refusal is keyed on a CLOSED row existing,
            # not on `base is None`. Passing --started already exits inside pick_open_row, but
            # with "Open rows: none" -- true and useless, since it names neither the status found
            # nor what to do; both paths now say the same thing.
            #
            # NOT a fold change: 4c ruled 2026-09-06 that `retracted` stays terminal and `amend`
            # is the field-only edit. exp.py's amend comment asserted `done` already "refuses --
            # the row is not open"; it did not.
            #
            # The `!= "running"` filter is redundant and kept as insurance: this branch runs only
            # when pick_open_row found no open row, so nothing running can be in scope. Dropping
            # it changes no behaviour -- measured, and it is why no world covers that clause. The
            # guard that keeps an open row closing is `base is None` itself.
            _closed = [r for r in rows() if r["name"] == a.name and r["status"] != "running"
                       and (a.started is None or r.get("started") == a.started)]
            if _closed:
                _r = _closed[-1]
                # THE ONE EXCEPTION: A MONITOR'S CLOSE IS NOT A RESULT (de-70, 4c's ruling (a),
                # 2026-09-08). A row whose only terminal event was written by the monitor reports
                # PROCESS STATE -- the pid returned 137 -- and a deliberate stop produces exactly
                # that, so the row reads `fail / exit 137 (signal 9)` for a run someone ended on
                # purpose at a chosen step. b0 hit this on (1.5b-a0.2b-e48_30b, 2026-09-07 05:15)
                # and no verb could reclassify it: `amend` excludes status by design, `retract`
                # withdraws a RESULT rather than restating one, `note` carries running forward.
                #
                # THE FOLD ALREADY RESOLVES THIS and is not touched. Its 2026-09-07 rule -- a
                # monitor's close loses to a human's, keyed on `writer` and not on the text -- makes
                # a human event win in EITHER file order, measured on a three-event fixture before
                # this was written (monitor-only -> fail/137; human after -> the human's; human
                # before, which a union merge can produce -> the human's). So the only thing
                # missing was a writer that would append the event; the semantics were in place.
                #
                # A --reason IS MANDATORY HERE and nowhere else in `done`. Overriding a close that
                # already exists is the one case where the ledger holds two terminal events for one
                # run and a reader has to know why the second one is there; without it the pair is
                # indistinguishable from a double-close bug. Same argument as `retract --reason`.
                #
                # NARROW BY CONSTRUCTION, and deliberately not generalised to (b), a `reclassify`
                # that could override any terminal event: overriding a HUMAN's close would need a
                # fold rule saying which human wins, and there is no principled answer. The
                # condition is every terminal event for this key being the monitor's, so one human
                # close already present makes this refuse again.
                _all_monitor = _closed_only_by_monitor(a.name, _r.get("started"))
                if _all_monitor and _r["status"] != "retracted":
                    if not a.reason:
                        sys.exit(
                            f"{a.name} ({_r.get('started')}) is closed as {_r['status']!r} by the "
                            f"MONITOR, result {str(_r.get('result', ''))[:60]!r}. A monitor reports "
                            f"process state, so a deliberate stop lands here as a failure and this "
                            f"is re-closable by hand -- but it needs --reason, because the ledger "
                            f"will then hold two terminal events for one run and a reader cannot "
                            f"tell a correction from a double-close bug without one. Re-run with "
                            f"--reason '<why the monitor's reading is not the result>'."
                        )
                    base = dict(_r)      # inherit cmd, hypothesis, started: this is the SAME run
                    _reclassifies = _r   # printed below, so the caller sees what was overridden
                else:
                    _how = ("A retraction is terminal by kind -- record the corrected result as a "
                            "NEW run (`start` under its own name, then `done`), and if only the "
                            "reading is missing use `amend`."
                            if _r["status"] == "retracted" else
                            "Re-close it explicitly with --started "
                            f"{_r.get('started')!r}, or `start` a new run if this is a new attempt.")
                    sys.exit(
                        f"{a.name} ({_r.get('started')}) is already closed as {_r['status']!r}"
                        + (f": {str(_r.get('retracted_reason', ''))[:80]!r}"
                           if _r["status"] == "retracted" else
                           f", result {str(_r.get('result', ''))[:60]!r}")
                        + f". Closing it again would append a row with no cmd, or an event the fold "
                          f"discards. {_how}"
                    )
        ev = dict(
            base
            or {
                "started": now(), "name": a.name, "cmd": "", "notes": "",
                "hypothesis": "", "commit": git_commit(),
            },
            status=a.status, result=a.result, finding=a.finding, decision=a.decision, ended=now(),
        )
        if a.reading_artifact:
            # CHECKED HERE, not only by harness.py. The field's whole job is to point at the
            # file a reader can open; a path that does not exist turns a scoring exemption
            # into an unfalsifiable claim, and harness.py would then FAIL the ledger AFTER
            # the close is already appended (append-only: it cannot be taken back).
            p = os.path.join(ROOT, a.reading_artifact)
            if not os.path.exists(p):
                sys.exit(f"--reading_artifact {a.reading_artifact} does not exist under "
                         f"{ROOT}; pull the file into the repo before closing the row")
            ev["reading_artifact"] = a.reading_artifact
        if a.writer:
            # ONLY WHEN SET. An absent field is a human's close, so writing "human" by default
            # would split the population into two spellings of the same thing and make fold()'s
            # rule depend on which era a row was written in.
            ev["writer"] = a.writer
        # THE REASON AND WHAT IT OVERRODE, on the event (de-70). Recorded as two fields rather than
        # folded into `result`, because `result` is the measurement and this is a statement ABOUT
        # another event -- the same separation `retract` keeps with retracted_result. A reader who
        # sees two terminal events for one run can then ask the ledger why, instead of inferring it
        # from the pair's existence. `writer` is deliberately NOT set: this close is a human's, and
        # setting it would make fold() treat it as the monitor's and discard it.
        if _reclassifies is not None:
            ev["reclassify_reason"] = a.reason
            ev["reclassifies"] = {
                "status": _reclassifies.get("status"),
                "result": _reclassifies.get("result"),
                "writer": _reclassifies.get("writer"),
            }
            ev.pop("writer", None)
        append(ev)
        if _reclassifies is not None:
            print(f"logged done: {a.name} -> {a.result}\n"
                  f"  RECLASSIFIED the monitor's close "
                  f"({_reclassifies.get('status')} / "
                  f"{str(_reclassifies.get('result'))[:50]}) -- its event is untouched; the fold "
                  f"now shows this one. Reason: {a.reason}")
        else:
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
    elif a.action == "retract":
        # WITHDRAWING A RESULT, not deleting a run. The row keeps its cmd, hypothesis, commit
        # and original result: what is being said is "this number does not stand", and a
        # reader who cannot see the number that was withdrawn cannot check the retraction.
        # Same discipline as facts/*.json's retracted_value, which exists because rewriting
        # `value` into a narration leaves one field holding both the dead number and its
        # replacement, separated by prose.
        #
        # APPENDS, like done and note. A rewrite means two branches retracting two different
        # runs keep both versions of each after a union merge.
        base = pick_closed_row(a.name, a.started, "retracting")
        if base.get("status") == "retracted":
            sys.exit(
                f"{a.name} ({base.get('started')}) is already retracted: "
                f"{str(base.get('retracted_reason', ''))[:100]!r}. Re-retracting would "
                f"overwrite the first reason with the second; append a note instead."
            )
        if a.superseded_by:
            # CHECKED, because an unchecked pointer is the defect this whole tool exists
            # against: a retraction saying "superseded by X" where X is not a row sends the
            # next reader looking for a run that does not exist, and reads as diligence.
            known = {r.get("started") for r in rows() if r["name"] == a.name}
            allk = {r.get("started") for r in rows()}
            if a.superseded_by not in allk:
                sys.exit(
                    f"--superseded_by {a.superseded_by!r} names no row in the ledger. "
                    f"Rows for {a.name}: {sorted(x for x in known if x)}"
                )
            if a.superseded_by == base.get("started"):
                sys.exit("--superseded_by names the row being retracted; a run cannot "
                         "supersede itself")
        append(dict(
            base,
            status="retracted",
            retracted_reason=a.reason,
            retracted_result=base.get("result", ""),
            superseded_by=a.superseded_by,
            retracted_at=now(),
        ))
        print(f"logged retract: {a.name} ({base.get('started')}) -- was "
              f"{str(base.get('result', ''))[:60]!r}")
    elif a.action == "amend":
        # CORRECTS A CLOSED ROW'S reading_artifact, finding OR decision WITHOUT TOUCHING status or
        # result, which is the one thing neither `done` nor `note` nor `retract` can do. Three rows
        # were closed `ok` with no reading, so score_matrix_present read them as unscored training
        # runs; the only tools available were `done` (refuses -- the row is not open, and on an
        # already-closed row that refusal had to be written: until de-46 it fabricated an orphan
        # and exited 0) and `retract` (withdraws the RESULT to satisfy a gate about its READING,
        # which is the wrong trade). 4c's ruling 2026-09-06: add this rather than change the fold.
        #
        # THE PROSE FIELDS ARE de-62's SECOND HALF, and the same argument as the reading: an
        # interpretation goes stale in a way its number does not. The result stands, what it means
        # changed. `note` cannot serve because it carries status=running forward, reopening a
        # finished run, and `retract` would withdraw a result that is still correct.
        #
        # NOT result, NOT status: those are the measurement and its validity, and `retract` owns
        # them precisely because it preserves the withdrawn value in `retracted_result` for a
        # reader to check. An amend that could rewrite `result` would be a silent retraction with
        # no record of the old number.
        #
        # NOT A PATH TO UN-RETRACT. A retracted row is refused below, because `retracted` is
        # terminal by KIND (see fold) and an amend that revived one would be a rewrite of that
        # rule wearing a different name -- exactly what the fold's comment guards against.
        if not (a.reading_artifact or a.finding or a.decision):
            sys.exit("amend needs at least one of --reading_artifact, --finding, --decision. "
                     "With none of them it would append an event identical to the row and report "
                     "success -- the silent no-op this command exists to replace.")
        base = pick_closed_row(a.name, a.started, "amending")
        if base.get("status") == "retracted":
            # THE REMEDY NAMED HERE IS NOT `done`. It said "un-retract deliberately with `done`"
            # until de-46, which was wrong twice over: `done` on a retracted row fabricated an
            # orphan rather than un-retracting anything, and it now refuses outright. A retraction
            # is terminal, so the corrected reading belongs to a new run under its own name.
            sys.exit(
                f"{a.name} ({base.get('started')}) is retracted, and amend does not revive a "
                f"row: `retracted` is terminal by kind, so the amended event would be dropped "
                f"by the fold and this command would report success while changing nothing. "
                f"Record it on a fresh run (`start` under its own name, then `done`)."
            )
        changes = {}
        if a.reading_artifact:
            # THE PATH MUST EXIST, checked here for the same reason `done` checks it: the field's
            # whole job is to name a file a reader can open, and a path that does not exist turns a
            # scoring exemption into an assertion nobody wrote.
            if not os.path.exists(os.path.join(ROOT, a.reading_artifact)):
                sys.exit(f"--reading_artifact {a.reading_artifact} does not exist under {ROOT}; "
                         f"pull the file into the repo before amending the row")
            changes["reading_artifact"] = a.reading_artifact
        for _f, _v in (("finding", a.finding), ("decision", a.decision)):
            if _v:
                changes[_f] = _v
        # NOTHING-TO-DO IS A REFUSAL, per field. An amend whose values already match the row would
        # append a duplicate event and print success, which is the shape that made the pre-de-46
        # `done` dangerous: a command that reports a change it did not make.
        unchanged = [k for k, v in changes.items() if base.get(k) == v]
        if unchanged and len(unchanged) == len(changes):
            sys.exit(f"{a.name} ({base.get('started')}) already carries "
                     + ", ".join(f"{k}={str(base.get(k))[:40]!r}" for k in unchanged)
                     + "; nothing to do")
        prev = {k: base.get(k) for k in changes}
        append(dict(base, **changes, amended_at=now()))
        print(f"logged amend: {a.name} ({base.get('started')}) "
              + "; ".join(f"{k} {str(prev[k])[:34]!r} -> {str(v)[:34]!r}"
                          for k, v in changes.items())
              + f" (status {base.get('status')!r} and result unchanged)")
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
