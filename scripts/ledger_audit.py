#!/usr/bin/env python3
# restartable: read-only git history walk. Minutes at most; nothing to shard, nothing to resume.
"""Find commits that silently ROLL BACK an append-only ledger.

    python3 scripts/ledger_audit.py --selftest      # run this before believing any output
    python3 scripts/ledger_audit.py                 # audit every union-merged ledger
    python3 scripts/ledger_audit.py --path runs/review.jsonl

THE PREDICATE IS DISPATCHED BY WRITE STYLE, because the seven ledgers are not all the same
kind of file and one predicate cannot be honest for both kinds (1e/44/de; every writer verified
in-repo, see WRITE_STYLE):

  APPEND-style (experiments, board, milestones, review, retro) -- rows accumulate and fold by
  key with LAST ROW WINS, so the live row must be SUBSUMED: every NON-EMPTY field of the
  parent's last row for a key must appear, same value, in some child row for that key. Filling a
  blank passes; changing a value, clearing it, or dropping the row fails (de's subsumption,
  measured across seven worlds):

    any(all(f in c and c[f] == v for f, v in head_last.items() if v not in ("", None, [], {}))
        for c in index_rows_of_key)

  REWRITE-style (tasks, score_matrix) -- the file IS the current state, rewritten whole
  (harness.py:5805 `open(..., "w")`) or read-modify-write by design (score_matrix.py:573, "the
  matrix is the current state, not a history"). In-place value changes are the intended
  behaviour, so the only rule is that a KEY must not vanish. Subsumption here is a false-alarm
  generator: de measured it rejecting 46 of tasks.jsonl's 317 plain commits, 18.6%, all
  legitimate. My first version applied subsumption to all seven.

Row counts and key sets are both wrong, each in its own direction, and so were my own two
attempts at a fix:

  LINE COUNT over-reports. 6018c62ad folds 127 duplicate rows -- 292 lines to 195 -- losing
  nothing. Acting on that alarm means "restoring" rows that were never lost.

  KEY PRESENCE under-reports, and this is the subtle one. An amendment carries the SAME key as
  the row it amends; that is what last-wins means. a59ac1f dropped the newest ab_zeroinit row
  while all four rows shared ('ab_zeroinit', '2026-09-02 16:39'), so the key set was untouched
  (192 -> 192) and the ledger's meaning silently reverted to the previous amendment. I reported
  "no losses beyond the restored one" on the strength of this predicate. That report was wrong.

  VERBATIM last-line (1e's first form) over-reports: these ledgers are edited in place by
  scripts/exp.py, so 6018c62ad legitimately rewrites a row to fill its empty fields
  (status running -> probe, result "" -> 130 chars) and a verbatim test reads that as a delete.

  FIELD-LENGTH dominance (mine) over-reports too: it rejects `status: "running" -> "ok"`
  because the new value is shorter. Subsumption compares VALUES, so it needs no narrative-field
  allowlist and no length heuristic -- both of which were me approximating what de measured.

DECLARED REWRITES ARE A SEPARATE CLASS, not a negative. 6018c62ad and 7359a56f9 both rewrite
history deliberately (a fold, and the 0830v1 reset). The scan reports them under their own
heading; a commit message carrying `ledger-rewrite:` is returned with its manifest so the caller
decides, rather than being silently exempted here.

SECOND READING, reported and never blocking: how many non-empty result/finding/decision/notes
VALUES vanish from the whole file after folding. 6018c62ad loses 71 of them -- including
sft_p324_v3's code-500 40.0% measurement, which afterwards survives only in a rendered artifact
(de). Subsumption cannot see this: those values belong to keys whose live row is intact, so no
key regresses while measurements still leave the ledger.

WHY THE SELFTEST IS THE POINT OF THIS FILE. Six predicates preceded this one and five would
have reported these ledgers CLEAN or raised false alarms:

  v1  `rev-list --merges` only -- the real loss is a PLAIN commit, so it reported clean.
  v2  all commits, DEFAULT history simplification, which prunes commits git judges
      uninteresting for a path: 66 walked of the 182 that touch the file, and a59ac1f was among
      the 116 pruned. Still missed the known loss.
  v3  --full-history, row count. Found a59ac1f, flagged six innocent commits.
  v4  key presence. Clean on the false alarms and BLIND to a59ac1f -- the wrong report above.
  v5  any field shrinking. Caught a59ac1f, false-alarmed on `status: running -> probe`.
  v6  narrative fields only, by length. Separated both known cases, but only by a heuristic
      that a status enum changing to a shorter value would have broken again.
  v7  de's subsumption on values. Separates all seven worlds.

So this file asserts BOTH known cases: a59ac1f flagged (positive) and c3a5a23 clean (negative --
a commit that only appends a done event, chosen because 6018c62ad is a declared rewrite and
therefore not a clean negative). A scan that sweeps the whole repository and cannot see the case
you already know about certifies nothing -- docs/lessons/gate_failure_shapes.md §69 one level
out: there the criterion had no power to fail, here the SEARCH had no power to find. And one
level further: v4 was tested on the false alarms it fixed, never on the true positive it broke.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Row identity per ledger -- the identity .gitattributes already names ("row identity is
# (name, started) / id, never line position").
KEYS = {
    "runs/experiments.jsonl":   lambda r: (r.get("name"), r.get("started")),
    "runs/tasks.jsonl":         lambda r: r.get("id"),
    "runs/review.jsonl":        lambda r: r.get("id"),
    "runs/board.jsonl":         lambda r: (r.get("ts"), r.get("from") or r.get("who")),
    "runs/milestones.jsonl":    lambda r: (r.get("name"), r.get("at") or r.get("ts")),
    # (ckpt, profile), NOT (ckpt, measured): write_records replaces same-(ckpt, profile) and
    # eval/score_matrix.py:578 records why -- "a milestone-profile record must never replace a
    # checkpoint's full record". Keying on `measured` would make every re-score a NEW audit key
    # while the writer treats it as the SAME row, so each legitimate replacement would read as a
    # vanished key. Measured on HEAD: 43 distinct (ckpt, profile) against 40 (ckpt, measured)
    # over 65 rows -- different partitions, and only one of them is the writer's.
    "runs/score_matrix.jsonl":  lambda r: (r.get("ckpt"), r.get("profile", "full")),
    "runs/retro.jsonl":         lambda r: r.get("owner") or r.get("id") or r.get("name"),
}

# HOW EACH LEDGER IS WRITTEN decides which predicate is honest for it (1e/44/de, verified at the
# writer). Subsumption on a REWRITE-style ledger is a false-alarm generator: de measured it
# rejecting 46 of tasks.jsonl's 317 plain commits, 18.6%, all legitimate.
#
#   append  -- rows only ever accumulate; the live row must survive       -> subsumption
#   rewrite -- the file is the current state and rows are replaced in     -> key must not vanish
#              place by design
WRITE_STYLE = {
    "runs/experiments.jsonl":  "append",   # scripts/exp.py:82
    "runs/board.jsonl":        "append",   # scripts/board.py:66, open(..., "a")
    "runs/milestones.jsonl":   "append",   # scripts/harness.py:11308, open(..., "a")
    "runs/review.jsonl":       "append",   # no in-repo writer; .gitattributes merge=union
    "runs/retro.jsonl":        "append",   # no in-repo writer; .gitattributes merge=union
    "runs/tasks.jsonl":        "rewrite",  # harness.py:5805 _write_tasks, open(..., "w")
    "runs/score_matrix.jsonl": "rewrite",  # score_matrix.py:573 write_records, read-modify-write
}

VALUE_FIELDS = ("result", "finding", "decision", "notes")   # the second reading's scope
REWRITE_MARKER = "ledger-rewrite:"  # a declared rewrite: return the manifest, do not judge
MAX_COMMITS = 20000                 # iteration cap: every history walk carries one

# de's predicates, imported rather than reimplemented (de-33, scripts/test_ledger_predicates.py,
# main 88155d9): nine worlds, including the assertion that key_present is strictly weaker than
# subsume. A second copy here would be a second thing to keep correct, and the copy that drifts
# is the one nobody runs the selftest for.
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from test_ledger_predicates import key_present, subsume  # noqa: E402

# WHICH PREDICATE EACH LEDGER GETS, decided by how its tool writes it. Every entry cites the
# writer; the two rewrite-style files are the ones where subsume would be a false-alarm
# generator (de measured it rejecting 46 of tasks.jsonl's 317 plain commits, 18.6%, all
# legitimate). This table DIFFERS from test_ledger_predicates.PREDICATE on score_matrix.jsonl:
# that module has it on subsume, and 44's reading of the writer moved it to key_present.
PREDICATE = {
    "runs/experiments.jsonl":   subsume,      # scripts/exp.py:82, append
    "runs/board.jsonl":         subsume,      # scripts/board.py:66, open(..., "a")
    "runs/milestones.jsonl":    subsume,      # scripts/harness.py:11308, open(..., "a")
    "runs/review.jsonl":        subsume,      # no in-repo writer; .gitattributes merge=union
    "runs/retro.jsonl":         subsume,      # no in-repo writer; .gitattributes merge=union
    "runs/tasks.jsonl":         key_present,  # harness.py:5805 _write_tasks, open(..., "w")
    # score_matrix.py:573 write_records is read-modify-write and replaces same-(ckpt, profile) BY
    # DESIGN -- ":574 the matrix is the current state, not a history". So a changed value is the
    # intended behaviour and only a vanished key is a fault.
    "runs/score_matrix.jsonl":  key_present,
}


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout


def _rows(path, text):
    """Parsed rows. Unparseable lines are SKIPPED: two ledgers carry pretty-printed JSON blocks
    whose braces sit on separate lines, and treating those as rows made four innocent merges
    look like 10-key losses in v3."""
    if path not in KEYS:
        raise KeyError(f"no key definition for {path}; add one to KEYS")
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def by_key(path, text):
    """{key: [rows in file order]}. Order matters: the LAST row for a key is the live one."""
    kf = KEYS[path]
    out = {}
    for r in _rows(path, text):
        try:
            out.setdefault(kf(r), []).append(r)
        except (AttributeError, TypeError):
            continue
    return out


def regressions(path, head_lines, index_lines, message=""):
    """[(key, row)] the child fails to preserve, via de's predicate for THIS ledger.

    de-33's pre-commit hook calls this with HEAD's blob and the index's blob rather than
    reimplementing the comparison: one predicate, one place to be wrong. When `message` carries
    `ledger-rewrite:` the manifest is still returned -- the caller decides whether to allow it,
    so a declared rewrite is never silently exempt here.

    KNOWN BLIND SPOT, score_matrix.jsonl: the .lock at score_matrix.py:581-585 serializes
    interleaved read-modify-write cycles but does NOT stop a STALE process from writing back
    state it read before someone else's update. Two writers that never overlap in time still
    lose a record if the second read early, and this audit cannot see it either -- the key
    survives, the row is simply older.
    """
    pred = PREDICATE.get(path, subsume)
    kf = KEYS[path]
    return pred([json.dumps(r) for r in _rows(path, head_lines)],
                [json.dumps(r) for r in _rows(path, index_lines)], kf)


def declared_rewrite(message):
    return REWRITE_MARKER in (message or "")


def vanished_values(path, head_lines, index_lines):
    """How many non-empty VALUE_FIELDS values leave the file entirely (second reading).

    Subsumption cannot see this: a value can disappear from a key whose live row is intact,
    which is how 6018c62ad drops 71 values -- among them sft_p324_v3's code-500 40.0%, left
    only in a rendered artifact.
    """
    def vals(text):
        out = set()
        for r in _rows(path, text):
            for f in VALUE_FIELDS:
                v = r.get(f)
                if isinstance(v, str) and v.strip():
                    out.add((f, v))
        return out
    return vals(head_lines) - vals(index_lines)


def _blob(rev, path):
    r = subprocess.run(["git", "show", f"{rev}:{path}"],
                       capture_output=True, text=True, cwd=ROOT)
    return None if r.returncode else r.stdout


def audit(path, rev="HEAD"):
    """([revs], [(commit, parent, bad, n_vanished, kind, subj, declared)]).

    --full-history is mandatory, not a refinement: default simplification pruned the known
    instance in v2.
    """
    revs = _git("rev-list", "--full-history", rev, "--", path).split()[:MAX_COMMITS]
    hits = []
    for c in revs:
        after = _blob(c, path)
        if after is None:
            continue
        msg = _git("log", "-1", "--format=%B", c)
        parents = _git("rev-parse", f"{c}^@").split()
        for p in parents:
            before = _blob(p, path)
            if before is None:
                continue
            bad = regressions(path, before, after, msg)
            if bad:
                hits.append((c, p, bad, len(vanished_values(path, before, after)),
                             "MERGE" if len(parents) > 1 else "plain",
                             _git("log", "-1", "--format=%s", c).strip(),
                             declared_rewrite(msg)))
    return revs, hits


def _selftest():
    fails = []
    P = "runs/experiments.jsonl"

    # 0. Every union-merged ledger must have a key, or the scan silently skips it.
    attrs = os.path.join(ROOT, ".gitattributes")
    if os.path.exists(attrs):
        declared = {ln.split()[0] for ln in open(attrs, encoding="utf-8")
                    if "merge=union" in ln and not ln.startswith("#")}
        missing = declared - set(KEYS)
        if missing:
            fails.append(f"union-merged but unaudited (no key definition): {sorted(missing)}")

    # 1. KNOWN POSITIVE: a59ac1f dropped the newest ab_zeroinit amendment, whose key is SHARED
    #    with the rows it amends -- invisible to merges-only, to default simplification, and to
    #    key presence.
    known = _git("rev-parse", "a59ac1f").strip()
    if not known:
        fails.append("a59ac1f absent; the known-positive case cannot run")
    else:
        revs, hits = audit(P)
        flagged = {c for c, *_ in hits}
        if known not in revs:
            fails.append("a59ac1f not in the walked set: SCOPE is wrong (v2's error)")
        elif known not in flagged:
            fails.append("a59ac1f walked but NOT flagged: the PREDICATE is wrong (v4's error -- "
                         "its key is shared with the rows it amends)")

        # 2. KNOWN NEGATIVE: c3a5a23 only appends a done event. 6018c62ad is NOT usable here --
        #    it is a declared rewrite, and 1e/44 ruled it a third class rather than a negative.
        clean = _git("rev-parse", "c3a5a23").strip()
        if clean and clean in flagged:
            fails.append("c3a5a23 flagged: appending a done event is not a rollback")

    # 3. The predicate on hand-built worlds, so it is not only tested through history.
    def R(**kw):
        return json.dumps({"name": "e", "started": "t", **kw})
    base = R(status="running", notes="aa", result="") + "\n"
    if regressions(P, base, base):
        fails.append("identical content reports a regression")
    if regressions(P, base, base + R(name="f", started="u", notes="b") + "\n"):
        fails.append("appending a NEW key reports a regression")
    if regressions(P, base, R(status="running", notes="aa", result="measured 1.23") + "\n"):
        fails.append("FILLING an empty field reports a regression (6018c62ad's shape)")
    amended = base + R(status="done", notes="aaLONGER") + "\n"
    if regressions(P, base, amended):
        fails.append("an amendment that adds a row reports a regression")
    if not regressions(P, amended, base):
        fails.append("DROPPING the amendment is not reported -- this is a59ac1f and the "
                     "predicate cannot fail")
    if not regressions(P, base, R(status="running", notes="CHANGED", result="") + "\n"):
        fails.append("CHANGING a non-empty value is not reported")
    if not regressions(P, base, R(status="running", result="") + "\n"):
        fails.append("CLEARING notes by omission is not reported")
    if not regressions(P, base, ""):
        fails.append("an emptied file is not reported")

    # 4. A shorter non-empty enum value is a CHANGE and must be caught -- v5/v6 got this wrong
    #    in both directions (v5 alarmed on it, v6 excused it by calling status non-narrative).
    if not regressions(P, base, R(status="ok", notes="aa", result="") + "\n"):
        fails.append("status running -> ok is not reported; subsumption compares VALUES, so a "
                     "changed enum is a change regardless of length")

    # 4b. DISPATCH. The same edit must be a rollback in an append ledger and legitimate in a
    #     rewrite ledger, or the dispatch is decorative. tasks.jsonl is rewritten whole, so an
    #     in-place value change is how harness.py:6081 closes a task.
    t_before = json.dumps({"id": "t1", "state": "open", "why": "x"}) + "\n"
    t_after = json.dumps({"id": "t1", "state": "done", "why": "x"}) + "\n"
    if regressions("runs/tasks.jsonl", t_before, t_after):
        fails.append("a rewrite-style ledger flags an in-place value change; that is how "
                     "harness.py closes a task (de: subsumption rejects 46 of 317 commits)")
    if not regressions("runs/tasks.jsonl", t_before, ""):
        fails.append("a rewrite-style ledger does not flag a VANISHED key")
    e_before = json.dumps({"name": "e", "started": "t", "notes": "x"}) + "\n"
    e_after = json.dumps({"name": "e", "started": "t", "notes": "CHANGED"}) + "\n"
    if not regressions("runs/experiments.jsonl", e_before, e_after):
        fails.append("an append-style ledger does NOT flag a changed value; the dispatch has "
                     "collapsed both styles onto the permissive predicate")
    if set(WRITE_STYLE) != set(KEYS):
        fails.append(f"WRITE_STYLE and KEYS disagree on which files exist: "
                     f"{sorted(set(KEYS) ^ set(WRITE_STYLE))}")

    # 4c. score_matrix's key must be the WRITER's key. Keying on `measured` instead makes every
    #     re-score a new key while write_records replaces the row, so each replacement reads as
    #     a vanished key. These are genuinely different partitions of the live file.
    sm = _blob("HEAD", "runs/score_matrix.jsonl")
    if sm:
        n_prof = len({(r.get("ckpt"), r.get("profile", "full")) for r in
                      _rows("runs/score_matrix.jsonl", sm)})
        n_meas = len({(r.get("ckpt"), r.get("measured")) for r in
                      _rows("runs/score_matrix.jsonl", sm)})
        if n_prof == n_meas:
            fails.append("(ckpt, profile) and (ckpt, measured) partition score_matrix.jsonl "
                         "identically here, so this check no longer distinguishes them")

    # 5. Pretty-printed blocks are not rows.
    if regressions(P, base + '  {"when": "x",\n   "evidence": "y"}\n', base):
        fails.append("a pretty-printed block is treated as a row; it is not")

    # 6. The declared-rewrite path returns the manifest rather than swallowing it.
    if not regressions(P, amended, base, "exp: fold\n\nledger-rewrite: dedupe"):
        fails.append("a ledger-rewrite message suppressed the manifest; the caller must decide")
    if not declared_rewrite("x\n\nledger-rewrite: y") or declared_rewrite("plain"):
        fails.append("declared_rewrite does not detect the marker")

    # 7. The second reading must see a value leaving the file even when no key regresses.
    twokeys = R(status="done", notes="aa", finding="F1") + "\n" + \
        json.dumps(dict(name="g", started="v", finding="F2")) + "\n"
    kept_live = R(status="done", notes="aa", finding="F1") + "\n"
    if regressions(P, twokeys, twokeys):
        fails.append("second-reading fixture regresses against itself")
    if not vanished_values(P, twokeys, kept_live):
        fails.append("vanished_values misses a finding that left the file entirely")

    for f in fails:
        print(f"  FAIL {f}")
    if fails:
        print(f"\n{len(fails)} failure(s)")
        return 1
    print("ledger_audit selftest OK: flags a59ac1f (a plain commit reverting an amendment whose "
          "key is shared with the rows it amends -- invisible to merges-only, default history "
          "simplification, and key presence), clean on c3a5a23 (appending a done event), "
          "accepts filling blanks and rejects changed/cleared/dropped values without any "
          "length heuristic, returns declared rewrites as manifests, and the second reading "
          "sees values leaving the file when no key regresses")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", action="append", help="ledger to audit (default: all)")
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()

    total = declared_n = 0
    for path in (a.path or sorted(KEYS)):
        revs, hits = audit(path, a.rev)
        if not revs:
            print(f"=== {path}: never committed on {a.rev}")
            continue
        print(f"=== {path}: {len(revs)} commit(s) touch it (--full-history)")
        for c, p, bad, nvan, kind, subj, decl in hits:
            tag = "DECLARED REWRITE" if decl else "ROLLBACK"
            print(f"    {tag} {c[:9]} {kind:5} vs {p[:7]}  {len(bad)} key(s)  {subj[:44]}")
            for key, _row in bad[:3]:
                print(f"          {str(key)[:66]}")
            if len(bad) > 3:
                print(f"          ... and {len(bad) - 3} more")
            if nvan:
                print(f"          second reading: {nvan} non-empty value(s) left the file")
            declared_n += bool(decl)
        if not hits:
            print("    clean: every key's live row is subsumed in the child")
        total += len(hits)
        print()
    print(f"TOTAL: {total} commit(s) with unsubsumed live rows "
          f"({declared_n} declared ledger-rewrite)")
    # A finding is for a human, not a build break: exit 1 would make the audit unrunnable in a
    # hook. de-33's hook checks HEAD-vs-index, where an undeclared rollback IS a break.
    return 0


if __name__ == "__main__":
    sys.exit(main())
