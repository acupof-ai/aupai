#!/usr/bin/env python3
"""A ledger field that is ABSENT must read as null, never as zero or as "".

44's queue item, 4c's ruling 2026-09-05. Three fields landed this week -- `class` and `cards` on
exp.py start, `defect_caught` on harness task done -- and each is written ONLY when given. That is
the whole design and it is easy to undo by accident: a `default=""` in the parser, or dropping the
conditional and writing the value unconditionally, turns "nobody stated this" into "stated, and the
answer was nothing". Both are one-character-looking changes and neither raises.

WHY IT MATTERS, in the words of the metric that consumes it: policy_metrics.py's metric 3 is
card-hours BY CLASS. A row with no class is a row from before the field existed, and there are 243
of them. If those read as "" the metric would either bucket them all together as one anonymous class
or, worse, count them as a class whose name happens to be empty -- and a backfilled class is a guess
the metric would then trust (4c). The same for cards: a run that held no cards and a run whose cards
nobody recorded are different facts, so a CPU job passes 'none' and the absence stays meaningful.

WHAT IS TESTED. Six worlds over the two writers, each in its own temp ledger (exp.py --root, and
harness's TASKS_PATH rebound), and the assertion is always on the WRITTEN JSON -- not on the parser's
namespace, because a flag declared and never wired reads correct in every other place. That is not
hypothetical: --cards was declared, passed by harness launch, and absent from the row, and only
reading the row showed cards: None (de, 2026-09-05).

  1 class given      -> the key is present with that value
  2 class omitted    -> THE KEY IS ABSENT. Not "", not null-valued: absent, so `"class" in row`
                        is False and a reader can distinguish it from a row that stated nothing
  3 cards given      -> present; and 'none' is a STATED answer, not a synonym for absent
  4 cards omitted    -> absent
  5 defect_caught "" -> PRESENT AND EMPTY. The one field where "" is a real answer: the reviewer
                        read the artifact and found nothing the owner had missed. A writer that
                        skipped "" would erase the difference between a clean review and no review,
                        which is metric 4's whole subject
  6 defect_caught
    omitted          -> absent, meaning no review has reported yet

Worlds 5 and 6 together are the pair that makes the rule non-trivial: for class/cards, "" and absent
must both be unstated; for defect_caught, "" is stated and absent is not. A writer that treated all
three fields the same way would pass a test that only checked one of them.

restartable: yes -- every world is a fresh temp dir removed in a finally; harness.TASKS_PATH and
exp's module globals are restored. Nothing reads or writes the repository's real ledgers.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _run(argv, cwd=ROOT):
    return subprocess.run([sys.executable] + argv, capture_output=True, text=True, timeout=180,
                          cwd=cwd)


def _rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def _exp_world(d, extra):
    """Write one experiments row through exp.py's real CLI and return it.

    Through the CLI, not by importing and calling the writer: the defect this guards against lives
    in the wiring between the parser and the dict, and an in-process call can pass a value the CLI
    never would.
    """
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    # --root is a TOP-LEVEL flag, before the subcommand: it redirects the ledger and is deliberately
    # not an env var (an ambient AUPAI_ROOT would silently redirect a production run's log).
    r = _run([os.path.join(ROOT, "scripts", "exp.py"), "--root", d, "start",
              "--name", "w", "--cmd", "/bin/true"] + extra)
    rows = _rows(os.path.join(d, "runs", "experiments.jsonl"))
    return r, (rows[-1] if rows else None)


def _report(fails):
    if fails:
        print("test_ledger_field_writers FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("test_ledger_field_writers ok: class and cards are written only when given and absent "
          "otherwise (so 243 pre-field rows read null, not \"\"), 'none' is a stated cards answer, "
          "and defect_caught keeps the opposite rule -- '' is written because a review that found "
          "nothing is a different fact from no review")
    return 0


def main():
    fails = []
    tmp = tempfile.mkdtemp(prefix="ledgerfields_")
    try:
        # ---- exp.py start: class and cards ----
        cases = [
            ("1 class given", ["--class", "incremental"], "class", "incremental", True),
            ("2 class omitted", [], "class", None, False),
            ("3 cards given", ["--cards", "1,2"], "cards", "1,2", True),
            ("3b cards 'none' is a STATED answer", ["--cards", "none"], "cards", "none", True),
            ("4 cards omitted", [], "cards", None, False),
        ]
        for label, extra, field, want, present in cases:
            d = os.path.join(tmp, label.split()[0] + field)
            os.makedirs(d)
            r, row = _exp_world(d, extra)
            if row is None:
                fails.append(f"{label}: exp.py start wrote no row at all (rc={r.returncode}): "
                             f"{(r.stderr or r.stdout).strip()[:200]}")
                continue
            if present:
                if field not in row:
                    fails.append(f"{label}: --{field} was passed and the written row has no "
                                 f"{field!r} key. A flag declared and not wired reads correct "
                                 f"everywhere except the row; keys present: {sorted(row)}")
                elif row[field] != want:
                    fails.append(f"{label}: row's {field} is {row[field]!r}, not {want!r}")
            else:
                if field in row:
                    fails.append(f"{label}: --{field} was NOT passed and the row carries "
                                 f"{field}={row[field]!r}. Absent must stay absent: a reader "
                                 f"cannot otherwise tell a row from before the field existed "
                                 f"(243 of them) from a row that stated nothing, and "
                                 f"policy_metrics metric 3 buckets by this value.")

        # ---- harness task done: defect_caught ----
        # The one field where "" is a real answer, so its two worlds are opposites of the above.
        #
        # task done VALIDATES its --commit and --evidence against the real repo: the sha must resolve,
        # must reach main, and must have touched the evidence path. That is deliberate (a delivery in
        # a worktree is not delivered), so this world cannot invent a sha -- it reads a real commit
        # off main and one file that commit actually touched. Rebinding TASKS_PATH keeps the WRITE in
        # a temp ledger while the VALIDATION reads the real history, which is the correct split: the
        # subject here is which keys the writer emits, not the gate in front of it.
        import harness
        # --no-merges, and that is not a detail: main's tip is usually a merge commit, and
        # `git show --name-only` on a merge prints NO files (the diff against two parents is
        # ambiguous, so git shows none by default). Asking for the tip found sha=8cba65f1 with
        # touched=[] and the world had no evidence path to cite -- a fixture that looked broken
        # while the code was fine.
        sha = subprocess.run(["git", "-C", ROOT, "rev-list", "-1", "--no-merges", "main"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        touched = subprocess.run(
            ["git", "-C", ROOT, "show", "--pretty=", "--name-only", sha],
            capture_output=True, text=True, timeout=60).stdout.split()
        ev_path = next((f for f in touched
                        if f.endswith((".py", ".md", ".json", ".jsonl", ".sh", ".txt"))), None)
        if not sha or not ev_path:
            fails.append(f"worlds 5-6 have no subject: could not read a commit on main with a "
                         f"file to cite (sha={sha!r}, touched={touched[:4]}). FAILING rather than "
                         f"skipping -- a field test that silently checks nothing is the shape this "
                         f"repo keeps paying for.")
            raise SystemExit(_report(fails))
        saved_tasks = harness.TASKS_PATH
        try:
            for label, extra, want_present, want_val in [
                ("5 defect_caught empty", ["--defect-caught", ""], True, ""),
                ("6 defect_caught omitted", [], False, None),
            ]:
                d = os.path.join(tmp, label.split()[0] + "task")
                os.makedirs(os.path.join(d, "runs"))
                harness.TASKS_PATH = os.path.join(d, "runs", "tasks.jsonl")
                harness._append_task({"id": "t1", "state": "open", "owner": "de",
                                      "deliverable": "x", "opened": "2026-09-05 00:00"})
                argv = ["done", "t1", "--evidence", ev_path, "--reviewer", "44",
                        "--commit", sha] + extra
                rc = harness.cmd_task(argv)
                rows = _rows(harness.TASKS_PATH)
                done = [r for r in rows if r.get("state") == "done"]
                if not done:
                    fails.append(f"{label}: task done wrote no done event (rc={rc}); the field "
                                 f"cannot be checked because the writer did not run")
                    continue
                row = done[-1]
                if want_present:
                    if "defect_caught" not in row:
                        fails.append(f"{label}: --defect-caught '' was passed and the key is "
                                     f"ABSENT. Empty is a real answer here -- the reviewer read "
                                     f"the artifact and found nothing the owner had missed -- and "
                                     f"dropping it collapses a clean review into no review, which "
                                     f"is metric 4's subject. keys: {sorted(row)}")
                    elif row["defect_caught"] != want_val:
                        fails.append(f"{label}: defect_caught is {row['defect_caught']!r}, not ''")
                else:
                    if "defect_caught" in row:
                        fails.append(f"{label}: no --defect-caught was passed and the row carries "
                                     f"defect_caught={row['defect_caught']!r}. Absent means no "
                                     f"review has reported yet; writing '' would make every "
                                     f"unreviewed close read as a review that found nothing.")
        finally:
            harness.TASKS_PATH = saved_tasks
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return _report(fails)


if __name__ == "__main__":
    sys.exit(main())
