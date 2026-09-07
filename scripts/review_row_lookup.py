#!/usr/bin/env python3
"""Who reviewed a sha: the one implementation, so a test can drive it.

merge_main.sh's review gate held this logic in a heredoc, where nothing could reach it. The
defect 4c reported on 2026-09-07 lived there: it read `runs/review.jsonl` from the invoking
worktree only, and review.jsonl is union-merged, so a branch behind main does not hold rows a
reviewer appended on main. The gate then refused a merge for want of a row that demonstrably
exists -- e1 and de each burned a merge cycle on it the same evening. A heredoc cannot have a
broken world, so the fix is a file with `--selftest` beside it.

BOTH SOURCES, UNIONED, NEITHER AUTHORITATIVE. main lacks a row written in the worktree minutes
ago; the worktree lacks a row main gained since its last merge. So both are read and any row from
either counts. A source that cannot be read contributes nothing and does not mask the other; when
both fail, nothing is printed and the caller refuses, which is the honest answer.

Exit 0 and print the reviewer when one is found; exit 1 and print nothing when none is. The
caller refuses on empty output, so an error path here must never print a name.

# restartable: this writes nothing. It reads two copies of one ledger and prints a name, so an
# interrupt costs one re-read of a file measured in kilobytes; there is no partial state to resume
# from and no shard to write.
"""

import json
import os
import subprocess
import sys


def _rows(text):
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # one malformed row must not hide the rest of the ledger
        if isinstance(obj, dict):
            out.append(obj)
    return out


def load_rows(root=".", ref="main", rel="runs/review.jsonl"):
    """The union of the worktree's ledger and `<ref>:<rel>`. Order is worktree first, which only
    affects which of two matching rows is reported, never whether one is found."""
    rows = []
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            rows += _rows(fh.read())
    except OSError:
        pass
    r = subprocess.run(["git", "-C", root, "show", f"{ref}:{rel}"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        rows += _rows(r.stdout)
    return rows


def reviewer_for(sha, branch, rows):
    """The reviewer of `sha` who is not `branch`, or None.

    SELF-REVIEW IS NOT A REVIEW, and the reviewer field is free text ("b0 (self-reported)"), so
    the test is whether the branch's own name appears in it rather than equality."""
    short = sha[:8]
    for r in rows:
        art = f"{r.get('artifact', '')} {r.get('item', '')}"
        if short not in art and sha not in art:
            continue
        rev = str(r.get("reviewer", "")).strip()
        if rev and branch.lower() not in rev.lower():
            return rev
    return None


def _selftest():
    import shutil
    import tempfile

    fails = []

    def case(name, got, want):
        if got != want:
            fails.append(f"{name}: want {want!r}, got {got!r}")
            print(f"  FAIL {name}: want {want!r}, got {got!r}", file=sys.stderr)
        else:
            print(f"  ok   {name} -> {got!r}")

    d = tempfile.mkdtemp(prefix="review_lookup_")
    try:
        def g(*a):
            return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)

        g("init", "-q", "-b", "main", ".")
        g("config", "user.email", "t@example.invalid")
        g("config", "user.name", "t")
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        led = os.path.join(d, "runs", "review.jsonl")
        sha = "abcdef1234567890abcdef1234567890abcdef12"

        # THE WORLD THE DEFECT LIVED IN: main HAS the row, the worktree does NOT. Built by
        # committing the row, then emptying the worktree copy -- which is what a branch behind
        # main actually looks like after a union merge landed the row elsewhere.
        with open(led, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"artifact": f"commit {sha[:8]}", "reviewer": "44"}) + "\n")
        g("add", "-A")
        g("commit", "-qm", "row on main")
        open(led, "w", encoding="utf-8").close()

        case("main has the row, the worktree does not (4c's defect)",
             reviewer_for(sha, "de", load_rows(d)), "44")
        case("worktree-only read finds nothing, which is what refused the merge",
             reviewer_for(sha, "de", _rows(open(led, encoding="utf-8").read())), None)

        # THE CONTROL, in the other direction: a row present ONLY in the worktree, uncommitted.
        # Without it a lookup that always read main would pass case 1 and silently stop seeing
        # rows written seconds ago.
        sha2 = "9876543210fedcba9876543210fedcba98765432"
        with open(led, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"artifact": f"commit {sha2[:8]}", "reviewer": "b0"}) + "\n")
        case("worktree has the row, main does not", reviewer_for(sha2, "de", load_rows(d)), "b0")

        # A SELF-REVIEW STILL DOES NOT COUNT, from either source. This is the assertion that
        # keeps the union from being a disarm: widening where rows come from must not widen who
        # may sign one.
        with open(led, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"artifact": f"commit {sha[:8]}",
                                 "reviewer": "de (self-reported)"}) + "\n")
        g("add", "-A")
        g("commit", "-qm", "self-review on main")
        case("a self-review on main is not a review", reviewer_for(sha, "de", load_rows(d)), None)

        # AN UNRELATED SHA finds nobody even with rows on both sides.
        case("an unreviewed sha finds nobody",
             reviewer_for("1111111111111111111111111111111111111111", "de", load_rows(d)), None)

        # A MALFORMED ROW does not hide the rest of the ledger.
        with open(led, "w", encoding="utf-8") as fh:
            fh.write("{not json\n")
            fh.write(json.dumps({"artifact": f"commit {sha2[:8]}", "reviewer": "e1"}) + "\n")
        case("a malformed row does not hide the ones after it",
             reviewer_for(sha2, "de", load_rows(d)), "e1")

        # NEITHER SOURCE READABLE: no ref, no file. Must be None, never a name.
        d2 = tempfile.mkdtemp(prefix="review_lookup_empty_")
        try:
            case("no ledger and no main ref reports nobody",
                 reviewer_for(sha, "de", load_rows(d2)), None)
        finally:
            shutil.rmtree(d2, ignore_errors=True)

        # THROUGH __main__, NOT reviewer_for(). Every case above calls the functions in-process,
        # so the argv parse, `load_rows(".")`'s cwd-dependent root and the exit-code contract
        # merge_main.sh consumes via `$(...)` and `[ -z "$row" ]` had no coverage: a mutant that
        # swapped the exit codes, or printed a name on the not-found path, left them all green
        # and made the gate unconditional. The cwd root is the specific line worth pinning --
        # tilerl and 62 each reasoned about what "." resolves to on 2026-09-07 and each got it
        # wrong in a different direction, because the answer is at merge_main.sh:245 (`cd
        # "$MAIN"`) and not in this file.
        def _exec(cwd, *argv):
            r = subprocess.run([sys.executable, os.path.abspath(__file__), *argv],
                               cwd=cwd, capture_output=True, text=True)
            return r.returncode, r.stdout.strip()

        # A FRESH SHA, because the worlds above left rows for `sha` and `sha2` committed on main.
        # Written as `sha`, the self-review case below passed for the wrong reason: the worktree
        # row was skipped as a self-review and the leftover "de (self-reported)" row on main was
        # returned instead, so rc was 0 and the assertion failed while the code was correct.
        sha3 = "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f"
        with open(led, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"artifact": f"commit {sha3[:8]}", "reviewer": "44"}) + "\n")
        case("__main__ found: rc 0 and the name on stdout", _exec(d, sha3, "de"), (0, "44"))
        case("__main__ self-review: rc 1 and nothing on stdout", _exec(d, sha3, "44"), (1, ""))
        case("__main__ unreviewed sha: rc 1 and nothing on stdout",
             _exec(d, "1" * 40, "de"), (1, ""))
        # THE ROOT IS THE CWD. Same ledger, same sha, run from a directory that has neither --
        # the row must not be found. This is the assertion that fails if `load_rows(".")` ever
        # becomes load_rows of anything else.
        d3 = tempfile.mkdtemp(prefix="review_lookup_cwd_")
        try:
            case("__main__ reads the cwd, not the script's directory",
                 _exec(d3, sha3, "de"), (1, ""))
        finally:
            shutil.rmtree(d3, ignore_errors=True)
        case("__main__ wrong argc: rc 2", _exec(d, sha3)[0], 2)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    if fails:
        print(f"review_row_lookup selftest: {len(fails)} failure(s)", file=sys.stderr)
        return 1
    print("review_row_lookup selftest OK: main-only, worktree-only and malformed rows all "
          "resolve; self-review and an unreviewed sha both report nobody")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    if len(sys.argv) != 3:
        print("usage: review_row_lookup.py <sha> <branch> | --selftest", file=sys.stderr)
        sys.exit(2)
    _sha, _branch = sys.argv[1], sys.argv[2]
    _rev = reviewer_for(_sha, _branch, load_rows("."))
    if _rev is None:
        sys.exit(1)
    print(_rev)
