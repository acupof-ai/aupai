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
    # `git` ITSELF CAN BE ABSENT, and this raised FileNotFoundError rather than returning what it
    # could read. Found 2026-09-07 while testing the PR source's exit codes with a stripped PATH:
    # the traceback surfaced as a bare exit 1, which merge_main's gate reads as "the lookup ran and
    # found nobody" -- a refusal naming the author for a failure that has nothing to do with them.
    # The docstring already promised "a source that cannot be read contributes nothing"; only the
    # `open()` half implemented it. OSError, not FileNotFoundError: a non-executable git, a
    # permission error and an absent one are all "this source cannot answer".
    try:
        r = subprocess.run(["git", "-C", root, "show", f"{ref}:{rel}"],
                           capture_output=True, text=True)
    except OSError:
        return rows
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


def approver_for(sha, branch, prs, require_body=True):
    """The PR reviewer who APPROVED `sha` with a substantive body, or None.

    THE FLIP'S REVIEW SOURCE (4c's ruling 2026-09-07, after I refused the plain form). Code goes
    through a GitHub PR and the second reader approves there instead of appending a review row.
    The danger in adopting that verbatim is that `gh pr view --json reviews` is NOT equivalent to
    a review row: GitHub approval is a click, while the row requires naming the artifact or
    failing case the reviewer actually opened, and `review_present` FAILs 30 minutes after a close
    if it names neither. Reading bare approval as the row keeps the ceremony and drops the one
    thing that makes a review here more than an ack.

    So approval is NECESSARY, NOT SUFFICIENT: the body must carry `artifact:` or `case:`. 4c
    accepted that; `require_body=False` exists only so the selftest can measure what the plain
    form would have accepted, which is how the difference stays visible rather than asserted.

    THE SHA MUST MATCH, not just the PR. A review approves a commit -- `commit_id` in the API --
    and a PR whose head moved after approval carries an approval of the OLD head. Without this a
    reviewer's approval would silently cover code they never saw, which is the same defect as a
    review row naming another sha (the gate already refuses that one).

    SELF-APPROVAL IS NOT A REVIEW, tested the same way reviewer_for does it: the branch name
    appearing in the login. GitHub refuses self-approval on its own, but the roster's identities
    are branch names rather than GitHub logins -- several sessions push under one account -- so
    GitHub's rule does not cover ours and this must.

    `prs` is the parsed `gh pr view --json reviews` payload (a list of review dicts), passed in
    rather than fetched, because a network call inside a merge gate must be the CALLER's to fail
    loudly on: the caller distinguishes "gh could not answer" from "nobody approved", exactly as
    merge_main distinguishes a broken lookup from a missing row.
    """
    short = sha[:8]
    for rv in prs or []:
        if not isinstance(rv, dict):
            continue
        if str(rv.get("state", "")).upper() != "APPROVED":
            continue
        cid = str(rv.get("commit_id") or rv.get("commitId") or "")
        if cid and not (cid.startswith(short) or short in cid or cid == sha):
            continue
        login = str((rv.get("author") or {}).get("login")
                    if isinstance(rv.get("author"), dict) else rv.get("author") or "").strip()
        if not login:
            continue
        if branch.lower() in login.lower():
            continue  # self-approval, by the roster's identity rather than GitHub's
        body = str(rv.get("body") or "")
        if require_body:
            low = body.lower()
            if "artifact:" not in low and "case:" not in low:
                continue
        return login
    return None


def gh_reviews(sha, branch, timeout=30):
    """(reviews, error) for the open PR whose head is `branch`. A non-empty error means gh could
    not answer, and the CALLER must refuse loudly -- never treat an unreachable GitHub as an
    unreviewed commit.

    THE ERROR IS RETURNED, NOT PRINTED, and the two outcomes are not interchangeable: `([], None)`
    is "gh answered, nobody approved" and the caller refuses with a missing-review message;
    `(None, "...")` is "the gate could not run" and the caller must say so instead. merge_main
    already makes exactly this distinction for the ledger lookup -- it reads stderr rather than the
    exit code, because python exits 1 on a traceback and this script exits 1 for a real answer --
    and a network source has the same trap with more ways to fail.
    """
    try:
        r = subprocess.run(
            ["gh", "pr", "view", branch, "--json", "reviews,headRefOid,state"],
            capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, "gh is not installed"
    except subprocess.TimeoutExpired:
        return None, f"gh did not answer within {timeout}s"
    except OSError as e:
        return None, f"gh could not run: {type(e).__name__}"
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()
        return None, f"gh pr view exited {r.returncode}: {err[-1][:120] if err else '(no stderr)'}"
    try:
        obj = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return None, "gh printed something that is not JSON"
    if not isinstance(obj, dict):
        return None, "gh printed JSON that is not an object"
    return obj.get("reviews") or [], None


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

        # ---- PR APPROVAL, the flip's review source. Ten cases, and the third is the ruling.
        SHA = "abc12345" + "0" * 32

        def rv(state="APPROVED", login="44", body="artifact: scripts/x.py", commit=SHA):
            return {"state": state, "author": {"login": login}, "body": body, "commit_id": commit}

        case("an approval naming an artifact counts",
             approver_for(SHA, "de", [rv()]), "44")
        case("a bare LGTM approval does NOT count",
             approver_for(SHA, "de", [rv(body="LGTM")]), None)
        # THE CONTROL FOR THE RULING ITSELF: the same payload under the plain form 4c first
        # proposed. It must be ACCEPTED there and refused above, or the body requirement is
        # unfalsifiable prose rather than a difference anyone can measure.
        case("...and the plain form would have accepted it, which is the whole objection",
             approver_for(SHA, "de", [rv(body="LGTM")], require_body=False), "44")
        case("'case:' also counts", approver_for(SHA, "de", [rv(body="case: the empty world")]),
             "44")
        case("COMMENTED is not APPROVED",
             approver_for(SHA, "de", [rv(state="COMMENTED")]), None)
        case("CHANGES_REQUESTED is not APPROVED",
             approver_for(SHA, "de", [rv(state="CHANGES_REQUESTED")]), None)
        case("self-approval by the branch's own name does not count",
             approver_for(SHA, "de", [rv(login="de")]), None)
        # AN APPROVAL OF AN EARLIER HEAD does not cover this sha: the reviewer never saw it.
        case("an approval of another commit does not cover this sha",
             approver_for(SHA, "de", [rv(commit="9" * 40)]), None)
        case("no reviews at all reports nobody", approver_for(SHA, "de", []), None)
        # gh RETURNING SOMETHING UNEXPECTED must report nobody, never crash: the caller refuses on
        # empty and the difference between "no approval" and "gh broke" is the caller's to make.
        case("a malformed payload reports nobody, and does not raise",
             approver_for(SHA, "de", [None, "nonsense", {}, {"state": "APPROVED"}]), None)

        # ---- THE CLI's FOUR EXIT CODES, driven through __main__ with a FAKE gh on PATH. The
        # functions above are covered in-process; what merge_main's gate actually branches on is
        # the exit code, and exit 3 (the PR source could not answer) is new. A stub of gh_reviews
        # would test the stub -- the branch under test is the one reading a real subprocess's
        # failure, so PATH gets a `gh` that fails, one that prints garbage, and no gh at all.
        #
        # THE ABSENT-gh CASE FOUND A REAL DEFECT, in code older than this change: an empty PATH
        # removes `git` too, and load_rows' `subprocess.run(["git", ...])` raised
        # FileNotFoundError, which surfaced as a bare exit 1 -- the code the gate reads as "ran
        # fine, nobody reviewed it", so it would have refused naming the author for a failure that
        # had nothing to do with them. Its docstring already promised that an unreadable source
        # contributes nothing; only the open() half implemented it.
        def _fake_gh(body, rc=0):
            bd = tempfile.mkdtemp(prefix="review_lookup_gh_")
            p = os.path.join(bd, "gh")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\n")
                if body:
                    fh.write("cat <<'JSON'\n" + body + "\nJSON\n")
                fh.write(f"exit {rc}\n")
            os.chmod(p, 0o755)
            return bd

        def _exec_path(bindir, *argv):
            """Run the CLI with `bindir` PREPENDED to PATH, or with a PATH holding nothing."""
            env = dict(os.environ)
            env["PATH"] = (bindir + os.pathsep + env["PATH"]) if bindir \
                else tempfile.mkdtemp(prefix="review_lookup_nopath_")
            r = subprocess.run([sys.executable, os.path.abspath(__file__), *argv],
                               cwd=d, capture_output=True, text=True, env=env)
            return r.returncode, r.stdout.strip(), ("did not answer" in (r.stderr or ""))

        _APPROVED = json.dumps({"reviews": [
            {"state": "APPROVED", "author": {"login": "44"},
             "body": "artifact: scripts/x.py", "commit_id": SHA}]})
        _BARE = json.dumps({"reviews": [
            {"state": "APPROVED", "author": {"login": "44"}, "body": "LGTM", "commit_id": SHA}]})
        case("--pr, an approval naming an artifact: rc 0 and the login",
             _exec_path(_fake_gh(_APPROVED), "--pr", SHA, "de"), (0, "44", False))
        case("--pr, a bare LGTM: rc 1, nothing printed",
             _exec_path(_fake_gh(_BARE), "--pr", SHA, "de"), (1, "", False))
        case("--pr, gh exits nonzero: rc 3 and the reason on stderr",
             _exec_path(_fake_gh("", rc=1), "--pr", SHA, "de"), (3, "", True))
        case("--pr, gh prints non-JSON: rc 3",
             _exec_path(_fake_gh("not json at all"), "--pr", SHA, "de"), (3, "", True))
        case("--pr, gh absent (and git absent with it): rc 3, never a traceback",
             _exec_path(None, "--pr", SHA, "de"), (3, "", True))
        case("--pr with too few args is still a usage error",
             _exec_path(_fake_gh(_APPROVED), "--pr", SHA)[0], 2)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    if fails:
        print(f"review_row_lookup selftest: {len(fails)} failure(s)", file=sys.stderr)
        return 1
    print("review_row_lookup selftest OK: main-only, worktree-only and malformed rows all "
          "resolve; self-review and an unreviewed sha both report nobody; a PR approval counts "
          "only with 'artifact:' or 'case:' in its body and only for the sha it approved, and "
          "the plain-form control shows what a bare LGTM would have bought")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    # --pr IS A SECOND SOURCE, NOT A REPLACEMENT (4c's split ruling 2026-09-07). Code lands via a
    # PR and the approval is the review; rulings and non-code keep review.jsonl. Either source
    # satisfies the gate, because a code commit reviewed on the PR has no row and a ruling
    # reviewed in the ledger has no PR -- requiring both would refuse every real case.
    #
    # THE EXIT CODES STAY AS THEY WERE, and gh's failure gets its own one. 0 = a reviewer, printed.
    # 1 = nobody, nothing printed, which is a REAL ANSWER the caller refuses on. 2 = usage. 3 = the
    # gate could not run, with the reason on stderr: merge_main must say "the gate is broken", not
    # "you have no reviewer", and it cannot tell them apart from an empty stdout alone. That
    # conflation is the defect 4c reported on the ledger lookup nine minutes after it shipped.
    _pr = "--pr" in sys.argv[1:]
    _args = [a for a in sys.argv[1:] if a != "--pr"]
    if len(_args) != 2:
        print("usage: review_row_lookup.py [--pr] <sha> <branch> | --selftest", file=sys.stderr)
        sys.exit(2)
    _sha, _branch = _args
    _rev = reviewer_for(_sha, _branch, load_rows("."))
    if _rev is None and _pr:
        _reviews, _err = gh_reviews(_sha, _branch)
        if _err:
            print(f"review_row_lookup: the PR approval source did not answer: {_err}",
                  file=sys.stderr)
            sys.exit(3)
        _rev = approver_for(_sha, _branch, _reviews)
    if _rev is None:
        sys.exit(1)
    print(_rev)
