#!/usr/bin/env python3
"""Merge gate: GO only when EVERY check on a PR is in a completed-passing state.

    python3 scripts/pr_merge_gate.py <pr-number|url|branch>
    echo $?   # 0 = GO, 1 = a check failed, 2 = a check pending, 3 = no checks / gh error

Why this exists: `gh pr checks` prints a table and returns 8 while checks are still
QUEUED/IN_PROGRESS, but a merge wrapper that parses prose, or merges on the CI *event*
rather than the settled set, can race a not-yet-started check (the #441 class). This gate
reads the machine form `gh pr checks --json` and decides from the whole set at once.

FIELD NOTE. `gh pr checks --json` has no `status`/`conclusion` fields -- only `state`
(QUEUED/IN_PROGRESS/SUCCESS/FAILURE/CANCELLED/...) and gh's own normalized `bucket`, one of
pass / fail / pending / skipping / cancel. We decide on `bucket` (documented categorization)
and print the raw `state` beside it. Decision, fail-closed:

  pass      -> GO
  skipping  -> GO (a terminal, conditionally-skipped check; reported, not blocking)
  pending   -> NO-GO (exit 2)
  fail      -> NO-GO (exit 1)
  cancel    -> NO-GO (a cancelled check is not a pass; re-run it) -- exit 1
  absent/"" -> NO-GO (exit 3): an empty check set means "all completed" is false, and a gh
              invocation that failed (auth/network) must never read as GO.

The decision lives in evaluate(), which takes the parsed list and does no IO, so the
mutation selftest can inject a fabricated in-progress check with no network.

MERGEABILITY. Checks alone are not the whole gate: a PR can have every check green and still
be unmergeable, because GitHub refuses on a conflict. genB measured that on 2026-09-19 -- the
gate printed GO while the PR was DIRTY. So `gh pr view --json mergeable,mergeStateStatus` is
read too, and its verdict is composed in decide() (also pure, so the selftest can inject the
gh response at the boundary instead of manufacturing a real conflict).

  CONFLICTING / DIRTY -> NO-GO, regardless of checks
  MERGEABLE + CLEAN|HAS_HOOKS|UNSTABLE -> checks decide (UNSTABLE means non-required checks
      are failing, and this gate already adjudicates the check set itself)
  MERGEABLE + BEHIND -> NO-GO ONLY if the repo requires up-to-date branches. MEASURED
      2026-09-19: `branches/main/protection` is 404 "Branch not protected", `rulesets` and
      `rules/branches/main` are both []. Nothing enforces up-to-date here, so BEHIND passes
      and the fact is printed rather than assumed.
  UNKNOWN / UNKNOWN -> fail open with a WARN. GitHub computes this field lazily and it goes
      stale, so a block on it would be a block on a caching artifact rather than on a real
      conflict; git-cannot-answer fails open elsewhere in this tree
      (scripts/integration_tree.py).
  gh error / field absent -> same fail-open WARN branch. An auth or network failure must never
      become either a GO-on-dirty or a block the checks did not earn.
"""
import argparse
import json
import subprocess
import sys

# buckets that mean "this check is settled and does not block a merge"
# pass/skipping are handled explicitly below; these name the blocking sets.
_FAIL_BUCKETS = frozenset({"fail", "cancel"})
_PENDING_BUCKETS = frozenset({"pending"})


def evaluate(checks):
    """Return (code, verdict, rows). Pure: no network, no sys.exit.

    code: 0 GO; 1 a failing/cancelled check; 2 a pending check (no failure); 3 empty set
    or an unrecognized bucket (fail closed). `rows` is a per-check human-readable list.
    Failure outranks pending, so the operator fixes the red rather than waiting on it.
    """
    rows = []
    n_fail = n_pending = n_pass = n_skip = 0
    for c in checks or []:
        bucket = (c.get("bucket") or "").strip()
        state = (c.get("state") or "").strip() or "?"
        name = c.get("name") or "?"
        wf = c.get("workflow") or ""
        label = f"{wf}/{name}" if wf else name
        if bucket == "pass":
            n_pass += 1
            rows.append(f"  PASS     {label} [{state}]")
        elif bucket == "skipping":
            n_skip += 1
            rows.append(f"  SKIP     {label} [{state}]")
        elif bucket in _PENDING_BUCKETS:
            n_pending += 1
            rows.append(f"  PENDING  {label} [{state}]")
        elif bucket in _FAIL_BUCKETS:
            n_fail += 1
            rows.append(f"  FAIL     {label} [{state}]")
        else:
            # Unknown/absent bucket: fail closed rather than guessing a new gh category is safe.
            n_fail += 1
            rows.append(f"  UNKNOWN  {label} [state={state!r} bucket={bucket!r}]")
    if not checks:
        return 3, "NO-GO: zero checks reported -- 'all completed' is not true", [
            "  (no checks; refusing to treat an empty set as green)"
        ]
    if n_fail:
        return 1, f"NO-GO: {n_fail} failing/cancelled/unknown check(s)", rows
    if n_pending:
        return 2, f"NO-GO: {n_pending} pending check(s)", rows
    return 0, f"GO: {n_pass} passing, {n_skip} skipped, none pending/failed", rows


def fetch(pr=None):
    """Run gh and return the parsed check list, or None if gh itself failed."""
    cmd = ["gh", "pr", "checks"]
    if pr is not None:
        cmd.append(str(pr))
    cmd += ["--json", "name,state,bucket,workflow,link"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode not in (0, 8):  # 8 = checks pending, which is a valid answer here
        sys.stderr.write(p.stderr.strip() or "gh pr checks failed\n")
        return None
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        sys.stderr.write("gh pr checks returned non-JSON output\n")
        return None


def fetch_mergeability(pr=None):
    """Run gh and return (mergeable, mergeStateStatus), or (None, None) if gh itself failed.

    BOTH-OR-NOTHING on failure: a half-read pair must take the fail-open branch, never one
    field's value standing in for the other.
    """
    cmd = ["gh", "pr", "view"]
    if pr is not None:
        cmd.append(str(pr))
    cmd += ["--json", "mergeable,mergeStateStatus"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return None, None
    if p.returncode != 0:
        return None, None
    try:
        obj = json.loads(p.stdout)
    except json.JSONDecodeError:
        return None, None
    return obj.get("mergeable"), obj.get("mergeStateStatus")


def decide(checks, mergeable, merge_state):
    """Compose the checks verdict with GitHub's mergeability. Pure: no IO, no sys.exit.

    Returns (code, verdict, rows). rc semantics are UNCHANGED for the checks-only cases
    (0 GO / 1 failing / 2 pending / 3 empty-or-gh-error); a merge conflict is a new rc 1.
    """
    code, verdict, rows = evaluate(checks)
    rows = list(rows)

    if mergeable is None and merge_state is None:
        rows.append("  WARN     mergeability unreadable (gh error) -- failing open on checks")
        return code, verdict, rows

    m = (mergeable or "").strip().upper()
    s = (merge_state or "").strip().upper()

    if m == "CONFLICTING":
        # UNCONDITIONAL, and independent of the checks code: a green check set on a conflicted
        # PR is exactly the false GO this exists to stop.
        return 1, (f"NO-GO: GitHub reports merge conflict "
                   f"(mergeable={mergeable}, mergeStateStatus={merge_state})"), rows + [
            "  BLOCK    GitHub reports merge conflict (mergeable=CONFLICTING) -- checks are "
            "irrelevant; resolve the conflict"]
    if s == "DIRTY":
        return 1, (f"NO-GO: GitHub reports the PR cannot merge "
                   f"(mergeable={mergeable}, mergeStateStatus={merge_state})"), rows + [
            "  BLOCK    mergeStateStatus=DIRTY -- GitHub cannot merge this PR"]
    if m == "UNKNOWN" or s == "UNKNOWN" or not m:
        rows.append(f"  WARN     mergeability not computed yet "
                    f"(mergeable={mergeable}, mergeStateStatus={merge_state}) -- failing open "
                    f"on checks; GitHub computes this lazily and it is often stale")
        return code, verdict, rows
    if s == "BEHIND":
        # MEASURED 2026-09-19: no branch protection and no rulesets on main, so nothing here
        # requires an up-to-date branch. Printed, not assumed -- re-read before changing.
        rows.append("  NOTE     mergeStateStatus=BEHIND, but main enforces no up-to-date "
                    "requirement (protection 404, rulesets []) -- not blocking on it")
        return code, verdict, rows
    rows.append(f"  MERGEABLE (mergeable={mergeable}, mergeStateStatus={merge_state})")
    return code, verdict, rows


def _selftest():
    def ch(bucket, state=None, name="check", wf="ci"):
        return {"name": name, "workflow": wf, "state": state or bucket.upper(), "bucket": bucket}

    # all pass -> GO
    code, verdict, _ = evaluate([ch("pass", "SUCCESS")] * 3)
    assert code == 0, ("all-pass", code, verdict)

    # THE REQUIRED MUTATION: a single in-progress check among passes must flip to NO-GO.
    for pending_state in ("IN_PROGRESS", "QUEUED", "PENDING"):
        code, verdict, _ = evaluate([ch("pass", "SUCCESS"), ch("pending", pending_state)])
        assert code == 2, (f"pending/{pending_state} must NO-GO", code, verdict)

    # a failed check outranks pending and exits 1
    code, _, _ = evaluate([ch("pending", "QUEUED"), ch("fail", "FAILURE")])
    assert code == 1, ("fail outranks pending", code)
    code, _, _ = evaluate([ch("cancel", "CANCELLED")])
    assert code == 1, ("cancel is not a pass", code)

    # a terminal skip does not block, but is reported
    code, verdict, _ = evaluate([ch("pass", "SUCCESS"), ch("skipping", "NEUTRAL")])
    assert code == 0, ("skip is terminal-neutral", code, verdict)

    # empty set and unknown bucket fail closed -- the thing that makes this a gate
    code, _, _ = evaluate([])
    assert code == 3, ("empty set must not be green", code)
    code, _, _ = evaluate(None)
    assert code == 3, ("None (gh failed) must not be green", code)
    code, _, _ = evaluate([{"name": "x", "state": "WEIRD", "bucket": "future-bucket"}])
    assert code == 1, ("unknown bucket fails closed", code)

    # raw-state-only (no bucket) must not sneak through
    code, _, _ = evaluate([{"name": "x", "state": "SUCCESS", "bucket": ""}])
    assert code == 1, ("missing bucket fails closed", code)
    # --- mergeability. decide() is pure, so the gh response is injected at its boundary: a
    # mutant that needs the network is not a selftest, and manufacturing a real conflict would
    # be a mutation of the live repo.
    green = [ch("pass", "SUCCESS")] * 3

    # a conflicted PR holding a fully green check set: the exact false GO this exists to stop.
    # RED ON OLD: before decide() existed, this set returned 0 GO.
    code, verdict, rows = decide(green, "CONFLICTING", "DIRTY")
    assert code == 1, ("CONFLICTING must NO-GO even on an all-green set", code, verdict)
    assert "merge conflict" in verdict, ("the verdict must name the conflict", verdict)
    assert any(r.strip().startswith("BLOCK") for r in rows), ("no BLOCK row", rows)

    # DIRTY without CONFLICTING: GitHub cannot merge for some other reason
    code, verdict, _ = decide(green, "MERGEABLE", "DIRTY")
    assert code == 1, ("DIRTY must NO-GO", code, verdict)

    # UNSTABLE means mergeable with non-required checks failing; this gate adjudicates the
    # check set itself, so the field must not block on a green set...
    code, verdict, rows = decide(green, "MERGEABLE", "UNSTABLE")
    assert code == 0, ("UNSTABLE on a green set must still GO", code, verdict)
    assert any("UNSTABLE" in r for r in rows), ("the field must be printed", rows)
    # ...but a failing check under UNSTABLE is still rc 1, via the checks path
    code, _, _ = decide([ch("fail", "FAILURE")], "MERGEABLE", "UNSTABLE")
    assert code == 1, ("a failing check under UNSTABLE must NO-GO", code)

    # UNKNOWN is lazily computed and goes stale, so it fails open with a WARN, never a block
    code, verdict, rows = decide(green, "UNKNOWN", "UNKNOWN")
    assert code == 0, ("UNKNOWN must fail open", code, verdict)
    assert any("WARN" in r and "not computed" in r for r in rows), ("no WARN for UNKNOWN", rows)

    # gh could not answer at all (both fields None): same fail-open branch, and it must be loud
    code, verdict, rows = decide(green, None, None)
    assert code == 0, ("gh error must fail open", code, verdict)
    assert any("WARN" in r and "gh error" in r for r in rows), ("no WARN for gh error", rows)

    # BEHIND passes ONLY because main enforces no up-to-date requirement (measured; see the
    # docstring). The row carries the reason so a reader can re-check the premise.
    code, verdict, rows = decide(green, "MERGEABLE", "BEHIND")
    assert code == 0, ("BEHIND must not block while nothing requires up-to-date", code, verdict)
    assert any("BEHIND" in r and "no up-to-date" in r for r in rows), ("no BEHIND note", rows)

    # a clean mergeable green set is still the ordinary GO
    assert decide(green, "MERGEABLE", "CLEAN")[0] == 0, "MERGEABLE/CLEAN must stay GO"

    # rc parity: composing mergeability must not shift the checks-only exit codes
    for cs, want in (([], 3), (None, 3), ([ch("pending", "QUEUED")], 2), ([ch("fail", "FAILURE")], 1)):
        got = decide(cs, "MERGEABLE", "CLEAN")[0]
        assert got == want, ("decide() changed the checks-only rc", cs, want, got)

    print("pr_merge_gate selftest ok: all-pass GO; pending/fail/cancel/empty/unknown NO-GO; "
          "CONFLICTING/DIRTY NO-GO; UNSTABLE/BEHIND/UNKNOWN/gh-error fail open")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pr", nargs="?", help="PR number, url, or branch (default: current branch)")
    ap.add_argument("--selftest", action="store_true", help="offline mutation checks; no network")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return 0
    checks = fetch(args.pr)
    if checks is None:
        print("NO-GO: could not read checks (gh error)")
        return 3
    mergeable, merge_state = fetch_mergeability(args.pr)
    code, verdict, rows = decide(checks, mergeable, merge_state)
    print(verdict)
    for r in rows:
        print(r)
    # AUDITABLE GO: the mergeability this run actually saw gets its own line, so a GO reads as
    # "GO and GitHub said MERGEABLE/CLEAN" rather than leaving the field unexamined.
    print(f"  mergeability: mergeable={mergeable} mergeStateStatus={merge_state}")
    return code


if __name__ == "__main__":
    sys.exit(main())
