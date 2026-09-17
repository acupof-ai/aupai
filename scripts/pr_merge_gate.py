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
    print("pr_merge_gate selftest ok: all-pass GO; pending/fail/cancel/empty/unknown NO-GO")


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
    code, verdict, rows = evaluate(checks)
    print(verdict)
    for r in rows:
        print(r)
    return code


if __name__ == "__main__":
    sys.exit(main())
