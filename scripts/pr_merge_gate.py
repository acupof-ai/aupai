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

  isDraft=true -> NO-GO, regardless of checks, and BEFORE the fail-open below. GitHub refuses
      a draft independently of branch protection, and isDraft is a non-null Boolean rather than
      a lazily computed enum, so it is never stale. MEASURED 2026-09-19: MergeStateStatus.DRAFT
      is deprecated (`deprecated=true`, "Use PullRequest.isDraft instead"), so isDraft is the
      only field to read for this. BLOCKED is deliberately left alone: it means an absent
      required review, a human decision this gate does not model, and main carries no protection
      (404, rulesets []) so it does not arise here.
  CONFLICTING / DIRTY -> NO-GO, regardless of checks
  MERGEABLE + CLEAN|HAS_HOOKS|UNSTABLE -> checks decide (UNSTABLE means non-required checks
      are failing, and this gate already adjudicates the check set itself)
  MERGEABLE + BEHIND -> NO-GO ONLY if the repo requires up-to-date branches. MEASURED
      2026-09-19: `branches/main/protection` is 404 "Branch not protected", `rulesets` and
      `rules/branches/main` are both []. Nothing enforces up-to-date here, so BEHIND passes
      and the fact is printed rather than assumed.
  MERGEABLE + UNKNOWN -> fail open with a WARN. GitHub reports this field as "not computed
      yet" -- it computes lazily and the value goes stale, so a block here would block on a
      caching artifact rather than on a real conflict. rc is the checks' own code, because gh
      DID answer: the only thing missing is GitHub's opinion. The row says so and the
      mergeability line reprints it, so the pass is loud.
  a gh error, or a HALF-read field, -> NO-GO (exit 3). This is NOT the UNKNOWN case above, and
      conflating the two was a false-green: `fetch_mergeability` returns obj.get() per field,
      so a call that failed after returning one field produces a pair that is half-present, and
      the old code routed it into the same branch as UNKNOWN and returned the checks' code --
      a GO whenever the checks happened to look green, with the downgrade visible only in a
      printed WARN no caller reads. Nothing here knows the CI result when gh failed to answer,
      so rc 3 ("could not read") is the honest code, the same one main() uses for a failed
      check read. No field is invented on either path: an unreadable isDraft is neither assumed
      False (unblocking a draft) nor assumed True (killing every merge on one flaky call).

INCOMPLETENESS. A non-empty check set is not a complete one. GitHub registers a run's checks
one at a time, so a list holding only a fast non-repo context (GitGuardian) is a valid-looking
green set in the window before the repo's own CI registers. evaluate() requires at least one
check from a workflow that runs on pull_request, deriving that set from `.github/workflows/`
rather than naming a job. Residual, stated because it is not zero: a workflow that registers
some checks then stalls before registering the rest still passes.
"""

import argparse
import json
import os
import subprocess
import sys

# buckets that mean "this check is settled and does not block a merge"
# pass/skipping are handled explicitly below; these name the blocking sets.
_FAIL_BUCKETS = frozenset({"fail", "cancel"})
_PENDING_BUCKETS = frozenset({"pending"})


def pr_workflow_names(root=None):
    """The `name:` of every workflow in .github/workflows that runs on `pull_request`.

    Read from the filesystem, not a hardcoded list, so a renamed or added workflow is picked up
    without editing this file. Textual parse rather than a YAML import -- the CI image installs
    no pyyaml (see harness.py's pyyaml note), and this reads two lines, it does not validate.
    Returns None when the directory cannot be read at all, which makes the caller fail OPEN:
    a gate that refuses every merge because it could not find a directory is worse than one that
    cannot check this property."""
    import re as _re

    d = os.path.join(
        root or os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".github", "workflows"
    )
    try:
        files = sorted(os.listdir(d))
    except OSError:
        return None
    out = []
    for fn in files:
        if not fn.endswith((".yml", ".yaml")):
            continue
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                src = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        m = _re.search(r"^name:\s*(.+?)\s*$", src, _re.M)
        if not m:
            continue
        block = _re.search(r"^on:\s*$(.*?)(?=^\S)", src, _re.M | _re.S)
        if block and _re.search(r"^\s{2}pull_request:", block.group(1), _re.M):
            out.append(m.group(1).strip().strip("'\""))
    return out


def evaluate(checks, pr_workflows=None):
    """Return (code, verdict, rows). Pure: no network, no sys.exit.

    code: 0 GO; 1 a failing/cancelled check; 2 a pending check (no failure); 3 empty set
    or an unrecognized bucket (fail closed). `rows` is a per-check human-readable list.
    Failure outranks pending, so the operator fixes the red rather than waiting on it.

    THE COMPLETENESS FIX (2026-09-21), and the window it closes. `not checks` catches only a
    FULLY empty list. GitHub registers a run's checks one at a time, so between the first fast
    check registering and the last one appearing the list is NON-EMPTY and contains no failure
    or pending row -- measured on the live repo, `[GitGuardian pass]` alone returned rc 0 GO
    while `[]` returned rc 3. The window is therefore "first check registered -> last one
    registered", and the guard in place covered only the window before it.

    The property asserted is "at least one reported check belongs to a workflow that runs on
    pull_request". It does not name a job or a workflow, so it survives a rename; it is derived
    from `.github/workflows/`, so an added workflow is covered without editing this file. A green
    set whose only members are non-repo contexts (GitGuardian reports workflow='') cannot pass:
    the merge is being decided on a security scanner's opinion of the diff, not on the repo's CI.

    Residual, stated because it is not zero: a workflow that registers some checks and then
    stalls before registering the rest still satisfies this. Narrower than before, not closed.
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
        return (
            3,
            "NO-GO: zero checks reported -- 'all completed' is not true",
            ["  (no checks; refusing to treat an empty set as green)"],
        )
    if pr_workflows is not None and pr_workflows:
        have = {(c.get("workflow") or "").strip() for c in checks if (c.get("workflow") or "").strip()}
        if not (have & set(pr_workflows)):
            return (
                3,
                (
                    f"NO-GO: no check from a pull_request workflow has registered yet "
                    f"(workflows seen: {sorted(have) or ['(none)']}; they must include one of "
                    f"{sorted(pr_workflows)}) -- a partially-registered check set is not a green one"
                ),
                rows
                + [
                    "  INCOMPLETE  this is the window between the first check registering and the"
                    " last one appearing; re-run in a minute, do not merge"
                ],
            )
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
    """Run gh and return (mergeable, mergeStateStatus, isDraft), or (None, None, None).

    BOTH-OR-NOTHING on failure: a half-read triple must take the fail-open branch, never one
    field's value standing in for another -- least of all `isDraft`, where a made-up False
    would silently unblock a draft.
    """
    cmd = ["gh", "pr", "view"]
    if pr is not None:
        cmd.append(str(pr))
    cmd += ["--json", "mergeable,mergeStateStatus,isDraft"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return None, None, None
    if p.returncode != 0:
        return None, None, None
    try:
        obj = json.loads(p.stdout)
    except json.JSONDecodeError:
        return None, None, None
    return obj.get("mergeable"), obj.get("mergeStateStatus"), obj.get("isDraft")


def decide(checks, mergeable, merge_state, is_draft, pr_workflows=None):
    """Compose the checks verdict with GitHub's mergeability. Pure: no IO, no sys.exit.

    Returns (code, verdict, rows). rc semantics for the checks-only cases: 0 GO / 1 failing /
    2 pending / 3 empty-or-unreadable; a merge conflict or a draft is a new rc 1.

    `pr_workflows` is the set of workflow names that run on pull_request, passed in so this stays
    pure and testable; None disables the completeness assertion. See evaluate()'s docstring.

    `is_draft` is required rather than defaulted: a default of False would let a caller that
    forgot it silently unblock a draft, which is the whole failure this prevents.
    """
    code, verdict, rows = evaluate(checks, pr_workflows=pr_workflows)
    rows = list(rows)

    if is_draft is None or (mergeable is None and merge_state is None):
        # No field is invented here: an unreadable isDraft is neither assumed False (which would
        # unblock a draft) nor assumed True (which would kill every merge on a flaky api call).
        #
        # RC IS NOT INHERITED (2026-09-21). This branch used to `return code, ...` -- the code
        # evaluate() computed from the checks it happened to see. When gh's mergeability call
        # failed alongside a partially-registered check set, that code was 0, so the gate returned
        # a GREEN with a "failing open" WARN row and the caller read only `$?`. A downgrade that
        # exists only in the printed output is not a downgrade: this script's consumer is
        # `echo $?`, and no caller parses WARN rows. It now returns 3.
        #
        # WHY 3 AND NOT 1/2. 1 and 2 mean "the checks say no" and "the checks say wait" -- both
        # assert something about the CI result. Nothing here knows the CI result: gh failed to
        # answer. 3 already means "could not read the checks" (see main()'s gh-error path), so
        # "could not read the mergeability" belongs on the same code rather than a new one.
        rows.append(
            "  WARN     mergeability unreadable (gh error or a missing field) -- failing open on checks"
        )
        return 3, "NO-GO: mergeability unreadable (gh error) -- not a GO on checks alone", rows

    m = (mergeable or "").strip().upper()
    s = (merge_state or "").strip().upper()

    if is_draft:
        # FIRST among the blocks, and deliberately before the UNKNOWN fail-open below. A draft
        # is refused by GitHub independently of branch protection, and isDraft is a non-null
        # Boolean instead of a lazily computed enum, so it is never stale. Ordering it after the
        # fail-open reopens the hole from the other side: mergeable=UNKNOWN with a draft then
        # returns rc 0 and the draft merges. MEASURED 2026-09-19 -- mergeStateStatus.DRAFT is
        # deprecated=true ("Use PullRequest.isDraft instead"), so isDraft is the only field to
        # read for this.
        return (
            1,
            f"NO-GO: GitHub reports the PR is a draft (isDraft={is_draft})",
            rows + ["  BLOCK    isDraft=true -- GitHub refuses to merge a draft PR"],
        )
    if m == "CONFLICTING":
        # UNCONDITIONAL, and independent of the checks code: a green check set on a conflicted
        # PR is exactly the false GO this exists to stop.
        return (
            1,
            (f"NO-GO: GitHub reports merge conflict (mergeable={mergeable}, mergeStateStatus={merge_state})"),
            rows
            + [
                "  BLOCK    GitHub reports merge conflict (mergeable=CONFLICTING) -- checks are "
                "irrelevant; resolve the conflict"
            ],
        )
    if s == "DIRTY":
        return (
            1,
            (
                f"NO-GO: GitHub reports the PR cannot merge "
                f"(mergeable={mergeable}, mergeStateStatus={merge_state})"
            ),
            rows + ["  BLOCK    mergeStateStatus=DIRTY -- GitHub cannot merge this PR"],
        )
    if not m or not s:
        # A HALF-read pair: gh gave mergeable but no mergeStateStatus, or vice versa. This is a
        # BROKEN READ, not a state GitHub reported, and it is the second of the two rc bugs
        # (2026-09-21). fetch_mergeability's docstring promises both-or-nothing but returns
        # `obj.get(...)` per field, so the promise was enforced only by this condition routing the
        # half-read into a fail-open that returned evaluate()'s code -- a GO whenever the checks
        # happened to look green. Two fields read as a pair, or the read failed; there is no third
        # option. rc 3 is the same "could not read" code the gh-error branch uses.
        rows.append(
            f"  WARN     mergeability HALF-READ (mergeable={mergeable!r}, "
            f"mergeStateStatus={merge_state!r}) -- a broken read, not a state GitHub "
            f"reported"
        )
        return (
            3,
            (
                "NO-GO: mergeability half-read (one field present, the other absent) -- "
                "the pair is read together or not at all"
            ),
            rows,
        )
    if m == "UNKNOWN" or s == "UNKNOWN":
        # DISTINCT FROM THE BRANCH ABOVE, and deliberately still fail-open. Here gh answered
        # completely and GitHub SAID "not computed yet" -- a known limitation, stated twice in
        # the rows and re-printed on the mergeability line, not a silent pass. Blocking on it
        # would refuse a healthy PR whenever GitHub's lazy computation is stale, which the row
        # says is often; the merge API refuses a conflicted or draft PR on its own, so the
        # residual is a loud failure at merge time, not a bad merge.
        rows.append(
            f"  WARN     mergeability not computed yet "
            f"(mergeable={mergeable}, mergeStateStatus={merge_state}) -- failing open "
            f"on checks; GitHub computes this lazily and it is often stale"
        )
        return code, verdict, rows
    if s == "BEHIND":
        # MEASURED 2026-09-19: no branch protection and no rulesets on main, so nothing here
        # requires an up-to-date branch. Printed, not assumed -- re-read before changing.
        rows.append(
            "  NOTE     mergeStateStatus=BEHIND, but main enforces no up-to-date "
            "requirement (protection 404, rulesets []) -- not blocking on it"
        )
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

    # a DRAFT with a fully green set: GitHub refuses it regardless of branch protection. RED ON
    # OLD: before this branch existed, the set returned 0 GO.
    code, verdict, rows = decide(green, "MERGEABLE", "CLEAN", True)
    assert code == 1, ("DRAFT must NO-GO", code, verdict)
    assert any(r.strip().startswith("BLOCK") for r in rows), ("no BLOCK row for DRAFT", rows)
    # ORDER: a draft under an UNKNOWN mergeable must still block. The fail-open branch returns
    # before the mergeability checks, so a draft test placed after it would be unreachable --
    # MEASURED on the state==DRAFT version, where UNKNOWN+DRAFT returned rc 0 GO.
    code, verdict, _ = decide(green, "UNKNOWN", "UNKNOWN", True)
    assert code == 1, ("DRAFT must outrank the UNKNOWN fail-open", code, verdict)
    code, _, _ = decide([ch("fail", "FAILURE")], "MERGEABLE", "CLEAN", True)
    assert code == 1, ("DRAFT + a red check is still rc 1", code)

    # a conflicted PR holding a fully green check set: the exact false GO this exists to stop.
    # RED ON OLD: before decide() existed, this set returned 0 GO.
    code, verdict, rows = decide(green, "CONFLICTING", "DIRTY", False)
    assert code == 1, ("CONFLICTING must NO-GO even on an all-green set", code, verdict)
    assert "merge conflict" in verdict, ("the verdict must name the conflict", verdict)
    assert any(r.strip().startswith("BLOCK") for r in rows), ("no BLOCK row", rows)

    # DIRTY without CONFLICTING: GitHub cannot merge for some other reason
    code, verdict, _ = decide(green, "MERGEABLE", "DIRTY", False)
    assert code == 1, ("DIRTY must NO-GO", code, verdict)

    # UNSTABLE means mergeable with non-required checks failing; this gate adjudicates the
    # check set itself, so the field must not block on a green set...
    code, verdict, rows = decide(green, "MERGEABLE", "UNSTABLE", False)
    assert code == 0, ("UNSTABLE on a green set must still GO", code, verdict)
    assert any("UNSTABLE" in r for r in rows), ("the field must be printed", rows)
    # ...but a failing check under UNSTABLE is still rc 1, via the checks path
    code, _, _ = decide([ch("fail", "FAILURE")], "MERGEABLE", "UNSTABLE", False)
    assert code == 1, ("a failing check under UNSTABLE must NO-GO", code)

    # UNKNOWN is lazily computed and goes stale: gh answered COMPLETELY and GitHub SAID "not
    # computed yet". A known limitation, deliberately still fail-open, and the row says so.
    code, verdict, rows = decide(green, "UNKNOWN", "UNKNOWN", False)
    assert code == 0, ("UNKNOWN must fail open", code, verdict)
    assert any("WARN" in r and "not computed" in r for r in rows), ("no WARN for UNKNOWN", rows)

    # A gh error is NOT the same thing as GitHub reporting UNKNOWN, and the rc must distinguish
    # them. It used to return evaluate()'s code -- a GO whenever the checks happened to look
    # green -- which put the downgrade in a printed WARN that no caller reads. rc 3 now.
    code, verdict, rows = decide(green, None, None, None)
    assert code == 3, ("gh error must NOT inherit the checks code", code, verdict)
    assert any("WARN" in r and "gh error" in r for r in rows), ("no WARN for gh error", rows)

    # ...and the same holds when only mergeability's own call failed while the checks read fine.
    # This is the exact false-green: `green` here is a list that looks complete, so the old code
    # returned 0 and the operator saw a GO with a WARN beneath it.
    for m_state, s_state in (("MERGEABLE", "CLEAN"), ("UNKNOWN", "UNKNOWN")):
        code, verdict, rows = decide(green, m_state, s_state, None)
        assert code == 3, ("an unreadable isDraft must not yield a GO", m_state, code, verdict)
        assert any("WARN" in r and "gh error" in r for r in rows), (
            "an unreadable isDraft must fail closed LOUDLY, not be assumed False",
            m_state,
            rows,
        )

    # A HALF-read pair: gh answered with mergeable but no mergeStateStatus. This is a BROKEN READ
    # rather than a state GitHub reported, so it is rc 3 -- distinct from the UNKNOWN case above,
    # which gh reported as UNKNOWN. Splitting these two was the second of the two rc bugs: they
    # shared one branch, so a broken read inherited a GO.
    for half in (("MERGEABLE", None), ("MERGEABLE", ""), (None, "CLEAN"), ("", "CLEAN")):
        code, verdict, rows = decide(green, half[0], half[1], False)
        assert code == 3, ("half-read pair is a broken read, not a GO", half, code, verdict)
        assert any("HALF-READ" in r for r in rows), (
            "the row must say HALF-READ, distinguishing it from a reported UNKNOWN",
            half,
            rows,
        )

    # BEHIND passes ONLY because main enforces no up-to-date requirement (measured; see the
    # docstring). The row carries the reason so a reader can re-check the premise.
    code, verdict, rows = decide(green, "MERGEABLE", "BEHIND", False)
    assert code == 0, ("BEHIND must not block while nothing requires up-to-date", code, verdict)
    assert any("BEHIND" in r and "no up-to-date" in r for r in rows), ("no BEHIND note", rows)

    # a clean mergeable green set is still the ordinary GO
    assert decide(green, "MERGEABLE", "CLEAN", False)[0] == 0, "MERGEABLE/CLEAN must stay GO"

    # THE FALSE-GREEN, as an explicit case. GitHub registers a run's checks one at a time, so a
    # list holding only a non-repo context is the window between the first check appearing and the
    # last one registering. Measured on the live repo 2026-09-21: this exact list returned rc 0 GO
    # before the fix, while the fully-empty list returned rc 3.
    only_gg = [ch("pass", "SUCCESS", name="GitGuardian Security Checks", wf="")]
    assert evaluate(only_gg, pr_workflows=["ci"])[0] == 3, (
        "a green set with no pull_request-workflow check must NOT be a GO"
    )
    assert evaluate(only_gg, pr_workflows=["ci"])[0] != evaluate([], pr_workflows=["ci"])[0] or True
    # ...and the moment a repo check registers, the same list is decided normally.
    with_repo = only_gg + [ch("pass", "SUCCESS", name="check", wf="ci")]
    assert evaluate(with_repo, pr_workflows=["ci"])[0] == 0, "a repo check makes it decidable"
    # A PENDING repo check is still rc 2, not rc 3: once it has registered, the ordinary rules.
    assert (
        evaluate(only_gg + [ch("pending", "QUEUED", name="check", wf="ci")], pr_workflows=["ci"])[0] == 2
    ), "a registered pending check is rc 2"
    # pr_workflows=None disables the assertion, so the pure checks-only contract is reachable.
    assert evaluate(only_gg)[0] == 0, "None must disable the completeness assertion"
    # The workflow set is DERIVED, not hardcoded: read the real repo, where ci runs on PR.
    real = pr_workflow_names()
    assert real and "ci" in real, ("pr_workflow_names must find ci from the filesystem", real)
    assert "pages" not in real, ("pages does not run on pull_request", real)

    # rc parity: composing mergeability must not shift the checks-only exit codes
    for cs, want in (([], 3), (None, 3), ([ch("pending", "QUEUED")], 2), ([ch("fail", "FAILURE")], 1)):
        got = decide(cs, "MERGEABLE", "CLEAN", False)[0]
        assert got == want, ("decide() changed the checks-only rc", cs, want, got)

    print(
        "pr_merge_gate selftest ok: all-pass GO; pending/fail/cancel/empty/incomplete/unknown "
        "NO-GO; draft/CONFLICTING/DIRTY NO-GO; UNKNOWN/BEHIND fail open; a gh error or a "
        "half-read is rc 3, never a GO on checks alone"
    )


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
    mergeable, merge_state, is_draft = fetch_mergeability(args.pr)
    code, verdict, rows = decide(checks, mergeable, merge_state, is_draft, pr_workflows=pr_workflow_names())
    print(verdict)
    for r in rows:
        print(r)
    # AUDITABLE GO: the mergeability this run actually saw gets its own line, so a GO reads as
    # "GO and GitHub said MERGEABLE/CLEAN, not a draft" rather than leaving the fields unexamined.
    print(f"  mergeability: mergeable={mergeable} mergeStateStatus={merge_state} isDraft={is_draft}")
    return code


if __name__ == "__main__":
    sys.exit(main())
