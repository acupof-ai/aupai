#!/usr/bin/env python3
"""Authoritative ci-selftests red list, reconciled against the issue->PR map.

PREP ONLY -- written read-only ahead of the trigger (fb ruling 2026-09-18). Do not
commit: it runs against whatever tree it is pointed at, and the trigger is ae's
broadcast plus a new #514 head, not a schedule.

Assertions this produces, in fb's numbering:
  1. the four #543 fixes (olses/ledger x3) are green under the driver
  2. every remaining red reconciles to another owner's issue
  3. driver timeout no longer yields 0 bytes (ae's fix)
  4. output is streamable/readable
  5. the F set -- SELFTEST_FILES members with ZERO occurrence in ci.yml -- goes to 0

The F set is COMPUTED here (hook map x ci.yml), never hardcoded: fb's "248->0" is a
direction, and a literal 248 would be a number that rots the moment anyone registers or
excludes a selftest.

Usage:
    python3 scripts/ci_redlist_reconcile.py [--run] [--head <sha>]

    --run    actually execute `harness ci-selftests` (slow, hours on a loaded box).
             Without it, only the F-set and the issue map are computed, and the run
             command is printed for confirmation.
"""
# restartable: read-only apart from one raw-output dump written whole at the end; the
# expensive part is the driver run it invokes, and an interrupt costs only that re-run.
import argparse
import ast
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The branch the red list is about. Its tip must equal the sha the review rows name.
PR_BRANCH = "ae-ci-map-coverage"
PR_NUMBER = 514
#: fb's trigger (2026-09-18): BOTH of these reviewers must have an approving row on main
#: naming the live tip. An earlier version of this check accepted ANY row naming the tip,
#: which passed on genB's row alone while de's was still missing -- and de is the one whose
#: review covers the partition/walker logic this run exists to validate.
REQUIRED_REVIEWERS = ("genB", "de")

# fb's map (2026-09-18). issue number -> the PR that fixes it, or None for "another
# owner, PR not open yet". A red is EXPECTED iff its issue's fix is not an ancestor of
# the HEAD actually run; that is what makes this a reconciliation rather than a checklist.
ISSUE_PR = {
    532: 543,
    533: 543,
    534: 543,
    535: 543,  # merged with #543 (9f88e9fb)
    536: None,
    537: None,  # ae: ledger main-ref / JSONL diagnostics
    538: None,  # ae: sweep fixture (os.getpid pollution)
    539: None,  # de: cache_mmap
    540: 545,
    541: 547,  # 66: phisft
    542: 546,  # de: fsync fixture
}
# #544 is a merge_main CI-gate issue and is NOT a driver red.
NOT_IN_REDLIST = {544}

FIXED_BY_543 = {532, 533, 534, 535}


def git(*a, cwd=ROOT):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)


def hook_selftest_members(root=ROOT):
    """SELFTEST_FILES members, read with ast from the hook's own source.

    ast, not a regex: the map is a local inside main(), and its entries are string
    literals that a line-scan happily matches inside the surrounding comments too.
    """
    src = open(os.path.join(root, "scripts", "hooks", "pre-commit"), encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "SELFTEST_FILES" for t in node.targets):
            continue
        return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    return []


def ci_run_lines(root=ROOT):
    """.yml lines ending in a `- run: <cmd>`, i.e. what CI actually executes."""
    out = []
    p = os.path.join(root, ".github", "workflows", "ci.yml")
    for ln in open(p, encoding="utf-8"):
        m = re.match(r"\s*-\s*run:\s*(.+?)\s*$", ln)
        if m:
            out.append(m.group(1))
    return out


def _py_carries_selftest(txt, rel):
    """FALLBACK ONLY: does this python file dispatch a selftest (not merely mention one)?

    Used when the tree has no `harness._filesystem_selftest_paths` (#514's walker). That
    walker is the primary source because fb's instruction is to reuse it rather than write
    a second enumeration, and because a hand-rolled parallel implementation drifts: measured
    2026-09-18, a structural version written here to match ae's missed 103 of 296 registered
    members against her 0, because her criterion also accepts `flag in sys.argv` compares and
    a print whose literal is the selftest result contract. Re-deriving a criterion from its
    docstring is how a second implementation diverges from the first.

    What THIS version must not do is string-match: `datagen/count_cleaned_code.py` contains
    `--selftest` only inside a docstring sentence, and a substring scan reported 56 such
    prose-only files as carriers -- padding the population, which manufactures gaps.
    """
    import ast

    try:
        tree = ast.parse(txt)
    except SyntaxError:
        return False
    flags = {"--selftest", "--self-check", "--selfcheck", "--selfcheck-only"}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and "selftest" in n.name:
            return True
        if isinstance(n, ast.Call):
            if getattr(n.func, "attr", "") == "add_argument":
                if any(isinstance(a, ast.Constant) and a.value in flags for a in n.args):
                    return True
        if isinstance(n, ast.Compare):
            if any(isinstance(x, ast.Constant) and x.value in flags for x in ast.walk(n)):
                return True
    return False


def _sh_carries_selftest(txt):
    """Shell: strip comments, then look for the flag -- a comment mentioning it is not a test."""
    for ln in txt.splitlines():
        code = ln.split("#", 1)[0]
        if "--selftest" in code or "--self-check" in code or "--selfcheck" in code:
            return True
    return False


def filesystem_selftest_candidates(root=ROOT):
    """Files that CARRY a runnable selftest, found by walking the tree -- not from the map.

    The independent side of the coverage subtraction. Asking SELFTEST_FILES (or the
    partition derived from it) who exists would make the criterion read its own subject --
    the shape de and genB both flagged, and what ae's filesystem enumeration is for.

    This is the INDEPENDENT COUNTERPART to `harness._filesystem_selftest_paths`; the two
    must agree on both trees (main: 248 uncovered; #514: 0). Four defects were measured
    and closed while building it, each of which made the population short or padded:
      * the flag is not always `--selftest` (corpus_fingerprint runs `--self-check`,
        code_fewshot `--selfcheck`) -- 19 of 275 missed by a `--selftest`-only scan;
      * a hardcoded directory list omitted `probes/`, where registered members live;
      * a `.py`-only walk missed every `.sh` member (merge_main.sh, pod_push.sh,
        pod_backup.sh) -- the same blind spot genB found in check_selftests_are_gated;
      * extension filtering dropped `scripts/hooks/pre-commit` (no suffix) and the
        directory derivation never reached root-level `fone.py`.
    """
    hits = set()
    # PRIMARY: the tree's own filesystem walker, when it has one (#514's
    # harness._filesystem_selftest_paths). fb's instruction and the measurement agree here:
    # a second enumeration written in parallel drifts, and ae's criterion -- AST plus the
    # argv-compare and print-contract forms -- is the more complete one. Reusing it is also
    # what makes the two sides of the coverage subtraction genuinely independent: hers walks
    # the filesystem, the partition reads the hook map.
    try:
        sys.path.insert(0, os.path.join(root, "scripts"))
        import harness  # noqa: PLC0415

        if hasattr(harness, "_filesystem_selftest_paths"):
            return set(harness._filesystem_selftest_paths(root)), "ae filesystem walker"
    except Exception:
        pass
    # FALLBACK: a tree without that helper (pre-#514, e.g. main). Structural, never
    # substring -- see _py_carries_selftest for the measurement that settled that.
    dirs = {"scripts", "datagen", "algorithms", "filters", "eval", "mathbank", "probes", "tests", "."}
    for m in hook_selftest_members(root):
        head = m.split("/", 1)[0]
        if head.endswith((".py", ".sh")):
            dirs.add(".")
        elif head:
            dirs.add(head)
    for sub in sorted(dirs):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            if sub == "." and dirpath != root:
                dirnames[:] = []  # root level only; the named dirs are walked above
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
            for fn in filenames:
                # .sh AND no-extension files (scripts/hooks/pre-commit is registered and
                # has no suffix -- the pod_drift SCOPE shape the repo already records).
                if not fn.endswith((".py", ".sh")) and "." in fn:
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                try:
                    txt = open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                if rel.endswith(".sh"):
                    if _sh_carries_selftest(txt):
                        hits.add(rel)
                elif _py_carries_selftest(txt, rel):
                    hits.add(rel)
    return hits, "local structural fallback (no _filesystem_selftest_paths on this tree)"


def partition_buckets(root=ROOT):
    """{path: bucket} from the tree's own driver, or None when it has no partition.

    Read only to ask "which paths does the driver CLAIM", never "which paths exist".
    """
    try:
        sys.path.insert(0, os.path.join(root, "scripts"))
        import harness  # noqa: PLC0415 -- the partition lives in the tree under test

        src = open(os.path.join(root, "scripts", "hooks", "pre-commit"), encoding="utf-8").read()
        return harness._hook_ci_partition(src)
    except Exception:
        return None


def f_set(root=ROOT):
    """(uncovered, how) -- the population with no CI coverage, from two INDEPENDENT sources.

    ON A DRIVER TREE (#514+), the answer fb specified:
        uncovered = filesystem candidates - (every partition bucket)
    The two sides come from different enumerations on purpose. The partition claiming full
    coverage is not evidence of full coverage: it is the object under audit, and a
    predicate that answers by reading its own map is exactly what this round has been
    fixing. Subtracting an independently-walked population from the union of the buckets
    means a file the partition forgot shows up as uncovered rather than as absent.

    ON A PRE-DRIVER TREE, there is no partition to subtract; the fallback is the
    name-matching gap against ci.yml's `- run:` lines. That is the DEFECT's size (genB
    measured 248/277), not a coverage metric, and it is reported as such.

    No number here is fixed: every count is recomputed from the tree at run time, because
    the previous head's figures (266/15/9, 248) move with every registration and every
    bucket the F-fix adds.
    """
    cands, walker = filesystem_selftest_candidates(root)
    buckets = partition_buckets(root)
    reg = set(hook_selftest_members(root))
    missed = sorted(reg - cands)
    # A WEAK WALKER MUST NOT BE SUBSTITUTED FOR A STRONG ONE, and it must not be unioned
    # with one either. Measured 2026-09-18: the local structural fallback misses 69 of 277
    # registered members (it lacks ae's argv-compare and print-contract cases), and
    # unioning it with the map INFLATED the uncovered count to 276/22 -- a weak enumerator
    # adds noise to a subtraction, it does not add coverage. So on a driver tree the
    # population comes from the tree's own walker, and its completeness is asserted by the
    # miss check below rather than patched by a union.
    miss_note = ""
    if missed:
        miss_note = f" | WALK MISSES {len(missed)} registered: {missed[:4]}"
        if buckets:
            # On a driver tree the walker is the tree's own; a miss means it is incomplete
            # and the answer would be answered by a blind spot. Refuse rather than report a
            # number that cannot be trusted.
            return set(), (
                f"WALKER INCOMPLETE ({walker}): misses {len(missed)} registered "
                f"members, first {missed[:3]} -- fix the walker before trusting F"
            )
    pop = set(cands) if buckets else set(cands) | reg
    if buckets:
        covered = set(buckets)
        return {p for p in pop if p not in covered}, (
            f"walk ({len(cands)} via {walker}) - partition buckets ({len(buckets)}){miss_note}"
        )
    runs = "\n".join(ci_run_lines(root))
    return (
        {p for p in pop if p not in runs and os.path.basename(p) not in runs},
        f"walk ({len(cands)} via {walker}) | map ({len(reg)}) vs ci.yml run lines, no partition{miss_note}",
    )


def pr_state(n):
    r = subprocess.run(
        ["gh", "pr", "view", str(n), "--json", "state,mergeCommit"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        return None, None, (r.stderr or "").strip()[:80]
    import json

    o = json.loads(r.stdout)
    mc = (o.get("mergeCommit") or {}).get("oid")
    return o.get("state"), mc, None


def is_ancestor(sha, head):
    if not sha:
        return False
    return git("merge-base", "--is-ancestor", sha, head).returncode == 0


def parse_reds(out):
    """Failing target paths from the driver's output, in order.

    TWO BLIND SPOTS, both measured on the 2026-09-18 digest run, which reported "0
    red(s)" against a driver that had failed 7 targets:

    1. CASE. The driver prints `CI-SELFTESTS:` (harness.py:2528/2545); this matched
       lowercase `ci-selftests:` only. A regex that never matches cannot fail -- it
       returns an empty list, and an empty list reads as "everything green", which is
       the one wrong answer nobody re-checks.
    2. TIMEOUT IS NOT "FAIL". A target killed at the deadline prints
       `CI-SELFTESTS: TIMEOUT after Ns running <rel>` (harness.py:2535), never the FAIL
       line. Fixing only the case would still have hidden every timing-out target --
       and a timeout is the failure most likely to be a real hang.

    The summary line is parsed too, and the two are CROSS-CHECKED by the caller: if the
    driver says "N target(s) failed" and this function's per-target scan found a
    different set, the parse is wrong and must refuse rather than print a short list.
    """
    reds = []
    for ln in out.splitlines():
        m = re.match(r"(?i)ci-selftests:\s+FAIL\s+(\S+)", ln)
        if not m:
            m = re.match(r"(?i)ci-selftests:\s+TIMEOUT\s+after\s+\S+\s+running\s+(\S+)", ln)
        if m and m.group(1) not in reds:
            reds.append(m.group(1).rstrip(":"))
    return reds


def driver_reported_failures(out):
    """The driver's own count and name list from its summary line, or (None, []).

    `CI-SELFTESTS: <n> of <m> target(s) failed: a, b, c` (harness.py:2552). Read so the
    per-target scan above can be checked against the driver's own answer instead of
    being trusted; a parser and its subject are two views, and only one of them is the
    authority on how many failed.
    """
    for ln in out.splitlines():
        m = re.match(r"(?i)ci-selftests:\s+(\d+)\s+of\s+\d+\s+target\(s\)\s+failed:\s*(.*)", ln)
        if m:
            names = [x.strip() for x in m.group(2).split(",") if x.strip()]
            return int(m.group(1)), names
    return None, []


def head_agrees_with_reviews(pr, root=ROOT):
    """(ok, detail) -- the live tip must BE the sha the review rows name.

    fb's trigger condition (2026-09-18): the red list may only run when the reviewers'
    rows are on main AND the branch tip is still the sha they reviewed. A head that moved
    after review makes every finding stale, and a red list is exactly the artifact that
    cannot tell you it was taken against the wrong commit -- it prints the same table.

    Checked at RUN time, not asserted once by hand, because the window is a race: ae may
    push while the run is being set up.
    """
    # FETCH FIRST. A stale local origin/main makes this gate report a FALSE refusal --
    # measured on digest 2026-09-18: its clone was at 856b815d while main had moved to
    # a57ca04f, so the gate named "no review row from ['de']" when de's row was on main
    # already. A refusal is the safe direction, but it is still a wrong answer, and it
    # costs a person a round trip to disprove. The tip read below is a live ls-remote, so
    # only the row source needs refreshing.
    git("fetch", "-q", "origin", "main")
    r = git("ls-remote", "origin", f"refs/heads/{PR_BRANCH}")
    tip = r.stdout.split()[0][:12] if r.stdout.strip() else None
    if not tip:
        return False, f"could not read the tip of {PR_BRANCH}"
    rows = []
    src = git("show", "origin/main:runs/review.jsonl").stdout
    for line in src.splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("pr") == pr:
            rows.append(o)
    if not rows:
        return False, f"no review row for #{pr} on main"
    # Per reviewer: does an APPROVING row from this reviewer name the live tip?
    missing, stale = [], []
    for who in REQUIRED_REVIEWERS:
        mine = [o for o in rows if str(o.get("reviewer", "")).lower() == who.lower()]
        if not mine:
            missing.append(who)
            continue
        naming = " ".join(str(o.get("artifact", "")) + str(o.get("case", "")) for o in mine)
        shas = re.findall(r"\b[0-9a-f]{7,40}\b", naming)
        if not any(s.startswith(tip[:8]) or tip.startswith(s[:8]) for s in shas):
            stale.append(f"{who}({sorted(set(shas))[:2]})")
    if missing:
        return False, f"tip={tip}: no review row from {missing} for #{pr}"
    if stale:
        return False, f"tip={tip}: row(s) name another sha -- {stale}"
    return True, (f"tip={tip} == sha named by {list(REQUIRED_REVIEWERS)}; {len(rows)} row(s) total")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--head", default=None, help="sha the run happens at (default: HEAD)")
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    head = a.head or git("rev-parse", "HEAD").stdout.strip()
    # Exclude the TOOLS COPIED IN, not just this file. fb's prep instruction is "write it,
    # do not commit", and this script travels with helpers (verify_fetch.py), so a
    # self-only exclusion still refused: measured on digest 2026-09-18, the guard named
    # `?? scripts/verify_fetch.py` as tree dirt. That is a tool, not a pending edit, and the
    # failure reads as "dirty tree" rather than as the guard's own scope being wrong.
    # Excluded by exact relpath, never by prefix: an unrelated untracked file under scripts/
    # must still count as dirt, or the guard would stop guarding the thing it exists for.
    tool_rels = {os.path.relpath(os.path.abspath(__file__), ROOT), "scripts/verify_fetch.py"}
    # EXACT PATH MATCH, not substring. `rel in ln` would excuse `scripts/verify_fetch.py.evil`
    # as readily as the real tool -- measured: the substring test returns True for it. A
    # guard a filename can defeat is not a guard, and the whole point of this one is that
    # only genuine tools are excused.
    dirty_lines = []
    for ln in git("status", "--porcelain").stdout.splitlines():
        path = ln[3:].strip().strip('"')
        if path not in tool_rels:
            dirty_lines.append(ln)
    dirty = "\n".join(dirty_lines).strip()
    print(f"HEAD {head[:12]}")
    if dirty:
        # A red list on a dirty tree corresponds to no commit, so the ancestry column
        # below would be answering a question about a tree that does not exist.
        print(
            f"REFUSING: tree is dirty ({len(dirty.splitlines())} path(s)); the red list "
            f"must come from a fixed commit. Commit, or run from a clean worktree."
        )
        print(dirty[:400])
        return 2

    F, how = f_set()
    print(f"F set ({how}): {len(F)}")
    for p in sorted(F)[:15]:
        print(f"    {p}")
    if len(F) > 15:
        print(f"    ... and {len(F) - 15} more")
    print(
        "  assertion 5 wants this 0. The DEFINITION differs by tree and both are printed:"
        "\n    on a driver tree this is the partition's uncovered set (expect 0);"
        "\n    on a pre-driver tree it is the name-matching gap (was 248, the defect's size)\n"
    )

    if not a.run:
        print(
            "dry (no --run). The command to execute on digest:\n"
            "  OMP_NUM_THREADS=4 taskset -c 0-7 python3 scripts/harness.py ci-selftests "
            f"--timeout {a.timeout:.0f}"
        )
        return 0

    # THE TRIGGER, ENFORCED RATHER THAN REMEMBERED. fb's condition: both reviewers' rows on
    # main naming the same sha, and the live tip still that sha. Without this check a red
    # list taken against a superseded head prints exactly as convincing a table as a correct
    # one -- the failure is invisible in the artifact it produces.
    ok, why = head_agrees_with_reviews(PR_NUMBER)
    print(f"trigger check: {why}")
    if not ok:
        print(
            "REFUSING to run: the review rows and the branch tip do not agree. A red list "
            "taken now would describe a commit nobody reviewed. Report the state; do not run."
        )
        return 2

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    cmd = [
        "taskset",
        "-c",
        "0-7",
        sys.executable,
        "scripts/harness.py",
        "ci-selftests",
        "--timeout",
        str(a.timeout),
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    # Assertion 3/4: a timeout that prints nothing is indistinguishable from all-green.
    if not out.strip():
        print(f"REFUSING: driver produced 0 bytes (rc={r.returncode}) -- assertion 3 FAILS")
        return 1
    print(f"driver rc={r.returncode}, {len(out.splitlines())} output line(s)")
    # PERSIST THE RAW OUTPUT BEFORE PARSING IT. The 2026-09-18 run's parser was blind
    # (case + missing TIMEOUT) and the only reason it was caught is that the run happened
    # to print its own summary line into the log. Raw output on disk makes the parse
    # re-runnable, so a parser defect costs a re-parse, not another 20-minute driver run.
    raw_path = os.path.join(ROOT, "runs", "redlist_514_driver_raw.txt")
    try:
        with open(raw_path, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"raw driver output -> {os.path.relpath(raw_path, ROOT)}")
    except OSError as e:
        print(
            f"WARNING: could not persist raw output ({e}) -- the parse below is not "
            f"re-runnable without re-running the driver"
        )
    tail = out.strip().splitlines()[-3:]
    for ln in tail:
        print("  tail:", ln[:160])

    reds = parse_reds(out)
    n_rep, names_rep = driver_reported_failures(out)
    # CROSS-CHECK. The driver's summary is the authority; this parser is a reader of it.
    # A disagreement means the parse is wrong, and printing the short list would be the
    # exact failure this function was written for (a reader that answers "0 reds").
    if n_rep is not None and n_rep != len(reds):
        print(
            f"\nREFUSING: the driver reported {n_rep} failed target(s) but this parser "
            f"found {len(reds)}. Parser is wrong; not printing a red list from it."
        )
        print(f"  driver named: {', '.join(names_rep)}")
        print(f"  parser found: {', '.join(reds) or '(none)'}")
        return 1
    if n_rep is None:
        print(
            "\nWARNING: driver printed no 'N of M target(s) failed' summary line; the "
            "red list below is the per-target scan alone and is UNVERIFIED."
        )
    print(f"\n{len(reds)} red(s) reported (driver agrees: {n_rep})\n")

    rows, no_issue, expected_but_green = [], [], []
    for p in reds:
        issues = [i for i, pr in ISSUE_PR.items() if pr is not None and _pr_touches(pr, p)]
        issues = issues or _issues_by_path(p)
        if not issues:
            no_issue.append(p)
            rows.append((p, "-", "-", "-"))
            continue
        for i in issues:
            pr = ISSUE_PR.get(i)
            st, mc, err = pr_state(pr) if pr else (None, None, None)
            anc = is_ancestor(mc, head) if mc else False
            rows.append(
                (
                    p,
                    f"#{i}",
                    f"#{pr}" if pr else "(no PR)",
                    f"{st or err or '?'}{'/ancestor' if anc else '/NOT-ancestor or unmerged'}",
                )
            )

    for i, pr in ISSUE_PR.items():
        if i in FIXED_BY_543 or i in NOT_IN_REDLIST:
            continue
        if pr:
            st, mc, _ = pr_state(pr)
            if st == "MERGED" and is_ancestor(mc, head):
                if not any(f"#{i}" in r[1] for r in rows):
                    expected_but_green.append((i, pr))

    print(f"{'red':46} {'issue':8} {'fix PR':10} merge-state-vs-HEAD")
    for r_ in rows:
        print(f"{r_[0]:46} {r_[1]:8} {r_[2]:10} {r_[3]}")

    print("\nANOMALIES")
    print(f"  red with no issue ({len(no_issue)}): " + (", ".join(no_issue) or "none"))
    print(
        f"  issue merged-but-not-red ({len(expected_but_green)}): "
        + (", ".join(f"#{i}(PR#{pr})" for i, pr in expected_but_green) or "none")
    )
    print(
        "\nAssertion 2 is 'no anomaly in either group'. Report both groups as-is; do NOT "
        "claim the set is complete -- the map is a snapshot and a PR may merge after it."
    )
    return 0


def _pr_touches(pr, path):
    """Does PR#<pr>'s diff include `path`? Joins a red to its fixing PR by CONTENT.

    Not by issue number alone: two issues can share one PR (#532-535 all landed in #543),
    and a path can be fixed by a PR the map does not name.
    """
    try:
        r = subprocess.run(
            ["gh", "pr", "view", str(pr), "--json", "files"], capture_output=True, text=True, timeout=60
        )
        if r.returncode != 0:
            return False
        import json

        return path in {f["path"] for f in json.loads(r.stdout).get("files", [])}
    except Exception:
        return False


def _issues_by_path(path):
    """Fallback: any mapped issue whose PR touches this path."""
    return [i for i, pr in ISSUE_PR.items() if pr and _pr_touches(pr, path)]


def _selftest():
    """Known-answer worlds for the parser that reported "0 red(s)" against 7 failures.

    The two defects were the case of the marker and TIMEOUT not being a FAIL line. Both are
    driven here on the EXACT strings the driver prints (harness.py:2528/2535/2545), because a
    parser tested on strings it invented proves only that it agrees with itself.
    """
    fail = "CI-SELFTESTS: FAIL datagen/fetch_corpus.py --selftest (exit 1)"
    to = ("CI-SELFTESTS: TIMEOUT after 180s running scripts/test_e1_thing.py -- its process "
          "group was killed (mark it slow/excluded, or fix the hang). Last output before the kill:")
    missing = "CI-SELFTESTS: FAIL scripts/gone.py: registered but missing on disk"
    assert parse_reds(fail + "\n") == ["datagen/fetch_corpus.py"], parse_reds(fail)
    assert parse_reds(to + "\n") == ["scripts/test_e1_thing.py"], parse_reds(to)
    assert parse_reds(missing + "\n") == ["scripts/gone.py"], "the 'missing on disk' form was missed"
    # the lowercase form the first version matched must NOT be the only one that works
    assert parse_reds(fail.replace("CI-SELFTESTS:", "ci-selftests:") + "\n") == ["datagen/fetch_corpus.py"]
    # a passing line is not a red
    assert parse_reds("CI-SELFTESTS: [1/2] RUN x.py --selftest\n") == []
    assert parse_reds("CI-SELFTESTS: 256 target(s) passed\n") == []

    n, names = driver_reported_failures(
        "CI-SELFTESTS: 3 of 256 target(s) failed: a.py, b.py, c.py\n"
    )
    assert (n, names) == (3, ["a.py", "b.py", "c.py"]), (n, names)
    assert driver_reported_failures("CI-SELFTESTS: 256 target(s) passed\n") == (None, []), (
        "a green run must report NO count, not 0 -- 0 is a claim nothing made"
    )
    print(
        "ci_redlist_reconcile selftest OK: FAIL (both cases, and the 'missing on disk' form) "
        "and TIMEOUT both parse; a passing line does not; the summary count is read and a "
        "green run reports None rather than a fabricated 0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
