#!/usr/bin/env python3
"""The single place this project's progress is checked, recorded, and advanced.

Two rules:
- A stage is done when the measurement that would falsify it exists and is recorded, not when it produced a file.
- A check without a failing case is not a check: every CHECKS entry carries broken(), and --selftest asserts FAIL on it.

python scripts/harness.py            # check + status
python scripts/harness.py check      # invariants only; exit 1 on any failure
python scripts/harness.py run <step> # the only verb that executes; refuses while check is red
python scripts/harness.py ledger     # provenance and score, one row per checkpoint
python scripts/harness.py gaps       # what is NOT measured, stated out loud
python scripts/harness.py measure    # ...then GO MEASURE IT (full matrix, records itself)
python scripts/harness.py --selftest # every check must fail on its broken world
"""

import argparse
import ast
import functools
import glob
import importlib.machinery
import importlib.util
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Captured HERE, at import, because main() pops every GIT_* from os.environ before anything
# runs (see the comment there: an inherited GIT_DIR let a selftest's `git init` reconfigure
# the shared repository). One reader needs this one variable back: _funcs_in_diff, whose
# `git diff --cached` must see the temporary index a path-scoped commit builds. Read via
# _staged_index_env(), never by putting it back in os.environ.
_ORIG_GIT_INDEX_FILE = os.environ.get("GIT_INDEX_FILE") or ""
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))
import corpus_fingerprint as cfp  # noqa: E402
import pod_drift  # noqa: E402
DATA = os.path.join(ROOT, "data")
SAMPLE_DOMAIN = "sample"  # the only corpus directory a git checkout ships

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"
#: A deadline hit is its OWN state, not a SKIP. A SKIP means "this check does not apply
#: here" -- the hook prints nothing for it and `check` exits 0, so a check that times out
#: on EVERY run is a permanent silent pass wearing a legitimate state's name (44, D5).
#: TIMEOUT says the check was meant to run and did not. It is non-blocking once, because
#: a slow machine is not a defect; twice in a row it is FAIL, because a deadline nothing
#: can meet is a check nobody has.
TIMEOUT = "TIME"

#: The string a committed excerpt of a live log must carry in its LAST lines, and how many
#: lines count as "where tail looks". Both are read by check_snapshot_logs_say_so_at_the_tail
#: and by the trailer in runs/data_leg_206m_8b.log; a change here without a change there
#: turns the check red, which is the intended coupling.
SNAPSHOT_MARK = "END OF SNAPSHOT"
SNAPSHOT_TAIL = 5

# Per-check deadline. A check that hangs blocks the pre-commit hook and trains
# people to --no-verify; a timed-out check reports TIMEOUT and names itself.
_CHECK_TIMEOUT = 5
# Checks that legitimately scan more data than the 5s default allows. The
# template scan reads ~850k text fields on a full-data checkout (27s measured).
_CHECK_TIMEOUTS = {
    "eval_sft_template_contamination": 90,
    # Measured on the pod, 2026-09-01: 0.8s to load the 1.5GB pack, 0.2s to flatten
    # 192M tokens, and 0.127s per probe x 76 probes = 9.7s of search. It was never
    # going to fit 5s, so it timed out on nine consecutive runs and FAILed with
    # "has not actually run since" -- the contamination guard was off for a day while
    # reading as a known-red rather than as absent. 60s is 4x the 14s measured total,
    # which leaves room for a bigger pack without leaving room for a hang.
    #
    # Not indexed instead: a first-4-token sorted index over 192M positions costs 44s
    # to build and 0.49s to search, so it is 3x SLOWER than the linear scan it would
    # replace. Measured before choosing (de).
    "sft_pack_uncontaminated": 60,
    # Measured on this checkout, 2026-09-03: 8.2s wall -- it walks git log once per
    # closed task (86 of them), so it was never going to fit 5s; it timed out on 2
    # consecutive runs and FAILed with "has not actually run since", blocking a commit
    # whose changes it has nothing to say about. 30s is ~4x the measured total.
    "tasks_closed_by_commit": 30,
    # Measured on this checkout under load (load avg 28, 25 users), 2026-09-03:
    # getattr_cfg_names_exist 6.5s, restartability 4.5s (291 files scanned) -- both
    # pass by hand but cross 5s when the shared machine is busy, and each banked
    # 2 consecutive timeout strikes and FAILed a commit with 'has not actually run
    # since'. ~4x the measured wall time, same ratio as the entries above.
    "getattr_cfg_names_exist": 30,
    "restartability": 20,
    # Measured by hand on this checkout, 2026-09-04 (load avg 6.5, 24 users): pod_stamp_is_main
    # WARN in 2.9s, snapshot_logs_say_so_at_the_tail PASS in 9.3s over 84 tracked logs. Both
    # banked 3 consecutive strikes and FAILed a commit with "has not actually run since", and
    # neither has anything to do with what the commit changed -- the fifth and sixth instance of
    # the same shape as the four entries above.
    #
    # WHAT THEY ACTUALLY SPEND, because the fix has to match the cause: pod_stamp_is_main shells
    # out to the pod to read data/pod_synced_head, so its wall time is a network round trip and
    # cannot be optimised locally at all. snapshot_logs reads the last 5 lines of every tracked
    # log and compares against the pod's copies; 9.3s over 84 files is ~0.11s each, so it grows
    # with the log count and will cross any fixed deadline as runs/ fills. That is cost growth,
    # not a hang, which is the distinction the strike mechanism cannot make on its own.
    #
    # ~4x the measured wall, the same ratio every entry here uses. Not larger: the deadline still
    # has to catch a real hang, and a network call that takes 40s IS a hang worth reporting.
    "pod_stamp_is_main": 20,
    "snapshot_logs_say_so_at_the_tail": 40,
}
#: Consecutive-timeout counts, keyed by check name. On disk, not in memory: the point is
#: to notice a check that times out run AFTER run, and each run is a fresh process.
_TIMEOUT_STATE = os.path.join(ROOT, "runs", "check_timeouts.json")
#: Consecutive timeouts before a TIMEOUT becomes a FAIL.
_TIMEOUT_STRIKES = 2


class SelftestSkip(Exception):
    """A broken world cannot be built on this checkout (missing untracked file).
    The check itself SKIPs for the same reason, so the selftest skips too, out loud."""

# --------------------------------------------------------------------------- workspace root
# One root, configured once. AUPAI_ROOT resolves to an absolute path; every data
# location the pipeline steps write derives from it. Default: the repo root.
# The harness refuses a pipeline step whose data path escapes the root, and
# check_root_durable verifies the root is not on a Kubernetes emptyDir.
#
# /work is a Kubernetes emptyDir on the host's root disk -- a pod deletion erases
# it. The durable NVMe drives (/data00-/data03, ~11 TB) are on the host but NOT
# mounted inside the container, so the check detects known-ephemeral mounts
# rather than comparing against durable ones. When the migration mounts /data00
# inside the container, add it to a durable list and invert the check.
EPHEMERAL_MOUNTS = ("/work",)
#: Host NVMe that survives a pod deletion. Not visible inside the container today -- that
#: absence is exactly what makes root_durable a WARN rather than a FAIL.
DURABLE_MOUNTS = ("/data00", "/data01", "/data02", "/data03")


def aupai_root():
    """Resolve AUPAI_ROOT to an absolute path. Default: the repo root."""
    env = os.environ.get("AUPAI_ROOT")
    return os.path.abspath(env) if env else ROOT


def _is_mount(path):
    """A durable drive is a separate filesystem, not merely a directory that exists.
    2026-08-30: another session created /data00/aupai_raw, so os.path.isdir("/data00")
    turned true and this check began advising a move from /work (/dev/vda2, a real disk)
    onto the container's own overlay -- strictly less durable, and it read as a hard red
    that blocked launches. Existence is not a mount."""
    try:
        return os.stat(path).st_dev != os.stat("/").st_dev
    except OSError:
        return False


#: The seven sessions in this round and each one's reviewer. A delivery gets a second
#: reader who is not its author: the controller review with 44 caught four evidenced
#: errors in one day and nobody else's work had one (user order, 2026-08-31 22:00).
REVIEW_PAIRS = {"de": "44", "44": "de", "tilerl": "b0", "b0": "tilerl", "3b": "b0", "e1": "3b", "fb": "44"}
#: How long a dirty or untracked file may sit before the check names it. ONE constant
#: for both: they measure the same thing (work parked in a tree others share) and split
#: values -- 30 min for dirty, 24 h for untracked -- meant the noisier half fired on
#: every session mid-edit while the quieter half slept through a whole day.
#: 6 h (user, 2026-09-01, cutting friction): long enough that an in-progress edit is
#: never named, short enough that nothing survives a working session unowned. Both
#: stay WARN; neither ever blocked a commit.
_AGE_HOURS = 6

#: A review that has not arrived within this many minutes of the done row WARNs.
#: There is no FAIL tier: the user cut the friction on 2026-09-01 -- a missing review
#: blocking a commit makes the reviewer the bottleneck for work already delivered, and
#: the check's job is to make an unread delivery VISIBLE, not to stop the tree. The
#: window still means something: inside it the WARN is routine, past it the evidence
#: line says overdue and names the pair.
REVIEW_GRACE_MIN = 30
#: The rule starts here. 41 tasks closed before it existed and cannot grow a reviewer;
#: failing them would be a permanent red nobody can act on, which is the same as no signal.
REVIEW_RULE_FROM = "2026-08-31 14:00"
#: From when a close must name a commit that reaches main and touches its evidence.
#: Rows closed before this keep their prose evidence; the rule is not retroactive.
TASK_COMMIT_FROM = "2026-09-01 13:30"
PAIR_PRIOR_FROM = "2026-09-02 00:00"
_TASK_STOPWORDS = {"which", "there", "these", "those", "their", "would", "could", "should",
                   "every", "after", "before", "because", "instead", "rather", "without",
                   "against", "already", "still", "cannot", "names", "naming", "state",
                   "write", "writes", "written", "read", "reads", "check", "checks"}

#: Rule bullet (prefix) -> the check that enforces it. The AGENTS.md "Rule coverage"
#: table is the human-readable copy of this map; agents_rules_covered keeps both honest.
_RULE_CHECKS = {
    "The hook runs `--selftest` on staged files in its `SELFTEST_FILES` map":
        "selftests_are_gated",
    # pinned_ids + tokenizer_roundtrip catch a REBUILD after the fact (moved specials,
    # a dropped byte). Neither can see the unfreeze decision itself.
    "Tokenizer frozen 2026-08-29": "pinned_ids",
    "Vocabulary identity": "vocab_id_on_load_path",
    "Long jobs detach": "no_foreground_pod_training",
    "CI gates": "CI",
    "Derived artifacts carry the fingerprint of what produced them": "corpus_fp_matches",
    "Check a launch line's shape against `facts/efficiency.json` before it reaches a card": "launch_line_vs_oom_facts",
    "A fact's source names only checkpoints that still exist": "ckpt_facts_sources_present",
    "setsid, not nohup": "no_foreground_pod_training",
    "CUDA_VISIBLE_DEVICES, not cuda:N": "device_set_honoured",
    "Push code via scripts/pod_push.sh <files>, never bare podput": "pod_drift",
    "Never git stash in this repository": "no_shared_stash",
    "Outbound network: curl -4, always": "curl_ipv4",
    "runs/.jsonl ledgers merge by union": "no_ghost_running",
    "scripts/pod_push.sh pushes only content reachable from main": "pod_drift",
    "A commit that touches a file in the manifest's scope is pushed by its committer": "pod_drift",
    "Corpus directories named by any ladder mix": "ladder_config_frozen",
    "The shared corpus, checkpoints, and GPUs on the pod are unchanged": "pod_drift",
    "8×H20, all usable": "pod_drift",
    "pod is at ~/bin/pod": "pod_drift",
    "uv sync after dependency changes": "env_importable",
}

#: Rule bullets in AGENTS.md that no check can enforce, and why. The count is
#: ratcheted (_MANUAL_BASELINE): "manual" must not become the default answer.
#: A rule enters this list only when enforcement is impossible, not merely awkward.
_MANUAL_RULES = {
    "Card claims live where the job runs":
        "the claim files sit in runs/claims/ of the tree the job runs from and no check "
        "here reads the pod's; scripts/test_launch_claims.py asserts the launch path acquires "
        "and the monitor releases, card_claim.py --selftest asserts a shell pid is refused",
    "A hook edit made in a branch worktree does not run until it is merged":
        "a fact about how git resolves .git/hooks symlinks across worktrees; no artifact "
        "records which hook BODY executed for a given commit, which is exactly why the "
        "mistake is invisible and has to be written down",
    "A dropped tn tunnel does not end the command it started":
        "the surviving process lives in the container and the only record of the dropped tunnel "
        "is a terminal the repo never sees; no_foreground_pod_training catches the launch shape "
        "that produces these orphans, which is the cause, not the post-drop verification",
    "Shared files": "announcing an edit happens in conversation, outside the repo",
    "GPUs": "card ownership is a controller decision, not a file state",
    "A PID is only meaningful in the namespace that read it.":
        "no artifact records which namespace a pid was read in -- the host and the "
        "container both print bare integers and both are correct. A check would need "
        "to know the reader's view, which is not in the repo. The enforceable half is "
        "already covered: pod_drift and lane_respected key on GPU UUID and cmdline",
    "A kill is not finished until nvidia-smi says the card is free":
        "the rule is an operator sequence -- kill, then read the card, then kill what remains. "
        "lane_respected sees the instant, so it catches an orphan that is holding a card NOW, "
        "but nothing in the repo records whether the reader looked after their own kill",
    "Lanes: a 7-card training block, and one lane card for everything else":
        "the lane/block split is allocation policy; lane_respected checks the instant, not the policy",
    "Small jobs queue on the lane card":
        "queueing is operator behaviour over time; lane_respected catches the instantaneous violation",
    "The lane holds one job at a time": "same: lane_respected sees now, not the queue discipline",
    "When there is no lane card at all — `NGPU=8`, as p500m_20b_0902 runs — co-residency "
    "is judged by host IO and seconds, not by metric class":
        "the deciding quantity is host bytes read, which nothing in the repo records per "
        "eval run. scripts/eval_load_cost.py classifies each eval by whether it reaches a "
        "token cache (static, checkable) and carries the three MEASURED costs, but the "
        "measurement itself needs a live training run to differ against",
    "Judge the cost in seconds against what the run already spends on itself, never by the "
    "printed ETA":
        "how a human reads a log field. The fix that IS checkable is on the instrument -- "
        "ETA as a window mean, or the per-interval overrun printed beside it -- and that "
        "edits train.py, frozen for p500m_20b_0902 (de-27, stop-window list)",
    "What is reachable, measured 2026-08-30 with -4": "a record of a measurement, not a rule to enforce",
    "Reachability changes without notice, so a fetcher carries a mirror chain":
        "fetchers do carry chains; asserting 'a chain is present' would match a comment",
    "File transfer into the container: podput <local> <remote-abs-path>":
        "the 100KB cap is enforced by podput itself, which refuses",
    "tn exec and ~/bin/pod are two different filesystem views":
        "a fact about the environment; the mistakes it prevents are interactive",
    "cd inside a backgrounded chain stays in it": "a shell fact; no artifact records the mistake",
    "The pod is frozen from a training launch until that run prints its first step":
        "the window is bounded by two events in different places -- a launch timestamp on the "
        "pod and a push from a laptop -- and nothing records the second. pod_drift sees the "
        "drift that results, which is the consequence; whether a push landed inside someone "
        "else's startup window is not recoverable from any artifact",
    "Stage by path, never `git add -A`": "git history cannot show which command staged a commit",
    "Never run `git checkout` / `git restore` on a file you did not write":
        "no record of who wrote an uncommitted change",
    "Run `ruff format` over a whole file only if you created it": "reformat scope is a review judgement",
    "Commit as soon as a change works": "dirty_aged/untracked_aged enforce the deadline; 'as soon as' is judgement",
    "Each session works in its own worktree on its own branch": "worktree topology is per-machine, not in the repo",
    "A deletion needs a per-file check for glob and runtime loaders":
        "no static analysis sees a runtime glob; reachability.py is a citation graph and its "
        "header says so. vet_programs.py:37 globs math_programs_l*_ext*.py -- 23 live generators "
        "a name scan reads as unreferenced (near-miss, 2026-08-31)",
    "Every delivery has a second reader": "review_present checks the row exists; whether the reviewer actually read the artifact cannot be checked, only that they named one",
    "cfg_default raises rather than returning None": "a note on how checks are written, not a rule to enforce",
    "The ledger takes names from the scores": "a note on how the ledger reads, not a rule to enforce",
    "Commit in your worktree as soon as a change works": "same deadline as above, enforced by dirty_aged",
    "pod_push only ever ADDS: a deletion on main needs a second explicit step on the pod":
        "the deletion is an operator SEQUENCE -- delete here, then delete there -- and the "
        "second half happens on a filesystem no check reads. pod_drift compares the manifest "
        "against the pod, and a file in neither is invisible to it by construction",
    "Only a refusing: line means nothing shipped":
        "how a human reads pod_push's stdout. The transcript is not an artifact, so nothing "
        "records whether the reader's filter could see a refusal at all",
    "data/pod_head_manifest.txt is NOT tracked. scripts/pod_push.sh generates it":
        "pod_drift gates the pod side and .gitignore stops the file being committed by "
        "accident, so the tracking half IS enforced. What stays manual is the push ORDER -- "
        "that the manifest ships AFTER the files, so an interrupted push leaves an old "
        "manifest that reads as drift rather than a new one vouching for files that never "
        "landed. pod_drift's selftest asserts that property against a pod-shaped fixture in "
        "both directions, but whether the real pod_push.sh ran in that order is not "
        "recoverable from any artifact it leaves behind",
    "The index must equal HEAD before you merge: commit your paths, or `git reset`":
        "which order a session ran merge and add in is not recoverable from the repo. What "
        "IS checked is the consequence: a wip commit lands on the branch where dirty_aged "
        "and the behind-main hook see it. The rule's own history is why it stays prose -- "
        "the version before it was a correct measurement of the wrong branch shape "
        "(fast-forward, not three-way), and no artifact records which shape a merge had",
    "A conflicting path needs a commit first, and read which path it is":
        "same -- the sequence happens in a terminal. The consequence IS checked: a wip "
        "commit lands on the branch where dirty_aged and the behind-main hook see it",
    "`harness task` and `harness friction` write the ledger of the tree they are invoked from":
        "the invoking directory is a shell fact no artifact records; the integration tree's "
        "pre-commit hook refuses the resulting non-controller commit, which is the consequence",
}
#: Ratchet, a LITERAL. `len(_MANUAL_RULES)` would move with the thing it pins and the
#: check could never fire -- the ratchet has to be a number a commit has to change.
#: Raising it needs a message saying which rule became unenforceable and why.
#:
#: 22 -> 23 on 2026-09-01: "a kill is not finished until nvidia-smi says the card is free".
#: Manual because the rule is an operator SEQUENCE -- kill, read the card, kill what
#: remains -- and no artifact records whether the second step happened. lane_respected
#: catches an orphan that is holding a card at the instant it runs, which is the
#: consequence, not the discipline. The rule was written because the consequence went
#: unnoticed for two minutes with nobody looking (eval/run_eval.py pid 313429 on GPU7).
#: 23 -> 24 on 2026-09-01 for the hook-symlink rule. Which became unenforceable and
#: why: .git/hooks/pre-commit is a symlink resolved against MAIN's worktree, so every
#: worktree runs main's copy and a hook edit in a branch is inert until merged. No
#: artifact records which hook BODY executed for a given commit -- that is precisely
#: what makes the mistake invisible, and precisely why it cannot be checked. The
#: companion rule in the same commit went the other way, to check_selftests_are_gated,
#: so the pair is +1 manual and +1 checked rather than +2 manual.
#: 24 -> 25 on 2026-09-01: "a PID is only meaningful in the namespace that read it".
#: The host and the container print bare integers for the same process and both are
#: correct; nothing in a command or a log records which view produced one. A check
#: would have to know the reader's namespace, which is not a repo fact. The
#: consequences ARE checked -- pod_drift and lane_respected key on GPU UUID and
#: cmdline, the two identities that survive the boundary -- but the discipline of
#: reading and killing in the same view is not, and a guard on [ -d /proc/<pid> ]
#: written across it evaluated false on its first pass and launched a job onto a
#: running probe's cards.
#: 25 -> 28 on 2026-09-02, three pod-side rules, all unenforceable for the same reason:
#: the mistake happens on the pod's filesystem or in a terminal, and neither is an
#: artifact this repo can read. pod_push's add-only asymmetry is a two-place sequence
#: whose second half no check sees -- 69 files deleted from main were still on the pod
#: with pod_drift green, because the manifest asserts that the files it LISTS match and
#: says nothing about files in neither side. The `refusing:` rule is about reading
#: stdout, and a `| tail -2` that ate the refusal leaves no trace of having done so.
#: --write vs --write-index is not recoverable after the fact: both produce the same
#: filename, and a manifest built from the pre-merge HEAD is well-formed and wrong.
#: 28 -> 30 on 2026-09-02, the two behind-main sequencing rules from tilerl-16. Both are
#: orders a person types in a terminal -- merge before staging, wip-commit only a
#: conflicting path -- and no artifact records the order. The rule they replace IS
#: checked (no_shared_stash), which is why the pair is +1 checked and +2 manual rather
#: than +3 manual: the enforceable half of "never stash" is the stack itself.
#:
#: 30 -> 32 (de-27, 2026-09-02). Two co-residency rules under Lanes, and each names the
#: exact quantity nothing here can read:
#:   1. "co-residency is judged by host IO and seconds": the deciding quantity is host
#:      bytes read per eval run, which no artifact records. scripts/eval_load_cost.py
#:      classifies statically (does this file reach a token cache) and carries the three
#:      measured costs, but the measurement needs a live training run to difference
#:      against -- 46s/109s/209s came from p500m_20b_0902's own rate series.
#:   2. "judge in seconds, never by the printed ETA": how a human reads a log field. The
#:      checkable half is on the instrument, not the operator -- ETA as a window mean, or
#:      the per-interval overrun printed beside it -- and that edits train.py, frozen for
#:      p500m_20b_0902. When that lands, this comes back to 31 or 30.
#: 32 -> 34 (de, 2026-09-02), and both are temporary for stated reasons, not new
#: unenforceable ground:
#:   "The pod is frozen from a training launch until that run prints its first step" is
#:   manual by nature -- the window is bounded by a launch on the pod and a push from a
#:   laptop, and nothing records the second, so no artifact can say whether a push landed
#:   inside someone's startup window. This one does not come back down.
#:   "Check a launch line's shape against facts/efficiency.json" is manual only until
#:   44-20 lands. Both sides are static -- the launch line's (batch, accum, seq, layers)
#:   and the fact store's config blocks -- so it is fully checkable; it just is not
#:   written yet. When 44-20 lands this returns to 33.
#: 34 -> 33 (44-20, 2026-09-02): the launch-line check landed as
#: launch_line_vs_oom_facts, so the rule above moved to _RULE_CHECKS. It was manual
#: only until written, not manual by nature -- both sides are static.
_MANUAL_BASELINE = 36


def _norm_rule(text):
    """Rule keys and AGENTS.md bullets compared on one normal form: markdown stripped,
    whitespace collapsed, trailing punctuation dropped. Without this the map needs a
    key per punctuation variant, and a bullet that gains a backtick silently unmaps."""
    return re.sub(r"\s+", " ", re.sub(r"[`*_]", "", text)).strip().rstrip(":. ").lower()


def _agents_rule_bullets(root):
    """Every bold-lead rule bullet under Hard constraints / Pod / the coordination block."""
    p = os.path.join(root, "AGENTS.md")
    if not os.path.exists(p):
        return None, "AGENTS.md missing"
    text = open(p, encoding="utf-8").read()
    lines = text.split("\n")
    spans, cur = [], None
    for i, line in enumerate(lines):
        if re.match(r"^## ", line):
            if cur:
                spans.append((cur, i))
                cur = None
            if re.match(r"^## (Hard constraints|Pod|Coordination)", line):
                cur = i
            # the coordination rules live under "Rules kept from before the reset"
            elif "Rules kept from before" in line:
                cur = i
    if cur:
        spans.append((cur, len(lines)))
    out = []
    for a, b in spans:
        for i in range(a, b):
            m = re.match(r"^\s*-\s+\*\*(.+?)\*\*", lines[i])
            if m:
                out.append(m.group(1).rstrip(":. "))
            elif re.match(r"^- [A-Z`]", lines[i]) and len(lines[i]) > 40:
                out.append(re.sub(r"[`*]", "", lines[i][2:])[:60].rstrip(":. "))
    return out, None


def check_reported_path_is_written(root):
    """A runner that reports a preds path reports the one it WROTE.

    open_artifact(path, run=...) versions the path -- it writes preds_x.<run>.jsonl and
    hands the real name back as fout.name. Four runners captured that into out_path,
    attested out_path correctly, and then printed and recorded the UNVERSIONED
    preds_path. The log's last line named a file that does not exist, and
    l1_fewshot's --out JSON carried it in a machine-readable field.

    It cost an hour on 2026-09-01: the 16B code cell finished, the log said
    `preds saved: ...k8.jsonl`, that file was absent, and the result read as a dead run
    until the versioned file turned up in a directory listing. The attest call was right
    the whole time, which is why nothing caught it -- the correct value was computed and
    then not used.

    Source-level, because it is a wiring defect: in a function that binds out_path from
    an open_artifact handle, a later `preds_path` in a print or a dict value is the
    stale name. Only the same function is examined, so a runner that never versions is
    not implicated.

    THE MARKER IS THE DEFECT'S SHAPE, NOT THE NAME (6e, from e1's 29b31367). This matched any
    LOAD of a Name called `preds_path`, so it refused a correct CALL to a function of that name
    -- and e1 renamed a new function to `artifact_path` to get past it, recording the rename's
    reason in its own docstring. A rename to satisfy a check is the check making the codebase
    worse: the identifier was never the defect. The defect is a variable ASSIGNED IN THIS
    FUNCTION and then reported instead of out_path, so both halves are now required:

      - the name must be BOUND somewhere in the same function (assignment, with-as, for
        target, walrus). `preds_path(...)` calls a function defined elsewhere and binds
        nothing here, so it is not the shape; `preds_path = build(...)` then
        `print(preds_path)` is.
      - it must be LOADED outside the open_artifact call that would version it.

    A module-level or imported `preds_path` reads as unbound here and passes, which is
    correct: it cannot hold this function's stale versioned name. Verified against both worlds
    -- the historical defect still FAILs, and e1's call form passes without a rename.
    """
    bad = []
    files = 0
    for fp in sorted(glob.glob(os.path.join(root, "eval", "*.py"))
                     + glob.glob(os.path.join(root, "probes", "*.py"))):
        try:
            with open(fp, encoding="utf-8") as fh:
                src = fh.read()
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        if "open_artifact" not in src:
            continue
        files += 1
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            body = ast.dump(fn)
            if "open_artifact" not in body or "out_path" not in body:
                continue
            # Is `preds_path` a LOCAL VARIABLE of this function? Every binding form, because a
            # walrus or a with-as holds a versioned name just as an assignment does.
            bound = False
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and node.id == "preds_path" \
                        and isinstance(getattr(node, "ctx", None), ast.Store):
                    bound = True
                elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name) \
                        and node.optional_vars.id == "preds_path":
                    bound = True
            if not bound:
                # A call to a function named preds_path, or an import, binds nothing here and
                # cannot carry a stale versioned path. e1's 29b31367 is this case.
                continue
            for node in ast.walk(fn):
                # print(f"... {preds_path}") and {"preds_path": preds_path}
                if isinstance(node, ast.Name) and node.id == "preds_path":
                    if isinstance(getattr(node, "ctx", None), ast.Store):
                        continue
                    parent_is_call = False
                    for anc in ast.walk(fn):
                        if isinstance(anc, ast.Call) and anc.func.__class__ is ast.Name \
                                and getattr(anc.func, "id", "") == "open_artifact" \
                                and any(a is node for a in anc.args):
                            parent_is_call = True
                        # ALSO not the defect: `preds_path(...)`, where the Name is the callee
                        # rather than a value being reported. Without this, a function that both
                        # binds the name and calls something of that name would be flagged at the
                        # call site -- the same false positive one level in.
                        if isinstance(anc, ast.Call) and anc.func is node:
                            parent_is_call = True
                    if not parent_is_call:
                        bad.append(f"{os.path.relpath(fp, root)}:{node.lineno} in "
                                   f"{fn.name}() reports preds_path, not out_path")
    if bad:
        return FAIL, "; ".join(sorted(set(bad))[:4])
    return PASS, f"{files} runner(s) using open_artifact report the path they wrote"


def _broken_reported_path():
    """eval/l1_fewshot.py AS IT WAS AT 47cb01c2~1 -- the historical defect itself, read out of
    git rather than reconstructed.

    THE PREVIOUS WORLD STOPPED REPRODUCING THE DEFECT AND STAYED GREEN (found 2026-09-04 while
    fixing the Name-vs-Call false positive). It took today's file and reverted the print from
    out_path to preds_path. That was the whole edit in 47cb01c2, so it was a faithful world at
    the time -- but e1's 29b31367 deleted the `preds_path` VARIABLE, moving the name into a
    function called artifact_path. Reverting only the print then leaves `preds_path` unbound: a
    NameError at runtime, not the stale-name defect, and once the check required the name to be
    a local variable that world reported PASS. The mutation was still applied, the assert on
    the fixed string still held, and nothing was red.

    The real shape, at 47cb01c2~1: `preds_path` assigned (:168), passed to open_artifact which
    versions it (:174), the versioned name captured into out_path (:176), attest(out_path)
    correct (:201), and then :208/:215 print and record preds_path. The correct value was
    computed and then not used, which is why nothing caught it for a day.

    Read from git, so it cannot drift with the live file again -- the failure above was a world
    coupled to code that moved underneath it.
    """
    import shutil

    d = _tmp_repo_shaped()
    os.remove(os.path.join(d, "eval"))
    os.makedirs(os.path.join(d, "eval"))
    for f in glob.glob(os.path.join(ROOT, "eval", "*.py")):
        shutil.copy(f, os.path.join(d, "eval", os.path.basename(f)))
    r = subprocess.run(["git", "-C", ROOT, "show", "47cb01c2~1:eval/l1_fewshot.py"],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode or not r.stdout:
        raise SelftestSkip("47cb01c2~1 is not in this repository; the historical defect is "
                           "unavailable and a reconstructed world is what just failed")
    src = r.stdout
    # The world must hold the defect, not merely an old file: assert the three parts.
    assert "preds_path = os.path.join(" in src, "the historical file does not bind preds_path"
    assert "out_path = fout.name" in src, "the historical file does not capture the versioned name"
    assert 'print(f"preds saved: {preds_path}")' in src, "the historical file does not report it"
    with open(os.path.join(d, "eval", "l1_fewshot.py"), "w", encoding="utf-8") as f:
        f.write(src)
    return d


def _positive_reported_path():
    """THE FALSE POSITIVE, as a world: e1's shape, where `preds_path` is a FUNCTION being
    called. Returns a tmp root the check must report PASS on.

    Not a broken world -- CHECKS holds one per row and that slot carries the defect. This is
    the other half, and without it the fix is unverified in the direction it was made: the
    check refused a correct call to a function of this name (e1's commit 29b31367 records
    renaming to artifact_path to get past it), so the world that matters asserts the call form
    is accepted. A check made permissive by a mistake would pass the FAIL world too.
    """
    import shutil

    d = _tmp_repo_shaped()
    os.remove(os.path.join(d, "eval"))
    os.makedirs(os.path.join(d, "eval"))
    for f in glob.glob(os.path.join(ROOT, "eval", "*.py")):
        shutil.copy(f, os.path.join(d, "eval", os.path.basename(f)))
    p = os.path.join(d, "eval", "l1_fewshot.py")
    with open(p, encoding="utf-8") as f:
        src = f.read()
    # e1's own code with artifact_path renamed BACK to the name the check used as its marker.
    # Nothing else changes, so if this FAILs, the marker is still the name.
    assert "def artifact_path(" in src, "l1_fewshot no longer defines artifact_path"
    src = src.replace("artifact_path(", "preds_path(").replace("def preds_path(", "def preds_path(")
    with open(p, "w", encoding="utf-8") as f:
        f.write(src)
    return d


def check_snapshot_logs_say_so_at_the_tail(root):
    """A tracked runs/*.log that is a TRUNCATED copy of a live pod log says so in its last
    lines, not only its first.

    A committed excerpt of a running job's log is a useful artifact -- it pins the numbers a
    fact cites without carrying 3 MB. It is also indistinguishable from the live log to the
    tool everyone reads it with. `tail -3` never shows line 1, so a header saying "excerpt,
    pulled 08:0xZ" is invisible at exactly the moment someone reads the file for current
    state. MEASURED COST (b0, 2026-09-03): tailed runs/data_leg_206m_8b.log and reported the
    leg at step 6450/42% to the controller while the pod was at 9950/65%. The header was
    already there and already correct. Both readings were well-formed.

    So the marker has to be where the reading tool looks. This check is the marker's guard:
    the file must carry SNAPSHOT_MARK inside its last few lines.

    SCOPE -- why "smaller than the pod's copy" and not "any tracked log": a log whose job has
    ENDED is a complete record, and its local copy matching the pod byte for byte is the
    normal, correct state for 43 of the 50 tracked logs the pod also has. Only a local copy
    materially SHORTER than the pod's is an excerpt, and only an excerpt can mislead about
    current state. A check that demanded the marker everywhere would put a "not live" trailer
    on 43 files where it is false.

    Runs on the pod only, because the comparison needs both copies."""
    pod = os.path.expanduser("~/bin/pod")
    if not os.path.exists(pod) or pod_drift.is_pod(root):
        return SKIP, "host-side check; needs ~/bin/pod to read the live copies"
    tracked = subprocess.run(["git", "ls-files", "runs/"], capture_output=True, text=True,
                             cwd=root).stdout.split()
    logs = [t for t in tracked if t.endswith(".log")]
    if not logs:
        return SKIP, "no tracked runs/*.log"
    script = ("cd /work/aupai && for f in %s; do if [ -f \"$f\" ]; then "
              "echo \"$f $(stat -c%%s \"$f\")\"; fi; done" % " ".join(logs))
    r = subprocess.run([pod, script], capture_output=True, text=True, timeout=240)
    if r.returncode != 0:
        return SKIP, f"cannot read the pod: {r.stderr.strip()[:80]}"
    sizes = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            sizes[parts[0]] = int(parts[1])
    bad, excerpts = [], 0
    for rel in logs:
        p = os.path.join(root, rel)
        if rel not in sizes or not os.path.exists(p):
            continue
        local = os.path.getsize(p)
        # 1.5x, not "any difference": a live log grows between the pull and this read, so a
        # few hundred bytes is the pull's own latency and not an excerpt.
        if sizes[rel] <= local * 1.5:
            continue
        excerpts += 1
        tail = open(p, encoding="utf-8", errors="replace").read().splitlines()[-SNAPSHOT_TAIL:]
        if not any(SNAPSHOT_MARK in ln for ln in tail):
            bad.append(f"{rel} (local {local:,}B vs pod {sizes[rel]:,}B)")
    if bad:
        return FAIL, (
            f"{len(bad)} truncated snapshot(s) of a live pod log carry no {SNAPSHOT_MARK!r} in "
            f"their last {SNAPSHOT_TAIL} lines, so `tail` on them reads as current state: "
            f"{'; '.join(bad[:4])}"
            + (f" ... and {len(bad) - 4} more" if len(bad) > 4 else ""))
    return PASS, (f"{excerpts} of {len(logs)} tracked log(s) are truncated excerpts, each "
                  f"marked in its last {SNAPSHOT_TAIL} lines; the rest match the pod")


def _broken_snapshot_logs_say_so_at_the_tail():
    raise SelftestSkip(
        "the broken world is a PAIR of filesystems -- a local excerpt beside a longer pod copy "
        "-- and this check reads the live pod through ~/bin/pod with a hardcoded /work/aupai. "
        "Staging it would mean truncating a real pod log. The FAIL path is exercised instead by "
        "the mutation recorded in the commit: dropping the trailer from "
        "runs/data_leg_206m_8b.log turns this check red while its header still says 'excerpt'.")


def check_cited_artifacts_attested(root):
    """A fact citing a gitignored artifact carries a sha256 some attestation matches.

    data/eval/preds_*.jsonl is gitignored and nothing reads it programmatically, so
    fact_refs_resolve skips those paths on every machine: a fact could cite an artifact
    that exists nowhere and nothing would notice. That is how an unlogged rerun
    overwrote preds_l1_d3.jsonl and left five facts pointing at 477 rows of a different
    run for hours (e1, 44's contract, 2026-08-31).

    What this proves is historical -- the cited bytes existed at that path when the
    citation was made. It deliberately does NOT compare against the current file: preds
    are regenerated every run, so a current-state check would fail on every legitimate
    rerun. The writer's attestation row is the proof.

    The match is on (path, sha256), not the hash alone. Hash-only accepted an
    attestation of a DIFFERENT path carrying the same bytes -- which is precisely the
    versioned-write case the attestation exists for: open_artifact(path, run=...) writes
    preds_x.r1.jsonl while a careless writer attests preds_x.jsonl, and identical bytes
    at two paths is the normal state during a rerun, not a coincidence
    (tilerl T7-1, probe probes/t7_attest_path.py@41587cb)."""
    refs = os.path.join(root, "runs", "artifact_refs.jsonl")
    attested = set()  # (basename, sha256)
    if os.path.exists(refs):
        for line in open(refs, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("sha256") and r.get("path"):
                # Basename, not the full path: a fact cites the repo-relative path while
                # the writer attests whatever path it opened, which differs by the
                # writer's cwd. The versioned SUFFIX -- the part T7-1 is about -- lives
                # in the basename, so it is still compared.
                attested.add((os.path.basename(r["path"]), r["sha256"]))
    # The contract starts here. 18 citations predate it and cannot grow an attestation
    # retroactively -- their artifacts were written before any writer attested, and
    # several no longer exist. Failing them is a red nobody can act on, which is the
    # same as no signal. New and re-measured facts carry the hash.
    contract_from = "2026-09-01"
    cited, bad, legacy, unattestable = 0, [], 0, []
    for fp in sorted(glob.glob(os.path.join(root, "facts", "*.json"))):
        try:
            obj = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for e in obj.get("facts", []):
            blob = json.dumps(e, ensure_ascii=False)
            for m in re.finditer(r"(data/eval/[\w./-]+\.jsonl)", blob):
                path = m.group(1)
                if (e.get("measured") or "") < contract_from:
                    legacy += 1
                    continue
                cited += 1
                base = os.path.basename(path)
                # artifact_sha256 is a string for a one-artifact fact and a
                # {basename: sha} object for a fact that cites several. A single string
                # could not express a restatement that rescores three artifacts at once,
                # and the alternative -- splitting one measurement across three facts so
                # the field fits -- would shape the record around the guard (de,
                # 2026-09-01).
                decl = e.get("artifact_sha256") or ""
                sha = decl.get(base, "") if isinstance(decl, dict) else decl
                # A fact measured after the contract may still cite an artifact written
                # BEFORE it -- a restatement rescores old files. Those have no ledger row
                # and never can. Declaring the leg in config.unattested_leg exempts it
                # and COUNTS it, so the evidence says how much of the citation is
                # unbacked; a date-inferred exemption would hide it (de, 2026-09-01).
                if base in str(e.get("config", {}).get("unattested_leg", "")):
                    unattestable.append(f"{e.get('id')}:{base}")
                    continue
                if not sha:
                    bad.append(f"{e.get('id')} cites {path} with no artifact_sha256")
                elif (base, sha) not in attested:
                    if any(s == sha for _b, s in attested):
                        # The bytes are attested, at some OTHER path. That is the T7-1
                        # defect, and it earns its own message: the citation and the
                        # attestation disagree about WHICH FILE holds these bytes.
                        other = sorted(b for b, s in attested if s == sha)
                        bad.append(f"{e.get('id')} cites {path} sha {sha[:12]}, but that hash "
                                   f"is attested for {other[:2]} -- wrong path")
                    else:
                        bad.append(f"{e.get('id')} cites {path} sha {sha[:12]} with no attestation")
    if not cited:
        return SKIP, (f"no fact measured since {contract_from} cites a data/eval artifact "
                      f"({legacy} predate the contract)")
    if bad:
        return FAIL, f"{len(bad)} of {cited} citation(s) unattested: {'; '.join(bad[:3])}"
    ua = (f"; {len(unattestable)} leg(s) declared unattestable: {', '.join(sorted(unattestable)[:3])}"
          if unattestable else "")
    return PASS, (f"{cited} artifact citation(s) since {contract_from}, every hash attested "
                  f"by its writer ({legacy} legacy citations exempt){ua}")


def _broken_cited_artifacts_attested():
    """The REAL facts, with one artifact-citing entry re-dated into the contract window
    and its hash removed -- the shape a new fact takes when someone forgets to attest."""
    d = _tmp_repo()
    os.makedirs(os.path.join(d, "facts"), exist_ok=True)
    import shutil as _sh

    hit = None
    for fp in sorted(glob.glob(os.path.join(ROOT, "facts", "*.json"))):
        obj = json.load(open(fp, encoding="utf-8"))
        for e in obj.get("facts", []):
            if re.search(r"data/eval/[\w./-]+\.jsonl", json.dumps(e, ensure_ascii=False)):
                hit = (fp, obj, e)
                break
        if hit:
            break
    if not hit:
        raise SelftestSkip("no fact cites a data/eval artifact yet")
    fp, obj, e = hit
    for other in glob.glob(os.path.join(ROOT, "facts", "*.json")):
        _sh.copy(other, os.path.join(d, "facts", os.path.basename(other)))
    # inside the contract window, so the legacy exemption does not hide it
    e["measured"] = "2099-01-01"
    e.pop("artifact_sha256", None)
    json.dump(obj, open(os.path.join(d, "facts", os.path.basename(fp)), "w"), ensure_ascii=False)
    return d


def check_milestone_ckpt_pinned(root):
    """Every milestone row's checkpoint still exists, or a pinned copy does.

    train.py keeps the newest 3 step checkpoints, so a milestone file lives about
    3 x save_every steps -- ~70 minutes at stage-1 speed. The 3.24B own-mix baseline
    was lost that way on 2026-08-31: the rescore sat in the lane queue, step3500
    rotated out, and the measurement is unrepeatable because the weights are gone.
    A milestone is a checkpoint we have promised to keep; the roller does not know
    that, so the promise has to be a file beside it."""
    ms = os.path.join(root, "runs", "milestones.jsonl")
    if not os.path.exists(ms):
        return SKIP, "no runs/milestones.jsonl"
    rows = []
    for line in open(ms, encoding="utf-8"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return SKIP, "milestones.jsonl has no rows"
    # Only the pod holds checkpoints; a dev box would report every row unpinned.
    if not glob.glob(os.path.join(root, "ckpt_*.pt")):
        return SKIP, "no checkpoints on this machine (pod holds them)"
    lost, gone, ok = [], [], 0
    for r in rows:
        ck = r.get("ckpt")
        if not ck:
            continue
        if os.path.exists(os.path.join(root, ck)):
            ok += 1
            continue
        tok = r.get("milestone")
        pins = glob.glob(os.path.join(root, f"*milestone_{tok}*.pt")) if tok else []
        if pins:
            ok += 1
        elif r.get("unrepeatable"):
            gone.append(f"{ck}: {r['unrepeatable']}")
        else:
            lost.append(f"{ck} (milestone {tok or '?'})")
    if lost:
        return FAIL, (f"{len(lost)} milestone row(s) whose checkpoint is gone with no pinned "
                      f"copy: {'; '.join(lost[:3])} -- that measurement cannot be repeated")
    # Named on every PASS, never counted as ok. Unlike a retired mix, nothing LOADS a
    # milestone row -- people read it -- so the marker's teeth are that it cannot become
    # invisible, not that something refuses it. Weights that are gone cannot be pinned
    # afterwards, so the alternatives were a permanent red or deleting the number.
    if gone:
        return PASS, (f"{ok} milestone checkpoint(s) present or pinned; "
                      f"{len(gone)} unrepeatable and marked: {'; '.join(gone)[:150]}")
    return PASS, f"{ok} milestone checkpoint(s) present or pinned"


def _broken_milestone_ckpt_pinned():
    """The REAL milestones ledger with a row naming a checkpoint that is not there."""
    d = _tmp_repo()
    src = os.path.join(ROOT, "runs", "milestones.jsonl")
    if not os.path.exists(src):
        raise SelftestSkip("no milestones ledger to mutate")
    rows = [json.loads(x) for x in open(src, encoding="utf-8") if x.strip()]
    if not rows:
        raise SelftestSkip("milestones ledger is empty")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    rows[0] = dict(rows[0], ckpt="ckpt_rotated_away.pt.step3500", milestone="3.24b")
    with open(os.path.join(d, "runs", "milestones.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # a checkpoint must exist or the check SKIPs instead of failing
    open(os.path.join(d, "ckpt_present.pt"), "w").write("x")
    return d


def check_cache_readers_set_vocab_id(root):
    """Every module that imports a token-cache reader also sets train.VOCAB_ID.

    val_seqs and domain_loss_seqs reach train._domain_seqs, whose freshness guard compares
    each cache stamp against train.VOCAB_ID. That global starts at None and only
    train.build_tokenizer sets it, which no eval calls. A module that reads a cache without
    setting it does not get a warning: every stamp reads as a mismatch against an empty
    right side, and the guard reports "cache dirty" when the process simply has no
    fingerprint. Before the guard existed, the same None retokenized nine live domains and
    re-stamped them with an empty vocabulary (fb 2026-09-02, caught on ppl.py two minutes
    in).

    MEASURED 2026-09-03: eval/domain_bpb.py imported val_seqs at :219 and never set it, so
    domain_bpb has never produced a value -- while score_matrix.py:1186 and
    domain_loss.py:624 each carry the call with a comment explaining why. Two of three
    callers remembered. This is the check for the one that did not.

    Accepts any of the three real routes, because they are all correct: set_vocab_id(cfg),
    an assignment to train.VOCAB_ID (scripts/test_domain_loss_val.py:103 does this from a
    fingerprint it computes), or build_tokenizer. Requiring set_vocab_id by name would
    fail a file that does the right thing another way -- and a check that fires on correct
    input is looser than none, because the next author silences it.

    IT ALSO CHECKS COVERAGE, NOT ONLY EXISTENCE. The first version of this check asked
    "does a setter appear anywhere in the module" and PASSED eval/domain_bpb.py while that
    file was still broken: the fix put set_vocab_id under `if not a.hf`, and the val_seqs
    call at :256 sat outside that branch, so the --hf control arm still walked into the
    guard. Three rows in runs/score_matrix.jsonl carry that error and one of them is the
    control. A setter guarded by a condition the reader is NOT guarded by covers only some
    of the calls.

    The coverage test is WITHIN ONE FUNCTION only. eval/score_matrix.py sets the
    fingerprint at :1185 inside main's `if` and reads at :232 inside metric_domain_loss,
    which main calls at :1189 -- correct, but a cross-function depth comparison flags it,
    and that false positive is what a first version of this coverage rule produced. Branch
    depth means nothing across function boundaries: the caller decides whether the callee
    runs. Restricting to one scope still catches the real defect, because domain_bpb's
    setter and reader were in the same function.
    """
    import ast

    READERS = ("val_seqs", "domain_loss_seqs", "_domain_seqs")
    SETTERS = ("set_vocab_id", "build_tokenizer")

    def _depths_in(fn, targets, kinds):
        """Every `if`/`try` nesting depth at which a target appears inside ONE function.

        Nested function bodies are not descended into: they are separate scopes whose
        execution their own caller decides.
        """
        found = []

        def walk(node, depth):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                hit = False
                if isinstance(child, ast.Call) and "call" in kinds:
                    name = (child.func.attr if isinstance(child.func, ast.Attribute)
                            else getattr(child.func, "id", ""))
                    hit = name in targets
                elif isinstance(child, ast.Assign) and "assign" in kinds:
                    hit = any(isinstance(t, ast.Attribute) and t.attr == "VOCAB_ID"
                              for t in child.targets)
                if hit:
                    found.append(depth)
                walk(child, depth + 1 if isinstance(child, (ast.If, ast.Try, ast.While)) else depth)

        walk(fn, 0)
        return found

    scan = []
    for sub in ("eval", "scripts", "probes"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        scan += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".py")]
    if not scan:
        return SKIP, "no eval/scripts/probes directory"

    bad, checked = [], 0
    for path in scan:
        try:
            src = open(path, encoding="utf-8").read()
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        # IMPORTS a reader, not merely mentions one: a comment or a string naming val_seqs
        # is not a call into the cache. eval_load_cost.py holds the names in a tuple and
        # test_cache_dir_knob.py in an error message; neither reads a cache.
        imports = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                imports |= {a.name for a in n.names}
            elif isinstance(n, ast.Import):
                imports |= {a.name.split(".")[-1] for a in n.names}
        if not (imports & set(READERS)):
            continue
        checked += 1
        # The file DEFINING a reader is where the guard lives, not a caller of it.
        defines = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        if imports & set(READERS) & defines:
            continue
        called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                  for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assigns_global = any(
            isinstance(t, ast.Attribute) and t.attr == "VOCAB_ID"
            for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
        )
        rel = os.path.relpath(path, root)
        names = ", ".join(sorted(imports & set(READERS)))
        if not (called & set(SETTERS)) and not assigns_global:
            bad.append(f"{rel} imports {names} but never sets train.VOCAB_ID")
            continue
        # COVERAGE, per function: a setter deeper in the branch nesting than a reader in the
        # SAME function protects only some of that function's paths.
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            rd = _depths_in(fn, set(READERS), {"call"})
            sd = _depths_in(fn, set(SETTERS), {"call", "assign"})
            if rd and sd and min(sd) > min(rd):
                bad.append(f"{rel}:{fn.name} sets train.VOCAB_ID at branch depth {min(sd)} "
                           f"but reads the cache at depth {min(rd)} in the same function, so "
                           f"the reader runs on paths the setter does not cover")
    if bad:
        return FAIL, ("; ".join(bad) + " -- every cache stamp will read as a mismatch against "
                      "an empty fingerprint, and the guard will say 'cache dirty' when the "
                      "process simply has none. Call cache_guard.set_vocab_id(cfg) before the "
                      "first read (see eval/domain_loss.py:624)")
    return PASS, f"{checked} cache-reading module(s) set train.VOCAB_ID"


def _broken_cache_readers_set_vocab_id():
    """The REAL eval/ with domain_bpb.py's fingerprint line put back under `if not a.hf` --
    the state this file was in between the two fixes on 2026-09-03, not a synthetic one.

    That intermediate state is the interesting world, not the original: the original had no
    setter at all, which the existence half already catches. This one HAS a setter and is
    still broken, because val_seqs at :256 sits outside the branch, so the --hf control arm
    walks into the guard anyway. It is what the coverage half exists for.

    Mutating the live file rather than writing a fixture, for the reason de-7.3 records: a
    fixture encodes the author's assumption twice.
    """
    d = _tmp_repo_shaped()
    src = os.path.join(ROOT, "eval", "domain_bpb.py")
    if not os.path.exists(src):
        return None
    text = open(src, encoding="utf-8").read()
    line = "    train.VOCAB_ID = vocab_fingerprint(ours_tok)\n"
    if line not in text:
        return None  # the fix moved or was renamed: this world cannot be built
    import shutil as _sh
    link = os.path.join(d, "eval")
    if os.path.islink(link):
        os.unlink(link)
    _sh.copytree(os.path.join(ROOT, "eval"), link, ignore=_sh.ignore_patterns("__pycache__"))
    open(os.path.join(d, "eval", "domain_bpb.py"), "w", encoding="utf-8").write(
        text.replace(line, "    if not a.hf:\n        train.VOCAB_ID = vocab_fingerprint(ours_tok)\n"))
    return d


def check_mutation_asserted_took(root):
    """Every world that mutates a file in place proves the mutation took, before blaming the subject.

    6e's ruling, 2026-09-04, from the world-8 race that blocked b0 mid-merge for half an hour. World
    8 replaced `return 0.0` with `return 1.0` in algorithms/rlvr_reward.py -- byte-for-byte the same
    length, 3565 -> 3565 -- ran the mapped test green BEFORE mutating (correct: a refusal proves
    nothing on an already-red world), and that green run left a __pycache__. Python invalidates a
    pyc on (source mtime in WHOLE SECONDS, size), so a mutation landing in the same wall-clock second
    reused the stale bytecode, the defect never executed, the inner hook exited 0, and the world
    reported "a staged defect in a mapped SUBJECT was allowed" about a hook that works. Six replica
    runs: rc 1,1,1,0,0,1.

    THE DEEPER FAILURE IS THAT ONE MESSAGE COVERED TWO CAUSES. "The hook did not run the mapped
    test" and "the mutation never took effect" printed the same sentence, and it was the second one
    all along -- which is why the first hypothesis (a leaked GIT_INDEX_FILE) survived a day. The
    same shape killed _broken_reported_path silently: its marker is still in eval/l1_fewshot.py
    today, but e1's 29b31367 deleted the `preds_path` VARIABLE, so the reverted print raised
    NameError instead of reproducing the stale-name defect, and the world PASSED with its defect
    unreproduced.

    So: a world that mutates bytes and then runs something must assert the mutation took. Two ways
    satisfy it, and either is enough --
      1. compare the mutated text against the source (`!=`, `assert <new> in`, a marker guard that
         FAILs when the target is absent), or
      2. delete the __pycache__ it could have inherited, which removes the mechanism entirely.

    WHAT THIS CANNOT SEE, stated because a check that reads as complete is worse than one that
    admits its edge: it matches the SHAPE of an assertion, not its correctness. A world asserting
    the wrong string passes here. That is the residual, and it is why the scan
    runs/audit_0904/dead_worlds.py exists beside this check -- it reads whether the marker literal
    is still in the file, which is the other half.
    """
    worlds = []
    for path in (os.path.join(root, "scripts", "harness.py"),
                 os.path.join(root, "scripts", "hooks", "pre-commit")):
        if not os.path.exists(path):
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError as e:
            return FAIL, f"{os.path.relpath(path, root)} does not parse: {e}"
        rel = os.path.relpath(path, root)
        for n in tree.body:
            if not isinstance(n, ast.FunctionDef):
                continue
            if not (n.name.startswith(("_broken_", "_positive_")) or n.name == "_selftest"):
                continue
            worlds.append((rel if n.name == "_selftest" else "", n.name, n))

    bad = []
    for rel, name, fn in worlds:
        writes = replaces = runs = proves = purges = False
        for x in ast.walk(fn):
            if isinstance(x, ast.Call):
                f = getattr(x.func, "attr", None) or getattr(x.func, "id", None)
                if f == "replace":
                    replaces = True
                if f in ("write", "writelines"):
                    writes = True
                if f in ("run", "check_output", "check_call", "call", "Popen", "system",
                         "import_module"):
                    runs = True
                if f == "rmtree":
                    purges = True
            # The PROOF: `assert <marker> in src`, or any comparison of the mutated text against
            # what it came from. Either shape says the world checked rather than assumed.
            if isinstance(x, ast.Assert):
                proves = True
            elif isinstance(x, ast.Compare) and any(
                    isinstance(o, (ast.NotEq, ast.Eq, ast.In, ast.NotIn)) for o in x.ops):
                proves = True
        if not (replaces and writes and runs):
            continue
        if proves or purges:
            continue
        bad.append(f"{rel + ':' if rel else ''}{name}")

    if bad:
        return FAIL, (
            f"{len(bad)} world(s) mutate a file and then run it without proving the mutation took: "
            f"{', '.join(sorted(bad))} -- a size-preserving edit inside the pyc's one-second mtime "
            f"key silently does nothing, and the world then blames its subject (world 8 blocked "
            f"b0's merge this way, e5b73d40)")
    return PASS, f"{len(worlds)} worlds; every in-place mutator asserts its edit or purges pycache"


def _broken_mutation_asserted_took():
    """The REAL _broken_gemm_dims with its proof removed, and made to RUN what it mutated.

    NOT the hook's world 8, though that is the incident. Two attempts at surgery on the hook both
    produced a FAIL for the wrong reason -- first a dangling `else:`, then an unclosed brace -- and
    a broken world that reports FAIL on a SyntaxError proves nothing about the property. The lesson
    is the check's own: a world must prove its mutation took, and text surgery across a 1900-line
    file cannot. _broken_gemm_dims is nine lines, mutates train.py in place with a size-preserving
    edit (`ffn_hidden = 3072` -> `3400`, 17 bytes both), and carries exactly the assert this check
    looks for -- so removing that one line, and adding the subprocess call that makes the pyc
    reachable, is the whole world.
    """
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "train.py"), os.path.join(d, "train.py"))
    src = open(os.path.join(ROOT, "scripts", "harness.py"), encoding="utf-8").read()
    real = '''def _broken_gemm_dims():
    # The REAL train.py with ffn_hidden 3072 -> 3400: 8-aligned (passes the cuBLAS
    # tier) but not 16-aligned, so _fp8_ok silently drops FP8. Mutated, not hand-written.
    import shutil

    d = _tmp_repo()
    p = os.path.join(d, "train.py")
    shutil.copy(os.path.join(ROOT, "train.py"), p)
    src = open(p, encoding="utf-8").read()
    src = src.replace("ffn_hidden = 3072", "ffn_hidden = 3400", 1)
    assert "ffn_hidden = 3400" in src, "real train.py no longer has \'ffn_hidden = 3072\'; update _broken_gemm_dims"
    open(p, "w", encoding="utf-8").write(src)
    return d'''
    assert real in src, ("_broken_gemm_dims is not the shape this world mutates; the target moved "
                         "and the world would pass without testing anything")
    stripped = "\n".join(ln for ln in real.splitlines() if not ln.strip().startswith("assert "))
    # The subprocess call is what makes a stale pyc reachable, and the check only judges worlds
    # that RUN what they mutated -- without it the stripped world is correctly out of scope.
    stripped = stripped.replace('    open(p, "w", encoding="utf-8").write(src)',
                                '    open(p, "w", encoding="utf-8").write(src)\n'
                                '    subprocess.run([sys.executable, p], capture_output=True)')
    open(os.path.join(d, "scripts", "harness.py"), "w", encoding="utf-8").write(
        src.replace(real, stripped, 1))
    return d



def check_selftests_are_gated(root):
    """Every file carrying its own --selftest is in the hook's SELFTEST_FILES map.

    2026-09-01: a readout commit landed with its selftest RED under five green hook
    lines, because the hook ran tree/blob/ruff/harness and none of them knew the
    edited file carried fifteen cases testing the guard the commit was changing. The
    hook checked what it happened to check rather than what the commit changed. A
    selftest not in the map is a selftest nobody runs, which is worse than none --
    it reads as coverage."""
    hook = os.path.join(root, "scripts", "hooks", "pre-commit")
    if not os.path.exists(hook):
        return SKIP, "no scripts/hooks/pre-commit"
    src = open(hook, encoding="utf-8").read()
    m = re.search(r"SELFTEST_FILES\s*=\s*\{([^}]*)\}", src)
    if not m:
        return FAIL, ("scripts/hooks/pre-commit has no SELFTEST_FILES map, so no staged "
                      "file's selftest runs at commit time")
    gated = set(re.findall(r'"([^"]+)"', m.group(1)))
    # THE PARSE MUST NOT BE SHIFTED BY A COMMENT. Quotes are paired positionally, so a
    # comment inside either map that contains an odd number of double quotes re-pairs every
    # quote below it and silently drops the entries that follow. Measured 2026-09-03: a
    # comment quoting the regex on this very line cost 20 entries, and the check then
    # reported 16 ungated selftests that were all present in the map -- a false FAIL that
    # reads exactly like a real one, and sent me through nine probes looking for the wrong
    # cause. The comment explaining the first version of this bug caused the second.
    #
    # Cross-check the positional parse against a line-shaped one: a real entry is a quoted
    # path alone on its line, ending in a comma. If the two disagree, the map is being
    # misread and the counts below are meaningless -- so it refuses instead of reporting.
    for label, block in (("SELFTEST_FILES", m.group(1)),
                         ("NEEDS_DATA", nd.group(1) if (
                             nd := re.search(r"NEEDS_DATA\s*=\s*\{(.*?)\n    \}", src, re.S)
                         ) else "")):
        line_shaped = set(re.findall(r'^\s+"([^"]+)"\s*[,:]', block, re.M))
        positional = set(re.findall(r'"([^"]+)"', block))
        missed = sorted(x for x in line_shaped if x not in positional)
        if missed:
            return FAIL, (
                f"{len(missed)} entry(ies) in {label} are invisible to this check's parse, "
                f"so nothing runs their selftests while the map appears to list them: "
                f"{', '.join(missed[:4])}. Cause is almost always a comment inside the map "
                f"containing an odd number of double-quote characters, which re-pairs every "
                f"quote below it. Remove the quotes from the comment."
            )
    # A file the hook cannot run here is still accounted for, with the reason recorded.
    # "not in the map" and "cannot run at commit time" are different facts, and only
    # the second is acceptable -- silence about the first is how a selftest goes unrun.
    gated |= set(re.findall(r'"([^"]+)":', nd.group(1))) if nd else set()
    # The map's other direction, which nothing watched: an entry naming a file that no
    # longer exists. Found by measurement, not by reasoning -- main deleted
    # mathbank/arith_curriculum.py and mathbank/procedure_curriculum.py and both stayed
    # in SELFTEST_FILES, invisible because this check only ever asked whether every
    # selftest is IN the map, never whether every entry still names a file (de-12,
    # 2026-09-02). A stale entry is not dangerous the way an ungated selftest is, but it
    # is a claim about coverage of something that is gone, and the count it inflates is
    # the number this check reports. Only path-shaped entries: the maps carry prose keys
    # in comments that the value regex also matches.
    stale = sorted(g for g in gated
                   if "/" in g and g.endswith((".py", ".sh"))
                   and not os.path.exists(os.path.join(root, g)))
    if stale:
        return FAIL, (f"{len(stale)} hook map entry(ies) name a file that does not exist, "
                      f"so the map claims coverage of something deleted: {', '.join(stale[:4])}")
    have = set()
    # walk_tracked, not an os.listdir over four hand-named directories. The list was
    # ("eval", "scripts", "datagen", "probes"), so mathbank/ was outside what the check
    # looked at entirely and "42 files, all gated" was true of that subset and silent
    # about the rest -- the same shape as the predicate bug recorded below, one level up:
    # a gate that cannot see a file cannot report it missing. mathbank/dist_check.py
    # carries a selftest and was invisible here (de, 2026-09-02, MEASURED at 42 vs 43).
    for p, body in walk_tracked(root, (".py",)):
        rel = os.path.relpath(p, root)
        # `--selftest` anywhere in the CODE, not `"--selftest"` next to `add_argument`.
        # The narrow predicate assumed every selftest is wired through argparse; nine
        # files dispatch on sys.argv instead (scripts/eval_artifacts.py:
        # `sys.exit(_selftest() if "--selftest" in sys.argv else 0)`), and the
        # gate reported "27 files, all gated" while those nine ran nowhere. A gate
        # that cannot see a file cannot report it missing, so its PASS counted only
        # the files it already understood -- the check encoding an assumption about
        # where the interesting case lives, which is this repo's named class, in
        # the check written to catch that class (de, 2026-09-01, on 62's gate).
        #
        # Docstrings blanked, because the widening reached one file too far: prose
        # SAYING a file carries no `--selftest` matched as if it carried one. MEASURED
        # 2026-09-02 at 44 raw vs 42 stripped -- scripts/test_resume_accumulates.py,
        # whose docstring explains why it deliberately has no selftest flag, and
        # scripts/test_serve_history.py, whose usage line quotes the flag its body never
        # reads. Only the first was ever reported, because the second happens to be in
        # the map: a false positive hides wherever the answer is right by accident.
        if "--selftest" in strip_docstrings(body):
            have.add(rel)
        # A RUNNABLE test_*.py IS A SELFTEST WHETHER OR NOT IT CARRIES THE FLAG (de,
        # 2026-09-04, measured: 63 test_*.py tracked, 53 runnable by `if __name__`, 19 of
        # those in neither map). The population above is "files containing the string --selftest", so a
        # test_*.py with a main() and no flag was outside this check by construction --
        # the third time this check's population has been narrower than its property, after
        # the four-directory listdir blind to mathbank/ and the argparse-only predicate
        # blind to nine sys.argv dispatchers. Both earlier widenings are recorded above as
        # the same lesson, and both times the PASS counted only the files the check already
        # understood.
        #
        # The case that made it concrete: scripts/test_score_exit.py, 13 cases over
        # run_ddp.sh's exit codes and row close, never run at any commit -- and the commit
        # that CHANGED run_ddp.sh printed `selftests 0.03s`, which reads as "ran, fast" and
        # means "ran zero". scripts/test_resume_accumulates.py is the demonstration that the
        # old population was not merely incomplete but blind on purpose: its docstring
        # explains that it deliberately carries no flag, so stripping docstrings correctly
        # excluded it, and it is a runnable test nobody ran.
        #
        # `if __name__` and not `def main(`: algorithms/test_rlvr_reward_suite.py asserts at
        # module scope with no main() at all, so a main()-shaped predicate would have missed
        # it -- the same defect one more level down. (I first wrote that three files were in
        # that shape; measured, the other two are already gated. One is enough to decide the
        # predicate.) The invoker passes --selftest to every entry, so a file here must
        # tolerate an unknown argument or get a SELFTEST_FLAG override; that is a property of
        # the file, checked by running it, not something this check can assert.
        elif (re.search(r"(^|/)test_[\w-]+\.py$", rel)
                and re.search(r"^if __name__", strip_docstrings(body), re.M)):
            have.add(rel)
    missing = sorted(have - gated)
    if missing:
        return FAIL, (f"{len(missing)} file(s) carry a selftest but are not in the hook's "
                      f"SELFTEST_FILES or NEEDS_DATA, so nothing runs them at commit time: "
                      f"{', '.join(missing[:4])}")
    # A NEEDS_DATA entry is a CLAIM about why a selftest cannot run at commit time, and
    # nothing recomputed it. scripts/harness.py's read "the hook already runs `harness
    # check`, which is its selftest" -- false: `check` runs run_checks(), `--selftest`
    # verifies every check FAILs on its broken world, and they share no code path. The
    # harness's core proof had never run at commit time and the exemption said it had
    # (Codex found it, de confirmed by reading both entry points, 2026-09-01).
    #
    # A reason cannot be verified in general, but the specific false form can: an
    # exemption that claims some OTHER command already covers it is the one shape that
    # asserts coverage rather than impossibility. Cost, missing data, and needing root
    # are claims about this machine; "X already runs it" is a claim about X.
    if nd:
        covered = [k for k, v in re.findall(r'"([^"]+)":\s*"([^"]*)"', nd.group(1))
                   if re.search(r"\bis its selftest\b|\balready runs\b|\bcovered by\b", v)]
        if covered:
            return FAIL, (f"{len(covered)} NEEDS_DATA reason(s) claim another command "
                          f"already runs the selftest, which is a coverage claim nothing "
                          f"recomputes: {', '.join(covered)} -- state why it cannot run "
                          f"here (cost, data, root), not what supposedly covers it")
    return PASS, f"{len(have)} selftest-carrying file(s), all gated by the hook"


def check_probe_numbers_unique(root):
    """Surface tNN numbers claimed by more than one probe. WARN, not FAIL.

    Three collisions in one afternoon (2026-09-01): t62, t63 and t64 each named
    two unrelated probes from two sessions, because a probe number is allocated
    by guessing the next free integer against a tree other sessions are writing
    to concurrently. Nothing fails loudly -- both files exist, both run -- but a
    fact citing "t63" then resolves to whichever the reader finds first.

    WHY THIS IS A WARN AND NOT A FAIL, which my first version got wrong: sharing
    a number is LEGITIMATE when the files are one task's sub-probes. t57 is
    t57_absmax@c3d02c4 / t57_outlier / t57_seam@1c470e7, three angles on one fp8-head question,
    cited by full filename and unambiguous. My FAIL version reddened the board on
    that convention -- a check that fires on correct practice trains people to
    ignore it, which is worse than no check.

    Nothing in the filenames distinguishes "one task, three probes" from "two
    sessions, one number", and git authorship does not either on a shared
    machine. So this reports the shared numbers and leaves the judgement to a
    human, which is the honest limit of what it can know.

    IF YOU ARE HERE TO ADD A SKIP-LIST AND PROMOTE THIS TO FAIL, read this
    first. That was my first instinct too: keep the FAIL, exempt t57. But every
    skip-list entry is a claim that reality is wrong, and t57 is not wrong --
    it is the normal case. **The exception would have been evidence I was
    measuring the wrong property.** One exemption is a smell; the second is the
    moment to re-read the check instead of the world. A check that fires on
    correct practice trains people to ignore it, which is worse than no check.
    """
    import collections

    d = os.path.join(root, "probes")
    if not os.path.isdir(d):
        return SKIP, "no probes/ directory"
    seen = collections.defaultdict(list)
    for f in sorted(os.listdir(d)):
        m = re.match(r"^(t\d+)_.*\.py$", f)
        if m:
            seen[m.group(1)].append(f)
    if not seen:
        return SKIP, "probes/ holds no tNN_*.py files"
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(dupes.items()))
        return WARN, (f"{len(dupes)} probe number(s) claimed by >1 file -- fine if they are one "
                      f"task's sub-probes, a collision if two sessions picked the same integer: {detail}")
    return PASS, f"{len(seen)} probe numbers, all unique"


def _broken_probe_numbers_unique():
    """A REAL probe plus a second file claiming its number -- the exact shape of
    today's t62/t63/t64 collisions. Built from a real repo path rather than two
    invented names, because the meta-check requires the broken world to mirror
    the tree it stands in, and it is right to: a world of invented paths tests
    the check against a repo that does not exist.

    WARN, not FAIL: the check cannot tell a collision from one task's sub-probes
    (t57_absmax@c3d02c4 / t57_outlier / t57_seam@1c470e7 are legitimately one number)."""
    import shutil as _sh

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "probes"), exist_ok=True)
    real = None
    for f in sorted(os.listdir(os.path.join(ROOT, "probes"))):
        if re.match(r"^t\d+_.*\.py$", f):
            real = f
            break
    if real is None:
        raise SelftestSkip("no tNN_*.py probe in the repo to build a collision from")
    _sh.copy(os.path.join(ROOT, "probes", real), os.path.join(d, "probes", real))
    num = re.match(r"^(t\d+)_", real).group(1)
    with open(os.path.join(d, "probes", f"{num}_collision.py"), "w", encoding="utf-8") as f:
        f.write("# a second probe claiming the same number\n")
    return d


def check_no_duplicate_defs(root):
    """No module defines the same top-level name twice.

    A merge can land two copies of one function without any conflict: tonight two
    sessions restored the same dropped selftest from different commits, and
    harness.py carried _selftest_gpu_descendants twice, 200 lines apart. Python
    silently binds the second, so the FIRST copy is dead and the two drift the day
    one is edited -- and the selftest ran twice, which looks like coverage.

    ruff's F811 does not fire on this: the copies were separated by other defs and
    the name is called from a list of selftests, not shadowed in an obvious way.
    """
    bad = []
    scanned = 0
    for rel in ("scripts", "eval", "datagen", "filters", "algorithms"):
        for path in sorted(glob.glob(os.path.join(root, rel, "**", "*.py"), recursive=True)):
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
            scanned += 1
            seen = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name in seen:
                        bad.append(f"{os.path.relpath(path, root)}:{node.lineno} "
                                   f"{node.name} (first at :{seen[node.name]})")
                    seen[node.name] = node.lineno
    for path in (os.path.join(root, f) for f in ("train.py", "sft.py", "sft_math.py")):
        if not os.path.exists(path):
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (OSError, SyntaxError):
            continue
        scanned += 1
        seen = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in seen:
                    bad.append(f"{os.path.basename(path)}:{node.lineno} {node.name} "
                               f"(first at :{seen[node.name]})")
                seen[node.name] = node.lineno
    if bad:
        return FAIL, f"{len(bad)} duplicate top-level def(s): {'; '.join(bad[:3])}"
    return PASS, f"{scanned} module(s), no duplicate top-level definitions"


def _broken_selftests_are_gated():
    """The REAL hook with one REAL selftest-carrying file dropped from the map.

    Mutating the live artifact rather than writing a fixture: the reweight gate passed
    every synthetic case and returned None for every real role, because the fixture
    encoded the author's assumption twice (7.3).

    The dropped file is scripts/eval_artifacts.py SPECIFICALLY, not an argparse-wired
    one. It dispatches on sys.argv --
    `sys.exit(_selftest() if "--selftest" in sys.argv else 0)` -- so under the old
    narrow predicate ('"--selftest"' near add_argument) the check could not see it at
    all and this world would have gone GREEN with the file unguarded. That is exactly
    the defect the widening fixes, and using an argparse file here would leave the
    widening untested (de, 2026-09-01).

    Built on _tmp_repo_shaped, not _tmp_repo, since de-12 added the stale-entry
    assertion: in a bare tree none of the map's 49 paths resolve, so the check FAILed
    there naming datagen/build_corpus.py and never reached the ungated selftest. A world
    failing for the wrong reason proves nothing about the mutation it was built for
    (de, 2026-09-02). scripts/ is a symlink into the real repo in a shaped world, so the
    mutated hook needs its own copied directory or the write lands in the repo itself.
    """
    d = _tmp_repo_shaped()
    hook = os.path.join(ROOT, "scripts", "hooks", "pre-commit")
    ev = os.path.join(ROOT, "scripts", "eval_artifacts.py")
    if not (os.path.exists(hook) and os.path.exists(ev)):
        return None
    text = open(hook, encoding="utf-8").read()
    if '"scripts/eval_artifacts.py"' not in text:
        return None
    # Replace the scripts/ symlink with a real copy: everything the map names must still
    # resolve, and only the hook may differ.
    import shutil
    link = os.path.join(d, "scripts")
    if os.path.islink(link):
        os.unlink(link)
    shutil.copytree(os.path.join(ROOT, "scripts"), link,
                    ignore=shutil.ignore_patterns("__pycache__"))
    open(os.path.join(d, "scripts", "hooks", "pre-commit"), "w", encoding="utf-8").write(
        text.replace('"scripts/eval_artifacts.py", ', "")
            .replace('"scripts/eval_artifacts.py"', '"scripts/harness.py"'))
    return d


def _broken_no_duplicate_defs():
    """The REAL harness.py with one of its own functions defined a second time --
    the shape a merge produces, appended rather than hand-written."""
    d = _tmp_repo()
    src = os.path.join(ROOT, "scripts", "harness.py")
    if not os.path.exists(src):
        return None
    text = open(src, encoding="utf-8").read()
    marker = "def cfg_default(field):"
    if marker not in text:
        return None
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w", encoding="utf-8").write(
        text + "\n\ndef cfg_default(field):\n    return None  # the duplicate a merge lands\n"
    )
    return d


def _agents_coverage_table(root):
    """The Rule coverage table's (rule, enforcer) pairs, or ({}, err).

    Scoped to that ONE section: AGENTS.md holds a dozen other two-column tables
    (layout, entry points, the checks table), and reading them all reports every
    markdown row as a rule -- 104 rows and 74 false drifts when I first tried it.
    """
    p = os.path.join(root, "AGENTS.md")
    if not os.path.exists(p):
        return {}, "AGENTS.md missing"
    rows, inside = {}, False
    for line in open(p, encoding="utf-8").read().split("\n"):
        if re.match(r"^## ", line):
            inside = line.startswith("## Rule coverage")
            continue
        if not inside:
            continue
        m = re.match(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|$", line)
        if not m:
            continue
        rule, val = m.group(1), m.group(2)
        if rule == "Rule" or set(rule) <= set("-: "):
            continue
        rows[rule] = val
    return rows, None


def check_shapes_table_covers_doc(root):
    """Every surviving incident § in gate_failure_incidents.md (model-project) and
    infra_incidents.md (pod/infra) appears exactly once in AGENTS.md's rule table, and
    every row's count equals the §refs it lists.

    2026-09-04 restructure: gate_failure_shapes.md became the rules doc (no per-incident
    headings); incidents split into two layer files with '### §N' headings. 33 closed
    incidents were deleted, so the surviving set is intentionally non-contiguous. The
    gap check (1..max contiguous) was dropped. The duplicate check (within and across
    files) and the table-vs-doc comparison stand.

    Same ceiling as before: this proves an incident is REFERENCED, not that it sits under
    the right rule. Which rule an incident belongs to is a judgement only a person
    re-reading the pair can make."""
    doc = set()
    for fname in ("gate_failure_incidents.md", "infra_incidents.md"):
        p = os.path.join(root, "docs", "lessons", fname)
        if not os.path.exists(p):
            return FAIL, f"docs/lessons/{fname} missing"
        nums = [int(m) for m in re.findall(r"^### §(\d+)", open(p, encoding="utf-8").read(), re.M)]
        if not nums:
            return FAIL, f"no '### §N' incident headings in {fname} -- the doc's heading style changed"
        dupes = sorted({n for n in nums if nums.count(n) > 1}, key=int)
        if dupes:
            return FAIL, f"incident heading number(s) used more than once in {fname}: " + ", ".join(
                f"§{d}" for d in dupes) + " -- two sessions wrote the same number; renumber the later one"
        overlap = doc & set(nums)
        if overlap:
            return FAIL, f"incident §s in both layer files: {sorted(overlap)} -- an incident belongs to one layer"
        doc |= set(nums)

    a = os.path.join(root, "AGENTS.md")
    if not os.path.exists(a):
        return FAIL, "AGENTS.md missing"
    # The compressed-rules table only: three columns, the third a run of §refs. Other
    # AGENTS.md tables have two columns and cannot match.
    rows = re.findall(r"^\|\s*(.+?)\s*\|\s*(\d+)\s*\|\s*((?:§\d+\s*)+)\|",
                      open(a, encoding="utf-8").read(), re.M)
    if not rows:
        return FAIL, "no rule rows with §refs found in AGENTS.md -- the table was restructured"

    seen, miscount = {}, []
    for rule, n, refstr in rows:
        refs = [int(x) for x in re.findall(r"§(\d+)", refstr)]
        if len(refs) != int(n):
            miscount.append(f"{rule[:32]}: says {n}, lists {len(refs)}")
        for r in refs:
            seen[r] = seen.get(r, 0) + 1

    missing = sorted(doc - set(seen))
    dangling = sorted(set(seen) - doc)
    twice = sorted(k for k, v in seen.items() if v > 1)
    problems = []
    if missing:
        problems.append(f"in the doc, in no rule: {['§%d' % m for m in missing]}")
    if dangling:
        problems.append(f"in the table, not in the doc: {['§%d' % d for d in dangling]}")
    if twice:
        problems.append(f"listed under more than one rule: {['§%d' % t for t in twice]}")
    if miscount:
        problems.append(f"count disagrees with refs: {miscount[:3]}")
    if problems:
        return FAIL, "; ".join(problems)
    return PASS, (f"{len(doc)} incidents (max §{max(doc)}) each referenced exactly once across "
                  f"{len(rows)} rules; every row's count matches")


def _broken_shapes_table_covers_doc():
    """The REAL AGENTS.md with one §ref deleted from a rule row, doc untouched.

    fb's specified world (2026-09-02): a shape that reaches no rule. Note the row's own
    count is left at its old value, so this breaks BOTH halves at once -- the coverage
    half (that § is now in no rule) and the arithmetic half (the count now exceeds the
    refs listed). Deleting the ref without touching the count is also what a hand edit
    actually does.

    The other direction -- a shape appended to the doc while the table stands still, which
    is what happened twice on 2026-09-02 -- is _broken_shapes_table_doc_grew below. CHECKS
    holds one broken() per row, so the selftest runs that one explicitly."""
    d = _tmp_repo_shaped()
    src = os.path.join(ROOT, "AGENTS.md")
    if not os.path.exists(src) or not os.path.exists(
            os.path.join(ROOT, "docs", "lessons", "gate_failure_incidents.md")):
        return None
    text = open(src, encoding="utf-8").read()
    # Drop the LAST §ref of the first rule row that has more than one, so the row keeps a
    # valid shape and only its coverage changes.
    m = None
    for cand in re.finditer(r"^\|\s*.+?\s*\|\s*\d+\s*\|\s*((?:§\d+\s*){2,})\|", text, re.M):
        m = cand
        break
    if m is None:
        raise SelftestSkip("no rule row with 2+ §refs; update _broken_shapes_table_covers_doc")
    refs = m.group(1)
    dropped = re.findall(r"§\d+", refs)[-1]
    text = text[:m.start(1)] + re.sub(r"\s*" + dropped + r"\s*$", " ", refs) + text[m.end(1):]
    open(os.path.join(d, "AGENTS.md"), "w", encoding="utf-8").write(text)
    return d


def _broken_shapes_table_doc_grew():
    """The REAL docs with a new incident appended and the table left alone -- exactly what
    happened twice on 2026-09-02, and what a merge conflict does not catch."""
    import shutil as _sh

    d = _tmp_repo_shaped()
    src = os.path.join(ROOT, "docs", "lessons", "gate_failure_incidents.md")
    if not os.path.exists(src) or not os.path.exists(os.path.join(ROOT, "AGENTS.md")):
        return None
    text = open(src, encoding="utf-8").read()
    nums = [int(m) for m in re.findall(r"^### §(\d+)", text, re.M)]
    if not nums:
        raise SelftestSkip("no incident headings to extend; update _broken_shapes_table_doc_grew")
    # Use a number above every § in BOTH layer files so the overlap check does not fire.
    infra_p = os.path.join(ROOT, "docs", "lessons", "infra_incidents.md")
    if os.path.exists(infra_p):
        nums += [int(m) for m in re.findall(r"^### §(\d+)", open(infra_p, encoding="utf-8").read(), re.M)]
    new_n = max(nums) + 1
    # _tmp_repo_shaped SYMLINKS docs/, so writing through that path would append this
    # fixture to the REAL incidents doc. Replace the link with a real directory holding a
    # real copy of the one file this world mutates.
    link = os.path.join(d, "docs")
    if os.path.islink(link):
        os.unlink(link)
    dst = os.path.join(d, "docs", "lessons", "gate_failure_incidents.md")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8").write(
        text + f"\n\n### §{new_n} (2026-09-04, R1) an incident added without touching the table (fixture)\nopen: none.\n")
    # The check reads both layer files; copy the unmutated one so the FAIL is about the
    # new incident, not a missing file.
    infra_src = os.path.join(ROOT, "docs", "lessons", "infra_incidents.md")
    if os.path.exists(infra_src):
        _sh.copy2(infra_src, os.path.join(d, "docs", "lessons", "infra_incidents.md"))
    _sh.copy2(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    return d


def _broken_shapes_table_duplicate_heading():
    """The REAL docs with one heading number duplicated -- what happened four times
    (§62, §63, §69, §70) when two sessions appended shapes in parallel. A merge
    conflict catches a collision on the same LINE, not two sessions writing the same
    NUMBER."""
    import shutil as _sh

    d = _tmp_repo_shaped()
    src = os.path.join(ROOT, "docs", "lessons", "gate_failure_incidents.md")
    if not os.path.exists(src) or not os.path.exists(os.path.join(ROOT, "AGENTS.md")):
        return None
    text = open(src, encoding="utf-8").read()
    nums = re.findall(r"^### §(\d+)", text, re.M)
    if not nums:
        raise SelftestSkip("no incident headings to duplicate; update _broken_shapes_table_duplicate_heading")
    # Same symlink hazard as _broken_shapes_table_doc_grew: docs/ is a link into the repo.
    link = os.path.join(d, "docs")
    if os.path.islink(link):
        os.unlink(link)
    dst = os.path.join(d, "docs", "lessons", "gate_failure_incidents.md")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8").write(
        text + f"\n\n### §{nums[-1]} (2026-09-04, R1) a duplicate heading number (fixture)\nopen: none.\n")
    infra_src = os.path.join(ROOT, "docs", "lessons", "infra_incidents.md")
    if os.path.exists(infra_src):
        _sh.copy2(infra_src, os.path.join(d, "docs", "lessons", "infra_incidents.md"))
    _sh.copy2(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    return d


def check_agents_rules_covered(root):
    """Every AGENTS.md rule maps to a check name or an explicit manual reason, and the
    table in the doc says the same thing as the map in the code.

    A rule that is only prose is one people break for cause: tonight the register
    refusal in a worktree pushed a session into the shared tree, and 'run it in the
    main checkout' was a documented instruction pointing at the one place sessions
    overwrite each other. Coverage cannot prove a mapping is honest -- it proves one
    was made, and the manual count is ratcheted so 'manual' cannot quietly win.

    Two things it now also proves, both from real misses:

    The named check must be the one that ENFORCES the rule, not merely a check that
    exists. The CUDA_VISIBLE_DEVICES rule named gemm_dims_aligned (GEMM shapes) while
    the enforcer is device_set_honoured; both are real checks, so the existence test
    passed and the rule was unenforced in fact while reading as covered (44 and 3b,
    independently, 2026-09-01). Ceiling, stated rather than papered over: existence is
    checkable, relevance is not. Nothing here can see a pair that names a real check
    which does not enforce that rule -- only a human re-reading the pair can.

    The doc's table must agree with the code's map. Nothing read the table at all -- it
    was a hand-maintained copy of _RULE_CHECKS, so the two could drift and, in the
    CUDA_VISIBLE_DEVICES row, were wrong together."""
    bullets, err = _agents_rule_bullets(root)
    if err:
        return FAIL, err
    if not bullets:
        return FAIL, "no rule bullets found -- the sections were renamed or emptied"
    known = {c[0] for c in CHECKS} | {"CI", "pre-commit hook", "podput", "pod_push.sh"}
    covered = _RULE_CHECKS
    unmapped = []
    for b in bullets:
        nb = _norm_rule(b)
        key = next((k for k in covered if nb.startswith(_norm_rule(k)[:38])), None)
        manual = next((k for k in _MANUAL_RULES if nb.startswith(_norm_rule(k)[:38])), None)
        if key is None and manual is None:
            unmapped.append(b[:55])
    if unmapped:
        return FAIL, f"{len(unmapped)} rule(s) map to neither a check nor a manual reason: {unmapped[:3]}"
    n_manual = len(_MANUAL_RULES)
    if n_manual > _MANUAL_BASELINE:
        return FAIL, f"manual rules rose to {n_manual} (baseline {_MANUAL_BASELINE}) -- say which became unenforceable"
    bad_ref = [v for v in covered.values() if v not in known]
    if bad_ref:
        return FAIL, f"rule maps to a check that does not exist: {bad_ref[:3]}"
    # The doc's copy of the map.
    table, terr = _agents_coverage_table(root)
    if terr:
        return FAIL, terr
    if not table:
        return FAIL, "the Rule coverage table is empty or its heading was renamed"
    drift = []
    for rule, val in table.items():
        nr = _norm_rule(rule)
        # The same prefix match the bullet loop uses: the table truncates long bullets,
        # so the row text and the map key are not equal strings.
        in_map = next((v for k, v in covered.items() if nr.startswith(_norm_rule(k)[:38])), None)
        in_manual = next((v for k, v in _MANUAL_RULES.items() if nr.startswith(_norm_rule(k)[:38])), None)
        v = val.strip("`").strip()
        if in_map is None and in_manual is None:
            drift.append(f"{rule[:34]}: in the table, in neither map")
        elif v.startswith("manual"):
            if in_manual is None:
                drift.append(f"{rule[:34]}: table says manual, code says {in_map}")
        elif in_map is None:
            drift.append(f"{rule[:34]}: table says {v}, code says manual")
        elif v != in_map:
            drift.append(f"{rule[:34]}: table says {v}, code says {in_map}")
    if drift:
        return FAIL, f"{len(drift)} coverage row(s) disagree with _RULE_CHECKS: {'; '.join(drift[:3])}"
    return PASS, (f"{len(bullets)} rules: {len(bullets) - n_manual} checked, {n_manual} manual "
                  f"(baseline {_MANUAL_BASELINE}); {len(table)} table rows agree with the code")


def _broken_agents_rules_covered():
    """The REAL AGENTS.md with its CUDA_VISIBLE_DEVICES coverage row reverted to the
    wrong check it carried until 2026-09-01 -- the exact defect 44 and 3b found.

    This breaks the TABLE half, which the old broken world (an unmapped bullet) never
    exercised. The bullet half is covered by _broken_agents_rules_unmapped, which CHECKS
    cannot hold (one broken() per row) and the selftest runs explicitly after the loop."""
    d = _tmp_repo()
    src = os.path.join(ROOT, "AGENTS.md")
    if not os.path.exists(src):
        return None
    text = open(src, encoding="utf-8").read()
    row = "| `CUDA_VISIBLE_DEVICES`, not `cuda:N` | `device_set_honoured` |"
    if row not in text:
        raise SelftestSkip("the coverage row moved; update _broken_agents_rules_covered")
    text = text.replace(row, "| `CUDA_VISIBLE_DEVICES`, not `cuda:N` | `gemm_dims_aligned` |", 1)
    open(os.path.join(d, "AGENTS.md"), "w", encoding="utf-8").write(text)
    return d


def _broken_agents_rules_unmapped():
    """The REAL AGENTS.md with a new unmapped rule bullet appended to Hard constraints."""
    d = _tmp_repo()
    src = os.path.join(ROOT, "AGENTS.md")
    if not os.path.exists(src):
        return None
    text = open(src, encoding="utf-8").read()
    marker = "- **CI gates.**"
    if marker not in text:
        return None
    text = text.replace(marker, "- **Invented rule nobody mapped.** Added by the broken world.\n" + marker, 1)
    open(os.path.join(d, "AGENTS.md"), "w", encoding="utf-8").write(text)
    return d


def _ppid_of(pid):
    """Parent pid on the pod, or None. ppid 1 means init adopted the process -- the
    launching shell exited and it survived, which is the property setsid provides.

    Single-pid form, kept for callers outside the check. The check itself reads every
    ppid in ONE ps (see _pod_ps_rows): a call per process cost 11 round trips and 6.3 s,
    which is what forced the 15 s deadline stopgap."""
    r = subprocess.run([os.path.expanduser("~/bin/pod"), f"ps -o ppid= -p {pid}"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def _pod_ps_rows(timeout=20):
    """Every process on the pod as (pid, sid, pgid, ppid, stat, args), in ONE remote read.

    The check needs pid/sid/pgid for the training rows AND the ppid of each -- two
    fields from the same table. Reading them as one `ps -eo` is one round trip
    regardless of how many training processes are up; the previous shape ran a
    `pod ps -o ppid= -p <pid>` per process, so cost scaled with the size of the
    training job it was watching. Measured 6.3 s at 11 ranks, 0.6 s batched.

    stat is here for the zombie case: a reaped-but-not-waited process keeps its argv
    in ps as `[run_ddp.sh] <defunct>`, matches any regex over the command line, and
    has no session of its own -- so a check that judges detachment by session reads
    it as a foreground trainer. It runs no code and holds no card. Only stat tells
    them apart (2026-09-01).

    Returns (rows, error). A non-empty error means the read failed and the caller
    must SKIP -- never treat an unreadable pod as a clean one.
    """
    pod = os.path.expanduser("~/bin/pod")
    try:
        r = subprocess.run([pod, "ps -eo pid,sid,pgid,ppid,stat,args --no-headers"],
                           capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, f"pod unreachable: {type(e).__name__}"
    if r.returncode != 0:
        return None, f"pod ps exit {r.returncode}"
    # stat is kept in the row: a zombie holds a pid slot, runs nothing, and holds no card,
    # but keeps its argv (`[run_ddp.sh] <defunct>`) so it matches any command regex. Only
    # stat tells it from a live trainer. judge_pod_ps drops them; the rows carry stat so
    # it can (2026-09-01).
    rows = []
    for ln in r.stdout.splitlines():
        parts = ln.split(None, 5)
        if len(parts) == 6 and parts[0].isdigit():
            rows.append(tuple(parts))
    if not rows:
        return None, "pod ps returned nothing"
    return rows, None


def judge_pod_ps(allrows):
    """(state, evidence) from a `ps -eo pid,sid,pgid,ppid,stat,args` table.

    A foreground training job is one whose SESSION LEADER is the crictl exec shell,
    because that shell dies with the tn tunnel and leaves the trainer holding a card.
    `pod "<cmd>"` runs the command as `bash -lc <cmd>`, so the leader's own argv says
    which it was: a detached launch names setsid there and the job it spawns lands in
    a new session; a foreground launch does not, and the job stays in the shell's.
    That is the whole rule -- read the leader, not the child.

    Four false positives in one day came from inferring detachment from the child
    instead: a launcher shell matched on its quoted argv, a trainer whose leader had
    become a zombie read as sessionless, a zombie trainer read as a live one, and a
    trainer adopted by init read as an orphan. Each refused a commit while the pod was
    behaving exactly as intended. The evidence those versions wanted -- an intact
    parent chain -- is reaped in the normal case, so they were reading absence and
    calling it a violation (de, 2026-09-01).

    Tested by scripts/test_pod_ps_judge.py against captured tables, which the check
    itself cannot be: it reads the live pod and its broken() raises SelftestSkip.
    """
    # Zombies keep their argv (`[run_ddp.sh] <defunct>`) so they match any command
    # regex, but they run no code and hold no card.
    live = [x for x in allrows if "Z" not in x[4]]
    leader = {x[0]: x[5] for x in live if x[0] == x[1]}
    rows = [x for x in live if re.search(r"train\.py|run_ddp", x[5])]
    # A leader that IS a training row is a detached launcher, never the exec shell.
    fg = [x for x in rows
          if x[0] != x[1]
          and leader.get(x[1], "").startswith("bash -lc")
          and "setsid" not in leader.get(x[1], "")]
    if fg:
        return FAIL, (f"{len(fg)} training process(es) in the crictl exec session "
                      f"(leader {fg[0][1]} is a bash -lc without setsid): pid {fg[0][0]}")
    if not rows:
        return PASS, "no training process on the pod"
    return PASS, f"{len(rows)} training process(es), none in a crictl exec session"


def check_no_foreground_pod_training(root):
    """No training process on the pod outside a setsid session.

    'Long jobs detach' is the rule; the failure it prevents is an orphan holding a
    whole card at 100% after the tn tunnel dies, which once contaminated a
    seven-card profile silently."""
    pod = os.path.expanduser("~/bin/pod")
    fake = os.environ.get("HARNESS_POD_PS")
    if fake:
        # Selftest injection. The check's broken world is a process TABLE, not a repo tree,
        # and the pod is shared -- staging the violation live would mean starting a
        # foreground trainer on the box running the 15B job.
        allrows = [tuple(p) for p in (ln.split(None, 5) for ln in open(fake, encoding="utf-8"))
                   if len(p) == 6 and p[0].isdigit()]
    else:
        if not os.path.exists(pod) or pod_drift.is_pod(root):
            return SKIP, "host-side check; needs ~/bin/pod"
        allrows, err = _pod_ps_rows()
        if err:
            return SKIP, err
    return judge_pod_ps(allrows)


def _broken_no_foreground_pod_training():
    """The FAIL table, captured live on the pod, fed through HARNESS_POD_PS.

    judge_pod_ps is covered by scripts/test_pod_ps_judge.py in CI, but the harness's own
    selftest skipped this check -- and a skip here means `harness --selftest` reports the
    check as unexercised, which is how it drifted into four false positives in one day.
    The predicate and the check are different things: the test proves the judgement, this
    proves the check WIRES that judgement to a FAIL.

    Rows are verbatim from de's capture: `pod "cd /work/aupai && ./run_ddp.sh --name
    fixture_fg_probe --help; sleep 40"` -- a real torchrun in a real crictl exec session,
    off every card before capture. Not hand-written: staging it live would mean starting
    a foreground trainer on a shared box, i.e. committing the incident the check prevents.
    """
    FOREGROUND = [
        "1389335 1389335 1389335       0 Ss   bash -lc cd /work/aupai && ./run_ddp.sh --name fixture_fg_probe --help >/dev/null 2>&1; sleep 40",
        "1389346 1389335 1389335 1389335 S    /bin/bash ./run_ddp.sh --name fixture_fg_probe --help",
        "1389348 1389335 1389335 1389346 Sl   /usr/bin/python3 /usr/local/bin/torchrun --nproc_per_node=8 train.py --fp8 --name fixture_fg_probe",
        "1389417 1389417 1389417 1389348 Rsl  /usr/bin/python3 -u train.py --fp8 --name fixture_fg_probe --help",
    ]
    d = _tmp_repo()
    ps = os.path.join(d, "data", "pod_ps.txt")
    os.makedirs(os.path.dirname(ps), exist_ok=True)
    with open(ps, "w", encoding="utf-8") as f:
        f.write("\n".join(FOREGROUND) + "\n")
    os.environ["HARNESS_POD_PS"] = ps
    return d


_SKIP_DIRS = {".git", "data", "runs", "node_modules", "__pycache__", ".venv", "venv"}


def walk_tracked(root, suffixes):
    """Yield (path, text) for every tracked source file under root, one definition.

    Excluding what cannot hold tracked source, rather than listing what can, means a new
    directory is covered by default. A hand-listed set of directories was missing probes/
    and the repo root while the evidence still read "every curl call passes -4" -- true
    of what it looked at, silent about what it did not.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [x for x in dirnames if x not in _SKIP_DIRS and not x.startswith(".")]
        for fn in sorted(filenames):
            if fn.endswith(suffixes):
                p = os.path.join(dirpath, fn)
                yield p, open(p, encoding="utf-8", errors="replace").read()


def strip_docstrings(text):
    """Blank docstring bodies, keeping their newlines so line numbers stay true.

    Blanking rather than deleting is the whole content of this function, and the reason
    it is one definition instead of three: `re.sub(..., "")` shifts every line number
    after the docstring, so a check that reports `path:n` names a line that holds nothing
    it was looking for. timestamps_are_utc had the blanking form and curl_ipv4 had the
    deleting one -- MEASURED 2026-09-02 on a file whose curl sits on line 10 after a
    six-line docstring: the deleting form reported line 5. Two copies of a traversal
    diverge; the copy without the fix is the one nobody was reading."""
    return re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
                  lambda m: "\n" * m.group(0).count("\n"), text)


def check_curl_ipv4(root):
    """Every curl invocation in tracked code passes -4.

    The pod's IPv6 egress is broken and curl tries IPv6 first; the failure surfaces
    as 'Errno 99 / Cannot assign requested address', which reads as 'host is
    unreachable' and is actually 'the local address family is unusable'. On
    2026-08-30 that produced a whole reachability matrix of false negatives."""
    bad = []
    # An invocation, not the word: `curl` inside a quoted argv element or a shell
    # command string. Matching the bare word finds docstrings -- including this
    # check's own, which is how the first version failed on itself.
    inv = re.compile(r"""(?:\[\s*|["'])curl["'\s]|^\s*curl\s|[;&|]\s*curl\s""")
    scanned = 0
    for p, text in walk_tracked(root, (".py", ".sh")):
        scanned += 1
        # Drop comments and blank docstrings before looking for invocations. This used a
        # deleting re.sub until 2026-09-02, so every line number it reported after a
        # docstring was too low -- the FAIL named a line holding no curl at all.
        text = strip_docstrings(text)
        for n, line in enumerate(text.split("\n"), 1):
            s = line.split("#", 1)[0]
            if inv.search(s) and not re.search(r"-4\b", s):
                bad.append(f"{os.path.relpath(p, root)}:{n}")
    if bad:
        return FAIL, f"{len(bad)} curl call(s) without -4: {bad[:3]}"
    return PASS, f"every curl call in {scanned} tracked .py/.sh passes -4"


def check_timestamps_are_utc(root):
    """Every timestamp written into the repo is UTC.

    The Mac runs CST and the pod container runs UTC, so the same instant was written
    as 22:04 and as 14:04 in the same format with nothing to tell them apart. Every
    runs/*.jsonl ledger mixes both clocks, and no_stale_running, review_present and
    tasks_closed_by_commit all compare those strings: a pod row reads eight hours old
    the moment it is written and a Mac row reads eight hours in the future."""
    bad = []
    scanned = 0
    for p, text in walk_tracked(root, (".py",)):
        scanned += 1
        text = strip_docstrings(text)
        lines = text.split("\n")
        for n, line in enumerate(lines, 1):
            # The call's arguments may wrap, so read the continuation too: a first
            # version named a strftime whose time.gmtime sat on the next line.
            s = "".join(lines[n - 1:n + 1]).split("#", 1)[0]
            if re.search(r"""(?<!["'])time\.strftime\(""", s) and "gmtime" not in s:
                bad.append(f"{os.path.relpath(p, root)}:{n}")
            elif re.search(r"""(?<!["'])time\.localtime\(""", s):
                bad.append(f"{os.path.relpath(p, root)}:{n}")
    if bad:
        return FAIL, f"{len(bad)} naive local-clock call(s): {bad[:3]}"
    return PASS, f"every strftime in {scanned} tracked .py passes time.gmtime()"


def _broken_timestamps_are_utc():
    """The REAL exp.py with its gmtime dropped, which is what it looked like today."""
    d = _tmp_repo()
    src = os.path.join(ROOT, "scripts", "exp.py")
    text = open(src, encoding="utf-8").read()
    if "time.gmtime()" not in text:
        return None
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "exp.py"), "w", encoding="utf-8").write(
        text.replace(", time.gmtime()", "", 1)
    )
    return d


def _broken_curl_ipv4():
    """A REAL fetcher with the -4 removed from its curl invocation."""
    d = _tmp_repo()
    src = os.path.join(ROOT, "datagen", "fetch_corpus.py")
    if not os.path.exists(src):
        return None
    os.makedirs(os.path.join(d, "datagen"), exist_ok=True)
    text = open(src, encoding="utf-8").read()
    if '"-4"' not in text:
        return None
    open(os.path.join(d, "datagen", "fetch_corpus.py"), "w", encoding="utf-8").write(
        text.replace('"-4",', "", 1)
    )
    return d


def check_running_sh_override_verified(root):
    """POD_PUSH_ALLOW_RUNNING_SH must reach the offset check, not return unconditionally.

    The override used to be one line -- `[ -z "${POD_PUSH_ALLOW_RUNNING_SH:-}" ] || return 1`
    -- so setting it skipped the running-script refusal entirely on the operator's word. The
    safety is a property of the DIFF, not of the flag: an edit whose every changed byte lands
    after a live shell's offset is safe, and the same flag on an edit touching an earlier byte
    is not, with no warning either way (de-48, 2026-09-04).

    Source-level because the alternative is running a push. What it catches is the wiring
    being removed -- pod_sh_offset.py's own 17 assertions all keep passing if nothing calls
    it, which is the shape a guard fails in silence.
    """
    sh = os.path.join(root, "scripts", "pod_push.sh")
    gate = os.path.join(root, "scripts", "pod_sh_offset.py")
    if not os.path.exists(sh):
        return SKIP, "scripts/pod_push.sh not present"
    if not os.path.exists(gate):
        return FAIL, ("scripts/pod_sh_offset.py is gone, so POD_PUSH_ALLOW_RUNNING_SH has "
                      "nothing to verify the operator's byte-offset claim against")
    text = open(sh, encoding="utf-8").read()
    body = strip_docstrings(text)
    lines = [ln.split("#", 1)[0] for ln in body.split("\n")]
    mentions = [ln for ln in lines if "POD_PUSH_ALLOW_RUNNING_SH" in ln]
    if not mentions:
        return FAIL, ("scripts/pod_push.sh no longer reads POD_PUSH_ALLOW_RUNNING_SH; the "
                      "running-script gate has no override and no offset check")
    # The defect shape verbatim: the flag short-circuiting straight to a permit.
    for ln in mentions:
        if re.search(r"POD_PUSH_ALLOW_RUNNING_SH[^\n]*\|\|\s*return\s", ln):
            return FAIL, (
                f"POD_PUSH_ALLOW_RUNNING_SH returns unconditionally ({ln.strip()[:70]}) -- "
                "the override permits a push on the operator's word. It must call "
                "scripts/pod_sh_offset.py --check and refuse when that refuses"
            )
    if "pod_sh_offset.py" not in body:
        return FAIL, ("scripts/pod_push.sh does not call scripts/pod_sh_offset.py, so "
                      "POD_PUSH_ALLOW_RUNNING_SH is trusted rather than verified")
    return PASS, "the running-.sh override calls the byte-offset check"


def _broken_running_sh_override_verified():
    """The REAL pod_push.sh with the override restored to its unconditional form."""
    import shutil as _sh
    d = _tmp_repo()
    src = os.path.join(ROOT, "scripts", "pod_push.sh")
    if not os.path.exists(src):
        return None
    text = open(src, encoding="utf-8").read()
    if "pod_sh_offset.py" not in text:
        return None
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    _sh.copy(os.path.join(ROOT, "scripts", "pod_sh_offset.py"),
             os.path.join(d, "scripts", "pod_sh_offset.py"))
    # Cut the verified branch back to the one line it replaced.
    start = text.index('  if [ -n "${POD_PUSH_ALLOW_RUNNING_SH:-}" ]; then')
    end = text.index("  ~/bin/pod \"ps -eo stat,args", start)
    broken = (text[:start]
              + '  [ -z "${POD_PUSH_ALLOW_RUNNING_SH:-}" ] || return 1\n'
              + text[end:])
    open(os.path.join(d, "scripts", "pod_push.sh"), "w", encoding="utf-8").write(broken)
    return d


def check_root_durable(root):
    """AUPAI_ROOT must not be on a Kubernetes emptyDir. A pod deletion destroys
    everything on /work; the durable NVMe drives are not visible inside the
    container, so this detects known-ephemeral mounts rather than comparing
    against durable ones. Reports FAIL on the pod today (root is /work/aupai)."""
    env = os.environ.get("AUPAI_ROOT")
    aupai = os.path.abspath(env) if env else root
    # Selftest override: a .ephemeral_mounts file in the root names the ephemeral list.
    ef = os.path.join(root, ".ephemeral_mounts")
    if os.path.exists(ef):
        ephemeral = [l.strip() for l in open(ef) if l.strip()]
    else:
        ephemeral = list(EPHEMERAL_MOUNTS)
    # Severity tracks whether a fix EXISTS, not whether the risk is real. With no durable
    # mount inside the container there is nothing any run can do about this, and a check no
    # run can turn green gets --force'd along with the real reds. It still speaks every time.
    # REVIVAL: the moment a durable drive is mounted in the container, moving AUPAI_ROOT
    # becomes an executable fix and this goes back to FAIL. That is what DURABLE_MOUNTS tests.
    df = os.path.join(root, ".durable_mounts")
    if os.path.exists(df):
        durable = [l.strip() for l in open(df) if l.strip()]
    else:
        durable = [m for m in DURABLE_MOUNTS if _is_mount(m)]
    for m in ephemeral:
        if aupai == m or aupai.startswith(m + os.sep):
            note = f"root {aupai} is on {m}, a Kubernetes emptyDir -- a pod deletion erases it"
            if durable:
                return FAIL, f"{note}; move AUPAI_ROOT to {durable[0]}"
            return WARN, f"{note}; no durable mount is visible in the container, so nothing to move to"
    return PASS, f"root {aupai} is not on a known-ephemeral mount"


def _broken_root_durable():
    """A root that IS under a (fake) ephemeral mount."""
    d = _tmp_repo()
    # A repo-real file so selftest does not skip this check as hand-written.
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w").close()
    # The root (d) is under its parent -- name the parent as the ephemeral mount.
    with open(os.path.join(d, ".ephemeral_mounts"), "w") as f:
        f.write(os.path.dirname(d) + "\n")
    # A durable mount exists here, so the violation is fixable and the check must FAIL.
    # Without one it is a WARN, which is the whole point of the severity split.
    with open(os.path.join(d, ".durable_mounts"), "w") as f:
        f.write(d + "\n")
    return d


# Pipeline step -> the data paths it writes, relative to the repo root.
# The harness refuses to run a step whose path escapes AUPAI_ROOT.
_PIPELINE_DATA_PATHS = {
    "fetch": ("data/raw",),
    "clean": ("data/corpus",),
    "score": ("data/scores",),
    "dedup": ("data/corpus",),
}


def _check_data_under_root(step):
    """Refuse if the step's data paths escape AUPAI_ROOT. Symlinks are allowed --
    they are the migration mechanism (data/raw -> /data00/aupai_raw)."""
    aupai = aupai_root()
    for name in _PIPELINE_DATA_PATHS.get(step, ()):
        target = os.path.normpath(os.path.join(ROOT, name))
        if target != aupai and not target.startswith(aupai + os.sep):
            raise ValueError(
                f"{step} writes to {target}, outside AUPAI_ROOT {aupai}. "
                f"Set AUPAI_ROOT or move the data under it."
            )


# --------------------------------------------------------------------------- facts


@functools.lru_cache(maxsize=None)
def cfg_default(field):
    """Read a Cfg field from train.py by AST -- importing train.py pulls torch, and this
    file must run on CPU-only CI. Raises on a field it cannot read: returning None once
    let a one-token annotation edit retire two checks while main() exited 0."""
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "Cfg":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and stmt.targets[0].id == field:
                    return ast.literal_eval(stmt.value)
                if isinstance(stmt, ast.AnnAssign) and getattr(stmt.target, "id", None) == field:
                    return ast.literal_eval(stmt.value)
    raise KeyError(f"train.py has no Cfg.{field}; the check that reads it cannot run")


def read_mix(path):
    """(domains, error). Never an empty dict: `"web" in {}` is False, so an unparseable
    mix would report a passing guard."""
    if not os.path.exists(path):
        return None, f"{os.path.relpath(path, ROOT)} does not exist"
    try:
        obj = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return None, f"unparseable: {e}"
    doms = obj.get("domains")
    if not isinstance(doms, dict) or not doms:
        return None, "no non-empty 'domains' map (schema drift, or an empty mix)"
    return list(doms), None


def _exp_open(row):
    """Is this experiments.jsonl event an OPEN one? exp.py's rule, not a fourth local copy.

    exp.py folds terminal-wins on `status != "running"` (:63) and pick_open_row's docstring states
    it: OPEN means the last event for this (name, started) is `running`. Every other status is
    terminal by kind, including the ones that are not `ok`/`fail` -- killed, stopped, retracted,
    dropped, provisional. Measured on the live ledger 2026-09-04: 18 distinct statuses, of which
    exactly one is open, and no row carries an empty or absent status. Written as a function rather
    than a set literal for the reason _exp_fold gives: this file held four re-implementations of
    exp.py's reduction and three were wrong.
    """
    return (row.get("status") or "") == "running"


def _exp_fold(evs):
    """The ledger's own fold, from scripts/exp.py. Lazy-imported, like _launch_shape.

    exp.py owns runs/experiments.jsonl (it is the only writer), so it owns the
    reduction; this file had four separate re-implementations of it and three of them
    were wrong in different ways (position-based last-wins in two, name-only keying in
    a third). Imported INSIDE the function rather than at module scope, for the reason
    launch_tests documents about launch_gate: a selftest world is a partial tree, and
    harness must still import where scripts/exp.py is absent. Falls back to the
    terminal-wins fold inline -- not to position-based -- so a missing exp.py degrades
    to the correct answer rather than the one this task exists to delete.
    """
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from exp import fold
        return fold(evs)
    except Exception:
        out = {}
        for r in evs:
            key = (r.get("name"), r.get("started"))
            prev = out.get(key)
            if (prev is not None and prev.get("status") != "running"
                    and r.get("status") == "running"):
                continue
            # Kept in step with exp.fold's retraction rule: a retracted row is terminal by
            # KIND, so an `ok` ordered after it by a union merge must not un-retract the run.
            # This fallback only runs when the import above fails, and a fallback that folds
            # differently from the real one is the divergence this function exists to end --
            # so the rule is duplicated deliberately rather than left to drift.
            if (prev is not None and prev.get("status") == "retracted"
                    and r.get("status") != "retracted"):
                continue
            out[key] = r
        return list(out.values())


@functools.lru_cache(maxsize=None)
def experiments(raw=False):
    """The experiment log, folded by (name, started) with a close beating a later start.

    The file is an event log -- exp.py appends a running row and later a terminal one
    rather than rewriting. A reader that does not fold sees a superseded status:
    t56_profile went ok 13:34 then fail 13:47, and an unfolded read failed
    score_matrix_present on the stale ok. This used to fold on POSITION, which reopens
    a closed run when a duplicate start lands after it (see _exp_fold); it now shares
    the ledger's one fold. raw=True yields every event."""
    p = os.path.join(ROOT, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return []
    evs = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                evs.append(json.loads(line))
            except Exception:
                pass
    return evs if raw else _exp_fold(evs)


CKPT_RE = re.compile(r"\bckpt_[A-Za-z0-9_.-]+?\.pt\b")
# Must take the number carrying the %: "math-hard 37/1032 = 3.6%" holds three numbers and
# only the last is the score. `[^%]` stops the window bleeding into the next metric.
SCORE_RE = re.compile(r"math-hard[^%]{0,40}?(\d+(?:\.\d+)?)\s*%")


def score_from(text):
    m = SCORE_RE.search(text or "")
    return float(m.group(1)) if m else None


def produced_checkpoint(cmd, run_name):
    """The checkpoint a run's cmd produced, or None. Priority: --out, then --name,
    then a single free ckpt_*.pt in the cmd. INPUTS are excluded: rl_direct resumed
    ckpt_k4 and scored its own output, and crediting k4 with that score is the
    loudest wrong-attribution bug this ledger had."""
    inputs = set(re.findall(r"--(?:resume|sft_path|tokenizer|ckpt)\s+(\S+)", cmd))
    m = re.search(r"--out\s+(ckpt_[A-Za-z0-9_.-]+)\.pt", cmd)
    if m:
        return m.group(1)
    if m := re.search(r"--name\s+([A-Za-z0-9_.-]+)", cmd):
        return f"ckpt_{m.group(1)}"
    if not cmd.strip():
        return f"ckpt_{run_name}"
    free = [n for n in CKPT_RE.findall(cmd) if n not in inputs]
    if len(free) == 1:
        return free[0][: -len(".pt")]
    return None


def recorded_scores():
    """checkpoint -> (math-hard %, source), plus scores that matched no checkpoint.
    eval_hard.sh takes the checkpoint positionally (matching only --out dropped every
    score it produced); inputs are excluded, or resuming ckpt_A credits A with the
    output's score."""
    scores, orphans = {}, []
    for row in experiments():
        s = score_from(str(row.get("result", "")))
        if s is None:
            continue
        cmd = str(row.get("cmd", ""))
        run = str(row.get("name", "?"))
        cand = produced_checkpoint(cmd, run)
        if cand is None:
            orphans.append((run, s, cmd[:60]))
            continue
        scores.setdefault(cand, (s, run))
    return scores, orphans


def checkpoint_names(scores):
    """Every checkpoint this repo knows about: on disk, named in a command, OR carrying a
    score. The last source once silently dropped the top of the ledger's own table."""
    names = {os.path.basename(p)[: -len(".pt")] for p in glob.glob(os.path.join(ROOT, "ckpt_*.pt"))}
    for row in experiments():
        names.update(n[: -len(".pt")] for n in CKPT_RE.findall(str(row.get("cmd", ""))))
    return sorted(names | set(scores))


def local_tokenizers():
    """path -> fingerprint, for every data/tokenizer*.json that loads."""
    out = {}
    try:
        from tokenizers import Tokenizer

        from loader import vocab_fingerprint
    except Exception:
        return out
    for p in sorted(glob.glob(os.path.join(DATA, "tokenizer*.json"))):
        try:
            out[os.path.basename(p)] = vocab_fingerprint(Tokenizer.from_file(p))
        except Exception:
            pass
    return out


# -------------------------------------------------------------------------- checks
#
# Each check is (name, asserts, incident, run, broken). `run(root)` -> (state, evidence);
# `broken()` -> a temp root violating the condition, where run() must report FAIL.


def _tmp_repo(mix_obj=None):
    """A throwaway tree shaped like the repo, for a check to fail against. The mix goes at
    cfg_default("mix") -- the path the checks actually read, not a made-up one."""
    import tempfile

    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "data", "corpus"), exist_ok=True)
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    if mix_obj is not None:
        p = os.path.join(d, cfg_default("mix"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(mix_obj, open(p, "w"))
    return d


def _tmp_repo_shaped(mix_obj=None):
    """A throwaway tree that SEES the real code, docs and data directories.

    A world built on the bare `_tmp_repo()` resolves nothing, so any check that reads a
    path FAILs there whether or not the mutation is present -- three worlds were green
    for exactly that reason (entrypoints_ran on 38 absent citations, pod_drift on 238,
    facts_well_formed on absent docs/ and data/eval). Symlinks, so the world costs
    nothing and the mutation is the only thing wrong with it. Write into a symlinked
    directory and you write into the repo, so a world that mutates a file under one must
    copy it in first (de, 2026-09-01)."""
    import shutil
    import subprocess

    d = _tmp_repo(mix_obj)
    for name in ("scripts", "eval", "datagen", "probes", "mathbank", "algorithms",
                 "filters", "docs", "facts"):
        if os.path.isdir(os.path.join(ROOT, name)) and not os.path.exists(os.path.join(d, name)):
            os.symlink(os.path.join(ROOT, name), os.path.join(d, name))
    for f in os.listdir(ROOT):
        if f.endswith((".py", ".sh")) and not os.path.exists(os.path.join(d, f)):
            os.symlink(os.path.join(ROOT, f), os.path.join(d, f))
    # A real `git init` plus a COPIED .gitignore. `_is_gitignored` shells out to
    # `git check-ignore` and only falls back to reading .gitignore itself, and that
    # fallback is weaker than git -- it missed data/corpus/math/, so every gitignored
    # pod-only artifact a fact cites read as rot. git also will not follow a symlinked
    # .gitignore, so this one is copied while everything else is linked.
    shutil.copy(os.path.join(ROOT, ".gitignore"), os.path.join(d, ".gitignore"))
    subprocess.run(["git", "init", "-q"], cwd=d, capture_output=True)
    for sub in os.listdir(os.path.join(ROOT, "data")):
        src, dst = os.path.join(ROOT, "data", sub), os.path.join(d, "data", sub)
        if not os.path.exists(dst):
            os.symlink(src, dst)
    for f in os.listdir(os.path.join(ROOT, "runs")):
        src, dst = os.path.join(ROOT, "runs", f), os.path.join(d, "runs", f)
        if not os.path.exists(dst):
            os.symlink(src, dst)
    return d


def _tiny_tokenizer_json(eos_id=1, with_num=True):
    """A minimal WordLevel tokenizer that is VALID but LOSSY, so the round-trip and
    pinned-id checks have something real to reject (an absent file only hits SKIP)."""
    vocab = {"<unk>": 0, "<eos>": eos_id, "a": 2, "b": 3}
    if with_num:
        vocab["[NUM]"] = 4
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }


def _broken_tokenizer(eos_id=1, with_num=True):
    if not os.path.isfile(os.path.join(ROOT, "data", "tokenizer.json")):
        raise SelftestSkip("no data/tokenizer.json -- check SKIPs without it")
    d = _tmp_repo()
    json.dump(
        _tiny_tokenizer_json(eos_id, with_num),
        open(os.path.join(d, "data", "tokenizer.json"), "w"),
    )
    return d


def _broken_stale_run():
    """The row is built by the REAL logger, not hand-written -- a hand-written row shares
    the check's own schema assumptions."""

    d = _tmp_repo()
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "exp.py"),
            "--root",
            d,
            "start",
            "--name",
            "killed_job",
            "--cmd",
            "x",
        ],
        check=True,
        capture_output=True,
    )
    p = os.path.join(d, "runs", "experiments.jsonl")
    rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    assert rows and rows[0]["status"] == "running", "exp.py start no longer opens a running row"
    rows[0]["started"] = "2020-01-01 00:00"
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    return d


def check_mix_not_unfiltered(root):
    doms, err = read_mix(os.path.join(root, cfg_default("mix")))
    if err:
        # NOT a pass: "could not check" must never read as "checked".
        return FAIL, f"cannot read the default mix: {err}"
    if "web" in doms:
        return FAIL, "the default mix names domain 'web' (the unfiltered 2,991,648-doc corpus)"
    return PASS, f"domains={doms}"


def _broken_mix():
    """Names 'web' AND has one domain resolving with the other absent -- the second half is
    what makes check_mix_shards report FAIL rather than the checkout SKIP."""
    d = _tmp_repo({"total_tokens": 1e9, "domains": {"web": {"weight": 0.5}, "gone": {"weight": 0.5}}})
    os.makedirs(os.path.join(d, "data", "corpus", "web"))
    open(os.path.join(d, "data", "corpus", "web", "a.jsonl"), "w").write("{}\n")
    return d


def check_test_record_after_last_stage(root):
    """A launch-test record is written AFTER the last stage, or it certifies an unfinished run.

    test_e2e.py called record_launch_test at the end of its try block while stage 11 ran in
    the `finally`, so a stage-11 AssertionError left runs/launch_tests.json saying
    "scripts/test_e2e.py: pass" for a run that exited nonzero (b0 measured it at the Stage E
    shape, 6e ruled 2026-09-04). The row is what gate_arch_tests believes, so a premature
    record is a launch cleared by evidence of a run that failed.

    READ FROM THE AST, not by grep: "record_launch_test appears after stage 11 in the file"
    is a text fact and "it executes after stage 11" is the one that matters. A call inside the
    try body and a call at the end of the finally are adjacent in the source and opposite in
    effect.

    Scope is every file that calls record_launch_test AND has a try/finally in the function
    that calls it -- if there is no finally, there is no later stage for the record to precede,
    and flagging it would be noise."""
    bad, seen = [], 0
    for p, body in walk_tracked(root, (".py",)):
        if "record_launch_test(" not in body:
            continue
        rel = os.path.relpath(p, root)
        if rel.endswith("launch_tests.py"):
            continue  # its own definition and selftest
        try:
            tree = ast.parse(body)
        except SyntaxError as e:
            bad.append(f"{rel}: unparseable ({e.lineno})")
            continue
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef,))]:
            tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
            if not tries:
                continue
            for t in tries:
                def _calls(nodes, name):
                    return [x.lineno for n in nodes for x in ast.walk(n)
                            if isinstance(x, ast.Call) and getattr(x.func, "id", None) == name]
                in_body = _calls(t.body, "record_launch_test")
                in_fin = _calls(t.finalbody, "record_launch_test")
                if not (in_body or in_fin):
                    continue
                seen += 1
                if in_body:
                    bad.append(f"{rel}:{in_body[0]} records inside the try body while the "
                               f"finally still runs -- a later failure leaves the row saying "
                               f"pass")
                    continue
                stages = _calls(t.finalbody, "stage")
                if stages and in_fin[0] < max(stages):
                    bad.append(f"{rel}:{in_fin[0]} records before the finally's last stage() "
                               f"at :{max(stages)}")
    if bad:
        return FAIL, "; ".join(bad)
    if not seen:
        return SKIP, "no file records a launch test inside a try/finally"
    return PASS, f"{seen} launch-test record(s), each written after the last stage"


def _broken_test_record_after_last_stage():
    """The REAL test_e2e.py with its record moved back into the try body -- the exact shape
    b0 hit. Mutated from the live file rather than hand-written: a fixture would encode my
    own idea of where the call sits, and the defect was about where it sits."""
    import shutil as _sh
    d = _tmp_repo_shaped()
    src = os.path.join(ROOT, "scripts", "test_e2e.py")
    if not os.path.exists(src):
        return None
    text = open(src, encoding="utf-8").read()
    if "record_launch_test(__file__" not in text or "    finally:" not in text:
        return None
    # Take the real call and re-plant it just before the `return 0` that precedes the finally.
    call = ('        record_launch_test(__file__, "pass", _record["shape"], real_kernel=True,\n'
            '                           mix=_record["mix"], stages=_record["stages"])\n')
    text = text.replace("        return 0\n    finally:", call + "        return 0\n    finally:", 1)
    real_scripts = os.path.join(ROOT, "scripts")
    if os.path.islink(os.path.join(d, "scripts")):
        os.unlink(os.path.join(d, "scripts"))
        os.makedirs(os.path.join(d, "scripts"))
        for f in os.listdir(real_scripts):
            if f != "test_e2e.py":
                os.symlink(os.path.join(real_scripts, f), os.path.join(d, "scripts", f))
    open(os.path.join(d, "scripts", "test_e2e.py"), "w", encoding="utf-8").write(text)
    _sh.copystat(src, os.path.join(d, "scripts", "test_e2e.py"))
    return d


def check_launch_line_vs_oom_facts(root):
    """A launch line whose shape exactly matches a recorded OOM config is a refusal.
    p200m_4b_0902 launched b32a1 twice on 2026-09-02 after eff.microbatch_32_oom had
    recorded that exact OOM (93.8/95.2 GB, ranks 3/6 first): the line had been checked
    against argparse, not against the facts. Exact match on (dim, layers, batch, accum,
    seq) only -- partial matches skip, no fuzzy matching. seq defaults to Cfg.seq
    (train.py:187, 4096; launch lines carry no --seq flag). grad_ckpt and world are
    printed in the FAIL message, never joined on: the fact store does not record them
    consistently, and a guard that silently assumes equality invents data."""
    key = ("dim", "layers", "batch", "accum", "seq")
    oom = []
    for fp in sorted(glob.glob(os.path.join(root, "facts", "*.json"))):
        try:
            obj = json.load(open(fp))
        except (OSError, ValueError):
            continue
        for e in obj.get("facts", []):
            cfg = e.get("config")
            if not isinstance(cfg, dict) or not all(k in cfg for k in key):
                continue  # incomplete config blocks skip: no fuzzy matching
            if "OOM" not in str(e.get("value", "")) + str(cfg.get("result", "")):
                continue
            oom.append((e.get("id", os.path.basename(fp)), cfg))
    if not oom:
        return FAIL, ("no OOM fact with a complete (dim, layers, batch, accum, seq) config "
                      "block -- the facts side of this check is empty")
    flag_re = re.compile(r"--(dim|layers|batch|accum)\s+(\d+)")
    bad = []

    def adjudicate(where, line):
        if "run_ddp.sh" not in line:
            return
        flags = {k: int(v) for k, v in flag_re.findall(line)}
        if not all(k in flags for k in ("dim", "layers", "batch", "accum")):
            return  # partial launch line: skip, no fuzzy matching
        flags.setdefault("seq", 4096)  # Cfg.seq, train.py:187
        for fid, cfg in oom:
            if all(flags[k] == cfg[k] for k in key):
                grad = ("--no-grad_ckpt" if "--no-grad_ckpt" in line else
                        "--grad_ckpt" if "--grad_ckpt" in line else "grad_ckpt unstated")
                m = re.search(r"NGPU=(\d+)", line)
                world = f"NGPU={m.group(1)}" if m else "world unstated"
                fgrad = cfg["grad_ckpt"] if "grad_ckpt" in cfg else "fact did not record grad_ckpt"
                fworld = f"cards={cfg['cards']}" if "cards" in cfg else "fact did not record world"
                bad.append(f"{where} matches OOM fact {fid} on "
                           f"dim/layers/batch/accum/seq={tuple(flags[k] for k in key)}; "
                           f"launch {grad}, {world}; fact {fgrad}, {fworld}")

    for doc in sorted(glob.glob(os.path.join(root, "docs", "lessons", "stop_window_*.md"))):
        try:
            lines = open(doc).read().splitlines()
        except OSError:
            continue
        for ln, line in enumerate(lines, 1):
            adjudicate(f"{os.path.basename(doc)}:{ln}", line)
    exp = os.path.join(root, "runs", "experiments.jsonl")
    if os.path.exists(exp):
        for e in _exp_events(root) or []:
            if e.get("status") == "running" and isinstance(e.get("cmd"), str):
                adjudicate(f"experiments.jsonl running row {e.get('name')} ({e.get('started')})",
                           e["cmd"])
    if bad:
        return FAIL, "; ".join(bad)
    return PASS, f"{len(oom)} joinable OOM fact(s); no launch line or running row matches"


def _broken_launch_line_oom():
    """The real stop-window doc with the 200M line set back to --batch 32 --accum 1:
    the exact shape eff.microbatch_32_oom records as OOM, and the shape p200m_4b_0902
    launched twice on 2026-09-02. docs/ is a symlink in a shaped world, so it is copied
    before the mutation -- writing through the link would write into the repo."""
    import shutil
    d = _tmp_repo_shaped()
    os.remove(os.path.join(d, "docs"))
    shutil.copytree(os.path.join(ROOT, "docs"), os.path.join(d, "docs"))
    doc = os.path.join(d, "docs", "lessons", "stop_window_2026-09-02.md")
    text = open(doc).read()
    mutated = re.sub(r"(run_ddp\.sh[^\n]*?--batch )16( --accum )2\b",
                     r"\g<1>32\g<2>1", text, count=1)
    assert mutated != text, "broken world found no run_ddp.sh --batch 16 --accum 2 line to mutate"
    open(doc, "w").write(mutated)
    return d


def _ckpt_names(text):
    """Concrete checkpoint filenames named in a fact's source/config text.

    Brace notation `X.pt.step{1500,2000,2500}` is an explicit enumeration and is
    expanded; everything else is exact-match only. A fact that shortens a name
    (`ckpt_p200m_4b_0902.pt.step832` for the on-disk `.pt.interrupt.step832`,
    b0's eff.kda_mla_growth_ratio_l12, 2026-09-02) is a DEFECT IN THE FACT, not
    something the check resolves away: the absent name FAILs as a dead source.
    `.pt` is required in the token so ckpt_health.py and friends never match;
    trailing doc extensions (.jsonl/.txt) are stripped so a citation of a
    readout sidecar resolves to its checkpoint."""
    text = re.sub(r"(ckpt_[\w.]+?)\.step\{([\d, ]+)\}",
                  lambda m: " ".join(f"{m.group(1)}.step{n.strip()}"
                                     for n in m.group(2).split(",")),
                  text)
    names = set()
    # (?<![A-Za-z0-9_.]) -- the token must START a name, not sit inside a longer one.
    # Without it, preds_l1_d3_ckpt_p200m_4b_0902.pt.en.jsonl mints a checkpoint called
    # ckpt_p200m_4b_0902.pt.en that has never existed, and the fact citing that prediction
    # file goes red forever with no action available. The two guards were in direct
    # tension: cited_artifacts_attested REQUIRES the artifact's basename in the fact, and
    # that basename embeds a checkpoint name by naming convention (fb, 2026-09-03).
    for tok in re.findall(r"(?<![A-Za-z0-9_.])ckpt_[\w.]+?\.pt[\w.]*", text):
        for ext in (".jsonl", ".txt", ".md"):
            if tok.endswith(ext):
                tok = tok[: -len(ext)]
        names.add(tok.rstrip("."))
    return names


def _parse_ckpt_listing(path):
    """-> (listing_date, keep_set, {candidate: (mtime, section)}).

    KEEP lines carry series shorthand (`X.pt.step2000, .pt.step2500`); a
    continuation attaches after the bare core OR after the `.pt` boundary, and
    both readings are kept -- the wrong reading names a file that cannot exist,
    so over-protection costs nothing and under-protection is the hazard."""
    keep, cands, date, section = set(), {}, None, "A"
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("# "):
            m = re.search(r"listed (\d{4}-\d{2}-\d{2} \d{2}:\d{2}Z)", line)
            if m:
                date = m.group(1)
            if re.match(r"# [A-Z]\.", line):
                section = line[2]
            if line.startswith("# KEEP"):
                base = pt = None
                for item in (s.strip() for s in line.split(",")):
                    # A claim line separates claims with "; " but shorthand continuations
                    # with ",", so one item can both continue the previous claim and name
                    # the next: attach the leading continuation FIRST (old base), then let
                    # tokens rebase. "NOT kept: X" is an explicit exclusion -- cut it.
                    kept = item.split("NOT kept")[0]
                    if item.startswith(".") and base:
                        cont = re.match(r"[\w.]+", item).group(0)
                        keep.add(base + cont)
                        if pt:
                            keep.add(pt + cont)
                    toks = re.findall(r"ckpt_[\w.]+?\.pt[\w.]*", kept)
                    if toks:
                        for t in toks:
                            keep.add(t.rstrip("."))
                        first = toks[0].rstrip(".")
                        base = re.match(r"ckpt_[\w.]+?(?=\.)", first).group(0)
                        mm = re.match(r"ckpt_[\w.]+?\.pt", first)
                        pt = mm.group(0) if mm else None
            continue
        m = re.match(r"(\d{4}-\d\d-\d\d_\d\d:\d\d) [\d.]+ (\S+)", line)
        if m:
            cands[m.group(2)] = (m.group(1), section)
    return date, keep, cands


def _noted_gone(entry, name, tier=None):
    """Whether the entry's uncertainty/boundary already names this checkpoint as
    deleted/pruned. A stale source with an honest note is a WARN, not a FAIL:
    b0's eff.kda_mla_growth_ratio_l32 keeps step1500 in its source (provenance) and
    records the pruning in uncertainty, and that is the right shape. The tail after
    `.pt` matches the note; tails under 5 chars (`ep1`) must appear in full, since
    `ep1` is a substring of `step1000` and friends. The gone-word list includes the
    check's own tier labels (`absent`, `zero`ed, `delet`ion-candidate): a note that
    says "[absent] -- not in the listing" is speaking this check's language.
    Name and gone-word must sit in the SAME sentence: concatenating the fields lets
    "measured on X. Y was pruned" disclose a death X never had (de, review of
    9420c8b, measured True). Semicolons stay inside a sentence -- they join an aside
    to the disclosure that owns it (e1's recal note names the ckpt, then "; its
    siblings ARE listed", then the pruning). ASCII "." is no boundary: checkpoint
    names are built from it.

    A QUOTED TIER LABEL ONLY DISCLOSES ITS OWN TIER (de, 2026-09-04, measured). Accepting
    the check's vocabulary was right, and tier-blind it downgrades a different complaint
    than the one the note answers. ds.n2_params_vs_data_matched_compute says both legs
    POSTDATE the listing, "and therefore read [absent]", and that they were verified
    present on the pod at 19:10Z, 892,199,291 and 1,854,896,463 bytes. Against the newer
    09-03 22:56Z listing the two legs are [deletion-candidate] -- on the prune plan,
    unclaimed -- and the single word `[absent]` was the whole credit: strip the bracket
    labels from that sentence and zero gone-words survive, so the FAIL AGENTS.md requires
    read WARN, on a fact whose own note asserts the files exist. A note answering "not in
    the snapshot" is not a note answering "scheduled for deletion"; the second needs a
    KEEP claim or prose, and a checkpoint deleted at 12:03Z takes its evidence with it.
    Prose disclosure is unrestricted -- the 12 other credited notes keep their WARN, 10 on
    prose that survives the strip and 2 on a label matching their own tier."""
    note = f"{entry.get('uncertainty') or ''} {entry.get('boundary') or ''}"
    if not note.strip():
        return False
    tail = name.split(".pt", 1)[-1].lstrip(".")
    for seg in re.split(r"[。！？!?]+", note):
        if not (name in seg or (len(tail) >= 5 and tail in seg)):
            continue
        # A label for a tier OTHER than the one being reported is not a disclosure of it.
        if tier:
            seg = re.sub(r"\[(?:zeroed|absent|deletion-candidate)\]",
                         lambda m: m.group(0) if m.group(0) == f"[{tier}]" else " ", seg)
        if re.search(r"prun|delet|zero|remov|gone|discard|absent|作废|删|丢|重置", seg, re.I):
            return True
    return False


def check_pod_stamp_is_main(root):
    """The pod's sync stamp names a commit that is main, or is reachable from it (de-14).

    Launch condition 2' has three clauses and only two had code: run_ddp.sh:23 refuses a
    dirty push and :34 refuses manifest drift, but "the stamp's sha is main's" was PRINTED
    and never compared -- a human read two hex strings off the log. The clause cannot be
    checked on the pod (no git, no route back), so it is checked here, where git is.

    TWO FAILURE SHAPES, and the second is the one that motivated it:
      not an ancestor   the stamp names a commit main does not contain -- pushed from an
                        unmerged branch. pod_push.sh's stamp_sync used `rev-parse HEAD`,
                        which in a per-session worktree is that BRANCH's tip: measured
                        2026-09-03, this tree's HEAD was 1b85dd0c while main was 69c8bd87.
                        Every pushed FILE is main's (push_one refuses any that differs), so
                        such a stamp describes a tree existing on no branch.
      behind main       main moved after the push. Expected and not a fault by itself, so it
                        WARNs with the distance rather than failing: the pod legitimately
                        runs an older commit until someone pushes again.

    WARN throughout: whether the pod should be re-pushed is a launch decision, and a FAIL
    here would go red on every tree the moment anyone merges to main.

    THE STAMP LIVES ON THE POD, so this reads it there via ~/bin/pod and resolves it HERE,
    where git is. A first version read data/pod_synced_head from the local tree and SKIPped
    on every real repository, because that file is pod-local and is not tracked -- the check
    would have shipped green and never once run on the value it exists to judge. Same
    two-filesystem join as pod_ledger_rows_home, and the same reason it cannot be one-sided."""
    fake = os.environ.get("HARNESS_POD_STAMP")
    if not os.path.exists(os.path.join(root, ".git")):
        return SKIP, "not a git checkout -- this clause is the one the pod cannot answer"
    if fake:
        text = open(fake, encoding="utf-8").read() if os.path.exists(fake) else ""
    else:
        local = os.path.join(root, "data", "pod_synced_head")
        if os.path.exists(local):
            text = open(local, encoding="utf-8").read()
        elif os.path.exists(os.path.expanduser("~/bin/pod")):
            r = subprocess.run([os.path.expanduser("~/bin/pod"),
                                "cat /work/aupai/data/pod_synced_head"],
                               capture_output=True, text=True)
            if r.returncode or not r.stdout.strip():
                return SKIP, "no stamp on the pod (a partial push clears it; nothing to compare)"
            text = r.stdout
        else:
            return SKIP, "needs ~/bin/pod to read the pod's stamp"
    parts = text.split()
    if not parts:
        return WARN, "the pod's stamp is empty"
    sha = parts[0]
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        return WARN, (f"the stamp holds {sha!r}, not a full 40-char hex sha -- an abbreviated "
                      f"sha is not an identity (de-38)")
    def _git(*a):
        return subprocess.run(["git", "-C", root, *a], capture_output=True, text=True)
    if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode:
        return WARN, f"the stamp names {sha[:12]}, which is no commit in this repository"
    main = _git("rev-parse", "main").stdout.strip()
    if not main:
        return SKIP, "no main in this repository"
    if sha == main:
        return PASS, f"the pod's stamp is main ({sha[:12]})"
    if _git("merge-base", "--is-ancestor", sha, main).returncode == 0:
        n = _git("rev-list", "--count", f"{sha}..main").stdout.strip() or "?"
        return WARN, (f"the pod is at {sha[:12]}, {n} commit(s) behind main ({main[:12]}) -- "
                      f"expected after a merge; re-push before a launch that needs them")
    return WARN, (f"the pod's stamp names {sha[:12]}, which main does NOT contain: it was "
                  f"pushed from an unmerged branch, so the stamp describes a tree that exists "
                  f"on no branch (the FILES are main's -- pod_push refuses any that differ)")


def _broken_pod_stamp_is_main():
    """A stamp naming a commit main does not contain: the real ledger of shas is git itself, so
    the world takes a REAL commit that is not an ancestor of main.

    Built from this repository's own refs rather than a made-up hex string, because a made-up
    sha fails at `cat-file -e` and would exercise the wrong branch -- the check would report
    "no such commit" and the ancestor comparison, which is the thing being tested, would never
    run. If no such commit exists here, the world cannot be built and says so."""
    d = _tmp_repo()
    r = subprocess.run(["git", "-C", ROOT, "rev-list", "--all", "--not", "main", "-n", "1"],
                       capture_output=True, text=True)
    sha = r.stdout.strip()
    if not sha:
        raise SelftestSkip("no commit outside main here; cannot build a non-ancestor stamp")
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    stamp = os.path.join(d, "data", "pod_synced_head")
    with open(stamp, "w", encoding="utf-8") as fh:
        fh.write(f"{sha} 0 2026-09-03T00:00:00Z\n")
    # HARNESS_POD_STAMP so the world does not depend on the pod being reachable: a verdict
    # that is right only when the tunnel is up proves nothing (same reason as
    # HARNESS_POD_LEDGERS).
    os.environ["HARNESS_POD_STAMP"] = stamp
    os.symlink(os.path.join(ROOT, ".git"), os.path.join(d, ".git"))
    return d


def check_pod_ledger_rows_home(root):
    """A ledger row written on the pod is invisible to every other check (de-36).

    pod_push only ever pushes, and pod_drift only asserts that the files it LISTS match --
    so an appended row on the pod's emptyDir is seen by nothing here. Five score_matrix rows
    behind the closed A/Bs lived only on the pod until someone moved them by hand, and two
    more (p500m_20b_0902 step1500/step2500, the live run's own measurements) had accumulated
    again by 2026-09-03.

    WARN, not FAIL: the rows are recoverable by running the puller, and a FAIL on a
    condition whose fix is one command trains people to bypass the gate. The count and the
    ledger are named so the WARN says what to run it for.

    This check runs where the REPOSITORY is and reads the pod, which is the opposite of the
    'pod-only, SKIP elsewhere' the task asked for -- the question is 'does the repo lack a
    row the pod has', and the repo is the side that must be present to answer it. On the pod
    itself there is no .git and the comparison has no local side, so it SKIPs there."""
    import pod_drift
    fake = os.environ.get("HARNESS_POD_LEDGERS")
    # is_pod is "this tree has no .git", and a _tmp_repo() world has none either -- so the
    # pod gate must come AFTER the injection, or every broken world reads as "we are on the
    # pod" and SKIPs. The selftest caught exactly that (de-36): SKIP on its own world, which
    # is the shape where a check cannot be made to fail.
    if not fake and pod_drift.is_pod(root):
        return SKIP, "on the pod: no local ledger to compare against (this check runs on main)"
    if not fake and not os.path.exists(os.path.expanduser("~/bin/pod")):
        return SKIP, "needs ~/bin/pod to read the pod's ledgers"
    try:
        sys.path.insert(0, os.path.join(root, "scripts"))
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import pod_pull_ledgers as ppl
        if fake:
            # Selftest injection, same shape as HARNESS_POD_PS: the broken world is a repo
            # whose ledger is short, and reading the REAL pod would make the world's verdict
            # depend on whether the pod is reachable from wherever the selftest runs -- a
            # world that passes for the wrong reason on a laptop with no tunnel.
            def _reader(rel, _pod_root, _dir=fake):
                p = os.path.join(_dir, os.path.basename(rel))
                if not os.path.exists(p):
                    return None, "empty or absent on the pod"
                return open(p, encoding="utf-8").read(), None
            rows = ppl.survey(root=root, reader=_reader)
        else:
            rows = ppl.survey(root=root)
    except Exception as e:
        return SKIP, f"could not read the pod's ledgers: {type(e).__name__}: {e}"
    behind = [(rel, len(missing)) for rel, _np, _nl, missing, _c, _n in rows if missing]
    # A key whose CURRENT rows disagree is the other half of "the pod's record did not come
    # home": the row is present but says something else. It cannot be auto-applied -- which
    # of two closes is right is a human's call -- so it WARNs like the missing case rather
    # than passing silently. Counting only `missing` here would have reported PASS on the
    # 14 disagreements the pull found on 2026-09-03.
    disagree = sum(len(coll) for _r, _np, _nl, _m, coll, _n in rows)
    if not behind and not disagree:
        return PASS, f"{len(rows)} ledger(s): every pod row's key is present locally"
    parts = [f"{rel} is missing {n} pod row(s)" for rel, n in behind]
    if disagree:
        parts.append(f"{disagree} key(s) whose current row differs between the pod and here")
    return WARN, "; ".join(parts) + " -- run scripts/pod_pull_ledgers.py"


def _broken_pod_ledger_rows_home():
    """A repo whose experiments ledger is EMPTY while the pod's holds rows: every pod row's
    key is then absent locally, so the check must WARN.

    Both sides are the REAL ledger -- copied, then one side emptied -- not hand-written, and
    the pod side is injected through HARNESS_POD_LEDGERS so the world's verdict does not
    depend on the pod being reachable from wherever the selftest runs. A world that reports
    the right state because the tunnel is down is a world that proves nothing."""
    import shutil

    d = _tmp_repo()
    fake = os.path.join(d, "_fake_pod")
    os.makedirs(fake, exist_ok=True)
    for rel in ("runs/score_matrix.jsonl", "runs/experiments.jsonl"):
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            raise SelftestSkip(f"{rel} absent; nothing real to build the world from")
        dst = os.path.join(d, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        shutil.copy(src, os.path.join(fake, os.path.basename(rel)))
    open(os.path.join(d, "runs", "experiments.jsonl"), "w").close()
    os.environ["HARNESS_POD_LEDGERS"] = fake
    return d


def check_run_commits_resolve(root):
    """Every experiments row's `commit` names an object this repository holds (de-38).

    A sha that resolves nowhere reads as provenance while answering nothing, and nothing
    looked: p500m_20b_0902's 00:03 row carried `cec145b`, which matches no commit here and no
    prefix of one. It surfaced only because the pod's copy of that row disagreed in this one
    field, and the disagreement was the ONLY reason anyone read it.

    The cause was width, not a wrong tree. exp.git_commit used `rev-parse --short` on the git
    path and a hardcoded `[:7]` on the pod-stamp path; --short is git's AUTO-SCALING
    abbreviation and began returning 8 characters once the object count grew, so one commit
    wrote two strings (8cd68340 vs 8cd6834). Both paths now store the full 40, which is why
    this check can be exact.

    WARN, not FAIL: the rows already written cannot be re-derived by this check, and a FAIL on
    unfixable history is a permanent red. `unknown` and a `+dirty<n>` suffix PASS -- both are
    honest statements about what the sha can say, and refusing them would push writers back to
    the blank that git_commit exists to eliminate."""
    p = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return SKIP, "no runs/experiments.jsonl"
    if not os.path.exists(os.path.join(root, ".git")):
        return SKIP, "not a git checkout (the pod cannot resolve a sha)"
    sys.path.insert(0, os.path.join(root, "scripts"))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import exp as _exp

    # THE FOLDED CURRENT ROW PER KEY, not every line. This ledger is an event log: a
    # correction is APPENDED and the last row under (name, started) is what the row says
    # now. Reading every line reports superseded values as live -- measured here, three rows
    # corrected to `unknown` still WARNed because their originals were still on disk, which
    # is exactly the sibling of the de-36 defect where an earlier pod row was compared
    # against a current local one. exp.fold is THE reduction; a second copy would be a
    # second thing to keep correct.
    evs = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            evs.append(json.loads(line))
        except ValueError:
            continue  # ledgers_one_line_per_row owns malformed lines
    seen, bad = {}, []
    for r in _exp.fold(evs):
        sha = r.get("commit")
        if sha is None or sha == "":
            continue  # rows predating the field; git_commit never returns ""
        if sha not in seen:
            seen[sha] = _exp.commit_resolves(sha, root)
        ok, why = seen[sha]
        if not ok:
            bad.append(f"{r.get('name')} {r.get('started')}: {why}")
    if not bad:
        return PASS, f"{len(seen)} distinct commit value(s), all resolve or are 'unknown'"
    return WARN, f"{len(bad)} row(s) name an unresolvable commit: {'; '.join(bad[:3])}"


def _broken_run_commits_resolve():
    """The REAL ledger with EVERY unresolvable commit repaired except one planted `cec145b` --
    the shape p500m_20b_0902's row actually had.

    Repairing the others is what makes the world a discriminator. The real tree already WARNs
    (three pod-side shas came home with de-36's pull), so a world that merely adds a fourth
    proves nothing: undo the mutation and it still WARNs. Here the mutation is the ONLY
    unresolvable value, so removing it would make the world PASS.

    THE PLANT GOES IN THE ROW THE FOLD KEEPS, not in rows[-1] (de, 2026-09-04). The check reads
    exp.fold -- the current row per (name, started) -- and this world wrote to the last LINE.
    Those are the same row only until someone appends a close: e1 appended a `running` row for
    e1_c11_doccu_rescore and then its `ok` close under the same key, so rows[-1] became a
    SUPERSEDED row, the fold dropped the planted `cec145b` before the check could see it, and
    the world reported PASS -- caught 40 minutes after a full selftest had passed on it. Nothing
    about the check or the world changed; another session closed a run. A fixture that targets a
    position in an event log is valid only as long as nobody appends, which in this repo is
    minutes. Written as "the last row of the last key the fold keeps", so it is the same
    reduction the check applies.
    """
    d = _tmp_repo()
    src = os.path.join(ROOT, "runs", "experiments.jsonl")
    if not os.path.exists(src):
        raise SelftestSkip("runs/experiments.jsonl absent; nothing real to mutate")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import exp as _exp

    rows = [json.loads(x) for x in open(src, encoding="utf-8") if x.strip()]
    assert rows, "the real ledger is empty"
    for r in rows:
        sha = r.get("commit")
        if sha and not _exp.commit_resolves(sha, ROOT)[0]:
            r["commit"] = "unknown"          # an honest non-answer; the check accepts it
    # Plant into a row the fold KEEPS: take the fold's own last row and mutate the ledger line
    # that IS it, matched by identity so no second copy of the reduction can disagree.
    kept = _exp.fold(rows)
    assert kept, "the fold kept no row, so there is nowhere to plant a defect"
    target = kept[-1]
    planted = False
    for r in rows:
        if r is target or ((r.get("name"), r.get("started")) == (target.get("name"),
                                                                 target.get("started"))
                           and r.get("status") == target.get("status")):
            r["commit"] = "cec145b"          # the planted defect, and now the only one
            planted = True
    assert planted, "the fold's last row is not in the ledger rows -- cannot plant"
    dst = os.path.join(d, "runs", "experiments.jsonl")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # The mutation must survive the check's own reduction, or this world certifies nothing --
    # which is exactly how it went green above. Asserted here rather than trusted.
    assert any(r.get("commit") == "cec145b"
               for r in _exp.fold([json.loads(x) for x in open(dst, encoding="utf-8")
                                   if x.strip()])), \
        "the planted commit does not survive exp.fold, so the world cannot fail the check"
    # The check needs an object database to ask `cat-file -e` of; the mutation is in the
    # ledger, not in git, so linking the real .git is what keeps the world minimal.
    os.symlink(os.path.join(ROOT, ".git"), os.path.join(d, ".git"))
    return d


def check_ckpt_facts_sources_present(root):
    """A measured fact whose only recomputable source is a checkpoint that is
    doomed or gone must be red at write time, not at prune time.

    Three tiers (44 review, fb ruling 2026-09-02):
    [deletion-candidate] a fact's source/config names a checkpoint on the pod
    deletion list (runs/pod_ckpt_candidates_*.txt, newest by date) that no KEEP
    line claims -- eff.kda_mla_growth_ratio_l32's step1500 was pruned with
    nothing red, and step2000/2500/3000 nearly followed it the same day.
    [absent] / [zeroed] a fact's source/config names a checkpoint not in the
    listing -- pruned (step1500), zeroed by the reset (section A), or misnamed
    (`.pt.step832` vs the on-disk `.pt.interrupt.step832`). FAIL, unless the
    entry's uncertainty/boundary already names it as gone: then WARN -- the
    fact is honest about its dead source, and source keeps its provenance.
    A listing is a snapshot: a checkpoint newer than it reads absent until the
    listing is refreshed, which the FAIL message says. A FAIL on such a checkpoint
    (de's interrupt.step1192, written 14:31Z against a 13:58Z listing) means REFRESH
    THE LISTING -- re-scan the pod -- not KEEP-claim: a checkpoint that was never a
    candidate needs no claim. The interim unblock is an uncertainty note naming the
    write time, which WARNs. Only source/config fields
    are scanned -- a ckpt mentioned in a value or uncertainty is prose, not a
    source claim. Names match exactly: a fact that shortens a name is a DEFECT
    IN THE FACT (b0's step832), never resolved away here."""
    listings = sorted(glob.glob(os.path.join(root, "runs", "pod_ckpt_candidates_*.txt")))
    if not listings:
        return FAIL, "no runs/pod_ckpt_candidates_*.txt -- the facts side of this check is empty"
    date, keep, cands = _parse_ckpt_listing(listings[-1])
    bad, warned, n_facts = [], [], 0
    for fp in sorted(glob.glob(os.path.join(root, "facts", "*.json"))):
        try:
            obj = json.load(open(fp))
        except (OSError, ValueError):
            continue
        for e in obj.get("facts", []):
            src = str(e.get("source", ""))
            cfg = e.get("config")
            if isinstance(cfg, dict):
                src += " " + json.dumps(cfg, ensure_ascii=False)
            else:
                src += " " + str(cfg or "")
            names = _ckpt_names(src)
            if not names:
                continue
            n_facts += 1
            fid = e.get("id", os.path.basename(fp))
            for name in sorted(names):
                if name in cands:
                    mtime, section = cands[name]
                    if name in keep:
                        continue
                    if section == "A":
                        tier = "zeroed"
                        msg = f"[zeroed] {fid} -> {name} (section A, zeroed by the reset)"
                    else:
                        tier = "deletion-candidate"
                        msg = f"[deletion-candidate] {fid} -> {name} (candidate {mtime}, not KEEP-claimed)"
                elif name not in keep:
                    tier = "absent"
                    msg = f"[absent] {fid} -> {name} (not in pod listing {date}; pruned, misnamed, or newer than the snapshot)"
                else:
                    continue
                (warned if _noted_gone(e, name, tier) else bad).append(msg)
    if bad:
        both = "; ".join(bad + warned)
        return FAIL, f"{len(bad)} FAIL + {len(warned)} WARN: fact source(s) name doomed/gone " \
                     f"checkpoints (listing {date}): {both}"
    if warned:
        return WARN, f"{n_facts} fact(s) cite checkpoints; {len(warned)} source(s) name a gone " \
                     f"checkpoint already disclosed in uncertainty/boundary (listing {date}): " + "; ".join(warned)
    return PASS, (f"{n_facts} fact(s) cite checkpoints; every name is KEEP-claimed or "
                  f"resolves against the listing ({date}, {len(cands)} candidates)")


def _broken_ckpt_facts_sources():
    """The real candidates listing with every KEEP line deleted: the checkpoints
    fb ruled to keep at 14:15Z become unkept deletion candidates, and the facts
    that cite them must FAIL. runs/ is not linked in a shaped world, so it is
    copied in first (2.2M) -- the same copy-before-mutate rule as docs/."""
    import shutil
    d = _tmp_repo_shaped()
    runs = os.path.join(d, "runs")
    if os.path.isdir(runs) and not os.path.islink(runs):
        shutil.rmtree(runs)  # _tmp_repo makes an empty runs/ dir
    shutil.copytree(os.path.join(ROOT, "runs"), runs)
    listings = sorted(glob.glob(os.path.join(runs, "pod_ckpt_candidates_*.txt")))
    assert listings, "broken world found no candidates listing to strip"
    path = listings[-1]
    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines()
             if not ln.startswith("# KEEP")]
    assert len(lines) < sum(1 for _ in open(path, encoding="utf-8")), \
        "broken world found no KEEP lines to strip"
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return d


def check_keep_claim_reasons_live(root):
    """A KEEP claim whose reason cites a RETRACTED fact must WARN named: the reason
    died but the claim did not. ckpt_facts_sources_present only checks fact->ckpt
    membership (is the ckpt named in a live fact), not whether the claim's reason
    still holds (44-25, 1e ruling 2026-09-03).

    step1192's claim was 'the ONLY evidence refuting ds.second_resume_rereads_one_segment';
    that fact was retracted the same day by 52aec31 and the claim stood. The checkpoint
    guard cannot see the reason's death -- it asks 'is the ckpt claimed', never 'is the
    fact it was claimed for still alive'. WARN, not FAIL: a retracted fact does not make
    the checkpoint worthless, it makes the claim's justification stale -- a human either
    re-justifies (b0's live-reason line) or lets the claim go."""
    listings = sorted(glob.glob(os.path.join(root, "runs", "pod_ckpt_candidates_*.txt")))
    if not listings:
        return SKIP, "no runs/pod_ckpt_candidates_*.txt -- no claims to check"
    status = {}
    for fp in sorted(glob.glob(os.path.join(root, "facts", "*.json"))):
        try:
            obj = json.load(open(fp))
        except (OSError, ValueError):
            continue
        for e in obj.get("facts", []):
            if e.get("id"):
                status[e["id"]] = str(e.get("status", "")).lower()
    stale = []
    for line in open(listings[-1], encoding="utf-8").read().splitlines():
        if not line.startswith("# KEEP"):
            continue
        for fid in sorted(set(re.findall(r"\b[a-z]{2,4}\.[a-z_0-9]+\b", line))):
            if status.get(fid) == "retracted":
                stale.append(f"{fid} (KEEP claim: {line[7:67]}...)")
    if stale:
        return WARN, f"{len(stale)} KEEP claim(s) cite retracted fact(s) -- re-justify or let the claim go: " \
                     + "; ".join(stale)
    return PASS, "every fact id cited in a KEEP claim is live"


def _broken_keep_claim_reasons():
    """A live fact cited by a KEEP claim is flipped to retracted: the check must WARN
    naming it. Citations of facts already retracted in the real repo are blanked first
    (dot -> underscore, so the id regex no longer matches) -- without that the world
    WARNs on the real repo's own stale claims and the mutation proves nothing. runs/ is
    copied, not linked, same as the sibling broken world."""
    import shutil
    d = _tmp_repo_shaped()
    runs = os.path.join(d, "runs")
    if os.path.isdir(runs) and not os.path.islink(runs):
        shutil.rmtree(runs)
    shutil.copytree(os.path.join(ROOT, "runs"), runs)
    # facts/ is symlinked into the world; copy it before mutating or the write lands
    # in the repo (the _tmp_repo_shaped rule -- measured: this world once flipped two
    # real facts to retracted before the copy was added).
    facts = os.path.join(d, "facts")
    if os.path.islink(facts):
        os.unlink(facts)
    elif os.path.isdir(facts):
        shutil.rmtree(facts)
    shutil.copytree(os.path.join(ROOT, "facts"), facts)
    listings = sorted(glob.glob(os.path.join(runs, "pod_ckpt_candidates_*.txt")))
    assert listings, "broken world found no candidates listing"
    path = listings[-1]
    text = open(path, encoding="utf-8").read()
    # Blank every citation of an ALREADY-retracted fact so a WARN can only come from
    # the mutation; otherwise the world WARNs on the real repo's own stale claims.
    retracted = set()
    for fp in glob.glob(os.path.join(d, "facts", "*.json")):
        try:
            obj = json.load(open(fp))
        except (OSError, ValueError):
            continue
        for e in obj.get("facts", []):
            if e.get("id") and str(e.get("status", "")).lower() == "retracted":
                retracted.add(e["id"])
    assert retracted, "broken world: no retracted facts to strip -- the repo changed shape"
    for fid in retracted:
        text = text.replace(fid, fid.replace(".", "_"))
    open(path, "w", encoding="utf-8").write(text)
    for fp in sorted(glob.glob(os.path.join(d, "facts", "*.json"))):
        obj = json.load(open(fp))
        for e in obj.get("facts", []):
            if e.get("id") == "be.l1_below_constant_guess":
                assert str(e.get("status", "")).lower() != "retracted", \
                    "broken world: target fact already retracted"
                e["status"] = "retracted"
                json.dump(obj, open(fp, "w"), indent=1, ensure_ascii=False)
                return d
    raise AssertionError("broken world: be.l1_below_constant_guess not found in facts/")


def _gpu_present():
    """Whether this machine can train. The strict branch of mix_shards_present guards the
    pod; a dev box with a partial corpus is normal. HARNESS_GPU_PRESENT=1/0 overrides -- the
    selftest forces 1 so the broken world exercises the strict branch."""
    forced = os.environ.get("HARNESS_GPU_PRESENT")
    if forced is not None:
        return forced == "1"
    return bool(glob.glob("/dev/nvidia[0-9]*"))


def check_non_shard_jsonl_excluded(root):
    """train.py must classify every .jsonl in a domain dir, and REFUSE the unclassifiable.

    Written when holdout_slice_<phase>.jsonl -- a per-phase family an exact-name list could
    never cover -- was globbed as a shard and three domains died on KeyError: 'content'. My
    first fix added a prefix skip. main's (044e5ed) is strictly better and replaced it: a
    whitelist for shards, a pattern for known non-shards, and a REFUSAL for anything matching
    neither. The difference matters -- a prefix skip silently drops a real shard someone
    misnames, which is the expensive failure; refusing costs two minutes at step 0.

    This checks the rule, not today's corpus: a dev box has no domain dirs, and a check that
    passes on an empty directory is the vacuous shape this file exists to retire."""
    src = os.path.join(root, "train.py")
    if not os.path.exists(src):
        return SKIP, "no train.py"
    body = open(src, encoding="utf-8").read()
    # Match the DEFINITION, not the name. `"SHARD_RE" in body` is satisfied by the comment
    # above it, so deleting the assignment left the check green on its own broken world --
    # a substring test passing on prose, which is the same defect as grepping a gate's
    # message for "ESTIMATED" (b0, 2026-09-01, twice in one day).
    for name in ("SHARD_RE", "NON_SHARD_RE"):
        if not re.search(rf"^{name}\s*=\s*re\.compile", body, re.M):
            return FAIL, (f"train.py does not define {name}: a new artifact written into a "
                          f"corpus dir will be tokenized as rows")
    # the refusal branch is the point -- a whitelist that silently skips is only half the fix
    if "REFUSING" not in body.split("def _domain_seqs")[1][:4000]:
        return FAIL, ("_domain_seqs classifies shards but does not REFUSE the unknown -- a "
                      "misnamed shard would be dropped from training in silence")
    return PASS, "train.py whitelists shards, patterns known non-shards, and refuses the rest"


def _broken_non_shard_jsonl_excluded():
    """train.py with the shard whitelist removed -- the pre-044e5ed world, where an unknown
    .jsonl in a corpus dir is read as data."""
    d = _tmp_repo()
    body = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    body = body.replace('SHARD_RE = re.compile(r"_\\d{3,}\\.jsonl$")', "")
    with open(os.path.join(d, "train.py"), "w", encoding="utf-8") as f:
        f.write(body)
    return d


def check_mix_shards(root):
    doms, err = read_mix(os.path.join(root, cfg_default("mix")))
    if err:
        return FAIL, f"cannot read the default mix: {err}"
    corpus = os.path.join(root, "data", "corpus")
    missing = [d for d in doms if not glob.glob(os.path.join(corpus, d, "*.jsonl"))]
    if not missing:
        return PASS, f"all {len(doms)} domains have shards"
    # Strictness follows the ability to train: a GPU box with a missing domain is about to
    # tokenize on missing data; dev boxes ship no corpus, and a permanent red is no signal.
    if not _gpu_present():
        return SKIP, f"no GPU on this machine: {len(missing)}/{len(doms)} domains lack shards (not the pod)"
    return FAIL, f"no shards for {missing}"


def _broken_mix_30b():
    """A 30B mix that silently drops a domain (weights sum to 0.9) AND names a frozen
    ladder directory ('en') -- the two halves the contract check exists to catch."""
    d = _tmp_repo()
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    json.dump(
        {
            "total_tokens": 30e9,
            "domains": {"en": {"weight": 0.5, "epochs": 1, "anneal": 0.5}},
            "_blocked": {"code_rp1t": {"weight": 0.4, "epochs": 1, "anneal": 0.4}},
        },
        open(os.path.join(d, "data", "mix_30b.json"), "w"),
    )
    # a ladder mix naming 'en', so the ladder-name ban has something to bite
    json.dump(
        {"total_tokens": 1e9, "domains": {"en": {"weight": 1.0}}},
        open(os.path.join(d, "data", "mix_scale_3.24b.json"), "w"),
    )
    return d


def check_mix_30b_contract(root, mix_rel="data/mix_30b.json"):
    """The 30B pretrain mix (t30) is a composition contract, not a launch-ready file:
    domains land one by one as 3b stamps them, and every not-yet-landed domain is declared
    in _blocked with its full contract. Three invariants keep the composition from silently
    shrinking to what exists: weights(domains)+weights(_blocked) sum to 1.0 (nothing dropped),
    no name is a frozen ladder directory (a stamped-30B-corpus name, never web_hq/en/math/...),
    and every landed domain is actually stamped. mix_rel lets a staged launcher check the
    stage's own file so the line a person reads at launch names the mix being launched."""
    p = os.path.join(root, mix_rel)
    if not os.path.exists(p):
        return SKIP, f"{mix_rel} not present"
    try:
        mix = json.load(open(p, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return FAIL, f"mix_30b.json will not parse: {e}"
    landed = mix.get("domains", {})
    blocked = mix.get("_blocked", {})
    if not isinstance(landed, dict) or not isinstance(blocked, dict):
        return FAIL, "domains and _blocked must both be objects"
    # 1. nothing dropped: the declared composition still sums to one
    w_sum = sum(float(d.get("weight", 0)) for d in list(landed.values()) + list(blocked.values()))
    if abs(w_sum - 1.0) > 1e-3:
        return FAIL, f"weights(domains)+weights(_blocked) = {w_sum:.5f}, not 1.0 -- a domain was dropped or reweighted silently"
    # 2. no frozen ladder directory names (a 30B domain binds to a NEW stamped corpus)
    ladder = set()
    for lf in glob.glob(os.path.join(root, "data", "mix_scale_*.json")):
        try:
            ladder |= set(json.load(open(lf, encoding="utf-8")).get("domains", {}))
        except Exception:  # noqa: BLE001
            pass
    reused = [n for n in list(landed) + list(blocked) if n in ladder]
    if reused:
        return FAIL, f"30B mix reuses frozen ladder directory name(s) {sorted(reused)} -- bind to a new stamped directory"
    # 3. every landed domain is stamped (build_corpus_stats.json), strict only on a train box
    corpus = os.path.join(root, "data", "corpus")
    if landed:
        unstamped = [n for n in landed if not os.path.exists(os.path.join(corpus, n, "build_corpus_stats.json"))]
        if unstamped:
            if not _gpu_present():
                return SKIP, f"no GPU on this machine: landed domains {unstamped} not yet stamped (not the pod)"
            return FAIL, f"landed domains not stamped: {unstamped}"
    return PASS, f"weights sum to {w_sum:.5f}; {len(landed)} landed, {len(blocked)} blocked ({', '.join(sorted(blocked)) or 'none'})"


def check_tokenizer_roundtrip(root):
    p = os.path.join(root, "data", "tokenizer.json")
    if not os.path.exists(p):
        return SKIP, "data/tokenizer.json not present"
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(p)
    except Exception as e:
        return FAIL, f"tokenizer will not load: {e}"
    probe = "a\x00b\tc 中文 42"
    got = tok.decode(tok.encode(probe, add_special_tokens=False).ids)
    if got != probe:
        return FAIL, f"round-trip lost bytes: {probe!r} -> {got!r}"
    return PASS, "NUL, tab, hanzi and digits survive"


def check_pinned_ids(root):
    p = os.path.join(root, "data", "tokenizer.json")
    if not os.path.exists(p):
        return SKIP, "data/tokenizer.json not present"
    try:
        from tokenizers import Tokenizer

        import loader

        v = Tokenizer.from_file(p).get_vocab()
    except Exception as e:
        return FAIL, f"cannot read vocabulary: {e}"
    eos, num = v.get("<eos>"), v.get("[NUM]")
    want_num = cfg_default("num_id")
    if eos != loader.EOS_ID:
        return FAIL, f"<eos> is {eos}, four files hardcode {loader.EOS_ID}"
    if num != want_num:
        return FAIL, f"[NUM] is {num}, Cfg.num_id is {want_num}"
    return PASS, f"<eos>={eos} [NUM]={num}"


MAX_TRACKED_MB = 5


def check_vocab_id_on_load_path(root):
    """Every trainer that loads an SFT pack COMPARES the pack's vocab_id to the checkpoint's.

    The rule is AGENTS.md "Vocabulary identity": a pack from another vocabulary trains silently
    at ~4x the loss, because every id is wrong, in range, and the sizes match. `data/tokenizer.json`
    is rebuilt in place, so nothing else distinguishes two vocabularies.

    THE ROW THIS CLOSES WAS `manual: enforced at load since 7aacbac`, and reading the tree for it
    found the enforcement is on ONE of the two pack loaders. 7aacbac fixed sft_math.py, where the
    guard had been keyed on `"vocab" in d` while the packer writes `vocab_id` -- so the assert never
    fired and the run printed "the pack predates vocabulary fingerprinting" about a pack that
    carried the fingerprint. sft.py loads a pack at sft.py:75 and compares nothing: it reads
    ck["vocab_id"] only to STAMP the checkpoints it writes (sft.py:168, :178), which propagates the
    id without ever checking it. The narrow fix is the shape memory/cause-named-one-site-too-narrow
    records: the cause was recorded as "this function read the wrong key" when it was "this repo has
    two pack loaders and only one asks the question".

    AST, NOT A SUBSTRING, and the reason is this rule's own history. The defect 7aacbac fixed was a
    guard present in the source, spelled correctly, reading a key that did not exist -- a grep for
    `vocab_id` was GREEN throughout. So the check requires, per loader: the pack dict is subscripted
    or .get() for a vocab key, AND that value reaches a comparison. A guard that reads the key and
    drops the value on the floor is the defect, not the fix.

    WHAT IT CANNOT SEE (the coverage table's manual column, kept here because the table takes only
    a check name): whether the ids themselves are right -- only that the question is asked at every
    load site. And the population is trainers taking `--sft_path`, so a third pack reader that
    invents its own flag is outside it. It also says nothing about `holdout_fp`: sft.py lacks that
    guard too, which sft_math.py has had since 2026-09-03, and that is a different rule.
    """
    import ast as _ast

    # (file, the argparse dest that names the pack) -- a trainer is in scope because it LOADS a
    # pack, so this list is derived from that, not from a hand-kept roster of trainers.
    loaders = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".py"):
            continue
        p = os.path.join(root, fn)
        try:
            with open(p, encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue
        if "--sft_path" not in src:
            continue
        loaders.append((fn, src))
    if not loaders:
        return SKIP, "no trainer takes --sft_path here"

    bad = []
    ok = []
    for fn, src in loaders:
        try:
            tree = _ast.parse(src)
        except SyntaxError as e:
            bad.append(f"{fn} does not parse: {e}")
            continue
        # the name the pack dict is bound to: `<name> = torch.load(args.sft_path...)`
        pack_names = set()
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Assign) or not isinstance(node.value, _ast.Call):
                continue
            call = _ast.unparse(node.value)
            if "torch.load" in call and "sft_path" in call:
                for t in node.targets:
                    if isinstance(t, _ast.Name):
                        pack_names.add(t.id)
        if not pack_names:
            bad.append(f"{fn} names --sft_path but no `x = torch.load(args.sft_path)` was found; "
                       f"this check can no longer see how the pack is read")
            continue
        # The variable the pack's vocab key is READ INTO, and whether THAT NAME reaches a
        # comparison against something else. Following the name, not the word "vocab": the first
        # version asked whether any Compare in the file mentioned "vocab", which is satisfied by
        # `Cfg.vocab == n` or by the checkpoint-side read, so both worlds where the comparison was
        # removed stayed GREEN. Verified by mutating: `assert pack_vocab == ck_vocab` weakened to
        # `assert pack_vocab is not None` passed, which is the whole defect this check exists for.
        read_into, reads = set(), []
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Assign) or not isinstance(node.targets[0], _ast.Name):
                continue
            rhs = _ast.unparse(node.value)
            if "vocab" not in rhs:
                continue
            if any(f"{n}[" in rhs or f"{n}.get(" in rhs for n in pack_names):
                read_into.add(node.targets[0].id)
                reads.append(f"{node.targets[0].id} = {rhs}"[:70])
        compared = False
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Compare):
                continue
            # the read variable on one side, and something OTHER than a constant on the other:
            # `pack_vocab is not None` is a presence test, not a vocabulary comparison.
            sides = [_ast.unparse(node.left)] + [_ast.unparse(c) for c in node.comparators]
            if not any(s in read_into for s in sides):
                continue
            others = [s for s in sides if s not in read_into]
            if any(s not in ("None", "True", "False", "0", "''", '""') for s in others):
                compared = True
        if not reads:
            bad.append(f"{fn} loads a pack ({', '.join(sorted(pack_names))}) and never reads its "
                       f"vocab_id -- a pack from another vocabulary trains at ~4x the loss with "
                       f"every id wrong and in range")
        elif not compared:
            bad.append(f"{fn} reads the pack's vocab key ({reads[0]}) but nothing compares that "
                       f"value against the checkpoint's; a guard that reads the key and drops the "
                       f"value, or only tests it for presence, is the 7aacbac defect")
        else:
            ok.append(fn)
    if bad:
        return FAIL, "; ".join(bad)
    return PASS, (f"{len(ok)} pack loader(s) compare the pack's vocab_id to the checkpoint's "
                  f"({', '.join(ok)})")


def _broken_vocab_id_load_path():
    """sft_math.py's guard with the KEY RENAMED, which is the 7aacbac defect itself.

    Not a deleted assert: a deleted one is caught by a substring search too, and would prove
    nothing about why this check reads the AST. The world here keeps the assert, keeps the word
    vocab_id in the file, and points the read at a key the packer does not write -- exactly the
    state the repo was in until 2026-09-02, when a grep was green and the check never fired.
    """
    import shutil

    d = _tmp_repo_shaped()
    for fn in ("sft.py", "sft_math.py"):
        src = os.path.join(ROOT, fn)
        if not os.path.isfile(src):
            raise SelftestSkip(f"{fn} absent")
        dst = os.path.join(d, fn)
        if os.path.islink(dst):
            os.remove(dst)
        shutil.copy(src, dst)
    p = os.path.join(d, "sft_math.py")
    with open(p, encoding="utf-8") as fh:
        s = fh.read()
    # the read that feeds the assert, pointed at a key nothing writes
    old = 'pack_vocab = d.get("vocab_id", d.get("vocab"))'
    if old not in s:
        raise SelftestSkip("sft_math.py no longer reads the pack vocab this way")
    s = s.replace(old, 'pack_vocab = None  # d.get("vocabulary_identity")', 1)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(s)
    return d





# Scripts harness.py shells out to, as (path relative to ROOT, what invokes it). A subprocess
# path is a reference the interpreter never resolves until the moment it runs, so a moved file
# leaves it looking correct and failing only when someone actually needs that command.
_SPAWNED_SCRIPTS = [
    ("scripts/exp.py", "the experiment ledger, several call sites"),
    ("scripts/pod_drift.py", "manifest regeneration"),
    ("datagen/pretokenize.py", "harness run pretokenize"),
    ("eval/eval_all.sh", "harness eval"),
    ("eval/eval_hard.sh", "harness eval --hard"),
    ("run_ddp.sh", "harness launch"),
    ("scripts/card_claim.py", "harness launch acquires the cards; the monitor releases them"),
]


def check_spawned_scripts_exist(root):
    """Every script harness.py spawns is where harness.py says it is.

    c3a47e8 moved the corpus-build scripts from scripts/ to datagen/ and three
    os.path.join(HERE, ...) call sites kept pointing at scripts/. Nothing noticed, because a
    subprocess path is only resolved when the command runs, and none of the three had run
    since: `harness run pretokenize` is the step that warms token caches -- the launch gate's
    own epochs prerequisite -- and it would have died on FileNotFoundError at the moment
    someone tried to clear that gate (b0, 2026-09-01).

    Checking the list rather than parsing the source: a regex over os.path.join(HERE, "x.py")
    would miss an f-string or a variable, and the failure mode here is a path that reads fine.
    The list is asserted complete by the selftest against a grep of the source."""
    missing = [(p, why) for p, why in _SPAWNED_SCRIPTS
               if not os.path.exists(os.path.join(root, p))]
    if missing:
        return FAIL, (
            f"{len(missing)} spawned script(s) absent: "
            + "; ".join(f"{p} ({why})" for p, why in missing[:3])
            + " -- a subprocess path resolves only when it runs, so this fails at use, not here"
        )
    # The list must also be COMPLETE, or the check reports green over a call site nobody
    # listed. Scan this file's own source for the literal-path shape and require every hit to
    # be covered. Known ceiling, stated rather than hidden: this finds os.path.join(<dir>,
    # "name.py"/"name.sh") literals, not f-strings or variables. Those exist and are not
    # covered -- a wider net would need a real call-graph, and a check that overstates its
    # coverage is the defect this whole file spent the day on.
    src = open(os.path.join(root, "scripts", "harness.py"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(root, "scripts", "harness.py")) else ""
    # Only paths built INSIDE a command list. The first version matched any
    # os.path.join(DIR, "x.py") and reported six extras -- a shutil.copy, three file reads,
    # a substring test, and a filename quoted in this docstring. It conflated "a path is
    # constructed" with "a script is spawned", which would have trained the next reader to
    # add unrelated files to the list until the check meant nothing.
    named = set(re.findall(
        r'(?:subprocess\.(?:run|Popen|check_output)\(|cmd\s*=\s*)\[[^]]*?'
        r'os\.path\.join\([A-Z]+,\s*(?:"[a-z_]+",\s*)?"([a-z_]+\.(?:py|sh))"', src, re.S))
    listed = {os.path.basename(p) for p, _ in _SPAWNED_SCRIPTS}
    unlisted = sorted(named - listed - {"harness.py"})
    if unlisted:
        return FAIL, (
            f"{len(unlisted)} spawned script(s) in the source but not in _SPAWNED_SCRIPTS: "
            f"{', '.join(unlisted)} -- add them, or this check is green over an unchecked path"
        )
    # EXISTING IS NOT ENOUGH: a script can be at its path and still fail on import. c3a47e8
    # moved datagen/pretokenize.py out of scripts/, where `sys.path.insert(0, ROOT)` had been
    # enough to find `import harness` -- from datagen/ it is not, and the file raised
    # ModuleNotFoundError the first time anyone ran it after the move, which was tonight,
    # warming caches for the launch gate. Presence and importability are different properties
    # and the same commit broke both.
    #
    # RESOLVED, NOT EXECUTED. Executing the module body cost 9s and timed out eight runs in
    # a row, blocking every commit in the repo while reporting "has not actually run since"
    # -- a red carrying no information that no fix could clear. Threading it did not help:
    # the cost is not interpreter startup but one import, fla.ops.kda, at 6.07s of 7.6s
    # (torch itself is 0.92s), and three of those contend rather than overlap.
    #
    # So each top-level import is resolved with find_spec under the sys.path the script
    # itself builds -- honouring its own inserts, or `import harness` from datagen/ reads as
    # broken when it is not. 0.008s for all three, and verified to still FAIL on a
    # reconstructed c3a47e8 tree (harness.py in scripts/, the file in datagen/, only ROOT
    # inserted). KNOWN CEILING, stated rather than hidden: this catches an import that
    # cannot be FOUND, not one that raises while executing. A module that imports fine and
    # then throws in its body passes here -- covering that needs the 9s, and a check nobody
    # can afford to run is worth less than a fast one that says what it covers.
    def _import_check(item):
        rel, why = item
        full = os.path.join(root, rel)
        try:
            tree = ast.parse(open(full, encoding="utf-8").read())
        except (OSError, SyntaxError) as e:
            return f"{rel}: {type(e).__name__} {e} ({why})"
        HERE, ROOT_ = os.path.dirname(os.path.abspath(full)), root
        extra = []
        mods = []
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "insert" and len(n.args) > 1
                    and getattr(getattr(n.func.value, "value", None), "id", "") == "sys"):
                try:
                    extra.append(eval(compile(ast.Expression(n.args[1]), "<p>", "eval"),
                                      {"os": os, "ROOT": ROOT_, "HERE": HERE}))
                except Exception:
                    pass  # a computed path we cannot evaluate: skip, do not guess
            elif isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods.append(n.module)
        # PathFinder against explicit paths, NOT find_spec. find_spec consults
        # sys.modules first, and harness.py has already imported the very modules it is
        # testing -- so it returned "resolvable" on a reconstructed c3a47e8 tree with
        # sys.path emptied. Replacing sys.path did not help for the same reason. The
        # finder takes the search path as an argument and ignores both sys.path and the
        # import cache, which is the only way to ask "could THAT script find this".
        #
        # Third-party and stdlib names are skipped rather than resolved: PathFinder over
        # the script's own dirs says False for `json` too, and reporting the stdlib as
        # missing would be a check that fails on every correct file.
        search = [HERE] + [e for e in extra if isinstance(e, str)]
        # Names the REPO owns, found anywhere in it -- not names the script can already
        # reach. Scanning only the search path inverts the test: harness.py lives in
        # scripts/, which the broken c3a47e8 script never adds, so it looked like a
        # third-party module and was skipped. The check then passed on the one tree it
        # exists for. What makes a name ours is that the file is in the repo; whether
        # this script can reach it is the question, not the filter.
        repo_names = set()
        for d in ("", "scripts", "datagen", "eval", "probes", "algorithms"):
            dd = os.path.join(root, d)
            if os.path.isdir(dd):
                repo_names |= {os.path.splitext(f)[0] for f in os.listdir(dd)
                               if f.endswith(".py")}
        bad = []
        for m in sorted(set(mods)):
            top = m.split(".")[0]
            # Only names this repo could own. A missing third-party dep is a different
            # problem and not what this check is for.
            if top not in repo_names:
                continue
            if importlib.machinery.PathFinder.find_spec(top, search) is None:
                bad.append(top)
        return f"{rel}: cannot resolve {', '.join(bad)} ({why})" if bad else None

    py = [(rel, why) for rel, why in _SPAWNED_SCRIPTS if rel.endswith(".py")]
    broken = [b for b in (_import_check(i) for i in py) if b]
    if broken:
        return FAIL, (
            f"{len(broken)} spawned script(s) present but not importable: "
            + "; ".join(broken[:3])
            + " -- being at the right path is not the same as running"
        )
    return PASS, (f"all {len(_SPAWNED_SCRIPTS)} spawned scripts present and importable, and the "
                  f"list covers every literal path in harness.py")


def _broken_spawned_scripts_exist():
    """A tree where pretokenize.py sits at the path the refactor left behind.

    The real defect, minimised: the file exists in the repo, just not where the caller looks.
    That is why it read as fine -- `ls datagen/` and `grep pretokenize` both succeed."""
    d = _tmp_repo()
    for p, _ in _SPAWNED_SCRIPTS:
        full = os.path.join(d, p)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w").write("# stub\n")
    # move the one the refactor moved, to where it used to live
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    os.rename(os.path.join(d, "datagen", "pretokenize.py"),
              os.path.join(d, "scripts", "pretokenize.py"))
    return d


def _broken_spawned_scripts_importable():
    """Every script at its correct path, but pretokenize.py with its scripts/ path entry
    removed -- the SECOND half of what c3a47e8 broke, and the half that survived the first
    fix. Reverting one line of the real file rather than writing a stub: a stub would import
    cleanly and prove nothing."""
    import shutil

    d = _tmp_repo()
    for rel, _ in _SPAWNED_SCRIPTS:
        full = os.path.join(d, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        src = os.path.join(ROOT, rel)
        if os.path.exists(src):
            shutil.copy(src, full)
        else:
            open(full, "w").write("# stub\n")
    # The module the mutation makes unreachable must EXIST in the world, or the resolver
    # reads `harness` as third-party and skips it -- the world passed for that reason, not
    # because the defect was absent.
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "scripts", "harness.py"),
                os.path.join(d, "scripts", "harness.py"))
    pre = os.path.join(d, "datagen", "pretokenize.py")
    body = open(pre).read().replace(
        'sys.path.insert(0, os.path.join(ROOT, "scripts"))', "")
    open(pre, "w").write(body)
    return d


def check_no_oversized_blob(root):
    """gitignore does not cover already-tracked paths, so the pattern list was never the
    guard. This fires on the next one."""
    import subprocess

    p = subprocess.run(["git", "-C", root, "ls-tree", "-r", "-l", "HEAD"], capture_output=True, text=True)
    if p.returncode:
        return SKIP, "not a git repository (the pod checkout is not one)"
    big = []
    for ln in p.stdout.splitlines():
        f = ln.split(maxsplit=4)
        if len(f) == 5 and f[1] == "blob" and f[3].isdigit() and int(f[3]) > MAX_TRACKED_MB * 2**20:
            big.append(f"{f[4]} ({int(f[3]) / 2**20:.0f}MB)")
    if big:
        return FAIL, f"{len(big)} tracked blob(s) over {MAX_TRACKED_MB}MB: {', '.join(big[:4])}"
    return PASS, f"no tracked blob over {MAX_TRACKED_MB}MB"


def _broken_blob():
    """A real blob through real git plumbing -- a synthesised ls-tree line shares the
    check's own assumptions."""

    d = _tmp_repo()
    big = os.path.join(d, "big.jsonl")
    with open(big, "wb") as f:
        f.write(b"x" * ((MAX_TRACKED_MB + 1) * 2**20))
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-f", "big.jsonl"],
        ["git", "commit", "-qm", "big"],
    ):
        subprocess.run(cmd, cwd=d, check=True, capture_output=True)
    return d


def merge_reverted_content(root, merge_sha="HEAD", max_files=40):
    """Definitions the merge base had that the merge result no longer has.

    Returns [(path, name, side_taken, already_dropped_at)]. `already_dropped_at` is None
    for the shape the caller must FAIL on, and a commit sha for the shape it must only
    WARN on -- see the asymmetry below.

    The complement of merge_took_one_side, and the more dangerous shape. That
    function only examines paths BOTH parents changed since the base, on the
    reasoning that taking an untouched side whole IS the merge. That reasoning is
    wrong whenever the untouched side is behind the base: it carries an older copy
    forward and silently reverts the other side's work. 21da619 did exactly that to
    _selftest_gpu_descendants -- base had it, ours had it, the merged side had never
    seen it, and main lost a test while keeping the function it tests.

    A side that never had the content is not deleting it. Deliberate deletion is
    tested operationally (fb's rule): the removal counts as intended only when the
    removing side's OWN commits, merge-base..parent, contain a diff that removes the
    name. Otherwise the content is simply older than that branch.

    THE TWO SHAPES ARE NOT SYMMETRIC, and treating them alike produced 117 reds in one
    day (fb's ruling, 2026-09-02, after de measured that no single flag fixes it):

      name in ours, gone from the result -- THIS merge lost it. FAIL. 21da619 is here:
      base had it, ours had it, theirs had never seen it, and the resolution took theirs.

      name absent from ours, present in theirs and in the base -- ours had ALREADY
      dropped it before this merge ran, and this merge changed nothing about that. The
      loss belongs to whichever merge dropped it, and that merge's own check owned it.
      WARN naming that commit, because every later merge from an older branch inherits
      the same absence and calling inheritance a defect is what made the check unusable.
      c8a4578 is here: _built_set was dropped in d5aac3d's conflict resolution, which
      states its reason, and 117 merges after it inherited the red.

    Why plain `-S` still gates the FAIL branch and no flag was added: `git log -S` shows
    no diff for a merge commit, so a deletion made inside a resolution is invisible to
    it. `-m` sees that but then also matches a merge which merely CARRIED a deletion in
    from one side, which silences 21da619 -- a false PASS on the founding case, with
    merge_took_one_side returning [] there too. `--diff-merges=first-parent` finds
    neither, because d5aac3d took parent1 whole and its first-parent diff is empty.
    d5aac3d and ef27df0 are structurally identical at the graph level, so no predicate
    over the graph separates deliberate from accidental (de, MEASURED). The first-parent
    split above sidesteps the question instead of answering it: it asks which merge lost
    the definition, which is decidable, rather than whether someone meant to.

    Scope: top-level `def NAME(` in tracked .py files. Definitions are what a merge
    can silently revert without breaking an import, and a name is cheap to test for.
    A body gutted while the signature survives is not caught here."""
    def git(*args):
        r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else ""

    parents = git("rev-list", "--parents", "-n", "1", merge_sha).split()
    if len(parents) < 3:
        return []
    m, ours, theirs = parents[0], parents[1], parents[2]
    base = git("merge-base", ours, theirs).strip()
    if not base:
        return []
    defs = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(", re.M)
    out = []
    changed = [p for p in set(git("diff", "--name-only", base, m).split()) if p.endswith(".py")]
    for path in sorted(changed)[:max_files]:
        base_src = git("show", f"{base}:{path}")
        merged_src = git("show", f"{m}:{path}")
        if not base_src or not merged_src:
            continue
        lost = set(defs.findall(base_src)) - set(defs.findall(merged_src))
        for name in sorted(lost):
            # Which side dropped it, and did that side's own history delete it?
            in_ours = name in set(defs.findall(git("show", f"{ours}:{path}")))
            in_theirs = name in set(defs.findall(git("show", f"{theirs}:{path}")))
            if not in_ours and not in_theirs:
                continue  # gone from both parents; the merge did not lose it
            # fb's operational test, applied to the side that DROPPED it rather than
            # the side that kept it. Someone deliberately retiring a function deletes
            # it on their branch; that branch is the one now missing it. A side that
            # merely never had the content deleted nothing, and taking that side whole
            # is the silent revert this function exists to find.
            dropper = theirs if in_ours else ours
            side = "ours" if in_ours else "theirs"  # the side whose copy survived in the base sense
            deleted_deliberately = bool(
                git("log", "--format=%h", "-S", f"def {name}(", f"{base}..{dropper}", "--", path).strip()
            )
            if deleted_deliberately:
                continue
            # Ours already lacked it before this merge: inherited, not lost here. Name
            # the last commit on ours' line that removed it, so the WARN points at the
            # merge that owns the decision instead of at this one. `-m` is right for
            # THIS lookup -- unlike the gate above it is not deciding intent, only
            # reporting where to look, and the removal is usually a resolution.
            if not in_ours:
                # %H, not %h: `git log --format=%h` uses git's AUTO-SCALING abbreviation,
                # which lengthens as the object count grows (core.abbrev is unset here).
                # d5aac3d printed as 7 chars when de-22 landed and as 8 once the repo
                # crossed the next threshold, so this selftest's `at == "d5aac3d"` went
                # red on a tree where nothing about the merge had changed -- a permanent
                # red, which is the same as no signal. A caller comparing an identity
                # needs one that does not depend on repository size, so the full sha is
                # returned and every reader abbreviates for itself (de-35).
                at = git("log", "--format=%H", "-m", "-S", f"def {name}(",
                         f"{base}..{ours}", "--", path).split()
                out.append((path, name, side, at[0] if at else "an unfound commit"))
            else:
                out.append((path, name, side, None))
    return out


def merge_took_one_side(root, merge_sha="HEAD"):
    """Files a merge resolved by taking one parent WHOLE, when both parents had
    changed them. Returns [(path, "ours"|"theirs", n_lost_commits)].

    This is the real class, found by de 2026-08-31 while resolving a file two
    sessions had edited: `git checkout --theirs <f>` took main's whole file and
    dropped four of their changes; `git checkout HEAD <f>` restored theirs and
    dropped my _gpu_descendants. Neither printed anything. A file-level resolution
    of a file both sides edited discards one side silently, and no later check can
    fail on code that is no longer there.

    The test is exact rather than heuristic: for each path both parents changed
    since the merge base, compare the merge's blob to each parent's. Byte-identical
    to one parent means the other parent's work on that file is gone. A genuine
    3-way merge blends and matches neither.

    Ceiling: a path only one side changed is not examined, correctly -- taking that
    side whole IS the merge. And a resolution that happens to reproduce one side
    byte-for-byte while intending to is indistinguishable from one that discarded
    the other; that case is rare and worth a human look, which is what FAIL asks for.

    Measured false-positive rate on real history (e1, 2026-08-31): over 93 merges in
    one day it reports 2. One (350210e) lists seven paths with 0 commits lost -- the
    other side had no commits touching them, so nothing was discarded. The other
    (583a54a) is a true "one side taken whole" and the file it took does drop a fact,
    eff.dynamo_recompile_from_dynamic_cu, but that fact had been deliberately
    retracted in 432c987 as a wrong measurement. So the check's one substantive hit
    on a real day is a correct detection of a correct outcome. That is the expected
    shape: it flags contested whole-file resolutions for a human, it does not know
    which were intended.

    Not this function's business (e1, 2026-08-31): a commit that never reached the
    branch being merged was never in the input, so its absence is not a drop. I
    reported one of those as a lost commit and was wrong. `git branch --contains
    <sha>` distinguishes the two in two seconds and comes first."""
    def git(*args):
        r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else ""

    parents = git("rev-list", "--parents", "-n", "1", merge_sha).split()
    if len(parents) < 3:
        return []  # not a merge
    m, ours, theirs = parents[0], parents[1], parents[2]
    base = git("merge-base", ours, theirs).strip()
    if not base:
        return []
    both = (set(git("diff", "--name-only", base, ours).split())
            & set(git("diff", "--name-only", base, theirs).split()))

    def _blobs_in(side, path):
        """Every blob this side's history has ever held at `path`.

        One `cat-file --batch-check` rather than one `rev-parse` per commit. The loop
        version cost 11.84s on data/pod_head_manifest.txt -- 880 commits, because the
        hook rewrites that file on every commit -- and timed the whole check out on its
        5s deadline, i.e. turned it into a gate that never ran (de, 2026-09-01). Batched:
        0.29s for the same 879 blobs. The cost was per-contested-file, not per-merge:
        five other merges the same night were 0.16s because none had a contested path."""
        shas = git("rev-list", side, "--", path).split()
        if not shas:
            return set()
        r = subprocess.run(["git", "-C", root, "cat-file", "--batch-check"],
                           input="".join(f"{s}:{path}\n" for s in shas),
                           capture_output=True, text=True)
        return {ln.split()[0] for ln in r.stdout.splitlines() if " blob " in ln}

    out = []
    for path in sorted(both):
        mv = git("rev-parse", f"{m}:{path}").strip()
        a = git("rev-parse", f"{ours}:{path}").strip()
        b = git("rev-parse", f"{theirs}:{path}").strip()
        if not mv or a == b:
            continue
        if mv == a:
            taken, other, side = ours, b, "ours"
        elif mv == b:
            taken, other, side = theirs, a, "theirs"
        else:
            continue
        # CHERRY-PICK, not a drop. If the side we took has itself HELD the other
        # side's exact blob at some point in its history, that content was never
        # discarded -- it was received and then built upon. This is what a cherry-pick
        # across worktrees produces, and the commit count cannot see it: the picked
        # commit has a different sha, so `rev-list ours..theirs` still counts it.
        #
        # de, 2026-09-01, on this check's own author: tilerl cherry-picked my 5927ed6
        # into main as b8cae37 to unblock a launch. The blobs are byte-identical
        # (adb4224 both), my branch then added a comment on top, and the merge
        # correctly took my newer file -- while this check reported "2 commit(s) from
        # the other side lost". Nothing was lost. A permanent red is the same as no
        # signal, and a false positive on the normal way an urgent fix reaches main
        # would have been read past within a day.
        #
        # Exact blob equality, so it cannot excuse a real drop: a resolution that
        # discarded work produces a blob the taken side never held.
        if other and other in _blobs_in(taken, path):
            continue
        lost = len(git("rev-list", f"{taken}..{ours if side == 'theirs' else theirs}",
                       "--", path).split())
        out.append((path, side, lost))
    return out


def _is_fresh_render(root, staged_blob):
    """Is the staged EXPERIMENTS.md exactly what exp.py render produces from the merged
    ledger?

    EXPERIMENTS.md is derived: its content is a function of runs/experiments.jsonl, which
    .gitattributes already union-merges. So a merge never RESOLVES it -- whichever side is
    taken whole is stale by construction, and the fix is to re-render, not to splice lines.
    merge_complete cannot see that: it compares lines, and a row whose newer side carries a
    finding the older side lacked reads as content lost (2026-09-03, the 3b merge -- three
    rows de-38 had amended with 'commit was c304b37, which names no object in this
    repository' read as three rows dropped).

    Exempting the path outright would be the wrong fix, and the repo has the incident for
    it: a derived artifact that stays valid after its source changes with nothing raising.
    So the exemption IS the freshness test -- stale gets no pass."""
    import shutil
    import tempfile

    led = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.isfile(led):
        return False
    r = subprocess.run(["git", "-C", root, "cat-file", "-p", staged_blob],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False
    sys.path.insert(0, os.path.join(root, "scripts"))
    try:
        import exp as _exp
    except Exception:
        return False
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "runs"))
        shutil.copy(led, os.path.join(t, "runs", "experiments.jsonl"))
        old = _exp.ROOT
        try:
            _exp.set_root(t)
            _exp.render()
            want = open(os.path.join(t, "EXPERIMENTS.md"), encoding="utf-8").read()
        except Exception:
            return False
        finally:
            _exp.set_root(old)
    return want == r.stdout


def _content_restored(root, base, losing, path, staged_blob):
    """Is the losing side's own new content back in the staged blob?

    The escape hatch used to ask whether the staged blob DIFFERS from the parent that
    was taken whole. Differing is not restoring: staging the offending content plus one
    comment changes the blob and restores nothing, and the hatch called that a fix.
    Measured on the check's own broken world, 2026-09-01 -- a comment line above the
    resolution turned FAIL into "1 contested file(s) re-resolved", a false GO reached by
    a legal edit. Same shape as the day's other holes: an honest record plus a criterion
    that does not read it.

    So the question becomes the one fb named: is the content still there. Scope is one
    already-contested path, not the whole merge, which is why line granularity is safe
    here and wrong for merge_reverted_content -- the lost marker in the real incident
    was a line inside a surviving function, and a def-level test cannot see it.

    Non-empty lines that are not comments, compared as a set: reindentation and
    reordering during a hand resolution are not losses, and a moved line is still
    present. A line the losing side introduced and the fix rewrote rather than copied
    reads as missing, which errs toward FAIL and asks for a human -- the direction this
    check is supposed to fail in."""
    def show(rev):
        r = subprocess.run(["git", "-C", root, "show", f"{rev}:{path}"],
                           capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else ""

    def code(src):
        return {ln.strip() for ln in src.splitlines()
                if ln.strip() and not ln.strip().startswith("#")}

    r = subprocess.run(["git", "-C", root, "cat-file", "-p", staged_blob],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False
    added = code(show(losing)) - code(show(base))
    if not added:
        return True  # the losing side added nothing here; there is nothing to restore
    return not (added - code(r.stdout))


def check_merge_complete(root):
    """A merge must not resolve a contested file by discarding one side.

    Judges the STAGED blob when one is staged for a contested path, not HEAD's. A
    bad merge is refused; the commit that FIXES it must not be. Without this the
    check deadlocks: the amend re-reads HEAD, HEAD is still the bad merge, and
    --no-verify becomes the only way out -- which trains people to bypass the check
    at exactly the moment it is working (de + fb, 2026-08-31, first real catch)."""
    if not os.path.exists(os.path.join(root, ".git")):
        return SKIP, "no .git (pod or partial checkout)"
    r = subprocess.run(["git", "-C", root, "rev-list", "--parents", "-n", "1", "HEAD"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return SKIP, "cannot read HEAD"
    if len(r.stdout.split()) < 3:
        return PASS, "HEAD is not a merge (1 commit examined)"
    took = merge_took_one_side(root)
    n_took_raw = len(took)
    # A ledger is union-merged by .gitattributes and legitimately equals one side
    # when only that side appended; that is the merge driver working, not a drop.
    took = [t for t in took if not t[0].endswith(".jsonl")
            and t[0] != "data/pod_head_manifest.txt"]
    # EXPERIMENTS.md is rendered from that union-merged ledger. Exempt it only when the
    # staged blob IS the render; a stale one still FAILs.
    took = [t for t in took
            if t[0] != "EXPERIMENTS.md"
            or not _is_fresh_render(root, subprocess.run(
                ["git", "-C", root, "rev-parse", ":EXPERIMENTS.md"],
                capture_output=True, text=True).stdout.strip())]
    # 0 commits lost means the other side had no commits touching that path: the file
    # matches one parent because only one parent's history reached it, not because a
    # resolution discarded anything. Seven of the nine hits over one day's 93 merges
    # were this shape.
    took = [t for t in took if t[2] > 0]
    # A path whose STAGED blob RESTORES the losing side's content is being fixed right
    # now. Judge what is about to be committed, not what was.
    parents = subprocess.run(["git", "-C", root, "rev-list", "--parents", "-n", "1", "HEAD"],
                             capture_output=True, text=True).stdout.split()
    ours, theirs = (parents[1], parents[2]) if len(parents) >= 3 else (None, None)
    base = subprocess.run(["git", "-C", root, "merge-base", ours, theirs],
                          capture_output=True, text=True).stdout.strip() if ours else ""
    fixed = []
    for path, side, n in list(took):
        staged = subprocess.run(["git", "-C", root, "rev-parse", f":{path}"],
                                capture_output=True, text=True).stdout.strip()
        if not staged:
            continue  # nothing staged for it; the merge's own blob stands
        losing = theirs if side == "ours" else ours
        if not base or not _content_restored(root, base, losing, path, staged):
            continue
        took.remove((path, side, n))
        fixed.append(path)
    if took:
        return FAIL, (
            f"{len(took)} contested file(s) resolved by taking one side whole: "
            + "; ".join(f"{p} == {side} ({n} commit(s) from the other side lost)"
                        for p, side, n in took[:3])
            + ". Re-resolve by hand and grep a marker from each side before committing."
        )
    if fixed:
        return PASS, f"{len(fixed)} contested file(s) re-resolved in the staged tree: {', '.join(fixed[:3])}"
    # The other shape: a side that never had the content carries an older copy forward
    # and silently reverts the other side. merge_took_one_side cannot see it, because
    # it only examines files BOTH parents changed (21da619, 2026-08-31).
    reverted = merge_reverted_content(root)
    lost_here = [r for r in reverted if r[3] is None]
    inherited = [r for r in reverted if r[3] is not None]
    if lost_here:
        return FAIL, (
            f"{len(lost_here)} definition(s) present in the merge base and in ours, gone "
            "from the result, with no side deleting them: "
            + "; ".join(f"{name} in {path}" for path, name, _, _ in lost_here[:3])
            + ". A side that never had the content did not delete it -- restore from the base."
        )
    if inherited:
        # Ours already lacked these before this merge, so this merge lost nothing. The
        # merge that dropped it owned the decision, and its own run of this check saw it.
        # FAILing on inheritance is what put the same red on 117 merges (fb, 2026-09-02).
        return WARN, (
            f"{len(inherited)} definition(s) the base had and ours had ALREADY dropped "
            "before this merge: "
            + "; ".join(f"{name} in {path}, last removed on ours by {at} (git log -m -S)"
                        for path, name, _, at in inherited[:3])
            + ". This merge did not change ours' state -- check whether that commit meant it."
        )
    # The count for the PASS line comes from the scan already done above, not a second
    # one. Recomputing it cost 12.57s of the check's 25.37s and timed the check out on
    # its 5s deadline (de, 2026-09-01) -- and the second call answered a different
    # question anyway: `took` has been filtered and mutated by then, so the two numbers
    # were never the same. Measure once, report what you measured.
    contested = n_took_raw
    n_both = len(set(subprocess.run(
        ["git", "-C", root, "diff", "--name-only", "HEAD^1", "HEAD"],
        capture_output=True, text=True).stdout.split()))
    # BOTH SIDES' COUNTS, because one of them is legitimately zero. `HEAD^1..HEAD` is what
    # the merge brought into OURS, and it is 0 whenever ours already contained everything
    # the other side had that survived -- a fast-forward-shaped merge, or one whose only
    # contested path resolved to a DELETION on our side (a0e401e0: 0 against parent 1, 6
    # against parent 2). The selftest's own vacuity rule then reads "0 file(s) changed,
    # 0 contested" as a scan that examined nothing, and refuses a correct PASS. Reporting
    # the second parent's count as well makes the difference visible: 0 and 0 is a merge
    # with no content anywhere, 0 and N is a merge whose changes all came from ours.
    n_theirs = len(set(subprocess.run(
        ["git", "-C", root, "diff", "--name-only", "HEAD^2", "HEAD"],
        capture_output=True, text=True).stdout.split()))
    return PASS, (f"{n_both} file(s) changed by the merge into ours, {n_theirs} vs the other "
                  f"parent, {contested} contested file(s) taken whole")


def _broken_merge_complete():
    """A real two-branch repo where the merge was resolved with `git checkout --theirs`.

    de's actual mistake, 2026-08-31: resolving a file two sessions had edited, they
    ran `git checkout --theirs scripts/harness.py`, which took main's whole file and
    dropped four of their own changes. Then `git checkout HEAD` on the same file
    restored theirs and dropped mine. Neither printed anything. This world is that
    sequence, minimised."""
    d = _tmp_repo()
    sh = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)
    sh("init", "-q")
    sh("config", "user.email", "t@t"); sh("config", "user.name", "t")
    # A repo-real path: the selftest's meta-check rejects a world built entirely from
    # invented paths, and rightly -- a world hand-written from the check's own
    # assumptions cannot show the check works on the tree it will actually run against.
    rel = os.path.join("scripts", "loader.py")
    src = os.path.join(d, rel)
    os.makedirs(os.path.dirname(src), exist_ok=True)
    open(src, "w").write("def f():\n    return 1\n")
    sh("add", "-A"); sh("commit", "-qm", "base")
    sh("branch", "other")

    # our side edits the function
    open(src, "w").write("def f():\n    OURS_MARKER = 'kept by us'\n    return 1\n")
    sh("add", "-A"); sh("commit", "-qm", "ours: add OURS_MARKER")

    # their side edits the same function differently
    sh("checkout", "-q", "other")
    open(src, "w").write("def f():\n    THEIRS_MARKER = 'kept by them'\n    return 1\n")
    sh("add", "-A"); sh("commit", "-qm", "theirs: add THEIRS_MARKER")

    # the resolution that loses a side silently
    sh("checkout", "-q", "master") if sh("rev-parse", "--verify", "-q", "master").returncode == 0 else sh("checkout", "-q", "main")
    sh("merge", "--no-commit", "other")
    sh("checkout", "--theirs", rel)
    sh("add", rel)
    sh("commit", "-qm", "merge other (resolved --theirs)")
    txt = open(src).read()
    assert "THEIRS_MARKER" in txt and "OURS_MARKER" not in txt, \
        f"the broken world must have lost our side: {txt!r}"
    return d


def _bad_help_strings(src):
    """argparse help strings in src that argparse cannot format.

    argparse %-formats every help string against a params dict, so a literal
    percent must be doubled. "55.8% of SFT generations" makes "% o" a %o
    conversion and --help dies with "TypeError: %o format: an integer is
    required, not dict" (eval/code_zh.py, ae2063f, live 25 hours).

    Parsed with ast, not a regex: these help strings are written as implicit
    concatenation across three lines, and a regex that misses the continuation
    reads only the first fragment -- which is how my first version PASSED its
    own broken world. ast.literal_eval joins them the way Python does.

    Tested by formatting, not by pattern: `%s` is legal, `%%` is legal, `50%`
    raises ValueError and `100% done` raises TypeError. Only argparse's own
    formatting knows which is which, so ask it."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            try:
                text = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError, TypeError):
                continue
            if not isinstance(text, str) or "%" not in text:
                continue
            try:
                text % {"prog": "p", "default": None}
            except (TypeError, ValueError) as e:
                bad.append((text[:60], type(e).__name__, str(e)[:40]))
    return bad


def check_entrypoint_help(root):
    """An argparse help string must be formattable, or --help dies.

    Static, not by running --help: every CLI here imports torch, which costs
    ~9.7 s each and 213 s for the 22 of them, against a 5 s per-check budget.
    Running them also cannot tell a malformed help string from a dependency
    missing on this machine -- my first version reported 7 failures, of which
    the real one was 1 and the rest were no triton on macOS and the like.

    Ceiling: this reads source, so it catches the formatting class and nothing
    else. A CLI broken by its imports still passes. It is the cheapest test for
    the defect that actually happened, not a liveness test for entrypoints."""
    import glob

    checked = 0
    bad = []
    # THE REPO-ROOT ENTRY POINTS WERE NOT SCANNED, WHICH IS WHERE THIS DEFECT LIVED LONGEST.
    # This loop covered five subdirectories and no root file, so train.py -- the entry point
    # every launch goes through -- was outside it. Measured 2026-09-03: train.py:1963 carried
    # "weights 14% off against fp64 truth" from 169da865, so `train.py --help` had been dead
    # with the exact TypeError this check names, and the check passed the whole time. A guard
    # that skips the most-used file in the repo reports on the files that matter least.
    roots = sorted(glob.glob(os.path.join(root, "*.py")))
    for d in ("eval", "scripts", "datagen", "probes", "algorithms"):
        roots += sorted(glob.glob(os.path.join(root, d, "*.py")))
    for path in roots:
        try:
            src = open(path, encoding="utf-8").read()
        except OSError:
            continue
        if "argparse" not in src:
            continue
        checked += 1
        for text, exc, msg in _bad_help_strings(src):
            bad.append(f"{os.path.relpath(path, root)}: {exc}: {text!r}")
    if not checked:
        return SKIP, "no argparse entrypoints found"
    if bad:
        return FAIL, (
            f"{len(bad)} unformattable argparse help string(s); --help will die: "
            + "; ".join(bad[:3]) + ". Double the literal percent (%% not %)."
        )
    return PASS, f"{checked} argparse entrypoints have formattable help strings"


def _broken_entrypoint_help():
    """The real defect, verbatim from ae2063f."""
    d = _tmp_repo()
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    with open(os.path.join(d, "eval", "code_zh.py"), "w", encoding="utf-8") as f:
        f.write(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--temperature', type=float, default=0.0,\n"
            "               help='0.0 = greedy. Raised only to test whether '\n"
            "                    'the loop (55.8% of SFT generations repeat an 8-gram) '\n"
            "                    'is produced by greedy decoding.')\n"
            "p.parse_args()\n"
        )
    return d


def _exp_events(root, folded=True):
    """runs/experiments.jsonl. Folded by (name, started): the last event for a run wins.

    The ledger is an EVENT LOG, not a table -- exp.py `done` appends a closing event
    carrying the start row's `started` rather than rewriting the start row, so that a
    union merge of two branches cannot produce two half-closed runs. A reader that
    walks raw lines and looks at `status` therefore sees every closed run as still
    running, forever.

    That is not hypothetical: p02_fp32m_s0 was correctly closed on 2026-09-01 with an
    appended event on the exact (name, started) pair, and check_no_stale_running kept
    failing on it, because the check re-implemented the read without the fold. exp.py
    has folded since it was written; four readers here had not."""
    p = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return None
    evs = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            evs.append(json.loads(line))
        except Exception:
            continue  # a line another session is mid-append
    if not folded:
        return evs
    return _exp_fold(evs)


#: A `running` row older than this is a job that died without exp.py done. Named rather
#: than inlined so a fixture can derive from it: _selftest_exp_fold hardcoded a date that
#: was recent when written and aged past 24h as the clock moved, turning the selftest into
#: a permanent red (e1 found it, 2026-09-01). Same defect as _broken_dirty_aged's, which
#: was fixed the same day by deriving from _AGE_HOURS -- a constant standing in for a live
#: threshold is a fixture that expires.
#:
#: Swept the other eight hardcoded timestamps in this file after the fix. Only this one
#: was wrong, and the rule that separates them is the DIRECTION the fixture leans:
#:   - a fixture that backdates to EXCEED a threshold (_broken_no_stale_running's
#:     2020-01-01, _broken_no_ghost_running's 2026-08-29) only gets safer as time passes
#:   - a fixture that must stay UNDER one expires, silently, on a date nobody chose
#:   - a fixture whose check never reads the clock (_selftest_monitor_settled: settled()
#:     tests status, not age) cannot expire at all
#: So the audit question is not "is this date hardcoded" but "which side of the threshold
#: does it need to be on, and does time carry it across".
_STALE_RUNNING_H = 24


def merge_drops(root, rev="HEAD"):
    """[(parent_sha, path)] for paths a parent of `rev` held that `rev` lacks unlisted.

    THE PREDICATE, in one place, because merge_main.sh needs the paths as data and the check
    needs them as a sentence -- two copies of a git predicate is how the two answers come to
    disagree. Returns [] when `rev` is not a merge.
    """
    r = subprocess.run(["git", "-C", root, "rev-list", "--parents", "-n", "1", rev],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode:
        return None
    parts = r.stdout.split()
    if len(parts) < 3:
        return []
    m, parents = parts[0], parts[1:]

    def paths(x):
        # stdin=DEVNULL on every git call here: b0's own first attempt piped rev-list
        # into a `while read` loop whose body ran `git cat-file -e`, which consumed the loop's
        # stdin, so it printed nothing and reported "no transitions" for a path already proven
        # deleted. A silent empty result is the failure mode this whole predicate exists for.
        out = subprocess.run(["git", "-C", root, "ls-tree", "-r", "--name-only", x],
                             capture_output=True, text=True, stdin=subprocess.DEVNULL)
        return set(out.stdout.split("\n")) - {""}

    here = paths(m)
    lost = []
    for p in parents:
        gone = sorted(paths(p) - here)
        if not gone:
            continue
        # Which of those a NON-MERGE commit on the merge's side actually deleted. `--no-merges`
        # is the whole discrimination and it took two wrong versions to find (de, 2026-09-04).
        # A merge commit that drops a path DOES record `D` against the parent that held it --
        # measured on d9c9614f, whose parent2 acbdbdd1 added the file: `git diff --diff-filter=D
        # acbdbdd1 d9c9614f` prints the path, and `git log -m --diff-filter=D` lists all seven
        # merges. So "some commit records a D" is true of the silent drops themselves and rules
        # every one of them deliberate. What no silent drop has is a deletion someone WROTE: a
        # single-parent commit removing the path, which is what `git rm` produces. The
        # distinction is authorship, not the presence of a D record -- and measured across this
        # repo, `--all --no-merges --diff-filter=D` over the lost file returns NOTHING, while
        # `-m --diff-filter=D` returns all seven merges.
        #
        # One `git log` per parent, not per path: a merge losing hundreds of paths would
        # otherwise make this the slowest thing in the suite.
        deleted = subprocess.run(
            ["git", "-C", root, "log", "--format=", "--no-merges", "--diff-filter=D",
             "--name-only", m, f"^{p}", "--"] + gone,
            capture_output=True, text=True, stdin=subprocess.DEVNULL)
        intended = set(deleted.stdout.split("\n")) - {""}
        lost += [(p, path) for path in gone if path not in intended]
    return lost


def check_merge_keeps_parent_paths(root):
    """HEAD, if it is a merge, holds every path EITHER parent held, unless someone deleted it.

    runs/redaction_handread_v14.tsv (44's v14 hand-read, 51 lines, committed acbdbdd1) left main
    with no written deletion anywhere, and was restored four times because each restore was
    dropped again: 6f8361ec, 8f13f5a8, 72ba3c92, 8499a3cf. Seven merges dropped it -- d9c9614f
    (the drop site), d42e766c, 3f3568ad, 26e060af, 0c787961, 74b67ca3, c2cc8bba.

    TWO NAIVE PREDICATES MISS IT, both verified against the real repository 2026-09-04:

      - first-parent only: 74b67ca3's parent1 bbf1e354 already lacked the path, so
        `git diff --diff-filter=D 74b67ca3^1 74b67ca3` is EMPTY. The loss is against parent2.
      - "the second parent added it": c2cc8bba's first parent held it instead (6e).

    And the shape that looks decisive is not: a merge that drops a path DOES record a `D`
    against the parent that held it, so "no commit records a deletion" is false of every one of
    these seven. What none of them has is a deletion someone WROTE -- `git log --all --no-merges
    --diff-filter=D` over the path returns nothing at all.

    Discrimination, measured over the 40 most recent merges on main: 33 clean, 7 flagged, and
    the 7 are exactly the list above, each naming that one file. A guard that flagged every
    merge would pass a fixture of failing cases; this one does not flag 33 of 40.

    WARN, not FAIL. An inherited absence -- ours already lacked the path before this merge --
    belongs to the merge that dropped it, and FAILing on inheritance is what put one red on 117
    merges (fb, 2026-09-02, on merge_complete's sibling). The seven above are all reported
    because each one really did produce a tree without the file.
    """
    lost = merge_drops(root)
    if lost is None:
        return SKIP, "cannot read HEAD"
    if not lost:
        r = subprocess.run(["git", "-C", root, "rev-list", "--parents", "-n", "1", "HEAD"],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        n_par = max(len(r.stdout.split()) - 1, 0)
        if n_par < 2:
            return PASS, "HEAD is not a merge (1 commit examined)"
        n = len(subprocess.run(["git", "-C", root, "ls-tree", "-r", "--name-only", "HEAD"],
                               capture_output=True, text=True,
                               stdin=subprocess.DEVNULL).stdout.split("\n")) - 1
        return PASS, (f"{n} path(s) in the merge; every path each of {n_par} parent(s) held is "
                      f"present or was deliberately deleted")
    return WARN, (
        f"{len(lost)} path(s) a parent held are absent from this merge with nobody deleting "
        "them: "
        + "; ".join(f"{path} (held by {p[:8]})" for p, path in lost[:3])
        + ". Restore: git checkout <parent> -- <path>. A first-parent D-entry walk reports this "
          "clean, which is how one file was lost seven times."
    )


def _broken_merge_keeps_parent_paths():
    """THE REAL HISTORY, at the real merge that dropped the file. Not a synthetic sequence.

    A synthetic world was written first and it certified nothing: `git rm --cached` + unlink,
    committed, records a `D` in a single-parent commit, so the fixed check correctly rules it
    deliberate and the world PASSes with its "defect" in place. Measured, both directions -- the
    deliberate-deletion world and the supposed silent-drop world came back identical.

    What separates the real drops from a deliberate deletion is that NOBODY EVER WROTE ONE:
    `git log --all --no-merges --diff-filter=D -- runs/redaction_handread_v14.tsv` returns
    nothing across every ref, while `git log -m --diff-filter=D` lists all seven merges. That
    property cannot be built by running git commands that delete a file, because every such
    command records the deletion. It has to be taken from the history that has it.

    So the world is a worktree of this repository at d9c9614f -- the drop site, whose second
    parent acbdbdd1 added the file and whose tree lacks it. The check reads HEAD, so a detached
    checkout at that commit IS the failing case, with the real object database behind it.
    """
    import shutil as _sh
    import tempfile as _tf

    if subprocess.run(["git", "-C", ROOT, "cat-file", "-e", "d9c9614f^{commit}"],
                      capture_output=True, stdin=subprocess.DEVNULL).returncode:
        raise SelftestSkip("d9c9614f is not in this repository; the drop site is unavailable")
    d = _tf.mkdtemp(prefix="merge_drop_")
    # A linked worktree would register itself in the shared .git and need removing; a clone of
    # the local repo is self-contained and cheap (--no-checkout, then a detached read).
    r = subprocess.run(["git", "clone", "-q", "--no-checkout", "--no-local", "--shared",
                        ROOT, os.path.join(d, "r")],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")})
    if r.returncode:
        _sh.rmtree(d, ignore_errors=True)
        raise SelftestSkip(f"cannot clone this repository for the world: {r.stderr[:80]}")
    w = os.path.join(d, "r")
    subprocess.run(["git", "-C", w, "checkout", "-q", "--detach", "d9c9614f"],
                   capture_output=True, stdin=subprocess.DEVNULL)
    return w


def check_no_stale_running(root):
    evs = _exp_events(root)  # folded: an appended close must clear its start row
    if evs is None:
        return SKIP, "runs/experiments.jsonl not present"
    rows = []
    for r in evs:
        if r.get("status") != "running":
            continue
        # The field is `started`, in exp.py's %Y-%m-%d %H:%M format. An unreadable date is
        # a FAIL: a check that cannot see its subject must not report on it.
        try:
            t = time.mktime(time.strptime(str(r.get("started", "")), "%Y-%m-%d %H:%M"))
        except Exception:
            return FAIL, f"row {r.get('name', '?')!r} has no readable `started`: {r.get('started')!r}"
        age_h = (time.time() - t) / 3600
        if age_h > _STALE_RUNNING_H:
            rows.append(f"{r.get('name', '?')} {age_h:.0f}h")
    if rows:
        return FAIL, f"{len(rows)} killed mid-run and never closed: {', '.join(rows[:6])}"
    return PASS, "no run has been 'running' for over a day"


def check_no_ghost_running(root):
    # no_stale_running's blind spot: a run that FINISHED but was never recorded stays
    # 'running' for up to 24h. On the pod, a running row older than 2h with no live
    # process is a ghost -- close it with exp.py done. Pod-only: processes live there.
    if not pod_drift.is_pod(root):
        return SKIP, "dev checkout; process state lives on the pod"
    p = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return SKIP, "runs/experiments.jsonl not present"
    import subprocess

    # FOLD FIRST. experiments.jsonl is append-only: one run emits many rows and the
    # readers fold by (name, started), a close beating a later start. Reading raw lines
    # asked pgrep 54 times where 13 were needed (measured on the pod, 0.083s each = the
    # whole 4.1s), and reported the same run as several ghosts because its history has
    # several rows. So the count was wrong as well as slow, and raising the deadline
    # would have preserved both. Through _exp_fold since e1-18: this fold was
    # position-based, so a start event landing after a close made a finished run look
    # like a ghost -- the check for ghosts inventing one.
    evs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                evs.append(json.loads(line))
            except Exception:
                continue
    ghosts = []
    for r in _exp_fold(evs):
        if r.get("status") != "running":
            continue
        try:
            t = time.mktime(time.strptime(str(r.get("started", "")), "%Y-%m-%d %H:%M"))
        except Exception:
            return FAIL, f"row {r.get('name', '?')!r} has no readable `started`: {r.get('started')!r}"
        if (time.time() - t) / 3600 < 2:
            continue  # grace: a launched run takes time to appear in ps
        name = r.get("name", "")
        if name and not subprocess.run(["pgrep", "-f", name], capture_output=True, text=True).stdout.strip():
            ghosts.append(f"{name} (started {r.get('started')})")
    if ghosts:
        return FAIL, f"running rows with no live process: {', '.join(ghosts[:6])}; close with exp.py done"
    return PASS, "every running row has a live process"


def check_guard_on_path(root):
    """Deleting the guard's call site must show up as a FAIL, not just a raise in CI."""
    src_path = os.path.join(root, "train.py")
    if not os.path.exists(src_path):
        return SKIP, "train.py not present"
    tree = ast.parse(open(src_path, encoding="utf-8").read())
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if fn is None:
        return FAIL, "train.py has no main()"
    called = {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    if "_assert_mix_domains" not in called:
        return FAIL, "main() does not call _assert_mix_domains; run_ddp.sh is unguarded"
    # The retirement marker's teeth. mix_supply stops gating a mix that carries _retired,
    # which is only honest while train.py refuses to start on one; without this assert the
    # marker would be a label that silences a check and permits the run it describes.
    asserts = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
    if not any('"_retired"' in ast.dump(a) or "'_retired'" in ast.dump(a) for a in asserts):
        return FAIL, ("main() does not refuse a mix carrying _retired, so the marker "
                      "silences mix_supply without stopping the run it describes")
    return PASS, "main() calls _assert_mix_domains and refuses a retired mix"


def check_gemm_dims(root):
    """vocab 32773 cost 2.23x on the LM head because nothing checked shapes: it left the logits'
    leading dimension 2-byte aligned and cuBLAS fell back to an SM75 align-1 kernel on a Hopper
    card. Parsed from source rather than imported -- this must not need torch or a GPU.
    Full audit incl. every nn.Linear: scripts/shape_audit.py."""
    src_path = os.path.join(root, "train.py")
    if not os.path.exists(src_path):
        return SKIP, "train.py not present"
    tree = ast.parse(open(src_path, encoding="utf-8").read())
    cfg = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Cfg"), None)
    if cfg is None:
        return FAIL, "train.py has no Cfg"
    dims = {}
    for node in cfg.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("vocab", "d", "ffn_hidden"):
                    if isinstance(node.value.value, int):
                        dims[t.id] = node.value.value
    if not dims:
        return FAIL, "Cfg names none of vocab/d/ffn_hidden as an int literal"
    bad = [f"{k}={v} (%8={v % 8})" for k, v in dims.items() if v % 8]
    if bad:
        return FAIL, f"GEMM dims not 8-aligned: {', '.join(bad)} -- cuBLAS drops to an align-1 kernel"
    bad16 = [f"{k}={v} (%16={v % 16})" for k, v in dims.items() if v % 16]
    if bad16:
        return FAIL, f"GEMM dims not 16-aligned: {', '.join(bad16)} -- _fp8_ok rejects them, the run silently stays bf16"
    return PASS, f"{', '.join(f'{k}={v}' for k, v in sorted(dims.items()))} all 8/16-aligned"


def check_restartability(root):
    """A two-hour job that writes once at the end loses everything when interrupted
    (datagen/train_quality_head.py, killed at 50%, lost 100%). Ratcheted against
    scripts/restartability_baseline.json: only a NEW offender fails. Full report:
    scripts/restartability_audit.py."""
    audit = os.path.join(root, "scripts", "restartability_audit.py")
    base = os.path.join(root, "scripts", "restartability_baseline.json")
    if not os.path.exists(audit):
        return FAIL, "scripts/restartability_audit.py missing -- the check cannot run"
    if not os.path.exists(base):
        return FAIL, "scripts/restartability_baseline.json missing -- every script would read as new"
    out = subprocess.run([sys.executable, audit], cwd=root, capture_output=True, text=True)
    if out.returncode == 0:
        return PASS, out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "no new offenders"
    new = [ln for ln in out.stdout.splitlines() if ln.startswith("[NEW]")]
    return FAIL, "; ".join(new)[:300] or "restartability_audit failed"


def check_corpus_filters_fp(root):
    """A domain must record WHICH filters built it, not only what it contains.

    The gap: PROVENANCE records the Build command, and the same command run before and after a
    filters/ edit produces different corpora. corpus_fp_matches sees that the content changed;
    it cannot say why, and cannot answer 'did this batch go through pass3'. build_corpus.py now
    stamps filters_fp beside the content fingerprint.

    Domains built before the stamp existed carry no filters_fp. That is baselined debt
    (facts/corpus_filters_baseline.json, only shrinks): a no-stamp domain NOT in the baseline
    is new debt and FAILs; a no-stamp domain IN the baseline is reported in gaps. A MISMATCH
    is always a failure: it means the shards predate the filters currently in the tree."""
    sys.path.insert(0, os.path.join(root, "scripts"))
    import corpus_fingerprint as cf

    live = cf.fp_filters(root)
    if live is None:
        return SKIP, "no filters/ directory"
    doms, err = read_mix(os.path.join(root, cfg_default("mix")))
    if err:
        return SKIP, f"mix unreadable ({err}); mix checks own that"
    corpus = os.path.join(root, "data", "corpus")
    present = [d for d in doms if os.path.isdir(os.path.join(corpus, d))]
    if not present:
        # No mix-domain corpus on this machine. data/corpus/sample/ is not a mix domain --
        # it ships with the checkout and was never a build_corpus.py product. A machine that
        # never built corpus has nothing to verify; mix_shards_present owns "corpus vanished
        # on a GPU box".
        return SKIP, "no mix-domain corpus on this machine"
    baseline_path = os.path.join(root, CORPUS_FILTERS_BASELINE)
    baseline = json.load(open(baseline_path, encoding="utf-8")) if os.path.exists(baseline_path) else {}
    stale, new_unstamped, baselined, ok = [], [], [], 0
    for dom in present:
        stats = os.path.join(corpus, dom, "build_corpus_stats.json")
        got = None
        if os.path.isfile(stats):
            with open(stats, encoding="utf-8") as f:
                got = json.load(f).get("filters_fp")
        if got is None:
            if dom in baseline:
                baselined.append(dom)
            else:
                new_unstamped.append(dom)
        elif got != live:
            # A ruled mismatch is forgiven only when the baseline names THIS exact pair and
            # cites the fact that measured it. Keyed on both fingerprints, so the next
            # filters edit produces a new pair and a new FAIL: this forgives one known
            # state, never mismatches in general. The check's rule that a mismatch is
            # always a failure held right up to a mismatch nobody could remove -- the
            # shards were built with the pod's weaker filters and rebuilding nine domains
            # buys 0.25% -- and a permanent red is the same as no signal.
            ruled = str(baseline.get(dom, ""))
            if got in ruled and live in ruled and "facts/" in ruled:
                baselined.append(f"{dom} (ruled mismatch)")
            else:
                stale.append(f"{dom} built with filters {got}, tree is {live}")
        else:
            ok += 1
    if stale:
        return FAIL, "; ".join(stale)
    if new_unstamped:
        return FAIL, (
            f"{len(new_unstamped)} domain(s) have no filters_fp and are not in the baseline "
            f"({', '.join(new_unstamped)}) -- rebuild to stamp, or register in {CORPUS_FILTERS_BASELINE}"
        )
    if ok == 0 and not baselined:
        return FAIL, f"0/{len(present)} mix domain(s) match filters {live}"
    note = ""
    if baselined:
        note = (f"; BASELINED debt: {len(baselined)} domain(s) built before filters_fp existed "
                f"({', '.join(baselined)}) -- rebuild to stamp and shrink the baseline")
    return PASS, f"{ok}/{len(present)} domain(s) match filters {live}{note}"


def _broken_corpus_filters_fp():
    """Two failure modes: a stale stamp (mismatch with live filters) and a no-stamp
    domain that is NOT in the baseline (new debt). Both must FAIL."""
    d = _tmp_repo(mix_obj={"domains": {"web_hq": 1.0, "en": 1.0}})
    os.makedirs(os.path.join(d, "filters"), exist_ok=True)
    with open(os.path.join(d, "filters", "pass1_garbage.py"), "w") as fh:
        fh.write("# a filter\n")
    dom = os.path.join(d, "data", "corpus", "web_hq")
    os.makedirs(dom, exist_ok=True)
    with open(os.path.join(dom, "build_corpus_stats.json"), "w") as fh:
        json.dump({"fingerprint": "deadbeef", "filters_fp": "0000000000000000"}, fh)
    # en: no stamp at all, no baseline file in this world -> new unstamped domain
    os.makedirs(os.path.join(d, "data", "corpus", "en"), exist_ok=True)
    return d


def check_score_input_fresh(root):
    """A score must record which corpus it scored, and that corpus must be the current one.

    The gap: re-running clean changes the corpus but leaves stale scores with nothing
    raising. score_corpus.py stamps input_fp (the corpus fingerprint at score time);
    this check compares it against the corpus's current fingerprint. A mismatch means
    the scores describe a corpus that no longer exists."""
    doms, err = read_mix(os.path.join(root, cfg_default("mix")))
    if err:
        return SKIP, f"mix unreadable ({err}); mix checks own that"
    scores_dir = os.path.join(root, "data", "scores")
    present = [d for d in doms if os.path.isdir(os.path.join(scores_dir, d))]
    if not present:
        return SKIP, "no mix-domain scores on this machine"
    stale, unrecorded, ok = [], [], 0
    for dom in present:
        sp = os.path.join(scores_dir, dom, "score_stats.json")
        if not os.path.isfile(sp):
            unrecorded.append(dom)
            continue
        with open(sp) as f:
            score_stats = json.load(f)
        input_fp = score_stats.get("input_fp")
        if input_fp is None:
            unrecorded.append(dom)
            continue
        cp = os.path.join(root, "data", "corpus", dom, "build_corpus_stats.json")
        if not os.path.isfile(cp):
            stale.append(f"{dom}: scores exist but corpus is gone")
            continue
        with open(cp) as f:
            current_fp = json.load(f).get("fingerprint")
        if current_fp != input_fp:
            stale.append(f"{dom}: scored {input_fp[:8]}, corpus is now {current_fp[:8] if current_fp else 'MISSING'}")
        else:
            ok += 1
    if stale:
        return FAIL, "; ".join(stale)
    if ok == 0:
        return FAIL, (
            f"0/{len(present)} mix domain(s) fresh; "
            f"{len(unrecorded)} have no input_fp ({', '.join(unrecorded)}) -- re-score to stamp them"
        )
    if unrecorded:
        note = (f"; UNKNOWN, not verified: {len(unrecorded)} domain(s) predate input_fp "
                f"({', '.join(unrecorded)}) -- re-score to stamp them")
    else:
        note = ""
    return PASS, f"{ok} domain(s) fresh{note}"


def _broken_score_input_fresh():
    """A domain whose scores were taken against a corpus that has since changed."""
    d = _tmp_repo(mix_obj={"domains": {"web_hq": 1.0}})
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w").close()
    dom = "web_hq"
    corp = os.path.join(d, "data", "corpus", dom)
    scor = os.path.join(d, "data", "scores", dom)
    os.makedirs(corp, exist_ok=True)
    os.makedirs(scor, exist_ok=True)
    with open(os.path.join(corp, "build_corpus_stats.json"), "w") as f:
        json.dump({"fingerprint": "aaaa1111", "filters_fp": "x"}, f)
    with open(os.path.join(scor, "score_stats.json"), "w") as f:
        json.dump({"domain": dom, "input_fp": "bbbb2222", "scorer_fp": "y"}, f)
    return d


def check_sft_pack_holdout(root):
    """An SFT pack must be built against the current holdout set, not a stale one.

    The gap: holdout_hashes.txt is regenerated when the eval set changes, but a pack
    built before the regeneration still carries the old hashes. Training on it leaks
    held-out questions. The pack stamps holdout_fp (the hash of holdout_hashes.txt at
    pack time); this check compares it against the current file. Same pattern as
    score_input_fresh: the derived artifact must match the source it was built from."""
    import hashlib

    pack_path = os.path.join(root, "data", "sft", "sft_all.pt")
    if not os.path.isfile(pack_path):
        return SKIP, "no SFT pack"
    holdout_path = os.path.join(root, "data", "eval", "holdout_hashes.txt")
    if not os.path.isfile(holdout_path):
        return SKIP, "no holdout_hashes.txt"
    try:
        d = _read_ckpt_dict(pack_path)
    except Exception as e:
        return FAIL, f"cannot read pack: {e}"
    pack_fp = d.get("holdout_fp")
    if pack_fp is None:
        return PASS, "pack predates holdout fingerprinting -- UNKNOWN, not verified (repack to stamp)"
    with open(holdout_path, "rb") as f:
        live_fp = hashlib.sha256(f.read()).hexdigest()[:16]
    if pack_fp != live_fp:
        return FAIL, f"pack built against holdout {pack_fp}, current is {live_fp} -- repack"
    return PASS, f"pack matches holdout {live_fp}"


def _holdout_registry(root):
    """(module, error) from the tree's own datagen/holdout.py, imported by path.

    Returns the MODULE, not just REGISTRY, so a caller can read KNOWN_ABSENT from the same
    object. My first version returned the dict and then looked KNOWN_ABSENT up in sys.modules
    under a name built from hash(root) -- which never matched, so every known-absent entry read
    as a real absence. The fix is not a better lookup; it is not splitting the object.

    Imported rather than mirrored. The mirror below used to be the pattern -- a hand-copied list
    with the comment "Must match datagen/holdout.py EVAL_FILES" -- and it had already drifted to
    THREE entries against that file's four, which is what a comment enforces (de, 2026-09-04).
    """
    p = os.path.join(root, "datagen", "holdout.py")
    if not os.path.exists(p):
        return None, "datagen/holdout.py not present"
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"_holdout_reg_{abs(hash(root))}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return None, f"datagen/holdout.py did not import: {type(e).__name__}: {e}"
    reg = getattr(mod, "REGISTRY", None)
    if not isinstance(reg, dict) or not reg:
        return None, "datagen/holdout.py has no non-empty REGISTRY dict"
    return mod, None


# Files under the eval-data locations that are eval OUTPUTS or the guard's own artifacts, not
# eval SOURCES. An enumerable prefix list with a reason each, never a regex over "looks like
# predictions": the pod holds ~16 preds_*.jsonl and a large number of hard_ckpt_*.jsonl per-card
# generation dumps, and a check that flags those gets silenced within a day (e1's point). A
# prefix is a claim someone can check; a regex over intent is not.
_NON_EVAL_PREFIXES = {
    "preds_": "model predictions written by an eval run",
    "hard_ckpt_": "math-hard per-card generation dumps, named for the checkpoint that made them",
    "holdout_slice_": "a slice of a holdout file, written by a scan",
    "holdout_hashes": "the guard's own hash set, derived from the registry",
    "sft_contamination_baseline": "a ratchet baseline for the SFT contamination check",
    "template_contamination_baseline": "a ratchet baseline for the template contamination check",
    # READ BEFORE RULING, both of them, and neither carries an eval question:
    # code_rp1t_handread50.jsonl is 50 rows of {content, lang} -- a hand-read corpus QUALITY
    # sample, read only by scripts/test_shard_glob.py, with no question and no answer.
    "code_rp1t_handread50": "50-row hand-read corpus quality sample ({content, lang}), not an eval",
    # lambada_zh_ids.jsonl is 523 rows of {id} alone -- the id list of lambada_zh_src, which IS
    # registered. Bare ids leak nothing; the passages they name are the surface, and those are
    # covered by the lambada_zh_src entry.
    "lambada_zh_ids": "523 rows of bare {id} for lambada_zh_src, which is itself registered",
    # fetch_stats.json records WHICH MIRROR served each file (AGENTS, Pod: every HF-hosted source
    # lists [hf-mirror, modelscope, huggingface.co] and the fetcher writes which host answered).
    # It sits inside data/eval/humaneval/ and data/eval/lambada_en/ beside the eval files it
    # describes, so the glob finds it; it holds hosts and byte counts, no questions. Flagged on
    # the pod's first run of this check (6e, 2026-09-04) -- the laptop has neither directory.
    "fetch_stats": "the fetcher's per-host record (which mirror served each file), not eval data",
}


def check_eval_registry_complete(root):
    """Every eval/held-out data file is in datagen/holdout.py's REGISTRY, and every entry exists.

    The bug: EVAL_FILES held four paths, the corpus builders took their exclusion population
    from it, and data/sft/control_sft_text_heldout.jsonl was not one of them -- so 2,114 of
    7,523 measurable held-out items reached the pretraining corpus with the guard green,
    fingerprinted and loud (facts/contamination.json#cont.heldout_in_pretrain_corpus, e1
    2026-09-04). The guard was correct on its own population. Its population was wrong.

    Both directions, because each hides a different failure: a file absent from the registry is
    unexcluded corpus contamination, and an entry whose path is gone is a hash set quietly
    covering less than it claims.

    `data/` is gitignored, so on a laptop this check sees almost nothing and says so rather than
    passing: the population is only real on the pod, which is where counting it found 10
    unregistered files against the 8 a laptop glob plus a code grep reported.
    """
    mod, err = _holdout_registry(root)
    if err:
        return SKIP if "not present" in err else FAIL, err
    reg = mod.REGISTRY
    registered = {e["path"] for e in reg.values()}
    known_absent = getattr(mod, "KNOWN_ABSENT", set())
    # A REGISTERED PATH THAT IS ABSENT IS ONLY NEWS WHERE THE DATA LIVES. `data/` is gitignored,
    # so a git checkout is missing most of the population by construction and this direction is a
    # permanent red there -- which AGENTS records as the same as no signal.
    #
    # THE PREDICATE IS is_pod, NOT A FRACTION. My first version asked whether more than half the
    # entries were present, which is not a test of "do we have the population" at all: it read 5
    # of 13 here and skipped, then 7 of 13 in the integration tree and FAILED on the same commit
    # (6e, 2026-09-04). A threshold over how much data happens to be lying around flips on the
    # difference between two checkouts. is_pod asks the question that actually decides it -- a
    # tree with no .git is the hand-pushed pod tree, where every path must exist -- and it is the
    # same predicate mix_shards_present uses for the same reason.
    on_pod = pod_drift.is_pod(root)
    missing_paths = sorted(
        f"{n} -> {e['path']}" for n, e in reg.items()
        if n not in known_absent and not os.path.exists(os.path.join(root, e["path"]))
    ) if on_pod else []
    found, scanned_dirs = [], 0
    for sub in ("data/eval", "data/synthetic", "data/sft"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        scanned_dirs += 1
        for dirpath, _dn, filenames in os.walk(d):
            for fn in sorted(filenames):
                if not fn.endswith((".jsonl", ".json")):
                    continue
                if any(fn.startswith(p) for p in _NON_EVAL_PREFIXES):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                found.append((fn, rel))
    # Name-matched, deliberately wide: a gap hides in whatever a narrow glob excludes, which is
    # the defect being fixed. `data/synthetic` holds training sources too, so there the test is
    # the eval-ish name; under data/eval every source counts.
    evalish = re.compile(r"(holdout|heldout|held_out|_eval|eval_|test_500|humaneval|lambada|ceval)", re.I)
    unregistered = sorted(
        rel for fn, rel in found
        if rel not in registered and (rel.startswith("data/eval/") or evalish.search(fn))
    )
    if not scanned_dirs:
        return SKIP, "no data/eval, data/synthetic or data/sft here -- the population is pod-side"
    problems = []
    if unregistered:
        problems.append(f"{len(unregistered)} file(s) not in the REGISTRY, so no corpus builder "
                        f"excludes them: {unregistered[:4]}")
    if missing_paths:
        problems.append(f"{len(missing_paths)} registry entry(ies) name a path that does not "
                        f"exist, so the hash set covers less than it claims: {missing_paths[:3]}")
    if problems:
        return FAIL, "; ".join(problems)
    where = ("" if on_pod else
             "; registry paths NOT checked for existence here (data/ is gitignored on a git "
             f"checkout -- {sum(1 for e in reg.values() if os.path.exists(os.path.join(root, e['path'])))}"
             f"/{len(reg)} present; that direction speaks on the pod)")
    return PASS, (f"{len(reg)} registry entry(ies); {len(found)} eval-location file(s) scanned, "
                  f"every eval-ish one registered{where}")


def _broken_eval_registry_complete():
    """The REAL holdout.py with one REAL registry entry removed, over a data tree where that
    file EXISTS.

    Mutated, never hand-written: a hand-made registry shares this check's assumptions about the
    schema. The entry removed is the one whose absence caused the incident.

    THE FIXTURE MUST SUPPLY THE FILE, and this is the whole reason the first version passed
    green: `data/` is gitignored, so data/sft/control_sft_text_heldout.jsonl does not exist on a
    laptop -- remove its registry entry and there is no file left over for the check to call
    unregistered. The broken world was "an entry gone AND its file gone", which is a consistent
    world, not a broken one. Symlinking the real data tree is not enough either; the file has to
    be present, so the fixture writes a small stand-in beside the real tree's contents.
    """
    import shutil as _sh
    d = _tmp_repo()
    src = os.path.join(ROOT, "datagen", "holdout.py")
    if not os.path.exists(src):
        return None
    text = open(src, encoding="utf-8").read()
    key = '    "control_sft_text_heldout": {'
    if key not in text:
        return None
    start = text.index(key)
    end = text.index("\n    },\n", start) + len("\n    },\n")
    entry = text[start:end]
    rel = re.search(r'"path":\s*"([^"]+)"', entry).group(1)
    os.makedirs(os.path.join(d, "datagen"), exist_ok=True)
    open(os.path.join(d, "datagen", "holdout.py"), "w", encoding="utf-8").write(
        text[:start] + text[end:]
    )
    # The de-registered file, present. Two rows of the real schema ({question, answer, id, src}),
    # so the check sees a file under an eval location that no registry entry covers -- the
    # incident's exact shape.
    p = os.path.join(d, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for i in (1, 2):
            fh.write(json.dumps({"id": f"x{i}", "question": f"held out {i}?",
                                 "answer": str(i), "src": "fixture"}) + "\n")
    # And the rest of the real eval tree, so the scan is over a realistic population rather
    # than one planted file. Copied per-file: a symlink at data/ would shadow the file above.
    for sub in ("data/eval", "data/synthetic"):
        realsub = os.path.join(ROOT, sub)
        if not os.path.isdir(realsub):
            continue
        for dirpath, _dn, filenames in os.walk(realsub):
            for fn in filenames:
                if not fn.endswith((".jsonl", ".json")):
                    continue
                s = os.path.join(dirpath, fn)
                t = os.path.join(d, os.path.relpath(s, ROOT))
                os.makedirs(os.path.dirname(t), exist_ok=True)
                if not os.path.exists(t):
                    _sh.copy(s, t)
    return d


def _broken_sft_pack_holdout():
    """A pack whose holdout_fp does not match the current holdout_hashes.txt."""
    import torch

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w").close()
    os.makedirs(os.path.join(d, "data", "sft"), exist_ok=True)
    os.makedirs(os.path.join(d, "data", "eval"), exist_ok=True)
    # The current holdout set:
    with open(os.path.join(d, "data", "eval", "holdout_hashes.txt"), "w") as f:
        f.write("current-holdout-hash\n")
    # A pack built against a DIFFERENT (stale) holdout set:
    torch.save(
        {"input_ids": torch.zeros(1, 4097, dtype=torch.int32),
         "labels": torch.zeros(1, 4097, dtype=torch.int32),
         "vocab_id": "test", "holdout_fp": "stale0000000000"},
        os.path.join(d, "data", "sft", "sft_all.pt"),
    )
    return d


# Must match datagen/holdout.py EVAL_FILES. Kept here (not imported) so the check
# reads from `root`, not from holdout.py's module-level ROOT.
_SFT_EVAL_FILES = [
    os.path.join("data", "eval", "math_test_500.jsonl"),
    os.path.join("data", "synthetic", "math_hard_eval_1k.jsonl"),
    os.path.join("data", "eval", "code_holdout_500.jsonl"),
]

# Eval files whose contamination is already accepted and recorded. A hit in a
# live file FAILs. A hit in a retired file reports and ratchets against a
# per-file baseline: FAIL on increase only. Retiring an eval is not accepting
# new contamination.
_RETIRED_EVALS = {
    "math_test_500.jsonl": "post-SFT inflated (30% near-dup in SFT corpus); base values clean",
    "math_hard_eval_1k.jsonl": "v1 retired as metric of record; continuity only",
}

_SFT_CONTAM_BASELINE = os.path.join("data", "eval", "sft_contamination_baseline.json")


def check_sft_pack_uncontaminated(root):
    """Directly test the pack for holdout contamination, not just the process that built it.

    Complements sft_pack_holdout: that proves the guard was alive at pack time; this
    proves the questions are not in the pack. Samples 40 questions per eval file,
    encodes the first 24 tokens, and searches for exact subsequence matches in the
    pack's flattened input_ids.

    Live eval files (code_holdout_500) FAIL on any hit — they are the metrics still
    used as evidence. Retired eval files (math_test_500, math_hard_eval_1k) report
    counts and ratchet against a per-file baseline: FAIL on increase only. The
    baseline is a committed snapshot of accepted contamination; a rise above it is
    new leakage.

    Verbatim matching only. Paraphrased questions (near-dup, not exact) need MinHash,
    the next layer. The 2026-08-30 contamination was 19/20 verbatim, so this covers
    the observed failure mode."""
    pack_path = os.path.join(root, "data", "sft", "sft_all.pt")
    if not os.path.isfile(pack_path):
        return SKIP, "no SFT pack"
    try:
        import numpy as np
        import torch
        from tokenizers import Tokenizer
    except ImportError:
        return SKIP, "torch/tokenizers not available"
    tok_path = os.path.join(root, "data", "tokenizer.json")
    if not os.path.isfile(tok_path):
        return SKIP, "no tokenizer"

    tok = Tokenizer.from_file(tok_path)
    NTOK = 24
    # Per-file probes: track which eval file each probe came from, so live and
    # retired files get different FAIL logic.
    file_probes = {}  # basename -> list of token lists
    for rel in _SFT_EVAL_FILES:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            continue
        lines = [l for l in open(p, encoding="utf-8") if l.strip()]
        step = max(1, len(lines) // 40)
        probes = []
        for line in lines[::step][:40]:
            try:
                q = json.loads(line)["instruction"]
            except (json.JSONDecodeError, KeyError):
                continue
            ids = tok.encode(q).ids[:NTOK]
            if len(ids) >= 8:  # too short to be a meaningful fingerprint
                probes.append(ids)
        if probes:
            file_probes[os.path.basename(rel)] = probes
    if not file_probes:
        return SKIP, "no eval questions found"

    d = torch.load(pack_path, map_location="cpu", weights_only=True)
    flat = d["input_ids"].numpy().flatten()
    del d

    # Per-file baseline of accepted exact-hit counts. A rise above it is new
    # contamination. The baseline is a committed snapshot, not a live measurement.
    baseline = {}
    bl_path = os.path.join(root, _SFT_CONTAM_BASELINE)
    if os.path.isfile(bl_path):
        try:
            with open(bl_path, encoding="utf-8") as f:
                baseline = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Narrow candidates token by token: after 3-4 tokens, almost nothing survives.
    worst = PASS
    parts = []
    for fname, probes in file_probes.items():
        hits = 0
        for probe in probes:
            plen = len(probe)
            candidates = np.where(flat[: -plen + 1] == probe[0])[0]
            for i, t in enumerate(probe[1:], 1):
                if len(candidates) == 0:
                    break
                candidates = candidates[flat[candidates + i] == t]
            if len(candidates) > 0:
                hits += 1
        n = len(probes)
        retired = fname in _RETIRED_EVALS
        if retired:
            reason = _RETIRED_EVALS[fname]
            base = baseline.get(fname)
            if base is None:
                if worst == PASS:
                    worst = WARN
                parts.append(f"{fname}: {hits}/{n} hits (retired: {reason}; no baseline — unratcheted)")
            elif hits > base:
                worst = FAIL
                parts.append(
                    f"{fname}: {hits}/{n} hits > baseline {base} "
                    f"(retired: {reason}; INCREASE = new contamination)"
                )
            else:
                parts.append(f"{fname}: {hits}/{n} hits (retired: {reason}; baseline {base})")
        else:
            if hits:
                worst = FAIL
                parts.append(f"{fname}: {hits}/{n} hits (LIVE — must be 0)")
            else:
                parts.append(f"{fname}: 0/{n} (live, clean)")
    suffix = " (verbatim only; paraphrased not detected)" if worst == PASS else ""
    return worst, "; ".join(parts) + suffix


def _broken_sft_pack_uncontaminated():
    """A pack that contains a real holdout question from a LIVE eval file, plus a
    retired-file question above its baseline. Either alone must FAIL."""
    import shutil

    import torch
    from tokenizers import Tokenizer

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w").close()

    # Real tokenizer and eval files, so the probe encodes real questions.
    tok_src = os.path.join(ROOT, "data", "tokenizer.json")
    if not os.path.isfile(tok_src):
        raise SelftestSkip("no data/tokenizer.json -- check SKIPs without it")
    shutil.copy(tok_src, os.path.join(d, "data", "tokenizer.json"))
    eval_dir = os.path.join(d, "data", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    for fname in ["math_test_500.jsonl", "code_holdout_500.jsonl"]:
        shutil.copy(os.path.join(ROOT, "data", "eval", fname), os.path.join(eval_dir, fname))

    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    # First question of each file is always in the every-Nth-line sample.
    rows = []
    for fname in ["code_holdout_500.jsonl", "math_test_500.jsonl"]:
        q = json.loads(
            open(os.path.join(ROOT, "data", "eval", fname), encoding="utf-8").readline()
        )["instruction"]
        ids = tok.encode(q).ids[:24]
        rows.append(ids + [0] * (4097 - len(ids)))

    # Baseline says math_test_500 has 0 accepted hits; the planted question is an increase.
    with open(os.path.join(eval_dir, "sft_contamination_baseline.json"), "w") as f:
        json.dump({"math_test_500.jsonl": 0, "code_holdout_500.jsonl": 0}, f)

    # Plant both questions' tokens in the pack.
    os.makedirs(os.path.join(d, "data", "sft"), exist_ok=True)
    torch.save(
        {
            "input_ids": torch.tensor(rows, dtype=torch.int32),
            "labels": torch.full((len(rows), 4097), -100, dtype=torch.int32),
            "vocab_id": "test",
            "holdout_fp": "x",
        },
        os.path.join(d, "data", "sft", "sft_all.pt"),
    )
    return d


_TEMPLATE_EVAL_FILES = [
    os.path.join("data", "eval", "code_holdout_500.jsonl"),
    os.path.join("data", "eval", "code_holdout_v2_500.jsonl"),
]
# Every file that fed an SFT pack. Mirrors SOURCES in scripts/census_code_v2.py.
_TEMPLATE_SFT_SOURCES = [
    "data/alpaca_gpt4_zh.jsonl",
    "data/coig.jsonl",
    "data/openo1_sft.jsonl",
    "data/gsm8k_zh.jsonl",
    "data/school_math_r1_zh.jsonl",
    "data/s1k.jsonl",
    "data/sft/fable5_cot.jsonl",
    "data/sft/v5_evol_code_2300.jsonl",
    "data/synthetic/knowledge_qa_zh.jsonl",
    "data/synthetic/math_gsm8k_zh.jsonl",
    "data/synthetic/code_python_zh.jsonl",
]
_TEMPLATE_RETIRED = {
    "code_holdout_500.jsonl": "v1: SFT trained on the same synthetic generator canon (code_python_zh); v2 is the live set",
}
_TEMPLATE_BASELINE = os.path.join("data", "eval", "template_contamination_baseline.json")
_TEMPLATE_TEXT_FIELDS = ("instruction", "output", "prompt", "response", "input", "q", "a")


def _template_norm(s):
    """Family-level key: lowercase, digit runs to #, quoted strings to ", whitespace collapsed.

    Two instances of one generator template with different parameters normalize equal."""
    s = str(s).lower()
    s = re.sub(r"\d+(?:st|nd|rd|th)", "#", s)  # ordinals: 17th and 92nd are one parameter
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'', '"', s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


_TEMPLATE_CACHE = os.path.join(".cache", "template_contamination.json")
_TEMPLATE_CACHE_VERSION = "v1"  # bump when the scan logic changes


def _template_inputs_key(root):
    """Content fingerprint of the scan's inputs, corpus_fingerprint-style: whole-file
    sha256 for the small eval files, head+tail 64KB for the large SFT sources.
    Head+tail catches same-size edits (a same-size rewrite moves head or tail);
    mtime-only would not, and a copy/podput changes mtime without touching a byte."""
    import hashlib

    h = hashlib.sha256(_TEMPLATE_CACHE_VERSION.encode())
    for rel in _TEMPLATE_EVAL_FILES:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                h.update(rel.encode() + b"\0" + hashlib.sha256(f.read()).digest())
    for rel in _TEMPLATE_SFT_SOURCES:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            continue
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            head = f.read(65536)
            if size > 65536:
                f.seek(-65536, os.SEEK_END)
                tail = f.read(65536)
            else:
                tail = b""
        h.update(
            f"{rel}:{size}:{hashlib.sha256(head).hexdigest()}:{hashlib.sha256(tail).hexdigest()}\n".encode()
        )
    return h.hexdigest()[:16]


def check_eval_sft_template_contamination(root):
    """Cache wrapper: the scan costs ~27s on a full-data checkout, which is the
    hook-cost class that bred the --no-verify habit. Keyed on a content fingerprint
    of its inputs, so a hit path is stat calls plus one small hash, under 0.2s."""
    cache_path = os.path.join(root, _TEMPLATE_CACHE)
    key = _template_inputs_key(root)
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("key") == key:
                return cached["state"], cached["evidence"] + " (cached)"
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass  # corrupt or stale cache: recompute

    state, evidence = _template_scan(root)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"key": key, "state": state, "evidence": evidence}, f)
    except OSError:
        pass
    return state, evidence


def _template_scan(root):
    """No code eval problem shares a generator template with an SFT source.

    sft_pack_uncontaminated matches verbatim token subsequences. The 2026-08-31
    code-500 v1 failure sat one level above it: SFT trained on the same synthetic
    generator, so the model learned the template itself and scored 40% on variants
    it had never seen. This check normalizes literals away and tests containment of
    the first 200 normalized chars of each sampled eval problem in every SFT
    source's text fields.

    Live eval files FAIL on any hit. Retired files ratchet against a per-file
    baseline: FAIL on increase, WARN when unbaselined hits exist (same shape as
    sft_pack_uncontaminated)."""
    file_probes = {}
    for rel in _TEMPLATE_EVAL_FILES:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            continue
        lines = [l for l in open(p, encoding="utf-8") if l.strip()]
        step = max(1, len(lines) // 40)
        needles = []
        for line in lines[::step][:40]:
            try:
                q = json.loads(line)["instruction"]
            except (json.JSONDecodeError, KeyError):
                continue
            n = _template_norm(q)[:200]
            if len(n) >= 32:
                needles.append(n)
        if needles:
            file_probes[os.path.basename(rel)] = needles
    if not file_probes:
        return SKIP, "no eval questions found"

    sources = [s for s in _TEMPLATE_SFT_SOURCES if os.path.isfile(os.path.join(root, s))]
    if not sources:
        return SKIP, "no SFT sources present"

    baseline = {}
    bl_path = os.path.join(root, _TEMPLATE_BASELINE)
    if os.path.isfile(bl_path):
        try:
            with open(bl_path, encoding="utf-8") as f:
                baseline = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    remaining = {name: set(range(len(v))) for name, v in file_probes.items()}
    # One alternation regex over all needles: a single C-level scan per field
    # instead of len(needles) Python-level `in` checks (31 live needles over
    # hundreds of thousands of source lines otherwise blows the check timeout).
    all_needles = [n for needles in file_probes.values() for n in needles]
    pattern = re.compile("|".join(re.escape(n) for n in all_needles))
    for rel in sources:
        if not any(remaining.values()):
            break
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            for line in f:
                if not any(remaining.values()):
                    break
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for field in _TEMPLATE_TEXT_FIELDS:
                    t = d.get(field)
                    if not isinstance(t, str) or len(t) < 32:
                        continue
                    tn = _template_norm(t)
                    if not pattern.search(tn):
                        continue
                    # A field matched: attribute exactly, needle by needle.
                    # Alternation alone reports one needle per start position,
                    # so a short needle that prefixes a long one would be lost.
                    for name, needles in file_probes.items():
                        for i in [i for i in remaining[name] if needles[i] in tn]:
                            remaining[name].discard(i)

    worst = PASS
    parts = []
    for name, needles in sorted(file_probes.items()):
        hits = len(needles) - len(remaining[name])
        n = len(needles)
        if name in _TEMPLATE_RETIRED:
            reason = _TEMPLATE_RETIRED[name]
            base = baseline.get(name)
            if base is None:
                if hits and worst == PASS:
                    worst = WARN
                parts.append(f"{name}: {hits}/{n} template hits (retired: {reason}; no baseline -- stamp {_TEMPLATE_BASELINE})")
            elif hits > base:
                worst = FAIL
                parts.append(f"{name}: {hits}/{n} template hits > baseline {base} (retired: {reason}; INCREASE = new leakage)")
            else:
                parts.append(f"{name}: {hits}/{n} template hits (retired: {reason}; baseline {base})")
        else:
            if hits:
                worst = FAIL
                parts.append(f"{name}: {hits}/{n} sampled problems share a template with an SFT source (LIVE -- must be 0)")
            else:
                parts.append(f"{name}: 0/{n} (live, template-clean)")
    return worst, "; ".join(parts)


def _broken_eval_sft_template_contamination():
    """An eval problem and an SFT doc from one generator template, different parameters."""
    d = _tmp_repo()
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w").close()
    os.makedirs(os.path.join(d, "data", "eval"), exist_ok=True)
    os.makedirs(os.path.join(d, "data", "sft"), exist_ok=True)
    with open(os.path.join(d, "data", "eval", "code_holdout_v2_500.jsonl"), "w") as f:
        f.write(json.dumps({"instruction": "Write a Python function fib(n) that returns the 17th Fibonacci number. Show the full recursive definition with docstring and type hints."}) + "\n")
    with open(os.path.join(d, "data", "sft", "v5_evol_code_2300.jsonl"), "w") as f:
        f.write(json.dumps({"instruction": "Write a Python function fib(n) that returns the 92nd Fibonacci number. Show the full recursive definition with docstring and type hints.", "output": "def fib(n): ..."}) + "\n")
    return d


def _broken_restartability():
    """The real regression: a new script that accumulates in a loop and saves once at the end."""
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    for f in ("restartability_audit.py", "restartability_baseline.json"):
        shutil.copy(os.path.join(ROOT, "scripts", f), os.path.join(d, "scripts", f))
    os.makedirs(os.path.join(d, "datagen"), exist_ok=True)
    with open(os.path.join(d, "datagen", "new_long_job.py"), "w") as fh:
        fh.write("import numpy as np\nxs = []\nfor i in range(10):\n    xs.append(i)\n"
                 "np.save('out.npy', xs)\n")
    return d


def _broken_gemm_dims():
    # The REAL train.py with ffn_hidden 3072 -> 3400: 8-aligned (passes the cuBLAS
    # tier) but not 16-aligned, so _fp8_ok silently drops FP8. Mutated, not hand-written.
    import shutil

    d = _tmp_repo()
    p = os.path.join(d, "train.py")
    shutil.copy(os.path.join(ROOT, "train.py"), p)
    src = open(p, encoding="utf-8").read()
    src = src.replace("ffn_hidden = 3072", "ffn_hidden = 3400", 1)
    assert "ffn_hidden = 3400" in src, "real train.py no longer has 'ffn_hidden = 3072'; update _broken_gemm_dims"
    open(p, "w", encoding="utf-8").write(src)
    return d


def _broken_guard():
    d = _tmp_repo()
    open(os.path.join(d, "train.py"), "w").write("def main():\n    pass\n")
    return d


# --------------------------------------------------------------------------- facts
#
# Measurements live in facts/*.json, never in prose. A fact carries its measurement
# config -- a value without one is this project's repeated failure class.

FACTS_DIR = os.path.join(ROOT, "facts")
FACT_REQUIRED = {"id", "value", "measured", "source", "config", "uncertainty", "status"}
FACT_STATUS = {"measured", "recorded", "unmeasured", "retracted"}
FACT_NEEDS_CLAIM = {"unmeasured", "retracted"}
# A retracted entry must say, as data, WHICH numbers died -- and `[]` is a real answer.
#
# THE FIELD EXISTS BECAUSE `value` CANNOT ANSWER IT. On a retracted entry, `value` is rewritten
# into a narration of the retraction, so one field holds the dead number AND the number that
# replaced it, separated only by prose: eff.depth_is_not_the_mfu_gap carries retracted 14.2/16.0
# beside correct 43.5/23.7, and eff.dynamo_recompile_not_a_lever's 54.9/260/72% are ALL from the
# measurement that superseded it. Any tool that greps a retracted entry's numbers hunts correct
# values, which is worse than not checking: it spends the reader's trust in everything the tool
# says (§159's addendum, de and e1, 2026-09-04).
#
# WHY A LIST AND NOT A FLAG, from reading all nine retracted entries rather than pattern-matching
# them: FIVE of the nine retract a CONCLUSION while their numbers stand. be.l1_3shot_retracted's
# rerun reproduced 0.2/63.6/8.9; cont.sft_all_code_holdout_leak's v2 2.2% and v3 40.0% are
# restored; ds.second_resume_rereads_one_segment's 8,192 rows was true of every checkpoint written
# before 52aec31. A field that marked all nine's numbers dead would kill five entries' correct
# values -- the same defect in a new place. So the list holds only values that are wrong, and an
# empty list states that no number died, which is the common case.
FACT_RETRACTED_VALUE = "retracted_value"
FACT_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
FACT_SOURCE_PATH = re.compile(
    # probes/ was absent until 2026-09-02 and its absence was a blind spot, not a scoping
    # decision: 44's 22 probe deletions in 30b9010 rewrote 39 refs to path@rev, and this
    # check would have passed them either way because it never looked in that directory.
    # The same retirements under eval/ or scripts/ FAILed the day de wrote them (de-21).
    # Added once @rev was understood here; all 27 probes/ citations resolve, 21 by rev and
    # profile_step.py live.
    r"(?<![\w/])(?:data|runs|scripts|docs|eval|datagen|filters|mathbank|algorithms|workflows|probes)/[\w./-]+"
)
# Debt register for tracked-missing sources: each entry carries a reason. Can only
# shrink -- a new missing source is a FAIL, not a baseline entry. Reported in `gaps`.
FACT_SOURCE_BASELINE = os.path.join("facts", "source_baseline.json")
CORPUS_FILTERS_BASELINE = os.path.join("facts", "corpus_filters_baseline.json")


def _cat_file_exists(root, specs):
    """{spec: True/False} for many `git cat-file` specs in ONE subprocess.

    `git cat-file --batch-check` reads specs on stdin and prints one line per line of
    input, in order, `<sha> <type> <size>` for a hit and `<spec> missing` for a miss.
    Order and one-line-per-input are what make the mapping safe, and both are asserted
    on the real repository in _selftest_batched_git_probes.

    WHY THIS EXISTS: the per-spec form was one subprocess per probe, and the probe count
    grows with the register. Measured on this repo 2026-09-03, 86 closed tasks: 86
    `rev-parse` calls cost 1.43 s of tasks_closed_by_commit's 2.15 s, and one
    `--batch-check` for the same 86 costs 0.023 s -- 62x. The 5 s deadline was not the
    defect; a cost that grows one subprocess per row is, and raising the deadline only
    moves the date the check goes permanently red (98 reported exactly that today, two
    consecutive timeouts on this check plus facts_well_formed).

    Returns every spec as False when git cannot answer at all (no .git, as on the pod),
    which is what both callers did before."""
    out = dict.fromkeys(specs, False)
    if not specs:
        return out
    try:
        r = subprocess.run(
            ["git", "-C", root, "cat-file", "--batch-check"],
            input="".join(s + "\n" for s in specs),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return out
    # Same split as _resolve_shas: rc != 0 is git declining to answer (not a repository),
    # so every spec stays False -- what `cat-file -e` returned there. A truncated
    # SUCCESSFUL answer is the different, louder case below.
    if r.returncode != 0:
        return out
    lines = r.stdout.split("\n")
    # A SHORT OUTPUT MUST NOT PASS SILENTLY. zip stops at the shorter side, so if git
    # printed fewer lines than specs, every unmatched spec would keep its `False` default
    # -- which reads as "this rev does not hold the path", i.e. a fact citation refused for
    # a reason that never happened. One line per input line is the property the whole
    # mapping rests on, so it is asserted rather than assumed.
    if len(lines) < len(specs):
        raise RuntimeError(
            f"git cat-file --batch-check returned {len(lines)} line(s) for {len(specs)} "
            f"spec(s) -- the one-line-per-input contract this mapping needs does not hold"
        )
    for spec, line in zip(specs, lines[: len(specs)], strict=True):
        out[spec] = bool(line) and not line.endswith(" missing")
    return out


def _rev_has_path(root, rev, path):
    """Does `path` exist in the tree at `rev`? False when git cannot answer.

    A `path@rev` citation is only durable if the rev actually holds the file. Accepting
    the syntax without checking would turn the retirement form into a way to make any
    dead citation pass -- which is the shape this repo keeps paying for (metadata is a
    claim, not a fact about content). False on the pod, where there is no .git; the
    caller only reaches this on a full checkout.

    One spec at a time. check_facts_well_formed batches instead (_cat_file_exists);
    this stays for the single-probe callers and for the selftest that pins the two
    against each other."""
    return _cat_file_exists(root, [f"{rev}:{path}"])[f"{rev}:{path}"]


def _gitignored_set(paths, root):
    """{path: True/False} for many paths in ONE `git check-ignore` subprocess.

    Same reasoning as _cat_file_exists: the per-path form was one subprocess per fact
    source, and 50 of them cost 0.84 s of check_facts_well_formed's 1.49 s on this repo
    (measured 2026-09-03); one batched call over the same 50 costs 0.017 s.

    `--stdin --verbose --non-matching` is the combination that yields one output line per
    input line whether or not it matched -- without --non-matching, non-ignored paths
    print nothing and the output can no longer be zipped to the input. Each line is
    `<source>:<lineno>:<pattern>\\t<path>`, and `::\\t<path>` for a non-match, so the
    ignored test is "the prefix before the tab is not `::`".

    Falls back to the same minimal .gitignore reader as the single-path form when git
    cannot answer (rc 128 on the pod, no .git). Each path is probed as itself AND with a
    trailing slash, so a directory pattern (data/corpus/*/) matches a source written
    without one -- the behaviour the single-path form had, kept because dropping it made
    every gitignored pod-only artifact a fact cites read as rot."""
    paths = list(dict.fromkeys(paths))
    if not paths:
        return {}
    probes = [p for path in paths for p in (path, path + "/")]
    hit = {}
    try:
        r = subprocess.run(
            ["git", "-C", root, "check-ignore", "--stdin", "--verbose", "--non-matching"],
            input="".join(p + "\n" for p in probes),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode in (0, 1):
            lines = r.stdout.split("\n")
            # A short output falls through to the reader rather than zipping against it:
            # the reader answers the same question and its disagreement with git is
            # bounded and asserted, while a truncated zip silently answers "not ignored"
            # for every probe past the end.
            if len(lines) >= len(probes):
                for probe, line in zip(probes, lines[: len(probes)], strict=True):
                    hit[probe] = not line.startswith("::\t")
                return {path: hit.get(path, False) or hit.get(path + "/", False) for path in paths}
        # 128: git unavailable or not a repo (pod) -> fall through to the reader
    except (OSError, subprocess.SubprocessError):
        pass
    return {path: _gitignore_reader(path, root) for path in paths}


def _gitignore_rx(pat):
    """One .gitignore pattern -> a compiled regex over a repo-relative path.

    fnmatch is the wrong tool and was the second half of the reader's defect: its `*`
    crosses `/`, git's does not. `data/*.jsonl` therefore matched
    `data/eval/math_test_500.jsonl` under fnmatch while git says it does not -- measured
    2026-09-03 in the agreement sweep below.

    Translated here rather than pulled from a library: `pathspec` is not a dependency of
    this repo and the pod installs nothing at check time."""
    anchored = "/" in pat.rstrip("/")
    body = pat.strip("/") if pat.startswith("/") else pat.rstrip("/")
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if body.startswith("**", i):
            out.append(".*")
            i += 2
            if body.startswith("/", i):
                i += 1
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    core = "".join(out)
    # Unanchored: match at any depth. Anchored: from the repo root.
    prefix = r"(?:.*/)?" if not anchored else ""
    # Trailing `(/.*)?$` so an ignored directory also ignores everything beneath it.
    return re.compile(rf"^{prefix}{core}(?:/.*)?$")


def _gitignore_reader(path, root):
    """The pod fallback: read .gitignore directly, when `git check-ignore` cannot answer.

    TWO DEFECTS, both found by the selftest that pins this against git rather than by
    reading it, and both invisible where they mattered -- this code only runs where git
    cannot answer, which is the pod, so a divergence FAILs a check nobody can reproduce
    on a laptop. The comment above check_facts_well_formed's broken world already
    recorded the first hazard ("it missed data/corpus/math/") and the reader was never
    fixed.

      1. A directory pattern was compared LITERALLY after its slash was stripped
         (`path == pat` or `path.startswith(pat + "/")`), so `data/corpus/*/` -- the
         pattern covering every corpus domain -- matched nothing: `data/corpus/web_hq`
         is neither equal to `data/corpus/*` nor under `data/corpus/*/`.
      2. fnmatch's `*` crosses `/` and git's does not, so `data/*.jsonl` matched
         `data/eval/math_test_500.jsonl`. Fixed in _gitignore_rx.

    Negation is honoured now (last match wins, as git does): once prefix matching worked,
    `!data/corpus/primary/` became reachable and the old "skips negation, no fact source
    points there" excuse stopped holding -- data/corpus/primary is exactly such a source.

    KNOWN, MEASURED DIVERGENCE from `git check-ignore`, and it is git's behaviour rather
    than a defect here: git consults the index, so a path with a TRACKED file under it
    reads as not-ignored (data/synthetic/ holds one tracked .jsonl; `check-ignore` says
    no, `check-ignore --no-index` says yes). This reader answers the question .gitignore
    asks and cannot see an index. The selftest asserts agreement on paths with no tracked
    content and records this one exception by name."""
    gi = os.path.join(root, ".gitignore")
    if not os.path.exists(gi):
        return False
    rel = path.rstrip("/")
    verdict = False
    for line in open(gi, encoding="utf-8"):
        line = line.rstrip("\n").strip()
        if not line or line.startswith("#"):
            continue
        neg = line.startswith("!")
        pat = line[1:] if neg else line
        if not pat:
            continue
        if _gitignore_rx(pat).match(rel):
            verdict = not neg
    return verdict


def _is_gitignored(path, root):
    """True if path is covered by .gitignore. One path at a time; batching callers use
    _gitignored_set. Kept so the selftest can pin the two implementations against each
    other on the real .gitignore -- a batched form that disagrees with the single form
    is the defect the batching would otherwise introduce silently."""
    return _gitignored_set([path], root)[path]


#: Ledgers that merge by union. Every one must be one JSON object per physical line.
def _union_ledgers(root):
    """Ledgers .gitattributes merges by union, read from .gitattributes itself.

    A hand-kept copy of this list drifts from the file that decides the behaviour:
    runs/review.jsonl was union-merged from the moment it existed but sat outside the
    hardcoded tuple, so a pretty-printed review row -- exactly the class this check
    exists for -- would not have been caught (44's review, 2026-08-31)."""
    p = os.path.join(root, ".gitattributes")
    if not os.path.exists(p):
        return ()
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line.startswith("#") or "merge=union" not in line:
            continue
        path = line.split()[0]
        if path.endswith(".jsonl"):
            out.append(path)
    return tuple(out)


def check_tasks_paired_and_prior(root):
    """Every task opened since the rule names a second session who agreed it and what
    was already known before it started.

    Three sessions took AUTHORITY at once and two of them duplicated a merge; separately
    a throughput number had no reference point in the literature at all, so "is this
    good" could not be answered. Both are the same absence: nobody stated, before
    starting, who agreed and what was already known (user order 2026-09-01).

    A DROPPED task is out of scope. _read_tasks already folds by id, so the check reads
    each task's latest state -- but it judged a dropped one anyway, and dropping is how a
    task with a bad prior is supposed to be retired. fb hit this: after dropping a task
    whose prior was wrong, the check still FAILed on it, and the only way through was to
    delete the uncommitted row -- so the register loses the record of the decision to
    satisfy a check about record-keeping."""
    rows = _read_tasks(os.path.join(root, "runs", "tasks.jsonl"))
    # Two scopes unioned, because a timestamp alone cannot separate them today: rows
    # written before the UTC fix carry CST, and 21:27 CST sorts after a 14:40 UTC
    # threshold. Every row the new `add` writes carries the prior key, so those are in
    # scope whatever their clock; the date takes over once every row is UTC.
    scope = [t for t in rows
             if (t.get("state") != "dropped"
                 and ("prior" in t or (t.get("opened") or "") >= PAIR_PRIOR_FROM))]
    if not scope:
        return SKIP, f"no task opened since the rule took effect ({PAIR_PRIOR_FROM})"
    bad = []
    for t in scope:
        pair, prior = t.get("pair"), (t.get("prior") or "")
        if not pair:
            bad.append(f"{t['id']}: no second session agreed it")
        elif pair == t.get("owner"):
            bad.append(f"{t['id']}: paired with its own owner")
        if not prior:
            bad.append(f"{t['id']}: does not say what was already known")
        elif prior != "defect-fix" and not re.search(r"\d{4}\.\d{4,5}|facts/\S+#\S+|https?://", prior):
            bad.append(f"{t['id']}: prior '{prior[:40]}' is neither a citation nor defect-fix")
    if bad:
        return FAIL, f"{len(bad)} of {len(scope)} task(s): {'; '.join(bad[:3])}"
    return PASS, f"{len(scope)} task(s), each agreed by a second session with its prior art named"


def _broken_tasks_paired_and_prior():
    """The REAL register plus one open row with no pair and no prior.

    Two rows, because the dropped-row exemption must not be a way through: x-2 is the
    same violation retired by a later `dropped` event, and the world must still FAIL --
    on x-1 only. A world with just the dropped row would pass and certify nothing."""
    import shutil as _sh
    d = _tmp_repo()
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    src = os.path.join(ROOT, "runs", "tasks.jsonl")
    dst = os.path.join(d, "runs", "tasks.jsonl")
    _sh.copy(src, dst)
    with open(dst, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "x-1", "owner": "de", "state": "open", "task": "t",
                            "opened": "2099-01-01 00:00"}) + "\n")
        f.write(json.dumps({"id": "x-2", "owner": "de", "state": "open", "task": "t",
                            "opened": "2099-01-01 00:00"}) + "\n")
        f.write(json.dumps({"id": "x-2", "owner": "de", "state": "dropped", "task": "t",
                            "opened": "2099-01-01 00:00"}) + "\n")
    return d


def check_tasks_closed_by_commit(root):
    """Every task closed since the rule carries a commit that reaches main and touches
    its evidence.

    Closing wrote free text, so a task closed on a path that never existed read as
    delivered and the register could not tell the difference. The rule dates from
    TASK_COMMIT_FROM; rows closed before it keep their prose evidence."""
    rows = _read_tasks(os.path.join(root, "runs", "tasks.jsonl"))
    # THE MAP MUST SEE MERGE COMMITS. `git log --name-only` prints no paths for a merge, so this
    # check silently could not verify any delivery that landed in one -- 607 of main's 2755
    # commits (22%) read as touching nothing, and closing de-30 against c889bc2 was refused with
    # `touches []` while `git show --stat` listed 7 files. Asserted here rather than trusted,
    # because the failure is invisible: the check stays green and simply cannot see a whole class
    # of commit. A merge on main with zero paths means _main_touched lost its `-m` or its union.
    touched = _main_touched(root)
    if touched:
        merges = subprocess.run(
            ["git", "-C", root, "log", "main", "--merges", "--format=%H", "-40"],
            capture_output=True, text=True).stdout.split()
        blind = [s[:8] for s in merges if s in touched and not touched[s]]
        if len(blind) > len(merges) // 2 and merges:
            return FAIL, (f"_main_touched sees no paths for {len(blind)} of the last "
                          f"{len(merges)} merges on main ({blind[:3]}) -- it lost `-m`, so any "
                          f"delivery inside a merge cannot be verified")
    scope = [t for t in rows
             if t.get("state") == "done" and (t.get("closed") or "") >= TASK_COMMIT_FROM]
    if not scope:
        return SKIP, f"no task closed since the rule took effect ({TASK_COMMIT_FROM})"
    bad = []
    # One subprocess for every sha, not one each: see _resolve_shas.
    resolved = _resolve_shas(root, [t.get("commit") for t in scope if t.get("commit")])
    for t in scope:
        sha = t.get("commit")
        if not sha:
            bad.append(f"{t['id']}: closed with no commit")
            continue
        why = _commit_delivers(sha, t.get("evidence") or "", root, t["id"], t.get("closed"),
                               resolved=resolved)
        if why:
            bad.append(f"{t['id']}: {why}")
    if bad:
        return FAIL, f"{len(bad)} of {len(scope)} closed task(s): {'; '.join(bad[:3])}"
    return PASS, f"{len(scope)} closed task(s), each delivered by a commit that reaches main"


def _broken_tasks_closed_by_commit():
    import shutil as _sh
    d = _tmp_repo()
    src = os.path.join(ROOT, "runs", "tasks.jsonl")
    dst = os.path.join(d, "runs", "tasks.jsonl")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    _sh.copy(src, dst)
    rows = [json.loads(x) for x in open(dst, encoding="utf-8") if x.strip()]
    seed = dict(rows[-1], id="broken-1", state="done", closed="2099-01-01 00:00",
                evidence="scripts/harness.py", commit="0" * 40, reviewer="44")
    with open(dst, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(seed, ensure_ascii=False) + "\n")
    return d


QUEUE_MIN_OPEN = 2
QUEUE_EXEMPT = {"fb", "98"}


def check_owner_queue_depth(root):
    """Every roster member has at least QUEUE_MIN_OPEN open, unblocked tasks.

    On 2026-09-02 the user found six sessions idle while the register showed 16 open rows:
    nine of them were blocked on the frozen training path and the rest were held by two
    owners. An idle session is a cost with no artifact, and nothing in the repo said so.
    An empty queue is FAIL: that session is idle now. A queue of one is WARN: the
    controller refills before it empties."""
    roster_p = os.path.join(root, "runs", "roster.json")
    if not os.path.exists(roster_p):
        return SKIP, "no runs/roster.json"
    members = [m["name"] for m in json.load(open(roster_p, encoding="utf-8"))["members"]
               if m["name"] not in QUEUE_EXEMPT]
    rows = _read_tasks(os.path.join(root, "runs", "tasks.jsonl"))
    depth = {m: 0 for m in members}
    for t in rows:
        if t.get("state") == "open" and not (t.get("blocked_on") or "").strip():
            if t.get("owner") in depth:
                depth[t["owner"]] += 1
    empty = [m for m, n in sorted(depth.items()) if n == 0]
    if empty:
        return FAIL, f"idle: no open unblocked task for {', '.join(empty)} -- controller assigns now"
    short = [f"{m}={n}" for m, n in sorted(depth.items()) if n < QUEUE_MIN_OPEN]
    if short:
        return WARN, f"queue under {QUEUE_MIN_OPEN} open unblocked task(s): {', '.join(short)} -- controller refills"
    return PASS, ", ".join(f"{m}={n}" for m, n in sorted(depth.items()))


def _broken_owner_queue_depth():
    import shutil as _sh
    d = _tmp_repo()
    for rel in ("runs/roster.json", "runs/tasks.jsonl"):
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        _sh.copy(os.path.join(ROOT, rel), os.path.join(d, rel))
    dst = os.path.join(d, "runs", "tasks.jsonl")
    rows = [json.loads(x) for x in open(dst, encoding="utf-8") if x.strip()]
    with open(dst, "w", encoding="utf-8") as fh:
        for r in rows:
            if r.get("state") == "open":
                r = dict(r, blocked_on="frozen until the run ends")
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return d


PEER_STALL_MIN = 120


def check_peer_stalled(root):
    """A roster member holding an open task with no commit and no ledger row for two hours.

    The register says who OWNS work; nothing said whether anyone is DOING it. The idleness
    owner_queue_depth catches is an empty queue; this is the opposite shape -- a full queue
    and no output, which reads identically to a session working hard on something not yet
    committed. That ambiguity is why it WARNs and never FAILs: the check cannot see a
    session mid-edit, so the honest reading is "nothing has reached the repo from here in
    two hours", not "this session is stopped".

    One source of truth with cmd_who: scripts/board.liveness, so the terminal column and
    this check can never disagree about how old a member is. Its docstring carries the two
    facts that decide the number -- every branch a member owns counts (b0's eponymous tip
    was 23h old while b0-ve-rownorms was 3 minutes old), and ledger rows are UTC while git
    dates are +08 local."""
    sys.path.insert(0, os.path.join(root, "scripts"))
    try:
        from board import liveness
    except Exception as e:
        return SKIP, f"scripts/board.py not importable: {type(e).__name__}: {e}"
    live = liveness(root)
    if not live:
        return SKIP, "no runs/roster.json"
    stalled = []
    for name, d in sorted(live.items()):
        if not d["open_tasks"]:
            continue
        ages = [x for x in (d["commit_min"], d["ledger_min"]) if x is not None]
        # No branch AND no ledger row is not a stall -- it is a member who has never
        # appeared, which is a roster question rather than a liveness one.
        if not ages:
            continue
        if min(ages) >= PEER_STALL_MIN:
            stalled.append(f"{name} {min(ages)}m ({d['open_tasks']} open)")
    if stalled:
        return WARN, (f"{len(stalled)} member(s) with an open task and nothing in the repo for "
                      f"{PEER_STALL_MIN}m: {', '.join(stalled)}")
    return PASS, f"{len(live)} roster member(s), none quiet for {PEER_STALL_MIN}m with work open"


def _broken_peer_stalled():
    """A REAL roster with one added member whose branch does not exist and whose only
    ledger row is three days old, so the check must name it.

    Mutated, not hand-written: the roster and the register are the real files, and the
    fake member's task row is a copy of a real row with the owner and dates changed.

    KNOWN CEILING, stated because the world is WEAKER than it looks: _tmp_repo carries no
    branches, so `commit_min` is None for EVERY member here and ledger age alone decides.
    The world therefore also names real members whose ledger row is old but whose commits
    are minutes fresh (98, tilerl) -- it goes WARN partly for a reason the real tree does
    not have. _selftest_peer_stalled_names_the_fixture pins the part that matters: the
    fixture member is named, and the same roster without it does not name it."""
    import shutil as _sh
    d = _tmp_repo()
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    for rel in ("runs/roster.json", "runs/tasks.jsonl", "runs/review.jsonl", "runs/board.jsonl"):
        src = os.path.join(ROOT, rel)
        if os.path.exists(src):
            _sh.copy(src, os.path.join(d, rel))
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    _sh.copy(os.path.join(ROOT, "scripts", "board.py"), os.path.join(d, "scripts", "board.py"))
    rp = os.path.join(d, "runs", "roster.json")
    with open(rp, encoding="utf-8") as fh:
        roster = json.load(fh)
    roster["members"].append({"name": "ghost", "role": "research", "socket": "uds:/tmp/none.sock",
                              "topics": [], "note": "fixture"})
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(roster, fh, ensure_ascii=False, indent=1)
    tp = os.path.join(d, "runs", "tasks.jsonl")
    with open(tp, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    seed = dict(rows[-1], id="ghost-1", owner="ghost", state="open",
                opened="2026-08-31 00:00", closed=None, commit=None, evidence=None)
    with open(tp, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(seed, ensure_ascii=False) + "\n")
    return d


def _selftest_peer_stalled_names_the_fixture():
    """peer_stalled names the stalled member and not the working ones, both directions.

    The broken world alone does not prove this: it has no branches, so it warns on ledger
    age for real members too, and "WARN happened" would pass with the check reading only
    one field. Two worlds over one roster -- with the fixture member and without -- plus
    the real tree, where every member has a fresh branch.
    """
    import shutil as _sh

    d = _broken_peer_stalled()
    state, ev = check_peer_stalled(d)
    assert state == WARN and "ghost" in ev, f"the fixture member was not named: {state} {ev}"

    # Same world, fixture removed from the roster: ghost must disappear from the message.
    rp = os.path.join(d, "runs", "roster.json")
    with open(rp, encoding="utf-8") as fh:
        roster = json.load(fh)
    roster["members"] = [m for m in roster["members"] if m["name"] != "ghost"]
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(roster, fh, ensure_ascii=False, indent=1)
    _s2, ev2 = check_peer_stalled(d)
    assert "ghost" not in ev2, f"ghost is named after leaving the roster: {ev2}"

    # A COMMIT IS ENOUGH TO CLEAR A STALL EVEN WITH AN OLD LEDGER ROW, which is the whole
    # reason liveness reads both. Give the world a git repo with a `ghost` branch committed
    # now, restore the fixture, and the member must stop being named.
    with open(rp, "w", encoding="utf-8") as fh:
        roster["members"].append({"name": "ghost", "role": "research", "socket": "uds:/x.sock",
                                  "topics": [], "note": "fixture"})
        json.dump(roster, fh, ensure_ascii=False, indent=1)
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_COMMON_DIR")}
    for args in (["init", "-q", "-b", "ghost"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"], ["add", "runs/roster.json"],
                 ["commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", d, *args], capture_output=True, env=env)
    _s3, ev3 = check_peer_stalled(d)
    assert "ghost" not in ev3, (
        f"a member with a commit made seconds ago is still called stalled: {ev3} -- "
        f"liveness is reading only the ledger")

    # And the real tree: nobody is stalled today, so a check that always warns fails here.
    state_real, ev_real = check_peer_stalled(ROOT)
    assert state_real in (PASS, WARN), (state_real, ev_real)
    _sh.rmtree(d, ignore_errors=True)
    print("  peer_stalled: fixture named; unnamed once off the roster; a fresh commit clears "
          "it despite an old ledger row")


def check_review_present(root):
    """Every done task has a review row from the reviewer it named.

    A delivery with one reader is a delivery nobody checked: the controller review
    caught four evidenced errors in a day while every other session's work shipped
    unread (user order, 2026-08-31 22:00). The row closes on --reviewer alone, so a
    sleeping reviewer never blocks the register; the review itself is due within
    REVIEW_GRACE_MIN of the close. Inside the window a missing review WARNs, after it
    FAILs -- the same logic as the 15-minute challenge rule, where a challenge that
    does not arrive does not block the ruling but its absence is not invisible."""
    tasks = _read_tasks(os.path.join(root, "runs", "tasks.jsonl"))
    done = [t for t in tasks if t.get("state") == "done"]
    in_scope = [t for t in done if (t.get("closed") or "") >= REVIEW_RULE_FROM]
    if not in_scope:
        return SKIP, f"no task closed since the rule took effect ({REVIEW_RULE_FROM})"
    reviews = {}
    p = os.path.join(root, "runs", "review.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # ledgers_one_line_per_row owns malformed rows
            if r.get("task"):
                reviews.setdefault(r["task"], []).append(r)
    now = time.time()
    overdue, pending, no_reviewer = [], [], []
    for t in done:
        tid = t.get("id")
        if (t.get("closed") or "") < REVIEW_RULE_FROM:
            continue  # closed before the rule; see REVIEW_RULE_FROM
        named = t.get("reviewer")
        task_reviews = reviews.get(tid, [])
        if not named:
            # pre-rule rows named no reviewer; any peer review row or a
            # legacy-unreviewed declaration (artifacts gone, 44-28) closes them
            if task_reviews:
                continue
            no_reviewer.append(tid)
            continue
        if any(r.get("reviewer") == named for r in task_reviews):
            continue
        if any(r.get("verdict") == "legacy-unreviewed" for r in task_reviews):
            continue  # artifacts gone, declared in the ledger (44-28)
        closed = t.get("closed", "")
        try:
            age_min = (now - time.mktime(time.strptime(closed, "%Y-%m-%d %H:%M"))) / 60
        except ValueError:
            age_min = REVIEW_GRACE_MIN + 1  # unparseable timestamp: treat as due
        (overdue if age_min > REVIEW_GRACE_MIN else pending).append(f"{tid}->{named}")
    if no_reviewer:
        return WARN, f"{len(no_reviewer)} done task(s) name no reviewer: {no_reviewer[:4]}"
    if overdue:
        return WARN, f"{len(overdue)} review(s) over {REVIEW_GRACE_MIN}min overdue: {overdue[:4]}"
    if pending:
        return WARN, f"{len(pending)} review(s) pending inside the {REVIEW_GRACE_MIN}min window: {pending[:4]}"
    return PASS, f"{len(in_scope)} task(s) closed since {REVIEW_RULE_FROM}, every one reviewed by the peer it named"


def _broken_review_present():
    """The REAL register and review ledger with one review row removed."""
    d = _tmp_repo()
    tp = os.path.join(ROOT, "runs", "tasks.jsonl")
    rp = os.path.join(ROOT, "runs", "review.jsonl")
    if not (os.path.exists(tp) and os.path.exists(rp)):
        return None
    reviews = []
    for line in open(rp, encoding="utf-8"):
        if line.strip():
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    reviewed = [r for r in reviews if r.get("task")]
    if not reviewed:
        raise SelftestSkip("no task-linked review rows yet; the check SKIPs the same way")
    import shutil as _sh

    drop = reviewed[0]["task"]
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    _sh.copy(tp, os.path.join(d, "runs", "tasks.jsonl"))
    with open(os.path.join(d, "runs", "review.jsonl"), "w", encoding="utf-8") as f:
        for r in reviews:
            if r.get("task") != drop:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # a done row naming NO reviewer with no review row of any kind: must WARN (44-28)
    with open(os.path.join(d, "runs", "tasks.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "selftest-norev", "state": "done", "owner": "t",
                            "closed": "2026-09-02 00:00"}) + "\n")
    return d


def check_ledgers_one_line_per_row(root):
    """Every union-merged ledger holds one JSON object per physical line.

    .gitattributes merges these by union, which concatenates lines. A pretty-printed
    row spans many lines, so union interleaves two branches' rows into syntactically
    broken JSON and row identity silently becomes position. 3b's retro row was
    pretty-printed across lines 3-12 (2026-08-31): 9 of 18 lines unparseable, and any
    merge touching it would have corrupted the neighbouring rows too."""
    bad = []
    checked = 0
    for rel in _union_ledgers(root):
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        checked += 1
        for n, line in enumerate(open(p, encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad.append(f"{rel}:{n}")
                continue
            if not isinstance(obj, dict):
                bad.append(f"{rel}:{n} is {type(obj).__name__}, not an object")
    if not checked:
        return SKIP, "no union ledgers present"
    if bad:
        return FAIL, f"{len(bad)} line(s) not one JSON object: {bad[:4]}"
    return PASS, f"{checked} union ledger(s), one JSON object per line"


def _broken_ledgers_one_line_per_row():
    """The REAL retro ledger with one row pretty-printed -- 3b's actual failure."""
    d = _tmp_repo()
    src = os.path.join(ROOT, "runs", "retro.jsonl")
    if not os.path.exists(src):
        return None
    # Raw lines, not json.loads: the real ledger may already hold a malformed row
    # (3b's, today), and a broken world that cannot be built while the bug is live
    # is a broken world that only works after the fix.
    lines = [x for x in open(src, encoding="utf-8") if x.strip()]
    good = None
    for x in lines:
        try:
            good = json.loads(x)
            break
        except json.JSONDecodeError:
            continue
    if good is None:
        return None
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    # .gitattributes too: the ledger list is derived from it, so a world without it
    # has no ledgers and the check SKIPs instead of failing.
    ga = os.path.join(ROOT, ".gitattributes")
    if not os.path.exists(ga):
        return None
    import shutil as _sh

    _sh.copy(ga, os.path.join(d, ".gitattributes"))
    with open(os.path.join(d, "runs", "retro.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(good, ensure_ascii=False, indent=1) + "\n")  # the pretty-printed row
    return d


def check_facts_well_formed(root):
    """Every fact carries its measurement config, and its source path must exist.

    Source standard: a fact's source is the command that produced it PLUS a durable
    artifact (score_matrix record / preds file / commit sha). runs/*.log is NOT a
    qualified source -- logs are deleted, overwritten, or replaced by the next
    same-named run: four facts cited logs a full-disk find on the pod could not produce
    (de-16, 2026-09-02). Committing a log makes the path RESOLVE, not the source qualify.
    Measured: overwriting runs/ab_vocab.log in the working tree left
    `git show HEAD:runs/ab_vocab.log` byte-identical at sha256 923862e4 -- so what a
    commit preserves is the blob, reachable only by sha, while the cited path now holds
    the next run's bytes. A citation naming the path still reads the wrong content, and
    this check cannot see the difference. Enforcing tracked-ness was tried and dropped:
    `git ls-files` returns nothing usable in a broken world with no real .git, so the
    branch could not be made to FAIL, and a check that cannot fail is an assertion.
    The remedy stays a durable artifact per fact.

    Three-state source check (same shape as corpus_filters_fp):
    - path exists -> OK
    - path missing AND gitignored -> SKIP (pod-only artifact; this machine doesn't
      have it, the pod does)
    - path missing AND in the baseline -> registered debt, reported in `gaps`
    - path missing AND not in baseline -> FAIL (new rot)

    The baseline is a ratchet: it can only shrink. A new missing source is a FAIL,
    never a silent baseline addition."""
    facts_dir = os.path.join(root, "facts")
    if not os.path.isdir(facts_dir):
        return FAIL, "facts/ does not exist -- measurements have nowhere to carry their config"
    files = sorted(
        p for p in glob.glob(os.path.join(facts_dir, "*.json"))
        if os.path.basename(p)
        not in (os.path.basename(FACT_SOURCE_BASELINE), os.path.basename(CORPUS_FILTERS_BASELINE))
    )
    if not files:
        return FAIL, "facts/ holds no *.json"
    baseline_path = os.path.join(root, FACT_SOURCE_BASELINE)
    source_baseline = json.load(open(baseline_path, encoding="utf-8")) if os.path.exists(baseline_path) else {}
    errors, ids, entries = [], {}, []
    baselined = []
    pending = []  # (tag, path, rev-or-None) for every source path absent from the tree
    for p in files:
        fn = os.path.basename(p)
        try:
            lst = json.load(open(p, encoding="utf-8"))["facts"]
            assert isinstance(lst, list) and lst
        except Exception as e:
            errors.append(f"{fn}: no readable non-empty 'facts' list ({e})")
            continue
        for e in lst:
            if not isinstance(e, dict):
                errors.append(f"{fn}: entry is not an object")
                continue
            tag = f"{fn}#{e.get('id', '?')}"
            if missing := FACT_REQUIRED - e.keys():
                errors.append(f"{tag}: missing {sorted(missing)}")
                continue
            if e["status"] not in FACT_STATUS:
                errors.append(f"{tag}: bad status {e['status']!r}")
            if not isinstance(e["config"], dict) or not e["config"]:
                errors.append(f"{tag}: config must be a non-empty object")
            if not FACT_DATE_RE.fullmatch(str(e["measured"])):
                errors.append(f"{tag}: measured must be YYYY-MM-DD, got {e['measured']!r}")
            if e["status"] in FACT_NEEDS_CLAIM:
                for k in ("claim", "audit", "refuted_by"):
                    if not e.get(k):
                        errors.append(f"{tag}: {e['status']} fact needs {k}")
            # RETRACTED ONLY, and `[]` counts as present -- the absence of the KEY is the defect,
            # not an empty list. Five of nine retracted entries have no dead number (see
            # FACT_RETRACTED_VALUE), so requiring a non-empty list would force someone to invent
            # one. `is None` rather than a falsy test for exactly that reason.
            if e["status"] == "retracted":
                rv = e.get(FACT_RETRACTED_VALUE)
                if rv is None:
                    errors.append(
                        f"{tag}: retracted fact needs {FACT_RETRACTED_VALUE} -- the list of values "
                        f"that are WRONG, as data. Use [] when the conclusion was retracted but "
                        f"its numbers stand (5 of 9 entries), which is a statement, not a gap")
                elif not isinstance(rv, list) or any(not isinstance(x, str) for x in rv):
                    errors.append(f"{tag}: {FACT_RETRACTED_VALUE} must be a list of strings, "
                                  f"got {type(rv).__name__}")
                else:
                    # A value listed as dead must appear in the entry, or the two disagree about
                    # what was retracted and neither can be trusted. Checked against value+claim
                    # because a retraction rewrites `value` into narration and the original number
                    # often survives only in `claim`.
                    hay = f"{e.get('value', '')} {e.get('claim', '')}"
                    for x in rv:
                        if x not in hay:
                            errors.append(
                                f"{tag}: {FACT_RETRACTED_VALUE} lists {x!r}, which appears in "
                                f"neither value nor claim -- a dead value nobody can find is not "
                                f"a retraction anyone can act on")
            if e["id"] in ids:
                errors.append(f"duplicate id {e['id']!r} in {fn} and {ids[e['id']]}")
            ids[e["id"]] = fn
            # Source-path half: a full-checkout check. The pod is a partial checkout (the
            # manifest's executing files, not the repo), so a path missing there is not rot
            # -- it was never there. CI and dev run this fully; the pod skips it. The config
            # half above runs everywhere.
            if not pod_drift.is_pod(root):
                src = str(e["source"])
                for m in FACT_SOURCE_PATH.findall(src):
                    if os.path.exists(os.path.join(root, m)):
                        continue
                    # `path@rev` is the repo's retirement form for a deleted file: the
                    # content is reachable at that sha, so the citation resolves even
                    # though the path does not. launch_gate learned this in 4676118; this
                    # check did not, and it FAILed on the seven retirements of de-21 while
                    # 44's 22 in 30b9010 passed only because `probes/` is absent from
                    # FACT_SOURCE_PATH's directory list -- the same deletions in eval/ or
                    # scripts/ would have failed. Verify the rev, do not just accept the
                    # syntax: a sha that names nothing is a dead citation wearing the
                    # durable form.
                    # COLLECTED, NOT PROBED HERE. Every git probe in this loop was one
                    # subprocess, and the loop runs once per fact source: 34 cat-file plus
                    # 50 check-ignore calls, 0.84 s of the check's 1.49 s, and the count
                    # grows with facts/. Two batched calls after the loop cost 0.04 s.
                    rev = re.search(re.escape(m) + r"@([0-9a-f]{7,40})\b", src)
                    pending.append((tag, m, rev.group(1) if rev else None))
            entries.append((fn, e))
    # Resolve every collected candidate in two subprocesses, then apply the SAME order of
    # precedence the loop used: rev holds the path -> gitignored -> baselined -> error.
    if pending:
        rev_specs = [f"{r}:{m}" for _tag, m, r in pending if r]
        rev_ok = _cat_file_exists(root, rev_specs) if rev_specs else {}
        ignored = _gitignored_set([m for _tag, m, _r in pending], root)
        for tag, m, rev in pending:
            if rev and rev_ok.get(f"{rev}:{m}"):
                continue
            if ignored.get(m):
                continue  # pod-only artifact; this machine doesn't have it
            if m in source_baseline:
                baselined.append(m)
                continue  # registered debt; gaps reports it
            errors.append(f"{tag}: source path {m} does not exist (not in baseline)")
    agents = os.path.join(root, "AGENTS.md")
    prose = open(agents, encoding="utf-8").read() if os.path.exists(agents) else ""
    for fn, e in entries:
        for phrase in e.get("guard_phrases", []):
            if phrase in prose:
                errors.append(f"AGENTS.md asserts {phrase!r}, recorded as {e['status']} in {fn}#{e['id']}")
    for p in files:
        if os.path.basename(p) not in prose:
            errors.append(f"AGENTS.md never mentions {os.path.basename(p)} -- an orphan fact file")
    for m in re.findall(r"facts/[\w.-]+\.json", prose):
        if not os.path.exists(os.path.join(root, m)):
            errors.append(f"AGENTS.md cites {m}, which does not exist")
    if errors:
        head = "; ".join(errors[:5])
        return FAIL, head + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else "")
    note = f"; {len(set(baselined))} baselined source(s) (debt register, see `harness gaps`)" if baselined else ""
    # State the population, not just the count. "every entry carries its config" over
    # 7 files reads identically whether 7 is all of them or 7 of 9 -- and it IS 7 of 9
    # here, the two baselines being deliberately excluded. A universal quantifier over
    # a self-constructed population is only as true as the construction, and the
    # reader cannot audit the construction from a bare N (fb's sweep, 2026-09-01,
    # after selftests_are_gated reported "27 files, all gated" over a real 36).
    _all = len(glob.glob(os.path.join(facts_dir, "*.json")))
    _pop = f"{len(files)} of {_all} facts/*.json" if _all != len(files) else f"all {_all} facts/*.json"
    return PASS, (f"{len(entries)} facts in {_pop} (baselines excluded), every entry "
                  f"carries its config{note}")


def _broken_facts():
    """The REAL facts files and REAL AGENTS.md, with one entry's config deleted and
    one entry's source pointing at a non-existent scripts/ path. A hand-written file
    would share the check's own assumptions.

    The source mutation uses a path under scripts/: a data/ path would be gitignored by
    data/*.jsonl and silently SKIPped, so it must be one the three-state check treats
    as FAIL. That coverage is what the world lacked when the source regex had no left
    anchor and no data/ prefix.

    The code and docs directories are symlinked in because OTHER entries' sources cite
    them. Without docs/, the world FAILed on `docs/lessons/base_eval_at_200m.md does not
    exist` for facts nobody mutated -- so the selftest was green on absence, and would
    have stayed green with both mutations removed. Verified by removing them
    (de, 2026-09-01).

    Four mutations now, the last two for retracted_value: one entry's key deleted, and one
    entry's list naming a value the entry does not contain. See the comments at each."""
    import shutil

    d = _tmp_repo_shaped()
    os.makedirs(os.path.join(d, ".git"), exist_ok=True)  # full checkout: the pod skips the path half
    os.remove(os.path.join(d, "facts"))
    os.makedirs(os.path.join(d, "facts"))
    for f in glob.glob(os.path.join(FACTS_DIR, "*.json")):
        shutil.copy(f, os.path.join(d, "facts"))
    obj = json.load(open(os.path.join(d, "facts", "tokenizer.json"), encoding="utf-8"))
    del obj["facts"][0]["config"]
    obj["facts"][0]["source"] = "scripts/no_such_script_xyz.py"
    json.dump(obj, open(os.path.join(d, "facts", "tokenizer.json"), "w"))
    # THE RETRACTED-VALUE HALF, on the real retracted entries. Two mutations, because the
    # branch has two ways to be wrong and one world that exercised only the missing key would
    # leave the disagreement half unguarded:
    #   1. the key deleted -> "retracted fact needs retracted_value"
    #   2. a value listed that appears in neither `value` nor `claim` -> the disagreement error.
    # Mutation 2 is the one that matters in practice: the field is hand-maintained, so the way
    # it rots is someone editing `value` and leaving the list behind, and a check that only
    # asserts presence would call that entry well-formed. Both are on REAL entries -- the
    # entries whose `value` narrates its own retraction are exactly what the field is for.
    cf = os.path.join(d, "facts", "efficiency.json")
    obj2 = json.load(open(cf, encoding="utf-8"))
    hit = [e for e in obj2["facts"] if e.get("status") == "retracted"]
    if len(hit) < 2:
        raise SelftestSkip(f"facts/efficiency.json holds {len(hit)} retracted entries; the world "
                           "needs two to break both halves of the retracted_value branch")
    hit[0].pop("retracted_value", None)
    hit[1]["retracted_value"] = ["1234.5678 no such number in this entry"]
    json.dump(obj2, open(cf, "w"))
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    return d


# The POPULATION, not just the predicate. This hand-listed four directories, so the
# entry-point table's own FIRST row -- train.py sft.py sft_math.py serve.py chat.py
# infer.py, the repo's actual entry points -- was invisible to a check whose evidence
# says "every tried entry-point command". Widened to any path ending .sh/.py, which is
# what a table row citing a script looks like regardless of where it lives. Four more
# rows come into scope and the cited-file-exists tier stays green on all of them; the
# citations are bare basenames (exp.py, tokenizer_report.py) in prose rows, so the
# existence tier is resolved against scripts/ as well as the root (fb's sweep for
# universals over self-built populations, de 2026-09-01).
ENTRY_SCRIPT_RE = re.compile(r"(?:[\w.-]+/)?[\w.-]+\.(?:sh|py)")
# Where a bare basename in a prose row may live. A row saying `exp.py done` cites
# scripts/exp.py; resolving only against the root would call a real script missing.
ENTRY_SEARCH_DIRS = ("", "scripts", "eval", "datagen", "probes", "mathbank", "algorithms", "filters")


def _entry_exists(root, s):
    """A cited path with a directory must exist AT that path -- `scripts/foo.py` naming a
    file that actually lives in eval/ is exactly the doc rot this tier catches. Only a
    bare basename in a prose row (`exp.py done`) is searched, because prose does not
    carry a directory to be wrong about."""
    if "/" in s:
        return os.path.exists(os.path.join(root, s))
    return any(os.path.exists(os.path.join(root, d, s)) for d in ENTRY_SEARCH_DIRS)


def check_unreached_files_ruled(root):
    """Every file reachability reports as unreached carries a FATE ruling (de-5).

    The listing is not a deletion oracle and never was, so the value of a scan is that
    somebody RULED on each name it raises: keep and make reachable, or delete. Without a
    check, the unruled ones accumulate -- 25 of them by 2026-09-03, and 23 were last touched
    that day or the day before, so this is a live accumulation and not a backlog anyone
    finishes once.

    WARN, not FAIL: an unruled file is a to-do, and the fix is a person reading it and
    deciding. A FAIL would go red the moment anyone adds a probe, which trains people to
    bypass rather than to rule.

    This replaces a one-off classifier script. That script sorted the scan's output AFTER the
    fact, so its 21 false candidates came back on every run and needed a person to re-read
    them each time; the real fix was teaching the scan the edge it could not see (the
    pre-commit hook's SELFTEST_FILES), which took 46 unreached to 25. A check on what remains
    is the part worth keeping every commit.

    READS THE COMMITTED LISTING, not a fresh scan. Running reachability.py here costs more
    than the 5 s check budget and timed out on the first attempt. The listing cannot go stale
    behind this check: scripts/test_reachability_fresh.py asserts runs/reachability.txt
    matches a live scan, it is in the hook's SELFTEST_FILES, and it fired on exactly that
    during this change."""
    listing = os.path.join(root, "runs", "reachability.txt")
    if not os.path.exists(listing):
        return SKIP, "runs/reachability.txt not present (run scripts/reachability.py > it)"
    unruled = []
    for ln in open(listing, encoding="utf-8"):
        # A row whose REACHED FROM is `none` and whose FATE column is empty. The fate text
        # is the last column, so a row with a ruling has something after `none`.
        if re.search(r"\s+none\s*$", ln.rstrip("\n")):
            unruled.append(ln.split()[0])
    if not unruled:
        return PASS, "every unreached file in runs/reachability.txt carries a FATE ruling"
    # NAME ALL OF THEM. This was `unruled[:6]` with a ` ...`, and the two it elided were the
    # two nobody could act on: a WARN is a to-do list, and an item a reader cannot see is an
    # item nobody works off. Measured 2026-09-03: 8 unruled, the printed 6 stopped at
    # datagen/cot_pilot.py, and datagen/sft_math_share.py and sft_sample_200_eqcheck.py were
    # invisible until the listing was read by hand. The list is bounded by the tree's own
    # unreached count, so there is no runaway to cap.
    return WARN, (f"{len(unruled)} unreached file(s) with no FATE ruling in "
                  f"scripts/reachability.py: {', '.join(unruled)}"
                  f" -- run each before judging it "
                  f"(a hook-registered or glob-loaded file is live and invisible here), then "
                  f"add KEEP or delete it")


def _broken_unreached_files_ruled():
    """The REAL listing with one ruled file's FATE text removed, so a file that IS ruled today
    reports unruled. Mutating the real artifact rather than writing a world: a hand-written
    listing would share this check's own idea of the column format."""
    d = _tmp_repo()
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    src = os.path.join(ROOT, "runs", "reachability.txt")
    if not os.path.exists(src):
        raise SelftestSkip("runs/reachability.txt absent; nothing real to mutate")
    out, hit = [], False
    for ln in open(src, encoding="utf-8"):
        m = re.match(r"^(\S+\.(?:py|sh))(\s+.*?)(KEEP|DELETE)\b.*$", ln.rstrip("\n"))
        if m and not hit:
            # Same row, ruling stripped and the REACHED FROM forced to `none`: this is what
            # an unruled file looks like, built from a real row.
            out.append(f"{m.group(1):<56} 1  deadbeef 2026-09-03   none\n")
            hit = True
        else:
            out.append(ln)
    if not hit:
        raise SelftestSkip("no ruled row in runs/reachability.txt to strip")
    with open(os.path.join(d, "runs", "reachability.txt"), "w", encoding="utf-8") as fh:
        fh.writelines(out)
    return d


def check_entrypoints_ran(root):
    """A cited script that does not exist is FAIL -- the doc is rotten. A command tried and
    never ok is WARN -- a to-do fixed by running it. Zero log matches is skipped: never
    tried is not tried and failed (wrappers log the inner command)."""
    agents = os.path.join(root, "AGENTS.md")
    if not os.path.exists(agents):
        return SKIP, "AGENTS.md not present"
    log = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(log):
        return SKIP, "runs/experiments.jsonl not present"
    rows = []
    for line in open(log, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    if not rows:
        # An empty log is the post-reset state, not a clean bill: zero matches must read
        # as "never tried", never as PASS.
        return SKIP, "runs/experiments.jsonl has no rows"
    missing, stale = [], []
    n_rows = 0
    for line in open(agents, encoding="utf-8"):
        if "|" not in line or not ENTRY_SCRIPT_RE.search(line):
            continue
        n_rows += 1
        # Task-cell tokens catch attempts logged under an inner command (the wrapper is
        # invisible to the log).
        task_tokens = {t for t in re.split(r"[^a-z0-9]+", line.split("|")[1].lower()) if len(t) >= 5}
        for s in sorted(set(re.findall(r"[\w/.-]+\.(?:sh|py)", line))):
            if not _entry_exists(root, s):
                missing.append(s)
                continue
            matched = [
                r
                for r in rows
                if s in str(r.get("cmd", ""))
                or any(
                    t in str(r.get("name", "")).lower() or t in str(r.get("cmd", "")).lower()
                    for t in task_tokens
                )
            ]
            if matched and not any(r.get("status") == "ok" for r in matched):
                latest = matched[-1]
                finding = " ".join(str(latest.get("finding", "")).split())[:120]
                stale.append(
                    f"{s}: {len(matched)} run(s), never ok, latest={latest.get('status')!r}"
                    + (f" -- {finding}" if finding else "")
                )
    if missing:
        return FAIL, f"entry-point table cites script(s) not in the repo: {missing}"
    if stale:
        return WARN, "; ".join(stale[:4])
    return PASS, f"{n_rows} script-citing row(s) in AGENTS.md; every tried entry-point command has an ok run"


def check_entrypoints_table_present(root):
    """The entry-point table is the doc's contract with the repo. Zero script-citing rows
    is the cfg_default failure shape: two corpus invariants reported SKIP 'chosen on
    purpose' and check exited 0 -- an empty list silences the guard. FAIL, never SKIP."""
    agents = os.path.join(root, "AGENTS.md")
    if not os.path.exists(agents):
        return SKIP, "AGENTS.md not present"
    n = sum(1 for line in open(agents, encoding="utf-8") if "|" in line and ENTRY_SCRIPT_RE.search(line))
    if n == 0:
        return FAIL, "no entry-point row cites a script -- an empty list silences the guard (cfg_default shape)"
    return PASS, f"{n} entry-point row(s) cite scripts"


def _broken_entrypoint():
    """The REAL AGENTS.md, in a tree holding the REAL scripts it cites, with two rows
    added citing scripts that do not exist. Two, because the population widened:
    `scripts/ghost_command.sh` is a pathed citation and `ghost_prose_only.py` is a bare
    basename in a prose row, which the old four-directory predicate could not see.

    The empty `_tmp_repo()` was not a broken world for this check. Every one of the 38
    real citations resolved to nothing there, so the world FAILed with or without a
    ghost -- the selftest was green on 38 false positives and would have stayed green
    if the ghost detection were deleted outright. Confirmed by running the check on the
    same world with both ghosts removed: still FAIL. Symlinking the cited directories
    makes the ghost the only thing wrong, which is what the world has to isolate
    (de, 2026-09-01).

    The WARN tier is live in the real repo (run_ablation.sh), so it needs no synthetic
    world. The log row is written by the REAL logger with --root d, so the check runs
    instead of SKIPping on an absent log."""
    import shutil, subprocess

    d = _tmp_repo()
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    for name in ENTRY_SEARCH_DIRS:
        if name and os.path.isdir(os.path.join(ROOT, name)):
            os.symlink(os.path.join(ROOT, name), os.path.join(d, name))
    for f in os.listdir(ROOT):
        if f.endswith((".py", ".sh")) and not os.path.exists(os.path.join(d, f)):
            os.symlink(os.path.join(ROOT, f), os.path.join(d, f))
    with open(os.path.join(d, "AGENTS.md"), "a") as f:
        f.write("| Ghost | `python scripts/ghost_command.sh` |\n")
        f.write("| Ghost prose | a rule enforced by `ghost_prose_only.py`, nowhere in the tree |\n")
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "exp.py"),
            "--root",
            d,
            "start",
            "--name",
            "broken_world",
            "--cmd",
            "./run_ddp.sh --mix x",
        ],
        check=True,
        capture_output=True,
    )
    return d


def _broken_entrypoints_table():
    """The REAL AGENTS.md with every script-citing table row deleted -- the check must
    FAIL, not SKIP. The doc carries several script-citing tables, so deleting only the
    entry-point block would leave the count above zero."""
    d = _tmp_repo()
    lines = open(os.path.join(ROOT, "AGENTS.md"), encoding="utf-8").read().splitlines(keepends=True)
    out = [ln for ln in lines if not ("|" in ln and ENTRY_SCRIPT_RE.search(ln))]
    with open(os.path.join(d, "AGENTS.md"), "w", encoding="utf-8") as f:
        f.writelines(out)
    return d


# --------------------------------------------------------------------------- docs layout
#
# docs/ has three subdirectories and zero .md files at its root. Research docs carry
# question/status/source frontmatter; lessons cite facts by facts/<file>.json#<id>.

DOCS_SUBDIRS = ("lessons", "audits")
FRONTMATTER_KEYS = ("question", "status", "source")
FRONTMATTER_STATUS = ("measured", "recorded", "open", "retracted")
#: A trailing `.` is sentence punctuation, not part of the id. Fact ids contain dots
#: (`dq.audit.protocol_400`), so the id class has to accept them -- but `[\w.]+` is greedy
#: and swallows the period that ends the sentence, turning a live citation into
#: `be.math_v2_likelihood_twin.` and a FAIL that names an id nobody wrote. Latent when
#: found, not live: zero of the 72 current doc citations end a sentence, so the check would
#: have gone wrong on the first one that did (de-16, 2026-09-02, after writing the same bug
#: into a one-off scanner and getting a false positive out of it -- the repo's own
#: greedy-regex-over-JSONL lesson, one field over).
FACT_REF_RE = re.compile(r"facts/([\w.-]+)\.json#([\w.]*[\w])")
CMD_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\n(.*?)```", re.S)
CMD_PATH_RE = re.compile(r"(?<![\w.-])([\w./-]+\.(?:sh|py))(?![\w.-])")


def _frontmatter(path):
    text = open(path, encoding="utf-8").read()
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None
    fields = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()
    return fields


def check_docs_root_clean(root):
    docs = os.path.join(root, "docs")
    if not os.path.isdir(docs):
        return FAIL, "docs/ missing"
    stray = sorted(f for f in os.listdir(docs) if f.endswith(".md") and os.path.isfile(os.path.join(docs, f)))
    if stray:
        return FAIL, f"docs/ root holds .md files: {stray[:5]} -- classify into lessons/, audits/, standards/"
    return PASS, "docs/ root holds no .md files"


def check_lessons_frontmatter(root):
    docs = os.path.join(root, "docs")
    if not os.path.isdir(docs):
        return FAIL, "docs/ missing"
    problems, n = [], 0
    for sub in DOCS_SUBDIRS:
        d = os.path.join(docs, sub)
        if not os.path.isdir(d):
            problems.append(f"docs/{sub}/ missing")
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".md") or f.startswith("README"):
                continue
            n += 1
            fm = _frontmatter(os.path.join(d, f))
            if fm is None:
                problems.append(f"docs/{sub}/{f}: no frontmatter")
                continue
            missing = [k for k in FRONTMATTER_KEYS if not fm.get(k)]
            if missing:
                problems.append(f"docs/{sub}/{f}: missing {missing}")
            elif fm["status"] not in FRONTMATTER_STATUS:
                problems.append(f"docs/{sub}/{f}: bad status {fm['status']!r}")
    if problems:
        return FAIL, "; ".join(problems[:5])
    if n == 0:
        return FAIL, "no lesson/audit files found -- an empty list silences the guard"
    return PASS, f"{n} research docs carry question/status/source"


def check_fact_refs(root):
    facts_dir = os.path.join(root, "facts")
    if not os.path.isdir(facts_dir):
        return FAIL, "facts/ missing"
    index = {}
    for f in glob.glob(os.path.join(facts_dir, "*.json")):
        try:
            obj = json.load(open(f, encoding="utf-8"))
            index[os.path.basename(f)] = {e["id"]: e for e in obj.get("facts", [])}
        except Exception as e:
            return FAIL, f"cannot parse {f}: {e}"
    bad, retracted, n = [], [], 0
    for sub in DOCS_SUBDIRS:
        d = os.path.join(root, "docs", sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".md"):
                continue
            for m in FACT_REF_RE.finditer(open(os.path.join(d, f), encoding="utf-8").read()):
                n += 1
                fname, fid = m.group(1) + ".json", m.group(2)
                if fname not in index:
                    # fname already carries .json; appending it again printed
                    # "facts/base_eval.json.json does not exist", a path nobody can act on.
                    bad.append(f"docs/{sub}/{f}: facts/{fname} does not exist")
                elif fid not in index[fname]:
                    bad.append(f"docs/{sub}/{f}: {fid} not in facts/{fname}")
                elif index[fname][fid].get("status") == "retracted":
                    retracted.append(f"docs/{sub}/{f} cites retracted {fname}#{fid}")
    if bad:
        return FAIL, "; ".join(bad[:5])
    if retracted:
        return WARN, f"{n} citation(s); " + "; ".join(retracted[:4])
    return PASS, f"{n} fact citation(s) all resolve"


def _broken_docs_root():
    """The REAL docs tree plus one stray .md at the root -- the rule is zero .md files
    directly under docs/, so any new root file FAILs until classified."""
    import shutil

    d = _tmp_repo()
    shutil.copytree(os.path.join(ROOT, "docs"), os.path.join(d, "docs"))
    open(os.path.join(d, "docs", "stray.md"), "w").write("# stray\n")
    return d


def _broken_lessons_fm():
    """The REAL docs tree with kept_methods.md's frontmatter stripped -- the check must
    FAIL on the missing fields, not on a hand-written file sharing the check's own
    assumptions."""
    import shutil

    d = _tmp_repo()
    shutil.copytree(os.path.join(ROOT, "docs"), os.path.join(d, "docs"))
    p = os.path.join(d, "docs", "lessons", "kept_methods.md")
    text = open(p, encoding="utf-8").read()
    if text.startswith("---\n"):
        text = text[text.find("\n---", 4) + 4 :].lstrip("\n")
    open(p, "w", encoding="utf-8").write(text)
    return d


def _broken_fact_ref():
    """The REAL lessons and facts trees, with one citation to a nonexistent fact appended
    to a real lesson."""
    import shutil

    d = _tmp_repo()
    shutil.copytree(os.path.join(ROOT, "docs"), os.path.join(d, "docs"))
    shutil.copytree(os.path.join(ROOT, "facts"), os.path.join(d, "facts"))
    with open(os.path.join(d, "docs", "lessons", "kept_methods.md"), "a", encoding="utf-8") as f:
        f.write("\n\nSee facts/tokenizer.json#tok.does_not_exist.\n")
    return d


DATA_PATH_RE = re.compile(r"data/[A-Za-z0-9_][A-Za-z0-9_./-]*")


@functools.lru_cache(maxsize=None)
def _tracked_paths(root):
    r = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    return None if r.returncode else frozenset(r.stdout.split("\n"))


def _cited_path_exists(root, tok):
    """A doc-cited data path that resolves. Gitignored artifacts (tokenizer.json, corpus
    bytes) are exempt -- absent from a clean checkout is their normal state; only a
    TRACKED path that is missing is rot. With no git (the pod), disk is the only truth."""
    if os.path.exists(os.path.join(root, tok)):
        return True
    tracked = _tracked_paths(root)
    if tracked is None:
        return False
    prefix = tok.rstrip("/") + "/"
    return tok not in tracked and not any(p.startswith(prefix) for p in tracked)


def _doc_data_paths(root):
    """Every data/ path cited in a doc, as (file, line, path). Templates (<domain>), globs
    (*), and brace expansions ({0.2b,...}) are not one path. A doc that recommends a path
    that does not exist is how 3b ran a wrong fingerprint off README's mix_v3.json."""
    out = []
    for pat in ("*.md", "docs/**/*.md", "data/*.md"):
        for md in glob.glob(os.path.join(root, pat), recursive=True):
            for i, line in enumerate(open(md, encoding="utf-8"), 1):
                for m in DATA_PATH_RE.finditer(line):
                    tok = m.group(0).rstrip("._-/")
                    if not tok or tok == "data":
                        continue
                    nxt = line[m.end()] if m.end() < len(line) else ""
                    if nxt in "{[*?<>":  # brace/glob expansion or template, not one path
                        continue
                    out.append((os.path.relpath(md, root), i, tok))
    return out


def check_doc_commands(root):
    """Every .sh/.py cited in an AGENTS.md command block exists, and every data/ path
    cited in any doc exists. A documented path that does not resolve is worse than none:
    README once recommended data/mix_v3.json, which has never existed, and a session ran
    a wrong fingerprint because of it. Only fenced blocks are scanned for scripts; prose
    citations of data files are scanned across all docs.

    Pod SKIP: the pod is a partial checkout (the manifest's executing files, not the
    repo), so a path missing there is not rot -- it was never there. CI and dev run
    this fully, where the files actually exist."""
    if pod_drift.is_pod(root):
        return SKIP, ("partial checkout: the pod holds the manifest's executing files, "
                      "not the repo -- doc-cited paths are checked on dev/CI, not here")
    agents = os.path.join(root, "AGENTS.md")
    missing = set()
    if os.path.exists(agents):
        for block in CMD_BLOCK_RE.findall(open(agents, encoding="utf-8").read()):
            for tok in CMD_PATH_RE.findall(block):
                if not os.path.exists(os.path.join(root, tok)):
                    missing.add(tok)
    for f, _ln, tok in _doc_data_paths(root):
        if not _cited_path_exists(root, tok):
            missing.add(f"{f}:{_ln} {tok}")
    if missing:
        return FAIL, f"doc(s) cite path(s) not in the repo: {sorted(missing)[:5]}"
    if not os.path.exists(agents) and not _doc_data_paths(root):
        return SKIP, "no docs present"
    return PASS, "every doc-cited script and data path exists"


# Retired phrases that must not reappear in README. The objective changed 2026-08-30;
# the old Chinese-LLM framing and the removed 1024 window are stale, not historical.
_README_RETIRED = [
    "中文推理模型",
    "sliding window 1024",
    "窗口 1024",
    "32,773",
]


def check_readme_current(root):
    """README reflects the current objective, not a retired one.

    (a) README's first paragraph carries the objective terms from AGENTS.md's first
        heading (derived, not hard-coded).
    (b) No guard phrase from facts/*.json and no retired phrase appears in README.
    (c) Every command block in README cites files that exist (same as doc_commands)."""
    readme = os.path.join(root, "README.md")
    agents = os.path.join(root, "AGENTS.md")
    if not os.path.exists(readme):
        return SKIP, "no README.md"
    if not os.path.exists(agents):
        return SKIP, "no AGENTS.md"
    text = open(readme, encoding="utf-8").read()
    # (a) objective terms from AGENTS.md's first heading
    heading = open(agents, encoding="utf-8").readline().strip()
    # Extract the objective part: after "—", before "("
    obj = heading.split("—", 1)[-1].split("(")[0].strip() if "—" in heading else heading.lstrip("# ").strip()
    # Content words: len > 3, lowercased, from the objective
    stop = {"with", "from", "that", "this", "have", "will", "been", "they", "their",
            "optional", "attention", "residuals", "gated", "hybrid"}
    terms = set()
    for w in obj.split():
        w = w.lower().strip(",.()—")
        if len(w) > 3 and w not in stop:
            terms.add(w)
    # First paragraph: skip the leading "# title" line
    lines = text.split("\n")
    body_start = 1 if lines and lines[0].startswith("# ") else 0
    first_para = "\n".join(lines[body_start:]).split("\n\n")[0].lower()
    missing = [t for t in terms if t not in first_para]
    if missing:
        return FAIL, f"README first paragraph missing objective terms from AGENTS.md heading: {missing}"
    # (b) guard phrases and retired phrases
    bad = []
    for gf in _read_guard_phrases(root):
        if gf in text:
            bad.append(f"guard phrase: {gf!r}")
    for rp in _README_RETIRED:
        if rp in text:
            bad.append(f"retired: {rp!r}")
    if bad:
        return FAIL, "; ".join(bad[:3])
    # (c) command blocks cite existing files
    missing_cmds = set()
    for block in CMD_BLOCK_RE.findall(text):
        for tok in CMD_PATH_RE.findall(block):
            if not os.path.exists(os.path.join(root, tok)):
                missing_cmds.add(tok)
    if missing_cmds:
        return FAIL, f"README cites path(s) not in the repo: {sorted(missing_cmds)[:5]}"
    return PASS, "README matches the current objective; no retired or guard phrases"


def _read_guard_phrases(root):
    """All guard_phrases entries from facts/*.json."""
    out = []
    for f in glob.glob(os.path.join(root, "facts", "*.json")):
        try:
            data = json.load(open(f, encoding="utf-8"))
            for entry in data.values() if isinstance(data, dict) else data:
                if isinstance(entry, dict) and isinstance(entry.get("guard_phrases"), list):
                    out.extend(entry["guard_phrases"])
        except (json.JSONDecodeError, OSError):
            pass
    return out


def _broken_readme_current():
    """The REAL README with a retired phrase spliced back in -- the FAIL tier.

    The world also needs the code directories: tier (c) resolves every path README's
    command blocks cite, and without them the world FAILed on `./run_ddp.sh`,
    `eval/score_matrix.py` and the rest whether or not the phrase was spliced. Same
    defect as entrypoints_ran's world (de, 2026-09-01)."""
    import shutil
    d = _tmp_repo()
    shutil.copy(os.path.join(ROOT, "README.md"), os.path.join(d, "README.md"))
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    for name in ("scripts", "eval", "datagen", "probes", "mathbank", "algorithms",
                 "filters", "docs", "facts"):
        if os.path.isdir(os.path.join(ROOT, name)):
            os.symlink(os.path.join(ROOT, name), os.path.join(d, name))
    for f in os.listdir(ROOT):
        if f.endswith((".py", ".sh")) and not os.path.exists(os.path.join(d, f)):
            os.symlink(os.path.join(ROOT, f), os.path.join(d, f))
    p = os.path.join(d, "README.md")
    text = open(p, encoding="utf-8").read()
    with open(p, "w", encoding="utf-8") as f:
        f.write("A 200M 中文推理模型.\n\n" + text)
    return d


def _broken_doc_commands():
    """The REAL README with one data path swapped to a name that does not exist -- the
    FAIL tier for the data-path half (the script half appends a fake command block).
    The other data paths README cites are created, so the world fails ONLY on the swap."""
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, ".git"), exist_ok=True)  # full checkout: the pod SKIPs this check
    shutil.copy(os.path.join(ROOT, "README.md"), os.path.join(d, "README.md"))
    p = os.path.join(d, "README.md")
    s = open(p, encoding="utf-8").read()
    assert "data/mix_scale_0.2b.json" in s, "real README no longer cites mix_scale_0.2b; update _broken_doc_commands"
    open(p, "w", encoding="utf-8").write(s.replace("data/mix_scale_0.2b.json", "data/mix_scale_nonexistent.json"))
    os.makedirs(os.path.join(d, "data", "corpus", "sample"), exist_ok=True)
    for f in ("data/mix_sample.json", "data/mix_30b.json", "data/mix_scale_0.2b.json",
              "data/tokenizer.json"):
        open(os.path.join(d, f), "w").write("{}")
    with open(os.path.join(d, "README.md"), "a", encoding="utf-8") as f:
        f.write("\n```bash\npython scripts/nonexistent_command.sh --flag\n```\n")
    return d


def _is_probe_mix(root, mix_path):
    """True when a mix is a smoke/probe fixture rather than a real corpus draw.

    Read from the mix itself: a `_comment` disclaiming its content, or a token budget
    far below the smallest ladder point. Not a filename match -- a name test would
    pass the day someone copies the fixture, and would miss a probe that used a
    differently-named one."""
    p = mix_path if os.path.isabs(mix_path) else os.path.join(root, mix_path)
    try:
        obj = json.load(open(p, encoding="utf-8"))
    except Exception:
        return False
    comment = " ".join(obj.get("_comment", [])) if isinstance(obj.get("_comment"), list) \
        else str(obj.get("_comment") or "")
    if "smoke" in comment.lower() or "content is irrelevant" in comment.lower():
        return True
    # 0.2b is the smallest ladder point; anything an order below it is not a real run
    return float(obj.get("total_tokens") or 0) < 2e8


def check_score_matrix(root):
    """Every status=ok training run has a score-matrix record for the checkpoint it
    produced. 'Trained but not scored' must be impossible: an ok row with no matrix
    record is a FAIL, not a gap. Eval/measure rows produce no checkpoint and are
    exempt. The matrix is runs/score_matrix.jsonl, one record per scored checkpoint."""
    log = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(log):
        return SKIP, "runs/experiments.jsonl not present"
    # Fold by (name, started) with a close beating a later start: the ledger is an event
    # log, so one run has a running row and then a terminal one. Reading raw events made
    # a superseded 'ok' outlive the 'fail' that replaced it -- t56_profile, ok 13:34 then
    # fail 13:47, failed this check as an unscored success (2026-08-31). Through
    # _exp_fold since e1-18; it was position-based, so a duplicate start after a close
    # reopened the run and this check then demanded a score for a finished one.
    evs = []
    for line in open(log, encoding="utf-8"):
        try:
            evs.append(json.loads(line))
        except Exception:
            continue
    rows = _exp_fold(evs)
    if not rows:
        return SKIP, "experiments.jsonl has no rows"
    scored = set()
    matrix = os.path.join(root, "runs", "score_matrix.jsonl")
    if os.path.exists(matrix):
        for line in open(matrix, encoding="utf-8"):
            try:
                scored.add(json.loads(line).get("ckpt"))
            except Exception:
                pass
    missing = []
    unverifiable = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        cmd = str(r.get("cmd", ""))
        if not any(t in cmd for t in ("train.py", "sft_math", "rlvr", "run_ddp.sh", "run_sft.sh")):
            continue
        # A probe is not a scoreable training run. The test is the MIX it read, not the
        # run's name: a mix whose own _comment says the content is irrelevant produces a
        # checkpoint whose scores mean nothing, and scoring it would waste a lane slot on
        # a number nobody can interpret. t56_profile, t57_recompile and t57_steady are
        # optimiser probes on mix_smoke_warmup.json; they blocked the ledger sync for two
        # sessions (fb, 2026-09-01).
        m_mix = re.search(r"--mix\s+(\S+)", cmd)
        if m_mix and _is_probe_mix(root, m_mix.group(1)):
            continue
        if "--profile" in cmd or "--profile_steps" in cmd:
            continue  # a torch profiler run stops after a handful of steps
        # A ROW MAY NAME ITS OWN READING instead of a score-matrix record, because for some runs the
        # matrix is a DIFFERENT QUANTITY than the one the row pre-registered. e1_31b_loop_500's
        # reading is a 2x2 humaneval-BPB diagonal against its own mismatch controls; the matrix's
        # domain-loss numbers would be a real record of something nobody asked about, and satisfying
        # this gate with it would cost a card (6e's ruling, 2026-09-04: not that).
        #
        # THE PATH MUST EXIST. Otherwise the field is a way to assert a reading nobody wrote, which
        # is worse than the gap it replaces -- the row would read as "scored by another instrument"
        # with nothing behind it.
        art = r.get("reading_artifact")
        if art:
            if os.path.exists(os.path.join(root, str(art))):
                continue
            missing.append(f"{r.get('name', '?')} reading_artifact {art} does not exist")
            continue
        cand = produced_checkpoint(cmd, str(r.get("name", "?")))
        if cand is None:
            # SILENTLY EXEMPT UNTIL 2026-09-04, and that is what this branch is for. A row whose cmd
            # is prose with no --out, no --name and no bare ckpt_*.pt returns None here and the gate
            # skipped it: e1_31_middle_layer_loop passed that way while its 500-step sibling, whose
            # cmd names --out honestly, was held to the record. The gate was being satisfied by cmd
            # FORMATTING rather than by runs being scored. WARN rather than FAIL because the row may
            # be legitimate -- but it is no longer invisible, and `reading_artifact` is the field
            # that resolves it (6e's ruling: make the escape visible, do not fix it by accident).
            unverifiable.append(str(r.get("name", "?")))
            continue
        if f"{cand}.pt" not in scored:
            missing.append(cand)
    if missing:
        return FAIL, f"ok training run(s) with no score-matrix record: {sorted(set(missing))[:5]}"
    if unverifiable:
        return WARN, (f"{len(unverifiable)} ok training row(s) whose cmd names no checkpoint, so "
                      f"'trained but not scored' cannot be checked for them: "
                      f"{sorted(set(unverifiable))[:5]} -- add reading_artifact: <path> naming the "
                      f"run's own reading, or a cmd that names its --out")
    return PASS, "every ok training run has a score-matrix record"


def _broken_score_matrix():
    """A REAL ok training row, written by the real exp.py, with no score-matrix
    record -- the FAIL tier."""

    d = _tmp_repo()
    for argv in (
        ["start", "--name", "x", "--cmd", "./run_ddp.sh --name x"],
        ["done", "--name", "x", "--status", "ok", "--result", "done"],
    ):
        subprocess.run(
            [sys.executable, os.path.join(HERE, "exp.py"), "--root", d, *argv],
            check=True, capture_output=True,
        )
    return d


def _broken_score_matrix_no_ckpt():
    """A REAL ok training row whose cmd names no checkpoint -- the WARN tier.

    This is the world e1_31_middle_layer_loop lives in: a cmd written as prose describing stages,
    with no --out, no --name and no bare ckpt_*.pt, so produced_checkpoint returns None. Until
    2026-09-04 the check skipped such rows entirely, which meant the gate was satisfied by cmd
    FORMATTING rather than by the run being scored -- and the sibling row that wrote --out honestly
    was the one held to the record.
    """
    d = _tmp_repo()
    for argv in (
        ["start", "--name", "prose", "--cmd",
         "sft_math.py both arms at 250 steps (Stage B), then eval/humaneval_bpb.py in the 2x2"],
        ["done", "--name", "prose", "--status", "ok", "--result", "done"],
    ):
        subprocess.run(
            [sys.executable, os.path.join(HERE, "exp.py"), "--root", d, *argv],
            check=True, capture_output=True,
        )
    return d


def _broken_score_matrix_dangling_artifact():
    """An ok training row whose reading_artifact names a path that does not exist -- the FAIL tier.

    The field is an escape hatch from the score-matrix requirement, so an unchecked one would let a
    row assert "scored by another instrument" with nothing behind it: strictly worse than the gap it
    replaces, because the gap is visible and the false claim reads as satisfied.
    """
    d = _tmp_repo()
    for argv in (
        ["start", "--name", "y", "--cmd", "sft_math.py --out ckpt_y.pt"],
        ["done", "--name", "y", "--status", "ok", "--result", "done"],
    ):
        subprocess.run(
            [sys.executable, os.path.join(HERE, "exp.py"), "--root", d, *argv],
            check=True, capture_output=True,
        )
    log = os.path.join(d, "runs", "experiments.jsonl")
    rows = [json.loads(x) for x in open(log, encoding="utf-8") if x.strip()]
    rows[-1]["reading_artifact"] = "runs/this_file_was_never_written.log"
    with open(log, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return d


def _read_ckpt_dict(path):
    """Read the full dict from a torch.save checkpoint without loading tensors.
    The .pt is a zip; the dict is plain data pickled in data.pkl. Tensor storages
    are referenced via persistent_load, which we stub, and torch rebuild
    functions resolve to dummies."""
    import pickle
    import zipfile

    class _Stub(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch"):
                return lambda *a, **kw: None
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                return lambda *a, **kw: None

        def persistent_load(self, pid):
            return None

    with zipfile.ZipFile(path) as z:
        pkl_name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        with z.open(pkl_name) as f:
            return _Stub(f).load()


def _read_ckpt_cfg(path):
    """Read the cfg dict from a torch.save checkpoint without importing torch."""
    return _read_ckpt_dict(path).get("cfg", {})


#: Checkpoints written before train.py stamped env_fp. They can never grow one, so failing
#: on them is a permanent red, and a permanent red is no signal -- it blocked a launch the
#: hour it landed. A RATCHET, like scripts/restartability_baseline.json: only a NEW
#: checkpoint without env_fp fails. The list may shrink, never grow.
#:
#: It lives in a FILE, not in this source: the checkpoints are on the pod and a dev
#: checkout has three of them, so a list written here was written against the wrong
#: world -- it passed locally and left 28 pod checkpoints failing.
_ENV_FP_BASELINE = os.path.join("data", "env_fp_baseline.json")


def _pre_env_fp(root):
    try:
        with open(os.path.join(root, _ENV_FP_BASELINE), encoding="utf-8") as f:
            return set(json.load(f)["pre_guard"])
    except Exception:
        return set()


def check_env_fp_present(root):
    """Every checkpoint written since the guard landed carries an environment fingerprint.

    A container restart can change the effective environment (dropping
    hand-installed packages) without anyone noticing. The fingerprint is
    compared on resume; a checkpoint without one cannot be safely resumed."""
    ckpts = sorted(glob.glob(os.path.join(root, "ckpt_*.pt")))
    if not ckpts:
        return SKIP, "no checkpoints"
    pre_guard = _pre_env_fp(root)
    missing, grandfathered = [], 0
    for p in ckpts:
        try:
            d = _read_ckpt_dict(p)
        except Exception:
            continue  # unreadable checkpoint is a different check's problem
        if "env_fp" in d:
            continue
        if os.path.basename(p) in pre_guard:
            grandfathered += 1
        else:
            missing.append(os.path.basename(p))
    if missing:
        return FAIL, f"{len(missing)} checkpoint(s) without env_fp: {', '.join(missing[:5])}"
    return PASS, f"all {len(ckpts) - grandfathered} post-guard checkpoints carry env_fp ({grandfathered} grandfathered)"


def _broken_env_fp_present():
    """A real torch checkpoint without env_fp."""
    import torch

    d = _tmp_repo()
    # A repo-real file so selftest does not skip this check as hand-written.
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w").close()
    # A REAL torch checkpoint (torch.save), just without env_fp -- the check
    # reads it the same way it reads a production one.
    torch.save(
        {"model": {}, "cfg": {}, "vocab_id": "test", "corpus_fp": {}},
        os.path.join(d, "ckpt_test.pt"),
    )
    return d


def check_opt_state_present(root):
    """A checkpoint that records a training step must carry optimizer state.

    A checkpoint with `step` but no `opt` cannot be safely resumed: Muon momentum
    and AdamW moments are zeroed, the loss dips and recovers, and it reads as
    noise rather than a bug. The ladder's short runs from scratch never resumed,
    so the gap stayed hidden. The 30B run will."""
    ckpts = sorted(glob.glob(os.path.join(root, "ckpt_*.pt")))
    if not ckpts:
        return SKIP, "no checkpoints"
    missing, resumable = [], 0
    for p in ckpts:
        try:
            d = _read_ckpt_dict(p)
        except Exception:
            continue  # unreadable checkpoint is a different check's problem
        if "step" not in d:
            continue  # final/eval checkpoint, not claiming to be resumable
        resumable += 1
        if "opt" not in d:
            missing.append(os.path.basename(p))
    if missing:
        return FAIL, f"{len(missing)} checkpoint(s) with step but no opt: {', '.join(missing[:5])}"
    if resumable == 0:
        return PASS, f"0/{len(ckpts)} checkpoints carry a step field -- guard for 30B .stepN checkpoints, not a live PASS"
    return PASS, f"all {resumable} resumable checkpoints carry opt state"


def _broken_opt_state_present():
    """A real torch checkpoint with step but no opt state."""
    import torch

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w").close()
    torch.save(
        {"model": {}, "cfg": {}, "vocab_id": "test", "corpus_fp": {}, "env_fp": "x", "step": 100},
        os.path.join(d, "ckpt_test.pt"),
    )
    return d


def _is_ladder_mix(mix):
    """The frozen recipe binds the six ladder points and nothing else.

    Membership was the filename prefix `mix_scale_`, which asks "is this named like a
    ladder point" rather than "is this a ladder point". The 30B delivery run has no
    ladder to be comparable with -- nothing is measured against it -- and its natural
    name, mix_scale_30b.json, would have silently put it under a recipe frozen for a
    purpose that does not apply to it, including warmup 20 (0.06% of a 32,697-step run).
    LADDER is the list, so ask LADDER."""
    return os.path.basename(mix) in {os.path.basename(m) for _, m in LADDER}


def check_ladder_config(root):
    """Every ladder checkpoint's cfg matches the single frozen run config.
    Scope: checkpoints whose experiments row was launched via run_ddp.sh
    (harness run point or a bare launch that landed in the ledger). A/B runs
    use torchrun directly and are not bound by the frozen config.
    A missing field (None) is UNKNOWN, not divergence: the checkpoint predates
    the stamp. SKIP without checkpoints or the frozen config file."""
    fpath = os.path.join(root, "data", "mix_scale_run_config.json")
    if not os.path.exists(fpath):
        return SKIP, "data/mix_scale_run_config.json not present"
    frozen = json.load(open(fpath, encoding="utf-8"))
    # A key declared frozen but absent from the config freezes nothing: `frozen[k]`
    # raised KeyError on 'warmdown' and the whole check died, so the other twenty keys
    # stopped being verified too. warmdown and anneal_frac were added to _FROZEN_KEYS
    # when the WSD schedule landed and never added to the JSON. A crash is better than a
    # silent skip and worse than a finding: report it as one, and keep checking the keys
    # that do have a frozen value (de, 2026-09-01).
    absent = [k for k in (*_FROZEN_KEYS, *_CODE_FROZEN_KEYS) if k not in frozen]
    if absent:
        # BEFORE the no-checkpoints SKIP: an unfrozen frozen key is a defect in the
        # config, true on a machine holding no checkpoints at all. Behind the SKIP it
        # would be invisible on every dev box and only visible on the pod.
        return FAIL, (f"{len(absent)} key(s) declared frozen but absent from "
                      f"data/mix_scale_run_config.json, so nothing freezes them: "
                      f"{', '.join(absent)} -- add the value or drop the key")
    ckpts = sorted(glob.glob(os.path.join(root, "ckpt_*.pt")))
    if not ckpts:
        return SKIP, "no checkpoints"
    # Ladder points are launched via run_ddp.sh; A/B runs use torchrun directly.
    ladder_names = set()
    # An A/B arm deviates on purpose, and the ledger already records that it did: the
    # deviating flag is written out in the launch cmd. Deriving the exemption from that row
    # beats a hand-written list, which is exactly the thing that leaves out whatever nobody
    # remembers (the env_fp baseline did, at three names against the pod's twenty-eight).
    declared = {}
    exp_path = os.path.join(root, "runs", "experiments.jsonl")
    if os.path.exists(exp_path):
        # Folded first (e1-18), then keyed by NAME, and the second half is a known
        # narrowing rather than an oversight. `declared` grants a frozen-key exemption,
        # so which run of a name wins decides which deviations are forgiven -- and two
        # names in this ledger ran with DIFFERENT flag sets: p02_s0 (4 runs, 3 sets) and
        # p500m_20b_0902 (00:03 with 10 flags, the 01:03 relaunch with 15). Dropping
        # `started` collapses those, so the exemption for ckpt_<name>.pt comes from
        # whichever row wins rather than from the run that produced the checkpoint.
        # Position order and start order agree on today's ledger (measured: 0 of 27
        # names differ), so this is latent, and widening the key is a behaviour change
        # to an exemption path -- left for a ruling, not folded in here. The fold does
        # remove the other half: a duplicate start landing after a close can no longer
        # be the row that grants an exemption.
        evs = []
        for line in open(exp_path, encoding="utf-8"):
            if line.strip():
                evs.append(json.loads(line))
        for r in _exp_fold(evs):
            if "run_ddp.sh" in r.get("cmd", ""):
                nm = r.get("name", "")
                ladder_names.add(nm)
                flags = {_FLAG_TO_CFG.get(t[2:].split("=", 1)[0], t[2:].split("=", 1)[0])
                         for t in r.get("cmd", "").split() if t.startswith("--")}
                declared[nm] = flags
    bad, unknown, exempt, checked = [], [], [], 0
    for p in ckpts:
        name = os.path.basename(p)[5:-3]  # ckpt_<name>.pt
        if name not in ladder_names:
            continue
        try:
            cfg = _read_ckpt_cfg(p)
        except Exception as e:
            return FAIL, f"{os.path.basename(p)}: cannot read cfg: {e}"
        if not _is_ladder_mix(cfg.get("mix", "")):
            continue
        checked += 1
        for k in (*_FROZEN_KEYS, *_CODE_FROZEN_KEYS):
            if k not in frozen:
                continue  # reported once, as `absent`, not per checkpoint
            v = cfg.get(k)
            if v is None:
                unknown.append(f"{os.path.basename(p)}:{k}")
            elif v != frozen[k]:
                if k in declared.get(name, ()):
                    exempt.append(f"{os.path.basename(p)}: {k}={v} (declared by {name})")
                else:
                    bad.append(f"{os.path.basename(p)}: {k}={v} != frozen {frozen[k]}")
    if bad:
        return FAIL, "; ".join(bad)
    if not checked:
        return SKIP, "no ladder checkpoints (experiments.jsonl is pod-authoritative; local copy may be stale)"
    msg = f"{checked} checkpoint(s) match the frozen config"
    if exempt:
        msg += f"; {len(exempt)} declared A/B deviation(s): {'; '.join(exempt)}"
    if unknown:
        msg += f"; {len(unknown)} field(s) unverifiable (pre-stamp): {', '.join(sorted(set(unknown)))}"
    return PASS, msg


def _broken_ladder_config():
    """A checkpoint with warmup changed from the frozen 20 to 30 -- the silent
    recipe drift that produces a completed point under a different config."""
    import io
    import pickle
    import shutil
    import zipfile

    d = _tmp_repo()
    shutil.copy(
        os.path.join(ROOT, "data", "mix_scale_run_config.json"),
        os.path.join(d, "data", "mix_scale_run_config.json"),
    )
    cfg = {"mix": "data/mix_scale_0.2b.json", "warmup": 30, "batch": 16, "accum": 2,
           "vocab": 32784, "bucket_cap_mb": 50}
    buf = io.BytesIO()
    pickle.dump({"cfg": cfg, "vocab_id": "fake"}, buf)
    with zipfile.ZipFile(os.path.join(d, "ckpt_test.pt"), "w") as z:
        z.writestr("data.pkl", buf.getvalue())
    # An exp row launched via run_ddp.sh -- the scope filter.
    with open(os.path.join(d, "runs", "experiments.jsonl"), "w") as f:
        f.write(json.dumps({"name": "test", "started": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
                            "status": "ok", "cmd": "bash run_ddp.sh --name test"}) + "\n")
    return d


def check_ladder_cfg_consistent(root):
    """All six ladder points record the same cfg (except mix). The frozen list
    prevents launch drift; this detects code-edit drift between points -- a
    chunk_size, layers, or optimizer-param edit that no CLI flag can make and
    no frozen key can see. Checkpoints record vars(Cfg): 46 fields, complete.
    The only legitimate per-point difference is mix (the D varies)."""
    names = [n for n, _ in LADDER]
    ckpts = [os.path.join(root, f"ckpt_{n}.pt") for n in names]
    ckpts = [p for p in ckpts if os.path.exists(p)]
    if len(ckpts) < 2:
        return SKIP, f"{len(ckpts)}/{len(names)} ladder checkpoints present; need 2+ to compare"
    cfgs = {}
    for p in ckpts:
        try:
            cfgs[p] = _read_ckpt_cfg(p)
        except Exception as e:
            return FAIL, f"{os.path.basename(p)}: cannot read cfg: {e}"
    base = cfgs[ckpts[0]]
    diffs, unknown = [], []
    for p in ckpts[1:]:
        cfg = cfgs[p]
        for k in sorted(set(base) | set(cfg)):
            if k == "mix":
                continue  # the D varies; everything else must not
            if k not in base or k not in cfg:
                unknown.append(f"{os.path.basename(p)}:{k}")
            elif base[k] != cfg[k]:
                diffs.append(f"{os.path.basename(p)}:{k} {base[k]!r}->{cfg[k]!r}")
    if diffs:
        return FAIL, f"{len(diffs)} field(s) differ: {'; '.join(diffs[:5])}"
    note = f"; {len(unknown)} unverifiable (pre-stamp): {', '.join(sorted(set(unknown))[:5])}" if unknown else ""
    return PASS, f"{len(ckpts)} checkpoints, {len(base)} fields, all consistent{note}"


def _broken_ladder_cfg_consistent():
    """Two ladder checkpoints with chunk_size changed in one -- the code-edit
    drift the frozen list cannot see (no CLI flag touches chunk_size)."""
    import io
    import pickle
    import shutil
    import zipfile

    d = _tmp_repo()
    shutil.copy(
        os.path.join(ROOT, "data", "mix_scale_run_config.json"),
        os.path.join(d, "data", "mix_scale_run_config.json"),
    )
    for name, cs in [("p02_s0", 32), ("p03", 64)]:
        cfg = {"mix": f"data/mix_scale_{'0.2b' if name == 'p02_s0' else '0.3b'}.json",
               "chunk_size": cs, "batch": 16}
        buf = io.BytesIO()
        pickle.dump({"cfg": cfg, "vocab_id": "fake"}, buf)
        with zipfile.ZipFile(os.path.join(d, f"ckpt_{name}.pt"), "w") as z:
            z.writestr("data.pkl", buf.getvalue())
    return d


def _train_parser_flags(train_py):
    """CLI flag names from train.py's argparse section. Two shapes: direct
    add_argument("--flag", ...) and the loop over a dict whose keys are flag names."""
    src = open(train_py, encoding="utf-8").read()
    section = src[src.index("ArgumentParser"):src.index("parse_args")]
    flags = set()
    for m in re.finditer(r'add_argument\(\s*["\']--(\w+)["\']', section):
        flags.add(m.group(1))
    for m in re.finditer(r'^\s+"(\w+)":\s*"', section, re.M):
        flags.add(m.group(1))
    return flags


def check_frozen_keys_complete(root):
    """Every train.py parser flag that changes a Cfg field is either in _FROZEN_KEYS
    or in _UNFROZEN_ALLOWLIST. The frozen set rotted once: eight architecture/recipe
    flags were missing and nothing noticed. This check is the tripwire."""
    train_py = os.path.join(root, "train.py")
    if not os.path.exists(train_py):
        return SKIP, "train.py missing"
    flags = _train_parser_flags(train_py)
    known = set(_FROZEN_KEYS) | _UNFROZEN_ALLOWLIST
    missing = []
    for f in sorted(flags):
        cfg_key = _FLAG_TO_CFG.get(f, f)
        if cfg_key not in known:
            missing.append(f"--{f} (Cfg.{cfg_key})")
    if missing:
        return FAIL, f"{len(missing)} flag(s) in neither frozen set nor allow-list: {'; '.join(missing)}"
    return PASS, f"{len(flags)} parser flags, all in frozen set or allow-list"


def _broken_frozen_keys_complete():
    """The real train.py with a new architecture flag added to the parser --
    exactly how the eight missing fields escaped notice."""
    d = _tmp_repo()
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    # Add a new add_argument call inside the parser section, before parse_args.
    patched = src.replace(
        'args = parser.parse_args()',
        '    parser.add_argument("--new_arch_flag", action="store_true",\n'
        '                        help="a new architecture flag the frozen set does not know about")\n'
        '    args = parser.parse_args()',
    )
    with open(os.path.join(d, "train.py"), "w", encoding="utf-8") as fh:
        fh.write(patched)
    return d


def _token_cache_dir():
    """The directory holding token caches, from train.py's TOKEN_CACHE constant.
    HARNESS_TOKEN_CACHE_DIR overrides (selftest)."""
    forced = os.environ.get("HARNESS_TOKEN_CACHE_DIR")
    if forced:
        return forced
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    m = re.search(r'^TOKEN_CACHE\s*=\s*["\']([^"\']+)["\']', src, re.M)
    if not m:
        raise KeyError("train.py has no TOKEN_CACHE; the check that reads it cannot run")
    return os.path.dirname(m.group(1))


def _cache_rows(path, seq):
    """Number of rows in a token cache, from the zip storage size (no torch)."""
    import zipfile

    with zipfile.ZipFile(path) as z:
        info = next(i for i in z.infolist() if i.filename.endswith("data/0"))
    return (info.file_size // 4) // (seq + 1)


def check_mix_supply(root, mix_glob=None):
    """Per-domain demand vs epoch-capped cache supply at every budget point.
    FAILs when demand exceeds the FULL cache (data would repeat even after
    train.py's cap). The val-split reduction (demand > pool but <= cache) is
    a known, accepted condition -- the gate doc documents it as 1.53% at
    3.24b, handled by the fit-protocol reading D from the log. The val carve
    lands entirely in the anneal phase (roughly 2x the per-domain loss the
    whole-budget figure suggests), not spread across both phases. SKIP without
    caches (CPU CI, dev box)."""
    # An all-blocked mix is answerable without a cache, so report it before the
    # cache-dir SKIP -- otherwise a dev box says "no cache" and the real state (every
    # domain deliberately blocked, pre-corpus) is invisible (fb, 2026-08-31).
    pat = mix_glob or os.path.join(root, "data", "mix_scale_[0-9]*.json")
    for f in glob.glob(pat if os.path.isabs(pat) else os.path.join(root, pat)):
        try:
            obj = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not obj.get("domains") and obj.get("_blocked"):
            return SKIP, (f"{os.path.basename(f)}: all {len(obj['_blocked'])} domains blocked "
                          f"(pre-corpus, by design) -- nothing to gate yet")
    cache_dir = _token_cache_dir()
    if not os.path.isdir(cache_dir):
        return SKIP, f"token cache dir {cache_dir} not present"
    seq = cfg_default("seq")
    anneal_frac = cfg_default("anneal_frac")
    val_frac = cfg_default("val_frac")
    val_rows_max = cfg_default("val_rows_max")
    # mix_glob lets a caller gate one file (mix_30b_stage2.json) rather than only
    # the ladder; the default keeps the ladder behaviour.
    pattern = mix_glob or os.path.join(root, "data", "mix_scale_[0-9]*.json")
    mixes = sorted(glob.glob(pattern if os.path.isabs(pattern) else os.path.join(root, pattern)))
    if not mixes:
        # Distinguish "no such file" from "the file exists and every domain is blocked":
        # a mix with all domains under _blocked is a deliberate pre-corpus state, and a
        # SKIP that calls it "no matching files" gets filed as a gap (fb, 2026-08-31).
        return SKIP, f"no mix files match {os.path.basename(pattern)}"
    # A retired mix is not gated, and the marker has teeth: train.py refuses to start on
    # one, so "retired" is a property of the artifact rather than a label this check
    # agreed to ignore. The ladder's anneal demand exceeds its cache supply and cannot be
    # corrected -- ladder_config_frozen forbids editing it -- so the only honest exits
    # were retirement or a permanent red, and a permanent red is the same as no signal.
    retired = [os.path.basename(m) for m in mixes
               if json.load(open(m, encoding="utf-8")).get("_retired")]
    mixes = [m for m in mixes if os.path.basename(m) not in retired]
    if not mixes:
        return SKIP, f"every matching mix is retired: {', '.join(retired)}"
    bad = []
    val_loss_tokens = 0  # val-split loss at the largest budget point, in tokens
    largest = max(
        json.load(open(m, encoding="utf-8"))["total_tokens"] for m in mixes
    )
    for mp in mixes:
        mix = json.load(open(mp, encoding="utf-8"))
        rows = mix["total_tokens"] / seq
        is_largest = mix["total_tokens"] == largest
        for name, d in mix["domains"].items():
            cache = os.path.join(cache_dir, f"tokens_{name}.pt")
            if not os.path.exists(cache):
                bad.append(f"{os.path.basename(mp)}: {name} has no cache")
                continue
            try:
                cache_rows = _cache_rows(cache, seq)
            except Exception as e:
                bad.append(f"{os.path.basename(mp)}: {name} cache unreadable: {e}")
                continue
            # The builder draws from the POOL, not the raw cache: train.py:1583 carves
            # the val holdout off first, then caps at pool x epochs. Checking raw supply
            # passes a mix the builder then silently under-draws -- stage-1 cot passed at
            # 3 x 424,056,227 = 1.272B and drew 1.210B, and the run scheduled 14.938B
            # instead of 15.000B (44, 2026-08-31). Row counts come from the cache, which
            # is authoritative over a log-rounded weight.
            n_val = min(max(1, int(cache_rows * val_frac)), val_rows_max)
            pool_rows = cache_rows - n_val
            used = 0
            for frac, key in ((1 - anneal_frac, "weight"), (anneal_frac, "anneal")):
                want = int(rows * frac * d.get(key, d["weight"]))
                cap = int(pool_rows * d.get("epochs", 1)) - used
                # 0.5% tolerance: weight->row rounding leaves sub-0.1% residue
                # at 3.24b (documented in the gate doc). Real oversupply FAILs.
                if want > cap * 1.005:
                    bad.append(f"{os.path.basename(mp)}: {name} {key} wants {want} rows, "
                               f"pool supplies {cap} (cache {cache_rows} - {n_val} val, "
                               f"x{d.get('epochs', 1)} epochs)")
                    break
                used += want
            else:
                # Both phases within cache: compute val-split loss for the report.
                if is_largest:
                    val_loss_tokens += max(0, used - pool_rows) * seq
    if bad:
        return FAIL, "; ".join(bad)
    pct = 100 * val_loss_tokens / largest if largest else 0
    return PASS, f"{len(mixes)} mixes, all within POOL supply (cache - val); val-split loss {pct:.2f}% at {largest / 1e9:.2f}B"


def _broken_mix_supply():
    """The real 0.2b mix with caches too small to supply it -- demand exceeds
    pool at every domain."""
    import shutil
    import zipfile

    d = _tmp_repo()
    dst = os.path.join(d, "data", "mix_scale_0.2b.json")
    shutil.copy(os.path.join(ROOT, "data", "mix_scale_0.2b.json"), dst)
    # Live, because a retired mix is not gated and the world would report SKIP -- which
    # is what it did the moment the ladder was retired, leaving this check unable to fail.
    obj = json.load(open(dst, encoding="utf-8"))
    obj.pop("_retired", None)
    json.dump(obj, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    cache_dir = os.path.join(d, "fake_caches")
    os.makedirs(cache_dir)
    seq = cfg_default("seq")
    for dom in ("web_hq", "textbook", "wiki", "en", "math", "code", "chat"):
        with zipfile.ZipFile(os.path.join(cache_dir, f"tokens_{dom}.pt"), "w") as z:
            z.writestr("data/0", b"\x00" * (4 * (seq + 1) * 10))
    os.environ["HARNESS_TOKEN_CACHE_DIR"] = cache_dir
    return d


def _provenance_fingerprints(path, domains):
    """{domain: fingerprint} parsed from data/PROVENANCE.md. A domain block is a heading
    whose text contains the domain as a whole token; its fingerprint is a
    `fingerprint: <16-hex>` line in that section. Only mix domains are attributed, so a
    fingerprint under an unrelated heading ('SFT-math candidates') cannot hijack 'math'."""
    if not os.path.isfile(path):
        return {}
    domset = {d.lower() for d in domains}
    out, section = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            section = m.group(1)
            continue
        fp = re.search(r"fingerprint[:\s=]+([0-9a-f]{16})", line, re.I)
        if fp and section:
            for tok in re.findall(r"[A-Za-z0-9_]+", section):
                if tok.lower() in domset:
                    out[tok.lower()] = fp.group(1)
    return out


def check_corpus_fp(root):
    """Every domain the default mix names must (a) carry a build-time fingerprint
    (build_corpus.py stamps build_corpus_stats.json) matching the live directory, and
    (b) have a provenance block in data/PROVENANCE.md whose recorded fingerprint also
    matches. A missing stamp or a missing block is FAIL, not SKIP: an unstamped or
    unrecorded domain cannot be distinguished from a swapped-in one -- the voided 0.2b
    run trained on CCI3 shards under web_hq's name, and fineweb2 web_hq was lost with no
    record of how it was built. Domains with no directory on this machine are
    mix_shards_present's beat, not this one."""
    doms, err = read_mix(os.path.join(root, cfg_default("mix")))
    if err:
        return FAIL, f"cannot read the default mix: {err}"
    corpus = os.path.join(root, "data", "corpus")
    present = [d for d in doms if os.path.isdir(os.path.join(corpus, d))]
    if not present:
        return SKIP, "no mix domain has a directory on this machine"
    prov = _provenance_fingerprints(os.path.join(root, "data", "PROVENANCE.md"), doms)
    problems, ok = [], 0
    for dom in present:
        stats = os.path.join(corpus, dom, "build_corpus_stats.json")
        try:
            with open(stats, encoding="utf-8") as f:
                stamped = json.load(f).get("fingerprint")
        except Exception:
            stamped = None
        live = cfp.fp_domain(dom, corpus)
        dom_ok = True
        if not stamped:
            problems.append(f"{dom}: no build-time fingerprint")
            dom_ok = False
        elif live != stamped:
            problems.append(f"{dom}: stamped {stamped} != live {live}")
            dom_ok = False
        if dom not in prov:
            problems.append(f"{dom}: no PROVENANCE.md block")
            dom_ok = False
        elif prov[dom] != live:
            problems.append(f"{dom}: PROVENANCE.md {prov[dom]} != live {live}")
            dom_ok = False
        if dom_ok:
            ok += 1
    if problems:
        return FAIL, f"{ok}/{len(present)} match; " + "; ".join(problems[:3])
    return PASS, f"{ok}/{len(present)} mix domains match their build-time and PROVENANCE.md fingerprints"


def check_pod_drift(root):
    # The pod is not a git repo: its files must match the manifest pod_push.sh shipped with
    # them. A dev checkout has nothing to check -- the manifest is generated at push time and
    # is not tracked, so there is no committed copy that could be stale (shape A, 2026-09-04).
    if pod_drift.is_pod(root):
        ok, evidence = pod_drift.check_pod(root)
        return (PASS if ok else FAIL), evidence
    return SKIP, "dev checkout; the pod gates file drift against the manifest shipped with it"


def _broken_ghost_running():
    """The REAL experiment log plus a fake running row whose process cannot exist: the
    pod-only ghost check must see it. The 2h grace is passed by backdating the row."""
    import shutil

    d = _tmp_repo()
    shutil.copy(os.path.join(ROOT, "runs", "experiments.jsonl"), os.path.join(d, "runs", "experiments.jsonl"))
    with open(os.path.join(d, "runs", "experiments.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"started": "2026-08-29 00:00", "name": "ghost_run_xyz", "status": "running"}) + "\n")
    return d


def _broken_pod_drift():
    """Every file the REAL manifest names, copied in, the manifest REGENERATED over
    those copies, then one file mutated. Two bugs made the old world green for free:

    It copied the manifest plus one file, so the other 238 named files were absent and
    the check reported "239 drifted: missing AGENTS.md; missing algorithms/..." with or
    without the mutation. Verified by restoring the appended file and rerunning: still
    FAIL, same 239. Selftest green on 238 absences.

    Copying all 239 was not enough either -- the manifest records committed hashes and a
    dev checkout has uncommitted edits, so the world drifted on whatever the session
    happened to have open. Regenerating over the copies makes the world self-consistent,
    and then the appended line is the only difference there is (de, 2026-09-01).

    THE FILE LIST COMES FROM pod_drift.scoped_paths, NOT from reading the real manifest, and that
    is a CI fix rather than a refactor. `data/pod_head_manifest.txt` stopped being tracked on
    2026-09-04 -- pod_push generates it from the HEAD it ships -- so every machine that has run
    pod_push has one on disk and this world was green there, while a fresh clone has none and the
    open() raised FileNotFoundError before any check ran. That is how `harness.py --selftest` came
    to tell CI nothing at all for every commit in that window: not "some worlds failed", nothing
    ran. scoped_paths is the same `git ls-files SCOPE` that pod_push feeds the generator, so the
    world now derives its population from the same place the real artifact does and cannot go
    absent again.

    The CI branch cannot be exercised here -- the selftest world has no .git."""
    import shutil

    d = _tmp_repo()
    man_rel = os.path.join("data", "pod_head_manifest.txt")
    rels = []
    for rel in pod_drift.scoped_paths(ROOT):
        src = os.path.join(ROOT, rel)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(d, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        rels.append(rel)
    # FOUR COLUMNS, from pod_drift's own classifier and mode reader. read_manifest tolerates a
    # missing class or mode (defaulting docs/644), so a three-column world would have PASSED while
    # asserting less than the real artifact does -- the mode column landed 2026-09-03 and a world
    # that omits it cannot notice a mode drift at all. sha_disk over the COPIES, not sha_head:
    # the point of regenerating here is that the world is self-consistent against its own files,
    # which is what makes the appended line the only difference.
    _classes = pod_drift._classify_files()
    _modes = pod_drift.git_modes(ROOT, "HEAD")
    with open(os.path.join(d, man_rel), "w", encoding="utf-8") as f:
        for rel in rels:
            f.write(f"{pod_drift.sha_disk(os.path.join(d, rel))}  {rel}  "
                    f"{_classes.get(rel, 'docs')}  {_modes.get(rel, '644')}\n")
    with open(os.path.join(d, "scripts", "harness.py"), "a", encoding="utf-8") as f:
        f.write("\n# broken world drift\n")
    return d


def _broken_corpus_fp():
    """The REAL default mix copied into the broken world, with two of its real-named domains
    present: one stamped correctly and then drifted, one carrying no stamp at all. Both tiers
    must report FAIL, and the evidence must carry the denominator -- '1 domain(s) match' once
    read as all-green when it was 1 of 7."""
    import shutil

    d = _tmp_repo()
    mix_rel = cfg_default("mix")
    shutil.copy(os.path.join(ROOT, mix_rel), os.path.join(d, mix_rel))
    doms, _ = read_mix(os.path.join(ROOT, mix_rel))
    corpus = os.path.join(d, "data", "corpus")
    real = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", doms[0], "*.jsonl")))
    real = real or sorted(glob.glob(os.path.join(ROOT, "data", "**", "*.jsonl"), recursive=True))
    drifted = os.path.join(corpus, doms[0])
    os.makedirs(drifted)
    shutil.copy(real[0], os.path.join(drifted, "real_shard.jsonl"))
    with open(os.path.join(drifted, "build_corpus_stats.json"), "w") as f:
        json.dump({"fingerprint": cfp.fp_domain(doms[0], corpus)}, f)
    with open(os.path.join(drifted, "real_shard.jsonl"), "a", encoding="utf-8") as f:
        f.write('{"question": "broken world drift", "output": "1"}\n')
    unstamped = os.path.join(corpus, doms[1])
    os.makedirs(unstamped)
    shutil.copy(real[0], os.path.join(unstamped, "real_shard.jsonl"))
    # A PROVENANCE.md at the real path: doms[0]'s block records a WRONG fingerprint
    # (the mismatched-block tier), doms[1] has no block at all.
    with open(os.path.join(d, "data", "PROVENANCE.md"), "w") as f:
        f.write(f"# provenance\n\n## {doms[0]}\n\nfingerprint: 0000000000000000\n")
    return d


# Every third-party module this repo imports, and the pip name that supplies it. The
# container's image already carries most of them; a restart keeps /work but drops the
# writable layer, so only the hand-installed ones vanish -- and which ones those are is
# not knowable without a written list. That is the whole reason this exists.
_REQUIRED = {
    "torch": "torch", "numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib",
    "pyarrow": "pyarrow", "tokenizers": "tokenizers", "transformers": "transformers",
    "datasets": "datasets", "huggingface_hub": "huggingface_hub", "flask": "flask",
    "opencc": "opencc", "trackio": "trackio", "liger_kernel": "liger-kernel",
    "fla": "flash-linear-attention", "torchao": "torchao", "triton": "triton",
    "flash_attn": "flash-attn",
}
# Absent on a dev Mac by design; only a box that can train is expected to have them.
_LINUX_ONLY = {"liger_kernel", "fla", "torchao", "triton", "flash_attn"}


def check_env_importable(root):
    """Every third-party module the repo imports is importable.

    2026-08-30: a container restart dropped the writable layer and with it liger_kernel,
    fla, flask, opencc and trackio. The code was untouched, so the first symptom was a
    ModuleNotFoundError on a line that had worked an hour earlier -- which reads as a
    broken import, not as missing infrastructure, and sends the next person to debug the
    wrong thing. This check names the cause and prints the command that fixes it.
    """
    import importlib.util

    extra = os.environ.get("HARNESS_REQUIRE_EXTRA")  # selftest injects an unsatisfiable name
    req = dict(_REQUIRED, **({extra: extra} if extra else {}))
    missing = []
    for mod in sorted(req):
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, ValueError):
            missing.append(mod)
            continue
        # `spec.origin is None` is a namespace package: the directory survived but the
        # module did not. 2026-08-30: flash_attn resolved this way after a reinstall, so
        # this check stayed green while `from flash_attn import flash_attn_func` raised
        # ImportError -- train.py caught it, set HAS_FA=False and trained the whole
        # fp32-master A/B on the fallback attention path, silently non-comparable with the
        # six ladder points that ran with fa True. find_spec is not import.
        if spec is None or spec.origin is None:
            # flash_attn is the one package where the shadow is expected: the base image
            # ships flash-attn 4, whose real code lives under flash_attn.cute, and train.py
            # imports from there when the v2 top level is absent. Ask for what the code
            # actually uses rather than for the layout it used to have.
            alt = _REQUIRED_ALT.get(mod)
            try:
                if alt and (a := importlib.util.find_spec(alt)) and a.origin:
                    continue
            except (ImportError, ValueError):
                pass
            missing.append(mod)
    if not missing:
        return PASS, f"all {len(req)} imported packages present"
    # Strictness follows the ability to train, as in check_mix_shards: a dev box ships
    # none of the CUDA half and a permanent red there is no signal. The incident this
    # guards is a pod restart, and the pod has cards.
    if not _gpu_present():
        return SKIP, f"{len(missing)} absent on a box that cannot train: {', '.join(missing)}"
    # pip refuses to uninstall Debian's blinker (no RECORD file), which is what flask
    # pulls on; --ignore-installed is the difference between a working command and a
    # half-applied one, so it goes in only when flask is actually among the missing.
    pre = "--ignore-installed blinker " if "flask" in missing else ""
    pkgs = " ".join(req[m] for m in missing)
    return FAIL, (
        f"{len(missing)} package(s) missing: {', '.join(missing)} -- if this box worked "
        f"before, the container restarted and lost its writable layer (/work survives, "
        f"installed packages do not). Restore: python3 -m pip install {pre}{pkgs}"
    )


def _broken_env():
    """A world whose requirement list names a package that cannot exist."""
    os.environ["HARNESS_REQUIRE_EXTRA"] = "aupai_no_such_module"
    return _tmp_repo()


# --------------------------------------------------------------------------- tasks

TASKS_PATH = os.path.join(ROOT, "runs", "tasks.jsonl")
FRICTION_PATH = os.path.join(ROOT, "runs", "friction.jsonl")
FRICTION_KINDS = ("merge", "hook", "check", "pod", "launch")


def _friction_rows(path=None):
    """Every row, newest last. No fold: each row is one blocking event, not a state."""
    p = path or FRICTION_PATH
    out = []
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def check_friction_minutes_required(root):
    """near_miss and process_failure friction rows must carry minutes_lost.

    These two kinds are the ones where the cost is the whole point: a near-miss with no
    minutes is a story, not a data point, and a process failure with no minutes cannot be
    ranked against the other causes. 2026-09-04: 3/3 rows of these kinds lacked
    minutes_lost, and the friction summary printed "minutes not reported" for the
    combined cause -- the second-largest unfixed friction item, invisible to ranking.

    Baseline 3 (b0's rows, 2026-09-03/04): the check FAILs if a FOURTH row is added
    without minutes_lost. When b0 fills in the historical minutes, lower the baseline."""
    p = os.path.join(root, "runs", "friction.jsonl")
    if not os.path.exists(p):
        return SKIP, "no runs/friction.jsonl"
    BASELINE = 3
    bad = []
    for i, ln in enumerate(open(p, encoding="utf-8"), 1):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("kind") in ("near_miss", "process_failure") and "minutes_lost" not in r:
            who = r.get("who", "?")
            what = r.get("what", "?")[:50]
            bad.append(f"line {i} ({who}: {what})")
    if len(bad) > BASELINE:
        return FAIL, f"{len(bad)} near_miss/process_failure rows missing minutes_lost (baseline {BASELINE}): " + "; ".join(bad[BASELINE:BASELINE + 3])
    return PASS, f"{len(bad)}/{BASELINE} baseline rows missing minutes_lost; no new violations"


def _broken_friction_minutes_required():
    """A temp repo whose friction.jsonl has 4 near_miss rows without minutes_lost
    (baseline is 3, so the 4th is a new violation)."""
    d = _tmp_repo_shaped()
    fpath = os.path.join(d, "runs", "friction.jsonl")
    if os.path.islink(fpath):
        os.remove(fpath)  # symlink to the real file; replace with a temp-local copy
    with open(fpath, "w", encoding="utf-8") as fh:
        for i in range(4):
            fh.write(json.dumps({"kind": "near_miss", "who": "x", "what": f"fixture {i}"}) + "\n")
    return d


def cmd_friction(argv):
    """The one ledger of things that blocked someone, and the counts by cause.

    User order 14:2xZ 2026-09-03: every time a merge, a hook, a check or a pod push blocks
    someone, the cause goes in one place, and the place is periodically summarised into
    fixes. Without it each session pays the same toll privately and nobody can see which
    toll is the expensive one -- six merges blocked on two derived files today, and that
    only became visible when one person happened to hit all six.

    `minutes_lost` IS A SELF-REPORT AND THE OUTPUT SAYS SO. It is the field a reader will
    want to sum, and a sum of estimates printed beside measured counts reads as measured
    (this repo's most common defect shape). So the summary prints the count as the number
    and the minutes as `~N min (self-reported)`, and a row may omit minutes entirely
    rather than inventing one.

    Rows are events, never folded: two merges blocked by the same cause are two rows, which
    is the whole point of counting by cause."""
    ap = argparse.ArgumentParser(prog="harness friction")
    sub = ap.add_subparsers(dest="op")
    a = sub.add_parser("add")
    a.add_argument("--kind", required=True, choices=FRICTION_KINDS)
    a.add_argument("--cause", required=True,
                   help="what actually blocked it, specific enough to fix: the file and the "
                        "mechanism, not 'merge conflict'")
    a.add_argument("--blocked", required=True, help="what could not proceed")
    a.add_argument("--fix", default="", help="what unblocked it, or empty if nothing did yet")
    a.add_argument("--minutes", type=int, default=None,
                   help="SELF-REPORTED minutes lost; omit rather than guess")
    a.add_argument("--who", default=None, help="defaults to the current branch")
    a.add_argument("--commit", action="store_true",
                   help="commit the row path-scoped in this call, so the ledger never sits "
                        "dirty and cannot abort someone's merge before the drivers run")
    # RESOLVED IS AN APPEND, NEVER AN EDIT, and that is forced by the merge semantics rather
    # than chosen. runs/friction.jsonl is `merge=union`: MEASURED 2026-09-04 on a throwaway
    # repo, when two branches edit the SAME line union keeps BOTH -- so rewriting a row's
    # cause in place resurrects the wrong cause beside the correction as soon as anyone
    # merges. The ledger is an event log for the same reason runs/tasks.jsonl is; a
    # correction is a new event that names what it supersedes.
    rs = sub.add_parser("resolved")
    rs.add_argument("--cause-was", required=True,
                    help="a distinctive substring of the superseded cause; must match at "
                         "least one existing row or this refuses")
    rs.add_argument("--now-known", required=True,
                    help="the measured mechanism that replaces it")
    rs.add_argument("--fixed-by", default="", help="commit or change that removed the cause")
    rs.add_argument("--who", default=None)
    rs.add_argument("--commit", action="store_true")
    args = ap.parse_args(argv)

    if args.op == "resolved":
        rows = _friction_rows()
        hit = [r for r in rows if args.cause_was.lower() in (r.get("cause") or "").lower()]
        if not hit:
            # REFUSE rather than append an orphan correction. A resolution naming a cause no
            # row carries is worse than no resolution: the summary would grow a line that
            # corrects nothing, and nobody reading it could tell which is which.
            print(f"no friction row's cause contains {args.cause_was!r} -- nothing to resolve; "
                  f"run `harness friction` and copy a substring from the cause you mean",
                  file=sys.stderr)
            return 1
        row = {
            "when": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
            "who": args.who or _current_branch(),
            "kind": "resolution",
            "supersedes_cause": args.cause_was,
            "supersedes_rows": len(hit),
            "now_known": args.now_known,
            "fixed_by": args.fixed_by,
            "sha": _head_sha(),
        }
        _append_task(row, path=FRICTION_PATH)
        print(f"friction <- resolution over {len(hit)} row(s): {args.now_known[:70]}")
        if args.commit:
            rel = os.path.relpath(FRICTION_PATH, ROOT)
            r = subprocess.run(["git", "-C", ROOT, "commit", "-m",
                                f"friction: resolution -- {args.now_known[:60]}", "--", rel],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  row appended but NOT committed: {(r.stderr or r.stdout).strip()[:300]}")
                return 1
            print(f"  committed {rel} path-scoped")
        return 0

    if args.op != "add":
        rows = _friction_rows()
        if not rows:
            print(f"no friction rows in {os.path.relpath(FRICTION_PATH, ROOT)}")
            return 0
        # Resolutions are not causes: they are corrections OVER causes, so they are pulled
        # out before grouping. Counting them as their own cause would put a line in the
        # table for every correction and make the top cause look less frequent than it was.
        resolutions = [r for r in rows if r.get("kind") == "resolution"]
        rows = [r for r in rows if r.get("kind") != "resolution"]
        by_cause = {}
        for r in rows:
            c = (r.get("cause") or "?")
            d = by_cause.setdefault(c, {"n": 0, "min": 0, "reported": 0, "kinds": set(),
                                        "fixed": 0, "last": ""})
            d["n"] += 1
            if isinstance(r.get("minutes_lost"), int):
                d["min"] += r["minutes_lost"]
                d["reported"] += 1
            d["kinds"].add(r.get("kind") or "?")
            if r.get("fix_applied"):
                d["fixed"] += 1
            d["last"] = max(d["last"], r.get("when") or "")
        print(f"{len(rows)} row(s), {len(by_cause)} cause(s), {len(resolutions)} resolution(s) "
              f"-- most rows first\n")
        for c, d in sorted(by_cause.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
            mins = (f"~{d['min']} min (self-reported, {d['reported']}/{d['n']} rows)"
                    if d["reported"] else "minutes not reported")
            print(f"{d['n']:>3}x  {','.join(sorted(d['kinds'])):<14} {mins}")
            print(f"      {c}")
            print(f"      {d['fixed']}/{d['n']} row(s) carry a fix; last {d['last']}")
            # PRINT THE SUPERSESSION UNDER THE CAUSE IT CORRECTS, not in a section of its own.
            # A resolution row filed away elsewhere leaves the refuted mechanism as the first
            # and last thing a reader sees -- which is the §159 shape: the retraction was
            # written down, in a place that did not reach the site that published the number.
            for r in resolutions:
                if (r.get("supersedes_cause") or "").lower() in c.lower():
                    print(f"      SUPERSEDED {r.get('when')} by {r.get('who')}: "
                          f"{r.get('now_known')}")
                    if r.get("fixed_by"):
                        print(f"      fixed by: {r['fixed_by']}")
        return 0

    row = {
        "when": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        "who": args.who or _current_branch(),
        "kind": args.kind,
        "blocked_what": args.blocked,
        "cause": args.cause,
        "fix_applied": args.fix,
        "minutes_lost": args.minutes,
        "sha": _head_sha(),
    }
    _append_task(row, path=FRICTION_PATH)
    print(f"friction <- {args.kind}: {args.cause[:70]}")
    if args.commit:
        # COMMIT THE ROW IN THIS CALL, because a dirty friction.jsonl is itself a friction
        # cause: git's pre-merge tree check aborts with "local changes would be overwritten"
        # BEFORE merge drivers are consulted, so union-merge cannot help a file that is
        # sitting dirty (6e, 2026-09-03; second row of that shape today). A ledger whose
        # own writer leaves the tree dirty manufactures the toll it exists to record.
        #
        # NOTHING IS OWED WHEN THIS RETURNS. The first version used --no-verify and printed
        # an "OWED: pod_drift --write" line -- print-and-continue, and what it owed was a
        # manifest refix, then the top cause in this very ledger (6e's ruling, 2026-09-03).
        # The second version regenerated the manifest here and committed both paths. Both are
        # gone with the manifest's tracking (shape A, 2026-09-04): there is nothing derived
        # left to keep in step, so one ledger path is the whole commit.
        rel = os.path.relpath(FRICTION_PATH, ROOT)
        paths = [rel]
        msg = f"friction: {args.kind} -- {args.cause[:60]}"
        r = subprocess.run(["git", "-C", ROOT, "commit", "-m", msg, "--", *paths],
                           capture_output=True, text=True)
        if r.returncode != 0:
            # A hook refusal is a real answer, not noise: it means something else in the tree
            # is wrong (behind main, a red check). Surface ITS reason and exit nonzero rather
            # than retrying with --no-verify, which would hide it. The row stays appended and
            # uncommitted, which the message says, so the caller knows the tree is dirty.
            print(f"  row appended but NOT committed: {(r.stderr or r.stdout).strip()[:400]}")
            return 1
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", *paths],
                               capture_output=True, text=True).stdout.strip()
        print(f"  committed {' '.join(paths)} path-scoped, so the ledger never sits dirty")
        if dirty:
            # The hook regenerates the manifest during the commit; if that left either path
            # dirty again, say so instead of reporting a clean tree that is not clean.
            print(f"  STILL DIRTY after the commit, needs a person: {dirty[:200]}")
            return 1
    return 0


def _current_branch():
    r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def _head_sha():
    r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or ""




def _read_tasks(path=None, raw=False):
    """The register, folded by id: last row for an id wins.

    The file is an EVENT LOG, not a table. `task done`/`reopen` append a new row
    carrying the same id and the new state instead of rewriting the old one,
    because runs/*.jsonl merges by union: when two branches rewrite the same row,
    union keeps BOTH and the register grows a duplicate id (2026-08-31, t39 and
    t40 -- an open row and a done row for each, and tasks_well_formed failed the
    merge). Appends from different branches union cleanly and fold to the same
    state whichever order they land in.

    raw=True returns every event, for the checks that must see collisions.
    """
    p = path or TASKS_PATH
    if not os.path.exists(p):
        return []
    # A concurrent append can be observed mid-write: the reader sees a torn line and
    # json.loads raises, so `harness check` failed inside a hook and passed 20 s later
    # by hand -- a flake that reads as a real refusal (fb, 2026-08-31). Skip a line
    # that will not parse; ledgers_one_line_per_row is what judges malformed rows, and
    # it runs when nobody is mid-write.
    rows = []
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if raw:
        return rows
    folded = {}
    for r in rows:
        folded[r.get("id")] = r  # dict preserves first-insertion order; the value is the last event
    return list(folded.values())


def _append_task(row, path=None):
    """One event. Append, never rewrite: see _read_tasks."""
    p = path or TASKS_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # One write() of one complete line, O_APPEND: concurrent appends under a page-sized
    # payload do not interleave, and no reader observes a partial row. Building the
    # line first matters -- f.write() of a str can flush at a buffer boundary.
    line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _write_tasks(rows, path=None):
    p = path or TASKS_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _main_when(root=None):
    """{full sha: commit time 'YYYY-MM-DD HH:MM'} for every commit on main, one git call.

    Replaces the rule "a commit naming the task id in its subject beats the cited one":
    reviews and register commits name task ids too, and a delivery spans several commits,
    so that rule failed 9 of 31 honest rows the first time it actually ran (2026-09-02).
    What the incident behind it needed is the time check in _commit_delivers: a commit
    cited on a row must have existed when the row closed. Applied to rows closed from
    2026-09-02, the day the register's timestamps became UTC; earlier rows are local time
    and cannot be compared to a commit date."""
    root = root or ROOT
    if root not in _MAIN_WHEN:
        r = subprocess.run(["git", "-C", root, "log", "main", "--format=%H %cd", "--date=format-local:%Y-%m-%d %H:%M"],
                           capture_output=True, text=True, env={**os.environ, "TZ": "UTC"})
        _MAIN_WHEN[root] = dict(ln.split(" ", 1) for ln in r.stdout.splitlines() if " " in ln)
    return _MAIN_WHEN[root]


_MAIN_WHEN = {}


_MAIN_TOUCHED = {}


def _main_touched(root):
    """{full sha: [paths touched]} for every commit reachable from main, one git call.

    The per-task form ran cat-file, merge-base and show for each closed task: 32 tasks
    were 96 subprocesses and 4.9 s alone, over the 5 s deadline under hook contention,
    so the check timed out three times running and became a permanent red (2026-09-02).

    `-m` IS THE FIX FOR MERGES, and `--first-parent` is NOT part of it. Plain
    `git log --name-only` prints no paths at all for a merge commit -- git suppresses merge diffs
    by default -- so a delivery that landed inside a merge read as `touches []` and could not
    close its task, while a task closed against such a sha would equally never be caught.
    MEASURED on main (de, 2026-09-03), three options:

        plain               2756 commits, 616 seen as touching nothing (607 of them merges)
        -m --first-parent   1471 commits,   2 -- but 1285 commits MISSING, a worse blind spot:
                            --first-parent stops walking merged branches, so every commit that
                            reached main THROUGH a merge disappears from the map entirely
        -m alone            2756 commits,  10 -- all commits kept, nothing missing

    So `-m` alone. The first attempt at this fix used `-m --first-parent`, which reads as an
    improvement (2 empties beats 616) and silently drops nearly half the history; the count of
    commits, not just the count of empties, is what separates them. `git show --stat` on one of
    the 607 lists 7 files, which is how the disagreement surfaced -- closing de-30 against
    c889bc2.

    A merge under `-m` emits ONE BLOCK PER PARENT, each repeating the same %H, so the parse must
    UNION rather than assign: `out[sha] = paths` keeps only the last block. Measured, 588 shas
    have more than one block and 192 of the first 200 have a union larger than their last block --
    the worst carries 4 paths across two 3-path blocks. A file delivered against the first parent
    and absent from the second would read as not delivered.
    """
    if root not in _MAIN_TOUCHED:
        r = subprocess.run(["git", "-C", root, "log", "main", "-m",
                            "--name-only", "--format=%x00%H"],
                           capture_output=True, text=True)
        out = {}
        for block in r.stdout.split("\x00")[1:]:
            lines = block.split("\n")
            sha = lines[0].strip()
            paths = [p for p in lines[1:] if p.strip()]
            if sha in out:
                seen = set(out[sha])
                out[sha].extend(p for p in paths if p not in seen)
            else:
                out[sha] = paths
        _MAIN_TOUCHED[root] = out
    return _MAIN_TOUCHED[root]


def _resolve_shas(root, shas):
    """{sha as given: full sha or None} in ONE subprocess (_cat_file_exists' sibling).

    _commit_delivers ran `rev-parse --verify` per task: 86 closed tasks were 86
    subprocesses and 1.43 s of the check's 2.15 s, measured 2026-09-03, and the count
    grows by one every time a task closes. `cat-file --batch-check` resolves all 86 in
    0.023 s. Its output is one line per input line, in order -- `<full sha> commit <n>`
    for a hit, `<spec> missing` otherwise.

    A short sha resolves to its full form here, which is what the caller needs to index
    _main_touched. Verified on the real repo in _selftest_batched_git_probes, including
    that a duplicate input yields a duplicate output line, so zip stays aligned."""
    out = dict.fromkeys(shas, None)
    todo = [s for s in dict.fromkeys(shas) if s]
    if not todo:
        return out
    try:
        r = subprocess.run(
            ["git", "-C", root, "cat-file", "--batch-check"],
            input="".join(f"{s}^{{commit}}\n" for s in todo),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return out
    # rc != 0 is git declining to answer at all -- not a repository (the broken worlds and
    # the pod), a bad object database. Every spec then reads as unresolved, which is
    # exactly what the per-sha `rev-parse --verify -q` did, so the callers' behaviour is
    # unchanged. Distinguished from a TRUNCATED successful answer below, because those two
    # want opposite handling and one empty line looks like both: caught by the full
    # --selftest, where a temp repo with no .git made the guard raise instead of report.
    if r.returncode != 0:
        return out
    lines = r.stdout.split("\n")
    # Same contract as _cat_file_exists, and the same reason to assert it: an unresolved
    # sha here becomes "is not a commit in this repo", so a truncated output would refuse
    # a real delivery.
    if len(lines) < len(todo):
        raise RuntimeError(
            f"git cat-file --batch-check returned {len(lines)} line(s) for {len(todo)} "
            f"sha(s) -- one line per input line does not hold"
        )
    for s, line in zip(todo, lines[: len(todo)], strict=True):
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "commit":
            out[s] = parts[0]
    return out


def _commit_delivers(sha, evidence, root=None, tid=None, closed=None, resolved=None):
    """Empty string if sha reaches main and its diff touches a path named in evidence.

    The register's evidence field was free text: a path that never existed closed a
    task, and the register read as delivered. A commit hash is the one claim the repo
    can refute by itself -- it either resolves, reaches main, and moved that file, or
    it does not (user ruling 2026-09-01: the conversation is notification, the commit
    is the truth).

    `resolved` is an optional {sha: full-or-None} from _resolve_shas, so a caller with
    many shas pays one subprocess instead of one each. Absent, this resolves its own."""
    root = root or ROOT
    g = ["git", "-C", root]
    main_log = _main_touched(root)
    if resolved is not None and sha in resolved:
        full = resolved[sha] or ""
    else:
        full = subprocess.run(
            g + ["rev-parse", "--verify", "-q", f"{sha}^{{commit}}"], capture_output=True, text=True
        ).stdout.strip()
    if not full:
        return f"{sha} is not a commit in this repo"
    if full not in main_log:
        return f"{sha} does not reach main -- a delivery in a worktree is not delivered"
    touched = main_log[full]
    # A fact citation facts/<f>.json#<id> is the form check_fact_refs requires, and the
    # done gate rejected it as a nonexistent path (44-26). Strip the fragment for the
    # touched-file comparison, then assert the id lives in that file at HEAD.
    fact_refs = FACT_REF_RE.findall(evidence)
    paths = []
    tried = []
    for w in re.split(r"\s+", evidence):
        # STRIP THE PUNCTUATION PROSE PUTS AROUND A PATH, both ends. `,;:'"` alone was not
        # enough: evidence is written as prose, so a path arrives parenthesised
        # ("(facts/efficiency.json)"), backticked, bracketed, or ending a sentence. MEASURED
        # 2026-09-04 on eight forms -- 4 missed, and the refusal then said the commit does
        # not touch the named files, which points at the wrong cause entirely (e1's report;
        # its trailing-comma case already passed, `,` was in the old set).
        #
        # The parenthesised FACT CITATION is the one that failed twice over: "(facts/x.json#id)"
        # never matched the `"#" in w` split either, so the id check below was skipped in
        # silence -- a citation nobody verified, reading as a citation that resolved.
        #
        # A trailing `.` is stripped only from a token that ALREADY looks like a path, never
        # before the test: stripping first turns "done." into "done" and prose starts matching.
        # And the test is on the EXTENSION, not on the slash -- my first version asked "is this
        # path-shaped" first, so "facts/efficiency.json." passed on its slash and kept the dot,
        # which is the defect being fixed, one form later (measured, 11/12 before this line).
        w = w.strip(" ,;:'\"`()[]{}<>")
        exts = (".py", ".json", ".md", ".sh", ".jsonl")
        if w.endswith(".") and ("/" in w or w.rstrip(".").endswith(exts)):
            w = w.rstrip(".")
        if not ("/" in w or w.endswith(exts)):
            continue
        if w.startswith("facts/") and "#" in w:
            w = w.split("#", 1)[0]
        tried.append(w)
        paths.append(w)
    if not paths:
        return f"evidence names no path, so nothing can be checked against {sha[:8]}"
    if not any(any(t == p or t.startswith(p.rstrip("/") + "/") for t in touched) for p in paths):
        # NAME THE TOKENS TRIED AS PATHS. Without them this refusal says "the commit does not
        # deliver what the evidence claims" for two different causes -- a genuinely wrong
        # commit, and a path this function failed to parse out of prose -- and the reader
        # cannot tell which (e1, 2026-09-04).
        return (f"{sha[:8]} touches {touched[:3]} but evidence names {paths[:3]} -- "
                f"the commit does not deliver what the evidence claims "
                f"(tokens read as paths: {tried[:5]})")
    for fname, fid in fact_refs:
        r = subprocess.run(g + ["show", f"HEAD:facts/{fname}.json"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return f"evidence cites facts/{fname}.json#{fid} but that file is not at HEAD"
        try:
            ids = {e.get("id") for e in json.loads(r.stdout).get("facts", [])}
        except ValueError:
            return f"evidence cites facts/{fname}.json#{fid} but that file is not valid JSON at HEAD"
        if fid not in ids:
            return (f"evidence cites facts/{fname}.json#{fid} but that id is not in the file "
                    f"at HEAD -- the citation does not resolve")
    when = _main_when(root).get(full, "")
    if closed and closed >= "2026-09-02" and when and when > closed[:16] + ":59":
        return (f"{sha[:8]} was committed at {when}, after the row closed at {closed} -- "
                "a delivery cited after the fact is a repair of the register, not the delivery; "
                "reopen and close again on the commit that exists")
    return ""


def cmd_task(argv):
    """harness task {add,done,list}. The controller's assignments, in a file rather than
    in a conversation -- a conversation gets compacted and the assignment goes with it.

      harness task add  --owner b0 --task "..." --why "..." [--reading "..."] [--blocked-on t03]
      harness task done b0-1 --evidence "runs/l1_p324_d0.log: 0-shot answer-present 41.2%"
      harness task reopen b0-1 --why "pack half landed; v4 run still open"
      harness task list [--all]

    `done` REQUIRES evidence, and the check enforces it: a task closed without an artifact
    is a session reporting itself complete, which is the one thing the board footer forbids.
    `reopen` keeps the prior evidence and appends the reason; the check accepts the transition.
    """
    # os.path.exists, not isdir: in a linked worktree .git is a FILE pointing at the
    # common dir, and isdir refused every task command from a worktree (2026-08-31,
    # the same shape as install-hooks' NotADirectoryError).
    if not os.path.exists(os.path.join(ROOT, ".git")):
        print("refusing: the register lives in the repo; run this in the tree", file=sys.stderr)
        return 1
    ap = argparse.ArgumentParser(prog="harness task")
    sub = ap.add_subparsers(dest="op", required=True)
    a = sub.add_parser("add")
    a.add_argument("--owner", required=True)
    a.add_argument("--socket", required=True,
                   help="the owner's socket address; names collide, sockets do not")
    a.add_argument("--task", required=True)
    a.add_argument("--why", required=True, help="why this is worth a session's time")
    a.add_argument("--reading", default=None, help="how to read the result, written BEFORE it exists")
    a.add_argument("--pair", required=True,
                   help="the second session who agreed this task before it started, and who "
                        "second-reads it after; NOT a co-executor and not the owner -- the pair "
                        "reviews, the owner writes")
    a.add_argument("--prior", required=True,
                   help="what is already known: an arXiv id, a facts/<f>.json#<id>, or the literal "
                        "'defect-fix' when the task repairs our own code and no prior art applies")
    a.add_argument("--dup-ok", dest="dup_ok", action="store_true",
                   help="proceed past the overlap refusal; say in --why how this differs")
    a.add_argument("--blocked-on", dest="blocked_on", default=None)
    d = sub.add_parser("done")
    d.add_argument("id")
    d.add_argument("--reviewer", required=True,
                   help=f"who reads this delivery; a roster member other than the owner {sorted(set(REVIEW_PAIRS))}")
    d.add_argument("--evidence", required=True, help="artifact path, command, or fact id -- not a claim")
    d.add_argument("--commit", required=True,
                   help="the commit that delivers it: must reach main and must touch --evidence")
    r = sub.add_parser("reopen")
    r.add_argument("id")
    r.add_argument("--why", required=True, help="why this task is being reopened")
    p = sub.add_parser("drop")
    p.add_argument("id")
    p.add_argument("--why", required=True, help="why this will not be done; names what superseded it")
    sub.add_parser("list").add_argument("--all", action="store_true", help="include closed tasks")
    args = ap.parse_args(argv)
    rows = _read_tasks()

    if args.op == "add":
        # Owner-scoped ids: a global max+1 collides when two branches allocate
        # concurrently and the union merge keeps both (t52 twice, 2026-08-31).
        # <owner>-<n> is collision-free across branches; existing t-ids stay.
        n = max([int(r["id"].split("-", 1)[1]) for r in rows
                 if re.fullmatch(rf"{re.escape(args.owner)}-\d+", r.get("id", ""))] or [0]) + 1
        if args.pair == args.owner:
            print(f"refusing: {args.owner} cannot pair with itself", file=sys.stderr)
            return 1
        if args.pair not in REVIEW_PAIRS:
            print(f"refusing: {args.pair} is not on the roster {sorted(set(REVIEW_PAIRS))}", file=sys.stderr)
            return 1
        # The same cache task went to two people nine minutes apart and both stayed open
        # all night; --pair cannot see it, because each row had one. Nothing compared the
        # rows to each other. Distinctive words, so two rows about different subjects that
        # share "the token cache" do not collide but two about the same one do.
        def _words(s):
            return {w for w in re.findall(r"[a-z_]{5,}", (s or "").lower())} - _TASK_STOPWORDS
        new = _words(args.task)
        for t in rows:
            if t.get("state") != "open" or not new:
                continue
            old = _words(t.get("task"))
            if old and len(new & old) / min(len(new), len(old)) >= 0.5 and not args.dup_ok:
                print(f"refusing: {t['id']} ({t.get('owner')}) overlaps this task -- "
                      f"{sorted(new & old)[:6]}. Fold into it, or pass --dup-ok saying why "
                      "they are different", file=sys.stderr)
                return 1
        row = {
            "id": f"{args.owner}-{n}",
            "owner": args.owner,
            "socket": args.socket,
            "pair": args.pair,
            "prior": args.prior,
            "state": "open",
            "task": args.task,
            "why": args.why,
            "reading": args.reading,
            "blocked_on": args.blocked_on,
            "opened": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
            "evidence": None,
        }
        _append_task(row)
        print(f"{row['id']} -> {args.owner}: {args.task[:70]}")
        return 0

    if args.op == "done":
        hit = [r for r in rows if r.get("id") == args.id]
        if not hit:
            print(f"no task {args.id}; `harness task list` shows what is open")
            return 1
        owner = hit[0].get("owner")
        if args.reviewer == owner:
            print(f"refusing: {args.reviewer} owns {args.id}; a delivery needs a second reader", file=sys.stderr)
            return 1
        if args.reviewer not in REVIEW_PAIRS:
            print(f"refusing: {args.reviewer} is not on the roster {sorted(set(REVIEW_PAIRS))}", file=sys.stderr)
            return 1
        bad = _commit_delivers(args.commit, args.evidence, None, args.id)
        if bad:
            print(f"refusing: {bad}", file=sys.stderr)
            return 1
        # Append the new state as an event; never rewrite the row (see _read_tasks).
        ev = dict(hit[0], state="done", evidence=args.evidence, reviewer=args.reviewer,
                  commit=args.commit, closed=time.strftime("%Y-%m-%d %H:%M", time.gmtime()))
        _append_task(ev)
        print(f"{args.id} done: {args.evidence[:80]}")
        return 0

    if args.op == "reopen":
        hit = [r for r in rows if r.get("id") == args.id]
        if not hit:
            print(f"no task {args.id}; `harness task list --all` shows what is closed")
            return 1
        if hit[0].get("state") != "done":
            print(f"{args.id} is {hit[0].get('state')}, not done -- only done tasks can reopen")
            return 1
        ev = dict(
            hit[0],
            state="open",
            reopen_reason=args.why,
            reopened=time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
            evidence=hit[0].get("evidence", ""),  # keep prior evidence; the check accepts open+evidence
        )
        ev.pop("closed", None)
        _append_task(ev)
        print(f"{args.id} reopened: {args.why[:80]}")
        return 0

    if args.op == "drop":
        hit = [r for r in rows if r.get("id") == args.id]
        if not hit:
            print(f"no task {args.id}; `harness task list` shows what is open")
            return 1
        if hit[0].get("state") != "open":
            print(f"{args.id} is {hit[0].get('state')}, not open")
            return 1
        _append_task(dict(hit[0], state="dropped", drop_reason=args.why,
                          closed=time.strftime("%Y-%m-%d %H:%M", time.gmtime())))
        print(f"{args.id} dropped: {args.why[:80]}")
        return 0

    show = rows if args.all else [r for r in rows if r.get("state") == "open"]
    for r in show:
        blocked = f"  [blocked on {r['blocked_on']}]" if r.get("blocked_on") else ""
        print(f"{r['id']} {r.get('state', '?'):5} {r.get('owner', '?'):8} {r.get('task', '')[:78]}{blocked}")
    print(f"\n{len(show)} task(s); {sum(1 for r in rows if r.get('state') == 'open')} open of {len(rows)}")
    return 0


def _task_open_run(name, hypothesis):
    """Auto-open a task row for a pipeline run. A run that nobody assigned still
    needs a row, or its landing closes nothing."""
    rows = _read_tasks()
    n = max([int(r["id"][1:]) for r in rows if re.fullmatch(r"t\d+", r.get("id", ""))] or [0]) + 1
    row = {
        "id": f"t{n:02d}",
        "owner": "pipeline",
        "state": "open",
        "task": f"run point {name}",
        "why": hypothesis or f"0830v1 budget point {name}",
        "reading": "score_matrix record is the result; the fit interprets",
        "blocked_on": None,
        "opened": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        "evidence": None,
    }
    _write_tasks(rows + [row])
    return row["id"]


def _task_close_run(name, evidence):
    """Auto-close the pipeline-opened task for a run that landed. Returns the
    closed task id, or None if no matching open row exists."""
    rows = _read_tasks()
    for r in reversed(rows):  # most recent first: a rerun opens a second row
        if (
            r.get("state") == "open"
            and r.get("owner") == "pipeline"
            and r.get("task") == f"run point {name}"
        ):
            r.update(state="done", evidence=evidence, closed=time.strftime("%Y-%m-%d %H:%M", time.gmtime()))
            _write_tasks(rows)
            return r["id"]
    return None


def check_tasks_well_formed(root):
    """A closed task carries an artifact; an open one carries an owner and a reason; a DROPPED
    one carries the reason it was dropped.

    `drop_reason` is a separate field from `why` and the distinction is the point: `why` is the
    justification the row was OPENED with, and it survives a drop unchanged, so a dropped row
    always has a `why` and that tells a reader nothing about why the work stopped. Measured
    2026-09-04 (tilerl's triage d039a32e, verified here): 52 dropped rows, 45 carried
    `drop_reason` and 7 did not -- e1-21, e1-25, e1-27, e1-29, e1-30, tilerl-1, tilerl-10 -- and
    this check passed on all seven, because it only ever asserted `why`.

    A PLAIN FAIL, and it was WARN for about two hours. Those seven rows belonged to e1 and
    tilerl, and nobody else could fill the field: the value IS the reason, so a value written by
    a third party is a fabricated one. FAILing immediately would have made `harness check
    --selftest` red on the real tree for every session until the owners acted, which _demo
    forbids (every check PASSes or SKIPs on the real tree at the moment it lands) and which is
    the permanent-red shape -- a red nobody can clear is the same as no signal. So the clause
    shipped with DROP_REASON_GRANDFATHERED: a dated literal list, WARN by name, shrink-only,
    to be deleted when it emptied. tilerl filled two within the hour and e1 all five at
    aaf02e47; at 482 events / 51 dropped / 51 carrying the field the list was empty and is gone,
    exactly as its own comment said. Recorded because the ratchet is the reusable part: a new
    required field on existing rows their author cannot fill is a dated WARN list plus a deletion
    condition, not a FAIL that blocks every session and not a rule left unenforced.
    """
    rows = _read_tasks(os.path.join(root, "runs", "tasks.jsonl"))
    if not rows:
        return SKIP, "no task register"
    bad = []
    # The register is an event log: repeated ids are state changes, not duplicates.
    # A real collision is two rows sharing an id but naming DIFFERENT tasks -- two
    # branches allocating the same id independently (t52 twice, 2026-08-31). `opened`
    # is the discriminator: one task is opened once, whatever its later events.
    opened_by_id = {}
    for r in _read_tasks(os.path.join(root, "runs", "tasks.jsonl"), raw=True):
        opened_by_id.setdefault(r.get("id"), set()).add(r.get("opened"))
    collisions = sorted(i for i, o in opened_by_id.items() if len(o) > 1)
    if collisions:
        bad.append(f"id collision (same id, different tasks): {', '.join(collisions)}")
    no_reason = []
    for r in rows:
        if r.get("state") == "done" and not (r.get("evidence") or "").strip():
            bad.append(f"{r.get('id')} done without evidence")
        if r.get("state") == "open" and not (r.get("owner") or "").strip():
            bad.append(f"{r.get('id')} open without an owner")
        if not (r.get("why") or "").strip():
            bad.append(f"{r.get('id')} has no why")
        if r.get("state") == "dropped" and not str(r.get("drop_reason") or "").strip():
            no_reason.append(str(r.get("id")))
    if no_reason:
        bad.append(f"{len(no_reason)} dropped without drop_reason: {', '.join(sorted(no_reason))}"
                   f" -- `why` is the reason it was OPENED and survives a drop, so it cannot say "
                   f"why the work stopped")
    if bad:
        return FAIL, "; ".join(bad[:3])
    n_open = sum(1 for r in rows if r.get("state") == "open")
    n_drop = sum(1 for r in rows if r.get("state") == "dropped")
    return PASS, (f"{len(rows)} task(s), {n_open} open, every closed one carries an artifact, "
                  f"every one of {n_drop} dropped carries a drop_reason")


def _broken_tasks_well_formed():
    """The REAL register with a genuine id collision: the first row's id reused by a
    DIFFERENT task (different `opened`), which is what two branches allocating max+1
    independently produce (t52 twice, 2026-08-31). An exact duplicate would not do --
    that is a legal state-change event under the event-log semantics."""
    d = _tmp_repo()
    p = os.path.join(d, "runs", "tasks.jsonl")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    rows = _read_tasks(raw=True)
    if not rows:  # nothing real to mutate; the check SKIPs and the selftest would be a fiction
        return None
    rows = rows + [dict(rows[0], opened="2020-01-01 00:00", task="a different task, same id")]
    _write_tasks(rows, p)
    return d


def _broken_tasks_drop_reason():
    """A SECOND world for the drop_reason clause: the real register, REPAIRED, then one real
    dropped row's `drop_reason` removed.

    The repair is what makes it a discriminator, and it was load-bearing when this was written:
    the live register missed the field on seven rows, so a world that merely added an eighth
    proved nothing -- undo the mutation and it reported the same tier. Every dropped row is
    filled in, so the only remaining offender is the planted one, and putting the field back
    makes the world PASS. The selftest asserts BOTH halves, which is what makes the FAIL a
    statement about the mutation rather than about the register. Kept after e1 filled the last
    five (51/51 now carry the field, so a bare copy would PASS and a single mutation would
    already discriminate): the fixture no longer depends on the register's state, and reverting
    it would make this world fragile again the next time a row lands without the field.

    Filling the others in is legitimate for a fixture and would not be for the register: the
    value written is the literal string "(filled by the fixture)", which no reader could mistake
    for a real reason.
    """
    d = _tmp_repo()
    p = os.path.join(d, "runs", "tasks.jsonl")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    raw = _read_tasks(raw=True)
    if not raw:
        return None
    dropped = [r.get("id") for r in _read_tasks() if r.get("state") == "dropped"]
    if not dropped:
        raise SelftestSkip("the real register holds no dropped row to mutate")
    victim = dropped[0]
    rows = []
    for r in raw:
        r = dict(r)
        if r.get("state") == "dropped":
            # repair every dropped row, then re-break exactly one
            r["drop_reason"] = "" if r.get("id") == victim else "(filled by the fixture)"
        rows.append(r)
    _write_tasks(rows, p)
    return d


#: A task open longer than this without movement is forgotten. 3 days: sessions are
#: hours-long, so a task surviving 3 days has crossed multiple session boundaries;
#: no_stale_running uses 24h because experiments have a shorter lifecycle.
_TASK_STALE_DAYS = 3


def check_tasks_stale(root):
    """Open tasks that are forgotten: unblocked but not picked up, or untouched for days.

    Two failure modes, same disease as no_stale_running:
    - blocked_on points to a done task — the work is unblocked, nobody moved it. FAIL.
    - open and unblocked for > _TASK_STALE_DAYS days — the owner or controller forgot. WARN.
    """
    rows = _read_tasks(os.path.join(root, "runs", "tasks.jsonl"))
    if not rows:
        return SKIP, "no task register"
    done_ids = {r["id"] for r in rows if r.get("state") == "done"}
    stale = []
    for r in rows:
        if r.get("state") != "open":
            continue
        tid = r.get("id", "?")
        blocked = r.get("blocked_on")
        if blocked and blocked in done_ids:
            stale.append(
                (FAIL, f"{tid} blocked on {blocked} which is done — unblocked, not picked up")
            )
        elif not blocked:
            opened = r.get("opened", "")
            if opened:
                try:
                    t = time.strptime(opened, "%Y-%m-%d %H:%M")
                    age_days = (time.time() - time.mktime(t)) / 86400
                    if age_days > _TASK_STALE_DAYS:
                        stale.append(
                            (WARN, f"{tid} open {age_days:.0f}d (owner {r.get('owner', '?')})")
                        )
                except (ValueError, OverflowError):
                    pass
    if not stale:
        return PASS, "no stale open tasks"
    worst = FAIL if any(s == FAIL for s, _ in stale) else WARN
    return worst, "; ".join(ev for _, ev in stale[:5])


def _broken_tasks_stale():
    """An open task blocked on a done task — unblocked but not picked up."""
    d = _tmp_repo()
    p = os.path.join(d, "runs", "tasks.jsonl")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    rows = _read_tasks()
    if not rows:
        return None
    done = next((r for r in rows if r.get("state") == "done"), None)
    if not done:
        return None
    rows.append(
        {
            "id": "t99",
            "owner": "test",
            "state": "open",
            "task": "stale: blocked on a done task",
            "why": "selftest",
            "reading": None,
            "blocked_on": done["id"],
            "opened": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
            "evidence": None,
        }
    )
    _write_tasks(rows, p)
    return d


# --------------------------------------------------------------------------- lane

_TRAINING_PROCS = ("train.py", "sft.py", "sft_math.py", "torchrun", "run_ddp.sh", "rlvr.py")


def _has_training_process():
    """Whether a training process runs in this container.

    nvidia-smi reports host PIDs that ps inside a container cannot resolve, so
    process-to-GPU mapping is unavailable. This checks the container's own
    process list instead: if a training process exists, GPU usage is assumed
    to be from training. The ceiling is occupancy, not identity.
    """
    fake = os.environ.get("HARNESS_TRAINING_PROC")
    if fake is not None:
        return fake == "1"
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10)
        return any(k in out.stdout for k in _TRAINING_PROCS)
    except (OSError, subprocess.SubprocessError):
        return False


def _busy_training_cards(train_cards):
    """(busy, error). busy is the subset of train_cards with a GPU compute app.
    error=None on success, "not_found" if nvidia-smi is missing, or a message
    if it exists but fails — a GPU machine with a broken instrument must FAIL,
    not go silent.

    HARNESS_BUSY_CARDS injects a comma-separated list for the selftest.
    """
    fake = os.environ.get("HARNESS_BUSY_CARDS")
    if fake is not None:
        return [c.strip() for c in fake.split(",") if c.strip() in train_cards], None
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except FileNotFoundError:
        return [], "not_found"
    except (OSError, subprocess.SubprocessError) as e:
        return [], f"nvidia-smi failed to run: {e}"
    if apps.returncode != 0:
        return [], f"nvidia-smi compute-apps exited {apps.returncode}: {apps.stderr[:200]}"
    if gpus.returncode != 0:
        return [], f"nvidia-smi gpu-query exited {gpus.returncode}: {gpus.stderr[:200]}"
    by_uuid = {}
    for line in gpus.stdout.splitlines():
        if "," in line:
            i, u = line.split(",", 1)
            by_uuid[u.strip()] = i.strip()
    busy_uuids = {l.strip() for l in apps.stdout.splitlines() if l.strip()}
    busy = [c for c in sorted(train_cards) if any(by_uuid.get(u) == c for u in busy_uuids)]
    return busy, None


CARD_HELD_MIB = 1000


def check_card_held_without_claim(root):
    """A card holding real memory that no live claim names.

    e1's Stage A held card 5 for 15 minutes with the claim written in its LAPTOP tree, so
    nothing on the pod said the card was held: claims live where the job runs
    (scripts/card_claim.py reads runs/claims/ in the tree it is invoked from), and a claim on
    the wrong side of the boundary is invisible to everyone who looks.

    THE SPEC'S MECHANISM DOES NOT WORK HERE AND THIS IS WHY THE CHECK IS SHAPED DIFFERENTLY.
    It asked to report a pid that is "not in our namespace" as foreign. Measured on the pod
    2026-09-03: container nvidia-smi reports HOST pids (2274285, 2274286, ...) and container
    `ps` resolves NONE of them -- not even our own live training ranks, whose claim sits right
    there in runs/claims/ naming container pid 3818363. `--query-compute-apps=process_name`
    returns `[Not Found]` for every one. So "unresolvable pid" is the normal case for every
    process on the machine, ours included, and a foreign/ours split built on it would mark
    every card foreign. AGENTS.md already states the general form: a PID is only meaningful
    in the namespace that read it, and GPU UUID plus cmdline are the only cross-boundary
    identities.

    What the check can therefore say, and does: this card holds N MiB and no live claim in
    THIS tree names it. It cannot say whose process it is. That is still the whole content of
    the incident -- e1's card 5 had memory and no claim here -- and it is honest about the
    limit instead of inventing an attribution. Foreign containers are named as a possible
    cause in the message, not decided between.

    WARN, never FAIL: an unclaimed card is a real state needing a person, but it is also what
    a legitimate job looks like in the seconds between allocating memory and writing its
    claim, and what another team's container looks like permanently."""
    sys.path.insert(0, os.path.join(root, "scripts"))
    try:
        import card_claim
    except Exception as e:
        return SKIP, f"scripts/card_claim.py not importable: {type(e).__name__}: {e}"
    # The claim dir of the tree being checked, not this process's default: on the pod the
    # harness runs from /work/aupai and must read /work/aupai/runs/claims.
    claim_dir = os.path.join(root, "runs", "claims")
    saved = card_claim.CLAIM_DIR
    try:
        card_claim.CLAIM_DIR = claim_dir
        live, _stale = card_claim.claims()
        held = card_claim.held_cards(live)
        mem = card_claim.card_memory()
        # A ZOMBIE-HELD CARD IS CLAIMED, AND THIS CHECK USED TO SAY "no claim" FOR IT (de-51).
        # claims() files a zombie as LIVE on purpose -- acquire deletes whatever claims() calls
        # stale, so calling it stale would hand a starting job's cards away -- so a card held by
        # a corpse lands in `held` and the WARN below never fires, or fires with the wrong
        # reason for a neighbouring card. The three causes the message offers (another tree, not
        # claimed yet, another container) all send the reader outside this tree, and the claim
        # is right here. Named separately, WARN either way, because the action differs: a
        # foreign card needs identifying, a zombie-held one needs `card_claim.py release`.
        zombie_held = {}
        for c in live:
            p = c.get("pid")
            if isinstance(p, int) and card_claim._is_zombie(p):
                for card in c.get("cards", []):
                    zombie_held[str(card)] = (c.get("name"), p)
    finally:
        card_claim.CLAIM_DIR = saved
    if mem is None:
        return SKIP, "no nvidia-smi here -- this check is pod-side"
    unclaimed = [(c, m) for c, m in sorted(mem.items(), key=lambda kv: int(kv[0]))
                 if m > CARD_HELD_MIB and c not in held]
    zombie_busy = [(c, m, zombie_held[c]) for c, m in sorted(mem.items(), key=lambda kv: int(kv[0]))
                   if m > CARD_HELD_MIB and c in zombie_held]
    if unclaimed or zombie_busy:
        parts = []
        if unclaimed:
            detail = ", ".join(f"card {c} {m} MiB" for c, m in unclaimed)
            parts.append(
                f"{len(unclaimed)} card(s) hold memory no live claim in "
                f"{os.path.relpath(claim_dir, root)} names: {detail} -- either a claim written in "
                f"another tree (claims live where the job runs), a job that has not claimed yet, "
                f"or another container. Whose it is cannot be read here: nvidia-smi gives host "
                f"pids that this namespace cannot resolve, so identify it by GPU UUID plus "
                f"cmdline (nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory)")
        if zombie_busy:
            zd = ", ".join(f"card {c} {m} MiB claimed by {nm} on pid {p} (Z)"
                           for c, m, (nm, p) in zombie_busy)
            parts.append(
                f"{len(zombie_busy)} card(s) are claimed by a ZOMBIE process, so the claim is "
                f"real and its job has ended: {zd} -- do not look for another tree or another "
                f"container, release it (card_claim.py release --name <name>). claims() keeps a "
                f"zombie LIVE on purpose, because acquire deletes what it calls stale")
        return WARN, " AND ".join(parts)
    n_busy = sum(1 for m in mem.values() if m > CARD_HELD_MIB)
    return PASS, (f"{len(mem)} card(s), {n_busy} above {CARD_HELD_MIB} MiB, every one named by a "
                  f"live claim ({len(live)} claim(s))")


def _broken_card_held_without_claim():
    """The REAL claim dir with one live claim's cards emptied, so its card holds memory that
    no claim names -- exactly e1's state, produced by mutation rather than by hand.

    SKIPs where there is no nvidia-smi, because the check SKIPs there too: a world that
    cannot reach the instrument certifies nothing about it. On a cardless box this is the
    honest outcome, and the pod is where the world has teeth."""
    import shutil as _sh
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import card_claim
    if card_claim.card_memory() is None:
        raise SelftestSkip("no nvidia-smi: card_held_without_claim SKIPs here, so its world can "
                           "only be built on the pod")
    real = os.path.join(ROOT, "runs", "claims")
    d = _tmp_repo()
    os.makedirs(os.path.join(d, "runs", "claims"), exist_ok=True)
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    _sh.copy(os.path.join(ROOT, "scripts", "card_claim.py"), os.path.join(d, "scripts", "card_claim.py"))
    names = sorted(os.listdir(real)) if os.path.isdir(real) else []
    if not names:
        raise SelftestSkip("no live claim to mutate: the world needs a real claim whose cards "
                           "can be emptied")
    for nm in names:
        src = os.path.join(real, nm)
        obj = json.load(open(src, encoding="utf-8"))
        obj["cards"] = []          # the claim stays live; it just names no card
        with open(os.path.join(d, "runs", "claims", nm), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
    return d


def check_lane_respected(root):
    """Training cards must not be partially occupied by non-training work.

    The criterion is partial vs full occupancy, not training vs non-training:
    - 0 busy cards → PASS (idle)
    - busy == world → PASS (the block is used as a block, regardless of what)
    - 0 < busy < world, no training process → FAIL (a small job tore the block)
    - 0 < busy < world, training process → PASS (training in progress)

    The rule protects the block's continuity: a 7-card DDP job needs all 7
    simultaneously, so a 10-minute eval on one card blocks a 55-minute run.
    A 7-card sharded eval that fills all 7 is legitimate — it uses the block
    as a block.

    Ceiling: a squatter that fills ALL world cards passes. This is deliberate —
    it does not happen in practice (filling 7 cards requires a 7-card task), and
    preventing it would need process identity, which is unavailable in a
    container (nvidia-smi reports host PIDs that ps cannot resolve).

    Selftest ceiling: the broken world injects HARNESS_BUSY_CARDS /
    HARNESS_TRAINING_PROC, bypassing the real nvidia-smi and ps reads. The
    selftest validates the decision logic, not the data acquisition — if the
    real nvidia-smi parse degrades to always-empty, selftest stays green. The
    only validation of the real read path is a manual run on the pod.

    Cardless machines SKIP. A GPU machine with a broken nvidia-smi FAILs.

    THE LANE CARD IS NOT A TRAINING CARD (de-49 / DL-10). The frozen config's `cards` is the
    ladder recipe's full set -- "0,1,2,3,4,5,6" -- and the controller narrows it per round in
    runs/card_assignment.json, which also names the lane. Deriving the training set from the
    frozen config alone counted the lane as one of ours, so a foreign process on the lane read
    as our own training and the check called the state healthy. MEASURED on the pod 2026-09-04
    from this check's own output: `[PASS] lane_respected training cards [0, 5]: 2/7 busy
    (training in progress)` while `lane_card` was "5" and card 5 held 2000 MiB from another
    container. That inverts the rule the check exists for -- a foreign process on the lane is
    exactly what made the controller move the lane off card 6 the day before.

    So: `block_cards` and `lane_card` from card_assignment.json take precedence when present,
    the frozen config is the fallback, and the lane is subtracted either way. No foreign/ours
    process test is added: nvidia-smi reports HOST pids the container resolves for nothing --
    our own train.py is not in compute-apps at all -- so any namespace split marks every card
    foreign including ones running our training (6e withdrew that proposal on this measurement).
    """
    if not _gpu_present():
        return SKIP, "no GPUs on this machine"
    config_path = os.path.join(root, "data", "mix_scale_run_config.json")
    if not os.path.isfile(config_path):
        return SKIP, "no mix_scale_run_config.json"
    try:
        config = json.load(open(config_path, encoding="utf-8"))
        train_cards = {c.strip() for c in config["cards"].split(",") if c.strip()}
        world = int(config.get("world", len(train_cards)))
    except (json.JSONDecodeError, KeyError, ValueError):
        return SKIP, "cannot read cards/world from mix_scale_run_config.json"
    lane, src = set(), "frozen config"
    apath = os.path.join(root, "runs", "card_assignment.json")
    if os.path.isfile(apath):
        try:
            grant = json.load(open(apath, encoding="utf-8"))
        except (OSError, ValueError):
            grant = {}
        # _expand_cards (harness.py:13180) returns ints and handles both spellings -- the grant
        # file writes ranges ("0-3"), the frozen config writes lists. The card sets here are
        # strings because that is what nvidia-smi indices are compared as.
        block = {str(c) for c in _expand_cards(grant.get("block_cards"))}
        if block:
            train_cards, world, src = block, len(block), "card_assignment.block_cards"
        lane = {str(c) for c in _expand_cards(grant.get("lane_card"))}
        if lane:
            # The lane is whichever card is not in `cards` (AGENTS.md), so a lane that appears
            # in the training set is the defect, not a conflict to resolve either way.
            train_cards = train_cards - lane
            world = min(world, len(train_cards)) if train_cards else 0
    if not train_cards:
        return SKIP, f"no training card left after removing the lane {sorted(lane)}"
    busy, err = _busy_training_cards(train_cards)
    if err == "not_found":
        return SKIP, "nvidia-smi not installed"
    if err is not None:
        return FAIL, f"nvidia-smi broken: {err}"
    lane_busy = []
    if lane:
        lb, lerr = _busy_training_cards(lane)
        if lerr is None:
            lane_busy = lb
    where = f"{sorted(train_cards)} (from {src}, lane {sorted(lane) if lane else 'unknown'} excluded)"
    lane_note = ""
    if lane_busy:
        lane_note = (f"; LANE {','.join(sorted(lane_busy))} is occupied -- one job at a time, and "
                     f"whose it is cannot be read here (host pids)")
    if not busy:
        return (WARN, f"training cards {where}: idle{lane_note}") if lane_busy else \
               (PASS, f"training cards {where}: idle")
    if len(busy) >= world:
        return (WARN if lane_busy else PASS), \
            f"training cards {sorted(busy)}: all {world} busy (block used as block){lane_note}"
    if _has_training_process():
        return (WARN if lane_busy else PASS), \
            f"training cards {busy}: {len(busy)}/{world} busy (training in progress){lane_note}"
    return FAIL, (
        f"training cards {busy}: {len(busy)}/{world} busy but no training process — "
        f"a small job is tearing the block. Small jobs go on the lane card "
        f"({sorted(lane) if lane else 'the one not in ' + str(sorted(train_cards))}).{lane_note}"
    )


def _broken_lane_respected():
    """The MEASURED pod state of 2026-09-04: block 0-3, lane 5, cards 0 and 5 busy, no
    training process. This exact world printed PASS before de-49 -- "training cards [0, 5]:
    2/7 busy (training in progress)" -- because card 5 was counted as one of ours.

    The positive is asserted here too, in the same world, because an assertion that only
    demands FAIL passes on an implementation that never returns PASS.
    """
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    # Real config so the training card set is real, not hand-written.
    shutil.copy(
        os.path.join(ROOT, "data", "mix_scale_run_config.json"),
        os.path.join(d, "data", "mix_scale_run_config.json"),
    )
    # The REAL grant file with its block and lane kept, so the lane the check subtracts is the
    # controller's, not one this world invented.
    real_grant = os.path.join(ROOT, "runs", "card_assignment.json")
    if os.path.isfile(real_grant):
        shutil.copy(real_grant, os.path.join(d, "runs", "card_assignment.json"))
        grant = json.load(open(real_grant, encoding="utf-8"))
        lane = [str(c) for c in _expand_cards(grant.get("lane_card"))]
        block = [str(c) for c in _expand_cards(grant.get("block_cards"))]
    else:
        lane, block = [], []
    if not block:
        raise SelftestSkip("runs/card_assignment.json names no block_cards; there is no lane "
                           "violation to build")
    if not lane:
        # A GRANT WITH NO LANE CANNOT EXPRESS THIS CHECK'S DEFECT, and the fallback that used to
        # stand here quietly built a world with no violation at all: it marked the frozen config's
        # FIRST card busy, which on the 2026-09-04 grant (`lane_card: ""`, `block_cards: 1,2,4,6`)
        # is a block card, so the world read "training cards [1,2,4,6]: idle" and the check
        # correctly PASSed -- reported by _demo as "lane_respected cannot be made to fail".
        # A world that cannot express the defect SKIPs by name; it does not build a green one and
        # let the selftest interpret it. Measured: grant 8a7a0662 set lane_card empty for the
        # head-hybrid A/B, which runs with no lane card at all, and this selftest went red on a
        # check that works.
        raise SelftestSkip(
            f"runs/card_assignment.json grants no lane card (block {','.join(block)}), and the "
            f"defect this check catches is a busy LANE counted as one of the block's cards -- "
            f"there is no such card to mark busy, so the world would assert nothing")
    # One block card plus the lane, no training process: the lane must not make up the count.
    os.environ["HARNESS_BUSY_CARDS"] = f"{block[0]},{lane[0]}"
    os.environ["HARNESS_TRAINING_PROC"] = "0"
    # The positive, in this same world: every block card busy with a training process is
    # healthy, and if that does not PASS the FAIL above proves nothing.
    _saved = os.environ["HARNESS_BUSY_CARDS"], os.environ["HARNESS_TRAINING_PROC"]
    os.environ["HARNESS_BUSY_CARDS"] = ",".join(block)
    os.environ["HARNESS_TRAINING_PROC"] = "1"
    st, ev = check_lane_respected(d)
    assert st == PASS, f"the positive world does not PASS ({st}: {ev}) -- the FAIL below is empty"
    os.environ["HARNESS_BUSY_CARDS"], os.environ["HARNESS_TRAINING_PROC"] = _saved
    return d


# A CUDA_VISIBLE_DEVICES assignment is safe when its value comes from the shard map
# eval/_devs.sh builds (${_DEVS[...]}) or defers to the caller
# (${CUDA_VISIBLE_DEVICES:-...}). Anything else -- a literal, a bare $i, a seq
# expansion -- is a physical index that REPLACES the caller's restriction.
_CVD_SAFE = re.compile(r"^\$\{_DEVS\[|^\$\{CUDA_VISIBLE_DEVICES:-|^(?:\"\"|''|-1)$")
_CVD_ASSIGN = re.compile(r"(?:^|\s)(?:export\s+)?CUDA_VISIBLE_DEVICES=(\S+)")
# `=""` and `=-1` are ACCEPTED: they mean NO device, which cannot escape a lane. Measured
# on the pod 2026-09-03: with CUDA_VISIBLE_DEVICES="" torch.cuda.device_count() is 0 and
# is_available() False (unset gives 8; -1 also gives 0). Added because the rule this check
# enforces is "never take a card the caller did not give you", and asking for none is the
# strongest possible compliance -- while the syntax alone reads identically to writing a
# physical index. Without this, the only way to pass was to leave the variable inherited,
# i.e. a pure-CPU script stays able to open every visible card; that ambiguity is what
# turned a CPU counting job into a card-ownership investigation (2026-09-03). Empty is the
# idiom because it is what makes "this step uses no card" visible in the process's env.
# Known false positive, left in on purpose: this matches the TEXT, so a script that
# names the variable inside an error message ("set CUDA_VISIBLE_DEVICES=<n> first")
# reads as an assignment. Hit once, 2026-09-01, on run_sampled_arm.sh's usage string.
# Teaching it to parse shell quoting is more surface than the false positive costs, and
# the failure direction is right -- it over-reports rather than missing an escape. The
# fix at the call site is to reword the message, not to exempt the file.
# A script that reads _DEVS must have sourced eval/_devs.sh, which is what refuses a
# shard with no device to land on. Reconstructing the array with a bare `read -ra`
# reintroduces the :-$i fallback the helper exists to remove.
_CVD_SOURCE = re.compile(r"source\s+eval/_devs\.sh|\.\s+eval/_devs\.sh")
# Scripts whose whole job is to own the block; they define the allocation rather
# than live inside someone else's. Each is exempt for a reason recorded here, not
# because it was inconvenient to fix.
_CVD_EXEMPT = {
    "run_ddp.sh",          # the block launcher itself; its default is the full 8
    "scripts/run_pretrain.sh",   # direct-invocation launchers; harness launch overrides
    "scripts/run_sft.sh",        # CUDA_VISIBLE_DEVICES from the controller's card list
    "scripts/run_pipeline.sh",
    "eval/_devs.sh",       # the helper itself: it BUILDS the map
}


def _repo_owned_files(root):
    """Paths the repo owns, relative to root. git ls-files in a checkout; the pod
    manifest on the pod, which has no .git. None when neither is available.

    The pod tree is not a checkout and holds untracked scratch files beside the
    repo's own, so "every file under root" is the wrong scope for any check that
    can turn `harness check` red there."""
    if os.path.exists(os.path.join(root, ".git")):
        r = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
        if r.returncode == 0:
            return set(r.stdout.split())
    manifest = os.path.join(root, "data", "pod_head_manifest.txt")
    if os.path.isfile(manifest):
        out = set()
        for line in open(manifest, encoding="utf-8"):
            parts = line.split()
            if len(parts) >= 2:
                out.add(parts[1])
        return out or None
    return None


def check_device_set_honoured(root):
    """A shard script must take its device from eval/_devs.sh, never write a physical index.

    CUDA_VISIBLE_DEVICES is not additive: setting it in a child REPLACES the parent's
    restriction. A script that writes `=0` or `=$i` therefore escapes whatever lane the
    caller confined it to. On 2026-08-31 a lane-card launch (CUDA_VISIBLE_DEVICES=7) of
    eval/code_zh.py landed on physical GPU 0, a training-block card, and blocked t01
    (2f97e4a). Two survivals of that fix show why the class needs a check rather than
    five hand-edits:

    | survival | shape |
    |---|---|
    | eval_all.sh:54,61 | never edited; kept a bare `=0` for nine more hours |
    | all five shard scripts | `${_DEVS[$i]:-$i}` still spills when N exceeds the caller's device count -- `CUDA_VISIBLE_DEVICES=7` with the default N=6 puts shards 1-5 on physical 1-5 |

    Ceiling: this reads the assignment's syntax, not what runs. A script that computes a
    physical index into a variable and assigns the variable passes. Detecting that needs
    shell dataflow; the cheaper guarantee is one accepted idiom that greps.

    Scope: repo-owned scripts only, taken from git or the pod manifest. The pod tree also
    holds a dozen untracked scratch launchers (ab_launch.sh, fleet6.sh, ...) that predate
    the rule; failing on those would turn `harness check` red on the pod, and `harness run`
    refuses while check is red, so an unowned scratch file would block the pipeline."""
    owned = _repo_owned_files(root)
    if owned is None:
        return SKIP, "cannot enumerate repo-owned files (no git, no manifest)"
    bad = []
    checked = 0
    for rel in sorted(owned):
        if not rel.endswith(".sh"):
            continue
        if rel in _CVD_EXEMPT or os.path.basename(rel) in _CVD_EXEMPT:
            continue
        path = os.path.join(root, rel)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        checked += 1
        body = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
        for n, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            m = _CVD_ASSIGN.search(line)
            if m and not _CVD_SAFE.match(m.group(1)):
                bad.append(f"{rel}:{n} CUDA_VISIBLE_DEVICES={m.group(1)}")
        if any("_DEVS" in l for l in body) and not _CVD_SOURCE.search("\n".join(body)):
            bad.append(f"{rel} reads _DEVS without sourcing eval/_devs.sh")
    if not checked:
        return SKIP, "no repo-owned shell scripts present"
    if bad:
        return FAIL, (
            f"{len(bad)} assignment(s) write a physical index instead of taking the "
            f"caller's: {', '.join(bad[:4])}. Use `source eval/_devs.sh \"$N\"` then "
            f"${{_DEVS[$i]}}."
        )
    return PASS, f"{checked} repo-owned shell scripts take their device from the caller's set"


def _broken_device_set_honoured():
    """The real eval_all.sh with its fixed assignment reverted to the bare `=0` it
    carried until 2026-08-31 -- the exact line the check was written for."""
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    src = os.path.join(ROOT, "eval", "eval_all.sh")
    dst = os.path.join(d, "eval", "eval_all.sh")
    shutil.copy(src, dst)
    shutil.copy(os.path.join(ROOT, "eval", "_devs.sh"), os.path.join(d, "eval", "_devs.sh"))
    # No .git in a _tmp_repo, so the check reads its scope from the manifest.
    with open(os.path.join(d, "data", "pod_head_manifest.txt"), "w", encoding="utf-8") as f:
        f.write("0  eval/eval_all.sh  docs\n0  eval/_devs.sh  docs\n")
    s = open(dst, encoding="utf-8").read()
    fixed = "CUDA_VISIBLE_DEVICES=${_DEVS[0]} python3 eval/run_eval.py"
    assert fixed in s, "eval_all.sh no longer carries the fixed assignment; update _broken_device_set_honoured"
    open(dst, "w", encoding="utf-8").write(
        s.replace(fixed, "CUDA_VISIBLE_DEVICES=0 python3 eval/run_eval.py")
    )
    return d


def check_untracked_aged(root):
    """Untracked files older than _AGE_HOURS in the shared tree -- someone's unfinished work.

    In a multi-session tree an untracked file belongs to the session that made it.
    Past the window it is either forgotten or blocked; either way the owner should give
    it a fate (commit, gitignore, delete). WARN, not FAIL: the file is not wrong,
    it is just unowned."""
    if not os.path.exists(os.path.join(root, ".git")):  # worktree .git is a file
        return SKIP, "no .git (pod or partial checkout)"
    r = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root, capture_output=True, text=True,
    )
    if r.returncode != 0:
        return SKIP, f"git ls-files failed: {r.stderr.strip()}"
    cutoff = time.time() - _AGE_HOURS * 3600
    aged = []
    for f in r.stdout.splitlines():
        p = os.path.join(root, f)
        if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
            aged.append(f)
    if aged:
        return WARN, f"{len(aged)} untracked file(s) older than {_AGE_HOURS}h: {', '.join(aged[:5])}"
    return PASS, "no aged untracked files"


def _broken_untracked_aged():
    """A real git repo with one aged untracked file — the violation this check catches."""
    import shutil
    import subprocess as sp

    d = _tmp_repo()
    sp.run(["git", "init"], cwd=d, capture_output=True)
    # A real tracked file so the selftest's repo-real-path check passes.
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    sp.run(["git", "add", "AGENTS.md"], cwd=d, capture_output=True)
    sp.run(["git", "commit", "-m", "init"], cwd=d, capture_output=True)
    # An untracked file with a 48h-old mtime.
    stale = os.path.join(d, "stale_untracked.py")
    open(stale, "w").write("# stale\n")
    old = time.time() - 48 * 3600
    os.utime(stale, (old, old))
    return d


def check_dirty_aged(root):
    """Tracked files dirty longer than _AGE_HOURS -- uncommitted work sitting in the
    shared tree. In a multi-session tree a dirty file is a landmine: it blocks anyone
    who needs to push it, and a broad `git add` sweeps it into someone else's commit
    (d535674 swept 26 files; 2026-08-31 ruled that nothing stays uncommitted).
    WARN, not FAIL: the file is not wrong, its owner just has to commit or revert.
    The owner is unknown; the path is named so the standup can assign it."""
    if not os.path.exists(os.path.join(root, ".git")):  # worktree .git is a file
        return SKIP, "no .git (pod or partial checkout)"
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root, capture_output=True, text=True,
    )
    if r.returncode != 0:
        return SKIP, f"git status failed: {r.stderr.strip()}"
    cutoff = time.time() - _AGE_HOURS * 3600
    aged = []
    for line in r.stdout.splitlines():
        # XY porcelain: X = index status, Y = worktree status. A file staged and then
        # further modified reads "AM" -- `line[:2].strip() not in ("M","A")` filtered
        # that out, and on a box without git identity every broken-world commit fails
        # and leaves "AM", so the selftest went green while the check saw nothing (CI
        # 2026-08-31). Any M or A in either column is uncommitted work.
        if len(line) < 4 or not (set(line[:2]) & {"M", "A"}):
            continue  # untracked (??) is untracked_aged's job; deletes have no mtime
        p = os.path.join(root, line[3:])
        if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
            aged.append(line[3:])
    if aged:
        return WARN, f"{len(aged)} tracked file(s) dirty >{_AGE_HOURS}h: {', '.join(aged[:5])}"
    return PASS, "no aged dirty files"


def _broken_dirty_aged():
    """A real git repo with one tracked file dirty for longer than _AGE_HOURS. No git
    identity is configured, so the commit fails and the file sits staged-and-modified
    ("AM" in porcelain) -- the exact shape the old line[:2].strip() parser missed on CI.

    The age is DERIVED from the check's constant, not written beside it. A hardcoded
    2 hours was aged under the old 30-minute threshold and is not under 6 hours: raising
    the threshold turned this broken world green and the selftest caught it, which is
    the one thing a hand-written age cannot promise to keep doing."""
    import shutil
    import subprocess as sp

    d = _tmp_repo()
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    sp.run(["git", "init"], cwd=d, capture_output=True, env=env)
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    sp.run(["git", "add", "AGENTS.md"], cwd=d, capture_output=True, env=env)
    sp.run(["git", "commit", "-m", "init"], cwd=d, capture_output=True, env=env)
    with open(os.path.join(d, "AGENTS.md"), "a") as f:
        f.write("\n# dirty\n")
    old = time.time() - (_AGE_HOURS * 3600 + 600)
    os.utime(os.path.join(d, "AGENTS.md"), (old, old))
    return d


_STASH_GRACE_DAYS = 30


#: Paths that must not change on main while a training run holds the block. Not "files
#: that matter" -- files whose content the RUNNING job re-reads or is defined by, so a
#: change to one makes the live run and the tree describe different experiments.
_FROZEN_PATHS = (
    "train.py", "run_ddp.sh", "scripts/supervise_run.sh", "scripts/pod_drift.py",
    "eval/score_matrix.py", "eval/domain_loss.py", "data/mix_500m.json",
    "data/tokenizer.json",
)


_RUN_LOG_FRESH_S = 900


def _run_holds_the_block(root):
    """(True, note) while a training run owns the cards.

    TWO SOURCES, because neither reaches both places. runs/card_assignment.json is the
    grant and it is current in a git tree -- but pod_push skips runs/ by design
    (pod_drift.py:270), so the pod's copy is whatever it was when someone last put it
    there: on 2026-09-02 the pod still read a 09-01 grant saying `launch_block_granted:
    false` while eight cards were training. A check keyed only on that file disarms
    itself on the pod, which is the one place a mid-run edit actually lands.

    So the pod's source is a run log still being written: mtime within
    _RUN_LOG_FRESH_S. That is the same evidence a person uses (`tail` it and see it
    move), it needs nothing synced, and it cannot claim a run that has stopped. The
    grant is checked first because it is the authority where it is fresh, and it also
    covers the window between a crash and its resume, when no log is being written."""
    try:
        with open(os.path.join(root, "runs", "card_assignment.json"), encoding="utf-8") as f:
            a = json.load(f)
        if a.get("launch_block_granted") and \
                (a.get("next_grant") or {}).get("blocked_on") == "the run itself":
            return True, str(a.get("note", ""))[:60]
    except (OSError, ValueError):
        pass
    now = time.time()
    for p in glob.glob(os.path.join(root, "runs", "*.log")):
        try:
            if now - os.path.getmtime(p) < _RUN_LOG_FRESH_S:
                return True, f"{os.path.basename(p)} written in the last {_RUN_LOG_FRESH_S // 60} min"
        except OSError:
            continue
    return False, "no block grant and no run log written recently"


def check_frozen_paths(root):
    """While a run holds the block, main does not change what the run is made of.

    A commit message saying HOLD binds the person who wrote it, and `git merge` does not
    read English: da06097 carried "do not merge to main while p500m_20b_0902 is training"
    in its own first line, and I merged the branch that contained it 40 minutes later
    (2026-09-02). The revert was clean and that is not the point -- the same mistake with
    train.py would have made the tree describe a model the running job is not training.

    THE BASELINE IS THE SHA IN THE RUN'S LOG, not the one in its exp row. The exp row is
    stamped once at `exp start` and never again, so a relaunch leaves it naming code the
    job stopped executing 36 minutes later: p500m_20b_0902's row says dca9762 while the
    log's banner and every byte on the pod say cdfa1db. Reading the row made this check
    FAIL on two files that are byte-identical between main and the pod -- a stale baseline
    reports drift where there is none, which is worse than not checking, because a red
    that is always red gets muted.

    The log line is `pod code: <sha> (clean, synced ...)`, written by run_ddp.sh:41 from
    data/pod_synced_head at launch. CEILING, stated rather than hidden: that log is live
    on the pod and reaches a git tree only once committed, i.e. after the run, so this is
    armed where a mid-run edit lands and SKIPs on a Mac meanwhile. No banner: SKIP rather
    than fall back to the exp row -- that value is available and wrong, and a wrong answer
    is not better than none.

    # ponytail: banner-scrape, upgrade to a launch-written runs/<name>.base sha when a
    # second reader needs it."""
    ok, note = _run_holds_the_block(root)
    if not ok:
        return SKIP, f"no run holds the block ({note})"
    # The pod has no .git, so no sha resolves there and a `git diff` baseline cannot work
    # -- and the banner names a pre-rewrite sha that stopped existing on 2026-09-02 when
    # history was rewritten. On the pod the question is answered without git at all: do
    # the frozen files still hash to what the committed manifest says? That is the same
    # comparison pod_drift makes, narrowed to the paths that must not move mid-run.
    if pod_drift.is_pod(root):
        manifest = pod_drift.read_manifest(os.path.join(root, "data", "pod_head_manifest.txt"))
        bad = []
        for p in _FROZEN_PATHS:
            want = (manifest.get(p) or (None,))[0]
            full = os.path.join(root, p)
            if want is None or not os.path.exists(full):
                continue
            if pod_drift.sha_disk(full) != want:
                bad.append(p)
        if bad:
            return FAIL, (f"{len(bad)} frozen path(s) on the pod differ from the committed "
                          f"manifest while a run holds the block: {', '.join(bad[:4])}")
        return PASS, f"{len(_FROZEN_PATHS)} frozen paths match the manifest ({note})"
    rows = [r for r in (_exp_events(root) or []) if r.get("status") == "running"]
    base = None
    for r in reversed(rows):
        # `pod code: <sha>` is written by run_ddp.sh:41 from data/pod_synced_head at
        # launch. On the pod that log is live; in a git tree it is there once the log has
        # been committed, which is after the run ends. So this check is armed on the pod
        # (where a mid-run edit would actually land) and SKIPs on a Mac until the log
        # arrives -- stated rather than papered over with the exp row.
        try:
            with open(os.path.join(root, "runs", f"{r.get('name')}.log"),
                      encoding="utf-8", errors="replace") as f:
                for ln in f:
                    m = re.match(r"pod code: ([0-9a-f]{7,40})\b", ln)
                    if m:
                        base = m.group(1)
                        break
        except OSError:
            continue
        if base:
            break
    if not base:
        # No banner: the log is pod-only until the run ends. SKIP rather than substitute
        # a baseline. Both available substitutes are wrong in a way that produces a
        # standing red, and a check that is always red gets muted -- which is how the
        # guard fails. The exp row's `commit` is stamped once at start and goes stale on
        # a relaunch (dca9762 vs the live cdfa1db). The last commit before the row's
        # `started` is EARLIER than the code the pod actually holds, because pod_push
        # ships mid-run: it reported 5 changed paths here while the pod matched main
        # byte for byte. This check is armed on the pod, which is where a mid-run edit
        # lands; a Mac says nothing rather than something false.
        return SKIP, ("no `pod code:` banner in a committed run log -- this is armed on "
                      "the pod, where the live log and the manifest both exist")
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
                       cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        return FAIL, f"the running row names commit {base}, which is not in this repo"
    r = subprocess.run(["git", "diff", "--name-only", base, "HEAD", "--", *_FROZEN_PATHS],
                       cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        return SKIP, f"git diff failed: {r.stderr.strip()[:80]}"
    changed = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if changed:
        return FAIL, (f"{len(changed)} frozen path(s) changed on main since {base[:7]}, the commit "
                      f"the running job executes: {', '.join(changed[:4])}")
    return PASS, f"{len(_FROZEN_PATHS)} frozen paths unchanged since the run's {base[:7]}"


def _broken_frozen_paths():
    """A clone with a block grant, a running exp row naming the base commit, and a later
    commit touching train.py. Real commits, because the check compares two shas."""
    import shutil
    import subprocess as sp

    d = _tmp_repo()
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    ident = ["-c", "user.email=t@t", "-c", "user.name=t"]
    sp.run(["git", "init"], cwd=d, capture_output=True, env=env)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "train.py"), os.path.join(d, "train.py"))
    with open(os.path.join(d, "runs", "card_assignment.json"), "w") as f:
        json.dump({"launch_block_granted": True, "note": "broken world",
                   "next_grant": {"blocked_on": "the run itself"}}, f)
    sp.run(["git", "add", "-A"], cwd=d, capture_output=True, env=env)
    sp.run(["git", *ident, "commit", "-m", "base"], cwd=d, capture_output=True, env=env)
    base = sp.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True,
                  env=env).stdout.strip()
    # The running row names the commit the job executes -- written after that commit
    # exists, which is the same order pod_push stamps it in.
    with open(os.path.join(d, "runs", "experiments.jsonl"), "w") as f:
        f.write(json.dumps({"started": "2026-09-02 01:03", "name": "brokenworld",
                            "status": "running", "commit": base, "ended": ""}) + "\n")
    with open(os.path.join(d, "runs", "brokenworld.log"), "w") as f:
        f.write(f"pod code: {base} (clean, synced 2026-09-02T00:00:00Z, manifest verified)\n")
    with open(os.path.join(d, "train.py"), "a") as f:
        f.write("\n# the change that must not happen mid-run\n")
    sp.run(["git", "add", "-A"], cwd=d, capture_output=True, env=env)
    sp.run(["git", *ident, "commit", "-m", "touch a frozen path"], cwd=d,
           capture_output=True, env=env)
    return d


def _cfg_known_names(root):
    """Every name a `cfg`/`Cfg` object can legitimately carry -> (body, published).

    Two sources, and leaving either out makes the check useless in opposite directions:
    the Cfg class body (48 names), and every `cfg.x = ` / `Cfg.x = ` assignment anywhere in
    the tree (25 more) -- build_mix publishes _row_cursor, _plan_domains and friends onto Cfg
    at runtime, so a body-only reading would flag all of them."""
    train_py = os.path.join(root, "train.py")
    if not os.path.exists(train_py):
        return None, None
    try:
        tree = ast.parse(open(train_py, encoding="utf-8", errors="replace").read())
    except SyntaxError as e:
        return None, f"train.py does not parse: {e}"
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "Cfg"), None)
    if cls is None:
        return None, "train.py has no class Cfg"
    body = {x.id for s in cls.body if isinstance(s, ast.Assign)
            for x in s.targets if isinstance(x, ast.Name)}
    body |= {s.target.id for s in cls.body
             if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
    published = set()
    for p, txt in walk_tracked(root, (".py",)):
        try:
            t2 = ast.parse(txt)
        except SyntaxError:
            continue
        for n in ast.walk(t2):
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                for x in tg:
                    if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) \
                            and x.value.id in ("cfg", "Cfg"):
                        published.add(x.attr)
    return (body, published), None


def check_getattr_cfg_names_exist(root):
    """`getattr(cfg, "name", <non-None>)` names a field cfg can actually carry.

    getattr returns the SAME value when the name is absent as when it is present and equal
    to the default, so the call site cannot tell "read it" from "did not find it". 62's
    probe read `getattr(cfg, "logit_softcap", 0.0)` -- softcap is the module constant
    SOFTCAP at model.py:63, so the field does not exist, 0.0 came back silently, and
    post-softcap logits were reported as pre. The three values (14.62/14.69/14.54) were
    perfect evidence for the conclusion being argued, so nothing looked wrong; what gave it
    away was 14.62 sitting 0.4 under a hard ceiling of 15.0.

    The benign and the fatal spelling are IDENTICAL in source -- train.py:756's
    `getattr(cfg, "attn_res_lr", 0.01)` names a real field at :221 with a matching default
    -- which is why a human reading the line cannot separate them and a name check can.

    WHICH POSITIVES THIS DELIBERATELY MISSES, asked before writing it rather than after
    (the rule from docs/lessons/fact_and_inference.md):
      - a None default is a presence PROBE, not a value read: `getattr(cfg, "x", None)`
        followed by an `is None` branch is the correct way to ask, so it is skipped.
      - a non-literal name (`getattr(cfg, key, d)`) is not statically decidable; skipped.
      - objects other than a bare `cfg`/`Cfg` name (`self.cfg`, `ck["cfg"]`) are not
        followed, so this covers the spelling that has bitten us and not every reader.
      - a checkpoint's cfg DICT legitimately lacks fields added after it was written; those
        go through `.get()`, not getattr, so they are outside this check by construction."""
    known, err = _cfg_known_names(root)
    if err:
        return FAIL, err
    body, published = known
    allow = body | published
    bad = []
    for p, txt in walk_tracked(root, (".py",)):
        try:
            tree = ast.parse(txt)
        except SyntaxError:
            continue
        rel = os.path.relpath(p, root)
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "getattr" and len(n.args) == 3):
                continue
            obj, nm, dflt = n.args
            if not (isinstance(obj, ast.Name) and obj.id in ("cfg", "Cfg")):
                continue
            if not (isinstance(nm, ast.Constant) and isinstance(nm.value, str)):
                continue
            if isinstance(dflt, ast.Constant) and dflt.value is None:
                continue
            if nm.value in allow:
                continue
            bad.append(f"{rel}:{n.lineno} getattr({obj.id}, {nm.value!r}, ...)")
    if bad:
        return FAIL, (f"{len(bad)} getattr site(s) name a field Cfg does not carry, so the "
                      f"default comes back silently and reads as a measurement: "
                      f"{'; '.join(bad[:4])}")
    return PASS, f"every getattr(cfg, ...) name is one of {len(allow)} Cfg fields"


def _broken_getattr_cfg_names():
    """The REAL train.py with one getattr name misspelled -- mutated, not hand-written.

    `attn_res_lr` at :756 is the benign instance the docstring cites, so breaking exactly
    it makes the broken world the same shape as the defect: a real field name, off by a
    suffix, with a plausible default beside it."""
    import shutil

    d = _tmp_repo_shaped()
    real_train = os.path.join(d, "train.py")
    if os.path.islink(real_train):
        os.unlink(real_train)
    shutil.copy(os.path.join(ROOT, "train.py"), real_train)
    src = open(real_train, encoding="utf-8").read()
    needle = 'getattr(cfg, "attn_res_lr", 0.01)'
    assert needle in src, "the benign getattr the broken world mutates has moved"
    open(real_train, "w", encoding="utf-8").write(
        src.replace(needle, 'getattr(cfg, "attn_res_lr_MISSING", 0.01)', 1))
    return d


def check_no_conflict_markers(root):
    """No tracked source or doc holds a merge/stash conflict marker.

    Found by reading, not by a gate: `docs/lessons/gate_failure_shapes.md:870` carried a
    bare `>>>>>>> Stashed changes` in 9420c8b, committed, with every hook line green (de,
    2026-09-02). Nothing in CHECKS looked for it, ruff does not read Markdown, and a
    trailing marker at the end of a long doc is invisible to a reviewer scrolling to the
    section they came for.

    Why this class survives a clean-looking commit: a marker is not a syntax error in
    Markdown, JSON-with-comments, or a shell heredoc, so the file still parses and still
    renders. In a .py it WOULD be a syntax error, which is why py_compile catches those
    and only those -- the gap is exactly the file types this repo keeps its evidence in.

    `Stashed changes` specifically points at the shared stash stack AGENTS.md forbids: a
    pop of another session's entry conflicts on paths the popper never touched, so the
    marker lands somewhere they were not looking.

    Matched at line start only, and `=======` is deliberately NOT one of the needles: a
    Markdown setext heading underline is a row of `=`, and so is a table rule in some
    docs, which would make this check fire on well-formed prose."""
    needles = ("<<<<<<< ", ">>>>>>> ", "|||||||  ")
    hits = []
    for p, txt in walk_tracked(root, (".md", ".py", ".json", ".jsonl", ".sh", ".txt", ".yml", ".yaml")):
        rel = os.path.relpath(p, root)
        # This file names the markers in its own docstring and needle list, so it would
        # match itself -- the §61 shape, a criterion whose needle sits in its own data.
        if rel == os.path.join("scripts", "harness.py"):
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if line.startswith(needles):
                hits.append(f"{rel}:{i} {line[:34]}")
                break
    if hits:
        return FAIL, (f"{len(hits)} file(s) hold a conflict marker -- a resolution was committed "
                      f"half-done: {'; '.join(hits[:4])}")
    return PASS, "no tracked file holds a conflict marker"


def _broken_no_conflict_markers():
    """The REAL shapes doc with the REAL marker put back at the line it was found on.

    Mutating the actual file, not a hand-written stub: the check reads tracked files under
    a repo-shaped tree, and a stub would share the check's assumption about where docs
    live. This is the exact byte sequence 9420c8b committed."""
    import shutil

    d = _tmp_repo_shaped()
    rel = os.path.join("docs", "lessons", "gate_failure_shapes.md")
    dst = os.path.join(d, rel)
    # _tmp_repo_shaped symlinks docs/, so writing through it would edit the real file.
    real_docs = os.path.join(d, "docs")
    if os.path.islink(real_docs):
        os.unlink(real_docs)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(ROOT, rel), dst)
    with open(dst, "a", encoding="utf-8") as f:
        f.write(">>>>>>> Stashed changes\n")
    return d


def check_no_shared_stash(root):
    """The stash stack is empty. There is exactly ONE of it per repository.

    `.git/refs/stash` is not per-worktree, so every session shares one stack: e1 and b0
    each ran push -> merge main -> pop within the same window on 2026-09-02 and each
    popped the other's entry. Nothing was lost that time, and that is luck, not a
    property -- a pop applies someone else's diff to your tree and the conflict, if any,
    is reported against files you never touched. The reflex it comes from is the
    behind-main gate, and that gate does not need a clean tree: a dirty `git merge main`
    fast-forwards fine (measured on three trees), and only a local change on a CONFLICTING
    path makes merge refuse. So the replacement is: merge directly, and when it refuses,
    a path-limited `wip:` commit first.

    Entries get an expiry rather than a whitelist, because a whitelist has no end. The
    creation time comes from the stash reflog, and the message names the DATE the entry
    turns from WARN to FAIL -- a relative "30 days" reads as "not yet" on every one of
    those days, and `git stash list` does not show ages at all."""
    if not os.path.exists(os.path.join(root, ".git")):
        return SKIP, "no .git (pod or partial checkout)"
    r = subprocess.run(["git", "stash", "list", "--format=%gd %ct %gs"],
                       cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        return SKIP, f"git stash list failed: {r.stderr.strip()}"
    entries = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if not entries:
        return PASS, "stash stack empty"
    now, expired, pending = time.time(), [], []
    for ln in entries:
        parts = ln.split(" ", 2)
        ref, stamp, rest = parts[0], parts[1], (parts[2] if len(parts) > 2 else "")
        # %ct is a Unix timestamp, so there is nothing to parse and no local clock to
        # get wrong. An unreadable stamp counts as expired, never as fresh.
        created = int(stamp) if stamp.isdigit() else 0
        exp = created + _STASH_GRACE_DAYS * 86400
        item = f"{ref} {rest[:40]} (expires {time.strftime('%Y-%m-%d', time.gmtime(exp))})"
        (expired if now > exp else pending).append(item)
    if expired:
        return FAIL, f"{len(expired)} stash entr(ies) past their {_STASH_GRACE_DAYS}-day grace: {'; '.join(expired[:3])}"
    return WARN, (f"{len(pending)} stash entr(ies) on the SHARED stack -- pop them into a branch "
                  f"or drop them: {'; '.join(pending[:3])}")


def _broken_no_shared_stash():
    """A real repo with a real stashed change. Not a fabricated ref: `git stash` is the
    thing under test, so the broken world runs it."""
    import shutil
    import subprocess as sp

    d = _tmp_repo()
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    sp.run(["git", "init"], cwd=d, capture_output=True, env=env)
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    sp.run(["git", "add", "AGENTS.md"], cwd=d, capture_output=True, env=env)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "init"],
           cwd=d, capture_output=True, env=env)
    with open(os.path.join(d, "AGENTS.md"), "a") as f:
        f.write("\n# stashed\n")
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "stash", "push",
            "-m", "broken world"], cwd=d, capture_output=True, env=env)
    return d


def check_allocation_reads_the_grant(root):
    """The cards a training launch GETS are the cards the grant file GIVES.

    The defect this exists for (b0, 2026-09-03): launch_gate.gate_cards read
    runs/card_assignment.json to decide GO, and _allocation_cards read
    data/mix_scale_run_config.json to decide which cards the job actually got. The grant
    was narrowed to 0-3 because cards 4 and 7 hold the user's own work in other
    containers; the gate went GO on 0-3 and the launcher would still have handed the run
    cards 0-6. Correcting the authoritative file changed nothing, because the acting code
    never read it. Same shape as recipe_provenance's, one file over: a record can be
    right, current, and read by a gate, and still not reach the code that acts (§142).

    READS `root`, and every sub-case is derived from what that tree holds rather than from
    a fixture written here. The first version built its own temp dirs and so returned the
    same answer for every tree it was handed -- `check --selftest` caught it immediately
    ("reported PASS on its broken world"), which is the §71 shape: a check that cannot be
    made to fail by damaging the thing it checks is not checking it.
    """
    gp = os.path.join(root, "runs", "card_assignment.json")
    lp = os.path.join(root, "data", "mix_scale_run_config.json")
    if not os.path.isfile(gp) or not os.path.isfile(lp):
        return SKIP, "needs both runs/card_assignment.json and data/mix_scale_run_config.json"
    try:
        with open(gp, encoding="utf-8") as fh:
            grant = json.load(fh)
        with open(lp, encoding="utf-8") as fh:
            ladder_cards = _expand_cards(json.load(fh).get("cards", ""))
    except (OSError, ValueError) as e:
        return FAIL, f"a card source is unreadable: {e}"
    granted = _expand_cards(grant.get("block_cards", "")) if grant.get("launch_block_granted") else []

    # AN UNGRANTED BOX IS THE LEGITIMATE STATE OF AN IDLE BOX (de, on 6e's report 2026-09-04).
    # An explicit launch_block_granted:false must stop a LAUNCH, and it does -- cmd_launch
    # calls with raise_on_false left set. It must not stop a READ: SystemExit is a
    # BaseException, so run_checks:10136's `except Exception` never saw it and `harness check`
    # printed ZERO of its check lines on a false-grant tree, measured. That took every other
    # red in the tree dark with it, and 6e -- unable to commit an honest false -- kept the
    # field true with the real decision in the prose. The instrument made the file lie.
    # Reported here as SKIP rather than FAIL: nothing is wrong with a box nobody has claimed.
    if "launch_block_granted" in grant and not grant["launch_block_granted"]:
        return SKIP, (f"the grant says launch_block_granted is false, so there is no block to "
                      f"compare an allocation against -- an ungranted box is a legitimate "
                      f"state, and a LAUNCH is where a false grant refuses (granted_by "
                      f"{str(grant.get('granted_by', 'unknown'))[:40]})")

    import io

    err = io.StringIO()
    _real, sys.stderr = sys.stderr, err
    try:
        block = [int(c) for c in
                 _allocation_cards(True, root=root, raise_on_false=False).split(",") if c.strip()]
        lane_s = _allocation_cards(False, root=root, raise_on_false=False)
    finally:
        sys.stderr = _real
    msg = err.getvalue()

    if granted:
        # 1. The grant decides. This is the whole defect.
        if set(block) != set(granted):
            return FAIL, (f"the grant gives {_csv(granted)} and a training launch would get "
                          f"{_csv(block)} -- the launcher does not read the file that says "
                          f"who owns the cards")
        # 2. A disagreement between the two sources must be announced, never resolved
        #    silently: one file said 0-3, another 0-6, and nothing said they differed.
        if set(ladder_cards) != set(granted) and "DISAGREE" not in msg:
            return FAIL, (f"the two card sources disagree (grant {_csv(granted)} vs ladder "
                          f"{_csv(ladder_cards)}) and nothing said so -- that silence is what "
                          f"let a launch target cards outside the grant")
        # 3. ...and agreement must NOT warn, or the warning is noise and gets waved past.
        if set(ladder_cards) == set(granted) and "DISAGREE" in msg:
            return FAIL, ("two AGREEING card sources reported a disagreement; a warning that "
                          "fires on correct input gets ignored (§142)")
        # 4. lane_card: null means NO lane, not "complement the block" -- with block 0-3
        #    the complement is 4-7, the cards the narrowing existed to protect.
        if "lane_card" in grant and grant["lane_card"] is None and lane_s:
            return FAIL, (f"the grant states lane_card: null and the lane came back "
                          f"{lane_s!r} -- complementing the block hands a non-training job "
                          f"cards the grant does not give")
        if lane_s:
            lane_set = set(int(c) for c in lane_s.split(",") if c.strip())
            outside = lane_set - set(_expand_cards(grant.get("lane_card", "")))
            if grant.get("lane_card") is not None and outside:
                return FAIL, (f"the lane {lane_s!r} includes card(s) {_csv(sorted(outside))} "
                              f"the grant does not name as the lane")
            # 5. The lane and the block must be DISJOINT. A lane card inside the block
            #    hands a non-training job a card the training block is already using, and
            #    DDP does not fail cleanly on that: on 2026-09-02 two probes shared cards
            #    twice and OOM'd each other. Read from the grant's own two fields, so a
            #    grant that contradicts itself is caught where it is written.
            both = lane_set & set(block)
            if both:
                return FAIL, (f"the grant's lane card(s) {_csv(sorted(both))} are inside its "
                              f"own block {_csv(block)} -- a non-training job would land on a "
                              f"card the training block holds, which OOMs both")
        return PASS, (f"grant {_csv(granted)} decides the block; "
                      f"{'disagreement announced' if set(ladder_cards) != set(granted) else 'sources agree'}; "
                      f"lane {lane_s or 'none (grant says null)'}")
    # 6. An explicit launch_block_granted:false must RAISE FOR A LAUNCH and NOT for a read.
    #    "I say no" and "I have not spoken" are different answers, and on the pod they were
    #    worlds apart: the pod held a false grant from 2026-09-01 and the launcher happily fell
    #    back to cards 0-6, the user's card 4 included. BOTH halves are asserted, because the
    #    raising half alone is what took this whole instrument dark -- SystemExit is a
    #    BaseException, run_checks' `except Exception` cannot catch it, and a false grant printed
    #    zero check lines (measured 2026-09-04). Built as a throwaway world rather than by
    #    damaging `root`, because the property is about a value this tree does not hold.
    import shutil as _sh
    import tempfile as _tf

    _d = _tf.mkdtemp(prefix="grant_false_")
    try:
        os.makedirs(os.path.join(_d, "runs"), exist_ok=True)
        with open(os.path.join(_d, "runs", "card_assignment.json"), "w") as fh:
            json.dump({"launch_block_granted": False, "block_cards": None}, fh)
        try:
            _grant_cards(_d)
            return FAIL, ("an explicit launch_block_granted:false did not refuse -- a "
                          "controller saying NO is being treated as a value to fall back "
                          "from, which is how the pod allocated 0-6 under a false grant")
        except SystemExit:
            pass
        try:
            _cards, _why = _grant_cards(_d, raise_on_false=False)
        except SystemExit:
            return FAIL, ("raise_on_false=False still raised -- a READ of a false grant kills "
                          "the whole check run, because SystemExit is a BaseException that "
                          "run_checks' `except Exception` cannot catch")
        if _cards is not None or "false" not in _why:
            return FAIL, (f"a read of a false grant returned {_cards!r}/{_why[:50]!r} -- it must "
                          f"return no cards and say the grant is false")
    finally:
        _sh.rmtree(_d, ignore_errors=True)

    # No grant: the fallback is the ladder config, and it must SAY so -- a silent fallback
    # is indistinguishable from a grant that happens to agree.
    if set(block) != set(ladder_cards):
        return FAIL, (f"no block grant, so the ladder config's {_csv(ladder_cards)} should "
                      f"decide, but the allocation is {_csv(block)}")
    if "mix_scale_run_config" not in msg:
        return FAIL, ("the fallback to data/mix_scale_run_config.json is SILENT -- nothing "
                      "tells a reader which of the two files decided the cards")
    return PASS, f"no grant; fell back to the ladder's {_csv(ladder_cards)} and said so"


def _broken_allocation_reads_the_grant():
    """A grant whose stated lane is a card the block already holds.

    THE DEFECT THIS CHECK WAS WRITTEN FOR IS IN CODE, NOT DATA -- _allocation_cards
    reading the wrong file -- and a broken TREE cannot express that: any tree fed to the
    fixed function gets the right answer. So this world damages the artifact instead, in
    the one way that is still a card error: it grants block 0-3 and names card 2 as the
    lane, so a non-training job and the training block are handed the same card. That is
    the collision the lane exists to prevent (2026-09-02: two probes shared cards twice and
    OOM'd each other), and it is what case 4 reads.

    Stated plainly because the distinction matters for reading this check's green: the
    code-level defect is covered by running the OLD implementation against the real tree
    (done at the terminal, 2026-09-03: grant 0-3, old code returns 0-6, red), not by this
    world. A world that cannot fail the property is worse than no world, so this one fails
    a property the check actually holds.
    """
    d = _tmp_repo()
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    with open(os.path.join(d, "runs", "card_assignment.json"), "w") as f:
        json.dump({"launch_block_granted": True, "block_cards": "0-3", "lane_card": "2"}, f)
    with open(os.path.join(d, "data", "mix_scale_run_config.json"), "w") as f:
        json.dump({"cards": "0,1,2,3", "world": 4}, f)
    return d


GHOST_STARTLESS_CEILING = 180


def check_no_ghost_close(root):
    """A close must fold onto the row it closes, not mint a second identity beside it.

    6e's ruling, 2026-09-04. `exp.py done` resolves its row through pick_open_row and then falls
    back to `dict(base or {"started": now(), ...})`, so a close that fails to find its start writes
    a row whose `started` is the CLOSE's timestamp. The result is two keys where the run had one:
    the real key stays `running` forever and a terminal row sits beside it carrying the measurement.
    b0_headmix_armA is the incident -- 09:09 stayed `running` while a `fail vanished` row appeared
    under 11:10, and the number that mattered (val 2.117, scoring rc=1) was on the pod under the
    09:09 key where nothing local could see it.

    FAIL is the NARROW predicate and WARN is the broad one, because they are different questions and
    only one has a bounded answer today.

    NARROW: a key with a terminal event whose `started` is later than a still-open key of the SAME
    name. That is a close which minted an identity while the row it should have folded onto was
    open. Measured on the live ledger 2026-09-04: 0 today, and exactly 1 -- the armA pair -- if the
    two events f4d48444 pulled home are removed. So the check goes green the moment the repair
    lands, which is the property that makes it a gate rather than a standing red.

    POSITION IS NOT IN THE PREDICATE, and this is the part that took three measurements to get
    right. exp.fold is terminal-wins, so a start event appearing after a close does not reopen the
    key. Three readings of the same ledger: a position-based scan over raw events found 3 hits (two
    of them already repaired, because the repair is an APPEND and the original ghost row is still
    in the file); folding first found 0, since folding collapses each key to its terminal row and
    destroys the evidence; keying on "is any event under this key terminal" found 1, the real one.
    A check that folds first cannot see this defect at all, and a check that reads positions
    reports repaired history as broken.

    BROAD, as a WARN with a dated ceiling: 180 keys hold no open event ever -- rows appended
    straight to a terminal status, with no start on record. Shipping that as FAIL would turn the
    whole ledger's history red, so the ceiling is the literal GHOST_STARTLESS_CEILING (180, measured
    2026-09-04) and only a NEW start-less close raises the count past it. Same only-shrinks pattern
    as tasks_well_formed's drop_reason grandfather list: the number can go down without a commit and
    cannot go up without one.
    """
    p = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return SKIP, "runs/experiments.jsonl not present"
    evs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                evs.append(json.loads(line))
            except Exception:
                continue
    if not evs:
        return SKIP, "runs/experiments.jsonl is empty"

    by_key = {}
    for r in evs:
        by_key.setdefault((r.get("name"), r.get("started")), []).append(r)
    open_keys = {k for k, v in by_key.items() if all(_exp_open(r) for r in v)}

    ghosts = []
    for k in sorted(by_key, key=lambda kk: (str(kk[0]), str(kk[1]))):
        if k in open_keys:
            continue
        stole = sorted(ok for ok in open_keys
                       if ok[0] == k[0] and str(ok[1]) < str(k[1]))
        if stole:
            sts = ",".join(sorted({str(r.get("status")) for r in by_key[k]}))
            ghosts.append(f"{k[0]} closed under {k[1]} [{sts}] while {stole[0][1]} was still open")
    if ghosts:
        return FAIL, (
            f"{len(ghosts)} close(s) minted a new identity instead of folding onto the open row: "
            f"{'; '.join(ghosts[:4])} -- the run's real key stays `running` forever and its "
            f"measurement sits under a key no reader joins on (b0_headmix_armA, 2026-09-04). Close "
            f"with the start row's own `started`, and append a void row for the ghost key")

    startless = [k for k in by_key if k not in open_keys
                 and not any(_exp_open(r) for r in by_key[k])]
    if len(startless) > GHOST_STARTLESS_CEILING:
        return WARN, (
            f"{len(startless)} keys have a terminal row and NO start event, above the "
            f"{GHOST_STARTLESS_CEILING} recorded 2026-09-04. The new ones were appended straight "
            f"to a terminal status, so nothing records that the run began or when. Start rows "
            f"first, or raise the ceiling in a commit saying which ones are legitimate")
    return PASS, (
        f"{len(by_key)} keys, {len(open_keys)} open, 0 ghost closes; {len(startless)} start-less "
        f"(ceiling {GHOST_STARTLESS_CEILING})")


def _broken_no_ghost_close():
    """The REAL ledger with the two events f4d48444 pulled home removed -- the world as it was.

    Mutated, not hand-written, and the mutation is a DELETION of real rows rather than an edit, so
    there is no size-preserving-pyc question here (this world runs no python it wrote). Removing
    armA's `error` and `dropped` events restores exactly the state 6e reported: the pod held the
    measurement, 09:09 read `running`, and a `fail vanished` row stood under 11:10. Measured
    2026-09-04: 1 ghost in this world, 0 in the real ledger.

    The check that this world is load-bearing and not merely different: it must hold a terminal row
    under a LATER `started` than a still-open key of the same name. Asserted here, because a future
    edit to armA's rows could leave the deletion valid and the property absent.
    """
    d = _tmp_repo()
    src = os.path.join(ROOT, "runs", "experiments.jsonl")
    kept = []
    for line in open(src, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("name") == "b0_headmix_armA" and str(r.get("status")) in ("error", "dropped"):
            continue
        kept.append(r)
    arm = [r for r in kept if r.get("name") == "b0_headmix_armA"]
    opens = {str(r.get("started")) for r in arm if _exp_open(r)}
    closes = {str(r.get("started")) for r in arm if not _exp_open(r)}
    assert opens and closes and min(opens) < max(closes), (
        f"b0_headmix_armA no longer holds an open key earlier than a closed one "
        f"(open {sorted(opens)}, closed {sorted(closes)}); this world would report no ghost and "
        f"the check would pass on it untested")
    with open(os.path.join(d, "runs", "experiments.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return d


CHECKS = [
    (
        "no_ghost_close",
        "a close folds onto the row it closes; it never mints a second identity beside it",
        "b0_headmix_armA: exp.py done could not find the 09:09 start, so it wrote a row under "
        "11:10 -- the run read `running` forever in main while its real result (val 2.117, "
        "scoring rc=1, no metrics) sat on the pod under the key nothing joined on",
        check_no_ghost_close,
        _broken_no_ghost_close,
    ),
    (
        "mutation_asserted_took",
        "every broken world that mutates a file and runs it proves the mutation took effect",
        "world 8's `return 0.0` -> `return 1.0` kept rlvr_reward.py at 3565 bytes, so a mutation "
        "landing in the same wall-clock second as the world's own green run reused a stale "
        "__pycache__ (invalidation is mtime-in-SECONDS plus size), the defect never executed, and "
        "the world reported 'a staged defect was allowed' -- blocking b0 mid-merge for half an "
        "hour while the message covered two different causes",
        check_mutation_asserted_took,
        _broken_mutation_asserted_took,
    ),
    (
        "allocation_reads_the_grant",
        "a training launch's cards come from runs/card_assignment.json, not the ladder config",
        "the grant was narrowed to 0-3 because cards 4 and 7 hold the user's own work; the "
        "gate read the grant and went GO while the launcher read mix_scale_run_config.json "
        "and would have handed the run cards 0-6",
        check_allocation_reads_the_grant,
        _broken_allocation_reads_the_grant,
    ),
    (
        "env_importable",
        "every third-party module the repo imports is installed",
        "a container restart dropped the writable layer; SFT died on ModuleNotFoundError and read as a code bug",
        check_env_importable,
        _broken_env,
    ),
    (
        "mix_not_unfiltered",
        "the mix train.py defaults to does not name 'web'",
        "the v2 mix gave 88% weight to the unfiltered corpus and Cfg.mix pointed at it by default",
        check_mix_not_unfiltered,
        _broken_mix,
    ),
    (
        "mix_shards_present",
        "every domain in the default mix has shards on disk",
        "a domain with no shards is only caught after the other domains are tokenized",
        check_mix_shards,
        _broken_mix,
    ),
    (
        "launch_line_vs_oom_facts",
        "no stop-window launch line or running experiments row matches a recorded OOM config on (dim, layers, batch, accum, seq)",
        "p200m_4b_0902 launched b32a1 twice on 2026-09-02 after eff.microbatch_32_oom had recorded that exact OOM; the line had been checked against argparse, not against the facts",
        check_launch_line_vs_oom_facts,
        _broken_launch_line_oom,
    ),
    (
        "ckpt_facts_sources_present",
        "no fact source/config names a checkpoint on the deletion list unkept, or one absent from the pod listing",
        "eff.kda_mla_growth_ratio_l32's step1500 source was pruned with nothing red; the same day's list nearly took step2000/2500/3000 too",
        check_ckpt_facts_sources_present,
        _broken_ckpt_facts_sources,
    ),
    (
        "run_commits_resolve",
        "every experiments row's commit names an object this repository holds",
        "p500m_20b_0902's 00:03 row carried cec145b, which resolves to nothing here; it surfaced only because the pod's copy of that row disagreed in that one field, and the cause was exp.git_commit writing 8 chars via rev-parse --short on one path and 7 via a hardcoded slice on the other",
        check_run_commits_resolve,
        _broken_run_commits_resolve,
    ),
    (
        "pod_stamp_is_main",
        "the pod's sync stamp names a commit main contains",
        "launch condition 2' clause three had no code: run_ddp.sh printed the stamp's sha and never compared it, so a human read two hex strings off a 66-hour log; pod_push stamped `rev-parse HEAD`, which in a per-session worktree is that branch's tip (1b85dd0c while main was 69c8bd87)",
        check_pod_stamp_is_main,
        _broken_pod_stamp_is_main,
    ),
    (
        "pod_ledger_rows_home",
        "every row in the pod's runs/*.jsonl has its key present in the repository's copy",
        "five score_matrix rows behind the closed A/Bs existed only on the pod's emptyDir; pod_push only pushes and pod_drift only asserts listed files match, so a pod-only ledger row is invisible to every check",
        check_pod_ledger_rows_home,
        _broken_pod_ledger_rows_home,
    ),
    (
        "keep_claim_reasons_live",
        "no KEEP claim in the candidates listing cites a fact whose status is retracted",
        "step1192's claim was 'the ONLY evidence refuting ds.second_resume_rereads_one_segment'; that fact was retracted the same day by 52aec31 and the claim stood",
        check_keep_claim_reasons_live,
        _broken_keep_claim_reasons,
    ),
    (
        "no_oversized_blob",
        f"no file over {MAX_TRACKED_MB}MB is tracked by git",
        "gitignore does not cover already-tracked paths; a 40MB file committed once because of it",
        check_no_oversized_blob,
        _broken_blob,
    ),
    (
        "non_shard_jsonl_excluded",
        "train.py's shard glob skips holdout_slice_*.jsonl and any other non-shard family",
        "the holdout slice is one file per PHASE, so an exact-name list could never cover it; "
        "three domains failed to tokenize with KeyError: 'content' and the launch gate's epochs "
        "item was blocked until it was found",
        check_non_shard_jsonl_excluded,
        _broken_non_shard_jsonl_excluded,
    ),
    (
        "spawned_scripts_exist",
        "every script harness.py shells out to is at the path harness.py uses",
        "c3a47e8 moved pretokenize.py to datagen/ and three call sites kept pointing at scripts/; "
        "a subprocess path resolves only when it runs, so `harness run pretokenize` -- the step "
        "that warms the token caches the launch gate requires -- was broken and silent",
        check_spawned_scripts_exist,
        _broken_spawned_scripts_exist,
    ),
    (
        "tokenizer_roundtrip",
        "data/tokenizer.json decodes back to the exact input bytes",
        "the k5 vocabulary silently dropped NUL and tab",
        check_tokenizer_roundtrip,
        _broken_tokenizer,
    ),
    (
        "pinned_ids",
        "<eos> is loader.EOS_ID and [NUM] is Cfg.num_id",
        "four files hardcode these ids and a vocabulary rebuild moves them silently",
        check_pinned_ids,
        lambda: _broken_tokenizer(eos_id=5),
    ),
    (
        "vocab_id_on_load_path",
        "every trainer that loads an SFT pack compares the pack's vocab_id to the checkpoint's",
        "a pack from another vocabulary trains silently at ~4x the loss -- every id is wrong, in range, and the sizes match; 7aacbac fixed sft_math.py's guard, which had read a key the packer never writes, and sft.py loads a pack and compares nothing",
        check_vocab_id_on_load_path,
        _broken_vocab_id_load_path,
    ),
    (
        "entrypoint_help",
        "every argparse help string can be formatted, so --help works",
        "eval/code_zh.py --help died with 'TypeError: %o format' for 25 hours and nobody noticed, because nothing runs --help; a literal percent in a help string is a %-conversion to argparse",
        check_entrypoint_help,
        _broken_entrypoint_help,
    ),
    (
        "merge_complete",
        "a merge does not resolve a contested file by discarding one side",
        "resolving a file two sessions had edited with `git checkout --theirs` took one whole file and dropped four changes; the reverse checkout dropped another session's. Neither printed anything, and no later check can fail on code that is no longer there",
        check_merge_complete,
        _broken_merge_complete,
    ),
    (
        "merge_keeps_parent_paths",
        "a merge's tree holds every path either parent held, unless a commit deleted it",
        "runs/redaction_handread_v14.tsv left main with no deletion commit anywhere and was restored four times, each restore dropped again; `git reset HEAD <path>` on a file that arrived from a merge records a DELETION against the merged parent. A first-parent D-entry walk reports both real merges clean, because the path was already absent in parent 1",
        check_merge_keeps_parent_paths,
        _broken_merge_keeps_parent_paths,
    ),
    (
        "no_stale_running",
        "no experiments.jsonl row has been 'running' for over 24h",
        "a killed job wrote its checkpoint, never ran its eval, and left the row open",
        check_no_stale_running,
        _broken_stale_run,
    ),
    (
        "no_ghost_running",
        "a running row older than 2h has a live process (pod only)",
        "a finished-but-unrecorded run looked alive for up to 24h under no_stale_running alone",
        check_no_ghost_running,
        _broken_ghost_running,
    ),
    (
        "corpus_filters_fp",
        "every stamped corpus domain records the filters that built it, and they still match",
        "PROVENANCE recorded the build command but not the filter version; the same command before and after a filters/ edit yields different corpora",
        check_corpus_filters_fp,
        _broken_corpus_filters_fp,
    ),
    (
        "score_input_fresh",
        "a score records which corpus it scored, and that corpus is still the current one",
        "re-running clean changes the corpus but leaves stale scores with nothing raising",
        check_score_input_fresh,
        _broken_score_input_fresh,
    ),
    (
        "eval_registry_complete",
        "every eval/held-out data file is in datagen/holdout.py's REGISTRY, and every entry's path exists",
        "EVAL_FILES held four paths and the corpus builders took their exclusion population from it, so control_sft_text_heldout.jsonl was never excluded and 2,114 of 7,523 held-out items reached the pretraining corpus with the guard green, fingerprinted and loud -- the guard was correct on its own population and its population was wrong (e1, 2026-09-04)",
        check_eval_registry_complete,
        _broken_eval_registry_complete,
    ),
    (
        "sft_pack_holdout",
        "an SFT pack is built against the current holdout set, not a stale one",
        "a pack built before a holdout regeneration leaks held-out questions into training",
        check_sft_pack_holdout,
        _broken_sft_pack_holdout,
    ),
    (
        "sft_pack_uncontaminated",
        "no holdout question appears verbatim in the SFT pack",
        "a pack can pass the holdout fingerprint check yet still contain held-out questions (stale hash set, missing EVAL_FILES entry)",
        check_sft_pack_uncontaminated,
        _broken_sft_pack_uncontaminated,
    ),
    (
        "eval_sft_template_contamination",
        "no code eval problem shares a generator template with an SFT source",
        "verbatim matching misses the family level: SFT on the same synthetic generator teaches the template itself, which made code-500 v1 useless",
        check_eval_sft_template_contamination,
        _broken_eval_sft_template_contamination,
    ),
    (
        "restartability",
        "no NEW script accumulates in a loop and writes only at the end",
        "a two-hour scoring job wrote once at the end; killed at 50% it lost 100% of the work",
        check_restartability,
        _broken_restartability,
    ),
    (
        "gemm_dims_aligned",
        "Cfg's GEMM dimensions are multiples of 16 (8 for cuBLAS fast kernels, 16 for _fp8_ok)",
        "vocab 32773 made cuBLAS pick an SM75 align-1 kernel on Hopper; the LM head ran at 41% of bf16 peak, unnoticed",
        check_gemm_dims,
        _broken_gemm_dims,
    ),
    (
        "guard_on_path",
        "train.py main() actually calls the mix guard",
        "the guard lived in a wrapper while the documented entry point bypassed it",
        check_guard_on_path,
        _broken_guard,
    ),
    (
        "tasks_paired_and_prior",
        "every task opened since the rule names a second session and its prior art",
        "three sessions took the same work at once and a throughput number had no reference point in the literature; both are the same absence, stated before starting (user order 2026-09-01)",
        check_tasks_paired_and_prior,
        _broken_tasks_paired_and_prior,
    ),
    (
        "tasks_closed_by_commit",
        "every task closed since the rule names a commit that reaches main and touches its evidence",
        "the register closed on free text, so a task closed on a path that never existed read as delivered; a whole evening's assignments lived only in chat and none was recoverable",
        check_tasks_closed_by_commit,
        _broken_tasks_closed_by_commit,
    ),
    (
        "owner_queue_depth",
        "every roster member has at least two open, unblocked tasks",
        "six sessions sat idle under 16 open rows, nine of them frozen with the training path; the register recorded the freeze and nobody read it as idleness (user, 2026-09-02)",
        check_owner_queue_depth,
        _broken_owner_queue_depth,
    ),
    (
        "peer_stalled",
        "no roster member holds an open task while nothing has reached the repo from them for two hours",
        "the register said who owned work and nothing said whether anyone was doing it; six sessions sat idle under 16 open rows and the freeze that caused it was recorded but never read as idleness (user, 2026-09-02)",
        check_peer_stalled,
        _broken_peer_stalled,
    ),
    (
        "review_present",
        "every done task carries a review row from the peer it named",
        "the controller review caught four evidenced errors in one day while every other session's deliveries shipped with one reader",
        check_review_present,
        _broken_review_present,
    ),
    (
        "ledgers_one_line_per_row",
        "every union-merged ledger holds one JSON object per physical line",
        "3b's retro row was pretty-printed across lines 3-12: union merge concatenates lines, so a multi-line row interleaves with another branch's rows and identity becomes position",
        check_ledgers_one_line_per_row,
        _broken_ledgers_one_line_per_row,
    ),
    (
        "facts_well_formed",
        "every facts/*.json entry carries its measurement config, and AGENTS.md asserts no guarded phrase",
        "a value without its measurement config is the project's repeated failure class",
        check_facts_well_formed,
        _broken_facts,
    ),
    (
        "unreached_files_ruled",
        "every file reachability reports as unreached carries a FATE ruling",
        "25 unreached files had accumulated with no ruling, 23 of them touched within two days -- and 21 more were false candidates the scan could not see the hook's edge to",
        check_unreached_files_ruled,
        _broken_unreached_files_ruled,
    ),
    (
        "entrypoints_ran",
        "every script the entry-point table cites exists (FAIL); every tried one has an ok run (WARN)",
        "run_ablation.sh shipped as the AttnRes A/B entry while its rows read killed and OOM-fail",
        check_entrypoints_ran,
        _broken_entrypoint,
    ),
    (
        "entrypoints_table_present",
        "AGENTS.md contains at least one entry-point row citing a script",
        "cfg_default: two corpus invariants reported SKIP 'chosen on purpose' and check exited 0 -- "
        "an empty list silences the guard",
        check_entrypoints_table_present,
        _broken_entrypoints_table,
    ),
    (
        "docs_root_clean",
        "zero .md files directly under docs/ -- research, audits, standards live in subdirs",
        "docs/ was flat with audit_*/data_recipe*/exp_* mixed at the root, no rule and no check",
        check_docs_root_clean,
        _broken_docs_root,
    ),
    (
        "lessons_have_frontmatter",
        "every docs/lessons|audits/*.md (README excepted) carries question/status/source",
        "research docs carried no machine-checkable contract; a doc could answer no question and cite nothing",
        check_lessons_frontmatter,
        _broken_lessons_fm,
    ),
    (
        "fact_refs_resolve",
        "every facts/<file>.json#<id> citation resolves; citing a retracted fact WARNs",
        "the citation regex dropped .json from the path, so this check passed on zero real citations until its fix on 2026-08-30 (36 citations now resolve)",
        check_fact_refs,
        _broken_fact_ref,
    ),
    (
        "corpus_fp_matches",
        "every domain the default mix names carries a build-time fingerprint matching its live directory; a missing stamp is FAIL, not SKIP",
        "the voided 0.2b run trained on CCI3 shards under web_hq's name and no fingerprint said so -- an unstamped domain cannot be distinguished from a swapped-in one",
        check_corpus_fp,
        _broken_corpus_fp,
    ),
    (
        "pod_drift",
        "pod files match the committed manifest; in CI, the manifest matches HEAD",
        "the pod ran 142 files behind HEAD and its harness had never run the full check set -- training happened under rules the repo no longer had",
        check_pod_drift,
        _broken_pod_drift,
    ),
    (
        "doc_commands_exist",
        "every .sh/.py cited in an AGENTS.md command block exists",
        "a documented command that does not run is worse than none",
        check_doc_commands,
        _broken_doc_commands,
    ),
    (
        "readme_current",
        "README reflects the current objective, not a retired one",
        "README opened with the retired Chinese-LLM framing after the objective changed; a stale README misdirects every new reader",
        check_readme_current,
        _broken_readme_current,
    ),
    (
        "score_matrix_present",
        "every status=ok training run has a score-matrix record for its checkpoint",
        "a base checkpoint reads zero on every generative eval, and an unscored ok run is invisible -- the matrix is the only score that moves on a base",
        check_score_matrix,
        _broken_score_matrix,
    ),
    (
        "ladder_config_frozen",
        "every ladder checkpoint's cfg matches data/mix_scale_run_config.json",
        "a silent recipe drift (wrong warmup, wrong bucket) produces a completed point that poisons the curve; the OOM was loud, the wrong-but-valid case is not",
        check_ladder_config,
        _broken_ladder_config,
    ),
    (
        "frozen_keys_complete",
        "every train.py parser flag is in _FROZEN_KEYS or _UNFROZEN_ALLOWLIST",
        "eight architecture/recipe flags escaped the frozen set and nothing noticed; the list rots the moment someone adds a flag",
        check_frozen_keys_complete,
        _broken_frozen_keys_complete,
    ),
    (
        "ladder_cfg_consistent",
        "all six ladder checkpoints record the same cfg (except mix)",
        "a code edit to chunk_size/layers/optimizer params between points is invisible to the frozen list (no CLI flag) and to pod_drift (manifest regenerated); this is the only check that sees it",
        check_ladder_cfg_consistent,
        _broken_ladder_cfg_consistent,
    ),
    (
        "mix_supply",
        "per-domain demand does not exceed epoch-capped pool at any budget point",
        "a mix that wants more rows than its pool allows trains on repeated data with nothing raising",
        check_mix_supply,
        _broken_mix_supply,
    ),
    (
        "reported_path_is_written",
        "a runner that versions its output reports the path it actually wrote",
        "the 16B code cell's log said `preds saved: ...k8.jsonl`, that file did not exist, and the result read as a dead run for an hour -- attest() had the right path all along, the print and the --out JSON used the pre-versioning one",
        check_reported_path_is_written,
        _broken_reported_path,
    ),
    (
        "snapshot_logs_say_so_at_the_tail",
        "a committed excerpt of a live pod log says 'END OF SNAPSHOT' where tail looks",
        "b0 tailed runs/data_leg_206m_8b.log and reported the leg at 42% while the pod was at 65%; the header already said 'excerpt' and tail never shows line 1",
        check_snapshot_logs_say_so_at_the_tail,
        _broken_snapshot_logs_say_so_at_the_tail,
    ),
    (
        "cited_artifacts_attested",
        "a fact citing a gitignored eval artifact carries a sha256 its writer attested",
        "preds_*.jsonl is gitignored so fact_refs_resolve skips it; an unlogged rerun overwrote preds_l1_d3.jsonl and five facts pointed at another run's rows for hours",
        check_cited_artifacts_attested,
        _broken_cited_artifacts_attested,
    ),
    (
        "milestone_ckpt_pinned",
        "every milestone row's checkpoint still exists or has a pinned copy",
        "the 3.24B own-mix baseline was lost when step3500 rotated out of train.py's newest-3 window while the rescore waited in the lane queue; the weights are gone and the measurement cannot be repeated",
        check_milestone_ckpt_pinned,
        _broken_milestone_ckpt_pinned,
    ),
    (
        "cache_readers_set_vocab_id",
        "every module importing a token-cache reader also sets train.VOCAB_ID",
        "eval/domain_bpb.py imported val_seqs and never set the fingerprint, so it has never produced a value: the global stayed None, every cache stamp read as a mismatch against an empty right side, and the guard reported 'cache dirty' when the process simply had none -- while score_matrix.py:1186 and domain_loss.py:624 both carry the call with a comment saying why",
        check_cache_readers_set_vocab_id,
        _broken_cache_readers_set_vocab_id,
    ),
    (
        "selftests_are_gated",
        "every file carrying its own --selftest is in the hook's SELFTEST_FILES map",
        "a readout commit landed with its selftest RED under five green hook lines: the hook ran tree/blob/ruff/harness and none of them knew the edited file carried fifteen cases testing the guard that commit was changing -- it checked what it happened to check, not what the commit changed",
        check_selftests_are_gated,
        _broken_selftests_are_gated,
    ),
    (
        "probe_numbers_unique",
        "tNN numbers claimed by more than one probe are surfaced for a human to judge",
        "three collisions in one afternoon: t62/t63/t64 each named two probes from two sessions, so a fact citing a number resolves to whichever file the reader finds",
        check_probe_numbers_unique,
        _broken_probe_numbers_unique,
    ),
    (
        "no_duplicate_defs",
        "no module defines the same top-level name twice",
        "two sessions restored one dropped selftest from different commits and harness.py carried it twice; Python binds the second, so the first is dead code that drifts, and ruff F811 does not fire across intervening defs",
        check_no_duplicate_defs,
        _broken_no_duplicate_defs,
    ),
    (
        "agents_rules_covered",
        "every AGENTS.md rule maps to a check name or an explicit manual reason",
        "the register refusal in a worktree pushed a session into the shared tree tonight: a rule that is only prose is one people break for cause",
        check_agents_rules_covered,
        _broken_agents_rules_covered,
    ),
    (
        "shapes_table_covers_doc",
        "every incident in the incidents doc is referenced exactly once in AGENTS.md's rule table",
        "three sessions added shapes on 2026-09-02 and the numbering collided twice (two §62s, two §63s), each caught only by a merge conflict -- which catches a same-line collision but never a shape that reaches no rule, or a row whose count says 14 beside fifteen refs",
        check_shapes_table_covers_doc,
        _broken_shapes_table_covers_doc,
    ),
    (
        "timestamps_are_utc",
        "every timestamp written into the repo is UTC",
        "the Mac writes CST and the pod container writes UTC in the same format with no marker, so a pod row reads eight hours old the moment it lands and every ledger age comparison is wrong by up to eight hours (2026-09-01)",
        check_timestamps_are_utc,
        _broken_timestamps_are_utc,
    ),
    (
        "curl_ipv4",
        "every curl call in tracked code passes -4",
        "the pod's IPv6 egress is broken; without -4 the failure reads as 'host unreachable' and produced a whole reachability matrix of false negatives (2026-08-30)",
        check_curl_ipv4,
        _broken_curl_ipv4,
    ),
    (
        "running_sh_override_verified",
        "POD_PUSH_ALLOW_RUNNING_SH reaches the byte-offset check instead of permitting on the operator's word",
        "the override was one line returning unconditionally: an edit to a RUNNING script was pushed on an assertion nobody recomputed, and the same flag on an edit touching an earlier byte would corrupt the live shell's resume position with no warning (de-48, 2026-09-04)",
        check_running_sh_override_verified,
        _broken_running_sh_override_verified,
    ),
    (
        "no_foreground_pod_training",
        "no training process on the pod outside a setsid session",
        "a foreground pod job becomes an orphan holding a whole card at 100% when the tn tunnel dies; one silently contaminated a seven-card profile",
        check_no_foreground_pod_training,
        _broken_no_foreground_pod_training,
    ),
    (
        "root_durable",
        "AUPAI_ROOT is on a durable mount (/data00-/data03), not a Kubernetes emptyDir",
        "the 94 GB corpus, every checkpoint, and the repo lived in a 365 GB emptyDir for weeks; a pod deletion would erase all of it",
        check_root_durable,
        _broken_root_durable,
    ),
    (
        "env_fp_present",
        "every checkpoint carries an environment fingerprint",
        "a container restart changed the effective environment and three sessions chased wrong hypotheses for an hour because nothing recorded what the environment WAS",
        check_env_fp_present,
        _broken_env_fp_present,
    ),
    (
        "opt_state_present",
        "a checkpoint with a step number carries optimizer state",
        "resuming from a checkpoint with step but no opt zeroes Muon momentum and AdamW moments; the loss dips and recovers, looking like noise",
        check_opt_state_present,
        _broken_opt_state_present,
    ),
    (
        "tasks_well_formed",
        "a closed task carries an artifact; an open one carries an owner and a reason",
        "the controller's assignments lived only in chat. Compaction ate them, and a task closed on a session's word is the same self-report the board footer forbids",
        check_tasks_well_formed,
        _broken_tasks_well_formed,
    ),
    (
        "tasks_stale",
        "open tasks are not forgotten: unblocked ones are picked up, old ones are flagged",
        "a task blocked on a done task sat idle for days; the controller assigns work and the register is the only place that remembers it",
        check_tasks_stale,
        _broken_tasks_stale,
    ),
    (
        "card_held_without_claim",
        "every card holding real memory is named by a live claim in this tree",
        "e1's Stage A held card 5 for 15 minutes with the claim written in its laptop tree, so nothing on the pod said the card was held (2026-09-03)",
        check_card_held_without_claim,
        _broken_card_held_without_claim,
    ),
    (
        "lane_respected",
        "non-training processes do not occupy training cards",
        "a 10-min eval on one training card blocks a 55-min 7-card run; the lane rule was announced in docs but nothing enforced it",
        check_lane_respected,
        _broken_lane_respected,
    ),
    (
        "device_set_honoured",
        "every shell script indexes the caller's CUDA_VISIBLE_DEVICES instead of writing a physical index",
        "a lane-card launch (CUDA_VISIBLE_DEVICES=7) landed on physical GPU 0 and blocked t01; three scripts were fixed by hand and eval_all.sh kept the same bug for another nine hours",
        check_device_set_honoured,
        _broken_device_set_honoured,
    ),
    (
        "untracked_aged",
        f"untracked files older than {_AGE_HOURS}h in the shared tree get a fate",
        "a session's unfinished work sits unowned for days; nobody knows if it is safe to delete",
        check_untracked_aged,
        _broken_untracked_aged,
    ),
    (
        "frozen_paths",
        "main does not change what a running job is made of, while it is running",
        "a HOLD in a commit message binds the person who wrote it; git merge does not read English (da06097)",
        check_frozen_paths,
        _broken_frozen_paths,
    ),
    (
        "no_shared_stash",
        "the stash stack is empty; it is shared by every worktree in this repo",
        "e1 and b0 each stashed, merged main and popped in the same window -- and each popped the other's entry",
        check_no_shared_stash,
        _broken_no_shared_stash,
    ),
    (
        "friction_minutes_required",
        "near_miss and process_failure friction rows carry minutes_lost",
        "3/3 rows of these kinds lacked minutes_lost, making the second-largest unfixed friction cause invisible to ranking",
        check_friction_minutes_required,
        _broken_friction_minutes_required,
    ),
    (
        "no_conflict_markers",
        "no tracked doc or source holds a merge/stash conflict marker",
        "a bare '>>>>>>> Stashed changes' sat committed at gate_failure_shapes.md:870 under green hooks (9420c8b)",
        check_no_conflict_markers,
        _broken_no_conflict_markers,
    ),
    (
        "getattr_cfg_names_exist",
        "every getattr(cfg, \"name\", <non-None>) names a field Cfg can carry",
        "getattr(cfg, 'logit_softcap', 0.0) on a nonexistent field reported post-softcap logits as pre, and the numbers matched the expected conclusion (lessons-62, 2026-09-03)",
        check_getattr_cfg_names_exist,
        _broken_getattr_cfg_names,
    ),
    (
        "dirty_aged",
        f"tracked files dirty longer than {_AGE_HOURS}h are named so the owner commits or reverts",
        "uncommitted work blocks pushes and gets swept into other sessions' commits (d535674, 26 files)",
        check_dirty_aged,
        _broken_dirty_aged,
    ),
    (
        "mix_30b_contract",
        "the 30B mix declares its full composition (landed + _blocked sum to 1.0) and names no frozen ladder directory",
        "a pre-launch mix silently shrinks to whatever 3b has stamped, and a reused ladder name trains on the frozen corpus",
        check_mix_30b_contract,
        _broken_mix_30b,
    ),
]

# Where each check's evidence lives (fb, 2026-09-01). "pod" = gitignored data or
# machine state that exists only on the pod (corpus, caches, checkpoints, GPUs,
# process table); "repo" = files or history in git. The launch gate partitions
# FAILs on this: a repo check answers on main and its pod FAIL does not gate (the
# pod holds 168 one-off files main never saw, so repo scans can never be clean
# there); a pod check answers on the pod. Declared in code, not observed at
# runtime -- both machines must read the same answer, and "does it SKIP here?"
# is a property of the machine, not the check. Every check declares; the selftest
# fails on a check added without a declaration or a stale name.
EVIDENCE = {
    # pod: evidence exists only on the training box
    "env_importable": "pod", "mix_shards_present": "pod", "tokenizer_roundtrip": "pod",
    "pinned_ids": "pod", "no_ghost_running": "pod", "corpus_filters_fp": "pod",
    "score_input_fresh": "pod", "sft_pack_holdout": "pod", "sft_pack_uncontaminated": "pod",
    # pod: `data/` is gitignored, so a laptop sees almost none of the population -- counting it
    # on the pod found 10 unregistered files where a laptop glob plus a code grep reported 8.
    "eval_registry_complete": "pod",
    "eval_sft_template_contamination": "pod", "corpus_fp_matches": "pod", "pod_drift": "pod",
    "ladder_config_frozen": "pod", "ladder_cfg_consistent": "pod", "mix_supply": "pod",
    "milestone_ckpt_pinned": "pod", "env_fp_present": "pod", "opt_state_present": "pod",
    "card_held_without_claim": "pod", "lane_respected": "pod", "no_foreground_pod_training": "pod", "root_durable": "pod",
    # repo: the two card-source files are both tracked, so this answers the same anywhere
    "allocation_reads_the_grant": "repo",
    # repo: harness.py and the hook are both tracked, so the worlds' shape answers the same
    # anywhere. `auth=?` is not a third value -- an unregistered check prints it and is then
    # neither mirrored on the pod nor gated, which is a check outside the rule rather than
    # exempt from it.
    "mutation_asserted_took": "repo",
    "no_ghost_close": "repo",
    # repo: evidence is in git; answers on main, never gated by a pod-side FAIL
    "mix_not_unfiltered": "repo", "no_oversized_blob": "repo", "non_shard_jsonl_excluded": "repo",
    "spawned_scripts_exist": "repo", "entrypoint_help": "repo", "merge_complete": "repo",
    "merge_keeps_parent_paths": "repo",
    "no_stale_running": "repo", "restartability": "repo", "gemm_dims_aligned": "repo",
    "guard_on_path": "repo", "tasks_paired_and_prior": "repo", "tasks_closed_by_commit": "repo", "owner_queue_depth": "repo",
    "peer_stalled": "repo",
    "review_present": "repo", "ledgers_one_line_per_row": "repo", "facts_well_formed": "repo",
    "unreached_files_ruled": "repo", "entrypoints_ran": "repo", "entrypoints_table_present": "repo", "docs_root_clean": "repo",
    "lessons_have_frontmatter": "repo", "fact_refs_resolve": "repo", "doc_commands_exist": "repo",
    "readme_current": "repo", "score_matrix_present": "repo", "reported_path_is_written": "repo",
    "cited_artifacts_attested": "repo", "selftests_are_gated": "repo", "probe_numbers_unique": "repo",
    "snapshot_logs_say_so_at_the_tail": "pod",
    # repo: the readers and their callers are all tracked source; an AST parse needs no pod
    "cache_readers_set_vocab_id": "repo",
    # repo: sft.py and sft_math.py are tracked, so the AST answers the same anywhere. It does NOT
    # read a pack or a checkpoint -- the ids themselves are pod-side and outside this check.
    "vocab_id_on_load_path": "repo",
    "no_duplicate_defs": "repo", "agents_rules_covered": "repo", "timestamps_are_utc": "repo",
    "shapes_table_covers_doc": "repo",
    "curl_ipv4": "repo", "tasks_well_formed": "repo", "tasks_stale": "repo",
    "running_sh_override_verified": "repo",
    "device_set_honoured": "repo", "untracked_aged": "repo", "dirty_aged": "repo",
    "no_shared_stash": "repo", "friction_minutes_required": "repo", "frozen_paths": "repo", "no_conflict_markers": "repo",
    "getattr_cfg_names_exist": "repo",
    "launch_line_vs_oom_facts": "repo",
    "ckpt_facts_sources_present": "repo",
    # "both": the question joins two filesystems -- the pod holds the rows, the repository
    # holds what it is missing -- so neither side alone can answer it. It runs wherever a
    # local ledger and ~/bin/pod are both present, and SKIPs on the pod, where there is no
    # local side to compare against.
    "pod_ledger_rows_home": "both",
    # repo: resolving a sha against main needs the object database, which is the whole reason
    # this clause cannot live in run_ddp.sh.
    "pod_stamp_is_main": "repo",
    # repo: resolving a sha needs the object database, which the pod's tree does not have.
    "run_commits_resolve": "repo",
    "keep_claim_reasons_live": "repo",
    "mix_30b_contract": "repo", "frozen_keys_complete": "repo",
}


# -------------------------------------------------------------------------- stages
#
# A stage is done when its POSTCONDITION exists, never when its artifact does.

STAGES = [
    (
        "tokenizer",
        ["tokenizer_roundtrip", "pinned_ids"],
        "a tokenizer_<name>.json pinned per live checkpoint",
    ),
    ("corpus", ["corpus_filters_fp", "mix_not_unfiltered", "mix_shards_present"], "contamination scan recorded for every source"),
    ("pretrain", ["restartability", "gemm_dims_aligned", "guard_on_path", "no_stale_running", "score_matrix_present"], "checkpoint carries vocab_id; val loss recorded"),
    ("sft", ["pinned_ids"], "pack fingerprint == checkpoint vocab_id; loss-mask test passes"),
    ("eval", [], "math-hard recorded in runs/experiments.jsonl"),
]


# ------------------------------------------------------------------------- reports


def _check_deadline(signum, frame):
    # The per-check deadline below arms signal.alarm; without this handler SIGALRM's default
    # disposition killed the whole harness run (exit 142, no output, no check named) --
    # found 2026-09-01 when no_foreground_pod_training first exceeded 5 s. The handler turns
    # the alarm into the TimeoutError that run_checks already catches as a named SKIP.
    raise TimeoutError("check deadline")


def _read_timeout_strikes():
    """Consecutive-timeout count per check, from the last run. Unreadable = empty:
    the ledger is an optimisation over 'this timed out before', never a gate on its own."""
    try:
        with open(_TIMEOUT_STATE, encoding="utf-8") as f:
            obj = json.load(f)
        return {k: int(v) for k, v in obj.items() if isinstance(v, (int, float))}
    except (OSError, ValueError, AttributeError):
        return {}


def _write_timeout_strikes(strikes):
    """Persist the counts. A failure here must not fail the run -- it only costs the
    NEXT run its memory, which degrades to the single-timeout behaviour."""
    try:
        os.makedirs(os.path.dirname(_TIMEOUT_STATE), exist_ok=True)
        with open(_TIMEOUT_STATE, "w", encoding="utf-8") as f:
            json.dump(strikes, f, indent=1, sort_keys=True)
    except OSError:
        pass


def tree_provenance(root=ROOT):
    """One line naming the tree a check result describes: branch, HEAD, how far
    behind main, and whether it is dirty.

    A check's conclusion has two inputs -- the check's code and the tree it ran on --
    and only the first was ever reported. On 2026-09-01 no_foreground_pod_training was
    fixed four times and 3b ran the version before the first fix; separately two
    sessions each read the other's item as red in their own tree while both items were
    done. "This check is broken" and "this check is broken in my tree" are different
    claims, and the output could not tell them apart (fb, user order, 2026-09-01)."""
    def git(*a):
        r = subprocess.run(["git", "-C", root, *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    head = git("rev-parse", "--short", "HEAD")
    if head is None:
        return "tree: not a git repository"
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "?"
    behind = git("rev-list", "--count", "HEAD..main")
    dirty = git("status", "--porcelain")
    parts = [f"branch {branch}", f"HEAD {head}"]
    if behind is None:
        parts.append("behind main: unknown (no main ref)")
    elif behind == "0":
        parts.append("up to date with main")
    else:
        parts.append(f"BEHIND main by {behind} -- `git merge --no-edit main` before "
                     f"trusting any red below")
    if dirty:
        parts.append(f"{len(dirty.splitlines())} uncommitted file(s)")
    return "tree: " + ", ".join(parts)


def run_checks(root=ROOT, quiet=False, persist_timeouts=True):
    results = []
    if not quiet:
        print(f"  {tree_provenance(root)}")
    prev_strikes = _read_timeout_strikes()
    strikes = {}
    _prev_alarm_handler = signal.signal(signal.SIGALRM, _check_deadline)
    for name, asserts, incident, fn, _broken in CHECKS:
        t0 = time.time()
        try:
            signal.alarm(_CHECK_TIMEOUTS.get(name, _CHECK_TIMEOUT))
            state, evidence = fn(root)
        except TimeoutError:
            # A deadline hit is never a SKIP: see the TIMEOUT constant. The strike count
            # is what separates "this machine was busy" from "this check never runs".
            n = prev_strikes.get(name, 0) + 1
            strikes[name] = n
            limit = _CHECK_TIMEOUTS.get(name, _CHECK_TIMEOUT)
            if n >= _TIMEOUT_STRIKES:
                state = FAIL
                evidence = (f"timed out after {limit}s on {n} consecutive runs -- this check "
                            f"has not actually run since; raise its deadline or fix it")
            else:
                state = TIMEOUT
                evidence = (f"timed out after {limit}s (strike {n}/{_TIMEOUT_STRIKES}; "
                            f"the next consecutive timeout FAILs)")
        except Exception as e:  # a check that crashes is a failed check, never a pass
            state, evidence = FAIL, f"the check itself raised: {type(e).__name__}: {e}"
        finally:
            signal.alarm(0)
        # THE MIRROR OF THE auth=pod SKIP. A pod-authoritative check SKIPs on a laptop because the
        # thing it reads is not here; a repo-authoritative check must SKIP on the POD for the same
        # reason, and until now it FAILed instead. Measured on the pod 2026-09-04 (6e): 8 FAILs
        # and launch_gate reading NO-GO, five of them auth=repo checks failing only because
        # /work/aupai is not a git checkout -- tasks_closed_by_commit reported 86 of 86 commits
        # "not a commit in this repo", shapes_table_covers_doc reported §75+ missing from a stale
        # partial copy of the doc, ckpt_facts_sources_present found no listing because those files
        # are outside the push scope. None of that is a defect in the repo; it is a check reading
        # a tree that does not hold its subject.
        #
        # A blanket red is worse than no red: launch_gate weighs these, so five structural FAILs
        # made every real signal on the pod unreadable, which is the permanent-red rule in AGENTS.
        # SKIP names the reason so nobody reads it as "checked and fine".
        if (state in (FAIL, WARN, TIMEOUT) and pod_drift.is_pod(root)
                and EVIDENCE.get(name) == "repo"):
            state = SKIP
            evidence = (f"repo check, not authoritative here: {evidence[:110]}"
                        if evidence else "repo check, not authoritative here")
        dur = time.time() - t0
        results.append((name, state, evidence, asserts, incident))
        if not quiet:
            print(f"  [{state:^4}] {name:<22} {evidence}  ({dur:.1f}s) auth={EVIDENCE.get(name, '?')}")
            if state in (FAIL, WARN, TIMEOUT):
                print(f"         asserts: {asserts}")
            if state == FAIL:
                print(f"         prevents: {incident}")
    signal.signal(signal.SIGALRM, _prev_alarm_handler)
    # Only checks that timed out THIS run keep a count; anything that ran resets to zero
    # by absence. Written after the loop so a partial run cannot bank a strike.
    if persist_timeouts:
        _write_timeout_strikes(strikes)
    return results


def ledger():
    scores, orphans = recorded_scores()
    toks = local_tokenizers()
    print(f"  {'checkpoint':<26}{'on disk':>8}{'math-hard':>11}   source of the score")
    for n in checkpoint_names(scores):
        on_disk = os.path.exists(os.path.join(ROOT, f"{n}.pt"))
        s, src = scores.get(n, (None, None))
        sc = f"{s:.1f}%" if s is not None else "-"
        print(f"  {n:<26}{'yes' if on_disk else 'record':>8}{sc:>11}   {src or ''}")
    if orphans:
        # Dropping unmatched scores silently turns real measurements into "never measured".
        print(f"\n  {len(orphans)} recorded score(s) matched NO checkpoint name:")
        for name, s, cmd in orphans:
            print(f"    {name}: {s:.1f}%   cmd={cmd!r}")
    if toks:
        print("\n  local tokenizers:")
        for k, v in toks.items():
            print(f"    {k:<26}{v}")


def gaps():
    """An unmeasured checkpoint whose weights are gone is not a gap, it is history.

    Listing the two together made `gaps` nag about names nobody can ever score, which is
    how a to-do list stops being read. Only the ones whose weights are here are actionable,
    and `measure` closes exactly those."""
    scores, _orphans = recorded_scores()
    unmeasured = [n for n in checkpoint_names(scores) if n not in scores]
    here = [n for n in unmeasured if os.path.exists(os.path.join(ROOT, f"{n}.pt"))]
    gone = [n for n in unmeasured if n not in here]
    print(f"  {len(here)} checkpoint(s) with weights here and NO math-hard -- run `harness.py measure`:")
    print("    " + (", ".join(here) if here else "(none)"))
    if gone:
        print(
            f"\n  {len(gone)} unmeasured checkpoint(s) whose weights are GONE. Not a to-do: they"
            "\n  were deleted, and EXPERIMENTS.md is now the whole of what is known about them."
        )
        print("    " + ", ".join(gone))
    md = os.path.join(ROOT, "EXPERIMENTS.md")
    if os.path.exists(md):
        markers = (
            "not controlled",
            "never was",
            "still untested",
            "no benefit measurement",
            "cannot resolve",
            "unexplained",
            "has never been",
        )
        hits = [
            (i, ln.strip())
            for i, ln in enumerate(open(md, encoding="utf-8"), 1)
            if any(m in ln.lower() for m in markers)
        ]
        print(f"\n  {len(hits)} claim(s) EXPERIMENTS.md marks as uncontrolled or unmeasured:")
        for i, ln in hits[:12]:
            print(f"    EXPERIMENTS.md:{i}  {ln[:96]}")
    bp = os.path.join(ROOT, FACT_SOURCE_BASELINE)
    if os.path.exists(bp):
        debt = json.load(open(bp, encoding="utf-8"))
        print(f"\n  {len(debt)} baselined fact source(s) -- debt register, can only shrink:")
        for path, reason in sorted(debt.items()):
            print(f"    {path}: {reason[:90]}")
    cfb = os.path.join(ROOT, CORPUS_FILTERS_BASELINE)
    if os.path.exists(cfb):
        debt = json.load(open(cfb, encoding="utf-8"))
        print(f"\n  {len(debt)} baselined corpus domain(s) without filters_fp -- debt register, can only shrink:")
        for dom, reason in sorted(debt.items()):
            print(f"    {dom}: {reason[:90]}")


def measure(only=None, ngpu=None, tokenizer=None, dry=False, full=False):
    """CLOSE the gaps instead of reporting them.

    `gaps` naming a checkpoint as unmeasured, over and over, is not progress -- somebody
    still has to type the command, and on this project that somebody produced three
    write-ups and zero runs of the metric of record in one night. This runs the FULL
    matrix (scripts/eval_all.sh: math-hard, math-500, the MC suite, and the digit head for
    a FoNE checkpoint) on every checkpoint that is on disk and has no score, and writes the
    result back through scripts/exp.py so the ledger picks it up on the next read.

    Needs GPUs and the checkpoints, i.e. the pod. A checkpoint whose vocabulary does not
    match the tokenizer is recorded as a FAILURE, not skipped: eval_all.sh stops on that
    mismatch by design; an unrecorded stop makes a gap permanent."""
    import subprocess

    scores, _ = recorded_scores()
    todo = [
        n
        for n in checkpoint_names(scores)
        if n not in scores and os.path.exists(os.path.join(ROOT, f"{n}.pt"))
    ]
    if only:
        todo = [n for n in todo if only in n]
    # Newest first. NOT capped: gaps must stop listing the same names forever, and
    # math-hard alone is ~5 min per checkpoint.
    todo.sort(key=lambda n: os.path.getmtime(os.path.join(ROOT, f"{n}.pt")), reverse=True)
    # gaps counts every unscored name; this can only close the ones whose weights exist. Say
    # which ones it cannot, or an empty todo reads as "nothing left" over gaps' remainder.
    absent = [n for n in checkpoint_names(scores) if n not in scores and n not in todo]
    if absent:
        print(
            f"  {len(absent)} unscored checkpoint(s) NOT on disk, so not closable here: {', '.join(absent)}"
        )
    if not todo:
        print("  nothing to measure: every checkpoint whose weights are here carries a score")
        return 0
    print(f"  {len(todo)} checkpoint(s) on disk with no score: {', '.join(todo)}")
    if dry:
        return 0
    env = {**os.environ, "NGPU": str(ngpu)} if ngpu else None
    for n in todo:
        ck = f"{n}.pt"
        # math-hard alone by default: it is the metric of record, the only thing score_from
        # reads, and the only thing that closes a gaps entry. The MC suite is ~30% of the
        # matrix's runtime and eval_all.sh's own comment says it sits at the chance line.
        if full:
            cmd = ["bash", os.path.join(ROOT, "eval", "eval_all.sh"), ck] + ([tokenizer] if tokenizer else [])
        else:
            cmd = ["bash", os.path.join(ROOT, "eval", "eval_hard.sh"), ck, str(ngpu or 6)]
        print(f"\n  === {ck} ===", flush=True)
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        # eval_all.sh writes runs/evalall_<ckpt>.log unconditionally, so a crash after the
        # math-hard stage still leaves the lines that DID land; re-capturing stdout loses them.
        log = os.path.join(ROOT, "runs", f"evalall_{n}.log")
        out = open(log, encoding="utf-8").read() if full and os.path.exists(log) else p.stdout + p.stderr
        # eval_all.sh:91's own extractor. Matching only "TOTAL" dropped the digit head and the
        # arithmetic rate -- the two things the matrix exists to report BESIDE the score.
        keep = re.compile(r"TOTAL|whole-number exact|^Average|% wrong|STOP:")
        hits = [ln.strip() for ln in out.splitlines() if keep.search(ln)]
        result = " | ".join(hits) if hits else f"eval produced no summary line (rc={p.returncode})"
        # `ok` has to mean what ledger and gaps mean by "measured", which is score_from() being
        # able to read a math-hard number out of this string. Deciding it on "some TOTAL line
        # appeared" lets measure record a gap as closed while gaps still lists it -- exactly the
        # failure this command exists to prevent.
        status = "ok" if p.returncode == 0 and score_from(result) is not None else "fail"
        print(f"  {result}")

        # start THEN done, not done alone: exp.py's done appends a row with cmd="" when no
        # running row matches, and recorded_scores attributes an empty-cmd row as
        # f"ckpt_{name}" -- so a done-only row here would score `ckpt_ckpt_k8` and leave the
        # gap open. The start row is also where the ledger reads provenance from.
        def exp(*argv):
            subprocess.run([sys.executable, os.path.join(HERE, "exp.py"), *argv], cwd=ROOT, check=True)

        exp("start", "--name", n, "--cmd", " ".join(cmd), "--hypothesis", "harness measure")
        exp("done", "--name", n, "--status", status, "--result", result)
    print(f"\n  measured {len(todo)}; re-run `harness.py gaps` to see what is left")
    return 0


def stages(res=None):
    res = {n: s for n, s, _e, _a, _i in (res or run_checks(quiet=True))}
    scores, _ = recorded_scores()
    print(f"  {'stage':<12}{'gates':>26}   postcondition")
    for name, gates, post in STAGES:
        bad = [g for g in gates if res.get(g) == FAIL]
        detail = f"BLOCKED: {','.join(bad)}" if bad else f"{len(gates)} gate(s) pass"
        print(f"  {name:<12}{detail:>26}   {post}")
    print(f"\n  eval postcondition: {len(scores)} checkpoint(s) carry a math-hard score.")


# --------------------------------------------------------------------------- board


def _val_nll(name):
    """Last val NLL from runs/<name>.log, or None. The log line is
    'ep 1/1 train 3.281 val 3.322 615s'."""
    log = os.path.join(ROOT, "runs", f"{name}.log")
    if not os.path.exists(log):
        return None
    val = None
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"val (\d+\.\d+)", line)
        if m:
            val = float(m.group(1))
    return val


def _30b_readiness(root=ROOT):
    """30B launch readiness, computed from artifacts only.

    Gates: t22's task deps (t01/t20/t21/t30) all done; mix_30b _blocked empty
    and landed domains stamped; pod drift clean; CI green on origin/main;
    block idle or fully training; no build_corpus writers active.
    Returns (ready, gates) where gates is a list of (name, state, detail)."""
    gates = []

    # 1. Task gates: t22's dependencies must all be done
    tasks = {r.get("id"): r for r in _read_tasks(os.path.join(root, "runs", "tasks.jsonl"))}
    for tid in ("t01", "t20", "t21", "t30"):
        t = tasks.get(tid)
        if not t:
            gates.append((f"task {tid}", FAIL, "not in register"))
        elif t.get("state") != "done":
            gates.append((f"task {tid}", FAIL, f"state={t.get('state')}"))
        else:
            gates.append((f"task {tid}", PASS, "done"))

    # 2. mix_30b: _blocked empty, landed domains stamped
    mix_path = os.path.join(root, "data", "mix_30b.json")
    if not os.path.exists(mix_path):
        gates.append(("mix_30b", FAIL, "data/mix_30b.json missing"))
    else:
        try:
            mix = json.load(open(mix_path, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            gates.append(("mix_30b", FAIL, f"parse error: {e}"))
        else:
            blocked = mix.get("_blocked", {})
            landed = mix.get("domains", {})
            if blocked:
                gates.append(("mix_30b blocked", FAIL, f"{len(blocked)} blocked: {', '.join(sorted(blocked))}"))
            else:
                gates.append(("mix_30b blocked", PASS, "empty"))
            corpus = os.path.join(root, "data", "corpus")
            unstamped = [n for n in landed if not os.path.exists(os.path.join(corpus, n, "build_corpus_stats.json"))]
            if unstamped:
                gates.append(("mix_30b stamps", FAIL, f"unstamped: {', '.join(unstamped)}"))
            elif landed:
                gates.append(("mix_30b stamps", PASS, f"{len(landed)} landed, all stamped"))
            else:
                gates.append(("mix_30b stamps", SKIP, "no landed domains"))

    # 3. Pod drift
    drift_py = os.path.join(root, "scripts", "pod_drift.py")
    if os.path.exists(drift_py):
        r = subprocess.run([sys.executable, drift_py, "--check"], capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            gates.append(("pod drift", PASS, "clean"))
        else:
            first = (r.stdout or r.stderr).strip().split("\n")[0][:100]
            gates.append(("pod drift", FAIL, first))
    else:
        gates.append(("pod drift", SKIP, "pod_drift.py not found"))

    # 4. CI conclusion on origin/main
    try:
        r = subprocess.run(
            ["gh", "api", "repos/{owner}/{repo}/commits/HEAD/check-runs",
             "--jq", ".check_runs[0].conclusion"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            conclusion = r.stdout.strip()
            state = PASS if conclusion == "success" else FAIL
            gates.append(("CI origin/main", state, conclusion))
        else:
            gates.append(("CI origin/main", SKIP, "gh api failed or no runs"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        gates.append(("CI origin/main", SKIP, "gh not available"))

    # 5. GPU occupancy (pod only)
    if _gpu_present():
        config_path = os.path.join(root, "data", "mix_scale_run_config.json")
        if os.path.exists(config_path):
            config = json.load(open(config_path, encoding="utf-8"))
            block = [c.strip() for c in config.get("cards", "").split(",") if c.strip()]
            busy, err = _busy_training_cards(block)
            if err:
                gates.append(("GPU block", SKIP, err))
            elif not busy:
                gates.append(("GPU block", PASS, "idle"))
            elif len(busy) == len(block) and _has_training_process():
                gates.append(("GPU block", PASS, f"all {len(block)} busy (training)"))
            else:
                gates.append(("GPU block", FAIL, f"{len(busy)}/{len(block)} busy, no training process"))
        else:
            gates.append(("GPU block", SKIP, "no run config"))
    else:
        gates.append(("GPU block", SKIP, "no GPU on this machine"))

    # 6. build_corpus writers: python processes only (not the launch chain), grouped by --domain.
    # Gate: ≤1 per domain (a running clean is the normal state before t22).
    try:
        r = subprocess.run(["pgrep", "-af", "build_corpus"], capture_output=True, text=True, timeout=10)
        writers = [l for l in r.stdout.strip().split("\n")
                   if l and "pgrep" not in l and ("python" in l.split(None, 1)[-1][:20] if len(l.split(None, 1)) > 1 else False)]
        # Group by --domain argument
        domains = {}
        for w in writers:
            m = re.search(r"--domain\s+(\S+)", w)
            dom = m.group(1) if m else "unknown"
            domains[dom] = domains.get(dom, 0) + 1
        over = {d: n for d, n in domains.items() if n > 1}
        if over:
            gates.append(("corpus writers", FAIL, f"multiple writers per domain: {over}"))
        elif writers:
            gates.append(("corpus writers", PASS, f"{len(writers)} writer(s), domains: {sorted(domains)}"))
        else:
            gates.append(("corpus writers", PASS, "none active"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        gates.append(("corpus writers", SKIP, "pgrep not available"))

    ready = all(s != FAIL for _n, s, _d in gates)
    return ready, gates


def _board_event(kind, msg):
    """Append an event to runs/events.jsonl. The harness knows when things
    happen; this is how it stops staying silent."""
    path = os.path.join(ROOT, "runs", "events.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "kind": kind, "msg": msg}) + "\n")


def _board_data():
    """All board state, gathered from the same artifacts the checks read."""
    res = run_checks(ROOT, quiet=True)
    checks = [{"name": n, "state": s, "evidence": e} for n, s, e, _a, _i in res]
    n_skip = sum(1 for c in checks if c["state"] == SKIP)
    n_fail = sum(1 for c in checks if c["state"] == FAIL)
    # score matrix: ckpt -> metrics
    sm = {}
    sm_path = os.path.join(ROOT, "runs", "score_matrix.jsonl")
    if os.path.exists(sm_path):
        for line in open(sm_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                sm[r["ckpt"]] = r.get("metrics", {})
    # ladder points
    ladder = []
    for name, mix in LADDER:
        ckpt = f"ckpt_{name}.pt"
        m = sm.get(ckpt, {})
        dl = m.get("domain_loss", {})
        ladder.append({
            "name": name,
            "mix": os.path.basename(mix),
            "val_nll": _val_nll(name),
            "domain_loss": dl.get("unweighted_mean"),
            "minimal_pairs": m.get("minimal_pairs", {}).get("overall"),
            "lambada_zh": m.get("lambada_zh", {}).get("two_way_acc"),
            "math_v2_like": m.get("math_v2_like", {}).get("acc") or m.get("math_v2_like", {}).get("pass1"),
            "ceval": m.get("mc_ceval", {}).get("Average"),
            "scored": ckpt in sm,
        })
    # recent experiments (last 8)
    exps = []
    exp_path = os.path.join(ROOT, "runs", "experiments.jsonl")
    if os.path.exists(exp_path):
        rows = [json.loads(l) for l in open(exp_path, encoding="utf-8") if l.strip()]
        for r in rows[-8:]:
            exps.append({"name": r.get("name"), "status": r.get("status"),
                         "started": r.get("started"), "cmd": r.get("cmd", "")[:80]})
    # events (last 10)
    events = []
    ev_path = os.path.join(ROOT, "runs", "events.jsonl")
    if os.path.exists(ev_path):
        rows = [json.loads(l) for l in open(ev_path, encoding="utf-8") if l.strip()]
        events = rows[-10:]
    # staleness: newest artifact mtime
    newest = 0.0
    for p in [sm_path, exp_path, os.path.join(ROOT, "train.py")]:
        if os.path.exists(p):
            newest = max(newest, os.path.getmtime(p))
    ready_30b, gates_30b = _30b_readiness()
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "staleness": {"newest_artifact": time.strftime("%Y-%m-%d %H:%M", time.gmtime(newest)) if newest else None,
                      "skip_count": n_skip, "fail_count": n_fail},
        "checks": checks,
        "ladder": ladder,
        "experiments": exps,
        "events": events,
        "tasks": _read_tasks(),
        "readiness_30b": {"ready": ready_30b, "gates": [{"name": n, "state": s, "detail": d} for n, s, d in gates_30b]},
    }


def _render_board_html(d):
    """Self-contained HTML: inline CSS, system fonts, no external deps."""
    def pct(v):
        return f"{v*100:.1f}%" if isinstance(v, (int, float)) and v <= 1 else (f"{v}" if v is not None else "—")

    def num(v, fmt="{:.3f}"):
        return fmt.format(v) if isinstance(v, (int, float)) else "—"

    colors = {PASS: "#2d7d32", FAIL: "#c62828", SKIP: "#888", WARN: "#f9a825"}
    rows = ""
    for c in d["checks"]:
        color = colors.get(c["state"], "#888")
        rows += f'<tr><td>{c["name"]}</td><td style="color:{color};font-weight:600">{c["state"]}</td><td>{c["evidence"][:100]}</td></tr>\n'

    lrows = ""
    for p in d["ladder"]:
        status = "✓ scored" if p["scored"] else "…"
        lrows += (f'<tr><td>{p["name"]}</td><td>{p["mix"]}</td><td>{num(p["val_nll"])}</td>'
                  f'<td>{num(p["domain_loss"])}</td><td>{pct(p["minimal_pairs"])}</td>'
                  f'<td>{pct(p["lambada_zh"])}</td><td>{pct(p["math_v2_like"])}</td>'
                  f'<td>{num(p["ceval"], "{:.1f}")}</td><td>{status}</td></tr>\n')

    erows = ""
    for e in d["events"]:
        erows += f'<tr><td>{e["ts"]}</td><td>{e["kind"]}</td><td>{e["msg"]}</td></tr>\n'

    xprows = ""
    for x in d["experiments"]:
        xprows += f'<tr><td>{x["name"]}</td><td>{x["status"]}</td><td>{x["started"]}</td></tr>\n'

    trows = ""
    for t in sorted(d["tasks"], key=lambda r: (r.get("state") != "open", r.get("id", ""))):
        colour = "#c62828" if t.get("state") == "open" else "#2d7d32"
        # `reading` is the pre-registration: how to read the result, written before it
        # exists. It is shown here because a reading rule that lives only in a session's
        # context is gone before the number it governs is read.
        note = t.get("reading") or t.get("evidence") or ""
        blocked = f' <i>blocked on {t["blocked_on"]}</i>' if t.get("blocked_on") else ""
        trows += (f'<tr><td>{t.get("id")}</td><td>{t.get("owner")}</td>'
                  f'<td style="color:{colour};font-weight:600">{t.get("state")}</td>'
                  f'<td>{t.get("task", "")}{blocked}<br><span class="meta">{note[:220]}</span></td></tr>\n')

    st = d["staleness"]
    stale_warn = ""
    if st["fail_count"]:
        stale_warn = f'<p style="color:#c62828;font-weight:700">{st["fail_count"]} CHECK(S) RED</p>'
    if st["skip_count"]:
        stale_warn += f'<p style="color:#888">{st["skip_count"]} check(s) SKIPped — guard not running, not guard passed</p>'

    # 30B launch readiness
    r30 = d.get("readiness_30b", {})
    r30_rows = ""
    for g in r30.get("gates", []):
        color = colors.get(g["state"], "#888")
        r30_rows += f'<tr><td>{g["name"]}</td><td style="color:{color};font-weight:600">{g["state"]}</td><td>{g["detail"][:120]}</td></tr>\n'
    r30_banner = ""
    if r30.get("ready"):
        r30_banner = '<p style="color:#2d7d32;font-weight:700;font-size:1.1em">30B LAUNCH READY</p>'
    elif r30:
        r30_banner = '<p style="color:#c62828;font-weight:700;font-size:1.1em">30B NOT READY</p>'

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>aupai board</title><style>
body{{font:14px/1.5 system-ui,sans-serif;margin:2em;background:#fafafa;color:#222}}
h1{{font-size:1.3em}} h2{{font-size:1.1em;margin-top:1.5em;border-bottom:1px solid #ddd;padding-bottom:3px}}
table{{border-collapse:collapse;width:100%;margin:.5em 0}}
th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid #eee;font-size:13px}}
th{{color:#666;font-weight:600}} .meta{{color:#888;font-size:12px}}
</style></head><body>
<h1>aupai monitoring board</h1>
<p class="meta">rendered {d["timestamp"]} · newest artifact {st["newest_artifact"] or "—"}</p>
{stale_warn}
{r30_banner}
<h2>30B launch readiness</h2><p class="meta">computed from artifacts only; READY when every gate is green and _blocked is empty</p>
<table><tr><th>gate</th><th>state</th><th>detail</th></tr>{r30_rows}</table>
<h2>tasks</h2><p class="meta">assignments and their reading rules, from runs/tasks.jsonl. A closed
task carries an artifact, never a session's word for it.</p>
<table><tr><th>id</th><th>owner</th><th>state</th><th>task / reading rule</th></tr>{trows}</table>
<h2>checks</h2><table><tr><th>check</th><th>state</th><th>evidence</th></tr>{rows}</table>
<h2>ladder points</h2><table><tr><th>point</th><th>mix</th><th>val NLL</th><th>domain loss</th>
<th>min pairs</th><th>lambada</th><th>math v2</th><th>ceval</th><th>status</th></tr>{lrows}</table>
<h2>recent experiments</h2><table><tr><th>name</th><th>status</th><th>started</th></tr>{xprows}</table>
<h2>events</h2><table><tr><th>time</th><th>kind</th><th>message</th></tr>{erows}</table>
</body></html>"""


def cmd_board(as_json=False, html_path=None):
    """harness board [--json | --html <path>]. Renders harness state as JSON or HTML.
    Default: writes runs/board.html. Every number is read at render time — nothing typed."""
    d = _board_data()
    if as_json:
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return 0
    path = html_path or os.path.join(ROOT, "runs", "board.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_render_board_html(d))
    print(f"board written to {path} ({len(d['checks'])} checks, {len(d['ladder'])} ladder points)")
    return 0


def _refresh_board():
    """Refresh runs/board.html after a state-changing step. Never blocks the step."""
    try:
        cmd_board(html_path=os.path.join(ROOT, "runs", "board.html"))
    except Exception as e:
        print(f"board refresh failed (non-blocking): {e}")


# ------------------------------------------------------------------------ selftest


def _selftest_milestone_reachable():
    """A milestone target past the run's final step refuses; the run-end checkpoint
    (no .step suffix) is visible to the watcher.

    Both halves of one near-miss: the 15B milestone was armed at step 16500 against a
    16281-step run, so it could never fire -- and the artifact it would have wanted,
    ckpt_<run>.pt written by train.py:2168, carries no .step suffix and the watcher's
    glob cannot see it. That checkpoint is also the stage-2 resume source, so missing
    it costs the first real per-role verdict AND the resume (fb, 2026-09-01)."""
    import inspect

    src = inspect.getsource(cmd_milestone)
    i = src.index("past the run's final step")
    assert "return 2" in src[i:i + 400], "an unreachable target must refuse, not wait forever"
    j = src.index('f"ckpt_{a.run}.pt"')
    assert "saved.setdefault(a.final_step" in src[j:j + 300], (
        "the run-end checkpoint must be registered at the final step")
    # the arithmetic that made it unreachable
    total, save_every = 16281, 500
    assert 16500 > total, "the armed target really was past the end"
    assert (total // save_every) * save_every == 16000, "last save_every multiple"
    print(f"  milestone: a target past step {total} refuses; ckpt_<run>.pt registers at the end")


def _selftest_cold_cache_refuses():
    """A training launch with no token caches refuses instead of taking a short gate.

    The fallback was backwards: _derive_gate_timeout returns None when no cache is on
    disk, and cmd_launch then used 120s -- so the emptier the cache, the shorter the
    deadline, while the work grows from a load into a single-process retokenize of
    hours (b0, 2026-08-31). Asserts on the source, because reproducing it needs a
    training launch and a card."""
    import inspect
    import tempfile

    # the derivation really does return None-with-that-reason for an empty cache dir
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    mix = os.path.join(d, "data", "m.json")
    json.dump({"domains": {"a": {}, "b": {}}, "total_tokens": 1e9}, open(mix, "w"))
    secs, note = _derive_gate_timeout(["--mix", mix], cache_dir=d)
    assert secs is None and "no token caches" in note, (secs, note)

    src = inspect.getsource(cmd_launch)
    i = src.index('"no token caches" in gate_note')
    branch = src[i:i + 900]
    assert "REFUSING" in branch, "a cold-cache training launch must refuse"
    assert "return 2" in branch, "the refusal must exit non-zero"
    # and the refusal must come BEFORE the fallback assignment it replaces
    assert i < src.index("args.gate_timeout = derived or"), (
        "the refusal must precede the 120s fallback, or the fallback still wins")
    print("  gate: a training launch with cold caches refuses rather than taking 120s")


def _selftest_provenance_states_the_tree():
    """Every `check` run must say which tree it describes, and say BEHIND when behind.

    Built on real git repositories, because the whole claim is about what git reports:
    a hand-written world would share this function's own assumptions about rev-list.
    The three cases are the three a reader acts on differently -- up to date, behind,
    and no main at all (a temp repo, a single-branch clone), where the honest answer is
    that the question has no answer rather than zero."""
    import shutil
    import subprocess as sp
    import tempfile

    d = tempfile.mkdtemp()
    try:
        def git(*a, cwd=d):
            return sp.run(["git", "-C", cwd, *a], capture_output=True, text=True)

        git("init", "-q", ".")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")
        open(os.path.join(d, "f"), "w").write("a")
        git("add", "f")
        git("commit", "-qm", "base")
        # a repo with no `main` ref at all: the count cannot be taken
        head_branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if head_branch != "main":
            line = tree_provenance(d)
            assert "unknown (no main ref)" in line, (
                f"a repo without main must say the question has no answer: {line}")
            git("branch", "-m", "main")

        line = tree_provenance(d)
        assert "up to date with main" in line, f"on main, at main: {line}"
        assert "BEHIND" not in line, line

        git("checkout", "-qb", "side")
        git("checkout", "-q", "main")
        open(os.path.join(d, "f"), "w").write("b")
        git("add", "f")
        git("commit", "-qm", "ahead")
        git("checkout", "-q", "side")
        line = tree_provenance(d)
        assert "BEHIND main by 1" in line, f"one commit behind must say so: {line}"
        assert "branch side" in line, f"the branch must be named: {line}"
        assert "git merge" in line, f"it must say what to do about it: {line}"

        open(os.path.join(d, "g"), "w").write("c")
        git("add", "g")
        assert "1 uncommitted file(s)" in tree_provenance(d), tree_provenance(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # and it must actually reach the output of every check run
    import inspect

    src = inspect.getsource(run_checks)
    assert "tree_provenance" in src, (
        "run_checks no longer prints the tree: a red is then indistinguishable from a "
        "red in a stale tree, which is the confusion this exists to end")
    print("  provenance: check names its tree; behind-main is stated, no-main is not 0")


def _selftest_refusal_writes_no_row():
    """A refused launch leaves the ledger untouched.

    The start row was written at step 1 and the lane check ran at 2a, so launching a
    second job under a LIVE run's name closed that run's row as fail: l1_rerun_0831
    read running/running/fail while pid 550586 was alive and writing (e1,
    2026-08-31). Asserts the ordering directly -- the refusal path must reach `return`
    before any exp.py call, so a ledger seeded with a running row is byte-identical
    after it."""
    import inspect

    src = inspect.getsource(cmd_launch)
    lane_at = src.index("_lane_occupant(lane_card)")
    row_at = src.index('"start", "--name", args.name')
    assert lane_at < row_at, (
        "the lane check must run BEFORE the start row is written, or a refused launch "
        "closes a live run's row")
    # and the refusal itself must write nothing
    refusal = src[lane_at:src.index("return 1", lane_at)]
    assert "exp.py" not in refusal, f"the refusal path writes a ledger row:\n{refusal}"
    assert "No ledger row" in refusal, "the refusal must say it wrote nothing"
    print("  launch: a lane refusal returns before the start row, writing no ledger row")


def _selftest_pool_not_raw_supply():
    """A mix whose demand fits the raw cache but exceeds the pool must FAIL.

    The gap the old check missed: the builder carves the val holdout first and caps
    at pool x epochs, so a mix sized against raw supply is accepted and then silently
    under-draws (stage-1 cot: passed at 1.272B, drew 1.210B, 44). This asserts the
    arithmetic directly -- the check itself needs a token cache, which a dev box has
    not got, so a world-based test would SKIP and prove nothing."""
    cache_rows, val_frac, val_rows_max, epochs = 100_000, 0.01, 2_000, 3
    n_val = min(max(1, int(cache_rows * val_frac)), val_rows_max)
    pool_rows = cache_rows - n_val
    assert n_val == 1000 and pool_rows == 99_000, (n_val, pool_rows)
    raw_cap, pool_cap = cache_rows * epochs, pool_rows * epochs
    # Fits 300,000 raw, and exceeds 297,000 pool by more than the check's own 0.5%
    # rounding tolerance (298,485). The previous 297,500 cleared pool but NOT the
    # tolerance, so the check would have accepted it -- the dead assert 44 found was
    # hiding a case that did not test what it claimed.
    want = 299_000
    assert want <= raw_cap, "sanity: the case must fit RAW supply"
    assert want > pool_cap, "sanity: the case must exceed POOL supply"
    # The check's own tolerance, not a restatement of the line above: 44 caught the
    # previous version as `want > pool_cap * 1.005 or want > pool_cap`, whose second
    # clause is the preceding assert, so it could not fail. The live question is
    # whether the case clears the 0.5% rounding tolerance the check actually applies.
    assert want > pool_cap * 1.005, (
        f"the case must exceed pool by more than the check's 0.5% tolerance: "
        f"{want} vs {pool_cap * 1.005:.0f}")
    shortfall = 1 - pool_cap / want
    print(f"  mix_supply: pool model rejects a raw-supply-sized draw ({shortfall:.2%} short)")




def _selftest_killpg_reaps_children():
    """A kill must reap a child running a DIFFERENT script.

    Local processes, not the pod: the defect is in how the target set is chosen, and
    that logic is the same either way. pgrep -f on the parent's cmdline cannot match
    a child with another name -- score_matrix shells out to math_zh, code_zh,
    run_eval and domain_loss, so one parent kill leaked a child holding 12.7GB on
    GPU7 while the parent exited cleanly (e1, 2026-08-31)."""
    import shutil
    import signal as sig
    import subprocess as sp
    import tempfile

    d = tempfile.mkdtemp(prefix="killpg_")
    child = os.path.join(d, "differently_named_child.py")
    parent = os.path.join(d, "parent_runner.py")
    open(child, "w").write("import time\nwhile True: time.sleep(0.2)\n")
    open(parent, "w").write(
        f"import subprocess, time\n"
        f"subprocess.Popen(['{sys.executable}', {child!r}])\n"
        f"while True: time.sleep(0.2)\n"
    )
    proc = sp.Popen([sys.executable, parent], start_new_session=True)
    try:
        time.sleep(1.5)
        pgid = os.getpgid(proc.pid)
        # the old approach: match the PARENT's cmdline. It cannot see the child.
        by_pattern = sp.run(["pgrep", "-f", "parent_runner.py"], capture_output=True, text=True)
        assert str(proc.pid) in by_pattern.stdout.split(), "sanity: parent must match its own pattern"
        child_match = sp.run(["pgrep", "-f", "parent_runner.py"], capture_output=True, text=True)
        kids = sp.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True).stdout.split()
        missed = [k for k in kids if k not in child_match.stdout.split()]
        assert missed, "the differently-named child must be INVISIBLE to a cmdline match"
        # the group is what sees everything
        os.killpg(pgid, sig.SIGTERM)
        try:
            proc.wait(timeout=5)
        except sp.TimeoutExpired:
            pass
        deadline = time.time() + 5
        while True:
            pids = sp.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True).stdout.split()
            left = []
            for k in pids:
                st = sp.run(["ps", "-o", "stat=", "-p", k], capture_output=True, text=True).stdout.strip()
                if st and not st.startswith("Z"):
                    left.append(k)
            if not left or time.time() > deadline:
                break
            time.sleep(0.2)
        assert not left, f"killpg must reap the whole group, {left} survived"
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), sig.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        shutil.rmtree(d, ignore_errors=True)
    print("  kill: a differently-named child is invisible to a cmdline match, reaped by the group")


def _selftest_milestone_selection():
    """The watcher never labels a below-target checkpoint as the milestone.

    Known answer from the live incident: target 3500, saves {2000,2500,3000},
    save_every 500. min(|s-target|) picks 3000 -- exactly save_every away, so the
    `> save_every` guard does not fire -- and the row then claims 3.24B for a
    checkpoint that saw 2.753B (e1, 2026-08-31)."""
    def pick(saved, target, save_every, alive):
        at_or_past = [x for x in saved if x >= target]
        if at_or_past:
            return min(at_or_past)
        if max(saved) >= target - save_every and alive:
            return None  # wait for the exact save
        return max(saved)

    assert pick([2000, 2500, 3000], 3500, 500, True) is None, "must wait, not take step3000"
    assert pick([2000, 2500, 3000, 3500], 3500, 500, True) == 3500, "exact save wins"
    assert pick([3500, 4000], 3500, 500, True) == 3500, "never overshoot past an exact hit"
    assert pick([3600], 3500, 500, True) == 3600, "a save just past the target is a real reading"
    # the run died before reaching the target: the last save is the best available read
    assert pick([2000, 2500, 3000], 3500, 500, False) == 3000, "a dead run must not wait forever"
    # token accounting: the shortfall the label would have hidden
    short = 1 - (3000 * TOKENS_PER_STEP) / 3.24e9
    assert 0.14 < short < 0.16, f"step3000 vs the 3.24B label is ~15%, got {short:.3f}"
    print(f"  milestone: waits for the exact save; step3000 would have been {short:.1%} short")


def _selftest_monitor_suppression():
    """The monitor writes no row once a run has a terminal one.

    Runs the real embedded monitor source against a temp ledger, because the bug
    was in that source and a reimplementation would test the wrong code: t56_profile
    closed ok 13:34, the monitor appended fail 'log silent' 13:47, and the log stops
    growing exactly when a run succeeds."""
    import shutil
    import subprocess as sp
    import tempfile

    d = tempfile.mkdtemp(prefix="monsup_")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    exp_log = os.path.join(d, "runs", "experiments.jsonl")
    with open(exp_log, "w", encoding="utf-8") as f:
        f.write(json.dumps({"name": "r1", "started": "2026-08-31 13:25", "status": "running"}) + "\n")

    # Extract settled() from the real monitor source rather than retyping it.
    src = _arm_monitor.__doc__ and None  # keep the reference explicit for readers
    code = inspect.getsource(_arm_monitor)
    body = code[code.index("def settled():"):code.index("while True:")]
    ns = {"os": os, "json": json, "exp_log": exp_log, "name": "r1"}
    exec(compile(body, "<monitor>", "exec"), ns)  # noqa: S102 -- the real source, by design
    settled = ns["settled"]
    assert settled() is False, "a running row is not settled"

    with open(exp_log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"name": "r1", "started": "2026-08-31 13:25", "status": "ok",
                            "result": "profile complete"}) + "\n")
    assert settled() is True, "an ok row must settle the run and silence the monitor"

    with open(exp_log, "w", encoding="utf-8") as f:
        f.write(json.dumps({"name": "r1", "started": "x", "status": "running"}) + "\n")
        f.write(json.dumps({"name": "r1", "started": "x", "status": "fail",
                            "result": "killed by harness kill"}) + "\n")
    assert settled() is True, "a kill record must silence the monitor too"
    shutil.rmtree(d, ignore_errors=True)
    del src, sp
    print("  monitor: no row after a run reaches ok or fail")


def _selftest_gate_timeout():
    """Known answer: the gate is cache bytes / _CACHE_READ_GIBPS x2, floored at 600 s.

    Tonight's real numbers are the case that matters -- 149 GiB must exceed the
    6m26s the run actually took, and the old 120 s default must not."""
    import shutil
    import tempfile

    d = tempfile.mkdtemp(prefix="gate_")
    os.makedirs(os.path.join(d, "cache"), exist_ok=True)
    mix = os.path.join(d, "m.json")
    json.dump({"domains": {"a": {}, "b": {}}}, open(mix, "w"))
    cache = os.path.join(d, "cache")

    def with_gib(total_gib):
        for f in glob.glob(os.path.join(cache, "*.pt")):
            os.remove(f)
        half = int(total_gib * 2**30 / 2)
        for name in ("tokens_a.pt", "tokens_b.pt"):
            with open(os.path.join(cache, name), "wb") as f:
                f.truncate(half)  # sparse: no real bytes written
        return _derive_gate_timeout(["--mix", mix], cache_dir=cache)

    secs, note = with_gib(149)
    expect = int(149 / _CACHE_READ_GIBPS * 2)
    assert secs == expect, f"149 GiB -> {secs}s, expected {expect}"
    # The case the constant exists for: tonight's real startup was 386 s. A rate taken
    # from single-stream warm reads (1.5 GiB/s) derives 198 s and kills a healthy run.
    assert secs > 386, f"the gate must exceed the 6m26s the real run took, got {secs}s"
    assert "149 GiB" in note, note

    secs, _ = with_gib(1)  # a small mix must still get the floor, not 1s
    assert secs == _GATE_FLOOR_S, f"floor not applied: {secs}"

    secs, note = _derive_gate_timeout(["--mix", os.path.join(d, "nope.json")], cache_dir=cache)
    assert secs is None and "unreadable" in note, f"a missing mix must not produce a gate: {secs} {note}"
    shutil.rmtree(d, ignore_errors=True)
    print(f"  gate: 149 GiB -> {int(149 / _CACHE_READ_GIBPS * 2)}s (> the 6m26s real startup), small mix -> {_GATE_FLOOR_S}s floor")


def _selftest_merge_fix_not_deadlocked():
    """The check must refuse a bad merge AND accept the commit that fixes it.

    Without the second half it deadlocks: the amend re-reads HEAD, HEAD is still the
    bad merge, and --no-verify becomes the only exit -- which teaches people to
    bypass the check at the moment it is working (de + fb, 2026-08-31)."""
    import shutil

    d = _broken_merge_complete()
    try:
        state, _ = check_merge_complete(d)
        assert state == FAIL, f"a bad merge must be refused, got {state}"
        rel = os.path.join("scripts", "loader.py")
        with open(os.path.join(d, rel), "w") as f:
            f.write("def f():\n    OURS_MARKER = 'kept by us'\n"
                    "    THEIRS_MARKER = 'kept by them'\n    return 1\n")
        subprocess.run(["git", "-C", d, "add", rel], capture_output=True)
        state, evidence = check_merge_complete(d)
        assert state == PASS, f"the fix for a bad merge must be accepted, got {state}: {evidence}"
        with open(os.path.join(d, rel), "w") as f:
            f.write("def f():\n    THEIRS_MARKER = 'kept by them'\n    return 1\n")
        subprocess.run(["git", "-C", d, "add", rel], capture_output=True)
        state, _ = check_merge_complete(d)
        assert state == FAIL, "restaging the offending content must not pass as a fix"
        # And the hole that made the hatch escapable: the criterion was "the staged blob
        # DIFFERS from the parent taken whole", so one comment above the unfixed
        # resolution changed the blob, restored nothing, and read as a fix. Measured
        # 2026-09-01: FAIL became "1 contested file(s) re-resolved". A legal edit that
        # buys a GO is the same shape as the day's other holes.
        with open(os.path.join(d, rel), "w") as f:
            f.write("# a comment that restores nothing\n"
                    "def f():\n    THEIRS_MARKER = 'kept by them'\n    return 1\n")
        subprocess.run(["git", "-C", d, "add", rel], capture_output=True)
        state, _ = check_merge_complete(d)
        assert state == FAIL, ("a staged blob that merely DIFFERS from the offending "
                               "parent is not a fix; the lost content is still lost")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  merge fix: bad merge refused, real fix accepted, restaged offender and "
          "cosmetic-only edit still refused")


def _selftest_merge_cherry_pick_not_a_drop():
    """A cherry-picked commit is not a lost commit.

    The real case, 2026-09-01, on this check's own author: tilerl cherry-picked de's
    5927ed6 into main as b8cae37 to unblock a launch. Byte-identical blobs, de's
    branch then added a comment on top, and the merge correctly took the newer file --
    while check_merge_complete reported "2 commit(s) from the other side lost".
    Nothing was lost. The commit count cannot see it: a cherry-pick has a different
    sha, so rev-list still counts it as absent.

    This matters because the cherry-pick is the NORMAL way an urgent fix reaches main
    here, so the false positive would have fired on the common path and a permanent
    red is the same as no signal. The fix is exact blob equality -- did the side we
    took ever HOLD the other side's blob -- which cannot excuse a real drop, and the
    third assertion below is what pins that.
    """
    import shutil

    d = _tmp_repo()
    sh = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)  # noqa: E731
    try:
        sh("init", "-q")
        sh("config", "user.email", "t@t")
        sh("config", "user.name", "t")
        rel = os.path.join("scripts", "loader.py")
        src = os.path.join(d, rel)
        os.makedirs(os.path.dirname(src), exist_ok=True)
        open(src, "w").write("def f():\n    return 1\n")
        sh("add", "-A")
        sh("commit", "-qm", "base")
        main = sh("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        sh("checkout", "-qb", "feature")

        # feature makes the fix
        fix = "def f():\n    FIX = 'urgent'\n    return 1\n"
        open(src, "w").write(fix)
        sh("add", "-A")
        sh("commit", "-qm", "feature: the urgent fix")

        # main CHERRY-PICKS it: same content, different sha
        sh("checkout", "-q", main)
        open(src, "w").write(fix)
        sh("add", "-A")
        sh("commit", "-qm", "main: cherry-pick of the urgent fix")

        # feature builds on top, then main merges feature
        sh("checkout", "-q", "feature")
        open(src, "w").write(fix.replace("return 1", "# note\n    return 1"))
        sh("add", "-A")
        sh("commit", "-qm", "feature: a comment on top")
        sh("checkout", "-q", main)
        # The merge CONFLICTS -- both sides changed the line -- so it must be resolved
        # and committed, or there is no merge commit and check_merge_complete returns
        # "HEAD is not a merge": a vacuous PASS that looks exactly like the real one.
        # My first version of this fixture stopped at `git merge` and asserted PASS,
        # and it passed with the fix neutered. Third instance today of a check that
        # agrees with the thing it is checking (de, 2026-09-01).
        sh("merge", "--no-commit", "feature")
        open(src, "w").write(fix.replace("return 1", "# note\n    return 1"))
        sh("add", rel)
        sh("commit", "-qm", "merge feature (took the newer file)")
        parents = sh("rev-list", "--parents", "-n", "1", "HEAD").stdout.split()
        assert len(parents) >= 3, (
            f"the fixture must produce a MERGE commit, got {len(parents) - 1} parent(s) "
            f"-- otherwise check_merge_complete returns 'not a merge' and the assertion "
            f"below passes without ever running the code under test")
        took = merge_took_one_side(d)
        assert not took, f"the cherry-picked blob must not count as a drop: {took}"

        state, evidence = check_merge_complete(d)
        assert state == PASS, (
            f"a cherry-pick is not a drop -- main HELD that exact blob before the "
            f"merge, and the merge took the newer file: {state} {evidence}")

        # and the real drop must STILL fail: a blob the taken side never held.
        d2 = _broken_merge_complete()
        try:
            state, _ = check_merge_complete(d2)
            assert state == FAIL, (
                "the cherry-pick exemption must not excuse a genuine one-side "
                f"resolution, got {state}")
        finally:
            shutil.rmtree(d2, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  merge cherry-pick: picked commit not counted as lost, genuine drop still refused")


def _selftest_merge_reverted_content():
    """Real merges as the cases, plus the deliberate deletion that must NOT fire.

    21da619 is the instance that exposed the gap: base and ours had
    _selftest_gpu_descendants, the merged side had never seen it, the merge took that
    side, and main lost a test while keeping the function it tests.
    merge_took_one_side saw nothing, because it only examines files BOTH parents
    changed.

    41294c1 is the near-miss I first reported alongside it and got wrong: no side had
    the content, so nothing was lost. Kept as a case precisely because I misread it.

    The constructed case is the one that decides whether the check is usable: someone
    retiring a function on purpose must not be flagged, or every intentional deletion
    becomes a red and the check gets bypassed.

    c8a4578 is the third case and the reason the return value grew a fourth field
    (de-22, 2026-09-02): _built_set was dropped in d5aac3d's conflict resolution, ours
    already lacked it when this merge ran, and calling that a defect put the same red on
    117 merges. It must land in the ALREADY-DROPPED class, naming d5aac3d, while 21da619
    stays a FAIL -- the two shapes are checked here together because a fix for either one
    alone is what the flag experiments produced."""
    import shutil
    import tempfile

    real = "/Users/bytedance/code/aupai"
    if os.path.exists(os.path.join(real, ".git")):
        hit = merge_reverted_content(real, "21da619")
        assert any(n == "_selftest_gpu_descendants" and at is None for _, n, _, at in hit), \
            f"21da619 must be caught with already_dropped_at None (a FAIL), got {hit}"
        assert not merge_reverted_content(real, "41294c1"), "41294c1 lost nothing; must be clean"
        # The inherited class, and the whole point of the fourth field: the same scan must
        # report _built_set with a sha, not None, or the caller FAILs on inheritance again.
        inh = merge_reverted_content(real, "c8a4578")
        # startswith, not ==, and the producer now returns %H (de-35). The previous form
        # pinned `at == "d5aac3d"` against git's auto-scaling abbreviation: it printed 7
        # chars when de-22 wrote this and 8 once the object count crossed a threshold, so
        # the selftest went red with nothing about the merge or the check having changed.
        # A permanent red is the same as no signal, and it sat red on main for a day.
        # The full sha is the stable identity; a prefix test states the fact being
        # asserted -- which commit -- without depending on how git chose to print it.
        assert any(n == "_built_set" and at and at.startswith("d5aac3d") for _, n, _, at in inh), \
            f"c8a4578 must report _built_set as already dropped by d5aac3d, got {inh}"
        # And the identity is the FULL sha, asserted directly: startswith alone is
        # satisfied by any longer-but-wrong value and by the abbreviation this fix
        # removes, so without this the same time bomb could be reintroduced at the
        # producer and this selftest would stay green.
        ats = [at for _, n, _, at in inh if n == "_built_set"]
        assert all(len(a) == 40 for a in ats), (
            f"already_dropped_at must be a full 40-char sha, got {ats} -- an abbreviated "
            f"sha changes length with the repository's object count, which is what made "
            f"this assertion a permanent red (de-35)")

    d = tempfile.mkdtemp(prefix="delib_")
    try:
        def sh(*a):
            return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)
        sh("init", "-q"); sh("config", "user.email", "t@t"); sh("config", "user.name", "t")
        os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
        f = os.path.join(d, "scripts", "loader.py")
        open(f, "w").write("def keep_me():\n    return 1\n\n\ndef retire_me():\n    return 2\n")
        sh("add", "-A"); sh("commit", "-qm", "base"); sh("branch", "other")
        open(f, "w").write("def keep_me():\n    return 1\n")          # deliberate retire
        sh("add", "-A"); sh("commit", "-qm", "retire retire_me on purpose")
        sh("checkout", "-q", "other")
        open(f, "w").write("def keep_me():\n    return 1\n\n\ndef retire_me():\n"
                           "    return 2\n\n\ndef added():\n    return 3\n")
        sh("add", "-A"); sh("commit", "-qm", "unrelated add")
        back = "master" if sh("rev-parse", "--verify", "-q", "master").returncode == 0 else "main"
        sh("checkout", "-q", back)
        sh("merge", "--no-commit", "other")
        sh("checkout", "--ours", "scripts/loader.py"); sh("add", "scripts/loader.py")
        sh("commit", "-qm", "merge")
        assert not merge_reverted_content(d), \
            "a deliberate deletion must not be flagged, or every intended removal is a red"
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # The verdicts, not just the scan. check_merge_complete reads HEAD, so each case is a
    # clone checked out AT that merge -- the real artifact, per the broken-world rule. The
    # scan asserts above would still pass if the caller collapsed both classes to FAIL,
    # which is exactly the bug being fixed, so the states are asserted here too.
    if os.path.exists(os.path.join(real, ".git")):
        w = tempfile.mkdtemp(prefix="mergecls_")
        try:
            for merge, want, needle in ((("21da619"), FAIL, "_selftest_gpu_descendants"),
                                        (("c8a4578"), WARN, "d5aac3d")):
                c = os.path.join(w, merge)
                assert subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", real, c],
                                      capture_output=True).returncode == 0
                assert subprocess.run(["git", "-C", c, "checkout", "-q", merge],
                                      capture_output=True).returncode == 0
                st, why = check_merge_complete(c)
                assert st == want, f"{merge} must be {want}, got {st}: {why[:160]}"
                assert needle in why, f"{merge}'s text must name {needle}: {why[:160]}"
        finally:
            shutil.rmtree(w, ignore_errors=True)

    print("  merge revert: 21da619 FAIL, c8a4578 WARN naming d5aac3d, 41294c1 clean, "
          "deliberate deletion not flagged")


def _selftest_scoped_index_is_read():
    """A path-scoped commit's staged diff is visible to _funcs_in_diff.

    THE NEGATIVE CONTROL, named: the code path whose removal makes this fail is
    _staged_index_env()'s use in _funcs_in_diff -- pass env=None there and case 2 below
    selects zero functions. (tilerl's rule, 2026-09-04: a negative selftest must name the
    path it guards, or it is an assertion that cannot fail.)

    Real commits in a throwaway repo, because the whole defect is a variable git exports only
    to a hook it invokes -- no fixture can produce GIT_INDEX_FILE=.git/next-index-<pid>.lock,
    and a test that stages with `git add` first is exactly the test that missed this: under a
    plain staged commit both index reads agree.
    """
    import shutil
    import tempfile

    global _ORIG_GIT_INDEX_FILE, ROOT
    d = tempfile.mkdtemp(prefix="scoped_index_")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

    def g(*a):
        return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True, env=env)

    g("init", "-q")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    src = os.path.join(d, "m.py")
    with open(src, "w") as f:
        f.write("def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n")
    with open(os.path.join(d, "other.txt"), "w") as f:
        f.write("x\n")
    g("add", "m.py", "other.txt")
    g("commit", "-q", "-m", "base")
    if g("rev-parse", "HEAD").returncode:
        shutil.rmtree(d, ignore_errors=True)
        print("  scoped index: SKIP (cannot create a git repo here)")
        return

    # Edit ONE function; the selector must name it and not the other.
    with open(src, "w") as f:
        f.write("def alpha():\n    return 1\n\n\ndef beta():\n    return 99\n")
    saved_root, saved_idx = ROOT, _ORIG_GIT_INDEX_FILE
    try:
        ROOT = d
        # 1. plain staged commit: .git/index, both forms agree. This is the case that
        #    passed while the scoped one was blind.
        g("add", "m.py")
        _ORIG_GIT_INDEX_FILE = ""
        assert _funcs_in_diff(["m.py"]) == {"beta"}, "staged diff must select the edited function"
        g("commit", "-q", "-m", "staged")

        # 2. path-scoped commit: git builds a temp index and names it ONLY in
        #    GIT_INDEX_FILE. Captured from a real hook run rather than constructed.
        with open(src, "w") as f:
            f.write("def alpha():\n    return 7\n\n\ndef beta():\n    return 99\n")
        with open(os.path.join(d, "other.txt"), "w") as f:
            f.write("y\n")
        hooks = os.path.join(d, ".git", "hooks")
        os.makedirs(hooks, exist_ok=True)
        stamp = os.path.join(d, "idxvar")
        hp = os.path.join(hooks, "pre-commit")
        with open(hp, "w") as f:
            f.write(f'#!/bin/sh\nprintf %s "$GIT_INDEX_FILE" > {stamp}\n')
        os.chmod(hp, 0o755)
        r = subprocess.run(["git", "-C", d, "commit", "-m", "scoped", "--", "m.py"],
                           capture_output=True, text=True, env=env)
        assert not r.returncode, f"the scoped commit failed: {r.stderr[:200]}"
        captured = open(stamp, encoding="utf-8").read().strip()
        assert captured, "git exported no GIT_INDEX_FILE to the hook -- the premise is gone"
        assert "index" in os.path.basename(captured), captured

        # The lock index is deleted when the commit completes, so replay the same shape:
        # a temp index holding the staged path, named only in the variable.
        idx = os.path.join(d, "replay-index")
        subprocess.run(["git", "-C", d, "read-tree", "HEAD"], capture_output=True,
                       env=dict(env, GIT_INDEX_FILE=idx))
        with open(src, "w") as f:
            f.write("def alpha():\n    return 7\n\n\ndef beta():\n    return 42\n")
        subprocess.run(["git", "-C", d, "add", "m.py"], capture_output=True,
                       env=dict(env, GIT_INDEX_FILE=idx))
        _ORIG_GIT_INDEX_FILE = idx
        got = _funcs_in_diff(["m.py"])
        assert got == {"beta"}, (
            f"a path-scoped commit's staged diff selected {sorted(got)}: with GIT_INDEX_FILE "
            f"stripped, `git diff --cached` reads .git/index, which that commit never touches "
            f"-- this is 7fd8bc68's 'no CHECK function is changed'")
        # and the guard itself: without the env, the same world selects nothing.
        _ORIG_GIT_INDEX_FILE = ""
        assert _funcs_in_diff(["m.py"]) == set(), (
            "the world does not reproduce the defect -- the default index sees the scoped "
            "staging, so case 2's pass proves nothing")
    finally:
        ROOT, _ORIG_GIT_INDEX_FILE = saved_root, saved_idx
        shutil.rmtree(d, ignore_errors=True)
    print("  scoped index: a path-scoped commit's staged diff selects the edited function; "
          "with GIT_INDEX_FILE dropped the same world selects nothing")


def _selftest_batched_git_probes():
    """The three batched git probes agree with the per-item form they replaced, on the
    REAL repository, and the properties the zip depends on hold.

    Batching a probe is a rewrite of a criterion, so the risk is not that it is slow --
    it is that it silently answers a different question. Two of the three properties the
    mapping rests on are undocumented behaviour of `git cat-file --batch-check` and
    `git check-ignore --stdin`: one output line per input line, in input order, INCLUDING
    for a repeated input and for a non-match. Asserted, not assumed.

    A vacuous version of this test is easy to write and worthless: if the candidate list
    were empty, or the two forms were called on inputs where every answer is the same,
    it would pass with the batched form completely broken. So the worlds below include a
    known hit AND a known miss on each probe, and assert the two answers differ."""
    # 1. _cat_file_exists / _rev_has_path -- a live path and one that never existed.
    head = subprocess.run(
        ["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if not head:
        print("  batched git probes: SKIP (not a git checkout)")
        return
    live, dead = "scripts/harness.py", "scripts/definitely_not_a_file_xyz.py"
    batched = _cat_file_exists(ROOT, [f"{head}:{live}", f"{head}:{dead}"])
    assert batched[f"{head}:{live}"] is True, "a live path read as missing at HEAD"
    assert batched[f"{head}:{dead}"] is False, "a nonexistent path read as present"
    assert _rev_has_path(ROOT, head, live) is True
    assert _rev_has_path(ROOT, head, dead) is False
    # A repeated spec must produce a repeated line, or every later spec shifts by one.
    rep = _cat_file_exists(ROOT, [f"{head}:{live}", f"{head}:{dead}", f"{head}:{live}"])
    assert rep[f"{head}:{live}"] is True and rep[f"{head}:{dead}"] is False, rep

    # 2. _gitignored_set / _is_gitignored -- a gitignored path and a tracked one. The
    # batched form must agree with the single form on BOTH, and they must differ from
    # each other, so an implementation that returns one constant fails here.
    ig, tracked = "data/corpus/web_hq", "scripts/harness.py"
    s = _gitignored_set([ig, tracked], ROOT)
    assert s[ig] is True, f"{ig} is gitignored but the batched form said no"
    assert s[tracked] is False, f"{tracked} is tracked but the batched form said ignored"
    assert _is_gitignored(ig, ROOT) is True and _is_gitignored(tracked, ROOT) is False

    # The READER FALLBACK must agree with git on the same population, because it is the
    # only implementation that runs on the pod -- a divergence there FAILs a check nobody
    # can reproduce on a laptop, and two such divergences were live until this assertion
    # was written (a literal compare that could not match `data/corpus/*/`, and fnmatch's
    # `*` crossing `/` so `data/*.jsonl` swallowed `data/eval/math_test_500.jsonl`).
    #
    # Swept over EVERY fact source path plus controls for each pattern form, not over a
    # hand-picked pair: a pair proves the two agree somewhere, which is what the broken
    # version also did.
    import glob as _glob

    probe = set()
    for f in _glob.glob(os.path.join(ROOT, "facts", "*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                for e in json.load(fh)["facts"]:
                    probe.update(FACT_SOURCE_PATH.findall(str(e.get("source", ""))))
        except Exception:
            continue
    probe.update(
        {
            "scripts/harness.py",
            "data/corpus/web_hq",
            "data/corpus/primary",
            "data/corpus/primary/x.jsonl",
            "runs/tasks.jsonl",
            "__pycache__/x.pyc",
            "x.pyc",
            "ckpt_foo.pt",
            "runs/trace_x.json",
            "data/math/a.jsonl",
            "data/sft/a.parquet",
            "data/eval/math_test_500.jsonl",
            "data/eval/preds_l1.jsonl",
            "data/deep/nested/x.pt",
        }
    )
    # THE EXCEPTION IS DERIVED FROM THE INDEX, NOT A LIST OF NAMES. git consults the index,
    # so a directory that HOLDS A TRACKED FILE reads as not-ignored even when .gitignore
    # covers it, while this reader answers only what .gitignore says; `check-ignore
    # --no-index` agrees with the reader on exactly those paths.
    #
    # The first version hard-coded {"data/synthetic/", "data/synthetic"} -- the one path that
    # diverged the day it was written. Adding a fact source that names data/corpus/sample/
    # (148 tracked files under it) turned the selftest red on a divergence of the SAME KIND,
    # and a two-name allow-list has to be edited every time. Asking the index instead means
    # the exception is the property, so a NEW divergence of a DIFFERENT kind still fails.
    git_says = _gitignored_set(sorted(probe), ROOT)
    tracked_under = set(
        subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True).stdout.split()
    )

    def _holds_tracked(p):
        pre = p.rstrip("/") + "/"
        return any(t == p.rstrip("/") or t.startswith(pre) for t in tracked_under)

    disagree = [
        p for p in sorted(probe)
        if git_says[p] != _gitignore_reader(p, ROOT)
        # git says not-ignored, reader says ignored, and the index explains exactly that gap.
        and not (git_says[p] is False and _holds_tracked(p))
    ]
    assert not disagree, (
        f"the .gitignore reader disagrees with git on {len(disagree)} of {len(probe)} "
        f"path(s): {disagree[:5]} -- the pod runs the reader, so this is a FAIL only "
        f"reproducible there"
    )
    # And the sweep must not be vacuous: it has to contain both answers.
    assert any(git_says.values()) and not all(git_says.values()), (
        "the agreement sweep is one-sided, so it would pass against a constant"
    )
    assert _gitignore_reader("data/corpus/web_hq", ROOT) is True
    assert _gitignore_reader("data/eval/math_test_500.jsonl", ROOT) is False, (
        "fnmatch's * crossing / is back: data/*.jsonl must not match data/eval/..."
    )
    assert _gitignore_reader("data/corpus/primary", ROOT) is False, (
        "!data/corpus/primary/ negation is not honoured"
    )

    # 3. _resolve_shas -- a short sha resolves to full, a non-sha resolves to None.
    short = head[:8]
    got = _resolve_shas(ROOT, [short, "notacommit", short])
    assert got[short] == head, f"a short sha did not resolve to full: {got[short]!r}"
    assert got["notacommit"] is None, "a non-sha resolved to something"
    # Empty input must not spawn a subprocess or raise.
    assert _resolve_shas(ROOT, []) == {} and _cat_file_exists(ROOT, []) == {}
    assert _gitignored_set([], ROOT) == {}
    print(
        "  batched git probes: cat-file/check-ignore/resolve agree with the per-item "
        "form on a hit and a miss each, order survives a repeat, reader matches git"
    )


def _selftest_flagless_test_is_gated():
    """The WIDENED arm of selftests_are_gated: a flagless runnable test_*.py must be seen.

    The check's own broken world drops scripts/eval_artifacts.py, which CARRIES the flag, so
    it exercises the old population only -- the widened arm would be unguarded there while
    --selftest read green. Same shape as the run_checks mirror above: a guard placed where no
    test looks. One broken world per check, so this arm asserts here.

    Mutates the REAL hook by removing one REAL registration, and uses a file that is flagless
    on purpose: scripts/test_resume_accumulates.py, whose docstring explains it deliberately
    carries no --selftest. Under the old population it was invisible with docstrings stripped,
    which is exactly the blindness being fixed -- pick a flag-carrying file here and the
    mutation proves nothing about the widening (measured: 63 test_*.py tracked, 53 runnable,
    19 in neither map before this).

    Also asserts the NEGATIVE, because a predicate that fires on every test_*.py would pass
    the positive case for the wrong reason: a test_*.py with no `if __name__` runs nothing when
    executed and must NOT be demanded."""
    victim = "scripts/test_resume_accumulates.py"
    hookp = os.path.join(ROOT, "scripts", "hooks", "pre-commit")
    if not (os.path.exists(hookp) and os.path.exists(os.path.join(ROOT, victim))):
        return
    text = open(hookp, encoding="utf-8").read()
    if f'"{victim}",' not in text:
        return
    d = _tmp_repo_shaped()
    # scripts/ is a symlink into the real repo in a shaped world, so the mutated hook needs
    # its own directory -- but COPYING a handful of files makes the other 96 map entries look
    # deleted and the check FAILs on the stale-entry assertion instead, which is a world
    # failing for the wrong reason (the same trap _broken_selftests_are_gated records).
    # Mirror the real scripts/ by symlinking every entry, then override only hooks/.
    real_scripts = os.path.join(ROOT, "scripts")
    if os.path.islink(os.path.join(d, "scripts")):
        os.unlink(os.path.join(d, "scripts"))
        os.makedirs(os.path.join(d, "scripts"))
        for f in os.listdir(real_scripts):
            if f != "hooks":
                os.symlink(os.path.join(real_scripts, f), os.path.join(d, "scripts", f))
    hd = os.path.join(d, "scripts", "hooks")
    os.makedirs(hd, exist_ok=True)
    open(os.path.join(hd, "pre-commit"), "w", encoding="utf-8").write(
        text.replace(f'"{victim}",', "", 1))
    st, ev = check_selftests_are_gated(d)
    assert st == FAIL and victim in ev, (
        f"a flagless runnable test_*.py dropped from the map must FAIL and be named; "
        f"got {st}: {ev[:200]}")
    # The negative: a test_*.py that runs nothing when executed is not demanded.
    inert = os.path.join(d, "scripts", "test_inert_probe.py")
    open(inert, "w", encoding="utf-8").write("def helper():\n    return 1\n")
    open(os.path.join(hd, "pre-commit"), "w", encoding="utf-8").write(text)
    st2, ev2 = check_selftests_are_gated(d)
    assert "test_inert_probe.py" not in ev2, (
        f"a test_*.py with no `if __name__` runs nothing and must not be demanded: {ev2[:200]}")
    print("  selftests_are_gated: a flagless runnable test must be gated; an inert test_*.py "
          "is not demanded")
    # The odd-quote arm (§74): a comment inside the map with an odd number of double quotes
    # re-pairs every quote below it and silently drops entries. The cross-validation must
    # refuse rather than report a false FAIL.
    mutated = text.replace("SELFTEST_FILES = {",
                           'SELFTEST_FILES = {\n    # odd quote " here', 1)
    if mutated != text:
        open(os.path.join(hd, "pre-commit"), "w", encoding="utf-8").write(mutated)
        st3, ev3 = check_selftests_are_gated(d)
        assert st3 == FAIL and "invisible to this check" in ev3, (
            f"an odd quote in a map comment must FAIL with the cross-validation message; "
            f"got {st3}: {ev3[:200]}")
        print("  selftests_are_gated: an odd quote in a map comment is refused, not reported "
              "as a false FAIL")


def _selftest_repo_auth_mirror():
    """An auth=repo FAIL becomes SKIP on the pod's shape, and NOWHERE else.

    THE MIRROR LIVES IN run_checks, WHICH THE BROKEN-WORLD LOOP NEVER CALLS. That loop invokes
    each check function directly -- check_x(d) -- so nothing above this exercises the scoping at
    all, and the first version of this commit shipped with the mirror unguarded while the
    selftest read green. A guard placed where no test looks is the shape this file exists to
    catch (de, 2026-09-04).

    Three directions, because a scoping rule that fires too widely is worse than the blanket red
    it replaces: it would silently disable every repo check in CI.
    """
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="authmirror_")
    try:
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        assert pod_drift.is_pod(d), "a tree with no .git must read as the pod's shape"
        res = run_checks(d, quiet=True, persist_timeouts=False)
        by = {n: (s, e) for n, s, e, _a, _i in res}
        repo = [n for n, v in EVIDENCE.items() if v == "repo"]
        pod = [n for n, v in EVIDENCE.items() if v == "pod"]
        bad = [n for n in repo if by.get(n, ("", ""))[0] == FAIL]
        assert not bad, f"auth=repo check(s) still FAIL on the pod's shape: {bad[:4]}"
        mirrored = [n for n in repo if "not authoritative here" in (by.get(n, ("", ""))[1] or "")]
        assert mirrored, "no auth=repo check was mirrored -- the rule did not fire at all"
        # The original evidence is kept, or a SKIP reads as "checked and fine".
        sample = by[mirrored[0]][1]
        assert len(sample) > len("repo check, not authoritative here"), (
            f"the mirrored SKIP dropped its evidence: {sample!r}")
        # auth=pod checks are NOT mirrored: the pod stays strict about itself.
        wrong = [n for n in pod if "not authoritative here" in (by.get(n, ("", ""))[1] or "")]
        assert not wrong, f"auth=pod check(s) wrongly mirrored on the pod: {wrong[:4]}"
        # THE POSITIVE, and the one that matters: on a real checkout nothing is mirrored, so CI
        # keeps every repo check. Without this every assertion above passes on a rule that
        # silences repo checks everywhere.
        res2 = run_checks(ROOT, quiet=True, persist_timeouts=False)
        here = [n for n, _s, e, _a, _i in res2 if "not authoritative here" in (e or "")]
        assert not here, f"the mirror fired on a real checkout, disabling repo checks: {here[:4]}"
        print(f"  repo-auth mirror: {len(mirrored)} auth=repo FAIL(s) -> SKIP on the pod's shape, "
              f"0 on a checkout, 0 auth=pod mirrored")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _selftest_commit_delivers_fact_ref():
    """_commit_delivers understands facts/<f>.json#<id>: the fragment is stripped for the
    touched-file comparison and the id must exist in that file at HEAD (44-26).

    Three worlds: a real id passes, a fake id is named-refused, a bare path (no fragment)
    is unchanged. The done gate rejected the citation form check_fact_refs requires."""
    import tempfile
    d = tempfile.mkdtemp(prefix="cdfr_")
    try:
        def sh(*a):
            return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)
        sh("init", "-q", "-b", "main"); sh("config", "user.email", "t@t"); sh("config", "user.name", "t")
        os.makedirs(os.path.join(d, "facts"), exist_ok=True)
        json.dump({"facts": [{"id": "eff.real", "status": "measured"}]},
                  open(os.path.join(d, "facts", "efficiency.json"), "w"))
        sh("add", "."); sh("commit", "-qm", "add facts")
        sha = sh("rev-parse", "HEAD").stdout.strip()
        # world 1: real id passes
        assert _commit_delivers(sha, "facts/efficiency.json#eff.real", d) == "", \
            "a real fact id was refused"
        # world 2: fake id named-refused
        why = _commit_delivers(sha, "facts/efficiency.json#eff.fake", d)
        assert "eff.fake" in why, f"a fake id was not named in the refusal: {why}"
        # world 3: bare path unchanged
        assert _commit_delivers(sha, "facts/efficiency.json", d) == "", \
            "a bare path was refused"
        # worlds 4+: THE PUNCTUATION PROSE PUTS AROUND A PATH. Evidence is written as prose,
        # so a path arrives parenthesised, backticked, bracketed, or ending a sentence, and
        # the refusal then said the commit does not touch the named files -- pointing at the
        # wrong cause entirely (e1, 2026-09-04; its trailing-comma case already passed).
        # The parenthesised FACT CITATION failed twice over: "(facts/x.json#id)" never matched
        # the fragment split either, so the id check was skipped in silence.
        for ev in ("the ratio (facts/efficiency.json) is measured",
                   "measured in facts/efficiency.json.",
                   "see `facts/efficiency.json` for the value",
                   "[facts/efficiency.json] holds it",
                   "<facts/efficiency.json>",
                   "facts/efficiency.json, and the run log"):
            why = _commit_delivers(sha, ev, d)
            assert why == "", f"punctuation around a real path was refused: {ev!r} -> {why}"
        # ...and the fragment is still stripped when the citation is parenthesised, so a FAKE
        # id inside parens is still caught. Without this, widening the strip could have made
        # every parenthesised citation pass unverified -- a looser gate reading as a fixed one.
        why = _commit_delivers(sha, "(facts/efficiency.json#eff.fake)", d)
        assert "eff.fake" in why, f"a fake id inside parens was not caught: {why}"
        assert _commit_delivers(sha, "(facts/efficiency.json#eff.real)", d) == "", \
            "a real id inside parens was refused"
        # NEGATIVES. Stripping `.` must not turn prose into a path: a gate that accepts
        # anything is the failure mode this widening could introduce, and it is silent.
        for ev in ("measured the ratio and it holds", "this is done.", "..."):
            why = _commit_delivers(sha, ev, d)
            assert "names no path" in why, f"prose was read as a path: {ev!r} -> {why}"
        # The refusal must NAME the tokens it read as paths: without them, a wrong commit and
        # an unparsed path give the same message and the reader cannot tell which (e1).
        why = _commit_delivers(sha, "scripts/nonexistent_xyz.py", d)
        assert "tokens read as paths" in why and "scripts/nonexistent_xyz.py" in why, \
            f"the refusal does not name the tokens tried: {why}"
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    print("  commit_delivers: fact-ref real id passes, fake id named-refused, bare path "
          "unchanged, prose punctuation stripped (6 forms), prose still names no path")


def _selftest_review_present_legacy():
    """A legacy-unreviewed declaration closes a no-reviewer done row; without it the row WARNs (44-28).

    Two worlds over one register: a done row naming no reviewer, with and without the
    declaration. The pre-rule rows (t01/t02/...) named no reviewer because the rule did
    not exist; classifying them once -- review row where the artifact survives,
    legacy-unreviewed where it is gone -- must silence the permanent WARN, while a NEW
    unreviewed row must still warn."""
    import tempfile
    d = tempfile.mkdtemp(prefix="rpl_")
    try:
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        task = {"id": "t-old", "state": "done", "owner": "t", "closed": "2026-09-02 00:00"}
        with open(os.path.join(d, "runs", "tasks.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(task) + "\n")
        legacy = {"id": "44-legacy-t-old", "reviewer": "44", "owner": "t", "task": "t-old",
                  "verdict": "legacy-unreviewed", "finding": "artifacts gone"}
        with open(os.path.join(d, "runs", "review.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(legacy) + "\n")
        verdict, msg = check_review_present(d)
        assert verdict == PASS, f"a legacy-declared row still warns: {msg}"
        # remove the declaration: the same row must WARN naming it
        open(os.path.join(d, "runs", "review.jsonl"), "w").close()
        verdict, msg = check_review_present(d)
        assert verdict == WARN and "t-old" in msg, f"an undeclared no-reviewer row did not warn: {msg}"
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    print("  review_present: legacy declaration silences a pre-rule row; without it the row WARNs")


def _selftest_attest_written_path():
    """attest must record the path that was WRITTEN, not the one requested.

    open_artifact(path, run=...) writes a versioned path, so the two differ exactly
    when versioning is in use -- which is exactly when the distinction matters. Every
    eval writer attested its requested path, so l1_15b_final recorded a hash for
    preds_l1_d3.jsonl: the 477-row overwrite it was versioned specifically to avoid
    touching. The attestation pointed at the wrong file and the citation check could
    not tell (e1, 2026-09-01).

    The fix is to attest the handle's `.name`. This asserts the two paths differ under
    --run, so a future refactor that drops versioning cannot make the test vacuous."""
    import shutil
    import tempfile

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from eval_artifacts import attest, open_artifact

    d = tempfile.mkdtemp(prefix="attest_")
    try:
        os.makedirs(os.path.join(d, "data", "eval"), exist_ok=True)
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        requested = os.path.join(d, "data", "eval", "preds_x.jsonl")
        with open_artifact(requested, run="r1") as f:
            written = f.name
            f.write("{}\n")
        assert written != requested, (
            "the premise: --run must version the path, or this test proves nothing")
        attest(written, root=d)
        refs = os.path.join(d, "runs", "artifact_refs.jsonl")
        rows = [json.loads(x) for x in open(refs, encoding="utf-8") if x.strip()]
        assert rows, "attest wrote no row"
        got = os.path.basename(rows[-1]["path"])
        assert got == os.path.basename(written), \
            f"attested {got}, but {os.path.basename(written)} is the file that exists"
        assert got != os.path.basename(requested), \
            "attesting the requested path is the defect this test exists for"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  attest: records the written path, not the requested one")


def _selftest_check_timeout_skips():
    """A slow check must report TIMEOUT naming its deadline, not kill the harness, and
    a SECOND consecutive timeout must FAIL.

    Two defects, one test. (a) signal.alarm() with no handler runs SIG_DFL, which
    terminates: the `except TimeoutError` in run_checks was dead code and a slow check
    exited -14/142 with empty stdout and stderr. The hook then refused the commit with
    no check named and told the reader to rerun by hand, where it passes -- the
    --no-verify training P8 exists to prevent. It refused the commit carrying the
    e1-4 review of itself (2026-09-01).

    (b) That fix reported SKIP, which the hook prints nothing for and `check` exits 0
    on. A check that times out on EVERY run was therefore a permanent silent pass
    wearing a legitimate state's name (44, D5). TIMEOUT is its own state, and the second
    consecutive strike is a FAIL -- a deadline nothing can meet is a check nobody has.

    Tests the PROPERTY -- run_checks turns an overrun into a named non-pass, and a
    repeat into a failure -- not the mechanism. My first version asserted a handler was
    installed at import time, which was true only of my own fix; de's is scoped to
    run_checks and restores the previous handler, which is better, and the test failed
    on the better code. A test that encodes one implementation rejects its replacement."""
    slow_name = "__selftest_slow__"
    saved_checks = list(CHECKS)
    saved_to = _CHECK_TIMEOUTS.get(slow_name)
    _CHECK_TIMEOUTS[slow_name] = 1
    try:
        # Time the SLOW CHECK, not the whole run: run_checks executes all 51, so a
        # wall-clock assertion over the run measures the suite and fails on a healthy
        # machine. The check under test sleeps 3s and must be cut off at 1s.
        marks = []
        real_sleep = time.sleep

        def timed_slow(_root):
            t = time.time()
            try:
                real_sleep(3)
            finally:
                marks.append(time.time() - t)
            return PASS, "should never be reached"

        CHECKS.append((slow_name, "a check that overruns", "the harness dying with no name",
                       timed_slow, lambda: _tmp_repo()))

        def run_and_read():
            # persist_timeouts=False: the real ledger must not carry this fixture's strikes.
            results = run_checks(ROOT, quiet=True, persist_timeouts=False)
            row = [r for r in results if r[0] == slow_name]
            assert row, f"{slow_name} produced no result -- the run died"
            return row[0][1], row[0][2]

        real_read = globals()["_read_timeout_strikes"]
        # Strike 1, from a clean slate: TIMEOUT, non-blocking, names the deadline.
        globals()["_read_timeout_strikes"] = lambda: {}
        try:
            state, evidence = run_and_read()
        finally:
            globals()["_read_timeout_strikes"] = real_read
        assert state == TIMEOUT, f"an overrunning check must report TIMEOUT, got {state}: {evidence}"
        assert state != SKIP, "TIMEOUT must not be a SKIP -- that is the silent pass"
        assert "timed out" in evidence, f"the TIMEOUT must name the deadline: {evidence}"
        assert marks and marks[0] < 2.5, f"the alarm did not interrupt the check ({marks}s)"

        # Strike 2, with the previous run's count in hand: FAIL, so `check` exits 1.
        globals()["_read_timeout_strikes"] = lambda: {slow_name: 1}
        try:
            state2, evidence2 = run_and_read()
        finally:
            globals()["_read_timeout_strikes"] = real_read
        assert state2 == FAIL, (
            f"a second consecutive timeout must FAIL, got {state2}: {evidence2} -- "
            f"otherwise a check that never runs never says so")
        assert "consecutive" in evidence2, f"the FAIL must say why: {evidence2}"
    finally:
        CHECKS[:] = saved_checks
        if saved_to is None:
            _CHECK_TIMEOUTS.pop(slow_name, None)
        else:
            _CHECK_TIMEOUTS[slow_name] = saved_to
    print("  check timeout: strike 1 TIMEOUTs naming its deadline, strike 2 FAILs; the run survives both")


def _selftest_exp_fold():
    """The ledger is an event log: a close clears its start, and a stray later start
    does not reopen a closed run.

    Both halves are real defects from 2026-09-01. p02_fp32m_s0 was correctly closed
    with an appended event on the exact (name, started) pair and check_no_stale_running
    kept failing, because it walked raw lines. Then (sft_p324_v3, 03:44) turned out to
    carry an ok event at line 44 and a running event at line 132 — folding on file
    order alone reported a run that finished in 32 minutes as 26 hours stale."""
    import shutil
    import tempfile

    d = tempfile.mkdtemp(prefix="expfold_")
    try:
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        p = os.path.join(d, "runs", "experiments.jsonl")
        # Run c's timestamp is RELATIVE to now, derived from the same constant the check
        # reads. It was a hardcoded 2026-08-31 12:45 -- recent when written, older than
        # 24h a day later -- so the final assertion (c is open AND recent, therefore
        # PASS) inverted on its own and the selftest went permanently red. A fixture that
        # expires. Half the threshold, so it is unambiguously inside the window whatever
        # _STALE_RUNNING_H becomes.
        recent = time.strftime("%Y-%m-%d %H:%M",
                               time.gmtime(time.time() - _STALE_RUNNING_H * 3600 / 2))
        ev = [
            {"name": "a", "started": "2026-08-31 05:08", "status": "running", "ended": ""},
            {"name": "a", "started": "2026-08-31 05:08", "status": "fail", "ended": "2026-09-01 05:29"},
            # a duplicate START appended AFTER the close, the sft_p324_v3 shape
            {"name": "b", "started": "2026-08-31 03:44", "status": "ok", "ended": "2026-08-31 04:16"},
            {"name": "b", "started": "2026-08-31 03:44", "status": "running", "ended": ""},
            # a genuinely open run must survive the fold
            {"name": "c", "started": recent, "status": "running", "ended": ""},
            # ONE NAME, TWO RUNS, and it is here to give the key WIDTH something to fail
            # on. Without it the fixture cannot tell (name, started) from name alone --
            # every name appeared once, so both keyings returned identical rows and a
            # reader folding by name only (check_ladder_flags_declared did, dropping
            # `started`) passed the agreement assertion below. Verified by blinding:
            # exp.fold keyed on name alone goes green on the other four rows and red on
            # these two. d's two runs must stay two rows.
            {"name": "d", "started": "2026-08-30 06:05", "status": "ok", "ended": "2026-08-30 06:06"},
            {"name": "d", "started": "2026-08-30 06:13", "status": "fail", "ended": "2026-08-30 06:15"},
        ]
        with open(p, "w", encoding="utf-8") as f:
            for r in ev:
                f.write(json.dumps(r) + "\n")

        folded = {(r["name"], r["started"]): r for r in _exp_events(d)}
        assert folded[("a", "2026-08-31 05:08")]["status"] == "fail", "an appended close must clear its start"
        assert folded[("b", "2026-08-31 03:44")]["status"] == "ok", \
            "a start appended after a close must NOT reopen the run"
        assert folded[("c", recent)]["status"] == "running", \
            "a genuinely open run must still read as running"
        assert len(_exp_events(d, folded=False)) == len(ev), "raw=False must return every event"

        # a and b keep fixed dates on purpose: both are CLOSED, and check_no_stale_running
        # only looks at rows whose status is still running, so no amount of clock movement
        # reaches them. Only the open row had to become relative.
        state, evidence = check_no_stale_running(d)
        assert state == PASS, f"only run c is open and it is recent: {state} {evidence}"

        # EVERY READER, NOT JUST THIS ONE (e1-18). Four re-implementations of this fold
        # lived here and in exp.py, and three were wrong in different ways: position-based
        # last-wins in experiments() and check_no_ghost_running/check_score_matrix, and
        # name-only keying in check_ladder_flags_declared. exp.py:34's docstring asserted
        # the shape above was impossible -- "union-merging two branches cannot produce a
        # running row and a done row for the same run" -- while the comment 200 lines up
        # from here records the ledger containing it. Assert the AGREEMENT, because the
        # defect was never one reader's answer, it was two readers giving different ones.
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import exp as _exp
        prev_log = _exp.LOG
        try:
            _exp.LOG = p
            by_exp = {(r["name"], r["started"]): r["status"] for r in _exp.rows()}
        finally:
            _exp.LOG = prev_log
        by_harness = {(r["name"], r["started"]): r["status"] for r in _exp_events(d)}
        assert by_exp == by_harness, (
            f"exp.py and harness fold the same ledger differently: {by_exp} vs {by_harness} "
            "-- one ledger, one reduction, or a check and the tool disagree about what ran")
        assert by_exp[("b", "2026-08-31 03:44")] == "ok", \
            "exp.py rows() must not reopen a closed run either (it folded on position)"
        # AGREEMENT IS NOT THE PROPERTY, and finding that out is why these two lines
        # exist. Both readers now reach one fold, so a regression IN that fold moves both
        # and they agree while both are wrong -- verified by blinding exp.fold to
        # name-only keying, which went green on the agreement assertion above. Assert the
        # KEY WIDTH against d's two runs directly: one name, two `started` values, two
        # rows. That is what check_ladder_flags_declared dropped, and it decides which
        # run's flags grant a frozen-key exemption.
        for reader, got in (("exp.py rows()", by_exp), ("harness _exp_events", by_harness)):
            assert got.get(("d", "2026-08-30 06:05")) == "ok" and \
                got.get(("d", "2026-08-30 06:13")) == "fail", \
                (f"{reader} folded two runs of one name into one row ({got}) -- the key is "
                 "(name, started), and dropping `started` makes a re-run replace its "
                 "predecessor's record")
        # And the fold reached through the lazy import is the same fold. A silent fallback
        # to position-based would pass every assertion above, because the real exp.py is
        # importable here; check the degraded path explicitly.
        assert {(r["name"], r["started"]): r["status"] for r in _exp_fold(ev)} == by_harness, \
            "_exp_fold must agree with the reader it replaced"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  exp fold: close clears its start; a later start does not reopen; open runs "
          "survive; exp.py and harness agree")


def _selftest_gpu_descendants():
    """Known answer: a child whose cmdline shares nothing with its parent's is still
    found, because descent is what is walked.

    The failing case is the real one -- score_matrix (parent) shells out to
    math_zh.py (child). On 2026-08-31 a kill matched children by the parent's
    cmdline pattern, math_zh.py could not match, and it survived holding 12.7 GB on
    GPU 7. The verification greped the same pattern, so `killed; exp row closed`
    printed over a live orphan.

    Pure-function test on the ppid walk: no pod, no GPU.
    """
    def walk(gpu_pids, ppid, root, limit=12):
        out = []
        for p in gpu_pids:
            seen, cur = 0, p
            while cur in ppid and seen < limit:
                cur = ppid[cur]
                seen += 1
                if cur == str(root):
                    out.append(p)
                    break
        return out

    # score_matrix 200 -> bash eval_math.sh 250 (NO GPU) -> math_zh 300; 400 is a stranger.
    # The intermediate shell is the case that broke the first implementation: a ppid
    # map built only over GPU-holding pids stops at 250 and finds nothing. The map
    # must come from the whole process table.
    ppid = {"200": "100", "250": "200", "300": "250", "400": "999"}
    got = walk(["300", "400"], ppid, 100)
    assert got == ["300"], f"descent must cross the non-GPU shell and exclude the stranger: {got}"

    # The regression itself: a map missing the shell must NOT find the child. If this
    # ever passes, the map has silently narrowed back to GPU pids only.
    gpu_only = {"200": "100", "300": "250"}
    assert walk(["300"], gpu_only, 100) == [], "a map missing the intermediate shell must find nothing"

    # The failing case: pattern matching cannot find it. This is what the old code did.
    parent_cmdline = "score_matrix.py --ckpt X --profile milestone"
    child_cmdline = "math_zh.py --ckpt X --shards 1"
    assert parent_cmdline not in child_cmdline, "the premise: the child shares no cmdline text"

    # A cycle must not hang the kill path.
    got = walk(["1"], {"1": "2", "2": "1"}, 999)
    assert got == [], "a ppid cycle must terminate and match nothing"

    # A direct child is found too, not just a grandchild.
    assert walk(["200"], {"200": "100"}, 100) == ["200"]
    print("  gpu descendants: child found across a non-GPU shell; narrowed map finds nothing; "
          "cycle terminates; stranger excluded")


def _selftest_devs_map():
    """Known answer: eval/_devs.sh maps shards onto the caller's cards, and refuses
    when there are more shards than cards.

    The grep check (device_set_honoured) proves the scripts spell the idiom; this
    proves the idiom does the right thing. The case that matters is the incident:
    CUDA_VISIBLE_DEVICES=7 with eval_hard.sh's default N=6. The 2f97e4a fix mapped
    shard 0 to card 7 and let shards 1-5 fall back to physical 1-5, five
    training-block cards, so the spill survived its own fix.
    """
    helper = os.path.join(ROOT, "eval", "_devs.sh")
    if not os.path.exists(helper):
        raise SelftestSkip("eval/_devs.sh not present")

    def devs(cvd, n):
        env = dict(os.environ)
        if cvd is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = cvd
        r = subprocess.run(
            ["bash", "-c", f'source eval/_devs.sh {n} && echo "${{_DEVS[*]}}"'],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()

    rc, out, err = devs("7", 6)
    assert rc != 0, f"lane card with 6 shards must refuse, got rc=0 and {out!r}"
    assert "1" in err and "6 shards" in err, err

    rc, out, _ = devs("7", 1)
    assert (rc, out) == (0, "7"), f"lane card with 1 shard -> {out!r} rc={rc}"

    rc, out, _ = devs("2,5", 2)
    assert (rc, out) == (0, "2 5"), f"two cards -> {out!r} rc={rc}"

    rc, out, _ = devs(None, 6)
    assert (rc, out) == (0, "0 1 2 3 4 5"), f"unset caller -> {out!r} rc={rc}"

    print("  devs: CVD=7 N=6 refuses; CVD=7 N=1 -> [7]; CVD=2,5 N=2 -> [2 5]; unset N=6 -> [0..5]")


def _selftest_register_union():
    """fb's case: two branches each close a DIFFERENT row, the files union-merge,
    and the result reads as both closed with no duplicate complaint.

    Union merge concatenates; that is only safe because done APPENDS an event
    rather than rewriting the row. This asserts the property directly on a
    concatenated file, which is what .gitattributes produces."""
    import shutil
    import tempfile

    d = tempfile.mkdtemp(prefix="register_")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    p = os.path.join(d, "runs", "tasks.jsonl")
    base = [
        {"id": "de-1", "owner": "de", "state": "open", "task": "A", "why": "w", "opened": "2026-08-31 10:00"},
        {"id": "b0-1", "owner": "b0", "state": "open", "task": "B", "why": "w", "opened": "2026-08-31 10:01"},
    ]
    # Two branches, each appending one close event to its own copy.
    side_de = dict(base[0], state="done", evidence="runs/a.log", closed="2026-08-31 11:00")
    side_b0 = dict(base[1], state="done", evidence="runs/b.log", closed="2026-08-31 11:05")
    _write_tasks(base + [side_de] + [side_b0], p)  # the union of both branches

    folded = {r["id"]: r for r in _read_tasks(p)}
    assert folded["de-1"]["state"] == "done", f"de-1 must fold to done: {folded['de-1']}"
    assert folded["b0-1"]["state"] == "done", f"b0-1 must fold to done: {folded['b0-1']}"
    assert folded["de-1"]["evidence"] == "runs/a.log"
    assert folded["b0-1"]["evidence"] == "runs/b.log"
    assert len(_read_tasks(p)) == 2, "folded view holds one row per id"
    assert len(_read_tasks(p, raw=True)) == 4, "raw view holds every event"

    state, evidence = check_tasks_well_formed(d)
    assert state == PASS, f"union-merged register must PASS, got {state}: {evidence}"

    # A real collision -- same id, a different task -- must still FAIL.
    _write_tasks(base + [side_de, side_b0, dict(base[0], task="C", opened="2026-08-30 09:00")], p)
    state, evidence = check_tasks_well_formed(d)
    assert state == FAIL and "collision" in evidence, f"a real id collision must FAIL: {state} {evidence}"
    shutil.rmtree(d, ignore_errors=True)
    print("  register: union-merged closes fold to done; a same-id different-task row still FAILs")


def _selftest_auto_resume():
    """de-1's three cases, on real child processes: a crash after a .step save is
    resumed once, a clean exit is not, the kill-criterion code is not.

    Real Popen children, not a mocked return code: the bug this supervisor replaced
    was a zombie whose exit the parent never observed, which no mock reproduces."""
    global ROOT
    import shutil
    import tempfile

    real_root, real_sleep = ROOT, time.sleep
    d = tempfile.mkdtemp(prefix="autoresume_")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    child = os.path.join(d, "child.py")
    # Writes a step checkpoint on the first run, then exits with the code it was given.
    # On a resumed run (--resume present) it exits 0, so a resumed job terminates.
    open(child, "w").write(
        "import os, sys\n"
        "rc = int(sys.argv[1])\n"
        "if '--resume' in sys.argv:\n"
        "    open(os.path.join(sys.argv[2], 'resumed.txt'), 'a').write(' '.join(sys.argv[3:]) + '\\n')\n"
        "    sys.exit(0)\n"
        "print('traceback: the crash scene', flush=True)\n"
        "open(os.path.join(sys.argv[2], 'ckpt_arts.pt.step500'), 'w').write('x')\n"
        "sys.exit(rc)\n"
    )

    class _A:
        name, training, output, auto_resume = "arts", False, None, 1

    def run(rc):
        for f in ("resumed.txt", "ckpt_arts.pt.step500"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                os.remove(p)
        log = os.path.join(d, "runs", "arts.log")
        cmd = [sys.executable, child, str(rc), d]
        with open(log, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, cwd=d)
        out = _supervise(_A(), cmd, proc, "", log, os.path.join(d, "runs", "arts.pid"), root=d)
        resumed = os.path.join(d, "resumed.txt")
        return out, (open(resumed).read() if os.path.exists(resumed) else "")

    try:
        ROOT = d
        time.sleep = lambda s: None  # the 60s backoff is policy, not behaviour under test
        rc, resumed = run(1)
        assert "--resume" in resumed and "step500" in resumed, f"a crash must resume from the step ckpt: {resumed!r}"
        assert resumed.count("\n") == 1, f"exactly one resume for N=1: {resumed!r}"
        assert rc == 0, f"a successful resume returns 0, got {rc}"

        rc, resumed = run(0)
        assert resumed == "", f"a clean exit must NOT resume: {resumed!r}"
        assert rc == 0

        rc, resumed = run(_KILL_CRITERION_EXIT)
        assert resumed == "", f"the kill criterion must NOT resume: {resumed!r}"
        assert rc == _KILL_CRITERION_EXIT, f"the kill-criterion code is returned as-is, got {rc}"

        # The crash scene survives the resume that appends over the live log, and a
        # newer interrupt save is preferred over the older periodic one.
        for f in glob.glob(os.path.join(d, "runs", "arts.log.died_*")):
            os.remove(f)
        rc, resumed = run(1)
        died = glob.glob(os.path.join(d, "runs", "arts.log.died_*"))
        assert len(died) == 1, f"one crash, one archive: {died!r}"
        with open(died[0]) as f:
            assert "the crash scene" in f.read(), "the archive is the scene, not an empty file"
        with open(os.path.join(d, "ckpt_arts.pt.interrupt.step900"), "w") as f:
            f.write("x")
        ck, step = _latest_step_ckpt("arts")
        assert step == 900 and "interrupt" in ck, f"an interrupt save at a later step wins: {ck!r}"
    finally:
        ROOT, time.sleep = real_root, real_sleep
        shutil.rmtree(d, ignore_errors=True)
    print("  auto-resume: crash resumes once, clean exit and kill criterion do not; "
          "the scene is archived and a newer interrupt save wins")


def _staged_index_env():
    """The env for a `git diff --cached` that must see what THIS commit is staging.

    main() strips GIT_INDEX_FILE from os.environ, and it must: a selftest running `git init`
    under an inherited GIT_DIR reconfigured the shared repository twice on 2026-09-02. But
    a path-scoped commit -- `git commit <paths>` -- builds a TEMPORARY index and names it
    only in GIT_INDEX_FILE. With the variable gone, `git diff --cached` reads .git/index,
    which for that commit holds nothing.

    MEASURED 2026-09-04 in a throwaway repo: under `git commit -- a.txt`, the hook sees
    GIT_INDEX_FILE=.git/next-index-<pid>.lock, `git diff --cached` names a.txt, and the same
    command with the variable unset names NOTHING. Under a plain staged `git commit` the
    variable is .git/index and both forms agree, which is why the defect was invisible:
    every test of the scoped selftest had staged with `git add` first.

    The consequence was real. 7fd8bc68 rewrote check_tasks_well_formed and added a broken
    world, was committed path-scoped, and the hook printed "no CHECK function is changed by
    the staged diff" -- zero of 80 verified, reported as nothing-to-do. That is the shape the
    scoped substitute exists to prevent, in the substitute itself.

    So: keep os.environ clean, and pass the captured value back only to the git calls that
    ask what is staged. _ORIG_GIT_INDEX_FILE is read once at import, before main() strips it.
    """
    if not _ORIG_GIT_INDEX_FILE:
        return None
    return dict(os.environ, GIT_INDEX_FILE=_ORIG_GIT_INDEX_FILE)


def _funcs_in_diff(paths, rev=None):
    """Top-level function names whose body the staged diff of `paths` touches.

    FUNCTION granularity, not file: every one of the CHECKS functions lives in harness.py, so a
    file-level filter selects all 79 and saves nothing -- measured, which is why the first
    version of this was useless for the only caller it has. Reads the changed line numbers from
    `git diff` and maps each to the enclosing `def` by AST line span.

    The --cached read carries _staged_index_env(): under a path-scoped commit the index is a
    temporary file named only in GIT_INDEX_FILE, which main() has stripped, and without it this
    selected zero functions for a commit that rewrote a check.
    """
    import ast
    import bisect

    out = set()
    for rel in paths:
        args = ["git", "diff", "--unified=0"]
        args += [rev] if rev else ["--cached"]
        r = subprocess.run(args + ["--", rel], capture_output=True, text=True, cwd=ROOT,
                           env=None if rev else _staged_index_env())
        lines = set()
        for line in r.stdout.splitlines():
            m = re.match(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            if m:
                start, count = int(m.group(1)), int(m.group(2) or 1)
                lines.update(range(start, start + max(count, 1)))
        if not lines:
            continue
        try:
            tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())
        except (OSError, SyntaxError):
            continue
        spans = sorted((n.lineno, n.end_lineno, n.name) for n in tree.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        starts = [s[0] for s in spans]
        for ln in lines:
            i = bisect.bisect_right(starts, ln) - 1
            if i >= 0 and spans[i][1] >= ln:
                out.add(spans[i][2])
    return out


def _checks_touching(paths, rev=None):
    """Check names whose run() or broken() the staged diff of `paths` changes.

    For the hook's cheap substitute: a commit that edits one check should verify THAT check's
    broken world, not all 79 (~4 min). Resolved by each function's own __code__.co_name and
    co_filename, so it cannot drift from where the code actually lives.

    An edit OUTSIDE any check function -- a helper both call, the CHECKS table itself -- returns
    nothing, and the caller must treat that as "no answer", never as a pass. That is the honest
    reading: a helper change can break any check, and the only cheap thing that covers it is
    the full run.
    """
    changed = _funcs_in_diff(paths, rev=rev)
    if not changed:
        return []
    want = {os.path.realpath(os.path.join(ROOT, p)) for p in paths}
    out = []
    for name, _a, _i, fn, broken in CHECKS:
        for f in (fn, broken):
            code = getattr(f, "__code__", None)
            if (code and os.path.realpath(code.co_filename) in want
                    and code.co_name in changed):
                out.append(name)
                break
    return out


def _demo(only=None):
    """Every check must FAIL on a world where its condition is violated."""
    import shutil

    # The step gap: a card idle on the first reading and busy on the second is busy.
    global _busy_once
    _real, seen = _busy_once, []
    _busy_once = lambda cards: seen.append(1) or ([] if len(seen) == 1 else ["3"])
    try:
        assert _busy_cards(["3", "4"], settle=3) == ["3"], "a one-sample gap let a held card through"
    finally:
        _busy_once = _real

    assert score_from("math-hard 37/1032 = 3.6%") == 3.6, "took the numerator, not the percentage"
    assert score_from("math-hard deferred to the bench stage") is None, "invented a score"
    assert score_from("math-hard 1.7% (18/1032) vs k5 1.9%") == 1.7

    # _noted_gone: name and gone-word must share a sentence; the tier vocabulary counts.
    # The cross-sentence case is de's review finding (9420c8b): concatenating the fields
    # read "measured on X. Y was pruned" as X's own death disclosure.
    assert _noted_gone({"uncertainty": "ckpt_k5_clean_0827.pt [absent] -- not in the listing"},
                       "ckpt_k5_clean_0827.pt")
    assert _noted_gone({"uncertainty": "step1500 pruned before this reading"},
                       "ckpt_p500m_20b_0902.pt.step1500")
    assert not _noted_gone({"uncertainty": "在 ckpt_x.pt.step1500 上测的。step2000 被剪了"},
                           "ckpt_x.pt.step1500")
    assert not _noted_gone({"uncertainty": "step1000 was fine"}, "ckpt_x.pt.ep1")
    assert not _noted_gone({"uncertainty": "nothing here"}, "ckpt_x.pt.step1500")

    # A QUOTED TIER LABEL ONLY DISCLOSES ITS OWN TIER (de-53, the N2 legs, 2026-09-04).
    # `_n2` is ds.n2_params_vs_data_matched_compute's own shape: it quotes [absent] to
    # explain that the checkpoints POSTDATE the listing and asserts they exist. Against a
    # newer listing they are deletion candidates, and that word was the entire credit.
    _n2 = {"uncertainty": "BOTH CHECKPOINTS POSTDATE THE LISTING and therefore read "
                          "[absent] to ckpt_facts_sources_present: ckpt_data_leg_206m_8b.pt "
                          "was verified present on the pod at 19:10Z, 892,199,291 bytes"}
    assert _noted_gone(_n2, "ckpt_data_leg_206m_8b.pt", "absent"), \
        "a note quoting [absent] must still disclose the absent tier it answers"
    assert not _noted_gone(_n2, "ckpt_data_leg_206m_8b.pt", "deletion-candidate"), \
        "[absent] credited a deletion-candidate: a FAIL for a doomed ckpt read WARN"
    # Tier-blind is the pre-fix behaviour; keep the untiered call answering the old question.
    assert _noted_gone(_n2, "ckpt_data_leg_206m_8b.pt"), "untiered call changed meaning"
    # Prose disclosure is unrestricted -- the tier does not narrow a real statement.
    _prose = {"uncertainty": "ckpt_z.pt was pruned on the 09-04 plan before this reading"}
    for _t in ("absent", "deletion-candidate", "zeroed", None):
        assert _noted_gone(_prose, "ckpt_z.pt", _t), f"prose disclosure lost at tier {_t}"
    # A matching label still counts, and the two other labels in one sentence do not leak.
    assert _noted_gone({"uncertainty": "ckpt_q.pt is [deletion-candidate], unclaimed"},
                       "ckpt_q.pt", "deletion-candidate")
    assert not _noted_gone({"uncertainty": "ckpt_q.pt reads [zeroed] and [absent] here"},
                           "ckpt_q.pt", "deletion-candidate")

    # run dispatch: a missing or unknown step is a usage error, not a silent exit 0
    assert run_dispatch([]) == 2 and run_dispatch(["bogus"]) == 2

    saved = os.path.join(ROOT, "runs", "experiments.jsonl")
    if os.path.exists(saved) and os.path.getsize(saved):
        # Only when the log carries math-hard-shaped results: a fresh log (the 0830v1 reset
        # wiped it) legitimately has none, and an empty parse of an empty-of-scores log is
        # not a parser regression. A ceval percentage in a row does not count -- recorded_scores
        # is the math-hard ledger.
        rows = [json.loads(l) for l in open(saved, encoding="utf-8") if l.strip()]
        if any(SCORE_RE.search(str(r.get("result", ""))) for r in rows):
            s, _o = recorded_scores()
            assert s, "math-hard-shaped results exist but none attributed: score_from stopped parsing"

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        json.dump({"domains": {}}, open(p, "w"))
        _d, e = read_mix(p)
        assert e, "an empty domains map read as a valid mix"

    # mix_shards_present's strict branch runs only on a GPU box; force it so the broken
    # world exercises the branch that runs on the pod.
    os.environ["HARNESS_GPU_PRESENT"] = "1"
    # A broken world must hold at least one file at a path the real repo also contains. A
    # world hand-written from the check's own assumptions cannot -- scan_math_contamination's
    # self-check wrote its own rows with its own field names and never saw a real corpus
    # shape. no_oversized_blob is the exception: its artifact is a >MAX file that cannot
    # exist in the repo by design; its reality comes from real git plumbing, not a repo file.
    # Known ceiling: this catches worlds built on made-up paths, not worlds that mutate one
    # real file and hand-write the rest -- the latter is a code-review property, not a tree one.
    # env_importable joins it for the same reason: its artifact is process import state,
    # not a tree, so no world it builds can hold a repo file. no_foreground_pod_training
    # joins for the same reason again -- its artifact is a process TABLE. Its rows are not
    # hand-written despite sitting in a fixture: they are de's verbatim capture of a real
    # foreground trainer on the pod, which is what the reality rule actually asks for.
    synthetic_world = {"no_oversized_blob", "env_importable", "no_foreground_pod_training"}

    # A check whose evidence lives on the POD cannot satisfy the repo-real rule either, and
    # for a reason worth stating rather than exempting: data/pod_synced_head is written only
    # on the pod and is not tracked here, so NO world built in this repository can hold a file
    # at that path. That is the other face of the check's own docstring -- a clause the pod
    # cannot answer is answered here, so its artifact is not a repo file.
    #
    # The exemption carries a SUBSTITUTE assertion rather than dropping the rule. The point of
    # "mutate a real artifact" is that the world must be built from something outside the
    # check's own assumptions; for this world that something is git itself. So the substitute
    # asserts the world's stamp names a commit REAL git resolves and that main REALLY does not
    # contain -- the two properties a hand-written hex string cannot have. A world that passes
    # this could not have been invented.
    def _stamp_world_is_real(world):
        p = os.path.join(world, "data", "pod_synced_head")
        if not os.path.exists(p):
            return "the world holds no stamp at data/pod_synced_head"
        sha = (open(p, encoding="utf-8").read().split() or [""])[0]
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            return f"the stamp holds {sha!r}, not a full 40-char hex sha"
        if subprocess.run(["git", "-C", ROOT, "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True).returncode:
            return f"{sha[:12]} is no commit in the real repository -- invented, not taken"
        if subprocess.run(["git", "-C", ROOT, "merge-base", "--is-ancestor", sha, "main"],
                          capture_output=True).returncode == 0:
            return (f"{sha[:12]} IS an ancestor of the real main, so the world does not hold "
                    f"the condition the check exists to catch")
        return None

    world_reality = {"pod_stamp_is_main": _stamp_world_is_real}
    # WARN-only checks: their broken world must produce WARN (or FAIL), not PASS/SKIP.
    # review_present joined them on 2026-09-01 when the user cut the blocking: a check
    # with no FAIL tier cannot have a FAILing broken world, and demanding one would
    # force the tier back. What its world must still prove is that removing a review row
    # is VISIBLE -- WARN is the signal, silence is the defect.
    warn_only = {"untracked_aged", "dirty_aged", "review_present", "probe_numbers_unique",
                 "no_shared_stash", "keep_claim_reasons_live", "pod_ledger_rows_home",
                 "run_commits_resolve", "pod_stamp_is_main", "unreached_files_ruled",
                 "peer_stalled", "card_held_without_claim", "merge_keeps_parent_paths"}
    untested = []
    skipped = []
    for name, _a, _i, fn, broken in CHECKS:
        if only is not None and name not in only:
            continue
        try:
            root = broken()
        except SelftestSkip as e:
            print(f"  SKIP {name}: {e}")
            skipped.append(name)
            continue
        try:
            if name in world_reality:
                why = world_reality[name](root)
                if why:
                    untested.append(f"{name}: broken world is not built from a real artifact -- {why}")
                    continue
            elif name not in synthetic_world and not any(
                os.path.exists(os.path.join(ROOT, os.path.relpath(os.path.join(dp, f), root)))
                for dp, _dn, fns in os.walk(root)
                for f in fns
            ):
                untested.append(f"{name}: broken world holds no file at a repo-real path -- hand-written?")
                continue
            state, evidence = fn(root)
            if name in warn_only:
                if state in (PASS, SKIP):
                    untested.append(f"{name} reported {state} on its broken world ({evidence})")
            elif state != FAIL:
                untested.append(f"{name} reported {state} on its broken world ({evidence})")
        except Exception as e:
            untested.append(f"{name} raised instead of reporting FAIL: {e}")
        finally:
            shutil.rmtree(root, ignore_errors=True)
            os.environ.pop("HARNESS_REQUIRE_EXTRA", None)  # _broken_env leaks this
            os.environ.pop("HARNESS_POD_PS", None)  # its world is a temp ps capture
            os.environ.pop("HARNESS_POD_LEDGERS", None)  # same: a temp dir of pod ledgers
            os.environ.pop("HARNESS_POD_STAMP", None)  # same: a temp pod sync stamp
    # HARNESS_GPU_PRESENT is set once before the loop and needed by several broken
    # worlds (mix_shards_present, lane_respected); clean up after the whole loop.
    os.environ.pop("HARNESS_GPU_PRESENT", None)
    # spawned_scripts_exist needs a SECOND world. Its registered one is "the file moved", which
    # FAILs before the importability half ever runs. c3a47e8 broke both properties and the first
    # fix covered only one: pretokenize.py sat at the right path and still raised
    # ModuleNotFoundError, which is how tonight's cache warming died (b0, 2026-09-01).
    _imp = _broken_spawned_scripts_importable()
    try:
        _st, _why = check_spawned_scripts_exist(_imp)
        if _st != FAIL:
            untested.append(f"spawned_scripts_exist reported {_st} on a present-but-"
                            f"unimportable script ({_why[:60]})")
    finally:
        shutil.rmtree(_imp, ignore_errors=True)

    # agents_rules_covered needs a SECOND world for the same structural reason: CHECKS
    # carries one broken() per row, and its registered world breaks the TABLE half (a
    # coverage row naming the wrong check). The BULLET half -- a rule mapped to neither a
    # check nor a manual reason -- has its world written and never run, which is why the
    # deletion audit read it as dead code. It is not: on that world the check FAILs with
    # "1 rule(s) map to neither a check nor a manual reason", a defect the table world
    # cannot produce. Measured before wiring, since a world nobody runs is indistinguishable
    # from one that cannot fail.
    _unm = _broken_agents_rules_unmapped()
    if _unm:
        try:
            _st, _why = check_agents_rules_covered(_unm)
            if _st != FAIL:
                untested.append(f"agents_rules_covered reported {_st} on an unmapped rule "
                                f"bullet ({_why[:60]})")
        finally:
            shutil.rmtree(_unm, ignore_errors=True)

    # shapes_table_covers_doc has two halves its registered world does not exercise: an
    # incident that reaches the doc but not the table, and a heading number written twice.
    # _broken_shapes_table_doc_grew existed but nothing ran it -- a broken world nobody
    # runs is the §71 shape itself.
    for _w, _label in ((_broken_shapes_table_doc_grew, "doc grew, table stood still"),
                       (_broken_shapes_table_duplicate_heading, "duplicate heading number")):
        _d = _w()
        if _d:
            try:
                _st, _why = check_shapes_table_covers_doc(_d)
                if _st != FAIL:
                    untested.append(f"shapes_table_covers_doc reported {_st} on {_label} ({_why[:60]})")
            finally:
                shutil.rmtree(_d, ignore_errors=True)

    # score_matrix_present gained two branches on 2026-09-04 and its registered world exercises
    # neither. Both are worlds where the check could SILENTLY PASS, which is the shape it was just
    # fixed for: reading_artifact is an escape hatch, so a path that does not exist must FAIL rather
    # than wave the row through; and a cmd that names no checkpoint used to be skipped outright,
    # which is how e1_31_middle_layer_loop passed while its honestly-written sibling did not.
    for _w, _want, _label, _needle in (
        (_broken_score_matrix_dangling_artifact, FAIL, "reading_artifact at a missing path",
         "does not exist"),
        (_broken_score_matrix_no_ckpt, WARN, "an ok training row whose cmd names no checkpoint",
         "names no checkpoint"),
    ):
        _d = _w()
        if _d:
            try:
                _st, _why = check_score_matrix(_d)
                if _st != _want:
                    untested.append(f"score_matrix_present reported {_st}, wanted {_want}, on "
                                    f"{_label} ({_why[:70]})")
                elif _needle not in _why:
                    untested.append(f"score_matrix_present hit the right tier on {_label} but does "
                                    f"not say why: {_why[:70]}")
            finally:
                shutil.rmtree(_d, ignore_errors=True)

    # tasks_well_formed's registered world breaks the ID half (a collision). The drop_reason
    # clause needs its own, and it needs a REVERSIBILITY assertion rather than only a FAIL. When
    # it was written the live register missed the field on seven rows, so a world built by copying
    # it reported the same tier whether or not the mutation was present -- three worlds were green
    # for exactly that reason (2171). So: repair every dropped row, break one, assert FAIL naming
    # that id, then fill the field back in and assert PASS. The second half is what proves the
    # FAIL came from the mutation, and it stays after e1 filled the last five: the fixture is now
    # independent of the register's state, which is the property that made it worth writing.
    # (A third assertion covered DROP_REASON_GRANDFATHERED's WARN tier and went with the list.)
    _d = _broken_tasks_drop_reason()
    if _d:
        try:
            _p = os.path.join(_d, "runs", "tasks.jsonl")
            _rows = _read_tasks(_p, raw=True)
            _victim = next((r["id"] for r in _read_tasks(_p)
                            if r.get("state") == "dropped"
                            and not str(r.get("drop_reason") or "").strip()), None)
            if not _victim:
                untested.append("tasks_well_formed's drop_reason world holds no row missing the "
                                "field -- the mutation did not land")
            else:
                _st, _why = check_tasks_well_formed(_d)
                if _st != FAIL:
                    untested.append(f"tasks_well_formed reported {_st} on a dropped row with no "
                                    f"drop_reason ({_why[:70]})")
                elif "drop_reason" not in _why or str(_victim) not in _why:
                    untested.append(f"tasks_well_formed FAILs on the drop_reason world but does "
                                    f"not name the field and the row: {_why[:90]}")
                _filled = [dict(r, drop_reason="(filled by the fixture)")
                           if r.get("state") == "dropped" else r for r in _rows]
                _write_tasks(_filled, _p)
                _st, _why = check_tasks_well_formed(_d)
                if _st != PASS:
                    untested.append(f"tasks_well_formed still reported {_st} after the drop_reason "
                                    f"was filled back in, so its FAIL was not caused by the "
                                    f"mutation ({_why[:70]})")
        finally:
            shutil.rmtree(_d, ignore_errors=True)

    # reported_path_is_written needs a POSITIVE world, and it is the only reason the fix is
    # verified in the direction it was made. The check used to match any LOAD of a Name called
    # preds_path, so it refused a correct CALL to a function of that name and e1 renamed to
    # artifact_path to get past it (29b31367 records the rename's reason in its own docstring).
    # A rename to satisfy a check is the check making the codebase worse. MEASURED both ways on
    # 2026-09-04: on the call-form world the pre-fix logic FAILs at l1_fewshot.py:518 and the
    # fixed logic PASSes, while on the defect world both FAIL -- so the fix is not merely a
    # loosening. The FAIL world alone could not tell those apart.
    _d = _positive_reported_path()
    if _d:
        try:
            _st, _why = check_reported_path_is_written(_d)
            if _st != PASS:
                untested.append(f"reported_path_is_written reported {_st} on a CALL to a "
                                f"function named preds_path -- the marker is still the name, "
                                f"not the defect ({_why[:80]})")
        finally:
            shutil.rmtree(_d, ignore_errors=True)

    if only is not None:
        # THE FILTERED RUN STOPS HERE, and says what it did not do. Everything below is a
        # second world for a NAMED check (spawned_scripts_exist, agents_rules_covered,
        # shapes_table_covers_doc, score_matrix_present), the repo-auth mirror, and the
        # non-vacuous-PASS sweep over every check's live evidence -- none of which is scoped to
        # a name, so running them under a filter would either re-run the full cost the filter
        # exists to avoid or silently skip properties the caller was not told about.
        #
        # A filtered green is NOT the selftest passing. Printed as a count of what ran against
        # the total, because the failure this whole item is about is a green line that describes
        # less coverage than the reader assumes.
        assert not untested, ("checks that cannot be made to fail:\n  "
                              + "\n  ".join(untested))
        print(f"harness selftest (FILTERED): {len(only)} of {len(CHECKS)} checks verified on "
              f"their broken worlds -- {', '.join(sorted(only))}. NOT run: the extra worlds, "
              f"the repo-auth mirror, and the non-vacuous-PASS sweep. Run the full "
              f"`harness check --selftest` before trusting this as coverage.")
        return 0

    assert not untested, "checks that cannot be made to fail:\n  " + "\n  ".join(untested)

    _selftest_repo_auth_mirror()
    _selftest_flagless_test_is_gated()

    # The other half of the selftest: a PASS must have verified something. A check that
    # examined zero items and returned PASS is vacuous -- the shape shared by score_matrix_present
    # (0 ok runs in the ledger), lane_respected (0 PIDs resolvable in a container), and the two
    # zero-count PASSes above. The broken-world loop asserts FAILs fire; this asserts PASSes are
    # non-vacuous. A count is a number followed by a unit word; a PASS whose evidence carries
    # counts ALL of which are zero verified nothing. (A legal zero -- "0 new (170 checked)" --
    # passes because not every count is zero.) A check with nothing to examine on this machine
    # must SKIP, not PASS.
    #
    # Ceiling: this only covers checks whose evidence happens to contain "digit space letter".
    # A check that degrades to `return PASS, "ok"` -- no digits at all -- is invisible here, and
    # so is a zero written as a fraction ("0/36 hits", sft_pack_uncontaminated's format) or with
    # "=" ("checked=0"). The lower-entropy fix is structured counts (a check returns n_examined
    # alongside evidence), not a smarter regex -- that is an arms race with string formats. Not
    # done; the next person should know this green does not cover a no-digit PASS.
    #
    # The leading boundary is load-bearing, not tidiness: without it the regex reads a count out
    # of a PATH. In a worktree named aupai-b0, root_durable's "root /.../aupai-b0 is not on a
    # known-ephemeral mount" yields the match "0 is", the only "count" in the string, so a
    # correct PASS was reported as vacuous and the selftest was red in that worktree and green
    # in the main one (b0, 2026-09-01). A count is preceded by whitespace or start-of-string;
    # a digit glued to a word is part of a name.
    #
    # The meta-check carries its own failing case: a fake check whose PASS is vacuous. Without
    # it nothing proves the meta-check fires -- the exact defect it guards against.
    def _vacuous_pass(_root):
        return PASS, "0 domain(s) match filters abc"

    # ...and its own false-positive case, the one above: a path-embedded digit is not a count.
    def _pass_with_digit_in_a_path(_root):
        return PASS, "root /Users/x/code/aupai-b0 is not on a known-ephemeral mount"

    vacuous = []
    for name, _a, _i, fn, _b in list(CHECKS) + [
        ("fake_vacuous_pass", "", "", _vacuous_pass, None),
        ("fake_digit_in_a_path", "", "", _pass_with_digit_in_a_path, None),
    ]:
        try:
            state, evidence = fn(ROOT)
        except Exception:
            continue  # a crash against the real repo is the broken-world loop's territory
        if state != PASS:
            continue
        counts = [int(m.group(1)) for m in re.finditer(r"(?:^|\s)(\d+)\s+[a-zA-Z]", str(evidence))]
        if counts and all(c == 0 for c in counts):
            vacuous.append(f"{name}: PASS with all-zero counts ({evidence})")
    assert any(v.startswith("fake_vacuous_pass") for v in vacuous), (
        "meta-check did not catch its own deliberately-vacuous PASS -- the regex or loop regressed"
    )
    assert not any(v.startswith("fake_digit_in_a_path") for v in vacuous), (
        "meta-check read a count out of a path -- the leading boundary regressed"
    )
    real = [v for v in vacuous
            if not v.startswith("fake_vacuous_pass") and not v.startswith("fake_digit_in_a_path")]
    assert not real, "PASS with nothing verified:\n  " + "\n  ".join(real)

    # sync selftest: a merge that loses a row must FAIL. The incident: a hand-merge
    # keyed by name dropped 9 rows (p02_s0 x4, p03 x5 share a name; identity is
    # (name, started)). Two failure modes, both on the REAL file:
    exp_path = os.path.join(ROOT, "runs", "experiments.jsonl")
    real = [l for l in open(exp_path, encoding="utf-8") if l.strip()]
    if len(real) >= 5:
        eid = lambda r: (r.get("name", ""), r.get("started", ""))
        pod_fake = real[:10] if len(real) >= 10 else real[:5]
        merged, err = _merge_jsonl(pod_fake, real, eid, "selftest")
        assert err is None, f"sync selftest: clean merge refused: {err}"
        # A repeated identity is LEGAL: a start row and a done row share (name, started)
        # by design. The old selftest asserted the opposite and locked in the refusal
        # that blocked every sync once done began appending (2026-08-31).
        dup, err = _merge_jsonl(pod_fake + [pod_fake[0]], real, eid, "selftest")
        assert err is None, f"sync selftest: a repeated identity must merge, not refuse: {err}"
        n_dup = len([x for x in dup.strip().split("\n") if x])
        assert n_dup == len([x for x in merged.strip().split("\n") if x]), (
            f"a byte-identical duplicate must collapse: {n_dup}")
        # A second EVENT for the same run (same identity, different bytes) is kept.
        ev = json.loads(pod_fake[0]); ev["status"] = "ok"; ev["result"] = "later event"
        two, err = _merge_jsonl(pod_fake + [json.dumps(ev)], real, eid, "selftest")
        assert err is None and len([x for x in two.strip().split("\n") if x]) == n_dup + 1, (
            "a second event for one run must survive the merge")
        # failure mode 1: a truncated pod row -> refuse, never ride into the ledger
        _, err = _merge_jsonl(pod_fake + [pod_fake[0][:20]], real, eid, "selftest")
        assert err and "parse" in err, f"sync selftest: truncated pod row not caught: {err}"
        # failure mode 2: a merged output missing a pod row -> verify catches it.
        # Drop an identity, not a LINE. Two events legitimately share (name, started)
        # -- a start and its close -- so dropping the last line can leave its identity
        # present through the sibling event and the check correctly reports nothing
        # lost. This selftest used to pass only because the real ledger's last line
        # happened to be identity-unique; closing p02_fp32m_s0 made it a duplicate and
        # the assert stopped firing (2026-09-01). It was testing the fixture.
        pod_ids = [eid(json.loads(l)) for l in pod_fake]
        pod_set = set(pod_ids)
        repo_only_ids = [eid(json.loads(l)) for l in real if eid(json.loads(l)) not in pod_set]
        merged_ids = [eid(json.loads(l)) for l in merged.strip().split("\n")]
        dropped = pod_ids[0]
        kept = [i for i in merged_ids if i != dropped]
        assert len(kept) < len(merged_ids), "the selftest must actually drop something"
        err = _verify_merge(pod_ids, repo_only_ids, kept, "selftest")
        assert err and "lost" in err, f"sync selftest: lost row not caught: {err}"

    # pod_push manifest freshness: --check-head must fail when a scoped file changed
    # in HEAD without a regenerated manifest. pod_push runs this before any transfer,
    # so a stale manifest refuses before a byte is pushed.
    import tempfile
    d = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=d, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d, capture_output=True)
    os.makedirs(os.path.join(d, "scripts"))
    open(os.path.join(d, "scripts", "real.py"), "w").write("# v1\n")
    subprocess.run(["git", "add", "."], cwd=d, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "v1"], cwd=d, capture_output=True)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    _old_manifest = pod_drift.MANIFEST
    pod_drift.MANIFEST = os.path.join(d, "data", "pod_head_manifest.txt")
    try:
        # check_head is gone with the manifest's tracking (shape A, 2026-09-04): "is the
        # committed manifest stale against HEAD" has no subject once the file is generated
        # from HEAD at push time. What survives, and is the property pod_push.sh depends on:
        # write_manifest describes the HEAD it ran against, and describes the NEW HEAD after
        # a commit changes a scoped file. Both directions, since a generator that returned
        # the same bytes regardless would pass the first assertion alone.
        n = pod_drift.write_manifest(d)
        assert n == 1, f"expected 1 scoped file, got {n}"
        first = pod_drift.read_manifest(pod_drift.MANIFEST)
        assert first["scripts/real.py"][0] == pod_drift.sha_head(d, "scripts/real.py"), (
            f"a freshly generated manifest must describe HEAD: {first}")
        open(os.path.join(d, "scripts", "real.py"), "w").write("# v2\n")
        subprocess.run(["git", "add", "."], cwd=d, capture_output=True)
        subprocess.run(["git", "commit", "-qm", "v2"], cwd=d, capture_output=True)
        stale = pod_drift.read_manifest(pod_drift.MANIFEST)
        assert stale["scripts/real.py"][0] != pod_drift.sha_head(d, "scripts/real.py"), (
            "the pre-existing manifest must NOT describe the new HEAD -- if it did, this "
            "fixture could not tell a regenerating generator from one that does nothing")
        pod_drift.write_manifest(d)
        after = pod_drift.read_manifest(pod_drift.MANIFEST)
        assert after["scripts/real.py"][0] == pod_drift.sha_head(d, "scripts/real.py"), (
            f"regeneration must pick up the new HEAD: {after}")
    finally:
        pod_drift.MANIFEST = _old_manifest

    # pre-commit hook selftest: a staged 6MB file must exit non-zero; a small
    # allowed data file must pass; a small unallowed data file must refuse.
    d = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=d, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d, capture_output=True)
    hook_dst = os.path.join(d, ".git", "hooks", "pre-commit")
    os.makedirs(os.path.dirname(hook_dst), exist_ok=True)
    os.symlink(os.path.join(ROOT, "scripts", "hooks", "pre-commit"), hook_dst)
    # 6MB file -> must refuse
    with open(os.path.join(d, "big.jsonl"), "w") as f:
        f.write("x" * (6 * 1024 * 1024))
    subprocess.run(["git", "add", "big.jsonl"], cwd=d, capture_output=True)
    r = subprocess.run([hook_dst], cwd=d, capture_output=True)
    assert r.returncode != 0, f"6MB staged file must refuse: {r.stdout}"
    # small allowed data file -> must pass
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    open(os.path.join(d, "data", "mix_test.json"), "w").write("{}")
    subprocess.run(["git", "reset", "-q"], cwd=d, capture_output=True)
    subprocess.run(["git", "add", "data/mix_test.json"], cwd=d, capture_output=True)
    r = subprocess.run([hook_dst], cwd=d, capture_output=True)
    assert r.returncode == 0, f"allowed data file must pass: {r.stdout} {r.stderr}"
    # small unallowed data file -> must refuse
    open(os.path.join(d, "data", "secret.jsonl"), "w").write("small")
    subprocess.run(["git", "add", "data/secret.jsonl"], cwd=d, capture_output=True)
    r = subprocess.run([hook_dst], cwd=d, capture_output=True)
    assert r.returncode != 0, f"unallowed data file must refuse: {r.stdout}"

    # pre-merge-commit: a non-ff merge bringing an unlisted data/ path must be
    # refused. git runs no pre-commit hook on a clean merge, so this is the only
    # commit-time gate on the merge path (2026-08-31: a bad fact landed in main
    # through a merge). The branch commit uses --no-verify: the point is the merge.
    dm = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=dm, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=dm, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=dm, capture_output=True)
    for hk in ("pre-commit", "pre-merge-commit"):
        dst = os.path.join(dm, ".git", "hooks", hk)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(os.path.join(ROOT, "scripts", "hooks", "pre-commit"), dst)
    open(os.path.join(dm, "README"), "w").write("base\n")
    subprocess.run(["git", "add", "README"], cwd=dm, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=dm, capture_output=True)
    subprocess.run(["git", "checkout", "-qb", "side"], cwd=dm, capture_output=True)
    os.makedirs(os.path.join(dm, "data"), exist_ok=True)
    open(os.path.join(dm, "data", "evil.bin"), "w").write("x")
    subprocess.run(["git", "add", "data/evil.bin"], cwd=dm, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "evil", "--no-verify"], cwd=dm, capture_output=True)
    subprocess.run(["git", "checkout", "-q", "main"], cwd=dm, capture_output=True)
    open(os.path.join(dm, "other"), "w").write("y")  # diverge so the merge is non-ff
    subprocess.run(["git", "add", "other"], cwd=dm, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "other"], cwd=dm, capture_output=True)
    r = subprocess.run(["git", "merge", "side", "--no-edit"], cwd=dm, capture_output=True, text=True)
    assert r.returncode != 0, f"merge with an unlisted data/ path must be refused: {r.stdout} {r.stderr}"
    head = subprocess.run(["git", "log", "--oneline", "-1"], cwd=dm, capture_output=True, text=True).stdout
    assert "other" in head, f"refused merge left a merge commit: {head}"

    # Manifest regeneration: stage a scoped edit, run the hook, commit, and
    # pod_drift.py --check-head must pass without a second commit.
    d2 = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=d2, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=d2, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d2, capture_output=True)
    hook_dst2 = os.path.join(d2, ".git", "hooks", "pre-commit")
    os.makedirs(os.path.dirname(hook_dst2), exist_ok=True)
    os.symlink(os.path.join(ROOT, "scripts", "hooks", "pre-commit"), hook_dst2)
    os.makedirs(os.path.join(d2, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(d2, "data"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "scripts", "pod_drift.py"), os.path.join(d2, "scripts", "pod_drift.py"))
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d2, "AGENTS.md"))
    subprocess.run(["git", "add", "-A"], cwd=d2, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=d2, capture_output=True)
    # Stage a scoped edit
    with open(os.path.join(d2, "AGENTS.md"), "a") as f:
        f.write("\n# test edit\n")
    subprocess.run(["git", "add", "AGENTS.md"], cwd=d2, capture_output=True)
    r = subprocess.run([hook_dst2], cwd=d2, capture_output=True)
    assert r.returncode == 0, f"hook must pass on scoped edit: {r.stdout} {r.stderr}"
    # THE HOOK MUST LEAVE THE MANIFEST ALONE (shape A, 2026-09-04). This world used to assert
    # the opposite -- that the hook staged a regenerated manifest and --check-head then passed
    # -- and both halves are now the defect: regenerating a derived file on every commit made
    # it the top friction cause, and in a merge commit the regen could only reach the index.
    # Asserting the absence rather than deleting the world, because "the hook no longer touches
    # this file" is a property worth a failing test if anyone puts it back.
    staged_files = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=d2, capture_output=True, text=True
    ).stdout
    assert "data/pod_head_manifest.txt" not in staged_files, (
        f"the hook staged the manifest -- it is untracked and generated by pod_push.sh from "
        f"the HEAD it ships: {staged_files}")
    subprocess.run(["git", "commit", "-m", "scoped edit"], cwd=d2, capture_output=True)
    # And the removed flag must REFUSE rather than silently succeed, so a caller still
    # running it is told the question is gone instead of reading exit 0 as an answer.
    r = subprocess.run(
        [sys.executable, os.path.join(d2, "scripts", "pod_drift.py"), "--check-head"],
        cwd=d2, capture_output=True, text=True,
    )
    assert r.returncode != 0 and "was removed" in (r.stderr + r.stdout), (
        f"--check-head must refuse loudly now that it is gone: {r.returncode} "
        f"{r.stdout} {r.stderr}")
    shutil.rmtree(d2, ignore_errors=True)

    # launch gate selftest: a log without the fa/doc_mask lines must time out;
    # a log with them must pass immediately.
    gate_log = os.path.join(tempfile.mkdtemp(), "gate.log")
    open(gate_log, "w").write("params 200M | device cuda | world 7 | fa True | fp8 True\n")
    ok, _ = _wait_for_startup(gate_log, timeout=3)
    assert not ok, "gate must FAIL without doc_mask line"
    open(gate_log, "a").write("cfg batch 16 doc_mask True attn_res 0/0\n")
    ok, _ = _wait_for_startup(gate_log, timeout=3)
    assert ok, "gate must PASS with both lines True"
    # fa False must kill immediately, not wait for timeout
    open(gate_log, "w").write("params 200M | device cuda | world 7 | fa False | fp8 True\n")
    open(gate_log, "a").write("cfg batch 16 doc_mask True attn_res 0/0\n")
    t0 = time.time()
    ok, reason = _wait_for_startup(gate_log, timeout=30)
    assert not ok and "fa False" in reason, f"gate must FAIL on fa False, got {ok} {reason}"
    assert time.time() - t0 < 5, "fa False must kill immediately, not wait for timeout"
    shutil.rmtree(os.path.dirname(gate_log), ignore_errors=True)

    # task reopen: done -> open keeps prior evidence, appends reason, check accepts the transition.
    # Operates on a temp copy of the real register, never the ledger itself.
    import tempfile as _tf
    tmp_root = os.path.join(_tf.mkdtemp())
    tmp_tasks = os.path.join(tmp_root, "runs", "tasks.jsonl")
    os.makedirs(os.path.dirname(tmp_tasks), exist_ok=True)
    real_rows = _read_tasks()
    if real_rows:
        # The subject here is the REOPEN transition, so every other tier the live register
        # carries has to be neutralised or this case reports someone else's row. Seven dropped
        # rows carried no drop_reason on 2026-09-04 and a bare copy of the register then failed
        # this case for a reason unrelated to a reopen. All 51 carry the field now, so the fill
        # below is a no-op today and stays anyway: the next row dropped without one would
        # otherwise turn this case red and point at the reopen path.
        real_rows = [dict(r, drop_reason=(r.get("drop_reason") or "(filled by the fixture)"))
                     if r.get("state") == "dropped" else r for r in real_rows]
        _write_tasks(real_rows, tmp_tasks)
        test_row = dict(real_rows[0])
        test_row.update(id="t_selftest", state="done", evidence="prior evidence",
                        owner="selftest", why="test", closed=time.strftime("%Y-%m-%d %H:%M", time.gmtime()))
        rows = _read_tasks(tmp_tasks) + [test_row]
        _write_tasks(rows, tmp_tasks)
        # Reopen: same transition as cmd_task
        rows = _read_tasks(tmp_tasks)
        hit = [r for r in rows if r.get("id") == "t_selftest"]
        assert hit and hit[0]["state"] == "done", "selftest row must start done"
        prior = hit[0].get("evidence", "")
        hit[0].update(state="open", reopen_reason="selftest reopen",
                      reopened=time.strftime("%Y-%m-%d %H:%M", time.gmtime()), evidence=prior)
        hit[0].pop("closed", None)
        _write_tasks(rows, tmp_tasks)
        # Verify: state=open, evidence preserved, reopen_reason present, check accepts
        rows = _read_tasks(tmp_tasks)
        hit = [r for r in rows if r.get("id") == "t_selftest"][0]
        assert hit["state"] == "open", "reopen must set state=open"
        assert hit["evidence"] == "prior evidence", "reopen must preserve prior evidence"
        assert hit.get("reopen_reason") == "selftest reopen", "reopen must carry the reason"
        assert "closed" not in hit, "reopen must remove closed"
        state, _ = check_tasks_well_formed(tmp_root)
        assert state == PASS, f"tasks_well_formed must accept a reopened row, got {state}"
        shutil.rmtree(tmp_root, ignore_errors=True)

    # PYTHONUNBUFFERED: a child launched with the launcher's env must see it,
    # so block-buffered stdout does not starve the log-silent monitor.
    r = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('PYTHONUNBUFFERED',''))"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert r.stdout.strip() == "1", f"child must see PYTHONUNBUFFERED=1, got {r.stdout.strip()!r}"

    # 30B readiness: a world with one _blocked domain must return NOT READY naming it.
    d30 = tempfile.mkdtemp()
    os.makedirs(os.path.join(d30, "data"), exist_ok=True)
    os.makedirs(os.path.join(d30, "runs"), exist_ok=True)
    json.dump(
        {"total_tokens": 30e9, "domains": {},
         "_blocked": {"code_rp1t": {"weight": 1.0, "epochs": 1, "anneal": 1.0}}},
        open(os.path.join(d30, "data", "mix_30b.json"), "w"),
    )
    open(os.path.join(d30, "runs", "tasks.jsonl"), "w").close()  # empty JSONL
    ready, gates = _30b_readiness(d30)
    assert not ready, "a blocked domain must make readiness NOT READY"
    blocked_gate = [g for g in gates if g[0] == "mix_30b blocked"]
    assert blocked_gate and blocked_gate[0][1] == FAIL, f"must have a FAIL mix_30b blocked gate: {gates}"
    assert "code_rp1t" in blocked_gate[0][2], f"gate must name the blocked domain: {blocked_gate[0][2]}"
    shutil.rmtree(d30, ignore_errors=True)

    _selftest_milestone_reachable()
    _selftest_cold_cache_refuses()
    _selftest_refusal_writes_no_row()
    _selftest_provenance_states_the_tree()
    _selftest_pool_not_raw_supply()
    _selftest_killpg_reaps_children()
    _selftest_milestone_selection()
    _selftest_monitor_suppression()
    _selftest_gate_timeout()
    _selftest_register_union()
    _selftest_auto_resume()
    _selftest_devs_map()
    _selftest_gpu_descendants()
    _selftest_exp_fold()
    _selftest_check_timeout_skips()
    _selftest_attest_written_path()
    _selftest_merge_fix_not_deadlocked()
    _selftest_merge_cherry_pick_not_a_drop()
    _selftest_merge_reverted_content()
    _selftest_commit_delivers_fact_ref()
    _selftest_batched_git_probes()
    _selftest_scoped_index_is_read()
    _selftest_peer_stalled_names_the_fixture()
    _selftest_review_present_legacy()

    # Every check must PASS or SKIP on the real tree at the moment it lands.
    # A check that is red on the real artifact the day it ships is the
    # permanent-red failure mode; selftest must catch it, not the first colleague.
    for name, _a, _i, fn, _broken in CHECKS:
        state, _evidence = fn(ROOT)
        assert state != FAIL, f"{name} FAILs on the real tree -- fix the check or the artifact before landing"

    # Every check declares where its evidence lives (EVIDENCE); a check added
    # without a declaration would be classified by nobody, and a stale name is
    # noise. Equality, not subset: both directions fail loudly.
    check_names = {n for n, *_ in CHECKS}
    assert set(EVIDENCE) == check_names, (
        f"EVIDENCE stale: {sorted(set(EVIDENCE) - check_names)}; "
        f"undeclared: {sorted(check_names - set(EVIDENCE))}")

    # THE COUNT MUST NOT INCLUDE WHAT WAS SKIPPED. `len(CHECKS)` claimed "81 checks each verified to
    # FAIL on a broken world" while a SelftestSkip meant some of them were never run -- the skip
    # printed above, but the closing line is what people read and quote, and it overclaimed by
    # exactly the checks whose worlds could not be built. Same class as the defect this run's own
    # commit fixes: a number that reads as coverage without being it.
    _verified = len(CHECKS) - len(skipped)
    _tail = f"; {len(skipped)} SKIPPED, not verified: {', '.join(sorted(skipped))}" if skipped else ""
    print(f"harness self-test OK ({_verified} of {len(CHECKS)} checks each verified to FAIL on a "
          f"broken world; every PASS verified a non-zero count{_tail})")


STEPS = ("pretokenize", "point", "ladder", "fetch", "clean", "score", "dedup")

# The six 0830v1 budget points, in order. Each is a mix_scale_* mix at the
# frozen run config. Names double as checkpoint names: ckpt_<name>.pt.
LADDER = [
    # p02_s0, not p02: the 0.2b point is the seed-0 run (ckpt_p02_s0.pt), already
    # scored. Naming the entry p02_s0 makes the skip regex match it -- the curve's
    # 0.2b point and the sigma-hat measurement come from the same checkpoint.
    ("p02_s0", "data/mix_scale_0.2b.json"),
    ("p03", "data/mix_scale_0.3b.json"),
    ("p04", "data/mix_scale_0.4b.json"),
    ("p08", "data/mix_scale_0.8b.json"),
    ("p16", "data/mix_scale_1.6b.json"),
    ("p324", "data/mix_scale_3.24b.json"),
]


# A step must not be blocked by the red it exists to clear. `run pretokenize` builds the
# token caches, and mix_supply is red precisely because they are missing -- gating one on
# the other is a deadlock with no way out but --force, which then also waves through the
# reds that DO matter.
#
# fetch -> clean -> score -> dedup: each step makes the next step's precondition green.
# fetch is step 1, so every corpus-chain red is red because the chain hasn't run yet --
# blocking fetch on any of them deadlocks the chain. clean rebuilds the corpus and clears
# mix_shards_present / corpus_filters_fp / corpus_fp directly. score re-scores and clears
# score_input_fresh. dedup writes a manifest without touching shards, so no check goes
# red because of it and it needs no exemption.
_REPAIRS = {
    "pretokenize": {"mix_supply"},
    "fetch": {"mix_shards_present", "corpus_filters_fp", "corpus_fp", "score_input_fresh"},
    "clean": {"mix_shards_present", "corpus_filters_fp", "corpus_fp"},
    "score": {"score_input_fresh"},
}


def _gate(force, step=None):
    """The red invariants, by name and evidence. A runnable step needs none: 'no GPU
    pretraining while harness is red' was a doc line nothing executed until this."""
    repaired = _REPAIRS.get(step, ())
    reds = [(n, ev) for n, s, ev, _a, _i in run_checks(ROOT, quiet=True)
            if s == FAIL and n not in repaired]
    for n, ev in reds:
        print(f"  RED {n}: {ev}")
    if reds and not force:
        print("REFUSING to run while harness is red. Pass --force to override "
              "(the reds are recorded in the exp row).")
    return reds


def _exp(action, **kw):
    cmd = [sys.executable, os.path.join(HERE, "exp.py"), action]
    for k, v in kw.items():
        cmd += [f"--{k}", str(v)]
    subprocess.run(cmd, cwd=ROOT, check=False)


def _run_pretokenize(step_args, forced):
    cmd = [sys.executable, os.path.join(ROOT, "datagen", "pretokenize.py"), *step_args]
    _exp("start", name="pretokenize", cmd=" ".join(cmd),
         hypothesis=f"tokenize every mix domain into its cache before training{forced}")
    r = subprocess.run(cmd, cwd=ROOT)
    _exp("done", name="pretokenize", result=f"exit {r.returncode}",
         finding="caches warm" if r.returncode == 0 else "pretokenize failed",
         decision="training can launch on warm caches" if r.returncode == 0 else "fix the failure before launching",
         status="ok" if r.returncode == 0 else "fail")
    return r.returncode


def _step_name(step, step_args):
    """fetch --source web_hq -> fetch_web_hq; bare step name when no source/domain."""
    for i, a in enumerate(step_args):
        if a in ("--source", "--domain") and i + 1 < len(step_args):
            return f"{step}_{step_args[i + 1]}"
        if a.startswith(("--source=", "--domain=")):
            return f"{step}_{a.split('=', 1)[1]}"
    return step


def _preflight_fetch(step_args):
    """Disk guard at the harness entry, before the exp row. A fetch that cannot fit
    must refuse here, not 200GB into the pull. The script keeps its own guard for
    direct invocation (defense-in-depth); this is the one that blocks the pipeline."""
    target = 0.0
    for i, a in enumerate(step_args):
        if a == "--target_bytes" and i + 1 < len(step_args):
            target = float(step_args[i + 1])
            break
        if a.startswith("--target_bytes="):
            target = float(a.split("=", 1)[1])
            break
    sys.path.insert(0, HERE)
    import fetch_corpus
    fetch_corpus.ensure_raw_location()
    return fetch_corpus.disk_ok(target)


def _run_pipeline_step(step, script, step_args, forced, env=None):
    """fetch/clean/score: gate (in dispatch) + exp start/done + run the script.
    The script owns the work, the output fingerprint, and shard-level resumability.
    Score pins CUDA_VISIBLE_DEVICES=0 -- a collision on GPU 0 is visible (benchmarks
    fail), a collision on 1-7 is silent (training corrupted)."""
    _check_data_under_root(step)
    cmd = [sys.executable, os.path.join(HERE, script), *step_args]
    name = _step_name(step, step_args)
    _exp("start", name=name, cmd=" ".join(cmd), hypothesis=f"{step} step{forced}")
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    ok = r.returncode == 0
    _exp("done", name=name, result=f"exit {r.returncode}",
         finding=f"{step} complete" if ok else f"{step} failed",
         decision="next step can run" if ok else "fix the failure before next step",
         status="ok" if ok else "fail")
    return r.returncode


_FROZEN_KEYS = (
    "batch", "accum", "warmup", "vocab", "bucket_cap_mb",  # recipe
    "warmdown", "anneal_frac",  # WSD schedule shape: recipe, must match across a staged run
    "attn_res_blocks", "attn_every", "attn_res", "attn_res_dyn_q",  # architecture
    # ARCHITECTURE, and the most load-bearing entry in this set: it decides whether a block holds
    # one mixer or both. A resume that changed it would not be a different hyperparameter, it
    # would be a different model wearing one run's name -- and unlike attn_every it also changes
    # the parameter count (+1.18% at d1024 L12 h8, measured 2026-09-04), so the tok/s and the
    # loss curve would both move for a reason the log does not record.
    "head_mixed",
    # ARCHITECTURE, and frozen for the ordinary reason plus a specific one. It changes what a
    # document attends to inside the KDA short_conv (eff.kda_document_isolation_violated), so two
    # segments of one run that disagree on it trained two different models -- the same argument as
    # attn_res. The specific reason: every checkpoint before 2026-09-04 trained with it effectively
    # False, and scripts/loader.py pins it False when a checkpoint's cfg lacks the key, so a resume
    # that flipped it would change the topology mid-run while the log kept one name for both halves.
    "conv_doc_isolated",
    # Numerics, not init: it changes every step, so a resume DOES honour it -- but two
    # segments of one run that disagree on it are still incomparable, which is what this
    # set is for. bf16 vs fp32 accumulation of the AttnRes logit dot product moves the
    # mixing weights 14% (measured vs fp64).
    "attn_res_fp32_logits",
    # Initialisation: FROZEN, not measurement. It changes the trajectory from step 0, so two
    # segments of one run that disagree on it are not comparable -- and because it only acts
    # at __init__, a resume silently ignores it, which is exactly the drift the frozen set
    # exists to catch (the arm's own weights carry the init; the flag does not).
    "zero_init_out", "muon_shape_lr", "value_embed",
    # b0-17: untie_head acts only at __init__ (model.py:359) -- the arm's weights carry the
    # architecture and a resume silently ignores the flag, which is the drift this set catches.
    # head_lr is NOT here: it is the A/B knob that exists to take two values (1e's ruling
    # 2026-09-03), and it moves into this set the day an untied head ships in the recipe.
    "untie_head",
    "seq", "grad_ckpt", "fone", "doc_mask",  # architecture / training comparability
    "d", "heads", "layers", "ffn_hidden",  # shape: CLI-settable from 2026-09-01 (500M; --dim sets d)
    # N7 Stage D (b0, 2026-09-03): --loop changes the TOPOLOGY from step 0 -- blocks LO..HI are
    # visited twice -- so two segments of one run that disagree on it are not comparable, and it
    # belongs here for zero_init_out's reason rather than in the allow-list for no_attn_res's.
    # It is an A/B knob, but unlike head_lr the two arms are two SEPARATE RUNS, never two
    # segments of one: a resume that dropped --loop would continue looped weights with an
    # unlooped body and the logs would show nothing. Cfg.loop_blocks in the checkpoint is the
    # other half of that guard -- the flag says what was asked for, the metadata says what ran.
    "loop",
)

# Architecture constants with no CLI flag. They cannot drift via a launch, so
# _strip_frozen and frozen_args do not touch them. But they can drift via a code
# edit, and ladder_config_frozen compares them against the JSON as documented
# intent -- closing the gap where all six points agree with each other but not
# with what was intended (fb regenerated the manifest mid-ladder, blinding pod_drift).
_CODE_FROZEN_KEYS = ("chunk_size",)  # the shape moved to _FROZEN_KEYS when it got flags

# CLI flags whose name differs from their Cfg field (--no_attn_res sets Cfg.attn_res).
# --no_doc_mask is gone: it existed because the attention fallback could not honour
# doc_mask, and now it can, so the flag was only a way to turn a frozen recipe key off.
_FLAG_TO_CFG = {"no_attn_res": "attn_res", "dim": "d"}

# Parser flags intentionally outside the frozen set. Criterion: a flag that changes the
# architecture or the recipe is frozen; these are run-management, measurement, or
# deliberately variable. check_frozen_keys_complete forces a decision when a new flag lands.
# A module whose real code may live one level down (see check_env_importable).
_REQUIRED_ALT = {"flash_attn": "flash_attn.cute"}

_UNFROZEN_ALLOWLIST = {
    "seed",               # the quantity that is supposed to vary
    "name", "mix", "resume", "max_steps",  # run management
    "save_every",         # checkpoint cadence, an operational knob, not a recipe key
    "fp8",                # training precision, not architecture
    "track", "profile", "profile_warmup", "profile_steps",  # measurement
    "allow_corpus_drift", "allow_pod_drift", "allow_env_drift", "allow_partial_cursor",  # safety overrides
    "lr_scale",           # optimizer multiplier, varies by experiment
    "no_static_graph", "no_bucket_view",  # DDP A/B, do not touch Cfg
    "val_every", "val_batches",  # validation cadence, not architecture
    # An A/B arm, like no_attn_res: it exists to take two values, so freezing it would
    # declare settled the very thing the experiment is run to settle. MOVE IT INTO
    # _FROZEN_KEYS the day the A/B says fp32 masters ship -- a decided setting left here
    # is one a launch can silently omit.
    "fp32_master",
    # b0-17's lr knob, same reasoning as fp32_master: freezing it would declare settled the very
    # thing arms 2 and 3 are run to settle. MOVE IT INTO _FROZEN_KEYS if an untied head ships.
    "head_lr",
    "frozen_probe",       # measurement switch; does not change what is measured
    # Not a recipe key: it changes how attention is computed, not what is computed. It
    # exists so the ~20x-slower fallback cannot be entered by accident, which is the
    # opposite of a knob a launch may vary quietly.
    "allow_slow_attn",
}


def _strip_frozen(passthrough, frozen):
    """Drop agreeing frozen flags from passthrough; refuse disagreeing ones.
    Returns (clean_passthrough, conflicts). An agreeing flag is accepted, not
    refused -- fb launched four runs with explicit --batch 16 --accum 2 that
    matched; refusing presence would block a correct launch.
    Bool flags: presence of --<bool> sets True; --no_<bool> sets False. An
    agreeing bool flag is kept (Cfg default may differ from frozen); a
    conflicting one is refused."""
    s = set(_FROZEN_KEYS)
    clean, conflicts = [], []
    i = 0
    while i < len(passthrough):
        a = passthrough[i]
        if not a.startswith("--"):
            clean.append(a)
            i += 1
            continue
        flag = a[2:].split("=", 1)[0]
        cfg_key = _FLAG_TO_CFG.get(flag, flag)
        if cfg_key not in s:
            clean.append(a)
            i += 1
            continue
        fv = frozen[cfg_key]
        if isinstance(fv, bool):
            sets_true = flag == cfg_key  # --attn_res, --fone, etc.
            sets_false = flag in _FLAG_TO_CFG  # --no_attn_res
            if (sets_true and not fv) or (sets_false and fv):
                conflicts.append(f"{a} (frozen {cfg_key}={fv})")
            else:
                clean.append(a)
            i += 1
        else:
            if "=" in a:
                v = a.split("=", 1)[1]
                i += 1
            elif i + 1 < len(passthrough):
                v = passthrough[i + 1]
                i += 2
            else:
                conflicts.append(f"{a} (no value)")
                continue
            if int(v) != fv:
                conflicts.append(f"{a} {v} (frozen {fv})")
    return clean, conflicts


def _run_point(step_args, forced):
    """One 0830v1 budget point. run_ddp.sh already scores the checkpoint on success,
    so this only has to launch it and record the row. --name is required; --mix defaults
    to train.py's default; everything else passes through to train.py.

    Ladder mixes (mix_scale_*) carry a frozen run config in
    data/mix_scale_run_config.json: run point sets the env + flags from it and refuses
    a disagreeing CLI flag. The six points must differ only in D."""
    name, mix, hypothesis, passthrough = None, None, None, []
    i = 0
    while i < len(step_args):
        a = step_args[i]
        if a == "--name" and i + 1 < len(step_args):
            name, i = step_args[i + 1], i + 2
        elif a.startswith("--name="):
            name, i = a.split("=", 1)[1], i + 1
        elif a == "--mix" and i + 1 < len(step_args):
            mix, i = step_args[i + 1], i + 2
        elif a.startswith("--mix="):
            mix, i = a.split("=", 1)[1], i + 1
        elif a == "--hypothesis" and i + 1 < len(step_args):
            hypothesis, i = step_args[i + 1], i + 2
        else:
            passthrough.append(a)
            i += 1
    if not name:
        print("run point: --name <n> is required")
        return 2
    mix = mix or cfg_default("mix")
    env = None
    frozen_args = []
    if _is_ladder_mix(mix):
        fpath = os.path.join(ROOT, "data", "mix_scale_run_config.json")
        if not os.path.exists(fpath):
            print(f"run point: {fpath} missing -- the ladder recipe is not optional")
            return 2
        frozen = json.load(open(fpath, encoding="utf-8"))
        passthrough, conflicts = _strip_frozen(passthrough, frozen)
        if conflicts:
            print(f"run point: refusing -- frozen config disagrees: {'; '.join(conflicts)}")
            print("  edit data/mix_scale_run_config.json to change the ladder recipe (reopens the ladder)")
            return 2
        cards = [c.strip() for c in frozen["cards"].split(",") if c.strip()]
        world = frozen.get("world", len(cards))
        # `world` is the recipe (card count moves the effective batch and the gradient
        # noise); `cards` is only which H20s. They used to be one string with NGPU split
        # out of it, so dropping a card to dodge a busy one silently changed the recipe.
        if len(cards) != world:
            print(f"run point: refusing -- cards={frozen['cards']} is {len(cards)} cards, "
                  f"but the recipe is world={world}. Card COUNT is the ladder; card IDENTITY "
                  f"is not. Reallocate, do not shrink.")
            return 2
        # 90 s: a point costs ~10 min of 7 cards, so confirming for 90 s is free, and it
        # covers the 55 s step gap that made a busy eval_all.sh look idle.
        busy = _busy_cards(cards, settle=90)
        if busy:
            print(f"run point: refusing -- card(s) {', '.join(busy)} are in use. DDP is "
                  f"synchronous, so one contended rank slows all {world}: a launch here "
                  f"forges a regression rather than measuring one.")
            return 2
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=frozen["cards"], NGPU=str(world))
        # _CFG_TO_FLAG, not f"--{k}": Cfg.d's flag is --dim, because "--d" is ambiguous
        # inside torchrun's own parser and run_ddp.sh's args pass through it.
        # Only the renames, not the negations: reversing no_attn_res->attn_res would emit
        # "--no_attn_res 1", the opposite of what it says. Bools never reach here today,
        # which is why an inverted map would have been silent.
        _cfg_to_flag = {"d": "dim"}
        frozen_args = [v for k in _FROZEN_KEYS if not isinstance(frozen[k], bool)
                       for v in (f"--{_cfg_to_flag.get(k, k)}", str(frozen[k]))]
        print(
            f"run point: frozen config -> cards={frozen['cards']} "
            + " ".join(f"{k}={frozen[k]}" for k in _FROZEN_KEYS)
        )
    cmd = ["bash", os.path.join(ROOT, "run_ddp.sh"), "--mix", mix, "--name", name, *frozen_args, *passthrough]
    _exp("start", name=name, cmd=" ".join(cmd),
         hypothesis=hypothesis or f"0830v1 budget point, mix {os.path.basename(mix)}{forced}")
    _task_open_run(name, hypothesis)
    rec = os.path.join(ROOT, "runs", "score_matrix.jsonl")
    ckpt = f"ckpt_{name}.pt"

    def _ckpt_record():
        if not os.path.exists(rec):
            return None
        for line in open(rec, encoding="utf-8"):
            if f'"ckpt": "{ckpt}"' in line:
                return line
        return None

    before = _ckpt_record()
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    # A rerun of the same ckpt must not pass on the FIRST run's record: the line has to
    # have changed, not merely be present (score_matrix --json replaces same-ckpt lines,
    # and only on success -- a failed rescore leaves the stale line in place).
    after = _ckpt_record()
    scored = r.returncode == 0 and after is not None and after != before
    _exp("done", name=name,
         result=f"exit {r.returncode}; {ckpt} scored in score_matrix" if scored else f"exit {r.returncode}",
         finding="score_matrix record is the result; the fit interprets" if scored else "run failed before scoring",
         decision="proceed to next point" if r.returncode == 0 else "investigate before next point",
         status="ok" if r.returncode == 0 else "fail")
    if r.returncode == 0 and scored:
        _task_close_run(name, f"ckpt_{name}.pt; score_matrix record")
        val = _val_nll(name)
        _board_event("point_landed", f"{name} scored: val {val:.3f}" if val else f"{name} scored")
    elif r.returncode != 0:
        _board_event("point_failed", f"{name} exited {r.returncode}")
    _refresh_board()
    return r.returncode


def _run_ladder(step_args, forced):
    """All six budget points, sequential, resumable. A point with a
    score-matrix record is skipped; a failed point stops the ladder.
    Each point runs through _run_point, which enforces the frozen config.
    The gate re-fires before every point: a red at hour two banks the points
    already done and stops, rather than launching the next point blind."""
    rec = os.path.join(ROOT, "runs", "score_matrix.jsonl")
    done = set()
    if os.path.exists(rec):
        for line in open(rec, encoding="utf-8"):
            m = re.search(r'"ckpt": "ckpt_(.+?)\.pt"', line)
            if m:
                done.add(m.group(1))
    for name, mix in LADDER:
        if name in done:
            print(f"ladder: {name} already scored, skipping")
            continue
        if _gate(bool(forced)) and not forced:
            print(f"ladder: harness red, stopping with {len(done)} point(s) banked")
            _board_event("check_red", f"ladder stopped at {name}: harness red, {len(done)} point(s) banked")
            _refresh_board()
            return 1
        print(f"ladder: starting {name} ({mix})")
        rc = _run_point(["--name", name, "--mix", mix], forced)
        if rc != 0:
            print(f"ladder: {name} failed (exit {rc}), stopping")
            _board_event("ladder_stopped", f"{name} failed (exit {rc}), {len(done)} point(s) banked")
            _refresh_board()
            return rc
    print("ladder: all six points complete")
    _board_event("ladder_complete", "all six points scored")
    _refresh_board()
    return 0


def _busy_cards(cards, settle=0):
    """Which of `cards` has a compute process, watched over `settle` seconds and unioned.

    One reading is not enough: a card is owned by the script still running, not by the row
    nvidia-smi prints. 2026-08-30 eval_all.sh showed 0-5 at 0 MiB for 55 s between two of
    its shard steps and then took all seven cards back. A launch inside that gap would have
    contended with it for an hour. Watching across a window turns a step gap into a
    sighting; nothing else here can see another container's process tree.
    # ponytail: a window only catches gaps shorter than it. The real fix is a claim file the
    # holder writes and a trap removes, so ownership is declared rather than inferred --
    # worth it once two sessions launch unattended.
    """
    seen, deadline = set(), time.time() + settle
    while True:
        seen |= set(_busy_once(cards))
        if time.time() >= deadline or len(seen) == len(cards):
            return [c for c in cards if c in seen]
        time.sleep(min(5, max(1, settle / 12)))


def _busy_once(cards):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        uuids = {l.strip() for l in out.stdout.splitlines() if l.strip()}
        idx = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []  # no nvidia-smi: a dev box, nothing to protect
    by_uuid = {}
    for line in idx.stdout.splitlines():
        if "," in line:
            i, u = line.split(",", 1)
            by_uuid[u.strip()] = i.strip()
    return [c for c in cards if any(by_uuid.get(u) == c for u in uuids)]


def run_dispatch(rest):
    """`harness run <step>` -- the only verb that executes. Thin dispatch, no new logic:
    every step refuses while check is red (--force records the reds in the exp row),
    writes its own exp.py start/done, and scores what it produced."""
    if not rest or rest[0] not in STEPS:
        print(f"usage: harness.py run <{'|'.join(STEPS)}> [step flags] [--force]")
        return 2
    step, step_args = rest[0], list(rest[1:])
    force = "--force" in step_args
    if force:
        step_args.remove("--force")
    reds = _gate(force, step)
    if reds and not force:
        return 1
    forced = f" [FORCED, red: {', '.join(n for n, _ in reds)}]" if reds else ""
    if step == "pretokenize":
        return _run_pretokenize(step_args, forced)
    if step == "point":
        return _run_point(step_args, forced)
    if step == "ladder":
        return _run_ladder(step_args, forced)
    if step == "fetch":
        if not _preflight_fetch(step_args):
            return 2
        return _run_pipeline_step("fetch", "fetch_corpus.py", step_args, forced)
    if step == "clean":
        return _run_pipeline_step("clean", "clean_corpus.py", step_args, forced)
    if step == "score":
        return _run_pipeline_step("score", "score_corpus.py", step_args, forced,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
    if step == "dedup":
        return _run_pipeline_step("dedup", "dedup_corpus.py", step_args, forced)
    return 2


# --------------------------------------------------------------------------- sync


def _verify_merge(pod_ids, repo_only_ids, merged_ids, label):
    """Assert the merge lost nothing. Returns an error string or None.

    Identity may repeat: the ledgers are event logs, so a start row and a done row
    deliberately share (name, started) and readers fold last-event-wins. The old
    version refused on a repeated identity, which contradicted the semantics it was
    meant to protect and blocked every sync once done started appending (2026-08-31,
    5 duplicate identities on the pod). What must hold is that no row disappears."""
    missing = [i for i in pod_ids if i not in set(merged_ids)]
    if missing:
        return f"{label}: {len(missing)} pod row(s) lost in merge"
    lost_repo = [i for i in repo_only_ids if i not in set(merged_ids)]
    if lost_repo:
        return f"{label}: {len(lost_repo)} repo-only row(s) lost in merge"
    if len(merged_ids) < len(pod_ids):
        return f"{label}: merged {len(merged_ids)} rows, fewer than the pod's {len(pod_ids)}"
    return None


def _merge_jsonl(pod_lines, repo_lines, identity_fn, label):
    """Union of pod and repo rows, byte-identical lines collapsed. Returns (text, error).

    Union, not identity-keyed replacement: these are event logs. The pod is the
    producer, so its rows come first and a repo row that repeats an identity is an
    additional EVENT for that run, not a competing copy -- keep both and let readers
    fold. Only a line that will not parse is refused: a truncated row would otherwise
    ride into the committed ledger and break every reader."""
    pod_rows, repo_rows = [], []
    for src, lines, out in (("pod", pod_lines, pod_rows), ("repo", repo_lines, repo_rows)):
        for n, l in enumerate(lines, 1):
            l = l.strip()
            if not l:
                continue
            try:
                out.append((json.loads(l), l))
            except json.JSONDecodeError as e:
                return None, f"{label}: {src} line {n} will not parse ({str(e)[:50]}); fix it before syncing"
    # Byte-identical lines collapse wherever they appear, including within one side:
    # a union merge of two branches that both appended the same row leaves two copies
    # (b0's retro row, 2026-08-31), and syncing that would carry the duplicate onward.
    seen, merged = set(), []
    for _, l in pod_rows + repo_rows:
        if l not in seen:
            seen.add(l)
            merged.append(l)
    pod_line_set = {l for _, l in pod_rows}
    err = _verify_merge(
        [identity_fn(r) for r, _ in pod_rows],
        [identity_fn(r) for r, l in repo_rows if l not in pod_line_set],
        [identity_fn(json.loads(l)) for l in merged],
        label,
    )
    if err:
        return None, err
    return "\n".join(merged) + "\n", None


def cmd_sync(rest):
    """`harness sync` -- pull runs/experiments.jsonl and runs/score_matrix.jsonl from
    the pod, merging by producer identity. The pod is the producer; repo-only rows
    (degen_t08, fetch_*) are kept. A merge that loses a row refuses."""
    import base64

    pod = os.path.expanduser("~/bin/pod")
    if not os.path.exists(pod):
        print("~/bin/pod not found -- sync runs from a dev box with pod access")
        return 2
    syncs = [
        ("runs/experiments.jsonl", lambda r: (r.get("name", ""), r.get("started", "")), "experiments"),
        ("runs/score_matrix.jsonl", lambda r: r.get("ckpt", ""), "score_matrix"),
    ]
    for relpath, idfn, label in syncs:
        r = subprocess.run(
            [pod, f"base64 /work/aupai/{relpath}"], capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            print(f"  {label}: pod read failed: {r.stderr.strip()[:200]}")
            return 1
        pod_text = base64.b64decode(r.stdout).decode("utf-8")
        repo_path = os.path.join(ROOT, relpath)
        repo_text = open(repo_path, encoding="utf-8").read() if os.path.exists(repo_path) else ""
        merged, err = _merge_jsonl(pod_text.splitlines(), repo_text.splitlines(), idfn, label)
        if err:
            print(f"  REFUSING {label}: {err}")
            return 1
        with open(repo_path, "w", encoding="utf-8") as f:
            f.write(merged)
        n_pod = len([l for l in pod_text.splitlines() if l.strip()])
        n_repo = len([l for l in repo_text.splitlines() if l.strip()])
        n_merged = len([l for l in merged.splitlines() if l.strip()])
        print(f"  {label}: {n_pod} pod + {n_merged - n_pod} repo-only = {n_merged} rows (was {n_repo})")
    return 0


# --------------------------------------------------------------------------- clean


def _expand_cards(spec):
    """Card spec to a sorted index list. Accepts "0,1,2", "0-7", and both mixed.

    The grant file writes ranges ("block_cards": "0-7") and the ladder config writes
    lists, so a reader that splits on commas turns eight cards into one -- and NGPU is
    len(cards.split(",")), which would launch a one-rank job under an eight-card grant."""
    out = set()
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _csv(cards):
    """Card list -> the comma string CUDA_VISIBLE_DEVICES and NGPU are built from.
    _expand_cards returns ints; every consumer here wants "0,1,2,3"."""
    return ",".join(str(c) for c in cards)


def _grant_cards(root=None, raise_on_false=True):
    """The controller's card grant from runs/card_assignment.json, or (None, why).

    THE GRANT FILE IS THE AUTHORITY, and before 2026-09-03 nothing read it for this
    (b0). launch_gate.gate_cards read card_assignment.json to decide GO/NO-GO, while
    _allocation_cards below read mix_scale_run_config.json to decide which cards the job
    actually gets -- so the gate's source and the actor's source were different files.
    The grant was narrowed to 0-3 because cards 4 and 7 hold the USER'S OWN work in other
    containers; the gate went GO on 0-3 and the launcher would still have handed the job
    cards 0-6, card 4 included. Correcting the grant file had no effect on the launch
    because the launch path never looked at it.

    Same shape as recipe_provenance's, one file over: a record can be right, current, and
    read by a gate, and still not reach the code that acts (gate_failure_shapes.md §142).

    Returns (cards, "") when the block is granted, (None, why) when no grant is READABLE,
    and RAISES SystemExit when the file says launch_block_granted is false.

    "I SAY NO" AND "I HAVE NOT SPOKEN" ARE DIFFERENT ANSWERS (6e's ruling, b0 2026-09-03).
    Both used to return None and fall back to the ladder config, which is the same
    collapse §140 is about: a value domain with no slot for "not answered" folds it into a
    normal answer. On the pod the two are worlds apart -- an explicit false is the
    controller saying the block is NOT available, and a missing file only means nothing was
    synced. Falling back on the explicit false is how the pod came to allocate cards 0-6
    with a false grant sitting right there, the user's card 4 included.

    The refusal is deliberately NARROW: only an explicit false raises. A missing or
    unreadable file still falls back with a note, because a blanket refusal would stop
    every training launch on a pod that has never received this file -- other sessions'
    included, and narrowing a rule for my own leg must not sweep them in (6e).

    AND IT RAISES FOR A LAUNCH, NOT FOR A READ (de, 2026-09-04, on 6e's report). SystemExit
    is a BaseException, so `except Exception` in run_checks:10136 does not catch it: with
    launch_block_granted false, MEASURED, `harness check` printed ZERO of its check lines and
    exited 1 on this message. One check's refusal took the whole instrument dark, and every
    other red in the tree with it -- the permanent-red rule in AGENTS, arrived at from the
    other direction. 6e hit the consequence at commit time and kept the field `true` with the
    real decision in the prose, which is the file lying because the instrument made honesty
    uncommittable.

    So `raise_on_false` is the caller's question. A LAUNCH asks with it set: an explicit false
    must stop a job from reaching a card, which is the 0-6 allocation this refusal was written
    for. A CHECK reads with it clear and gets (None, why) naming the false, because an
    ungranted box is the legitimate state of an idle box and a check may only report it.
    """
    root = ROOT if root is None else root
    p = os.path.join(root, "runs", "card_assignment.json")
    if not os.path.isfile(p):
        return None, "no runs/card_assignment.json"
    try:
        with open(p, encoding="utf-8") as fh:
            a = json.load(fh)
    except (OSError, ValueError) as e:
        return None, f"card_assignment.json unreadable: {e}"
    if "launch_block_granted" in a and not a["launch_block_granted"]:
        why = (f"{os.path.relpath(p, root)} says launch_block_granted is false -- the "
               f"controller has NOT granted a card block here "
               f"(granted_by {a.get('granted_by', 'unknown')}, {a.get('granted', 'undated')})")
        if not raise_on_false:
            return None, why
        raise SystemExit(
            f"REFUSING: {why}. That is a decision, not a missing value, so it is not "
            f"something to fall back from: falling back on it is what put a launch on cards "
            f"0-6 with this file reading false. "
            f"Get a grant, or push the current one -- runs/card_assignment.json is in "
            f"pod_drift's SCOPE as of 2026-09-03 and reaches the pod with `pod_push.sh --all`.")
    if not a.get("launch_block_granted"):
        return None, "card_assignment.json has no launch_block_granted key"
    cards = _expand_cards(a.get("block_cards", ""))
    if not cards:
        return None, "card_assignment.json grants the block but names no block_cards"
    return cards, ""


def _allocation_cards(training, root=None, raise_on_false=True):
    """Card set from the controller's allocation file, never from the caller.

    Training jobs get the block: runs/card_assignment.json's block_cards when that file
    grants one, else mix_scale_run_config.json's cards. Non-training jobs get the lane
    (the cards not in the block).

    TWO FILES, ONE OF THEM AUTHORITATIVE, AND A DISAGREEMENT IS REPORTED (b0
    2026-09-03). mix_scale_run_config.json is the ladder's FROZEN RUN CONFIG -- its
    `cards` and `world` record what the six mix_scale_* budget points ran on, and its own
    _comment says a change to any value reopens the ladder. `cards` and `world` are in
    neither _FROZEN_KEYS nor _CODE_FROZEN_KEYS, so `cards` is operationally editable and
    has been edited before (1-7 -> 0-6, 2026-08-30). `world` is NOT: the six points ran at
    world 7 and editing that field to describe a run that has not happened would falsify
    the record of runs that did (6e's ruling). So the grant file carries today's
    allocation and the ladder config keeps its history.

    A CONFLICT PRINTS RATHER THAN RESOLVING. Silently preferring either file is how this
    defect worked in the first place: one file said 0-3, another said 0-6, and nothing
    said they disagreed. World size follows the cards -- cmd_launch derives NGPU from
    len(cards) -- so a 4-card grant cannot produce a 7-rank launch through this path.
    """
    root = ROOT if root is None else root
    config_path = os.path.join(root, "data", "mix_scale_run_config.json")
    ladder = []
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                ladder = _expand_cards(json.load(fh).get("cards", ""))
        except (OSError, ValueError):
            ladder = []
    granted, why = _grant_cards(root, raise_on_false=raise_on_false)
    if granted is None:
        # Fall back to the old source, SAYING SO. A silent fallback here would look
        # identical to a grant that happens to match, and the message is the only thing
        # that tells a reader which file decided.
        if ladder:
            print(f"note   cards {_csv(ladder)} from data/mix_scale_run_config.json "
                  f"({why}); the grant file is the authority when it has one",
                  file=sys.stderr)
            block = ladder
        else:
            return os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    else:
        if ladder and set(ladder) != set(granted):
            # SAY IT HERE, REFUSE AT THE LAUNCH. This function is not the place to exit from:
            # `harness check` calls it twice (check_allocation_reads_the_grant, 8678-8679) and so
            # does free-card (12535), so raising here took the WHOLE check run down -- measured
            # 2026-09-04, `harness check` printed this message and zero of 58 checks. A guard that
            # disables the instrument that would have caught it is worse than the warning it
            # replaced.
            #
            # The refusal lives in cmd_launch, where the thing being refused is a launch and the
            # blast radius is one job. The message is identical; only the exit moved.
            print(f"WARNING: card sources DISAGREE -- runs/card_assignment.json grants "
                  f"{_csv(granted)}, data/mix_scale_run_config.json says "
                  f"{_csv(ladder)}. Using the grant. The ladder config's `cards` "
                  f"records what the six mix_scale_* points ran on and is not today's "
                  f"allocation; if today's block really changed, narrow it there too.",
                  file=sys.stderr)
        block = granted
    if training:
        return _csv(block)
    # THE LANE IS NOT THE COMPLEMENT OF THE BLOCK when the grant says there is no lane
    # (b0, 2026-09-03). With block_cards 0-3, "everything else" is 4,5,6,7 -- and cards 4
    # and 7 hold the USER'S OWN work in other containers while 5 and 6 are other sessions'.
    # Computing the lane by set subtraction invents a lane out of exactly the cards the
    # grant was narrowed to protect. card_assignment.json states lane_card: null, which is
    # a decision ("no lane under a 4-card block; small jobs queue on 5/6 by arrangement"),
    # not a missing value -- so an explicit null returns no lane and the caller refuses,
    # rather than being handed somebody else's card.
    if granted is not None:
        try:
            with open(os.path.join(root, "runs", "card_assignment.json"), encoding="utf-8") as fh:
                _a = json.load(fh)
        except (OSError, ValueError):
            _a = {}
        if "lane_card" in _a and _a["lane_card"] is None:
            print("note   the grant states lane_card: null -- no lane card under this "
                  "block, so a non-training GPU job has nowhere to land here. Complementing "
                  "the block would hand it cards outside the grant.", file=sys.stderr)
            return ""
        if _a.get("lane_card") is not None:
            return _csv(_expand_cards(_a["lane_card"]))
    lane = sorted(set(range(8)) - set(block))
    return _csv(lane) if lane else _csv(block)


def _lane_occupant(card):
    """PID of a process using the given GPU card, or None. Uses nvidia-smi on
    the pod; returns None on machines without GPUs."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
             f"--id={card}"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().split("\n")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def cmd_free_card(argv):
    """`harness free-card [--wait N] [--settle N]` -- print a lane card measured free.

    Exists for the scoring that fires without a person: run_ddp.sh scored inside the
    training shell, so CUDA_VISIBLE_DEVICES was still the seven-card block and the
    scorer took whatever card 0 happened to be doing. On 2026-09-01 that was another
    process holding 14.37 GiB, and the scorer died asking for 96 MiB. Nothing read a
    card; the card number came from the environment (fb's ruling: the free judgement
    must come from a measurement at that moment, never from a default card number).

    Prints one index and exits 0, or waits for one and exits 1 if none frees --
    queue, never spill into the block.
    """
    ap = argparse.ArgumentParser(prog="harness free-card")
    ap.add_argument("--wait", type=int, default=0, help="seconds to wait for a card to free")
    ap.add_argument("--settle", type=int, default=8, help="window over which a card must stay idle")
    a = ap.parse_args(argv)
    lane = [c.strip() for c in _allocation_cards(False).split(",") if c.strip()]
    if not lane:
        print("no lane card in the allocation", file=sys.stderr)
        return 1
    deadline = time.time() + a.wait
    while True:
        free = [c for c in lane if c not in _busy_cards(lane, settle=a.settle)]
        if free:
            print(free[0])
            return 0
        if time.time() >= deadline:
            held = {c: _lane_occupant(c) for c in lane}
            print(f"no free lane card: {held}. Queue, do not spill into the block.",
                  file=sys.stderr)
            return 1
        time.sleep(min(30, max(5, a.wait / 20)))


def _wait_for_startup(log_path, timeout):
    """Poll the log for the training startup gate lines.

    train.py prints two runlog lines at startup:
      params ... | device cuda | world 7 | fa True | fp8 True
      cfg batch ... doc_mask True attn_res ...
    Both must read True. fa False kills immediately (the incident: a log printed
    fa False beside doc_mask True and nothing objected)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.isfile(log_path):
            text = open(log_path, errors="ignore").read()
            if "fa False" in text:
                return False, "fa False in log"
            if "fa True" in text and "doc_mask True" in text:
                return True, "gate passed"
        time.sleep(2)
    return False, "timeout"


def _arm_monitor(name, pid, log_path, output_path=None):
    """Start a background monitor that marks the exp row when the process dies
    or the log (and declared output) goes silent for 10 minutes.

    The monitor never writes a row for a run that already reached a terminal state:
    t56_profile closed ok at 13:34 and the monitor appended 'log silent' fail at
    13:47, because the log stops growing precisely when a run finishes. A fail row
    after a real result is worse than no row -- it inverts the verdict, and
    score_matrix_present read that stale fail's predecessor as an unscored ok."""
    output_repr = repr(output_path) if output_path else "None"
    rc_repr = repr(os.path.join(ROOT, "runs", f"{name}.rc"))
    monitor_code = f'''
import json, os, subprocess, sys, time
pid, log, name, exp_py = {pid}, "{log_path}", "{name}", "{os.path.join(HERE, "exp.py")}"
output = {output_repr}
rc_file = {rc_repr}
exp_log = os.path.join(os.path.dirname(exp_py), "..", "runs", "experiments.jsonl")
silent_limit = 600
last_size, last_grow = 0, time.time()

def settled():
    """True once this run has a terminal row: someone closed it, or harness kill did."""
    try:
        with open(exp_log, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("name") == name and r.get("status") in ("ok", "fail", "retracted"):
                    return True
    except (OSError, ValueError):
        pass
    return False

while True:
    time.sleep(60)
    if settled():
        # A terminal row exists, so someone closed this run -- `exp.py done` by hand, or
        # `harness kill`. Release here too: the row is the signal that the job is over, and
        # the exit below is the ONLY other place that releases. Without this, every run closed
        # by a human leaves its claim behind, which is the majority of runs. Release is
        # idempotent -- "no claim for X" on an already-released name is not an error.
        subprocess.run([sys.executable, os.path.join(os.path.dirname(exp_py), "card_claim.py"),
                        "release", "--name", name], capture_output=True)
        break  # a result exists; anything the monitor adds now can only contradict it
    # os.kill(pid, 0) accepts a ZOMBIE: an exited child nobody reaped keeps its pid,
    # so the monitor waits forever on a finished run (harness.py:6418 documents the
    # same trap, tilerl re-hit it). /proc state Z is dead.
    alive = True
    try:
        os.kill(pid, 0)
        try:
            with open(f"/proc/{pid}/stat") as sf:
                alive = sf.read().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            pass  # no procfs (macOS): fall back to the signal probe
    except OSError:
        alive = False
    if not alive:
        # The exit code, not the disappearance. A process that is gone has either
        # finished or been killed, and the monitor cannot tell those apart by watching
        # a pid -- it used to write status=ok for both. The 22B milestone was killed at
        # 04:22 to yield the lane and its row reads `ok | process exited`, identical to
        # a completed score; nothing on the board could see that the 22B reading did
        # not exist (de, 2026-09-01). cmd_launch's wrapper writes runs/<name>.rc after
        # the child returns, so the verdict is an artifact.
        #
        # No .rc means the wrapper itself died -- SIGKILL to the group, the machine
        # went down -- which is the killed case, not the finished one. Fail closed:
        # a run whose fate is unknown is not a success. The row says which of the two
        # it was, because "no rc" and "rc 137" call for different responses.
        rc, why = None, ""
        try:
            with open(rc_file, encoding="utf-8") as rf:
                rc = int(rf.read().strip())
        except (OSError, ValueError):
            why = "no exit code recorded: the wrapper died with the job (killed group, or the box went away)"
        if rc == 0:
            status, result, finding = "ok", "exit 0", "monitor: process exited cleanly"
        elif rc is None:
            status, result, finding = "fail", "vanished", "monitor: " + why
        else:
            sig = " (signal %d)" % (rc - 128) if rc > 128 else ""
            status, result = "fail", "exit %d%s" % (rc, sig)
            finding = "monitor: process exited %d%s" % (rc, sig)
        subprocess.run([sys.executable, exp_py, "done", "--name", name,
            "--result", result, "--finding", finding,
            "--decision", "check the log", "--status", status], capture_output=True)
        # RELEASE THE CARDS HERE, beside the row that records the death. cmd_launch cannot:
        # it returns while the job is still running, so releasing there would free a card
        # under a live job. The monitor is the only thing that outlives the job and sees it
        # end, and a claim nobody releases is the ORPHAN-SHELL state one step on -- a card
        # that reads held forever, which is what made `card_claim.py status` report all
        # eight pod cards as orphans (de-30/de-34).
        subprocess.run([sys.executable, os.path.join(os.path.dirname(exp_py), "card_claim.py"),
                        "release", "--name", name], capture_output=True)
        break
    grew = False
    for p in ([log] + ([output] if output else [])):
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            if sz > last_size:
                last_size, last_grow = sz, time.time()
                grew = True
                break
    if not grew and time.time() - last_grow > silent_limit:
        # The process is ALIVE -- the liveness probe above would have broken out of the
        # loop otherwise. So silence is not death: score_matrix --profile milestone is
        # silent for long stretches inside a generative eval, and this branch used to
        # write status=fail on two runs that then produced complete score records and
        # readouts 22 and 54 minutes LATER (ms_..._15b_s1.pt.step16000,
        # ms_..._30b_s2.pt.step17500; de-8 D6). The fail row outlived the incident and
        # the ledger has disagreed with its own artifacts since.
        #
        # A monitor that cannot see the process must not overwrite the verdict the
        # process will produce itself. Warn into the run's own log -- the file the
        # operator is already tailing -- and say the observation is a PROXY: log bytes,
        # not process state. Appending to the log rather than the ledger is deliberate:
        # exp.py has no note verb, and inventing a row state for "suspicious" would put a
        # third value in a field every reader folds on.
        try:
            with open(log, "a", encoding="utf-8") as lf:
                lf.write(
                    f"\\n[monitor {{time.strftime('%H:%M:%S', time.gmtime())}}] no log growth in {{silent_limit}}s, "
                    f"but pid {{pid}} is ALIVE -- stalled_suspected, NOT failed. Liveness here is "
                    f"inferred from LOG BYTES, not from the process; a generative eval is silent "
                    f"by construction. Leaving the row open: the run decides its own verdict.\\n"
                )
        except OSError:
            pass
        last_grow = time.time()  # re-arm: one note per silent window, not one per minute
'''
    monitor_proc = subprocess.Popen(
        [sys.executable, "-c", monitor_code],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return monitor_proc.pid


#: Effective cache-load rate, GiB/s, as the STARTUP sees it: 149 GiB of mix reached the
#: first step in 6m26s on 2026-08-31 (7 ranks, shared page cache) -> 0.39 GiB/s aggregate.
#: Not the 1.5-1.6 GiB/s warm single-stream figure -- that is one reader on an idle disk,
#: and using it derives a 198 s gate for a startup that actually took 386 s. Contended
#: (two tokenizers running) the same read collapsed to 0.07 GiB/s, hence the x2.
_CACHE_READ_GIBPS = 0.39
_GATE_FLOOR_S = 600


def _derive_gate_timeout(cmd, cache_dir=None):
    """Startup-gate seconds derived from the mix the command names, or None.

    train.py:1396 loads every domain's FULL token cache on every rank before the
    first step. On 2026-08-31 that was 149 GiB and the first step line came 6m26s
    after launch -- the 120 s default would have killed a healthy run. The gate is
    a property of the mix, not something an operator should have to measure again:
    total cache bytes / _CACHE_READ_GIBPS, doubled for contention, floor 600 s.
    """
    mix = None
    for i, c in enumerate(cmd):
        if c == "--mix" and i + 1 < len(cmd):
            mix = cmd[i + 1]
            break
    if not mix:
        return None, None
    doms, err = read_mix(mix if os.path.isabs(mix) else os.path.join(ROOT, mix))
    if err:
        return None, f"mix unreadable ({err})"
    cache_dir = cache_dir or os.path.dirname("/data00/pretrain_1b_tokens.pt")
    total = 0
    missing = []
    for d in doms:
        p = os.path.join(cache_dir, f"tokens_{d}.pt")
        if os.path.exists(p):
            total += os.path.getsize(p)
        else:
            missing.append(d)
    if not total:
        return None, f"no token caches on disk for {len(doms)} domain(s)"
    gib = total / 2**30
    secs = max(_GATE_FLOOR_S, int(gib / _CACHE_READ_GIBPS * 2))
    note = f"{gib:.0f} GiB of cache / {_CACHE_READ_GIBPS} GiB/s x2 -> {secs}s"
    if missing:
        note += f" (not yet tokenized: {', '.join(missing[:3])})"
    return secs, note


def _job_pids_for(pid):
    """[pid] of the python/torchrun processes under pid, via card_claim's ppid walk.

    Imported rather than reimplemented: card_claim owns the "is this a shell" and "is this the
    job" predicates, and a second copy is a second thing to keep right (de-34's argv0-vs-substring
    distinction is exactly the kind that drifts).
    """
    try:
        sys.path.insert(0, HERE)
        import card_claim
    except ImportError:
        return []
    try:
        return [p for p, _a in card_claim._job_descendants(pid)]
    except Exception:  # noqa: BLE001 -- a claim helper must never take the launch down
        return []


def _acquire_cards(name, cards, pid, note):
    """(ok, message). Claim `cards` for `name` on behalf of `pid`."""
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "card_claim.py"), "acquire",
         "--name", name, "--cards", cards, "--pid", str(pid), "--note", note],
        capture_output=True, text=True,
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def _release_cards(name):
    """Release `name`'s claim. Never raises -- a stuck claim is reported by status, and a
    release that crashes the caller would leave the job unsupervised."""
    if not name:
        return
    try:
        subprocess.run([sys.executable, os.path.join(HERE, "card_claim.py"), "release",
                        "--name", name], capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        pass


def cmd_launch(rest):
    """`harness launch <name> [--training] [--hypothesis "..."] -- <cmd>`

    Unified launcher: exp row first, setsid nohup with runs/<name>.log,
    card allocation from the controller's config (never the caller), startup
    gate for training jobs (kill + fail row if the fa/doc_mask line never
    appears), monitor armed on process-gone and log-silent.

    Usage:
      harness launch p02_s0 --training --hypothesis "0.2b point" -- ./run_ddp.sh --mix data/mix_scale_0.2b.json --name p02_s0
      harness launch eval_ckpt -- eval/math_hard.py --ckpt ckpt_k9.pt
    """
    ap = argparse.ArgumentParser(prog="harness launch")
    ap.add_argument("name", help="run name (also the log and exp row name)")
    ap.add_argument("--training", action="store_true", help="training job (block cards, startup gate)")
    ap.add_argument("--hypothesis", default="", help="what this run is meant to test")
    ap.add_argument("--gate-timeout", type=int, default=None,
                    help="startup gate timeout in seconds (default: 120, 300 for --resume)")
    ap.add_argument("--output", default=None, help="declared output path for non-training jobs (monitored for growth alongside the log)")
    ap.add_argument("--no-gpu", action="store_true", help="corpus/CPU job: no card assigned, no lane check")
    ap.add_argument("--auto-resume", type=int, default=0, metavar="N",
                    help="on a non-zero exit, relaunch with --resume <latest step ckpt>, up to N times "
                         "(blocks: detach the whole command with setsid nohup)")
    # Manual split on -- : argparse REMAINDER greedily captures our own --training flag.
    if "--" not in rest:
        ap.error("no command given after --")
    idx = rest.index("--")
    args = ap.parse_args(rest[:idx])
    cmd = rest[idx + 1:]
    if not cmd:
        ap.error("no command given after --")
    # Resume loads a checkpoint; 120s is too short for a 959MB file (tilerl-4c, t38).
    # For a training job the gate comes from the mix's cache size instead: 120 s would
    # have killed tonight's healthy 15B run, whose first step came 6m26s in.
    gate_note = None
    if args.gate_timeout is None:
        if args.training:
            derived, gate_note = _derive_gate_timeout(cmd)
            if derived is None and gate_note and "no token caches" in gate_note:
                # REFUSE rather than fall back. The fallback was backwards: the emptier
                # the cache, the shorter the deadline, while the work grows from a load
                # into a single-process retokenize of hours. A 120s gate then kills a
                # healthy job at its most expensive moment (b0, 2026-08-31).
                print(f"REFUSING: {args.name} -- {gate_note}. A training launch with cold "
                      f"caches cannot be gated on time: tokenizing is hours, and the "
                      f"derived gate would be shorter than a warm load. Run "
                      f"`harness pretokenize --workers 8` first, or pass an explicit "
                      f"--gate-timeout if you accept the risk.", file=sys.stderr)
                return 2
            args.gate_timeout = derived or (300 if "--resume" in cmd else 120)
            if derived is None and gate_note:
                gate_note = f"{gate_note}; falling back to {args.gate_timeout}s"
        else:
            args.gate_timeout = 300 if "--resume" in cmd else 120
    if args.training:
        print(f"startup gate: {args.gate_timeout}s" + (f" ({gate_note})" if gate_note else " (explicit)"))

    # Popen does not use a shell: a bare foo.py fails with Permission denied.
    # Prepend the interpreter when the command is a .py file in the repo.
    if cmd[0].endswith(".py") and os.path.isfile(os.path.join(ROOT, cmd[0])):
        cmd = [sys.executable] + cmd

    # Infer --no-gpu: corpus commands need no card
    _CORPUS_CMDS = ("fetch_corpus", "build_corpus", "count_cleaned", "clean_corpus")
    if not args.no_gpu and not args.training:
        if any(any(c in part for c in _CORPUS_CMDS) for part in cmd):
            args.no_gpu = True

    # 1. Allocation and the lane check FIRST: a refusal must write no ledger row.
    # Card allocation from the controller's config
    if args.no_gpu:
        cards = ""
    else:
        cards = _allocation_cards(args.training)

    # 1b. TWO ALLOCATION SOURCES THAT DISAGREE REFUSE THE LAUNCH (6e's ruling 2026-09-04).
    # _allocation_cards resolves the disagreement correctly -- the grant wins -- and says so on
    # stderr, which is where this stopped being enough: the params leg launched with
    # "card sources DISAGREE -- grants 0,1,2,3, says 0,1,2,3,4,5,6. Using the grant" in its own
    # launch output, and nobody read it until afterwards. The resolution was right and the
    # staleness stayed, so the next launcher faced the same ambiguity.
    #
    # REFUSED HERE, not inside _allocation_cards: that function is called by `harness check`
    # itself (check_allocation_reads_the_grant) and by free-card, and raising there took the whole
    # 58-check run down with it -- measured while writing this. The blast radius of a refusal
    # belongs at the launch, where the thing refused is one job.
    #
    # NOT A CHECK AGAINST THE FROZEN `world`: the grant is 4 cards and the ladder's world is 7,
    # and `world` is deliberately un-editable because the six points ran at 7 (the config's own
    # _comment). The rank-vs-card check below is the one that can be made -- two live quantities.
    if args.training and not args.no_gpu:
        _gp = os.path.join(ROOT, "runs", "card_assignment.json")
        _lp = os.path.join(ROOT, "data", "mix_scale_run_config.json")
        try:
            with open(_gp, encoding="utf-8") as _fh:
                _grant = json.load(_fh)
            with open(_lp, encoding="utf-8") as _fh:
                _ladder = _expand_cards(json.load(_fh).get("cards", ""))
        except (OSError, ValueError):
            _grant, _ladder = {}, []
        _granted = _expand_cards(_grant.get("block_cards", "")) if _grant.get("launch_block_granted") else []
        if _granted and _ladder and set(_granted) != set(_ladder):
            print(f"REFUSING: {args.name} -- two allocation sources disagree, so which cards a "
                  f"training job owns depends on which file the reader trusts.\n"
                  f"  runs/card_assignment.json grants {_csv(_granted)}  <- the authority\n"
                  f"  data/mix_scale_run_config.json says {_csv(_ladder)}  <- the ladder's record\n"
                  f"The ladder config's `cards` records what the six mix_scale_* points ran on, "
                  f"not today's allocation. If today's block is {_csv(_granted)}, narrow `cards` "
                  f"there too; the launch then agrees with both files. No ledger row written.",
                  file=sys.stderr)
            return 2

    # 2a-0. World size follows the CARDS, and a command that states its own rank count
    # must agree with them (6e's ruling, 2026-09-03). NGPU below is derived from
    # len(cards), so this path cannot produce a rank/card mismatch on its own -- but the
    # COMMAND can carry one: run_ab_speedrun.sh computes NGPU from its own $CARDS, and a
    # torchrun --nproc_per_node written into the command line is read by torchrun before
    # anything here sees it. Refuse rather than let 7 ranks start on 4 cards, which does
    # not fail cleanly: ranks 4-6 land on cards 0-2 a second time and OOM the ones that
    # were healthy.
    if args.training and cards:
        _ncards = len([c for c in cards.split(",") if c.strip()])
        _stated = None
        for _part in cmd:
            _m = re.match(r"--nproc_per_node=(\d+)$", _part) or re.match(r"NGPU=(\d+)$", _part)
            if _m:
                _stated = int(_m.group(1))
        if _stated is not None and _stated != _ncards:
            print(f"REFUSING: {args.name} -- the command states {_stated} ranks but the "
                  f"allocation is {_ncards} card(s) ({cards}). Card COUNT is the recipe: "
                  f"{_stated} ranks on {_ncards} cards double-books cards and OOMs the "
                  f"ranks that were healthy. No ledger row written.", file=sys.stderr)
            return 2

    # 2a. Lane-occupancy refusal: a non-training GPU job must not start while the
    # lane is occupied. Queue, never spill. Training jobs use the block, not the lane.
    if not args.training and not args.no_gpu and cards:
        # Which lane card, measured now -- not the first one in the list. A single
        # nvidia-smi reading misses a step gap, so this watches a window (_busy_cards);
        # the card number must come from that measurement, never from a default.
        lane = [c.strip() for c in cards.split(",") if c.strip()]
        busy = _busy_cards(lane, settle=8)
        free = [c for c in lane if c not in busy]
        lane_card = free[0] if free else lane[0]
        occupant = _lane_occupant(lane_card) if not free else None
        if occupant or not free:
            # No ledger row. The refusal happens BEFORE the start row is written, so a
            # second launch under a live run's name cannot close that run's row: on
            # 2026-08-31 l1_rerun_0831 read running/running/fail while pid 550586 was
            # alive and writing, because the row was written at step 1 and this refusal
            # ran at 2a (e1). A job that never starts leaves no trace in the ledger.
            print(f"REFUSED: {args.name} - lane GPU {lane_card} occupied by pid {occupant}. "
                  f"No ledger row written; the lane holds one job at a time.", file=sys.stderr)
            return 1
        cards = lane_card


    # 2. The start row, once the job is known to be runnable.
    launcher = f"--gate-timeout {args.gate_timeout}"
    if args.auto_resume:
        launcher += f" --auto-resume {args.auto_resume}"
    if args.training:
        launcher += " --training"
    subprocess.run(
        [sys.executable, os.path.join(HERE, "exp.py"),
         "start", "--name", args.name,
         "--cmd", " ".join(cmd),
         "--notes", f"launcher: harness launch {launcher}" + (f"; gate {gate_note}" if gate_note else ""),
         "--hypothesis", args.hypothesis],
        check=True,
    )

    # 2b. Training jobs: verify training-scope drift before launch.
    # A corpus-scope drift (e.g. fetch_corpus.py mid-push) must not stop a training launch.
    if args.training:
        drift_py = os.path.join(HERE, "pod_drift.py")
        if os.path.exists(drift_py):
            r = subprocess.run(
                [sys.executable, drift_py, "--check", "--scope", "training"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                detail = (r.stdout or r.stderr).strip().split("\n")[0][:150]
                subprocess.run(
                    [sys.executable, os.path.join(HERE, "exp.py"),
                     "done", "--name", args.name,
                     "--result", "refused: training-scope drift",
                     "--finding", detail,
                     "--decision", "push the drifted training file or wait for the push to finish",
                     "--status", "fail"],
                    capture_output=True,
                )
                print(f"REFUSED: {args.name} — training-scope drift: {detail}", file=sys.stderr)
                return 1

    # 3. Launch with setsid nohup
    log_path = os.path.join(ROOT, "runs", f"{args.name}.log")
    pid_path = os.path.join(ROOT, "runs", f"{args.name}.pid")
    rc_path = os.path.join(ROOT, "runs", f"{args.name}.rc")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cards
    env["PYTHONUNBUFFERED"] = "1"  # Python block-buffers stdout when it is a file
    if args.training and cards:
        env["NGPU"] = str(len(cards.split(",")))  # run_ddp.sh defaults to 8; the block is 7

    # A stale .rc from a previous run of this name would be read as this run's verdict.
    if os.path.exists(rc_path):
        os.unlink(rc_path)
    # The exit code must outlive the process, because the only thing that can read it
    # otherwise is whoever reaped the child. The monitor cannot: it polls a pid, sees it
    # vanish, and has no way to distinguish "finished" from "killed". It wrote status=ok
    # for both -- the 22B milestone was killed at 04:22 while yielding the lane and its
    # ledger row reads `ok | process exited`, indistinguishable from a completed score
    # (de, 2026-09-01). Wrapping the command so the shell records $? makes the verdict an
    # artifact rather than a guess. `exec` is deliberately absent: the wrapper must
    # survive the child to write the file.
    wrapped = ["bash", "-c", 'set -o pipefail; "$@"; rc=$?; printf %s "$rc" > "$0"; exit "$rc"',
               rc_path, *cmd]
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            wrapped,
            stdout=log_f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env, cwd=ROOT,
            start_new_session=True,  # setsid: detach from the controlling terminal
        )
    # Pid file: container pid AND the cmdline. A kill that uses the container pid
    # from the host fails silently (host has no such pid -- 2026-08-31, twice);
    # `harness kill <name>` resolves the host pid from the cmdline via tn exec.
    with open(pid_path, "w") as f:
        f.write(f"{proc.pid}\n{' '.join(cmd)}\n")

    # DECLARE THE CARDS. de-30: card_claim.py existed and harness launch -- the documented way
    # to start any GPU job -- never called it, so `card_claim.py status` on the pod reported all
    # eight cards ORPHAN. Ownership was inferred from nvidia-smi instead of declared, and on
    # 2026-09-02 two probes shared cards twice and OOM'd each other.
    #
    # WHICH PID. Not proc.pid: that is `bash -c 'set -o pipefail; "$@"; ...'`, a shell by
    # construction, and card_claim refuses a shell because a claim on one fails both ways (de-34,
    # measured 2026-09-03 on both of tonight's incidents). The claim must name the process that
    # dies WITH the job, so acquire on the job descendant. Measured on harness's own wrapper
    # shape: the descendant exists by the time Popen returns, both for a python payload and for a
    # shell script that execs one, as run_ddp.sh does. If it has not appeared yet -- a slow
    # interpreter start -- claim nothing and say so rather than claim the wrapper.
    claim_name = None
    if cards and not args.no_gpu:
        job_pids = _job_pids_for(proc.pid)
        if job_pids:
            ok_claim, claim_msg = _acquire_cards(args.name, cards, job_pids[0], f"harness launch {args.name}")
            if ok_claim:
                claim_name = args.name
            else:
                print(f"note   cards {cards} not claimed: {claim_msg}", file=sys.stderr)
        else:
            print(f"note   cards {cards} not claimed: no job process under {proc.pid} yet "
                  f"(a claim on the wrapper shell is worse than none -- de-34)", file=sys.stderr)

    # 4. Training jobs: verify the startup gate line
    if args.training:
        ok, reason = _wait_for_startup(log_path, args.gate_timeout)
        if not ok:
            # The group, not the pid: run_ddp.sh spawns torchrun which spawns ranks,
            # and killing the shell leaves them running (tilerl's matrix: bash rc -15,
            # torchrun alive). getpgid rather than assuming proc.pid leads its group --
            # a job launched without start_new_session does not (tilerl's review).
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                os.kill(proc.pid, signal.SIGTERM)
            subprocess.run(
                [sys.executable, os.path.join(HERE, "exp.py"),
                 "done", "--name", args.name,
                 "--result", f"killed: startup gate {reason}",
                 "--finding", f"gate failed: {reason}",
                 "--decision", "fix the startup issue",
                 "--status", "fail"],
                capture_output=True,
            )
            print(f"FAILED: {args.name} killed — {reason}", file=sys.stderr)
            _release_cards(claim_name)
            return 1

    # 5. Arm monitor
    monitor_pid = _arm_monitor(args.name, proc.pid, log_path, output_path=args.output)

    print(f"launched {args.name} (pid {proc.pid}, monitor {monitor_pid}) on cards {cards}")
    print(f"  log: {log_path}")
    print(f"  exp: python scripts/exp.py done --name {args.name} --result ... --finding ... --decision ...")

    # 6. Auto-resume: relaunch on a crash, never on a clean exit (de-1).
    # Blocking by design -- the supervisor must outlive the child, so the caller
    # detaches this whole command (setsid nohup), exactly as it detaches training.
    if args.auto_resume:
        return _supervise(args, cmd, proc, cards, log_path, pid_path)
    return 0


#: Reserved for train.py's NaN / kill-criterion stop: a deliberate abort, never resumed.
#: train.py does not raise it yet (it rolls back to good_state instead, train.py:2034);
#: the guard exists so that adding the stop does not also need a change here, and so a
#: future exit(_KILL_CRITERION_EXIT) cannot be silently treated as a crash.
_KILL_CRITERION_EXIT = 42


def _latest_step_ckpt(name):
    """(path, step) of the newest resumable checkpoint, or (None, None).

    Two names are resumable: ckpt_<name>.pt.step<N> from the periodic save and
    ckpt_<name>.pt.interrupt.step<N> from train.py's SIGTERM handler (:2479). The
    interrupt file is by construction the newest thing on disk when a signal arrives --
    written at the step the signal hit, after the last periodic save -- so matching only
    `.pt.step*` resumes from up to save_every steps earlier and silently discards the
    save whose whole purpose was to keep them. p500m_20b_0902 resumed from
    .interrupt.step83 on 2026-09-02; this glob would have taken step0 or nothing.
    Ties go to the interrupt file: it is the later write."""
    best, best_step, best_int = None, None, False
    for p in glob.glob(os.path.join(ROOT, f"ckpt_{name}.pt.step*")) + \
             glob.glob(os.path.join(ROOT, f"ckpt_{name}.pt.interrupt.step*")):
        m = re.search(r"\.step(\d+)$", p)
        if not m:
            continue
        step, is_int = int(m.group(1)), ".interrupt.step" in p
        if best_step is None or (step, is_int) > (best_step, best_int):
            best, best_step, best_int = p, step, is_int
    return best, best_step


def _supervise(args, cmd, proc, cards, log_path, pid_path, root=None):
    """Wait on a launched job; on a crash relaunch it with --resume, up to N times.

    `root` redirects the exp row and suppresses the monitor: the selftest must not
    write into the real ledger (see _close_row)."""
    resumes = []
    for attempt in range(args.auto_resume + 1):
        rc = proc.wait()
        if rc == 0:
            _close_row(args.name, "ok", f"exited 0 after {len(resumes)} resume(s)",
                       "clean exit", "none", root)
            return 0
        if rc == _KILL_CRITERION_EXIT:
            _close_row(args.name, "fail", f"kill criterion (exit {rc}) after {len(resumes)} resume(s)",
                       "deliberate stop: NaN or kill criterion, not a crash",
                       "diagnose the stop; auto-resume does not relaunch it", root)
            return rc
        if attempt == args.auto_resume:
            _close_row(args.name, "fail", f"exit {rc}, auto-resume exhausted ({args.auto_resume})",
                       f"crashed {len(resumes) + 1} times; resumed at steps {resumes}",
                       "investigate the crash before relaunching", root)
            return rc
        ckpt, step = _latest_step_ckpt(args.name)
        if ckpt is None:
            _close_row(args.name, "fail", f"exit {rc}, no step checkpoint to resume from",
                       "crashed before the first --save_every save",
                       "relaunch from scratch", root)
            return rc
        # The env fingerprint is part of what the checkpoint was trained under. A
        # changed environment makes a resume a different run wearing the same name.
        fp_now = _env_fp_now()
        fp_ckpt = _ckpt_env_fp(ckpt)
        if fp_now and fp_ckpt and fp_now != fp_ckpt:
            _close_row(args.name, "fail", f"exit {rc}, REFUSING resume: env fingerprint changed",
                       f"checkpoint {fp_ckpt} vs current {fp_now}",
                       "resume by hand after deciding the environment change is safe", root)
            print(f"REFUSING resume: env fingerprint {fp_ckpt} -> {fp_now}", file=sys.stderr)
            return rc
        print(f"auto-resume {attempt + 1}/{args.auto_resume}: exit {rc}, resuming from step {step} in 60s",
              flush=True)
        # Copy the crash scene before the resume appends over it. `cp`, not `mv`: the
        # dead run's fd is gone but the resumed child inherits log_f by the same path,
        # and moving the file out from under a live writer leaves it on an unlinked
        # inode with the visible log empty. The step-83 scene survived only because a
        # person remembered to archive it by hand at 01:38.
        try:
            import shutil

            died = f"{log_path}.died_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            shutil.copy2(log_path, died)
            print(f"crash scene archived: {os.path.basename(died)}", flush=True)
        except OSError as e:
            print(f"crash scene NOT archived ({type(e).__name__}) -- the resume will append over it",
                  file=sys.stderr, flush=True)
        time.sleep(60)
        resumes.append(step)
        rcmd = [c for c in cmd if not c.startswith("--resume")] + ["--resume", ckpt]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cards
        env["PYTHONUNBUFFERED"] = "1"
        if args.training and cards:
            env["NGPU"] = str(len(cards.split(",")))
        with open(log_path, "a") as log_f:
            log_f.write(f"\n=== auto-resume {attempt + 1}: exit {rc}, --resume {os.path.basename(ckpt)} ===\n")
            log_f.flush()
            proc = subprocess.Popen(rcmd, stdout=log_f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, cwd=ROOT,
                                    start_new_session=True)
        with open(pid_path, "w") as f:
            f.write(f"{proc.pid}\n{' '.join(rcmd)}\n")
        if not root:
            _arm_monitor(args.name, proc.pid, log_path, output_path=args.output)
    return 0


def _close_row(name, status, result, finding, decision, root=None):
    """Close an exp row. `root` exists for the selftest: exp.py takes no ambient
    override (the ledger gets no env var), so a test that cannot redirect it writes
    into the real ledger -- which is exactly what happened (four 'arts' rows,
    2026-08-31, one pair sharing an identity that then failed the sync guard)."""
    cmd = [sys.executable, os.path.join(HERE, "exp.py"), "done", "--name", name,
           "--result", result, "--finding", finding, "--decision", decision, "--status", status]
    if root:
        cmd += ["--root", root]
    subprocess.run(cmd, capture_output=True)


def _env_fp_now():
    try:
        sys.path.insert(0, HERE)
        from env_fp import env_fingerprint
        return env_fingerprint()
    except Exception:
        return None


def _ckpt_train_mix(path):
    """The mix a checkpoint was TRAINED on, from its own cfg.

    Scoring domain loss on a different mix's heads compares two models on text
    neither shares: the 3.24B milestone was scored on the ladder's heads
    (chat/code/en/math/textbook/web_hq/wiki) while stage 1 trains on
    code_rp1t/cot/en_c4/math_owm/textbook_30b/wiki_chat/zh_web -- zero overlap, and
    the readout called it a 6-of-7-domain regression (2026-08-31). The checkpoint
    knows what it read; asking it beats a default."""
    try:
        import torch

        cfg = torch.load(path, map_location="cpu", weights_only=False).get("cfg", {})
        mix = cfg.get("mix") if isinstance(cfg, dict) else getattr(cfg, "mix", None)
        return mix or None
    except Exception:
        return None


def _ckpt_env_fp(path):
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False).get("env_fp")
    except Exception:
        return None


def _drop_zombies(host_pids):
    """Keep only pids that are actually running.

    `pgrep -f X` matches on argv, and a zombie KEEPS its argv (`[run_ddp.sh] <defunct>`)
    while running nothing and holding no card. So a name match is not evidence of a live
    process -- on 2026-09-01 `pgrep -f compile_worker | wc -l` returned 1577 where 1570
    were zombies and 38 were live, and the miscount was read as CPU saturation on an idle
    machine. On a kill path the same substitution is worse than a miscount: it makes the
    caller act on processes that already exited.

    Filtered here rather than remembered at each call site, because knowing the trap
    exists and recalling it at the moment of use are different things -- the miscount
    above was made 15 minutes after merging the fix whose docstring states it.
    """
    if not host_pids:
        return []
    r = subprocess.run(["tn", "exec", f"ps -o pid=,stat= -p {','.join(host_pids)}"],
                       capture_output=True, text=True)
    live = set()
    for ln in r.stdout.splitlines():
        f = ln.split()
        if len(f) >= 2 and not f[1].startswith("Z"):
            live.add(f[0])
    return [p for p in host_pids if p in live]


def _gpu_descendants(root_host_pid):
    """Host pids holding GPU memory that descend from root_host_pid.

    The cmdline pattern cannot see them: score_matrix shells out to math_zh.py,
    code_zh.py, run_eval.py and domain_loss.py, so a child's cmdline shares no
    text with its parent's. Walk /proc/<pid>/stat's PPid field up from each pid
    nvidia-smi reports instead -- descent is the real relation, cmdline was a
    proxy for it (de + e1, 2026-08-31: a kill left math_zh.py holding 12.7 GB
    and the verification greped the same blind pattern, so it printed success).

    The chain runs through NON-GPU processes: score_matrix (383102) -> bash
    eval_math.sh (400242, no GPU) -> math_zh.py (400379, 6.5 GB). A ppid map
    built only over GPU pids stops at the shell and reports nothing, which is
    exactly how the first version of this function failed. Read the whole
    process table's ppid map, not just the GPU pids'."""
    r = subprocess.run(
        ["tn", "exec", "nvidia-smi --query-compute-apps=pid --format=csv,noheader"],
        capture_output=True, text=True,
    )
    gpu_pids = [p.strip() for p in r.stdout.split() if p.strip().isdigit()]
    if not gpu_pids:
        return []
    # Whole-table ppid map in one call: intermediate shells are not GPU processes,
    # so a map over gpu_pids alone breaks the chain.
    r = subprocess.run(["tn", "exec", "ps -eo pid=,ppid="], capture_output=True, text=True)
    ppid = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            ppid[parts[0]] = parts[1]
    out = []
    for p in gpu_pids:
        seen, cur = 0, p
        while cur in ppid and seen < 64:  # bounded: a cycle must not hang a kill
            cur = ppid[cur]
            seen += 1
            if cur == str(root_host_pid):
                out.append(p)
                break
    return out


def cmd_kill(argv):
    """`harness kill <name> [--dry]` — kill a launched job by name, not pid.

    The pid file holds the CONTAINER pid and cmdline; a host-side kill with that
    pid no-ops (the host has no such pid -- 2026-08-31, twice, while 32 workers
    kept writing). Resolves host pids with `tn exec pgrep -f` on the cmdline,
    matches the parent via /proc NSpid, kills workers first, then the parent and
    the monitor, and closes the exp row. --dry prints the resolution, kills nothing.

    Children whose cmdline differs from the parent's are found by DESCENT, not by
    pattern: see _gpu_descendants. The pattern alone cannot see them, and using it
    to verify the kill made the failure silent."""
    ap = argparse.ArgumentParser(prog="harness kill")
    ap.add_argument("name")
    ap.add_argument("--dry", action="store_true", help="print resolved pids, kill nothing")
    a = ap.parse_args(argv)
    pid_path = f"/work/aupai/runs/{a.name}.pid"
    r = subprocess.run(
        [os.path.expanduser("~/bin/pod"), f"cat {pid_path}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"no pid file on the pod: runs/{a.name}.pid", file=sys.stderr)
        return 1
    lines = r.stdout.splitlines()
    cpid, cmdline = lines[0], lines[1] if len(lines) > 1 else ""
    if not cmdline:
        print(f"pid file predates the cmdline record; resolve by hand: tn exec pgrep -f {a.name}", file=sys.stderr)
        return 1
    pattern = re.escape(cmdline)
    r = subprocess.run(["tn", "exec", f"pgrep -f '{pattern}'"], capture_output=True, text=True)
    host_pids = _drop_zombies([p for p in r.stdout.split() if p.strip()])
    parent, children = None, []
    for hp in host_pids:
        s = subprocess.run(["tn", "exec", f"grep -h NSpid /proc/{hp}/status"], capture_output=True, text=True)
        nsp = s.stdout.split()
        innermost = nsp[-1] if len(nsp) >= 2 else (nsp[0] if nsp else None)
        if innermost == cpid:
            parent = hp
        else:
            children.append(hp)
    # The monitor's embedded code names the container pid; it is the only other match.
    r = subprocess.run(["tn", "exec", f"pgrep -f '{cpid}'"], capture_output=True, text=True)
    monitor_pids = _drop_zombies([p for p in r.stdout.split() if p.strip() and p not in host_pids])
    # Differently-named children holding GPU memory (score_matrix -> math_zh.py etc).
    gpu_kids = [p for p in _gpu_descendants(parent) if p not in host_pids] if parent else []
    print(f"container pid {cpid} -> parent {parent or '?'}, workers {children or 'none'}, "
          f"monitor {monitor_pids or 'none'}, gpu descendants {gpu_kids or 'none'}")
    if a.dry:
        return 0
    # Two mechanisms, because they catch different escapes (de + e1, 2026-08-31):
    # the process GROUP sweeps everything the setsid'd launch spawned, including
    # processes holding no card, but misses anything that has left the group (a
    # double-setsid, a re-parented orphan). GPU descent catches exactly what holds a
    # card regardless of group. Group kill is the sweep; GPU occupancy is the
    # acceptance test, because a card held is the thing that actually costs us.
    pgids = set()
    for hp in ([parent] if parent else []) + children:
        r = subprocess.run(["tn", "exec", f"ps -o pgid= -p {hp}"], capture_output=True, text=True)
        if r.stdout.strip():
            pgids.add(r.stdout.strip())
    for pg in pgids:
        subprocess.run(["tn", "exec", f"kill -TERM -{pg}"], capture_output=True)
    time.sleep(2)
    for hp in children + gpu_kids:  # anything the group missed
        subprocess.run(["tn", "exec", f"kill {hp}"], capture_output=True)
    if parent:
        time.sleep(1)
        subprocess.run(["tn", "exec", f"kill {parent}"], capture_output=True)
    for hp in monitor_pids:
        subprocess.run(["tn", "exec", f"kill {hp}"], capture_output=True)
    time.sleep(2)
    # Verify three ways, because each is blind to what the others see: the group
    # (everything the launch spawned), the cmdline pattern (the historical check),
    # and GPU occupancy (the resource). The pattern that could not SEE the orphan
    # cannot prove it is gone, and a kill that reports success while 12.7 GB stays
    # held is worse than one that fails loudly.
    left = []
    for pg in pgids:
        r = subprocess.run(["tn", "exec", f"pgrep -g {pg}"], capture_output=True, text=True)
        left += [x for x in r.stdout.split() if x.strip() and x not in left]
    chk = subprocess.run(["tn", "exec", f"pgrep -f '{pattern}'"], capture_output=True, text=True)
    left += [p for p in chk.stdout.split() if p.strip() and p not in left]
    still_gpu = [p for p in _gpu_descendants(parent)] if parent else []
    reparented = [p for p in still_gpu if p not in left]
    left += reparented
    if left:
        for hp in left:  # a job that ignores TERM still must not hold a card
            subprocess.run(["tn", "exec", f"kill -9 {hp}"], capture_output=True)
        time.sleep(2)
        again = []
        for pg in pgids:
            r = subprocess.run(["tn", "exec", f"pgrep -g {pg}"], capture_output=True, text=True)
            again += [x for x in r.stdout.split() if x.strip()]
        again += [p for p in (_gpu_descendants(parent) if parent else []) if p not in again]
        if again:
            print(f"STILL ALIVE after KILL: {' '.join(again)}", file=sys.stderr)
            return 1
        if reparented:
            # Out of the group but still on a card: the escape the group sweep cannot
            # see. Worth naming rather than silently reaping -- it means something
            # re-parented, and the next one may not hold a card to be found by.
            print(f"  REPARENTED (outside the job's group, found by GPU occupancy): "
                  f"{' '.join(reparented)}", file=sys.stderr)
        print(f"  {len(left)} process(es) needed SIGKILL: {' '.join(left)}", file=sys.stderr)
    subprocess.run(
        [os.path.expanduser("~/bin/pod"),
         f"cd /work/aupai && python3 scripts/exp.py done --name {a.name} --status fail "
         f"--result 'killed by harness kill' --finding 'operator kill of container pid {cpid}' "
         f"--decision 'relaunch or close'"],
        capture_output=True,
    )
    print(f"killed {a.name}; exp row closed")
    return 0


#: Tokens per optimizer step for the stage-1 recipe: batch 16 x accum 2 x seq 4096
#: x 7 ranks = 917,504. Used to state what a checkpoint actually saw, against the
#: milestone's nominal budget.
TOKENS_PER_STEP = 16 * 2 * 4096 * 7


def _paired_profile(paired, matrix=None):
    """The profile the pair's record actually carries: 'milestone' when one exists,
    else 'full'. Asking the ledger beats assuming -- a pair scored only under the
    milestone profile has no full record, and a missing record reads as ABSENT
    rather than as an error."""
    path = matrix or os.path.join(ROOT, "runs", "score_matrix.jsonl")
    have = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ckpt") == paired:
                have.add(r.get("profile", "full"))
    return "milestone" if "milestone" in have and "full" not in have else "full"


def _pin_milestone(watch_dir, run, ckpt, token):
    """Take a milestone checkpoint out of train.py's rotation. Returns the pinned path.

    HARDLINK, not copy: instant, and it adds no 959MB write to the training box's disk
    during a save window. The inode survives the pruner's os.remove of the .step name,
    since that only drops one link.

    The name must sit outside the pruner's glob. train.py:2091 globs
    `ckpt_<run>.pt.step*`; `ckpt_<run>.milestone_<token>.pt` has no `.pt.step`, so the
    roller cannot see it. A name the glob matches is not a pin.

    Called the MOMENT the watcher detects the milestone save, before scoring is even
    queued: at save_every 500 the save-to-rotation window is ~22 minutes, so a pin at
    scoring start loses the same race the 3.24B rescore lost (b0, fb, 2026-08-31)."""
    src = os.path.join(watch_dir, ckpt)
    # The STEP goes in the name. Once the roller deletes ckpt_<run>.pt.step8500 the
    # pinned copy is the only survivor, and a name carrying only the token cannot say
    # which step it holds -- the milestone label is nominal (8b) while the checkpoint is
    # 8500 steps and 7.799B tokens, 2.5% apart (fb, 2026-09-01).
    m_step = re.search(r"\.step(\d+)$", ckpt)
    step_part = f"_step{m_step.group(1)}" if m_step else "_final"
    dst = os.path.join(watch_dir, f"ckpt_{run or 'run'}.milestone_{token}{step_part}.pt")
    if os.path.exists(dst) or not os.path.exists(src):
        return dst if os.path.exists(dst) else None
    try:
        os.link(src, dst)  # same filesystem by construction: both live beside the run
        return dst
    except OSError:
        try:
            import shutil as _sh

            _sh.copy2(src, dst)
            return dst
        except OSError as e:
            print(f"WARNING: could not pin {ckpt}: {e}; the rescore window is ~3 saves wide",
                  file=sys.stderr)
            return None


def _run_alive(run):
    """True if the training run still has a process. A watcher that waits for an
    exact save must not wait forever after the run ends."""
    try:
        r = subprocess.run([os.path.expanduser("~/bin/pod"), f"pgrep -f 'name {run}' | head -1"],
                           capture_output=True, text=True, timeout=20)
        return bool(r.stdout.strip())
    except Exception:
        return True  # unknown: prefer waiting over mislabelling


MILESTONE_TOKENS = {"3.24b": 3.24e9, "8b": 8e9, "15b": 15e9, "16b": 16e9,
                    "22b": 22e9, "30b": 30e9}
#: Longest-first so "3.24b" wins over a shorter token that prefixes it; built FROM the
#: dict rather than hand-written beside it, because the hand-written copy is the defect:
#: adding 22b to the dict alone leaves the parser blind to it.
#:
#: Anchored on `.milestone_<tok>_`, the ONE name _pin_milestone writes. The previous
#: pattern was `_(tok)(?:_|[.\-])` anywhere in the name, which matches the RUN name:
#: ckpt_pretrain_30b_s2.pt.step21000 parsed as "30b" and would have recorded a 19.3B
#: checkpoint under the 30B milestone's budget. Same class as the step-3000-labelled-3.24b
#: incident, one level up -- the milestone is a property of the pin, not of the run's name.
_MILESTONE_RE = re.compile(
    r"\.milestone_(" + "|".join(re.escape(t) for t in sorted(MILESTONE_TOKENS, key=len, reverse=True))
    + r")(?:_|\.)")


def _milestone_token(name):
    """The milestone this checkpoint IS, read from its pin name, or None.

    None is the fail-closed answer: the caller then requires --tokens rather than
    inferring a budget from a substring of the run name."""
    m = _MILESTONE_RE.search(name)
    return m.group(1) if m else None


def _exp_row_status(name):
    """Last status of an exp row; rows are append-only, last row with the name wins."""
    # Folded by (name, started), then the latest start for that name. Folding by
    # NAME alone attaches a close to whichever row came last in the file, which is
    # how a close landed on the wrong start of p02_fp32m_s0 (fb, 2026-09-01).
    evs = _exp_events(ROOT)
    if not evs:
        return None
    mine = [r for r in evs if r.get("name") == name]
    if not mine:
        return None
    return max(mine, key=lambda r: str(r.get("started", ""))).get("status")


def _wait_launched(name, timeout):
    """Wait for a job launched in-process by cmd_launch. Returns its exit code,
    or None on timeout.

    waitpid, not a poll of the exp row: cmd_launch's Popen child is OUR child, and
    an unreaped child that exits becomes a zombie whose pid `os.kill(pid, 0)` still
    accepts. The monitor therefore never sees it die, never closes the row, and a
    row-polling wait hangs for its full timeout with the work long finished
    (2026-08-31, the t39 dry run: score_matrix done in 50 min, driver still waiting).
    Reaping is also what yields the exit code, which auto-resume needs.
    """
    pid_path = os.path.join(ROOT, "runs", f"{name}.pid")
    try:
        pid = int(open(pid_path).readline().strip())
    except (OSError, ValueError):
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return _exp_row_exit(name)  # not our child (resumed run): fall back to the row
        if done == pid:
            return os.waitstatus_to_exitcode(status)
        time.sleep(5)
    return None


def _exp_row_exit(name):
    """0 if the row closed ok, 1 if it closed failed, None while still running."""
    st = _exp_row_status(name)
    return None if st in (None, "running") else (0 if st == "ok" else 1)


def cmd_milestone(argv):
    """`harness milestone <ckpt> [--paired <ckpt>] [--tokens T] [--mix M] [--watch <dir>] [--interval N] [--dry]`

    Score a milestone checkpoint unattended (t39): score_matrix --profile milestone
    on the lane via `harness launch`, then eval/readout_30b.py against the
    pre-registration, then append the milestone ledger row (runs/milestones.jsonl)
    and the facts entry, and print the three-state verdict per metric.

    --watch polls a directory for the run's step checkpoints (ckpt_<run>.pt.step<N>)
    and scores the saved step nearest each milestone step (3.24B ~= step 3531 at
    16x2x4096x7 -> the step-3500 save), so a 3am milestone is scored without a
    person. The pair defaults to the previous milestone's checkpoint, else
    ckpt_p324.pt (the 3.24B ladder stand-in). The paired ckpt file may already be
    deleted (train.py keeps the newest 3): readout_30b reads the pair's score
    record by name, which persists in runs/score_matrix.jsonl.

    domain_loss needs no launch of its own: the milestone profile scores it on the
    3.24b mix, the same heads as the ladder records, and readout_30b falls back to
    the score record when no explicit domain-loss file is given.
    """
    ap = argparse.ArgumentParser(prog="harness milestone")
    ap.add_argument("ckpt", nargs="?", help="checkpoint file (single-run mode)")
    ap.add_argument("--paired", default=None, help="paired checkpoint (default: previous milestone, else ckpt_p324.pt)")
    ap.add_argument("--unpaired", action="store_true",
                    help="score and record only, no comparison: the own-mix baseline a later "
                         "milestone differences against (b0's ruling, 2026-08-31)")
    ap.add_argument("--tokens", type=float, default=None, help="milestone token budget (default: parsed from the name)")
    ap.add_argument("--mix", default=None,
                    help="domain-loss heads mix (default: the mix each checkpoint was trained on)")
    ap.add_argument("--watch", default=None, help="poll this directory for step checkpoints")
    ap.add_argument("--run", default=None, help="watch: run name (checkpoints are ckpt_<run>.pt.step<N>)")
    ap.add_argument("--milestones", default=None, help="watch: '3.24b=3500,8b=8500,15b=16500' (token=nearest-saved-step)")
    ap.add_argument("--final-step", type=int, default=None,
                    help="the run's total step count; registers the suffix-less run-end "
                         "checkpoint (ckpt_<run>.pt) at this step so a final milestone fires")
    ap.add_argument("--save-every", type=int, default=500, help="watch: max distance from the milestone step to score a save")
    ap.add_argument("--interval", type=int, default=120, help="watch poll interval in seconds")
    ap.add_argument("--dry", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args(argv)

    def run_one(ckpt, paired, tokens, milestone=None):
        stem = ckpt[:-3] if ckpt.endswith(".pt") else ckpt
        # Heads come from the checkpoint's own training mix unless overridden. A
        # hardcoded default scored the 3.24B milestone on the ladder's heads while
        # stage 1 trained on a disjoint set, and the readout reported it as a
        # 6-of-7-domain regression (2026-08-31).
        mix = a.mix or _ckpt_train_mix(os.path.join(a.watch or ROOT, ckpt)) \
            or os.path.join(ROOT, "data/mix_scale_3.24b.json")
        if not os.path.isabs(mix):
            mix = os.path.join(ROOT, mix)
        if a.dry:
            print(f"harness launch ms_{stem} -- eval/score_matrix.py --ckpt {ckpt} "
                  f"--profile milestone --mix {os.path.relpath(mix, ROOT)} --json runs/score_matrix.jsonl")
            if a.unpaired:
                print(f"(unpaired baseline: no readout_30b call; record + "
                      f"runs/readout_{stem}.txt written from the score record)")
                return "dry"
            print(f"python3 eval/readout_30b.py --milestone {ckpt} --paired {paired} "
                  f"--milestone-tokens {tokens} --milestone-profile milestone --paired-profile full "
                  f"> runs/readout_{stem}.txt")
            return "dry"
        # Idempotent second attempt: the watcher already pinned this at detection, but a
        # single-run invocation (harness milestone <ckpt>) has no watcher to have done it.
        if milestone:
            _pin_milestone(a.watch or ROOT, a.run, ckpt, milestone)
        rc = cmd_launch([
            f"ms_{stem}", "--hypothesis", f"milestone {tokens / 1e9:.2f}B score_matrix profile", "--",
            "eval/score_matrix.py", "--ckpt", ckpt, "--profile", "milestone",
            "--mix", mix, "--json", "runs/score_matrix.jsonl",
        ])
        if rc != 0:
            return "refused"  # lane occupied; the watcher retries next poll
        st = _wait_launched(f"ms_{stem}", 5400)
        if st is None:
            return "score_matrix timeout"
        if st != 0:
            return f"score_matrix exit {st}"
        m_step = re.search(r"\.step(\d+)$", ckpt)
        actual_tokens = int(m_step.group(1)) * TOKENS_PER_STEP if m_step else None
        # The pair's budget: a milestone row for it, else the ladder point's nominal.
        paired_tokens = None
        ms_path = os.path.join(ROOT, "runs", "milestones.jsonl")
        if os.path.exists(ms_path):
            for line in open(ms_path, encoding="utf-8"):
                if line.strip():
                    pr = json.loads(line)
                    if pr.get("ckpt") == paired:
                        paired_tokens = pr.get("actual_tokens") or pr.get("tokens")
        if paired_tokens is None and paired == "ckpt_p324.pt":
            paired_tokens = 3.24e9
        readout_path = os.path.join(ROOT, "runs", f"readout_{stem}.txt")
        if a.unpaired:
            # No valid pair exists yet: stage-1's heads are disjoint from the ladder's,
            # so both directions are confounded -- the ladder-heads read is the OOD
            # penalty and a stage-1-heads read against p324 is its mirror. Record the
            # baseline; 8B and 15B share stage-1's domains and difference against it
            # (b0 ruled, 44 reviewed).
            rec = None
            smp = os.path.join(ROOT, "runs", "score_matrix.jsonl")
            if os.path.exists(smp):
                for _l in open(smp, encoding="utf-8"):
                    if not _l.strip():
                        continue
                    try:
                        _r = json.loads(_l)
                    except json.JSONDecodeError:
                        continue
                    if _r.get("ckpt") == ckpt and _r.get("profile", "full") == "milestone":
                        rec = _r
            dl = (rec or {}).get("metrics", {}).get("domain_loss", {})
            lines = [f"=== {ckpt}: UNPAIRED own-mix baseline ===",
                     f"mix: {os.path.relpath(mix, ROOT)}",
                     f"tokens: {actual_tokens / 1e9:.3f}B (nominal {tokens / 1e9:.2f}B)",
                     "",
                     "No verdict: this milestone has no comparable pair. Recorded so 8B and",
                     "15B, which share these heads, can difference against it.",
                     ""]
            for k in sorted(x for x in dl if isinstance(dl[x], dict)):
                lines.append(f"  {k:15s} {dl[k].get('loss')}")
            for k in sorted(x for x in (rec or {}).get("metrics", {}) if x != "domain_loss"):
                lines.append(f"  {k:15s} {(rec or {}).get('metrics', {})[k]}")
            body = "\n".join(lines) + "\n"
            with open(readout_path, "w", encoding="utf-8") as f:
                f.write(body)
            print(body)
            r = None
        else:
            r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "eval", "readout_30b.py"),
                 "--milestone", ckpt, "--paired", paired, "--milestone-tokens", str(tokens),
                 "--milestone-profile", "milestone",
                 # The PAIR's profile, resolved from what exists rather than assumed.
                 # Hardcoding "full" was right while the pair was ckpt_p324 (a ladder
                 # checkpoint with a full record) and wrong the moment a milestone
                 # became the pair: the 8B baseline exists only under profile=milestone,
                 # so the lookup missed and the 15B readout printed ABSENT on every
                 # metric -- a "no metric moved" verdict resting on a lookup miss, which
                 # is worse than a wrong number because it reads as a measurement (fb).
                 "--paired-profile", _paired_profile(paired),
                 "--milestone-mix", mix]
                + (["--paired-mix", _ckpt_train_mix(os.path.join(a.watch or ROOT, paired)) or ""]
                   if _ckpt_train_mix(os.path.join(a.watch or ROOT, paired)) else [])
                + (["--actual-tokens", str(actual_tokens)] if actual_tokens else [])
                + (["--paired-tokens", str(paired_tokens)] if paired_tokens else []),
                capture_output=True, text=True,
            )
            with open(readout_path, "w", encoding="utf-8") as f:
                f.write(r.stdout)
            print(r.stdout)
            if r.returncode != 0:
                return f"readout rc={r.returncode}: {r.stderr[:200]}"
        moved = r.stdout.count("verdict: moved") if r else 0
        preds = [p for p in (
            os.path.join(ROOT, "data", "eval", f"preds_{ckpt}.jsonl"),
            os.path.join(ROOT, "data", "eval", f"preds_code_{ckpt}.jsonl"),
            os.path.join(ROOT, "data", "eval", f"preds_code_v2_{ckpt}.jsonl"),
        ) if os.path.exists(p)]
        # actual_step/actual_tokens were computed once above and passed to the
        # readout's budget gate. Reusing them here rather than recomputing keeps the
        # number that GATES and the number that is RECORDED from drifting apart --
        # the shape that produced tonight's l1 retraction, where a fact cited an
        # artifact whose contents had moved underneath it (e1).
        actual_step = int(m_step.group(1)) if m_step else None
        row = {
            "ckpt": ckpt, "paired": (None if a.unpaired else paired), "tokens": tokens,
            "unpaired_baseline": bool(a.unpaired), "mix": os.path.relpath(mix, ROOT),
            "step": actual_step, "actual_tokens": actual_tokens,
            "token_shortfall": (round(1 - actual_tokens / tokens, 4) if actual_tokens and tokens else None),
            "milestone": milestone, "launcher": "harness", "score_matrix": "runs/score_matrix.jsonl",
            "preds": [os.path.relpath(p, ROOT) for p in preds],
            "readout": f"runs/readout_{stem}.txt", "metrics_moved": moved,
            "measured": time.strftime("%Y-%m-%d", time.gmtime()),
        }
        with open(os.path.join(ROOT, "runs", "milestones.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        facts_path = os.path.join(ROOT, "facts", "base_eval.json")
        facts = json.load(open(facts_path, encoding="utf-8"))
        facts["facts"] = [e for e in facts["facts"] if e.get("id") != f"be.milestone_{stem}"]
        facts["facts"].append({
            "id": f"be.milestone_{stem}",
            "value": f"{moved} metric(s) moved past threshold vs {paired}; full verdict in runs/readout_{stem}.txt",
            "measured": row["measured"],
            "source": f"runs/readout_{stem}.txt",
            "config": f"score_matrix --profile milestone --mix {row['mix']}; readout_30b prereg thresholds; paired {paired}",
            "uncertainty": "per-metric n and threshold printed in the readout file",
            "status": "measured",
        })
        json.dump(facts, open(facts_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return "ok"

    if a.watch:
        if not a.run or not a.milestones:
            ap.error("--watch needs --run and --milestones 'token=step,...'")
        spec = {}  # insertion order = pairing order: each milestone pairs the previous
        for item in a.milestones.split(","):
            tok, step = item.split("=")
            tok = tok.strip()
            if tok not in MILESTONE_TOKENS:
                ap.error(f"unknown milestone token {tok!r}; known: {sorted(MILESTONE_TOKENS)}")
            spec[tok] = int(step)
        if a.final_step:
            unreachable = {t: v for t, v in spec.items() if v > a.final_step}
            if unreachable:
                print(f"REFUSING: milestone target(s) past the run's final step "
                      f"{a.final_step}: {unreachable}. They can never fire -- the run ends "
                      f"first and the milestone is silently skipped.", file=sys.stderr)
                return 2
        print(f"watching {a.watch} for ckpt_{a.run}.pt.step<N>; "
              f"milestones {spec}; pair = previous milestone, else ckpt_p324.pt", flush=True)
        ms = os.path.join(ROOT, "runs", "milestones.jsonl")
        while True:
            scored = {}  # milestone token -> ckpt file, from the ledger
            if os.path.exists(ms):
                for l in open(ms, encoding="utf-8"):
                    if l.strip():
                        r = json.loads(l)
                        # is not None, not truthiness: light-profile rows carry
                        # milestone=null and must be skipped, but a token of "" or 0
                        # would be skipped too -- the --seed 0 class (tilerl, t58).
                        if r.get("milestone") is not None:
                            scored[r["milestone"]] = r["ckpt"]
            saved = {}
            for p in glob.glob(os.path.join(a.watch, f"ckpt_{a.run}.pt.step*")):
                m = re.search(r"\.step(\d+)$", p)
                if m:
                    saved[int(m.group(1))] = os.path.basename(p)
            # The run-end checkpoint has NO .step suffix (train.py:2168 writes
            # ckpt_<run>.pt), so the glob above cannot see it and a final-step milestone
            # would never fire. It is also the stage-2 resume source, so it is the one
            # artifact that must never be missed. Register it at the run's true final
            # step, which a target past the end (16500 armed against a 16281-step run)
            # would otherwise skip entirely (fb, 2026-09-01).
            final = os.path.join(a.watch, f"ckpt_{a.run}.pt")
            if os.path.exists(final) and a.final_step:
                saved.setdefault(a.final_step, os.path.basename(final))
            for tok, target in spec.items():
                if tok in scored or not saved:
                    continue
                # Never score BELOW the target while the exact save is still coming.
                # min(|s-target|) took step3000 for target 3500 because 500 is not
                # > save_every, labelled a 2.753B checkpoint as the 3.24B milestone --
                # 15% short, and run_one records the nominal budget regardless (e1,
                # 2026-08-31). A save at or past the target is a real reading; one
                # before it is a different budget wearing the milestone's name.
                at_or_past = [x for x in saved if x >= target]
                if at_or_past:
                    step = min(at_or_past)
                elif max(saved) >= target - a.save_every and _run_alive(a.run):
                    continue  # the exact save is one interval away and training is up
                else:
                    step = max(saved)  # run ended short; the last save is the best read
                if abs(step - target) > a.save_every:
                    continue  # nearest save is too far from the milestone step; wait
                ckpt = saved[step]
                prev = [t for t in spec if t in scored]
                paired = a.paired or (scored[prev[-1]] if prev else "ckpt_p324.pt")
                print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] milestone {tok} @ step {step} "
                      f"(target {target}): {ckpt} vs {paired}", flush=True)
                # Pin BEFORE scoring: run_one may queue behind an occupied lane, and the
                # rotation does not wait for the queue.
                pinned = _pin_milestone(a.watch, a.run, ckpt, tok)
                if pinned:
                    print(f"  pinned -> {os.path.basename(pinned)} (outside the roller)", flush=True)
                res = run_one(ckpt, paired, MILESTONE_TOKENS[tok], milestone=tok)
                print(f"  -> {res}", flush=True)
            time.sleep(a.interval)

    if not a.ckpt:
        ap.error("ckpt required (or --watch)")
    tokens = a.tokens or MILESTONE_TOKENS.get(_milestone_token(a.ckpt) or "")
    if not tokens:
        ap.error("--tokens required (cannot parse a milestone token from the checkpoint name)")
    paired = a.paired or "ckpt_p324.pt"
    # Label the row with the milestone this run represents. The watcher dedups on it,
    # and a null makes a hand-run milestone invisible to the watcher that follows it.
    label = _milestone_token(a.ckpt) or next(
        (t for t, v in MILESTONE_TOKENS.items() if abs(v - tokens) / v < 0.05), None)
    res = run_one(a.ckpt, paired, tokens, milestone=label)
    if res not in ("ok", "dry"):
        print(f"FAILED: {res}", file=sys.stderr)
        return 1
    return 0


def cmd_install_hooks(rest):
    """`harness install-hooks` -- symlink .git/hooks/{pre-commit,pre-merge-commit,
    post-merge,post-commit,commit-msg} to scripts/hooks/. pre-commit covers direct
    commits; pre-merge-commit (git >= 2.24) covers non-fast-forward merges, which
    otherwise run no hook at all (2026-08-31: a bad fact entered main through a clean
    merge). post-merge covers fast-forward merges, which also run no hook (2026-09-03
    friction: a wip commit landed on main unchecked via fast-forward, ~10 min).
    The hook refuses staged files >5MB and new data/ paths not in the allow-list."""
    # The common dir: in a linked worktree .git is a FILE, and the common dir's
    # hooks/ is what git consults -- installing there covers every worktree once.
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], cwd=ROOT,
        capture_output=True, text=True,
    ).stdout.strip()
    if not common:
        print("cannot resolve git common dir (run inside the repo)")
        return 1
    common_abs = common if os.path.isabs(common) else os.path.join(ROOT, common)
    main_root = os.path.dirname(common_abs)  # .../aupai from .../aupai/.git (worktree-safe)
    # The hook source is the MAIN tree's copy, never this worktree's: a symlink into
    # a worktree breaks when the worktree is removed and tracks a branch, not main.
    hook_src = os.path.join(main_root, "scripts", "hooks", "pre-commit")
    if not os.path.exists(hook_src):
        print(f"hook source missing: {hook_src}")
        return 1
    hooks_dir = os.path.join(common_abs, "hooks")
    # post-commit has its OWN source: it is not the same script under another name.
    # It repairs the shared index that pre-commit's manifest `git add` leaves stale
    # under `git commit -- <paths>` -- see scripts/hooks/post-commit.
    for name in ("pre-commit", "pre-merge-commit", "post-merge", "post-commit", "commit-msg"):
        # commit-msg is its own file, not the pre-commit script under another name: it is the
        # only hook git hands THIS commit's message (argv[1]). pre-commit cannot read it --
        # .git/COMMIT_EDITMSG there still holds the PREVIOUS commit's message, measured four ways
        # (de-33, 2026-09-03) -- so the ledger guard grants its exception by env var in
        # pre-commit and requires the reason here, where the message is real.
        if name in ("post-commit", "commit-msg", "post-merge"):
            src = os.path.join(main_root, "scripts", "hooks", name)
        else:
            src = hook_src
        if not os.path.exists(src):
            print(f"hook source missing: {src}")
            return 1
        hook_dst = os.path.join(hooks_dir, name)
        os.makedirs(hooks_dir, exist_ok=True)
        if os.path.lexists(hook_dst):
            os.remove(hook_dst)
        os.symlink(os.path.relpath(src, hooks_dir), hook_dst)
        print(f"installed: {hook_dst} -> {os.path.relpath(src, main_root)}")
    # THE MERGE DRIVERS BELONG HERE, not in .gitattributes alone. The attribute names a
    # driver; `git config merge.<name>.driver` defines it, per clone and untracked -- so a
    # committed attribute with no config is a merge that cannot find its driver, which is
    # the same shape as a hook edited in a branch worktree: installed-looking and never run.
    # Installing hooks is the one step every session already runs in every tree.
    sys.path.insert(0, os.path.join(main_root, "scripts"))
    try:
        import merge_drivers
        merge_drivers.install(root=ROOT)
        ok, msg = merge_drivers.check(root=ROOT)
        print(f"merge drivers: {'OK' if ok else 'INCOMPLETE'} -- {msg}")
    except Exception as e:
        print(f"merge drivers NOT installed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def main():
    for _k in [k for k in os.environ if k.startswith("GIT_")]:
        os.environ.pop(_k)
    # Drop inherited GIT_* before anything runs. git sets GIT_DIR and GIT_INDEX_FILE for
    # its hooks, and a hook that runs a selftest passes them down: any `git init` in a
    # temp directory then RECONFIGURES the real repository, and `core.bare = true` on a
    # repo with a worktree makes every git command in every session fatal. That happened
    # twice on 2026-09-02, from two different selftests.
    #
    # Here, not at the eight `git init` call sites, and not only in the hook: the
    # invariant is "this file's git work is about ROOT, never about whatever invoked us",
    # which holds for `harness check` under a hook, under CI, and typed by hand. Guarding
    # the call sites means every future one must remember; guarding the hook means only
    # that entry point is safe. pod_drift.py keeps its own copy -- it runs standalone.
    for _v in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY",
               "GIT_COMMON_DIR"):
        os.environ.pop(_v, None)
    # argparse with choices, not a hand-rolled scan: a bare-flag filter once resolved
    # cmd="7", matched no branch, printed nothing and exited 0 -- a silent no-op, the
    # failure mode this file exists to prevent.
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        return run_dispatch(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "task":
        return cmd_task(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "friction":
        return cmd_friction(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        return cmd_sync(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "install-hooks":
        return cmd_install_hooks(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "launch":
        return cmd_launch(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "kill":
        return cmd_kill(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "milestone":
        return cmd_milestone(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "free-card":
        return cmd_free_card(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "cmd", nargs="?", default="all", choices=["all", "check", "ledger", "gaps", "measure", "stages", "board"]
    )
    ap.add_argument("--json", action="store_true", help="board: emit state as JSON instead of HTML")
    ap.add_argument("--html", default=None, help="board: output path (default runs/board.html)")
    ap.add_argument("--only", help="measure: substring filter on the checkpoint name")
    ap.add_argument("--ngpu", help="measure: shards for eval_all.sh")
    ap.add_argument("--tokenizer", help="measure: the vocabulary these checkpoints were trained on")
    ap.add_argument("--dry", action="store_true", help="measure: list what would run")
    ap.add_argument("--full", action="store_true", help="measure: the whole matrix, not just math-hard")
    ap.add_argument("--merge-drops", action="store_true",
                    help="print `path<TAB>parent` for every path HEAD's parents held and HEAD "
                         "lacks with no written deletion; exit 1 if any. For merge_main.sh")
    ap.add_argument("--selftest", action="store_true", help="every check must FAIL on its broken world")
    ap.add_argument("--selftest-touching", metavar="PATHS",
                    help="comma-separated files: verify only the checks whose run() or broken() "
                         "is defined in them. For the pre-commit hook, which cannot afford the "
                         "full ~4min run; prints what it did NOT cover")
    a = ap.parse_args()
    if a.merge_drops:
        _lost = merge_drops(ROOT)
        if _lost is None:
            print("cannot read HEAD", file=sys.stderr)
            return 2
        for _p, _path in _lost:
            print(f"{_path}\t{_p}")
        return 1 if _lost else 0
    if a.selftest_touching:
        _paths = [p.strip() for p in a.selftest_touching.split(",") if p.strip()]
        _names = _checks_touching(_paths)
        if not _names:
            print(f"no CHECK function is changed by the staged diff of {', '.join(_paths)} -- "
                  f"nothing scoped to verify. THIS IS NOT A PASS for those files: an edit to a "
                  f"shared helper or to the CHECKS table can break any check, and only the full "
                  f"`harness check --selftest` covers that.")
            return 0
        return _demo(only=set(_names)) or 0
    if a.selftest:
        return _demo() or 0
    cmd = a.cmd
    res = []
    if cmd in ("all", "check"):
        print("INVARIANTS  (a check that cannot run is a FAILURE, never a pass)")
        res = run_checks()
        bad = [n for n, s, *_ in res if s == FAIL]
        warns = [n for n, s, *_ in res if s == WARN]
        timed = [n for n, s, *_ in res if s == TIMEOUT]
        skipped = [n for n, s, *_ in res if s == SKIP]
    else:
        bad, warns, timed = [], [], []
        skipped = []
    if cmd in ("all", "ledger"):
        print("\nLEDGER  (provenance and score on one line)")
        ledger()
    if cmd in ("all", "gaps"):
        print("\nGAPS  (stated out loud, never inferred from an absence)")
        gaps()
    if cmd == "measure":
        return measure(only=a.only, ngpu=a.ngpu, tokenizer=a.tokenizer, dry=a.dry, full=a.full)
    if cmd == "board":
        return cmd_board(as_json=a.json, html_path=a.html)
    if cmd in ("all", "stages"):
        print("\nSTAGES  (a stage is done when its falsifying measurement exists)")
        stages(res)
    if bad:
        print(f"\n{len(bad)} invariant(s) FAILED: {', '.join(bad)}")
        return 1
    if timed:
        # Non-blocking on the first strike, but never silent: a check that did not run
        # is not a check that passed, and the next consecutive timeout exits 1.
        print(f"\n{len(timed)} check(s) TIMED OUT and did not run: {', '.join(timed)} "
              f"-- a second consecutive timeout FAILs")
    if skipped:
        # 0 FAIL means nothing without the denominator. Same sha 2dfe207a: Mac printed
        # 0 FAIL over 38 checks while the pod FAILed 9 -- the ones that skip here are
        # where they live, and the last line a reader acts on never said so.
        print(f"\n{len(bad)} FAIL of {len(res) - len(skipped)} run; {len(skipped)} did "
              f"NOT run here: {', '.join(skipped)} -- green here is not green on the pod")
    if warns:
        print(f"\n{len(warns)} non-blocking warning(s) (to-dos, not failures): {', '.join(warns)}")
    # 44-10: this names the scope of what ran; the "did NOT run here" line above is the refusal.
    n_pod = sum(1 for v in EVIDENCE.values() if v == "pod")
    print(f"\nauthority: {len(EVIDENCE) - n_pod} repo checks (green here = green on main), "
          f"{n_pod} pod checks (green here = green on the pod only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
